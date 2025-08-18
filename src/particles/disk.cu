#include "particles/disk.cuh"
#include "kernels/common.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

namespace md::disk {

__constant__ DiskConst g_disk;

void bind_disk_globals(const double* d_e_interaction, const double* d_mass, const double* d_rad, unsigned int* d_rebuild_flag, const double* d_thresh2) {
    DiskConst h { d_e_interaction, d_mass, d_rad, d_rebuild_flag, d_thresh2 };
    cudaMemcpyToSymbol(g_disk, &h, sizeof(DiskConst));
}

// Disk-specific kernels
namespace kernels {

// Calculate the threshold squared for each system
__global__ void calculate_thresh2_kernel(
    const double* __restrict__ skin,
    double* __restrict__ thresh2
) {
    const int S = md::geo::g_sys.n_systems;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S) return;

    const double skin_i_half = skin[i] * verlet_skin_to_threshold_factor;
    thresh2[i] = skin_i_half * skin_i_half;
}

// Update positions and displacements given velocities and a scale factor - used for cell neighbor list
__global__ void update_positions_kernel_cell(
    double* __restrict__ x,
    double* __restrict__ y,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ last_x,
    const double* __restrict__ last_y,
    double* __restrict__ disp2,
    double scale
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double pos_x = x[i];
    double pos_y = y[i];
    const double vel_x = vx[i];
    const double vel_y = vy[i];

    // Update positions
    pos_x += vel_x * scale;
    pos_y += vel_y * scale;
    x[i] = pos_x;
    y[i] = pos_y;

    // Calculate squared displacement for neighbor list update
    double dx = pos_x - last_x[i];
    double dy = pos_y - last_y[i];
    double d2 = dx * dx + dy * dy;
    disp2[i] = d2;

    // Determine per-system rebuild flag
    const int sid = md::geo::g_sys.id[i];
    const double thresh2 = g_disk.thresh2[sid];
    if (d2 > thresh2) atomicOr(&g_disk.rebuild_flag[sid], 1u);
}

// Update positions and displacements given velocities and a scale factor - used for naive neighbor list
__global__ void update_positions_kernel_naive(
    double* __restrict__ x,
    double* __restrict__ y,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    double scale
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x[i] += vx[i] * scale;
    y[i] += vy[i] * scale;
}

// Update velocities given forces and a scale factor
__global__ void update_velocities_kernel(
    double* __restrict__ vx,
    double* __restrict__ vy,
    const double* __restrict__ fx,
    const double* __restrict__ fy,
    double scale)
{
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double scaled_mass_inv = scale / g_disk.mass[i];

    const double vxi = vx[i];
    const double vyi = vy[i];
    const double fxi = fx[i];
    const double fyi = fy[i];

    vx[i] = vxi + fxi * scaled_mass_inv;
    vy[i] = vyi + fyi * scaled_mass_inv;
}

// Compute the pairwise forces on the particles using the neighbor list
__global__ void compute_pair_forces_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
          double* __restrict__ fx,
          double* __restrict__ fy,
          double* __restrict__ pe)
{
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int sid = md::geo::g_sys.id[i];
    const double e_i = g_disk.e_interaction[sid];
    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];
    const double box_inv_x = md::geo::g_box.inv_x[sid];
    const double box_inv_y = md::geo::g_box.inv_y[sid];

    const double xi = x[i], yi = y[i];
    const double ri = g_disk.rad[i];
    double fxi = 0.0, fyi = 0.0, pei = 0.0;

    const int beg = md::geo::g_neigh.start[i];
    const int end = md::geo::g_neigh.start[i+1];

    for (int k = beg; k < end; ++k) {
        const int j = md::geo::g_neigh.ids[k];
        const double xj = x[j], yj = y[j];
        const double rj = g_disk.rad[j];

        double dx, dy;
        double r2 = md::geo::disp_pbc_L(xi, yi, xj, yj, box_size_x, box_size_y, box_inv_x, box_inv_y, dx, dy);

        // Early reject if no overlap: r^2 >= (ri+rj)^2
        const double radsum = ri + rj;
        const double radsum2 = radsum * radsum;
        if (r2 >= radsum2) continue;

        // Overlap: compute r and invr once
        const double r   = sqrt(r2);
        const double inv = 1.0 / r;
        const double nx  = dx * inv;
        const double ny  = dy * inv;

        const double delta = radsum - r;
        const double fmag  = e_i * delta;

        // Force on i is along -n (repulsion)
        fxi -= fmag * nx;
        fyi -= fmag * ny;

        // Single-count the pair energy (each pair gets half)
        pei += (0.5 * e_i * delta * delta) * 0.5;
    }
    
    fx[i] = fxi;
    fy[i] = fyi;
    pe[i] = pei;
}

// Compute the wall forces on the particles (MUST FOLLOW AFTER compute_pair_forces_kernel OR AN EQUIVALENT FORCE-ZEROING OPERATION!)
__global__ void compute_wall_forces_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ fx,
    double* __restrict__ fy,
    double* __restrict__ pe)
{
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int sid = md::geo::g_sys.id[i];
    const double e_i = g_disk.e_interaction[sid];
    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];

    const double xi = x[i], yi = y[i];
    const double ri = g_disk.rad[i];
    
    double fxi = 0.0, fyi = 0.0, pei = 0.0;

    if (xi < ri) {
        const double delta = ri - xi;
        const double fmag = e_i * delta;
        fxi -= fmag;
        pei += (0.5 * e_i * delta * delta) * 0.5;
    }
    if (xi > box_size_x - ri) {
        const double delta = ri - (box_size_x - xi);
        const double fmag = e_i * delta;
        fxi -= fmag;
        pei += (0.5 * e_i * delta * delta) * 0.5;
    }
    if (yi < ri) {
        const double delta = ri - yi;
        const double fmag = e_i * delta;
        fyi -= fmag;
        pei += (0.5 * e_i * delta * delta) * 0.5;
    }
    if (yi > box_size_y - ri) {
        const double delta = ri - (box_size_y - yi);
        const double fmag = e_i * delta;
        fyi -= fmag;
        pei += (0.5 * e_i * delta * delta) * 0.5;
    }

    fx[i] += fxi;
    fy[i] += fyi;
    pe[i] += pei;
}

// Compute the damping forces
__global__ void compute_damping_forces_kernel(
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    double* __restrict__ fx,
    double* __restrict__ fy,
    double scale
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    fx[i] += -vx[i] * scale;
    fy[i] += -vy[i] * scale;
}

// Compute the kinetic energy of each particle
__global__ void compute_ke_kernel(
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    double* __restrict__ ke
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    ke[i] = 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
}

} // namespace kernels

void Disk::compute_forces_impl() {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::compute_pair_forces_kernel, G, B,
        this->pos.xptr(), this->pos.yptr(),
        this->force.xptr(), this->force.yptr(),
        this->pe.ptr()
    );
}

void Disk::compute_wall_forces_impl() {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::compute_wall_forces_kernel, G, B,
        this->pos.xptr(), this->pos.yptr(),
        this->force.xptr(), this->force.yptr(),
        this->pe.ptr()
    );
}

void Disk::compute_damping_forces_impl(double scale) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::compute_damping_forces_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(),
        this->force.xptr(), this->force.yptr(),
        scale
    );
}

void Disk::update_positions_impl(double scale) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    switch (Base::neighbor_method) {
        case NeighborMethod::Cell:
            CUDA_LAUNCH(kernels::update_positions_kernel_cell, G, B,
                this->pos.xptr(), this->pos.yptr(),
                this->vel.xptr(), this->vel.yptr(),
                this->last_pos.xptr(), this->last_pos.yptr(),
                this->disp2.ptr(),
                scale
            );
            break;
        case NeighborMethod::Naive:
            CUDA_LAUNCH(kernels::update_positions_kernel_naive, G, B,
                this->pos.xptr(), this->pos.yptr(),
                this->vel.xptr(), this->vel.yptr(),
                scale
            );
            break;
    }
}

void Disk::update_velocities_impl(double scale) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::update_velocities_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(),
        this->force.xptr(), this->force.yptr(),
        scale
    );
}

void Disk::sync_class_constants_impl() {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(this->n_systems());
    CUDA_LAUNCH(kernels::calculate_thresh2_kernel, G, B,
        this->verlet_skin.ptr(),
        this->thresh2.ptr()
    );
    cudaDeviceSynchronize();
    bind_disk_globals(this->e_interaction.ptr(), this->mass.ptr(), this->rad.ptr(), this->rebuild_flag.ptr(), this->thresh2.ptr());
}

void Disk::reset_displacements_impl() {
    if (this->last_pos.size() != this->pos.size()) {
        throw std::runtime_error("Disk::reset_displacements_impl: last_pos and pos must have the same size (likely need to call allocate_particles after setting the neighbor method)");
    }
    last_pos.copy_from(this->pos);
    disp2.fill(0.0);
}

void Disk::reorder_particles_impl() {
    auto src = thrust::make_zip_iterator(
        thrust::make_tuple(
            this->pos.x.begin(), this->pos.y.begin(),
            this->vel.x.begin(), this->vel.y.begin(),
            this->force.x.begin(), this->force.y.begin(),
            this->rad.begin(),
            this->mass.begin(),
            this->cell_id.begin()
        )
    );
    auto dst = thrust::make_zip_iterator(
        thrust::make_tuple(
            this->pos.x.swap_begin(), this->pos.y.swap_begin(),
            this->vel.x.swap_begin(), this->vel.y.swap_begin(),
            this->force.x.swap_begin(), this->force.y.swap_begin(),
            this->rad.swap_begin(),
            this->mass.swap_begin(),
            this->cell_id.swap_begin()
        )
    );
    thrust::gather(this->order.begin(), this->order.end(), src, dst);
    this->pos.swap(); this->vel.swap(); this->force.swap(); this->rad.swap(); this->mass.swap(); this->cell_id.swap();
}

bool Disk::check_cell_neighbors_impl() {
    const int S = this->n_systems();
    if (S == 0) return false;
    // use thrust to check if any rebuild_flag is non-zero
    auto first = thrust::device_pointer_cast(this->rebuild_flag.ptr());
    // OR-reduce all flags; non-zero means rebuild
    unsigned int flags = thrust::reduce(thrust::device, first, first + S, 0u, thrust::bit_or<unsigned int>());
    if (flags == 0u) return false;
    // reset the rebuild flags
    this->rebuild_flag.fill(0u);
    return true;
}

void Disk::compute_ke_impl() {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::compute_ke_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(),
        this->ke.ptr()
    );
}

} // namespace md::disk