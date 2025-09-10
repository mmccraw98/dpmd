#include "particles/disk.cuh"
#include "kernels/base_particle_kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

namespace md::disk {

__constant__ DiskConst g_disk;

void bind_disk_globals(const double* d_e_interaction, const double* d_mass, const double* d_rad, unsigned int* d_rebuild_flag) {
    DiskConst h { d_e_interaction, d_mass, d_rad, d_rebuild_flag };
    cudaMemcpyToSymbol(g_disk, &h, sizeof(DiskConst));
}

// Disk-specific kernels
namespace kernels {

// Update positions and displacements given velocities and a scale factor - used for cell neighbor list
__global__ void update_positions_kernel_cell(
    double* __restrict__ x,
    double* __restrict__ y,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ last_x,
    const double* __restrict__ last_y,
    double* __restrict__ disp2,
    const double* __restrict__ scale,
    const double scale2
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    const double s = scale[sid] * scale2;

    double pos_x = x[i];
    double pos_y = y[i];
    const double vel_x = vx[i];
    const double vel_y = vy[i];

    // Update positions
    pos_x += vel_x * s;
    pos_y += vel_y * s;
    x[i] = pos_x;
    y[i] = pos_y;

    // Calculate squared displacement for neighbor list update
    double dx = pos_x - last_x[i];
    double dy = pos_y - last_y[i];
    double d2 = dx * dx + dy * dy;
    disp2[i] = d2;

    // Determine per-system rebuild flag
    const double thresh2 = md::geo::g_neigh.thresh2[sid];
    if (d2 > thresh2) atomicOr(&g_disk.rebuild_flag[sid], 1u);
}

// Update positions and displacements given velocities and a scale factor - used for naive neighbor list
__global__ void update_positions_kernel_naive(
    double* __restrict__ x,
    double* __restrict__ y,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ scale,
    const double scale2
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    const double s = scale[sid] * scale2;
    x[i] += vx[i] * s;
    y[i] += vy[i] * s;
}

// Update velocities given forces and a scale factor
__global__ void update_velocities_kernel(
    double* __restrict__ vx,
    double* __restrict__ vy,
    const double* __restrict__ fx,
    const double* __restrict__ fy,
    const double* __restrict__ scale,
    const double scale2)
{
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];

    const double scaled_mass_inv = scale[sid] * scale2 / g_disk.mass[i];

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

// Compute the forces on the particles due to the walls and the interactions with their neighbors
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

    // compute wall forces - do not divide pe by 2 here since it is only given to one particle
    if (xi < ri) {
        const double delta = ri - xi;
        const double fmag = e_i * delta;
        fxi += fmag;
        pei += (0.5 * e_i * delta * delta);
    }
    if (xi > box_size_x - ri) {
        const double delta = ri - (box_size_x - xi);
        const double fmag = e_i * delta;
        fxi -= fmag;
        pei += (0.5 * e_i * delta * delta);
    }
    if (yi < ri) {
        const double delta = ri - yi;
        const double fmag = e_i * delta;
        fyi += fmag;
        pei += (0.5 * e_i * delta * delta);
    }
    if (yi > box_size_y - ri) {
        const double delta = ri - (box_size_y - yi);
        const double fmag = e_i * delta;
        fyi -= fmag;
        pei += (0.5 * e_i * delta * delta);
    }

    // compute pair forces
    const int beg = md::geo::g_neigh.start[i];
    const int end = md::geo::g_neigh.start[i+1];
    for (int k = beg; k < end; ++k) {
        const int j = md::geo::g_neigh.ids[k];
        const double xj = x[j], yj = y[j];
        const double rj = g_disk.rad[j];

        double dx = xj - xi;
        double dy = yj - yi;
        double r2 = dx * dx + dy * dy;

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

// Compute the damping forces
__global__ void compute_damping_forces_kernel(
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    double* __restrict__ fx,
    double* __restrict__ fy,
    const double* __restrict__ scale
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    const double s = scale[sid];
    fx[i] += -vx[i] * s;
    fy[i] += -vy[i] * s;
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
    ke[i] = 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]) * g_disk.mass[i];
}

// Scale the velocities of the particles in each system
__global__ void scale_velocities_kernel(
    double* __restrict__ vx,
    double* __restrict__ vy,
    const double* __restrict__ scale
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    vx[i] *= scale[sid];
    vy[i] *= scale[sid];
}

// Scale the positions of the particles in each system
__global__ void scale_positions_kernel(
    double* __restrict__ x,
    double* __restrict__ y,
    const double* __restrict__ scale
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    x[i] *= scale[sid];
    y[i] *= scale[sid];
}


// Mix velocities and forces - system-level alpha, primarily used for FIRE
__global__ void mix_velocities_and_forces_kernel(
    double* __restrict__ vx,
    double* __restrict__ vy,
    const double* __restrict__ fx,
    const double* __restrict__ fy,
    const double* __restrict__ alpha
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    const double a = alpha[sid];
    double vxi = vx[i];
    double vyi = vy[i];
    double fxi = fx[i];
    double fyi = fy[i];
    double force_norm = sqrt(fxi * fxi + fyi * fyi);
    double vel_norm = sqrt(vxi * vxi + vyi * vyi);
    double mixing_ratio = 0.0;
    if (force_norm > 1e-16) {
        mixing_ratio = vel_norm / force_norm * a;
    } else {
        vxi = 0.0;
        vyi = 0.0;
        mixing_ratio = 0.0;
    }
    vx[i] = vxi * (1 - a) + fxi * mixing_ratio;
    vy[i] = vyi * (1 - a) + fyi * mixing_ratio;
}

// Functor to compute power: force_x * vel_x + force_y * vel_y
struct PowerFunctor {
    __device__ double operator()(const thrust::tuple<double, double, double, double>& t) const {
        double fx = thrust::get<0>(t);
        double vx = thrust::get<1>(t);
        double fy = thrust::get<2>(t);
        double vy = thrust::get<3>(t);
        return fx * vx + fy * vy;
    }
};

__global__ void save_particle_state_kernel(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ last_x,
    double* __restrict__ last_y,
    double* __restrict__ rad,
    double* __restrict__ last_rad,
    double* __restrict__ mass,
    double* __restrict__ last_mass,
    int* __restrict__ flag,
    int true_val
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = md::geo::g_sys.n_particles;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    if (flag[sid] != true_val) return;
    last_x[i] = x[i];
    last_y[i] = y[i];
    last_rad[i] = rad[i];
    last_mass[i] = mass[i];
}

__global__ void save_system_state_kernel(
    double* __restrict__ box_size_x,
    double* __restrict__ box_size_y,
    double* __restrict__ last_box_size_x,
    double* __restrict__ last_box_size_y,
    int* __restrict__ flag,
    int true_val
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (i >= S) return;
    const int sid = md::geo::g_sys.id[i];
    if (flag[sid] != true_val) return;
    last_box_size_x[i] = box_size_x[i];
    last_box_size_y[i] = box_size_y[i];
}

__global__ void restore_particle_state_kernel(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ last_x,
    double* __restrict__ last_y,
    double* __restrict__ rad,
    double* __restrict__ last_rad,
    double* __restrict__ mass,
    double* __restrict__ last_mass,
    int* __restrict__ flag,
    int true_val
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = md::geo::g_sys.n_particles;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    if (flag[sid] != true_val) return;
    x[i] = last_x[i];
    y[i] = last_y[i];
    rad[i] = last_rad[i];
    mass[i] = last_mass[i];
}

__global__ void restore_system_state_kernel(
    double* __restrict__ box_size_x,
    double* __restrict__ box_size_y,
    double* __restrict__ last_box_size_x,
    double* __restrict__ last_box_size_y,
    int* __restrict__ flag,
    int true_val
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (i >= S) return;
    const int sid = md::geo::g_sys.id[i];
    if (flag[sid] != true_val) return;
    box_size_x[i] = last_box_size_x[i];
    box_size_y[i] = last_box_size_y[i];
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

void Disk::compute_damping_forces_impl(df::DeviceField1D<double> scale) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::compute_damping_forces_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(),
        this->force.xptr(), this->force.yptr(),
        scale.ptr()
    );
}

void Disk::update_positions_impl(df::DeviceField1D<double> scale, double scale2) {
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
                scale.ptr(), scale2
            );
            break;
        case NeighborMethod::Naive:
            CUDA_LAUNCH(kernels::update_positions_kernel_naive, G, B,
                this->pos.xptr(), this->pos.yptr(),
                this->vel.xptr(), this->vel.yptr(),
                scale.ptr(), scale2
            );
            break;
    }
}

void Disk::update_velocities_impl(df::DeviceField1D<double> scale, double scale2) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::update_velocities_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(),
        this->force.xptr(), this->force.yptr(),
        scale.ptr(), scale2
    );
}

void Disk::scale_velocities_impl(df::DeviceField1D<double> scale) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::scale_velocities_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(),
        scale.ptr()
    );
}

void Disk::scale_positions_impl(df::DeviceField1D<double> scale) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::scale_positions_kernel, G, B,
        this->pos.xptr(), this->pos.yptr(),
        scale.ptr()
    );
}

void Disk::mix_velocities_and_forces_impl(df::DeviceField1D<double> alpha) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::mix_velocities_and_forces_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(),
        this->force.xptr(), this->force.yptr(),
        alpha.ptr()
    );
}

void Disk::compute_fpower_total_impl() {
    cudaStream_t stream = 0;
    const int S = this->n_systems();
    if (this->fpower_total.size() != S) {
        this->fpower_total.resize(S);
    }

    void* d_temp = this->cub_sys_agg.ptr();
    size_t temp_bytes = this->cub_sys_agg.size();

    // Create transform iterator that computes force·velocity on-the-fly
    auto input_iter = thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(
            this->force.xptr(),
            this->vel.xptr(),
            this->force.yptr(),
            this->vel.yptr()
        )),
        kernels::PowerFunctor()
    );

    // 1) size request
    cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_bytes,
        input_iter, this->fpower_total.ptr(),
        S,
        this->system_offset.ptr(),
        this->system_offset.ptr() + 1,
        stream);

    // 2) ensure workspace
    if (temp_bytes > static_cast<size_t>(this->cub_sys_agg.size())) {
        this->cub_sys_agg.resize(static_cast<int>(temp_bytes));
        d_temp = this->cub_sys_agg.ptr();
    }

    // 3) run
    cub::DeviceSegmentedReduce::Sum(
        d_temp, temp_bytes,
        input_iter, this->fpower_total.ptr(),
        S,
        this->system_offset.ptr(),
        this->system_offset.ptr() + 1,
        stream);
}


void Disk::sync_class_constants_impl() {
    bind_disk_globals(this->e_interaction.ptr(), this->mass.ptr(), this->rad.ptr(), this->rebuild_flag.ptr());
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

void Disk::set_random_positions_impl(double box_pad_x, double box_pad_y) {
    throw std::runtime_error("Disk::set_random_positions_impl: not implemented");
}

void Disk::save_state_impl(df::DeviceField1D<int> flag, int true_val) {
    if (this->last_state_pos.size() != this->pos.size()) {
        this->last_state_pos.resize(this->pos.size());
    }
    if (this->last_state_rad.size() != this->rad.size()) {
        this->last_state_rad.resize(this->rad.size());
    }
    if (this->last_state_mass.size() != this->mass.size()) {
        this->last_state_mass.resize(this->mass.size());
    }
    if (this->last_state_box_size.size() != this->box_size.size()) {
        this->last_state_box_size.resize(this->box_size.size());
    }
    
    const int N = n_particles();
    const int S = n_systems();
    auto B = md::launch::threads_for();
    auto G_N = md::launch::blocks_for(N);
    auto G_S = md::launch::blocks_for(S);
    CUDA_LAUNCH(kernels::save_particle_state_kernel, G_N, B,
        this->pos.xptr(), this->pos.yptr(), this->last_state_pos.xptr(), this->last_state_pos.yptr(),
        this->rad.ptr(), this->last_state_rad.ptr(),
        this->mass.ptr(), this->last_state_mass.ptr(),
        flag.ptr(), true_val
    );
    CUDA_LAUNCH(kernels::save_system_state_kernel, G_S, B,
        this->box_size.xptr(), this->box_size.yptr(), this->last_state_box_size.xptr(), this->last_state_box_size.yptr(),
        flag.ptr(), true_val
    );
}

void Disk::restore_state_impl(df::DeviceField1D<int> flag, int true_val) {
    if (this->last_state_pos.size() != this->pos.size()) {
        throw std::runtime_error("Disk::restore_state_impl: last_state_pos is not initialized");
    }
    if (this->last_state_rad.size() != this->rad.size()) {
        throw std::runtime_error("Disk::restore_state_impl: last_state_rad is not initialized");
    }
    if (this->last_state_mass.size() != this->mass.size()) {
        throw std::runtime_error("Disk::restore_state_impl: last_state_mass is not initialized");
    }
    if (this->last_state_box_size.size() != this->box_size.size()) {
        throw std::runtime_error("Disk::restore_state_impl: last_state_box_size is not initialized");
    }
    const int N = n_particles();
    const int S = n_systems();
    auto B = md::launch::threads_for();
    auto G_N = md::launch::blocks_for(N);
    auto G_S = md::launch::blocks_for(S);
    CUDA_LAUNCH(kernels::restore_particle_state_kernel, G_N, B,
        this->pos.xptr(), this->pos.yptr(), this->last_state_pos.xptr(), this->last_state_pos.yptr(),
        this->rad.ptr(), this->last_state_rad.ptr(),
        this->mass.ptr(), this->last_state_mass.ptr(),
        flag.ptr(), true_val
    );
    CUDA_LAUNCH(kernels::restore_system_state_kernel, G_S, B,
        this->box_size.xptr(), this->box_size.yptr(), this->last_state_box_size.xptr(), this->last_state_box_size.yptr(),
        flag.ptr(), true_val
    );
    Base::sync_box();
    Base::sync_class_constants();
    Base::check_neighbors();
}

void Disk::load_static_from_hdf5_point_extras_impl(hid_t group) {
    // nothing to do
}

void Disk::load_from_hdf5_point_extras_impl(hid_t group) {
    // nothing to do
}

std::string Disk::get_class_name_impl() {
    return "Disk";
}

std::vector<std::string> Disk::get_static_field_names_point_extras_impl() {
    return {};  // nothing extra for disks
}

std::vector<std::string> Disk::get_state_field_names_point_extras_impl() {
    return {};  // nothing extra for disks
}

void Disk::output_build_registry_point_extras_impl(io::OutputRegistry& reg) {
    // nothing extra for disks
}

} // namespace md::disk