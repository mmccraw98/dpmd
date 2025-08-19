#include "particles/rigid_bumpy.cuh"
#include "kernels/common.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

namespace md::rigid_bumpy {

__constant__ RigidBumpyConst g_rigid_bumpy;

void bind_rigid_bumpy_globals(const double* d_e_interaction, const double* d_vertex_rad) {
    RigidBumpyConst h { d_e_interaction, d_vertex_rad };
    cudaMemcpyToSymbol(g_rigid_bumpy, &h, sizeof(RigidBumpyConst));
}

// RigidBumpy-specific kernels
namespace kernels {

// Compute the pairwise forces on the particles using the neighbor list
__global__ void compute_pair_forces_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
          double* __restrict__ fx,
          double* __restrict__ fy,
          double* __restrict__ pe)
{
    const int Nv = md::geo::g_sys.n_vertices;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nv) return;

    const int sid = md::poly::g_vertex_sys.id[i];
    const double e_i = g_rigid_bumpy.e_interaction[sid];
    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];
    const double box_inv_x = md::geo::g_box.inv_x[sid];
    const double box_inv_y = md::geo::g_box.inv_y[sid];

    const double xi = x[i], yi = y[i];
    const double ri = g_rigid_bumpy.vertex_rad[i];
    double fxi = 0.0, fyi = 0.0, pei = 0.0;

    const int beg = md::geo::g_neigh.start[i];
    const int end = md::geo::g_neigh.start[i+1];

    for (int k = beg; k < end; ++k) {
        const int j = md::geo::g_neigh.ids[k];
        const double xj = x[j], yj = y[j];
        const double rj = g_rigid_bumpy.vertex_rad[j];

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
    const int Nv = md::geo::g_sys.n_vertices;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nv) return;

    const int sid = md::geo::g_sys.id[i];
    const double e_i = g_rigid_bumpy.e_interaction[sid];
    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];

    const double xi = x[i], yi = y[i];
    const double ri = g_rigid_bumpy.vertex_rad[i];
    
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

// Compute the forces on the particles
__global__ void compute_particle_forces_kernel(
    const double* __restrict__ vertex_force_x,
    const double* __restrict__ vertex_force_y,
    const double* __restrict__ vertex_pe,
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ pe
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Sum over all vertices
    double fxi = 0.0, fyi = 0.0, pei = 0.0;
    int beg = md::poly::g_poly.particle_offset[i];
    int end = md::poly::g_poly.particle_offset[i+1];
    for (int j = beg; j < end; ++j) {
        fxi += vertex_force_x[j];
        fyi += vertex_force_y[j];
        pei += vertex_pe[j];
    }

    force_x[i] = fxi;
    force_y[i] = fyi;
    pe[i] = pei;
}

} // namespace kernels

void RigidBumpy::compute_particle_forces() {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::compute_particle_forces_kernel, G, B,
        this->vertex_force.xptr(), this->vertex_force.yptr(), this->vertex_pe.ptr(),
        this->force.xptr(), this->force.yptr(), this->pe.ptr()
    );
}

void RigidBumpy::compute_forces_impl() {
    const int Nv = n_vertices();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(Nv);
    CUDA_LAUNCH(kernels::compute_pair_forces_kernel, G, B,
        this->vertex_pos.xptr(), this->vertex_pos.yptr(),
        this->vertex_force.xptr(), this->vertex_force.yptr(),
        this->vertex_pe.ptr()
    );
}

void RigidBumpy::compute_wall_forces_impl() {
    const int Nv = n_vertices();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(Nv);
    CUDA_LAUNCH(kernels::compute_wall_forces_kernel, G, B,
        this->vertex_pos.xptr(), this->vertex_pos.yptr(),
        this->vertex_force.xptr(), this->vertex_force.yptr(),
        this->vertex_pe.ptr()
    );
}

void RigidBumpy::compute_damping_forces_impl(double scale) {
    throw std::runtime_error("RigidBumpy::compute_damping_forces_impl: not implemented");
}

void RigidBumpy::update_positions_impl(double scale) {
    throw std::runtime_error("RigidBumpy::update_positions_impl: not implemented");
}

void RigidBumpy::update_velocities_impl(double scale) {
    throw std::runtime_error("RigidBumpy::update_velocities_impl: not implemented");
}

void RigidBumpy::sync_class_constants_poly_extras_impl() {
    bind_rigid_bumpy_globals(this->e_interaction.ptr(), this->vertex_rad.ptr());
}

void RigidBumpy::reset_displacements_impl() {
    throw std::runtime_error("RigidBumpy::reset_displacements_impl: not implemented");
}

void RigidBumpy::reorder_particles_impl() {
    throw std::runtime_error("RigidBumpy::reorder_particles_impl: not implemented");
}

bool RigidBumpy::check_cell_neighbors_impl() {
    throw std::runtime_error("RigidBumpy::check_cell_neighbors_impl: not implemented");
}

void RigidBumpy::compute_ke_impl() {
    throw std::runtime_error("RigidBumpy::compute_ke_impl: not implemented");
}

void RigidBumpy::allocate_poly_extras_impl(int N) {
    // throw std::runtime_error("RigidBumpy::allocate_poly_extras_impl: not implemented");
    std::cout << "RigidBumpy::allocate_poly_extras_impl: not implemented\n";
}

void RigidBumpy::allocate_poly_vertex_extras_impl(int Nv) {
    // throw std::runtime_error("RigidBumpy::allocate_poly_vertex_extras_impl: not implemented");
    std::cout << "RigidBumpy::allocate_poly_vertex_extras_impl: not implemented\n";
}

void RigidBumpy::allocate_poly_system_extras_impl(int S) {
    // throw std::runtime_error("RigidBumpy::allocate_poly_system_extras_impl: not implemented");
    std::cout << "RigidBumpy::allocate_poly_system_extras_impl: not implemented\n";
}

void RigidBumpy::enable_poly_swap_extras_impl(bool enable) {
    // throw std::runtime_error("RigidBumpy::enable_poly_swap_extras_impl: not implemented");
    std::cout << "RigidBumpy::enable_poly_swap_extras_impl: not implemented\n";
}

} // namespace md::rigid_bumpy