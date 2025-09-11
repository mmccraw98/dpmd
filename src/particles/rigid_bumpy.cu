#include "particles/rigid_bumpy.cuh"
#include "kernels/base_particle_kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

namespace md::rigid_bumpy {

__constant__ RigidBumpyConst g_rigid_bumpy;

void bind_rigid_bumpy_globals(const double* d_e_interaction, const double* d_vertex_rad, const double* d_mass, const double* d_moment_inertia) {
    RigidBumpyConst h { d_e_interaction, d_vertex_rad, d_mass, d_moment_inertia };
    cudaMemcpyToSymbol(g_rigid_bumpy, &h, sizeof(RigidBumpyConst));
}

// RigidBumpy-specific kernels
namespace kernels {

// Update the positions of the particles and their vertices
__global__ void update_positions_kernel(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ theta,
    double* __restrict__ vertex_x,
    double* __restrict__ vertex_y,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vtheta,
    const double* __restrict__ scale,
    const double scale2
)
{
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int sid = md::geo::g_sys.id[i];
    const double sc = scale[sid] * scale2;
    const double x_i = x[i];
    const double y_i = y[i];
    const double angle_i = theta[i];

    double new_x_i = x_i + vx[i] * sc;
    double new_y_i = y_i + vy[i] * sc;
    double new_angle_i = angle_i + vtheta[i] * sc;

    // Update the position of the particle
    x[i] = new_x_i;
    y[i] = new_y_i;
    theta[i] = new_angle_i;

    // Update the position of the vertices
    const int beg = md::poly::g_poly.particle_offset[i];
    const int end = md::poly::g_poly.particle_offset[i+1];
    double s, c; sincos(new_angle_i - angle_i, &s, &c);
    for (int j = beg; j < end; ++j) {
        double dx = vertex_x[j] - x_i;
        double dy = vertex_y[j] - y_i;
        double rx = c*dx - s*dy;
        double ry = s*dx + c*dy;
        vertex_x[j] = new_x_i + rx;
        vertex_y[j] = new_y_i + ry;
    }
}

// Update the velocities of the particles
__global__ void update_velocities_kernel(
    double* __restrict__ vx,
    double* __restrict__ vy,
    double* __restrict__ vtheta,
    const double* __restrict__ force_x,
    const double* __restrict__ force_y,
    const double* __restrict__ torque,
    const double* __restrict__ scale,
    const double scale2
)
{
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double inertia_i = g_rigid_bumpy.moment_inertia[i];
    const int sid = md::geo::g_sys.id[i];
    const double sc = scale[sid] * scale2;
    const double scaled_mass_inv = sc / g_rigid_bumpy.mass[i];
    const double scaled_inertia_inv = inertia_i != 0.0 ? sc / inertia_i : 0.0;

    const double vxi = vx[i];
    const double vyi = vy[i];
    const double vthetai = vtheta[i];
    const double fxi = force_x[i];
    const double fyi = force_y[i];
    const double tqi = torque[i];

    vx[i] = vxi + fxi * scaled_mass_inv;
    vy[i] = vyi + fyi * scaled_mass_inv;
    vtheta[i] = vthetai + tqi * scaled_inertia_inv;
}

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

    const int v_sid = md::poly::g_vertex_sys.id[i];
    const double e_i = g_rigid_bumpy.e_interaction[v_sid];
    const double box_size_x = md::geo::g_box.size_x[v_sid];
    const double box_size_y = md::geo::g_box.size_y[v_sid];
    const double box_inv_x = md::geo::g_box.inv_x[v_sid];
    const double box_inv_y = md::geo::g_box.inv_y[v_sid];

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

// Compute the forces on the particles due to the walls and the interactions with their neighbors
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

    const int v_sid = md::poly::g_vertex_sys.id[i];
    const double e_i = g_rigid_bumpy.e_interaction[v_sid];
    const double box_size_x = md::geo::g_box.size_x[v_sid];
    const double box_size_y = md::geo::g_box.size_y[v_sid];

    const double xi = x[i], yi = y[i];
    const double ri = g_rigid_bumpy.vertex_rad[i];
    
    double fxi = 0.0, fyi = 0.0, pei = 0.0;

    // compute the wall forces - do not divide pe by 2 here since it is only given to one particle
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

    const int beg = md::geo::g_neigh.start[i];
    const int end = md::geo::g_neigh.start[i+1];

    for (int k = beg; k < end; ++k) {
        const int j = md::geo::g_neigh.ids[k];
        const double xj = x[j], yj = y[j];
        const double rj = g_rigid_bumpy.vertex_rad[j];

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

// Compute the damping forces on the particles
__global__ void compute_damping_forces_kernel(
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vtheta,
    double* __restrict__ fx,
    double* __restrict__ fy,
    double* __restrict__ torque,
    const double* __restrict__ scale
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int sid = md::geo::g_sys.id[i];
    const double sc = scale[sid];
    fx[i] += -vx[i] * sc;
    fy[i] += -vy[i] * sc;
    torque[i] += -vtheta[i] * sc;
}

// Compute the forces on the particles
__global__ void compute_particle_forces_kernel(
    const double* __restrict__ vertex_force_x,
    const double* __restrict__ vertex_force_y,
    const double* __restrict__ vertex_pe,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ px,
    const double* __restrict__ py,
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    double* __restrict__ torque,
    double* __restrict__ pe
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Sum over all vertices
    const double px_i = px[i], py_i = py[i];
    double fxi = 0.0, fyi = 0.0, tqi = 0.0, pei = 0.0;
    double dx, dy, vertex_force_x_i, vertex_force_y_i;
    int beg = md::poly::g_poly.particle_offset[i];
    int end = md::poly::g_poly.particle_offset[i+1];
    for (int j = beg; j < end; ++j) {
        vertex_force_x_i = vertex_force_x[j];
        vertex_force_y_i = vertex_force_y[j];
        fxi += vertex_force_x_i;
        fyi += vertex_force_y_i;
        pei += vertex_pe[j];
        dx = vx[j] - px_i;
        dy = vy[j] - py_i;
        tqi += vertex_force_y_i * dx - vertex_force_x_i * dy;
    }

    force_x[i] = fxi;
    force_y[i] = fyi;
    torque[i] = tqi;
    pe[i] = pei;
}

// Set random positions within the box with padding
// Angle is chosen between 0 and 2pi/Nv to maintain periodicity
__global__ void set_random_positions_in_box_kernel(
    curandStatePhilox4_32_10_t* __restrict__ states,
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ angle,
    double* __restrict__ vertex_pos_x,
    double* __restrict__ vertex_pos_y,
    const double box_pad_x,
    const double box_pad_y
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandStatePhilox4_32_10_t st = states[i];

    const int n_vertices_per_particle = md::poly::g_poly.n_vertices_per_particle[i];
    const int particle_offset = md::poly::g_poly.particle_offset[i];
    const int sid = md::geo::g_sys.id[i];
    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];

    double pos_xi = pos_x[i];
    double pos_yi = pos_y[i];
    double angle_i = angle[i];
    
    // Generate new random position and angle for the particle
    double new_pos_xi = curand_uniform_double(&st) * (box_size_x - 2 * box_pad_x) + box_pad_x;
    double new_pos_yi = curand_uniform_double(&st) * (box_size_y - 2 * box_pad_y) + box_pad_y;
    double angle_period_inv = (n_vertices_per_particle > 1) ? 1.0 / n_vertices_per_particle : 0.0;
    double new_angle_i = curand_uniform_double(&st) * 2 * M_PI * angle_period_inv;

    // Displace and rotate all vertices
    double s, c; sincos(new_angle_i - angle_i, &s, &c);
    for (int j = 0; j < n_vertices_per_particle; ++j) {
        double dx = vertex_pos_x[particle_offset + j] - pos_xi;
        double dy = vertex_pos_y[particle_offset + j] - pos_yi;
        double rx = c*dx - s*dy;
        double ry = s*dx + c*dy;
        vertex_pos_x[particle_offset + j] = new_pos_xi + rx;
        vertex_pos_y[particle_offset + j] = new_pos_yi + ry;
    }
    
    pos_x[i] = new_pos_xi;
    pos_y[i] = new_pos_yi;
    angle[i] = new_angle_i;

    states[i] = st;
}

// Compute the kinetic energy for each particle
__global__ void compute_ke_kernel(
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vtheta,
    double* __restrict__ ke
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double vxi = vx[i];
    const double vyi = vy[i];
    const double vthetai = vtheta[i];
    const double m = g_rigid_bumpy.mass[i];
    const double I = g_rigid_bumpy.moment_inertia[i];
    const double kei = 0.5 * ((vxi * vxi + vyi * vyi) * m + vthetai * vthetai * I);
    ke[i] = kei;
}

// Functor to compute power: force_x * vel_x + force_y * vel_y + torque * ang_vel
struct PowerFunctor {
    __device__ double operator()(const thrust::tuple<double, double, double, double, double, double>& t) const {
        double fx = thrust::get<0>(t);
        double vx = thrust::get<1>(t);
        double fy = thrust::get<2>(t);
        double vy = thrust::get<3>(t);
        double torque = thrust::get<4>(t);
        double ang_vel = thrust::get<5>(t);
        return fx * vx + fy * vy + torque * ang_vel;
    }
};

// Scale the velocities of the particles
__global__ void scale_velocities_kernel(
    double* __restrict__ vx,
    double* __restrict__ vy,
    double* __restrict__ vtheta,
    const double* __restrict__ scale
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    vx[i] *= scale[sid];
    vy[i] *= scale[sid];
    vtheta[i] *= scale[sid];
}

__global__ void scale_positions_kernel(
    double* __restrict__ x, double* __restrict__ y,
    double* __restrict__ vertex_x, double* __restrict__ vertex_y,
    const double* __restrict__ scale_factor
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = md::geo::g_sys.n_particles;
    if (i >= N) return;

    const int sid = md::geo::g_sys.id[i];

    const int beg = md::poly::g_poly.particle_offset[i];
    const int end = md::poly::g_poly.particle_offset[i+1];

    double pos_x = x[i];
    double pos_y = y[i];

    double new_pos_x = pos_x * scale_factor[sid];
    double new_pos_y = pos_y * scale_factor[sid];

    x[i] = new_pos_x;
    y[i] = new_pos_y;

    for (int j = beg; j < end; j++) {
        vertex_x[j] += (new_pos_x - pos_x);
        vertex_y[j] += (new_pos_y - pos_y);
    }
}

// Mix velocities and forces - system-level alpha, primarily used for FIRE
__global__ void mix_velocities_and_forces_kernel(
    double* __restrict__ vx,
    double* __restrict__ vy,
    double* __restrict__ vtheta,
    const double* __restrict__ force_x,
    const double* __restrict__ force_y,
    const double* __restrict__ torque,
    const double* __restrict__ alpha
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    const int sid = md::geo::g_sys.id[i];
    const double a = alpha[sid];
    if (a == 0.0) { return; }
    double vxi = vx[i];
    double vyi = vy[i];
    double vthetai = vtheta[i];
    double fxi = force_x[i];
    double fyi = force_y[i];
    double torquei = torque[i];
    double force_norm = sqrt(fxi * fxi + fyi * fyi);
    double vel_norm = sqrt(vxi * vxi + vyi * vyi);
    double mixing_ratio = 0.0;
    if (force_norm > 1e-16) {
        mixing_ratio = vel_norm / force_norm * a;
    } else {
        vxi = 0.0;
        vyi = 0.0;
    }
    double torque_norm = fabs(torquei);
    double ang_vel_norm = fabs(vthetai);
    double torque_mixing_ratio = 0.0;
    if (torque_norm > 1e-16) {
        torque_mixing_ratio = ang_vel_norm / torque_norm * a;
    } else {
        vthetai = 0.0;
        torque_mixing_ratio = 0.0;
    }
    vx[i] = vxi * (1 - a) + fxi * mixing_ratio;
    vy[i] = vyi * (1 - a) + fyi * mixing_ratio;
    vtheta[i] = vthetai * (1 - a) + torquei * torque_mixing_ratio;
}


__global__ void save_particle_state_kernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y, double* __restrict__ last_pos_x, double* __restrict__ last_pos_y,
    const double* __restrict__ angle, double* __restrict__ last_angle,
    const double* __restrict__ mass, double* __restrict__ last_mass,
    const double* __restrict__ moment_inertia, double* __restrict__ last_moment_inertia,
    const int* __restrict__ n_vertices_per_particle, int* __restrict__ last_n_vertices_per_particle,
    const int* __restrict__ flag, const int true_val
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = md::geo::g_sys.n_particles;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    if (flag[sid] == true_val) {
        last_pos_x[i] = pos_x[i];
        last_pos_y[i] = pos_y[i];
        last_angle[i] = angle[i];
        last_mass[i] = mass[i];
        last_moment_inertia[i] = moment_inertia[i];
        last_n_vertices_per_particle[i] = n_vertices_per_particle[i];
    }
}

__global__ void save_vertex_state_kernel(
    const double* __restrict__ vertex_rad, double* __restrict__ last_vertex_rad,
    const double* __restrict__ vertex_pos_x, const double* __restrict__ vertex_pos_y, double* __restrict__ last_vertex_pos_x, double* __restrict__ last_vertex_pos_y,
    const int* __restrict__ vertex_particle_id, int* __restrict__ last_vertex_particle_id,
    const int* __restrict__ flag, const int true_val
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Nv = md::geo::g_sys.n_vertices;
    if (i >= Nv) return;
    const int v_sid = md::poly::g_vertex_sys.id[i];
    if (flag[v_sid] == true_val) {
        last_vertex_rad[i] = vertex_rad[i];
        last_vertex_pos_x[i] = vertex_pos_x[i];
        last_vertex_pos_y[i] = vertex_pos_y[i];
        last_vertex_particle_id[i] = vertex_particle_id[i];
    }
}

__global__ void save_system_state_kernel(
    const double* __restrict__ box_size_x, const double* __restrict__ box_size_y, double* __restrict__ last_box_size_x, double* __restrict__ last_box_size_y,
    const int* __restrict__ flag, const int true_val
) {
    const int sid = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (sid >= S) return;
    if (flag[sid] == true_val) {
        last_box_size_x[sid] = box_size_x[sid];
        last_box_size_y[sid] = box_size_y[sid];
    }
}

__global__ void restore_particle_state_kernel(
    double* __restrict__ pos_x, double* __restrict__ pos_y, const double* __restrict__ last_pos_x, const double* __restrict__ last_pos_y,
    double* __restrict__ angle, const double* __restrict__ last_angle,
    double* __restrict__ mass, const double* __restrict__ last_mass,
    double* __restrict__ moment_inertia, const double* __restrict__ last_moment_inertia,
    int* __restrict__ n_vertices_per_particle, const int* __restrict__ last_n_vertices_per_particle,
    const int* __restrict__ flag, const int true_val
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = md::geo::g_sys.n_particles;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    if (flag[sid] == true_val) {
        pos_x[i] = last_pos_x[i];
        pos_y[i] = last_pos_y[i];
        angle[i] = last_angle[i];
        mass[i] = last_mass[i];
        moment_inertia[i] = last_moment_inertia[i];
        n_vertices_per_particle[i] = last_n_vertices_per_particle[i];
    }
}

__global__ void restore_vertex_state_kernel(
    double* __restrict__ vertex_rad, const double* __restrict__ last_vertex_rad,
    double* __restrict__ vertex_pos_x, const double* __restrict__ last_vertex_pos_x,
    double* __restrict__ vertex_pos_y, const double* __restrict__ last_vertex_pos_y,
    int* __restrict__ vertex_particle_id, const int* __restrict__ last_vertex_particle_id,
    const int* __restrict__ flag, const int true_val
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Nv = md::geo::g_sys.n_vertices;
    if (i >= Nv) return;
    const int v_sid = md::poly::g_vertex_sys.id[i]; 
    if (flag[v_sid] == true_val) {
        vertex_rad[i] = last_vertex_rad[i];
        vertex_pos_x[i] = last_vertex_pos_x[i];
        vertex_pos_y[i] = last_vertex_pos_y[i];
        vertex_particle_id[i] = last_vertex_particle_id[i];
    }
}

__global__ void restore_system_state_kernel(
    double* __restrict__ box_size_x, double* __restrict__ box_size_y, const double* __restrict__ last_box_size_x, const double* __restrict__ last_box_size_y,
    const int* __restrict__ flag, const int true_val
) {
    const int sid = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (sid >= S) return;
    if (flag[sid] == true_val) {
        box_size_x[sid] = last_box_size_x[sid];
        box_size_y[sid] = last_box_size_y[sid];
    }
}


} // namespace kernels

void RigidBumpy::compute_particle_forces() {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::compute_particle_forces_kernel, G, B,
        this->vertex_force.xptr(), this->vertex_force.yptr(), this->vertex_pe.ptr(),
        this->vertex_pos.xptr(), this->vertex_pos.yptr(), this->pos.xptr(), this->pos.yptr(),
        this->force.xptr(), this->force.yptr(), this->torque.ptr(), this->pe.ptr()
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

    // sum up the vertex forces and potential energies to the particle level
    this->compute_particle_forces();
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

    // sum up the vertex forces and potential energies to the particle level
    this->compute_particle_forces();
}

void RigidBumpy::compute_damping_forces_impl(df::DeviceField1D<double> scale) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::compute_damping_forces_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(), this->angular_vel.ptr(),
        this->force.xptr(), this->force.yptr(), this->torque.ptr(), scale.ptr());
}

void RigidBumpy::update_positions_impl(df::DeviceField1D<double> scale, double scale2) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::update_positions_kernel, G, B,
        this->pos.xptr(), this->pos.yptr(), this->angle.ptr(),
        this->vertex_pos.xptr(), this->vertex_pos.yptr(),
        this->vel.xptr(), this->vel.yptr(), this->angular_vel.ptr(), scale.ptr(), scale2);
}

void RigidBumpy::update_velocities_impl(df::DeviceField1D<double> scale, double scale2) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::update_velocities_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(), this->angular_vel.ptr(),
        this->force.xptr(), this->force.yptr(), this->torque.ptr(), scale.ptr(), scale2);
}

void RigidBumpy::scale_velocities_impl(df::DeviceField1D<double> scale) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::scale_velocities_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(), this->angular_vel.ptr(), scale.ptr());
}

void RigidBumpy::scale_positions_impl(df::DeviceField1D<double> scale) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::scale_positions_kernel, G, B,
        this->pos.xptr(), this->pos.yptr(), this->vertex_pos.xptr(), this->vertex_pos.yptr(), scale.ptr());
}

void RigidBumpy::mix_velocities_and_forces_impl(df::DeviceField1D<double> alpha) {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::mix_velocities_and_forces_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(), this->angular_vel.ptr(), this->force.xptr(), this->force.yptr(), this->torque.ptr(), alpha.ptr());
}

void RigidBumpy::sync_class_constants_poly_extras_impl() {
    bind_rigid_bumpy_globals(this->e_interaction.ptr(), this->vertex_rad.ptr(), this->mass.ptr(), this->moment_inertia.ptr());
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
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::compute_ke_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(), this->angular_vel.ptr(),
        this->ke.ptr()
    );
}

void RigidBumpy::allocate_poly_extras_impl(int N) {
    this->torque.resize(N);
    this->angular_vel.resize(N);
    this->angle.resize(N);
    this->torque.fill(0.0);
    this->angular_vel.fill(0.0);
    this->angle.fill(0.0);
}

void RigidBumpy::allocate_poly_vertex_extras_impl(int Nv) {
    // nothing to do
}

void RigidBumpy::allocate_poly_system_extras_impl(int S) {
    // nothing to do
}

void RigidBumpy::enable_poly_swap_extras_impl(bool enable) {
    // nothing to do
}


void RigidBumpy::set_random_positions_impl(double box_pad_x, double box_pad_y) {
    if (!this->pos.rng_enabled()) { this->pos.enable_rng(); }  // enable RNG if not already enabled
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::set_random_positions_in_box_kernel, G, B,
        this->pos.rng_states.data().get(),
        this->pos.xptr(), this->pos.yptr(), this->angle.ptr(),
        this->vertex_pos.xptr(), this->vertex_pos.yptr(), box_pad_x, box_pad_y);
}

// Compute the total power for each system
void RigidBumpy::compute_fpower_total_impl() {
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
            this->vel.yptr(),
            this->torque.ptr(),
            this->angular_vel.ptr()
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

void RigidBumpy::save_state_impl(df::DeviceField1D<int> flag, int true_val) {
    if (this->last_state_pos.size() != this->pos.size()) {
        this->last_state_pos.resize(this->pos.size());
    }
    if (this->last_state_angle.size() != this->angle.size()) {
        this->last_state_angle.resize(this->angle.size());
    }
    if (this->last_state_mass.size() != this->mass.size()) {
        this->last_state_mass.resize(this->mass.size());
    }
    if (this->last_state_moment_inertia.size() != this->moment_inertia.size()) {
        this->last_state_moment_inertia.resize(this->moment_inertia.size());
    }
    if (this->last_state_vertex_rad.size() != this->vertex_rad.size()) {
        this->last_state_vertex_rad.resize(this->vertex_rad.size());
    }
    if (this->last_state_vertex_pos.size() != this->vertex_pos.size()) {
        this->last_state_vertex_pos.resize(this->vertex_pos.size());
    }
    if (this->last_state_vertex_particle_id.size() != this->vertex_particle_id.size()) {
        this->last_state_vertex_particle_id.resize(this->vertex_particle_id.size());
    }
    if (this->last_state_box_size.size() != this->box_size.size()) {
        this->last_state_box_size.resize(this->box_size.size());
    }
    if (this->last_state_n_vertices_per_particle.size() != this->n_vertices_per_particle.size()) {
        this->last_state_n_vertices_per_particle.resize(this->n_vertices_per_particle.size());
    }
    
    const int N = n_particles();
    const int V = n_vertices();
    const int S = n_systems();
    auto B = md::launch::threads_for();
    auto G_N = md::launch::blocks_for(N);
    auto G_S = md::launch::blocks_for(S);
    auto G_V = md::launch::blocks_for(V);
    CUDA_LAUNCH(kernels::save_particle_state_kernel, G_N, B,
        this->pos.xptr(), this->pos.yptr(), this->last_state_pos.xptr(), this->last_state_pos.yptr(),
        this->angle.ptr(), this->last_state_angle.ptr(),
        this->mass.ptr(), this->last_state_mass.ptr(),
        this->moment_inertia.ptr(), this->last_state_moment_inertia.ptr(),
        this->n_vertices_per_particle.ptr(), this->last_state_n_vertices_per_particle.ptr(),
        flag.ptr(), true_val);
    CUDA_LAUNCH(kernels::save_vertex_state_kernel, G_V, B,
        this->vertex_rad.ptr(), this->last_state_vertex_rad.ptr(),
        this->vertex_pos.xptr(), this->vertex_pos.yptr(), this->last_state_vertex_pos.xptr(), this->last_state_vertex_pos.yptr(),
        this->vertex_particle_id.ptr(), this->last_state_vertex_particle_id.ptr(),
        flag.ptr(), true_val);

    CUDA_LAUNCH(kernels::save_system_state_kernel, G_S, B,
        this->box_size.xptr(), this->box_size.yptr(), this->last_state_box_size.xptr(), this->last_state_box_size.yptr(),
        flag.ptr(), true_val);
    
}

void RigidBumpy::restore_state_impl(df::DeviceField1D<int> flag, int true_val) {
    if (this->last_state_pos.size() != this->pos.size()) {
        throw std::runtime_error("RigidBumpy::restore_state_impl: last_state_pos is not initialized");
    }
    if (this->last_state_angle.size() != this->angle.size()) {
        throw std::runtime_error("RigidBumpy::restore_state_impl: last_state_angle is not initialized");
    }
    if (this->last_state_mass.size() != this->mass.size()) {
        throw std::runtime_error("RigidBumpy::restore_state_impl: last_state_mass is not initialized");
    }
    if (this->last_state_moment_inertia.size() != this->moment_inertia.size()) {
        throw std::runtime_error("RigidBumpy::restore_state_impl: last_state_moment_inertia is not initialized");
    }
    if (this->last_state_vertex_rad.size() != this->vertex_rad.size()) {
        throw std::runtime_error("RigidBumpy::restore_state_impl: last_state_vertex_rad is not initialized");
    }
    if (this->last_state_vertex_pos.size() != this->vertex_pos.size()) {
        throw std::runtime_error("RigidBumpy::restore_state_impl: last_state_vertex_pos is not initialized");
    }
    if (this->last_state_vertex_particle_id.size() != this->vertex_particle_id.size()) {
        throw std::runtime_error("RigidBumpy::restore_state_impl: last_state_vertex_particle_id is not initialized");
    }
    if (this->last_state_box_size.size() != this->box_size.size()) {
        throw std::runtime_error("RigidBumpy::restore_state_impl: last_state_box_size is not initialized");
    }
    if (this->last_state_n_vertices_per_particle.size() != this->n_vertices_per_particle.size()) {
        throw std::runtime_error("RigidBumpy::restore_state_impl: last_state_n_vertices_per_particle is not initialized");
    }

    const int N = n_particles();
    const int V = n_vertices();
    const int S = n_systems();
    auto B = md::launch::threads_for();
    auto G_N = md::launch::blocks_for(N);
    auto G_S = md::launch::blocks_for(S);
    auto G_V = md::launch::blocks_for(V);
    CUDA_LAUNCH(kernels::restore_particle_state_kernel, G_N, B,
        this->pos.xptr(), this->pos.yptr(), this->last_state_pos.xptr(), this->last_state_pos.yptr(),
        this->angle.ptr(), this->last_state_angle.ptr(),
        this->mass.ptr(), this->last_state_mass.ptr(),
        this->moment_inertia.ptr(), this->last_state_moment_inertia.ptr(),
        this->n_vertices_per_particle.ptr(), this->last_state_n_vertices_per_particle.ptr(),
        flag.ptr(), true_val);
    CUDA_LAUNCH(kernels::restore_vertex_state_kernel, G_V, B,
        this->vertex_rad.ptr(), this->last_state_vertex_rad.ptr(),
        this->vertex_pos.xptr(), this->last_state_vertex_pos.xptr(),
        this->vertex_pos.yptr(), this->last_state_vertex_pos.yptr(),
        this->vertex_particle_id.ptr(), this->last_state_vertex_particle_id.ptr(),
        flag.ptr(), true_val);
    CUDA_LAUNCH(kernels::restore_system_state_kernel, G_S, B,
        this->box_size.xptr(), this->box_size.yptr(), this->last_state_box_size.xptr(), this->last_state_box_size.yptr(),
        flag.ptr(), true_val);

    // recalculate particle offset using an exclusive scan of n_vertices_per_particle
    thrust::exclusive_scan(
        thrust::device,
        this->n_vertices_per_particle.ptr(),
        this->n_vertices_per_particle.ptr() + N,
        this->particle_offset.ptr()
    );

    // sync vertex system constants
    Base::sync_box();
    Base::sync_class_constants();
    Base::check_neighbors();
}

void RigidBumpy::load_static_from_hdf5_poly_extras_impl(hid_t group) {
    this->mass.from_host(read_vector<double>(group, "mass"));
    this->moment_inertia.from_host(read_vector<double>(group, "moment_inertia"));
}

void RigidBumpy::load_from_hdf5_poly_extras_impl(hid_t group) {
    this->angle.from_host(read_vector<double>(group, "angle"));
    this->pos.from_host(read_vector_2d<double>(group, "pos"));
    if (h5_link_exists(group, "vel")) {
        this->vel.from_host(read_vector_2d<double>(group, "vel"));
    }
    if (h5_link_exists(group, "force")) {
        this->force.from_host(read_vector_2d<double>(group, "force"));
    }
    if (h5_link_exists(group, "torque")) {
        this->torque.from_host(read_vector<double>(group, "torque"));
    }
    if (h5_link_exists(group, "angular_vel")) {
        this->angular_vel.from_host(read_vector<double>(group, "angular_vel"));
    }
}

std::string RigidBumpy::get_class_name_impl() {
    return "RigidBumpy";
}

std::vector<std::string> RigidBumpy::get_static_field_names_poly_extras_impl() {
    return {"mass", "moment_inertia"};
}

std::vector<std::string> RigidBumpy::get_state_field_names_poly_extras_impl() {
    return {"pos", "vel", "force", "torque", "angular_vel", "angle"};
}

void RigidBumpy::output_build_registry_poly_extras_impl(io::OutputRegistry& reg) {
    // Register rigid bumpy specific fields
    using io::FieldSpec1D; using io::FieldSpec2D;
    std::string order_inv_str = "order_inv";
    {
        FieldSpec1D<double> p;
        p.index_by = [order_inv_str]{ return order_inv_str; };
        p.get_device_field = [this]{ return &this->mass; };
        reg.fields["mass"] = p;
    }
    {
        FieldSpec1D<double> p; 
        p.index_by = [order_inv_str]{ return order_inv_str; };
        p.get_device_field = [this]{ return &this->moment_inertia; };
        reg.fields["moment_inertia"] = p;
    }
    {
        FieldSpec2D<double> p; 
        p.index_by = [order_inv_str]{ return order_inv_str; };
        p.get_device_field = [this]{ return &this->pos; };
        reg.fields["pos"] = p;
    }
    {
        FieldSpec2D<double> p; 
        p.index_by = [order_inv_str]{ return order_inv_str; };
        p.get_device_field = [this]{ return &this->vel; };
        reg.fields["vel"] = p;
    }
    {
        FieldSpec2D<double> p; 
        p.index_by = [order_inv_str]{ return order_inv_str; };
        p.get_device_field = [this]{ return &this->force; };
        reg.fields["force"] = p;
    }
    {
        FieldSpec1D<double> p; 
        p.index_by = [order_inv_str]{ return order_inv_str; };
        p.get_device_field = [this]{ return &this->angle; };
        reg.fields["angle"] = p;
    }
    {
        FieldSpec1D<double> p; 
        p.index_by = [order_inv_str]{ return order_inv_str; };
        p.get_device_field = [this]{ return &this->torque; };
        reg.fields["torque"] = p;
    }
    {
        FieldSpec1D<double> p; 
        p.index_by = [order_inv_str]{ return order_inv_str; };
        p.get_device_field = [this]{ return &this->angular_vel; };
        reg.fields["angular_vel"] = p;
    }
}


} // namespace md::rigid_bumpy