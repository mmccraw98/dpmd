#include "particles/rigid_bumpy.cuh"
#include "kernels/base_particle_kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

namespace md::rigid_bumpy {

__constant__ RigidBumpyConst g_rigid_bumpy;

void bind_rigid_bumpy_globals(const double* d_e_interaction, const double* d_vertex_rad, const double* d_mass, const double* d_moment_inertia, unsigned int* d_rebuild_flag) {
    RigidBumpyConst h { d_e_interaction, d_vertex_rad, d_mass, d_moment_inertia, d_rebuild_flag };
    cudaMemcpyToSymbol(g_rigid_bumpy, &h, sizeof(RigidBumpyConst));
}

// RigidBumpy-specific kernels
namespace kernels {

// Update the positions of the particles and their vertices
__global__ void update_positions_kernel_naive(
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
        const int j_index = md::poly::g_poly.static_particle_order[j];
        double dx = vertex_x[j_index] - x_i;
        double dy = vertex_y[j_index] - y_i;
        double rx = c*dx - s*dy;
        double ry = s*dx + c*dy;
        vertex_x[j_index] = new_x_i + rx;
        vertex_y[j_index] = new_y_i + ry;
    }
}

// Update the positions of the particles and their vertices - used for cell neighbor list
__global__ void update_positions_kernel_cell(
    double* __restrict__ x,
    double* __restrict__ y,
    const double* __restrict__ last_x,
    const double* __restrict__ last_y,
    double* __restrict__ disp2,
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
        const int j_index = md::poly::g_poly.static_particle_order[j];
        double dx = vertex_x[j_index] - x_i;
        double dy = vertex_y[j_index] - y_i;
        double rx = c*dx - s*dy;
        double ry = s*dx + c*dy;
        vertex_x[j_index] = new_x_i + rx;
        vertex_y[j_index] = new_y_i + ry;
    }
    // Calculate squared displacement for neighbor list update
    // use the position of the first vertex in each particle
    const int first_vertex_index = md::poly::g_poly.static_particle_order[beg];
    double dx = vertex_x[first_vertex_index] - last_x[i];
    double dy = vertex_y[first_vertex_index] - last_y[i];
    double d2 = dx * dx + dy * dy;
    disp2[i] = d2;

    // Determine per-system rebuild flag
    const double thresh2 = md::geo::g_neigh.thresh2[sid];
    if (d2 > thresh2) atomicOr(&g_rigid_bumpy.rebuild_flag[sid], 1u);
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
        const int j_index = md::poly::g_poly.static_particle_order[j];
        vertex_force_x_i = vertex_force_x[j_index];
        vertex_force_y_i = vertex_force_y[j_index];
        fxi += vertex_force_x_i;
        fyi += vertex_force_y_i;
        pei += vertex_pe[j_index];
        dx = vx[j_index] - px_i;
        dy = vy[j_index] - py_i;
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
        const int j_index = md::poly::g_poly.static_particle_order[particle_offset + j];
        double dx = vertex_pos_x[j_index] - pos_xi;
        double dy = vertex_pos_y[j_index] - pos_yi;
        double rx = c*dx - s*dy;
        double ry = s*dx + c*dy;
        vertex_pos_x[j_index] = new_pos_xi + rx;
        vertex_pos_y[j_index] = new_pos_yi + ry;
    }
    
    pos_x[i] = new_pos_xi;
    pos_y[i] = new_pos_yi;
    angle[i] = new_angle_i;

    states[i] = st;
}

// Set random positions of particles within their domains
// the domains are assumed to be convex polygons
__global__ void set_random_positions_in_domains_kernel(
    const int N_domains,
    curandStatePhilox4_32_10_t* __restrict__ states,
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ angle,
    double* __restrict__ vertex_pos_x,
    double* __restrict__ vertex_pos_y,
    const double* __restrict__ domain_pos_x,
    const double* __restrict__ domain_pos_y,
    const double* __restrict__ domain_centroid_x,
    const double* __restrict__ domain_centroid_y,
    const int* __restrict__ domain_offset,
    const int* __restrict__ domain_particle_id,
    const double* __restrict__ domain_fractional_area
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_domains) return;

    const int d_beg = domain_offset[i];
    const int d_end = domain_offset[i+1];
    
    curandStatePhilox4_32_10_t st = states[i];
    double rng_t = curand_uniform_double(&st);  // random triangle
    double rng_u = curand_uniform_double(&st);  // random point in triangle
    double rng_v = curand_uniform_double(&st);  // random point in triangle
    double rng_w = curand_uniform_double(&st);  // random angle

    double d_frac_area_prev = 0.0;
    int j_sel = 0;
    // find the index within the fractional triangle areas that is greater than or equal to the target
    for (int j = d_beg; j < d_end; ++j) {
        const double d_frac_area = domain_fractional_area[j];
        if ((rng_t > d_frac_area_prev) && (rng_t <= d_frac_area)) { j_sel = j; break; }
        d_frac_area_prev = d_frac_area;
    }

    // selected triangle vertices
    double a_x = domain_centroid_x[i];
    double a_y = domain_centroid_y[i];
    double b_x = domain_pos_x[j_sel];
    double b_y = domain_pos_y[j_sel];
    int c_index = ((j_sel + 1 - d_beg) % (d_end - d_beg)) + d_beg;
    double c_x = domain_pos_x[c_index];
    double c_y = domain_pos_y[c_index];

    // uniform sample inside triangle
    if (rng_u + rng_v > 1.0) { rng_u = 1.0 - rng_u; rng_v = 1.0 - rng_v; }

    const int particle_id = domain_particle_id[i];
    const int n_vertices_per_particle = md::poly::g_poly.n_vertices_per_particle[particle_id];
    const int particle_offset = md::poly::g_poly.particle_offset[particle_id];

    double new_pos_x = a_x + rng_u * (b_x - a_x) + rng_v * (c_x - a_x);
    double new_pos_y = a_y + rng_u * (b_y - a_y) + rng_v * (c_y - a_y);
    double angle_period_inv = (n_vertices_per_particle > 1) ? 1.0 / n_vertices_per_particle : 0.0;
    double new_angle_i = rng_w * 2 * M_PI * angle_period_inv;

    // Displace and rotate all vertices
    double s, c; sincos(new_angle_i - angle[particle_id], &s, &c);
    double pos_x_i = pos_x[particle_id];
    double pos_y_i = pos_y[particle_id];
    for (int j = 0; j < n_vertices_per_particle; ++j) {
        const int j_index = md::poly::g_poly.static_particle_order[particle_offset + j];
        double dx = vertex_pos_x[j_index] - pos_x_i;
        double dy = vertex_pos_y[j_index] - pos_y_i;
        double rx = c*dx - s*dy;
        double ry = s*dx + c*dy;
        vertex_pos_x[j_index] = new_pos_x + rx;
        vertex_pos_y[j_index] = new_pos_y + ry;
    }
    
    pos_x[particle_id] = new_pos_x;
    pos_y[particle_id] = new_pos_y;
    angle[particle_id] = new_angle_i;

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
        const int j_index = md::poly::g_poly.static_particle_order[j];
        vertex_x[j_index] += (new_pos_x - pos_x);
        vertex_y[j_index] += (new_pos_y - pos_y);
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
    const int* __restrict__ static_particle_order, int* __restrict__ last_static_particle_order,
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
        last_static_particle_order[i] = static_particle_order[i];
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
    int* __restrict__ static_particle_order, const int* __restrict__ last_static_particle_order,
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
        static_particle_order[i] = last_static_particle_order[i];
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

__global__ void set_n_dof_kernel(int* __restrict__ n_dof, const double* __restrict__ moment_inertia) {
    const int sid = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (sid >= S) return;
    const int beg = md::geo::g_sys.offset[sid];
    const int end = md::geo::g_sys.offset[sid+1];
    int n_dof_sum = 0;
    for (int i = beg; i < end; i++) {
        n_dof_sum += 2;
        if (moment_inertia[i] > 0.0) {
            n_dof_sum += 1;
        }
    }
    n_dof[sid] = n_dof_sum;
}

__global__ void count_particle_contacts_kernel(
    const double* __restrict__ vertex_pos_x, const double* __restrict__ vertex_pos_y,
    const int* __restrict__ particle_neighbor_start, const int* __restrict__ particle_neighbor_ids,
    int* __restrict__ pair_vertex_contacts_i, int* __restrict__ pair_vertex_contacts_j,
    int* __restrict__ pair_ids_i, int* __restrict__ pair_ids_j,
    int* __restrict__ contacts
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = md::geo::g_sys.n_particles;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];
    const double box_inv_x = md::geo::g_box.inv_x[sid];
    const double box_inv_y = md::geo::g_box.inv_y[sid];
    const int beg = particle_neighbor_start[i];
    const int end = particle_neighbor_start[i+1];
    int contact_count = 0;
    for (int p_k = beg; p_k < end; p_k++) {  // loop over the particle neighbors
        int j = particle_neighbor_ids[p_k];
        int num_interacting_vertices = 0;
        for (int v_i = md::poly::g_poly.particle_offset[i]; v_i < md::poly::g_poly.particle_offset[i+1]; v_i++) {  // loop over the vertices of the particle
            const int v_i_index = md::poly::g_poly.static_particle_order[v_i];  // get the real vertex index
            bool found_interaction = false;  // for each vertex of the current particle, check if it is interacting with the neighboring particle
            const double xi = vertex_pos_x[v_i_index], yi = vertex_pos_y[v_i_index];
            const double ri = g_rigid_bumpy.vertex_rad[v_i_index];
            const int v_beg = md::geo::g_neigh.start[v_i_index];
            const int v_end = md::geo::g_neigh.start[v_i_index + 1];
            for (int k = v_beg; k < v_end; k++) {  // loop over the vertices of the particle neighbor
                const int v_j = md::geo::g_neigh.ids[k];
                if (j != md::poly::g_poly.particle_id[v_j]) continue;  // skip the vertex if it is not in the neighboring particle
                // check if the vertex is overlapping with the neighboring vertex
                const double xj = vertex_pos_x[v_j], yj = vertex_pos_y[v_j];
                const double rj = g_rigid_bumpy.vertex_rad[v_j];
                double dx, dy;
                double r2 = md::geo::disp_pbc_L(xi, yi, xj, yj, box_size_x, box_size_y, box_inv_x, box_inv_y, dx, dy);
                const double radsum = ri + rj;
                const double radsum2 = radsum * radsum;
                if (r2 >= radsum2) continue;
                found_interaction = true;
                break;
            }
            if (found_interaction) {
                num_interacting_vertices++;
            }
        }
        // once done aggregating the interacting vertices for the pair i-j, store the number of interacting vertices
        pair_vertex_contacts_i[p_k] = num_interacting_vertices;  // for this pair
        // find the index of the pair j-i in the particle neighbor list, store the number of interacting vertices for this pair
        const int rev_beg = particle_neighbor_start[j];
        const int rev_end = particle_neighbor_start[j+1];
        for (int p_l = rev_beg; p_l < rev_end; p_l++) {
            if (particle_neighbor_ids[p_l] == i) {
                pair_vertex_contacts_j[p_l] = num_interacting_vertices;
                break;
            }
        }
        pair_ids_i[p_k] = i;
        pair_ids_j[p_k] = j;
        // if there are any interacting vertices, increment the contact count by 1
        if (num_interacting_vertices > 0) {
            contact_count++;
        }
    }
    // finally, store the total number of contacts for the particle
    contacts[i] = contact_count;
}

__global__ void compute_friction_coeff_kernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y,
    const double* __restrict__ vertex_pos_x, const double* __restrict__ vertex_pos_y,
    const int* __restrict__ particle_neighbor_start, const int* __restrict__ particle_neighbor_ids,
    int* __restrict__ pair_ids_i, int* __restrict__ pair_ids_j,
    double* __restrict__ friction_coeff
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = md::geo::g_sys.n_particles;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    const double e_i = g_rigid_bumpy.e_interaction[sid];
    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];
    const double box_inv_x = md::geo::g_box.inv_x[sid];
    const double box_inv_y = md::geo::g_box.inv_y[sid];
    const double xi = pos_x[i], yi = pos_y[i];
    const int beg = particle_neighbor_start[i];
    const int end = particle_neighbor_start[i+1];
    for (int p_k = beg; p_k < end; p_k++) {  // loop over the particle neighbors
        double force_x = 0.0, force_y = 0.0;
        double dist_x, dist_y;
        int j = particle_neighbor_ids[p_k];
        const double xj = pos_x[j], yj = pos_y[j];
        double r_particle = sqrt(md::geo::disp_pbc_L(xi, yi, xj, yj, box_size_x, box_size_y, box_inv_x, box_inv_y, dist_x, dist_y));
        for (int v_i = md::poly::g_poly.particle_offset[i]; v_i < md::poly::g_poly.particle_offset[i+1]; v_i++) {  // loop over the vertices of the particle
            const int v_i_index = md::poly::g_poly.static_particle_order[v_i];  // get the real vertex index
            const double vxi = vertex_pos_x[v_i_index], vyi = vertex_pos_y[v_i_index];
            const double ri = g_rigid_bumpy.vertex_rad[v_i_index];
            const int v_beg = md::geo::g_neigh.start[v_i_index];
            const int v_end = md::geo::g_neigh.start[v_i_index + 1];
            for (int k = v_beg; k < v_end; k++) {  // loop over the vertices of the particle neighbor
                const int v_j = md::geo::g_neigh.ids[k];
                if (j != md::poly::g_poly.particle_id[v_j]) continue;  // skip the vertex if it is not in the neighboring particle
                // check if the vertex is overlapping with the neighboring vertex
                const double vxj = vertex_pos_x[v_j], vyj = vertex_pos_y[v_j];
                const double rj = g_rigid_bumpy.vertex_rad[v_j];
                double dx, dy;
                double r2 = md::geo::disp_pbc_L(vxi, vyi, vxj, vyj, box_size_x, box_size_y, box_inv_x, box_inv_y, dx, dy);
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
                force_x -= fmag * nx;
                force_y -= fmag * ny;
            }
        }
        double normal_force_x = force_x * dist_x / r_particle;
        double normal_force_y = force_y * dist_y / r_particle;
        double normal_force = sqrt(normal_force_x * normal_force_x + normal_force_y * normal_force_y);
        if (normal_force == 0.0) continue;
        double tangential_force_x = force_x - normal_force_x;
        double tangential_force_y = force_y - normal_force_y;
        double tangential_force = sqrt(tangential_force_x * tangential_force_x + tangential_force_y * tangential_force_y);
        friction_coeff[p_k] = tangential_force / normal_force;
        pair_ids_i[p_k] = i;
        pair_ids_j[p_k] = j;
    }
}

__global__ void compute_pair_dist_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const int* __restrict__ particle_neighbor_start, const int* __restrict__ particle_neighbor_ids,
    int* __restrict__ pair_ids_i, 
    int* __restrict__ pair_ids_j,
    double* __restrict__ pair_dist
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];
    const double box_inv_x = md::geo::g_box.inv_x[sid];
    const double box_inv_y = md::geo::g_box.inv_y[sid];
    const double xi = pos_x[i], yi = pos_y[i];
    const int beg = particle_neighbor_start[i];
    const int end = particle_neighbor_start[i+1];
    for (int p_k = beg; p_k < end; p_k++) {
        const int j = particle_neighbor_ids[p_k];
        const double xj = pos_x[j], yj = pos_y[j];
        double dx, dy;
        double r2 = md::geo::disp_pbc_L(xi, yi, xj, yj, box_size_x, box_size_y, box_inv_x, box_inv_y, dx, dy);
        pair_dist[p_k] = sqrt(r2);
        pair_ids_i[p_k] = i;
        pair_ids_j[p_k] = j;
    }
}

__global__ void compute_stress_tensor_kernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ vertex_pos_x,
    const double* __restrict__ vertex_pos_y,
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ angular_vel,
    const double* __restrict__ vertex_mass,
    double* __restrict__ stress_tensor_xx,
    double* __restrict__ stress_tensor_xy,
    double* __restrict__ stress_tensor_yx,
    double* __restrict__ stress_tensor_yy
) {
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int sid = md::geo::g_sys.id[i];
    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];
    const double box_inv_x = md::geo::g_box.inv_x[sid];
    const double box_inv_y = md::geo::g_box.inv_y[sid];
    const double e_i = g_rigid_bumpy.e_interaction[sid];
    const double xi = pos_x[i], yi = pos_y[i];
    const double vel_xi = vel_x[i], vel_yi = vel_y[i];
    const double angular_veli = angular_vel[i];

    double stress_tensor_x_x_acc = 0.0;
    double stress_tensor_x_y_acc = 0.0;
    double stress_tensor_y_x_acc = 0.0;
    double stress_tensor_y_y_acc = 0.0;

    double box_area = box_size_x * box_size_y;

    int beg = md::poly::g_poly.particle_offset[i];
    int n_vertices_per_particle = md::poly::g_poly.n_vertices_per_particle[i];
    int end = beg + n_vertices_per_particle;
    int condition = (n_vertices_per_particle > 1) ? 1.0 : 0.0;

    for (int v = beg; v < end; v++) {
        const int v_index = md::poly::g_poly.static_particle_order[v];  // get the real vertex index
        const double vxi = vertex_pos_x[v_index], vyi = vertex_pos_y[v_index];
        const double ri = g_rigid_bumpy.vertex_rad[v_index];
        const double mass = vertex_mass[v_index];
        const double rel_pos_x = vxi - xi;
        const double rel_pos_y = vyi - yi;
        const double p_rad = sqrt(rel_pos_x * rel_pos_x + rel_pos_y * rel_pos_y);
        double vertex_vel_x = vel_xi - angular_veli * rel_pos_y * (condition);
        double vertex_vel_y = vel_yi + angular_veli * rel_pos_x * (condition);

        stress_tensor_x_x_acc += mass * vertex_vel_x * vertex_vel_x;
        stress_tensor_x_y_acc += mass * vertex_vel_x * vertex_vel_y;
        stress_tensor_y_x_acc += mass * vertex_vel_y * vertex_vel_x;
        stress_tensor_y_y_acc += mass * vertex_vel_y * vertex_vel_y;

        const int v_beg = md::geo::g_neigh.start[v_index];
        const int v_end = md::geo::g_neigh.start[v_index + 1];
        for (int k = v_beg; k < v_end; k++) {
            const int v_j = md::geo::g_neigh.ids[k];
            const double vxj = vertex_pos_x[v_j], vyj = vertex_pos_y[v_j];
            const double rj = g_rigid_bumpy.vertex_rad[v_j];

            double dx, dy;
            double r2 = md::geo::disp_pbc_L(vxi, vyi, vxj, vyj, box_size_x, box_size_y, box_inv_x, box_inv_y, dx, dy);
    
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
            double force_x = -fmag * nx;
            double force_y = -fmag * ny;

            // divide by 2 to avoid double counting
            stress_tensor_x_x_acc += dx * force_x / 2.0;
            stress_tensor_x_y_acc += dx * force_y / 2.0;
            stress_tensor_y_x_acc += dy * force_x / 2.0;
            stress_tensor_y_y_acc += dy * force_y / 2.0;
        }
    }
    stress_tensor_xx[i] = stress_tensor_x_x_acc / box_area;
    stress_tensor_xy[i] = stress_tensor_x_y_acc / box_area;
    stress_tensor_yx[i] = stress_tensor_y_x_acc / box_area;
    stress_tensor_yy[i] = stress_tensor_y_y_acc / box_area;
}

__global__ void reset_displacements_kernel(
    double* __restrict__ last_pos_x,
    double* __restrict__ last_pos_y,
    const double* __restrict__ vertex_pos_x,
    const double* __restrict__ vertex_pos_y,
    double* __restrict__ disp2
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = md::geo::g_sys.n_particles;
    if (i >= N) return;
    const int beg = md::poly::g_poly.particle_offset[i];
    const int first_vertex_index = md::poly::g_poly.static_particle_order[beg];
    last_pos_x[i] = vertex_pos_x[first_vertex_index];
    last_pos_y[i] = vertex_pos_y[first_vertex_index];
    disp2[i] = 0.0;
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
    switch (Base::neighbor_method) {
        case NeighborMethod::Cell:
            CUDA_LAUNCH(kernels::update_positions_kernel_cell, G, B,
                this->pos.xptr(), this->pos.yptr(),
                this->last_pos.xptr(), this->last_pos.yptr(),this->disp2.ptr(),
                this->angle.ptr(),
                this->vertex_pos.xptr(), this->vertex_pos.yptr(),
                this->vel.xptr(), this->vel.yptr(), this->angular_vel.ptr(), scale.ptr(), scale2);
            break;
        case NeighborMethod::Naive:
            CUDA_LAUNCH(kernels::update_positions_kernel_naive, G, B,
                this->pos.xptr(), this->pos.yptr(), this->angle.ptr(),
                this->vertex_pos.xptr(), this->vertex_pos.yptr(),
                this->vel.xptr(), this->vel.yptr(), this->angular_vel.ptr(), scale.ptr(), scale2);
                break;
    }
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
    bind_rigid_bumpy_globals(this->e_interaction.ptr(), this->vertex_rad.ptr(), this->mass.ptr(), this->moment_inertia.ptr(), this->rebuild_flag.ptr());
}

void RigidBumpy::reset_displacements_impl() {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::reset_displacements_kernel, G, B,
        this->last_pos.xptr(), this->last_pos.yptr(), this->vertex_pos.xptr(), this->vertex_pos.yptr(), this->disp2.ptr());
}

void RigidBumpy::reorder_particles_impl() {
    auto src = thrust::make_zip_iterator(
        thrust::make_tuple(
            this->vertex_pos.x.begin(), this->vertex_pos.y.begin(),
            this->vertex_force.x.begin(), this->vertex_force.y.begin(),
            this->vertex_mass.begin(), this->vertex_rad.begin(),
            this->vertex_particle_id.begin(),
            this->cell_id.begin()
        )
    );
    auto dst = thrust::make_zip_iterator(
        thrust::make_tuple(
            this->vertex_pos.x.swap_begin(), this->vertex_pos.y.swap_begin(),
            this->vertex_force.x.swap_begin(), this->vertex_force.y.swap_begin(),
            this->vertex_mass.swap_begin(), this->vertex_rad.swap_begin(),
            this->vertex_particle_id.swap_begin(),
            this->cell_id.swap_begin()
        )
    );
    thrust::gather(this->order.begin(), this->order.end(), src, dst);
    this->vertex_pos.swap(); this->vertex_force.swap(); this->vertex_mass.swap(); this->vertex_rad.swap(); this->vertex_particle_id.swap(); this->cell_id.swap();

    this->build_static_particle_order();
}

bool RigidBumpy::check_cell_neighbors_impl() {
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

void RigidBumpy::compute_ke_impl() {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::compute_ke_kernel, G, B,
        this->vel.xptr(), this->vel.yptr(), this->angular_vel.ptr(),
        this->ke.ptr()
    );
}

void RigidBumpy::set_n_dof_impl() {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(this->n_systems());
    CUDA_LAUNCH(kernels::set_n_dof_kernel, G, B,
        this->n_dof.ptr(), this->moment_inertia.ptr()
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

void RigidBumpy::set_random_positions_in_domains_impl() {
    if (!this->pos.rng_enabled()) { this->pos.enable_rng(); }  // enable RNG if not already enabled
    const int N_domains = this->domain_particle_id.size();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N_domains);
    CUDA_LAUNCH(kernels::set_random_positions_in_domains_kernel, G, B,
        N_domains,
        this->pos.rng_states.data().get(),
        this->pos.xptr(), this->pos.yptr(), this->angle.ptr(),
        this->vertex_pos.xptr(), this->vertex_pos.yptr(),
        this->domain_pos.xptr(), this->domain_pos.yptr(),
        this->domain_centroid.xptr(), this->domain_centroid.yptr(),
        this->domain_offset.ptr(), this->domain_particle_id.ptr(),
        this->domain_fractional_area.ptr());
}

// Compute the total power for each system
void RigidBumpy::compute_fpower_total_impl() {
    cudaStream_t stream = 0;
    const int S = this->n_systems();
    if (this->fpower_total.size() != S) {
        this->fpower_total.resize(S);
    }

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

    this->segmented_sum(input_iter, this->fpower_total.ptr(), stream);
}

void RigidBumpy::compute_contacts_impl() {
    this->build_particle_neighbors();
    const int N = n_particles();
    if (this->contacts.size() != N) {
        this->contacts.resize(N);
    }
    if (this->pair_ids.size() != this->particle_neighbor_ids.size()) {
        this->pair_ids.resize(this->particle_neighbor_ids.size());
    }
    if (this->pair_vertex_contacts.size() != this->particle_neighbor_ids.size()) {
        this->pair_vertex_contacts.resize(this->particle_neighbor_ids.size());
    }
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::count_particle_contacts_kernel, G, B,
        this->vertex_pos.xptr(), this->vertex_pos.yptr(),
        this->particle_neighbor_start.ptr(), this->particle_neighbor_ids.ptr(),
        this->pair_vertex_contacts.xptr(), this->pair_vertex_contacts.yptr(),
        this->pair_ids.xptr(), this->pair_ids.yptr(),
        this->contacts.ptr());
}

void RigidBumpy::compute_friction_coeff() {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    if (this->particle_neighbor_ids.size() == 0) {
        throw std::runtime_error("RigidBumpy::compute_friction_coeff: particle neighbor ids is empty");
    }
    if (this->pair_ids.size() != this->particle_neighbor_ids.size()) {
        this->pair_ids.resize(this->particle_neighbor_ids.size());
    }
    if (this->friction_coeff.size() != this->particle_neighbor_ids.size()) {
        this->friction_coeff.resize(this->particle_neighbor_ids.size());
    }
    this->friction_coeff.fill(0.0);
    CUDA_LAUNCH(kernels::compute_friction_coeff_kernel, G, B,
        this->pos.xptr(), this->pos.yptr(),
        this->vertex_pos.xptr(), this->vertex_pos.yptr(),
        this->particle_neighbor_start.ptr(), this->particle_neighbor_ids.ptr(),
        this->pair_ids.xptr(), this->pair_ids.yptr(),
        this->friction_coeff.ptr());
}

void RigidBumpy::compute_pair_dist_impl() {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    if (this->particle_neighbor_ids.size() == 0) {
        throw std::runtime_error("RigidBumpy::compute_pair_dist_impl: particle neighbor ids is empty");
    }
    if (this->pair_ids.size() != this->particle_neighbor_ids.size()) {
        this->pair_ids.resize(this->particle_neighbor_ids.size());
    }
    if (this->pair_dist.size() != this->particle_neighbor_ids.size()) {
        this->pair_dist.resize(this->particle_neighbor_ids.size());
    }
    CUDA_LAUNCH(kernels::compute_pair_dist_kernel, G, B,
        this->pos.xptr(), this->pos.yptr(),
        this->particle_neighbor_start.ptr(), this->particle_neighbor_ids.ptr(),
        this->pair_ids.xptr(), this->pair_ids.yptr(),
        this->pair_dist.ptr());
}

void RigidBumpy::compute_stress_tensor_impl() {
    const int N = n_particles();
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(N);
    CUDA_LAUNCH(kernels::compute_stress_tensor_kernel, G, B,
        this->pos.xptr(), this->pos.yptr(),
        this->vertex_pos.xptr(), this->vertex_pos.yptr(),
        this->vel.xptr(), this->vel.yptr(), this->angular_vel.ptr(),
        this->vertex_mass.ptr(),
        this->stress_tensor_x.xptr(), this->stress_tensor_x.yptr(), this->stress_tensor_y.xptr(), this->stress_tensor_y.yptr());
}

void RigidBumpy::init_cell_neighbors_poly_extras_impl() {
    this->last_pos.resize(this->n_particles());
    this->disp2.resize(this->n_particles());
    this->rebuild_flag.resize(this->n_systems());
    this->rebuild_flag.fill(0u);
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
    if (this->last_state_static_particle_order.size() != this->static_particle_order.size()) {
        this->last_state_static_particle_order.resize(this->static_particle_order.size());
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
        this->static_particle_order.ptr(), this->last_state_static_particle_order.ptr(),
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
    if (this->last_state_static_particle_order.size() != this->static_particle_order.size()) {
        throw std::runtime_error("RigidBumpy::restore_state_impl: last_state_static_particle_order is not initialized");
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
        this->static_particle_order.ptr(), this->last_state_static_particle_order.ptr(),
        flag.ptr(), true_val);
    CUDA_LAUNCH(kernels::restore_system_state_kernel, G_S, B,
        this->box_size.xptr(), this->box_size.yptr(), this->last_state_box_size.xptr(), this->last_state_box_size.yptr(),
        flag.ptr(), true_val);

    // TODO: i am not sure we need this anymore
    // recalculate particle offset using an exclusive scan of n_vertices_per_particle
    thrust::exclusive_scan(
        thrust::device,
        this->n_vertices_per_particle.ptr(),
        this->n_vertices_per_particle.ptr() + N,
        this->particle_offset.ptr()
    );
}

void RigidBumpy::load_static_from_hdf5_poly_extras_impl(hid_t group) {
    this->mass.from_host(read_vector<double>(group, "mass"));
    this->moment_inertia.from_host(read_vector<double>(group, "moment_inertia"));
    this->area.from_host(read_vector<double>(group, "area"));
}

void RigidBumpy::load_from_hdf5_poly_extras_impl(hid_t group) {
    if (h5_link_exists(group, "angle")) {
        this->angle.from_host(read_vector<double>(group, "angle"));
    }
    if (h5_link_exists(group, "pos")) {
        this->pos.from_host(read_vector_2d<double>(group, "pos"));
    }
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
    return {"mass", "moment_inertia", "area"};
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
        p.get_device_field = [this]{ return &this->mass; };
        reg.fields["mass"] = p;
    }
    {
        FieldSpec1D<double> p; 
        p.get_device_field = [this]{ return &this->moment_inertia; };
        reg.fields["moment_inertia"] = p;
    }
    {
        FieldSpec2D<double> p; 
        p.get_device_field = [this]{ return &this->pos; };
        reg.fields["pos"] = p;
    }
    {
        FieldSpec2D<double> p; 
        p.get_device_field = [this]{ return &this->vel; };
        reg.fields["vel"] = p;
    }
    {
        FieldSpec2D<double> p; 
        p.get_device_field = [this]{ return &this->force; };
        reg.fields["force"] = p;
    }
    {
        FieldSpec1D<double> p; 
        p.get_device_field = [this]{ return &this->angle; };
        reg.fields["angle"] = p;
    }
    {
        FieldSpec1D<double> p; 
        p.get_device_field = [this]{ return &this->torque; };
        reg.fields["torque"] = p;
    }
    {
        FieldSpec1D<double> p; 
        p.get_device_field = [this]{ return &this->angular_vel; };
        reg.fields["angular_vel"] = p;
    }
    {
        FieldSpec1D<double> p;
        p.preprocess = [this]{ this->compute_contacts(); this->compute_friction_coeff(); };
        p.get_device_field = [this]{ return &this->friction_coeff; };
        reg.fields["friction_coeff"] = p;
    }
    {
        FieldSpec2D<int> p;
        p.preprocess = [this]{ this->compute_contacts(); };
        p.get_device_field = [this]{ return &this->pair_vertex_contacts; };
        reg.fields["pair_vertex_contacts"] = p;
    }
    {
        FieldSpec1D<double> p; // overrides base class
        p.preprocess = [this]{ this->build_particle_neighbors(); this->compute_pair_dist(); };
        p.get_device_field = [this]{ return &this->pair_dist; };
        reg.fields["pair_dist"] = p;
    }
}


} // namespace md::rigid_bumpy
