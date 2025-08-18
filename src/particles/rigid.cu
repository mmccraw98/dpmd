// #include "particles/rigid.hpp"
// #include "kernels/common.cuh"
// #include "utils/cuda_debug.hpp"
// #include <thrust/transform.h>
// #include <thrust/reduce.h>
// #include <thrust/scan.h>
// #include <thrust/iterator/permutation_iterator.h>
// #include <thrust/execution_policy.h>
// #include <cub/cub.cuh>

// // ---- Per-system constants ----
// struct RigidConst {
//     const double* e_interaction;
//     const int* particle_id;
//     const int* particle_offset;
//     const int* n_vertices_per_particle;
//     int n_vertices;
// };

// __constant__ RigidConst g_rigid;

// namespace {  // TU-local

// __host__ void bind_rigid_globals(const double* d_e_interaction, const int* d_particle_id, const int* d_particle_offset, const int* d_n_vertices_per_particle, int n_vertices) {
//     RigidConst h { d_e_interaction, d_particle_id, d_particle_offset, d_n_vertices_per_particle, n_vertices };
//     cudaMemcpyToSymbol(g_rigid, &h, sizeof(RigidConst));
// }

// struct CountMinusOneClamp {
//     __host__ __device__ int operator()(int m) const {
//         int c = m - 1;
//         return (c > 0) ? c : 0;
//     }
// };

// __global__ void fill_naive_neighbor_list_kernel(
//     const int* __restrict__ neighbor_start,     // N+1
//     int*       __restrict__ neighbor_ids        // total
// ){
//     const int N = g_rigid.n_vertices;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= N) return;

//     const int sid   = md::geo::g_sys.id[i];
//     const int begin = md::geo::g_sys.offset[sid];
//     const int end   = md::geo::g_sys.offset[sid+1];
//     const long pid = g_rigid.particle_id[i];

//     int w = neighbor_start[i];
//     for (int j = begin; j < end; ++j) {
//         if (j == i) continue;
//         const long pj = g_rigid.particle_id[j];
//         if (pid == pj) continue;
//         neighbor_ids[w++] = j;
//     }
// }

// __global__ void disk_force_kernel(
//     const double* __restrict__ x,
//     const double* __restrict__ y,
//     const double* __restrict__ rad,
//           double* __restrict__ fx,
//           double* __restrict__ fy,
//           double* __restrict__ pe)
// {
//     const int N = g_rigid.n_vertices;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= N) return;

//     const int sid = md::geo::g_sys.id[i];
//     const double e_i = g_rigid.e_interaction[sid];
//     const double box_size_x = md::geo::g_box.size_x[sid];
//     const double box_size_y = md::geo::g_box.size_y[sid];
//     const double box_inv_x = md::geo::g_box.inv_x[sid];
//     const double box_inv_y = md::geo::g_box.inv_y[sid];

//     const double xi = x[i], yi = y[i], ri = rad[i];
//     double fxi = 0.0, fyi = 0.0, pei = 0.0;

//     const int beg = md::geo::g_neigh.start[i];
//     const int end = md::geo::g_neigh.start[i+1];

//     for (int k = beg; k < end; ++k) {
//         const int j = md::geo::g_neigh.ids[k];
//         if (j == -1) continue;
//         const double xj = x[j], yj = y[j], rj = rad[j];

//         double dx, dy;
//         double r2 = md::geo::disp_pbc_L(xi, yi, xj, yj, box_size_x, box_size_y, box_inv_x, box_inv_y, dx, dy);

//         // Early reject if no overlap: r^2 >= (ri+rj)^2
//         const double radsum = ri + rj;
//         const double radsum2 = radsum * radsum;
//         if (r2 >= radsum2) continue;

//         // Overlap: compute r and invr once
//         const double r   = sqrt(r2);
//         const double inv = 1.0 / r;
//         const double nx  = dx * inv;
//         const double ny  = dy * inv;

//         const double delta = radsum - r;
//         const double fmag  = e_i * delta;

//         // Force on i is along -n (repulsion)
//         fxi -= fmag * nx;
//         fyi -= fmag * ny;

//         // Single-count the pair energy (each pair gets half)
//         pei += (0.5 * e_i * delta * delta) * 0.5;
//     }
    
//     fx[i] = fxi;
//     fy[i] = fyi;
//     pe[i] = pei;
// }

// __global__ void disk_wall_force_kernel(
//     const double* __restrict__ x,
//     const double* __restrict__ y,
//     const double* __restrict__ rad,
//     double* __restrict__ fx,
//     double* __restrict__ fy,
//     double* __restrict__ pe)
// {
//     const int N = g_rigid.n_vertices;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= N) return;

//     const int sid = md::geo::g_sys.id[i];
//     const double e_i = g_rigid.e_interaction[sid];
//     const double box_size_x = md::geo::g_box.size_x[sid];
//     const double box_size_y = md::geo::g_box.size_y[sid];
//     const double box_inv_x = md::geo::g_box.inv_x[sid];
//     const double box_inv_y = md::geo::g_box.inv_y[sid];

//     const double xi = x[i], yi = y[i], ri = rad[i];
    
//     double fxi = 0.0, fyi = 0.0, pei = 0.0;

//     if (xi < ri) {
//         const double delta = ri - xi;
//         const double fmag = e_i * delta;
//         fxi -= fmag;
//         pei += (0.5 * e_i * delta * delta) * 0.5;
//     }
//     if (xi > box_size_x - ri) {
//         const double delta = ri - (box_size_x - xi);
//         const double fmag = e_i * delta;
//         fxi -= fmag;
//         pei += (0.5 * e_i * delta * delta) * 0.5;
//     }
//     if (yi < ri) {
//         const double delta = ri - yi;
//         const double fmag = e_i * delta;
//         fyi -= fmag;
//         pei += (0.5 * e_i * delta * delta) * 0.5;
//     }
//     if (yi > box_size_y - ri) {
//         const double delta = ri - (box_size_y - yi);
//         const double fmag = e_i * delta;
//         fyi -= fmag;
//         pei += (0.5 * e_i * delta * delta) * 0.5;
//     }

//     fx[i] += fxi;
//     fy[i] += fyi;
//     pe[i] += pei;
// }

// __global__ void set_random_particle_positions_kernel(
//     curandStatePhilox4_32_10_t* __restrict__ states,
//     const int* __restrict__ particle_id,
//     double* __restrict__ pos_x,
//     double* __restrict__ pos_y,
//     double* __restrict__ angle,
//     double* __restrict__ vertex_pos_x,
//     double* __restrict__ vertex_pos_y,
//     double x_min,
//     double x_max,
//     double y_min,
//     double y_max
// ) {
//     // TODO:
//     // 1) build a kernel over PARTICLES not vertices
//     // 2) for each particle, generate a random position within the bounds and an angle
//     // 3) use the angle to rotate the vertex positions around the particle's position
//     // 4) use the new particle position to update the vertex positions
//     // 5) update the particle positions
//     const int N = md::geo::g_sys.n_particles;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= N) return;

//     curandStatePhilox4_32_10_t st = states[i];

//     const int n_vertices_per_particle = g_rigid.n_vertices_per_particle[i];
//     const int particle_offset = g_rigid.particle_offset[i];

//     double pos_xi = pos_x[i];
//     double pos_yi = pos_y[i];
//     double angle_i = angle[i];
    
//     double new_pos_xi = curand_uniform_double(&st) * (x_max - x_min) + x_min;
//     double new_pos_yi = curand_uniform_double(&st) * (y_max - y_min) + y_min;
//     double angle_period_inv = (n_vertices_per_particle > 1) ? 1.0 / n_vertices_per_particle : 0.0;
//     double new_angle_i = curand_uniform_double(&st) * 2 * M_PI * angle_period_inv;

//     double delta_x = new_pos_xi - pos_xi;
//     double delta_y = new_pos_yi - pos_yi;
//     double dtheta = new_angle_i - angle_i;

//     double s, c; sincos(dtheta, &s, &c);
//     for (int j = 0; j < n_vertices_per_particle; ++j) {
//         double dx = vertex_pos_x[particle_offset + j] - pos_xi;
//         double dy = vertex_pos_y[particle_offset + j] - pos_yi;
//         double rx = c*dx - s*dy;
//         double ry = s*dx + c*dy;
//         vertex_pos_x[particle_offset + j] = new_pos_xi + rx;
//         vertex_pos_y[particle_offset + j] = new_pos_yi + ry;
//     }
    
//     pos_x[i] = new_pos_xi;
//     pos_y[i] = new_pos_yi;
//     angle[i] = new_angle_i;

//     states[i] = st;
// }

// __global__ void sum_vertex_pe_to_particle_pe_kernel(
//     const double* __restrict__ pe,
//     double* __restrict__ particle_pe
// ) {
//     const int N = md::geo::g_sys.n_particles;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= N) return;

//     const int n_vertices_per_particle = g_rigid.n_vertices_per_particle[i];
//     const int particle_offset = g_rigid.particle_offset[i];

//     double pei = 0.0;
//     for (int j = 0; j < n_vertices_per_particle; ++j) {
//         pei += pe[particle_offset + j];
//     }

//     particle_pe[i] = pei;
// }

// } // anon namespace

// namespace md {

// void Rigid::set_random_particle_positions(double x_min, double x_max, double y_min, double y_max) {
//     const int N = n_particles();
//     const int B = 256;
//     dim3 Bdim(B), Gdim((N + B - 1)/B);
//     CUDA_LAUNCH(set_random_particle_positions_kernel, Gdim, Bdim,
//         pos.ptr_rng2d(), particle_id.ptr(), pos.xptr(), pos.yptr(), angle.ptr(), vertex_pos.xptr(), vertex_pos.yptr(), x_min, x_max, y_min, y_max
//     );
// }

// void Rigid::init_naive_neighbors_impl() {
//     // 1) counts[i] = system_size[ system_id[i] ] - 1
//     int N = n_vertices();
//     neighbor_count.resize(N);
//     auto sys_size_begin = system_size.data.begin();
//     auto sid_begin      = system_id.data.begin();
//     auto per_particle_m = thrust::make_permutation_iterator(sys_size_begin, sid_begin);

//     thrust::transform(
//         per_particle_m, per_particle_m + N,
//         neighbor_count.data.begin(),
//         CountMinusOneClamp()
//     );

//     // 2) exclusive scan -> neighbor_start (N+1)
//     neighbor_start.resize(N + 1);
//     thrust::exclusive_scan(
//         neighbor_count.data.begin(), neighbor_count.data.end(),
//         neighbor_start.data.begin()
//     );

//     // 3) total edges = sum(counts) (computed on device; result returned to host as an int)
//     int total = thrust::reduce(
//         neighbor_count.data.begin(), neighbor_count.data.end(),
//         0, thrust::plus<int>()
//     );

//     // 4) set start[N] on device (no cudaMemcpy)
//     thrust::fill_n(neighbor_start.data.begin() + N, 1, total);

//     // 5) size neighbor_ids
//     neighbor_ids.resize(total);
//     neighbor_ids.fill(-1);

//     // 6) fill neighbor_ids
//     int B = 256;
//     dim3 Bdim(B), Gdim((N + B - 1)/B);
//     CUDA_LAUNCH(fill_naive_neighbor_list_kernel, Gdim, Bdim,
//         neighbor_start.ptr(), neighbor_ids.ptr()
//     );
// }

// void Rigid::compute_forces_impl() {
//     const int N = n_particles();
//     const int B = 256;
//     dim3 Bdim(B), Gdim((N + B - 1)/B);
//     CUDA_LAUNCH(disk_force_kernel, Gdim, Bdim,
//         vertex_pos.xptr(), vertex_pos.yptr(), rad.ptr(), force.xptr(), force.yptr(), pe.ptr()
//     );
// }

// void Rigid::compute_wall_forces_impl() {
//     const int N = n_particles();
//     const int B = 256;
//     dim3 Bdim(B), Gdim((N + B - 1)/B);
//     CUDA_LAUNCH(disk_wall_force_kernel, Gdim, Bdim,
//         vertex_pos.xptr(), vertex_pos.yptr(), rad.ptr(), force.xptr(), force.yptr(), pe.ptr()
//     );
// }

// void Rigid::sync_class_constants_impl() {
//     bind_rigid_globals(e_interaction.ptr(), particle_id.ptr(), particle_offset.ptr(), n_vertices_per_particle.ptr(), n_vertices());
// }

// void Rigid::sum_vertex_pe_to_particle_pe() {
//     const int N = n_particles();
//     const int B = 256;
//     dim3 Bdim(B), Gdim((N + B - 1)/B);
//     CUDA_LAUNCH(sum_vertex_pe_to_particle_pe_kernel, Gdim, Bdim, pe.ptr(), particle_pe.ptr());
// }

// } // namespace md