#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

namespace md { namespace geo {

// -----------------------------
// Global constants
// -----------------------------
struct BoxConst {
    const double* size_x;     // length = n_systems
    const double* size_y;     // length = n_systems
    const double* inv_x;      // length = n_systems
    const double* inv_y;      // length = n_systems
};

struct SystemConst {
    const int*    offset;  // length = n_systems+1
    const int*    id;      // length = n_particles
    int           n_systems;
    int           n_particles;
    int           n_vertices;
};

struct NeighborConst {
    const int*    start; // length = n_particles+1
    const int*    ids;   // length = total_neighbors
    const double* cutoff;  // length = n_systems
    const double* thresh2; // length = n_systems
};

struct CellConst {
    const double* size_x;    // length = n_systems
    const double* size_y;    // length = n_systems
    const double* inv_x;     // length = n_systems
    const double* inv_y;     // length = n_systems
    const int*    dim_x;     // length = n_systems
    const int*    dim_y;     // length = n_systems
    const int*    sys_start; // length = n_systems+1
};

// Declared here; defined in a .cu TU.
extern __constant__ BoxConst g_box;
extern __constant__ SystemConst g_sys;
extern __constant__ NeighborConst g_neigh;
extern __constant__ CellConst g_cell;

// Host helpers, defined in common_globals.cu
__host__ void bind_box_globals(const double* d_box_size_x,
                               const double* d_box_size_y,
                               const double* d_box_inv_x,
                               const double* d_box_inv_y);
__host__ void bind_system_globals(const int*    d_system_offset,
                                  const int*    d_system_id,
                                  int           n_systems,
                                  int           n_particles,
                                  int           n_vertices);
__host__ void bind_neighbor_globals(const int*    d_neighbor_start,
                                    const int*    d_neighbor_ids,
                                    const double* d_neighbor_cutoff,
                                    const double* d_thresh2);
__host__ void bind_cell_globals(const double* d_cell_size_x,
                                const double* d_cell_size_y,
                                const double* d_cell_inv_x,
                                const double* d_cell_inv_y,
                                const int*    d_cell_dim_x,
                                const int*    d_cell_dim_y,
                                const int*    d_cell_system_start);

__global__ void calculate_box_inv_kernel(
    const double* __restrict__ box_size_x,
    const double* __restrict__ box_size_y,
    double*       __restrict__ box_inv_x,
    double*       __restrict__ box_inv_y,
    int S
);

// Calculate the cell size and its inverse for each system
__global__ void init_cell_sizes_kernel(
    int S,
    const double* __restrict__ box_size_x,
    const double* __restrict__ box_size_y,
    const int*    __restrict__ cell_dim_x,
    const int*    __restrict__ cell_dim_y,
    double*       __restrict__ cell_size_x,
    double*       __restrict__ cell_size_y,
    double*       __restrict__ cell_inv_x,
    double*       __restrict__ cell_inv_y,
    int*          __restrict__ ncell_out);

__global__ void compute_fractional_packing_fraction_kernel(
    const double* __restrict__ area,
    double* __restrict__ packing_fraction_per_particle
);

__global__ void compute_temperature_kernel(
    const double* __restrict__ ke_total,
    const int*    __restrict__ n_dof,
    double*       __restrict__ temperature
);

// Structs
struct StressTrace2D {
	__host__ __device__
	double operator()(const thrust::tuple<double, double>& t) const {
		return 0.5 * (thrust::get<0>(t) + thrust::get<1>(t));
	}
};

// -----------------------------
// Geometry helpers (ASAP / __forceinline__)
// -----------------------------

// Wrap u in [0,1)
__device__ __forceinline__ double wrap01(double u) {
    u -= floor(u);
    return (u >= 1.0) ? 0.0 : u;
}

// nearest-image: dx -> dx - L * nint(dx/L)   (branchless, numerically robust)
__device__ __forceinline__ double min_image(double dx, double L) {
    return (L > 0.0) ? (dx - L * nearbyint(dx / L)) : dx;
}

__device__ __forceinline__ void min_image_vec(double dx, double dy, double Lx, double Ly,
                                              double& out_dx, double& out_dy) {
    out_dx = (Lx > 0.0) ? (dx - Lx * nearbyint(dx / Lx)) : dx;
    out_dy = (Ly > 0.0) ? (dy - Ly * nearbyint(dy / Ly)) : dy;
}

// ---------- No-PBC: displacement & distances ----------
__device__ __forceinline__ double disp_no_pbc(double xi, double yi,
                                              double xj, double yj,
                                              double& dx, double& dy) {
    dx = xj - xi;
    dy = yj - yi;
    return dx*dx + dy*dy;               // r^2
}

__device__ __forceinline__ double dist2_no_pbc(double xi, double yi,
                                               double xj, double yj) {
    double dx, dy;
    return disp_no_pbc(xi, yi, xj, yj, dx, dy);
}

__device__ __forceinline__ double dist_no_pbc(double xi, double yi,
                                              double xj, double yj) {
    return sqrt(dist2_no_pbc(xi, yi, xj, yj));
}

// ---------- PBC (explicit arrays): displacement & distances ----------
__device__ __forceinline__ double disp_pbc_arrays(double xi, double yi,
                                                  double xj, double yj,
                                                  int sid,
                                                  const double* __restrict__ box_size_x,
                                                  const double* __restrict__ box_size_y,
                                                  double& dx, double& dy) {
    const double lx = (box_size_x ? box_size_x[sid] : 0.0);
    const double ly = (box_size_y ? box_size_y[sid] : 0.0);
    dx = xj - xi; dy = yj - yi;
    // branchless nearest image
    dx = (lx > 0.0) ? (dx - lx * nearbyint(dx / lx)) : dx;
    dy = (ly > 0.0) ? (dy - ly * nearbyint(dy / ly)) : dy;
    return dx*dx + dy*dy;
}

__device__ __forceinline__ double disp_pbc_L(double xi, double yi,
                                             double xj, double yj,
                                             double Lx, double Ly,
                                             double Lx_inv, double Ly_inv,
                                             double& dx, double& dy) {
    dx = xj - xi; dy = yj - yi;
    // branchless nearest image
    dx = (Lx > 0.0) ? (dx - Lx * nearbyint(dx * Lx_inv)) : dx;
    dy = (Ly > 0.0) ? (dy - Ly * nearbyint(dy * Ly_inv)) : dy;
    return dx*dx + dy*dy;
}

__device__ __forceinline__ double dist2_pbc_arrays(double xi, double yi,
                                                   double xj, double yj,
                                                   int sid,
                                                   const double* __restrict__ box_size_x,
                                                   const double* __restrict__ box_size_y) {
    double dx, dy;
    return disp_pbc_arrays(xi, yi, xj, yj, sid, box_size_x, box_size_y, dx, dy);
}

__device__ __forceinline__ double dist_pbc_arrays(double xi, double yi,
                                                  double xj, double yj,
                                                  int sid,
                                                  const double* __restrict__ box_size_x,
                                                  const double* __restrict__ box_size_y) {
    return sqrt(dist2_pbc_arrays(xi, yi, xj, yj, sid, box_size_x, box_size_y));
}

// ---------- PBC (globals): displacement & distances ----------
__device__ __forceinline__ double disp_pbc_global(double xi, double yi,
                                                  double xj, double yj,
                                                  int sid,
                                                  double& dx, double& dy) {
    const double* bx = g_box.size_x;
    const double* by = g_box.size_y;
    if (!bx || !by || sid < 0 || sid >= g_sys.n_systems) {
        dx = xj - xi; dy = yj - yi;
        return dx*dx + dy*dy;
    }
    const double lx = bx[sid];
    const double ly = by[sid];
    dx = xj - xi; dy = yj - yi;
    dx = (lx > 0.0) ? (dx - lx * nearbyint(dx / lx)) : dx;
    dy = (ly > 0.0) ? (dy - ly * nearbyint(dy / ly)) : dy;
    return dx*dx + dy*dy;
}

__device__ __forceinline__ double dist2_pbc_global(double xi, double yi,
                                                   double xj, double yj,
                                                   int sid) {
    double dx, dy;
    return disp_pbc_global(xi, yi, xj, yj, sid, dx, dy);
}

__device__ __forceinline__ double dist_pbc_global(double xi, double yi,
                                                  double xj, double yj,
                                                  int sid) {
    return sqrt(dist2_pbc_global(xi, yi, xj, yj, sid));
}

// Compute the temperature scale factor
__global__ void compute_temperature_scale_factor_kernel(
    const double* __restrict__ temperature,
    const double* __restrict__ temperature_target,
    double* __restrict__ scale_factor
);

// Scale the box size given a packing fraction increment
__global__ void scale_box_by_increment_kernel(
    const int S,
    double* __restrict__ box_size_x,
    double* __restrict__ box_size_y,
    const double* __restrict__ packing_fraction,
    const double* __restrict__ increment,
    double* __restrict__ scale_factor
);

// Common cell list kernels

// Update the static index
__global__ void update_static_index_kernel(
    const int N,
    const int* __restrict__ order_inv,
    int* __restrict__ static_index
);

// Update the cell size given the box size and the number of cells per dimension
__global__ void update_cell_size_kernel(
    const int S,
    const double* __restrict__ box_size_x,
    const double* __restrict__ box_size_y,
    const int* __restrict__ cell_dim_x,
    const int* __restrict__ cell_dim_y,
    double* __restrict__ cell_size_x,
    double* __restrict__ cell_size_y,
    double* __restrict__ cell_inv_x,
    double* __restrict__ cell_inv_y
);

// Assign the cell ids to the particles based on their positions
__global__ void assign_cell_ids_kernel(
    const int N,
    const int* __restrict__ system_id,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int* __restrict__ cell_id);

// Count the number of particles in each cell
__global__ void count_cells_kernel(
    int N,
    const int* __restrict__ cell_id,
    int* __restrict__ counts);

// Scatter the particle order based on the cell ids
__global__ void scatter_order_kernel(
    const int N,
    const int* __restrict__ cell_id,
    int* __restrict__ write_ptr,
    int* __restrict__ order,
    int* __restrict__ order_inv);


}} // namespace md::geo