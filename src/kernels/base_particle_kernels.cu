#include <cuda_runtime.h>
#include "kernels/base_particle_kernels.cuh"
#include "utils/cuda_utils.cuh"

namespace md { namespace geo {

__constant__ BoxConst g_box;
__constant__ SystemConst g_sys;
__constant__ NeighborConst g_neigh;
__constant__ CellConst g_cell;

void bind_box_globals(const double* d_box_size_x,
                      const double* d_box_size_y,
                      const double* d_box_inv_x,
                      const double* d_box_inv_y) {
    BoxConst h { d_box_size_x, d_box_size_y, d_box_inv_x, d_box_inv_y };
    cudaMemcpyToSymbol(g_box, &h, sizeof(BoxConst));
}

void bind_system_globals(const int* d_system_offset,
                         const int* d_system_id,
                         int n_systems,
                         int n_particles,
                         int n_vertices) {
    SystemConst h { d_system_offset, d_system_id, n_systems, n_particles, n_vertices };
    cudaMemcpyToSymbol(g_sys, &h, sizeof(SystemConst));
}

void bind_neighbor_globals(const int* d_neighbor_start,
                           const int* d_neighbor_ids,
                           const double* d_verlet_skin,
                           const double* d_thresh2) {
    NeighborConst h { d_neighbor_start, d_neighbor_ids, d_verlet_skin, d_thresh2 };
    cudaMemcpyToSymbol(g_neigh, &h, sizeof(NeighborConst));
}

void bind_cell_globals(const double* d_cell_size_x,
                       const double* d_cell_size_y,
                       const double* d_cell_inv_x,
                       const double* d_cell_inv_y,
                       const int* d_cell_dim_x,
                       const int* d_cell_dim_y,
                       const int* d_cell_system_start) {
    CellConst h { d_cell_size_x, d_cell_size_y, d_cell_inv_x, d_cell_inv_y,
                  d_cell_dim_x, d_cell_dim_y, d_cell_system_start };
    cudaMemcpyToSymbol(g_cell, &h, sizeof(CellConst));
}

__global__ void calculate_box_inv_kernel(
    const double* __restrict__ box_size_x,
    const double* __restrict__ box_size_y,
    double*       __restrict__ box_inv_x,
    double*       __restrict__ box_inv_y,
    int S
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;

    const double lx = box_size_x[s];
    const double ly = box_size_y[s];

    box_inv_x[s] = (lx > 0.0) ? (1.0 / lx) : 0.0;
    box_inv_y[s] = (ly > 0.0) ? (1.0 / ly) : 0.0;
}

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
    int*          __restrict__ ncell_out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;

    const double lx = box_size_x[s];
    const double ly = box_size_y[s];
    const int    nx = cell_dim_x[s];
    const int    ny = cell_dim_y[s];

    // cell sizes and inverses (guard nx/ny)
    const double csx = (nx > 0) ? (lx / nx) : 0.0;
    const double csy = (ny > 0) ? (ly / ny) : 0.0;

    cell_size_x[s]   = csx;
    cell_size_y[s]   = csy;
    cell_inv_x[s]    = (csx > 0.0) ? (1.0 / csx) : 0.0;
    cell_inv_y[s]    = (csy > 0.0) ? (1.0 / csy) : 0.0;

    // per-system number of cells
    ncell_out[s] = nx * ny;
}

// Kernel to compute the fractional packing fraction for each particle in the system
__global__ void compute_fractional_packing_fraction_kernel(
    const double* __restrict__ area,
    double* __restrict__ packing_fraction_per_particle
) {
    const int N = g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    const int sid = g_sys.id[i];
    const double box_area = g_box.size_x[sid] * g_box.size_y[sid];
    packing_fraction_per_particle[i] = area[i] / box_area;
}

__global__ void compute_temperature_kernel(
    const double* __restrict__ ke_total,
    const int*    __restrict__ n_dof,
    double*       __restrict__ temperature
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = g_sys.n_systems;
    if (s >= S) return;
    temperature[s] = ke_total[s] * 2.0 / (n_dof[s]);
}

__global__ void assign_cell_ids_kernel(
    const int N,
    const int* __restrict__ system_id,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int* __restrict__ cell_id) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int sid = system_id[i];

    const double box_inv_x  = g_box.inv_x[sid];
    const double box_inv_y  = g_box.inv_y[sid];

    const int cell_dim_x     = g_cell.dim_x[sid];
    const int cell_dim_y     = g_cell.dim_y[sid];

    const double xi = x[i], yi = y[i];

    double u = geo::wrap01(xi * box_inv_x);
    double v = geo::wrap01(yi * box_inv_y);

    int cell_id_x = (int)floor(u * cell_dim_x);
    int cell_id_y = (int)floor(v * cell_dim_y);

    int local_cell_id = cell_id_x + cell_id_y * cell_dim_x;
    cell_id[i] = local_cell_id + geo::g_cell.sys_start[sid];
}

__global__ void count_cells_kernel(
    int N,
    const int* __restrict__ cell_id,
    int* __restrict__ counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cid = cell_id[i];
    if (cid >= 0) atomicAdd(&counts[cid], 1);
}

__global__ void scatter_order_kernel(
    const int N,
    const int* __restrict__ cell_id,
    int* __restrict__ write_ptr,
    int* __restrict__ order,
    int* __restrict__ order_inv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cid = cell_id[i];
    int dst = atomicAdd(&write_ptr[cid], 1);
    order[dst] = i;
    if (order_inv) order_inv[i] = dst;
}

}} // namespace md::geo