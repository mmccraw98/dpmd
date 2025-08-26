#include <cuda_runtime.h>
#include "kernels/base_particle_kernels.cuh"
#include "utils/cuda_debug.hpp"

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

}} // namespace md::geo