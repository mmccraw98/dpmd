#include <cuda_runtime.h>
#include "kernels/common.cuh"

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
                         int n_particles) {
    SystemConst h { d_system_offset, d_system_id, n_systems, n_particles };
    cudaMemcpyToSymbol(g_sys, &h, sizeof(SystemConst));
}

void bind_neighbor_globals(const int* d_neighbor_start,
                           const int* d_neighbor_ids,
                           const double* d_verlet_skin) {
    NeighborConst h { d_neighbor_start, d_neighbor_ids, d_verlet_skin };
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

}} // namespace md::geo