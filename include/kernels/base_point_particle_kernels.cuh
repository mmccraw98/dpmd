#pragma once

#include "kernels/base_particle_kernels.cuh"

namespace md::point {

__global__ void set_naive_neighbor_count(
    int* __restrict__ neighbor_count
);

__global__ void fill_naive_neighbor_list_kernel(
    const int* __restrict__ neighbor_start,
    int*       __restrict__ neighbor_ids
);

__global__ void assign_cell_ids_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    int* __restrict__ cell_id
);

__global__ void count_cells_kernel(const int* __restrict__ cell_id,
                                   int N,
                                   int* __restrict__ counts);

__global__ void scatter_order_kernel(const int* __restrict__ cell_id,
                                     int N,
                                     int* __restrict__ write_ptr,
                                     int* __restrict__ order,
                                     int* __restrict__ order_inv);

// Count the number of neighbors for each particle by enumerating over the 9-cell stencil (particle's cell and 8 surrounding cells)
__global__ void count_cell_neighbors_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ rad,
    const int*    __restrict__ cell_id,
    const int*    __restrict__ cell_start,
    int*          __restrict__ neighbor_count
);

// Fill the neighbor list for each particle by enumerating over the 9-cell stencil (particle's cell and 8 surrounding cells)
__global__ void fill_neighbors_cell_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ rad,
    const int*    __restrict__ cell_id,
    const int*    __restrict__ cell_start,
    const int*    __restrict__ neighbor_start,
    int*          __restrict__ neighbor_ids
);

}