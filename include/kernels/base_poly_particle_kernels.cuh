#pragma once

#include "kernels/base_particle_kernels.cuh"
#include <stdio.h>

namespace md::poly {

// Global constants for all poly particles
struct PolyConst {
    const int* particle_id;
    const int* particle_offset;
    const int* n_vertices_per_particle;
    const int* static_particle_order;
};

struct PolySystemConst {
    const int*    offset;  // length = n_systems+1
    const int*    id;      // length = n_vertices
    const int*    size;    // length = n_systems
};

// Global constants for all poly particles
extern __constant__ PolyConst g_poly;
extern __constant__ PolySystemConst g_vertex_sys;

// Helper for binding the poly class globals
__host__ void bind_poly_globals(const int* d_particle_id, const int* d_particle_offset, const int* d_n_vertices_per_particle, const int* d_static_particle_order);

// Helper for binding the poly system constants
__host__ void bind_poly_system_globals(const int* d_vertex_system_offset, const int* d_vertex_system_id, const int* d_vertex_system_size);

// Build the static particle order list
__global__ void build_static_particle_order_kernel(
    const int* __restrict__ order_inv,
    int* __restrict__ static_particle_order
);

// Count the number of naive vertex neighbors
__global__ void count_naive_vertex_neighbors_kernel(
    int* __restrict__ neighbor_count
);

// Fill the naive vertex neighbor list
__global__ void fill_naive_vertex_neighbor_list_kernel(
    const int* __restrict__ neighbor_start,
    int* __restrict__ neighbor_ids
);

// Build particle-level pair keys (primary, neighbor)
__global__ void fill_particle_neighbor_pair_keys_kernel(
    unsigned long long* __restrict__ pair_keys
);

// Count the number of vertices in each cell
__global__ void count_vertex_cell_neighbors_kernel(
    const double* __restrict__ vertex_pos_x,
    const double* __restrict__ vertex_pos_y,
    const double* __restrict__ vertex_rad,
    const int* __restrict__ vertex_cell_id,
    const int* __restrict__ cell_start,
    int* __restrict__ neighbor_count
);

// Fill the neighbor list for each vertex by enumerating over the 9-cell stencil
__global__ void fill_vertex_cell_neighbor_list_kernel(
    const double* __restrict__ vertex_pos_x,
    const double* __restrict__ vertex_pos_y,
    const double* __restrict__ vertex_rad,
    const int* __restrict__ vertex_cell_id,
    const int* __restrict__ cell_start,
    const int* __restrict__ neighbor_start,
    int* __restrict__ neighbor_ids
);

}
