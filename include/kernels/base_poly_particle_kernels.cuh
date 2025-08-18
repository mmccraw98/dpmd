#pragma once

#include "kernels/common.cuh"

namespace md::poly {

// Global constants for all poly particles
struct PolyConst {
    const int* particle_id;
    const int* particle_offset;
    const int* vertex_system_id;
    const int* vertex_system_offset;
};

// Global constants for all poly particles
extern __constant__ PolyConst g_poly;

// Helper for binding the poly globals
__host__ void bind_poly_globals(const int* d_vertex_particle_id, const int* d_particle_offset, const int* d_vertex_system_id, const int* d_vertex_system_offset);

// Count the number of naive vertex neighbors
__global__ void count_naive_vertex_neighbors_kernel(
    int* __restrict__ neighbor_count
);


}