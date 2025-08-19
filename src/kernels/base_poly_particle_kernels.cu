#include "kernels/base_poly_particle_kernels.cuh"
#include "stdio.h"

namespace md::poly {

__constant__ PolyConst g_poly;
__constant__ PolySystemConst g_vertex_sys;

void bind_poly_globals(const int* d_particle_id, const int* d_particle_offset, const int* d_n_vertices_per_particle) {
    PolyConst h { d_particle_id, d_particle_offset, d_n_vertices_per_particle };
    cudaMemcpyToSymbol(g_poly, &h, sizeof(PolyConst));
}
void bind_poly_system_globals(const int* d_vertex_system_offset, const int* d_vertex_system_id, const int* d_vertex_system_size) {
    PolySystemConst h { d_vertex_system_offset, d_vertex_system_id, d_vertex_system_size };
    cudaMemcpyToSymbol(g_vertex_sys, &h, sizeof(PolySystemConst));
}

__global__ void count_naive_vertex_neighbors_kernel(
    int* __restrict__ neighbor_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Nv = md::geo::g_sys.n_vertices;
    if (i >= Nv) return;

    const int v_sid = g_vertex_sys.id[i];
    const int v_sys_size = g_vertex_sys.size[v_sid];
    const int v_pid = g_poly.particle_id[i];
    const int n_vertices_in_particle = g_poly.n_vertices_per_particle[v_pid];
    // if there are N_v_t total vertices in the system, a vertex of particle i has a maximum of (N_v_t - N_v_i) vertex neighbors
    const int n_neighbors = v_sys_size - n_vertices_in_particle;
    neighbor_count[i] = n_neighbors;
}

__global__ void fill_naive_vertex_neighbor_list_kernel(
    const int* __restrict__ neighbor_start,
    int* __restrict__ neighbor_ids
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Nv = md::geo::g_sys.n_vertices;
    if (i >= Nv) return;

    const int v_sid = g_vertex_sys.id[i];
    const int v_sys_beg = g_vertex_sys.offset[v_sid];
    const int v_sys_end = g_vertex_sys.offset[v_sid+1];
    const int v_pid = g_poly.particle_id[i];

    int neigh_pid;

    int w = neighbor_start[i];
    for (int j = v_sys_beg; j < v_sys_end; ++j) {
        neigh_pid = g_poly.particle_id[j];
        if (neigh_pid == v_pid) continue;
        neighbor_ids[w++] = j;
    }
}

}

