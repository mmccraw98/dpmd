#include "kernels/base_poly_particle_kernels.cuh"

namespace md::poly {

__constant__ PolyConst g_poly;

void bind_poly_globals(const int* d_vertex_particle_id, const int* d_particle_offset, const int* d_vertex_system_id, const int* d_vertex_system_offset) {
    PolyConst h { d_vertex_particle_id, d_particle_offset, d_vertex_system_id, d_vertex_system_offset };
    cudaMemcpyToSymbol(g_poly, &h, sizeof(PolyConst));
}

__global__ void count_naive_vertex_neighbors_kernel(
    int* __restrict__ neighbor_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Nv = 
    if (i >= Nv) return;

    const int sys_begin = md::geo::g_sys.off

    // const int pid = g_poly.particle_id[i];
    // const int offset = g_poly.particle_offset[pid];
    // const int n_neighbors = g_poly.particle_offset[pid+1] - offset;
    // neighbor_count[i] = n_neighbors;
}

}

