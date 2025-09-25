#include "kernels/base_poly_particle_kernels.cuh"
#include "stdio.h"

namespace md::poly {

__constant__ PolyConst g_poly;
__constant__ PolySystemConst g_vertex_sys;

void bind_poly_globals(const int* d_particle_id, const int* d_particle_offset, const int* d_n_vertices_per_particle, const int* d_static_particle_order) {
    PolyConst h { d_particle_id, d_particle_offset, d_n_vertices_per_particle, d_static_particle_order };
    cudaMemcpyToSymbol(g_poly, &h, sizeof(PolyConst));
}
void bind_poly_system_globals(const int* d_vertex_system_offset, const int* d_vertex_system_id, const int* d_vertex_system_size) {
    PolySystemConst h { d_vertex_system_offset, d_vertex_system_id, d_vertex_system_size };
    cudaMemcpyToSymbol(g_vertex_sys, &h, sizeof(PolySystemConst));
}

__global__ void build_static_particle_order_kernel(
    const int* __restrict__ order_inv,
    int* __restrict__ static_particle_order
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = md::geo::g_sys.n_particles;
    if (i >= N) return;
    for (int k = g_poly.particle_offset[i]; k < g_poly.particle_offset[i + 1]; k++) {
        const int old_vertex_index = static_particle_order[k];
        static_particle_order[k] = order_inv[old_vertex_index];
    }
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

__global__ void fill_particle_neighbor_pair_keys_kernel(
    unsigned long long* __restrict__ pair_keys
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Nv = md::geo::g_sys.n_vertices;
    if (i >= Nv) return;

    const int pid = md::poly::g_poly.particle_id[i];
    const int beg = md::geo::g_neigh.start[i];
    const int end = md::geo::g_neigh.start[i+1];

    for (int k = beg; k < end; ++k) {
        const int j = md::geo::g_neigh.ids[k];
        const int neighbor_pid = md::poly::g_poly.particle_id[j];
        const unsigned long long key =
            (static_cast<unsigned long long>(static_cast<unsigned int>(pid)) << 32) |
            static_cast<unsigned int>(neighbor_pid);
        pair_keys[k] = key;
    }
}

__global__ void count_vertex_cell_neighbors_kernel(
    const double* __restrict__ vertex_pos_x,
    const double* __restrict__ vertex_pos_y,
    const double* __restrict__ vertex_rad,
    const int* __restrict__ vertex_cell_id,
    const int* __restrict__ cell_start,
    int* __restrict__ neighbor_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Nv = md::geo::g_sys.n_vertices;
    if (i >= Nv) return;

    const int cid = vertex_cell_id[i];
    const int pid = md::poly::g_poly.particle_id[i];
    const int sid = md::poly::g_vertex_sys.id[i];
    const int cell_dim_x = md::geo::g_cell.dim_x[sid];
    const int cell_dim_y = md::geo::g_cell.dim_y[sid];
    const int cell_sys_start = md::geo::g_cell.sys_start[sid];

    const int local_cid = cid - cell_sys_start;
    const int cid_x = local_cid % cell_dim_x;
    const int cid_y = local_cid / cell_dim_x;

    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];
    const double box_inv_x = md::geo::g_box.inv_x[sid];
    const double box_inv_y = md::geo::g_box.inv_y[sid];
    const double cutoff = md::geo::g_neigh.cutoff[sid];

    const double xi = vertex_pos_x[i], yi = vertex_pos_y[i], ri = vertex_rad[i];

    int count = 0;

    // 3x3 stencil, wrap with PBC
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        int yy = cid_y + dy; if (yy < 0) yy += cell_dim_y; else if (yy >= cell_dim_y) yy -= cell_dim_y;
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = cid_x + dx; if (xx < 0) xx += cell_dim_x; else if (xx >= cell_dim_x) xx -= cell_dim_x;

            const int ncell = cell_sys_start + (yy * cell_dim_x + xx);
            const int beg = cell_start[ncell];
            const int end = cell_start[ncell + 1];

            for (int j = beg; j < end; ++j) {
                const int j_pid = md::poly::g_poly.particle_id[j];
                if (j == i || j_pid == pid) continue;
                const double xj = vertex_pos_x[j], yj = vertex_pos_y[j], rj = vertex_rad[j];
                double dxp, dyp;
                const double r2 = md::geo::disp_pbc_L(xi, yi, xj, yj, box_size_x, box_size_y, box_inv_x, box_inv_y, dxp, dyp);
                const double cut = (ri + rj + cutoff);
                if (r2 < cut * cut) ++count;
            }
        }
    }
    neighbor_count[i] = count;
}

__global__ void fill_vertex_cell_neighbor_list_kernel(
    const double* __restrict__ vertex_pos_x,
    const double* __restrict__ vertex_pos_y,
    const double* __restrict__ vertex_rad,
    const int* __restrict__ vertex_cell_id,
    const int* __restrict__ cell_start,
    const int* __restrict__ neighbor_start,
    int* __restrict__ neighbor_ids
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int Nv = md::geo::g_sys.n_vertices;
    if (i >= Nv) return;

    const int cid = vertex_cell_id[i];
    const int pid = md::poly::g_poly.particle_id[i];
    const int sid = md::poly::g_vertex_sys.id[i];
    const int cell_dim_x = md::geo::g_cell.dim_x[sid];
    const int cell_dim_y = md::geo::g_cell.dim_y[sid];
    const int cell_sys_start = md::geo::g_cell.sys_start[sid];

    const int local_cid = cid - cell_sys_start;
    const int cid_x = local_cid % cell_dim_x;
    const int cid_y = local_cid / cell_dim_x;

    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];
    const double box_inv_x = md::geo::g_box.inv_x[sid];
    const double box_inv_y = md::geo::g_box.inv_y[sid];
    const double cutoff = md::geo::g_neigh.cutoff[sid];

    const double xi = vertex_pos_x[i], yi = vertex_pos_y[i], ri = vertex_rad[i];

    int w = neighbor_start[i];

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        int yy = cid_y + dy; if (yy < 0) yy += cell_dim_y; else if (yy >= cell_dim_y) yy -= cell_dim_y;
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = cid_x + dx; if (xx < 0) xx += cell_dim_x; else if (xx >= cell_dim_x) xx -= cell_dim_x;

            const int ncell = cell_sys_start + (yy * cell_dim_x + xx);
            const int beg = cell_start[ncell];
            const int end = cell_start[ncell + 1];

            for (int j = beg; j < end; ++j) {
                const int j_pid = md::poly::g_poly.particle_id[j];
                if (j == i || j_pid == pid) continue;
                const double xj = vertex_pos_x[j], yj = vertex_pos_y[j], rj = vertex_rad[j];
                double dxp, dyp;
                const double r2 = md::geo::disp_pbc_L(xi, yi, xj, yj, box_size_x, box_size_y, box_inv_x, box_inv_y, dxp, dyp);
                const double cut = (ri + rj + cutoff);
                if (r2 < cut * cut) neighbor_ids[w++] = j;
            }
        }
    }
}

}
