#include "kernels/base_point_particle_kernels.cuh"
#include <cstdio>

namespace md::point {

__global__ void set_naive_neighbor_count(
    int* __restrict__ neighbor_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = geo::g_sys.n_particles;
    if (i >= N) return;

    const int sid = geo::g_sys.id[i];
    const int n_neighbors = (geo::g_sys.offset[sid+1] - geo::g_sys.offset[sid]) - 1;
    neighbor_count[i] = n_neighbors;
}

__global__ void fill_naive_neighbor_list_kernel(
    const int* __restrict__ neighbor_start,
    int*       __restrict__ neighbor_ids
) {
    const int N = geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int sid   = geo::g_sys.id[i];
    const int begin = geo::g_sys.offset[sid];
    const int end   = geo::g_sys.offset[sid+1];

    int w = neighbor_start[i];
    for (int j = begin; j < end; ++j) {
        if (j == i) continue;
        neighbor_ids[w++] = j;
    }
}


__global__ void assign_cell_ids_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    int* __restrict__ cell_id) 
{
    const int N = geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int sid = geo::g_sys.id[i];

    const double box_inv_x  = geo::g_box.inv_x[sid];
    const double box_inv_y  = geo::g_box.inv_y[sid];

    const int cell_dim_x     = geo::g_cell.dim_x[sid];
    const int cell_dim_y     = geo::g_cell.dim_y[sid];

    const double xi = x[i], yi = y[i];

    double u = geo::wrap01(xi * box_inv_x);
    double v = geo::wrap01(yi * box_inv_y);

    int cell_id_x = (int)floor(u * cell_dim_x);
    int cell_id_y = (int)floor(v * cell_dim_y);

    int local_cell_id = cell_id_x + cell_id_y * cell_dim_x;
    cell_id[i] = local_cell_id + geo::g_cell.sys_start[sid];
}

__global__ void count_cells_kernel(const int* __restrict__ cell_id,
                                   int N,
                                   int* __restrict__ counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cid = cell_id[i];
    if (cid >= 0) atomicAdd(&counts[cid], 1);
}

__global__ void scatter_order_kernel(const int* __restrict__ cell_id,
                                     int N,
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


__global__ void count_cell_neighbors_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ rad,
    const int*    __restrict__ cell_id,
    const int*    __restrict__ cell_start,
    int*          __restrict__ neighbor_count
){
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int my_cell        = cell_id[i];
    const int sid            = md::geo::g_sys.id[i];
    const int cell_dim_x     = md::geo::g_cell.dim_x[sid];
    const int cell_dim_y     = md::geo::g_cell.dim_y[sid];
    const int cell_sys_start = md::geo::g_cell.sys_start[sid];

    // Decode local (cx, cy) from global cell id
    const int local_cell_id = my_cell - cell_sys_start;
    const int cell_id_x     = local_cell_id % cell_dim_x;
    const int cell_id_y     = local_cell_id / cell_dim_x;

    const double box_size_x   = md::geo::g_box.size_x[sid];
    const double box_size_y   = md::geo::g_box.size_y[sid];
    const double box_inv_x    = md::geo::g_box.inv_x[sid];
    const double box_inv_y    = md::geo::g_box.inv_y[sid];
    const double skin         = md::geo::g_neigh.skin[sid];

    const double xi = x[i], yi = y[i], ri = rad[i];

    int count = 0;

    // 3x3 stencil, wrap with PBC
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        int yy = cell_id_y + dy; if (yy < 0) yy += cell_dim_y; else if (yy >= cell_dim_y) yy -= cell_dim_y;
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = cell_id_x + dx; if (xx < 0) xx += cell_dim_x; else if (xx >= cell_dim_x) xx -= cell_dim_x;

            const int ncell = cell_sys_start + (yy * cell_dim_x + xx);
            const int beg   = cell_start[ncell];
            const int end   = cell_start[ncell + 1];

            for (int p = beg; p < end; ++p) {
                const int j  = p;
                if (j == i) continue;

                const double xj = x[j], yj = y[j], rj = rad[j];

                double dxp, dyp;
                const double r2 = md::geo::disp_pbc_L(xi, yi, xj, yj, box_size_x, box_size_y, box_inv_x, box_inv_y, dxp, dyp);

                const double cut = (ri + rj + skin);
                if (r2 < cut * cut) ++count;
            }
        }
    }
    neighbor_count[i] = count;
    
    #if MD_DEBUG
    if (count > 1000) {  // Warn about unusually high neighbor counts
        printf("WARNING: particle %d has %d neighbors (very high)\n", i, count);
    }
    if (count < 0) {
        printf("ERROR: particle %d has negative neighbor count %d\n", i, count);
    }
    #endif
}

__global__ void fill_neighbors_cell_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ rad,
    const int*    __restrict__ cell_id,
    const int*    __restrict__ cell_start,
    const int*    __restrict__ neighbor_start,
    int*          __restrict__ neighbor_ids
){
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int my_cell = cell_id[i];
    const int sid     = md::geo::g_sys.id[i];
    const int cell_dim_x  = md::geo::g_cell.dim_x[sid];
    const int cell_dim_y  = md::geo::g_cell.dim_y[sid];
    const int cell_sys_start = md::geo::g_cell.sys_start[sid];

    const int local_cell_id = my_cell - cell_sys_start;
    const int cell_id_x     = local_cell_id % cell_dim_x;
    const int cell_id_y     = local_cell_id / cell_dim_x;

    const double box_size_x   = md::geo::g_box.size_x[sid];
    const double box_size_y   = md::geo::g_box.size_y[sid];
    const double box_inv_x    = md::geo::g_box.inv_x[sid];
    const double box_inv_y    = md::geo::g_box.inv_y[sid];
    const double skin         = md::geo::g_neigh.skin[sid];

    const double xi = x[i], yi = y[i], ri = rad[i];

    int w = neighbor_start[i];

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        int yy = cell_id_y + dy; if (yy < 0) yy += cell_dim_y; else if (yy >= cell_dim_y) yy -= cell_dim_y;
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = cell_id_x + dx; if (xx < 0) xx += cell_dim_x; else if (xx >= cell_dim_x) xx -= cell_dim_x;

            const int ncell = cell_sys_start + (yy * cell_dim_x + xx);
            const int beg   = cell_start[ncell];
            const int end   = cell_start[ncell + 1];

            for (int p = beg; p < end; ++p) {
                const int j = p;
                if (j == i) continue;

                const double xj = x[j], yj = y[j], rj = rad[j];

                double dxp, dyp;
                const double r2 = md::geo::disp_pbc_L(xi, yi, xj, yj, box_size_x, box_size_y, box_inv_x, box_inv_y, dxp, dyp);

                const double cut = (ri + rj + skin);
                if (r2 < cut * cut) {
                    neighbor_ids[w++] = j;
                }
            }
        }
    }
}

}