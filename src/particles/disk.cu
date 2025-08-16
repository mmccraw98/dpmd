#include "particles/disk.hpp"
#include "kernels/common.cuh"
#include "utils/cuda_debug.hpp"
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>

// ---- Per-system constants ----
struct DiskConst {
    const double* e_interaction;
};

__constant__ DiskConst g_disk;

namespace {  // TU-local

__host__ void bind_disk_globals(const double* d_e_interaction) {
    DiskConst h { d_e_interaction };
    cudaMemcpyToSymbol(g_disk, &h, sizeof(DiskConst));
}

struct CountMinusOneClamp {
    __host__ __device__ int operator()(int m) const {
        int c = m - 1;
        return (c > 0) ? c : 0;
    }
};

__global__ void fill_naive_neighbor_list_kernel(
    const int* __restrict__ neighbor_start,     // N+1
    int*       __restrict__ neighbor_ids        // total
){
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int sid   = md::geo::g_sys.id[i];
    const int begin = md::geo::g_sys.offset[sid];
    const int end   = md::geo::g_sys.offset[sid+1];

    int w = neighbor_start[i];
    for (int j = begin; j < end; ++j) {
        if (j == i) continue;
        neighbor_ids[w++] = j;
    }
}

__global__ void disk_force_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ rad,
          double* __restrict__ fx,
          double* __restrict__ fy,
          double* __restrict__ pe)
{
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int sid = md::geo::g_sys.id[i];
    const double e_i = g_disk.e_interaction[sid];
    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];
    const double box_inv_x = md::geo::g_box.inv_x[sid];
    const double box_inv_y = md::geo::g_box.inv_y[sid];

    const double xi = x[i], yi = y[i], ri = rad[i];
    double fxi = 0.0, fyi = 0.0, pei = 0.0;

    const int beg = md::geo::g_neigh.start[i];
    const int end = md::geo::g_neigh.start[i+1];

    for (int k = beg; k < end; ++k) {
        const int j = md::geo::g_neigh.ids[k];
        const double xj = x[j], yj = y[j], rj = rad[j];

        double dx, dy;
        double r2 = md::geo::disp_pbc_L(xi, yi, xj, yj, box_size_x, box_size_y, box_inv_x, box_inv_y, dx, dy);

        // Early reject if no overlap: r^2 >= (ri+rj)^2
        const double radsum = ri + rj;
        const double radsum2 = radsum * radsum;
        if (r2 >= radsum2) continue;

        // Overlap: compute r and invr once
        const double r   = sqrt(r2);
        const double inv = 1.0 / r;
        const double nx  = dx * inv;
        const double ny  = dy * inv;

        const double delta = radsum - r;
        const double fmag  = e_i * delta;

        // Force on i is along -n (repulsion)
        fxi -= fmag * nx;
        fyi -= fmag * ny;

        // Single-count the pair energy (each pair gets half)
        pei += (0.5 * e_i * delta * delta) * 0.5;
    }
    
    fx[i] = fxi;
    fy[i] = fyi;
    pe[i] = pei;
}

__global__ void disk_wall_force_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ rad,
    double* __restrict__ fx,
    double* __restrict__ fy,
    double* __restrict__ pe)
{
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int sid = md::geo::g_sys.id[i];
    const double e_i = g_disk.e_interaction[sid];
    const double box_size_x = md::geo::g_box.size_x[sid];
    const double box_size_y = md::geo::g_box.size_y[sid];
    const double box_inv_x = md::geo::g_box.inv_x[sid];
    const double box_inv_y = md::geo::g_box.inv_y[sid];

    const double xi = x[i], yi = y[i], ri = rad[i];
    
    double fxi = 0.0, fyi = 0.0, pei = 0.0;

    if (xi < ri) {
        const double delta = ri - xi;
        const double fmag = e_i * delta;
        fxi -= fmag;
        pei += (0.5 * e_i * delta * delta) * 0.5;
    }
    if (xi > box_size_x - ri) {
        const double delta = ri - (box_size_x - xi);
        const double fmag = e_i * delta;
        fxi -= fmag;
        pei += (0.5 * e_i * delta * delta) * 0.5;
    }
    if (yi < ri) {
        const double delta = ri - yi;
        const double fmag = e_i * delta;
        fyi -= fmag;
        pei += (0.5 * e_i * delta * delta) * 0.5;
    }
    if (yi > box_size_y - ri) {
        const double delta = ri - (box_size_y - yi);
        const double fmag = e_i * delta;
        fyi -= fmag;
        pei += (0.5 * e_i * delta * delta) * 0.5;
    }

    fx[i] += fxi;
    fy[i] += fyi;
    pe[i] += pei;
}

__global__ void init_per_system_cell_params(
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

__global__ void assign_cell_ids_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    int* __restrict__ cell_id) 
{
    const int N = md::geo::g_sys.n_particles;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int sid = md::geo::g_sys.id[i];

    const double box_inv_x  = md::geo::g_box.inv_x[sid];
    const double box_inv_y  = md::geo::g_box.inv_y[sid];

    const int cell_dim_x     = md::geo::g_cell.dim_x[sid];
    const int cell_dim_y     = md::geo::g_cell.dim_y[sid];

    const double xi = x[i], yi = y[i];

    double u = md::geo::wrap01(xi * box_inv_x);
    double v = md::geo::wrap01(yi * box_inv_y);

    int cell_id_x = (int)floor(u * cell_dim_x);
    int cell_id_y = (int)floor(v * cell_dim_y);

    int local_cell_id = cell_id_x + cell_id_y * cell_dim_x;
    cell_id[i] = local_cell_id + md::geo::g_cell.sys_start[sid];
}

__global__ void reorder_by_kernel(const double* __restrict__ x,
                                  const double* __restrict__ y,
                                  double* __restrict__ x_new,
                                  double* __restrict__ y_new,
                                  const double* __restrict__ vx,
                                  const double* __restrict__ vy,
                                  double* __restrict__ vx_new,
                                  double* __restrict__ vy_new,
                                  const double* __restrict__ rad,
                                  double* __restrict__ rad_new,
                                  const double* __restrict__ mass,
                                  double* __restrict__ mass_new,
                                  const double* __restrict__ area,
                                  double* __restrict__ area_new,
                                  const int* __restrict__ order,
                                  int* __restrict__ order_inv)
{
    const int N = md::geo::g_sys.n_particles;
    int i_new = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_new >= N) return;

    int i = order[i_new];
    x_new[i_new] = x[i];
    y_new[i_new] = y[i];
    vx_new[i_new] = vx[i];
    vy_new[i_new] = vy[i];
    rad_new[i_new] = rad[i];
    mass_new[i_new] = mass[i];
    area_new[i_new] = area[i];
    order_inv[i] = i_new;
}


__global__ void count_neighbors_cell_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ rad,
    const int*    __restrict__ cell_id,     // (N,) global cell id per particle (sorted layout)
    const int*    __restrict__ cell_start,  // (total_cells+1,) CSR into sorted particles
    int*          __restrict__ neighbor_count    // (N,) neighbor counts
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
}

__global__ void fill_neighbors_cell_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ rad,
    const int*    __restrict__ cell_id,       // (N,)
    const int*    __restrict__ cell_start,    // (total_cells+1,)
    const int*    __restrict__ neighbor_start,   // (N+1,)
    int*          __restrict__ neighbor_ids      // (total)
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
                const int j  = p;
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
    // w should end at neighbor_start[i+1] — optional assert under DEBUG
}


} // anon namespace

namespace md {

void Disk::init_naive_neighbors_impl() {
    // 1) counts[i] = system_size[ system_id[i] ] - 1
    int N = n_particles();
    neighbor_count.resize(N);
    auto sys_size_begin = system_size.data.begin();
    auto sid_begin      = system_id.data.begin();
    auto per_particle_m = thrust::make_permutation_iterator(sys_size_begin, sid_begin);

    thrust::transform(
        per_particle_m, per_particle_m + N,
        neighbor_count.data.begin(),
        CountMinusOneClamp()
    );

    // 2) exclusive scan -> neighbor_start (N+1)
    neighbor_start.resize(N + 1);
    thrust::exclusive_scan(
        neighbor_count.data.begin(), neighbor_count.data.end(),
        neighbor_start.data.begin()
    );

    // 3) total edges = sum(counts) (computed on device; result returned to host as an int)
    int total = thrust::reduce(
        neighbor_count.data.begin(), neighbor_count.data.end(),
        0, thrust::plus<int>()
    );

    // 4) set start[N] on device (no cudaMemcpy)
    thrust::fill_n(neighbor_start.data.begin() + N, 1, total);

    // 5) size neighbor_ids
    neighbor_ids.resize(total);

    // 6) fill neighbor_ids
    int B = 256;
    dim3 Bdim(B), Gdim((N + B - 1)/B);
    CUDA_LAUNCH(fill_naive_neighbor_list_kernel, Gdim, Bdim,
        neighbor_start.ptr(), neighbor_ids.ptr()
    );
}

void Disk::update_naive_neighbors_impl() {
    // do nothing
}

void Disk::init_cell_neighbors_impl() {
    const int S = n_systems();
    const int N = n_particles();
    if (S == 0 || N == 0) return;

    // 0) Resize initial arrays
    cell_size.resize(S);
    cell_inv.resize(S);

    // 1) Make sure per-system arrays are sized (you said cell_dim is provided).
    //    We'll compute: cell_size, cell_inv, box_inv and ncell per system.
    //    Temporary device vector to hold per-system ncell before scan:
    thrust::device_vector<int> d_ncell(S, 0);

    // 2) Compute per-system params in one pass
    {
        const int B = 256;
        dim3 Bdim(B), Gdim((S + B - 1)/B);
        CUDA_LAUNCH(init_per_system_cell_params, Gdim, Bdim,
            S,
            box_size.xptr(), box_size.yptr(),
            cell_dim.xptr(),  cell_dim.yptr(),
            cell_size.xptr(), cell_size.yptr(),
            cell_inv.xptr(),  cell_inv.yptr(),
            thrust::raw_pointer_cast(d_ncell.data())
        );
    }

    // 3) Build system→cell offset (S+1). You called this "cell_system_start".
    //    NOTE: This is SIZE S+1 (not total_cells+1). The "+1" gets total cells at tail.
    cell_system_start.resize(S + 1);

    // Exclusive scan d_ncell -> first S entries of cell_system_start
    thrust::exclusive_scan(d_ncell.begin(), d_ncell.end(), cell_system_start.data.begin());

    // Fetch total_cells and write tail
    int total_cells = thrust::reduce(d_ncell.begin(), d_ncell.end(), 0, thrust::plus<int>());
    thrust::fill_n(cell_system_start.data.begin() + S, 1, total_cells);

    // 4) Size per-cell CSR buffers and per-particle aux
    cell_count.resize(total_cells);       // will be filled in update_build
    cell_start.resize(total_cells + 1);   // CSR rows for particles per cell
    cell_id.resize(N);                    // per-particle cell ids
    order.resize(N); order_inv.resize(N);
    neighbor_count.resize(N);
    neighbor_start.resize(N + 1);

    // 5) Initialize skin bookkeeping baseline
    last_pos.resize(N);
    disp2.resize(N);
    last_pos.copy_from(pos);
    disp2.fill(0.0, 0.0);

    std::cout << "init_cell_neighbors_impl" << std::endl;
}

void Disk::update_cell_neighbors_impl() {
    sync_cells();
    std::cout << "--1" << std::endl;
    // ---- 0) Assign cell ids ----
    const int S = n_systems();
    const int N = n_particles();
    const int B = 256;
    dim3 Bdim(B), Gdim((N + B - 1)/B);
    CUDA_LAUNCH(assign_cell_ids_kernel, Gdim, Bdim,
        pos.xptr(), pos.yptr(),
        cell_id.ptr()
    );
    std::cout << "--2" << std::endl;
    std::cout << "cell_system_start: " << cell_system_start.get_element(S) << std::endl;
    if (cell_id.size() != N || order.size() != N) {
    throw std::runtime_error("size mismatch: cell_id/order != N");
    }
    int _total_cells = cell_system_start.get_element(n_systems());
    int min_cid = *thrust::min_element(cell_id.data.begin(), cell_id.data.end());
    int max_cid = *thrust::max_element(cell_id.data.begin(), cell_id.data.end());
    std::cout << "cell_id range: [" << min_cid << ", " << max_cid << ") of total_cells=" << _total_cells << "\n";

    // 1) order := [0..N)
    thrust::sequence(order.data.begin(), order.data.end(), 0);
    std::cout << "--3" << std::endl;
    // 2) sort by cell id
    thrust::sort_by_key(
        cell_id.data.begin(), cell_id.data.end(),
        order.data.begin()
    );
    std::cout << "--4" << std::endl;

    // ---- 3) Apply permutation to SoA ----
    {
        const int B = 256;
        dim3 Bdim(B), Gdim((N + B - 1)/B);
        CUDA_LAUNCH(reorder_by_kernel, Gdim, Bdim,
            pos.xptr(), pos.yptr(),
            pos.xptr_swap(), pos.yptr_swap(),
            vel.xptr(), vel.yptr(),
            vel.xptr_swap(), vel.yptr_swap(),
            rad.ptr(),
            rad.ptr_swap(),
            mass.ptr(),
            mass.ptr_swap(),
            area.ptr(),
            area.ptr_swap(),
            order.ptr(), 
            order_inv.ptr()
        );
        pos.swap();
        vel.swap();
        rad.swap();
        mass.swap();
        area.swap();
    }
    std::cout << "--5" << std::endl;
    // ---- 4) Build cell_count and cell_start on the *sorted* cell_id ----
    int total_cells = cell_system_start.get_element(S);
    std::cout << "--6" << std::endl;
    auto unique_cells_out = order.data.begin();            // int*
    auto counts_out       = neighbor_count.data.begin();   // int*

    // reduce_by_key over sorted cell_id with values=1
    auto ones = thrust::make_constant_iterator<int>(1);
    auto rbk_end = thrust::reduce_by_key(
        cell_id.data.begin(), cell_id.data.end(),   // sorted keys
        ones,                                       // values
        unique_cells_out,                           // out: unique cell ids (R entries)
        counts_out                                  // out: counts per unique cell (R entries)
    );
    std::cout << "--7" << std::endl;
    // R = number of non-empty cells encountered in this system layout
    int R = static_cast<int>(rbk_end.first - unique_cells_out);
    std::cout << "--8" << std::endl;
    // Zero the full cell_count[total_cells], then scatter the R non-empty results
    cell_count.resize(total_cells);
    thrust::fill(cell_count.data.begin(), cell_count.data.end(), 0);
    std::cout << "--9" << std::endl;
    thrust::scatter(counts_out, counts_out + R,
                    unique_cells_out,        // indices are global cell ids
                    cell_count.data.begin());
    std::cout << "--10" << std::endl;
    // Exclusive scan → cell_start (size total_cells+1)
    cell_start.resize(total_cells + 1);
    thrust::exclusive_scan(cell_count.data.begin(), cell_count.data.end(),
                           cell_start.data.begin());
    std::cout << "--11" << std::endl;
    // terminal element is N (in the sorted particle indexing)
    thrust::fill_n(cell_start.data.begin() + total_cells, 1, N);
    std::cout << "--12" << std::endl;   
    // ---- 5) Build neighbor list ----
    sync_cells();
    build_neighbor_list();
    std::cout << "update_cell_neighbors_impl" << std::endl;
}

void Disk::build_neighbor_list() {
    const int N = n_particles();
    {
        const int B = 256;
        dim3 Bdim(B), Gdim((N + B - 1) / B);
        CUDA_LAUNCH(count_neighbors_cell_kernel, Gdim, Bdim,
            pos.xptr(), pos.yptr(), rad.ptr(),
            cell_id.ptr(), cell_start.ptr(),
            neighbor_count.ptr()
        );
    }
    // ---- Scan -> neighbor_start ----
    thrust::exclusive_scan(
        neighbor_count.data.begin(), neighbor_count.data.end(),
        neighbor_start.data.begin()
    );

    // tail
    int total = neighbor_start.get_element(N-1) + neighbor_count.get_element(N-1);
    neighbor_ids.resize(total);

    thrust::fill_n(neighbor_start.data.begin() + N, 1, total);

    // ---- Pass 2: fill ids ----
    {
        const int B = 256;
        dim3 Bdim(B), Gdim((N + B - 1) / B);
        CUDA_LAUNCH(fill_neighbors_cell_kernel, Gdim, Bdim,
            pos.xptr(), pos.yptr(), rad.ptr(),
            cell_id.ptr(), cell_start.ptr(),
            neighbor_start.ptr(),
            neighbor_ids.ptr()
        );
    }
    sync_neighbors();
    std::cout << "build_neighbor_list" << std::endl;
}

void Disk::check_cell_neighbors_impl() {
}

void Disk::compute_forces_impl() {
    const int N = n_particles();
    const int B = 256;
    dim3 Bdim(B), Gdim((N + B - 1)/B);
    CUDA_LAUNCH(disk_force_kernel, Gdim, Bdim,
        pos.xptr(), pos.yptr(), rad.ptr(), force.xptr(), force.yptr(), pe.ptr()
    );
}

void Disk::compute_wall_forces_impl() {
    const int N = n_particles();
    const int B = 256;
    dim3 Bdim(B), Gdim((N + B - 1)/B);
    CUDA_LAUNCH(disk_wall_force_kernel, Gdim, Bdim,
        pos.xptr(), pos.yptr(), rad.ptr(), force.xptr(), force.yptr(), pe.ptr()
    );
}

void Disk::sync_class_constants_impl() {
    bind_disk_globals(e_interaction.ptr());
}


void Disk::update_positions_impl(double /*scale*/) {
    // do nothing
}

void Disk::update_velocities_impl(double /*scale*/) {
    // do nothing
}

} // namespace md