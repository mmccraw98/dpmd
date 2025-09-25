#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/minimizers.cuh"
#include "integrators/velocity_verlet.cuh"
#include "utils/output_manager.hpp"
#include "kernels/base_point_particle_kernels.cuh"

namespace local_kernels {
    // one off kernel used to set the cell size and its inverse given the box size
    __global__ void set_cell_size_and_inv_kernel_ONE_OFF(
        const int S,
        const double* __restrict__ box_size_x,
        const double* __restrict__ box_size_y,
        const int* __restrict__ cell_dim_x,
        const int* __restrict__ cell_dim_y,
        double* __restrict__ cell_size_x,
        double* __restrict__ cell_size_y,
        double* __restrict__ cell_inv_x,
        double* __restrict__ cell_inv_y
    ) {
        int s = blockIdx.x * blockDim.x + threadIdx.x;
        if (s >= S) return;
        cell_size_x[s] = box_size_x[s] / cell_dim_x[s];
        cell_size_y[s] = box_size_y[s] / cell_dim_y[s];
        cell_inv_x[s] = 1.0 / cell_size_x[s];
        cell_inv_y[s] = 1.0 / cell_size_y[s];
    }

    // could redefine this in the base point particle and elevate to base particle level
    __global__ void assign_vertex_cell_ids_kernel(
        const int Nv,
        const double* __restrict__ vertex_pos_x,
        const double* __restrict__ vertex_pos_y,
        const int* __restrict__ cell_system_start,
        const int* __restrict__ cell_dim_x,
        const int* __restrict__ cell_dim_y,
        int* __restrict__ vertex_cell_id
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= Nv) return;
        const double x = vertex_pos_x[i];
        const double y = vertex_pos_y[i];
        const int sid = md::poly::g_vertex_sys.id[i];
        const double box_inv_x = md::geo::g_box.inv_x[sid];
        const double box_inv_y = md::geo::g_box.inv_y[sid];

        double u = md::geo::wrap01(x * box_inv_x);
        double v = md::geo::wrap01(y * box_inv_y);
        const int cell_id_x = (int)floor(u * cell_dim_x[sid]);
        const int cell_id_y = (int)floor(v * cell_dim_y[sid]);
        vertex_cell_id[i] = cell_id_x + cell_id_y * cell_dim_x[sid] + cell_system_start[sid];
    }

    // could redefine this in the base point particle and elevate to base particle level
    __global__ void count_cells_kernel(
        const int Nv,
        const int* __restrict__ vertex_cell_id,
        int* __restrict__ cell_count
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= Nv) return;
        int cid = vertex_cell_id[i];
        if (cid >= 0) atomicAdd(&cell_count[cid], 1);
    }

    // could redefine this in the base point particle and elevate to base particle level
    __global__ void scatter_order_kernel(
        const int Nv,
        const int* __restrict__ vertex_cell_id,
        int* __restrict__ write_ptr,
        int* __restrict__ order,
        int* __restrict__ order_inv
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= Nv) return;
        int cid = vertex_cell_id[i];
        int dst = atomicAdd(&write_ptr[cid], 1);
        order[dst] = i;
        if (order_inv) order_inv[i] = dst;
    }

    __global__ void reorder_vertex_data_kernel(
        const int Nv,
        const int* __restrict__ order_inv,
        const double* __restrict__ vertex_pos_x,
        const double* __restrict__ vertex_pos_y,
        const double* __restrict__ vertex_force_x,
        const double* __restrict__ vertex_force_y,
        const double* __restrict__ vertex_rad,
        const int* __restrict__ vertex_cell_id,
        const int* __restrict__ vertex_particle_id,
        double* __restrict__ vertex_pos_x_new,
        double* __restrict__ vertex_pos_y_new,
        double* __restrict__ vertex_force_x_new,
        double* __restrict__ vertex_force_y_new,
        double* __restrict__ vertex_rad_new,
        int* __restrict__ vertex_particle_id_new,
        int* __restrict__ vertex_cell_id_new
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= Nv) return;
        int dst = order_inv[i];
        vertex_pos_x_new[dst] = vertex_pos_x[i];
        vertex_pos_y_new[dst] = vertex_pos_y[i];
        vertex_force_x_new[dst] = vertex_force_x[i];
        vertex_force_y_new[dst] = vertex_force_y[i];
        vertex_rad_new[dst] = vertex_rad[i];
        vertex_particle_id_new[dst] = vertex_particle_id[i];
        vertex_cell_id_new[dst] = vertex_cell_id[i];
    }

    // build the block of new vertex ids in the expected particle_offset block order
    __global__ void build_static_particle_order_kernel(
        const int N,
        const int* __restrict__ particle_offset,
        const int* __restrict__ n_vertices_per_particle,
        const int* __restrict__ order_inv,
        int* __restrict__ static_particle_order
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        for (int k = particle_offset[i]; k < particle_offset[i + 1]; k++) {
            static_particle_order[k] = order_inv[k];
        }
    }

    __global__ void count_vertex_cell_neighbors_kernel(
        const double* __restrict__ vertex_pos_x,
        const double* __restrict__ vertex_pos_y,
        const double* __restrict__ vertex_rad,
        const int* __restrict__ vertex_cell_id,
        const int* __restrict__ cell_start,
        int* __restrict__ neighbor_count,
        const int* __restrict__ cell_dim_x_array,
        const int* __restrict__ cell_dim_y_array,
        const int* __restrict__ cell_system_start_array
    ) {
        const int Nv = md::geo::g_sys.n_vertices;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= Nv) return;

        const int cid = vertex_cell_id[i];
        const int pid = md::poly::g_poly.particle_id[i];
        const int sid = md::poly::g_vertex_sys.id[i];
        const int cell_dim_x = cell_dim_x_array[sid];
        const int cell_dim_y = cell_dim_y_array[sid];
        const int cell_sys_start = cell_system_start_array[sid];

        // Decode local (cx, cy) from global cell id
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
        int* __restrict__ neighbor_ids,
        const int* __restrict__ cell_dim_x_array,
        const int* __restrict__ cell_dim_y_array,
        const int* __restrict__ cell_system_start_array
    ) {
        const int Nv = md::geo::g_sys.n_vertices;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= Nv) return;

        const int cid = vertex_cell_id[i];
        const int pid = md::poly::g_poly.particle_id[i];
        const int sid = md::poly::g_vertex_sys.id[i];
        const int cell_dim_x = cell_dim_x_array[sid];
        const int cell_dim_y = cell_dim_y_array[sid];
        const int cell_sys_start = cell_system_start_array[sid];

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

int main(int argc, char** argv) {
    std::string in_path = "/home/mmccraw/dev/data/09-09-25/new-initializations/rb_new";
    md::rigid_bumpy::RigidBumpy P;
    P.load_from_hdf5(in_path, "init");
    std::cout << "running" << std::endl;

    // build the vertex level cell list
    const int Nv = P.n_vertices();
    const int N = P.n_particles();
    const int S = P.n_systems();

    // define the relevant arrays
    df::DeviceField2D<double> cell_size; cell_size.resize(S);
    df::DeviceField2D<double> cell_inv; cell_inv.resize(S);
    df::DeviceField2D<int> cell_dim; cell_dim.resize(S);
    df::DeviceField1D<int> cell_system_start; cell_system_start.resize(S+1);
    df::DeviceField1D<int> cell_id; cell_id.resize(Nv);
    df::DeviceField1D<int> cell_count;
    df::DeviceField1D<int> cell_start;

    df::DeviceField1D<int> neighbor_count; neighbor_count.resize(Nv);
    df::DeviceField1D<int> neighbor_start; neighbor_start.resize(Nv + 1);
    df::DeviceField1D<int> neighbor_ids; neighbor_ids.resize(0);

    // arbitary number of cells per dimension
    const int cell_dim_lin = 12;
    cell_dim.fill(cell_dim_lin, cell_dim_lin);
    // compute the cell_system_start array
    df::DeviceField1D<int> throwaway_cell_size; throwaway_cell_size.resize(S); throwaway_cell_size.fill(cell_dim_lin * cell_dim_lin);
    thrust::exclusive_scan(throwaway_cell_size.begin(), throwaway_cell_size.end(), cell_system_start.begin());
    cell_system_start.set_element(S, cell_dim_lin * cell_dim_lin * S);
    const int N_cells = cell_system_start.get_element(S);
    // set the size of cell_count and cell_start
    cell_count.resize(N_cells);
    cell_start.resize(N_cells + 1);

    df::DeviceField1D<int> cell_aux; cell_aux.resize(N_cells);
    df::DeviceField1D<int> order; order.resize(Nv);
    df::DeviceField1D<int> order_inv; order_inv.resize(Nv);

    P.neighbor_cutoff.resize(S); P.neighbor_cutoff.fill(2.0);
    P.sync_neighbors();

    // initialization is done

    // assign the cell ids to the vertices
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(Nv);
    CUDA_LAUNCH(local_kernels::assign_vertex_cell_ids_kernel, G, B, Nv, P.vertex_pos.xptr(), P.vertex_pos.yptr(), cell_system_start.ptr(), cell_dim.xptr(), cell_dim.yptr(), cell_id.ptr());

    // rebuild the cell layout
    // count the number of vertices in each cell
    cell_count.fill(0);
    CUDA_LAUNCH(local_kernels::count_cells_kernel, G, B, Nv, cell_id.ptr(), cell_count.ptr());
    // determine the starting vertex index of each cell
    thrust::exclusive_scan(cell_count.begin(), cell_count.end(), cell_start.begin(), 0);
    cell_start.set_element(N_cells, Nv);
    thrust::copy(cell_start.begin(), cell_start.begin() + N_cells, cell_aux.begin());
    // determine the vertex order and inverse order
    CUDA_LAUNCH(local_kernels::scatter_order_kernel, G, B, Nv, cell_id.ptr(), cell_aux.ptr(), order.ptr(), order_inv.ptr());

    // reorder the vertex level data
    P.vertex_pos.enable_swap();
    P.vertex_force.enable_swap();
    P.vertex_rad.enable_swap();
    P.vertex_particle_id.enable_swap();
    cell_id.enable_swap();
    CUDA_LAUNCH(local_kernels::reorder_vertex_data_kernel, G, B, Nv, order_inv.ptr(), P.vertex_pos.xptr(), P.vertex_pos.yptr(), P.vertex_force.xptr(), P.vertex_force.yptr(), P.vertex_rad.ptr(), P.vertex_particle_id.ptr(), cell_id.ptr(), P.vertex_pos.xptr_swap(), P.vertex_pos.yptr_swap(), P.vertex_force.xptr_swap(), P.vertex_force.yptr_swap(), P.vertex_rad.ptr_swap(), P.vertex_particle_id.ptr_swap(), cell_id.ptr_swap());
    P.vertex_pos.swap();
    P.vertex_force.swap();
    P.vertex_rad.swap();
    P.vertex_particle_id.swap();
    cell_id.swap();
    // sync the vertex particle id
    P.sync_class_constants();
    // build the static particle order list
    df::DeviceField1D<int> static_particle_order; static_particle_order.resize(Nv);
    CUDA_LAUNCH(local_kernels::build_static_particle_order_kernel, G, B, N, P.particle_offset.ptr(), P.n_vertices_per_particle.ptr(), order_inv.ptr(), static_particle_order.ptr());

    // count the number of neighbors for each vertex
    CUDA_LAUNCH(local_kernels::count_vertex_cell_neighbors_kernel, G, B, P.vertex_pos.xptr(), P.vertex_pos.yptr(), P.vertex_rad.ptr(), cell_id.ptr(), cell_start.ptr(), neighbor_count.ptr(), cell_dim.xptr(), cell_dim.yptr(), cell_system_start.ptr());

    // scan the neighbor counts to get the starting index of each vertices neighbor list
    const int total_neighbors = thrust::reduce(neighbor_count.begin(), neighbor_count.end(),0, thrust::plus<int>());
    thrust::exclusive_scan(neighbor_count.begin(), neighbor_count.end(), neighbor_start.begin());
    neighbor_start.set_element(Nv, total_neighbors);
    neighbor_ids.resize(total_neighbors);

    // fill the neighbor list
    CUDA_LAUNCH(local_kernels::fill_vertex_cell_neighbor_list_kernel, G, B, P.vertex_pos.xptr(), P.vertex_pos.yptr(), P.vertex_rad.ptr(), cell_id.ptr(), cell_start.ptr(), neighbor_start.ptr(), neighbor_ids.ptr(), cell_dim.xptr(), cell_dim.yptr(), cell_system_start.ptr());

    // verify that the neighbor list is filled correctly
    std::vector<int> neighbor_count_host; neighbor_count.to_host(neighbor_count_host);
    std::vector<int> neighbor_start_host; neighbor_start.to_host(neighbor_start_host);
    std::vector<int> neighbor_ids_host; neighbor_ids.to_host(neighbor_ids_host);
    std::vector<int> vertex_particle_id_host; P.vertex_particle_id.to_host(vertex_particle_id_host);
    for (int i = 0; i < Nv; i++) {
        const int neighbor_start_i = neighbor_start_host[i];
        const int neighbor_end_i = neighbor_start_host[i+1];
        const int pid = vertex_particle_id_host[i];
        assert(neighbor_count_host[i] == neighbor_end_i - neighbor_start_i);
        for (int neighbor_id = neighbor_start_i; neighbor_id < neighbor_end_i; neighbor_id++) {
            const int j = neighbor_ids_host[neighbor_id];
            const int j_pid = vertex_particle_id_host[j];
            assert(j != i);
            assert(j_pid != pid);
            // if i is a neighbor of j, j should be a neighbor of i
            bool is_neighbor = false;
            for (int k = neighbor_start_host[j]; k < neighbor_start_host[j+1]; k++) {
                if (neighbor_ids_host[k] == i) {
                    is_neighbor = true;
                    break;
                }
            }
            assert(is_neighbor);
        }
    }
}