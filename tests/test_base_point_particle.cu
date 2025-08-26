// test_base_point_particle.cu
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "particles/base_point_particle.hpp"
#include "kernels/base_particle_kernels.cuh"

namespace md {

// Trivial concrete type: uses BaseParticle defaults and implements a basic reordering
struct Dummy : BasePointParticle<Dummy> {
    void reorder_particles_impl() {
        auto src = thrust::make_zip_iterator(
            thrust::make_tuple(
                this->pos.x.begin(), this->pos.y.begin(),
                this->vel.x.begin(), this->vel.y.begin(),
                this->force.x.begin(), this->force.y.begin(),
                this->rad.begin(),
                this->mass.begin(),
                this->cell_id.begin()
            )
        );
        auto dst = thrust::make_zip_iterator(
            thrust::make_tuple(
                this->pos.x.swap_begin(), this->pos.y.swap_begin(),
                this->vel.x.swap_begin(), this->vel.y.swap_begin(),
                this->force.x.swap_begin(), this->force.y.swap_begin(),
                this->rad.swap_begin(),
                this->mass.swap_begin(),
                this->cell_id.swap_begin()
            )
        );
        thrust::gather(this->order.begin(), this->order.end(), src, dst);
        this->pos.swap(); this->vel.swap(); this->force.swap(); this->rad.swap(); this->mass.swap(); this->cell_id.swap();
    }
};
}

int main() {
    md::Dummy P;

    // Two systems: S=2
    const int S = 2;
    const int cell_dim = 4;
    const int n_cells_per_system = cell_dim * cell_dim;
    const int N = S * n_cells_per_system;
    const int num_particles_per_system = N/S;
    P.allocate_particles(N);
    P.allocate_systems(S);

    double box_size = 10.0;
    P.box_size.fill(box_size, box_size);
    P.sync_box();

    const double rad = std::sqrt((0.5 * box_size * box_size) / (M_PI * num_particles_per_system));
    P.rad.fill(rad);
    const double verlet_skin = box_size * box_size;  // overestimation to force all possible particles to be neighbors
    P.verlet_skin.fill(verlet_skin);
    const double thresh2 = (0.5 * verlet_skin) * (0.5 * verlet_skin);  // threshold squared for each system for neighbor list rebuild
    P.thresh2.fill(thresh2);

    std::vector<double> pos_x(N), pos_y(N);
    for (int i = 0; i < N; i++) {
        pos_x[i] = (i % cell_dim) * box_size / cell_dim;
        pos_y[i] = (i / cell_dim) * box_size / cell_dim;
    }
    P.pos.from_host(pos_x, pos_y);

    std::vector<int> system_sizes(S);
    std::vector<int> system_ids(N);
    std::vector<int> system_offsets(S+1);
    for (int i = 0; i < N; i++) {
        system_ids[i] = std::floor(i / num_particles_per_system);
        system_sizes[system_ids[i]]++;
    }
    system_offsets[0] = 0;
    for (int i = 0; i < S; i++) {
        system_offsets[i+1] = system_offsets[i] + system_sizes[i];
    }
    P.system_id.from_host(system_ids);
    P.system_size.from_host(system_sizes);
    P.system_offset.from_host(system_offsets);
    P.sync_system();

    // test naive neighbor method
    P.set_neighbor_method(md::NeighborMethod::Naive);
    P.init_neighbors();

    std::vector<int> neighbor_ids(P.neighbor_ids.size());
    std::vector<int> neighbor_counts(N);
    std::vector<int> neighbor_starts(N+1);
    P.neighbor_ids.to_host(neighbor_ids);
    P.neighbor_count.to_host(neighbor_counts);
    P.neighbor_start.to_host(neighbor_starts);
    for (int i = 0; i < N; i++) {
        int theoretical_neighbor_count = num_particles_per_system - 1;
        // check that the number of neighbors is correct
        assert(neighbor_counts[i] == theoretical_neighbor_count);
        std::vector<int> seen_neighbors;
        for (int j = neighbor_starts[i]; j < neighbor_starts[i+1]; j++) {
            // check that the particle is not its own neighbor
            assert(i != neighbor_ids[j]);
            // check that the particle is in the same system as its neighbor
            assert(system_ids[i] == system_ids[neighbor_ids[j]]);
            // check that we haven't seen this neighbor before
            assert(std::find(seen_neighbors.begin(), seen_neighbors.end(), neighbor_ids[j]) == seen_neighbors.end());
            seen_neighbors.push_back(neighbor_ids[j]);
        }
    }
    cudaDeviceSynchronize();
    std::cout << "BasePointParticle naive neighbor method test passed.\n";

    // test cell neighbor method
    std::vector<int> cell_system_start(S + 1);
    cell_system_start[0] = 0;
    for (int i = 0; i < S; i++) {
        cell_system_start[i+1] = cell_system_start[i] + n_cells_per_system;
    }
    P.cell_system_start.from_host(cell_system_start);
    P.cell_dim.resize(S);
    P.cell_dim.fill(cell_dim, cell_dim);
    P.sync_cells();
    P.set_neighbor_method(md::NeighborMethod::Cell);
    P.init_neighbors();

    // Pull device data to host
    std::vector<int> cell_start, order, order_inv, cell_id;
    cell_start.resize(P.cell_start.size());
    order.resize(P.order.size());
    order_inv.resize(P.order_inv.size());
    cell_id.resize(P.cell_id.size());

    P.cell_start.to_host(cell_start);
    P.order.to_host(order);
    P.order_inv.to_host(order_inv);
    P.cell_id.to_host(cell_id);
    
    // compute cell_count from cell_start
    std::vector<int> cell_count(P.cell_count.size());
    for (int c=0; c< (int)cell_count.size(); ++c) {
        cell_count[c] = cell_start[c+1] - cell_start[c];
    }

    // 1) With 1 particle per cell: every cell_count[c] == 1
    for (int c=0; c< (int)cell_count.size(); ++c) {
        assert(cell_count[c] == 1);
    }

    // 2) cell_start must be [0,1,2,...,C] and cell_start[C] == N
    const int C = (int)cell_count.size();
    for (int c=0; c<C; ++c) {
        assert(cell_start[c] == c);
    }
    assert(cell_start[C] == N);

    // 3) order_inv is truly inverse of order
    for (int new_idx=0; new_idx<N; ++new_idx) {
        int old_idx = order[new_idx];
        assert(old_idx >= 0 && old_idx < N);
        assert(order_inv[old_idx] == new_idx);
    }

    // 4) Because there is exactly one particle per global cell, and you reorder,
    //    the reordered cell_id must be strictly increasing: cell_id[new_idx] == new_idx.
    for (int i=0; i<N; ++i) {
        assert(cell_id[i] == i);
    }

    // 5) Neighbor count sanity (9-cell stencil): with 1 per cell, expect 8 neighbors
    //    provided your cutoff accepts all 8 surrounding cells.
    std::vector<int> neigh_ct(N);
    P.neighbor_count.to_host(neigh_ct);
    for (int i=0; i<N; ++i) {
        assert(neigh_ct[i] == 8);
    }

    // 6) Verify reordering is done correctly
    // make a a new order vector and pair that with redefined rad (set to be in decreasing order)
    // then sort and verify that rad is then increasing and contains all the original rad values
    std::vector<int> host_order(N);
    std::vector<double> host_rad(N);
    std::vector<int> host_system_start(S+1);
    P.system_offset.to_host(host_system_start);
    for (int s_i = 0; s_i < S; s_i++) {
        int beg = host_system_start[s_i];
        int end = host_system_start[s_i+1];
        for (int i = beg; i < end; i++) {
            host_order[i] = beg + (end - i - 1);
            host_rad[i] = rad * (end - i);
        }
    }
    P.order.from_host(host_order);
    P.rad.from_host(host_rad);
    P.reorder_particles();
    std::vector<double> host_rad_sorted(N);
    std::vector<int> host_order_sorted(N);
    P.order.to_host(host_order_sorted);
    P.rad.to_host(host_rad_sorted);
    for (int s_i = 0; s_i < S; s_i++) {
        int beg = host_system_start[s_i];
        int end = host_system_start[s_i+1];
        // check that rad is increasing
        // check that each rad is unique (no duplicates)
        // check that each rad is in the original rad vector
        std::vector<double> seen_rads;
        for (int i = beg; i < end; i++) {
            if (i != end-1) {
                assert(host_rad_sorted[i] < host_rad_sorted[i+1]);
            }
            assert(std::find(seen_rads.begin(), seen_rads.end(), host_rad_sorted[i]) == seen_rads.end());
            seen_rads.push_back(host_rad_sorted[i]);
            assert(std::find(host_rad.begin(), host_rad.end(), host_rad_sorted[i]) != host_rad.end());
        }
    }
    cudaDeviceSynchronize();
    std::cout << "BasePointParticle simple cell neighbor method test passed.\n";

    {
        const int particles_per_cell = 2;
        const int N2 = particles_per_cell * n_cells_per_system * S;
        md::Dummy Q;
        Q.allocate_particles(N2);
        Q.allocate_systems(S);

        Q.box_size.fill(box_size, box_size);
        Q.sync_box();
        Q.rad.fill(rad);
        Q.verlet_skin.fill(10.0 * box_size * box_size);

        // Build host positions: 2 per global cell
        std::vector<double> x2(N2), y2(N2);
        std::vector<int> sys_id2(N2), sys_sizes2(S,0), sys_off2(S+1,0);
        int idx = 0;
        for (int sid=0; sid<S; ++sid) {
            const int nx = (int)std::ceil(std::sqrt(particles_per_cell));
            const int ny = (particles_per_cell + nx - 1) / nx; // ceil(p/nx)
            for (int k=0;k<particles_per_cell;++k) {
                for (int gy=0; gy<cell_dim; ++gy) for (int gx=0; gx<cell_dim; ++gx) {
                    int kx = k % nx;
                    int ky = k / nx;
                    // strictly interior offsets: (kx+0.5)/nx, (ky+0.5)/ny ∈ (0,1)
                    double u = (gx + (kx + 0.5) / nx) / cell_dim;  // ∈ (gx/cell_dim, (gx+1)/cell_dim)
                    double v = (gy + (ky + 0.5) / ny) / cell_dim;

                    x2[idx] = u * box_size;
                    y2[idx] = v * box_size;
                    sys_id2[idx] = sid;
                    sys_sizes2[sid]++; idx++;
                }
            }
        }

        Q.pos.from_host(x2, y2);
        Q.system_id.from_host(sys_id2);
        Q.system_size.from_host(sys_sizes2);
        Q.system_offset.from_host(sys_off2);
        Q.sync_system();

        // Cells (same as before)
        std::vector<int> cell_sys_start2 = cell_system_start; // reuse from above
        Q.cell_system_start.from_host(cell_sys_start2);
        Q.cell_dim.resize(S);
        Q.cell_dim.fill(cell_dim, cell_dim);
        Q.sync_cells();

        Q.set_neighbor_method(md::NeighborMethod::Cell);
        Q.init_neighbors(); // runs full pipeline

        // Pull cell_start, compute cell_count, verify counts and starts
        std::vector<int> cs2(Q.cell_start.size()), cc2(Q.cell_count.size());
        Q.cell_start.to_host(cs2);
        for (int c=0; c< (int)cc2.size(); ++c) {
            cc2[c] = cs2[c+1] - cs2[c];
            assert(cc2[c] == particles_per_cell);
        }
        assert(cs2.back() == N2);

        std::vector<int> cell_id2(N2);
        Q.cell_id.to_host(cell_id2);
        std::vector<int> cell_dim_x2(S), cell_dim_y2(S);
        Q.cell_dim.to_host(cell_dim_x2, cell_dim_y2);

        // Neighbor counts: expected = sum counts over 9 cells - 1
        std::vector<int> neigh_count2(N2); Q.neighbor_count.to_host(neigh_count2);
        for (int i = 0; i < N2; i++) {
            const int my_cell = cell_id2[i];
            const int sid     = sys_id2[i];
            const int cell_dim_x  = cell_dim_x2[sid];
            const int cell_dim_y  = cell_dim_y2[sid];
            const int cell_sys_start = cell_sys_start2[sid];

            const int local_cell_id = my_cell - cell_sys_start;
            const int cell_id_x     = local_cell_id % cell_dim_x;
            const int cell_id_y     = local_cell_id / cell_dim_x;
            std::vector<int> seen_neighbors;
            int num_neighbors = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = cell_id_y + dy; if (yy < 0) yy += cell_dim_y; else if (yy >= cell_dim_y) yy -= cell_dim_y;
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = cell_id_x + dx; if (xx < 0) xx += cell_dim_x; else if (xx >= cell_dim_x) xx -= cell_dim_x;

                    const int ncell = cell_sys_start + (yy * cell_dim_x + xx);
                    const int beg   = cs2[ncell];
                    const int end   = cs2[ncell + 1];
                    for (int j = beg; j < end; ++j) {
                        if (j == i) continue;
                        assert(std::find(seen_neighbors.begin(), seen_neighbors.end(), j) == seen_neighbors.end());
                        seen_neighbors.push_back(j);
                        num_neighbors++;
                    }
                }
            }
            assert(num_neighbors == neigh_count2[i]);
        }
    }
    cudaDeviceSynchronize();
    std::cout << "BasePointParticle complex cell neighbor method test passed.\n";
    return 0;
}