#pragma once
#include "utils/device_fields.hpp"
#include "kernels/common.cuh"
#include "kernels/launch_config.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <stdexcept>
#include <cstddef>

namespace md {

// Enum for the different neighbor methods
enum class NeighborMethod : int { Naive = 0, Cell = 1 };

// Base class for all particle types
template <class Derived>
class BaseParticle {
public:
    // Particle dynamics fields
    df::DeviceField2D<double> pos;            // (N,2)
    df::DeviceField2D<double> vel;            // (N,2)
    df::DeviceField2D<double> force;          // (N,2)
    df::DeviceField1D<double> pe;             // (N,)
    df::DeviceField1D<double> ke;             // (N,)

    // Neighbor list fields
    df::DeviceField1D<int>    neighbor_count; // (N,) - number of neighbors for each particle
    df::DeviceField1D<int>    neighbor_start; // (N+1,) - starting index of the neighbor list for a given particle in the neighbor_ids list
    df::DeviceField1D<int>    neighbor_ids;   // (total_neighbors,) - list of neighbor ids for all particles
    
    // Cell list fields
    df::DeviceField1D<int>    cell_id;        // (N,) - id of the cell that each particle belongs to
    df::DeviceField1D<int>    cell_count;     // (total_cells,) - number of particles in each cell
    df::DeviceField1D<int>    cell_start;     // (total_cells+1,) - starting particle index of the particles in each cell
    df::DeviceField1D<int>    order;          // (N,) - sorted particle index
    df::DeviceField1D<int>    order_inv;      // (N,) - inverse of the sorted particle index
    df::DeviceField2D<double> cell_size;         // (S,2) [Lx,Ly] - size of the cell for each system
    df::DeviceField2D<double> cell_inv;          // (S,2) [1/Lx,1/Ly] - inverse of the cell size for each system
    df::DeviceField2D<int>    cell_dim;          // (S,2) [Nx,Ny] - number of cells in each dimension for each system
    df::DeviceField1D<int>    cell_system_start; // (S+1,) - starting index of the cells for each system in the cell_ids list


    // System fields
    df::DeviceField1D<int>    system_id;         // (N,) — assumed static - effectively true if systems arent permuted
    df::DeviceField1D<int>    system_size;       // (S,) - number of particles in each system
    df::DeviceField1D<int>    system_offset;     // (S+1) - starting index of the particles in each system
    
    // Box fields
    df::DeviceField2D<double> box_size;          // (S,2) [Lx,Ly] - size of the box for each system
    df::DeviceField2D<double> box_inv;           // (S,2) [1/Lx,1/Ly] - inverse of the box size for each system
    
    // System constants
    df::DeviceField1D<double> verlet_skin;       // (S,) - verlet skin for each system
    df::DeviceField1D<double> packing_fraction;  // (S,) - packing fraction for each system
    df::DeviceField1D<double> pressure;          // (S,) - pressure for each system
    df::DeviceField1D<double> temperature;       // (S,) - temperature for each system
    df::DeviceField1D<double> pe_total;          // (S,) - total potential energy for each system
    df::DeviceField1D<double> ke_total;          // (S,) - total kinetic energy for each system

    // Neighbor method
    NeighborMethod neighbor_method = NeighborMethod::Naive;

    // Set neighbor method to one of the enum values
    void set_neighbor_method(NeighborMethod method) {
        neighbor_method = method;
    }

    // Initialize variables for neighbor lists
    void init_neighbors() {
        switch (neighbor_method) {
            case NeighborMethod::Naive: init_naive_neighbors(); break;
            case NeighborMethod::Cell:  init_cell_neighbors();  break;
        }
        sync_neighbors();  // sync all neighbor related data
        update_neighbors();
    }

    // Update and build neighbor lists, sync to device once done
    void update_neighbors() {
        switch (neighbor_method) {
            case NeighborMethod::Naive: update_naive_neighbors(); break;
            case NeighborMethod::Cell:  update_cell_neighbors();  break;
        }
        sync_neighbors();
    }

    // Check if neighbors need to be updated, if so, call update_neighbors()
    void check_neighbors() {
        switch (neighbor_method) {
            case NeighborMethod::Naive: break;  // do nothing
            case NeighborMethod::Cell:  check_cell_neighbors();   break;
        }
    }

    // Total number of particles
    int n_particles() const { return pos.size(); }

    // Total number of systems
    int n_systems()  const { return box_size.size(); }

    // Total number of neighbors
    int n_neighbors() const {
        if (neighbor_count.size() == 0) {
            throw std::runtime_error("BaseParticle::n_neighbors: neighbor_count is not set");
        }
        int total = thrust::reduce(
            neighbor_count.begin(), neighbor_count.end(),
            0, thrust::plus<int>()
        );
        return total;
    }

    // Total number of cells
    int n_cells() const { return _n_cells; }

    // Allocate particle-level data for total number of particles
    void allocate_particles(int N) {
        pos.resize(N); vel.resize(N); force.resize(N); pe.resize(N); ke.resize(N);
        system_id.resize(N);
        derived().allocate_particles_impl(N);
    }

    // Allocate system-level data for total number of systems
    void allocate_systems(int S) {
        box_size.resize(S); box_inv.resize(S); system_size.resize(S); system_offset.resize(S+1);
        packing_fraction.resize(S); pressure.resize(S); temperature.resize(S); pe_total.resize(S); ke_total.resize(S);
        verlet_skin.resize(S);
        derived().allocate_systems_impl(S);
    }

    // Bind box data to device memory and calculate its inverse
    void sync_box() {
        init_box_sizes();
        geo::bind_box_globals(box_size.xptr(), box_size.yptr(), box_inv.xptr(), box_inv.yptr());
    }

    // Bind system data to device memory
    void sync_system() {
        geo::bind_system_globals(system_offset.ptr(), system_id.ptr(), n_systems(), n_particles());
    }

    // Bind neighbor data to device memory
    void sync_neighbors() {
        geo::bind_neighbor_globals(neighbor_start.ptr(), neighbor_ids.ptr(), verlet_skin.ptr());
    }

    // Bind cell data to device memory
    void sync_cells() {
        geo::bind_cell_globals(cell_size.xptr(), cell_size.yptr(), cell_inv.xptr(), cell_inv.yptr(), cell_dim.xptr(), cell_dim.yptr(), cell_system_start.ptr());
    }

    // Bind class constants to device memory
    void sync_class_constants() {derived().sync_class_constants_impl();}

    // Enable swap, allocate aux memory in relevant particle-level fields
    void enable_swap(bool enable) {
        if (enable) {
            pos.enable_swap();
            vel.enable_swap();
            force.enable_swap();
        } else {
            pos.disable_swap();
            vel.disable_swap();
            force.disable_swap();
        }
        derived().enable_swap_impl(enable);
    }

    // Compute pairwise forces
    void compute_forces()    { derived().compute_forces_impl(); }

    // Compute wall forces
    void compute_wall_forces() { derived().compute_wall_forces_impl(); }

    // Compute damping forces
    void compute_damping_forces(double scale) { derived().compute_damping_forces_impl(scale); }

    // Update positions
    void update_positions(double scale) { derived().update_positions_impl(scale); }

    // Update velocities
    void update_velocities(double scale) { derived().update_velocities_impl(scale); }

    // Initialize naive (all-to-all, N^2) neighbor list
    void init_naive_neighbors() { derived().init_naive_neighbors_impl(); }

    // Update and build the naive neighbor list
    void update_naive_neighbors() { derived().update_naive_neighbors_impl(); }

    // Initialize the cell neighbor list and enable array swapping
    void init_cell_neighbors() { enable_swap(true); init_cell_sizes(); sync_cells(); derived().init_cell_neighbors_impl(); }

    // Update the cell neighbor list
    void update_cell_neighbors() { derived().update_cell_neighbors_impl(); sync_cells(); reset_displacements(); }

    // Check the cell neighbor list
    void check_cell_neighbors() { if (derived().check_cell_neighbors_impl()) update_cell_neighbors(); }

    // Reorder the particle data by the order array (or some other internal ordering logic)
    void reorder_particles() { derived().reorder_particles_impl(); }

    // Reset the displacements of the particles to the current positions
    void reset_displacements() { derived().reset_displacements_impl(); }

    // Compute the kinetic energy of each particle
    void compute_ke() { derived().compute_ke_impl(); }

    // Compute the total potential energy of each system
    void compute_pe_total() { derived().compute_pe_total_impl(); }

    // Compute the total kinetic energy of each system
    void compute_ke_total() { derived().compute_ke_total_impl(); }

    // Initialize box sizes
    void init_box_sizes() {
        if (box_size.size() == 0) {
            throw std::runtime_error("BaseParticle::init_box_sizes: box_size is not set");
        }
        if (box_inv.size() == 0) {
            throw std::runtime_error("BaseParticle::init_box_sizes: box_inv is not set");
        }
        const int S = n_systems();
        if (S == 0) return;
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(S);
        CUDA_LAUNCH(md::geo::calculate_box_inv_kernel, G, B,
            box_size.xptr(), box_size.yptr(), box_inv.xptr(), box_inv.yptr(), S
        );
    }

    // Initialize cell sizes
    void init_cell_sizes() {
        if (box_size.size() == 0) {
            throw std::runtime_error("BaseParticle::init_cell_sizes: box_size is not set");
        }
        if (cell_dim.size() == 0) {
            throw std::runtime_error("BaseParticle::init_cell_sizes: cell_dim is not set");
        }
        if (box_size.size() != cell_dim.size()) {
            throw std::runtime_error("BaseParticle::init_cell_sizes: box_size and cell_dim must have the same size");
        }
        const int S = n_systems();
        if (S == 0) return;

        cell_size.resize(S);
        cell_inv.resize(S);

        thrust::device_vector<int> n_cell(S, 0);

        const int B = 256;
        auto G = md::launch::blocks_for(S);
        CUDA_LAUNCH(md::geo::init_cell_sizes_kernel, G, B,
            S,
            box_size.xptr(), box_size.yptr(),
            cell_dim.xptr(),  cell_dim.yptr(),
            cell_size.xptr(), cell_size.yptr(),
            cell_inv.xptr(),  cell_inv.yptr(),
            thrust::raw_pointer_cast(n_cell.data())
        );

        cell_system_start.resize(S + 1);

        thrust::exclusive_scan(n_cell.begin(), n_cell.end(), cell_system_start.begin());

        int total_cells = thrust::reduce(n_cell.begin(), n_cell.end(), 0, thrust::plus<int>());
        thrust::fill_n(cell_system_start.begin() + S, 1, total_cells);

        cell_count.resize(total_cells);
        cell_start.resize(total_cells + 1);

        _n_cells = cell_system_start.get_element(n_systems());
    }

protected:
    Derived&       derived()       { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    // Default no-op hooks
    void allocate_particles_impl(int) {}
    void allocate_systems_impl(int) {}
    void enable_swap_impl(bool) {}
    void init_naive_neighbors_impl() {}
    void update_naive_neighbors_impl() {}
    void init_cell_neighbors_impl() {}
    void update_cell_neighbors_impl() {}
    bool check_cell_neighbors_impl() { return false; }
    void compute_forces_impl() {}
    void compute_wall_forces_impl() {}
    void compute_damping_forces_impl(double) {}
    void sync_class_constants_impl() {}
    void update_positions_impl(double) {}
    void update_velocities_impl(double) {}
    void reorder_particles_impl() {}
    void reset_displacements_impl() {}
    void compute_ke_impl() {}
    void compute_pe_total_impl() {}
    void compute_ke_total_impl() {}
private:
    int _n_cells;
};

} // namespace md