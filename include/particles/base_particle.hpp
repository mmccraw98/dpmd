#pragma once
#include "utils/device_fields.cuh"
#include "utils/h5_io.hpp"
#include "kernels/base_particle_kernels.cuh"
#include "utils/cuda_utils.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <cub/device/device_segmented_reduce.cuh>
#include <stdexcept>
#include <cstddef>
#include <map>
#include <vector>
#include <string>
#include <limits>
#include "utils/output_manager.hpp"
#include <filesystem>

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
    df::DeviceField1D<double> area;           // (N,)

    // Miscellaneous particle-level fields
    df::DeviceField1D<int> contacts;  // (N,) - number of particle-particle contacts for each particle

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
    df::DeviceField1D<int>    static_index;   // (N,) - index of the particle in the static order, used for undoing the cell-list sorting
    df::DeviceField2D<double> cell_size;         // (S,2) [Lx,Ly] - size of the cell for each system
    df::DeviceField2D<double> cell_inv;          // (S,2) [1/Lx,1/Ly] - inverse of the cell size for each system
    df::DeviceField2D<int>    cell_dim;          // (S,2) [Nx,Ny] - number of cells in each dimension for each system
    df::DeviceField1D<int>    cell_system_start; // (S+1,) - starting index of the cells for each system in the cell_ids list
    df::DeviceField1D<double> neighbor_cutoff;   // (S,) - neighbor cutoff for each system
    df::DeviceField1D<double> thresh2;           // (S,) - threshold squared for each system for neighbor list rebuild


    // System fields
    df::DeviceField1D<int>           system_id;             // (N,) — assumed static - effectively true if systems arent permuted
    df::DeviceField1D<int>           system_size;           // (S,) - number of particles in each system
    df::DeviceField1D<int>           system_offset;         // (S+1) - starting index of the particles in each system
    df::DeviceField1D<unsigned char> cub_sys_agg;           // (S,) temporary field for system-level aggregation
    
    // Box fields
    df::DeviceField2D<double> box_size;          // (S,2) [Lx,Ly] - size of the box for each system
    df::DeviceField2D<double> box_inv;           // (S,2) [1/Lx,1/Ly] - inverse of the box size for each system
    
    // System constants
    df::DeviceField1D<double> packing_fraction;  // (S,) - packing fraction for each system
    df::DeviceField1D<double> pressure;          // (S,) - pressure for each system
    df::DeviceField1D<double> temperature;       // (S,) - temperature for each system
    df::DeviceField1D<double> pe_total;          // (S,) - total potential energy for each system
    df::DeviceField1D<double> ke_total;          // (S,) - total kinetic energy for each system
    df::DeviceField1D<double> fpower_total;      // (S,) - total power for each system (used for the FIRE algorithm)
    df::DeviceField1D<int>    n_dof;             // (S,) - number of degrees of freedom for each system
    df::DeviceField1D<int>    n_contacts_total;  // (S,) - total number of contacts for each system

    // Stresses
    df::DeviceField2D<double> stress_tensor_x; // (N,2) - x-components of the stress tensor for each system (xx, xy)
    df::DeviceField2D<double> stress_tensor_y; // (N,2) - y-components of the stress tensor for each system (yx, yy)
    df::DeviceField2D<double> stress_tensor_total_x; // (S, 2) - x-components of the total stress tensor for each system (xx, xy)
    df::DeviceField2D<double> stress_tensor_total_y; // (S, 2) - y-components of the total stress tensor for each system (yx, yy)

    // Domain sampling fields
    df::DeviceField2D<double> domain_pos;             // (N_domain_vertices,2) - position of the domain
    df::DeviceField1D<double> domain_fractional_area; // (N_domain_vertices,) - fractional area of the domain
    df::DeviceField2D<double> domain_centroid;        // (N_domains,2) - centroid of the domain
    df::DeviceField1D<int>    domain_offset;          // (N_domains+1,) - offset of the domain
    df::DeviceField1D<int>    domain_particle_id;     // (N_domains,) - id of particle in the domain

    // Pairwise fields
    df::DeviceField2D<int>    pair_ids;   // (N_neighbors,2) - id of the two particles in the pairwise interaction
    df::DeviceField1D<double> pair_dist;  // (N_neighbors,) - distance between the pair of particles given by pair_ids

    // Neighbor method
    NeighborMethod neighbor_method = NeighborMethod::Naive;

    NeighborMethod get_neighbor_method() const { return neighbor_method; }

    // String representation of neighbor method for I/O
    std::string neighbor_method_to_string() const {
        switch (neighbor_method) {
            case NeighborMethod::Naive: return "Naive";
            case NeighborMethod::Cell:  return "Cell";
        }
        throw std::runtime_error("BaseParticle::neighbor_method_to_string: invalid neighbor method");
    }

    // Load from hdf5
    void load_from_hdf5(std::string path, std::string location, std::vector<std::string> optional_loading = {}) {
        std::string meta_path = path + "/meta.h5";
        // check if the meta file exists
        if (!std::filesystem::exists(meta_path)) {
            throw std::runtime_error("BaseParticle::load_from_hdf5: meta file does not exist: " + meta_path);
        }
        // open the meta file
        hid_t meta_file = H5Fopen(meta_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (meta_file < 0) {
            throw std::runtime_error("BaseParticle::load_from_hdf5: failed to open meta file: " + meta_path);
        }
        ::H5Handle meta_file_handle(meta_file, &H5Fclose);

        // load static
        auto static_group = ::open_group_checked(meta_file, "static");
        load_static_from_hdf5_group(static_group.get());  // load everything that is required
        load_from_hdf5_group(static_group.get());  // load everything that we can find in the static group
        load_optional_from_hdf5_group(static_group.get(), optional_loading);  // load any one-offs

        // location: init, final, restart
        if (location == "init") {
            auto init_group = ::open_group_checked(meta_file, location);
            load_from_hdf5_group(init_group.get());
        } else if (location == "final") {
            auto final_group = ::open_group_checked(meta_file, location);
            load_from_hdf5_group(final_group.get());
        } else if (location == "restart") {
            auto restart_group = ::open_group_checked(meta_file, location);
            load_from_hdf5_group(restart_group.get());
        } else {
            throw std::runtime_error("BaseParticle::load_from_hdf5: invalid location");
        }

        sync_box();
        sync_system();
        sync_neighbors();
        sync_cells();
        sync_class_constants();
        init_neighbors();
        set_n_dof();
    }

    // Load static data and initialize the particle using a specified group from the hdf5 file
    // Load everything that is strictly required to define thet particle
    // note: if any of these are not present, there will be errors by design
    void load_static_from_hdf5_group(hid_t group) {
        const int S = read_scalar<int>(group, "n_systems");
        const int N = read_scalar<int>(group, "n_particles");
        const int Nv = read_scalar<int>(group, "n_vertices");
        allocate_systems(S);
        allocate_particles(N);
        allocate_vertices(Nv);

        // system data
        system_id.from_host(read_vector<int>(group, "system_id"));
        system_size.from_host(read_vector<int>(group, "system_size"));
        system_offset.from_host(read_vector<int>(group, "system_offset"));

        // neighbor list data
        {
            std::string neighbor_method_str = read_scalar<std::string>(group, "neighbor_method");
            if (neighbor_method_str == "Naive") {
                set_neighbor_method(NeighborMethod::Naive);
            } else if (neighbor_method_str == "Cell") {
                set_neighbor_method(NeighborMethod::Cell);
                cell_size.from_host(read_vector_2d<double>(group, "cell_size"));
                cell_dim.from_host(read_vector_2d<int>(group, "cell_dim"));
                cell_system_start.from_host(read_vector<int>(group, "cell_system_start"));
                neighbor_cutoff.from_host(read_vector<double>(group, "neighbor_cutoff"));
                thresh2.from_host(read_vector<double>(group, "thresh2"));
            } else {
                throw std::runtime_error("BaseParticle::load_static_from_hdf5_group: invalid neighbor method");
            }
        }
        // run the impl for class-specific static loading
        derived().load_static_from_hdf5_group_impl(group);
    }

    // If there are any one-off, weird sets of fields that need to be loaded, this is the place to do it
    void load_optional_from_hdf5_group(hid_t group, std::vector<std::string> optional_loading) {
        for (const std::string& loading_scheme : optional_loading) {
            if (loading_scheme == "domain") {
                domain_pos.from_host(read_vector_2d<double>(group, "domain_pos"));
                domain_centroid.from_host(read_vector_2d<double>(group, "domain_centroid"));
                domain_fractional_area.from_host(read_vector<double>(group, "domain_fractional_area"));
                domain_offset.from_host(read_vector<int>(group, "domain_offset"));
                domain_particle_id.from_host(read_vector<int>(group, "domain_particle_id"));
            } else {
                throw std::runtime_error("BaseParticle::load_optional_from_hdf5_group: invalid loading scheme");
            }
        }
    }

    // Load from group within hdf5
    // Load everything that is NOT strictly required to define the particle
    // note: though these arent required, they still may cause errors if not loaded
    // note: pos, vel, force are loaded in the sub-classes since not all particles have them (i.e. DPM)
    // note: state variables (like pos) that are essentially required, are loaded outside of if-exists checks to force them to be loaded
    void load_from_hdf5_group(hid_t group) {
        if (h5_link_exists(group, "box_size")) {  // if the box size has been changed, load it again
            box_size.from_host(read_vector_2d<double>(group, "box_size"));
        }
        if (h5_link_exists(group, "cell_size")) {  // if the cell size has been changed, load it again
            cell_size.from_host(read_vector_2d<double>(group, "cell_size"));
        }
        if (h5_link_exists(group, "cell_dim")) {  // if the cell dim has been changed, load it again
            cell_dim.from_host(read_vector_2d<int>(group, "cell_dim"));
        }
        if (h5_link_exists(group, "packing_fraction")) {
            packing_fraction.from_host(read_vector<double>(group, "packing_fraction"));
        }
        if (h5_link_exists(group, "pressure")) {
            pressure.from_host(read_vector<double>(group, "pressure"));
        }
        if (h5_link_exists(group, "temperature")) {
            temperature.from_host(read_vector<double>(group, "temperature"));
        }
        if (h5_link_exists(group, "area")) {
            area.from_host(read_vector<double>(group, "area"));
        }
        if (h5_link_exists(group, "pe")) {
            pe.from_host(read_vector<double>(group, "pe"));
        }
        if (h5_link_exists(group, "ke")) {
            ke.from_host(read_vector<double>(group, "ke"));
        }
        if (h5_link_exists(group, "pe_total")) {
            pe_total.from_host(read_vector<double>(group, "pe_total"));
        }
        if (h5_link_exists(group, "ke_total")) {
            ke_total.from_host(read_vector<double>(group, "ke_total"));
        }
        if (h5_link_exists(group, "fpower_total")) {
            fpower_total.from_host(read_vector<double>(group, "fpower_total"));
        }
        // run the impl for class-specific loading
        derived().load_from_hdf5_group_impl(group);
    }

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
        sync_neighbors();
        update_neighbors();
    }

    // Update and build neighbor lists
    void update_neighbors() {
        switch (neighbor_method) {
            case NeighborMethod::Naive: update_naive_neighbors(); break;
            case NeighborMethod::Cell:  update_cell_neighbors();  break;
        }
    }

    // Check if neighbors need to be updated, if so, call update_neighbors()
    void check_neighbors(bool force_update=false) {
        switch (neighbor_method) {
            case NeighborMethod::Naive: break;  // do nothing
            case NeighborMethod::Cell:  check_cell_neighbors(force_update);   break;
        }
    }

    // Total number of particles
    int n_particles() const { return pos.size(); }

    // Total number of vertices
    int n_vertices() const { return derived().n_vertices_impl(); }

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
        area.resize(N);
        pos.fill(0.0, 0.0); vel.fill(0.0, 0.0); force.fill(0.0, 0.0); pe.fill(0.0); ke.fill(0.0);
        area.fill(0.0);
        derived().allocate_particles_impl(N);
    }

    // Allocate vertex-level data for total number of vertices
    void allocate_vertices(int Nv) {
        derived().allocate_vertices_impl(Nv);
    }

    // Allocate system-level data for total number of systems
    void allocate_systems(int S) {
        box_size.resize(S); box_inv.resize(S); system_size.resize(S); system_offset.resize(S+1);
        packing_fraction.resize(S); pressure.resize(S); temperature.resize(S); pe_total.resize(S); ke_total.resize(S);
        neighbor_cutoff.resize(S); thresh2.resize(S); cub_sys_agg.resize(S);
        box_size.fill(0.0, 0.0); box_inv.fill(0.0, 0.0); system_size.fill(0); system_offset.fill(0);
        packing_fraction.fill(0.0); pressure.fill(0.0); temperature.fill(0.0); pe_total.fill(0.0); ke_total.fill(0.0);
        neighbor_cutoff.fill(0.0); thresh2.fill(0.0); cub_sys_agg.fill(0);
        n_dof.resize(S); n_dof.fill(0.0);
        derived().allocate_systems_impl(S);
    }

    // Set the number of degrees of freedom for each system
    void set_n_dof() {
        derived().set_n_dof_impl();
    }

    // Bind box data to device memory and calculate its inverse
    void sync_box() {
        init_box_sizes();
        geo::bind_box_globals(box_size.xptr(), box_size.yptr(), box_inv.xptr(), box_inv.yptr());
    }

    // Bind system data to device memory
    void sync_system() {
        geo::bind_system_globals(system_offset.ptr(), system_id.ptr(), n_systems(), n_particles(), n_vertices());
        derived().bind_system_globals_impl();
    }

    // Bind neighbor data to device memory
    void sync_neighbors() {
        geo::bind_neighbor_globals(neighbor_start.ptr(), neighbor_ids.ptr(), neighbor_cutoff.ptr(), thresh2.ptr());
    }

    // Bind cell data to device memory
    void sync_cells() {
        geo::bind_cell_globals(cell_size.xptr(), cell_size.yptr(), cell_inv.xptr(), cell_inv.yptr(), cell_dim.xptr(), cell_dim.yptr(), cell_system_start.ptr());
    }

    // Bind class constants to device memory
    void sync_class_constants() {derived().sync_class_constants_impl();}

    // Enable swap, allocate aux memory in relevant particle-level fields
    void enable_swap(bool enable) { derived().enable_swap_impl(enable); }

    // Set random positions within the box with padding defaulting to 0.0
    void set_random_positions(double box_pad_x=0.0, double box_pad_y=0.0) { derived().set_random_positions_impl(box_pad_x, box_pad_y); }

    // Set random positions of particles within their domains
    void set_random_positions_in_domains() { derived().set_random_positions_in_domains_impl(); }

    // Compute pairwise forces (zeros out force and potential energy initially)
    void compute_forces()    { derived().compute_forces_impl(); }

    // Compute pairwise forces and wall forces (zeros out force and potential energy initially)
    void compute_wall_forces() { derived().compute_wall_forces_impl(); }

    // Compute damping forces - system-level damping scale
    void compute_damping_forces(df::DeviceField1D<double> scale) { derived().compute_damping_forces_impl(scale); }

    // Update positions - system-level dt with optional global scaling factor (i.e. for half-step)
    void update_positions(df::DeviceField1D<double> scale, double scale2=1.0) { derived().update_positions_impl(scale, scale2); }

    // Update velocities - system-level velocity scale with optional global scaling factor (i.e. for half-step)
    void update_velocities(df::DeviceField1D<double> scale, double scale2=1.0) { derived().update_velocities_impl(scale, scale2); }

    // Mix velocities and forces - system-level alpha, primarily used for FIRE
    void mix_velocities_and_forces(df::DeviceField1D<double> alpha) { derived().mix_velocities_and_forces_impl(alpha); }

    // Initialize naive (all-to-all, N^2) neighbor list
    void init_naive_neighbors() { derived().init_naive_neighbors_impl(); }

    // Update and build the naive neighbor list
    void update_naive_neighbors() { derived().update_naive_neighbors_impl(); sync_neighbors(); }

    // Initialize the cell neighbor list and enable array swapping
    void init_cell_neighbors() { enable_swap(true); init_cell_sizes(); sync_cells(); derived().init_cell_neighbors_impl(); }

    // Update the static index
    void update_static_index() { derived().update_static_index_impl(); }

    // Update the cell neighbor list
    void update_cell_neighbors() {
        derived().update_cell_neighbors_impl();
        update_static_index();
        sync_cells();
        sync_neighbors();
        reset_displacements();
    }

    // Check the cell neighbor list
    void check_cell_neighbors(bool force_update=false) { if (force_update || derived().check_cell_neighbors_impl()) update_cell_neighbors(); }

    // Reorder the particle data by the order array (or some other internal ordering logic)
    // resync the class constants after reordering to ensure the globals are updated
    void reorder_particles() { derived().reorder_particles_impl(); sync_class_constants(); }

    // Reset the displacements of the particles to the current positions
    void reset_displacements() { derived().reset_displacements_impl(); }

    // Update the cell size given the box size and the number of cells per dimension
    void update_cell_size() {
        const int S = n_systems();
        if (S == 0) return;
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(S);
        CUDA_LAUNCH(md::geo::update_cell_size_kernel, G, B,
            S, box_size.xptr(), box_size.yptr(), cell_dim.xptr(), cell_dim.yptr(), cell_size.xptr(), cell_size.yptr(), cell_inv.xptr(), cell_inv.yptr());
        sync_cells();
    }

    // Set the temperature of the systems by scaling the velocities
    void set_temperature(df::DeviceField1D<double> temperature_target) {
        compute_temperature();  // get current temperature
        df::DeviceField1D<double> scale; scale.resize(n_systems());
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(n_systems());
        CUDA_LAUNCH(md::geo::compute_temperature_scale_factor_kernel, G, B,
            temperature.ptr(), temperature_target.ptr(), scale.ptr()
        );
        set_average_velocity();
        scale_velocities(scale);
    }

    // Overload for a single temperature target
    void set_temperature(double temperature_target) {
        df::DeviceField1D<double> temperature_target_df; temperature_target_df.resize(n_systems()); temperature_target_df.fill(temperature_target);
        set_temperature(temperature_target_df);
    }

    // Scale the velocities of the particles
    void scale_velocities(df::DeviceField1D<double> scale) { derived().scale_velocities_impl(scale); }

    // Scale the velocities of the particles uniformly
    void scale_velocities(double scale) {
        const int S = n_systems();
        df::DeviceField1D<double> scale_df; scale_df.resize(S); scale_df.fill(scale);
        scale_velocities(scale_df);
    }

    // Calculate the average velocity of the systems
    df::DeviceField2D<double> calculate_average_velocity() { return derived().calculate_average_velocity_impl(); }

    // Set the average velocity of the systems to a desired vector
    void set_average_velocity(df::DeviceField2D<double> average_velocity) { derived().set_average_velocity_impl(average_velocity); }

    // Overload for a single average velocity
    void set_average_velocity(double average_velocity_x, double average_velocity_y) {
        df::DeviceField2D<double> average_velocity; average_velocity.resize(n_systems()); average_velocity.fill(average_velocity_x, average_velocity_y);
        set_average_velocity(average_velocity);
    }

    // Overload for zero velocity
    void set_average_velocity() {
        df::DeviceField2D<double> average_velocity; average_velocity.resize(n_systems()); average_velocity.fill(0.0, 0.0);
        set_average_velocity(average_velocity);
    }

    // Scale the positions of the particles
    void scale_positions(df::DeviceField1D<double> scale) { derived().scale_positions_impl(scale); }

    // Change the packing fraction by a set increment using an affine scaling of the box size and positions
    void increment_packing_fraction(df::DeviceField1D<double> increment) {
        compute_packing_fraction();
        const int S = n_systems();
        if (S == 0) return;
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(S);
        df::DeviceField1D<double> scale_factor; scale_factor.resize(S); scale_factor.fill(1.0);
        CUDA_LAUNCH(md::geo::scale_box_by_increment_kernel, G, B,
            S, box_size.xptr(), box_size.yptr(), packing_fraction.ptr(), increment.ptr(), scale_factor.ptr()
        );
        scale_positions(scale_factor);
        if (neighbor_method == NeighborMethod::Cell) {
            update_cell_size();
        }
        sync_box();
        check_neighbors(true);
        compute_packing_fraction();
    }

    // Overload for a single increment
    void increment_packing_fraction(double increment) {
        const int S = n_systems();
        df::DeviceField1D<double> increment_df; increment_df.resize(S); increment_df.fill(increment);
        increment_packing_fraction(increment_df);
    }

    // Compute the kinetic energy of each particle
    void compute_ke() { derived().compute_ke_impl(); }

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

        auto B = md::launch::threads_for();
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

    // Compute the total potential energy of each system
    void compute_pe_total() {
        cudaStream_t stream = 0;
        segmented_sum(this->pe.ptr(), this->pe_total.ptr(), stream);
    }

    // Compute the kinetic energy of each system
    void compute_ke_total() {
        compute_ke();  // compute the kinetic energy of each particle
        cudaStream_t stream = 0;
        segmented_sum(this->ke.ptr(), this->ke_total.ptr(), stream);
    }

    // Compute the temperature of each system
    void compute_temperature() {
        const int S = this->n_systems();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(S);
        CUDA_LAUNCH(md::geo::compute_temperature_kernel, G, B,
            this->ke_total.ptr(), this->n_dof.ptr(), this->temperature.ptr()
        );
    }

    // Compute the total particle area of each system
    df::DeviceField1D<double> compute_particle_area_total() {
        cudaStream_t stream = 0;
        const int S = this->n_systems();
        df::DeviceField1D<double> area_total(S);
        segmented_sum(this->area.ptr(), area_total.ptr(), stream);
        return area_total;
    }

    // Compute the packing fraction of each system
    void compute_packing_fraction() {
        const int N = n_particles();
        const int S = n_systems();
        
        // Create temporary array for per-particle packing fractions
        static df::DeviceField1D<double> pf_per_particle;
        if (pf_per_particle.size() != N) {
            pf_per_particle.resize(N);
        }
        
        // Compute per-particle packing fractions
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(N);
        CUDA_LAUNCH(md::geo::compute_fractional_packing_fraction_kernel, G, B,
            this->area.ptr(), pf_per_particle.ptr()
        );
        
        // Sum per system using CUB
        cudaStream_t stream = 0;
        segmented_sum(pf_per_particle.ptr(), this->packing_fraction.ptr(), stream);
    }

    // Compute the number of contacts for each particle
    void compute_contacts() { derived().compute_contacts_impl(); }

    // Compute the total number of contacts for each system
    void compute_n_contacts_total() {
        cudaStream_t stream = 0;
        segmented_sum(this->contacts.ptr(), this->n_contacts_total.ptr(), stream);
    }

    // Compute the distances between each pair of particles
    void compute_pair_dist() { derived().compute_pair_dist_impl(); }

    // Compute the stress tensor for each system
    void compute_stress_tensor() {
        if (this->stress_tensor_x.size() != this->n_particles()) {
            this->stress_tensor_x.resize(this->n_particles());
        }
        if (this->stress_tensor_y.size() != this->n_particles()) {
            this->stress_tensor_y.resize(this->n_particles());
        }
        derived().compute_stress_tensor_impl();
    }

    // Compute the total stress tensor for each system
    void compute_stress_tensor_total() {
        if (this->stress_tensor_total_x.size() != this->n_systems()) {
            this->stress_tensor_total_x.resize(this->n_systems());
        }
        if (this->stress_tensor_total_y.size() != this->n_systems()) {
            this->stress_tensor_total_y.resize(this->n_systems());
        }
        this->segmented_sum(this->stress_tensor_x.xptr(), this->stress_tensor_total_x.xptr(), 0);
        this->segmented_sum(this->stress_tensor_y.yptr(), this->stress_tensor_total_y.yptr(), 0);
        this->segmented_sum(this->stress_tensor_x.yptr(), this->stress_tensor_total_y.xptr(), 0);
        this->segmented_sum(this->stress_tensor_y.xptr(), this->stress_tensor_total_x.yptr(), 0);
    }

    // Compute the pressure for each system (trace of the stress tensor divided by the number of dimensions, 2)
    void compute_pressure() {
        compute_stress_tensor();

        cudaStream_t stream = 0;
        const int S = this->n_systems();
        if (this->pressure.size() != S) {
            this->pressure.resize(S);
        }

        auto input_iter = thrust::make_transform_iterator(
            thrust::make_zip_iterator(thrust::make_tuple(
                this->stress_tensor_x.xptr(),  // sigma_xx
                this->stress_tensor_y.yptr()   // sigma_yy
            )),
            md::geo::StressTrace2D()
        );

        this->segmented_sum(input_iter, this->pressure.ptr(), stream);
    }

    // Compute the total power of each system (used for the FIRE algorithm)
    void compute_fpower_total() { derived().compute_fpower_total_impl(); }

    // Save current state using a flag array (true_val == save, otherwise don't save)
    void save_state(df::DeviceField1D<int> flag, int true_val) { derived().save_state_impl(flag, true_val); }

    // Save current state
    void save_state() {
        const int S = n_systems();
        const int true_val = 1;
        df::DeviceField1D<int> flag; flag.resize(S); flag.fill(true_val);
        save_state(flag, true_val);
    }

    // Restore to last saved state using a flag array (true_val == restore, otherwise don't restore)
    void restore_state(df::DeviceField1D<int> flag, int true_val) {
        derived().restore_state_impl(flag, true_val);
        sync_box();
        sync_class_constants();
        check_neighbors(true);  // force a neighbor list rebuild
    }

    // Restore to last saved state
    void restore_state() {
        const int S = n_systems();
        const int true_val = 1;
        df::DeviceField1D<int> flag; flag.resize(S); flag.fill(true_val);
        restore_state(flag, true_val);
    }

    // =============================
    // Field maps for OutputManager
    // =============================

    // Get the class name as a string
    std::string get_class_name() { return derived().get_class_name_impl(); }

    // Get the names of the fields that should be saved as static
    std::vector<std::string> get_static_field_names() {
        std::vector<std::string> static_names {"system_id", "system_size", "system_offset", "box_size"};
        if (get_neighbor_method() == NeighborMethod::Cell) {
            std::vector<std::string> cell_names {"cell_size", "cell_dim", "cell_system_start", "neighbor_cutoff", "thresh2"};
            static_names.insert(static_names.end(), cell_names.begin(), cell_names.end());
        }
        std::vector<std::string> derived_names = derived().get_static_field_names_impl();
        static_names.insert(static_names.end(), derived_names.begin(), derived_names.end());
        return static_names;
    }

    // Get the names of the fields that should be saved as state
    std::vector<std::string> get_state_field_names()  {
        // there are no state fields for the base particle
        return derived().get_state_field_names_impl();
    }

    // Base provides common fields; derived can extend via CRTP hooks
    void output_build_registry(io::OutputRegistry& reg) {
        using io::FieldSpec1D; using io::FieldSpec2D;
        {
            FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->area; };
            reg.fields["area"] = p;
        }
        // {  // not sure how to handle this one
        //     Provider1D p; p.ensure_ready = []{};
        //     p.get_device = [this]{ return &this->neighbor_count; };
        //     p.index_space = ;
        //     reg.fields["neighbor_count"] = FieldDesc{ Dimensionality::D1, p, {}};
        // }
        // {  // not sure how to handle this one
        //     Provider1D p; p.ensure_ready = []{};
        //     p.get_device = [this]{ return &this->neighbor_start; };
        //     p.index_space = ;
        //     reg.fields["neighbor_start"] = FieldDesc{ Dimensionality::D1, p, {} };
        // }
        // {  // not sure how to handle this one
        //     Provider1D p; p.ensure_ready = []{};
        //     p.get_device = [this]{ return &this->neighbor_ids; };
        //     p.index_space = ;
        //     reg.fields["neighbor_ids"] = FieldDesc{ Dimensionality::D1, p, {} };
        // }
        // {  // this may need to be in the point subclass only as cell id may be a vertex level thing
        //     Provider1D p; p.ensure_ready = []{};
        //     p.get_device = [this]{ return &this->cell_id; };
        //     p.index_space = IndexSpace::System;
        //     reg.fields["cell_id"] = FieldDesc{ Dimensionality::D1, IndexSpace::Particle, p, {} };
        // }
        // {  // this may need to be in the point subclass only as cell id may be a vertex level thing
        //     Provider1D p; p.ensure_ready = []{};
        //     p.get_device = [this]{ return &this->cell_count; };
        //     p.index_space = IndexSpace::System;
        //     reg.fields["cell_count"] = FieldDesc{ Dimensionality::D1, IndexSpace::Particle, p, {} };
        // }
        // {  // this may need to be in the point subclass only as cell id may be a vertex level thing
        //     Provider1D p; p.ensure_ready = []{};
        //     p.get_device = [this]{ return &this->cell_start; };
        //     p.index_space = IndexSpace::System;
        //     reg.fields["cell_start"] = FieldDesc{ Dimensionality::D1, IndexSpace::Particle, p, {} };
        // }
        // {
        //     Provider1D p; p.ensure_ready = []{};
        //     p.get_device = [this]{ return &this->order; };
        //     p.index_space = IndexSpace::System;
        //     reg.fields["order"] = FieldDesc{ Dimensionality::D1, IndexSpace::Particle, p, {} };
        // }
        // {
        //     Provider1D p; p.ensure_ready = []{};
        //     p.get_device = [this]{ return &this->order_inv; };
        //     p.index_space = IndexSpace::System;
        //     reg.fields["order_inv"] = FieldDesc{ Dimensionality::D1, IndexSpace::Particle, p, {} };
        // }
        {
            FieldSpec2D<double> p; 
            p.get_device_field = [this]{ return &this->cell_size; };
            reg.fields["cell_size"] = p;
        }
        {
            FieldSpec2D<int> p; 
            p.get_device_field = [this]{ return &this->cell_dim; };
            reg.fields["cell_dim"] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.get_device_field = [this]{ return &this->cell_system_start; };
            reg.fields["cell_system_start"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->neighbor_cutoff; };
            reg.fields["neighbor_cutoff"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->thresh2; };
            reg.fields["thresh2"] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.get_device_field = [this]{ return &this->system_id; };
            reg.fields["system_id"] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.get_device_field = [this]{ return &this->system_size; };
            reg.fields["system_size"] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.get_device_field = [this]{ return &this->system_offset; };
            reg.fields["system_offset"] = p;
        }
        {
            FieldSpec2D<double> p; 
            p.get_device_field = [this]{ return &this->box_size; };
            reg.fields["box_size"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.preprocess = [this]{ this->compute_packing_fraction(); };
            p.get_device_field = [this]{ return &this->packing_fraction; };
            reg.fields["packing_fraction"] = p;
        }
        // {  // not supported yet
        //     Provider1D p; p.ensure_ready = []{};
        //     p.get_device = [this]{ return &this->temperature; };
        //     p.index_space = IndexSpace::System;
        //     reg.fields["temperature"] = FieldDesc{ Dimensionality::D1, IndexSpace::System, p, {} };
        // }
        {
            FieldSpec1D<double> p; 
            p.preprocess = [this]{ this->compute_pe_total(); };
            p.get_device_field = [this]{ return &this->pe_total; };
            reg.fields["pe_total"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.preprocess = [this]{ this->compute_ke_total(); };
            p.get_device_field = [this]{ return &this->ke_total; };
            reg.fields["ke_total"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.preprocess = [this]{ this->compute_ke_total(); this->compute_temperature(); };
            p.get_device_field = [this]{ return &this->temperature; };
            reg.fields["temperature"] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.preprocess = [this]{ this->compute_contacts(); };
            p.get_device_field = [this]{ return &this->contacts; };
            reg.fields["contacts"] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.preprocess = [this]{ this->compute_contacts(); this->compute_n_contacts_total(); };
            p.get_device_field = [this]{ return &this->n_contacts_total; };
            reg.fields["n_contacts_total"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.preprocess = [this]{ this->compute_pair_dist(); };
            p.get_device_field = [this]{ return &this->pair_dist; };
            reg.fields["pair_dist"] = p;
        }
        {
            FieldSpec2D<int> p; 
            p.preprocess = [this]{ this->compute_pair_dist(); };
            p.get_device_field = [this]{ return &this->pair_ids; };
            reg.fields["pair_ids"] = p;
        }
        {
            FieldSpec2D<double> p; 
            p.preprocess = [this]{ this->compute_stress_tensor(); };
            p.get_device_field = [this]{ return &this->stress_tensor_x; };
            reg.fields["stress_tensor_x"] = p;
        }
        {
            FieldSpec2D<double> p; 
            p.preprocess = [this]{ this->compute_stress_tensor(); };
            p.get_device_field = [this]{ return &this->stress_tensor_y; };
            reg.fields["stress_tensor_y"] = p;
        }
        {
            FieldSpec2D<double> p; 
            p.preprocess = [this]{ this->compute_stress_tensor_total(); };
            p.get_device_field = [this]{ return &this->stress_tensor_total_x; };
            reg.fields["stress_tensor_total_x"] = p;
        }
        {
            FieldSpec2D<double> p;
            p.preprocess = [this]{ this->compute_stress_tensor_total(); };
            p.get_device_field = [this]{ return &this->stress_tensor_total_y; };
            reg.fields["stress_tensor_total_y"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.preprocess = [this]{ this->compute_stress_tensor(); this->compute_pressure(); };
            p.get_device_field = [this]{ return &this->pressure; };
            reg.fields["pressure"] = p;
        }
        derived().output_build_registry_impl(reg);
    }

    void output_capture_inverse_orders(std::vector<int>& inv_particles, std::vector<int>& inv_vertices) const {
        inv_particles.clear(); inv_vertices.clear();
        if (order_inv.size() == n_particles() && n_particles() > 0) {
            order_inv.to_host(inv_particles);
        }
    }

protected:
    Derived&       derived()       { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    template <typename InputIterator, typename OutputIterator>
    void segmented_sum(InputIterator input,
                       OutputIterator output,
                       int num_segments,
                       const int* begin_offsets,
                       const int* end_offsets,
                       cudaStream_t stream = 0) {
        if (num_segments <= 0) {
            return;
        }

        size_t temp_bytes = 0;
        cub::DeviceSegmentedReduce::Sum(
            nullptr, temp_bytes,
            input, output,
            num_segments,
            begin_offsets,
            end_offsets,
            stream);

        void* workspace = reserve_sys_agg_storage(temp_bytes);
        cub::DeviceSegmentedReduce::Sum(
            workspace, temp_bytes,
            input, output,
            num_segments,
            begin_offsets,
            end_offsets,
            stream);
    }

    template <typename InputIterator, typename OutputIterator>
    void segmented_sum(InputIterator input,
                       OutputIterator output,
                       cudaStream_t stream = 0) {
        const int S = this->n_systems();
        segmented_sum(input, output, S,
                      this->system_offset.ptr(),
                      this->system_offset.ptr() + 1,
                      stream);
    }

    void* reserve_sys_agg_storage(size_t temp_bytes) {
        if (temp_bytes == 0) {
            return nullptr;
        }
        const size_t current_bytes = static_cast<size_t>(this->cub_sys_agg.size());
        if (temp_bytes > current_bytes) {
            if (temp_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("BaseParticle::reserve_sys_agg_storage: temp_bytes exceeds DeviceField1D capacity");
            }
            this->cub_sys_agg.resize(static_cast<int>(temp_bytes));
        }
        return static_cast<void*>(this->cub_sys_agg.ptr());
    }

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
    void mix_velocities_and_forces_impl(double) {}
    void scale_velocities_impl(double) {}
    df::DeviceField2D<double> calculate_average_velocity_impl() { return df::DeviceField2D<double>(0, 0); }
    void set_average_velocity_impl(df::DeviceField2D<double>) {}
    void update_positions_impl(double, double) {}
    void update_velocities_impl(double, double) {}
    void reorder_particles_impl() {}
    void reset_displacements_impl() {}
    void compute_ke_impl() {}
    void allocate_vertices_impl(int) {}
    void bind_system_globals_impl() {}
    int n_vertices_impl() const { return 0; }
    void set_random_positions_impl(double, double) {}
    void set_random_positions_in_domains_impl() {}
    void compute_fpower_total_impl() {}
    void compute_contacts_impl() {}
    void compute_pair_dist_impl() {}
    void save_state_impl(df::DeviceField1D<int>, int) {}
    void restore_state_impl(df::DeviceField1D<int>, int) {}
    void load_from_hdf5_group_impl(hid_t) {}
    void load_static_from_hdf5_group_impl(hid_t) {}
    std::string get_class_name_impl() { return ""; }
    std::vector<std::string> get_static_field_names_impl() { return {}; }
    std::vector<std::string> get_state_field_names_impl() { return {}; }
    void output_build_registry_impl(io::OutputRegistry&) {}
    void set_n_dof_impl() {}
private:
    int _n_cells;
};

} // namespace md
