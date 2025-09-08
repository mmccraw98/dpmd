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
    df::DeviceField1D<double> verlet_skin;       // (S,) - verlet skin for each system
    df::DeviceField1D<double> thresh2;           // (S,) - threshold squared for each system for neighbor list rebuild


    // System fields
    df::DeviceField1D<int>    system_id;             // (N,) — assumed static - effectively true if systems arent permuted
    df::DeviceField1D<int>    system_size;           // (S,) - number of particles in each system
    df::DeviceField1D<int>    system_offset;         // (S+1) - starting index of the particles in each system
    df::DeviceField1D<unsigned char> cub_sys_agg;    // (S,) temporary field for system-level aggregation
    
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

    // Neighbor method
    NeighborMethod neighbor_method = NeighborMethod::Naive;

    NeighborMethod get_neighbor_method() const { return neighbor_method; }

    // Load from hdf5
    void load_from_hdf5(std::string meta_path, std::string location) {
        // open the meta file
        hid_t meta_file = H5Fopen(meta_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (meta_file < 0) {
            throw std::runtime_error("BaseParticle::load_from_hdf5: failed to open meta file: " + meta_path);
        }
        
        // load static
        hid_t static_group = H5Gopen(meta_file, "static", H5P_DEFAULT);
        load_static_from_hdf5_group(static_group);  // load everything that is required
        load_from_hdf5_group(static_group);  // load everything that is not required
        H5Gclose(static_group);
        
        // location: init, final, restart
        if (location == "init") {
            hid_t init_group = H5Gopen(meta_file, "init", H5P_DEFAULT);
            load_from_hdf5_group(init_group);
            H5Gclose(init_group);
        } else if (location == "final") {
            hid_t final_group = H5Gopen(meta_file, "final", H5P_DEFAULT);
            load_from_hdf5_group(final_group);
            H5Gclose(final_group);
        } else if (location == "restart") {
            hid_t restart_group = H5Gopen(meta_file, "restart", H5P_DEFAULT);
            load_from_hdf5_group(restart_group);
            H5Gclose(restart_group);
        } else {
            throw std::runtime_error("BaseParticle::load_from_hdf5: invalid location");
        }
        H5Fclose(meta_file);

        sync_box();
        sync_system();
        sync_neighbors();
        sync_cells();
        sync_class_constants();
        init_neighbors();
    }

    // Load static data and initialize the particle using a specified group from the hdf5 file
    // Load everything that is strictly required to define thet particle
    // note: if any of these are not present, there will be errors by design
    void load_static_from_hdf5_group(hid_t group) {
        const int S = read_scalar<int>(group, "n_systems");
        const int N = read_scalar<int>(group, "n_particles");
        const int Nv = read_scalar<int>(group, "n_vertices");
        const std::string neighbor_method_str = read_scalar<std::string>(group, "neighbor_method");
        if (neighbor_method_str == "naive") {
            set_neighbor_method(NeighborMethod::Naive);
        } else if (neighbor_method_str == "cell") {
            set_neighbor_method(NeighborMethod::Cell);
        } else {
            throw std::runtime_error("BaseParticle::load_static_from_hdf5_group: invalid neighbor method");
        }
        allocate_systems(S);
        allocate_particles(N);
        allocate_vertices(Nv);
        set_neighbor_method(neighbor_method);

        // system data
        system_id.from_host(read_vector<int>(group, "system_id"));
        system_size.from_host(read_vector<int>(group, "system_size"));
        system_offset.from_host(read_vector<int>(group, "system_offset"));
        box_size.from_host(read_vector_2d<double>(group, "box_size"));

        // load cell list data
        if (get_neighbor_method() == NeighborMethod::Cell) {
            cell_size.from_host(read_vector_2d<double>(group, "cell_size"));
            cell_dim.from_host(read_vector_2d<int>(group, "cell_dim"));
            cell_system_start.from_host(read_vector<int>(group, "cell_system_start"));
            verlet_skin.from_host(read_vector<double>(group, "verlet_skin"));
            thresh2.from_host(read_vector<double>(group, "thresh2"));
        }

        // run the impl for class-specific static loading
        derived().load_static_from_hdf5_group_impl(group);
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
        if 
        this->packing_fraction.fill(0.0);
        if (h5_link_exists(group, "packing_fraction")) {
            packing_fraction.from_host(read_vector<double>(group, "packing_fraction"));
        }
        this->pressure.fill(0.0);
        if (h5_link_exists(group, "pressure")) {
            pressure.from_host(read_vector<double>(group, "pressure"));
        }
        this->temperature.fill(0.0);
        if (h5_link_exists(group, "temperature")) {
            temperature.from_host(read_vector<double>(group, "temperature"));
        }
        this->area.fill(0.0);
        if (h5_link_exists(group, "area")) {
            area.from_host(read_vector<double>(group, "area"));
        }
        this->pe.fill(0.0);
        if (h5_link_exists(group, "pe")) {
            pe.from_host(read_vector<double>(group, "pe"));
        }
        this->ke.fill(0.0);
        if (h5_link_exists(group, "ke")) {
            ke.from_host(read_vector<double>(group, "ke"));
        }
        this->pe_total.fill(0.0);
        if (h5_link_exists(group, "pe_total")) {
            pe_total.from_host(read_vector<double>(group, "pe_total"));
        }
        this->ke_total.fill(0.0);
        if (h5_link_exists(group, "ke_total")) {
            ke_total.from_host(read_vector<double>(group, "ke_total"));
        }
        this->fpower_total.fill(0.0);
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
        verlet_skin.resize(S); thresh2.resize(S); cub_sys_agg.resize(S);
        derived().allocate_systems_impl(S);
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
        geo::bind_neighbor_globals(neighbor_start.ptr(), neighbor_ids.ptr(), verlet_skin.ptr(), thresh2.ptr());
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

    // Set random positions within the box with padding defaulting to 0.0
    void set_random_positions(double box_pad_x=0.0, double box_pad_y=0.0) { derived().set_random_positions_impl(box_pad_x, box_pad_y); }

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
    void update_naive_neighbors() { derived().update_naive_neighbors_impl(); }

    // Initialize the cell neighbor list and enable array swapping
    void init_cell_neighbors() { enable_swap(true); init_cell_sizes(); sync_cells(); derived().init_cell_neighbors_impl(); }

    // Update the cell neighbor list
    void update_cell_neighbors() { derived().update_cell_neighbors_impl(); sync_cells(); sync_neighbors(); reset_displacements(); }

    // Check the cell neighbor list
    void check_cell_neighbors() { if (derived().check_cell_neighbors_impl()) update_cell_neighbors(); }

    // Reorder the particle data by the order array (or some other internal ordering logic)
    // resync the class constants after reordering to ensure the globals are updated
    void reorder_particles() { derived().reorder_particles_impl(); sync_class_constants(); }

    // Reset the displacements of the particles to the current positions
    void reset_displacements() { derived().reset_displacements_impl(); }

    // Scale the velocities of the particles
    void scale_velocities(df::DeviceField1D<double> scale) { derived().scale_velocities_impl(scale); }

    // Scale the velocities of the particles uniformly
    void scale_velocities(double scale) {
        const int S = n_systems();
        df::DeviceField1D<double> scale_df; scale_df.resize(S); scale_df.fill(scale);
        scale_velocities(scale_df);
    }

    // Scale the positions of the particles
    void scale_positions(df::DeviceField1D<double> scale) { derived().scale_positions_impl(scale); }

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

    // Compute the total potential energy of each system
    void compute_pe_total() {
        cudaStream_t stream = 0;
        const int S = this->n_systems();

        void* d_temp = this->cub_sys_agg.ptr();
        size_t temp_bytes = this->cub_sys_agg.size();

        // 1) size request
        cub::DeviceSegmentedReduce::Sum(
            nullptr, temp_bytes,
            this->pe.ptr(), this->pe_total.ptr(),
            S,
            this->system_offset.ptr(),
            this->system_offset.ptr() + 1,
            stream);

        // 2) ensure workspace
        if (temp_bytes > static_cast<size_t>(this->cub_sys_agg.size())) {
            this->cub_sys_agg.resize(static_cast<int>(temp_bytes));
            d_temp = this->cub_sys_agg.ptr();
        }

        // 3) run
        cub::DeviceSegmentedReduce::Sum(
            d_temp, temp_bytes,
            this->pe.ptr(), this->pe_total.ptr(),
            S,
            this->system_offset.ptr(),
            this->system_offset.ptr() + 1,
            stream);
    }

    // Compute the kinetic energy of each system
    void compute_ke_total() {
        compute_ke();  // compute the kinetic energy of each particle
        cudaStream_t stream = 0;
        const int S = this->n_systems();

        void* d_temp = this->cub_sys_agg.ptr();
        size_t temp_bytes = this->cub_sys_agg.size();

        // 1) size request
        cub::DeviceSegmentedReduce::Sum(
            nullptr, temp_bytes,
            this->ke.ptr(), this->ke_total.ptr(),
            S,
            this->system_offset.ptr(),
            this->system_offset.ptr() + 1,
            stream);

        // 2) ensure workspace
        if (temp_bytes > static_cast<size_t>(this->cub_sys_agg.size())) {
            this->cub_sys_agg.resize(static_cast<int>(temp_bytes));
            d_temp = this->cub_sys_agg.ptr();
        }

        // 3) run
        cub::DeviceSegmentedReduce::Sum(
            d_temp, temp_bytes,
            this->ke.ptr(), this->ke_total.ptr(),
            S,
            this->system_offset.ptr(),
            this->system_offset.ptr() + 1,
            stream);
    }

    // Compute the total particle area of each system
    df::DeviceField1D<double> compute_particle_area_total() {
        cudaStream_t stream = 0;
        const int S = this->n_systems();

        void* d_temp = this->cub_sys_agg.ptr();
        size_t temp_bytes = this->cub_sys_agg.size();

        df::DeviceField1D<double> area_total(S);

        // 1) size request
        cub::DeviceSegmentedReduce::Sum(
            nullptr, temp_bytes,
            this->area.ptr(), area_total.ptr(),
            S,
            this->system_offset.ptr(),
            this->system_offset.ptr() + 1,
            stream);

        // 2) ensure workspace
        if (temp_bytes > static_cast<size_t>(this->cub_sys_agg.size())) {
            this->cub_sys_agg.resize(static_cast<int>(temp_bytes));
            d_temp = this->cub_sys_agg.ptr();
        }

        // 3) run
        cub::DeviceSegmentedReduce::Sum(
            d_temp, temp_bytes,
            this->area.ptr(), area_total.ptr(),
            S,
            this->system_offset.ptr(),
            this->system_offset.ptr() + 1,
            stream);

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
        void* d_temp = this->cub_sys_agg.ptr();
        size_t temp_bytes = this->cub_sys_agg.size();
        
        // 1) size request
        cub::DeviceSegmentedReduce::Sum(
            nullptr, temp_bytes,
            pf_per_particle.ptr(), this->packing_fraction.ptr(),
            S,
            this->system_offset.ptr(),
            this->system_offset.ptr() + 1,
            stream);

        // 2) ensure workspace
        if (temp_bytes > static_cast<size_t>(this->cub_sys_agg.size())) {
            this->cub_sys_agg.resize(static_cast<int>(temp_bytes));
            d_temp = this->cub_sys_agg.ptr();
        }

        // 3) run
        cub::DeviceSegmentedReduce::Sum(
            d_temp, temp_bytes,
            pf_per_particle.ptr(), this->packing_fraction.ptr(),
            S,
            this->system_offset.ptr(),
            this->system_offset.ptr() + 1,
            stream);
    }

    // Compute the total power of each system (used for the FIRE algorithm)
    void compute_fpower_total() { derived().compute_fpower_total_impl(); }

    // Save current state using a flag array (true_val == save, otherwise don't save)
    void save_state(df::DeviceField1D<int> flag, int true_val) { derived().save_state_impl(flag, true_val); }

    // Save current state
    void save_state() {
        const int N = n_particles();
        const int true_val = 1;
        df::DeviceField1D<int> flag; flag.resize(N); flag.fill(true_val);
        save_state(flag, true_val);
    }

    // Restore to last saved state using a flag array (true_val == restore, otherwise don't restore)
    void restore_state(df::DeviceField1D<int> flag, int true_val) { derived().restore_state_impl(flag, true_val); }

    // Restore to last saved state
    void restore_state() {
        const int N = n_particles();
        const int true_val = 1;
        df::DeviceField1D<int> flag; flag.resize(N); flag.fill(true_val);
        restore_state(flag, true_val);
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
    void mix_velocities_and_forces_impl(double) {}
    void scale_velocities_impl(double) {}
    void update_positions_impl(double, double) {}
    void update_velocities_impl(double, double) {}
    void reorder_particles_impl() {}
    void reset_displacements_impl() {}
    void compute_ke_impl() {}
    void allocate_vertices_impl(int) {}
    void bind_system_globals_impl() {}
    int n_vertices_impl() const { return 0; }
    void set_random_positions_impl(double, double) {}
    void compute_fpower_total_impl() {}
    void save_state_impl(df::DeviceField1D<int>, int) {}
    void restore_state_impl(df::DeviceField1D<int>, int) {}
    void load_from_hdf5_group_impl(hid_t) {}
    void load_static_from_hdf5_group_impl(hid_t) {}
private:
    int _n_cells;
};

} // namespace md