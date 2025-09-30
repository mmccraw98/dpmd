#pragma once
#include "particles/base_particle.hpp"
#include "utils/h5_io.hpp"
#include "utils/cuda_utils.cuh"
#include "kernels/base_poly_particle_kernels.cuh"
#include <cub/device/device_segmented_reduce.cuh>
#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/scatter.h>
#include <thrust/unique.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <type_traits>
#include <utility>

template<class T, class = void>
struct has_allocate_poly_extras_impl : std::false_type {};
template<class T>
struct has_allocate_poly_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().allocate_poly_extras_impl(0))>> : std::true_type {};

template<class T, class = void>
struct has_allocate_poly_vertex_extras_impl : std::false_type {};
template<class T>
struct has_allocate_poly_vertex_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().allocate_poly_vertex_extras_impl(0))>> : std::true_type {};

template<class T, class = void>
struct has_allocate_poly_system_extras_impl : std::false_type {};
template<class T>
struct has_allocate_poly_system_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().allocate_poly_system_extras_impl(0))>> : std::true_type {};

template<class T, class = void>
struct has_enable_poly_swap_extras_impl : std::false_type {};
template<class T>
struct has_enable_poly_swap_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().enable_poly_swap_extras_impl(false))>> : std::true_type {};

template<class T, class = void>
struct has_sync_class_constants_poly_extras_impl : std::false_type {};
template<class T>
struct has_sync_class_constants_poly_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().sync_class_constants_poly_extras_impl())>> : std::true_type {};

template<class T, class = void>
struct has_load_static_from_hdf5_poly_extras_impl : std::false_type {};
template<class T>
struct has_load_static_from_hdf5_poly_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().load_static_from_hdf5_poly_extras_impl(std::declval<hid_t>()))>> : std::true_type {};

template<class T, class = void>
struct has_load_from_hdf5_poly_extras_impl : std::false_type {};
template<class T>
struct has_load_from_hdf5_poly_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().load_from_hdf5_poly_extras_impl(std::declval<hid_t>()))>> : std::true_type {};


template<class T, class = void>
struct has_get_static_field_names_poly_extras_impl : std::false_type {};
template<class T>
struct has_get_static_field_names_poly_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().get_static_field_names_poly_extras_impl())>> : std::true_type {};


template<class T, class = void>
struct has_get_state_field_names_poly_extras_impl : std::false_type {};
template<class T>
struct has_get_state_field_names_poly_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().get_state_field_names_poly_extras_impl())>> : std::true_type {};

template<class T, class = void>
struct has_output_build_registry_poly_extras_impl : std::false_type {};
template<class T>
struct has_output_build_registry_poly_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().output_build_registry_poly_extras_impl(std::declval<io::OutputRegistry&>()))>> : std::true_type {};

template<class T, class = void>
struct has_init_cell_neighbors_poly_extras_impl : std::false_type {};
template<class T>
struct has_init_cell_neighbors_poly_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().init_cell_neighbors_poly_extras_impl())>> : std::true_type {};

namespace md {

// Enum for the different cell sort methods
enum class CellSortMethod { Bucket, Standard };

// Base class for point particles
template<class Derived>
class BasePolyParticle : public BaseParticle<Derived> {
public:
    using Base = BaseParticle<Derived>;

    df::DeviceField1D<double>        e_interaction;            // (S,) interaction energy
    df::DeviceField1D<int>           n_vertices_per_particle;  // (N,) number of vertices per particle
    df::DeviceField1D<int>           particle_offset;          // (N+1,) index of the first vertex of each particle
    df::DeviceField1D<int>           vertex_particle_id;       // (Nv,) particle id of each vertex
    df::DeviceField1D<int>           vertex_system_id;         // (Nv,) system id of each vertex
    df::DeviceField1D<int>           vertex_system_offset;     // (S+1,) index of the first vertex of each system
    df::DeviceField1D<int>           vertex_system_size;       // (S,) number of vertices in each system
    df::DeviceField2D<double>        vertex_pos;               // (Nv,2) vertex positions
    df::DeviceField2D<double>        vertex_vel;               // (Nv,2) vertex velocities
    df::DeviceField2D<double>        vertex_force;             // (Nv,2) vertex forces
    df::DeviceField1D<double>        vertex_pe;                // (Nv,) vertex potential energy
    df::DeviceField1D<double>        vertex_mass;              // (Nv,) vertex mass
    df::DeviceField1D<double>        vertex_rad;               // (Nv,) vertex radius
    df::DeviceField1D<int>           cell_aux;                 // (C,) auxiliary data for each cell
    df::DeviceField1D<int>           particle_neighbor_ids;    // (N_particle_neighbors,) - list of particle ids for the vertex neighbors
    df::DeviceField1D<int>           particle_neighbor_start;  // (N+1,) - starting index of the neighbor list for a given particle in the particle_neighbor_ids list
    df::DeviceField1D<int>           particle_neighbor_count;  // (N,) - number of neighbors for each particle

    inline static constexpr CellSortMethod cell_sort_method = CellSortMethod::Bucket;  // default sort method for the cell list

    // Allocate the particles
    void allocate_particles_impl(int N) {
        this->n_vertices_per_particle.resize(N);
        this->particle_offset.resize(N+1);
        this->n_vertices_per_particle.fill(0);
        this->particle_offset.fill(0);
        if constexpr (has_allocate_poly_extras_impl<Derived>::value)
            this->derived().allocate_poly_extras_impl(N);
    }

    // Allocate the vertices
    void allocate_vertices_impl(int Nv) {
        this->vertex_pos.resize(Nv);
        this->vertex_vel.resize(Nv);
        this->vertex_force.resize(Nv);
        this->vertex_pe.resize(Nv);
        this->vertex_mass.resize(Nv);
        this->vertex_rad.resize(Nv);
        this->vertex_particle_id.resize(Nv);
        this->vertex_system_id.resize(Nv);
        this->vertex_pos.fill(0.0, 0.0);
        this->vertex_vel.fill(0.0, 0.0);
        this->vertex_force.fill(0.0, 0.0);
        this->vertex_pe.fill(0.0);
        this->vertex_mass.fill(0.0);
        this->vertex_rad.fill(0.0);
        this->vertex_particle_id.fill(0);
        this->vertex_system_id.fill(0);
        this->static_index.resize(Nv);
        thrust::sequence(this->static_index.begin(), this->static_index.end(), 0);
        if constexpr (has_allocate_poly_vertex_extras_impl<Derived>::value)
            this->derived().allocate_poly_vertex_extras_impl(Nv);
    }

    // Allocate the systems
    void allocate_systems_impl(int S) {
        this->e_interaction.resize(S);
        this->vertex_system_offset.resize(S+1);
        this->vertex_system_size.resize(S);
        this->e_interaction.fill(0.0);
        this->vertex_system_offset.fill(0);
        this->vertex_system_size.fill(0);
        if constexpr (has_allocate_poly_system_extras_impl<Derived>::value)
            this->derived().allocate_poly_system_extras_impl(S);
    }

    // Enable/disable the swap for the point particle system
    void enable_swap_impl(bool enable) {
        if (enable) {
            this->vertex_pos.enable_swap();
            this->vertex_vel.enable_swap();
            this->vertex_force.enable_swap();
            this->vertex_mass.enable_swap();
            this->vertex_rad.enable_swap();
            this->vertex_particle_id.enable_swap();
            this->cell_id.enable_swap();
        } else {
            this->vertex_pos.disable_swap();
            this->vertex_vel.disable_swap();
            this->vertex_force.disable_swap();
            this->vertex_mass.disable_swap();
            this->vertex_rad.disable_swap();
            this->vertex_particle_id.disable_swap();
            this->cell_id.disable_swap();
        }
        if constexpr (has_enable_poly_swap_extras_impl<Derived>::value)
            this->derived().enable_poly_swap_extras_impl(enable);
    }

    // Initialize the naive neighbors
    void init_naive_neighbors_impl() {
        int Nv = Base::n_vertices();
        this->neighbor_count.resize(Nv);
        this->neighbor_start.resize(Nv + 1);
    }

    // Update the naive neighbors (should only be done once per simulation - never needs to be called again)
    void update_naive_neighbors_impl() {
        int Nv = this->n_vertices();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(Nv);
        // 1) count the neighbors (every particle in system i has system_size[i] - 1 neighbors)
        CUDA_LAUNCH(md::poly::count_naive_vertex_neighbors_kernel, G, B,
            this->neighbor_count.ptr()
        );
        // 2) scan the neighbor counts to get the starting index of each particle's neighbor list
        thrust::exclusive_scan(this->neighbor_count.begin(), this->neighbor_count.end(), this->neighbor_start.begin());
        // 3) count the total number of neighbors
        int total_neighbors = this->n_neighbors();
        // 4) set start[N] on device
        this->neighbor_start.set_element(Nv, total_neighbors);
        // 5) size neighbor_ids
        this->neighbor_ids.resize(total_neighbors);
        // 6) fill neighbor_ids
        CUDA_LAUNCH(md::poly::fill_naive_vertex_neighbor_list_kernel, G, B,
            this->neighbor_start.ptr(), this->neighbor_ids.ptr()
        );
    }

    // Initialize the cell neighbors
    void init_cell_neighbors_impl() {
        int Nv = this->n_vertices();
        this->cell_id.resize(Nv);
        this->order.resize(Nv);
        this->order_inv.resize(Nv);
        this->static_index.resize(Nv);
        thrust::sequence(this->static_index.begin(), this->static_index.end(), 0);
        this->neighbor_count.resize(Nv);
        this->neighbor_start.resize(Nv + 1);
        this->cell_aux.resize(this->n_cells());
        if constexpr (has_init_cell_neighbors_poly_extras_impl<Derived>::value)
            this->derived().init_cell_neighbors_poly_extras_impl();
    }

    // Update the cell neighbors
    void update_cell_neighbors_impl() {
        int C = this->n_cells();
        int Nv = this->n_vertices();
        // 1) assign cell ids to each vertex
        this->_assign_cell_ids();
        // 2) rebuild the cell layout using bucket, 
        // count the number of vertices in each cell,
        // determine the starting vertex index for each cell,
        // and determine the vertex order and inverse order
        this->_rebuild_cell_layout();
        // 3) use the order array to rearrange the vertex data so that spatially nearby vertices are nearby in the relevant data arrays
        this->reorder_particles();
        // 4) count the number of neighbors for each vertex
        this->_count_cell_neighbors();
        // 5) scan the neighbor counts to get the starting index of each vertex's neighbor list
        const int total_neighbors = this->n_neighbors();
        thrust::exclusive_scan(this->neighbor_count.begin(), this->neighbor_count.end(), this->neighbor_start.begin());
        this->neighbor_start.set_element(Nv, total_neighbors);
        this->neighbor_ids.resize(total_neighbors);
        // 6) create the neighbor list, again enumerating over the 9-cell stencil
        this->_fill_cell_neighbors();
    }

    // Update the static index
    void update_static_index_impl() {
        int Nv = this->n_vertices();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(Nv);
        CUDA_LAUNCH(md::geo::update_static_index_kernel, G, B, Nv, this->order_inv.ptr(), this->static_index.ptr());
    }

    // Sync class constants with pass-through to sub-class for any extra constants
    void sync_class_constants_impl() {
        md::poly::bind_poly_globals(this->vertex_particle_id.ptr(), this->particle_offset.ptr(), this->n_vertices_per_particle.ptr(), this->static_index.ptr());
        if constexpr (has_sync_class_constants_poly_extras_impl<Derived>::value)
            this->derived().sync_class_constants_poly_extras_impl();
        cudaDeviceSynchronize();
    }

    // Bind the system constants to the device memory
    void bind_system_globals_impl() {
        md::poly::bind_poly_system_globals(this->vertex_system_offset.ptr(), this->vertex_system_id.ptr(), this->vertex_system_size.ptr());
    }

    // Return total number of vertices
    int n_vertices_impl() const {
        return this->vertex_pos.size();
    }

    // Compute the stress tensor for each system
    void compute_stress_tensor_total_impl() {
        this->compute_stress_tensor();
        const int S = this->n_systems();
        this->segmented_sum(this->stress_tensor_x.xptr(), this->stress_tensor_total_x.xptr(), S, this->vertex_system_offset.ptr(), this->vertex_system_offset.ptr() + 1);
        this->segmented_sum(this->stress_tensor_y.yptr(), this->stress_tensor_total_y.yptr(), S, this->vertex_system_offset.ptr(), this->vertex_system_offset.ptr() + 1);
        this->segmented_sum(this->stress_tensor_x.yptr(), this->stress_tensor_total_y.xptr(), S, this->vertex_system_offset.ptr(), this->vertex_system_offset.ptr() + 1);
        this->segmented_sum(this->stress_tensor_y.xptr(), this->stress_tensor_total_x.yptr(), S, this->vertex_system_offset.ptr(), this->vertex_system_offset.ptr() + 1);
    }

    // Build the particle-level neighbor list from the vertex-level neighbor list
    void build_particle_neighbors() {
        const int N = this->n_particles();
        const int Nv = this->n_vertices();
        const int total_vertex_neighbors = this->n_neighbors();

        this->particle_neighbor_count.resize(N); this->particle_neighbor_count.fill(0);
        this->particle_neighbor_start.resize(N + 1); this->particle_neighbor_start.fill(0);
        this->particle_neighbor_ids.resize(0);

        if (total_vertex_neighbors == 0) {
            this->particle_neighbor_start.set_element(N, 0);
            return;
        }

        df::DeviceField1D<unsigned long long> pair_keys; pair_keys.resize(total_vertex_neighbors);

        auto threads = md::launch::threads_for();
        auto blocks = md::launch::blocks_for(Nv);
        CUDA_LAUNCH(md::poly::fill_particle_neighbor_pair_keys_kernel,
                    blocks, threads,
                    pair_keys.ptr());

        auto key_begin = pair_keys.begin();
        auto key_end = key_begin + total_vertex_neighbors;
        thrust::sort(thrust::device, key_begin, key_end);

        auto unique_end = thrust::unique(thrust::device, key_begin, key_end);
        const int num_unique = static_cast<int>(unique_end - key_begin);
        pair_keys.resize(num_unique);

        this->particle_neighbor_ids.resize(num_unique);

        auto extract_primary = [] __device__ (unsigned long long key) -> int {
            return static_cast<int>(key >> 32);
        };
        auto extract_neighbor = [] __device__ (unsigned long long key) -> int {
            return static_cast<int>(key & 0xFFFFFFFFull);
        };

        thrust::transform(
            thrust::device,
            pair_keys.begin(),
            pair_keys.begin() + num_unique,
            this->particle_neighbor_ids.begin(),
            extract_neighbor);

        df::DeviceField1D<int> primary_ids; primary_ids.resize(num_unique);
        df::DeviceField1D<int> per_particle_counts; per_particle_counts.resize(num_unique);

        auto primary_begin = thrust::make_transform_iterator(pair_keys.begin(), extract_primary);
        auto primary_end = primary_begin + num_unique;

        auto particle_reduce_result = thrust::reduce_by_key(
            thrust::device,
            primary_begin,
            primary_end,
            thrust::make_constant_iterator<int>(1),
            primary_ids.begin(),
            per_particle_counts.begin());

        const int n_particles_with_neighbors = static_cast<int>(particle_reduce_result.first - primary_ids.begin());

        if (n_particles_with_neighbors > 0) {
            thrust::scatter(
                thrust::device,
                per_particle_counts.begin(),
                per_particle_counts.begin() + n_particles_with_neighbors,
                primary_ids.begin(),
                this->particle_neighbor_count.begin());
        }

        thrust::exclusive_scan(
            this->particle_neighbor_count.begin(),
            this->particle_neighbor_count.end(),
            this->particle_neighbor_start.begin());
        this->particle_neighbor_start.set_element(N, num_unique);
    }


    // Load static data from hdf5 group and initialize the particle
    void load_static_from_hdf5_group_impl(hid_t group) {
        this->e_interaction.from_host(read_vector<double>(group, "e_interaction"));
        this->vertex_mass.from_host(read_vector<double>(group, "vertex_mass"));
        this->vertex_rad.from_host(read_vector<double>(group, "vertex_rad"));
        this->n_vertices_per_particle.from_host(read_vector<int>(group, "n_vertices_per_particle"));
        this->particle_offset.from_host(read_vector<int>(group, "particle_offset"));
        this->vertex_particle_id.from_host(read_vector<int>(group, "vertex_particle_id"));
        this->vertex_system_id.from_host(read_vector<int>(group, "vertex_system_id"));
        this->vertex_system_offset.from_host(read_vector<int>(group, "vertex_system_offset"));
        this->vertex_system_size.from_host(read_vector<int>(group, "vertex_system_size"));
        if constexpr (has_load_static_from_hdf5_poly_extras_impl<Derived>::value)
            this->derived().load_static_from_hdf5_poly_extras_impl(group);
    }

    // Load from hdf5 group and initialize the particle
    void load_from_hdf5_group_impl(hid_t group) {
        if (h5_link_exists(group, "vertex_pos")) {
            this->vertex_pos.from_host(read_vector_2d<double>(group, "vertex_pos"));
        }
        if (h5_link_exists(group, "vertex_vel")) {
            this->vertex_vel.from_host(read_vector_2d<double>(group, "vertex_vel"));
        }
        if (h5_link_exists(group, "vertex_force")) {
            this->vertex_force.from_host(read_vector_2d<double>(group, "vertex_force"));
        }
        if (h5_link_exists(group, "vertex_pe")) {
            this->vertex_pe.from_host(read_vector<double>(group, "vertex_pe"));
        }
        if constexpr (has_load_from_hdf5_poly_extras_impl<Derived>::value)
            this->derived().load_from_hdf5_poly_extras_impl(group);
    }


    // Get the names of the fields that should be saved as static
    std::vector<std::string> get_static_field_names_impl() {
        std::vector<std::string> static_names {"e_interaction", "vertex_mass", "vertex_rad", "n_vertices_per_particle", "particle_offset", "vertex_particle_id", "vertex_system_id", "vertex_system_offset", "vertex_system_size"};
        std::vector<std::string> derived_names = this->derived().get_static_field_names_poly_extras_impl();
        static_names.insert(static_names.end(), derived_names.begin(), derived_names.end());
        return static_names;
    }

    // Get the names of the fields that should be saved as state
    std::vector<std::string> get_state_field_names_impl() {
        std::vector<std::string> state_names {"vertex_pos"};
        std::vector<std::string> derived_names = this->derived().get_state_field_names_poly_extras_impl();
        state_names.insert(state_names.end(), derived_names.begin(), derived_names.end());
        return state_names;
    }

    // Build the output registry
    void output_build_registry_impl(io::OutputRegistry& reg) {
        // Register poly-specific fields
        using io::FieldSpec1D; using io::FieldSpec2D;
        std::string order_str = "static_index";
        // Provide the canonical->current vertex permutation so OutputManager can rebuild the original layout
        {
            FieldSpec1D<int> p;
            p.get_device_field = [this]{ return &this->static_index; };
            reg.fields[order_str] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.preprocess = [this]{ this->compute_contacts(); };
            p.get_device_field = [this]{ return &this->contacts; };
            reg.fields["contacts"] = p;
        }
        {
            FieldSpec2D<double> p;
            p.preprocess = [this]{ this->compute_overlaps(); };
            p.get_device_field = [this]{ return &this->overlaps; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["overlaps"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->pe; };
            reg.fields["pe"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.preprocess = [this]{ this->compute_ke(); };
            p.get_device_field = [this]{ return &this->ke; };
            reg.fields["ke"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->e_interaction; };
            reg.fields["e_interaction"] = p;
        }
        {
            FieldSpec1D<int> p;
            p.get_device_field = [this]{ return &this->n_vertices_per_particle; };
            reg.fields["n_vertices_per_particle"] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.get_device_field = [this]{ return &this->particle_offset; };
            reg.fields["particle_offset"] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.get_device_field = [this]{ return &this->vertex_particle_id; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["vertex_particle_id"] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.get_device_field = [this]{ return &this->vertex_system_id; };
            reg.fields["vertex_system_id"] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.get_device_field = [this]{ return &this->vertex_system_offset; };
            reg.fields["vertex_system_offset"] = p;
        }
        {
            FieldSpec1D<int> p; 
            p.get_device_field = [this]{ return &this->vertex_system_size; };
            reg.fields["vertex_system_size"] = p;
        }
        {
            FieldSpec2D<double> p; 
            p.get_device_field = [this]{ return &this->vertex_pos; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["vertex_pos"] = p;
        }
        {
            FieldSpec2D<double> p; 
            p.get_device_field = [this]{ return &this->vertex_vel; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["vertex_vel"] = p;
        }
        {
            FieldSpec2D<double> p; 
            p.get_device_field = [this]{ return &this->vertex_force; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["vertex_force"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->vertex_pe; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["vertex_pe"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->vertex_mass; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["vertex_mass"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->vertex_rad; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["vertex_rad"] = p;
        }
        {
            FieldSpec2D<double> p; 
            p.preprocess = [this]{ this->compute_stress_tensor(); };
            p.get_device_field = [this]{ return &this->stress_tensor_x; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["stress_tensor_x"] = p;
        }
        {
            FieldSpec2D<double> p; 
            p.preprocess = [this]{ this->compute_stress_tensor(); };
            p.get_device_field = [this]{ return &this->stress_tensor_y; };
            p.index_by = [order_str]{ return order_str; };
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
        if constexpr (has_output_build_registry_poly_extras_impl<Derived>::value)
            this->derived().output_build_registry_poly_extras_impl(reg);
    }

private:
    // Assign global cell ids to each vertex based on their position
    void _assign_cell_ids() {
        const int Nv = this->n_vertices();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(Nv);
        CUDA_LAUNCH(md::geo::assign_cell_ids_kernel, G, B,
            Nv,
            this->vertex_system_id.ptr(),
            this->vertex_pos.xptr(), this->vertex_pos.yptr(),
            this->cell_id.ptr()
        );
    }

    // Rebuild the cell layout using bucket sort
    void _rebuild_cell_layout() {
        const int Nv = this->n_vertices();
        const int C = this->n_cells();
        // 1) count the number of vertices in each cell by building a histogram using atomicAdd at the vertex-level
        this->cell_count.fill(0);
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(Nv);
        CUDA_LAUNCH(md::geo::count_cells_kernel, G, B,
            Nv, this->cell_id.ptr(), this->cell_count.ptr()
        );
        // 2) determine the starting vertex index (id of first vertex) for each cell
        thrust::exclusive_scan(this->cell_count.begin(), this->cell_count.end(), this->cell_start.begin(), 0);
        this->cell_start.set_element(C, Nv);
        thrust::copy(this->cell_start.begin(), this->cell_start.begin() + C, this->cell_aux.begin());
        // compute the order and its inverse using atomicAdd at the vertex-level
        CUDA_LAUNCH(md::geo::scatter_order_kernel, G, B,
            Nv, this->cell_id.ptr(), this->cell_aux.ptr(), this->order.ptr(), this->order_inv.ptr()
        );
    }

    // Count the number of vertices in each cell
    void _count_cell_neighbors() {
        const int Nv = this->n_vertices();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(Nv);
        CUDA_LAUNCH(md::poly::count_vertex_cell_neighbors_kernel, G, B,
            this->vertex_pos.xptr(), this->vertex_pos.yptr(), this->vertex_rad.ptr(),
            this->cell_id.ptr(), this->cell_start.ptr(), this->neighbor_count.ptr()
        );
    }

    // Fill the neighbor list for each vertex by enumerating over the 9-cell stencil
    void _fill_cell_neighbors() {
        const int Nv = this->n_vertices();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(Nv);
        CUDA_LAUNCH(md::poly::fill_vertex_cell_neighbor_list_kernel, G, B,
            this->vertex_pos.xptr(), this->vertex_pos.yptr(), this->vertex_rad.ptr(),
            this->cell_id.ptr(), this->cell_start.ptr(), this->neighbor_start.ptr(), this->neighbor_ids.ptr()
        );
    }

};

} // namespace md
