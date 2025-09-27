#pragma once
#include "particles/base_particle.hpp"
#include "utils/cuda_utils.cuh"
#include "utils/h5_io.hpp"
#include "kernels/base_point_particle_kernels.cuh"
#include <cub/device/device_segmented_reduce.cuh>
#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/execution_policy.h>
#include <type_traits>
#include <utility>

template<class T, class = void>
struct has_allocate_point_extras_impl : std::false_type {};
template<class T>
struct has_allocate_point_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().allocate_point_extras_impl(0))>> : std::true_type {};

template<class T, class = void>
struct has_allocate_point_system_extras_impl : std::false_type {};
template<class T>
struct has_allocate_point_system_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().allocate_point_system_extras_impl(0))>> : std::true_type {};

template<class T, class = void>
struct has_enable_point_swap_extras_impl : std::false_type {};
template<class T>
struct has_enable_point_swap_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().enable_point_swap_extras_impl(false))>> : std::true_type {};

template<class T, class = void>
struct has_load_static_from_hdf5_point_extras_impl : std::false_type {};
template<class T>
struct has_load_static_from_hdf5_point_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().load_static_from_hdf5_point_extras_impl(std::declval<hid_t>()))>> : std::true_type {};

template<class T, class = void>
struct has_load_from_hdf5_point_extras_impl : std::false_type {};
template<class T>
struct has_load_from_hdf5_point_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().load_from_hdf5_point_extras_impl(std::declval<hid_t>()))>> : std::true_type {};


template<class T, class = void>
struct has_get_static_field_names_point_extras_impl : std::false_type {};
template<class T>
struct has_get_static_field_names_point_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().get_static_field_names_point_extras_impl())>> : std::true_type {};


template<class T, class = void>
struct has_get_state_field_names_point_extras_impl : std::false_type {};
template<class T>
struct has_get_state_field_names_point_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().get_state_field_names_point_extras_impl())>> : std::true_type {};


template<class T, class = void>
struct has_output_build_registry_point_extras_impl : std::false_type {};
template<class T>
struct has_output_build_registry_point_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().output_build_registry_point_extras_impl(std::declval<io::OutputRegistry&>()))>> : std::true_type {};

template<class T, class = void>
struct has_init_cell_neighbors_extras_impl : std::false_type {};
template<class T>
struct has_init_cell_neighbors_extras_impl<T,
    std::void_t<decltype(std::declval<T&>().init_cell_neighbors_extras_impl())>> : std::true_type {};

namespace md {

// Enum for the different cell sort methods
enum class CellSortMethod { Bucket, Standard };

// Base class for point particles
template<class Derived>
class BasePointParticle : public BaseParticle<Derived> {
public:
    using Base = BaseParticle<Derived>;

    df::DeviceField1D<double>        e_interaction;  // (S,) interaction energy
    df::DeviceField1D<double>        mass;           // (N,) mass
    df::DeviceField1D<double>        rad;            // (N,) radius
    df::DeviceField1D<int>           cell_aux;       // (C,) auxiliary data for each cell

    inline static constexpr CellSortMethod cell_sort_method = CellSortMethod::Bucket;  // default sort method for the cell list

    // Allocate the particles
    void allocate_particles_impl(int N) {
        this->system_id.resize(N);
        this->mass.resize(N);
        this->rad.resize(N);
        if constexpr (has_allocate_point_extras_impl<Derived>::value)
            this->derived().allocate_point_extras_impl(N);
    }

    // Allocate the systems
    void allocate_systems_impl(int S) {
        this->e_interaction.resize(S);
        if constexpr (has_allocate_point_system_extras_impl<Derived>::value)
            this->derived().allocate_point_system_extras_impl(S);
    }

    // Enable/disable the swap for the point particle system
    void enable_swap_impl(bool enable) {
        if (enable) {
            this->pos.enable_swap();
            this->vel.enable_swap();
            this->force.enable_swap();
            this->mass.enable_swap();
            this->rad.enable_swap();
            this->cell_id.enable_swap();
        } else {
            this->pos.disable_swap();
            this->vel.disable_swap();
            this->force.disable_swap();
            this->mass.disable_swap();
            this->rad.disable_swap();
            this->cell_id.disable_swap();
        }
        if constexpr (has_enable_point_swap_extras_impl<Derived>::value)
            this->derived().enable_point_swap_extras_impl(enable);
    }

    // Initialize the naive neighbors
    void init_naive_neighbors_impl() {
        int N = this->n_particles();
        this->neighbor_count.resize(N);
        this->neighbor_start.resize(N + 1);
    }

    // Update the naive neighbors (should only be done once per simulation - never needs to be called again)
    void update_naive_neighbors_impl() {
        int N = this->n_particles();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(N);
        // 1) count the neighbors (every particle in system i has system_size[i] - 1 neighbors)
        CUDA_LAUNCH(md::point::set_naive_neighbor_count, G, B,
            this->neighbor_count.ptr()
        );
        // 2) scan the neighbor counts to get the starting index of each particle's neighbor list
        thrust::exclusive_scan(this->neighbor_count.begin(), this->neighbor_count.end(), this->neighbor_start.begin());
        // 3) count the total number of neighbors
        int total_neighbors = this->n_neighbors();
        // 4) set start[N] on device
        this->neighbor_start.set_element(N, total_neighbors);
        // 5) size neighbor_ids
        this->neighbor_ids.resize(total_neighbors);
        // 6) fill neighbor_ids
        CUDA_LAUNCH(md::point::fill_naive_neighbor_list_kernel, G, B,
            this->neighbor_start.ptr(), this->neighbor_ids.ptr()
        );
    }

    // Initialize the cell neighbors
    void init_cell_neighbors_impl() {
        int N = this->n_particles();
        this->cell_id.resize(N);
        this->order.resize(N);
        this->order_inv.resize(N);
        this->static_index.resize(N);
        thrust::sequence(this->static_index.begin(), this->static_index.end(), 0);
        this->neighbor_count.resize(N);
        this->neighbor_start.resize(N + 1);
        if constexpr (cell_sort_method == CellSortMethod::Bucket) {
            this->cell_aux.resize(this->n_cells());
        } else {
            this->cell_aux.resize(this->n_particles());
        }
        if constexpr (has_init_cell_neighbors_extras_impl<Derived>::value)
            this->derived().init_cell_neighbors_extras_impl();
    }

    // Update the cell neighbors
    void update_cell_neighbors_impl() {
        int C = this->n_cells();
        int N = this->n_particles();
        // 1) assign cell ids to each particle
        this->_assign_cell_ids();
        // 2) rebuild the cell layout using either bucket or sort, 
        // count the number of particles in each cell,
        // determine the starting particle index for each cell,
        // and determine the particle order and inverse order
        this->_rebuild_cell_layout();
        // 3) use the order array to rearrange the particle data so that spatially nearby particles are nearby in the relevant data arrays
        this->reorder_particles();
        // 4) count the number of neighbors for each particle
        this->_count_cell_neighbors();
        // 5) scan the neighbor counts to get the starting index of each particle's neighbor list
        const int total_neighbors = this->n_neighbors();
        thrust::exclusive_scan(this->neighbor_count.begin(), this->neighbor_count.end(), this->neighbor_start.begin());
        this->neighbor_start.set_element(N, total_neighbors);
        this->neighbor_ids.resize(total_neighbors);
        // 6) create the neighbor list, again enumerating over the 9-cell stencil
        this->_fill_cell_neighbors();
    }

    // Update the static index
    void update_static_index_impl() {
        int N = this->n_particles();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(N);
        CUDA_LAUNCH(md::geo::update_static_index_kernel, G, B, N, this->order_inv.ptr(), this->static_index.ptr());
    }

    // Compute the stress tensor for each system
    void compute_stress_tensor_total_impl() {
        this->compute_stress_tensor();
        this->segmented_sum(this->stress_tensor_x.xptr(), this->stress_tensor_total_x.xptr(), 0);
        this->segmented_sum(this->stress_tensor_y.yptr(), this->stress_tensor_total_y.yptr(), 0);
        this->segmented_sum(this->stress_tensor_x.yptr(), this->stress_tensor_total_y.xptr(), 0);
        this->segmented_sum(this->stress_tensor_y.xptr(), this->stress_tensor_total_x.yptr(), 0);
    }

    // Load static data from hdf5 group and initialize the particle
    void load_static_from_hdf5_group_impl(hid_t group) {
        this->e_interaction.from_host(read_vector<double>(group, "e_interaction"));
        this->mass.from_host(read_vector<double>(group, "mass"));
        this->rad.from_host(read_vector<double>(group, "rad"));
        if constexpr (has_load_static_from_hdf5_point_extras_impl<Derived>::value)
            this->derived().load_static_from_hdf5_point_extras_impl(group);
    }

    // Load from hdf5 group and initialize the particle
    void load_from_hdf5_group_impl(hid_t group) {
        if (h5_link_exists(group, "pos")) {
            this->pos.from_host(read_vector_2d<double>(group, "pos"));
        }
        if (h5_link_exists(group, "vel")) {
            this->vel.from_host(read_vector_2d<double>(group, "vel"));
        }
        if (h5_link_exists(group, "force")) {
            this->force.from_host(read_vector_2d<double>(group, "force"));
        }
        if constexpr (has_load_from_hdf5_point_extras_impl<Derived>::value)
            this->derived().load_from_hdf5_point_extras_impl(group);
    }
 
    // Get the names of the fields that should be saved as static
    std::vector<std::string> get_static_field_names_impl() {
        std::vector<std::string> static_names {"e_interaction", "mass", "rad", "area"};
        std::vector<std::string> derived_names = this->derived().get_static_field_names_point_extras_impl();
        static_names.insert(static_names.end(), derived_names.begin(), derived_names.end());
        return static_names;
    }

    // Get the names of the fields that should be saved as state
    std::vector<std::string> get_state_field_names_impl() {
        std::vector<std::string> state_names {"pos", "vel", "force"};
        std::vector<std::string> derived_names = this->derived().get_state_field_names_point_extras_impl();
        state_names.insert(state_names.end(), derived_names.begin(), derived_names.end());
        return state_names;
    }

    // Build the output registry
    void output_build_registry_impl(io::OutputRegistry& reg) {
        // Register point-specific fields
        using io::FieldSpec1D; using io::FieldSpec2D;
        std::string order_str = "static_index";
        // Expose canonical->current permutation so OutputManager can restore the original ordering
        {
            FieldSpec1D<int> p;
            p.get_device_field = [this]{ return &this->static_index; };
            reg.fields[order_str] = p;
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
        {
            FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->pe; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["pe"] = p;
        }
        {
            FieldSpec1D<double> p; 
            p.preprocess = [this]{ this->compute_ke(); };
            p.get_device_field = [this]{ return &this->ke; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["ke"] = p;
        }
        {
            io::FieldSpec2D<double> p; 
            p.get_device_field = [this]{ return &this->pos; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["pos"] = p;
        }
        {
            io::FieldSpec2D<double> p; 
            p.get_device_field = [this]{ return &this->vel; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["vel"] = p;
        }
        {
            io::FieldSpec2D<double> p; 
            p.get_device_field = [this]{ return &this->force; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["force"] = p;
        }
        {
            io::FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->e_interaction; };
            reg.fields["e_interaction"] = p;
        }
        {
            io::FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->mass; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["mass"] = p;
        }
        {
            io::FieldSpec1D<double> p; 
            p.get_device_field = [this]{ return &this->rad; };
            p.index_by = [order_str]{ return order_str; };
            reg.fields["rad"] = p;
        }
        if constexpr (has_output_build_registry_point_extras_impl<Derived>::value)
            this->derived().output_build_registry_point_extras_impl(reg);
    }


private:
    // Assign global cell ids to each particle based on their position + the system cell offsets
    void _assign_cell_ids() {
        const int N = this->n_particles();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(N);
        CUDA_LAUNCH(md::geo::assign_cell_ids_kernel, G, B,
            N,
            this->system_id.ptr(),
            this->pos.xptr(), this->pos.yptr(),
            this->cell_id.ptr()
        );
    }

    // Count the number of particles in each cell
    void _count_cell_neighbors() {
        const int N = this->n_particles();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(N);
        CUDA_LAUNCH(md::point::count_cell_neighbors_kernel, G, B,
            this->pos.xptr(), this->pos.yptr(), this->rad.ptr(),
            this->cell_id.ptr(), this->cell_start.ptr(), this->neighbor_count.ptr()
        );
    }

    // Fill the neighbor list for each particle by enumerating over the 9-cell stencil (particle's cell and 8 surrounding cells)
    void _fill_cell_neighbors() {
        const int N = this->n_particles();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(N);
        CUDA_LAUNCH(md::point::fill_neighbors_cell_kernel, G, B,
            this->pos.xptr(), this->pos.yptr(), this->rad.ptr(), 
            this->cell_id.ptr(), this->cell_start.ptr(), this->neighbor_start.ptr(), this->neighbor_ids.ptr()
        );
    }

    // Rebuild the cell layout using bucket sort
    void _rebuild_cell_layout_bucket() {
        const int N = this->n_particles();
        const int C = this->n_cells();
        // 1) count the number of particles in each cell by building a histogram using atomicAdd at the particle-level
        this->cell_count.fill(0);
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(N);
        CUDA_LAUNCH(md::geo::count_cells_kernel, G, B,
            N, this->cell_id.ptr(), this->cell_count.ptr()
        );
        // 2) determine the starting particle index (id of first particle) for each cell
        thrust::exclusive_scan(this->cell_count.begin(), this->cell_count.end(), this->cell_start.begin(), 0);
        this->cell_start.set_element(C, N);
        thrust::copy(thrust::device, this->cell_start.begin(), this->cell_start.begin() + C, this->cell_aux.begin());
        // compute the order and its inverse using atomicAdd at the particle-level
        CUDA_LAUNCH(md::geo::scatter_order_kernel, G, B,
            N, this->cell_id.ptr(), this->cell_aux.ptr(), this->order.ptr(), this->order_inv.ptr()
        );

    }

    // Rebuild the cell layout using standard sort
    void _rebuild_cell_layout_sort() {
        const int N = this->n_particles();
        const int C = this->n_cells();
        // 1) generate the particle order and sort by a copied cell id since it is expected that cell_id is not sorted
        thrust::sequence(this->order.begin(), this->order.end(), 0);
        thrust::copy(this->cell_id.begin(), this->cell_id.end(), this->cell_aux.begin());
        thrust::sort_by_key(this->cell_aux.begin(), this->cell_aux.end(), this->order.begin());
        // compute the order and its inverse using the scatter kernel
        thrust::scatter(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + N,
            this->order.begin(),
            this->order_inv.begin());
        // 2) determine the starting particle index (id of first particle) for each cell
        // use binary search to find the first particle in each cell and store the result in cell_start
        auto keys_begin = this->cell_aux.begin(); // now sorted ascending
        thrust::counting_iterator<int> c0(0), cend(C);
        thrust::lower_bound(
            thrust::device,
            keys_begin, keys_begin + N,
            c0, cend,
            this->cell_start.begin());
        this->cell_start.set_element(C, N);
    }

    // Rebuild the cell layout using the currently selected method
    void _rebuild_cell_layout() {
        if constexpr (cell_sort_method == CellSortMethod::Bucket) {
            this->_rebuild_cell_layout_bucket();
        } else {
            this->_rebuild_cell_layout_sort();
        }   
    }
};

} // namespace md
