#pragma once
#include "particles/base_particle.hpp"
#include "kernels/base_poly_particle_kernels.cuh"
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

    inline static constexpr CellSortMethod cell_sort_method = CellSortMethod::Bucket;  // default sort method for the cell list

    // Allocate the particles
    void allocate_particles_impl(int N) {
        this->n_vertices_per_particle.resize(N);
        this->particle_offset.resize(N+1);
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
        if constexpr (has_allocate_poly_vertex_extras_impl<Derived>::value)
            this->derived().allocate_poly_vertex_extras_impl(Nv);
    }

    // Allocate the systems
    void allocate_systems_impl(int S) {
        this->e_interaction.resize(S);
        this->vertex_system_offset.resize(S+1);
        this->vertex_system_size.resize(S);
        if constexpr (has_allocate_poly_system_extras_impl<Derived>::value)
            this->derived().allocate_poly_system_extras_impl(S);
    }

    // Enable/disable the swap for the point particle system
    void enable_swap_impl(bool enable) {
        if (enable) {
            this->n_vertices_per_particle.enable_swap();
            this->vertex_pos.enable_swap();
            this->vertex_vel.enable_swap();
            this->vertex_force.enable_swap();
            this->vertex_pe.enable_swap();
            this->vertex_mass.enable_swap();
            this->vertex_rad.enable_swap();
            this->vertex_particle_id.enable_swap();
            this->cell_id.enable_swap();
        } else {
            this->n_vertices_per_particle.disable_swap();
            this->vertex_pos.disable_swap();
            this->vertex_vel.disable_swap();
            this->vertex_force.disable_swap();
            this->vertex_pe.disable_swap();
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
        throw std::runtime_error("BasePolyParticle::init_cell_neighbors_impl is not implemented");
    }

    // Update the cell neighbors
    void update_cell_neighbors_impl() {
        throw std::runtime_error("BasePolyParticle::update_cell_neighbors_impl is not implemented");
    }

    // Sync class constants with pass-through to sub-class for any extra constants
    void sync_class_constants_impl() {
        md::poly::bind_poly_globals(this->vertex_particle_id.ptr(), this->particle_offset.ptr(), this->n_vertices_per_particle.ptr());
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

private:

};

} // namespace md