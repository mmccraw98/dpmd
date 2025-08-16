#pragma once
#include "particles/base_particle.hpp"

namespace md {

class Rigid : public BaseParticle<Rigid> {
public:
    using Base = BaseParticle<Rigid>;

    df::DeviceField1D<double> angle;
    df::DeviceField2D<double> vertex_pos;
    df::DeviceField1D<int> particle_id;
    df::DeviceField1D<int> particle_offset;
    df::DeviceField1D<int> n_vertices_per_particle;
    df::DeviceField1D<double> particle_pe;

    int n_vertices()  const { return vertex_pos.size(); }

    Rigid() = default;

    // Persistent buffers for radix sort
    thrust::device_vector<unsigned long long> sort_keys_in_;   // N keys before sort
    thrust::device_vector<unsigned long long> sort_keys_out_;  // N keys after sort
    thrust::device_vector<int> order_in_;                      // N particle indices before sort
    thrust::device_vector<int> order_out_;                     // N particle indices after sort

    // Persistent CUB temp storage for radix sort
    void* cub_temp_storage_ = nullptr;
    size_t cub_temp_bytes_ = 0;

    // ===== CRTP-required methods =====
    void init_naive_neighbors_impl();
    void update_naive_neighbors_impl() {};
    void init_cell_neighbors_impl() {};
    void update_cell_neighbors_impl() {};
    void check_cell_neighbors_impl() {};
    void compute_forces_impl();
    void compute_wall_forces_impl();
    void sync_class_constants_impl();
    void update_positions_impl(double /*scale*/) {};
    void update_velocities_impl(double /*scale*/) {};

    void set_random_particle_positions(double x_min, double x_max, double y_min, double y_max);

    void sum_vertex_pe_to_particle_pe();

    // ===== Optional CRTP hooks =====
    // Reorder disk-only fields in one fused pass (once you add them).
    void reorder_extra_impl(const int* /*d_perm*/) {
        // No extra fields yet; nothing to do.
    }

    // Seed RNG for disk-only fields (positions/vel/rad/mass already seeded by Base).
    void seed_rng_extra_impl(unsigned long long /*seed*/, unsigned long long /*base_tag*/) {
        // No extra fields yet; nothing to do.
    }

    // Toggle swap buffers for disk-only fields.
    void enable_swap_impl(bool /*enable*/) {
        // No extra fields yet; nothing to do.
    }

    // Disk-specific methods
    void build_neighbor_list();

    // ===== Convenience views (if you want disk-specific aliases) =====
    // using Base::pos_view; // already exposed
    // Add more views if/when you add disk-only arrays.
};

} // namespace md