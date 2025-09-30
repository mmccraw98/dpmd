#pragma once

#include "particles/base_poly_particle.hpp"
#include "utils/output_manager.hpp"

namespace md::rigid_bumpy {

// ---- Rigid Bumpy-specific constants ----
struct RigidBumpyConst {  // TODO: should add more to this once the class is fully implemented
    const double* e_interaction;
    const double* vertex_rad;
    const double* mass;
    const double* moment_inertia;
    unsigned int* rebuild_flag;
};

// Rigid Bumpy-specific device constants
extern __constant__ RigidBumpyConst g_rigid_bumpy;

// Bind the rigid bumpy constants to the device
void bind_rigid_bumpy_globals(const double* d_e_interaction, const double* d_vertex_rad, const double* d_mass, const double* d_moment_inertia, unsigned int* d_rebuild_flag);

// Rigid Bumpy particle class
class RigidBumpy : public md::BasePolyParticle<md::rigid_bumpy::RigidBumpy> {
    template<class> friend class md::BaseParticle;
    template<class> friend class md::BasePolyParticle;
public:
    using Base = md::BasePolyParticle<md::rigid_bumpy::RigidBumpy>;

    // ---- Rigid Bumpy-specific fields ----
    df::DeviceField2D<double>       last_pos;       // (N,2) - positions of the first vertex in each particle when neighbor list was last built
    df::DeviceField1D<double>       disp2;          // (N,) - displacement squared since last pos was written
    df::DeviceField1D<unsigned int> rebuild_flag;   // (S,) - rebuild flag for each system
    df::DeviceField1D<double>       angle;          // (N,) - angle of the particle
    df::DeviceField1D<double>       torque;         // (N,) - torque on the particle
    df::DeviceField1D<double>       angular_vel;    // (N,) - angular velocity of the particle
    df::DeviceField1D<double>       mass;           // (N,) - mass of the particle
    df::DeviceField1D<double>       moment_inertia; // (N,) - moment of inertia of the particle
    df::DeviceField1D<double>       friction_coeff; // (N_particle_neighbors,) - friction coefficient for each pair of particles
    df::DeviceField2D<int>          pair_vertex_contacts;     // (N_particle_neighbors,2) - number of vertex contacts for each pair of particles

    // Sum up the forces on the particles
    void compute_particle_forces();

    // Compute the pairwise forces on the particles
    void compute_forces_impl();

    // Compute the wall forces
    void compute_wall_forces_impl();

    // Compute the damping forces
    void compute_damping_forces_impl(df::DeviceField1D<double> scale);

    // Update the positions of the particles
    void update_positions_impl(df::DeviceField1D<double> scale, double scale2);

    // Update the velocities of the particles
    void update_velocities_impl(df::DeviceField1D<double> scale, double scale2);

    // Scale the velocities of the particles
    void scale_velocities_impl(df::DeviceField1D<double> scale);

    // Calculate the average velocity of the systems
    df::DeviceField2D<double> calculate_average_velocity_impl();

    // Set the average velocity of the systems
    void set_average_velocity_impl(df::DeviceField2D<double> average_velocity);

    // Scale the positions of the particles
    void scale_positions_impl(df::DeviceField1D<double> scale);

    // Mix velocities and forces - system-level alpha, primarily used for FIRE
    void mix_velocities_and_forces_impl(df::DeviceField1D<double> alpha);

    // Allocate the poly particle-level extras
    void allocate_poly_extras_impl(int N);

    // Allocate the poly vertex-level extras
    void allocate_poly_vertex_extras_impl(int Nv);

    // Allocate the poly system extras
    void allocate_poly_system_extras_impl(int S);

    // Sync the class constants
    void sync_class_constants_poly_extras_impl();

    // Enable/disable the swap for the poly particle system
    void enable_poly_swap_extras_impl(bool);

    // Reorder particles by the internal order array
    void reorder_particles_impl();

    // Reset the displacements of the particles to the current positions
    void reset_displacements_impl();

    // Check if the cell neighbors need to be rebuilt
    bool check_cell_neighbors_impl();

    // Compute the kinetic energy of each particle
    void compute_ke_impl();

    // Set the number of degrees of freedom for each system
    void set_n_dof_impl();

    // Set random positions within the box with padding defaulting to 0.0 inherited from BaseParticle
    void set_random_positions_impl(double box_pad_x, double box_pad_y);

    // Set random positions of particles within their domains
    void set_random_positions_in_domains_impl();

    // Compute the total power for each system
    void compute_fpower_total_impl();

    // Compute the number of contacts for each particle
    void compute_contacts_impl();

    // Compute the friction coefficients for each pair of particles
    void compute_friction_coeff();

    // Compute the distances between each pair of particles
    void compute_pair_dist_impl();

    // Compute the overlaps for each particle
    void compute_overlaps_impl();

    // Compute the stress tensor for each system
    void compute_stress_tensor_impl();

    // Initialize the cell neighbors
    void init_cell_neighbors_poly_extras_impl();

    void save_state_impl(df::DeviceField1D<int> flag, int true_val);

    void restore_state_impl(df::DeviceField1D<int> flag, int true_val);

    void load_static_from_hdf5_poly_extras_impl(hid_t group);

    void load_from_hdf5_poly_extras_impl(hid_t group);

    // Get the class name
    std::string get_class_name_impl();

    // Get the names of the fields that should be saved as static
    std::vector<std::string> get_static_field_names_poly_extras_impl();

    // Get the names of the fields that should be saved as state
    std::vector<std::string> get_state_field_names_poly_extras_impl();

    // Build the output registry
    void output_build_registry_poly_extras_impl(io::OutputRegistry& reg);

private:
    df::DeviceField2D<double> last_state_pos;
    df::DeviceField1D<double> last_state_angle;
    df::DeviceField1D<double> last_state_mass;
    df::DeviceField1D<int> last_state_n_vertices_per_particle;
    df::DeviceField1D<int> last_state_particle_offset;
    df::DeviceField1D<double> last_state_moment_inertia;
    df::DeviceField1D<double> last_state_vertex_rad;
    df::DeviceField2D<double> last_state_vertex_pos;
    df::DeviceField1D<int> last_state_vertex_particle_id;
    df::DeviceField2D<double> last_state_box_size;
    df::DeviceField1D<int> last_state_static_index;
};

}
