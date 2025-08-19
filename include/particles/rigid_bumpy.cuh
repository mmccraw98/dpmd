#pragma once

#include "particles/base_poly_particle.hpp"

namespace md::rigid_bumpy {

// ---- Rigid Bumpy-specific constants ----
struct RigidBumpyConst {  // TODO: should add more to this once the class is fully implemented
    const double* e_interaction;
    const double* vertex_rad;
};

// Rigid Bumpy-specific device constants
extern __constant__ RigidBumpyConst g_rigid_bumpy;

// Bind the rigid bumpy constants to the device
void bind_rigid_bumpy_globals(const double* d_e_interaction, const double* d_vertex_rad);

// Rigid Bumpy particle class
class RigidBumpy : public md::BasePolyParticle<md::rigid_bumpy::RigidBumpy> {
    template<class> friend class md::BaseParticle;
    template<class> friend class md::BasePolyParticle;
public:
    using Base = md::BasePolyParticle<md::rigid_bumpy::RigidBumpy>;

    // ---- Rigid Bumpy-specific fields ----
    df::DeviceField2D<double>       last_pos;       // (N,2) - positions of the particles when neighbor list was last built
    df::DeviceField1D<double>       disp2;          // (N,) - displacement squared since last pos was written
    df::DeviceField1D<unsigned int> rebuild_flag;   // (S,) - rebuild flag for each system
    df::DeviceField1D<double>       angle;          // (N,) - angle of the particle
    df::DeviceField1D<double>       torque;         // (N,) - torque on the particle
    df::DeviceField1D<double>       angular_vel;    // (N,) - angular velocity of the particle
    df::DeviceField1D<double>       moment_inertia; // (N,) - moment of inertia of the particle

    // Sum up the forces on the particles
    void compute_particle_forces();  // TODO: could raise this to BasePolyParticle????

    // Compute the pairwise forces on the particles
    void compute_forces_impl();  // TODO: could raise this to BasePolyParticle - or at least separate into vertex-level and particle-level

    // Compute the wall forces
    void compute_wall_forces_impl();  // TODO: could raise this to BasePolyParticle

    // Compute the damping forces
    void compute_damping_forces_impl(double scale);

    // Update the positions of the particles
    void update_positions_impl(double scale);

    // Update the velocities of the particles
    void update_velocities_impl(double scale);

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

    // Set random positions within the box with padding defaulting to 0.0 inherited from BaseParticle
    void set_random_positions_impl(double box_pad_x, double box_pad_y);
};

}