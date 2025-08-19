#pragma once

#include "particles/base_point_particle.hpp"

namespace md::disk {

// ---- Disk-specific constants ----
struct DiskConst {
    const double* e_interaction;
    const double* mass;
    const double* rad;
    unsigned int* rebuild_flag;
};

// Disk-specific device constants
extern __constant__ DiskConst g_disk;

// Bind the disk constants to the device
void bind_disk_globals(const double* d_e_interaction, const double* d_mass, const double* d_rad, unsigned int* d_rebuild_flag, const double* d_thresh2);

// Disk particle class
class Disk : public md::BasePointParticle<md::disk::Disk> {
    template<class> friend class md::BaseParticle;
    template<class> friend class md::BasePointParticle;
public:
    using Base = md::BasePointParticle<md::disk::Disk>;

    // ---- Disk-specific fields ----
    df::DeviceField2D<double>       last_pos;     // (N,2) - positions of the particles when neighbor list was last built
    df::DeviceField1D<double>       disp2;        // (N,) - displacement squared since last pos was written
    df::DeviceField1D<unsigned int> rebuild_flag; // (S,) - rebuild flag for each system

    // Compute the pairwise forces on the particles
    void compute_forces_impl();

    // Compute the wall forces
    void compute_wall_forces_impl();

    // Compute the damping forces
    void compute_damping_forces_impl(double scale);

    // Update the positions of the particles
    void update_positions_impl(double scale);

    // Update the velocities of the particles
    void update_velocities_impl(double scale);

    // Sync the class constants
    void sync_class_constants_impl();

    // Allocate the point extras - nothing extra for disks
    void allocate_point_extras_impl(int N) {
        switch (Base::neighbor_method) {
            case NeighborMethod::Naive:
                break;
            case NeighborMethod::Cell:
                this->last_pos.resize(N);
                this->disp2.resize(N);
                break;
        }
    }

    // Allocate the point system extras - nothing extra for disks
    void allocate_point_system_extras_impl(int S) {
        this->rebuild_flag.resize(S);
        this->rebuild_flag.fill(0u);
    }

    // Enable/disable the swap for the point particle system - nothing extra for disks
    void enable_point_swap_extras_impl(bool) {}

    // Reorder particles by the internal order array
    void reorder_particles_impl();

    // Reset the displacements of the particles to the current positions
    void reset_displacements_impl();

    // Check if the cell neighbors need to be rebuilt
    bool check_cell_neighbors_impl();

    // Compute the kinetic energy of each particle
    void compute_ke_impl();

    // Set random positions within the box
    void set_random_positions_impl(double box_pad_x, double box_pad_y);
};

}
