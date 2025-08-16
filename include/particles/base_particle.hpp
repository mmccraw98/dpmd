#pragma once
#include "utils/device_fields.hpp"
#include "kernels/common.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <stdexcept>
#include <cstddef>

namespace md {

// Simple pointer views
template <class T> struct Vec1  { T* p; };
template <class T> struct CVec1 { const T* p; };
template <class T> struct Vec2  { T* x; T* y; };
template <class T> struct CVec2 { const T* x; const T* y; };

enum class NeighborMethod : int { Naive = 0, Cell = 1 };

template <class Derived>
class BaseParticle {
public:
    // === Particle-level SoA (owned, device) ===
    df::DeviceField2D<double> pos;            // (N,2)
    df::DeviceField2D<double> last_pos;       // (N,2)
    df::DeviceField2D<double> disp2;          // (N,2)
    df::DeviceField2D<double> vel;            // (N,2)
    df::DeviceField2D<double> force;          // (N,2)
    df::DeviceField1D<double> rad;            // (N,)
    df::DeviceField1D<double> area;           // (N,)
    df::DeviceField1D<double> mass;           // (N,)
    df::DeviceField1D<double> pe;             // (N,)
    df::DeviceField1D<double> ke;             // (N,)
    df::DeviceField1D<int>    system_id;      // (N,) — assumed static - effectively true if systems arent permuted
    df::DeviceField1D<int>    neighbor_count; // (N,) - number of neighbors for each particle
    df::DeviceField1D<int>    neighbor_start; // (N+1,) - starting index of the neighbor list for a given particle in the neighbor_ids list
    df::DeviceField1D<int>    neighbor_ids;   // (total_neighbors,) - list of neighbor ids for all particles
    df::DeviceField1D<int>    cell_id;        // (N,) - id of the cell that each particle belongs to
    df::DeviceField1D<int>    cell_count;     // (total_cells,) - number of particles in each cell
    df::DeviceField1D<int>    cell_start;     // (total_cells+1,) - starting particle index of the particles in each cell
    df::DeviceField1D<int>    order;          // (N,) - sorted particle index
    df::DeviceField1D<int>    order_inv;      // (N,) - inverse of the sorted particle index

    // === Per-system (owned, device) ===
    df::DeviceField1D<int>    system_size;       // (S,) - number of particles in each system
    df::DeviceField1D<int>    system_offset;     // (S+1) - starting index of the particles in each system
    df::DeviceField2D<double> box_size;          // (S,2) [Lx,Ly] - size of the box for each system
    df::DeviceField2D<double> box_inv;           // (S,2) [1/Lx,1/Ly] - inverse of the box size for each system
    df::DeviceField2D<double> cell_size;         // (S,2) [Lx,Ly] - size of the cell for each system
    df::DeviceField2D<double> cell_inv;          // (S,2) [1/Lx,1/Ly] - inverse of the cell size for each system
    df::DeviceField2D<int>    cell_dim;          // (S,2) [Nx,Ny] - number of cells in each dimension for each system
    df::DeviceField1D<int>    cell_system_start; // (S+1,) - starting index of the cells for each system in the cell_ids list
    df::DeviceField1D<double> verlet_skin;       // (S,) - verlet skin for each system
    df::DeviceField1D<double> e_interaction;     // (S,) - interaction energy for each system
    df::DeviceField1D<double> packing_fraction;  // (S,) - packing fraction for each system

    // ---- Neighbor method ----
    NeighborMethod neighbor_method = NeighborMethod::Naive;

    // ---- Allocation (call once up-front) ----
    void allocate_particles(int N) {
        pos.resize(N); vel.resize(N); force.resize(N);
        rad.resize(N); mass.resize(N); system_id.resize(N);
        pe.resize(N); ke.resize(N);
        last_pos.resize(N); disp2.resize(N);
    }
    void allocate_systems(int S) {
        box_size.resize(S); box_inv.resize(S); system_size.resize(S); system_offset.resize(S+1);
        e_interaction.resize(S); packing_fraction.resize(S);
    }

    // ---- Views for kernels ----
    Vec2<double>  pos_view()                 { return { pos.xptr(),    pos.yptr()    }; }
    CVec2<double> pos_view()           const { return { pos.xptr(),    pos.yptr()    }; }
    Vec2<double>  vel_view()                 { return { vel.xptr(),    vel.yptr()    }; }
    CVec2<double> vel_view()           const { return { vel.xptr(),    vel.yptr()    }; }
    Vec2<double>  force_view()               { return { force.xptr(),  force.yptr()  }; }
    CVec2<double> force_view()         const { return { force.xptr(),  force.yptr()  }; }
    CVec1<double> radius_view()        const { return { rad.ptr()       }; }
    CVec1<double> mass_view()          const { return { mass.ptr()      }; }

    // ---- Bind globals ----
    void sync_box() {
        geo::bind_box_globals(box_size.xptr(), box_size.yptr(), box_inv.xptr(), box_inv.yptr());
    }

    void sync_system() {
        geo::bind_system_globals(system_offset.ptr(), system_id.ptr(), n_systems(), n_particles());
    }

    void sync_neighbors() {
        geo::bind_neighbor_globals(neighbor_start.ptr(), neighbor_ids.ptr(), verlet_skin.ptr());
    }

    void sync_cells() {
        geo::bind_cell_globals(cell_size.xptr(), cell_size.yptr(), cell_inv.xptr(), cell_inv.yptr(), cell_dim.xptr(), cell_dim.yptr(), cell_system_start.ptr());
    }

    void sync_class_constants() {derived().sync_class_constants_impl();}

    // ---- Reordering (base + extensible hook) ----
    // d_perm maps new[i] <- old[d_perm[i]]  (length=N)
    void reorder_by(const thrust::device_vector<int>& d_perm) {
        derived().reorder_extra_impl(thrust::raw_pointer_cast(d_perm.data())); // subclass hook
    }

    // ---- RNG seeding across base fields; subclass can extend ----
    void seed_rng(unsigned long long seed) {
        // TODO: rename this to enable_rng
        // TODO: add a stateless and stateful version
        pos.enable_rng(seed);
        vel.enable_rng(seed+101ULL);
        force.enable_rng(seed+202ULL);
        rad.enable_rng(seed+303ULL);
        mass.enable_rng(seed+404ULL);
        derived().seed_rng_extra_impl(seed);
    }

    // ---- Enable/disable swap ----
    void enable_swap(bool enable) {
        pos.enable_swap();
        vel.enable_swap();
        force.enable_swap();
        rad.enable_swap();
        mass.enable_swap();
        area.enable_swap();
        cell_id.enable_swap();
        derived().enable_swap_impl(enable);
    }
    // Entry points (CRTP)
    void compute_forces()    { derived().compute_forces_impl(); }
    void compute_wall_forces() { derived().compute_wall_forces_impl(); }
    void update_positions(double scale) { derived().update_positions_impl(scale); }
    void update_velocities(double dt) { derived().update_velocities_impl(dt); }

    // ---- Neighbor Methods ----
    void set_neighbor_method(NeighborMethod method) {
        neighbor_method = method;
    }

    void init_naive_neighbors() { derived().init_naive_neighbors_impl(); }
    void update_naive_neighbors() { derived().update_naive_neighbors_impl(); }

    void init_cell_neighbors() { derived().init_cell_neighbors_impl(); sync_cells(); }
    void update_cell_neighbors() { derived().update_cell_neighbors_impl(); sync_cells(); }
    void check_cell_neighbors() { derived().check_cell_neighbors_impl(); }

    void init_neighbors() {
        switch (neighbor_method) {
            case NeighborMethod::Naive: init_naive_neighbors(); break;
            case NeighborMethod::Cell:  init_cell_neighbors();  break;
        }
        update_neighbors();
    }

    void update_neighbors() {
        switch (neighbor_method) {
            case NeighborMethod::Naive: update_naive_neighbors(); break;
            case NeighborMethod::Cell:  update_cell_neighbors();  break;
        }
        sync_neighbors();
    }

    void check_neighbors() {
        switch (neighbor_method) {
            case NeighborMethod::Naive: break;  // do nothing
            case NeighborMethod::Cell:  check_cell_neighbors();   break;
        }
    }

    // Counts
    int n_particles() const { return pos.size(); }
    int n_systems()  const { return box_size.size(); }

protected:
    Derived&       derived()       { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    // Default no-op hooks
    void reorder_extra_impl(const int*) {}
    void seed_rng_extra_impl(unsigned long long, unsigned long long) {}
    void enable_swap_impl(bool) {}
    void init_naive_neighbors_impl() {}
    void update_naive_neighbors_impl() {}
    void init_cell_neighbors_impl() {}
    void update_cell_neighbors_impl() {}
    void check_cell_neighbors_impl() {}
    void compute_forces_impl() {}
    void compute_wall_forces_impl() {}
    void sync_class_constants_impl() {}
    void update_positions_impl(double) {}
    void update_velocities_impl(double) {}
};

} // namespace md