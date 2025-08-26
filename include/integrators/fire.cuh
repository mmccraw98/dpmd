#pragma once

#include "integrators/base_integrator.hpp"
#include "utils/device_fields.cuh"

namespace k {

} // namespace k

namespace md { namespace integrators {

// Base class for all FIRE integrators
template <class Derived, class ParticleT>
class BaseFIRE : public BaseIntegrator<Derived, ParticleT> {
    using Base = BaseIntegrator<Derived, ParticleT>;
public:
    using Base::Base;  // inherit Base(p)

    BaseFIRE(ParticleT& p, double dt) : Base(p) {
        dt.resize(p.n_systems());     dt.fill(dt);
        dt_half.resize(p.n_systems()); dt_half.fill(0.5 * dt);
    }
    BaseFIRE(ParticleT& p, df::DeviceField1D<double> dt) : Base(p) {
        dt.resize(p.n_systems());
        dt_half.resize(p.n_systems());
        dt_half.copy_from(dt);
        dt_half.scale(0.5);
    }

    // The actual VV step; Base::step() will call this
    inline void step_impl() {
        auto& P = this->particle();
        P.update_velocities(dt_half);
        P.update_positions(dt);
        P.check_neighbors();
        this->derived().compute_particle_forces_impl();
        P.update_velocities(dt_half);
    }

    inline void init_impl() {
        this->derived().compute_particle_forces_impl();
    }

protected:
    df::DeviceField1D<double> dt, dt_half, dt_max, dt_min, dt_reverse;

    inline void define_arrays() {
        const int S = this->particle().n_systems();
        dt_max.resize(dt.size()); dt_max.copy_from(dt); dt_max.scale(dt_max_scale);
        dt_min.resize(dt.size()); dt_min.copy_from(dt); dt_min.scale(dt_min_scale);
        dt_reverse.resize(S); dt_reverse.fill(0.0);
        velocity_scale.resize(S); velocity_scale.fill(velocity_scale_init);
        alpha.resize(S); alpha.fill(alpha_init);
        N_good.resize(S); N_good.fill(0);
        df::DeviceField1D<int> N_bad; N_bad.resize(S); N_bad.fill(0);
        df::DeviceField1D<double> pe_total_prev; pe_total_prev.resize(S); pe_total_prev.fill(1e9);
    }
};


// // Velocity Verlet with periodic boundaries - forces must be computed before integration
// template <class ParticleT>
// class VelocityVerlet final
// : public BaseVelocityVerlet<VelocityVerlet<ParticleT>, ParticleT> {
//     using Base = BaseVelocityVerlet<VelocityVerlet<ParticleT>, ParticleT>;
// public:
//     VelocityVerlet(ParticleT& p, double dt) : Base(p, dt) {}
//     VelocityVerlet(ParticleT& p, const df::DeviceField1D<double>& dt) : Base(p, dt) {}

//     inline void compute_particle_forces_impl() {
//         this->particle().compute_forces();
//     }
// };

// // Velocity Verlet with closed boundaries - forces must be computed before integration
// template <class ParticleT>
// class VelocityVerletWall final
// : public BaseVelocityVerlet<VelocityVerletWall<ParticleT>, ParticleT> {
//     using Base = BaseVelocityVerlet<VelocityVerletWall<ParticleT>, ParticleT>;
// public:
//     VelocityVerletWall(ParticleT& p, double dt) : Base(p, dt) {}
//     VelocityVerletWall(ParticleT& p, const df::DeviceField1D<double>& dt) : Base(p, dt) {}

//     inline void compute_particle_forces_impl() {
//         this->particle().compute_wall_forces();
//     }
// };

} // namespace integrators
} // namespace md