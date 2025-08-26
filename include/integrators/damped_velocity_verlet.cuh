#pragma once

#include "integrators/base_integrator.hpp"
#include "utils/device_fields.cuh"


namespace md { namespace integrators {

// Base class for all velocity verlet integrators
template <class Derived, class ParticleT>
class BaseDampedVelocityVerlet : public BaseIntegrator<Derived, ParticleT> {
    using Base = BaseIntegrator<Derived, ParticleT>;
public:
    using Base::Base;  // inherit Base(p)

    BaseDampedVelocityVerlet(ParticleT& p, double dt, double damping) : Base(p) {
        dt_.resize(p.n_systems());     dt_.fill(dt);
        dt_half_.resize(p.n_systems()); dt_half_.fill(0.5 * dt);
        damping_.resize(p.n_systems()); damping_.fill(damping);
    }
    BaseDampedVelocityVerlet(ParticleT& p, df::DeviceField1D<double> dt, df::DeviceField1D<double> damping) : Base(p) {
        dt_ = dt;
        dt_half_.resize(p.n_systems());
        dt_half_.copy_from(dt_);
        dt_half_.scale(0.5);
        damping_ = damping;
    }

    // The actual VV step; Base::step() will call this
    inline void step_impl() {
        auto& P = this->particle();
        P.update_velocities(dt_half_);
        P.update_positions(dt_);
        P.check_neighbors();
        this->derived().compute_particle_forces_impl();
        P.compute_damping_forces(damping_);
        P.update_velocities(dt_half_);
    }

    inline void init_impl() {
        this->derived().compute_particle_forces_impl();
    }

protected:
    df::DeviceField1D<double> dt_, dt_half_;
    df::DeviceField1D<double> damping_;
};


// Damped Velocity Verlet with periodic boundaries - forces must be computed before integration
template <class ParticleT>
class DampedVelocityVerlet final
: public BaseDampedVelocityVerlet<DampedVelocityVerlet<ParticleT>, ParticleT> {
    using Base = BaseDampedVelocityVerlet<DampedVelocityVerlet<ParticleT>, ParticleT>;
public:
    DampedVelocityVerlet(ParticleT& p, double dt, double damping) : Base(p, dt, damping) {}
    DampedVelocityVerlet(ParticleT& p, const df::DeviceField1D<double>& dt, const df::DeviceField1D<double>& damping) : Base(p, dt, damping) {}

    inline void compute_particle_forces_impl() {
        this->particle().compute_forces();
    }
};

// Damped Velocity Verlet with closed boundaries - forces must be computed before integration
template <class ParticleT>
class DampedVelocityVerletWall final
: public BaseDampedVelocityVerlet<DampedVelocityVerletWall<ParticleT>, ParticleT> {
    using Base = BaseDampedVelocityVerlet<DampedVelocityVerletWall<ParticleT>, ParticleT>;
public:
    DampedVelocityVerletWall(ParticleT& p, double dt, double damping) : Base(p, dt, damping) {}
    DampedVelocityVerletWall(ParticleT& p, const df::DeviceField1D<double>& dt, const df::DeviceField1D<double>& damping) : Base(p, dt, damping) {}

    inline void compute_particle_forces_impl() {
        this->particle().compute_wall_forces();
    }
};

} // namespace integrators
} // namespace md