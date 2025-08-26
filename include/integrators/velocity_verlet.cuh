#pragma once

#include "integrators/base_integrator.hpp"
#include "utils/device_fields.cuh"


namespace md { namespace integrators {

// Base class for all velocity verlet integrators
template <class Derived, class ParticleT>
class BaseVelocityVerlet : public BaseIntegrator<Derived, ParticleT> {
    using Base = BaseIntegrator<Derived, ParticleT>;
public:
    using Base::Base;  // inherit Base(p)

    BaseVelocityVerlet(ParticleT& p, double dt) : Base(p) {
        dt_.resize(p.n_systems());     dt_.fill(dt);
        dt_half_.resize(p.n_systems()); dt_half_.fill(0.5 * dt);
    }
    BaseVelocityVerlet(ParticleT& p, df::DeviceField1D<double> dt) : Base(p) {
        dt_ = dt;
        dt_half_.resize(p.n_systems());
        dt_half_.copy_from(dt_);
        dt_half_.scale(0.5);
    }

    // The actual VV step; Base::step() will call this
    inline void step_impl() {
        auto& P = this->particle();
        P.update_velocities(dt_half_);
        P.update_positions(dt_);
        P.check_neighbors();
        this->derived().compute_particle_forces_impl();
        P.update_velocities(dt_half_);
    }

    inline void init_impl() {
        this->derived().compute_particle_forces_impl();
    }

protected:
    df::DeviceField1D<double> dt_, dt_half_;
};


// Velocity Verlet with periodic boundaries - forces must be computed before integration
template <class ParticleT>
class VelocityVerlet final
: public BaseVelocityVerlet<VelocityVerlet<ParticleT>, ParticleT> {
    using Base = BaseVelocityVerlet<VelocityVerlet<ParticleT>, ParticleT>;
public:
    VelocityVerlet(ParticleT& p, double dt) : Base(p, dt) {}
    VelocityVerlet(ParticleT& p, const df::DeviceField1D<double>& dt) : Base(p, dt) {}

    inline void compute_particle_forces_impl() {
        this->particle().compute_forces();
    }
};

// Velocity Verlet with closed boundaries - forces must be computed before integration
template <class ParticleT>
class VelocityVerletWall final
: public BaseVelocityVerlet<VelocityVerletWall<ParticleT>, ParticleT> {
    using Base = BaseVelocityVerlet<VelocityVerletWall<ParticleT>, ParticleT>;
public:
    VelocityVerletWall(ParticleT& p, double dt) : Base(p, dt) {}
    VelocityVerletWall(ParticleT& p, const df::DeviceField1D<double>& dt) : Base(p, dt) {}

    inline void compute_particle_forces_impl() {
        this->particle().compute_wall_forces();
    }
};

} // namespace integrators
} // namespace md