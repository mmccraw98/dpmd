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

    BaseVelocityVerlet(ParticleT& p, double dt_init) : Base(p) {
        dt.resize(p.n_systems());
        dt.fill(dt_init);
        dt_half.resize(p.n_systems());
        dt_half.fill(0.5 * dt_init);
    }
    BaseVelocityVerlet(ParticleT& p, df::DeviceField1D<double> dt_init) : Base(p) {
        dt.resize(p.n_systems());
        dt_half.resize(p.n_systems());
        dt.copy_from(dt_init);
        dt_half.copy_from(dt_init);
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
    df::DeviceField1D<double> dt, dt_half;
};


// Velocity Verlet with periodic boundaries - forces must be computed before integration
template <class ParticleT>
class VelocityVerlet final
: public BaseVelocityVerlet<VelocityVerlet<ParticleT>, ParticleT> {
    using Base = BaseVelocityVerlet<VelocityVerlet<ParticleT>, ParticleT>;
public:
    VelocityVerlet(ParticleT& p, double dt_init) : Base(p, dt_init) {}
    VelocityVerlet(ParticleT& p, const df::DeviceField1D<double>& dt_init) : Base(p, dt_init) {}

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
    VelocityVerletWall(ParticleT& p, double dt_init) : Base(p, dt_init) {}
    VelocityVerletWall(ParticleT& p, const df::DeviceField1D<double>& dt_init) : Base(p, dt_init) {}

    inline void compute_particle_forces_impl() {
        this->particle().compute_wall_forces();
    }
};

} // namespace integrators
} // namespace md