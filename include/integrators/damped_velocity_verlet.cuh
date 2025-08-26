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

    BaseDampedVelocityVerlet(ParticleT& p, double dt_init, double damping_init) : Base(p) {
        dt.resize(p.n_systems());
        dt.fill(dt_init);
        dt_half.resize(p.n_systems());
        dt_half.fill(0.5 * dt_init);
        damping.resize(p.n_systems());
        damping.fill(damping_init);
    }
    BaseDampedVelocityVerlet(ParticleT& p, df::DeviceField1D<double> dt_init, df::DeviceField1D<double> damping_init) : Base(p) {
        dt.resize(p.n_systems());
        dt_half.resize(p.n_systems());
        dt.copy_from(dt_init);
        dt_half.copy_from(dt_init);
        dt_half.scale(0.5);
        damping.resize(p.n_systems());
        damping.copy_from(damping_init);
    }

    // The actual VV step; Base::step() will call this
    inline void step_impl() {
        auto& P = this->particle();
        P.update_velocities(dt_half);
        P.update_positions(dt);
        P.check_neighbors();
        this->derived().compute_particle_forces_impl();
        P.compute_damping_forces(damping);
        P.update_velocities(dt_half);
    }

    inline void init_impl() {
        this->derived().compute_particle_forces_impl();
    }

protected:
    df::DeviceField1D<double> dt, dt_half;
    df::DeviceField1D<double> damping;
};


// Damped Velocity Verlet with periodic boundaries - forces must be computed before integration
template <class ParticleT>
class DampedVelocityVerlet final
: public BaseDampedVelocityVerlet<DampedVelocityVerlet<ParticleT>, ParticleT> {
    using Base = BaseDampedVelocityVerlet<DampedVelocityVerlet<ParticleT>, ParticleT>;
public:
    DampedVelocityVerlet(ParticleT& p, double dt_init, double damping_init) : Base(p, dt_init, damping_init) {}
    DampedVelocityVerlet(ParticleT& p, const df::DeviceField1D<double>& dt_init, const df::DeviceField1D<double>& damping_init) : Base(p, dt_init, damping_init) {}

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
    DampedVelocityVerletWall(ParticleT& p, double dt_init, double damping_init) : Base(p, dt_init, damping_init) {}
    DampedVelocityVerletWall(ParticleT& p, const df::DeviceField1D<double>& dt_init, const df::DeviceField1D<double>& damping_init) : Base(p, dt_init, damping_init) {}

    inline void compute_particle_forces_impl() {
        this->particle().compute_wall_forces();
    }
};

} // namespace integrators
} // namespace md