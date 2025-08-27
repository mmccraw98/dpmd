#pragma once

#include "integrators/base_integrator.hpp"
#include "utils/device_fields.cuh"

namespace md::integrators {

namespace kernels {


// Handle the logic for the FIRE update
__global__ void fire_update_kernel(
    double* dt_reverse,
    double* velocity_scale,
    int* N_good,
    int* N_bad,
    double* power,
    double* dt,
    double* alpha,
    double* dt_max,
    double* dt_min,
    double f_inc,
    double f_dec,
    double f_alpha,
    double alpha_init,
    int N_min,
    int N_bad_max
);

// Check if the system has converged
__global__ void fire_convergence_kernel(
    double* __restrict__ pe_total,
    double* __restrict__ pe_total_prev,
    double* __restrict__ dt,
    double avg_pe_diff_target,
    double avg_pe_target
);


} // namespace kernels

// Base class for all FIRE integrators
template <class Derived, class ParticleT>
class BaseFIRE : public BaseIntegrator<Derived, ParticleT> {
    using Base = BaseIntegrator<Derived, ParticleT>;
public:
    using Base::Base;  // inherit Base(p)

    BaseFIRE(ParticleT& p, double dt_init) : Base(p) {
        dt.resize(p.n_systems());
        dt.fill(dt_init);
        define_arrays();
    }
    BaseFIRE(ParticleT& p, df::DeviceField1D<double> dt_init) : Base(p), dt(dt_init) {
        define_arrays();
    }

    // The actual VV step; Base::step() will call this
    inline void step_impl() {
        auto& P = this->particle();

        // calculate power for each system
        P.compute_fpower_total();

        // update the timestep and velocity scale for each system
        const int S = P.n_systems();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(S);
        CUDA_LAUNCH(
            kernels::fire_update_kernel, G, B,
            dt_reverse.ptr(), velocity_scale.ptr(), N_good.ptr(), N_bad.ptr(), P.fpower_total.ptr(), dt.ptr(), alpha.ptr(),
            dt_max.ptr(), dt_min.ptr(), f_inc, f_dec, f_alpha, alpha_init, N_min, N_bad_max
        );
        // reverse the positions and scale the velocities for the systems that are moving uphill
        P.update_positions(dt_reverse, 1.0);  // does nothing unless dt_reverse[s] != 0.0
        P.scale_velocities(velocity_scale);  // does nothing unless velocity_scale[s] != 1.0
        
        // proceed with velocity verlet with velocity mixing
        P.update_velocities(dt, 0.5);
        P.mix_velocities_and_forces(alpha);
        P.scale_velocities(velocity_scale);  // does nothing unless velocity_scale[s] != 1.0
        P.update_positions(dt, 0.5);

        P.check_neighbors();
        this->derived().compute_particle_forces_impl();
        
        P.update_velocities(dt, 0.5);
    }

    inline void init_impl() {
        // start with the velocities set to zero and the forces calculated
        velocity_scale.fill(0.0);
        this->particle().scale_velocities(velocity_scale);
        velocity_scale.fill(1.0);
        this->derived().compute_particle_forces_impl();
    }

    bool converged(double avg_pe_diff_target, double avg_pe_target) {
        auto& P = this->particle();
        // check convergence
        P.compute_pe_total();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(P.n_systems());
        CUDA_LAUNCH(kernels::fire_convergence_kernel, G, B,
            P.pe_total.ptr(), pe_total_prev.ptr(), dt.ptr(), avg_pe_diff_target, avg_pe_target
        );
        // if the sum of dt is 0, all systems are converged
        return thrust::reduce(thrust::device, dt.ptr(), dt.ptr() + P.n_systems(), 0.0) == 0.0;
    }

    df::DeviceField1D<double> get_dt() {
        return dt;
    }

protected:
    double alpha_init = 0.1;       // initial mixing factor (alpha)
    double f_inc = 1.1;            // dt increase factor
    double f_dec = 0.5;            // dt decrease factor
    double f_alpha = 0.99;         // mixing factor (alpha) decrease factor
    long N_min = 5;                // minimum number of steps
    long N_bad_max = 10;           // maximum number of bad steps
    double dt_max_scale = 10.0;    // maximum dt scale
    double dt_min_scale = 0.001;   // minimum dt scale

    df::DeviceField1D<double> dt;             // (S,) time-step for each system
    df::DeviceField1D<double> dt_half;        // (S,) half time-step for velocity verlet update
    df::DeviceField1D<double> dt_max;         // (S,) maximum dt for each system
    df::DeviceField1D<double> dt_min;         // (S,) minimum dt for each system
    df::DeviceField1D<double> dt_reverse;     // (S,) dt used to reverse the dynamics - typically 0, meaning no reversal
    df::DeviceField1D<double> velocity_scale; // (S,) velocity scale for each system - typically 1, meaning no scaling
    df::DeviceField1D<double> alpha;          // (S,) mixing factor for each system
    df::DeviceField1D<int> N_good;            // (S,) number of good steps for each system
    df::DeviceField1D<int> N_bad;             // (S,) number of bad steps for each system
    df::DeviceField1D<double> pe_total_prev;  // (S,) previous potential energy for each system

    inline void define_arrays() {
        const int S = this->particle().n_systems();
        dt_half.resize(S); dt_half.copy_from(dt); dt_half.scale(0.5);
        dt_max.resize(dt.size()); dt_max.copy_from(dt); dt_max.scale(dt_max_scale);
        dt_min.resize(dt.size()); dt_min.copy_from(dt); dt_min.scale(dt_min_scale);
        dt_reverse.resize(S); dt_reverse.fill(0.0);
        velocity_scale.resize(S); velocity_scale.fill(1.0);
        alpha.resize(S); alpha.fill(alpha_init);
        N_good.resize(S); N_good.fill(0);
        N_bad.resize(S); N_bad.fill(0);
        pe_total_prev.resize(S); pe_total_prev.fill(1e9);
    }
};


// FIRE integrator with periodic boundaries - forces must be computed before integration
template <class ParticleT>
class FIRE final
: public BaseFIRE<FIRE<ParticleT>, ParticleT> {
    using Base = BaseFIRE<FIRE<ParticleT>, ParticleT>;
public:
    FIRE(ParticleT& p, double dt) : Base(p, dt) {}
    FIRE(ParticleT& p, const df::DeviceField1D<double>& dt) : Base(p, dt) {}

    inline void compute_particle_forces_impl() {
        this->particle().compute_forces();
    }
};

// FIRE with closed boundaries - forces must be computed before integration
template <class ParticleT>
class FIREWall final
: public BaseFIRE<FIREWall<ParticleT>, ParticleT> {
    using Base = BaseFIRE<FIREWall<ParticleT>, ParticleT>;
public:
    FIREWall(ParticleT& p, double dt) : Base(p, dt) {}
    FIREWall(ParticleT& p, const df::DeviceField1D<double>& dt) : Base(p, dt) {}

    inline void compute_particle_forces_impl() {
        this->particle().compute_wall_forces();
    }
};

} // namespace md::integrators