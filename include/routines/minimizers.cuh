#pragma once
#include "particles/base_particle.hpp"
#include "utils/device_fields.cuh"
#include "integrators/fire.cuh"

namespace md::routines {

// Minimize potential energy using FIRE in PBC
template <class ParticleT>
void minimize_fire(
    ParticleT& P,
    df::DeviceField1D<double>& dt,
    int max_steps,
    double avg_pe_target,
    double avg_pe_diff_target
) {
    md::integrators::FIRE<ParticleT> fire(P, dt);
    fire.init();

    bool converged = false;

    int step = 0;
    while (step < max_steps && !converged) {
        fire.step();
        converged = fire.converged(avg_pe_target, avg_pe_diff_target);  // check if converged
        step++;
    }
    if (!converged) {
        std::cout << "Did not converge in " << max_steps << " steps" << std::endl;
    }
}

// Minimize potential energy using FIRE in PBC, overwriting the dt with the final dt
template <class ParticleT>
void minimize_fire_copy_dt(
    ParticleT& P,
    df::DeviceField1D<double>& dt,
    int max_steps,
    double avg_pe_target,
    double avg_pe_diff_target
) {
    md::integrators::FIRE<ParticleT> fire(P, dt);
    fire.init();

    bool converged = false;

    int step = 0;
    while (step < max_steps && !converged) {
        fire.step();
        converged = fire.converged(avg_pe_target, avg_pe_diff_target);  // check if converged
        step++;
    }
    
    // overwrite the dt with the final dt
    dt = fire.get_dt();
    
    if (!converged) {
        std::cout << "Did not converge in " << max_steps << " steps" << std::endl;
    }

    // set the velocities to zero to avoid after-effects
    P.scale_velocities(0.0);
}


// Minimize potential energy using FIRE in closed boundary conditions
template <class ParticleT>
void minimize_fire_wall(
    ParticleT& P,
    df::DeviceField1D<double>& dt,
    int max_steps,
    double avg_pe_target,
    double avg_pe_diff_target
) {
    md::integrators::FIREWall<ParticleT> fire(P, dt);
    fire.init();

    bool converged = false;

    int step = 0;
    while (step < max_steps && !converged) {
        fire.step();
        converged = fire.converged(avg_pe_target, avg_pe_diff_target);  // check if converged
        step++;
    }
    if (!converged) {
        std::cout << "Did not converge in " << max_steps << " steps" << std::endl;
    }
}

// Minimize potential energy using FIRE in closed boundary conditions, overwriting the dt with the final dt
template <class ParticleT>
void minimize_fire_wall_copy_dt(
    ParticleT& P,
    df::DeviceField1D<double>& dt,
    int max_steps,
    double avg_pe_target,
    double avg_pe_diff_target
) {
    md::integrators::FIREWall<ParticleT> fire(P, dt);
    fire.init();

    bool converged = false;

    int step = 0;
    while (step < max_steps && !converged) {
        fire.step();
        converged = fire.converged(avg_pe_target, avg_pe_diff_target);  // check if converged
        step++;
    }
    
    // overwrite the dt with the final dt
    dt = fire.get_dt();
    
    if (!converged) {
        std::cout << "Did not converge in " << max_steps << " steps" << std::endl;
    }

    // set the velocities to zero to avoid after-effects
    P.scale_velocities(0.0);
}


}  // namespace md::routines