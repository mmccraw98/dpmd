#pragma once
#include "particles/base_particle.hpp"
#include "utils/device_fields.cuh"
#include "integrators/fire.cuh"
#include "routines/minimizers.cuh"


#include <iostream>

namespace md::routines {

namespace kernels {

// Remove systems that did not converge (set action to 0)
__global__ void remove_badly_initialized_systems_kernel(
    int* __restrict__ action,
    double* __restrict__ dt
);

// Reset the dt for the systems that have not jammed yet (action != 0)
__global__ void reinit_dt_kernel(
    double* __restrict__ dt,
    const int* __restrict__ action,
    double* __restrict__ dt_copy
);

// Update the binary search parameters
__global__ void jamming_update_kernel(
    const double* __restrict__ pe_total,
    double* __restrict__ phi,
    double* __restrict__ phi_low,
    double* __restrict__ phi_high,
    int* __restrict__ action,
    double phi_increment,
    double phi_tolerance,
    double avg_pe_target
);

// Scale the box size
__global__ void scale_box_kernel(
    double* __restrict__ box_size_x,
    double* __restrict__ box_size_y,
    const double* __restrict__ area_total,
    const double* __restrict__ packing_fraction_target,
    int* __restrict__ action,
    double* __restrict__ scale_factor
);

struct functor_abs_val {
    __host__ __device__
    double operator()(const double& x) const {
        return x < 0 ? -x : x;
    }
};

}  // namespace kernels

// Jam particles using binary search and FIRE minimization with PBC
template <class ParticleT>
void jam_binary_search_pbc(
    ParticleT& P,
    df::DeviceField1D<double> dt,
    int max_compression_steps,
    int max_minimization_steps,
    double avg_pe_target,
    double avg_pe_diff_target,
    double phi_increment,
    double phi_tolerance
) {
    // define the launch configurations
    const int S = P.n_systems();
    const int N = P.n_particles();
    auto B = md::launch::threads_for();
    auto G_S = md::launch::blocks_for(S);
    auto G_N = md::launch::blocks_for(N);

    df::DeviceField1D<double> dt_copy; dt_copy.resize(S); dt_copy.copy_from(dt);
    df::DeviceField1D<int> action; action.resize(S); action.fill(-1);  // -2 default, -1 unjammed (save state), 0 converged, 1 jammed (revert state)
    df::DeviceField1D<double> scale_factor; scale_factor.resize(S); scale_factor.fill(0.0);
    df::DeviceField1D<double> phi, phi_low, phi_high; phi.resize(S); phi_low.resize(S); phi_high.resize(S);

    // run initial minimization
    minimize_fire_copy_dt(P, dt, max_minimization_steps, avg_pe_target, avg_pe_diff_target);

    // remove systems that did not converge (set action to 0)
    CUDA_LAUNCH(kernels::remove_badly_initialized_systems_kernel, G_S, B, action.ptr(), dt.ptr());

    // save the state for all systems with action != 0
    P.save_state(action, -1);

    // set the initial packing fraction
    P.compute_packing_fraction();
    phi.copy_from(P.packing_fraction);
    phi_low.copy_from(P.packing_fraction);
    phi_high.fill(-1.0);

    int compression_step = 0;

    while (compression_step < max_compression_steps) {

        // reset the dt for the systems that are still running
        CUDA_LAUNCH(kernels::reinit_dt_kernel, G_S, B, dt.ptr(), action.ptr(), dt_copy.ptr());

        // run the minimization
        minimize_fire_copy_dt(P, dt, max_minimization_steps, avg_pe_target, avg_pe_diff_target);

        // call the jamming kernel
        CUDA_LAUNCH(kernels::jamming_update_kernel, G_S, B,
            P.pe_total.ptr(), phi.ptr(), phi_low.ptr(), phi_high.ptr(), action.ptr(), phi_increment, phi_tolerance, avg_pe_target
        );

        // save the state for all systems with action = -1
        P.save_state(action, -1);

        // revert the state for all systems with action = 1
        P.restore_state(action, 1);

        // scale the box size
        df::DeviceField1D<double> area_total = P.compute_particle_area_total();
        CUDA_LAUNCH(kernels::scale_box_kernel, G_S, B,
            P.box_size.xptr(), P.box_size.yptr(), area_total.ptr(), phi.ptr(), action.ptr(), scale_factor.ptr()
        );

        // sync the box, system, neighbors, cells, and class constants - TODO: find a better way to do this?
        if (P.neighbor_method == NeighborMethod::Cell) {
            P.update_cell_size();
        }
        P.sync_box();
        P.sync_system();
        P.sync_neighbors();
        P.sync_cells();
        P.sync_class_constants();
        P.check_neighbors(true);
        P.compute_packing_fraction();

        // scale the positions
        P.scale_positions(scale_factor);

        // check if the sum of abs(action) is 0, we are done
        double sum_abs_action = thrust::transform_reduce(thrust::device, action.ptr(), action.ptr() + S, kernels::functor_abs_val(), 0.0, thrust::plus<double>());
        double avg_phi = thrust::reduce(thrust::device, phi.ptr(), phi.ptr() + S, 0.0, thrust::plus<double>()) / S;
        double avg_action = thrust::reduce(thrust::device, action.ptr(), action.ptr() + S, 0.0, thrust::plus<double>()) / S;
        double avg_pe = thrust::reduce(thrust::device, P.pe_total.ptr(), P.pe_total.ptr() + S, 0.0, thrust::plus<double>()) / S;
        std::cout << "Avg abs(action): " << sum_abs_action / S << " Avg phi: " << avg_phi << " Avg action: " << avg_action << " Avg pe: " << avg_pe << std::endl;
        if (sum_abs_action == 0) {
            break;
        }
        compression_step++;
    }
    P.compute_forces();
    P.compute_pe_total();
    P.compute_packing_fraction();
}


// Jam particles using binary search and FIRE minimization with closed boundary conditions
template <class ParticleT>
void jam_binary_search_wall(
    ParticleT& P,
    df::DeviceField1D<double> dt,
    int max_compression_steps,
    int max_minimization_steps,
    double avg_pe_target,
    double avg_pe_diff_target,
    double phi_increment,
    double phi_tolerance
) {
    // define the launch configurations
    const int S = P.n_systems();
    const int N = P.n_particles();
    auto B = md::launch::threads_for();
    auto G_S = md::launch::blocks_for(S);
    auto G_N = md::launch::blocks_for(N);

    df::DeviceField1D<double> dt_copy; dt_copy.resize(S); dt_copy.copy_from(dt);
    df::DeviceField1D<int> action; action.resize(S); action.fill(-1);  // -2 default, -1 unjammed (save state), 0 converged, 1 jammed (revert state)
    df::DeviceField1D<double> scale_factor; scale_factor.resize(S); scale_factor.fill(0.0);
    df::DeviceField1D<double> phi, phi_low, phi_high; phi.resize(S); phi_low.resize(S); phi_high.resize(S);

    // run initial minimization
    minimize_fire_wall_copy_dt(P, dt, max_minimization_steps, avg_pe_target, avg_pe_diff_target);

    // remove systems that did not converge (set action to 0)
    CUDA_LAUNCH(kernels::remove_badly_initialized_systems_kernel, G_S, B, action.ptr(), dt.ptr());

    // save the state for all systems with action != 0
    P.save_state(action, -1);

    // set the initial packing fraction
    P.compute_packing_fraction();
    phi.copy_from(P.packing_fraction);
    phi_low.copy_from(P.packing_fraction);
    phi_high.fill(-1.0);

    int compression_step = 0;

    while (compression_step < max_compression_steps) {

        // reset the dt for the systems that are still running
        CUDA_LAUNCH(kernels::reinit_dt_kernel, G_S, B, dt.ptr(), action.ptr(), dt_copy.ptr());

        // run the minimization
        minimize_fire_wall_copy_dt(P, dt, max_minimization_steps, avg_pe_target, avg_pe_diff_target);

        // call the jamming kernel
        CUDA_LAUNCH(kernels::jamming_update_kernel, G_S, B,
            P.pe_total.ptr(), phi.ptr(), phi_low.ptr(), phi_high.ptr(), action.ptr(), phi_increment, phi_tolerance, avg_pe_target
        );

        // save the state for all systems with action = -1
        P.save_state(action, -1);

        // revert the state for all systems with action = 1
        P.restore_state(action, 1);

        // scale the box size
        df::DeviceField1D<double> area_total = P.compute_particle_area_total();
        CUDA_LAUNCH(kernels::scale_box_kernel, G_S, B,
            P.box_size.xptr(), P.box_size.yptr(), area_total.ptr(), phi.ptr(), action.ptr(), scale_factor.ptr()
        );

        // sync the box, system, neighbors, cells, and class constants - TODO: find a better way to do this?
        if (P.neighbor_method == NeighborMethod::Cell) {
            P.update_cell_size();
        }
        P.sync_box();
        P.sync_system();
        P.sync_neighbors();
        P.sync_cells();
        P.sync_class_constants();
        // P.update_neighbors();
        P.check_neighbors(true);
        P.compute_packing_fraction();

        // scale the positions
        P.scale_positions(scale_factor);

        // check if the sum of abs(action) is 0, we are done
        double sum_abs_action = thrust::transform_reduce(thrust::device, action.ptr(), action.ptr() + S, kernels::functor_abs_val(), 0.0, thrust::plus<double>());
        double avg_phi = thrust::reduce(thrust::device, phi.ptr(), phi.ptr() + S, 0.0, thrust::plus<double>()) / S;
        double avg_action = thrust::reduce(thrust::device, action.ptr(), action.ptr() + S, 0.0, thrust::plus<double>()) / S;
        double avg_pe = thrust::reduce(thrust::device, P.pe_total.ptr(), P.pe_total.ptr() + S, 0.0, thrust::plus<double>()) / S;
        std::cout << "Avg abs(action): " << sum_abs_action / S << " Avg phi: " << avg_phi << " Avg action: " << avg_action << " Avg pe: " << avg_pe << std::endl;
        if (sum_abs_action == 0) {
            break;
        }
        compression_step++;
    }
    P.compute_forces();
    P.compute_pe_total();
    P.compute_packing_fraction();
}

}  // namespace md::routines