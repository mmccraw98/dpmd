#include "routines/jammers.cuh"

namespace md::routines {

namespace kernels {

__global__ void remove_badly_initialized_systems_kernel(
    int* __restrict__ action,
    double* __restrict__ dt
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (s >= S) return;
    if (dt[s] != 0.0) {
        action[s] = 0;
    }
}

__global__ void reinit_dt_kernel(
    double* __restrict__ dt,
    const int* __restrict__ action,
    double* __restrict__ dt_copy
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (s >= S) return;
    dt[s] = (action[s] != 0) ? dt_copy[s] : 0.0;
}

__global__ void jamming_update_kernel(
    const double* __restrict__ pe_total,
    double* __restrict__ phi,
    double* __restrict__ phi_low,
    double* __restrict__ phi_high,
    int* __restrict__ action,
    double phi_increment,
    double phi_tolerance,
    double avg_pe_target
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (s >= S) return;

    if (action[s] == 0) {  // system is converged, do nothing
        return;
    }

    // default flag
    action[s] = -2;

    int dof = (md::geo::g_sys.offset[s+1] - md::geo::g_sys.offset[s]);  // TODO: this should likely be a class-level variable - tracking degrees of freedom in each system!
    double avg_pe = pe_total[s] / dof;
    if (avg_pe > avg_pe_target) {  // jammed
        phi_high[s] = phi[s];
        phi[s] = (phi_high[s] + phi_low[s]) / 2.0;
        action[s] = 1;  // jammed state is found, revert to last unjammed state
    } else {  // unjammed
        action[s] = -1;  // unjammed state is found, set current state as last unjammed state
        phi_low[s] = phi[s];
        if (phi_high[s] > 0) {
            phi[s] = (phi_high[s] + phi_low[s]) / 2.0;
        } else {
            phi[s] += phi_increment;
        }
    }
    if (std::abs(phi_high[s] / phi_low[s] - 1) < phi_tolerance && phi_high[s] > 0) {  // converged
        action[s] = 0;  // final state is found, do nothing
    }
}

__global__ void scale_box_kernel(
    double* __restrict__ box_size_x,
    double* __restrict__ box_size_y,
    const double* __restrict__ area_total,
    const double* __restrict__ packing_fraction_target,
    int* __restrict__ action,
    double* __restrict__ scale_factor
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (s >= S) return;

    if (action[s] == 0) { scale_factor[s] = 1.0; return; }
    if (packing_fraction_target[s] <= 0.0) { scale_factor[s] = 1.0; return; }

    double new_box_size = sqrt(area_total[s] / packing_fraction_target[s]);
    // Use current array value, not md::geo::g_box (may be stale until P.sync_box)
    scale_factor[s] = new_box_size / box_size_x[s];
    box_size_x[s] = new_box_size;
    box_size_y[s] = new_box_size;
}

}  // namespace kernels

}  // namespace md::routines