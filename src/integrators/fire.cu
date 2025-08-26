#include "integrators/fire.cuh"

namespace md::integrators {

namespace kernels {

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
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (s >= S) return;

    // initialize placeholders (will have no action unless the branch is taken)
    dt_reverse[s] = 0;
    velocity_scale[s] = 1.0;

    if (dt[s] == 0.0) {  // if the timestep is 0, the system is stopped, so do nothing
        velocity_scale[s] = 0.0;
        alpha[s] = 0.0;
        return;
    }

    if (power[s] > 0) {  // if moving downhill, increase the inertia
        N_good[s]++;
        N_bad[s] = 0;
        if (N_good[s] > N_min) {  // if the system has been moving downhill for enough steps, increase the timestep and mixing ratio
            dt[s] = fmin(dt[s] * f_inc, dt_max[s]);
            alpha[s] *= f_alpha;
        }
    } else {  // if moving uphill, decrease the inertia
        N_good[s] = 0;
        N_bad[s]++;
        if (N_bad[s] > N_bad_max) {  // system is stopped.  do nothing.
            dt[s] = 0.0;
            return;
        }
        dt[s] = fmax(dt[s] * f_dec, dt_min[s]);
        alpha[s] = alpha_init;
        dt_reverse[s] = -dt[s] * 0.5;  // move the positions back a half step
        velocity_scale[s] = 0.0;  // stop the motion
    }
}

__global__ void fire_convergence_kernel(
    double* __restrict__ pe_total,
    double* __restrict__ pe_total_prev,
    double* __restrict__ dt,
    double avg_pe_diff_target,
    double avg_pe_target
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (s >= S) return;

    int dof = (md::geo::g_sys.offset[s+1] - md::geo::g_sys.offset[s]);  // TODO: this should likely be a class-level variable - tracking degrees of freedom in each system!
    double avg_pe = pe_total[s] / dof;
    double avg_pe_prev = pe_total_prev[s] / dof;
    double avg_pe_diff = avg_pe_prev == 0 ? 0 : std::abs(avg_pe / avg_pe_prev - 1);
    if (avg_pe_diff < avg_pe_diff_target || avg_pe < avg_pe_target) {
        dt[s] = 0.0;  // set the timestep to 0 to stop the system
    }
    pe_total_prev[s] = avg_pe;  // update the previous potential energy
}

} // namespace kernels

} // namespace md::integrators