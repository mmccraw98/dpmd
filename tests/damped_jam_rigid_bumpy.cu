#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_debug.hpp"

namespace k {

// Check if the system has converged
__global__ void damped_md_convergence_kernel(
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

// Perform a single FIRE step
void damped_md_step(
    md::rigid_bumpy::RigidBumpy& P,
    df::DeviceField1D<double>& velocity_scale,
    df::DeviceField1D<double>& dt,
    df::DeviceField1D<double>& damping_coefficient
) {
    P.update_velocities(dt, 0.5);
    P.scale_velocities(velocity_scale);
    P.update_positions(dt, 1.0);
    P.compute_forces();
    P.compute_wall_forces();
    P.compute_damping_forces(damping_coefficient);
    P.compute_particle_forces();
    P.update_velocities(dt, 0.5);
    P.scale_velocities(velocity_scale);
}

void minimize_damped_md(
    md::rigid_bumpy::RigidBumpy& P,
    df::DeviceField1D<double>& dt,
    int n_steps, double avg_pe_diff_target, double avg_pe_target
) {
    // parameters
    const double damping_coefficient_scale = 1.0;

    const int S = P.n_systems();

    df::DeviceField1D<double> velocity_scale; velocity_scale.resize(S);
    df::DeviceField1D<double> damping_coefficient; damping_coefficient.resize(S); damping_coefficient.fill(damping_coefficient_scale);
    df::DeviceField1D<double> pe_total_prev; pe_total_prev.resize(S); pe_total_prev.fill(1e9);

    // start with the velocities set to zero and the forces calculated
    velocity_scale.fill(0.0);
    P.scale_velocities(velocity_scale);
    velocity_scale.fill(1.0);

    P.compute_forces();
    P.compute_wall_forces();
    P.compute_damping_forces(damping_coefficient);
    P.compute_particle_forces();

    int step = 0;
    while (step < n_steps) {
        k::damped_md_step(P, velocity_scale, dt, damping_coefficient);
        P.compute_pe_total();
        auto B = md::launch::threads_for();
        auto G = md::launch::blocks_for(S);
        CUDA_LAUNCH(k::damped_md_convergence_kernel, G, B,
            P.pe_total.ptr(), pe_total_prev.ptr(), dt.ptr(), avg_pe_diff_target, avg_pe_target
        );
        step++;
        // if the sum of dt is 0, we are done
        double sum_dt = thrust::reduce(thrust::device, dt.ptr(), dt.ptr() + S, 0.0);
        if (sum_dt == 0) {
            break;
        }
    }
    std::cout << "Minimization exited after " << step << " steps" << std::endl;

    // set the velocities to zero to avoid any FIRE artifacts in preceding steps
    velocity_scale.fill(0.0);
    P.scale_velocities(velocity_scale);
}

__global__ void jamming_update_kernel(
    const double* __restrict__ pe_total,
    double* __restrict__ phi,
    double* __restrict__ phi_low,
    double* __restrict__ phi_high,
    int* __restrict__ should_revert,
    double phi_increment,
    double phi_tolerance,
    double avg_pe_target
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (s >= S) return;

    if (should_revert[s] == 0) {  // system is converged, do nothing
        return;
    }

    // default flag
    should_revert[s] = -2;

    int dof = (md::geo::g_sys.offset[s+1] - md::geo::g_sys.offset[s]);  // TODO: this should likely be a class-level variable - tracking degrees of freedom in each system!
    double avg_pe = pe_total[s] / dof;
    if (avg_pe > avg_pe_target) {  // jammed
        phi_high[s] = phi[s];
        phi[s] = (phi_high[s] + phi_low[s]) / 2.0;
        should_revert[s] = 1;  // jammed state is found, revert to last unjammed state
    } else {  // unjammed
        should_revert[s] = -1;  // unjammed state is found, set current state as last unjammed state
        phi_low[s] = phi[s];
        if (phi_high[s] > 0) {
            phi[s] = (phi_high[s] + phi_low[s]) / 2.0;
        } else {
            phi[s] += phi_increment;
        }
    }
    if (std::abs(phi_high[s] / phi_low[s] - 1) < phi_tolerance && phi_high[s] > 0) {  // converged
        should_revert[s] = 0;  // final state is found, do nothing
    }
}

__global__ void jamming_revert_kernel(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ angle,
    double* __restrict__ last_x,
    double* __restrict__ last_y,
    double* __restrict__ last_angle,
    double* __restrict__ vertex_x,
    double* __restrict__ vertex_y,
    double* __restrict__ last_vertex_x,
    double* __restrict__ last_vertex_y,
    int* __restrict__ should_revert
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = md::geo::g_sys.n_particles;
    if (i >= N) return;

    const int sid = md::geo::g_sys.id[i];
    const int action = should_revert[sid];
    if (action == 0) { return; }

    const int beg = md::poly::g_poly.particle_offset[i];
    const int end = md::poly::g_poly.particle_offset[i+1];

    if (action == 1) {  // revert to last unjammed state
        x[i] = last_x[i];
        y[i] = last_y[i];
        angle[i] = last_angle[i];
        for (int j = beg; j < end; j++) {
            vertex_x[j] = last_vertex_x[j];
            vertex_y[j] = last_vertex_y[j];
        }
    } else if (action == -1) {  // set current state as last unjammed state
        last_x[i] = x[i];
        last_y[i] = y[i];
        last_angle[i] = angle[i];
        for (int j = beg; j < end; j++) {
            last_vertex_x[j] = vertex_x[j];
            last_vertex_y[j] = vertex_y[j];
        }
    }
}


__global__ void scale_positions_kernel(
    double* __restrict__ x, double* __restrict__ y,
    double* __restrict__ vertex_x, double* __restrict__ vertex_y,
    const double* __restrict__ scale_factor
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = md::geo::g_sys.n_particles;
    if (i >= N) return;

    const int sid = md::geo::g_sys.id[i];

    const int beg = md::poly::g_poly.particle_offset[i];
    const int end = md::poly::g_poly.particle_offset[i+1];

    double pos_x = x[i];
    double pos_y = y[i];

    double new_pos_x = pos_x * scale_factor[sid];
    double new_pos_y = pos_y * scale_factor[sid];

    x[i] = new_pos_x;
    y[i] = new_pos_y;

    for (int j = beg; j < end; j++) {
        vertex_x[j] += (new_pos_x - pos_x);
        vertex_y[j] += (new_pos_y - pos_y);
    }
}

__global__ void reinit_dt_kernel(double* __restrict__ dt,
                                 const int* __restrict__ should_revert,
                                 double dt_scale) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (s >= S) return;
    // Only re-enable FIRE for systems that are not converged
    if (should_revert[s] != 0) dt[s] = dt_scale;
    else dt[s] = 0.0;
}



__global__ void scale_box_kernel(
    double* __restrict__ box_size_x,
    double* __restrict__ box_size_y,
    const double* __restrict__ area_total,
    const double* __restrict__ packing_fraction_target,
    int* __restrict__ should_revert,
    double* __restrict__ scale_factor
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (s >= S) return;

    if (should_revert[s] == 0) { scale_factor[s] = 1.0; return; }
    if (packing_fraction_target[s] <= 0.0) { scale_factor[s] = 1.0; return; }

    double new_box_size = sqrt(area_total[s] / packing_fraction_target[s]);
    // Use current array value, not md::geo::g_box (may be stale until P.sync_box)
    scale_factor[s] = new_box_size / box_size_x[s];
    box_size_x[s] = new_box_size;
    box_size_y[s] = new_box_size;
}

__global__ void remove_bad_initial_systems_kernel(
    int* __restrict__ should_revert,
    double* __restrict__ dt
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (s >= S) return;
    if (dt[s] != 0.0) {
        should_revert[s] = 0;
    }
}

void print_data(md::rigid_bumpy::RigidBumpy& P, df::DeviceField1D<double>& phi, df::DeviceField1D<double>& phi_low, df::DeviceField1D<double>& phi_high, df::DeviceField1D<int>& should_revert, df::DeviceField1D<double>& scale_factor, df::DeviceField1D<double>& dt, std::vector<int>& system_offset, int sid, std::string prefix) {
    int pid = system_offset[sid];
    std::cout << prefix << std::endl;
    std::cout << "\t Pos0: " << P.pos.x.get_element(pid) << " " << P.pos.y.get_element(pid) << " " << P.angle.get_element(pid) << std::endl;
    std::cout << "\t Pos1: " << P.pos.x.get_element(pid + 1) << " " << P.pos.y.get_element(pid + 1) << " " << P.angle.get_element(pid + 1) << std::endl;
    std::cout << "\t F0: " << P.force.x.get_element(pid) << " " << P.force.y.get_element(pid) << " " << P.torque.get_element(pid) << std::endl;
    std::cout << "\t F1: " << P.force.x.get_element(pid + 1) << " " << P.force.y.get_element(pid + 1) << " " << P.torque.get_element(pid + 1) << std::endl;
    std::cout << "\t Box: " << P.box_size.x.get_element(sid) << " " << P.box_size.y.get_element(sid) << std::endl;
    std::cout << "\t\tSR: " << should_revert.get_element(sid) << " SF: " << scale_factor.get_element(sid) << " PH: " << phi.get_element(sid) << " PH_LOW: " << phi_low.get_element(sid) << " PH_HIGH: " << phi_high.get_element(sid) << " PE: " << P.pe_total.get_element(sid) << " DT: " << dt.get_element(sid) << std::endl;
}

}

int main(int argc, char** argv) {
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    const double box_length = std::stod(argv[3]);
    const int rng_seed = std::stoi(argv[4]);
    // std::string in_path = "/home/mmccraw/dev/data/08-11-25/jamming/test/in.h5";
    // std::string out_path = "/home/mmccraw/dev/data/08-11-25/jamming/test/out.h5";
    // const int rng_seed = 10;
    const double energy_scale = 1.0;

    hid_t in_file = H5Fopen(in_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (in_file < 0) { std::cerr << "Failed to open " << in_path << "\n"; return 1; }

    int n_particles = read_scalar<int>(in_file, "n_particles");
    int n_systems = read_scalar<int>(in_file, "n_systems");
    int n_vertices = read_scalar<int>(in_file, "n_vertices");
    double box_pad = read_scalar<double>(in_file, "box_pad");
    int n_particles_per_system = read_scalar<int>(in_file, "n_particles_per_system");
    int n_vertices_per_system = read_scalar<int>(in_file, "n_vertices_per_system");
    std::vector<double> vertex_pos_x, vertex_pos_y;
    std::tie(vertex_pos_x, vertex_pos_y) = read_vector_2d<double>(in_file, "vertex_pos");
    std::vector<double> angle = read_vector<double>(in_file, "angle");
    std::vector<int> system_id = read_vector<int>(in_file, "system_id");
    std::vector<int> system_size = read_vector<int>(in_file, "system_size");
    std::vector<int> system_offset = read_vector<int>(in_file, "system_offset");
    std::vector<int> vertex_particle_id = read_vector<int>(in_file, "vertex_particle_id");
    std::vector<int> vertex_system_id = read_vector<int>(in_file, "vertex_system_id");
    std::vector<int> vertex_system_offset = read_vector<int>(in_file, "vertex_system_offset");
    std::vector<int> vertex_system_size = read_vector<int>(in_file, "vertex_system_size");
    std::vector<int> particle_offset = read_vector<int>(in_file, "particle_offset");
    std::vector<double> vertex_rad = read_vector<double>(in_file, "vertex_rad");
    std::vector<double> pos_x, pos_y;
    std::tie(pos_x, pos_y) = read_vector_2d<double>(in_file, "pos");
    std::vector<int> n_vertices_per_particle = read_vector<int>(in_file, "n_vertices_per_particle");
    std::vector<double> mass = read_vector<double>(in_file, "mass");
    std::vector<double> moment_inertia = read_vector<double>(in_file, "inertia");
    std::vector<double> area = read_vector<double>(in_file, "area");
    std::vector<double> pe;

    H5Fclose(in_file);

    md::rigid_bumpy::RigidBumpy P;
    
    P.set_neighbor_method(md::NeighborMethod::Naive);

    P.allocate_systems(n_systems);
    P.allocate_particles(n_particles);
    P.allocate_vertices(n_vertices);

    P.area.from_host(area);
    P.angle.from_host(angle);
    P.pos.from_host(pos_x, pos_y);
    P.vertex_pos.from_host(vertex_pos_x, vertex_pos_y);
    P.e_interaction.fill(energy_scale);
    P.box_size.fill(box_length, box_length);
    P.system_id.from_host(system_id);
    P.system_size.from_host(system_size);
    P.system_offset.from_host(system_offset);
    P.n_vertices_per_particle.from_host(n_vertices_per_particle);
    P.particle_offset.from_host(particle_offset);
    P.vertex_particle_id.from_host(vertex_particle_id);
    P.vertex_system_id.from_host(vertex_system_id);
    P.vertex_system_offset.from_host(vertex_system_offset);
    P.vertex_system_size.from_host(vertex_system_size);
    P.mass.from_host(mass);
    P.moment_inertia.from_host(moment_inertia);
    P.vertex_mass.fill(1.0);  // I dont think we need this
    P.vertex_pe.fill(0.0);
    P.vertex_force.fill(0.0, 0.0);
    P.force.fill(0.0, 0.0);
    P.torque.fill(0.0);
    P.pe.fill(0.0);
    P.vertex_rad.from_host(vertex_rad);
    P.pos.enable_rng(rng_seed);

    P.sync_box();
    P.sync_system();
    P.sync_neighbors();
    P.sync_cells();
    P.sync_class_constants();

    P.init_neighbors();
    P.set_random_positions(0.1, 0.1);
    // P.set_random_positions(0.5, 0.5);

    // for storing the box size update for each system
    df::DeviceField1D<double> box_update; box_update.resize(P.n_systems()); box_update.fill(0.0);


    const double dt_scale = 1e-2;
    const int n_minimization_steps = 1e5;
    const int n_compression_steps = 1e4;
    double avg_pe_diff_target = 1e-16;
    double avg_pe_target = 1e-16;
    double avg_pe_target_jamming = avg_pe_target;
    double phi_increment = 1e-3;  // good to use 1e-2
    double phi_tolerance = 1e-10;



    const int S = P.n_systems();
    const int N = P.n_particles();
    const int V = P.n_vertices();

    df::DeviceField1D<double> dt; dt.resize(S);

    // set last state
    df::DeviceField2D<double> last_pos; last_pos.resize(N); last_pos.copy_from(P.pos);
    df::DeviceField1D<double> last_angle; last_angle.resize(N); last_angle.copy_from(P.angle);
    df::DeviceField2D<double> last_vertex_pos; last_vertex_pos.resize(V); last_vertex_pos.copy_from(P.vertex_pos);
    df::DeviceField1D<int> should_revert; should_revert.resize(S); should_revert.fill(-1);
    df::DeviceField1D<double> scale_factor; scale_factor.resize(S); scale_factor.fill(0.0);

    // calculate the total particle area for each system - ASSUMED CONSERVED!
    df::DeviceField1D<double> area_total = P.compute_particle_area_total();

    // define the launch configuration
    auto B = md::launch::threads_for();
    auto G_S = md::launch::blocks_for(S);
    auto G_N = md::launch::blocks_for(N);

    // run initial minimization
    dt.fill(dt_scale);  // reset dt to the initial value
    k::minimize_damped_md(P, dt, n_minimization_steps, avg_pe_diff_target, avg_pe_target);
    // if after the first minimzation, there are some systems that are not converged, they should be marked as final so as to not slow down the next minimization
    CUDA_LAUNCH(k::remove_bad_initial_systems_kernel, G_S, B, should_revert.ptr(), dt.ptr());


    // set the initial packing fraction
    df::DeviceField1D<double> phi, phi_low, phi_high; phi.resize(S); phi_low.resize(S); phi_high.resize(S);
    P.compute_packing_fraction();
    phi.copy_from(P.packing_fraction);
    phi_low.copy_from(P.packing_fraction);
    phi_high.fill(-1.0);

    int compression_step = 0;

    while (compression_step < n_compression_steps) {

        // reset the dt for the systems that are still running
        CUDA_LAUNCH(k::reinit_dt_kernel, G_S, B, dt.ptr(), should_revert.ptr(), dt_scale);

        // run the minimization
        k::minimize_damped_md(P, dt, n_minimization_steps, avg_pe_diff_target, avg_pe_target);

        // call the jamming kernel
        CUDA_LAUNCH(k::jamming_update_kernel, G_S, B,
            P.pe_total.ptr(), phi.ptr(), phi_low.ptr(), phi_high.ptr(), should_revert.ptr(), phi_increment, phi_tolerance, avg_pe_target_jamming  // TODO: might be a hack
        );

        // revert the systems that need to be reverted
        CUDA_LAUNCH(k::jamming_revert_kernel, G_N, B,
            P.pos.xptr(), P.pos.yptr(), P.angle.ptr(), last_pos.xptr(), last_pos.yptr(), last_angle.ptr(), P.vertex_pos.xptr(), P.vertex_pos.yptr(), last_vertex_pos.xptr(), last_vertex_pos.yptr(), should_revert.ptr()
        );

        // scale the box size
        CUDA_LAUNCH(k::scale_box_kernel, G_S, B,
            P.box_size.xptr(), P.box_size.yptr(), area_total.ptr(), phi.ptr(), should_revert.ptr(), scale_factor.ptr()
        );

        P.sync_box();
        // P.sync_cells();  // TODO?
        // P.update_neighbors();  // TODO?
        P.compute_packing_fraction();

        // scale the positions
        CUDA_LAUNCH(k::scale_positions_kernel, G_N, B,
            P.pos.xptr(), P.pos.yptr(), P.vertex_pos.xptr(), P.vertex_pos.yptr(), scale_factor.ptr()
        );

        // check if the sum of should_revert is 0, we are done
        int sum_should_revert = thrust::reduce(thrust::device, should_revert.ptr(), should_revert.ptr() + S, 0);
        int max_should_revert = thrust::reduce(thrust::device, should_revert.ptr(), should_revert.ptr() + S, 0, thrust::maximum<int>());
        double phi_avg = thrust::reduce(thrust::device, phi.ptr(), phi.ptr() + S, 0.0) / S;
        double pe_avg = thrust::reduce(thrust::device, P.pe_total.ptr(), P.pe_total.ptr() + S, 0.0) / S;
        std::cout << "Sum of should_revert: " << sum_should_revert << " PHI_AVG: " << phi_avg << " PE_AVG: " << pe_avg << std::endl;
        if (sum_should_revert == 0 && max_should_revert == 0) {
            break;
        }
        compression_step++;
    }

    hid_t out_file = H5Fcreate(out_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (out_file < 0) { std::cerr << "Failed to create " << out_path << "\n"; return 1; }

    std::vector<int> out_vertex_sys_offset; P.vertex_system_offset.to_host(out_vertex_sys_offset); write_vector(out_file, "vertex_system_offset", out_vertex_sys_offset);
    std::vector<int> out_vertex_offset; P.particle_offset.to_host(out_vertex_offset); write_vector(out_file, "vertex_offset", out_vertex_offset);
    std::vector<int> out_sys_offset; P.system_offset.to_host(out_sys_offset); write_vector(out_file, "system_offset", out_sys_offset);
    std::vector<double> out_pos_x, out_pos_y; P.pos.to_host(out_pos_x, out_pos_y); write_vector_2d(out_file, "pos", out_pos_x, out_pos_y);
    std::vector<double> out_angle; P.angle.to_host(out_angle); write_vector(out_file, "angle", out_angle);
    std::vector<double> out_vertex_pos_x, out_vertex_pos_y; P.vertex_pos.to_host(out_vertex_pos_x, out_vertex_pos_y); write_vector_2d(out_file, "vertex_pos", out_vertex_pos_x, out_vertex_pos_y);
    std::vector<double> out_vertex_rad; P.vertex_rad.to_host(out_vertex_rad); write_vector(out_file, "vertex_rad", out_vertex_rad);
    std::vector<double> out_vertex_force_x, out_vertex_force_y; P.vertex_force.to_host(out_vertex_force_x, out_vertex_force_y); write_vector_2d(out_file, "vertex_force", out_vertex_force_x, out_vertex_force_y);
    std::vector<double> out_vertex_mass; P.vertex_mass.to_host(out_vertex_mass); write_vector(out_file, "vertex_mass", out_vertex_mass);
    std::vector<double> out_packing_fraction; P.packing_fraction.to_host(out_packing_fraction); write_vector(out_file, "packing_fraction", out_packing_fraction);
    std::vector<double> out_pe_total; P.pe_total.to_host(out_pe_total); write_vector(out_file, "pe_total", out_pe_total);
    std::vector<double> out_box_size_x, out_box_size_y; P.box_size.to_host(out_box_size_x, out_box_size_y); write_vector_2d(out_file, "box_size", out_box_size_x, out_box_size_y);
    H5Fclose(out_file);
}