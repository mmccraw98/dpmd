#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/jammers.cuh"
#include "integrators/velocity_verlet.cuh"

namespace kernels {

__global__ void scale_box_kernel(
    double* __restrict__ box_size_x,
    double* __restrict__ box_size_y,
    const double* __restrict__ total_particle_area,
    const double* __restrict__ packing_fraction,
    const double* __restrict__ delta_phi,
    double* __restrict__ scale_factor
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int S = md::geo::g_sys.n_systems;
    if (s >= S) return;

    double new_packing_fraction = packing_fraction[s] + delta_phi[s];
    double new_box_size = sqrt(total_particle_area[s] / new_packing_fraction);

    scale_factor[s] = new_box_size / box_size_x[s];
    box_size_x[s] = new_box_size;
    box_size_y[s] = new_box_size;
}

}

int main(int argc, char** argv) {
    if (argc != 12) {
        std::cerr << "Usage: " << argv[0] << " <in_path> <out_path> <rng_seed> <vel_scale> <dt_scale> <n_steps> <n_fire_steps> <avg_pe_target> <avg_pe_diff_target> <phi_increment> <phi_tolerance>" << std::endl;
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    const int rng_seed = std::stoi(argv[3]);
    const double vel_scale = std::stod(argv[4]);
    const double dt_scale = std::stod(argv[5]);
    const int n_steps = std::stoi(argv[6]);
    const int n_fire_steps = std::stoi(argv[7]);
    const double avg_pe_target = std::stod(argv[8]);
    const double avg_pe_diff_target = std::stod(argv[9]);
    const double phi_increment = std::stod(argv[10]);
    const double phi_tolerance = std::stod(argv[11]);
    const double energy_scale = 1.0;

    hid_t in_file = H5Fopen(in_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (in_file < 0) { std::cerr << "Failed to open " << in_path << "\n"; return 1; }

    int n_particles = read_scalar<int>(in_file, "n_particles");
    int n_systems = read_scalar<int>(in_file, "n_systems");
    int n_vertices = read_scalar<int>(in_file, "n_vertices");
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
    std::vector<double> box_size_x, box_size_y;
    std::tie(box_size_x, box_size_y) = read_vector_2d<double>(in_file, "box_size");
    std::vector<double> delta_phi = read_vector<double>(in_file, "delta_phi");

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
    P.box_size.from_host(box_size_x, box_size_y);
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

    P.sync_box();
    P.sync_system();
    P.sync_neighbors();
    P.sync_cells();
    P.sync_class_constants();

    // do initial save
    hid_t out_file = H5Fcreate(out_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (out_file < 0) { std::cerr << "Failed to create " << out_path << "\n"; return 1; }

    std::vector<double> out_pos_x, out_pos_y;
    std::vector<double> out_angle;
    std::vector<double> out_force_x, out_force_y;
    std::vector<double> out_torque;
    std::vector<double> out_vertex_pos_x, out_vertex_pos_y;
    std::vector<double> out_vertex_force_x, out_vertex_force_y;
    std::vector<double> out_packing_fraction;
    std::vector<double> out_box_size_x, out_box_size_y;
    std::vector<double> out_pe_total;
    std::vector<double> out_ke_total;
    std::vector<double> out_delta_phi;
    std::vector<int> out_sys_offset; P.system_offset.to_host(out_sys_offset); write_vector(out_file, "system_offset", out_sys_offset);
    std::vector<int> out_vertex_sys_offset; P.vertex_system_offset.to_host(out_vertex_sys_offset); write_vector(out_file, "vertex_system_offset", out_vertex_sys_offset);
    std::vector<int> out_vertex_offset; P.particle_offset.to_host(out_vertex_offset); write_vector(out_file, "vertex_offset", out_vertex_offset);
    std::vector<double> out_vertex_rad; P.vertex_rad.to_host(out_vertex_rad); write_vector(out_file, "vertex_rad", out_vertex_rad);

    P.init_neighbors();

    P.compute_wall_forces();

    std::string group_name = "init";
    hid_t group = H5Gcreate2(out_file, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    P.compute_pe_total();
    P.compute_packing_fraction();
    P.pos.to_host(out_pos_x, out_pos_y); write_vector_2d(group, "pos", out_pos_x, out_pos_y);
    P.angle.to_host(out_angle); write_vector(group, "angle", out_angle);
    P.force.to_host(out_force_x, out_force_y); write_vector_2d(group, "force", out_force_x, out_force_y);
    P.torque.to_host(out_torque); write_vector(group, "torque", out_torque);
    P.vertex_pos.to_host(out_vertex_pos_x, out_vertex_pos_y); write_vector_2d(group, "vertex_pos", out_vertex_pos_x, out_vertex_pos_y);
    P.vertex_force.to_host(out_vertex_force_x, out_vertex_force_y); write_vector_2d(group, "vertex_force", out_vertex_force_x, out_vertex_force_y);
    P.packing_fraction.to_host(out_packing_fraction); write_vector(group, "packing_fraction", out_packing_fraction);
    P.box_size.to_host(out_box_size_x, out_box_size_y); write_vector_2d(group, "box_size", out_box_size_x, out_box_size_y);
    P.pe_total.to_host(out_pe_total); write_vector(group, "pe_total", out_pe_total);
    write_vector(group, "delta_phi", delta_phi);

    // scale by delta phi
    df::DeviceField1D<double> scale_factor; scale_factor.resize(P.n_systems());
    df::DeviceField1D<double> area_total = P.compute_particle_area_total();
    df::DeviceField1D<double> delta_phi_d; delta_phi_d.resize(P.n_systems()); delta_phi_d.from_host(delta_phi);
    P.compute_packing_fraction();
    auto B = md::launch::threads_for();
    auto G_S = md::launch::blocks_for(P.n_systems());
    CUDA_LAUNCH(kernels::scale_box_kernel, G_S, B,
        P.box_size.xptr(), P.box_size.yptr(), area_total.ptr(), P.packing_fraction.ptr(), delta_phi_d.ptr(), scale_factor.ptr()
    );
    P.sync_box();
    P.sync_system();
    P.sync_neighbors();
    P.sync_cells();
    P.sync_class_constants();
    P.scale_positions(scale_factor);

    // run dynamics
    df::DeviceField1D<double> dt; dt.resize(P.n_systems()); dt.fill(dt_scale);
    P.vel.stateless_rand_uniform(-vel_scale, vel_scale, -vel_scale, vel_scale);
    P.angular_vel.stateless_rand_uniform(-vel_scale, vel_scale);
    md::integrators::VelocityVerletWall vvw(P, dt);
    vvw.init();
    for (int i = 0; i < n_steps; i++) {
        vvw.step();
    }

    // jam the final configurations
    md::routines::jam_binary_search_wall(P, dt, 1e4, n_fire_steps, avg_pe_target, avg_pe_diff_target, phi_increment, phi_tolerance);

    group_name = "final";
    group = H5Gcreate2(out_file, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    P.compute_pe_total();
    P.compute_packing_fraction();
    P.pos.to_host(out_pos_x, out_pos_y); write_vector_2d(group, "pos", out_pos_x, out_pos_y);
    P.angle.to_host(out_angle); write_vector(group, "angle", out_angle);
    P.force.to_host(out_force_x, out_force_y); write_vector_2d(group, "force", out_force_x, out_force_y);
    P.torque.to_host(out_torque); write_vector(group, "torque", out_torque);
    P.vertex_pos.to_host(out_vertex_pos_x, out_vertex_pos_y); write_vector_2d(group, "vertex_pos", out_vertex_pos_x, out_vertex_pos_y);
    P.vertex_force.to_host(out_vertex_force_x, out_vertex_force_y); write_vector_2d(group, "vertex_force", out_vertex_force_x, out_vertex_force_y);
    P.packing_fraction.to_host(out_packing_fraction); write_vector(group, "packing_fraction", out_packing_fraction);
    P.box_size.to_host(out_box_size_x, out_box_size_y); write_vector_2d(group, "box_size", out_box_size_x, out_box_size_y);
    P.pe_total.to_host(out_pe_total); write_vector(group, "pe_total", out_pe_total);
    delta_phi_d.to_host(out_delta_phi); write_vector(group, "delta_phi", out_delta_phi);

    write_scalar(out_file, "n_particles", P.n_particles());
    write_scalar(out_file, "n_systems", P.n_systems());
    write_scalar(out_file, "n_vertices", P.n_vertices());

    H5Fclose(out_file);
    std::cout << "done" << std::endl;
}