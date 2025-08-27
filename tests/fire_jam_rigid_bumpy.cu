#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/jammers.cuh"

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

    const double dt_scale = 1e-2;
    const int n_minimization_steps = 1e5;
    const int n_compression_steps = 1e4;
    double avg_pe_diff_target = 1e-16;
    double avg_pe_target = 1e-16;
    double phi_increment = 1e-3;  // good to use 1e-2
    double phi_tolerance = 1e-10;

    df::DeviceField1D<double> dt; dt.resize(P.n_systems()); dt.fill(dt_scale);

    md::routines::jam_binary_search_wall(P, dt, n_compression_steps, n_minimization_steps, avg_pe_target, avg_pe_diff_target, phi_increment, phi_tolerance);

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