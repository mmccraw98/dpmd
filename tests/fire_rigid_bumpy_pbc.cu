#include "particles/rigid_bumpy.cuh"
#include "integrators/fire.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <in_path> <out_path> <rng_seed> <n_steps>" << std::endl;
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    const int rng_seed = std::stoi(argv[3]);
    const int n_steps = std::stoi(argv[4]);
    const double energy_scale = 1.0;

    hid_t in_file = H5Fopen(in_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (in_file < 0) { std::cerr << "Failed to open " << in_path << "\n"; return 1; }

    int n_particles = read_scalar<int>(in_file, "n_particles");
    int n_systems = read_scalar<int>(in_file, "n_systems");
    int n_vertices = read_scalar<int>(in_file, "n_vertices");
    double box_pad = read_scalar<double>(in_file, "box_pad");
    int n_particles_per_system = read_scalar<int>(in_file, "n_particles_per_system");
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

    P.set_random_positions(0.1, 0.1);

    P.init_neighbors();

    df::DeviceField1D<double> dt; dt.resize(P.n_systems()); dt.fill(1e-2);

    md::integrators::FIRE fire(P, dt);
    fire.init();

    int n_saves = 0;
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
    std::vector<int> out_sys_offset; P.system_offset.to_host(out_sys_offset); write_vector(out_file, "system_offset", out_sys_offset);
    std::vector<int> out_vertex_sys_offset; P.vertex_system_offset.to_host(out_vertex_sys_offset); write_vector(out_file, "vertex_system_offset", out_vertex_sys_offset);
    std::vector<int> out_vertex_offset; P.particle_offset.to_host(out_vertex_offset); write_vector(out_file, "vertex_offset", out_vertex_offset);
    std::vector<double> out_vertex_rad; P.vertex_rad.to_host(out_vertex_rad); write_vector(out_file, "vertex_rad", out_vertex_rad);

    int save_increment = std::max(1, n_steps / 1000);

    for (int i = 0; i < n_steps; i++) {
        fire.step();
        if (i % save_increment == 0) {
            P.compute_packing_fraction();
            P.compute_pe_total();
            P.compute_ke_total();

            std::string group_name = "step_" + std::to_string(n_saves);
            hid_t group = H5Gcreate2(out_file, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            P.pos.to_host(out_pos_x, out_pos_y); write_vector_2d(group, "pos", out_pos_x, out_pos_y);
            P.angle.to_host(out_angle); write_vector(group, "angle", out_angle);
            P.force.to_host(out_force_x, out_force_y); write_vector_2d(group, "force", out_force_x, out_force_y);
            P.torque.to_host(out_torque); write_vector(group, "torque", out_torque);
            P.vertex_pos.to_host(out_vertex_pos_x, out_vertex_pos_y); write_vector_2d(group, "vertex_pos", out_vertex_pos_x, out_vertex_pos_y);
            P.vertex_force.to_host(out_vertex_force_x, out_vertex_force_y); write_vector_2d(group, "vertex_force", out_vertex_force_x, out_vertex_force_y);
            P.packing_fraction.to_host(out_packing_fraction); write_vector(group, "packing_fraction", out_packing_fraction);
            P.box_size.to_host(out_box_size_x, out_box_size_y); write_vector_2d(group, "box_size", out_box_size_x, out_box_size_y);
            P.pe_total.to_host(out_pe_total); write_vector(group, "pe_total", out_pe_total);
            P.ke_total.to_host(out_ke_total); write_vector(group, "ke_total", out_ke_total);

            n_saves++;
        }
    }



    write_scalar(out_file, "n_saves", n_saves);
    write_scalar(out_file, "n_particles", P.n_particles());
    write_scalar(out_file, "n_systems", P.n_systems());
    write_scalar(out_file, "n_vertices", P.n_vertices());

    H5Fclose(out_file);
    std::cout << "done" << std::endl;
}