#include "particles/disk.cuh"
#include "integrators/velocity_verlet.cuh"
#include "integrators/damped_velocity_verlet.cuh"
#include <cmath>
#include <algorithm>
#include "utils/h5_io.hpp"

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <num_particles_per_system> <dt_scale> <n_steps> <S> <out_path>" << std::endl;
        return 1;
    }
    const int num_particles_per_system = atoi(argv[1]);
    const double dt_scale = atof(argv[2]);
    const int n_steps = atoi(argv[3]);
    const int S = atoi(argv[4]);
    std::string out_path = argv[5];

    const int n_cell_dim = 4;
    const double packing_fraction = 0.3;
    const double rad = 0.5;
    const double mass = 1.0;
    const double e_interaction = 1.0;
    const int N = num_particles_per_system * S;
    const double box_size = std::sqrt(num_particles_per_system * M_PI * rad * rad / packing_fraction);

    std::vector<int> host_cell_size_dim(S);
    std::vector<int> host_system_size(S);
    std::vector<int> host_system_start(S + 1);
    std::vector<int> host_cell_system_start(S + 1);
    std::vector<double> host_rad(N);
    std::vector<double> host_mass(N);
    std::vector<double> host_e_interaction(S);
    std::vector<double> host_skin(S);
    std::vector<double> host_thresh2(S);
    std::vector<double> host_box_size(S);
    std::vector<int> host_system_id(N);
    std::vector<double> host_pos_x(N), host_pos_y(N), host_force_x(N), host_force_y(N), host_pe(N);
    std::vector<int> host_neighbor_ids;
    std::vector<int> host_neighbor_start;
    std::vector<double> host_area(N);
    host_system_start[0] = 0;
    host_cell_system_start[0] = 0;
    for (int i = 0; i < S; i++) {
        host_cell_size_dim[i] = n_cell_dim;
        host_system_size[i] = num_particles_per_system;
        host_system_start[i + 1] = host_system_start[i] + num_particles_per_system;
        host_cell_system_start[i + 1] = host_cell_system_start[i] + n_cell_dim * n_cell_dim;
        host_box_size[i] = box_size;
        host_e_interaction[i] = e_interaction;
        host_skin[i] = 2.0 * rad;
        host_thresh2[i] = (0.5 * host_skin[i]) * (0.5 * host_skin[i]);
        for (int j = 0; j < num_particles_per_system; j++) {
            host_system_id[host_system_start[i] + j] = i;
        }
    }
    for (int i = 0; i < N; i++) {
        host_mass[i] = mass;
        host_rad[i] = rad;
        host_area[i] = M_PI * rad * rad;
    }

    df::DeviceField1D<double> dt; dt.resize(S); dt.fill(dt_scale);


    md::disk::Disk P;
    P.set_neighbor_method(md::NeighborMethod::Naive); // set this before allocating particles

    P.allocate_systems(S);
    P.allocate_particles(N);

    P.neighbor_cutoff.from_host(host_skin);
    P.thresh2.from_host(host_thresh2);
    P.system_id.from_host(host_system_id);
    P.system_size.from_host(host_system_size);
    P.system_offset.from_host(host_system_start);
    P.cell_dim.from_host(host_cell_size_dim, host_cell_size_dim);
    P.cell_system_start.from_host(host_cell_system_start);
    P.box_size.from_host(host_box_size, host_box_size);
    P.e_interaction.from_host(host_e_interaction);
    P.rad.from_host(host_rad);
    P.mass.from_host(host_mass);
    P.rad.from_host(host_rad);
    P.pos.stateless_rand_uniform(0.0, box_size, 0.0, box_size, 0);
    P.area.from_host(host_area);

    P.sync_box();
    P.sync_system();
    P.sync_neighbors();
    P.sync_cells();
    P.sync_class_constants();
    P.init_neighbors();
    P.compute_packing_fraction();

    {  // Equilibrate initially
        df::DeviceField1D<double> damping(S); damping.fill(1.0);
        md::integrators::DampedVelocityVerlet dvv(P, dt, damping);
        dvv.init();
        for (int i = 0; i < n_steps; i++) {
            dvv.step();
        }
    }

    double vel_scale = 5e-2;
    P.vel.stateless_rand_uniform(-vel_scale, vel_scale, -vel_scale, vel_scale);

    md::integrators::VelocityVerlet vv(P, dt);
    vv.init();

    int n_saves = 0;
    hid_t out_file = H5Fcreate(out_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (out_file < 0) { std::cerr << "Failed to create " << out_path << "\n"; return 1; }

    std::vector<double> out_pos_x, out_pos_y;
    std::vector<double> out_force_x, out_force_y;
    std::vector<double> out_packing_fraction;
    std::vector<double> out_box_size_x, out_box_size_y;
    std::vector<double> out_pe_total;
    std::vector<double> out_ke_total;
    std::vector<double> out_rad; P.rad.to_host(out_rad); write_vector(out_file, "rad", out_rad);
    std::vector<int> out_sys_offset; P.system_offset.to_host(out_sys_offset); write_vector(out_file, "system_offset", out_sys_offset);

    int save_increment = std::max(1, n_steps / 1000);

    for (int i = 0; i < n_steps; i++) {
        vv.step();
        if (i % save_increment == 0) {
            P.compute_packing_fraction();
            P.compute_pe_total();
            P.compute_ke_total();

            std::string group_name = "step_" + std::to_string(n_saves);
            hid_t group = H5Gcreate2(out_file, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            P.pos.to_host(out_pos_x, out_pos_y); write_vector_2d(group, "pos", out_pos_x, out_pos_y);
            P.force.to_host(out_force_x, out_force_y); write_vector_2d(group, "force", out_force_x, out_force_y);
            P.packing_fraction.to_host(out_packing_fraction); write_vector(group, "packing_fraction", out_packing_fraction);
            P.box_size.to_host(out_box_size_x, out_box_size_y); write_vector_2d(group, "box_size", out_box_size_x, out_box_size_y);
            P.pe_total.to_host(out_pe_total); write_vector(group, "pe_total", out_pe_total);
            P.ke_total.to_host(out_ke_total); write_vector(group, "ke_total", out_ke_total);

            n_saves++;
        }
    }

    write_scalar(out_file, "n_saves", n_saves);
    write_scalar(out_file, "n_particles", N);
    write_scalar(out_file, "n_systems", S);

    
    // std::vector<int> out_vertex_sys_offset; P.vertex_system_offset.to_host(out_vertex_sys_offset); write_vector(out_file, "vertex_system_offset", out_vertex_sys_offset);
    // std::vector<int> out_vertex_offset; P.particle_offset.to_host(out_vertex_offset); write_vector(out_file, "vertex_offset", out_vertex_offset);
    // std::vector<double> out_angle; P.angle.to_host(out_angle); write_vector(out_file, "angle", out_angle);
    // std::vector<double> out_vertex_pos_x, out_vertex_pos_y; P.vertex_pos.to_host(out_vertex_pos_x, out_vertex_pos_y); write_vector_2d(out_file, "vertex_pos", out_vertex_pos_x, out_vertex_pos_y);
    // std::vector<double> out_vertex_rad; P.vertex_rad.to_host(out_vertex_rad); write_vector(out_file, "vertex_rad", out_vertex_rad);
    // std::vector<double> out_vertex_force_x, out_vertex_force_y; P.vertex_force.to_host(out_vertex_force_x, out_vertex_force_y); write_vector_2d(out_file, "vertex_force", out_vertex_force_x, out_vertex_force_y);
    // std::vector<double> out_vertex_mass; P.vertex_mass.to_host(out_vertex_mass); write_vector(out_file, "vertex_mass", out_vertex_mass);
    H5Fclose(out_file);
    std::cout << "done" << std::endl;
}