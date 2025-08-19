#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"

int main() {
    std::string path = "/home/mmccraw/dev/analysis/fall-25/test-dp3/rigid_box_sample.h5";
    const double box_length = 10.0;
    const double energy_scale = 1.0;

    hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) { std::cerr << "Failed to open " << path << "\n"; return 1; }

    int n_particles = read_scalar<int>(f, "n_particles");
    int n_systems = read_scalar<int>(f, "n_systems");
    int n_vertices = read_scalar<int>(f, "n_vertices");
    int n_particles_per_system = read_scalar<int>(f, "n_particles_per_system");
    int n_vertices_per_system = read_scalar<int>(f, "n_vertices_per_system");
    std::vector<double> vertex_pos_x, vertex_pos_y;
    std::tie(vertex_pos_x, vertex_pos_y) = read_vector_2d<double>(f, "vertex_pos");
    std::vector<double> angle = read_vector<double>(f, "angle");
    std::vector<int> system_id = read_vector<int>(f, "system_id");
    std::vector<int> system_size = read_vector<int>(f, "system_size");
    std::vector<int> system_offset = read_vector<int>(f, "system_offset");
    std::vector<int> vertex_particle_id = read_vector<int>(f, "vertex_particle_id");
    std::vector<int> vertex_system_id = read_vector<int>(f, "vertex_system_id");
    std::vector<int> vertex_system_offset = read_vector<int>(f, "vertex_system_offset");
    std::vector<int> vertex_system_size = read_vector<int>(f, "vertex_system_size");
    std::vector<int> particle_offset = read_vector<int>(f, "particle_offset");
    std::vector<double> vertex_rad = read_vector<double>(f, "vertex_rad");
    std::vector<double> pos_x, pos_y;
    std::tie(pos_x, pos_y) = read_vector_2d<double>(f, "pos");
    std::vector<int> n_vertices_per_particle = read_vector<int>(f, "n_vertices_per_particle");


    md::rigid_bumpy::RigidBumpy P;
    
    P.set_neighbor_method(md::NeighborMethod::Naive);

    P.allocate_systems(n_systems);
    P.allocate_particles(n_particles);
    P.allocate_vertices(n_vertices);

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
    P.vertex_mass.fill(1.0);
    P.vertex_pe.fill(0.0);
    P.vertex_force.fill(0.0, 0.0);
    P.force.fill(0.0, 0.0);
    P.pe.fill(0.0);
    P.vertex_rad.from_host(vertex_rad);

    P.sync_box();
    P.sync_system();
    P.sync_neighbors();
    P.sync_cells();
    P.sync_class_constants();

    P.init_neighbors();

    P.set_random_positions();
    P.compute_forces();
    P.compute_particle_forces();
}