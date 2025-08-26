#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"

// Create a 2D dataset [n_samples, n_particles]
hid_t make_empty_ds_2d(int n_samples, int n_particles, const char* name, hid_t group) {
    hsize_t dims[2] = {static_cast<hsize_t>(n_samples), static_cast<hsize_t>(n_particles)};
    hsize_t chunk[2] = {1, static_cast<hsize_t>(n_particles)}; // One sample per chunk
    
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 2, chunk);
    hid_t ds = H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    
    H5Pclose(dcpl);
    H5Sclose(space);
    return ds;
}

// Write one sample (row) to a 2D dataset
void write_sample_to_ds(hid_t dataset, int sample_idx, const std::vector<double>& data) {
    // Select which row (sample) to write to
    hsize_t start[2] = {static_cast<hsize_t>(sample_idx), 0};
    hsize_t count[2] = {1, static_cast<hsize_t>(data.size())};
    
    hid_t fspace = H5Dget_space(dataset);
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, nullptr, count, nullptr);
    
    // Memory space is just a 1D array
    hsize_t mem_dims[1] = {static_cast<hsize_t>(data.size())};
    hid_t mspace = H5Screate_simple(1, mem_dims, nullptr);
    
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, data.data());
    
    H5Sclose(mspace);
    H5Sclose(fspace);
}

int main(int argc, char** argv) {
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    const double box_length = std::stod(argv[3]);
    const int n_samples = std::stoi(argv[4]);
    const int rng_seed = std::stoi(argv[5]);
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
    std::vector<double> pe;

    H5Fclose(in_file);

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
    P.angular_vel.fill(0.0);
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

    // write to out file
    hid_t out_file = H5Fcreate(out_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (out_file < 0) { std::cerr << "Failed to open " << out_path << "\n"; return 1; }

    // get buffers to save the data
    std::vector<double> h_box_size_x, h_box_size_y, h_pe, h_pos_x, h_pos_y, h_angle;
    std::vector<int> h_system_offset, h_particle_offset, h_n_vertices_per_particle;

    std::vector<double> h_vertex_pos_x, h_vertex_pos_y;

    // create groups
    hid_t meta = H5Gcreate(out_file, "meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t samples = H5Gcreate(out_file, "samples", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // write metadata
    write_scalar(meta, "n_particles", P.n_particles());
    write_scalar(meta, "n_vertices", P.n_vertices());
    write_scalar(meta, "n_systems", P.n_systems());
    write_scalar(meta, "n_samples", n_samples);
    write_scalar(meta, "box_pad", box_pad);
    P.system_offset.to_host(h_system_offset); write_vector(meta, "system_offset", h_system_offset);
    P.particle_offset.to_host(h_particle_offset); write_vector(meta, "particle_offset", h_particle_offset);
    P.n_vertices_per_particle.to_host(h_n_vertices_per_particle); write_vector(meta, "n_vertices_per_particle", h_n_vertices_per_particle);
    P.box_size.to_host(h_box_size_x, h_box_size_y); write_vector_2d(meta, "box_size", h_box_size_x, h_box_size_y);

    // Create datasets for time series data
    hid_t pos_x_ds = make_empty_ds_2d(n_samples, P.n_particles(), "pos_x", samples);
    hid_t pos_y_ds = make_empty_ds_2d(n_samples, P.n_particles(), "pos_y", samples);
    hid_t angle_ds = make_empty_ds_2d(n_samples, P.n_particles(), "angle", samples);
    hid_t pe_ds = make_empty_ds_2d(n_samples, P.n_particles(), "pe", samples);
    // hid_t vertex_pos_x_ds = make_empty_ds_2d(n_samples, P.n_vertices(), "vertex_pos_x", samples);
    // hid_t vertex_pos_y_ds = make_empty_ds_2d(n_samples, P.n_vertices(), "vertex_pos_y", samples);

    for (int i = 0; i < n_samples; ++i) {
        // generate random positions and calculate forces, sum to particle level
        P.set_random_positions(box_pad, box_pad);
        P.compute_forces();
        P.compute_wall_forces();
        P.compute_particle_forces();

        // transfer data to host
        P.pos.to_host(h_pos_x, h_pos_y);
        P.angle.to_host(h_angle);
        P.pe.to_host(h_pe);
        // P.vertex_pos.to_host(h_vertex_pos_x, h_vertex_pos_y);
        
        // write to datasets
        write_sample_to_ds(pos_x_ds, i, h_pos_x);
        write_sample_to_ds(pos_y_ds, i, h_pos_y);
        write_sample_to_ds(angle_ds, i, h_angle);
        write_sample_to_ds(pe_ds, i, h_pe);
        // write_sample_to_ds(vertex_pos_x_ds, i, h_vertex_pos_x);
        // write_sample_to_ds(vertex_pos_y_ds, i, h_vertex_pos_y);
    }

    H5Fclose(out_file);
}