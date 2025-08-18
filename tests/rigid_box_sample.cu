// #include "utils/device_fields.hpp"
// #include "particles/base_particle.hpp"
// #include "particles/rigid.hpp"
// #include "utils/h5_io.hpp"
// #include <iostream>
// #include <vector>
// #include <numeric>

// int main(int argc, char** argv) {
//     int n_samples = std::stoi(argv[1]);
//     unsigned long long seed = std::stoull(argv[2]);
//     double box_length = std::stod(argv[3]);
//     std::string path = std::string(argv[4]);
//     std::string filename = std::string(argv[5]);

//     hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
//     if (f < 0) { std::cerr << "Failed to open " << path << "\n"; return 1; }

//     int n_particles = read_scalar<int>(f, "n_particles");
//     int n_systems = read_scalar<int>(f, "n_systems");
//     int n_vertices = read_scalar<int>(f, "n_vertices");
//     int n_particles_per_system = read_scalar<int>(f, "n_particles_per_system");
//     int n_vertices_per_system = read_scalar<int>(f, "n_vertices_per_system");
//     std::vector<double> vertex_pos_x, vertex_pos_y;
//     std::tie(vertex_pos_x, vertex_pos_y) = read_vector_2d<double>(f, "vertex_pos");
//     std::vector<double> angles = read_vector<double>(f, "angles");
//     std::vector<int> particle_id = read_vector<int>(f, "particle_id");
//     std::vector<int> particle_offset = read_vector<int>(f, "particle_offset");
//     std::vector<double> radii = read_vector<double>(f, "radii");
//     std::vector<double> pos_x, pos_y;
//     std::tie(pos_x, pos_y) = read_vector_2d<double>(f, "pos");
//     std::vector<int> n_vertices_per_particle = read_vector<int>(f, "n_vertices_per_particle");


//     md::Rigid rigid;
//     rigid.allocate_particles(n_particles);
//     rigid.allocate_systems(n_systems);
//     rigid.vertex_pos.from_host(vertex_pos_x, vertex_pos_y);
//     rigid.particle_id.from_host(particle_id);
//     rigid.particle_offset.from_host(particle_offset);
//     rigid.rad.from_host(radii);
//     rigid.angle.from_host(angles);
//     rigid.pos.from_host(pos_x, pos_y);
//     rigid.n_vertices_per_particle.from_host(n_vertices_per_particle);

//     double range_low = 0;  // should add a slight offset if using rigid bumpy particles
//     double range_high = box_length;  // should add a slight offset if using rigid bumpy particles

//     std::vector<int> system_ids(n_vertices);
//     for (int i = 0; i < n_vertices; i++) {
//         system_ids[i] = i / n_vertices_per_system;
//     }
//     rigid.system_id.from_host(system_ids);

//     std::vector<double> box_sizes_x(n_systems), box_sizes_y(n_systems);
//     std::vector<double> box_inv_x(n_systems), box_inv_y(n_systems);
//     for (int i = 0; i < n_systems; i++) {
//         box_sizes_x[i] = box_length;
//         box_sizes_y[i] = box_length;
//         box_inv_x[i] = 1.0 / box_sizes_x[i];
//         box_inv_y[i] = 1.0 / box_sizes_y[i];
//     }
//     rigid.box_size.from_host(box_sizes_x, box_sizes_y);
//     rigid.box_inv.from_host(box_inv_x, box_inv_y);
//     rigid.system_size.fill(n_vertices_per_system);
//     std::vector<int> system_offsets(n_systems + 1, 0);
//     for (int i = 0; i < n_systems; i++) {
//         system_offsets[i+1] = system_offsets[i] + n_vertices_per_system;
//     }
//     rigid.system_offset.from_host(system_offsets);
//     rigid.e_interaction.fill(1.0);
//     rigid.force.resize(n_vertices);
//     rigid.pe.resize(n_vertices);
//     rigid.particle_pe.resize(n_particles);

//     rigid.sync_box();
//     rigid.sync_system();
//     rigid.sync_neighbors();
//     rigid.sync_cells();
//     rigid.sync_class_constants();
//     rigid.set_neighbor_method(md::NeighborMethod::Naive);
//     rigid.init_neighbors();

//     rigid.pos.enable_rng(seed);

//     rigid.set_random_particle_positions(range_low, range_high, range_low, range_high);

//     // create an hdf5 file
//     hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
//     hid_t meta = H5Gcreate(file, "meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//     hid_t samples = H5Gcreate(file, "samples", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//     write_scalar(meta, "n_particles", n_particles);
//     write_scalar(meta, "n_vertices", n_vertices);
//     write_scalar(meta, "n_systems", n_systems);
//     write_scalar(meta, "n_samples", n_samples);
//     write_vector(meta, "system_offset", system_offsets);
//     write_vector_2d(meta, "box_size", box_sizes_x, box_sizes_y);
//     std::vector<double> h_pe(n_particles), h_pos_x(n_particles), h_pos_y(n_particles), h_angle(n_particles);

//     hsize_t S = n_samples, P = n_particles;

//     auto make_ds_2d = [&](const char* name)->hid_t {
//         hsize_t dims[2]   = {S, P};
//         hsize_t chunk[2]  = {std::min<hsize_t>(64,S), P}; // tune 64 if you want
//         hid_t space = H5Screate_simple(2, dims, nullptr);
//         hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
//         H5Pset_chunk(dcpl, 2, chunk);
//         // optional: H5Pset_deflate(dcpl, 3);  // gzip if you want compression
//         hid_t ds = H5Dcreate2(samples, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
//         H5Pclose(dcpl); H5Sclose(space);
//         return ds;
//     };

//     hid_t d_pos_x = make_ds_2d("pos_x");
//     hid_t d_pos_y = make_ds_2d("pos_y");
//     hid_t d_angle = make_ds_2d("angle");
//     hid_t d_pe    = make_ds_2d("pe");
//     hsize_t count[2] = {1, P};
//     hid_t   mspace   = H5Screate_simple(2, count, nullptr);  // fixed shape [1, P]

//     hid_t fspace_x = H5Dget_space(d_pos_x);
//     hid_t fspace_y = H5Dget_space(d_pos_y);
//     hid_t fspace_angle = H5Dget_space(d_angle);
//     hid_t fspace_pe= H5Dget_space(d_pe);

//     std::cout << "starting loop" << std::endl;

//     // start the timer
//     auto start = std::chrono::high_resolution_clock::now();

//     int total = 0;
//     int sum = 0;

//     for (int i = 0; i < n_samples; i++) {
//         rigid.set_random_particle_positions(range_low, range_high, range_low, range_high);
//         rigid.compute_forces();
//         rigid.compute_wall_forces();  // not needed for disks because it is trivial to place them in the box with proper boundary selection
//         rigid.sum_vertex_pe_to_particle_pe();
//         cudaDeviceSynchronize();

//         rigid.pos.to_host(h_pos_x, h_pos_y);
//         rigid.angle.to_host(h_angle);
//         rigid.particle_pe.to_host(h_pe);

//         for (int j = 0; j < h_pe.size() / 2; j++) {
//             if ((h_pe[j * 2] + h_pe[j * 2 + 1]) == 0) {
//                 total += 1;
//             }
//             sum += 1;
//         }

//         hsize_t start[2] = {static_cast<hsize_t>(i), 0};
//         H5Sselect_hyperslab(fspace_x, H5S_SELECT_SET, start, nullptr, count, nullptr);
//         H5Dwrite(d_pos_x, H5T_NATIVE_DOUBLE, mspace, fspace_x, H5P_DEFAULT, h_pos_x.data());

//         H5Sselect_hyperslab(fspace_y, H5S_SELECT_SET, start, nullptr, count, nullptr);
//         H5Dwrite(d_pos_y, H5T_NATIVE_DOUBLE, mspace, fspace_y, H5P_DEFAULT, h_pos_y.data());

//         H5Sselect_hyperslab(fspace_angle, H5S_SELECT_SET, start, nullptr, count, nullptr);
//         H5Dwrite(d_angle, H5T_NATIVE_DOUBLE, mspace, fspace_angle, H5P_DEFAULT, h_angle.data());

//         H5Sselect_hyperslab(fspace_pe, H5S_SELECT_SET, start, nullptr, count, nullptr);
//         H5Dwrite(d_pe,    H5T_NATIVE_DOUBLE, mspace, fspace_pe, H5P_DEFAULT, h_pe.data());
//     }

//     std::cout << "box length " << box_length << std::endl;
//     std::cout << "fraction " << static_cast<double>(total) / sum << std::endl;

//     // end the timer
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//     std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

//     // close the file
//     H5Sclose(mspace);
//     H5Sclose(fspace_x);
//     H5Sclose(fspace_y);
//     H5Sclose(fspace_pe);
//     H5Sclose(fspace_angle);
//     H5Dclose(d_pos_x);
//     H5Dclose(d_pos_y);
//     H5Dclose(d_pe);
//     H5Dclose(d_angle);
//     H5Gclose(samples);
//     H5Gclose(meta);
//     H5Fclose(file);

//     return 0;
// }