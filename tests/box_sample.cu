// #include "utils/device_fields.hpp"
// #include "particles/base_particle.hpp"
// #include "particles/disk.hpp"
// #include "utils/h5_io.hpp"
// #include <iostream>
// #include <vector>


// int main(int argc, char** argv) {
//     int n_particles_per_system = 2;
//     int n_systems = std::stoi(argv[1]);
//     int n_particles = n_particles_per_system * n_systems;
//     int n_samples = std::stoi(argv[2]);
//     unsigned long long seed = std::stoull(argv[3]);
//     double box_length = std::stod(argv[4]);
//     double radius = std::stod(argv[5]);
//     std::string filename = argv[6];

//     md::Disk disk;
//     disk.enable_swap(false);
//     disk.allocate_particles(n_particles);
//     disk.allocate_systems(n_systems);

//     double range_low = radius;  // should add a slight offset if using rigid bumpy particles
//     double range_high = box_length - radius;  // should add a slight offset if using rigid bumpy particles

//     disk.pos.stateless_rand_uniform(range_low, range_high, range_low, range_high);
//     disk.rad.fill(radius);
//     std::vector<int> system_ids(n_particles);
//     for (int i = 0; i < n_particles; i++) {
//         system_ids[i] = i / n_particles_per_system;
//     }
//     disk.system_id.from_host(system_ids);

//     std::vector<double> box_sizes_x(n_systems), box_sizes_y(n_systems);
//     std::vector<double> box_inv_x(n_systems), box_inv_y(n_systems);
//     for (int i = 0; i < n_systems; i++) {
//         box_sizes_x[i] = box_length;
//         box_sizes_y[i] = box_length;
//         box_inv_x[i] = 1.0 / box_sizes_x[i];
//         box_inv_y[i] = 1.0 / box_sizes_y[i];
//     }
//     disk.box_size.from_host(box_sizes_x, box_sizes_y);
//     disk.box_inv.from_host(box_inv_x, box_inv_y);
//     disk.system_size.fill(n_particles_per_system);
//     std::vector<int> system_offsets(n_systems + 1, 0);
//     for (int i = 0; i < n_systems; i++) {
//         system_offsets[i+1] = system_offsets[i] + n_particles_per_system;
//     }
//     disk.system_offset.from_host(system_offsets);
//     disk.e_interaction.fill(1.0);

//     disk.sync_box();
//     disk.sync_system();
//     disk.sync_neighbors();
//     disk.sync_cells();
//     disk.sync_class_constants();
//     disk.set_neighbor_method(md::NeighborMethod::Naive);
//     disk.init_neighbors();

//     disk.pos.enable_rng(seed);

//     // create an hdf5 file
//     hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
//     hid_t meta = H5Gcreate(file, "meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//     hid_t samples = H5Gcreate(file, "samples", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//     write_scalar(meta, "n_particles", n_particles);
//     write_scalar(meta, "n_systems", n_systems);
//     write_scalar(meta, "n_samples", n_samples);
//     write_vector(meta, "system_offset", system_offsets);
//     write_vector_2d(meta, "box_size", box_sizes_x, box_sizes_y);
//     std::vector<double> pe(n_particles), pos_x(n_particles), pos_y(n_particles);

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
//     hid_t d_pe    = make_ds_2d("pe");
//     hsize_t count[2] = {1, P};
//     hid_t   mspace   = H5Screate_simple(2, count, nullptr);  // fixed shape [1, P]

//     hid_t fspace_x = H5Dget_space(d_pos_x);
//     hid_t fspace_y = H5Dget_space(d_pos_y);
//     hid_t fspace_pe= H5Dget_space(d_pe);

//     std::cout << "starting loop" << std::endl;

//     // start the timer
//     auto start = std::chrono::high_resolution_clock::now();

//     for (int i = 0; i < n_samples; i++) {
//         disk.pos.rand_uniform(range_low, range_high, range_low, range_high);
//         disk.compute_forces();
//         disk.compute_wall_forces();  // not needed for disks because it is trivial to place them in the box with proper boundary selection
//         cudaDeviceSynchronize();

//         disk.pos.to_host(pos_x, pos_y);
//         disk.pe.to_host(pe);
//         // hid_t sample_i = H5Gcreate(samples, std::to_string(i).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//         // write_vector_2d(sample_i, "pos", pos_x, pos_y);
//         // write_vector(sample_i, "pe", pe);
//         // H5Gclose(sample_i);
//         hsize_t start[2] = {static_cast<hsize_t>(i), 0};
//         // select slice on each filespace, then write
//         H5Sselect_hyperslab(fspace_x, H5S_SELECT_SET, start, nullptr, count, nullptr);
//         H5Dwrite(d_pos_x, H5T_NATIVE_DOUBLE, mspace, fspace_x, H5P_DEFAULT, pos_x.data());

//         H5Sselect_hyperslab(fspace_y, H5S_SELECT_SET, start, nullptr, count, nullptr);
//         H5Dwrite(d_pos_y, H5T_NATIVE_DOUBLE, mspace, fspace_y, H5P_DEFAULT, pos_y.data());

//         H5Sselect_hyperslab(fspace_pe, H5S_SELECT_SET, start, nullptr, count, nullptr);
//         H5Dwrite(d_pe,    H5T_NATIVE_DOUBLE, mspace, fspace_pe, H5P_DEFAULT, pe.data());
//     }

//     // end the timer
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//     std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

//     // close the file
//     H5Sclose(mspace);
//     H5Sclose(fspace_x);
//     H5Sclose(fspace_y);
//     H5Sclose(fspace_pe);

//     H5Dclose(d_pos_x);
//     H5Dclose(d_pos_y);
//     H5Dclose(d_pe);
//     H5Gclose(samples);
//     H5Gclose(meta);
//     H5Fclose(file);

//     return 0;
// }