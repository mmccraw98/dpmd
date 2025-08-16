// tests/test_particles.cu
#include "utils/device_fields.hpp"
#include "particles/base_particle.hpp"
#include "particles/disk.hpp"
#include <hdf5.h>
#include <iostream>
#include <vector>

template<typename T>
T load_scalar_h5(hid_t loc, const std::string& dset_path) {
    hid_t dset = H5Dopen2(loc, dset_path.c_str(), H5P_DEFAULT);
    if (dset < 0) {
        throw std::runtime_error("load_scalar_h5: dataset not found: " + dset_path);
    }

    // Ensure it’s actually a scalar dataspace
    hid_t space = H5Dget_space(dset);
    if (space < 0) {
        H5Dclose(dset);
        throw std::runtime_error("load_scalar_h5: H5Dget_space failed for " + dset_path);
    }
    if (H5Sget_simple_extent_ndims(space) != 0) {
        H5Sclose(space);
        H5Dclose(dset);
        throw std::runtime_error("load_scalar_h5: dataset is not scalar: " + dset_path);
    }

    T value;
    if (H5Dread(dset, df::h5_native<T>(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, &value) < 0) {
        H5Sclose(space);
        H5Dclose(dset);
        throw std::runtime_error("load_scalar_h5: read failed: " + dset_path);
    }

    H5Sclose(space);
    H5Dclose(dset);
    return value;
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "/home/mmccraw/dev/analysis/fall-25/test-dp3/data.h5";
    hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) { std::cerr << "Failed to open " << path << "\n"; return 1; }

    int n_particles = load_scalar_h5<int>(f, "n_particles");
    int n_systems = load_scalar_h5<int>(f, "n_systems");

    std::cout << "1" << std::endl;

    md::Disk disk;
    disk.enable_swap(true);
    disk.allocate_particles(n_particles);
    disk.allocate_systems(n_systems);

    std::cout << "2" << std::endl;

    disk.pos.load_h5(f, "positions");
    disk.rad.load_h5(f, "radii");
    disk.mass.load_h5(f, "masses");
    disk.system_id.load_h5(f, "system_ids");
    disk.box_size.load_h5(f, "box_sizes");
    disk.box_inv.load_h5(f, "box_inverse");
    disk.system_size.load_h5(f, "system_sizes");
    disk.system_offset.load_h5(f, "system_offsets");
    disk.e_interaction.load_h5(f, "interaction_energies");
    disk.cell_dim.load_h5(f, "cell_dim");
    disk.area.load_h5(f, "area");
    disk.verlet_skin.load_h5(f, "verlet_skin");

    std::cout << "3" << std::endl;

    H5Fclose(f);

    std::vector<int> system_ids(n_particles);
    disk.system_id.to_host(system_ids);
    int max_system_id = *std::max_element(system_ids.begin(), system_ids.end());
    std::cout << "Max system id: " << max_system_id << std::endl;

    disk.sync_box();
    std::cout << "4" << std::endl;
    disk.sync_system();
    std::cout << "5" << std::endl;
    disk.sync_neighbors();
    std::cout << "6" << std::endl;
    disk.sync_cells();
    std::cout << "7" << std::endl;
    disk.sync_class_constants();
    std::cout << "8" << std::endl;
    disk.set_neighbor_method(md::NeighborMethod::Cell);
    std::cout << "9" << std::endl;
    disk.init_neighbors();
    std::cout << "10" << std::endl;

    // start the timer
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100000; i++) {
        disk.compute_forces();
    }

    // end the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    return 0;
}