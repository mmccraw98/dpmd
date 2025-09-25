// test_disk_system_parallelism.cu

#include "particles/disk.cuh"
#include "integrators/velocity_verlet.cuh"
#include <cmath>
#include <algorithm>

int main(int argc, char** argv) {
    // TODO: something causes an illegal memory access in the compute_pair_forces_kernel when N (total across all systems) is large
    const int n_steps = 1000;
    // determine S from command line if provided, otherwise default to 10
    const int S = (argc > 1) ? atoi(argv[1]) : 10;
    const double dt_scale = 1e-2;
    const int num_particles_per_system = 1000;
    const int n_cell_dim = 4;
    const double packing_fraction = 0.5;
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
    }

    df::DeviceField1D<double> dt; dt.resize(S); dt.fill(dt_scale);

    {  // Test Cell neighbor method
        std::cout << "Testing Cell neighbor method for S = " << S << " and N = " << N << std::endl;
        md::disk::Disk P;
        P.set_neighbor_method(md::NeighborMethod::Cell); // set this before allocating particles

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

        P.sync_box();
        P.sync_system();
        P.sync_neighbors();
        P.sync_cells();
        P.sync_class_constants();
        P.init_neighbors();

        md::integrators::VelocityVerlet vv(P, dt);
        vv.init();

        // start the timer
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_steps; i++) {
            vv.step();
        }
        cudaDeviceSynchronize();

        // stop the timer
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    }
}