#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/minimizers.cuh"
#include "integrators/velocity_verlet.cuh"
#include "utils/output_manager.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <in_path> <out_path>" << std::endl;
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    const double dt_scale = 1e-2;

    md::rigid_bumpy::RigidBumpy P;
    P.load_from_hdf5(in_path, "init");
    
    io::OutputManager<md::rigid_bumpy::RigidBumpy> om(P, out_path, 1, false);
    om.set_trajectory_interval(1);
    om.set_extra_init_fields({"pe_total"});
    om.set_extra_final_fields({"pe_total"});
    om.initialize();

    df::DeviceField1D<double> dt; dt.resize(P.n_systems()); dt.fill(dt_scale);
    std::cout << "Minimizing" << std::endl;
    md::routines::minimize_fire_wall(P, dt, 1e6, 1e-4, 1e-4);
    std::cout << "Done" << std::endl;

    P.build_particle_neighbors();
    P.compute_contacts();
    std::vector<int> contacts; P.contacts.to_host(contacts);
    std::vector<int> particle_neighbor_start; P.particle_neighbor_start.to_host(particle_neighbor_start);

    std::vector<int> pair_vertex_contacts_i, pair_vertex_contacts_j; P.pair_vertex_contacts.to_host(pair_vertex_contacts_i, pair_vertex_contacts_j);
    std::vector<int> pair_ids_i, pair_ids_j; P.pair_ids.to_host(pair_ids_i, pair_ids_j);
    for (int i = 0; i < contacts.size(); i++) {
        std::cout << i << ": " << contacts[i] << std::endl;
        for (int j = particle_neighbor_start[i]; j < particle_neighbor_start[i+1]; j++) {
            std::cout << pair_ids_i[j] << " " << pair_ids_j[j] << ": " << pair_vertex_contacts_i[j] << " " << pair_vertex_contacts_j[j] << std::endl;
        }
        std::cout << std::endl;
    }

    om.finalize();
}