#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/minimizers.cuh"
#include "integrators/velocity_verlet.cuh"
#include "utils/output_manager.hpp"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <in_path> <out_path> <n_steps> <rng_seed>" << std::endl;
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    const int n_steps = std::stoi(argv[3]);
    const int rng_seed = std::stoi(argv[4]);

    md::rigid_bumpy::RigidBumpy P;
    P.load_from_hdf5(in_path, "init", {"domain"});  // use the domain optional loading scheme
    P.pos.enable_rng(rng_seed);
    
    io::OutputManager<md::rigid_bumpy::RigidBumpy> om(P, out_path, 10, true);
    om.set_extra_init_fields({"pe_total", "packing_fraction"});
    om.set_extra_final_fields({"pe_total", "packing_fraction"});
    om.set_trajectory_fields({"pos", "angle", "pe_total"});
    om.set_trajectory_interval(1);
    om.initialize();

    for (int i = 0; i < n_steps; i++) {
        P.set_random_positions_in_domains();
        P.compute_wall_forces();
        // TODO: add a way to manually save an array to trajectory under the current step with optional array indices to save:
        // run some kernel to determine the indices where the system pe is zero -> zero_indices
        // om.save_trajectory_field(i, P.pos, zero_indices)  // works for 2d and 1d fields
        // om.save_trajectory_field(i, P.angle, zero_indices)
        om.step(i);
    }
    om.finalize();
}