#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/minimizers.cuh"
#include "integrators/velocity_verlet.cuh"
#include "utils/output_manager.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <in_path> <out_path> <n_steps>" << std::endl;
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    const int n_steps = std::stoi(argv[3]);
    const double dt_scale = 1e-2;

    md::rigid_bumpy::RigidBumpy P;
    P.load_from_hdf5(in_path, "init");
    
    df::DeviceField1D<double> dt; dt.resize(P.n_systems()); dt.fill(dt_scale);
    md::integrators::VelocityVerletWall vvw(P, dt);
    vvw.init();

    io::OutputManager<md::rigid_bumpy::RigidBumpy> om(P, out_path, 10, true);
    om.set_extra_init_fields({"pe_total", "ke_total", "packing_fraction"});
    om.set_extra_final_fields({"pe_total", "ke_total", "packing_fraction"});
    om.set_trajectory_fields({"pos", "angle", "vertex_pos"});
    om.set_trajectory_interval(100);
    om.initialize();

    // TODO: add pre-req calculations

    std::cout << "Running for " << n_steps << " steps" << std::endl;
    for (int i = 0; i < n_steps; i++) {
        vvw.step();
        om.step(i);
    }
    om.finalize();
    std::cout << "Done" << std::endl;
}