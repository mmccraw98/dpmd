#include "particles/disk.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/minimizers.cuh"
#include "integrators/velocity_verlet.cuh"
#include "integrators/damped_velocity_verlet.cuh"
#include "utils/output_manager.hpp"

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <in_path> <out_path> <n_steps> <save_increment> <dt_scale>" << std::endl;
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    const int n_steps = std::stoi(argv[3]);
    const int save_increment = std::stoi(argv[4]);
    const double dt_scale = std::stod(argv[5]);

    md::disk::Disk P;
    P.load_from_hdf5(in_path, "init");
    
    df::DeviceField1D<double> dt; dt.resize(P.n_systems()); dt.fill(dt_scale);
    md::integrators::VelocityVerlet vv(P, dt);
    vv.init();

    io::OutputManager<md::disk::Disk> om(P, out_path, 10, false);
    om.set_extra_init_fields({});
    om.set_extra_final_fields({});
    // om.set_trajectory_fields({"pos", "vel", "pe_total", "ke_total", "temperature", "pressure", "stress_tensor_total_x", "stress_tensor_total_y"});
    // om.set_trajectory_fields({"pos", "pressure", "temperature", "overlaps", "pe_total"});
    // om.set_trajectory_fields({"pos", "pe_total", "ke_total", "temperature", "pressure", "stress_tensor_total_x", "stress_tensor_total_y"});
    om.set_trajectory_fields({"pos", "pe_total", "ke_total", "temperature", "stress_tensor_total_x", "stress_tensor_total_y", "pressure"});
    om.set_trajectory_interval(save_increment);
    om.initialize();

    // TODO: add pre-req calculations

    std::cout << "Running for " << n_steps << " steps" << std::endl;
    for (int i = 0; i < n_steps; i++) {
        vv.step();
        om.step(i);
        if (i % 10000 == 0) {
            std::cout << "Step " << i << std::endl;
        }
    }
    om.finalize();
    std::cout << "Done" << std::endl;
}