#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/minimizers.cuh"
#include "integrators/velocity_verlet.cuh"
#include "integrators/damped_velocity_verlet.cuh"
#include "utils/output_manager.hpp"

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <in_path> <out_path> <n_steps> <save_increment> <damping_scale> <compression_frequency> <temperature_target>" << std::endl;
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    const int n_steps = std::stoi(argv[3]);
    const int save_increment = std::stoi(argv[4]);
    const double damping_scale = std::stod(argv[5]);
    const int compression_frequency = std::stoi(argv[6]);
    const double temperature_target = std::stod(argv[7]);
    const double dt_scale = 1e-2;
    const double phi_increment = 1e-3;
    
    
    md::rigid_bumpy::RigidBumpy P;
    P.load_from_hdf5(in_path, "init");
    
    df::DeviceField1D<double> dt; dt.resize(P.n_systems()); dt.fill(dt_scale);
    df::DeviceField1D<double> damping; damping.resize(P.n_systems()); damping.fill(damping_scale);
    
    md::integrators::VelocityVerlet vv(P, dt);
    vv.init();

    io::OutputManager<md::rigid_bumpy::RigidBumpy> om(P, out_path, 10, false);
    om.set_extra_init_fields({"packing_fraction", "box_size"});
    om.set_extra_final_fields({"packing_fraction", "box_size"});
    om.set_trajectory_fields({"pos", "angle", "pe_total", "ke_total", "box_size", "packing_fraction"});
    om.set_trajectory_interval(save_increment);
    om.initialize();

    std::cout << "Running for " << n_steps << " steps" << std::endl;
    for (int i = 0; i < n_steps; i++) {
        vv.step();
        P.set_temperature(temperature_target);
        om.step(i);
        if (i % compression_frequency == 0) {
            P.increment_packing_fraction(phi_increment);
        }
        if (i % 10000 == 0) {
            std::cout << "Step " << i << std::endl;
        }
    }
    om.finalize();
    std::cout << "Done" << std::endl;
}