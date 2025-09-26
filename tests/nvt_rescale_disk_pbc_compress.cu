#include "particles/disk.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/minimizers.cuh"
#include "integrators/velocity_verlet.cuh"
#include "integrators/damped_velocity_verlet.cuh"
#include "utils/output_manager.hpp"

// Run NVE dynamics with a rescaling thermostat
// For the first half of the run, incrementally (de)compress the system while maintaining the desired temperature
// The second half of the run is meant to equilibrate the system at the desired temperature

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <in_path> <out_path> <n_steps> <phi_increment> <temperature_target>" << std::endl;
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    const int n_steps = std::stoi(argv[3]);
    const double phi_increment = std::stod(argv[4]);  // total amount we will be (de)compressing the system
    const double temperature_target = std::stod(argv[5]);  // target temperature for the system
    const double dt_scale = 1e-2;

    const int compression_frequency = std::min(n_steps, 100);  // how often we will be (de)compressing the system
    const double phi_step = phi_increment / (n_steps / compression_frequency);  // increment we will be (de)compressing the system by
    const int temperature_frequency = std::min(n_steps, 10);  // how often we will be setting the temperature of the system

    md::disk::Disk P;
    P.load_from_hdf5(in_path, "init");
    
    df::DeviceField1D<double> dt; dt.resize(P.n_systems()); dt.fill(dt_scale);
    md::integrators::VelocityVerlet vv(P, dt);
    vv.init();

    io::OutputManager<md::disk::Disk> om(P, out_path, 10, false);
    om.set_extra_init_fields({"packing_fraction", "box_size", "pos", "vel"});
    om.set_extra_final_fields({"packing_fraction", "box_size", "pos", "vel"});
    om.set_trajectory_fields({"pos", "vel", "pe_total", "ke_total", "box_size", "temperature"});
    om.set_trajectory_interval(1e2);
    om.initialize();

    std::cout << "Running for " << n_steps << " steps" << std::endl;
    int i = 0;
    while (i < n_steps) {
        vv.step();
        om.step(i);
        if (i % 10000 == 0) {
            std::cout << "Step " << i << std::endl;
        }
        if (i % compression_frequency == 0) {
            P.increment_packing_fraction(phi_step);
        }
        if (i % temperature_frequency == 0) {
            P.set_temperature(temperature_target);
        }
        i++;
    }
    while (i < 2 * n_steps) {
        vv.step();
        om.step(i);
        if (i % 10000 == 0) {
            std::cout << "Step " << i << std::endl;
        }
        if (i % temperature_frequency == 0) {
            P.set_temperature(temperature_target);
        }
        i++;
    }
    om.finalize();
    std::cout << "Done" << std::endl;
}