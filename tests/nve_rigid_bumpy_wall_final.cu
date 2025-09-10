#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/minimizers.cuh"
#include "integrators/velocity_verlet.cuh"
#include "utils/output_manager.hpp"

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <in_path> <out_path> <rng_seed> <vel_scale> <dt_scale> <n_steps>" << std::endl;
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    const int rng_seed = std::stoi(argv[3]);
    const double vel_scale = std::stod(argv[4]);
    const double dt_scale = std::stod(argv[5]);
    const int n_steps = std::stoi(argv[6]);
    const double energy_scale = 1.0;

    md::rigid_bumpy::RigidBumpy P;
    P.load_from_hdf5(in_path, "init");
    
    P.set_random_positions(0.2, 0.2);
    df::DeviceField1D<double> dt; dt.resize(P.n_systems()); dt.fill(dt_scale);
    md::routines::minimize_fire_wall(P, dt, 1e4, 1e-16, 1e-16);

    P.vel.stateless_rand_uniform(-vel_scale, vel_scale, -vel_scale, vel_scale);
    P.angular_vel.stateless_rand_uniform(-vel_scale, vel_scale);

    md::integrators::VelocityVerletWall vvw(P, dt);
    vvw.init();

    io::OutputManager<md::rigid_bumpy::RigidBumpy> om(P, out_path, 1, false);
    om.set_trajectory_fields({"pos", "angle", "pe"});
    om.set_trajectory_interval(100);
    om.initialize();

    // TODO: add pre-req calculations

    std::cout << "running for " << n_steps << " steps" << std::endl;
    for (int i = 0; i < n_steps; i++) {
        vvw.step();
        om.step(i);
    }
    om.finalize();
    std::cout << "done" << std::endl;
}