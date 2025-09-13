#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/minimizers.cuh"
#include "integrators/velocity_verlet.cuh"
#include "utils/output_manager.hpp"
#include "routines/jammers.cuh"

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
    
    const int n_minimization_steps = 1e5;
    const int n_compression_steps = 1e4;
    double avg_pe_diff_target = 1e-16;
    double avg_pe_target = 1e-16;
    double phi_increment = 1e-3;  // good to use 1e-2
    double phi_tolerance = 1e-10;

    io::OutputManager<md::rigid_bumpy::RigidBumpy> om(P, out_path, 1, false);
    om.set_extra_init_fields({"pe_total", "packing_fraction", "box_size "});
    om.set_extra_final_fields({"pe_total", "packing_fraction", "box_size"});
    om.set_trajectory_interval(1);
    om.initialize();

    df::DeviceField1D<double> dt; dt.resize(P.n_systems()); dt.fill(dt_scale);
    md::routines::jam_binary_search_wall(P, dt, n_compression_steps, n_minimization_steps, avg_pe_target, avg_pe_diff_target, phi_increment, phi_tolerance);

    om.finalize();
}