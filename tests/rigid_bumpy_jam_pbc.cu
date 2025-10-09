#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/jammers.cuh"
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

    int max_compression_steps = 1e4;
    int max_minimization_steps = 1e5;
    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;
    double phi_increment = 1e-3;
    double phi_tolerance = 1e-10;

    md::rigid_bumpy::RigidBumpy P;
    P.load_from_hdf5(in_path, "init");


    std::vector<double> pos_x, pos_y; P.pos.to_host(pos_x, pos_y);
    double pos_x_sum = 0, pos_y_sum = 0;
    for (int i = 0; i < pos_x.size(); i++) {
        pos_x_sum += pos_x[i];
        pos_y_sum += pos_y[i];
    }
    std::cout << "pos_x_sum: " << pos_x_sum << " pos_y_sum: " << pos_y_sum << std::endl;

    
    io::OutputManager<md::rigid_bumpy::RigidBumpy> om(P, out_path, 1, false);
    om.set_extra_init_fields({"pe_total", "box_size"});
    om.set_extra_final_fields({"pe_total", "box_size", "packing_fraction"});
    om.initialize();

    df::DeviceField1D<double> dt; dt.resize(P.n_systems()); dt.fill(dt_scale);

    std::cout << "JAMMING" << std::endl;
    md::routines::jam_binary_search_pbc(P, dt, max_compression_steps, max_minimization_steps, avg_pe_target, avg_pe_diff_target, phi_increment, phi_tolerance);

    om.finalize();
}