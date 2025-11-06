#include "particles/disk.cuh"
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

    md::disk::Disk P;
    P.load_from_hdf5(in_path, "init");

    

    P.compute_forces();

    io::OutputManager<md::disk::Disk> om(P, out_path, 1, false);
    om.set_extra_final_fields({"pe_total", "box_size", "packing_fraction", "n_contacts_total", "hessian_xx", "hessian_xy", "hessian_yx", "hessian_yy", "pair_forces", "pair_ids"});
    om.initialize();
    om.finalize();
}