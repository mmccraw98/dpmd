#include "particles/rigid_bumpy.cuh"
#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"
#include "utils/cuda_utils.cuh"
#include "routines/minimizers.cuh"
#include "integrators/velocity_verlet.cuh"
#include "utils/output_manager.hpp"
#include "kernels/base_point_particle_kernels.cuh"

int main(int argc, char** argv) {
    std::string in_path = "/home/mmccraw/dev/data/09-09-25/new-initializations/rb_new";
    md::rigid_bumpy::RigidBumpy P;
    P.load_from_hdf5(in_path, "init");
    std::cout << "running" << std::endl;
}