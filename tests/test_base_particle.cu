// test_base_particle.cu
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "particles/base_particle.hpp"
#include "kernels/base_particle_kernels.cuh"

namespace md {
// Trivial concrete type: uses BaseParticle defaults (no-op impls)
struct Dummy : BaseParticle<Dummy> {};
}

static inline void check_close(double a, double b, double tol=1e-12) {
    assert(std::isfinite(a) && std::isfinite(b));
    double rel = std::abs(a-b) / std::max(1.0, std::abs(b));
    if (rel > tol) {
        std::cerr << "check_close fail: got " << a << " expected " << b << "\n";
        std::abort();
    }
}

int main() {
    md::Dummy P;

    // Two systems: S=2
    const int S = 2;
    P.allocate_systems(S);

    // Set box sizes: (Lx, Ly) per system
    std::vector<double> h_Lx(S), h_Ly(S);
    h_Lx[0] = 10.0; h_Ly[0] = 5.0;
    h_Lx[1] = 20.0; h_Ly[1] = 5.0;

    // Set cell dims: (Nx, Ny) per system
    std::vector<int> h_Nx(S), h_Ny(S);
    h_Nx[0] = 5; h_Ny[0] = 2;
    h_Nx[1] = 4; h_Ny[1] = 3;

    // Copy to device fields
    P.box_size.from_host(h_Lx, h_Ly);
    P.cell_dim.from_host(h_Nx, h_Ny);
    P.cell_dim.resize(S); // ensure sized
    P.sync_box();

    std::vector<double> box_inv_x(S), box_inv_y(S);
    std::vector<double> box_size_x(S), box_size_y(S);
    P.box_inv.to_host(box_inv_x, box_inv_y);
    P.box_size.to_host(box_size_x, box_size_y);
    for (int s = 0; s < S; s++) {
        assert(box_inv_x[s] == 1.0 / h_Lx[s]);
        assert(box_inv_y[s] == 1.0 / h_Ly[s]);
        assert(box_size_x[s] == h_Lx[s]);
        assert(box_size_y[s] == h_Ly[s]);
    }
    // Exercise: build cell sizing + offsets (prepass runs inside)
    P.init_cell_neighbors();  // calls init_cell_sizes() + Derived::init_cell_neighbors_impl() (no-op) + sync_cells()

    // Pull results back
    std::vector<double> csx(S), csy(S), invx(S), invy(S);
    std::vector<int>    sysstart(S+1);

    P.cell_size.to_host(csx, csy);
    P.cell_inv.to_host(invx, invy);
    P.cell_system_start.to_host(sysstart);
    cudaDeviceSynchronize();

    // Expected per-system values
    // S0: Lx=10, Nx=5 => 2.0; Ly=5, Ny=2 => 2.5
    // S1: Lx=20, Nx=4 => 5.0; Ly=5, Ny=3 => 5/3
    check_close(csx[0], 2.0);
    check_close(csy[0], 2.5);
    check_close(invx[0], 1.0/2.0);
    check_close(invy[0], 1.0/2.5);

    check_close(csx[1], 5.0);
    check_close(csy[1], 5.0/3.0);
    check_close(invx[1], 1.0/5.0);
    check_close(invy[1], 3.0/5.0);

    // Expected cell counts: S0: 5*2=10, S1: 4*3=12
    // Exclusive scan -> [0, 10, 22]
    assert(sysstart[0] == 0);
    assert(sysstart[1] == 10);
    assert(sysstart[2] == 22);

    // Sanity on sized outputs
    assert(P.cell_count.size() == 22);
    assert(P.cell_start.size() == 23);

    std::cout << "BaseParticle cell sizing test passed.\n";
    return 0;
}