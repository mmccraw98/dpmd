#include "particles/base_particle.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Dummy : public md::BaseParticle<Dummy> {
    using Base = md::BaseParticle<Dummy>;
    // Required by CRTP:
    void compute_forces_impl() { /* no-op for now */ }
    void rebuild_neighbors_impl() { /* no-op for now */ }
    // Optional hooks (we leave reordering fused kernel for later)
    void reorder_extra_impl(const int*) {}
    void seed_rng_extra_impl(unsigned long long, unsigned long long) {}
    void enable_swap_impl(bool) {}
};

int main() {
    // Parameters
    const int N = 16;     // particles total
    const int S = 2;      // systems
    const int sizes[S] = {10, 6};
    const double Lx[S] = {10.0, 12.0};
    const double Ly[S] = {10.0,  8.0};

    // Construct and allocate
    Dummy P;
    P.enable_swap(true);
    P.allocate_particles(N);
    P.allocate_systems(S);

    // Seed RNGs for base fields
    P.seed_rng(123456789ULL, 7ULL);

    // Initialize base fields
    // positions ~ U([0,Lx), [0,Ly)) per system (quick-and-dirty using one range for whole batch)
    P.pos.rand_uniform_xy(0.0, 12.0, 0.0, 12.0);
    P.vel.fill(0.0, 0.0);
    P.force.fill(0.0, 0.0);
    P.rad.fill(0.5);
    P.mass.fill(1.0);

    // system_id: first 10 are system 0, next 6 are system 1
    {
        thrust::host_vector<int> h_sid(N, 0);
        for (int i = sizes[0]; i < N; ++i) h_sid[i] = 1;
        P.system_id.data = h_sid;
    }

    // box_size: write Lx, Ly
    {
        thrust::host_vector<double> hx(S), hy(S);
        for (int s=0; s<S; ++s) { hx[s]=Lx[s]; hy[s]=Ly[s]; }
        P.box_size.x.data = hx;
        P.box_size.y.data = hy;
    }

    // system_size: write sizes
    {
        thrust::host_vector<int> hs(S); for (int s=0; s<S; ++s) hs[s]=sizes[s];
        P.system_size.data = hs;
    }

    // system_offset: compute exclusive scan on host and upload
    {
        std::vector<int> off(S+1); off[0]=0;
        for (int s=0; s<S; ++s) off[s+1]=off[s]+sizes[s];
        thrust::host_vector<int> h_off(S+1);
        for (int i=0;i<S+1;++i) h_off[i]=off[i];
        P.system_offset.data = h_off;
    }

    // Pull some values back and sanity check sizes match
    assert(static_cast<int>(P.n_particles()) == N);
    assert(static_cast<int>(P.n_systems()) == S);

    std::vector<double> x,y; P.pos.to_host(x,y);
    std::vector<int> sid;     P.system_id.to_host(sid);

    // Print a couple values to prove life
    std::cout << "N=" << P.n_particles() << " S=" << P.n_systems() << "\n";
    std::cout << "pos[0]=(" << x[0] << "," << y[0] << "), sid[0]=" << sid[0] << "\n";
    std::cout << "pos[N-1]=(" << x.back() << "," << y.back() << "), sid[N-1]=" << sid.back() << "\n";

    std::cout << "BaseParticle smoke test OK\n";
    return 0;
}