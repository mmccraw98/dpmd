// test_base_point_particle.cu
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "particles/base_poly_particle.hpp"
#include "kernels/common.cuh"

namespace md {

// Trivial concrete type: uses BaseParticle defaults
struct Dummy : BasePolyParticle<Dummy> {
    // Empty for now
};

}

int main() {
    const int S = 100000;
    const int num_particles_per_system = 2;
    const double packing_fraction = 0.5;
    const double rad = 0.5;
    const double vertex_rad = 0.1;
    const int num_vertices_per_particle = 3;
    const double mass = 1.0;
    const double vertex_mass = mass / num_vertices_per_particle;
    const double e_interaction = 1.0;
    const int N = num_particles_per_system * S;
    const int num_vertices_per_system = num_vertices_per_particle * num_particles_per_system;
    const int Nv = num_vertices_per_system * S;
    const double box_size = std::sqrt(num_particles_per_system * M_PI * rad * rad / packing_fraction);
    const int expected_total_vertex_neighbors = num_vertices_per_particle * (num_particles_per_system - 1) * num_vertices_per_system * S;

    std::vector<int> host_n_vertices_per_particle(N); // [x]
    std::vector<int> host_particle_offset(N+1);  // [x]
    std::vector<int> host_vertex_particle_id(Nv); // [x]
    std::vector<int> host_vertex_system_id(Nv); // [x]
    std::vector<int> host_vertex_system_offset(S+1); // [x]
    std::vector<int> host_vertex_system_size(S); // [x]
    std::vector<double> host_vertex_pos_x(Nv), host_vertex_pos_y(Nv), host_vertex_force_x(Nv), host_vertex_force_y(Nv), host_vertex_pe(Nv);
    std::vector<double> host_vertex_mass(Nv); // [x]
    std::vector<double> host_vertex_rad(Nv); // [x]

    std::vector<int> host_neighbor_ids;
    std::vector<int> host_neighbor_start;

    std::vector<int> host_system_size(S);  // [x]
    std::vector<int> host_system_offset(S + 1); // [x]
    std::vector<double> host_rad(N);  // [x]
    std::vector<double> host_mass(N);  // [x]
    std::vector<double> host_e_interaction(S); // [x]
    std::vector<double> host_box_size(S); // [x]
    std::vector<int> host_system_id(N); // [x]
    std::vector<double> host_pos_x(N), host_pos_y(N), host_force_x(N), host_force_y(N), host_pe(N);
    host_system_offset[0] = 0;
    host_vertex_system_offset[0] = 0;
    host_particle_offset[0] = 0;
    for (int i = 0; i < S; i++) {
        host_system_size[i] = num_particles_per_system;
        host_vertex_system_size[i] = num_vertices_per_system;
        int particle_system_begin = host_system_offset[i];
        int vertex_system_begin = host_vertex_system_offset[i];
        host_system_offset[i + 1] = particle_system_begin + num_particles_per_system;
        host_vertex_system_offset[i + 1] = vertex_system_begin + num_vertices_per_system;
        host_box_size[i] = box_size;
        host_e_interaction[i] = e_interaction;
        for (int j = 0; j < num_particles_per_system; j++) {
            int particle_id = particle_system_begin + j;
            host_system_id[particle_id] = i;
            host_mass[particle_id] = mass;
            host_rad[particle_id] = rad;
            host_n_vertices_per_particle[particle_id] = num_vertices_per_particle;
            host_particle_offset[particle_id + 1] = host_particle_offset[particle_id] + num_vertices_per_particle;
            for (int k = 0; k < num_vertices_per_particle; k++) {
                int vertex_id = host_particle_offset[particle_id] + k;
                host_vertex_particle_id[vertex_id] = particle_id;
                host_vertex_system_id[vertex_id] = i;
                host_vertex_mass[vertex_id] = vertex_mass;
                host_vertex_rad[vertex_id] = vertex_rad;
            }
        }
    }

    {  // test naive neighbor method
        md::Dummy P;
        P.set_neighbor_method(md::NeighborMethod::Naive);

        P.allocate_systems(S);
        P.allocate_particles(N);
        P.allocate_vertices(Nv);

        P.system_id.from_host(host_system_id);
        P.system_size.from_host(host_system_size);
        P.system_offset.from_host(host_system_offset);
        P.box_size.from_host(host_box_size, host_box_size);
        P.e_interaction.from_host(host_e_interaction);
        P.n_vertices_per_particle.from_host(host_n_vertices_per_particle);
        P.particle_offset.from_host(host_particle_offset);
        P.vertex_particle_id.from_host(host_vertex_particle_id);
        P.vertex_system_id.from_host(host_vertex_system_id);
        P.vertex_system_offset.from_host(host_vertex_system_offset);
        P.vertex_system_size.from_host(host_vertex_system_size);
        P.vertex_mass.from_host(host_vertex_mass);
        P.vertex_rad.from_host(host_vertex_rad);

        P.sync_box();
        P.sync_system();
        P.sync_neighbors();
        P.sync_cells();
        P.sync_class_constants();

        P.init_neighbors();
        assert(P.neighbor_ids.size() == expected_total_vertex_neighbors);

        P.neighbor_ids.to_host(host_neighbor_ids);
        P.neighbor_start.to_host(host_neighbor_start);

        for (int i = 0; i < Nv; i++) {
            int _n_vertices_in_particle = host_n_vertices_per_particle[host_vertex_particle_id[i]];
            int _n_vertices_in_system = host_vertex_system_size[host_vertex_system_id[i]];
            int _n_neighbors = _n_vertices_in_system - _n_vertices_in_particle;
            int _v_pid = host_vertex_particle_id[i];
            int _v_sid = host_vertex_system_id[i];
            int calculated_n_neighbors = host_neighbor_start[i+1] - host_neighbor_start[i];
            assert(_n_neighbors == calculated_n_neighbors);  // check that the number of neighbors is correct
            std::vector<int> seen_neighbors;
            for (int j = host_neighbor_start[i]; j < host_neighbor_start[i+1]; j++) {
                int neighbor_id = host_neighbor_ids[j];
                assert(neighbor_id < Nv);  // check that neighbor is a valid vertex
                assert(neighbor_id != i);  // check that neighbor is not the same vertex
                assert(host_vertex_particle_id[neighbor_id] != _v_pid);  // check that neighbor is not in the same particle
                assert(host_vertex_system_id[neighbor_id] == _v_sid);  // check that neighbor is in the same system
                // check that we haven't seen this neighbor before
                assert(std::find(seen_neighbors.begin(), seen_neighbors.end(), neighbor_id) == seen_neighbors.end());
                seen_neighbors.push_back(neighbor_id);
            }
        }
        cudaDeviceSynchronize();
        std::cout << "BasePolyParticle naive neighbor method test passed.\n";
    }
    return 0;
}