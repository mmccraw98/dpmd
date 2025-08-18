// test_disk.cu

#include "particles/disk.cuh"
#include <cmath>
#include <algorithm>

inline bool feq(double a, double b, double abs_scale=1.0, int M=100) {
    const double eps = 2.220446049250313e-16;
    const double frac_tol = std::max(1e-15, 10*eps*M);   // ~2e-13 for M=100
    const double tol      = std::max(1e-14, frac_tol * std::max({std::abs(a), std::abs(b), abs_scale}));
    if (std::abs(a-b) <= tol) return true;
    if (std::max(std::abs(a),std::abs(b)) < 1e-12*abs_scale) return false;
    return std::abs(1 - a/b) <= frac_tol;
}

struct RelStd {
    long double mean = 0.0L, M2 = 0.0L;
    long long n = 0;
    void push(double x) {
        ++n;
        long double dx = (long double)x - mean;
        mean += dx / n;
        M2   += dx * ((long double)x - mean);
    }
    double rel() const {
        if (n == 0) return 0.0;
        long double var = M2 / n;                // population variance
        long double sd  = var > 0 ? std::sqrt((double)var) : 0.0;
        long double m   = std::abs(mean);
        return m > 0 ? (double)(sd / m) : 0.0;
    }
};

double fit_loglog_slope_filtered(const std::vector<double>& dt,
                                 const std::vector<double>& y,
                                 double abs_floor = 1e-18,
                                 double rel_floor = 1e-14)
{
    // Build filtered lists (drop y <= floor)
    double ymax = 0.0;
    for (double v : y) ymax = std::max(ymax, v);
    std::vector<double> xs, ys;
    for (size_t i=0;i<y.size();++i) {
        if (y[i] > abs_floor && y[i] > rel_floor * ymax) {
            xs.push_back(std::log(dt[i]));
            ys.push_back(std::log(y[i]));
        }
    }
    // Need at least 2 points
    if (xs.size() < 2) return std::numeric_limits<double>::quiet_NaN();

    double sx=0, sy=0, sxx=0, sxy=0;
    for (size_t i=0;i<xs.size();++i) {
        sx  += xs[i];  sy  += ys[i];
        sxx += xs[i]*xs[i]; sxy += xs[i]*ys[i];
    }
    double denom = xs.size()*sxx - sx*sx;
    return (xs.size()*sxy - sx*sy) / denom;
}

int main() {
    // Two systems, each with 20 particles
    const int S = 2;
    const int num_particles_per_system = 20;
    const int n_cell_dim = 4;
    const double packing_fraction = 0.5;
    const double rad = 0.5;
    const double mass = 1.0;
    const double e_interaction = 1.0;
    const int N = num_particles_per_system * S;
    const double box_size = std::sqrt(num_particles_per_system * M_PI * rad * rad / packing_fraction);

    std::vector<int> host_cell_size_dim(S);
    std::vector<int> host_system_size(S);
    std::vector<int> host_system_start(S + 1);
    std::vector<int> host_cell_system_start(S + 1);
    std::vector<double> host_rad(N);
    std::vector<double> host_mass(N);
    std::vector<double> host_e_interaction(S);
    std::vector<double> host_skin(S);
    std::vector<double> host_box_size(S);
    std::vector<int> host_system_id(N);
    std::vector<double> host_pos_x(N), host_pos_y(N), host_force_x(N), host_force_y(N), host_pe(N);
    std::vector<int> host_neighbor_ids;
    std::vector<int> host_neighbor_start;
    host_system_start[0] = 0;
    host_cell_system_start[0] = 0;
    for (int i = 0; i < S; i++) {
        host_cell_size_dim[i] = n_cell_dim;
        host_system_size[i] = num_particles_per_system;
        host_system_start[i + 1] = host_system_start[i] + num_particles_per_system;
        host_cell_system_start[i + 1] = host_cell_system_start[i] + n_cell_dim * n_cell_dim;
        host_box_size[i] = box_size;
        host_e_interaction[i] = e_interaction;
        host_skin[i] = 2.0 * rad;  // hows this?
        for (int j = 0; j < num_particles_per_system; j++) {
            host_system_id[host_system_start[i] + j] = i;
        }
    }
    for (int i = 0; i < N; i++) {
        host_mass[i] = mass;
        host_rad[i] = rad;
    }

    std::vector<double> host_force_x_naive(N), host_force_y_naive(N);
    std::vector<double> host_force_x_cell(N), host_force_y_cell(N);
    std::vector<double> host_pos_x_cell(N), host_pos_y_cell(N);
    std::vector<double> host_pos_x_naive(N), host_pos_y_naive(N);

    {  // Naive neighbor method
        md::disk::Disk P;
        P.set_neighbor_method(md::NeighborMethod::Naive); // set this before allocating particles

        P.allocate_systems(S);
        P.allocate_particles(N);

        P.verlet_skin.from_host(host_skin);
        P.system_id.from_host(host_system_id);
        P.system_size.from_host(host_system_size);
        P.system_offset.from_host(host_system_start);
        P.cell_dim.from_host(host_cell_size_dim, host_cell_size_dim);
        P.cell_system_start.from_host(host_cell_system_start);
        P.box_size.from_host(host_box_size, host_box_size);
        P.e_interaction.from_host(host_e_interaction);
        P.rad.from_host(host_rad);
        P.mass.from_host(host_mass);
        P.rad.from_host(host_rad);
        P.pos.enable_rng(0);
        P.pos.stateless_rand_uniform(0.0, box_size, 0.0, box_size, 0);

        P.sync_box();
        P.sync_system();
        P.sync_neighbors();
        P.sync_cells();
        P.sync_class_constants();
        P.init_neighbors();

        P.compute_forces();

        P.neighbor_ids.to_host(host_neighbor_ids);
        P.neighbor_start.to_host(host_neighbor_start);
        P.pos.to_host(host_pos_x, host_pos_y);
        P.force.to_host(host_force_x, host_force_y);
        P.pe.to_host(host_pe);
        for (int i = 0; i < N; i++) {
            bool is_isolated = host_pe[i] == 0.0;
            if (is_isolated) {
                assert(host_force_x[i] == 0.0);
                assert(host_force_y[i] == 0.0);
            }
            for (int j = host_neighbor_start[i]; j < host_neighbor_start[i + 1]; j++) {
                int neighbor_id = host_neighbor_ids[j];
                if (neighbor_id == i) continue;
                double dx = host_pos_x[neighbor_id] - host_pos_x[i];
                double dy = host_pos_y[neighbor_id] - host_pos_y[i];
                double r2 = dx * dx + dy * dy;
                double r = std::sqrt(r2);
                if (is_isolated) {
                    assert(r > 2.0 * rad);
                }
                if (r < 2.0 * rad) {
                    assert(std::abs(host_force_x[i]) > 0.0);
                    assert(std::abs(host_force_y[i]) > 0.0);
                    assert(std::abs(host_force_x[neighbor_id]) > 0.0);
                    assert(std::abs(host_force_y[neighbor_id]) > 0.0);
                    assert(host_pe[i] > 0.0);
                    assert(host_pe[neighbor_id] > 0.0);
                }
            }
        }
        P.force.to_host(host_force_x_naive, host_force_y_naive);
        P.pos.to_host(host_pos_x_naive, host_pos_y_naive);
        std::cout << "Naive neighbor test passed" << std::endl;
    }

    {  // Cell neighbor method
        md::disk::Disk P;
        P.set_neighbor_method(md::NeighborMethod::Cell); // set this before allocating particles

        P.allocate_systems(S);
        P.allocate_particles(N);

        P.verlet_skin.from_host(host_skin);
        P.system_id.from_host(host_system_id);
        P.system_size.from_host(host_system_size);
        P.system_offset.from_host(host_system_start);
        P.cell_dim.from_host(host_cell_size_dim, host_cell_size_dim);
        P.cell_system_start.from_host(host_cell_system_start);
        P.box_size.from_host(host_box_size, host_box_size);
        P.e_interaction.from_host(host_e_interaction);
        P.rad.from_host(host_rad);
        P.mass.from_host(host_mass);
        P.rad.from_host(host_rad);
        // P.pos.enable_rng(0);
        // P.pos.stateless_rand_uniform(0.0, box_size, 0.0, box_size, 0);
        P.pos.from_host(host_pos_x, host_pos_y);  // copy the positions from the naive method

        P.sync_box();
        P.sync_system();
        P.sync_neighbors();
        P.sync_cells();
        P.sync_class_constants();
        P.init_neighbors();

        P.compute_forces();

        P.neighbor_ids.to_host(host_neighbor_ids);
        P.neighbor_start.to_host(host_neighbor_start);
        P.pos.to_host(host_pos_x, host_pos_y);
        P.force.to_host(host_force_x, host_force_y);
        P.pe.to_host(host_pe);
        for (int i = 0; i < N; i++) {
            bool is_isolated = host_pe[i] == 0.0;
            if (is_isolated) {
                assert(host_force_x[i] == 0.0);
                assert(host_force_y[i] == 0.0);
            }
            for (int j = host_neighbor_start[i]; j < host_neighbor_start[i + 1]; j++) {
                int neighbor_id = host_neighbor_ids[j];
                if (neighbor_id == i) continue;
                double dx = host_pos_x[neighbor_id] - host_pos_x[i];
                double dy = host_pos_y[neighbor_id] - host_pos_y[i];
                double r2 = dx * dx + dy * dy;
                double r = std::sqrt(r2);
                if (is_isolated) {
                    assert(r > 2.0 * rad);
                }
                if (r < 2.0 * rad) {
                    assert(std::abs(host_force_x[i]) > 0.0);
                    assert(std::abs(host_force_y[i]) > 0.0);
                    assert(std::abs(host_force_x[neighbor_id]) > 0.0);
                    assert(std::abs(host_force_y[neighbor_id]) > 0.0);
                    assert(host_pe[i] > 0.0);
                    assert(host_pe[neighbor_id] > 0.0);
                }
            }
        }
        // reorder the particles for comparison to the naive method
        auto zip_src = thrust::make_zip_iterator(thrust::make_tuple(
            P.force.x.begin(), P.force.y.begin(),
            P.pos.x.begin(), P.pos.y.begin()
        ));
        auto zip_dst = thrust::make_zip_iterator(thrust::make_tuple(
            P.force.x.swap_begin(), P.force.y.swap_begin(),
            P.pos.x.swap_begin(), P.pos.y.swap_begin()
        ));
        thrust::scatter(zip_src, zip_src + P.n_particles(), P.order.begin(), zip_dst);
        P.force.swap();
        P.pos.swap();
        P.force.to_host(host_force_x_cell, host_force_y_cell);
        P.pos.to_host(host_pos_x_cell, host_pos_y_cell);
        std::cout << "Cell neighbor test passed" << std::endl;
    }

    // check that the two neighbor lists give the same force result
    for (int i = 0; i < N; i++) {
        assert(host_pos_x_naive[i] == host_pos_x_cell[i]);
        assert(host_pos_y_naive[i] == host_pos_y_cell[i]);
        assert(feq(host_force_x_naive[i], host_force_x_cell[i]));
        assert(feq(host_force_y_naive[i], host_force_y_cell[i]));
    }
    std::cout << "Cell-Naive neighbor comparison passed" << std::endl;


    // energy conservation check

    // first generate some initial conditions
    std::vector<double> equil_pos_x(N), equil_pos_y(N);
    std::vector<double> init_vel_x(N), init_vel_y(N);
    {
        md::disk::Disk P;
        P.set_neighbor_method(md::NeighborMethod::Naive); // set this before allocating particles

        P.allocate_systems(S);
        P.allocate_particles(N);

        P.verlet_skin.from_host(host_skin);
        P.system_id.from_host(host_system_id);
        P.system_size.from_host(host_system_size);
        P.system_offset.from_host(host_system_start);
        P.cell_dim.from_host(host_cell_size_dim, host_cell_size_dim);
        P.cell_system_start.from_host(host_cell_system_start);
        P.box_size.from_host(host_box_size, host_box_size);
        P.e_interaction.from_host(host_e_interaction);
        P.rad.from_host(host_rad);
        P.mass.from_host(host_mass);
        P.rad.from_host(host_rad);

        P.pos.enable_rng(10);
        P.pos.stateless_rand_uniform(0.0, box_size, 0.0, box_size, 0);

        P.sync_box();
        P.sync_system();
        P.sync_neighbors();
        P.sync_cells();
        P.sync_class_constants();
        P.init_neighbors();

        double dt = 1e-2;

        // quickly equilibrate the system and get initial conditions
        for (int rep = 0; rep < 10; rep++) {
            for (int i = 0; i < 500; i++) {
                P.update_velocities(dt * 0.5);
                P.update_positions(dt);
                P.check_neighbors();
                P.compute_forces();
                P.compute_damping_forces(1e0);
                P.update_velocities(dt * 0.5);
            }
            P.force.fill(0.0, 0.0);
            P.vel.fill(0.0, 0.0);
        }
        P.pos.to_host(equil_pos_x, equil_pos_y);
        P.vel.stateless_rand_normal(-1.0, 1.0, -1.0, 1.0);
        P.vel.scale(1e-3, 1e-3);
        P.vel.to_host(init_vel_x, init_vel_y);
    }

    {  // Test Naive neighbor method
        std::cout << "Testing Naive neighbor energy conservation" << std::endl;
        md::disk::Disk P;
        P.set_neighbor_method(md::NeighborMethod::Naive); // set this before allocating particles

        P.allocate_systems(S);
        P.allocate_particles(N);

        P.verlet_skin.from_host(host_skin);
        P.system_id.from_host(host_system_id);
        P.system_size.from_host(host_system_size);
        P.system_offset.from_host(host_system_start);
        P.cell_dim.from_host(host_cell_size_dim, host_cell_size_dim);
        P.cell_system_start.from_host(host_cell_system_start);
        P.box_size.from_host(host_box_size, host_box_size);
        P.e_interaction.from_host(host_e_interaction);
        P.rad.from_host(host_rad);
        P.mass.from_host(host_mass);
        P.rad.from_host(host_rad);
        P.pos.from_host(equil_pos_x, equil_pos_y);
        P.vel.from_host(init_vel_x, init_vel_y);

        P.sync_box();
        P.sync_system();
        P.sync_neighbors();
        P.sync_cells();
        P.sync_class_constants();
        P.init_neighbors();

        std::vector<double> dts_to_test = {1e-1, 1e-2, 1e-3};
        std::vector<int> n_steps_to_test = {1000, 10000, 100000};
        std::vector<double> energy_fluctuations(dts_to_test.size());
        std::vector<double> relstd; relstd.reserve(dts_to_test.size());
        for (int i = 0; i < static_cast<int>(dts_to_test.size()); i++) {
            P.vel.from_host(init_vel_x, init_vel_y);
            P.pos.from_host(equil_pos_x, equil_pos_y);
            P.compute_forces();

            int n_steps = n_steps_to_test[i];
            double dt_test = dts_to_test[i];

            std::vector<double> ke_total(S);
            std::vector<double> pe_total(S);
            std::vector<double> te_hist(n_steps, 0.0);
            RelStd rs;

            for (int step = 0; step < n_steps; step++) {
                P.update_velocities(dt_test * 0.5);
                P.update_positions(dt_test);
                P.check_neighbors();
                P.compute_forces();
                P.update_velocities(dt_test * 0.5);

                // log the total energy of each system
                P.compute_ke_total();
                P.compute_pe_total();
                P.pe_total.to_host(pe_total);
                P.ke_total.to_host(ke_total);
                for (int s = 0; s < S; s++) {
                    te_hist[step] += pe_total[s] + ke_total[s];
                }
                rs.push(te_hist[step]);
            }
            double r = rs.rel();
            relstd.push_back(r);
            std::cout << "dt: " << dt_test << " energy fluctuations: " << r << std::endl;
        }
        double p = fit_loglog_slope_filtered(dts_to_test, relstd);
        std::cout << "p: " << p << std::endl;
        assert(std::abs(p - 2.0) < 1 / std::sqrt(N));
        std::cout << "Naive neighbor energy conservation test passed" << std::endl;
    }

    {  // Test Cell neighbor method
        std::cout << "Testing Cell neighbor energy conservation" << std::endl;
        md::disk::Disk P;
        P.set_neighbor_method(md::NeighborMethod::Cell); // set this before allocating particles

        P.allocate_systems(S);
        P.allocate_particles(N);

        P.verlet_skin.from_host(host_skin);
        P.system_id.from_host(host_system_id);
        P.system_size.from_host(host_system_size);
        P.system_offset.from_host(host_system_start);
        P.cell_dim.from_host(host_cell_size_dim, host_cell_size_dim);
        P.cell_system_start.from_host(host_cell_system_start);
        P.box_size.from_host(host_box_size, host_box_size);
        P.e_interaction.from_host(host_e_interaction);
        P.rad.from_host(host_rad);
        P.mass.from_host(host_mass);
        P.rad.from_host(host_rad);
        P.pos.from_host(equil_pos_x, equil_pos_y);
        P.vel.from_host(init_vel_x, init_vel_y);

        P.sync_box();
        P.sync_system();
        P.sync_neighbors();
        P.sync_cells();
        P.sync_class_constants();
        P.init_neighbors();

        std::vector<double> dts_to_test = {1e-1, 1e-2, 1e-3};
        std::vector<int> n_steps_to_test = {1000, 10000, 100000};
        std::vector<double> energy_fluctuations(dts_to_test.size());
        std::vector<double> relstd; relstd.reserve(dts_to_test.size());
        for (int i = 0; i < static_cast<int>(dts_to_test.size()); i++) {
            P.vel.from_host(init_vel_x, init_vel_y);
            P.pos.from_host(equil_pos_x, equil_pos_y);
            P.compute_forces();

            int n_steps = n_steps_to_test[i];
            double dt_test = dts_to_test[i];

            std::vector<double> ke_total(S);
            std::vector<double> pe_total(S);
            std::vector<double> te_hist(n_steps, 0.0);
            RelStd rs;

            for (int step = 0; step < n_steps; step++) {
                P.update_velocities(dt_test * 0.5);
                P.update_positions(dt_test);
                P.check_neighbors();
                P.compute_forces();
                P.update_velocities(dt_test * 0.5);

                // log the total energy of each system
                P.compute_ke_total();
                P.compute_pe_total();
                P.pe_total.to_host(pe_total);
                P.ke_total.to_host(ke_total);
                for (int s = 0; s < S; s++) {
                    te_hist[step] += pe_total[s] + ke_total[s];
                }
                rs.push(te_hist[step]);
            }
            double r = rs.rel();
            relstd.push_back(r);
            std::cout << "dt: " << dt_test << " energy fluctuations: " << r << std::endl;
        }
        double p = fit_loglog_slope_filtered(dts_to_test, relstd);
        std::cout << "p: " << p << std::endl;
        assert(std::abs(p - 2.0) < 1 / std::sqrt(N));
        std::cout << "Cell neighbor energy conservation test passed" << std::endl;
    }
    
}