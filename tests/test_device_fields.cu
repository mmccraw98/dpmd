// tests/test_device_fields.cu

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include "utils/cuda_debug.hpp"
#include "utils/device_fields.hpp"

// -----------------------------
// Helpers
// -----------------------------
template<typename T>
bool approx_eq(T a, T b, T rtol, T atol){ return std::fabs(a-b) <= (atol + rtol*std::fabs(b)); }

template<typename T>
bool vec_approx_eq(const std::vector<T>& A, const std::vector<T>& B, T rtol, T atol){
    if (A.size()!=B.size()) return false;
    for (size_t i=0;i<A.size();++i) if (!approx_eq(A[i],B[i],rtol,atol)) return false;
    return true;
}

template<typename T>
bool vec_exact_eq(const std::vector<T>& A, const std::vector<T>& B){
    return A.size()==B.size() && std::equal(A.begin(),A.end(),B.begin());
}

template<typename T>
bool df_approx_eq(const df::DeviceField1D<T>& A, const df::DeviceField1D<T>& B, T rtol, T atol){
    if (A.size()!=B.size()) return false;
    std::vector<T> A_host, B_host;
    A.to_host(A_host);
    B.to_host(B_host);
    return vec_approx_eq(A_host, B_host, rtol, atol);
}

template<typename T>
bool df_exact_eq(const df::DeviceField1D<T>& A, const df::DeviceField1D<T>& B){
    if (A.size()!=B.size()) return false;
    std::vector<T> A_host, B_host;
    A.to_host(A_host);
    B.to_host(B_host);
    return vec_exact_eq(A_host, B_host);
}

// -----------------------------
// Tiny test harness
// -----------------------------
struct Runner {
    int passed = 0;
    int failed = 0;

    void expect(bool cond, const std::string& label, const std::string& why_on_fail){
        if (cond) {
            ++passed;
        } else {
            ++failed;
            std::cerr << "FAIL [" << label << "]: " << why_on_fail << "\n";
        }
    }
    void summary() const {
        std::cout << "==== Test Summary ====\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";
        std::cout << (failed==0 ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    }
};

// -----------------------------
// Simple stats on host
// -----------------------------
template<typename T>
T mean_of(const std::vector<T>& v){
    if (v.empty()) return T(0);
    long double s = 0.0;
    for (auto x : v) s += static_cast<long double>(x);
    return static_cast<T>(s / v.size());
}
template<typename T>
T var_of(const std::vector<T>& v, T mu){
    if (v.size() < 2) return T(0);
    long double s = 0.0;
    for (auto x : v) {
        long double d = static_cast<long double>(x) - static_cast<long double>(mu);
        s += d*d;
    }
    return static_cast<T>(s / (v.size()-1));
}
template<typename T>
T corr_xy(const std::vector<T>& x, const std::vector<T>& y){
    size_t n = std::min(x.size(), y.size());
    if (n < 2) return T(0);
    long double mx = 0.0, my = 0.0;
    for (size_t i=0;i<n;++i){ mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    long double num=0.0, dx2=0.0, dy2=0.0;
    for (size_t i=0;i<n;++i){
        long double dx = x[i]-mx;
        long double dy = y[i]-my;
        num += dx*dy; dx2 += dx*dx; dy2 += dy*dy;
    }
    if (dx2==0.0 || dy2==0.0) return 0;
    return static_cast<T>( num / std::sqrt(dx2*dy2) );
}

// -----------------------------
// Tests
// -----------------------------
void test_basic_construct_resize_swap(Runner& R){
    const int N0 = 1000, N1 = 1500;
    df::DeviceField1D<double> a(N0, /*swap*/true, /*rng*/false, /*seed*/0);

    R.expect(a.size()==N0, "construct/size(1D)", "initial size mismatch");
    R.expect(a.swap_enabled(), "construct/swap_enabled", "swap should be enabled after ctor");
    R.expect(static_cast<int>(a.swap_buffer.size())==N0, "construct/swap_buffer_alloc", "swap buffer not allocated to parity");

    a.resize(N1);
    R.expect(a.size()==N1, "resize/data(1D)", "resize did not update data size");
    R.expect(static_cast<int>(a.swap_buffer.size())==N1, "resize/swap_buffer(1D)", "swap buffer not resized to parity");

    a.disable_swap();
    R.expect(!a.swap_enabled(), "disable_swap", "swap flag did not disable");
    R.expect(a.swap_buffer.size()==0, "disable_swap/free", "swap buffer not freed");
}

void test_fill_scale_and_swap(Runner& R){
    const int N = 2048;
    df::DeviceField1D<float> a(N, /*swap*/true, /*rng*/false, 0);
    a.fill(3.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host;
    a.to_host(host);
    bool okfill = std::all_of(host.begin(), host.end(), [](float v){ return v==3.0f; });
    R.expect(okfill, "fill(1D)", "fill did not set all elements to 3.0f");

    // Put 5.0f into swap buffer, then swap
    thrust::fill(a.swap_buffer.begin(), a.swap_buffer.end(), 5.0f);
    a.swap();
    CUDA_CHECK(cudaDeviceSynchronize());
    a.to_host(host);
    bool okswap = std::all_of(host.begin(), host.end(), [](float v){ return v==5.0f; });
    R.expect(okswap, "swap(1D)", "swap did not swap content with swap_buffer");

    a.scale(0.5f);
    CUDA_CHECK(cudaDeviceSynchronize());
    a.to_host(host);
    bool okscale = std::all_of(host.begin(), host.end(), [](float v){ return v==2.5f; });
    R.expect(okscale, "scale(1D)", "scale did not multiply by 0.5");
}

void test_copy_from_and_clear(Runner& R){
    const int N = 4096;
    df::DeviceField1D<int> a(N, /*swap*/true, /*rng*/false, 0);
    df::DeviceField1D<int> b(N, /*swap*/false, /*rng*/false, 0);

    // init a with i
    thrust::sequence(a.data.begin(), a.data.end(), 0);
    b.copy_from(a);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> ha, hb;
    a.to_host(ha); b.to_host(hb);
    R.expect(vec_exact_eq(ha,hb), "copy_from(1D)", "data content mismatch after copy_from");

    a.clear();
    R.expect(a.size()==0, "clear/size(1D)", "clear did not reset size to 0");
    R.expect(!a.swap_enabled(), "clear/swap_flag(1D)", "clear did not disable swap flag");
}

void test_get_element_sync_async(Runner& R){
    const int N = 1024;
    df::DeviceField1D<double> a(N, /*swap*/false, /*rng*/false, 0);
    // a[i] = i + 0.5
    std::vector<double> h(N);
    for (int i=0;i<N;++i) h[i] = static_cast<double>(i) + 0.5;
    a.from_host(h);

    // sync
    double v10 = a.get_element(10);
    R.expect(v10 == 10.5, "get_element", "value mismatch at i=10");

    // async
    cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
    double v777 = a.get_element_async(777, s);
    CUDA_CHECK(cudaStreamDestroy(s));
    R.expect(v777 == 777.5, "get_element_async", "value mismatch at i=777");
}

void test_stateful_rng_1d_uniform_double(Runner& R){
    const int N = 200000;
    const double lo=0.0, hi=1.0, target_mean=0.5;
    df::DeviceField1D<double> a(N, /*swap*/false, /*rng*/false, 0);
    df::DeviceField1D<double> b(N, /*swap*/false, /*rng*/false, 0);

    const unsigned long long seed = 123456789ull;
    a.enable_rng(seed);
    b.enable_rng(seed);

    a.rand_uniform(lo,hi);
    b.rand_uniform(lo,hi);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Determinism across identically seeded runs
    R.expect(df_exact_eq(a,b), "rng1d_uniform_double/determinism", "identical seeds produced different sequences");

    // Range and mean sanity
    std::vector<double> h; a.to_host(h);
    bool in_range = std::all_of(h.begin(), h.end(), [&](double x){ return x>=lo && x<hi; });
    R.expect(in_range, "rng1d_uniform_double/range", "values escaped [lo,hi)");

    double mu = mean_of(h);
    double se = std::sqrt(1.0/12.0 / N); // std error of mean for U(0,1)
    R.expect(std::fabs(mu - target_mean) < 5.0*se, "rng1d_uniform_double/mean", "sample mean too far from 0.5");
}

void test_stateful_rng_1d_normal_float(Runner& R){
    const int N = 200000;
    const float mean=2.0f, sigma=3.0f;
    df::DeviceField1D<float> a(N, /*swap*/false, /*rng*/false, 0);
    a.enable_rng(987654321ull);

    a.rand_normal(mean, sigma);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h; a.to_host(h);
    float mu = mean_of(h);
    float v  = var_of(h, mu);

    // SE of mean ~ sigma/sqrt(N)
    float se_mu = sigma/std::sqrt(static_cast<float>(N));
    // Relative tolerance on variance ~ few %
    R.expect(std::fabs(mu - mean) < 5.0f*se_mu, "rng1d_normal_float/mean", "sample mean far from target");
    R.expect(std::fabs(v - sigma*sigma) < 0.1f*(sigma*sigma), "rng1d_normal_float/var", "sample variance far from sigma^2");
}

void test_stateful_rng_2d_uniform_and_normal(Runner& R){
    const int N = 200000;
    df::DeviceField2D<double> P(N, /*swap*/false, /*rng*/true, /*seed*/42ull);

    // Uniform
    P.rand_uniform(-1.0, +1.0, 2.0, 3.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> hx, hy; P.to_host(hx, hy);
    bool inx = std::all_of(hx.begin(), hx.end(), [](double x){ return x>=-1.0 && x<1.0; });
    bool iny = std::all_of(hy.begin(), hy.end(), [](double y){ return y>= 2.0 && y<3.0; });
    R.expect(inx && iny, "rng2d_uniform/range", "uniform 2D values escaped their ranges");

    // Weak independence check
    double rho_u = corr_xy(hx, hy);
    R.expect(std::fabs(rho_u) < 0.02, "rng2d_uniform/corr", "x/y appear correlated beyond tolerance");

    // Normal
    P.rand_normal(1.0, 2.0, -3.0, 4.0);
    CUDA_CHECK(cudaDeviceSynchronize());
    P.to_host(hx, hy);

    double mx = mean_of(hx), my = mean_of(hy);
    R.expect(std::fabs(mx - 1.0) < 5.0*(2.0/std::sqrt(N)), "rng2d_normal/mean_x", "x mean far from 1.0");
    R.expect(std::fabs(my + 3.0) < 5.0*(4.0/std::sqrt(N)), "rng2d_normal/mean_y", "y mean far from -3.0");

    double vx = var_of(hx,mx), vy = var_of(hy,my);
    R.expect(std::fabs(vx - 4.0) < 0.1*4.0, "rng2d_normal/var_x", "x variance far from 4");
    R.expect(std::fabs(vy - 16.0) < 0.1*16.0, "rng2d_normal/var_y", "y variance far from 16");

    double rho_n = corr_xy(hx, hy);
    R.expect(std::fabs(rho_n) < 0.02, "rng2d_normal/corr", "x/y normals appear correlated beyond tolerance");
}
void test_stateless_explicit_vs_pseudo_1d(Runner& R){
    const int N = 100000;
    const unsigned long long seed = 5555ull;

    // A: generate two sequences using the pseudo-stateless counter (auto++)
    df::DeviceField1D<float> A(N);
    A.enable_rng(seed);               // sets _rng_seed; curand states are unused here
    A.stateless_rand_uniform(0.0f, 1.0f);      // uses call_counter = 0
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> A0; A.to_host(A0);

    A.stateless_rand_uniform(0.0f, 1.0f);      // uses call_counter = 1
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> A1; A.to_host(A1);

    // B: generate the same two sequences but with EXPLICIT counts 0 and 1
    df::DeviceField1D<float> B(N);
    B.enable_rng(seed);
    B.stateless_rand_uniform(0.0f, 1.0f, /*rng_count*/0ull);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> B0; B.to_host(B0);

    B.stateless_rand_uniform(0.0f, 1.0f, /*rng_count*/1ull);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> B1; B.to_host(B1);

    // Checks
    R.expect(vec_exact_eq(A0,B0), "stateless_1d_uniform/explicit0_eq_auto0",
             "explicit count=0 did not match pseudo (auto++) count=0");
    R.expect(vec_exact_eq(A1,B1), "stateless_1d_uniform/explicit1_eq_auto1",
             "explicit count=1 did not match pseudo (auto++) count=1");

    // Different counts must differ: check at least one element differs
    bool all_equal = vec_exact_eq(A0, A1);
    R.expect(!all_equal, "stateless_1d_uniform/different_counts_differ",
             "count 0 and count 1 produced identical sequences");
}

void test_stateless_explicit_vs_pseudo_2d(Runner& R){
    const int N = 100000;
    const unsigned long long seed = 99999ull;

    // P: pseudo-stateless (auto++) normal draws
    df::DeviceField2D<double> P(N);
    P.enable_rng(seed);

    P.stateless_rand_normal(1.0, 2.0, -1.0, 3.0);     // call_counter = 0
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> x0,y0; P.to_host(x0,y0);

    P.stateless_rand_normal(1.0, 2.0, -1.0, 3.0);     // call_counter = 1
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> x1,y1; P.to_host(x1,y1);

    // Q: explicit counts 0, then 1 with the same seed
    df::DeviceField2D<double> Q(N);
    Q.enable_rng(seed);

    Q.stateless_rand_normal(1.0, 2.0, -1.0, 3.0, /*rng_count*/0ull);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> u0,v0; Q.to_host(u0,v0);

    Q.stateless_rand_normal(1.0, 2.0, -1.0, 3.0, /*rng_count*/1ull);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> u1,v1; Q.to_host(u1,v1);

    // Checks
    R.expect(vec_exact_eq(x0,u0) && vec_exact_eq(y0,v0),
             "stateless_2d_normal/explicit0_eq_auto0",
             "explicit count=0 did not match pseudo (auto++) count=0");
    R.expect(vec_exact_eq(x1,u1) && vec_exact_eq(y1,v1),
             "stateless_2d_normal/explicit1_eq_auto1",
             "explicit count=1 did not match pseudo (auto++) count=1");

    // Different counts must differ on at least one axis
    bool same_x = vec_exact_eq(x0,x1);
    bool same_y = vec_exact_eq(y0,y1);
    R.expect(!(same_x && same_y), "stateless_2d_normal/different_counts_differ",
             "count 0 and 1 produced identical sequences on both axes");
}

void test_resize_rng_parity(Runner& R){
    const int N0=1000, N1=3000;
    df::DeviceField1D<double> a(N0, /*swap*/false, /*rng*/false, 0);
    a.enable_rng(123ull);
    R.expect(static_cast<int>(a.rng_states.size())==N0, "rng_parity/after_enable", "rng_states not allocated to parity");

    a.resize(N1);
    R.expect(static_cast<int>(a.rng_states.size())==N1, "rng_parity/after_resize", "rng_states not resized to parity");
}

void test_zero_length_safety(Runner& R){
    df::DeviceField1D<float> a(0);
    df::DeviceField2D<double> P(0);
    // should no-op without error
    a.fill(1.0f);
    a.scale(2.0f);
    a.enable_rng(1ull);
    a.rand_uniform(0.0f, 1.0f);
    a.rand_normal(0.0f, 1.0f);
    a.stateless_rand_uniform(0.0f, 1.0f, 0ull);
    a.stateless_rand_normal(0.0f, 1.0f, 1ull);
    a.clear();

    P.enable_rng(2ull);
    P.rand_uniform(-1.0,1.0,-2.0,2.0);
    P.rand_normal(0.0,1.0,0.0,1.0);
    P.stateless_rand_uniform(-1.0,1.0,-2.0,2.0, 0ull);
    P.stateless_rand_normal(0.0,1.0,0.0,1.0, 1ull);
    P.clear();

    R.expect(true, "zero_length/no_crash", "zero-length operations threw or crashed");
}

void test_reorder_by_throws(Runner& R){
    df::DeviceField1D<int> a(10);
    std::vector<int> idx(10,0);
    bool threw = false;
    try {
        a.reorder_by(idx);
    } catch (const std::logic_error&) {
        threw = true;
    }
    R.expect(threw, "reorder_by/throws", "reorder_by did not throw logic_error as expected");
}

void test_integer_fill_scale(Runner& R){
    const int N=1024;
    df::DeviceField1D<long> a(N);
    a.fill(7);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<long> h; a.to_host(h);
    bool ok7 = std::all_of(h.begin(), h.end(), [](long v){ return v==7; });
    R.expect(ok7, "int_fill", "fill(7) failed for integer field");

    a.scale(3);
    CUDA_CHECK(cudaDeviceSynchronize());
    a.to_host(h);
    bool ok21 = std::all_of(h.begin(), h.end(), [](long v){ return v==21; });
    R.expect(ok21, "int_scale", "scale(3) failed for integer field");
}

// -----------------------------
// main
// -----------------------------
int main(){
    Runner R;
    // Optional: choose device explicitly
    // CUDA_CHECK(cudaSetDevice(0));

    test_basic_construct_resize_swap(R);
    test_fill_scale_and_swap(R);
    test_copy_from_and_clear(R);
    test_get_element_sync_async(R);

    test_stateful_rng_1d_uniform_double(R);
    test_stateful_rng_1d_normal_float(R);
    test_stateful_rng_2d_uniform_and_normal(R);

    test_stateless_explicit_vs_pseudo_1d(R);
    test_stateless_explicit_vs_pseudo_2d(R);

    test_resize_rng_parity(R);
    test_zero_length_safety(R);
    test_reorder_by_throws(R);
    test_integer_fill_scale(R);

    R.summary();
    return (R.failed==0) ? 0 : 1;
}