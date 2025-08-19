#pragma once
// include/utils/device_fields.cuh

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

#include <curand_kernel.h>
#include "utils/cuda_debug.hpp"
#include "kernels/launch_config.cuh"

#include <vector>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <cstdint>
#include <climits>
#include <cmath>

namespace df {


// ===============================
// Low-level device kernels
// ===============================
template <class T>
__global__ void k_fill(T* __restrict__ a, int N, T v){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) a[i] = v;
}

template <class T>
__global__ void k_scale(T* __restrict__ a, int N, T alpha){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) a[i] *= alpha;
}

// Header-safe (non-templated) RNG state init kernel.
static __global__ void k_init_rng_states(curandStatePhilox4_32_10_t* __restrict__ states,
                                         int N,
                                         unsigned long long seed,
                                         unsigned long long subseq_offset = 0ULL) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    curand_init(seed,
                static_cast<unsigned long long>(i) + subseq_offset,
                0ULL,
                &states[i]);
}

// 1D: pre-initialized uniform
template <class T>
__global__ void k_rand_uniform_1d(T* __restrict__ arr, int N,
                                  curandStatePhilox4_32_10_t* __restrict__ states,
                                  T lo, T hi){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandStatePhilox4_32_10_t st = states[i];

    if constexpr (std::is_same<T,float>::value) {
        float u = curand_uniform(&st); if (u == 1.0f) u = 0.0f;
        arr[i] = lo + (hi - lo) * u;
    } else {
        double u = curand_uniform_double(&st); if (u == 1.0) u = 0.0;
        arr[i] = lo + (hi - lo) * static_cast<T>(u);
    }

    states[i] = st; // write-back
}

// 1D: pre-initialized normal
template <class T>
__global__ void k_rand_normal_1d(T* __restrict__ arr, int N,
                                 curandStatePhilox4_32_10_t* __restrict__ states,
                                 T mean, T sigma){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandStatePhilox4_32_10_t st = states[i];

    if constexpr (std::is_same<T,float>::value) {
        float z = curand_normal(&st);
        arr[i] = mean + sigma * static_cast<T>(z);
    } else {
        double z = curand_normal_double(&st);
        arr[i] = mean + sigma * static_cast<T>(z);
    }

    states[i] = st; // write-back
}

// 2D: fused uniform using a SINGLE state array (one state per particle)
template <class T>
__global__ void k_rand_uniform_2d_one_state(T* __restrict__ arr_x,
                                            T* __restrict__ arr_y,
                                            int N,
                                            curandStatePhilox4_32_10_t* __restrict__ states,
                                            T lo_x, T hi_x, T lo_y, T hi_y){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandStatePhilox4_32_10_t st = states[i];

    if constexpr (std::is_same<T,float>::value) {
        // Vectorized draw amortizes state advancement
        float4 r = curand_uniform4(&st);
        if (r.x == 1.0f) r.x = 0.0f;
        if (r.y == 1.0f) r.y = 0.0f;
        arr_x[i] = lo_x + (hi_x - lo_x) * r.x;
        arr_y[i] = lo_y + (hi_y - lo_y) * r.y;
    } else {
        // No uniform4 for doubles; draw twice
        double ux = curand_uniform_double(&st); if (ux == 1.0) ux = 0.0;
        double uy = curand_uniform_double(&st); if (uy == 1.0) uy = 0.0;
        arr_x[i] = lo_x + (hi_x - lo_x) * static_cast<T>(ux);
        arr_y[i] = lo_y + (hi_y - lo_y) * static_cast<T>(uy);
    }

    states[i] = st;
}

// 2D: fused normal using a SINGLE state array (one state per particle)
template <class T>
__global__ void k_rand_normal_2d_one_state(T* __restrict__ arr_x,
                                           T* __restrict__ arr_y,
                                           int N,
                                           curandStatePhilox4_32_10_t* __restrict__ states,
                                           T mean_x, T sigma_x, T mean_y, T sigma_y){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandStatePhilox4_32_10_t st = states[i];

    if constexpr (std::is_same<T,float>::value) {
        float4 r = curand_normal4(&st);
        arr_x[i] = mean_x + sigma_x * static_cast<T>(r.x);
        arr_y[i] = mean_y + sigma_y * static_cast<T>(r.y);
    } else {
        // curand_normal2_double returns two doubles efficiently
        double2 r = curand_normal2_double(&st);
        arr_x[i] = mean_x + sigma_x * static_cast<T>(r.x);
        arr_y[i] = mean_y + sigma_y * static_cast<T>(r.y);
    }

    states[i] = st;
}

// Minimal Philox4x32-10 (Random123-style)
__device__ __forceinline__ uint32_t mulhi32(uint32_t a, uint32_t b) {
    return __umulhi(a, b);
}
struct PhiloxKey32 { uint32_t k0, k1; };
struct PhiloxCtr4  { uint32_t c0, c1, c2, c3; };

__device__ __forceinline__ PhiloxCtr4 philox_round(PhiloxCtr4 c, PhiloxKey32 k) {
    constexpr uint32_t M0 = 0xD2511F53u;
    constexpr uint32_t M1 = 0xCD9E8D57u;

    uint32_t hi0 = mulhi32(M0, c.c0), lo0 = M0 * c.c0;
    uint32_t hi1 = mulhi32(M1, c.c2), lo1 = M1 * c.c2;

    PhiloxCtr4 r;
    r.c0 = hi1 ^ c.c1 ^ k.k0;
    r.c1 = lo1;
    r.c2 = hi0 ^ c.c3 ^ k.k1;
    r.c3 = lo0;
    return r;
}

__device__ __forceinline__ PhiloxKey32 philox_bump(PhiloxKey32 k) {
    constexpr uint32_t W0 = 0x9E3779B9u;
    constexpr uint32_t W1 = 0xBB67AE85u;
    k.k0 += W0; k.k1 += W1; return k;
}

__device__ __forceinline__ PhiloxCtr4 philox10(PhiloxCtr4 c, PhiloxKey32 k) {
#pragma unroll
    for (int r = 0; r < 10; ++r) { c = philox_round(c, k); k = philox_bump(k); }
    return c;
}

__device__ __forceinline__ PhiloxCtr4 make_ctr(uint32_t idx, uint32_t call, uint32_t axis=0u) {
    PhiloxCtr4 c{idx, call, axis, 0u};
    return c;
}
__device__ __forceinline__ PhiloxKey32 make_key(uint64_t seed) {
    PhiloxKey32 k{
        static_cast<uint32_t>(seed & 0xFFFFFFFFull),
        static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFFull)
    };
    return k;
}

// Map to (0,1)
__device__ __forceinline__ float u01f(uint32_t x) {
    constexpr float SCALE = 1.0f / 4294967296.0f;
    return (static_cast<float>(x) + 0.5f) * SCALE;
}

// Map to (0,1)
__device__ __forceinline__ double u01d(uint32_t hi, uint32_t lo) {
    uint64_t bits = (static_cast<uint64_t>(hi) << 21) | (static_cast<uint64_t>(lo) >> 11);
    constexpr double SCALE = 1.0 / 9007199254740992.0; // 2^53
    return (static_cast<double>(bits) + 0.5) * SCALE;
}

// Box–Muller
__device__ __forceinline__ void box_muller(double u0, double u1, double& z0, double& z1) {
    double r = sqrt(-2.0 * log(u0));
    double th = 6.28318530717958647692 * u1; // 2*pi
    z0 = r * cos(th);
    z1 = r * sin(th);
}

// 1D stateless uniform (explicit call counter)
template <class T>
__global__ void k_stateless_uniform_1d(T* __restrict__ arr, int N,
                                       uint64_t seed, uint64_t call_counter,
                                       T lo, T hi){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    PhiloxCtr4 ctr = make_ctr(static_cast<uint32_t>(i), static_cast<uint32_t>(call_counter), 0u);
    PhiloxKey32 key = make_key(seed);
    PhiloxCtr4 out = philox10(ctr, key);

    if constexpr (std::is_same<T,float>::value) {
        float u = u01f(out.c0); if (u == 1.0f) u = 0.0f;
        arr[i] = lo + (hi - lo) * u;
    } else {
        double u = u01d(out.c0, out.c1); if (u == 1.0) u = 0.0;
        arr[i] = lo + (hi - lo) * static_cast<T>(u);
    }
}

// 1D stateless normal (explicit call counter)
template <class T>
__global__ void k_stateless_normal_1d(T* __restrict__ arr, int N,
                                      uint64_t seed, uint64_t call_counter,
                                      T mean, T sigma){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    PhiloxCtr4 ctr = make_ctr(static_cast<uint32_t>(i), static_cast<uint32_t>(call_counter), 0u);
    PhiloxKey32 key = make_key(seed);
    PhiloxCtr4 out = philox10(ctr, key);

    double u0 = u01d(out.c0, out.c1);
    double u1 = u01d(out.c2, out.c3);
    double z0, z1; box_muller(u0, u1, z0, z1);
    arr[i] = mean + sigma * static_cast<T>(z0);
}

// 2D stateless fused uniform (explicit call counter)
template <class T>
__global__ void k_stateless_uniform_2d(T* __restrict__ arr_x,
                                       T* __restrict__ arr_y,
                                       int N,
                                       uint64_t seed, uint64_t call_counter,
                                       T lo_x, T hi_x, T lo_y, T hi_y){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    PhiloxCtr4 ctr = make_ctr(static_cast<uint32_t>(i), static_cast<uint32_t>(call_counter), 0u);
    PhiloxKey32 key = make_key(seed);
    PhiloxCtr4 out = philox10(ctr, key);

    if constexpr (std::is_same<T,float>::value) {
        float ux = u01f(out.c0); if (ux == 1.0f) ux = 0.0f;
        float uy = u01f(out.c1); if (uy == 1.0f) uy = 0.0f;
        arr_x[i] = lo_x + (hi_x - lo_x) * ux;
        arr_y[i] = lo_y + (hi_y - lo_y) * uy;
    } else {
        double ux = u01d(out.c0, out.c1); if (ux == 1.0) ux = 0.0;
        double uy = u01d(out.c2, out.c3); if (uy == 1.0) uy = 0.0;
        arr_x[i] = lo_x + (hi_x - lo_x) * static_cast<T>(ux);
        arr_y[i] = lo_y + (hi_y - lo_y) * static_cast<T>(uy);
    }
}

// 2D stateless fused normal (explicit call counter)
template <class T>
__global__ void k_stateless_normal_2d(T* __restrict__ arr_x,
                                      T* __restrict__ arr_y,
                                      int N,
                                      uint64_t seed, uint64_t call_counter,
                                      T mean_x, T sigma_x, T mean_y, T sigma_y){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    PhiloxCtr4 ctr = make_ctr(static_cast<uint32_t>(i), static_cast<uint32_t>(call_counter), 0u);
    PhiloxKey32 key = make_key(seed);
    PhiloxCtr4 out = philox10(ctr, key);

    double u0 = u01d(out.c0, out.c1);
    double u1 = u01d(out.c2, out.c3);
    double z0, z1; box_muller(u0, u1, z0, z1);
    arr_x[i] = mean_x + sigma_x * static_cast<T>(z0);
    arr_y[i] = mean_y + sigma_y * static_cast<T>(z1);
}

// ===============================
// DeviceField1D<T>
// ===============================
template<class T>
struct DeviceField1D {
    // Data array
    thrust::device_vector<T> data;
    // Swap buffer array
    thrust::device_vector<T> swap_buffer;       // used for reorder/swap
    // RNG state array
    thrust::device_vector<curandStatePhilox4_32_10_t> rng_states;  // used for RNG
    
    DeviceField1D() = default;
    explicit DeviceField1D(int N, bool swap=false, bool rng=false, unsigned long long seed=0ULL)
    : data(static_cast<size_t>(N)), _enable_swap(swap), _enable_rng(rng), _rng_seed(seed) {
        if (_enable_swap) swap_buffer.resize(static_cast<size_t>(N));
        if (_enable_rng) _init_rng(_rng_seed, 0ULL);
    }

    // Check if swap is enabled
    bool swap_enabled() const { return _enable_swap; }

    // Check if RNG is enabled
    bool rng_enabled()  const { return _enable_rng; }

    // Get the RNG seed
    unsigned long long rng_seed() const { return _rng_seed; }

    // Get the size of the data
    int size() const { return static_cast<int>(data.size()); }

    // Resize the data array
    void resize(int N) {
        if (N == size()) return;
        data.resize(static_cast<size_t>(N));
        if (_enable_swap) swap_buffer.resize(static_cast<size_t>(N));
        if (_enable_rng) _init_rng(_rng_seed, 0ULL);
    }

    // Clear the data array and aux arrays
    void clear() {
        thrust::device_vector<T>().swap(data);
        if (_enable_swap) { thrust::device_vector<T>().swap(swap_buffer); _enable_swap = false; }
        if (_enable_rng) { thrust::device_vector<curandStatePhilox4_32_10_t>().swap(rng_states); _enable_rng = false; _rng_seed = 0ULL; }
    }

    T*       ptr()       { return data.data().get(); }
    T const* ptr() const { return data.data().get(); }
    T*       ptr_swap()       { return _enable_swap ? swap_buffer.data().get() : nullptr; }
    T const* ptr_swap() const { return _enable_swap ? swap_buffer.data().get() : nullptr; }
    curandStatePhilox4_32_10_t* ptr_rng()       { return _enable_rng ? rng_states.data().get() : nullptr; }
    curandStatePhilox4_32_10_t const* ptr_rng() const { return _enable_rng ? rng_states.data().get() : nullptr; }
    auto begin() { return data.begin(); }
    auto end()   { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end()   const { return data.end(); }
    auto swap_begin() { return swap_buffer.begin(); }
    auto swap_end()   { return swap_buffer.end(); }
    auto swap_begin() const { return swap_buffer.begin(); }
    auto swap_end()   const { return swap_buffer.end(); }
    auto rng_begin() { return rng_states.begin(); }
    auto rng_end()   { return rng_states.end(); }
    auto rng_begin() const { return rng_states.begin(); }
    auto rng_end()   const { return rng_states.end(); }
    
    // Set element i to value v
    void set_element(std::size_t i, T v);

    // Fetch element i from device to host
    T get_element(std::size_t i) const;

    // Fetch element i from device to host asynchronously
    T get_element_async(std::size_t i, cudaStream_t stream) const;

    // Enable swap, allocate aux memory
    void enable_swap() {
        _enable_swap = true;
        const size_t need = static_cast<size_t>(size());
        if (swap_buffer.size() != need) swap_buffer.resize(need);
    }

    // Disable swap, free aux memory
    void disable_swap() {
        _enable_swap = false;
        thrust::device_vector<T>().swap(swap_buffer);
    }

    // Enable statefull RNG, initialize RNG state array
    void enable_rng(unsigned long long seed = 0ULL) {
        _enable_rng = true;
        _rng_seed = seed;
        _init_rng(_rng_seed, 0ULL);
    }

    // Disable statefull RNG, free RNG state array memory
    void disable_rng() {
        _enable_rng = false;
        _rng_seed = 0ULL;
        thrust::device_vector<curandStatePhilox4_32_10_t>().swap(rng_states);
    }

    // Swap the data in the buffer array and the main array - useful for reordering
    void swap() {
        if (!_enable_swap) throw std::runtime_error("DeviceField1D::swap: enable_swap==false");
        thrust::swap(data, swap_buffer);
    }

    // Copy data from another DeviceField1D
    void copy_from(const DeviceField1D<T>& other) {
        if (other.size() != size()) throw std::runtime_error("DeviceField1D::copy_from: size mismatch");
        thrust::copy(other.data.begin(), other.data.end(), data.begin());
    }

    // Reorder the data by index vector
    template<class IndexVec>
    void reorder_by(const IndexVec& /*idx*/) {
        throw std::logic_error("DeviceField1D::reorder_by: not implemented");
    }

    // Copy data from device to host
    void to_host(std::vector<T>& out) const {
        if (out.size() != static_cast<size_t>(size())) out.resize(static_cast<size_t>(size()));
        thrust::copy(data.begin(), data.end(), out.begin());
    }

    // Copy data from host to device
    void from_host(const std::vector<T>& in) {
        if (in.size() != static_cast<size_t>(size())) resize(static_cast<int>(in.size()));
        data.assign(in.begin(), in.end());
    }

    // Fill the data with a set value
    void fill(T v);

    // Scale the data with a set value
    void scale(T alpha);

    // Statefull RNG (uniform) - requires rng_enabled() = true
    void rand_uniform(T lo, T hi);

    // Statefull RNG (normal) - requires rng_enabled() = true
    void rand_normal(T mean, T sigma);

    // Stateless RNG (uniform)
    void stateless_rand_uniform(T lo, T hi, unsigned long long rng_count);

    // Pseudo-stateless RNG (uniform)
    void stateless_rand_uniform(T lo, T hi);

    // Stateless RNG (normal)
    void stateless_rand_normal(T mean, T sigma, unsigned long long rng_count);

    // Pseudo-stateless RNG (normal)
    void stateless_rand_normal(T mean, T sigma);

private:
    bool _enable_swap = false;
    bool _enable_rng = false;
    unsigned long long _rng_seed = 0ULL;
    unsigned long long _rng_count = 0ULL;

    // Reinitialize RNG with given seed and optional subsequence offset (for disjoint streams).
    void _init_rng(unsigned long long seed = 0ULL, unsigned long long subseq_offset = 0ULL);
};

// ===============================
// DeviceField2D<T>   (SoA x/y) with SINGLE RNG state array
// ===============================
template<class T>
struct DeviceField2D {
    // Single RNG state array shared across x and y (one state per particle)
    thrust::device_vector<curandStatePhilox4_32_10_t> rng_states;

    // Two DeviceField1D objects for x and y axes
    DeviceField1D<T> x, y;

    DeviceField2D() = default;
    explicit DeviceField2D(int N, bool swap=false, bool rng=false, unsigned long long seed=0ULL)
    : x(N,swap,false,0ULL), y(N,swap,false,0ULL), _enable_rng(rng), _rng_seed(seed) {
        if (_enable_rng) _init_rng(_rng_seed);
    }

    // Resize the data for both axes
    void resize(int N) {
        x.resize(N);
        y.resize(N);
        if (_enable_rng) _init_rng(_rng_seed);
    }

    // Drop memory footprint of data and aux (size->0) for both axes and RNG state
    void clear() {
        x.clear();
        y.clear();
        if (_enable_rng) { thrust::device_vector<curandStatePhilox4_32_10_t>().swap(rng_states); _enable_rng = false; _rng_seed = 0ULL; }
    }

    // Swap the data in the buffer array and the main array - useful for reordering
    void swap() { x.swap(); y.swap(); }
    
    // Get the size of the data
    int size() const { return x.size(); }

    // Check if swap is enabled for both axes
    bool swap_enabled() const { return x.swap_enabled() && y.swap_enabled(); }

    // Check if RNG is enabled
    bool rng_enabled() const { return _enable_rng; }

    // Get the RNG seed
    unsigned long long rng_seed() const { return _rng_seed; }


    T*       xptr()       { return x.ptr(); }
    T*       yptr()       { return y.ptr(); }
    T const* xptr() const { return x.ptr(); }
    T const* yptr() const { return y.ptr(); }
    T*       xptr_swap()       { return x.ptr_swap(); }
    T*       yptr_swap()       { return y.ptr_swap(); }
    T const* xptr_swap() const { return x.ptr_swap(); }
    T const* yptr_swap() const { return y.ptr_swap(); }
    curandStatePhilox4_32_10_t* ptr_rng2d()       { return _enable_rng ? rng_states.data().get() : nullptr; }
    curandStatePhilox4_32_10_t const* ptr_rng2d() const { return _enable_rng ? rng_states.data().get() : nullptr; }
    auto rng_begin() { return rng_states.begin(); }
    auto rng_end()   { return rng_states.end(); }
    auto rng_begin() const { return rng_states.begin(); }
    auto rng_end()   const { return rng_states.end(); }

    // Reorder data by index vector
    template<class IndexVec>
    void reorder_by(const IndexVec& idx) { x.reorder_by(idx); y.reorder_by(idx); }

    // Copy data from another DeviceField2D
    void copy_from(const DeviceField2D<T>& other) { x.copy_from(other.x); y.copy_from(other.y); }

    // Set element i to value v
    void set_element(std::size_t i, T vx, T vy) {
        x.set_element(i, vx);
        y.set_element(i, vy);
    }

    // Fetch element i from device to host
    std::pair<T,T> get_element(std::size_t i) const {
        return { x.get_element(i), y.get_element(i) };
    }

    // Fetch element i from device to host asynchronously
    std::pair<T,T> get_element_async(std::size_t i, cudaStream_t stream) const {
        return { x.get_element_async(i, stream), y.get_element_async(i, stream) };
    }

    // Copy data from device to host
    void to_host(std::vector<T>& hx, std::vector<T>& hy) const { x.to_host(hx); y.to_host(hy); }

    // Copy data from host to device
    void from_host(const std::vector<T>& hx, const std::vector<T>& hy) { x.from_host(hx); y.from_host(hy); }

    // Fill both axes with set values
    void fill(T vx, T vy){ x.fill(vx); y.fill(vy); }

    // Scale both axes with set values
    void scale(T ax, T ay){ x.scale(ax); y.scale(ay); }

    // Enable swap, allocate aux memory
    void enable_swap() { x.enable_swap(); y.enable_swap(); }

    // Disable swap, free aux memory
    void disable_swap() { x.disable_swap(); y.disable_swap(); }

    // Enable statefull RNG, initialize RNG state array
    void enable_rng(unsigned long long seed = 0ULL) {
        _enable_rng = true;
        _rng_seed = seed;
        _init_rng(_rng_seed);
    }

    // Disable statefull RNG, free RNG state array memory
    void disable_rng() {
        _enable_rng = false;
        _rng_seed = 0ULL;
        thrust::device_vector<curandStatePhilox4_32_10_t>().swap(rng_states);
        // Ensure x/y 1D RNGs are disabled to avoid duplicate memory/usage confusion.
        if (x.rng_enabled()) x.disable_rng();
        if (y.rng_enabled()) y.disable_rng();
    }

    // Statefull RNG (uniform) - requires rng_enabled() = true
    void rand_uniform(T lx, T hx, T ly, T hy);

    // Statefull RNG (normal) - requires rng_enabled() = true
    void rand_normal(T mx, T sx, T my, T sy);

    // Stateless RNG (uniform)
    void stateless_rand_uniform(T lx, T hx, T ly, T hy, unsigned long long rng_count);

    // Pseudo-stateless RNG (uniform)
    void stateless_rand_uniform(T lx, T hx, T ly, T hy);

    // Stateless RNG (normal)
    void stateless_rand_normal(T mx, T sx, T my, T sy, unsigned long long rng_count);
    
    // Pseudo-stateless RNG (normal)
    void stateless_rand_normal(T mx, T sx, T my, T sy);

private:
    bool _enable_rng = false;
    unsigned long long _rng_seed = 0ULL;
    unsigned long long _rng_count = 0ULL;

    void _init_rng(unsigned long long seed);
};

} // namespace df