#include "utils/device_fields.cuh"

namespace df {

template <typename T>
void DeviceField1D<T>::set_element(std::size_t i, T v) {
    CUDA_CHECK(cudaMemcpy(ptr() + i, &v, sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
T DeviceField1D<T>::get_element(std::size_t i) const {
    T h;
    CUDA_CHECK(cudaMemcpy(&h, ptr() + i, sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

template <typename T>
T DeviceField1D<T>::get_element_async(std::size_t i, cudaStream_t stream) const {
    T h;
    CUDA_CHECK(cudaMemcpyAsync(&h, ptr() + i, sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h;
}

template <typename T>
void DeviceField1D<T>::fill(T v) {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_fill, G, B, ptr(), size(), v);
}

template <typename T>
void DeviceField1D<T>::scale(T alpha) {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_scale, G, B, ptr(), size(), alpha);
}

template <typename T>
void DeviceField1D<T>::rand_uniform(T lo, T hi) {
    if (!_enable_rng) throw std::runtime_error("DeviceField1D::rand_uniform: enable_rng==false");
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_rand_uniform_1d<T>, G, B, ptr(), size(), ptr_rng(), lo, hi);
}

template <typename T>
void DeviceField1D<T>::rand_normal(T mean, T sigma) {
    if (!_enable_rng) throw std::runtime_error("DeviceField1D::rand_normal: enable_rng==false");
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_rand_normal_1d<T>, G, B, ptr(), size(), ptr_rng(), mean, sigma);
}

template <typename T>
void DeviceField1D<T>::stateless_rand_uniform(T lo, T hi, unsigned long long rng_count) {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_stateless_uniform_1d<T>, G, B, ptr(), size(), _rng_seed, rng_count, lo, hi);
}

template <typename T>
void DeviceField1D<T>::stateless_rand_uniform(T lo, T hi) {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_stateless_uniform_1d<T>, G, B, ptr(), size(), _rng_seed, _rng_count++, lo, hi);
}

template <typename T>
void DeviceField1D<T>::stateless_rand_normal(T mean, T sigma, unsigned long long rng_count) {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_stateless_normal_1d<T>, G, B, ptr(), size(), _rng_seed, rng_count, mean, sigma);
}

template <typename T>
void DeviceField1D<T>::stateless_rand_normal(T mean, T sigma) {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_stateless_normal_1d<T>, G, B, ptr(), size(), _rng_seed, _rng_count++, mean, sigma);
}

template <typename T>
void DeviceField1D<T>::_init_rng(unsigned long long seed, unsigned long long subseq_offset) {
    if (!_enable_rng) _enable_rng = true;
    _rng_seed = seed;
    const size_t need = static_cast<size_t>(size());
    if (rng_states.size() != need) rng_states.resize(need);
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_init_rng_states, G, B, rng_states.data().get(), size(), _rng_seed, subseq_offset);
}

template <typename T>
void DeviceField2D<T>::rand_uniform(T lx, T hx, T ly, T hy) {
    if (!_enable_rng) throw std::runtime_error("DeviceField2D::rand_uniform: rng_enabled()==false");
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_rand_uniform_2d_one_state<T>, G, B,
                xptr(), yptr(), size(),
                ptr_rng2d(),
                lx, hx, ly, hy);
}

template <typename T>
void DeviceField2D<T>::rand_normal(T mx, T sx, T my, T sy) {
    if (!_enable_rng) throw std::runtime_error("DeviceField2D::rand_normal: rng_enabled()==false");
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_rand_normal_2d_one_state<T>, G, B,
                xptr(), yptr(), size(),
                ptr_rng2d(),
                mx, sx, my, sy);
}

template <typename T>
void DeviceField2D<T>::stateless_rand_uniform(T lx, T hx, T ly, T hy, unsigned long long rng_count) {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_stateless_uniform_2d<T>, G, B, xptr(), yptr(), size(), _rng_seed, rng_count, lx, hx, ly, hy);
}

template <typename T>
void DeviceField2D<T>::stateless_rand_uniform(T lx, T hx, T ly, T hy) {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_stateless_uniform_2d<T>, G, B, xptr(), yptr(), size(), _rng_seed, _rng_count++, lx, hx, ly, hy);
}

template <typename T>
void DeviceField2D<T>::stateless_rand_normal(T mx, T sx, T my, T sy, unsigned long long rng_count) {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_stateless_normal_2d<T>, G, B, xptr(), yptr(), size(), _rng_seed, rng_count, mx, sx, my, sy);
}

template <typename T>
void DeviceField2D<T>::stateless_rand_normal(T mx, T sx, T my, T sy) {
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_stateless_normal_2d<T>, G, B, xptr(), yptr(), size(), _rng_seed, _rng_count++, mx, sx, my, sy);
}

template <typename T>
void DeviceField2D<T>::_init_rng(unsigned long long seed) {
    if (!_enable_rng) _enable_rng = true;
    _rng_seed = seed;
    const size_t need = static_cast<size_t>(size());
    if (rng_states.size() != need) rng_states.resize(need);
    auto B = md::launch::threads_for();
    auto G = md::launch::blocks_for(size());
    CUDA_LAUNCH(k_init_rng_states, G, B, rng_states.data().get(), size(), _rng_seed, 0ULL);
}

// Explicit instantiation for DeviceField1D types
template class DeviceField1D<float>;
template class DeviceField1D<double>;
template class DeviceField1D<int>;
template class DeviceField1D<unsigned int>;
template class DeviceField1D<unsigned long long>;
template class DeviceField1D<bool>;
template class DeviceField1D<long>;
template class DeviceField1D<unsigned char>;

// Explicit instantiation for DeviceField2D types
template class DeviceField2D<float>;
template class DeviceField2D<double>;
template class DeviceField2D<int>;
template class DeviceField2D<unsigned int>;
template class DeviceField2D<unsigned long long>;
template class DeviceField2D<bool>;
template class DeviceField2D<long>;
template class DeviceField2D<unsigned char>;
} // namespace df