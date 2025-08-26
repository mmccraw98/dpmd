#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
  #define MD_HD __host__ __device__
#else
  #define MD_HD
#endif

namespace md::launch {
inline constexpr int kBlock = 256;  // kernel fixed block size

MD_HD inline int grid_size(int N, int B = kBlock) {
    int G = (N + B - 1) / B;
    return (G > 0) ? G : 1;
}
MD_HD inline dim3 blocks_for(int N, int B = kBlock) { return dim3(grid_size(N,B)); }
MD_HD inline dim3 threads_for(int B = kBlock)       { return dim3(B); }
}

// MD_DEBUG = 1 enables heavy sync+checks after every launch.
#ifndef MD_DEBUG
#define MD_DEBUG 0
#endif

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error: %s\n  at %s:%d\n  expr: %s\n",        \
                   cudaGetErrorString(_e), __FILE__, __LINE__, #call);        \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

#if MD_DEBUG
  // Launch (default stream)
  #define CUDA_LAUNCH(kernel, grid, block, ...)                               \
    do {                                                                      \
      kernel<<<(grid), (block)>>>(__VA_ARGS__);                               \
      cudaError_t _e = cudaGetLastError();                                    \
      if (_e != cudaSuccess) {                                                \
        std::fprintf(stderr, "Kernel launch error: %s\n  at %s:%d\n  kernel: %s\n", \
                     cudaGetErrorString(_e), __FILE__, __LINE__, #kernel);    \
        std::abort();                                                         \
      }                                                                       \
      _e = cudaDeviceSynchronize();                                           \
      if (_e != cudaSuccess) {                                                \
        std::fprintf(stderr, "Kernel runtime error: %s\n  at %s:%d\n  kernel: %s\n", \
                     cudaGetErrorString(_e), __FILE__, __LINE__, #kernel);    \
        std::abort();                                                         \
      }                                                                       \
    } while (0)

  // Launch (explicit stream)
  #define CUDA_LAUNCH_STREAM(kernel, grid, block, shmem, stream, ...)         \
    do {                                                                      \
      kernel<<<(grid), (block), (shmem), (stream)>>>(__VA_ARGS__);            \
      cudaError_t _e = cudaGetLastError();                                    \
      if (_e != cudaSuccess) {                                                \
        std::fprintf(stderr, "Kernel launch error: %s\n  at %s:%d\n  kernel: %s\n", \
                     cudaGetErrorString(_e), __FILE__, __LINE__, #kernel);    \
        std::abort();                                                         \
      }                                                                       \
      _e = cudaStreamSynchronize(stream);                                     \
      if (_e != cudaSuccess) {                                                \
        std::fprintf(stderr, "Kernel runtime error: %s\n  at %s:%d\n  kernel: %s\n", \
                     cudaGetErrorString(_e), __FILE__, __LINE__, #kernel);    \
        std::abort();                                                         \
      }                                                                       \
    } while (0)
#else
  #define CUDA_LAUNCH(kernel, grid, block, ...)                               \
    do { kernel<<<(grid), (block)>>>(__VA_ARGS__); } while (0)
  #define CUDA_LAUNCH_STREAM(kernel, grid, block, shmem, stream, ...)         \
    do { kernel<<<(grid), (block), (shmem), (stream)>>>(__VA_ARGS__); } while (0)
#endif