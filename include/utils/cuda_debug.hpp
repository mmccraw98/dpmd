#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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