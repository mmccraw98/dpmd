#pragma once
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