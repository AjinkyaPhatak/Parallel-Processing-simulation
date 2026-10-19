// S02 kernels. Convention for 2D kernels: threadIdx.x / blockIdx.x -> column (contiguous in
// memory, row-major storage), threadIdx.y / blockIdx.y -> row. Block SHAPE therefore decides
// how many consecutive addresses a warp touches (coalescing).
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// ---- 1. Vector add (memory-bound, 1D) ------------------------------------------------------
__global__ void vecadd_kernel(const float* __restrict__ a, const float* __restrict__ b,
                              float* __restrict__ c, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ---- 2a. Naive matrix multiply C = A x B (baseline of the study) -------------------------
__global__ void matmul_naive_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float s = 0.0f;
        for (int k = 0; k < N; ++k) s += A[(size_t)row * N + k] * B[(size_t)k * N + col];
        C[(size_t)row * N + col] = s;
    }
}

// ---- 2b. Shared-memory tiled matmul (SECONDARY comparison only; blockDim = (T, T)) --------
template <int T>
__global__ void matmul_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int N) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * T + tx, row = blockIdx.y * T + ty;
    float s = 0.0f;
    for (int t = 0; t < N; t += T) {
        As[ty][tx] = (row < N && t + tx < N) ? A[(size_t)row * N + t + tx] : 0.0f;
        Bs[ty][tx] = (t + ty < N && col < N) ? B[(size_t)(t + ty) * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < T; ++k) s += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < N && col < N) C[(size_t)row * N + col] = s;
}

// ---- 3. 2D convolution, KxK filter in constant memory, zero padding ----------------------
__constant__ float c_filter[25];

template <int K>
__global__ void conv2d_kernel(const float* __restrict__ in, float* __restrict__ out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    constexpr int R = K / 2;
    float s = 0.0f;
#pragma unroll
    for (int dy = 0; dy < K; ++dy) {
        int yy = y + dy - R;
#pragma unroll
        for (int dx = 0; dx < K; ++dx) {
            int xx = x + dx - R;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) s += in[(size_t)yy * W + xx] * c_filter[dy * K + dx];
        }
    }
    out[(size_t)y * W + x] = s;
}

// ---- Empty kernel for launch / block-scheduling overhead ----------------------------------
__global__ void empty_kernel() {}
