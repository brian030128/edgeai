#pragma once
// =============================================================================
//  kernels.cuh  --  THIS IS THE FILE YOU EDIT
// =============================================================================
//  Row-major, FP32 only:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//
//  Rules:
//   * FP32 CUDA cores only. No tensor cores, no TF32, no WMMA / mma.sync.
//   * No cuBLAS / CUTLASS / other libraries inside your kernels.
//   * Every kernel must pass the harness correctness check for every size
//     its launcher accepts. Return false from the launcher for sizes you
//     do not support.
//
//  To add a kernel: write the __global__ function, write a launcher, and add
//  one line to kKernels[] at the bottom of this file.
// =============================================================================
#include "common.cuh"

// -----------------------------------------------------------------------------
// Kernel 1: naive (GIVEN -- do not modify, it is your baseline)
// -----------------------------------------------------------------------------
// One thread computes one element of C.
// NOTE the index mapping: threadIdx.x -> ROW. Think about which addresses the
// 32 threads of one warp touch at the same moment.
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B, float beta,
                            float* __restrict__ C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    }
    const size_t idx = (size_t)row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
  }
}

bool launch_naive(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C, cudaStream_t s) {
  dim3 block(32, 32);
  dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
  sgemm_naive<<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

// -----------------------------------------------------------------------------
// Kernel 2: coalesced global memory access            (Part 2)
// -----------------------------------------------------------------------------
// Same algorithm as the naive kernel. Change only how threads map to (row, col)
// so that consecutive threads in a warp touch consecutive addresses.
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B, float beta,
                                float* __restrict__ C) {
  // TODO(Part 2)
}

bool launch_coalesced(int M, int N, int K, float alpha, const float* A,
                      const float* B, float beta, float* C, cudaStream_t s) {
  // TODO(Part 2): choose grid/block, launch, and return true.
  return false;
}

// -----------------------------------------------------------------------------
// Kernel 3: shared-memory tiling, one output per thread   (Part 3a)
// -----------------------------------------------------------------------------
// Each block computes a TILE x TILE tile of C. Loop over K in steps of TILE:
// cooperatively load a tile of A and a tile of B into __shared__ memory,
// __syncthreads(), accumulate, __syncthreads().
// Must handle ANY M, N, K (zero-pad out-of-range tile elements).
template <int TILE>
__global__ void sgemm_smem(int M, int N, int K, float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B, float beta,
                           float* __restrict__ C) {
  // TODO(Part 3a)
}

template <int TILE>
bool launch_smem(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C, cudaStream_t s) {
  // TODO(Part 3a)
  return false;
}

// -----------------------------------------------------------------------------
// Kernel 4: shared-memory tiling + 2D register blocking   (Part 3b)
// -----------------------------------------------------------------------------
// A block of (BM/TM)*(BN/TN) threads computes a BM x BN tile of C.
// Each thread computes a TM x TN sub-tile held in registers.
// Per K-step of size BK: load As (BM x BK) and Bs (BK x BN) into shared
// memory, then for each k in [0, BK): load TM values of A and TN values of B
// into registers and do TM*TN FMAs (an outer product).
//
// You may assume M % BM == 0, N % BN == 0, K % BK == 0 (the launcher below
// rejects other sizes). Part 4 asks what to do about the other sizes.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_reg2d(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  // TODO(Part 3b)
}

// Set to true once your sgemm_reg2d works.
constexpr bool kReg2DImplemented = false;

template <int BM, int BN, int BK, int TM, int TN>
bool launch_reg2d(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C, cudaStream_t s) {
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide block tile");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads <= 1024, "too many threads per block");
  if (!kReg2DImplemented) return false;
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_reg2d<BM, BN, BK, TM, TN>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

// -----------------------------------------------------------------------------
// Kernel 5+: your own kernels                          (Parts 4 and 5)
// -----------------------------------------------------------------------------
// e.g. a GEMV kernel + a shape-dispatching launcher, a vectorized / double
// buffered / cp.async / swizzled variant of kernel 4, ...

// =============================================================================
// Registry. The harness refers to kernels by index or by name.
// Index 0 is always cuBLAS (provided by the harness).
// Add new configurations / kernels here.
// =============================================================================
static const KernelEntry kKernels[] = {
    {"naive", launch_naive},                                   // 1
    {"coalesced", launch_coalesced},                           // 2
    {"smem16", launch_smem<16>},                               // 3
    {"smem32", launch_smem<32>},                               // 4
    {"reg2d_128x128x8_8x8", launch_reg2d<128, 128, 8, 8, 8>},  // 5
    {"reg2d_64x64x8_4x4", launch_reg2d<64, 64, 8, 4, 4>},      // 6
    {"reg2d_128x64x8_8x4", launch_reg2d<128, 64, 8, 8, 4>},    // 7
    // {"my_gemv", launch_my_gemv},
};
