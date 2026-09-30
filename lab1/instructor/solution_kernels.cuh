#pragma once
// =============================================================================
//  INSTRUCTOR REFERENCE SOLUTIONS -- do not distribute.
//  Build with `make solution`. Used to validate the harness, check_all.sh,
//  and to sanity-check student claims on the grader's GPU.
//  Deliberately "textbook" versions: no vectorization, no double buffering,
//  no swizzling -- those are Part 5 territory.
// =============================================================================
#include "../src/common.cuh"

// ---------------- Kernel 1: naive (identical to student file) ----------------
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B, float beta,
                            float* __restrict__ C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    const size_t idx = (size_t)row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
  }
}
bool launch_naive(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C, cudaStream_t s) {
  dim3 block(32, 32), grid(ceil_div(M, 32), ceil_div(N, 32));
  sgemm_naive<<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

// ---------------- Kernel 2: coalesced ----------------------------------------
// threadIdx.x -> column: a warp reads 32 consecutive B elements (1 request,
// 4 sectors) and broadcasts one A element; C writes are contiguous.
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B, float beta,
                                float* __restrict__ C) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    const size_t idx = (size_t)row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
  }
}
bool launch_coalesced(int M, int N, int K, float alpha, const float* A,
                      const float* B, float beta, float* C, cudaStream_t s) {
  dim3 block(32, 32), grid(ceil_div(N, 32), ceil_div(M, 32));
  sgemm_coalesced<<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

// ---------------- Kernel 3: shared-memory tiling ------------------------------
template <int TILE>
__global__ void sgemm_smem(int M, int N, int K, float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B, float beta,
                           float* __restrict__ C) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) {
    const size_t idx = (size_t)row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
  }
}
template <int TILE>
bool launch_smem(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C, cudaStream_t s) {
  dim3 block(TILE, TILE), grid(ceil_div(N, TILE), ceil_div(M, TILE));
  sgemm_smem<TILE><<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

// ---------------- Kernel 4: smem + 2D register blocking ----------------------
// As is stored transposed (As[k][m]) so the inner loop reads TM contiguous
// values. The transposed store causes smem bank conflicts on the write side --
// a good thing for students to find in ncu during Part 5.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_reg2d(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];
  const int tid = threadIdx.x;
  const int tRow = tid / (BN / TN);
  const int tCol = tid % (BN / TN);

  A += (size_t)blockIdx.y * BM * K;
  B += (size_t)blockIdx.x * BN;
  C += (size_t)blockIdx.y * BM * N + (size_t)blockIdx.x * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
  float regA[TM], regB[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = tid; i < BM * BK; i += NT) {
      const int r = i / BK, c = i % BK;
      As[c][r] = A[(size_t)r * K + c];
    }
    for (int i = tid; i < BK * BN; i += NT) {
      const int r = i / BN, c = i % BN;
      Bs[r][c] = B[(size_t)r * N + c];
    }
    __syncthreads();
    A += BK;
    B += (size_t)BK * N;
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regA[i] = As[k][tRow * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) regB[j] = Bs[k][tCol * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const size_t idx = (size_t)(tRow * TM + i) * N + tCol * TN + j;
      C[idx] = alpha * acc[i][j] + beta * C[idx];
    }
}

constexpr bool kReg2DImplemented = true;

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

// ---------------- Part 4 reference: GEMV kernels + shape dispatch -------------
// N == 1:  y (M) = A (MxK) @ x (K). One warp per row: coalesced reads of the
// A row, warp-shuffle reduction. Memory bound -> judge by GB/s.
__global__ void sgemv_warp_per_row(int M, int K, float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ x, float beta,
                                   float* __restrict__ y) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= M) return;  // warp-uniform, so full-mask shuffles are safe
  const float* a = A + (size_t)warp * K;
  float acc = 0.f;
  for (int k = lane; k < K; k += 32) acc += a[k] * x[k];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) y[warp] = alpha * acc + beta * y[warp];
}

// M == 1:  y (N) = x (K) @ B (KxN). One thread per output column, loop over
// K: each warp reads 32 consecutive B elements per k (coalesced).
// Parallelism is only N threads -- for small N a split-K version is needed.
__global__ void sgemv_col_per_thread(int N, int K, float alpha,
                                     const float* __restrict__ x,
                                     const float* __restrict__ B, float beta,
                                     float* __restrict__ y) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += x[k] * B[(size_t)k * N + n];
  y[n] = alpha * acc + beta * y[n];
}

bool launch_dispatch(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C, cudaStream_t s) {
  if (N == 1) {
    const int threads = 256;
    sgemv_warp_per_row<<<ceil_div(M * 32, threads), threads, 0, s>>>(M, K, alpha, A, B, beta, C);
    return true;
  }
  if (M == 1) {
    sgemv_col_per_thread<<<ceil_div(N, 256), 256, 0, s>>>(N, K, alpha, A, B, beta, C);
    return true;
  }
  if (launch_reg2d<128, 128, 8, 8, 8>(M, N, K, alpha, A, B, beta, C, s)) return true;
  if (launch_reg2d<64, 64, 8, 4, 4>(M, N, K, alpha, A, B, beta, C, s)) return true;
  return launch_smem<32>(M, N, K, alpha, A, B, beta, C, s);  // any size
}

static const KernelEntry kKernels[] = {
    {"naive", launch_naive},                                   // 1
    {"coalesced", launch_coalesced},                           // 2
    {"smem16", launch_smem<16>},                               // 3
    {"smem32", launch_smem<32>},                               // 4
    {"reg2d_128x128x8_8x8", launch_reg2d<128, 128, 8, 8, 8>},  // 5
    {"reg2d_64x64x8_4x4", launch_reg2d<64, 64, 8, 4, 4>},      // 6
    {"reg2d_128x64x8_8x4", launch_reg2d<128, 64, 8, 8, 4>},    // 7
    {"dispatch", launch_dispatch},                             // 8
};
