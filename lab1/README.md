# Homework: FP32 Matrix Multiplication Kernel in CUDA

## Rules

1. **FP32 on CUDA cores only.** No tensor cores, TF32, WMMA, or `mma.sync`.
2. **No libraries inside your kernels.** cuBLAS is only the reference.
3. **Every kernel must pass `scripts/check_all.sh`.**

## Setup

Requirements: NVIDIA GPU (compute capability ≥ 6.0), CUDA Toolkit ≥ 11.6, Nsight Systems (`nsys`), Nsight Compute (`ncu`).

```bash
make                         # builds ./sgemm, prints registers/smem per kernel
./sgemm --list               # kernel ids
./sgemm all 4096 4096 4096   # benchmark + check every kernel
./sgemm naive 1024 1024 1024 # one kernel
scripts/check_all.sh         # correctness on many shapes
scripts/profile.sh naive     # nsys + ncu reports into reports/
```

If `make` fails on `-arch=native`, use e.g. `make ARCH=-arch=sm_86`.

If `ncu` reports `ERR_NVGPUCTRPERM`: on Linux run with `sudo` or have the counters enabled; on Windows enable "Allow access to GPU performance counters to all users" in the NVIDIA Control Panel.

### Harness output

```
id  kernel          ms   GFLOP/s   GB/s  %cuBLAS  %peakF  %peakBW  check
```

GFLOP/s is `2·M·N·K / time`. GB/s is the minimum DRAM traffic (read A and B, write C) divided by time. Peak numbers are estimated from device attributes; use the spec-sheet values in your report. Timings are the median of 20 launches with L2 flushed before each launch.

---

## Part 0 — Know your GPU

Report your GPU's name, compute capability, SM count, peak FP32 GFLOP/s, and peak DRAM GB/s (from the spec sheet, compared with the harness estimate).

1. What is the ridge point of your GPU's roofline, in FLOP/byte?
2. What is the arithmetic intensity of SGEMM with M = N = K = 4096, assuming each matrix is read from DRAM exactly once? What about GEMV (N = 1, M = K = 4096)?
3. Which one should be compute-bound and which memory-bound on your GPU?

## Part 1 — The naive kernel and the profiler

`sgemm_naive` in `src/kernels.cuh` is given. Do not modify it.

1. Benchmark `naive` against `cublas` at 1024, 2048, and 4096.
2. Run `scripts/profile.sh naive 2048 2048 2048`.
3. In Nsight Systems (`--stats` output or `nsys-ui`), find the naive kernel and the cuBLAS kernel in the kernel summary. How much of the total runtime is each?

Open the `.ncu-rep` in Nsight Compute and look at:

- **GPU Speed Of Light Throughput.** Compare compute (SM) % and memory % against the peaks.
- **Memory Workload Analysis.** L1/L2/DRAM throughput, and the ratio of sectors to requests for global loads.
- **Warp State Statistics.** The dominant stall reason.
- **Source Counters / Source view.** Which source line is flagged for uncoalesced global accesses.

**Questions**

1. For one warp at a single iteration `k` of the loop, list which addresses of `A`, `B`, and `C` its 32 threads touch. How many 32-byte sectors does each load need?
2. Using ncu numbers, show that the kernel is limited by memory access, not by FP32 math. Quote the specific metrics you used.
3. Why does this kernel reach such a small fraction of cuBLAS even though SGEMM should be compute-bound (Part 0)?

## Part 2 — Coalescing

Implement `sgemm_coalesced`. Use the same algorithm as the naive kernel; change only how threads map to `(row, col)`.

Before you run it, write down your predicted speedup and your reasoning. Then measure it and profile it.

**Questions**

1. Compare the global-load sectors-per-request (or the equivalent ncu metric) before and after. Does the change match your answer to Part 1, Q1?
2. The code performs the same FLOPs and the same loads in the same order. Explain in one paragraph why the speed changed.
3. In the Memory Workload Analysis chart, which level of the memory hierarchy (L1, L2, DRAM) is now busiest? Each element of A is loaded K times in total. Where do those repeated loads get served from?
4. How did your prediction compare with the measurement? If it was off, why?

## Part 3 — Tiling with shared memory and registers

### 3a. Shared-memory tiling

Implement `sgemm_smem<TILE>` (one output per thread). It must handle any M, N, and K. Registered configurations: `smem16`, `smem32`.

### 3b. 2D register blocking

Implement `sgemm_reg2d<BM, BN, BK, TM, TN>`. Each thread computes a TM×TN block of outputs held in registers; each block of threads computes a BM×BN tile. The launcher is provided. Set `kReg2DImplemented = true` once it works.

### Experiment

Add at least six configurations to `kKernels[]` that vary BM/BN, BK, and TM/TN. For example: BM, BN ∈ {64, 128}; BK ∈ {8, 16, 32}; TM, TN ∈ {4, 8}. For each configuration, fill in a table with: time; registers/thread (from `make` output); shared memory per block; theoretical and achieved occupancy (ncu Occupancy section); dominant warp stall; shared-memory bank conflicts (ncu Memory Workload Analysis → Shared Memory table).

**Questions**

1. In 3a, how many shared-memory loads does each thread do per FMA? In 3b with TM = TN = 8? Relate this to why 3b is faster.
2. Which configuration is fastest on your GPU? Pick one that is clearly slower and explain the difference using ncu metrics (e.g. register pressure → lower occupancy, bank conflicts, too few blocks to fill the SMs, barrier stalls).
3. Find a case where higher occupancy was slower, or explain why you could not find one.
4. Compare with one classmate who has a different GPU. Is the best configuration the same? Give a hypothesis for why or why not.

## Part 4 — Different shapes, and GEMV

Run `scripts/sweep.sh`. It benchmarks all kernels on square, non-tile-multiple, skinny, GEMV (N = 1 or M = 1), small-K, and large-K shapes, and writes a CSV to `results/`. Make plots from it.

**Questions**

1. For GEMV (`4096 1 4096` and `8192 1 8192`), is GFLOP/s the right metric? Which metric is the right one, and how close does each of your kernels get to your GPU's peak DRAM bandwidth?
2. Your reg2d kernel rejects N = 1. If you ran a BN = 128 tiled kernel on it anyway (padding N up to 128), what fraction of its work would be useful? How many thread blocks would there be, compared with your number of SMs?
3. `64 64 65536` has plenty of FLOPs but runs poorly on most kernels. Why? (Hint: count the blocks.)
4. `4096 1 4096` and `1 4096 4096` have the same FLOPs and bytes. Should one kernel design fit both? Consider which dimension is contiguous in memory.

**Task:** implement at least one improvement for the shapes that did badly, and register it in `kKernels[]`. It must pass `check_all.sh`. Options: a dedicated GEMV kernel (one warp per row with a shuffle reduction, vectorized loads, …), split-K for small M×N with large K, or a `dispatch` launcher that picks a kernel based on the shape and falls back to a general kernel for sizes your tiled kernels reject. Report the before and after, with ncu evidence (for GEMV, DRAM throughput % in Speed of Light).

## Part 5 — Your own optimizations

Try as many optimizations as you like. Ideas:

| Optimization | Notes |
|---|---|
| Vectorized loads (`float4`) | For global and/or shared memory. Watch alignment. |
| Transposed / padded shared-memory layout | Find the bank conflicts in ncu first, then fix them. |
| Shared-memory swizzling | XOR-based index remapping as an alternative to padding. |
| Thread-block swizzling / rasterization | Changes the launch order of blocks to improve L2 hit rate. |
| Warp tiling | Adds a warp-level tile between block tiles and thread tiles. |
| Double buffering | Two shared-memory stages, or register prefetch of the next tile while computing the current one. |
| Async copies (`cp.async`, `cuda::memcpy_async`) | Hardware-async on sm_80+ only. |
| `__launch_bounds__`, `maxrregcount` | Trade registers against occupancy. |
| Autotuning | Automatically search the configuration space. |
| Split-K, stream-K | For shapes with few output tiles. |
| Anything else | Loop unrolling, `__ldg`, `-use_fast_math`, … |

For each optimization, write up:

1. **Hypothesis.** Which profiler-identified bottleneck does this target? What do you predict will happen, and roughly by how much?
2. **Change.** What did you change? Name the kernel in `kKernels[]`.
3. **Result.** Time and %cuBLAS before and after, at one or more sizes.
4. **Evidence.** The ncu metrics that moved (or didn't), before vs after (bank conflicts, stall reasons, L2 hit rate, registers, occupancy, instruction counts). Include screenshots or tables and attach the `.ncu-rep` files.
5. **Explanation.** Why it worked, or why it didn't. If it failed, what was the real bottleneck?

---

## What to submit

- `src/kernels.cuh`: all your kernels, registered in `kKernels[]`.
- `REPORT.md` (use the template), exported to PDF if you like.
- `results/`: your sweep CSV(s) and plots.
- `reports/`: the `.ncu-rep` files you cite. Include at least the naive, coalesced, and best reg2d kernels, plus one per Part 5 experiment. Profile one launch at a modest size, such as 2048.

## Profiling cheat sheet

```bash
# timeline + per-kernel summary
nsys profile --stats=true -o reports/x ./sgemm <kernel> 4096 4096 4096 --reps 5

# one launch of your kernel, all sections, clocks not locked
ncu --set full --import-source yes --clock-control none \
    --kernel-name-base function -k "regex:^(sgemm|sgemv)" \
    -o reports/x ./sgemm <kernel> 2048 2048 2048 --profile

# print specific metrics in the terminal
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    -k "regex:^sgemm" ./sgemm <kernel> 2048 2048 2048 --profile
```

| You suspect… | Look at |
|---|---|
| Uncoalesced access | Global-load sectors/request (4 is ideal for 32 × fp32); "Uncoalesced Global Accesses" rule in Source Counters |
| Memory-bound | Speed of Light: memory % much higher than compute %; stall reason *Long Scoreboard* |
| Bank conflicts | Memory Workload Analysis → Shared Memory table; stall reason *MIO Throttle* / *Short Scoreboard* |
| Low occupancy | Occupancy section: the limiter (registers, shared memory, or block size) |
| Sync overhead | Stall reason *Barrier* |
| Not enough parallelism | Launch Statistics: grid size vs. SM count, "waves per SM" |

## Notes

- Never profile or benchmark a `-G` (debug) build.
- `ncu` locks clocks to base frequency by default; the scripts pass `--clock-control none`. Compare ncu metrics with each other, not ncu time with harness time.
- A missing `__syncthreads()` often passes at small sizes and fails at large ones. Run `check_all.sh`.
- `row * K` overflows `int` for large matrices; use `size_t` for indices.
- `float4` loads require 16-byte-aligned addresses.
