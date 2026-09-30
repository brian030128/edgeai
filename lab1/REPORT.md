# SGEMM Homework Report

**Name:**
**GPU:**  (name, compute capability, #SMs)
**Peak FP32 (spec sheet):**  GFLOP/s    **Peak DRAM BW (spec sheet):**  GB/s
**Driver / CUDA version:**
**Laptop or desktop? Clocks locked?**

`scripts/check_all.sh` output (last line):  ALL PASSED / ...

## Part 0 — Roofline
- Ridge point:
- AI of SGEMM 4096³:          AI of GEMV 4096×4096:
- Expected bound for each:

## Part 1 — Naive kernel
| size | naive ms | cuBLAS ms | % cuBLAS |
|---|---|---|---|

nsys kernel summary (screenshot / table):

Q1 (addresses & sectors per load):
Q2 (evidence it is memory-bound — metric names + values):
Q3:

## Part 2 — Coalescing
Prediction (written BEFORE measuring):
| metric | naive | coalesced |
|---|---|---|
| time (ms) @ 2048 | | |
| global ld sectors/request | | |
| L1 / L2 / DRAM throughput % | | |

Q1–Q4:

## Part 3 — Tiling
| config | ms | %cuBLAS | regs | smem/block | theo. occ | achieved occ | top stall | smem bank conflicts |
|---|---|---|---|---|---|---|---|---|

Q1–Q4:

## Part 4 — Shapes
Plots (GFLOP/s for GEMM-like shapes, GB/s for GEMV shapes):

Q1–Q4:

Improvement implemented (kernel name in kKernels):
| shape | before | after | evidence (metric) |
|---|---|---|---|

## Part 5 — Experiments
(copy this block once per optimization)

### Experiment N: <name>   — kernel `<name in kKernels>`
- **Hypothesis & prediction:**
- **Change:**
- **Result:**
  | size | before ms | after ms | Δ |
  |---|---|---|---|
- **Evidence:** (metrics before → after; screenshot; `reports/<file>.ncu-rep`)
  | metric | before | after |
  |---|---|---|
- **Explanation:** (why it worked / didn't; if it failed, what was the real bottleneck?)

## Summary
Final ranking of your kernels on your GPU, and the single most important thing you learned.
