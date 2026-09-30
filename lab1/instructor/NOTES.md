# Instructor notes (do not distribute)

Remove the `instructor/` folder before handing out the starter code.

## Validating the setup
    make solution && ./sgemm_solution all 2048 2048 2048
    scripts/check_all.sh ./sgemm_solution     # should print ALL PASSED

The reference kernels are syntax-checked and their indexing was verified in a
CPU emulation, but run the line above on a real GPU before release.

## Grading workflow
1. Drop the student's `kernels.cuh` into `src/`, `make`, run `scripts/check_all.sh`
   -> correctness points (25). Any FAIL for a kernel = 0 for that kernel.
   Kernels still returning false (SKIPPED) = not implemented.
2. Read the report. Speed is never graded. Check that:
   - numbers are internally consistent (e.g. GFLOP/s = 2MNK/ms, %cuBLAS matches),
   - cited metrics exist in the attached .ncu-rep (spot-check 1-2 per student),
   - explanations follow from the evidence shown.
3. Hardware-independent claims can be spot-checked on the grader GPU:
   sectors/request naive vs coalesced, bank-conflict counts before/after
   padding/swizzle, registers per thread. Absolute times cannot.
4. Optional: 10-minute walkthrough with a random sample of students
   ("show me where in ncu you saw X", "why did this config lose?").

## Rough expectations (vary a lot by GPU; do not grade on these)
naive ~1-3% of cuBLAS; coalesced ~5-15%; smem ~10-20%;
reg2d 128x128x8_8x8 ~50-80%; with vectorization + bank-conflict fixes +
warp tiling, 80-95% is achievable. GEMV warp-per-row: 60-90% of peak DRAM BW.

## Common student mistakes worth feedback
- Claiming "memory bound" from nsys alone (nsys has no memory metrics).
- Comparing ncu durations to harness times (clock locking).
- Reporting GFLOP/s for GEMV instead of GB/s.
- float4 kernels that silently require N % 4 == 0 but accept any N
  (check_all.sh includes 1023x257x511 and 7x13x3 to catch this).
- Missing __syncthreads() after the compute loop (race on next tile load),
  which may pass at small sizes.
- Attributing a slowdown to "occupancy" without showing the occupancy numbers.
