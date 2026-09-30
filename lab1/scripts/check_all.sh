#!/usr/bin/env bash
# Correctness sweep. This is what the graders run (on THEIR GPU).
# Speed is not graded; every kernel must PASS (or be SKIPPED) on every shape.
#   usage: scripts/check_all.sh [binary]
BIN=${1:-./sgemm}
SHAPES=(
  "128 128 128"   "1024 1024 1024" "4096 4096 4096"
  "1000 1000 1000" "1023 257 511"  "1 1 1" "7 13 3"
  "4096 1 4096"   "1 4096 4096"    "8192 1 8192"
  "16 4096 4096"  "4096 4096 64"   "64 64 65536"
)
fail=0
for s in "${SHAPES[@]}"; do
  for beta in 0 0.5; do
    out=$($BIN all $s --beta $beta --alpha 1.5 --reps 1 --warmup 0 --csv)
    echo "$out" | awk -F, -v b=$beta '{printf "%-22s %6s %6s %6s beta=%-4s %s\n",$2,$3,$4,$5,b,$12}'
    echo "$out" | grep -q ",FAIL$" && fail=1
  done
done
[ $fail -eq 0 ] && echo "ALL PASSED" || { echo "SOME KERNELS FAILED"; exit 1; }
