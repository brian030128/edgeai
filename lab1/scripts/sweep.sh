#!/usr/bin/env bash
# Performance sweep over shapes -> results/sweep_<gpu>.csv  (Part 4)
#   usage: scripts/sweep.sh [binary]
BIN=${1:-./sgemm}
mkdir -p results
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT=results/sweep_${GPU}.csv
echo "gpu,kernel,M,N,K,ms,gflops,gbps,pct_cublas,pct_peak_flops,pct_peak_bw,check" > "$OUT"
SHAPES=(
  # square
  "512 512 512" "1024 1024 1024" "2048 2048 2048" "4096 4096 4096" "8192 8192 8192"
  # not multiples of the tile size
  "1000 1000 1000" "3000 3000 3000"
  # GEMV / skinny
  "4096 1 4096" "8192 1 8192" "1 4096 4096" "1 8192 8192"
  "16 4096 4096" "4096 16 4096"
  # small K / large K
  "4096 4096 64" "64 64 65536"
)
for s in "${SHAPES[@]}"; do
  echo ">> $s"
  $BIN all $s --csv --reps 10 | tee -a "$OUT"
done
echo "wrote $OUT"
