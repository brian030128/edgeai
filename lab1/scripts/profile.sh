#!/usr/bin/env bash
# Profile ONE kernel with nsys and ncu.   usage: scripts/profile.sh <kernel> [M N K]
# Reports go to reports/. Open .nsys-rep in Nsight Systems, .ncu-rep in Nsight Compute.
set -e
KER=${1:?kernel name or id}; shift
SIZE=${*:-4096 4096 4096}
mkdir -p reports
TAG=$(echo "$KER $SIZE" | tr ' ' '_')

# 1) Nsight Systems: timeline + per-kernel time summary (your kernel vs cuBLAS)
nsys profile --force-overwrite true -o reports/nsys_$TAG --stats=true \
  ./sgemm $KER $SIZE --reps 5 --no-flush

# 2) Nsight Compute: WHY it is slow. Only your kernel, one launch, all sections.
#    --clock-control none : same clocks the harness sees (ncu locks to base by default)
ncu --set full --import-source yes --clock-control none \
    --kernel-name-base function -k "regex:^(sgemm|sgemv)" \
    -f -o reports/ncu_$TAG ./sgemm $KER $SIZE --profile
echo "open reports/ncu_$TAG.ncu-rep in Nsight Compute (ncu-ui)"
