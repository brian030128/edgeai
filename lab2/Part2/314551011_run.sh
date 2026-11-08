#!/bin/bash
# Lab 2 Part 2 entry point: install deps, compress + evaluate all four cases.
# Usage: bash 314551011_run.sh
# Only the final Compression Ratio / PPL of every case is printed to stdout;
# installation and compression logs go to logs/.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PY=${PYTHON:-python}
$PY -m pip install -r requirements.txt > logs/pip.log 2>&1 || {
    echo "pip install failed, see logs/pip.log" >&2; exit 1; }

# case : target compression ratio : weight bits
CASES="1:0.75:4 2:0.81:3 3:0.50:8 4:0.85:3"

for spec in $CASES; do
    IFS=: read -r c rate wbits <<< "$spec"
    out=$($PY run_eval.py --case "$c" --rate "$rate" --wbits "$wbits" 2> "logs/case$c.log")
    rc=$?
    ppl=$(echo "$out" | sed -n 's/^ppl: //p')
    cr=$(echo "$out" | sed -n 's/^comp_rate: //p')
    echo "Case $c"
    if [ $rc -ne 0 ] || [ -z "$ppl" ]; then
        echo "FAILED (exit $rc), see logs/case$c.log"
    else
        echo "Compression Ratio: $cr"
        echo "PPL: $ppl"
    fi
    echo
done
