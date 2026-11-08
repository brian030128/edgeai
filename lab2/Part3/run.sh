#!/bin/bash
# Lab 2 Part 3: install dependencies, quantize Qwen3-4B (calibration-free) and
# evaluate WikiText-2 PPL. Usage: bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY -m pip install -q -r requirements.txt
$PY run.py
