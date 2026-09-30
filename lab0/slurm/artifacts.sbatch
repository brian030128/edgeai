#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=lab0_art
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -uo pipefail
cd /home/u4320956/edgeai/lab0
export STUDENT_ID=314551011
export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg

uv run python make_screenshots.py "${STUDENT_ID}.ipynb"
uv run python onnx_graph.py model.onnx

rm -f "${STUDENT_ID}.zip"
zip -j "${STUDENT_ID}.zip" \
    "${STUDENT_ID}.ipynb" \
    "${STUDENT_ID}_acc_1.png" \
    "${STUDENT_ID}_acc_2.png" \
    "${STUDENT_ID}_onnx.png" \
    "${STUDENT_ID}_speedup.png"
unzip -l "${STUDENT_ID}.zip"
echo "=== done $(date -Iseconds) ==="
