#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=lab2_p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# usage: sbatch slurm/sweep.sbatch CONFIG [CONFIG ...]

set -uo pipefail
cd /home/u4320956/edgeai/.claude/worktrees/lab2/lab2/Part3
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME" results
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT

echo "=== node: $(hostname) configs: $* start $(date -Iseconds) ==="
uv run python sweep.py "$@"
echo "=== done rc=$? $(date -Iseconds) ==="
