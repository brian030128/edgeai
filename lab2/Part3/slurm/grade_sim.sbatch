#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=lab2_p3_grade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Fresh Python 3.11 + torch 2.5.1+cu121, then the README steps on a clean copy.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
SRC=/home/u4320956/edgeai/.claude/worktrees/lab2/lab2/Part3
WORK="/tmp/$USER/p3grade_$SLURM_JOB_ID"
export HF_HOME="$WORK/hf"
mkdir -p "$WORK/sub" "$HF_HOME"
trap 'rm -rf "$WORK"' EXIT

uv venv --seed --python 3.11 "$WORK/env" >/dev/null
"$WORK/env/bin/pip" install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
cp "$SRC"/{run.sh,run.py,requirements.txt,evaluate.py,aquant.py,mxq.py,README.md} "$WORK/sub/"
cd "$WORK/sub"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=== start $(date -Iseconds) ==="
PATH="$WORK/env/bin:$PATH" bash run.sh 2>"$SRC/results/grade_sim_$SLURM_JOB_ID.err" | tee "$SRC/results/grade_sim_$SLURM_JOB_ID.txt"
echo "=== end $(date -Iseconds) ==="
