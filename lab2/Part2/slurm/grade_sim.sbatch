#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=lab2_grade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Reproduce the grading flow: fresh Python 3.11 + torch 2.5.1+cu121, then
# `bash 314551011_run.sh` on a clean copy of the submission files.

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
SRC=/home/u4320956/edgeai/.claude/worktrees/lab2/lab2/Part2
WORK="/tmp/$USER/grade_$SLURM_JOB_ID"
export HF_HOME="$WORK/hf"
mkdir -p "$WORK/sub" "$HF_HOME"
trap 'rm -rf "$WORK"' EXIT

# --- base grading environment ---
uv venv --seed --python 3.11 "$WORK/env" >/dev/null
"$WORK/env/bin/pip" install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$WORK/env/bin/python" -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__)"

# --- clean copy of the submission (only what would be zipped) ---
cp "$SRC"/{314551011_run.sh,requirements.txt,run_eval.py,evaluate.py,common.py,compress_lib.py,gptq.py,cases_common.py,case1.py,case2.py,case3.py,case4.py} "$WORK/sub/"
cd "$WORK/sub"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=== run.sh start $(date -Iseconds) ==="
PATH="$WORK/env/bin:$PATH" bash 314551011_run.sh | tee "$SRC/results/grade_sim_$SLURM_JOB_ID.txt"
echo "=== run.sh end $(date -Iseconds) ==="
mkdir -p "$SRC/results/grade_sim_logs_$SLURM_JOB_ID"
cp logs/*.log "$SRC/results/grade_sim_logs_$SLURM_JOB_ID/"
grep -h "\[check\]" logs/case*.log
