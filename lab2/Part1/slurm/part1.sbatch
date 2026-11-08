#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=lab2_part1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -uo pipefail
cd /home/u4320956/edgeai/.claude/worktrees/lab2/lab2/Part1
export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "=== node: $(hostname) start $(date -Iseconds) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
    --allow-errors 314551011_Part1.ipynb
echo "=== nbconvert rc=$? ==="

uv run python - 314551011_Part1.ipynb <<'PY'
import json, sys
nb = json.load(open(sys.argv[1]))
bad = passed = 0
for i, c in enumerate(nb["cells"]):
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            bad += 1
            print(f"ERROR cell {i}: {o['ename']}: {o['evalue']}")
        passed += "".join(o.get("text", "")).count("* Test passed.")
print(f"torch printed: {''.join(nb['cells'][6]['outputs'][0].get('text', '')).strip() if nb['cells'][6]['outputs'] else '?'}")
print(f"tests passed: {passed}, error cells: {bad}")
PY
echo "=== done $(date -Iseconds) ==="
