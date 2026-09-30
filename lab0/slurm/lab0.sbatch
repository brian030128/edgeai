#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=lab0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -uo pipefail
cd /home/u4320956/edgeai/lab0

export STUDENT_ID=314551011

# --- environment ---
source /etc/profile.d/lmod.sh
module load cuda/12.4
export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# --- HF cache on node-local scratch, never /home or /work ---
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT
# the usual ~/.cache/huggingface/token does not exist on this account
export HF_TOKEN="${HF_TOKEN:-$(cat /work/$USER/hf/token 2>/dev/null)}"

mkdir -p slurm/logs results data/cifar10

echo "=== node: $(hostname)  start $(date -Iseconds) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- CIFAR-10: www.cs.toronto.edu is unusably slow from this cluster
#     (~500 B/s), so fetch the identical tarball from a mirror and let
#     torchvision verify the md5 and extract it. ------------------------
TGZ=data/cifar10/cifar-10-python.tar.gz
MD5=c58f30108f718f92721af3b95e74349a
if [ -d data/cifar10/cifar-10-batches-py ]; then
  echo "[cifar] already extracted"
elif [ -f "$TGZ" ] && [ "$(md5sum "$TGZ" | cut -d' ' -f1)" = "$MD5" ]; then
  echo "[cifar] tarball already present and verified"
else
  for M in \
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz" \
    "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz" \
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" ; do
    echo "[cifar] trying $M"
    if curl -fL --max-time 600 --retry 2 -o "$TGZ" "$M"; then
      got=$(md5sum "$TGZ" | cut -d' ' -f1)
      if [ "$got" = "$MD5" ]; then echo "[cifar] md5 ok"; break; fi
      echo "[cifar] md5 mismatch ($got), trying next mirror"
    fi
    rm -f "$TGZ"
  done
  [ -f "$TGZ" ] || { echo "[cifar] FAILED to obtain dataset"; exit 1; }
fi

# --- heartbeat so the log shows progress while nbconvert is quiet ---
( while true; do
    sleep 120
    echo "[hb $(date +%H:%M:%S)] gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"
  done ) &
HB=$!
trap 'kill $HB 2>/dev/null; rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT

# --- execute the notebook in place, no per-cell timeout ---
uv run jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    --allow-errors \
    "${STUDENT_ID}.ipynb"
NBRC=$?
kill $HB 2>/dev/null
echo "=== notebook executed rc=$NBRC $(date -Iseconds) ==="

# --- report any cell that raised, so failures are not silent ---
uv run python - "${STUDENT_ID}.ipynb" <<'PY'
import json, sys
nb = json.load(open(sys.argv[1]))
bad = 0
for i, c in enumerate(nb["cells"]):
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            bad += 1
            print(f"[cell {i}] {o.get('ename')}: {o.get('evalue')}")
print(f"=== cells with errors: {bad} ===")
PY

# --- hand-in artefacts ---
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
