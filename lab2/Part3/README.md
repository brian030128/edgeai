# Lab 2 Part 3 — Calibration-Free Quantization of Qwen3-4B

**Student ID:** 314551011

* **Result:** WikiText-2 PPL **14.19** at a realized effective bit rate of **5.296 bits**.
* **Baselines:** fp16 scores 13.64; the AdaptQuant reference is 13.98.
* **No calibration data, fully deterministic:** every quantization decision is computed from the tensor being quantized.

## 1. Environment

Tested with Python 3.11 and `torch==2.5.1+cu121`, the course's base grading environment. Any CUDA GPU with ≥ 12 GB works, including a T4: peak GPU memory is about 10.4 GiB.

```bash
# (optional) fresh environment
python3.11 -m venv venv && source venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run (quantize + evaluate) — one command

```bash
bash run.sh          # = pip install -r requirements.txt && python run.py
```

`run.py` does the following:

1. Downloads `Qwen/Qwen3-4B` from the Hugging Face Hub (fp16).
2. Quantizes every target Linear layer: weights offline, activations on the fly. This takes a few seconds.
3. Evaluates WikiText-2 test perplexity with `evaluate.eval_ppl_wikitext2`. This is the same evaluation as Part 2, with 2048-token windows.

Expected output:

```
PPL: 14.1944
Weight bits (avg, incl. scales + tile tags): 4.3420
Activation bits (avg, incl. scales): 6.2500
Realized effective bit rate (W + A) / 2: 5.2960
```

`python run.py --no-quant` evaluates the unquantized fp16 model (PPL 13.6383) for comparison.

## 4. Files

| File | Content |
|---|---|
| `run.sh` | entry point (install + run) |
| `run.py` | final configuration, quantize + evaluate, prints PPL and bit rates |
| `aquant.py` | `QLinear` (online activation quantization), per-tile weight-format selection, bit accounting |
| `mxq.py` | FP4/FP6/FP8 element rounding, MX (E8M0) and E4M3 two-level block scaling, block-Hadamard rotation (ablations only) |
| `evaluate.py` | provided model loader + PPL evaluation (unchanged procedure) |
| `sweep.py`, `sweep_configs.py`, `results/sweep.jsonl` | every exploration / ablation run quoted in the report |
| `slurm/` | cluster job scripts used for the runs |
| `314551011_report.pdf` | report |

## 5. Method in one paragraph

* **Target layers:** every `q/k/v/o/gate/up/down_proj` of the 36 decoder layers. Both their weights and their input activations are quantized.
* **Weights:**
  * FP4 E2M1 elements with one **FP8 E4M3 scale per 32 weights**, plus one fp32 scale per tensor.
  * The block maximum maps exactly onto 6, so no weight is clipped.
* **Per-tile selection:** the weights are split into tiles of 16 rows × 32 columns. The **4.5 % of tiles** whose squared error drops the most when stored as FP6 E2M3 (same scaling) are promoted, with a 1-bit format tag per tile.
* **Activations:** **MXFP6**: E2M3 with an E8M0 scale per 32, computed from each block on the fly.
* **Effective bit rate:**
  * Weights: 4.25 + 0.045·2 + 1/512 = 4.342 bits. Activations: 6.25 bits.
  * (4.342 + 6.25) / 2 = **5.296 ≤ 5.3**.
* **Why it needs no calibration:** no calibration set, no forward-pass statistics, no random seed and no tuned threshold are used.

Weights and activations are fake-quantized: quantized and de-quantized to fp16. This reproduces the numerics of the formats on any GPU without Blackwell kernels.
