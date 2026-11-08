"""Lab 2 Part 3: calibration-free W/A quantization of Qwen3-4B, then
WikiText-2 PPL with the provided evaluation (evaluate.eval_ppl_wikitext2).

Final method (see README.md / report), applied to every target Linear
(q/k/v/o/gate/up/down_proj of all 36 decoder layers):
  * weights: FP4 E2M1 elements with one FP8 E4M3 scale per 32 weights
    (block max mapped exactly to 6, so nothing is clipped) and one fp32 scale
    per tensor; the 4.5% of 16x32 weight tiles whose error drops most when
    stored as FP6 E2M3 (same two-level scaling) are promoted to FP6;
  * activations (inputs of those Linear layers): MXFP6 (E2M3, E8M0 scale per
    32), scales computed on the fly from each block.
Fully deterministic; nothing is fitted on data: no calibration set is loaded.
"""
import argparse
import json

import torch

from aquant import quantize_model, summarize
from evaluate import DEFAULT_MODEL, eval_ppl_wikitext2, load_model


def mx(fmt, rule="floor", block=32):
    return {"kind": "mx", "fmt": fmt, "rule": rule, "block": block}


def nv(fmt, block=32):
    """FP element format with an E4M3 scale per block and an fp32 per-tensor scale."""
    return {"kind": "nv", "fmt": fmt, "block": block, "g": "tensor"}


FINAL = {
    "w_lo": nv("fp4"), "w_hi": nv("fp6"), "w_frac": 0.045, "w_tile_rows": 16,
    "a_lo": mx("fp6"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--no-quant", action="store_true", help="evaluate the fp16 model")
    args = ap.parse_args()
    torch.manual_seed(0)
    model, tok = load_model(args.model, args.dtype)
    qs = None if args.no_quant else quantize_model(model, FINAL)
    ppl = eval_ppl_wikitext2(model, tok)
    print(f"PPL: {ppl:.4f}")
    if qs is not None:
        s = summarize(qs)
        print(f"Weight bits (avg, incl. scales + tile tags): {s['w_bits']:.4f}")
        print(f"Activation bits (avg, incl. scales): {s['a_bits']:.4f}")
        print(f"Realized effective bit rate (W + A) / 2: {s['eff_bits']:.4f}")
        print(json.dumps({"ppl": round(ppl, 4), **s}))
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
