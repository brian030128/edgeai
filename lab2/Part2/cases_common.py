"""Glue shared by case1..case4: run a spec and report the case constraints."""
import torch

from compress_lib import compress
from common import log


def q(bits, gs=-1, sym=False, actorder=True):
    """GPTQ config: bits, group size (-1 = one scale per output row)."""
    return {"bits": bits, "groupsize": gs, "sym": sym, "actorder": actorder}


def run(model, tok, spec, checks):
    """checks: list of (label, stat_key, op, threshold) evaluated after compression."""
    torch.manual_seed(0)
    compress(model, tok, spec)
    st = model._stats
    for label, key, op, thr in checks:
        v = st[key]
        ok = v > thr if op == ">" else v >= thr if op == ">=" else abs(v - thr) < 0.01
        log(f"[check] {label}: {v:.4%} {op} {thr:.0%} -> {'OK' if ok else 'VIOLATED'}")
    return model
