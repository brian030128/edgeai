"""Layer-wise compression pipeline: GPTQ quantization, whitened SVD and
structured MLP pruning, with exact storage accounting (see common.py).

A *spec* is a plain dict:

    {
      "nsamples": 128, "seqlen": 2048,
      "quant": [rule, ...],      # first matching rule wins; cfg None = keep fp16
      "svd":   {"rules": [rule, ...], "factor_quant": [rule, ...]},
      "prune": {"rules": [rule, ...]},   # cfg {"ratio": r}: drop r of MLP channels
    }

    rule = {"layers": [idx...] | None, "names": ["q_proj", ...] | None, "cfg": {...} | None}

quant cfg : {"bits": 4, "groupsize": 128, "sym": False, "mse": False, "actorder": True}
svd cfg   : {"ratio": 0.2}   parameter reduction of that matrix
"""
import math

import torch
import torch.nn as nn

from common import (FP_BITS, current_ratio, hessians_for, get_calib, log,
                    quant_bits, sequential, set_bits)
from gptq import gptq_quantize

STAGES = [("self_attn", ["q_proj", "k_proj", "v_proj"]),
          ("self_attn", ["o_proj"]),
          ("mlp", ["gate_proj", "up_proj"]),
          ("mlp", ["down_proj"])]


def match(rules, idx, name):
    for r in rules or []:
        if r.get("layers") is not None and idx not in r["layers"]:
            continue
        if r.get("names") is not None and name not in r["names"]:
            continue
        return r.get("cfg")
    return None


class LowRankLinear(nn.Module):
    """y = A(B(x)) with B: in -> k, A: k -> out."""

    def __init__(self, A, B):
        super().__init__()
        out_f, k = A.shape
        in_f = B.shape[1]
        self.B = nn.Linear(in_f, k, bias=False, device=B.device, dtype=B.dtype)
        self.A = nn.Linear(k, out_f, bias=False, device=A.device, dtype=A.dtype)
        self.B.weight.data.copy_(B)
        self.A.weight.data.copy_(A)
        self.in_features, self.out_features, self.rank = in_f, out_f, k

    def forward(self, x):
        return self.A(self.B(x))


def svd_rank(out_f, in_f, ratio):
    return int(math.floor((1 - ratio) * out_f * in_f / (out_f + in_f)))


@torch.no_grad()
def whitened_svd(W, H, k, damp=1e-4):
    """Truncation-aware SVD (SVD-LLM): minimizes ||(W - AB) X||_F with H = X X^T.
    Returns balanced factors A [out, k], B [k, in] in fp32."""
    W = W.float()
    H = H.double()
    lam = damp * torch.mean(torch.diag(H))
    eye = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    for _ in range(6):
        S, info = torch.linalg.cholesky_ex(H + lam * eye)
        if info.item() == 0:
            break
        lam *= 10
    S = S.float()
    U, sig, Vh = torch.linalg.svd(W @ S, full_matrices=False)
    U, sig, Vh = U[:, :k], sig[:k], Vh[:k]
    rs = sig.sqrt()
    A = U * rs.unsqueeze(0)
    B = torch.linalg.solve_triangular(S, rs.unsqueeze(1) * Vh, upper=False, left=False)
    return A, B


def quantize_matrix(W, H, cfg):
    return gptq_quantize(W, H, cfg["bits"], groupsize=cfg.get("groupsize", 128),
                         sym=cfg.get("sym", False), actorder=cfg.get("actorder", True),
                         mse=cfg.get("mse", False), percdamp=cfg.get("percdamp", 0.01))


def cfg_bits(rows, cols, cfg):
    return quant_bits(rows, cols, cfg["bits"], cfg.get("groupsize", 128), cfg.get("sym", False))


@torch.no_grad()
def prune_mlp(mlp, H_down, ratio):
    """Remove `ratio` of the intermediate channels of a gated MLP.

    Score (OBS, per channel): ||W_down[:, j]||^2 / [H^-1]_jj, H = Hessian of
    the down_proj input. down_proj is then refit on the kept channels by
    least squares, W' = W H[:, K] H[K, K]^-1 (no gradient training)."""
    Wd = mlp.down_proj.weight.float()
    n = Wd.shape[1]
    keep_n = n - int(round(ratio * n))
    H = H_down.double()
    lam = 0.01 * torch.mean(torch.diag(H))
    Hd = H + lam * torch.eye(n, device=H.device, dtype=H.dtype)
    Hinv_diag = torch.diag(torch.cholesky_inverse(torch.linalg.cholesky(Hd)))
    score = (Wd.double() ** 2).sum(0) / Hinv_diag
    keep = torch.sort(torch.topk(score, keep_n).indices).values
    # least-squares compensation of down_proj
    HK = Hd[keep][:, keep]
    rhs = Hd[:, keep]                                   # [n, K]
    Wnew = torch.linalg.solve(HK, (Wd.double() @ rhs).t()).t()   # [out, K]
    dev, dt = mlp.down_proj.weight.device, mlp.down_proj.weight.dtype

    def new_linear(w):
        lin = nn.Linear(w.shape[1], w.shape[0], bias=False, device=dev, dtype=dt)
        lin.weight.data.copy_(w)
        return lin

    mlp.gate_proj = new_linear(mlp.gate_proj.weight[keep])
    mlp.up_proj = new_linear(mlp.up_proj.weight[keep])
    mlp.down_proj = new_linear(Wnew.to(dt))
    return keep_n


@torch.no_grad()
def compress(model, tokenizer, spec):
    calib = get_calib(tokenizer, spec.get("nsamples", 128), spec.get("seqlen", 2048), spec.get("seed", 0))
    qrules = spec.get("quant") or []
    svd = spec.get("svd") or {}
    prune = spec.get("prune") or {}
    st = model._stats
    st.setdefault("svd_params_removed", 0)
    st.setdefault("pruned_params", 0)

    def process(idx, layer, xs, kw):
        for parent_name, names in STAGES:
            parent = getattr(layer, parent_name)
            # structured MLP pruning happens right before gate/up are quantized
            pcfg = match(prune.get("rules"), idx, "mlp") if names[0] == "gate_proj" else None
            mods = {n: getattr(parent, n) for n in names}
            hook_mods = dict(mods)
            if pcfg:
                hook_mods["down_proj"] = parent.down_proj
            Hs = hessians_for(layer, {n: m for n, m in hook_mods.items()}, xs, kw)
            if pcfg:
                before = sum(getattr(parent, n).weight.numel() for n in ("gate_proj", "up_proj", "down_proj"))
                prune_mlp(parent, Hs.pop("down_proj"), pcfg["ratio"])
                after = sum(getattr(parent, n).weight.numel() for n in ("gate_proj", "up_proj", "down_proj"))
                st["pruned_params"] += before - after
                for n in ("gate_proj", "up_proj", "down_proj"):
                    w = getattr(parent, n).weight
                    set_bits(model, f"{idx}.{parent_name}.{n}", FP_BITS * w.numel())
                mods = {n: getattr(parent, n) for n in names}
            for n, mod in mods.items():
                key = f"{idx}.{parent_name}.{n}"
                W = mod.weight.data
                rows, cols = W.shape
                scfg = match(svd.get("rules"), idx, n)
                if scfg:
                    k = svd_rank(rows, cols, scfg["ratio"])
                    A, B = whitened_svd(W, Hs[n], k)
                    fq = match(svd.get("factor_quant"), idx, n)
                    if fq:
                        Bq = quantize_matrix(B, Hs[n], fq)
                        HA = Bq @ Hs[n] @ Bq.t()
                        A = quantize_matrix(A, HA, fq)
                        B = Bq
                        bits = cfg_bits(k, cols, fq) + cfg_bits(rows, k, fq)
                    else:
                        bits = FP_BITS * k * (rows + cols)
                    setattr(parent, n, LowRankLinear(A.to(W.dtype), B.to(W.dtype)))
                    st["svd_params_removed"] += rows * cols - k * (rows + cols)
                    set_bits(model, key, bits)
                    continue
                qcfg = match(qrules, idx, n)
                if qcfg:
                    mod.weight.data = quantize_matrix(W, Hs[n], qcfg)
                    set_bits(model, key, cfg_bits(rows, cols, qcfg))
            del Hs
        log(f"[layer {idx:2d}] running compression ratio {current_ratio(model):.4f}")

    sequential(model, calib, process)
    base = model._base_linear_numel
    st["svd_ratio"] = st["svd_params_removed"] / base
    st["prune_ratio"] = st["pruned_params"] / base
    st["comp_ratio"] = current_ratio(model)
    st["avg_bits"] = sum(model._linear_bits.values()) / base
    log("[stats] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items()))
    return model
