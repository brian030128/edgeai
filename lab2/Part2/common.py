"""Shared helpers for the four compression cases.

Size accounting
---------------
The compression ratio is measured over the target Linear layers: every
nn.Linear inside the decoder blocks (q/k/v/o_proj, gate/up/down_proj).
``record_base`` stores their original element count; every compression step
then records the *storage bits* of the layer it replaced in
``model._linear_bits``. Original weights are 16-bit, so

    compression ratio = 1 - sum(bits) / (16 * base_numel)

Weights are fake-quantized (de-quantized back to fp16 for the matmul), but the
recorded bits are what a packed representation needs: b bits per integer
weight, plus every fp16 scale, every b-bit zero point, every SVD factor and
anything else the method introduces.
"""
import math
import random

import torch
import torch.nn as nn
from datasets import load_dataset

TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FP_BITS = 16
SCALE_BITS = 16  # fp16 scales


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
#                               size bookkeeping                              #
# --------------------------------------------------------------------------- #
def layer_linears(layer):
    """Target Linear modules of one decoder layer, as {short_name: module}."""
    out = {}
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in TARGET_NAMES:
            out[name] = mod
    return out


def record_base(model):
    """Remember the uncompressed size of the target Linear layers."""
    numel, bits = 0, {}
    for i, layer in enumerate(model.model.layers):
        for name, mod in layer_linears(layer).items():
            n = mod.weight.numel() + (mod.bias.numel() if mod.bias is not None else 0)
            numel += n
            bits[f"{i}.{name}"] = FP_BITS * n
    model._base_linear_numel = numel
    model._base_linear_bits = dict(bits)
    model._linear_bits = bits
    model._stats = {}


def set_bits(model, key, bits):
    model._linear_bits[key] = bits


def current_ratio(model):
    return 1.0 - sum(model._linear_bits.values()) / (FP_BITS * model._base_linear_numel)


def quant_bits(rows, cols, wbits, groupsize, sym):
    """Storage of a b-bit quantized [rows, cols] matrix with per-group (or
    per-row, groupsize=-1) fp16 scale and, if asymmetric, a b-bit zero point."""
    g = cols if groupsize in (-1, None) else groupsize
    n_groups = rows * math.ceil(cols / g)
    return rows * cols * wbits + n_groups * (SCALE_BITS + (0 if sym else wbits))


# --------------------------------------------------------------------------- #
#                               calibration data                              #
# --------------------------------------------------------------------------- #
def get_calib(tokenizer, nsamples=128, seqlen=2048, seed=0):
    """Random seqlen-token windows from the WikiText-2 *train* split
    (the test split is used only by the provided evaluation)."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    rng = random.Random(seed)
    out = []
    for _ in range(nsamples):
        i = rng.randint(0, ids.shape[1] - seqlen - 1)
        out.append(ids[:, i : i + seqlen])
    return torch.cat(out, 0)


# --------------------------------------------------------------------------- #
#                       sequential layer-by-layer driver                      #
# --------------------------------------------------------------------------- #
class _Stop(Exception):
    pass


@torch.no_grad()
def capture_layer0_inputs(model, calib):
    """Run the embedding and return (hidden states of every sample, layer kwargs)."""
    layers = model.model.layers
    dev = next(model.parameters()).device
    store = {"x": [], "kw": None}
    orig = layers[0]

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, **kw):
            store["x"].append(x)
            store["kw"] = kw
            raise _Stop

    layers[0] = Catcher(orig)
    for i in range(calib.shape[0]):
        try:
            model(calib[i : i + 1].to(dev))
        except _Stop:
            pass
    layers[0] = orig
    kw = {k: v for k, v in store["kw"].items() if k not in ("past_key_values", "past_key_value", "use_cache")}
    return torch.cat(store["x"], 0), kw


def _run_layer(layer, xs, kw, bs=8):
    out = torch.empty_like(xs)
    for i in range(0, xs.shape[0], bs):
        y = layer(xs[i : i + bs], **kw)
        out[i : i + bs] = y[0] if isinstance(y, tuple) else y
    return out


@torch.no_grad()
def sequential(model, calib, process, bs=8):
    """For every decoder layer: feed the (already compressed) previous outputs,
    let ``process(idx, layer, xs, kw)`` compress the layer in place, then
    recompute the layer outputs as the next layer's inputs."""
    xs, kw = capture_layer0_inputs(model, calib)
    for idx, layer in enumerate(model.model.layers):
        process(idx, layer, xs, kw)
        xs = _run_layer(layer, xs, kw, bs)
        torch.cuda.empty_cache()
    del xs
    torch.cuda.empty_cache()


@torch.no_grad()
def collect_inputs(layer, mods, xs, kw, fn, bs=8):
    """Forward the layer over xs and call fn(name, input_2d) for each module in
    ``mods`` ({name: module}) on every batch."""
    hooks = []
    for name, m in mods.items():
        def hook(_m, inp, _out, name=name):
            fn(name, inp[0].reshape(-1, inp[0].shape[-1]))
        hooks.append(m.register_forward_hook(hook))
    _run_layer(layer, xs, kw, bs)
    for h in hooks:
        h.remove()


class HessianAccumulator:
    """Running E[x x^T] (scaled by 2) of a Linear layer input, as in GPTQ."""

    def __init__(self, cols, device):
        self.H = torch.zeros(cols, cols, device=device, dtype=torch.float32)
        self.n = 0

    def add(self, x):
        x = x.float()
        t = x.shape[0]
        self.H *= self.n / (self.n + t)
        self.n += t
        x = math.sqrt(2 / self.n) * x
        self.H += x.t() @ x


def hessians_for(layer, mods, xs, kw, bs=8):
    dev = xs.device
    acc = {n: HessianAccumulator(m.weight.shape[1], dev) for n, m in mods.items()}
    collect_inputs(layer, mods, xs, kw, lambda n, x: acc[n].add(x), bs)
    return {n: a.H for n, a in acc.items()}
