"""Calibration-free weight + activation quantization of Qwen3 Linear layers
with per-tile block-format selection.

Config (dict):
  w_lo, w_hi   : weight block formats (see mxq.qdq spec); w_hi optional
  w_frac       : fraction of weight tiles promoted to w_hi (global, data-free)
  w_tile_rows  : rows per weight tile (tile = w_tile_rows x block columns)
  a_lo, a_hi   : activation block formats; a_hi optional
  a_crest      : promote an activation tile to a_hi if amax/rms > a_crest
  a_tile_tokens: tokens per activation tile (all share one format)
  rot          : block-Hadamard size (0 = none), rot_seed
Format selection uses only the tensor being quantized (weights: the weight
itself; activations: the current input), so nothing is calibrated.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mxq import BlockRotation, qdq, spec_bits

TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class ActStats:
    def __init__(self):
        self.elems = 0
        self.bits = 0.0
        self.hi_tiles = 0
        self.tiles = 0

    def avg_bits(self):
        return self.bits / max(self.elems, 1)


class QLinear(nn.Module):
    def __init__(self, lin, Wq, cfg, rot, stats):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.weight = nn.Parameter(Wq, requires_grad=False)
        self.bias = lin.bias
        self.cfg, self.rot, self.stats = cfg, rot, stats

    def quant_act(self, x):
        cfg = self.cfg
        lo, hi = cfg["a_lo"], cfg.get("a_hi")
        blk = lo.get("block", 32)
        if lo["kind"] == "none":
            return x
        shp = x.shape
        if hi is None:
            y = qdq(x, lo)
            n = x.numel()
            self.stats.elems += n
            self.stats.bits += n * spec_bits(lo)
            return y.to(x.dtype)
        # per-tile dynamic selection: tile = T tokens x blk channels
        T = cfg.get("a_tile_tokens", 1)
        xt = x.reshape(-1, shp[-1]).float()
        ntok = xt.shape[0]
        pad = (-ntok) % T
        if pad:
            xt = torch.cat([xt, xt.new_zeros(pad, shp[-1])], 0)
        v = xt.reshape(-1, T, shp[-1] // blk, blk)            # [tiles_t, T, kb, blk]
        amax = v.abs().amax(dim=(1, 3))
        rms = v.pow(2).mean(dim=(1, 3)).sqrt().clamp(min=1e-30)
        promote = (amax / rms) > cfg["a_crest"]               # [tiles_t, kb]
        ylo = qdq(xt, lo).reshape(v.shape)
        yhi = qdq(xt, hi).reshape(v.shape)
        y = torch.where(promote[:, None, :, None], yhi, ylo).reshape(-1, shp[-1])[:ntok]
        frac = promote.float().mean().item()
        n = x.numel()
        tag_bits = 1.0 / (T * blk)                            # 1-bit format id per tile
        self.stats.elems += n
        self.stats.bits += n * ((1 - frac) * spec_bits(lo) + frac * spec_bits(hi) + tag_bits)
        self.stats.hi_tiles += promote.sum().item()
        self.stats.tiles += promote.numel()
        return y.reshape(shp).to(x.dtype)

    def forward(self, x):
        if self.rot is not None:
            x = self.rot(x)
        return F.linear(self.quant_act(x), self.weight, self.bias)


def target_linears(model):
    for i, layer in enumerate(model.model.layers):
        for pname in ("self_attn", "mlp"):
            parent = getattr(layer, pname)
            for n in TARGET_NAMES:
                if hasattr(parent, n) and isinstance(getattr(parent, n), nn.Linear):
                    yield f"{i}.{pname}.{n}", parent, n


def _tile_err(W, Wq, rows, blk):
    out, inn = W.shape
    d = (W.float() - Wq.float()).pow(2)
    d = d.reshape(out // rows, rows, inn // blk, blk).sum(dim=(1, 3))
    return d                                                    # [out/rows, in/blk]


@torch.no_grad()
def quantize_model(model, cfg, log=print):
    """Replace every target Linear by a QLinear. Returns a stats dict."""
    dev = next(model.parameters()).device
    stats = ActStats()
    w_lo, w_hi = cfg["w_lo"], cfg.get("w_hi")
    rows = cfg.get("w_tile_rows", 1)
    blk = w_lo.get("block", 32)
    rot_size = cfg.get("rot", 0)
    rots = {}
    items = list(target_linears(model))

    mixed = w_hi is not None and cfg.get("w_frac", 0) > 0

    # ---- rotations: one per (layer, input group); q/k/v and gate/up share ----
    GROUP = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "o_proj": "o",
             "gate_proj": "gu", "up_proj": "gu", "down_proj": "down"}
    mode = cfg.get("rot_mode", "shared")      # shared | per_layer | plain | search
    # group -> member weights (references only; dropped before the swap so the
    # original fp16 weights can be freed layer by layer)
    members = {}
    if rot_size:
        for key, parent, n in items:
            members.setdefault((key.split(".")[0], GROUP[n]), []).append(getattr(parent, n).weight.data)

    def make_rot(gkey, nin, j=0):
        if mode == "shared":
            return BlockRotation(nin, rot_size, dev, cfg.get("rot_seed", 0) + nin)
        if mode == "plain":
            return BlockRotation(nin, rot_size, dev, None)
        seed = (cfg.get("rot_seed", 0) * 1000003 + int(gkey[0]) * 97
                + ["qkv", "o", "gu", "down"].index(gkey[1]) * 13 + j * 7919)
        return BlockRotation(nin, rot_size, dev, seed)

    if rot_size:
        for gkey, Ws in members.items():
            nin = Ws[0].shape[1]
            if mode != "search":
                rots[gkey] = make_rot(gkey, nin)
                continue
            # data-free choice: candidate whose rotated weights quantize best
            best, best_err = None, float("inf")
            for j in range(cfg.get("rot_cands", 8)):
                R = make_rot(gkey, nin, j)
                err = 0.0
                for W in Ws:
                    Wr = R(W)
                    err += ((qdq(Wr, w_lo) - Wr.float()).pow(2).sum() / Wr.float().pow(2).sum()).item()
                if err < best_err:
                    best, best_err = R, err
            rots[gkey] = best
    members.clear()
    for key, parent, n in items:
        on = rot_size and GROUP[n] in cfg.get("rot_groups", ("qkv", "o", "gu", "down"))
        getattr(parent, n)._rot = rots.get((key.split(".")[0], GROUP[n])) if on else None

    ol_frac = cfg.get("w_outlier_frac", 0.0)

    def prep(lin, key=None):
        """Returns (W_target, W_lo, [W_hi]) in the (rotated) compute domain.
        With w_outlier_frac, the largest |w| / rms(W) entries of the layer are
        kept exactly (stored sparsely in fp16), the rest is block-quantized."""
        W = lin.weight.data
        S = None
        if ol_frac > 0:
            a = W.float().abs().flatten()
            k = max(1, int(ol_frac * a.numel()))
            idx = torch.topk(a, k).indices
            S = torch.zeros_like(W).flatten()
            S[idx] = W.flatten()[idx]
            S = S.view_as(W)
            W = W - S
        if lin._rot is not None:
            W = lin._rot(W)
            S = lin._rot(S) if S is not None else None
        Wlo = qdq(W, w_lo).to(W.dtype)
        Whi = qdq(W, w_hi).to(W.dtype) if mixed else None
        if S is not None:
            # error is measured on the dense remainder; the sparse part is exact
            Wt = W + S
            Wlo = Wlo + S
            Whi = Whi + S if Whi is not None else None
            W = Wt
        return (W, Wlo) if Whi is None else (W, Wlo, Whi)

    def gain(W, Wlo, Whi):
        # per-tile squared-error reduction, scale-free across layers
        return (_tile_err(W, Wlo, rows, blk) - _tile_err(W, Whi, rows, blk)) / W.float().pow(2).mean()

    # pass 1: per-tile promotion scores only (kept on CPU, one layer at a time)
    gains = []
    if mixed:
        for key, parent, n in items:
            gains.append(gain(*prep(getattr(parent, n))).flatten().cpu())
    thr = None
    if gains:
        allg = torch.cat(gains)
        k = int(cfg["w_frac"] * allg.numel())
        thr = torch.topk(allg, k).values.min() if k > 0 else float("inf")
        del allg

    # pass 2: build quantized weights and swap modules
    wbits = wnum = 0.0
    promoted = tiles = 0
    for key, parent, n in items:
        lin = getattr(parent, n)
        c = prep(lin)
        if len(c) == 3:
            W, Wlo, Whi = c
            m = gain(W, Wlo, Whi) >= thr
            mask = m.repeat_interleave(rows, 0).repeat_interleave(blk, 1)
            Wq = torch.where(mask, Whi, Wlo)
            f = m.float().mean().item()
            promoted += m.sum().item()
            tiles += m.numel()
            bits = (1 - f) * spec_bits(w_lo) + f * spec_bits(w_hi) + 1.0 / (rows * blk)
        else:
            W, Wq = c
            bits = spec_bits(w_lo)
        if ol_frac > 0:
            k = max(1, int(ol_frac * W.numel()))
            bits += k * (16 + 32) / W.numel()       # fp16 value + 32-bit index per outlier
        wbits += bits * W.numel()
        wnum += W.numel()
        rot = lin._rot
        setattr(parent, n, QLinear(lin, Wq.contiguous(), cfg, rot, stats))
        del W, Wq, c
    torch.cuda.empty_cache()
    out = {"w_bits": wbits / wnum, "w_promoted_frac": promoted / max(tiles, 1), "act": stats}
    log(f"[quant] weight bits {out['w_bits']:.4f}, promoted tiles {out['w_promoted_frac']:.4f}")
    return out


def summarize(qs):
    a = qs["act"]
    abits = a.avg_bits()
    return {"w_bits": round(qs["w_bits"], 4), "a_bits": round(abits, 4),
            "a_promoted_frac": round(a.hi_tiles / max(a.tiles, 1), 4),
            "eff_bits": round((qs["w_bits"] + abits) / 2, 4)}
