"""GPTQ (Frantar et al., 2023) with uniform per-row / per-group quantizers.

Follows the reference implementation (github.com/IST-DASLab/gptq):
blocked column-wise quantization with Hessian-based error feedback.
"""
import math

import torch


class Quantizer:
    """Uniform b-bit quantizer whose parameters are fitted per row of a 2D slab.

    sym=True  : q in [-(2^(b-1)), 2^(b-1)-1], zero point 0
    sym=False : q in [0, 2^b - 1], b-bit zero point
    mse=True  : grid-search a shrink factor of the range (clipping) that
                minimizes the |w - w_hat|^2.4 error, as in GPTQ's --mse option.
    Scales are rounded to fp16 because that is how they are stored.
    """

    def __init__(self, bits, sym=False, mse=False, grid=100, maxshrink=0.8, norm=2.4):
        self.bits, self.sym, self.mse = bits, sym, mse
        self.grid, self.maxshrink, self.norm = grid, maxshrink, norm
        self.maxq = 2 ** bits - 1

    def _params(self, xmin, xmax):
        if self.sym:
            amax = torch.maximum(xmin.abs(), xmax).clamp(min=1e-8)
            scale = 2 * amax / self.maxq
            zero = torch.full_like(scale, (self.maxq + 1) / 2)
        else:
            scale = ((xmax - xmin) / self.maxq).clamp(min=1e-8)
            zero = torch.round(-xmin / scale)
        scale = scale.half().float()
        return scale, zero

    def fit(self, w):
        """w: [rows, cols] -> scale, zero of shape [rows, 1]."""
        w = w.float()
        xmin = w.min(1, keepdim=True)[0].clamp(max=0)
        xmax = w.max(1, keepdim=True)[0].clamp(min=0)
        scale, zero = self._params(xmin, xmax)
        if self.mse:
            best = torch.full((w.shape[0], 1), float("inf"), device=w.device)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                s, z = self._params(p * xmin, p * xmax)
                err = (self.qdq(w, s, z) - w).abs_().pow_(self.norm).sum(1, keepdim=True)
                better = err < best
                if better.any():
                    best = torch.where(better, err, best)
                    scale = torch.where(better, s, scale)
                    zero = torch.where(better, z, zero)
        return scale, zero

    def qdq(self, w, scale, zero):
        q = torch.clamp(torch.round(w / scale) + zero, 0, self.maxq)
        return scale * (q - zero)


@torch.no_grad()
def gptq_quantize(W, H, bits, groupsize=128, sym=False, actorder=True, mse=False,
                  percdamp=0.01, blocksize=128):
    """Quantize W [rows, cols] (any float dtype) given the input Hessian H
    [cols, cols]. Returns the de-quantized weight in W's dtype.

    With actorder, groups are *static* (fitted on the original column order
    before permuting), so the packed format needs no per-column group index.
    """
    dtype = W.dtype
    W = W.float().clone()
    H = H.clone()
    rows, cols = W.shape
    q = Quantizer(bits, sym=sym, mse=mse)
    g = cols if groupsize in (-1, None) else groupsize

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # static group parameters on the original column order, expanded per column
    S = torch.empty_like(W)
    Z = torch.empty_like(W)
    for j in range(0, cols, g):
        s, z = q.fit(W[:, j : j + g])
        S[:, j : j + g] = s
        Z[:, j : j + g] = z

    perm = None
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        S, Z = S[:, perm], Z[:, perm]
        invperm = torch.argsort(perm)

    damp = percdamp * torch.mean(torch.diag(H))
    idx = torch.arange(cols, device=W.device)
    H[idx, idx] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            col = i1 + i
            qw = q.qdq(w, S[:, col], Z[:, col])
            Q1[:, i] = qw
            err = (w - qw) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    if perm is not None:
        Q = Q[:, invperm]
    return Q.to(dtype)


@torch.no_grad()
def rtn_quantize(W, bits, groupsize=128, sym=False, mse=False):
    """Round-to-nearest with the same quantizer (no Hessian)."""
    dtype = W.dtype
    W = W.float()
    rows, cols = W.shape
    g = cols if groupsize in (-1, None) else groupsize
    q = Quantizer(bits, sym=sym, mse=mse)
    out = torch.empty_like(W)
    for j in range(0, cols, g):
        s, z = q.fit(W[:, j : j + g])
        out[:, j : j + g] = q.qdq(W[:, j : j + g], s, z)
    return out.to(dtype)
