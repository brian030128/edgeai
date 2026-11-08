"""Calibration-free fake quantization with OCP-MX / NVFP4 block formats.

Element formats (sign + ebits exponent + mbits mantissa, IEEE-like with
subnormals, saturating, round-to-nearest-even):
    fp4  E2M1  max 6       fp6  E2M3  max 7.5     fp6' E3M2 max 28
    fp8  E4M3  max 448     int8 (symmetric, max 127)
Block scaling:
    MX     : 32 elements share an 8-bit power-of-two scale (E8M0)
             -> element bits + 8/32
    NVFP4  : 16 E2M1 elements share an FP8 E4M3 scale, plus one fp32
             per-tensor scale -> 4 + 8/16 = 4.5 bits
Every scale is computed from the tensor being quantized itself (weights
offline, activations on the fly), so no calibration data is involved.
"""
import math

import torch

FORMATS = {
    # name: (ebits, mbits, emax of largest normal, max value)
    "fp4": (2, 1, 2, 6.0),
    "fp6": (2, 3, 2, 7.5),
    "fp6e3": (3, 2, 4, 28.0),
    "fp8": (4, 3, 8, 448.0),
}
ELEM_BITS = {"fp4": 4, "fp6": 6, "fp6e3": 6, "fp8": 8, "int8": 8, "int4": 4}


def fp_round(x, ebits, mbits, maxv):
    """Round x (fp32, already divided by the block scale) to the element grid."""
    bias = 2 ** (ebits - 1) - 1
    emin = 1 - bias
    a = x.abs().clamp(max=maxv)
    e = torch.floor(torch.log2(a.clamp(min=1e-30))).clamp(min=emin)
    step = torch.exp2(e - mbits)
    q = (torch.round(a / step) * step).clamp(max=maxv)
    return torch.copysign(q, x)


def elem_round(x, fmt):
    if fmt.startswith("int"):
        qmax = 2 ** (ELEM_BITS[fmt] - 1) - 1
        return torch.round(x).clamp(-qmax, qmax)
    eb, mb, _, mx = FORMATS[fmt]
    return fp_round(x, eb, mb, mx)


def elem_max(fmt):
    if fmt.startswith("int"):
        return 2 ** (ELEM_BITS[fmt] - 1) - 1
    return FORMATS[fmt][3]


def elem_emax(fmt):
    if fmt.startswith("int"):
        return ELEM_BITS[fmt] - 2   # floor(log2(qmax))
    return FORMATS[fmt][2]


def _blocks(x, block):
    shape = x.shape
    assert shape[-1] % block == 0, (shape, block)
    return x.float().reshape(-1, block), shape


def mx_qdq(x, fmt, block=32, rule="floor"):
    """MX quantize-dequantize along the last dim.

    rule = "floor": OCP spec, X = 2^(floor(log2 amax) - emax_elem)
           "ceil" : smallest power of two with amax / X <= max (never clips)
           "best" : per block, whichever of the two has lower squared error
    """
    xb, shape = _blocks(x, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=2.0 ** -126)
    e_floor = torch.floor(torch.log2(amax)) - elem_emax(fmt)
    if rule == "floor":
        e = e_floor
    else:
        e_ceil = torch.ceil(torch.log2(amax / elem_max(fmt)))
        if rule == "ceil":
            e = e_ceil
        else:
            outs = []
            for ee in (e_floor, e_ceil):
                s = torch.exp2(ee.clamp(-127, 127))
                outs.append(elem_round(xb / s, fmt) * s)
            err0 = (outs[0] - xb).pow(2).sum(-1, keepdim=True)
            err1 = (outs[1] - xb).pow(2).sum(-1, keepdim=True)
            return torch.where(err0 <= err1, outs[0], outs[1]).reshape(shape)
    s = torch.exp2(e.clamp(-127, 127))
    return (elem_round(xb / s, fmt) * s).reshape(shape)


def nv_qdq(x, block=16, fmt="fp4", g_mode="row"):
    """NVFP4-style two-level scaling: element format `fmt`, one FP8 E4M3 scale
    per block (amax / elem_max, so the block maximum is never clipped), and an
    fp32 second-level scale per row of x (per output channel for weights, per
    token for activations) that keeps the E4M3 scales in range."""
    shape = x.shape
    rows = x.float().reshape(-1, shape[-1])
    emax = elem_max(fmt)
    if g_mode == "tensor":
        g = (rows.abs().amax() / (448.0 * emax)).clamp(min=1e-30).expand(rows.shape[0], 1)
    else:
        g = (rows.abs().amax(-1, keepdim=True) / (448.0 * emax)).clamp(min=1e-30)   # [R, 1]
    xb = rows.reshape(rows.shape[0], -1, block)                                 # [R, nb, B]
    s = xb.abs().amax(-1, keepdim=True) / emax
    s = fp_round(s / g[:, :, None], 4, 3, 448.0) * g[:, :, None]
    s = torch.where(s > 0, s, torch.ones_like(s))
    return (elem_round(xb / s, fmt) * s).reshape(shape)


def nvfp4_qdq(x, block=16):
    return nv_qdq(x, block, "fp4")


def qdq(x, spec):
    """spec: {"fmt": "fp4"|..., "kind": "mx"|"nv", "block": 32, "rule": "floor"}"""
    if spec["kind"] == "none":
        return x
    if spec["kind"] == "nv":
        return nv_qdq(x, spec.get("block", 16), spec.get("fmt", "fp4"), spec.get("g", "row"))
    return mx_qdq(x, spec["fmt"], spec.get("block", 32), spec.get("rule", "floor"))


def spec_bits(spec):
    if spec["kind"] == "none":
        return 16.0
    if spec["kind"] == "nv":
        # element bits + E4M3 block scale + fp32 per-row scale (row_len given
        # by the caller via spec["row"]; 2560 = smallest input dim)
        g_bits = 0.0 if spec.get("g", "row") == "tensor" else 32 / spec.get("row", 2560)
        return ELEM_BITS[spec.get("fmt", "fp4")] + 8 / spec.get("block", 16) + g_bits
    return ELEM_BITS[spec["fmt"]] + 8 / spec.get("block", 32)


# --------------------------------------------------------------------------- #
#                     randomized block-Hadamard rotation                       #
# --------------------------------------------------------------------------- #
def hadamard(n, device):
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


class BlockRotation:
    """R = blockdiag(diag(s_i) H) along the input dimension; orthogonal, so
    x W^T = (x R)(W R)^T. Signs come from a fixed seed (data-free)."""

    def __init__(self, n, size, device, seed):
        assert n % size == 0
        if seed is None:                       # plain (deterministic) Hadamard
            self.signs = torch.ones(n, device=device)
        else:
            g = torch.Generator().manual_seed(seed)
            self.signs = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(device)
        self.H = hadamard(size, device)
        self.size = size

    def __call__(self, x):
        dt = x.dtype
        y = (x.float() * self.signs).reshape(*x.shape[:-1], -1, self.size) @ self.H
        return y.reshape(x.shape).to(dt)
