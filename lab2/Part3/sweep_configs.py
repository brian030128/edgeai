"""Named configurations for the exploration sweeps (see aquant.py)."""

NONE = {"kind": "none"}


def mx(fmt, rule="floor", block=32):
    return {"kind": "mx", "fmt": fmt, "rule": rule, "block": block}


NV = {"kind": "nv", "block": 16}


def cfg(w, a, **kw):
    return {"w_lo": w, "a_lo": a, **kw}


CONFIGS = {
    "fp16": cfg(NONE, NONE),
    # --- one side only ---
    "w_fp4": cfg(mx("fp4"), NONE),
    "w_fp4best": cfg(mx("fp4", "best"), NONE),
    "w_nv": cfg(NV, NONE),
    "w_fp6": cfg(mx("fp6"), NONE),
    "a_fp6": cfg(NONE, mx("fp6")),
    "a_fp4": cfg(NONE, mx("fp4")),
    "a_fp8": cfg(NONE, mx("fp8")),
    # --- full W+A, eff <= 5.3 ---
    "w4a6": cfg(mx("fp4"), mx("fp6")),
    "w4ba6": cfg(mx("fp4", "best"), mx("fp6")),
    "w6a4": cfg(mx("fp6"), mx("fp4")),
    # --- with block-Hadamard rotation ---
    "w4ba6_r32": cfg(mx("fp4", "best"), mx("fp6"), rot=32),
    "w4ba6_r128": cfg(mx("fp4", "best"), mx("fp6"), rot=128),
    "w4ba6_r512": cfg(mx("fp4", "best"), mx("fp6"), rot=512),
    "w6a4_r128": cfg(mx("fp6"), mx("fp4"), rot=128),
    # --- references above budget ---
    "w4ba8": cfg(mx("fp4", "best"), mx("fp8")),
    "wnva6": cfg(NV, mx("fp6")),
}

# ---- round 2: FP8 (E4M3) block scales; budget-feasible combinations ----
NV32 = {"kind": "nv", "fmt": "fp4", "block": 32}
NV16 = {"kind": "nv", "fmt": "fp4", "block": 16}
ANV6 = {"kind": "nv", "fmt": "fp6", "block": 32}
CONFIGS.update({
    "w_nv32": cfg(NV32, NONE),
    "w_nv32_r32": cfg(NV32, NONE, rot=32),
    "wnv32a6": cfg(NV32, mx("fp6")),                        # 5.256
    "wnv32a6_r32": cfg(NV32, mx("fp6"), rot=32),
    "wnv16a6b128": cfg(NV16, mx("fp6", block=128)),         # 5.288
    "wnv16a6b128_r32": cfg(NV16, mx("fp6", block=128), rot=32),
    "wnv32anv6": cfg(NV32, ANV6),                           # 5.263
    "wnv32anv6_r32": cfg(NV32, ANV6, rot=32),
    "w4ba6_r32_ceil": cfg(mx("fp4", "ceil"), mx("fp6"), rot=32),
    "w4ba6best_r32": cfg(mx("fp4", "best"), mx("fp6", "best"), rot=32),
    # per-tile weight promotion fp4 -> fp6 using the 0.1-bit slack (tiles 16x32)
    "w4ba6_r32_mix": cfg(mx("fp4", "best"), mx("fp6"), rot=32,
                         w_hi=mx("fp6", "best"), w_frac=0.045, w_tile_rows=16),
})

# ---- ablations of the final method ----
_F = dict(w_hi=mx("fp6", "best"), w_frac=0.045, w_tile_rows=16)
CONFIGS.update({
    "final_seed1": cfg(mx("fp4", "best"), mx("fp6"), rot=32, rot_seed=1, **_F),
    "final_seed2": cfg(mx("fp4", "best"), mx("fp6"), rot=32, rot_seed=2, **_F),
    "final_norot": cfg(mx("fp4", "best"), mx("fp6"), **_F),
    "final_rows128": cfg(mx("fp4", "best"), mx("fp6"), rot=32, **{**_F, "w_tile_rows": 128}),
    "final_rows1": cfg(mx("fp4", "best"), mx("fp6"), rot=32, **{**_F, "w_tile_rows": 1, "w_frac": 0.03}),
    **{f"wonly_shared_s{s}": cfg(mx("fp4", "best"), NONE, rot=32, rot_seed=s, **_F) for s in range(3)},
    **{f"aonly_shared_s{s}": cfg(NONE, mx("fp6"), rot=32, rot_seed=s) for s in range(3)},
    **{f"perlayer_s{s}": cfg(mx("fp4", "best"), mx("fp6"), rot=32, rot_mode="per_layer", rot_seed=s, **_F)
       for s in range(3)},
    "plainhad": cfg(mx("fp4", "best"), mx("fp6"), rot=32, rot_mode="plain", **_F),
    **{f"search_s{s}": cfg(mx("fp4", "best"), mx("fp6"), rot=32, rot_mode="search", rot_cands=8,
                           rot_seed=s, **_F) for s in range(3)},
    # --- sparse fp16 outliers (top |w| of each layer) + slightly smaller promotion ---
    "ol_norot": cfg(mx("fp4", "best"), mx("fp6"), w_outlier_frac=2e-4, **{**_F, "w_frac": 0.04}),
    **{f"ol_shared_s{s}": cfg(mx("fp4", "best"), mx("fp6"), rot=32, rot_seed=s, w_outlier_frac=2e-4,
                              **{**_F, "w_frac": 0.04}) for s in range(3)},
    **{f"ol_perlayer_s{s}": cfg(mx("fp4", "best"), mx("fp6"), rot=32, rot_mode="per_layer", rot_seed=s,
                                w_outlier_frac=2e-4, **{**_F, "w_frac": 0.04}) for s in range(3)},
    "ol1e3_norot": cfg(mx("fp4", "best"), mx("fp6"), w_outlier_frac=1e-3, **{**_F, "w_frac": 0.02}),
    # --- which group makes the rotation seed-sensitive? (weights only) ---
    **{f"wg_{g}_s{s}": cfg(mx("fp4", "best"), NONE, rot=32, rot_seed=s, rot_groups=(g,), **_F)
       for g in ("qkv", "o", "gu", "down") for s in (1, 2)},
    # --- deterministic (no rotation) refinements, W <= 4.35 bits ---
    "d_nv32row_mix": cfg(NV32, mx("fp6"), w_hi={**ANV6}, w_frac=0.04, w_tile_rows=16),
    "d_nv32t_mix": cfg({**NV32, "g": "tensor"}, mx("fp6"), w_hi={**ANV6, "g": "tensor"},
                       w_frac=0.045, w_tile_rows=16),
    "d_nv32t_mix_ol": cfg({**NV32, "g": "tensor"}, mx("fp6"), w_hi={**ANV6, "g": "tensor"},
                          w_frac=0.04, w_tile_rows=16, w_outlier_frac=2e-4),
    "d_fp8hi_ol": cfg(mx("fp4", "best"), mx("fp6"), w_hi=mx("fp8", "best"), w_frac=0.02,
                      w_tile_rows=16, w_outlier_frac=2e-4),
    "d_ol_rows128": cfg(mx("fp4", "best"), mx("fp6"), w_outlier_frac=2e-4,
                        **{**_F, "w_frac": 0.04, "w_tile_rows": 128}),
    "final_fp8hi": cfg(mx("fp4", "best"), mx("fp6"), rot=32,
                       **{**_F, "w_hi": mx("fp8", "best"), "w_frac": 0.022}),
})
