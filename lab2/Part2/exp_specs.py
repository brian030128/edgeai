"""Named compression specs used for the exploration sweeps (see compress_lib)."""


def q(bits, gs=128, sym=False, mse=False, actorder=True):
    return {"bits": bits, "groupsize": gs, "sym": sym, "mse": mse, "actorder": actorder}


def all_quant(cfg):
    return [{"cfg": cfg}]


SPECS = {
    # --- uniform GPTQ baselines ---
    "q4g128": {"quant": all_quant(q(4, 128))},
    "q4pc": {"quant": all_quant(q(4, -1))},
    "q4pc_mse": {"quant": all_quant(q(4, -1, mse=True))},
    "q3g128": {"quant": all_quant(q(3, 128))},
    "q3pc": {"quant": all_quant(q(3, -1))},
    "q3pc_mse": {"quant": all_quant(q(3, -1, mse=True))},
    # --- SVD / pruning alone ---
    "svd20": {"svd": {"rules": [{"cfg": {"ratio": 0.2}}]}},
    "prune275": {"prune": {"rules": [{"cfg": {"ratio": 0.275}}]}},
    # --- case 2 candidates: 3-bit with large groups (<= 3.04 bits) ---
    "c2_g512": {"quant": all_quant(q(3, 512))},
    "c2_g640": {"quant": all_quant(q(3, 640))},
    # --- case 1 candidates: 4-bit per-row + a few 3-bit g128 matrices ---
    "c1_kv": {"quant": [{"names": ["k_proj", "v_proj"], "cfg": q(3, 128)}, {"cfg": q(4, -1)}]},
    "c1_down4": {"quant": [{"names": ["down_proj"], "layers": [16, 17, 18, 19], "cfg": q(3, 128)},
                           {"cfg": q(4, -1)}]},
    # --- case 3 candidate: SVD 20% + INT8 factors (MLP + o_proj of layers 0-5) ---
    "c3_a": {"svd": {"rules": [{"cfg": {"ratio": 0.2}}],
                     "factor_quant": [{"names": ["gate_proj", "up_proj", "down_proj"], "cfg": q(8, -1)},
                                      {"names": ["o_proj"], "layers": list(range(6)), "cfg": q(8, -1)}]}},
    # --- case 4 candidate: prune 28% MLP channels + 3-bit per-channel ---
    "c4_a": {"prune": {"rules": [{"cfg": {"ratio": 0.28}}]}, "quant": all_quant(q(3, -1))},
    "c4_b": {"prune": {"rules": [{"cfg": {"ratio": 0.29}}]}, "quant": all_quant(q(3, 640))},
    # --- sensitivity: only one projection type at 3-bit per-row, rest fp16 ---
    **{f"sens_{n}": {"quant": [{"names": [n], "cfg": q(3, -1)}]}
       for n in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
    # --- sensitivity by depth: 3-bit per-row on 9 consecutive layers, rest fp16 ---
    **{f"sens_L{a}": {"quant": [{"layers": list(range(a, a + 9)), "cfg": q(3, -1)}]}
       for a in [0, 9, 18, 27]},
    "c4_mse":{"prune": {"rules": [{"cfg": {"ratio": 0.28}}]}, "quant": all_quant(q(3, -1, mse=True))},
}
