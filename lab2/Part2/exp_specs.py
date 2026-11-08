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
}
