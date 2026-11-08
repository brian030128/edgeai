"""Case 4: pruning rate >= 20%, overall compression ratio >= 85%, PPL < 50.

1. Structured pruning of 28% of the intermediate channels of every MLP
   (rows of gate/up_proj, columns of down_proj). Channels are ranked by the OBS
   saliency ||W_down[:, j]||^2 / [H^-1]_jj and down_proj is refit on the kept
   channels by least squares. Structured, so no sparse index/mask is stored.
   -> 20.7% of the target parameters removed.
2. 3-bit GPTQ (per-row fp16 scale + 3-bit zero) on all remaining weights.
   -> overall compression 85.1%.
"""
from cases_common import q, run


def compress(model, tok, rate=0.85, wbits=3):
    spec = {"prune": {"rules": [{"cfg": {"ratio": 0.28}}]},
            "quant": [{"cfg": q(wbits or 3, -1)}]}
    return run(model, tok, spec, [("pruning rate", "prune_ratio", ">=", 0.20),
                                  ("overall compression ratio", "comp_ratio", ">=", rate)])
