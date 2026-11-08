"""Case 3: SVD compression ratio >= 20%, overall compression ratio = 50%, PPL < 50.

1. Truncation-aware (activation-whitened) SVD, as in SVD-LLM, on every target
   Linear layer: W ~= A B with rank k = floor(0.8 * m n / (m + n)), i.e. every
   matrix keeps 80% of its parameters -> SVD compression ratio 20.0%.
2. The factors of all MLP projections, and of o_proj in layers 0-5, are
   quantized to INT8 with GPTQ (one fp16 scale + 8-bit zero per row); the other
   factors stay fp16. This lands the overall ratio at ~50.3%.
"""
from cases_common import q, run

MLP = ["gate_proj", "up_proj", "down_proj"]


def compress(model, tok, rate=0.50, wbits=8):
    fq = q(wbits or 8, -1)
    spec = {"svd": {"rules": [{"cfg": {"ratio": 0.2}}],
                    "factor_quant": [{"names": MLP, "cfg": fq},
                                     {"names": ["o_proj"], "layers": list(range(6)), "cfg": fq}]}}
    return run(model, tok, spec, [("SVD compression ratio", "svd_ratio", ">=", 0.20),
                                  ("overall compression ratio", "comp_ratio", "=", rate)])
