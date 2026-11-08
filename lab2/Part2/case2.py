"""Case 2: compression ratio >= 81%, PPL < 20.

3-bit GPTQ (act-order, asymmetric) with group size 512: one fp16 scale + 3-bit
zero per 512 weights of a row, on every target Linear layer: 3 + 19/512 bits
-> 3.037 bits per weight -> 81.02% compression (per-row scales: PPL 19.2).
"""
from cases_common import q, run


def compress(model, tok, rate=0.81, wbits=3):
    spec = {"quant": [{"cfg": q(wbits or 3, 512)}]}
    return run(model, tok, spec, [("compression ratio", "comp_ratio", ">=", rate)])
