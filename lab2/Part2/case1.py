"""Case 1: compression ratio > 75%, PPL < 15.

Mixed-precision GPTQ (act-order, asymmetric):
  * 4-bit, one fp16 scale + 4-bit zero per output row, for every target
    Linear layer (4.006 bits/weight -> 74.96%, just short of 75%);
  * down_proj of the four middle layers 16-19 at 3-bit with group size 128.
    Middle-layer MLPs are the least sensitive; moving k/v_proj to 3 bits
    instead costs much more PPL (14.95 vs 14.26).
-> 3.983 bits/weight on average -> 75.1% compression.
"""
from cases_common import q, run


def compress(model, tok, rate=0.75, wbits=4):
    spec = {"quant": [{"names": ["down_proj"], "layers": [16, 17, 18, 19], "cfg": q(3, 128)},
                      {"cfg": q(wbits or 4, -1)}]}
    return run(model, tok, spec, [("compression ratio", "comp_ratio", ">", rate)])
