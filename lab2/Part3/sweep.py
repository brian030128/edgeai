"""python sweep.py NAME [NAME ...]  -- configs in sweep_configs.py.
Loads Qwen3-4B once; restores the original Linear modules between configs."""
import json
import os
import sys
import time

import torch

from aquant import quantize_model, summarize, target_linears
from evaluate import eval_ppl_wikitext2, load_model
from sweep_configs import CONFIGS


def main():
    os.makedirs("results", exist_ok=True)
    model, tok = load_model()
    orig = {key: (parent, n, getattr(parent, n)) for key, parent, n in target_linears(model)}
    for name in sys.argv[1:]:
        t0 = time.time()
        qs = quantize_model(model, CONFIGS[name])
        ppl = eval_ppl_wikitext2(model, tok)
        rec = {"name": name, "ppl": round(ppl, 4), **summarize(qs), "sec": round(time.time() - t0)}
        print("RESULT " + json.dumps(rec), flush=True)
        with open("results/sweep.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        for parent, n, lin in orig.values():
            setattr(parent, n, lin)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
