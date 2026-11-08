"""Experiment driver: python exp.py NAME [NAME ...]  (specs in exp_specs.py)."""
import json
import sys
import time

import torch

import common
from compress_lib import compress
from evaluate import compression_rate, eval_ppl_wikitext2, load_model
from exp_specs import SPECS


def main():
    out = "results/exp.jsonl"
    for name in sys.argv[1:]:
        t0 = time.time()
        torch.manual_seed(0)
        model, tok = load_model()
        common.record_base(model)
        if name != "base":
            compress(model, tok, SPECS[name])
        t1 = time.time()
        ppl = eval_ppl_wikitext2(model, tok)
        rec = {"name": name, "ppl": round(ppl, 4), "comp_rate": round(compression_rate(model), 5),
               **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in model._stats.items()},
               "t_compress": round(t1 - t0), "t_eval": round(time.time() - t1)}
        print("RESULT " + json.dumps(rec), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
