"""Laptop smoke of both drivers on tiny random models (CPU, fp32 weights loaded as bf16 by mdb's
attach path is not used here: the tiny models are attached directly). Checks plumbing only.

    python smoke.py OUT_DIR
"""
import json
import re
import sys
from pathlib import Path

import torch

import q3_recheck
import q25_recheck
from mdb.ops.debugger import attach


def tiny(kind, vocab):
    torch.manual_seed(0)
    if kind == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=28,
                          num_attention_heads=16, num_key_value_heads=8, head_dim=8,
                          max_position_embeddings=128, attn_implementation="eager")
        m = Qwen3ForCausalLM(cfg)
    else:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=vocab, hidden_size=48, intermediate_size=4224, num_hidden_layers=28,
                          num_attention_heads=12, num_key_value_heads=2, max_position_embeddings=128,
                          attn_implementation="eager")
        m = Qwen2ForCausalLM(cfg)
    return m.to(torch.bfloat16).eval()


class Tok:
    def __init__(self):
        self.ids = {}

    def encode(self, s, add_special_tokens=False):
        return [self.ids.setdefault(w, len(self.ids) + 5) for w in re.findall(r"\w+|[^\w\s]", s)]


def fake_q3_data(d):
    g = torch.Generator().manual_seed(1)
    fams = ["currency", "city"]
    items, panel, gt = [], [], {}
    for f in fams:
        for i in range(20):
            n = 10 + (i % 3)
            ids = torch.randint(0, 200, (n,), generator=g).tolist()
            span = list(range(n - 1 - (i % 2) - 1, n - 1))
            items.append({"id": f"{f}:{i}", "family": f, "answer": 300 + i, "ids": ids, "span": span,
                          "row": span[-1]})
    for it in items[:3] + items[20:22]:
        fam = [x for x in items if x["family"] == it["family"] and x["id"] != it["id"]]
        panel.append({"id": it["id"], "donors": [fam[0]["id"], fam[1]["id"]]})
        gt[it["id"]] = {"row": it["span"][-1], "layer": 20, "kv": 6}
    json.dump({"items": items}, open(d / "candidates.json", "w"))
    json.dump({"panel": panel}, open(d / "job1-panel.json", "w"))
    json.dump(gt, open(d / "gt.json", "w"))
    prompts = [{"k": k, "ids": [k % 19] + torch.randint(0, 200, (6,), generator=g).tolist(), "t1": 1, "t2": 2,
                "M": 0.0} for k in range(40)]
    json.dump({"prompts": prompts, "top1pct": [{"prompt": k, "row": 0, "orig_position_dM": 0.0,
                                                "orig_global_dM": 0.0} for k in (3, 7)]},
              open(d / "tp1_items.json", "w"))


def main(out):
    out = Path(out)
    d = out / "data"
    d.mkdir(parents=True, exist_ok=True)
    fake_q3_data(d)
    m3 = tiny("qwen3", 400)
    q3_recheck._load = lambda *a, **k: attach(m3, None)
    q3_recheck.main("x", d, out / "q3", 3)
    s = json.load(open(out / "q3" / "q3_summary.json"))
    print("Q3 crosscheck", s["crosscheck"])
    print("Q3 C6", [(r["prompt"], r["pooled"]["label"]) for r in s["C6"]])
    j2 = json.load(open(Path(__file__).parent / "data/j2_items.json"))
    json.dump(j2, open(d / "j2_items.json", "w"))
    m2 = tiny("qwen2", 400)
    dbg = attach(m2, None)
    dbg.tokenizer = Tok()
    q25_recheck._load = lambda *a, **k: dbg
    q25_recheck.main("x", d, out / "q25")
    s = json.load(open(out / "q25" / "q25_records.json"))
    print("Q25 crosscheck", s["crosscheck"])
    r = s["records"][0]["rows"]
    print("Q25 arms", sorted({x["arm"] for x in r}), len(r))


if __name__ == "__main__":
    main(sys.argv[1])
