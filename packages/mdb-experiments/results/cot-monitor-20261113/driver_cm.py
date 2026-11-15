"""Chain-mask precedence monitor (CM) on the 11-13 cot-monitor items (prereg-cm.json).

One model load (FP32 eager CUDA, exact upcast of the BF16 weights, TF32 off). Per (item, f):
  mdb (attach + Run): base letters at the answer row; CM1 = p[h, chain cols] = 0 at every layer and
    head of the answer row (no renormalisation); at f = 1 also PL1 = the same on placebo rows.
  HF cache path (one prefill of prompt + chain[:k]):
    CM1r  = answer row only, attention_mask 0 on chain columns (renormalised);
    CM1rs = every suffix row, attention_mask 0 on chain columns (renormalised; 11-04 arm m).
The log carries timing and labels only (no scores). Records go to OUT (jsonl).

Usage: python driver_cm.py MODEL_DIR ITEMS_JSON OUT_JSONL --start I --count N [--budget-s S]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

SUFFIX = "\n</think>\n\nThe answer is ("
LETTERS = "ABCD"


def r6(x):
    return round(float(x), 6)


def spans_for(tok, text, ids, body, hint):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    assert enc["input_ids"] == ids, "prompt re-tokenization mismatch"
    b0 = text.index(body)
    b1 = b0 + len(body)
    h0, h1 = (b1, b1 + len(hint))
    spans = {"sink": [], "question": [], "hint": [], "template": []}
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        if t == 0:
            spans["sink"].append(t)
        elif a < b1 and b > b0:
            spans["question"].append(t)
        elif a < h1 and b > h0:
            spans["hint"].append(t)
        else:
            spans["template"].append(t)
    return spans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10 ** 9)
    ap.add_argument("--budget-s", type=float, default=560.0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    import mdb

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 attn_implementation="eager").to("cuda").eval()
    cfg = model.config
    assert cfg.model_type == "qwen2", cfg.model_type
    dbg = mdb.attach(model, tok)
    L = cfg.num_hidden_layers
    suf = tok.encode(SUFFIX, add_special_tokens=False)
    letter_ids = []
    for c in LETTERS:
        full = tok.encode(SUFFIX + c, add_special_tokens=False)
        assert full[:-1] == suf
        letter_ids.append(full[-1])
    lt = torch.tensor(letter_ids, device="cuda")
    print(f"CM load {time.time() - t0:.1f}s torch={torch.__version__} transformers={transformers.__version__}",
          flush=True)
    items = json.loads(Path(args.items).read_text())["items"]
    if args.only:
        want = set(args.only.split(","))
        items = [it for it in items if it["uid"] in want]
    else:
        items = items[args.start:args.start + args.count]
    out = open(args.out, "a")
    emitted = 0

    def hf_pass(cache, ids, pos0, mask_cols, total):
        """Run ``ids`` at positions pos0.. on ``cache`` (holding pos0 rows); attention_mask over all
        ``total`` columns with zeros at ``mask_cols``; returns letter logits of the last row."""
        am = torch.ones(1, total, dtype=torch.long, device="cuda")
        if mask_cols:
            am[0, mask_cols] = 0
        pos = torch.arange(pos0, pos0 + len(ids), device="cuda")[None]
        o = model(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache, attention_mask=am,
                  position_ids=pos, use_cache=True)
        return o.logits[0, -1][lt].double()

    def mdb_masked(run, ck, cols):
        run.restore(ck)
        for layer in range(L):
            run.until(f"L{layer}.h")
            mask = torch.zeros(run.address_shape(f"L{layer}.p"), dtype=torch.bool)
            mask[..., cols] = True
            run.set(f"L{layer}.p", 0.0, "replace", mask=mask)
        run.until("logits")
        return run.read("logits").reshape(-1).to("cuda")[lt].double()

    n_item = 0
    for n_item, it in enumerate(items):
        if time.time() - t0 > args.budget_s:
            print(f"CM budget reached after {n_item} items", flush=True)
            break
        p_ids = tok.encode(it["prompt"], add_special_tokens=False)
        chain = tok.encode(it["chain"], add_special_tokens=False)
        spans = spans_for(tok, it["prompt"], p_ids, it["body"], it["hint"])
        n_p = len(p_ids)
        xi = LETTERS.index(it["x"])
        pool = spans["question"] + spans["template"]
        for f in it["fracs"]:
            ta = time.time()
            k = len(chain) if f == 1.0 else int(round(f * len(chain)))
            n_pre = n_p + k
            seq = p_ids + chain[:k] + suf
            T = len(seq)
            last = T - 1
            chain_rows = list(range(n_p, n_pre))
            rec = {"uid": it["uid"], "f": f, "k": k, "T": T, "label": it["label"], "set": it["set"],
                   "hint_type": it["hint_type"], "n_chain_rows": len(chain_rows)}
            # ---- HF cache path
            pre = model.model(input_ids=torch.tensor([seq[:n_pre]], device="cuda"), use_cache=True).past_key_values
            if chain_rows:
                rec["hf_cm1rs_letters"] = [r6(v) for v in hf_pass(copy.deepcopy(pre), suf, n_pre, chain_rows, T)]
            ans = copy.deepcopy(pre)
            if len(suf) > 1:
                model.model(input_ids=torch.tensor([suf[:-1]], device="cuda"), past_key_values=ans,
                            position_ids=torch.arange(n_pre, last, device="cuda")[None], use_cache=True)
            base_hf = hf_pass(copy.deepcopy(ans), [suf[-1]], last, [], T)
            rec["hf_letters"] = [r6(v) for v in base_hf]
            if chain_rows:
                rec["hf_cm1r_letters"] = [r6(v) for v in hf_pass(ans, [suf[-1]], last, chain_rows, T)]
            del pre, ans
            # ---- mdb path
            run = dbg.run(seq)
            ck = run.checkpoint()
            run.until("logits")
            lg = run.read("logits").reshape(-1).to("cuda")[lt].double()
            rec["mdb_letters"] = [r6(v) for v in lg]
            if chain_rows:
                rec["mdb_cm1_letters"] = [r6(v) for v in mdb_masked(run, ck, chain_rows)]
            if f == 1.0:
                n = min(len(chain_rows), len(pool))
                rng = random.Random(int(hashlib.sha256(it["uid"].encode()).hexdigest()[:16], 16))
                rows = sorted(rng.sample(pool, n)) if n else []
                rec["placebo_rows"] = len(rows)
                rec["placebo_is_whole_pool"] = n == len(pool)
                if rows:
                    rec["mdb_pl1_letters"] = [r6(v) for v in mdb_masked(run, ck, rows)]
            del run
            rec["seconds"] = round(time.time() - ta, 2)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            emitted += 1
            print(f"CM {hashlib.sha256(it['uid'].encode()).hexdigest()[:8]} f={f} T={T} "
                  f"{rec['seconds']}s t={time.time() - t0:.0f}", flush=True)
        torch.cuda.empty_cache()
    print(f"CM done emitted={emitted} items={n_item + 1 if items else 0} wall={time.time() - t0:.1f}s "
          f"peak_cuda_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
