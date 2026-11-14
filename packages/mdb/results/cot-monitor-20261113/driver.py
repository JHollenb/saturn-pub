"""Causal CoT-hint monitor on DeepSeek-R1-Distill-Qwen-1.5B (mdb tracker, experiment 3).

One model load (FP32 eager on CUDA, exact upcast of the stored BF16 weights, TF32 off), stored chains
re-encoded (no generation). For each item and chain fraction f (1.0 = the full chain; 0.25/0.5/0.75 =
the chain cut there, then the answer suffix), at the answer row (the last suffix token, "("):

  mdb (``mdb.attach`` + ``Run``):
    - symbols.table with the context partitioned into declared symbols (sink, question, hint,
      template, chain, suffix); consumer M[X, R] (X = hinted letter, R = the strongest other letter
      at that point). Per binding: attention mass and exact to_consumer (P19). The hint binding's
      per-(layer, head) mass and effect grids.
    - native ablation: p[h, row, hint columns] = 0 at every layer and head of the answer row
      (masked replace at each layer's p port), margin re-measured natively.
    - the old monitor recomputed: L22 KV-head 1 value-weighted hint share (10-15 worker ``route``).
  Hugging Face cache path (same model, incremental: prefill to the end of the chain, then the suffix):
    - base letters; suffix-wide value removal (the hint rows' V cells zeroed for every suffix row);
    - at f = 1: hint-row swaps to the Y prompt's rows (rows from the first letter-differing token to
      the end of the prompt, as the 10-15 A1y): K and V, K only, V only.

Writes one JSON line per (item, f) to OUT. Usage:
  python driver.py MODEL_DIR ITEMS_JSON OUT_JSONL --start I --count N [--budget-s S]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch

SUFFIX = "\n</think>\n\nThe answer is ("
LETTERS = "ABCD"
SITE = (22, 1)


def r6(x):
    return round(float(x), 6)


def letter_regex(xl: str):
    # the 10-15 worker's mentions_hint_letter pattern
    return re.compile(rf"\({xl}\)|\b{xl}\)|option {xl}\b|answer is {xl}\b|answer is \({xl}")


def spans_for(tok, text, ids, body, hint):
    """10-15 worker spans_for: prompt token spans by character offsets."""
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
    ap.add_argument("--only", default="", help="comma-separated uids (smoke)")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    import mdb
    from mdb.analysis import symbols
    from mdb.analysis.forecast import Margin
    from mdb.analysis.reads import head_mix

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 attn_implementation="eager").to("cuda").eval()
    cfg = model.config
    assert cfg.model_type == "qwen2", cfg.model_type
    dbg = mdb.attach(model, tok)
    L, H, G = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    dh = cfg.hidden_size // H
    suf = tok.encode(SUFFIX, add_special_tokens=False)
    letter_ids = []
    for c in LETTERS:
        full = tok.encode(SUFFIX + c, add_special_tokens=False)
        assert full[:-1] == suf
        letter_ids.append(full[-1])
    lt = torch.tensor(letter_ids, device="cuda")
    print(f"COT load {time.time() - t0:.1f}s model_type={cfg.model_type} L={L} H={H}/{G} "
          f"torch={torch.__version__} transformers={transformers.__version__} letters={letter_ids}",
          flush=True)

    body = json.loads(Path(args.items).read_text())
    items = body["items"]
    if args.only:
        want = set(args.only.split(","))
        items = [it for it in items if it["uid"] in want]
    else:
        items = items[args.start:args.start + args.count]
    out = open(args.out, "a")
    emitted = 0

    def letters_of(logits):
        lg = logits.reshape(-1)[lt].double()
        return lg

    def suffix_pass(cache, ids):
        o = model(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache, use_cache=True)
        return letters_of(o.logits[0, -1])

    def kv_layers(cache):
        return [(lay.keys, lay.values) for lay in cache.layers]

    for n_item, it in enumerate(items):
        if time.time() - t0 > args.budget_s:
            print(f"COT budget reached after {n_item} items", flush=True)
            break
        p_ids = tok.encode(it["prompt"], add_special_tokens=False)
        y_ids = tok.encode(it["donor_y"], add_special_tokens=False)
        chain = tok.encode(it["chain"], add_special_tokens=False)
        spans = spans_for(tok, it["prompt"], p_ids, it["body"], it["hint"])
        hint_rows = spans["hint"]
        n_p = len(p_ids)
        swap = None
        if len(y_ids) == n_p:
            diff = [i for i, (a, b) in enumerate(zip(p_ids, y_ids)) if a != b]
            swap = (diff[0], n_p) if diff else None
        checks = {"bos": p_ids[0] == tok.bos_token_id,
                  "n_prompt_matches_old": (it["old"]["swap_rows"] is None
                                           or it["old"]["swap_rows"][1] == n_p),
                  "swap_rows_match_old": (it["old"]["swap_rows"] is None or swap is None
                                          or list(swap) == it["old"]["swap_rows"]),
                  "chain_tokens_reencoded": len(chain), "chain_tokens_old": it["chain_tokens"]}
        xi = LETTERS.index(it["x"])
        for f in it["fracs"]:
            ta = time.time()
            k = len(chain) if f == 1.0 else int(round(f * len(chain)))
            n_pre = n_p + k
            seq = p_ids + chain[:k] + suf
            last = len(seq) - 1
            prefix_text = tok.decode(chain[:k])
            rec = {"uid": it["uid"], "f": f, "k": k, "T": len(seq), "label": it["label"],
                   "set": it["set"], "hint_type": it["hint_type"], "primary": it["primary"],
                   "checks": checks, "n_hint_rows": len(hint_rows),
                   "prefix_names_x_letter": bool(letter_regex(it["x"]).search(prefix_text)),
                   "prefix_names_x_option": it["options"][xi].lower() in prefix_text.lower()}
            # ---- Hugging Face incremental path
            pre = model.model(input_ids=torch.tensor([seq[:n_pre]], device="cuda"),
                              use_cache=True).past_key_values
            base_hf = suffix_pass(copy.deepcopy(pre), suf)
            rec["hf_letters"] = [r6(v) for v in base_hf]
            vz = copy.deepcopy(pre)
            for _, v in kv_layers(vz):
                v[:, :, hint_rows] = 0
            rec["hf_vzero_suffix_letters"] = [r6(v) for v in suffix_pass(vz, suf)]
            del vz
            if f == 1.0 and swap is not None:
                donor = model.model(input_ids=torch.tensor([y_ids], device="cuda"),
                                    use_cache=True).past_key_values
                s0, s1 = swap
                for arm, parts in (("swapKV_y", "kv"), ("swapK_y", "k"), ("swapV_y", "v")):
                    c = copy.deepcopy(pre)
                    for (kk, vv), (dk, dv) in zip(kv_layers(c), kv_layers(donor)):
                        if "k" in parts:
                            kk[:, :, s0:s1] = dk[:, :, s0:s1]
                        if "v" in parts:
                            vv[:, :, s0:s1] = dv[:, :, s0:s1]
                    rec[arm + "_letters"] = [r6(v) for v in suffix_pass(c, suf)]
                    del c
                del donor
            del pre
            # ---- mdb path
            run = dbg.run(seq)
            ck = run.checkpoint()
            run.until("logits")
            lg = letters_of(run.read("logits").to("cuda"))
            rec["mdb_letters"] = [r6(v) for v in lg]
            others = [j for j in range(4) if j != xi]
            ri = max(others, key=lambda j: float(lg[j]))
            rec["R"] = LETTERS[ri]
            margin = float(lg[xi] - lg[ri])
            rec["margin"] = margin
            rec["margin_hf"] = float(base_hf[xi] - base_hf[ri])
            parts = {"sink": spans["sink"], "question": spans["question"], "hint": hint_rows,
                     "template": spans["template"], "chain": list(range(n_p, n_pre)),
                     "suffix": list(range(n_pre, last + 1))}
            for name, rows in parts.items():
                if rows:
                    symbols.declare(run, name, [f"t{s}" for s in rows])
            tab = symbols.table(run, consumer=Margin(last, letter_ids[xi], letter_ids[ri]),
                                limit=16, readers=3)
            rep = tab.report
            rec["bindings"] = {b["symbol"]: {"mass": r6(b["mass"]["value"]),
                                             "to_consumer": (r6(b["to_consumer"]["value"])
                                                             if "value" in b["to_consumer"] else None),
                                             "readers": [[r["name"], r6(r["mass"]["value"]),
                                                          r6(r["effect"].get("value", float("nan")))]
                                                         for r in b["readers"]]}
                               for b in rep["bindings"]}
            rec["consumer_total"] = r6(rep["consumer_total"]["value"])
            rec["unbound_readout"] = r6(rep["unbound"]["readout"]["value"])
            rec["unbound_parts"] = {k: (r6(v["value"]) if isinstance(v, dict) and "value" in v else
                                        (r6(v["value"]["value"]) if isinstance(v, dict) and isinstance(v.get("value"), dict) else None))
                                    for k, v in rep["unbound"].get("parts", {}).items() if k != "meaning"}
            rec["accounting_within_bound"] = rep["accounting"]["within_bound"]
            names = tab.tensors["bindings"]
            bh = names.index("hint")
            rec["hint_effect_grid"] = [[r6(v) for v in tab.tensors["effect"][(last, l)][bh]] for l in range(L)]
            rec["hint_mass_grid"] = [[r6(v) for v in tab.tensors["mass"][(last, l)][bh]] for l in range(L)]
            # old monitor: value-weighted hint share at L22 KV1 (10-15 worker route)
            mix = head_mix(run, last, SITE[0], lazy=True)
            w = mix.weight.reshape(mix.weight.shape[0], H, dh)
            per = H // G
            tot = torch.zeros(mix.p.shape[1], dtype=torch.float64)
            for h in range(SITE[1] * per, (SITE[1] + 1) * per):
                tot += mix.p[h] * (mix.values[mix.kappa[h]] @ w[:, h, :].T).norm(dim=-1)
            rec["mass_L22_kv1"] = r6(tot[hint_rows].sum() / tot.sum())
            del mix, tab
            # native ablation of the answer row's reads of the hint rows (all layers, heads)
            run.restore(ck)
            for layer in range(L):
                run.until(f"L{layer}.h")
                mask = torch.zeros(run.address_shape(f"L{layer}.p"), dtype=torch.bool)
                mask[..., hint_rows] = True
                run.set(f"L{layer}.p", 0.0, "replace", mask=mask)
            run.until("logits")
            ab = letters_of(run.read("logits").to("cuda"))
            rec["mdb_ablate_letters"] = [r6(v) for v in ab]
            rec["margin_ablated"] = float(ab[xi] - ab[ri])
            rec["seconds"] = round(time.time() - ta, 2)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            emitted += 1
            print(f"COT {it['uid']} f={f} T={len(seq)} label={it['label']} X={it['x']} "
                  f"mdb={LETTERS[int(lg.argmax())]} margin={margin:.3f} hf={rec['margin_hf']:.3f} "
                  f"contrib={rec['bindings']['hint']['to_consumer']} abl={rec['margin_ablated']:.3f} "
                  f"vz={rec['hf_vzero_suffix_letters'][xi] - rec['hf_vzero_suffix_letters'][ri]:.3f} "
                  f"mass22={rec['mass_L22_kv1']:.3f} old={it['old']['mass_L22_kv1']} "
                  f"{rec['seconds']}s t={time.time() - t0:.0f}", flush=True)
            del run
        torch.cuda.empty_cache()
    print(f"COT done emitted={emitted} items={n_item + 1 if items else 0} wall={time.time() - t0:.1f}s "
          f"peak_cuda_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
