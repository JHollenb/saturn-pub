"""B2 live watchpoint driver (prereg watch/prereg-b2.json), run through mdb verbs.

Per item: ``run = dbg.run(prompt + chain)``; ``probe.suffix_sweep`` over 34 steps (arms native, M2,
CM1, M3; exact answer-row shares; null_vs_fresh at k = 0 and K); ``rows.row_table`` once (the
next-token lens and per-row hint ablation over the chain, co-readout); the watch predicates are
evaluated by ``breakpoints.replay`` over the trace. On spot items (index % 10 == 0): ``Run.feed``
spot-check at k = round(K/2) and K (native letters + mdb p-zero hint ablation at the answer row)
and ``row_table(check=True)``.

``--smoke`` prints only seconds, memory and exactness diffs (no scores). Writes one JSON line per
item (gzip at the end is done by the caller). Usage:
  python driver_b2.py MODEL SUBSET R_JSON OUT_JSONL --start I --count N [--budget-s S] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

SUFFIX = "\n</think>\n\nThe answer is ("
LETTERS = "ABCD"
P1 = 'argmax_letter == X and ablated["hint"] < 0'
P2 = 'share["hint"] > share["chain_so_far"]'
P1_M3 = 'argmax_letter == X and ablated["hint_suffix"] < 0'


def r5(x):
    return None if x is None else round(float(x), 5)


def spans_for(tok, text, ids, body, hint):
    """10-15 worker spans_for (as driver.py)."""
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


def steps_for(K):
    return sorted({round(j * K / 33) for j in range(34)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("subset")
    ap.add_argument("rjson")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10 ** 9)
    ap.add_argument("--budget-s", type=float, default=560.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-spot", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import mdb
    from mdb.analysis import probe, rows
    from mdb.engine.breakpoints import replay

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
    print(f"B2 load {time.time() - t0:.1f}s torch={torch.__version__} "
          f"transformers={transformers.__version__} letters={letter_ids} suffix={len(suf)}",
          flush=True)
    items = json.loads(Path(args.subset).read_text())["items"]
    Rfix = json.loads(Path(args.rjson).read_text())["R"]
    out = open(args.out, "a")
    chosen = list(enumerate(items))[args.start:args.start + args.count]
    for index, it in chosen:
        if time.time() - t0 > args.budget_s:
            print(f"B2 budget reached before item {index}", flush=True)
            break
        ta = time.time()
        torch.cuda.reset_peak_memory_stats()
        p_ids = tok.encode(it["prompt"], add_special_tokens=False)
        chain = tok.encode(it["chain"], add_special_tokens=False)
        spans = spans_for(tok, it["prompt"], p_ids, it["body"], it["hint"])
        n_p, K = len(p_ids), len(chain)
        X, R = it["x"], Rfix[it["uid"]]
        steps = steps_for(K)
        run = dbg.run(p_ids + chain)
        tb = time.time()
        sweep = probe.suffix_sweep(run, steps, suf, n_prompt=n_p, bindings=spans,
                                   letters=letter_ids, X=X, R=R, hint="hint",
                                   null_steps=(0, K))
        tc = time.time()
        spot = (index % 10 == 0) and not args.no_spot
        binds = dict(spans, chain=list(range(n_p, n_p + K)))
        rt = rows.row_table(run, binds, letters=letter_ids, X=X, R=R,
                            arms={"hint": spans["hint"]}, chunk=512, check=spot)
        td = time.time()
        # R agreement: the strongest other letter at k = K, native
        top = sweep.records[-1]["letters"]["native"]
        r_job = LETTERS[max((j for j in range(4) if j != LETTERS.index(X)), key=lambda j: top[j])]
        watches = {name: replay(pred, sweep.records, X=X, R=R, mode=mode)
                   for name, pred, mode in (("P1", P1, "edge"), ("P2", P2, "level"),
                                            ("P1_M3", P1_M3, "edge"))}
        checks = {"null_vs_fresh_max_abs": sweep.meta["null_vs_fresh_max_abs"],
                  "trace_check_max_abs": rt.meta.get("trace_check_max_abs"),
                  "R_job": r_job, "R_agrees": r_job == R}
        if spot:
            diffs = []
            for k in (round(0.5 * K), K):
                rec = next(r for r in sweep.records if r["step"] == k) if k in steps else None
                if rec is None:  # k not a probe step: read it with one extra sweep step
                    rec = probe.suffix_sweep(run, [k], suf, n_prompt=n_p, bindings=spans,
                                             letters=letter_ids, X=X, R=R,
                                             arms=("native", "M2")).records[0]
                fed = dbg.run(p_ids + chain[:k] + suf[:1])
                fed.feed(suf[1:])
                ck = fed.checkpoint()
                fed.until("logits")
                lg = fed.read("logits").reshape(-1)[letter_ids].double()
                nat = [float(v) for v in lg]
                fed.restore(ck)
                last = n_p + k + len(suf) - 1
                for layer in range(L):
                    fed.until(f"L{layer}.h")
                    mask = torch.zeros(fed.address_shape(f"L{layer}.p"), dtype=torch.bool)
                    mask[..., spans["hint"]] = True
                    fed.set(f"L{layer}.p", 0.0, "replace", mask=mask)
                fed.until("logits")
                ab = fed.read("logits").reshape(-1)[letter_ids].double()
                xi, ri = LETTERS.index(X), LETTERS.index(R)
                diffs.append({
                    "k": k, "row": last, "supplied": len(fed.supplied_tokens()),
                    "native_max_abs": max(abs(a - b) for a, b in
                                          zip(nat, rec["letters"]["native"])),
                    "M2_abs": abs(float(ab[xi] - ab[ri]) - rec["ablated"]["hint"]),
                })
                del fed
            checks["feed_spot"] = diffs
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        secs = {"run": round(tb - ta, 2), "sweep": round(tc - tb, 2),
                "row_table": round(td - tc, 2), "total": round(time.time() - ta, 2)}
        if args.smoke:
            print("B2SMOKE " + json.dumps({"index": index, "T": n_p + K, "steps": len(steps),
                                          "seconds": secs, "sweep_meta_seconds":
                                          sweep.meta["seconds"], "row_table_seconds":
                                          rt.meta["seconds"], "peak_gib": round(peak, 2),
                                          "checks": checks}), flush=True)
            del run, sweep, rt
            torch.cuda.empty_cache()
            continue
        chain_rows = rt.records[n_p:]
        rec = {
            "index": index, "uid": it["uid"], "label": it["label"], "set": it["set"],
            "cluster": it["cluster"], "hint_type": it["hint_type"], "X": X, "R": R,
            "n_prompt": n_p, "K": K, "n_hint_rows": len(spans["hint"]),
            "margin_1113": it["margin_1113"],
            "sweep": [{"step": r["step"], "frac": r5(r["frac"]), "margin": r5(r["margin"]),
                       "argmax_letter": r["argmax_letter"],
                       "letters": {a: [r5(v) for v in lg] for a, lg in r["letters"].items()},
                       "ablated": {a: r5(v) for a, v in r["ablated"].items()},
                       "ablated_argmax": r["ablated_argmax"],
                       "to_consumer": {b: r5(v) for b, v in r["to_consumer"].items()},
                       "mass": {b: r5(v) for b, v in r["mass"].items()},
                       "attn_total": r5(r["attn_total"])} for r in sweep.records],
            "watches": {n: {k: w[k] for k in ("holds", "hits", "fired_at", "crossings",
                                              "fired_fraction", "undefined")}
                        for n, w in watches.items()},
            "rows": {"margin": [r5(r["margin"]) for r in chain_rows],
                     "ablated_hint": [r5(r["ablated"]["hint"]) for r in chain_rows],
                     "argmax_letter": "".join(r["argmax_letter"] for r in chain_rows),
                     "to_consumer": {b: [r5(r["to_consumer"][b]) for r in chain_rows]
                                     for b in binds},
                     "mass_hint": [r5(r["mass"]["hint"]) for r in chain_rows]},
            "checks": checks, "seconds": secs, "peak_gib": round(peak, 2),
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"B2 item {index} T={n_p + K} steps={len(steps)} secs={secs} peak={peak:.2f}GiB "
              f"null={checks['null_vs_fresh_max_abs']:.2e} R_agrees={checks['R_agrees']}"
              + (f" spot={[(d['native_max_abs'], d['M2_abs']) for d in checks['feed_spot']]}"
                 f" trace_check={checks['trace_check_max_abs']}" if spot else ""), flush=True)
        del run, sweep, rt
        torch.cuda.empty_cache()
    print(f"B2 done {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
