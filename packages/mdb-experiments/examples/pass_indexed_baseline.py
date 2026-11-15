"""Pass-indexed baseline (R0 drift) as an mdb BenchSession recipe.

This expresses the six-line recipe from `research/papers/pass-indexed-baseline/paper.md`
section 7 in mdb's own primitives, on Qwen2.5-1.5B (base):

    prefill      -> the scene prefix becomes the pass-0 (fixed) parent cut
    stage_input  -> teacher-force the exact question tokens onto a context
    + step         (a committed AR boundary per supplied token)
    step         -> commit the model's OWN greedy answer into the context
    checkpoint   -> the per-turn cut whose cache the next turn continues
    fork / restore -> a retained-cache fork that shares the immutable prefix
                      ancestor, so the fixed baseline and the pass-indexed native
                      read the identical fact-span K/V

For each scene it walks the four preregistered turns. At turn t it scores the
turn's answer token two ways and reports the drift between them:

    (a) pass-indexed native  = continue the retained cache that already holds the
        model's own answers A0..A(t-1) (the trunk) through the question Qt;
    (b) fixed parent         = fork the pass-0 prefix cut and continue it through Qt.

    Drift(t) = logp_rebaselined(answer_t) - logp_fixed(answer_t)

It also reports the fact-span K/V bit-identity canary: the prefix (fact-span)
keys/values read on the pass-indexed trunk and on the fixed fork must be
bit-identical, because both descend from one immutable prefill ancestor and the
prefix is a strict common prefix of every context (identical tokens and RoPE
positions). Expected max-abs diff 0.0.

Numeric regime. mdb's Qwen lane (CausalContextVM) runs BF16; it refuses FP32.
The paper's primary R0 numbers are FP32. This example reproduces the *direction
and magnitude* of the base R0 drift in BF16 (a different numeric regime than the
FP32 primary, the same regime as the paper's 7B replication and its BF16 VM
cross-check), not a bit match.

Run inside an admitted mrun worker that has already loaded the model, or let the
script load it from a local checkpoint path:

    python pass_indexed_baseline.py --model-path /mnt/big/llm-models/Qwen2.5-1.5B \
        --root /mnt/usb-storage/mdb/pass-indexed --scenes S0            # smoke
    python pass_indexed_baseline.py --model-path /mnt/big/llm-models/Qwen2.5-1.5B \
        --root /mnt/usb-storage/mdb/pass-indexed                        # all 4 scenes

`--self-test` runs the recipe's *mechanics* on a tiny random CPU Qwen2 (no model
download, no tokenizer, synthetic token-id scenes). It validates that the
primitives compose and that the fact-span canary is exact; its drift numbers are
meaningless (random weights).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch

from mdb import BenchSession

REPORT_MARKER = "MDB_PASS_INDEXED_BASELINE_GZIP_B64="

# The four preregistered scenes and the four-turn script
# (saturn/experiments/2026-10-02-pass-indexed-baseline/PREREG.md; worker.py).
SCENES = [
    {"id": "S0", "lc": "red", "la": "fox", "rc": "blue", "ra": "cat"},
    {"id": "S1", "lc": "green", "la": "frog", "rc": "brown", "ra": "dog"},
    {"id": "S2", "lc": "black", "la": "bird", "rc": "white", "ra": "sheep"},
    {"id": "S3", "lc": "gray", "la": "wolf", "rc": "pink", "ra": "pig"},
]
BAND = tuple(range(22, 28))  # Qwen2.5-1.5B late K/V interface L22-27 (zero-based)


def _prefix_text(s: dict) -> str:
    return f"On the left is a {s['lc']} {s['la']}. On the right is a {s['rc']} {s['ra']}."


def _turns(s: dict) -> list[dict]:
    return [
        {"name": "T0_identity_left", "suffix": " The animal on the left is a", "answer": s["la"]},
        {"name": "T1_color_right", "suffix": " The color of the animal on the right is", "answer": s["rc"]},
        {"name": "T2_color_left", "suffix": " The color of the animal on the left is", "answer": s["lc"]},
        {"name": "T3_recap_left", "suffix": " The animal on the left was the", "answer": s["la"]},
    ]


def _single_leading_token(tokenizer, word: str) -> int:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is not one leading-space token: {ids}")
    return int(ids[0])


def _suffix_ids(tokenizer, prefix: str, suffix: str) -> list[int]:
    pid = tokenizer.encode(prefix, add_special_tokens=False)
    full = tokenizer.encode(prefix + suffix, add_special_tokens=False)
    if full[: len(pid)] != pid:
        raise ValueError("suffix is not a strict common-prefix fork of the prefix")
    return full[len(pid):]


def _fact_span(tokenizer, prefix: str, s: dict) -> list[int]:
    enc = tokenizer(prefix, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    positions = set()
    for key in ("lc", "la", "rc", "ra"):
        start = prefix.find(" " + s[key]) + 1
        end = start + len(s[key])
        hit = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if len(hit) != 1:
            raise ValueError(f"content word {s[key]!r} did not map to one token: {hit}")
        positions.add(hit[0])
    return sorted(positions)


def _logits(session) -> torch.Tensor:
    return session.resolve_value("logits").detach().float().reshape(-1)


def _logprob(logits: torch.Tensor, token_id: int) -> float:
    return float(torch.log_softmax(logits, dim=-1)[token_id].item())


def _feed(session, token_ids: list[int]) -> None:
    """Teacher-force exact tokens: one supplied-input AR boundary per token."""
    for tok in token_ids:
        session.stage_input(int(tok))
        session.step("token")


def _factspan_kv(session, band, positions) -> torch.Tensor:
    """Concatenate the fact-span K and V rows across the band into one tensor."""
    parts = []
    idx = torch.tensor(positions, dtype=torch.long)
    for layer in band:
        for comp in ("key", "value"):
            full = session.resolve_value(f"kv_cache.layer_{layer}.{comp}")  # [1, n_kv, rows, hd]
            parts.append(full.index_select(2, idx.to(full.device)).detach().float().reshape(-1).cpu())
    return torch.cat(parts)


def run_scene(session, scene_id: str, prefix_ids, turns, band, fact_span, gen_budget, eos_ids):
    """One scene: build the pass-indexed trunk, score each turn fixed vs pass-indexed."""
    trunk = scene_id
    c0 = session.prefill(prefix_ids, branch=trunk)  # pass-0 / fixed parent cut
    passes = []
    canary_maxabs = 0.0
    for t, tn in enumerate(turns):
        suffix, answer = tn["suffix_ids"], tn["answer_id"]

        # (b) fixed-parent continuation: a retained-cache fork of the pure prefix.
        session.fork(f"{trunk}-fix{t}", cut=c0)
        _feed(session, suffix)
        fix_lp = _logprob(_logits(session), answer)
        fix_kv = _factspan_kv(session, band, fact_span)

        # (a) pass-indexed native continuation: continue the retained trunk cache,
        # which already holds the model's own answers A0..A(t-1).
        session.checkout(trunk)
        _feed(session, suffix)
        reb_lp = _logprob(_logits(session), answer)
        reb_kv = _factspan_kv(session, band, fact_span)

        maxabs = float((reb_kv - fix_kv).abs().max().item()) if reb_kv.numel() else 0.0
        canary_maxabs = max(canary_maxabs, maxabs)

        # commit the model's own greedy answer into the trunk (advance the pass).
        answer_ids = []
        for _ in range(gen_budget):
            row = session.step("token")
            answer_ids.append(int(row["token_id"]))
            if int(row["token_id"]) in eos_ids:
                break
        session.checkpoint(f"{trunk}-hist{t + 1}")

        passes.append({
            "pass": t, "turn": tn["name"],
            "fixed_logprob": fix_lp, "rebaselined_logprob": reb_lp,
            "drift_logprob": reb_lp - fix_lp,
            "answer_first_is_gold": bool(answer_ids and answer_ids[0] == answer),
            "factspan_kv_maxabs": maxabs,
        })
    return {"scene": scene_id, "passes": passes, "canary_factspan_kv_maxabs": canary_maxabs}


def _summarize(scene_results):
    by_pass: dict[int, list[float]] = {}
    for sc in scene_results:
        for p in sc["passes"]:
            by_pass.setdefault(p["pass"], []).append(p["drift_logprob"])
    median_drift = {t: statistics.median(v) for t, v in sorted(by_pass.items())}
    ge1 = [d for t, v in by_pass.items() if t >= 1 for d in v]
    return {
        "median_drift_by_pass": median_drift,
        "median_abs_drift_passes_ge1": statistics.median([abs(d) for d in ge1]) if ge1 else None,
        "all_passes_ge1_positive": all(d > 0 for d in ge1) if ge1 else None,
        "canary_factspan_kv_maxabs": max((sc["canary_factspan_kv_maxabs"] for sc in scene_results), default=0.0),
    }


def _load_real(model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval().to(device)
    return model, tok


def _build_real(tok, scene):
    prefix = _prefix_text(scene)
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    turns = _turns(scene)
    for tn in turns:
        tn["suffix_ids"] = _suffix_ids(tok, prefix, tn["suffix"])
        tn["answer_id"] = _single_leading_token(tok, tn["answer"])
    fact_span = _fact_span(tok, prefix, scene)
    eos = {int(tok.eos_token_id)} if tok.eos_token_id is not None else set()
    return prefix_ids, turns, fact_span, eos


def _self_test(root: Path):
    """Mechanics only: tiny random CPU Qwen2, synthetic token-id scenes."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(113)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
        attn_implementation="eager")).eval().to(torch.bfloat16)
    band = tuple(range(model.config.num_hidden_layers))
    scene_results = []
    with BenchSession.from_resident(model, root=str(root / "session"),
                                    session_id="pass-indexed-selftest",
                                    continuation_capacity=64) as session:
        for s in range(2):
            prefix_ids = [1, 5, 2, 9, 3, 7]
            fact_span = [1, 3]  # two positions inside the prefix
            turns = [
                {"name": f"T{t}", "suffix_ids": [4 + t, 6], "answer_id": 7 + t}
                for t in range(4)
            ]
            scene_results.append(run_scene(session, f"X{s}", prefix_ids, turns, band,
                                            fact_span, gen_budget=3, eos_ids=set()))
    summary = _summarize(scene_results)
    report = {"self_test": True, "scenes": scene_results, "summary": summary}
    print(json.dumps(report, indent=2))
    assert summary["canary_factspan_kv_maxabs"] == 0.0, "fact-span K/V canary not bit-identical"
    assert all(math.isfinite(d) for v in [report["scenes"]] for sc in v for p in sc["passes"]
               for d in [p["drift_logprob"]]), "non-finite drift"
    print("SELF-TEST OK: primitives compose; fact-span K/V canary exact (max-abs 0.0).")


def _run_scenes(model, tok, chosen, root, gen_budget, capacity=256, resident_kv_bytes=256 << 20):
    scene_results = []
    with BenchSession.from_resident(model, tok, root=root, model_id="Qwen/Qwen2.5-1.5B",
                                    session_id="pass-indexed-baseline",
                                    continuation_capacity=capacity,
                                    resident_kv_bytes=resident_kv_bytes) as session:
        contract = session.capabilities().get("numerical_contract")
        for scene in chosen:
            prefix_ids, turns, fact_span, eos = _build_real(tok, scene)
            scene_results.append(run_scene(session, scene["id"], prefix_ids, turns, BAND,
                                            fact_span, gen_budget, eos))
    return scene_results, contract


def _worker_entry():
    """Run as an mrun worker: read the sealed config, emit a gzip+b64 report marker."""
    cfg = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(int(cfg.get("num_threads", 4)))
    model, tok = _load_real(cfg["model_path"], cfg.get("device", "cuda"))
    chosen = SCENES if not cfg.get("scenes") else [s for s in SCENES if s["id"] in set(cfg["scenes"])]
    scene_results, contract = _run_scenes(model, tok, chosen, cfg["root"], int(cfg.get("gen_budget", 3)))
    report = {"schema": "mdb-pass-indexed-baseline-example-v1", "model_id": "Qwen/Qwen2.5-1.5B",
              "model_path": cfg["model_path"], "dtype": str(next(model.parameters()).dtype),
              "numerical_contract": contract, "band": list(BAND),
              "gen_budget": int(cfg.get("gen_budget", 3)), "scenes": scene_results,
              "summary": _summarize(scene_results)}
    print(REPORT_MARKER + base64.b64encode(gzip.compress(json.dumps(report).encode())).decode(), flush=True)
    return 0


def main(argv=None):
    if argv is None and os.environ.get("SATURN_RUN_CONFIG") and len(sys.argv) == 1:
        return _worker_entry()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="tiny random CPU model, mechanics only")
    ap.add_argument("--model-path", help="local Qwen2.5-1.5B (base) checkpoint directory")
    ap.add_argument("--root", default="./pass-indexed-session", help="durable session root (new dir)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenes", nargs="*", default=None, help="scene ids to run (default: all four)")
    ap.add_argument("--gen-budget", type=int, default=3, help="greedy answer tokens per turn (FROZEN.json=3)")
    ap.add_argument("--out", help="write the report JSON here")
    args = ap.parse_args(argv)

    if args.self_test:
        _self_test(Path(args.root))
        return 0

    if not args.model_path:
        ap.error("--model-path is required unless --self-test")
    model, tok = _load_real(args.model_path, args.device)
    chosen = SCENES if not args.scenes else [s for s in SCENES if s["id"] in set(args.scenes)]
    scene_results, contract = _run_scenes(model, tok, chosen, args.root, args.gen_budget)
    summary = _summarize(scene_results)
    report = {"schema": "mdb-pass-indexed-baseline-example-v1", "model_id": "Qwen/Qwen2.5-1.5B",
              "dtype": str(next(model.parameters()).dtype), "numerical_contract": contract,
              "band": list(BAND), "gen_budget": args.gen_budget, "scenes": scene_results,
              "summary": summary}
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    md = summary["median_drift_by_pass"]
    print(f"\nmedian drift by pass: " + ", ".join(f"p{t}={md[t]:+.3f}" for t in sorted(md)),
          file=sys.stderr)
    print(f"fact-span K/V canary max-abs: {summary['canary_factspan_kv_maxabs']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
