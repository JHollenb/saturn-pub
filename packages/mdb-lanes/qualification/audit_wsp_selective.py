"""Audit selective repair from sealed saved outcomes, including full logits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def metrics(logits, targets, anchor):
    """Independent FP64 reductions over the complete retained vocabulary."""
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(anchor, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite full-vocabulary arrays required")
    def logsoftmax(v):
        v = v - v.max(axis=-1, keepdims=True)
        return v - np.log(np.exp(v).sum(axis=-1, keepdims=True))
    logp, base = logsoftmax(x), logsoftmax(y)
    n = len(targets)
    target = logp[np.arange(n), targets]
    winners = x[:n].argmax(axis=-1)
    kl = (np.exp(base[n:]) * (base[n:] - logp[n:])).sum(axis=-1)
    return {"correct": int((winners == targets).sum()), "winner_ids": winners.tolist(),
        "target_logp": target.tolist(), "mean_target_logp": float(target.mean()),
        "preservation_kl_rows": kl.tolist(), "preservation_kl": float(kl.mean())}


def audit(root, publication):
    from mdb_lanes.forecast_transaction import load_forecast_tree
    from mdb_lanes.wsp import verify_weight_study
    from mdb_lanes.wsp_discovery import _plain

    root = Path(root).resolve()
    artifact = publication["artifact"]
    for name, key in (("files.json", "files_manifest_sha256"), ("summary.json", "summary_sha256")):
        if sha(root / name) != artifact[key]:
            raise ValueError("published identity differs")
    inventory = read(root / "files.json")
    for relative, ref in inventory.items():
        p = Path(relative)
        if p.is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/")):
            raise ValueError("unsafe inventory path")
        p = root / p
        if any(q.is_symlink() for q in (p, *p.parents)) or sha(p) != ref["sha256"] or p.stat().st_size != ref["bytes"]:
            raise ValueError("inventory bytes differ: " + relative)
    summary, request, config, inputs, selection = (read(root / name) for name in
        ("summary.json", "request.json", "config.json", "inputs.json", "selection.json"))
    body = {k: v for k, v in request.items() if k != "request_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("request seal differs")
    if config != request["config"] or summary != publication["summary"]:
        raise ValueError("published summary or config differs")
    development = {(r["a"], r["b"]) for r in inputs["discovery"]}
    fresh = {(r["a"], r["b"]) for r in inputs["heldout"]}
    if development & fresh or len(fresh) != len(inputs["heldout"]):
        raise ValueError("task split overlap or duplicate fresh cases")
    if set(inputs["discovery_preservation"]) & set(inputs["heldout_preservation"]):
        raise ValueError("preservation split overlap")
    if any(str((r["a"] + r["b"]) % 10) != r["answer"] for split in ("discovery", "heldout") for r in inputs[split]):
        raise ValueError("oracle labels differ")
    verifications = {name: verify_weight_study(root / name) for name in ("capture", "discovery", "heldout")}

    def observation(stage, arm):
        record = read(root / stage / "observations" / (arm + ".json"))
        return load_forecast_tree(root / stage, record["value"])

    for row in selection["records"]:
        result = _plain(observation("discovery", row["arm"])["result"])
        if "error" not in result:
            if any(result["observations"][k] != row[k] for k in result["observations"]):
                raise ValueError("selection record differs from sealed observation")
    best = max((r for r in selection["records"] if r["arm"] != "whole-write" and r["objective"] is not None),
               key=lambda row: row["objective"])
    if best["candidate"] != selection["selected"] or best["arm"] != selection["selected_arm"] or summary["selected"] != best["candidate"]:
        raise ValueError("selection policy differs")
    logits, rows, generated = {}, {}, {}
    for arm in ("native", "whole-write", "virtual", "compiled", "restored", "wrong-sign"):
        value = observation("heldout", arm)
        if "error" in value["result"]:
            rows[arm] = {"instrument_status": "evidence-incomplete", "error": dict(value["result"]["error"])}
            continue
        observed = value["result"]["observations"]
        logits[arm] = np.asarray(observed["logits"])
        if "native" not in logits:
            raise ValueError("missing native anchor")
        measured = metrics(logits[arm], np.array(observed["target_ids"]), logits["native"])
        if measured["winner_ids"] != list(observed["winner_ids"]) or measured["correct"] != observed["correct"]:
            raise ValueError("native decisions differ from full-vocabulary saved argmax")
        measured["reported_target_logp_max_difference"] = float(np.max(np.abs(np.array(measured["target_logp"]) - observed["target_logp"])))
        measured["reported_kl_max_difference"] = float(np.max(np.abs(np.array(measured["preservation_kl_rows"]) - observed["preservation_kl_rows"])))
        measured["generated_ids"] = [list(r) for r in observed["generated_ids"]]
        measured["generated_text"] = list(observed["generated_text"])
        measured["instrumented_work"] = dict(value["instrumented_work"])
        rows[arm], generated[arm] = measured, measured["generated_ids"]
    comparisons = {}
    for a, b in (("virtual", "compiled"), ("native", "restored"), ("compiled", "native"), ("compiled", "whole-write"), ("wrong-sign", "native")):
        if a not in logits or b not in logits:
            continue
        comparisons[a + "_minus_" + b] = {
            "full_logits_max_abs_difference": float(np.max(np.abs(logits[a] - logits[b]))),
            "first_token_choices_equal": rows[a]["winner_ids"] == rows[b]["winner_ids"],
            "correct_delta": rows[a]["correct"] - rows[b]["correct"],
            "mean_logp_delta": rows[a]["mean_target_logp"] - rows[b]["mean_target_logp"],
            "preservation_kl_delta": rows[a]["preservation_kl"] - rows[b]["preservation_kl"],
            "generation_equal": generated[a] == generated[b]}
    return {"schema": "mdb-selective-saved-audit-v1", "mechanics_status": "valid",
        "instrument_status": "observed", "terminal_status": "not-evaluated",
        "job_id": summary["job_id"], "files_verified": len(inventory),
        "request_sha256": request["request_sha256"], "selected": selection["selected"],
        "heldout": rows, "comparisons": comparisons, "study_verifications": verifications,
        "split_counts": {key: len(inputs[key]) for key in ("discovery", "discovery_preservation", "heldout", "heldout_preservation")},
        "limits": ["saved-only audit, not independent pretrained rerun", "rank bounds correction dimension, not changed-weight sparsity",
            "same arithmetic template, 20 fresh tasks and 12 fresh preservation prompts; no broad capability claim"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, read(args.publication))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("mechanics_status", "selected", "comparisons")}, sort_keys=True))
