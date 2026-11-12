"""Saved-only reader for an automatic affine discovery campaign.

No model loading, fitting or native execution. Identity checks protect evidence;
consumer effects and forecast differences remain signed observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def audit(root, publication):
    from mdb.forecast_transaction import load_forecast_tree
    from mdb.wsp import verify_weight_study

    root = Path(root).resolve()
    artifact = publication["artifact"]
    if sha(root / "files.json") != artifact["files_manifest_sha256"] or sha(root / "summary.json") != artifact["summary_sha256"]:
        raise ValueError("native publication hashes differ")
    inventory = read(root / "files.json")
    for relative, reference in inventory.items():
        path = Path(relative)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/")):
            raise ValueError("unsafe result inventory path")
        target = root / path
        if any(p.is_symlink() for p in (target, *target.parents)) or sha(target) != reference["sha256"] or target.stat().st_size != reference["bytes"]:
            raise ValueError("result bytes differ: " + relative)
    summary, request, config = (read(root / name) for name in ("summary.json", "request.json", "config.json"))
    if summary["job_id"] != publication["summary"]["job_id"] or request["config"] != config:
        raise ValueError("job/config binding differs")
    request_body = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(request_body, sort_keys=True).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("request identity differs")
    verification = {}
    for path in sorted((root / "discovery").iterdir()):
        if path.is_dir() and (path / "seal.json").exists():
            verification[path.name] = verify_weight_study(path)
    verification["heldout"] = verify_weight_study(root / "heldout")
    selection = read(root / "discovery/selection.json")
    valid = [row for row in selection["records"] if row["objective"] is not None]
    best = max(valid, key=lambda row: row["objective"])
    if selection["selected"] != best["edit"] or selection["selected_objective"] != best["objective"]:
        raise ValueError("selected candidate differs from recorded selection policy")
    if selection["selected"] != summary["selected"]:
        raise ValueError("heldout summary selected a different edit")
    rows, recomputed = {}, {}
    for arm in ("native", "virtual", "compiled", "restored"):
        record = read(root / "heldout/observations" / (arm + ".json"))
        value = load_forecast_tree(root / "heldout", record["value"])
        if "error" in value["result"]:
            rows[arm] = {"instrument_status": "evidence-incomplete", "error": dict(value["result"]["error"])}
            continue
        observed = value["result"]["observations"]
        correct = sum(int(a == b) for a, b in zip(observed["winner_ids"], observed["target_ids"], strict=True))
        if correct != observed["correct"]:
            raise ValueError("reported correctness differs from retained token decisions")
        logp = sum(observed["target_logp"]) / observed["rows"]
        kl = sum(observed["preservation_kl_rows"]) / len(observed["preservation_kl_rows"])
        rows[arm] = {"correct": correct, "rows": observed["rows"],
            "mean_target_logp": logp, "preservation_kl": kl,
            "objective_recomputed": logp - config["preserve_weight"] * kl,
            "reported_objective": observed["objective"],
            "score_reduction_difference": logp - config["preserve_weight"] * kl - observed["objective"],
            "generated_ids": observed["generated_ids"], "generated_text": observed["generated_text"],
            "instrumented_work": dict(value["instrumented_work"])}
        recomputed[arm] = observed
    comparisons = {}
    for a, b in (("virtual", "compiled"), ("native", "restored"), ("compiled", "native")):
        if a not in recomputed or b not in recomputed:
            continue
        x, y = recomputed[a], recomputed[b]
        comparisons[a + "_minus_" + b] = {"correct_delta": x["correct"] - y["correct"],
            "mean_logp_delta": rows[a]["mean_target_logp"] - rows[b]["mean_target_logp"],
            "preservation_kl_delta": rows[a]["preservation_kl"] - rows[b]["preservation_kl"],
            "max_target_logp_difference": max(abs(u - v) for u, v in zip(x["target_logp"], y["target_logp"], strict=True)),
            "generation_equal": x["generated_ids"] == y["generated_ids"]}
    return {"schema": "mdb-auto-affine-saved-audit-v1", "mechanics_status": "valid",
        "instrument_status": "observed", "terminal_status": "not-evaluated",
        "job_id": summary["job_id"], "request_sha256": request["request_sha256"],
        "files_verified": len(inventory), "selected": selection["selected"],
        "discovery_candidates": len(selection["records"]), "heldout": rows,
        "comparisons": comparisons, "study_verifications": verification,
        "limits": ["saved-only audit; no independent full-model rerun",
                   "parameter restoration is instrumented by the executed runner",
                   "task effects are observations on eight heldout examples, not broad capability certification"]}


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
    print(json.dumps({key: result[key] for key in ("mechanics_status", "job_id", "selected", "comparisons")}, sort_keys=True))
