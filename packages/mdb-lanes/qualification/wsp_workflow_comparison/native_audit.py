"""Independent saved-array consumer/cost reader; no target-model execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from protocol import ARMS, digest, save


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def analyze(config, root):
    bundle = json.loads(Path("audit-input.json").read_text())
    source_roots = {n: Path(p) for n, p in config["saved_roots"].items()}
    checked = 0
    for name, source in source_roots.items():
        pin = bundle["source_pins"][name]
        if sha(source / "files.json") != pin["inventory_sha256"]:
            raise ValueError("saved source inventory differs")
        for filename, h in json.loads((source / "files.json").read_text()).items():
            path = (source / filename).resolve()
            if not path.is_relative_to(source.resolve()) or sha(path) != h:
                raise ValueError("saved source bytes differ")
            checked += 1
    selections = {r["id"]: r for r in bundle["selections"]}
    if any(r["plan_identity"] != bundle["plan"]["identity"] or r["execution_plan_identity"] != config["execution_plan_identity"] for r in selections.values()):
        raise ValueError("selection plan identity differs")
    candidates = {}
    for row in bundle["discovery"]:
        if not row["parameter_drift"] and not row["input_drift"]:
            candidates.setdefault(row["trial"], []).extend(row["probes"])
    for tid, selected in selections.items():
        native = {"edits": [], "result": bundle["packets"][selected["specimen"]["id"]]["baseline"]}
        best = max([native, *candidates.get(tid, [])], key=lambda p: p["result"]["objective"])
        if best["edits"] != selected["selection"] or best["result"]["objective"] != selected["discovery_objective"]:
            raise ValueError("discovery-only selection differs")
    costs = {}
    for arm in ARMS:
        rows = [r for r in bundle["discovery"] if r["arm"] == arm]
        proposals = [r for r in bundle["proposals"] if r["arm"] == arm]
        usages = [u for p in proposals for u in p["editor_receipt"]["usage"]]
        costs[arm] = {"editor_calls": len(proposals), "input_tokens": sum(u.get("input_tokens", 0) for u in usages),
            "output_tokens": sum(u.get("output_tokens", 0) for u in usages),
            "editor_seconds_sum": sum(p["editor_receipt"]["seconds"] for p in proposals),
            "program_seconds": sum(r["program_seconds"] for r in rows),
            "discovery_native_calls": sum(r["native_work"]["started_native_calls"] for r in rows),
            "discovery_native_rows": sum(r["native_work"]["native_rows"] for r in rows),
            "local_projection_replays": sum(r["native_work"].get("local_projection_replays", 0) for r in rows),
            "common_setup_anchor_forwards": sum(r["common_setup"]["anchor_forwards"] for r in rows),
            "incomplete_programs": sum(r["instrument_status"] != "observed" for r in rows),
            "completed_candidate_probes": sum(len(r["probes"]) for r in rows),
            "editor_timeouts": sum(p["editor_receipt"].get("inference_status") == "timeout" for p in proposals),
            "token_usage_complete": all(p["editor_receipt"].get("usage_status") != "unavailable" for p in proposals),
            "token_scope": "known completed calls only; lower bound if a provider call timed out"}
    outcomes, full_panel = [], []
    evaluation = source_roots["evaluate"]
    for row in bundle["evaluation"]:
        selection = selections[row["trial"]]
        if row["selection_identity"] != digest(selection):
            raise ValueError("evaluation selection pin differs")
        totals = {name: {"correct": 0, "task_nll": [], "preservation_kl": []} for name in ("native", "edited", "restored")}
        for batch in row["batches"]:
            directory = evaluation / row["trial"]
            compact = {}
            for name, receipt in batch["outcomes"].items():
                path = directory / receipt["readout"]["path"]
                if sha(path) != receipt["readout"]["sha256"]:
                    raise ValueError("readout pin differs")
                with np.load(path, allow_pickle=False) as saved:
                    arrays = {k: saved[k].copy() for k in saved.files}
                if not all(np.isfinite(a).all() for a in arrays.values()):
                    raise ValueError("nonfinite saved readout")
                if not np.array_equal(arrays["rows"], np.arange(16)):
                    raise ValueError("saved row order differs")
                compact[name] = arrays
                nll = -arrays["correct_logp"][:8].astype(np.float64)
                kl = arrays["kl"][8:].astype(np.float64)
                correct = int(np.count_nonzero(arrays["winner_ids"][:8] == np.asarray(receipt["target_ids"])))
                if correct != receipt["correct"]:
                    raise ValueError("saved target extraction differs")
                totals[name]["correct"] += correct
                totals[name]["task_nll"].extend(nll.tolist())
                totals[name]["preservation_kl"].extend(kl.tolist())
            if not np.array_equal(compact["native"]["logp_sha256"], compact["restored"]["logp_sha256"]):
                raise ValueError("native/restored full-vocabulary digest differs")
            if batch["batch"] == 0:
                native_id = row["specimen"]+"-"+ARMS[0]
                native_path = evaluation / native_id / "native-full.npz"
                edited_path = directory / "edited-full.npz"
                with np.load(native_path, allow_pickle=False) as a, np.load(edited_path, allow_pickle=False) as b:
                    original, edited = a["logp"].astype(np.float64), b["logp"].astype(np.float64)
                    native_digest = hashlib.sha256(a["logp"].tobytes()).digest()
                    edited_digest = hashlib.sha256(b["logp"].tobytes()).digest()
                if native_digest != compact["native"]["logp_sha256"].tobytes() or edited_digest != compact["edited"]["logp_sha256"].tobytes():
                    raise ValueError("full audit panel differs from compact readout")
                kl = (np.exp(original)*(original-edited)).sum(-1)
                targets = np.asarray(batch["outcomes"]["edited"]["target_ids"])
                correct = edited[np.arange(8), targets]
                if not np.array_equal(correct, compact["edited"]["correct_logp"][:8]):
                    raise ValueError("full-panel target logp differs")
                full_panel.append({"trial": row["trial"], "task_nll_fp64": float(-correct.mean()),
                    "preservation_kl_fp64": float(kl[8:].mean()),
                    "maximum_kl_reduction_difference_from_native_fp32": float(np.max(np.abs(kl-compact["edited"]["kl"])))})
        outcomes.append({"trial": row["trial"], "arm": row["arm"], "specimen": row["specimen"], "selection": row["selection"],
            "outcomes": {name: {"correct": values["correct"], "task_rows": len(values["task_nll"]),
                "task_nll": float(np.mean(values["task_nll"])), "preservation_kl": float(np.mean(values["preservation_kl"]))}
                for name, values in totals.items()}, "original_restored_digests_equal": True})
    report = {"schema": "mdb-native-workflow-comparison-audit-v1", "source_pins": bundle["source_pins"],
        "verified_files": checked, "costs": costs, "outcomes": outcomes, "independent_full_vocabulary_panel": full_panel,
        "work": {"target_model_loads": 0, "target_model_forwards": 0}, "mechanics_status": "valid",
        "instrument_status": "observed", "trend_status": "uninterpreted", "terminal_status": "not-evaluated",
        "scope": "three paired authored arithmetic discovery samples; two fresh batches per sample; independent saved arithmetic and selection/cost checks, not independent investigator replication or generation benchmark",
        "limits": ["Full-vocabulary KL independently recomputed on first fresh batch only; compact native KL retained on second batch.",
            "Reported provider input/output tokens and per-call elapsed sums are not dollar costs or campaign wall time.",
            "Same named editor and ceilings, without a provider seed or immutable backend revision."]}
    save(root / "comparison.json", report)
    return {"trials": len(outcomes), "full_vocabulary_audit_rows": len(full_panel), "verified_files": checked, "target_model_forwards": 0}
