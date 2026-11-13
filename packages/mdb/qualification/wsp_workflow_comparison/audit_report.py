"""Saved-only independent NumPy consumer and paired cost/quality reader."""
from __future__ import annotations

import numpy as np


def numerical_forward(inputs, weights, order):
    value = np.asarray(inputs, dtype=np.float64).copy()
    for name in order:
        value = value + np.tanh(value) @ np.asarray(weights[name + ".weight"], dtype=np.float64).T
    return value


def analyze(bundle):
    packets = bundle["packets"]
    selections = {r["id"]: r for r in bundle["selections"]}
    rows, costs = [], {}
    for trial in bundle["evaluation"]:
        tid, arm = trial["trial"], trial["arm"]
        packet = packets[trial["specimen"]]
        history = [r for r in bundle["discovery"] if r["trial"] == tid]
        candidates = [{"edits": [], "result": packet["baseline"]}]
        for record in history:
            if not record["parameter_drift"]:
                candidates.extend(record["probes"])
        expected = max(candidates, key=lambda r: r["result"]["objective"])
        if expected["edits"] != selections[tid]["selection"] or expected["edits"] != trial["selection"]:
            raise ValueError("sealed final selection does not follow the declared discovery-only policy")
        weights = {n: np.asarray(w, dtype=np.float64).copy() for n, w in packet["weights"].items()}
        discovery_checks = []
        for record in history:
            for probe in record["probes"]:
                probe_weights = {n: w.copy() for n, w in weights.items()}
                for edit in probe["edits"]:
                    probe_weights[edit["site"] + ".weight"] *= edit["gain"]
                checks = {}
                for split, output in (("discovery", "task_outputs"), ("preservation", "preservation_outputs")):
                    predicted = numerical_forward(packet[split][0], probe_weights, packet["sites"])
                    observed = np.asarray(probe["result"][output], dtype=np.float64)
                    checks[split] = float(np.max(np.abs(predicted-observed)))
                discovery_checks.append({"edits": probe["edits"], "maximum_numpy_output_difference": checks})
        changed = {n: w.copy() for n, w in weights.items()}
        for edit in trial["selection"]:
            changed[edit["site"] + ".weight"] *= edit["gain"]
        native, edited = trial["native"], trial["edited"]
        numerical, scalar_differences = {}, {}
        for label, source_weights, retained in (("native", weights, native), ("edited", changed, edited)):
            for split, input_key, target_key, output_key, mse_key in (
                    ("task", "evaluation_inputs", "evaluation_targets", "task_outputs", "task_mse"),
                    ("preservation", "preservation_inputs", "preservation_targets", "preservation_outputs", "preservation_mse")):
                predicted = numerical_forward(trial[input_key], source_weights, packet["sites"])
                observed = np.asarray(retained[output_key], dtype=np.float64)
                target = np.asarray(trial[target_key], dtype=np.float64)
                if predicted.shape != observed.shape or not np.isfinite(observed).all() or observed.shape != target.shape:
                    raise ValueError("invalid retained native arrays")
                numerical[label + "_" + split] = {"maximum_absolute_difference": float(np.max(np.abs(predicted-observed))),
                    "rms_difference": float(np.sqrt(np.mean((predicted-observed)**2)))}
                scalar_differences[label + "_" + split] = float(np.mean((observed-target)**2)) - retained[mse_key]
        if trial["native"] != trial["restored"]:
            raise ValueError("saved restoration does not reproduce native")
        usages = [u for r in bundle["proposals"] if r["id"] == tid for u in r["editor_receipt"]["usage"]]
        cost = {key: sum(u.get(key, 0) for u in usages) for key in (
            "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")}
        cost.update({"editor_seconds": sum(r["editor_receipt"]["seconds"] for r in bundle["proposals"] if r["id"] == tid),
            "program_seconds": sum(r["program_seconds"] for r in history),
            "native_calls": sum(r["native_work"]["started_native_calls"] for r in history),
            "native_rows": sum(r["native_work"]["native_rows"] for r in history),
            "numerical_code_characters": sum(len(r["proposal"]["analysis_python"]) for r in bundle["proposals"] if r["id"] == tid),
            "script_errors": [r["error"] for r in history if r["error"]],
            "completed_probes": sum(len(r["probes"]) for r in history), "billing_amount": None,
            "billing_scope": "account login exposes usage counters, not an attributable dollar charge"})
        costs[tid] = cost
        rows.append({"trial": tid, "arm": arm, "specimen": trial["specimen"],
            "fixture_kind": trial["oracle_reveal"]["kind"], "selection": trial["selection"],
            "native_task_mse": native["task_mse"], "edited_task_mse": edited["task_mse"],
            "native_preservation_mse": native["preservation_mse"], "edited_preservation_mse": edited["preservation_mse"],
            "target_mse_change": edited["task_mse"]-native["task_mse"],
            "preservation_mse_change": edited["preservation_mse"]-native["preservation_mse"],
            "numpy_consumer_comparisons": numerical, "saved_scalar_differences": scalar_differences,
            "discovery_probe_comparisons": discovery_checks,
            "restored_native_equal": True, "cost": cost})
    by_arm = {}
    for arm in bundle["plan"]["arms"]:
        selected = [r for r in rows if r["arm"] == arm]
        by_arm[arm] = {"specimens": len(selected), "sum_task_mse_change": sum(r["target_mse_change"] for r in selected),
            "sum_preservation_mse_change": sum(r["preservation_mse_change"] for r in selected),
            "input_tokens": sum(r["cost"]["input_tokens"] for r in selected),
            "output_tokens": sum(r["cost"]["output_tokens"] for r in selected),
            "editor_seconds": sum(r["cost"]["editor_seconds"] for r in selected),
            "program_seconds": sum(r["cost"]["program_seconds"] for r in selected),
            "native_calls": sum(r["cost"]["native_calls"] for r in selected),
            "native_rows": sum(r["cost"]["native_rows"] for r in selected),
            "script_errors": sum(len(r["cost"]["script_errors"]) for r in selected)}
    return {"schema": "mdb-paired-workflow-independent-report-v1", "rows": rows, "by_arm": by_arm,
        "mechanics_status": "valid", "trend_status": "uninterpreted", "terminal_status": "not-evaluated",
        "native_torch_forwards": 0, "independent_numpy_model_evaluations":
            4*len(rows) + 2*sum(len(r["discovery_probe_comparisons"]) for r in rows),
        "scope": "three constructed paired specimens; finite gain grammar; cost/quality pilot, not a powered superiority test"}
