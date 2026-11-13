"""Framework-free native footprint, head-use and output audit with signed trends."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from attention_allocation_audit import array, bf16, competition
from link_attention_symbols import captured_bits, native_array, sha
from mdb_lanes.project import Project
from mdb_lanes.value_coalition import LATE_CELLS, plan


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit(root, output):
    output.mkdir(exist_ok=False)
    inventory = read(root / "output-manifest.json")["files"]
    for item in inventory:
        p = root / item["path"]
        if p.stat().st_size != item["bytes"] or sha(p) != item["sha256"]:
            raise ValueError("native inventory changed")
    project = Project(root / "project", read_only=True)
    experiment = read(root / "experiment.json")
    arms = {p.stem: read(p) for p in (root / "arms").glob("*.json")}
    planned = {row["arm"]: row for row in json.loads(json.dumps(plan()))}
    if sorted(arms) != sorted(row["arm"] for row in plan()):
        raise ValueError("missing/duplicated planned branch")
    parents = experiment["native_parents"]
    source = experiment["source_row"]
    wrong = experiment["wrong_row"]
    if experiment["token_labels"]["recipient"][wrong] != ":":
        raise ValueError("wrong-source control is not the declared colon")
    receipts = {sid: {r.fingerprint: r.to_dict() for r in project.receipts(sid)}
        for sid in ("public-recipient", "public-donor")}
    counts = {"acts": 0, "exact_value_layer_checks": 0, "unchanged_slot_checks": 0,
        "native_reader_prefix_checks": 0, "decisions": 0, "own_feedback_links": 0,
        "canonical_attention_uses": 0, "canonical_readouts": 0, "exact_attention_checks": 0,
        "source_persistence_checks": 0}
    summaries, math_rows, head_rows = {}, [], []
    for name, arm in arms.items():
        config = arm["arm"]
        if config != planned[name]:
            raise ValueError("branch configuration differs from the frozen plan")
        kind = config["kind"]
        side = "donor" if kind in {"donor-native", "reciprocal"} else "recipient"
        reference = "native-" + side
        base_logits = array(root, arms[reference]["trajectory"][0]["logits"])[0]
        base_p = np.exp(base_logits.astype(np.float64) - float(base_logits.max()))
        base_p /= base_p.sum()
        baseline_traces = arms[reference]["trajectory"][0]["traces"]
        sid = arm["session_id"]
        if arm["before_act_cut"] != parents[side]:
            raise ValueError("branch did not return to its common parent")
        first = arm["trajectory"][0]
        before = read(project.store.root / "cuts" / (arm["before_act_cut"] + ".json"))
        after = read(project.store.root / "cuts" / (first["parent_cut"] + ".json"))
        if arm["act"]:
            act = arm["act"]
            if act["parent"] != arm["before_act_cut"] or act["result"] != first["parent_cut"]:
                raise ValueError("Act parent/result differs from native branch")
            for slot in before["slots"]:
                if slot != "values":
                    if before["slots"][slot] != after["slots"][slot]:
                        raise ValueError("Act changed a nonwritten native slot")
                    counts["unchanged_slot_checks"] += 1
            counts["acts"] += 1
        for layer in range(28):
            expected = native_array(project, parents[side], "values", layer).copy()
            if arm["act"]:
                donor = native_array(project, parents[arm["payload_side"]], "values", layer)
                for chosen, kv in config["cells"]:
                    if chosen == layer:
                        payload = donor[:, kv, arm["payload_row"]]
                        if config["dose"] == .5:
                            current = (expected[:, kv, arm["target_row"]].astype("<u4") << 16).view("<f4")
                            incoming = (payload.astype("<u4") << 16).view("<f4")
                            payload = (bf16(current + np.float32(.5) * (incoming - current)).view("<u4") >> 16).astype("<u2")
                        expected[:, kv, arm["target_row"]] = payload
            actual = native_array(project, first["parent_cut"], "values", layer)
            if not np.array_equal(expected, actual):
                raise ValueError(f"wrong complete value footprint: {name}/L{layer}")
            counts["exact_value_layer_checks"] += 1
        for step, row in enumerate(arm["trajectory"]):
            readout = receipts[sid][row["readout_receipt"]]
            commit = receipts[sid][row["commit_receipt"]]
            if (readout["parent"] != row["parent_cut"] or readout["result"] != row["readout_cut"]
                    or commit["parent"] != row["readout_cut"] or commit["result"] != row["result_cut"]):
                raise ValueError("native receipt closure lost")
            logits = array(root, row["logits"])[0]
            if not np.array_equal(native_array(project, row["readout_cut"], "logits"), captured_bits(root, row["logits"])):
                raise ValueError("native retained logits differ")
            if int(logits.argmax()) != row["next_token_id"] or arm["own_ids"][step] != row["next_token_id"]:
                raise ValueError("full vocabulary argmax mismatch")
            if native_array(project, row["result_cut"], "tokens").tolist()[0] != row["input_ids"] + [row["next_token_id"]]:
                raise ValueError("native sample/commit mismatch")
            if step and row["input_ids"] != arm["trajectory"][step-1]["input_ids"] + [arm["own_ids"][step-1]]:
                raise ValueError("future did not use its own token")
            if step:
                for layer in range(28):
                    initial_source = native_array(project, first["parent_cut"], "values", layer)[:, :, source]
                    future_source = native_array(project, row["parent_cut"], "values", layer)[:, :, source]
                    if not np.array_equal(initial_source, future_source):
                        raise ValueError("source carrier changed during its own future")
                    counts["source_persistence_checks"] += 1
            counts["decisions"] += 1
            counts["own_feedback_links"] += int(step > 0)
            lifecycle = row["output_lifecycle"]
            if project.symbol_lifecycle(sid, lifecycle["fingerprint"])["lifecycle"] != lifecycle:
                raise ValueError("canonical output lifecycle differs")
            counts["canonical_readouts"] += 1
        words = competition(array(root, first["logits"])[0], {"cold": 9255, "hot": 4017, "boiling": 49837})
        logits = array(root, first["logits"])[0].astype(np.float64)
        p = np.exp(logits - logits.max()); p /= p.sum()
        mid = (p + base_p) / 2
        jsd = .5 * (np.sum(p * np.log(p / mid)) + np.sum(base_p * np.log(base_p / mid)))
        summary = {"arm": name, "kind": kind, "own_text": arm["own_text"], "cells": config["cells"],
            "words": words, "hot_minus_cold": words["hot"]["logit"] - words["cold"]["logit"],
            "native_reference": reference,
            "parent_margin_effect": float(logits[4017] - logits[9255] - (base_logits[4017] - base_logits[9255])),
            "hot_or_boiling_probability": words["hot"]["probability"] + words["boiling"]["probability"],
            "native_total_variation": float(.5 * np.abs(p - base_p).sum()),
            "native_jsd_nats": float(jsd), "native_logit_l2": float(np.linalg.norm(logits - base_logits)),
            "first_changed_attention_layer": None, "first_changed_head_return_layer": None}
        source_lifecycles = {row["metadata"]["layer"]: row for row in arm["source_symbols"]}
        for trace in first["traces"]:
            layer = trace["layer"]
            q, k, v, a, returns = [array(root, trace[key]) for key in ("query", "key", "value", "attention", "head_return")]
            if a.shape != (1, 12, 1, len(first["input_ids"])):
                raise ValueError("attention geometry differs from its receiver")
            for slot, key in (("keys", "key"), ("values", "value")):
                prior = native_array(project, first["parent_cut"], slot, layer)
                captured = captured_bits(root, trace[key])[:, :, :prior.shape[2]]
                if not np.array_equal(prior, captured):
                    raise ValueError("reader did not consume the installed native cache")
                counts["native_reader_prefix_checks"] += 1
            baseline = baseline_traces[layer]
            if summary["first_changed_attention_layer"] is None and not np.array_equal(a, array(root, baseline["attention"])):
                summary["first_changed_attention_layer"] = layer
            if summary["first_changed_head_return_layer"] is None and not np.array_equal(returns, array(root, baseline["head_return"])):
                summary["first_changed_head_return_layer"] = layer
            scores = bf16(bf16(q @ np.repeat(k, 6, axis=1).swapaxes(-1, -2)) * np.float32(q.shape[-1] ** -.5))
            exp = np.exp(scores - scores.max(-1, keepdims=True))
            predicted_a = bf16(exp / exp.sum(-1, keepdims=True))
            predicted_return = bf16(a @ np.repeat(v, 6, axis=1)).transpose(0, 2, 1, 3)
            math_rows.append({"arm": name, "layer": layer,
                "attention_max_abs": float(np.abs(a - predicted_a).max()),
                "return_max_abs": float(np.abs(returns - predicted_return).max())})
            if layer < 19:
                continue
            lifecycle = source_lifecycles[layer]
            if project.symbol_lifecycle(sid, lifecycle["fingerprint"])["lifecycle"] != lifecycle:
                raise ValueError("canonical head-use record differs")
            if len(lifecycle["uses"]) != 12:
                raise ValueError("head uses incomplete")
            for use in lifecycle["uses"]:
                head, kv = use["metadata"]["query_head"], use["metadata"]["kv_head"]
                if kv != head // 6 or use["receipt"] != first["readout_receipt"]:
                    raise ValueError("head/group/receipt association differs")
                if use["metadata"]["output_binding"] != first["output_lifecycle"]["binding_fingerprint"]:
                    raise ValueError("head use points to another native readout")
                locations = {(loc["slot"], tuple(loc["selector"])) for loc in use["locations"]}
                if locations != {(slot, (layer, 0, kv, source)) for slot in ("keys", "values")}:
                    raise ValueError("head use points to another source address")
                observed = {m["observable"]: m["value"] for m in use["signed_observations"]}
                if observed["source_mass"] != float(a[0, head, 0, source]):
                    raise ValueError("canonical source mass differs from actual attention")
                counts["exact_attention_checks"] += 1
                counts["canonical_attention_uses"] += 1
                head_rows.append({"arm": name, "layer": layer, "head": head, "kv_head": kv,
                    "source_mass": observed["source_mass"], "use_id": use["use_id"],
                    "source_binding": lifecycle["binding_fingerprint"],
                    "source_cut": first["parent_cut"], "output_binding": use["metadata"]["output_binding"],
                    "source_value_return_norm": observed["source_value_return_norm"]})
        summaries[name] = summary
    controls = []
    for name in ("replay-recipient", "sham-late"):
        if not all(np.array_equal(array(root, a["logits"]), array(root, b["logits"]))
                for a, b in zip(arms[name]["trajectory"], arms["native-recipient"]["trajectory"])):
            raise ValueError("same-parent replay or sham failed")
        controls.append({"arm": name, "three_logit_arrays_exact": True})
    m0 = summaries["native-recipient"]["hot_minus_cold"]
    full = summaries["donor-late"]["hot_minus_cold"]
    cells = []
    for layer, kv in LATE_CELLS:
        single, complement = summaries[f"only-L{layer}-KV{kv}"], summaries[f"without-L{layer}-KV{kv}"]
        cells.append({"layer": layer, "kv_head": kv,
            "singleton_effect": single["hot_minus_cold"] - m0,
            "complement_loss": full - complement["hot_minus_cold"],
            "singleton_output": single["own_text"], "complement_output": complement["own_text"]})
    interactions = [{"layer": layer, "paired_effect": summaries[f"only-L{layer}-both"]["hot_minus_cold"] - m0,
        "interaction": summaries[f"only-L{layer}-both"]["hot_minus_cold"]
            - summaries[f"only-L{layer}-KV0"]["hot_minus_cold"]
            - summaries[f"only-L{layer}-KV1"]["hot_minus_cold"] + m0} for layer in range(19, 28)]
    trends = {"native_margin": m0, "full_margin": full, "full_effect": full - m0,
        "cells": cells, "layer_interactions": interactions,
        "KV_group_interaction": full - summaries["only-KV0"]["hot_minus_cold"] - summaries["only-KV1"]["hot_minus_cold"] + m0,
        "sum_singleton_effects": sum(c["singleton_effect"] for c in cells),
        "interpretation": "Conditional effects within one supplied task; complement loss is relative to full transplanted coalition, not individual-head necessity."}
    # Reopen/navigation must not mutate even SQLite analysis metadata.
    for item in inventory:
        if sha(root / item["path"]) != item["sha256"]:
            raise ValueError("audit mutated original native evidence")
    report = {"mechanics_status": "valid", "instrument_status": "native consumer, verified cell footprints and observed head uses",
        "trend_status": "signed per-cell, coalition, control and interaction effects retained",
        "terminal_status": "not-requested; exploratory", "raw_files_reverified": len(inventory),
        "counts": counts, "controls": controls, "canonical_bindings": sum(len(project.symbols(s)) for s in receipts),
        "frameworks_imported": [name for name in ("torch", "transformers", "saturn") if name in sys.modules],
        "math": {"attention_max_abs": max(r["attention_max_abs"] for r in math_rows),
            "return_max_abs": max(r["return_max_abs"] for r in math_rows)}}
    write(output / "audit.json", report)
    write(output / "arms.json", list(summaries.values()))
    write(output / "trends.json", trends)
    write(output / "head-uses.json", head_rows)
    write(output / "arithmetic.json", math_rows)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output.resolve())
