"""Independent saved-native audit of the prospective fixed-pair experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from attention_allocation_audit import array, competition
from link_attention_symbols import captured_bits, native_array, sha
from mdb.frozen_value_pair import CAPTURE_LAYERS, arms, cases
from mdb.project import Project


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
            raise ValueError("native inventory differs")
    project = Project(root / "project", read_only=True)
    counters = {"acts": 0, "value_layers": 0, "unchanged_slots": 0, "decisions": 0,
        "own_feedback_links": 0, "reader_prefixes": 0, "canonical_attention_uses": 0,
        "canonical_readouts": 0, "source_persistence": 0, "exact_shams": 0}
    summaries, case_summaries, head_rows, inapplicable = [], [], [], []
    all_sessions = set()
    for expected_case in cases():
        case_id = expected_case["id"]
        skipped = root / "cases" / (case_id + "-inapplicable.json")
        if skipped.exists():
            inapplicable.append(read(skipped))
            continue
        case = read(root / "cases" / (case_id + ".json"))
        if case["case"] != expected_case:
            raise ValueError("prompt variant differs from predeclared context")
        source = case["source_row"]
        parents = case["native_parents"]
        records = {a["arm"]: read(root / "arms" / (case_id + "--" + a["arm"] + ".json")) for a in arms()}
        sessions = {side: "public-" + case_id + "-" + side for side in ("recipient", "donor")}
        all_sessions.update(sessions.values())
        receipts = {sid: {r.fingerprint: r.to_dict() for r in project.receipts(sid)} for sid in sessions.values()}
        r_ids, d_ids = case["token_ids"]["recipient"], case["token_ids"]["donor"]
        if [i for i, (a, b) in enumerate(zip(r_ids, d_ids)) if a != b] != [source] or len(r_ids) != len(d_ids):
            raise ValueError("input contrast differs beyond supplied source")
        local = {}
        for expected_arm in arms():
            short = expected_arm["arm"]
            arm = records[short]
            expected = json.loads(json.dumps({**expected_arm, "arm": case_id + "--" + short}))
            if arm["arm"] != expected:
                raise ValueError("fixed pair/control was retuned")
            side = "donor" if expected_arm["kind"] in {"donor-native", "reciprocal"} else "recipient"
            target = case["wrong_row"] if expected_arm["kind"] == "wrong-target" else source
            payload_row = case["wrong_row"] if expected_arm["kind"] == "wrong-payload" else source
            payload_side = "recipient" if expected_arm["kind"] in {"sham", "reciprocal"} else "donor"
            if (arm["source_row"] != source or arm["target_row"] != target
                    or arm["payload_row"] != payload_row or arm["payload_side"] != payload_side):
                raise ValueError("source, target or donor control addressing differs")
            sid = sessions[side]
            first = arm["trajectory"][0]
            if arm["before_act_cut"] != parents[side] or arm["session_id"] != sid:
                raise ValueError("branch did not use its declared common parent")
            before = read(project.store.root / "cuts" / (parents[side] + ".json"))
            after = read(project.store.root / "cuts" / (first["parent_cut"] + ".json"))
            if arm["act"]:
                if arm["act"]["parent"] != parents[side] or arm["act"]["result"] != first["parent_cut"]:
                    raise ValueError("typed Act closure differs")
                counters["acts"] += 1
                for slot in before["slots"]:
                    if slot != "values":
                        if before["slots"][slot] != after["slots"][slot]:
                            raise ValueError("Act changed a nonwritten slot")
                        counters["unchanged_slots"] += 1
            for layer in range(28):
                value = native_array(project, parents[side], "values", layer).copy()
                if arm["act"]:
                    payload = native_array(project, parents[arm["payload_side"]], "values", layer)
                    for selected_layer, kv in expected_arm["cells"]:
                        if selected_layer == layer:
                            value[:, kv, arm["target_row"]] = payload[:, kv, arm["payload_row"]]
                if not np.array_equal(value, native_array(project, first["parent_cut"], "values", layer)):
                    raise ValueError("complete native value footprint differs")
                counters["value_layers"] += 1
            for step, row in enumerate(arm["trajectory"]):
                readout, commit = [receipts[sid][row[key]] for key in ("readout_receipt", "commit_receipt")]
                if (readout["parent"] != row["parent_cut"] or readout["result"] != row["readout_cut"]
                        or commit["parent"] != row["readout_cut"] or commit["result"] != row["result_cut"]):
                    raise ValueError("native continuation receipt differs")
                if not np.array_equal(native_array(project, row["readout_cut"], "logits"), captured_bits(root, row["logits"])):
                    raise ValueError("retained logits differ from saved native array")
                if int(array(root, row["logits"])[0].argmax()) != row["next_token_id"] or arm["own_ids"][step] != row["next_token_id"]:
                    raise ValueError("full-vocabulary decision differs")
                if native_array(project, row["result_cut"], "tokens").tolist()[0] != row["input_ids"] + [row["next_token_id"]]:
                    raise ValueError("actual native token commit differs")
                if step:
                    previous = arm["trajectory"][step-1]
                    if row["input_ids"] != previous["input_ids"] + [previous["next_token_id"]]:
                        raise ValueError("future did not consume its own output")
                    counters["own_feedback_links"] += 1
                    for layer in range(28):
                        initial = native_array(project, first["parent_cut"], "values", layer)[:, :, source]
                        later = native_array(project, row["parent_cut"], "values", layer)[:, :, source]
                        if not np.array_equal(initial, later):
                            raise ValueError("source value did not persist")
                        counters["source_persistence"] += 1
                lifecycle = row["output_lifecycle"]
                if project.symbol_lifecycle(sid, lifecycle["fingerprint"])["lifecycle"] != lifecycle:
                    raise ValueError("canonical readout differs")
                counters["decisions"] += 1
                counters["canonical_readouts"] += 1
            symbols = {s["metadata"]["layer"]: s for s in arm["source_symbols"]}
            if sorted(symbols) != list(CAPTURE_LAYERS) or [t["layer"] for t in first["traces"]] != list(CAPTURE_LAYERS):
                raise ValueError("declared capture coverage differs")
            for trace in first["traces"]:
                layer = trace["layer"]
                a = array(root, trace["attention"])
                if a.shape != (1, 12, 1, len(first["input_ids"])):
                    raise ValueError("attention shape differs from native receiver")
                for slot, key in (("keys", "key"), ("values", "value")):
                    prior = native_array(project, first["parent_cut"], slot, layer)
                    if not np.array_equal(prior, captured_bits(root, trace[key])[:, :, :prior.shape[2]]):
                        raise ValueError("reader did not consume the actual installed cache")
                    counters["reader_prefixes"] += 1
                lifecycle = symbols[layer]
                if project.symbol_lifecycle(sid, lifecycle["fingerprint"])["lifecycle"] != lifecycle or len(lifecycle["uses"]) != 12:
                    raise ValueError("canonical source lifecycle differs")
                for use in lifecycle["uses"]:
                    h, kv = use["metadata"]["query_head"], use["metadata"]["kv_head"]
                    locations = {(loc["slot"], tuple(loc["selector"])) for loc in use["locations"]}
                    if (kv != h // 6 or locations != {(slot, (layer, 0, kv, source)) for slot in ("keys", "values")}
                            or use["receipt"] != first["readout_receipt"]
                            or use["metadata"]["output_binding"] != first["output_lifecycle"]["binding_fingerprint"]):
                        raise ValueError("head/source/readout association differs")
                    metrics = {o["observable"]: o["value"] for o in use["signed_observations"]}
                    if metrics["source_mass"] != float(a[0, h, 0, source]):
                        raise ValueError("canonical attention mass differs from actual observation")
                    head_rows.append({"case": case_id, "arm": short, "layer": layer, "head": h, "kv_head": kv,
                        "source_mass": metrics["source_mass"], "source_row": source, "use_id": use["use_id"],
                        "binding_fingerprint": lifecycle["binding_fingerprint"], "source_cut": first["parent_cut"],
                        "output_binding": use["metadata"]["output_binding"]})
                    counters["canonical_attention_uses"] += 1
            words = competition(array(root, first["logits"])[0], {"cold": 9255, "hot": 4017, "boiling": 49837})
            row = {"case": case_id, "arm": short, "own_text": arm["own_text"],
                "first_token_id": first["next_token_id"], "first_token_text": first["next_token_text"],
                "hot_minus_cold": words["hot"]["logit"] - words["cold"]["logit"],
                "hot_or_boiling_probability": words["hot"]["probability"] + words["boiling"]["probability"], "words": words,
                "source_row": source, "native_parent": parents[side], "actual_source_cut": first["parent_cut"],
                "output_binding": first["output_lifecycle"]["binding_fingerprint"]}
            local[short] = row
        if not all(np.array_equal(array(root, a["logits"]), array(root, b["logits"])) for a, b in zip(
                records["native-recipient"]["trajectory"], records["sham-pair"]["trajectory"])):
            raise ValueError("pair sham changed native continuation")
        counters["exact_shams"] += 1
        m0, md = local["native-recipient"]["hot_minus_cold"], local["native-donor"]["hot_minus_cold"]
        for name, row in local.items():
            row["parent_reference"] = "native-donor" if name in {"native-donor", "reciprocal-pair"} else "native-recipient"
            row["margin_effect"] = row["hot_minus_cold"] - (md if row["parent_reference"] == "native-donor" else m0)
        case_summaries.append({"case": case_id, "axis": expected_case["axis"], "split": expected_case["split"],
            "source_row": source, "receiver_row": len(r_ids)-1,
            "native_margin": m0, "donor_margin": md, "native_contrast": md-m0,
            "pair_margin": local["pair"]["hot_minus_cold"], "pair_effect": local["pair"]["margin_effect"],
            "full_late_effect": local["full-late"]["margin_effect"],
            "singleton_KV0_effect": local["only-KV0"]["margin_effect"], "singleton_KV1_effect": local["only-KV1"]["margin_effect"],
            "interaction": local["pair"]["hot_minus_cold"]-local["only-KV0"]["hot_minus_cold"]-local["only-KV1"]["hot_minus_cold"]+m0,
            "pair_output": local["pair"]["first_token_text"], "native_output": local["native-recipient"]["first_token_text"],
            "donor_output": local["native-donor"]["first_token_text"], "full_output": local["full-late"]["first_token_text"],
            "reciprocal_effect": local["reciprocal-pair"]["margin_effect"],
            "wrong_payload_effect": local["wrong-payload-pair"]["margin_effect"],
            "wrong_target_effect": local["wrong-target-pair"]["margin_effect"],
            "native_expected_separation": local["native-recipient"]["first_token_id"] == 9255 and local["native-donor"]["first_token_id"] == 4017})
        summaries.extend(local.values())
        print(json.dumps({"audited_case": case_id, "pair_effect": case_summaries[-1]["pair_effect"]}), flush=True)
    for item in inventory:
        if sha(root / item["path"]) != item["sha256"]:
            raise ValueError("read-only audit changed raw evidence")
    report = {"mechanics_status": "valid", "instrument_status": "per-context native controls and fixed pair; supplied source spans",
        "trend_status": "per-context signed effects retained", "terminal_status": "not-requested; exploratory",
        "raw_files_reverified": len(inventory), "counts": counters, "applicable_contexts": len(case_summaries),
        "inapplicable_contexts": inapplicable, "canonical_bindings": sum(len(project.symbols(sid)) for sid in all_sessions),
        "frameworks_imported": [name for name in ("torch", "transformers", "saturn") if name in sys.modules]}
    write(output / "audit.json", report)
    write(output / "arms.json", summaries)
    write(output / "contexts.json", case_summaries)
    write(output / "head-uses.json", head_rows)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output.resolve())
