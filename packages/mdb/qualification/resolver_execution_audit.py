"""Framework-free independent audit; evaluator spans enter only this post-run tool."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from attention_allocation_audit import array, competition
from link_attention_symbols import captured_bits, native_array, sha
from mdb.attention_source_resolver import CAPTURE_LAYERS, arms
from mdb.project import Project


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit(root, output, evaluator):
    output.mkdir(exist_ok=False)
    inventory = read(root / "output-manifest.json")["files"]
    for item in inventory:
        p = root / item["path"]
        if p.stat().st_size != item["bytes"] or sha(p) != item["sha256"]:
            raise ValueError("native inventory differs")
    references = {row["id"]: row for row in read(evaluator)["contexts"]}
    request = read(root / "request.json")
    project = Project(root / "project", read_only=True)
    counters = {key: 0 for key in ("acts", "value_layers", "unchanged_slots", "decisions",
        "own_feedback_links", "reader_prefixes", "canonical_attention_uses", "canonical_readouts",
        "source_persistence", "exact_shams", "discovery_restores", "discovery_prefixes", "resolver_proof_links")}
    summaries, contexts, head_rows, resolutions, skipped = [], [], [], [], []
    all_sessions = set()
    for expected_case in request["config"]["cases"]:
        case_id = expected_case["id"]
        case = read(root / "cases" / (case_id + ".json"))
        if case["case"] != expected_case or set(expected_case) != {"id", "recipient", "donor"}:
            raise ValueError("runtime case differs from frozen text-only plan")
        if any(set(expected_case[s]) != {"text"} for s in ("recipient", "donor")):
            raise ValueError("runtime case contains annotations")
        parents = case["native_parents"]
        sessions = {s: "public-" + case_id + "-" + s for s in parents}
        all_sessions.update(sessions.values())
        receipts = {sid: {r.fingerprint: r.to_dict() for r in project.receipts(sid)} for sid in sessions.values()}
        for side, resolution in case["resolutions"].items():
            parent = parents[side]
            if resolution["parent"] != parent or resolution["restored_parent"] != parent:
                raise ValueError("discovery restore is not exact")
            receipt = receipts[sessions[side]][resolution["receipt"]]
            if (receipt["parent"] != parent or receipt["result"] != resolution["discovery_cut"]
                    or receipt["steps"] != 22 or len(receipt["transitions"]) != 22
                    or receipt["transitions"][-1]["structural_operation"] != "decoder_layer.20"):
                raise ValueError("resolver did not execute its declared native pass")
            counters["discovery_restores"] += 1
            trace = resolution["trace"]
            for slot, key in (("keys", "key"), ("values", "value")):
                prior = native_array(project, parent, slot, 20)
                if not np.array_equal(prior, captured_bits(root, trace[key])[:, :, :prior.shape[2]]):
                    raise ValueError("discovery reader cache differs from restored parent")
                counters["discovery_prefixes"] += 1
            weights = array(root, trace["attention"])[0, 5, 0]
            if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("invalid native attention weights")
            predicted = int(weights.argmax())
            cached = native_array(project, parent, "values", 20).shape[2]
            nominee = predicted if predicted < cached else None
            decision = resolution["nomination"]
            if (decision["predicted_row"] != predicted or decision["source_row"] != nominee
                    or decision["source_mass"] != float(weights[predicted]) or resolution["cached_rows"] != cached):
                raise ValueError("execution nomination differs from independent native argmax")
            # Only now inspect evaluator annotations. Token labels independently recover ASCII offsets.
            labels = case["token_labels"][side]
            if ''.join(labels) != expected_case[side]["text"]:
                raise ValueError("cannot independently recover evaluator character offsets")
            span = references[case_id][side]["source_span"]
            offset, reference_rows = 0, []
            for index, text in enumerate(labels):
                if offset < span[1] and offset + len(text) > span[0]: reference_rows.append(index)
                offset += len(text)
            if not reference_rows:
                raise ValueError("empty withheld reference mapping")
            resolutions.append({"case": case_id, "side": side, "source_row": nominee,
                "predicted_row": predicted, "reference_rows": reference_rows, "matches_reference": predicted in reference_rows,
                "source_token": labels[predicted], "reference_text": references[case_id][side]["source_word"],
                "source_mass": float(weights[predicted]), "reference_mass": float(weights[reference_rows].sum()),
                "gap": float(np.sort(weights)[-1] - np.sort(weights)[-2]), "cached_rows": cached,
                "receiver_row": len(labels)-1, "discovery_receipt": resolution["receipt"],
                "attention_sha256": trace["attention"]["sha256"], "parent": parent, "wall_s": resolution["wall_s"]})
        local, records = {}, {}
        skipped_names = {row["arm"]["arm"] for row in case["skipped"]}
        skipped.extend({"case": case_id, **row} for row in case["skipped"])
        for config in arms():
            short, kind = config["arm"], config["kind"]
            full_name = case_id + "--" + short
            if full_name in skipped_names: continue
            arm = read(root / "arms" / (full_name + ".json"))
            records[short] = arm
            if arm["arm"] != json.loads(json.dumps({**config, "arm": full_name})):
                raise ValueError("fixed route/control was retuned")
            side = "donor" if kind in {"donor-native", "reciprocal"} else "recipient"
            payload_side = "recipient" if kind in {"sham", "reciprocal"} else "donor"
            source = case["resolutions"][side]["nomination"]["source_row"]
            target = source
            payload_row = case["resolutions"][payload_side]["nomination"]["source_row"]
            if kind == "wrong-target" and target is not None: target -= 1
            if kind == "wrong-payload" and payload_row is not None: payload_row -= 1
            if kind == "fixed-distance":
                target, payload_row = len(case["token_ids"][side])-4, len(case["token_ids"][payload_side])-4
            if (arm["source_row"], arm["target_row"], arm["payload_row"], arm["payload_side"]) != (source, target, payload_row, payload_side):
                raise ValueError("Act addressing did not consume actual resolver decisions")
            sid, first = sessions[side], arm["trajectory"][0]
            if arm["before_act_cut"] != parents[side] or arm["session_id"] != sid:
                raise ValueError("branch did not use declared parent")
            before = read(project.store.root / "cuts" / (parents[side] + ".json"))
            after = read(project.store.root / "cuts" / (first["parent_cut"] + ".json"))
            if arm["act"]:
                if arm["act"]["parent"] != parents[side] or arm["act"]["result"] != first["parent_cut"]:
                    raise ValueError("Act closure differs")
                counters["acts"] += 1
                for slot in before["slots"]:
                    if slot != "values":
                        if before["slots"][slot] != after["slots"][slot]: raise ValueError("nonwritten slot changed")
                        counters["unchanged_slots"] += 1
            for layer in range(28):
                value = native_array(project, parents[side], "values", layer).copy()
                if arm["act"]:
                    payload = native_array(project, parents[payload_side], "values", layer)
                    for selected_layer, kv in config["cells"]:
                        if selected_layer == layer: value[:, kv, target] = payload[:, kv, payload_row]
                if not np.array_equal(value, native_array(project, first["parent_cut"], "values", layer)):
                    raise ValueError("complete Act footprint differs")
                counters["value_layers"] += 1
            for step, row in enumerate(arm["trajectory"]):
                readout, commit = [receipts[sid][row[key]] for key in ("readout_receipt", "commit_receipt")]
                if (readout["parent"], readout["result"], commit["parent"], commit["result"]) != (
                        row["parent_cut"], row["readout_cut"], row["readout_cut"], row["result_cut"]):
                    raise ValueError("continuation receipt differs")
                if not np.array_equal(native_array(project, row["readout_cut"], "logits"), captured_bits(root, row["logits"])):
                    raise ValueError("retained native logits differ")
                if int(array(root, row["logits"])[0].argmax()) != row["next_token_id"] or arm["own_ids"][step] != row["next_token_id"]:
                    raise ValueError("full-vocabulary decision differs")
                if native_array(project, row["result_cut"], "tokens").tolist()[0] != row["input_ids"] + [row["next_token_id"]]:
                    raise ValueError("native commit differs")
                if step:
                    previous = arm["trajectory"][step-1]
                    if row["input_ids"] != previous["input_ids"] + [previous["next_token_id"]]: raise ValueError("not own feedback")
                    counters["own_feedback_links"] += 1
                    if source is not None:
                        for layer in range(28):
                            if not np.array_equal(native_array(project, first["parent_cut"], "values", layer)[:, :, source],
                                    native_array(project, row["parent_cut"], "values", layer)[:, :, source]):
                                raise ValueError("nominated source did not persist")
                            counters["source_persistence"] += 1
                lifecycle = row["output_lifecycle"]
                if project.symbol_lifecycle(sid, lifecycle["fingerprint"])["lifecycle"] != lifecycle: raise ValueError("canonical readout differs")
                counters["decisions"] += 1
                counters["canonical_readouts"] += 1
            symbols = {s["metadata"]["layer"]: s for s in arm["source_symbols"]}
            if sorted(symbols) != (list(CAPTURE_LAYERS) if source is not None else []): raise ValueError("symbol coverage differs")
            for trace in first["traces"]:
                layer = trace["layer"]
                a = array(root, trace["attention"])
                if a.shape != (1, 12, 1, len(first["input_ids"])): raise ValueError("attention geometry differs")
                for slot, key in (("keys", "key"), ("values", "value")):
                    prior = native_array(project, first["parent_cut"], slot, layer)
                    if not np.array_equal(prior, captured_bits(root, trace[key])[:, :, :prior.shape[2]]): raise ValueError("installed reader cache differs")
                    counters["reader_prefixes"] += 1
                if source is None: continue
                lifecycle = symbols[layer]
                if project.symbol_lifecycle(sid, lifecycle["fingerprint"])["lifecycle"] != lifecycle or len(lifecycle["uses"]) != 12:
                    raise ValueError("canonical source lifecycle differs")
                proof = case["resolutions"][side]
                if any("cas://sha256/"+value not in lifecycle["binding"]["evidence"] for value in (
                        proof["receipt"], proof["trace"]["attention"]["sha256"])): raise ValueError("resolver evidence unlinked")
                counters["resolver_proof_links"] += 1
                for use in lifecycle["uses"]:
                    h, kv = use["metadata"]["query_head"], use["metadata"]["kv_head"]
                    locations = {(loc["slot"], tuple(loc["selector"])) for loc in use["locations"]}
                    if (kv != h//6 or locations != {(slot, (layer, 0, kv, source)) for slot in ("keys", "values")}
                            or use["receipt"] != first["readout_receipt"]
                            or use["metadata"]["output_binding"] != first["output_lifecycle"]["binding_fingerprint"]):
                        raise ValueError("source/head/readout link differs")
                    metrics = {o["observable"]: o["value"] for o in use["signed_observations"]}
                    if metrics["source_mass"] != float(a[0,h,0,source]): raise ValueError("attention mass differs")
                    head_rows.append({"case":case_id,"arm":short,"layer":layer,"head":h,"kv_head":kv,
                        "source_mass":metrics["source_mass"],"source_row":source,"use_id":use["use_id"],
                        "binding_fingerprint":lifecycle["binding_fingerprint"],"source_cut":first["parent_cut"],
                        "output_binding":use["metadata"]["output_binding"]})
                    counters["canonical_attention_uses"] += 1
            words = competition(array(root,first["logits"])[0], {"cold":9255,"hot":4017,"boiling":49837})
            local[short] = {"case":case_id,"arm":short,"own_text":arm["own_text"],"first_token_text":first["next_token_text"],
                "hot_minus_cold":words["hot"]["logit"]-words["cold"]["logit"],"words":words,
                "source_row":source,"target_row":target,"payload_row":payload_row,"native_parent":parents[side],
                "actual_source_cut":first["parent_cut"],"output_binding":first["output_lifecycle"]["binding_fingerprint"]}
        if "auto-sham" in records:
            if not all(np.array_equal(array(root,a["logits"]),array(root,b["logits"])) for a,b in zip(
                    records["native-recipient"]["trajectory"],records["auto-sham"]["trajectory"])): raise ValueError("sham changed output")
            counters["exact_shams"] += 1
        for name,row in local.items():
            reference = "native-donor" if name in {"native-donor","auto-reciprocal"} else "native-recipient"
            row["margin_effect"] = row["hot_minus_cold"]-local[reference]["hot_minus_cold"]
        contexts.append({"case":case_id,"split":references[case_id]["split"],
            "arms":{k:{f:v[f] for f in ("margin_effect","hot_minus_cold","first_token_text","source_row","target_row","payload_row")} for k,v in local.items()}})
        summaries.extend(local.values())
        print(json.dumps({"audited_case":case_id,"arms":len(local)}),flush=True)
    for item in inventory:
        if sha(root/item["path"]) != item["sha256"]: raise ValueError("read-only audit changed raw evidence")
    report = {"mechanics_status":"valid","instrument_status":"executed native resolver; evaluator withheld until audit",
        "trend_status":"per-context signed effects and nomination outcomes retained","terminal_status":"not-requested; exploratory",
        "raw_files_reverified":len(inventory),"counts":counters,"contexts":len(contexts),"skipped_arms":skipped,
        "canonical_bindings":sum(len(project.symbols(sid)) for sid in all_sessions),
        "frameworks_imported":[n for n in ("torch","transformers","saturn") if n in sys.modules],
        "evaluator_sha256":sha(evaluator),"source_matches":sum(r["matches_reference"] for r in resolutions),"nominations":len(resolutions)}
    for name,value in (("audit",report),("arms",summaries),("contexts",contexts),("head-uses",head_rows),("resolutions",resolutions)):
        write(output/(name+".json"),value)
    print(json.dumps(report))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--evaluation",type=Path,required=True)
    args=parser.parse_args()
    audit(args.root.resolve(),args.output.resolve(),args.evaluation.resolve())
