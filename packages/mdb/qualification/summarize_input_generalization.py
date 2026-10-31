"""Index saved input-generalization evidence and optionally publish a new atlas.

No model, source equations, replay or ISA oracle is executed. Artifact references
are relative to the supplied campaign root so that the index remains portable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

from mdb.computation_atlas import ComputationAtlas
from mdb.equation_comparison import EquationCatalog


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path, root):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def choices(heads):
    result = {}
    for name, value in heads.items():
        if name in {"dest", "src", "width", "length"}:
            result[name] = {"argmax_index": int(value[0].argmax())}
            if name == "length":
                result[name]["decoded_bytes"] = result[name]["argmax_index"] + 1
        elif name == "regs":
            result[name] = [sum(1 << bit for bit, logit in enumerate(word) if logit > 0) for word in value[0]]
        elif name in {"imm", "store_addr", "store_value"}:
            result[name] = sum(1 << bit for bit, logit in enumerate(value[0]) if logit > 0)
        else:
            result[name] = (value > 0).astype(int).reshape(-1).tolist()
    return result


def hard_distance(changes):
    """Descriptive Hamming count plus changed length/effect decisions."""
    return (changes["register_changed_bit_count"] + len(changes["flags_changed"])
            + sum(int(changes[field]) for field in ("length_changed", "store_changed", "syscall_changed")))


def oracle_correct(score):
    if score["oracle_status"] != "scored":
        return None
    return not score["register_mismatches"] and not score["flag_mismatches"] and score["length_matches"]


def exact_arrays(left, right):
    return set(left) == set(right) and all(left[name].shape == right[name].shape
        and left[name].dtype == right[name].dtype and left[name].tobytes() == right[name].tobytes() for name in left)


def summarize(root, *, atlas_root=None):
    root = Path(root).resolve()
    helper_path = Path(__file__).with_name("summarize_context_window.py")
    helpers = runpy.run_path(str(helper_path))
    result = {"schema": "mdb-input-generalization-publication-v1", "artifact_reference_base": "supplied campaign root",
              "campaign_root_name": root.name, "runs": {}, "model_calls": 0, "equation_evaluations": 0,
              "oracle_evaluations": 0, "terminal_status": "unassessed",
              "publisher_source_sha256": sha(Path(__file__)), "custody_helper_sha256": sha(helper_path),
              "limits": ["Source equations and all frozen weights were supplied; no learned rule or semantic discovery claim.",
                         "The initial six-case panel and four-case encoding followup are separate selected cohorts with different horizons.",
                         "Numerical model prediction accuracy and posthoc architectural correctness are separate observations.",
                         "Signed effects, P2-restoration counter-trends and native ISA errors are preserved without scientific pass thresholds.",
                         "Later instruction contrasts can follow different own-state roots; no forecast is reconditioned on native state.",
                         "Atlas comparisons are caller-selected family candidates, not validated rules or independent statistical replications."]}
    atlas_inputs = {}
    for directory in sorted(root.glob("run-*")):
        required = [directory / name for name in ("analysis-v1.json", "isa-score-all-arms-v1.json")]
        if not all(path.is_file() for path in required):
            result["runs"][directory.name] = {"status": "pending-required-audit-or-oracle", "included_rows": 0}
            continue
        # Checks transaction/custody before any numerical leaf is loaded.
        run = helpers["source_run"](directory, "sequence-result")
        audit, oracle = (read(path) for path in required)
        require(audit["mechanics_status"] == "valid", "independent audit is not mechanics-valid: " + directory.name)
        for document in (audit, oracle):
            require(all(document["source"][key] == run["source"][key] for key in
                        ("job_id", "request_sha256", "archive_sha256", "transaction")), "audit/oracle source custody differs")
        require(sha(directory / "analysis-v1-source.py") == audit["auditor_source_sha256"], "independent audit source snapshot differs")
        observation_path = run["transaction"] / "observations/native.json"
        require(oracle["source_observation_sha256"] == sha(observation_path), "oracle scores address other observations")
        observations = run["tree"].decode(read(observation_path)["value"])
        forecasts = run["tree"].decode(read(run["transaction"] / "forecasts/campaign.json")["predictions"])
        require(set(observations) == set(audit["rows"]) == set(oracle["rows"]), "audit/oracle/raw row inventories differ")
        comparisons = {row["row_id"]: row for row in read(run["data"] / "comparisons.json")["rows"]}
        require(set(comparisons) == set(observations), "worker comparison rows differ")
        plan, rows, counterexamples = run["plan"], {}, []
        for row_id, row in observations.items():
            measured, score = audit["rows"][row_id], oracle["rows"][row_id]
            case_id, arm, step = row_id.rsplit(".", 2)
            predicted = forecasts[case_id + "." + arm]["steps"].get(step)
            require(score["before"] == row["before"] and score["program_hex"] == row["program_hex"], "oracle row frontier differs")
            require(score["model"]["register_words"] == row["decoded"]["register_words"]
                    and score["model"]["flags"] == row["decoded"]["flags"] and score["model"]["length"] == row["decoded"]["length"], "oracle scored another hard state")
            effect = measured.get("intervention_effect")
            rows[row_id] = {"case_id": case_id, "arm": arm, "instruction": int(step),
                "program_hex": row["program_hex"], "before": row["before"], "actual_decoded": row["decoded"],
                "forecast_decoded": measured["forecast_decode"], "actual_head_choices": choices(row["heads"]),
                "forecast_head_choices": None if predicted is None else choices(predicted["heads"]),
                "head_decision_differences": {key: value["decision_differences"] for key, value in (measured["head_comparison"] or {}).items()},
                "head_max_abs": {key: value["residual"]["max_abs"] for key, value in (measured["head_comparison"] or {}).items()},
                "stage_max_abs": {key: value["max_abs"] for key, value in measured["stage_comparison"].items()},
                "actual_head_margins": measured["native_head_margins"], "forecast_head_margins": measured["forecast_head_margins"],
                "forecast_native_root_joins": measured["forecast_native_root_joins"],
                "native_chain": measured["native_chain"], "equation_record": comparisons[row_id].get("equation_record"),
                "RSI_bit1": {"actual": float(row["heads"]["regs"][0, 6, 1]),
                             "forecast": None if predicted is None else float(predicted["heads"]["regs"][0, 6, 1]),
                             "effect_vs_native": None if effect is None else effect["RSI_bit1_effect"]},
                "intervention_effect": effect, "posthoc_ISA": {**score, "correct_on_scored_outputs": oracle_correct(score)}}
            if arm == "native" and oracle_correct(score) is False:
                counterexamples.append({"row_id": row_id, "claim_id": "native-implements-supplied-ISA", "status": "posthoc-oracle-mismatch"})
            if arm == "U_swap" and int(step) == 0 and effect is not None and effect["RSI_bit1_effect"] > 0:
                counterexamples.append({"row_id": row_id, "claim_id": "U-swap-RSI-bit1-effect-always-negative", "status": "positive-effect-observed", "signed_effect": effect["RSI_bit1_effect"]})
        first_step = {}
        for case, effects in audit["first_step_full_state_effects"].items():
            distance_u = hard_distance(effects["U_swap_vs_native"]["full_hard_state_changes"])
            distance_p2 = hard_distance(effects["repair_vs_native"]["full_hard_state_changes"])
            first_step[case] = {"U_swap_vs_native": effects["U_swap_vs_native"], "repair_vs_native": effects["repair_vs_native"],
                                "repair_vs_U_swap": effects["repair_vs_U_swap"], "U_distance_from_native": distance_u,
                                "P2_restored_distance_from_native": distance_p2, "distance_change_after_P2_restoration": distance_p2 - distance_u,
                                "hard_distance_units": "changed register bits + flag bits + length/store/syscall decisions"}
            if distance_p2 > distance_u:
                counterexamples.append({"row_id": case + ".U_swap_P2native.0", "claim_id": "P2-restoration-never-increases-hard-distance-from-native",
                                        "status": "counterexample-observed", "distance_before": distance_u, "distance_after": distance_p2})
        summary = read(run["data"] / "summary.json")
        refs = {name: reference(path, root) for name, path in {"audit": required[0], "oracle": required[1],
                "audit_source": directory / "analysis-v1-source.py", "worker_summary": run["data"] / "summary.json",
                "comparisons": run["data"] / "comparisons.json", "plan": run["data"] / "plan.json",
                "transaction_report": run["data"] / "transaction-report.json", "forecast": run["transaction"] / "forecasts/campaign.json",
                "observations": observation_path, "external_request": directory / "request.json", "external_report": directory / "report.json"}.items()}
        result["runs"][directory.name] = {"status": "custody-verified-and-audited", "source": run["source"], "sources": refs,
            "sealed_run_summary": summary, "cases": plan["cases"], "arms": plan["arms"], "max_instructions": plan["max_instructions"],
            "counts": audit["counts"], "oracle_counts": oracle["counts"], "oracle_instrument": oracle["oracle"],
            "first_step": first_step, "rows": rows, "counterexamples": counterexamples,
            "native_oracle_correct_rows": sum(row["posthoc_ISA"]["correct_on_scored_outputs"] is True for row in rows.values() if row["arm"] == "native"),
            "native_oracle_scored_rows": sum(row["posthoc_ISA"]["correct_on_scored_outputs"] is not None for row in rows.values() if row["arm"] == "native"),
            "limits": [*plan.get("scope_notes", []), *audit["limits"], *oracle["limits"]]}
        atlas_inputs[directory.name] = {"catalog": run["data"] / "equations", "observations": observations, "audit": audit, "comparisons": comparisons}
    first_inputs = {}
    for label, source in atlas_inputs.items():
        for row_id, row in source["observations"].items():
            if not row_id.endswith(".native.0"):
                continue
            digest = hashlib.sha256()
            for name, value in sorted(row["roots"].items()):
                digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
                digest.update(value.tobytes())
            first_inputs.setdefault(digest.hexdigest(), []).append(label + "/" + row_id)
    result["first_step_input_inventory"] = {"case_contexts": sum(len(group) for group in first_inputs.values()),
        "distinct_literal_model_inputs": len(first_inputs), "groups": first_inputs,
        "scope": "Native first-step fetch15/regs/flags bytes; arms and successor rows are not independent new inputs."}
    if {"run-01", "run-02"} <= set(atlas_inputs):
        repeated = {}
        for arm in ("native", "U_swap", "U_swap_P2native"):
            left = atlas_inputs["run-01"]["observations"]["rax_add_xor." + arm + ".0"]
            right = atlas_inputs["run-02"]["observations"]["enc01_trailing." + arm + ".0"]
            repeated[arm] = {field: exact_arrays(left[field], right[field]) for field in ("roots", "heads", "stages")}
            repeated[arm]["decoded"] = left["decoded"] == right["decoded"]
        result["repeated_corner"] = {"left": "run-01/rax_add_xor", "right": "run-02/enc01_trailing",
            "first_step_byte_exact": repeated, "independent_novel_input": False,
            "scope": "Changed plan/source provenance can change applicability cohorts while model input bytes repeat."}
    if atlas_root is not None:
        result["atlas"] = publish_atlas(Path(atlas_root), result, atlas_inputs)
        location = Path(atlas_root).resolve()
        result["atlas"]["location"] = (location.relative_to(root).as_posix() if location.is_relative_to(root)
                                         else {"external_destination_name": location.name})
    result["counts"] = {"completed_runs": len(atlas_inputs), "pending_runs": len(result["runs"]) - len(atlas_inputs),
                        "native_calls": sum(run["counts"]["native_calls"] for run in result["runs"].values() if "counts" in run),
                        "rows": sum(len(run.get("rows", {})) for run in result["runs"].values())}
    return result


def publish_atlas(atlas_root, index, inputs):
    require(not atlas_root.exists(), "optional atlas destination must be a new directory; existing atlases remain unchanged")
    atlas_root.mkdir(parents=True)
    selected = {"schema": "mdb-input-generalization-atlas-selected-v1", "runs": index["runs"]}
    selected_path = atlas_root / "selected-effects.json"
    save(selected_path, selected)
    selected_ref = {"path": "selected-effects.json", "sha256": sha(selected_path)}
    atlas, imported, families, query_count = ComputationAtlas(atlas_root), {}, [], 0
    for label, source in inputs.items():
        run, ids, effects, futures, counter = index["runs"][label], [], {}, {}, {}
        catalog = EquationCatalog(source["catalog"])
        for row_id, row in run["rows"].items():
            identifier = row["equation_record"]
            require(identifier is not None, "atlas row lacks a published equation record")
            record = catalog.show(identifier)
            require(record["report"] == source["comparisons"][row_id]["head_report"], "catalog/worker comparison differs")
            require(record["binding"]["input_identity"] == source["observations"][row_id]["input_identity"], "catalog serving input differs")
            ids.append(identifier)
            pointer = "/runs/" + label + "/rows/" + row_id
            provenance = [{**selected_ref, "json_pointer": pointer}, run["sources"]["audit"], run["sources"]["oracle"]]
            effect = row["intervention_effect"]
            if effect is not None:
                scope = "matched-input" if effect["same_raw_roots"] else "different-own-state-roots"
                effects[identifier] = [{"consumer_id": "regs[6,1]", "contrast_id": "observed-minus-native/" + scope,
                                        "units": "logit", "value": effect["RSI_bit1_effect"], "provenance": provenance}]
            if row["arm"] == "U_swap_P2native" and row["instruction"] == 0:
                restoration = run["first_step"][row["case_id"]]["repair_vs_U_swap"]
                require(restoration["same_raw_roots"], "first-step P2 restoration has divergent raw roots")
                effects.setdefault(identifier, []).append({"consumer_id": "regs[6,1]",
                    "contrast_id": "P2-restoration-minus-U-swap/matched-input", "units": "logit",
                    "value": restoration["RSI_bit1_effect"], "provenance": provenance})
            key = row["case_id"] + "." + row["arm"]
            futures[identifier] = {"status": "observed-own-state-chain", "value": {
                "native": source["audit"]["native_chains"][key], "forecast": source["audit"]["forecast_chains"][key],
                "horizon": run["max_instructions"], "from_instruction": row["instruction"]}, "provenance": provenance}
            counter[identifier] = [{**item, "provenance": provenance} for item in run["counterexamples"] if item["row_id"] == row_id]
        imported[label] = {row_id: item["fingerprint"] for row_id, item in zip(run["rows"], atlas.ingest(source["catalog"], ids,
            provenance={"run_id": run["source"]["job_id"], "artifacts": [selected_ref, *run["sources"].values()]},
            signed_effects=effects, continuation=futures, counterexamples=counter), strict=True)}
        for arm in ("U_swap", "U_swap_P2native"):
            rows = [row_id for row_id, row in run["rows"].items() if row["arm"] == arm and row["instruction"] == 0]
            contrast = ("observed-minus-native/matched-input" if arm == "U_swap"
                        else "P2-restoration-minus-U-swap/matched-input")
            family = atlas.compare([imported[label][row] for row in rows], consumer_id="regs[6,1]",
                                   contrast_id=contrast)
            families.append({"run": label, "arm": arm, "horizon": run["max_instructions"], "row_ids": rows,
                             "contrast_id": contrast, "fingerprint": family["fingerprint"],
                             "cohorts": family["cohorts"], "rule_status": family["rule_status"]})
    matches = atlas.find(consumer_id="regs[6,1]")
    for record in matches:
        context = record["equation"]["tested_context"]
        require(len(atlas.find(applicable_to=context)) == 1, "exact atlas applicability context is ambiguous")
        require(not atlas.find(applicable_to={**context, "source_identity": "0" * 64}), "changed source still matches atlas context")
        query_count += 1
    value = {"schema": "mdb-input-generalization-atlas-index-v1", "path_scope": "explicit new atlas destination",
             "records": imported, "consumer_matches": len(matches), "exact_context_unique": query_count,
             "changed_source_matches": 0, "family_candidates": families, "selected_effects": selected_ref,
             "rule_status": "not-validated", "cross_run_pooling": "none; run/horizon cohorts and routes compared separately",
             "new_model_calls": 0, "equation_evaluations": 0}
    save(atlas_root / "index.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--atlas-root", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    value = summarize(args.root, atlas_root=args.atlas_root)
    save(args.output, value)
    print(json.dumps({"counts": value["counts"], "output": str(args.output), "model_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
