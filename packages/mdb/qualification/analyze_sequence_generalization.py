"""Audit sealed per-case architectural trajectories from saved arrays only.

No model, ISA-model, replay, or source-equation execution. Scientific head and
state discrepancies are retained separately from custody/instrument errors.
"""
from __future__ import annotations

import argparse
import json
import runpy
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from mdb.forward_ports import NativePortCatalog
from mdb.forward_symbols import verify_trace


@lru_cache(maxsize=1)
def helpers():
    return runpy.run_path(str(Path(__file__).with_name("summarize_context_window.py")))


@lru_cache(maxsize=1)
def transition_helpers():
    return runpy.run_path(str(Path(__file__).with_name("analyze_state_transition_campaign.py")))


def state(value):
    return {"register_words": [int(word, 16) if isinstance(word, str) else int(word)
                               for word in value["register_words"]],
            "flags": list(value.get("flags", value.get("flags_CF_ZF_SF_OF"))),
            "offset": int(value.get("offset", value.get("byte_offset", 0)))}


def head_margins(heads):
    """Signed threshold logits or top-two categorical margins; no pass cutoff."""
    result = {}
    for name, value in sorted(heads.items()):
        if name in {"dest", "src", "width", "length"}:
            ordered = np.sort(value.astype(np.float64), axis=-1)
            margin = ordered[..., -1] - ordered[..., -2]
            result[name] = {"rule": "argmax", "minimum_top_two_margin": float(margin.min()),
                            "ties": int(np.count_nonzero(margin == 0))}
        else:
            value = value.astype(np.float64)
            index = np.unravel_index(int(np.argmin(np.abs(value))), value.shape)
            result[name] = {"rule": "strict-logit>0", "minimum_absolute_logit": float(np.abs(value).min()),
                            "nearest_signed_logit": float(value[index]), "nearest_coordinate": list(map(int, index)),
                            "exact_zero_logits": int(np.count_nonzero(value == 0))}
    return result


def chain_checks(steps, case, program_hex, max_instructions, *, native):
    """Independently follow each trajectory's own heads; never use other roots."""
    h, result, previous = helpers(), {}, None
    initial = state({**case, "offset": 0})
    program_size = len(bytes.fromhex(program_hex))
    for position, number in enumerate(sorted(steps, key=int)):
        row, errors = steps[number], []
        try:
            shapes = transition_helpers()["SHAPES"]
            if set(row["heads"]) != set(shapes) or any(
                    row["heads"][name].shape != shape or not np.isfinite(row["heads"][name]).all()
                    for name, shape in shapes.items()):
                errors.append("eleven-head own-chain ABI/finite differs")
            before, decoded = state(row["before"]), h["decoded"](row["heads"])
            if int(number) != position or position >= max_instructions:
                errors.append("instruction index/budget differs")
            if decoded != row["decoded"]:
                errors.append("stored decode differs from own eleven raw heads")
            expected = initial if previous is None else {
                "register_words": previous[1]["register_words"], "flags": previous[1]["flags"],
                "offset": previous[0]["offset"] + previous[1]["length"]}
            if previous is not None and (previous[1]["store"] or previous[1]["syscall"]):
                errors.append("successor after blocked effect")
            if previous is not None and native and not previous[2]["driver_commit"].get("committed"):
                errors.append("successor after uncommitted native state")
            if before != expected:
                errors.append("own-state predecessor/initial state join differs")
            if not 0 <= before["offset"] < program_size:
                errors.append("step fetch outside declared program")
            literal = h["raw_roots"](program_hex, before)
            joins = {key: bool(np.array_equal(value, row["roots"].get(key))
                               and row["roots"][key].dtype == np.float32) for key, value in literal.items()}
            if set(row["roots"]) != set(literal) or not all(joins.values()):
                errors.append("literal own-state fetch/regs/flags differ")
            if native:
                commit = row["driver_commit"]
                blocked = decoded["store"] or decoded["syscall"]
                if bool(commit.get("committed")) == blocked:
                    errors.append("native atomic commit/effect policy differs")
                if commit.get("committed"):
                    after = {"register_words": decoded["register_words"], "flags": decoded["flags"],
                             "offset": before["offset"] + decoded["length"]}
                    if state(commit["before"]) != before or state(commit["after"]) != after:
                        errors.append("driver before/after differs from raw-head decisions")
                lineage = row.get("lineage", {})
                if lineage and (lineage.get("program_hex") != program_hex or lineage.get("step") != int(number)
                                or lineage.get("case_id") != case["case_id"]):
                    errors.append("native case/program/step lineage differs")
                expected_previous = None if previous is None else previous[2]["row_id"]
                if lineage and lineage.get("previous_row") != expected_previous:
                    errors.append("native predecessor row identity differs")
            result[number] = {"errors": errors, "own_state_join": before == expected,
                              "literal_root_joins": joins, "before": before, "expected_before": expected,
                              "decoded": decoded, "head_margins": head_margins(row["heads"])}
            previous = (before, decoded, row)
        except (KeyError, ValueError, TypeError, IndexError) as error:
            result[number] = {"errors": [*errors, f"chain instrument: {type(error).__name__}: {error}"]}
    if previous is not None:
        before, decoded, _ = previous
        ended = (decoded["store"] or decoded["syscall"] or len(steps) >= max_instructions
                 or before["offset"] + decoded["length"] >= program_size)
        if not ended:
            result[sorted(result, key=int)[-1]]["errors"].append("chain truncated before declared termination")
    elif not steps:
        result["missing"] = {"errors": ["declared trajectory has no first instruction"]}
    return result


def row_checks(run, row_id, row, prediction, program_hex):
    h, errors = helpers(), []

    def check(condition, message):
        if not condition:
            errors.append(message)

    trace = h["read"](h["file_path"](run["data"], "captures/" + row_id + "/trace.json"))
    verify_trace(trace)
    ports = NativePortCatalog(trace)
    check(trace["fingerprint"] == row["trace_fingerprint"] and trace["input_identity"] == row["input_identity"],
          "capture trace/input binding differs")
    check(trace["model_identity"] == run["plan"]["model_identity"] and trace["execution"] == run["plan"]["execution"],
          "capture model/numerical execution differs")
    check(row["program_hex"] == program_hex, "observed per-case program differs")
    heads = h["HEADS"]
    shapes = transition_helpers()["SHAPES"]
    check(set(row["heads"]) == set(trace["outputs"]) == heads, "eleven native heads incomplete")
    for name, shape in shapes.items():
        check(row["heads"][name].shape == shape and row["heads"][name].dtype == np.float32
              and np.isfinite(row["heads"][name]).all(), "native head ABI/finite differs: " + name)
        encoded = trace["outputs"][name]
        h["descriptor_join"](row["heads"][name], trace["values"][encoded["$value"]], "raw head/trace differs: " + name)
        if prediction is not None:
            check(prediction["heads"][name].shape == shape and np.isfinite(prediction["heads"][name]).all(),
                  "forecast head ABI/finite differs: " + name)
    expected_ports = {**h["expected_stage_ports"](), "tokens": "module:inp:input",
                      "layer0.S": "module:mixers.0:input", "layer1.S": "module:mixers.1:input"}
    check(set(row["stages"]) == set(row["actual_ports"]) == set(expected_ports), "39-port native inventory differs")
    stages = {}
    for name, binding in row["actual_ports"].items():
        path = ["args", 0] if name in {"tokens", "layer0.S", "layer1.S", "layer2.S", "layer2.h"} else None
        check(binding["selector"] == expected_ports.get(name) and ports.resolve(binding["selector"], path=path) == binding,
              "authoritative native port differs: " + name)
        h["descriptor_join"](row["stages"][name], trace["values"][binding["value_id"]], "stage/trace differs: " + name)
        if prediction is not None:
            check(name in prediction["stages"], "native port lacks sealed prediction: " + name)
            if name in prediction["stages"]:
                stages[name] = h["difference"](row["stages"][name], prediction["stages"][name])
    traced = sorted(h["sha"](h["canonical"]({key: value[key] for key in ("shape", "dtype", "sha256")}))
                    for value in trace["values"].values() if value["origin"] == "input")
    serving = sorted(h["sha"](h["canonical"](h["tensor_descriptor"](value))) for value in row["roots"].values())
    check(traced == serving, "native serving arrays differ from trace root descriptors")
    joins = {name: bool(prediction is not None and np.array_equal(value, prediction["roots"].get(name)))
             for name, value in row["roots"].items()}
    check(all(joins.values()) == row["forecast_root_join"], "reported cross-trajectory root join differs")
    check(set(row["replay_native_exact"]) == heads and all(row["replay_native_exact"].values()),
          "independent replay/native head equality differs")
    return {"trace": trace, "ports": ports, "errors": errors, "report": {
        "forecast_status": "sealed forecast available" if prediction is not None else "forecast horizon ended",
        "forecast_native_root_joins": joins, "same_forecast_native_roots": all(joins.values()),
        "comparison_scope": "same raw roots" if all(joins.values()) else "different own-state roots; conditional comparison",
        "native_decode": row["decoded"], "forecast_decode": None if prediction is None else prediction["decoded"],
        "native_head_margins": head_margins(row["heads"]),
        "forecast_head_margins": None if prediction is None else head_margins(prediction["heads"]),
        "head_comparison": None if prediction is None else h["head_comparison"](row["heads"], prediction["heads"]),
        "stage_comparison": stages,
        "forecast_only_stages": [] if prediction is None else sorted(set(prediction["stages"]) - set(row["stages"]))}}


def intervention_effect(row, baseline):
    h = helpers()
    return {"same_raw_roots": all(np.array_equal(value, baseline["roots"][key]) for key, value in row["roots"].items()),
            "all_head_signed_effects": {name: h["difference"](value, baseline["heads"][name]) for name, value in row["heads"].items()},
            "full_hard_state_changes": h["hard_changes"](row["heads"], baseline["heads"]),
            "RSI_bit1_effect": float(row["heads"]["regs"][0, 6, 1]) - float(baseline["heads"]["regs"][0, 6, 1]),
            "RDI_logit_effect_by_bit": (row["heads"]["regs"][0, 7].astype(np.float64) - baseline["heads"]["regs"][0, 7]).tolist()}


def audit(run_dir):
    h, t = helpers(), transition_helpers()
    # source_run verifies external request/archive/file inventory, transaction
    # seal and event chain before any prediction/observation array is decoded.
    run = h["source_run"](run_dir, "sequence-result")
    doc = h["read"](run["transaction"] / "forecasts/campaign.json")
    forecasts = run["tree"].decode(doc["predictions"])
    observations = run["tree"].decode(h["read"](run["transaction"] / "observations/native.json")["value"])
    payloads = {key: run["tree"].decode(value["value"]) for key, value in doc["payloads"].items()}
    plan, rows, proofs, errors = run["plan"], {}, {}, []
    order, native_chains, forecast_chains, payload_checks = [], {}, {}, {}
    for case in plan["cases"]:
        program = case.get("program_hex", plan.get("program_hex"))
        horizon = case.get("max_instructions", plan["max_instructions"])
        for arm in plan["arms"]:
            key = case["case_id"] + "." + arm
            prediction = forecasts[key]
            if prediction["program_hex"] != program or prediction["arm"] != arm:
                errors.append("sealed trajectory program/arm differs: " + key)
            steps = {number: {**row, "row_id": key + "." + number} for number, row in
                     ((name.rsplit(".", 1)[1], value) for name, value in observations.items() if name.startswith(key + "."))}
            native_chains[key] = chain_checks(steps, case, program, horizon, native=True)
            forecast_chains[key] = chain_checks(prediction["steps"], case, program, horizon, native=False)
            for number in sorted(steps, key=int):
                row_id, row = key + "." + number, steps[number]
                order.append(row_id)
                try:
                    proof = row_checks(run, row_id, row, prediction["steps"].get(number), program)
                    proofs[row_id] = proof
                    rows[row_id] = {"mechanics_errors": [*proof["errors"], *native_chains[key][number]["errors"]],
                                    "native_chain": native_chains[key][number], **proof["report"]}
                    lineage = row.get("lineage", {})
                    if lineage and lineage.get("arm") != arm:
                        rows[row_id]["mechanics_errors"].append("native arm lineage differs")
                except (KeyError, ValueError, TypeError, IndexError) as error:
                    rows[row_id] = {"mechanics_errors": [f"row instrument: {type(error).__name__}: {error}"]}
            if arm != "native":
                native = forecasts[case["case_id"] + ".native"]["steps"]["0"]
                expected = {"post0_zero": np.zeros((1, 64, 48), dtype=np.float32),
                            "U_native": native["stages"]["layer2.U"], "P2_native": native["stages"]["layer2.P2"]}
                for suffix, value in expected.items():
                    name, problems = key + "." + suffix, []
                    supplied = payloads[name]
                    descriptor = doc["payloads"][name]
                    original = run["tree"].decode(descriptor["original"])
                    if supplied.dtype != np.float32 or not np.array_equal(supplied, np.asarray(value, dtype=np.float32)):
                        problems.append("frozen payload differs from declared analytical endpoint")
                    if (not np.array_equal(supplied, np.asarray(original, dtype=np.float32))
                            or descriptor["cast"]["from"] != str(original.dtype) or descriptor["cast"]["to"] != str(supplied.dtype)):
                        problems.append("original/staged cast lineage differs")
                    payload_checks[name] = {"errors": problems, "provenance": descriptor.get("provenance")}
    if set(order) != set(observations):
        errors.append("unplanned observed case/arm rows")
    if set(payload_checks) != set(payloads):
        errors.append("frozen payload inventory differs from declared arm recipes")
    # The worker emits flat receipts in exactly case/arm/step native order.
    receipts, cursor = h["read"](run["data"] / "public-replay-receipts.json"), 0
    for row_id in order:
        row, problems = observations[row_id], rows[row_id]["mechanics_errors"]
        case_id, arm, number = row_id.rsplit(".", 2)
        source_id = case_id + ".native.0" if number == "0" else row_id
        try:
            lineage = row.get("lineage", {})
            if lineage.get("replay_source_row", source_id) != source_id:
                problems.append("declared common replay source differs")
            proof = proofs[source_id]
            count = len(proof["trace"]["events"])
            selected = receipts[cursor:cursor + count]
            cursor += count
            recipes, wrappers = [], []
            if number == "0" and arm != "native":
                changes = [("module:post.0:return", "post0_zero"), ("module:mixers.2.out_proj:return", "U_native")]
                if arm == "U_swap_P2native":
                    changes.append(("module:post.2:return", "P2_native"))
                for receipt, (selector, suffix) in zip(row["acts"], changes):
                    recipe = {"selector": selector, "payload_name": case_id + "." + arm + "." + suffix, "patch_kind": "replace"}
                    recipes.append(recipe)
                    wrappers.append({"receipt": receipt, "payload_name": recipe["payload_name"], "port": proof["ports"].resolve(selector)})
                if len(row["acts"]) != len(changes):
                    problems.append("first-step Act count differs")
            replay = t["audit_replay"](row_id, {**row, "acts": wrappers}, proofs[row_id], proof, selected, recipes, payloads)
            if not wrappers and row["acts"]:
                replay["errors"].append("unexpected native/later-step Act")
            rows[row_id]["replay"] = replay
            problems.extend(replay["errors"])
        except (KeyError, ValueError, TypeError, IndexError) as error:
            problems.append("replay instrument: " + str(error))
        native_id = case_id + ".native." + number
        if native_id in observations:
            rows[row_id]["native_baseline_row"] = native_id
            rows[row_id]["intervention_effect"] = intervention_effect(row, observations[native_id])
    if cursor != len(receipts):
        errors.append("flat transition receipt inventory/order differs")
    summary = h["read"](run["data"] / "summary.json")
    count = run["transaction_report"]["work"]["native_calls"]
    if len(observations) != count or summary["counts"]["physical_forwards"] != count:
        errors.append("native physical count/observation/summary differs")
    events = h["read"](run["data"] / "native-events.json")
    if len(events) != count or [event["index"] for event in events] != list(range(1, count + 1)):
        errors.append("native root-entry ledger differs")
    if count > plan["budgets"]["max_physical_forwards"] or summary["counts"]["replay_operations"] != len(receipts):
        errors.append("finite work budget/replay summary differs")
    if summary["counts"]["acts"] != sum(len(row["acts"]) for row in observations.values()):
        errors.append("Act summary count differs")
    first_step = {}
    for case in plan["cases"]:
        prefix = case["case_id"]
        baseline, changed, repair = (observations.get(prefix + "." + arm + ".0") for arm in ("native", "U_swap", "U_swap_P2native"))
        if baseline is not None and changed is not None and repair is not None:
            first_step[prefix] = {"U_swap_vs_native": intervention_effect(changed, baseline),
                                 "repair_vs_native": intervention_effect(repair, baseline),
                                 "repair_vs_U_swap": intervention_effect(repair, changed)}
    chain_errors = sum(len(row["errors"]) for chain in forecast_chains.values() for row in chain.values())
    native_chain_errors = sum(len(row["errors"]) for chain in native_chains.values() for row in chain.values())
    valid = not errors and not chain_errors and not native_chain_errors and not any(row["mechanics_errors"] for row in rows.values()) and not any(row["errors"] for row in payload_checks.values())
    return {"schema": "mdb-sequence-generalization-audit-v1", "source": run["source"],
            "audit_helpers": {name: h["sha"](Path(__file__).with_name(name).read_bytes()) for name in
                              ("summarize_context_window.py", "analyze_state_transition_campaign.py")},
            "plan_sha256": h["sha"]((run["data"] / "plan.json").read_bytes()),
            "mechanics_status": "valid" if valid else "instrument-errors-retained", "mechanics_errors": errors,
            "rows": rows, "native_chains": native_chains, "forecast_chains": forecast_chains,
            "payloads": payload_checks, "first_step_full_state_effects": first_step,
            "counts": {"native_calls": count, "rows": len(rows), "replay_transitions": len(receipts),
                       "head_decision_mismatches": sum(head["decision_differences"] for row in rows.values() for head in (row.get("head_comparison") or {}).values()),
                       "forecast_chain_errors": chain_errors,
                       "native_chain_errors": native_chain_errors,
                       "native_chain_steps": sum(len(chain) for chain in native_chains.values()),
                       "forecast_chain_steps": sum(len(chain) for chain in forecast_chains.values()),
                       "cross_root_divergence_rows": sum(not row.get("same_forecast_native_roots", False) for row in rows.values()),
                       "row_error_counts": dict(Counter(error for row in rows.values() for error in row["mechanics_errors"]))},
            "trend_status": "per-row signed effects and discrepancies retained", "terminal_status": "unassessed",
            "limits": ["Saved-array audit only; no model, replay, or source-equation evaluation.",
                       "Source equations and full weights were supplied; this is not semantic operation discovery.",
                       "Each forecast and native trajectory uses its own prior heads; later cross-root comparisons may differ.",
                       "Replay equality is the source-bound worker instrument plus typed receipts, not a second replay run.",
                       "No ISA correctness oracle, held-out population claim, or scientific numerical pass threshold."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output already exists")
    result = audit(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.with_name(args.output.stem + "-source.py")
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    result["auditor_source_sha256"] = helpers()["sha"](snapshot.read_bytes())
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"mechanics_status": result["mechanics_status"], "counts": result["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
