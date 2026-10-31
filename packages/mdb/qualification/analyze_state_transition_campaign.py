"""Independently audit saved segmented futures and matched-path interventions.

No model, replay or source-equation execution. Custody failures refuse the
artifact; row instrument errors and scientific discrepancies remain separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from mdb.discovery_catalog import StructuralCatalog
from mdb.equation_comparison import EquationCatalog
from mdb.forward_ports import NativePortCatalog
from mdb.forward_symbols import verify_trace

HEADS = {"regs", "flags", "dest", "src", "width", "length", "syscall", "imm", "store", "store_addr", "store_value"}
SHAPES = {"regs": (1, 8, 64), "flags": (1, 4), "dest": (1, 8), "src": (1, 8), "width": (1, 3),
          "length": (1, 15), "syscall": (1,), "imm": (1, 64), "store": (1,), "store_addr": (1, 64), "store_value": (1, 64)}
DIRECT = [10, 11, *range(284, 293)]


@lru_cache(maxsize=1)
def helpers():
    return runpy.run_path(str(Path(__file__).with_name("summarize_context_window.py")))


def state(value):
    words = value["register_words"]
    return {"register_words": [int(word, 16) if isinstance(word, str) else int(word) for word in words],
            "flags": list(value.get("flags", value.get("flags_CF_ZF_SF_OF"))),
            "offset": int(value.get("offset", value.get("byte_offset", 0)))}


def select(value, path):
    for component in path:
        value = value[component]
    return value


def direct_columns_exact(before, after, payload, observed):
    if before.shape != (1, 64, 293) or after.shape != before.shape or payload.shape != (1, 64, 11):
        return False
    untouched = [index for index in range(293) if index not in DIRECT]
    return bool(np.array_equal(after[:, :, DIRECT], payload) and np.array_equal(before[:, :, untouched], after[:, :, untouched])
                and np.array_equal(after, observed))


def chain_checks(steps, plan, *, native):
    """Check each chain against its own preceding hard decisions only."""
    h = helpers()
    segments = {row["id"]: (index, row) for index, row in enumerate(plan["segments"])}
    result, previous = {}, None
    initial = {"register_words": list(plan["register_words"]), "flags": list(plan["flags"]), "offset": 0}
    ordered = sorted(steps, key=int)
    for number in ordered:
        row, errors = steps[number], []
        try:
            before = state(row["before"])
            decoded = h["decoded"](row["heads"])
            if set(row["heads"]) != HEADS or any(row["heads"][name].shape != shape for name, shape in SHAPES.items()):
                errors.append("full eleven-head chain ABI differs")
            if decoded != row["decoded"]:
                errors.append("stored chain decode differs from its own raw heads")
            lineage = row["lineage"] if native else row
            segment_index, segment = segments[lineage["segment_id"]]
            segment_step = lineage["segment_step"]
            if int(number) != len(result):
                errors.append("noncontiguous global instruction index")
            if lineage["program_hex"] != segment["program_hex"]:
                errors.append("program differs from declared host segment")
            if not 0 <= segment_step < segment["max_instructions"]:
                errors.append("segment instruction budget/index differs")
            if before["offset"] < 0 or before["offset"] >= len(bytes.fromhex(segment["program_hex"])):
                errors.append("native/forecast step after segment program end")
            if previous is None:
                expected, host_start = initial, True
                if segment_index != 0 or segment_step != 0:
                    errors.append("first instruction does not start the declared first segment")
                if native and lineage.get("previous_row") is not None:
                    errors.append("first instruction has an unexpected native predecessor")
            else:
                previous_number, prior, prior_decoded, prior_before, prior_lineage = previous
                prior_index, prior_segment = segments[prior_lineage["segment_id"]]
                host_start = segment_index != prior_index
                if prior_decoded["store"] or prior_decoded["syscall"]:
                    errors.append("successor after blocked store/syscall")
                expected = {"register_words": prior_decoded["register_words"], "flags": prior_decoded["flags"],
                            "offset": 0 if host_start else prior_before["offset"] + prior_decoded["length"]}
                if host_start:
                    if segment_index != prior_index + 1 or segment_step != 0:
                        errors.append("host segment order/reset differs")
                    ended = (prior_lineage["segment_step"] + 1 >= prior_segment["max_instructions"]
                             or prior_before["offset"] + prior_decoded["length"] >= len(bytes.fromhex(prior_segment["program_hex"])))
                    if not ended:
                        errors.append("host reset before declared segment termination")
                elif segment_step != prior_lineage["segment_step"] + 1:
                    errors.append("within-segment instruction index differs")
                if native:
                    if not prior["driver_commit"].get("committed"):
                        errors.append("successor after uncommitted native state")
                    if lineage.get("previous_row") != prior["row_id"]:
                        errors.append("native predecessor row binding differs")
            if bool(lineage["host_segment_start"]) != host_start:
                errors.append("host segment start marker differs")
            own_join = before == expected
            if not own_join:
                errors.append("own-state predecessor/initial state join differs")
            literal = h["raw_roots"](segment["program_hex"], before)
            root_joins = {key: bool(np.array_equal(value, row["roots"].get(key))
                and row["roots"][key].dtype == np.float32) for key, value in literal.items()}
            if not all(root_joins.values()):
                errors.append("literal own-state input packing differs")
            if native:
                commit = row["driver_commit"]
                blocked = decoded["store"] or decoded["syscall"]
                if bool(commit.get("committed")) == blocked:
                    errors.append("native commit/effect policy differs")
                if commit.get("committed"):
                    after = {"register_words": decoded["register_words"], "flags": decoded["flags"],
                             "offset": before["offset"] + decoded["length"]}
                    if state(commit["before"]) != before or state(commit["after"]) != after:
                        errors.append("atomic driver before/after differs from actual head decisions")
            result[number] = {"own_state_join": own_join, "literal_root_joins": root_joins,
                "host_segment_start": host_start, "before": before, "expected_before": expected,
                "decoded": decoded, "errors": errors}
            previous = (number, row, decoded, before, lineage)
        except (KeyError, ValueError, TypeError, IndexError) as error:
            result[number] = {"errors": [*errors, f"chain metadata: {type(error).__name__}: {error}"]}
    if previous is not None:
        number, row, decoded, before, lineage = previous
        index, segment = segments[lineage["segment_id"]]
        finished = lineage["segment_step"] + 1 >= segment["max_instructions"] or before["offset"] + decoded["length"] >= len(bytes.fromhex(segment["program_hex"]))
        if not (decoded["store"] or decoded["syscall"]) and (index != len(segments) - 1 or not finished):
            result[number]["errors"].append("chain ends before declared host continuation completes")
    return result


def audit_row(run, row_id, row, prediction):
    h, errors = helpers(), []

    def check(condition, message):
        if not condition:
            errors.append(message)

    for name, shape in SHAPES.items():
        check(name in row["heads"] and row["heads"][name].shape == shape, "native head ABI differs: " + name)
        if prediction is not None:
            check(name in prediction["heads"] and prediction["heads"][name].shape == shape, "forecast head ABI differs: " + name)

    trace = h["read"](h["file_path"](run["data"], "captures/" + row_id + "/trace.json"))
    try:
        verify_trace(trace)
        check(trace["fingerprint"] == row["trace_fingerprint"] and trace["input_identity"] == row["input_identity"],
              "capture trace/input binding differs")
        check(trace["model_identity"] == run["seal"]["declarations"]["model_identity"], "capture model binding differs")
        check(trace["execution"] == run["plan"]["execution"], "capture numerical execution differs")
        ports = NativePortCatalog(trace)
        check(set(row["heads"]) == set(trace["outputs"]) == HEADS, "native eleven-head inventory differs")
        expected_ports = {**h["expected_stage_ports"](), "tokens": "module:inp:input",
                          "layer0.S": "module:mixers.0:input", "layer1.S": "module:mixers.1:input"}
        required_stages = set(expected_ports)
        check(required_stages <= set(row["stages"]), "required native intermediate inventory is incomplete")
        check(set(row["stages"]) == set(row["actual_ports"]), "saved native intermediate/actual port inventory differs")
        traced_inputs = sorted(h["sha"](h["canonical"]({key: value[key] for key in ("shape", "dtype", "sha256")}))
                               for value in trace["values"].values() if value["origin"] == "input")
        declared_inputs = sorted(h["sha"](h["canonical"](h["tensor_descriptor"](value))) for value in row["roots"].values())
        check(traced_inputs == declared_inputs, "traced serving inputs differ from literal root arrays")
        for name, encoded in trace["outputs"].items():
            h["descriptor_join"](row["heads"][name], trace["values"][encoded["$value"]], "native head/trace return differs: " + name)
        for name, binding in row["actual_ports"].items():
            path = ["args", 0] if name in {"tokens", "layer0.S", "layer1.S", "layer2.S", "layer2.h", "h"} else None
            check(binding["selector"] == expected_ports.get(name), "native stage name/authoritative module binding differs: " + name)
            check(ports.resolve(binding["selector"], path=path) == binding, "actual port binding differs: " + name)
            h["descriptor_join"](row["stages"][name], trace["values"][binding["value_id"]], "actual stage/trace descriptor differs: " + name)
    except (KeyError, ValueError, TypeError) as error:
        errors.append("capture/port instrument: " + str(error))
    root_joins = {key: bool(prediction is not None and np.array_equal(prediction["roots"].get(key), value))
                  for key, value in row["roots"].items()}
    check(all(root_joins.values()) == row["forecast_root_join"], "reported forecast/native root join differs")
    literal = h["raw_roots"](row["lineage"]["program_hex"], row["before"])
    check(set(literal) == set(row["roots"]) and all(np.array_equal(value, row["roots"].get(key)) for key, value in literal.items()),
          "actual literal root packing differs")
    if not row_id.startswith("sequence."):
        initial = {"register_words": run["plan"]["register_words"], "flags": run["plan"]["flags"], "offset": 0}
        program = (run["plan"]["contexts"]["old"] if row_id.startswith("path.")
                   else run["plan"]["mlp_programs"][row_id.split(".")[1]])
        check(state(row["before"]) == initial and row["lineage"]["program_hex"] == program,
              "matched-panel serving state differs from declared literal initial roots")
        expected = h["raw_roots"](program, initial)
        check(prediction is not None and all(np.array_equal(value, prediction["roots"].get(key))
                                            for key, value in expected.items()), "matched-panel forecast roots differ from declared raw frontier")
    check(h["decoded"](row["heads"]) == row["decoded"], "reported native decode differs from raw heads")
    if prediction is not None:
        check(h["decoded"](prediction["heads"]) == prediction["decoded"], "reported forecast decode differs from sealed raw heads")
    check(set(row["replay_native_exact"]) == HEADS and all(row["replay_native_exact"].values()), "independent native/replay head equality differs")
    stages = {}
    for name, value in row["stages"].items():
        if prediction is None:
            continue
        if name in prediction["stages"]:
            stages[name] = h["difference"](value, prediction["stages"][name])
        else:
            errors.append("saved native intermediate has no sealed prediction: " + name)
    return {"errors": errors, "trace": trace, "ports": NativePortCatalog(trace), "root_joins": root_joins,
            "report": {"native_decode": row["decoded"], "forecast_decode": None if prediction is None else prediction["decoded"],
                "forecast_status": "horizon-ended" if prediction is None else "sealed forecast available",
                "RSI_bit1": {"observed": float(row["heads"]["regs"][0, 6, 1]),
                             "forecast": None if prediction is None else float(prediction["heads"]["regs"][0, 6, 1])},
                "forecast_native_root_joins": root_joins,
                "comparison_input_scope": "same literal roots" if all(root_joins.values()) else "different own-state roots; retain conditional comparison",
                "head_comparison": None if prediction is None else h["head_comparison"](row["heads"], prediction["heads"]),
                "stage_comparison": stages, "forecast_only_stages": [] if prediction is None else sorted(set(prediction["stages"]) - set(row["stages"]))}}


def audit_replay(row_id, row, proof, source, receipts, recipes, payloads):
    h, errors = helpers(), []
    trace, ports = source["trace"], source["ports"]
    program = StructuralCatalog(trace).program_id
    if len(receipts) != len(trace["events"]):
        errors.append("typed primitive receipt count differs from complete source replay")
    diagnostics = Counter()
    for pc, receipt in enumerate(receipts):
        try:
            h["public_receipt"](receipt)
            if (receipt["operation"] != "mdb.discovery.native-transition" or receipt["trace_fingerprint"] != trace["fingerprint"]
                    or receipt["program_id"] != program or receipt["before_pc"] != pc or receipt["after_pc"] != pc + 1
                    or receipt["operator"] != trace["events"][pc]["operator"] or receipt["mechanics_status"] != "valid"):
                errors.append("typed primitive/source binding differs atPC" + str(pc))
            diagnostics[str(receipt.get("exact_outputs"))] += 1
        except (KeyError, ValueError, IndexError) as error:
            errors.append("typed primitive integrity atPC" + str(pc) + ":" + str(error))
    if len(row["acts"]) != len(recipes):
        errors.append("typed Act inventory differs from declared recipe")
    acts = []
    for wrapper, recipe in zip(row["acts"], recipes):
        try:
            receipt = wrapper["receipt"]
            h["public_receipt"](receipt)
            port = ports.resolve(recipe["selector"], path=recipe.get("path"))
            if (wrapper["payload_name"] != recipe["payload_name"] or wrapper["port"] != port
                    or receipt["operation"] != "mdb.discovery.ssa-act" or receipt["trace_fingerprint"] != trace["fingerprint"]
                    or receipt["pc"] != port["available_pc"] or receipt["value_id"] != port["value_id"]
                    or receipt["act"]["reads"] != [port["value_id"]] or receipt["act"]["writes"] != [port["value_id"]]
                    or receipt["act"]["operation"]["id"] != "saturn.builtin.replace"
                    or receipt["act"]["numerical_contract"] != "behavioral-intervention"
                    or receipt["mechanics_status"] != "valid"):
                errors.append("typed Act source/port/recipe binding differs")
            value = payloads[recipe["payload_name"]]
            if recipe["patch_kind"] == "direct-window-columns":
                before, after = wrapper["before_value"], wrapper["after_value"]
                h["descriptor_join"](before, receipt["before"], "direct-window before descriptor differs")
                h["descriptor_join"](after, receipt["after"], "direct-window after descriptor differs")
                if not direct_columns_exact(before, after, value, row["stages"]["tokens"]):
                    errors.append("direct-window eleven columns or preserved282columns differ")
                applied = after
            else:
                if recipe["patch_kind"] != "replace":
                    errors.append("unknown declared Act patch kind")
                applied = value
                matched = [name for name, binding in row["actual_ports"].items()
                           if binding["selector"] == recipe["selector"]]
                if not matched or any(not np.array_equal(row["stages"][name], value) for name in matched):
                    errors.append("native controlled module return differs from frozen payload")
            h["descriptor_join"](applied, receipt["after"], "Act after differs from frozen replacement bytes")
            parameters = receipt["act"]["parameters"]["items"]
            if len(parameters) != 1 or parameters[0]["key"]["value"] != "value":
                errors.append("typed Act parameter inventory differs")
            else:
                h["descriptor_join"](applied, parameters[0]["value"], "Act implementation parameter bytes differ")
            acts.append({"fingerprint": receipt["fingerprint"], "payload_name": recipe["payload_name"], "port": port})
        except (KeyError, ValueError, TypeError, IndexError) as error:
            errors.append("typed Act integrity: " + str(error))
    return {"errors": errors, "source_trace": trace["fingerprint"], "source_program": program,
            "transitions": len(receipts), "source_descriptor_diagnostics": dict(diagnostics), "acts": acts}


def expected_source(row_id):
    parts = row_id.split(".")
    if parts[0] == "sequence":
        return "sequence.native.0" if parts[2] == "0" else row_id
    if parts[0] == "path":
        return "sequence.native.0"
    return ".".join(parts[:2]) + ".native"


def frozen_payload_checks(forecasts, payloads, descriptors, tree):
    native = forecasts["trajectories"]["sequence.native"]["steps"]["0"]
    expected = {"zero": np.zeros((1, 64, 48), dtype=np.float32),
                "sequence.U_native": native["stages"]["layer2.U"], "sequence.P2_native": native["stages"]["layer2.P2"],
                "path.z_new": forecasts["path_rows"]["path.new.old"]["stages"]["z"],
                "path.M_other": forecasts["path_rows"]["path.new.new"]["stages"]["layer2.M"],
                "path.P2_own": forecasts["path_rows"]["path.old.old"]["stages"]["layer2.P2"]}
    for name in ("path.old.new", "path.new.new"):
        expected[name + ".direct"] = forecasts["path_rows"][name]["stages"]["tokens"][:, :, DIRECT]
    result = {}
    for name in sorted(set(expected) | set(payloads)):
        errors = []
        if name not in expected or name not in payloads:
            result[name] = {"errors": ["frozen payload inventory differs"]}
            continue
        supplied, descriptor = payloads[name], descriptors[name]
        original = tree.decode(descriptor["original"])
        if supplied.dtype != np.float32 or not np.array_equal(supplied, np.asarray(expected[name], dtype=np.float32)):
            errors.append("frozen FP32 payload differs from its declared sealed forecast endpoint")
        if (descriptor["cast"]["from"] != str(original.dtype) or descriptor["cast"]["to"] != str(supplied.dtype)
                or not np.array_equal(supplied, np.asarray(original, dtype=supplied.dtype))):
            errors.append("payload original/staged cast lineage differs")
        result[name] = {"errors": errors, "shape": list(supplied.shape), "value_npy_sha256": descriptor["value"]["sha256"],
                        "original_npy_sha256": descriptor["original"]["sha256"], "provenance": descriptor["provenance"]}
    return result


def worker_comparison(row, retained, catalog, input_identity):
    errors = []
    if row["head_comparison"] is None:
        return {"errors": [], "status": "forecast horizon ended; no fabricated forecast comparison"}
    if catalog is None:
        return {"errors": ["retained generic equation catalog missing"]}
    try:
        record = catalog.show(retained["equation_record"])
        if record["report"] != retained["head_report"] or record["tested_context"]["input_identity"] != input_identity:
            errors.append("generic equation record/report/input binding differs")
        for name, independent in row["head_comparison"].items():
            actual = retained["head_report"]["fields"][name]
            if (actual["residual"]["max_abs"] != independent["residual"]["max_abs"]
                    or actual["decisions"]["different"] != independent["decision_differences"]):
                errors.append("retained head comparison differs: " + name)
        for name, independent in row["stage_comparison"].items():
            if retained["stage_report"]["fields"][name]["residual"]["max_abs"] != independent["max_abs"]:
                errors.append("retained intermediate comparison differs: " + name)
        return {"errors": errors, "equation_record": record["fingerprint"]}
    except (KeyError, ValueError, TypeError, OSError) as error:
        return {"errors": [*errors, "retained comparison/catalog instrument: " + str(error)]}


def prior_attempt_checks(run, current_forecast):
    h, data, errors = helpers(), run["data"], []
    path = data / "prior-attempt.json"
    if not path.is_file():
        return {"status": "not supplied", "errors": []}
    retained = h["read"](path)
    copied = {}
    for name in ("transaction-report.json", "native-events.json", "failure.json", "transaction/seal.json", "transaction/forecasts/campaign.json"):
        copy_path = h["file_path"](data, "prior-" + name.replace("/", "-"))
        if (h["sha"](copy_path.read_bytes()) != retained["original_files"][name]["sha256"]
                or copy_path.stat().st_size != retained["original_files"][name]["bytes"]):
            errors.append("copied original metadata differs: " + name)
        copied[name] = h["read"](copy_path)
    report, seal = copied["transaction-report.json"], copied["transaction/seal.json"]
    seal_sha = h["sha"]((data / "prior-transaction-seal.json").read_bytes())
    if seal_sha != report["seal_sha256"] or seal_sha != retained["verification"]["seal_sha256"]:
        errors.append("original forecast seal identity differs")
    previous = None
    for event in report["events"]:
        if event["previous"] != previous or event["work"]["native_calls"] != 0:
            errors.append("original phase metadata chain/native exclusion differs")
        previous = h["sha"](h["canonical"](event))
    if previous != report["event_head"] or copied["native-events.json"] or report["work"]["native_calls"] != 0:
        errors.append("original failed attempt had native exposure or changed event head")
    original_raw = (data / "prior-transaction-forecasts-campaign.json").read_bytes()
    if h["sha"](original_raw) != seal["files"]["forecasts/campaign.json"]["sha256"]:
        errors.append("original copied forecasts differ from original pre-exposure seal")

    def identities(value, path=()):
        result = {}
        if isinstance(value, dict):
            if value.get("kind") == "array":
                result[path] = {key: value[key] for key in ("sha256", "content_sha256", "shape", "dtype")}
            else:
                for key, child in value.items():
                    result.update(identities(child, (*path, key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                result.update(identities(child, (*path, index)))
        return result

    old, current = identities(copied["transaction/forecasts/campaign.json"]), identities(current_forecast)
    changed = [list(key) for key, value in current.items() if old.get(key) != value]
    if changed or retained["changed_retained_array_paths"] or len(current) != retained["retained_array_occurrences"]:
        errors.append("retained forecast leaf identities changed after instrument/publication repair")
    if retained["status"] != "verified" or not retained["verification"]["verified"]:
        errors.append("worker original-full-custody check did not verify")
    return {"status": "verified metadata" if not errors else "differs", "errors": errors, "job_id": retained["job_id"],
        "metadata_sha256": h["sha"](path.read_bytes()), "original_seal_sha256": seal_sha,
        "original_event_head": report["event_head"], "original_native_calls": report["work"]["native_calls"],
        "original_failure": copied["failure.json"], "retained_array_occurrences": len(current), "changed_paths": changed,
        "omitted_original_unused_array_occurrences": len(set(old) - set(current)),
        "scope": "copied original seals/events/forecast descriptors checked offline; original numerical bytes were checked by the source-bound worker, not reloaded here"}


def consumer_effects(observations, forecasts, retained):
    result = {}
    for row_id, row in observations.items():
        baseline_id = ("sequence.native." + row_id.rsplit(".", 1)[1] if row_id.startswith("sequence.") else
                       "sequence.U_swap.0" if row_id.startswith("path.") else ".".join(row_id.split(".")[:2]) + ".native")
        baseline = observations.get(baseline_id)
        try:
            prediction = select(forecasts, row["prediction_path"])
        except KeyError:
            prediction = None
        word = row["decoded"]["register_words"][7]
        prior_word = row["before"]["register_words"][7]
        bits = {0, 1, 2} | {bit for bit in range(64) if (word ^ prior_word) & (1 << bit)}
        if baseline is not None:
            bits.update(bit for bit in range(64) if (word ^ baseline["decoded"]["register_words"][7]) & (1 << bit))
        terms = {} if prediction is None else prediction["register_terms"]
        result[row_id] = {"baseline_row": baseline_id, "row_program_hex": row["lineage"]["program_hex"],
            "baseline_program_hex": None if baseline is None else baseline["lineage"]["program_hex"],
            "same_serving_roots": baseline is not None and all(np.array_equal(row["roots"][key], baseline["roots"][key]) for key in row["roots"]),
            "contrast_scope": retained.get(row_id, {}).get("contrast_scope"),
            "RSI_bit1": {"observed": float(row["heads"]["regs"][0, 6, 1]),
                "effect_vs_baseline": None if baseline is None else float(row["heads"]["regs"][0, 6, 1]) - float(baseline["heads"]["regs"][0, 6, 1]),
                "forecast": None if prediction is None else float(prediction["heads"]["regs"][0, 6, 1]),
                "sealed_receiving_terms": {name: float(value[0, 6, 1]) for name, value in terms.items()}},
            "RDI": {"before": prior_word, "observed": word,
                "forecast": None if prediction is None else prediction["decoded"]["register_words"][7],
                "changed_bits_vs_baseline": [] if baseline is None else [bit for bit in range(64) if (word ^ baseline["decoded"]["register_words"][7]) & (1 << bit)],
                "logit_effect_by_bit": None if baseline is None else (row["heads"]["regs"][0, 7].astype(np.float64) - baseline["heads"]["regs"][0, 7]).tolist(),
                "sealed_receiving_terms": {str(bit): {name: float(value[0, 7, bit]) for name, value in terms.items()} for bit in sorted(bits)}},
            "receiving_term_scope": "sealed forecast decomposition only; no native readout/model equations recomputed"}
    return result


def audit(run_dir, derived_dir=None, failed=False):
    h = helpers()
    run = h["source_run"](run_dir, "state-transition-result", failed=failed)
    tx, plan = run["transaction"], run["plan"]
    publication = Path(derived_dir) if derived_dir is not None else run["data"]
    forecast_doc = h["read"](tx / "forecasts/campaign.json")
    # NumericalTree is a thin public load_forecast_tree wrapper that converts
    # immutable mapping containers to ordinary JSON-compatible containers.
    forecasts = run["tree"].decode(forecast_doc["predictions"])
    observations = run["tree"].decode(h["read"](tx / "observations/native.json")["value"])
    payloads = {key: run["tree"].decode(value["value"]) for key, value in forecast_doc["payloads"].items()}
    receipts = defaultdict(list)
    for wrapper in h["read"](run["data"] / "public-replay-receipts.json"):
        receipts[wrapper["row_id"]].append(wrapper["receipt"])
    rows, proofs, global_errors = {}, {}, []
    comparison_path = publication / "comparisons.json"
    comparison_rows = h["read"](comparison_path)["rows"] if comparison_path.is_file() else []
    optional_publication_failed = failed and derived_dir is None and not comparison_rows
    retained = {row["row_id"]: row for row in comparison_rows}
    if len(retained) != len(comparison_rows):
        global_errors.append("duplicate retained worker comparison row")
    catalog_path = publication / "equations"
    catalog = EquationCatalog(catalog_path) if catalog_path.is_dir() and not catalog_path.is_symlink() else None
    for row_id, row in observations.items():
        try:
            try:
                prediction = select(forecasts, row["prediction_path"])
            except KeyError:
                if not row_id.startswith("sequence."):
                    raise
                arm, step = row_id.split(".")[1:]
                if row["prediction_path"] != ["trajectories", "sequence." + arm, "steps", step]:
                    raise ValueError("sequence prediction path differs") from None
                prediction = None
            proof = audit_row(run, row_id, row, prediction)
            proofs[row_id] = proof
            rows[row_id] = {"panel": row["panel"], "mechanics_errors": proof["errors"], **proof["report"]}
            comparison = ({"errors": [], "status": "optional worker publication failed; independently audited from retained raw arrays"}
                          if optional_publication_failed else worker_comparison(rows[row_id], retained.get(row_id, {}), catalog, row["input_identity"]))
            rows[row_id]["retained_worker_comparison"] = comparison
            rows[row_id]["mechanics_errors"].extend(comparison["errors"])
        except (KeyError, ValueError, TypeError, IndexError) as error:
            rows[row_id] = {"panel": row.get("panel"), "mechanics_errors": [f"row instrument: {type(error).__name__}: {error}"]}
    for row_id, row in observations.items():
        try:
            source_id = row["replay_source_row"]
            if source_id != expected_source(row_id):
                rows[row_id]["mechanics_errors"].append("declared replay source differs from panel/common-parent policy")
            replay = audit_replay(row_id, row, proofs[row_id], proofs[source_id], receipts[row_id], plan["row_recipes"].get(row_id, []), payloads)
            rows[row_id]["replay"] = replay
            rows[row_id]["mechanics_errors"].extend(replay["errors"])
        except (KeyError, ValueError, TypeError) as error:
            rows[row_id]["mechanics_errors"].append("replay source instrument: " + str(error))
    native_chains, forecast_chains = {}, {}
    for arm in plan["arms"]:
        prefix = "sequence." + arm + "."
        steps = {key[len(prefix):]: {**dict(value), "row_id": key} for key, value in observations.items() if key.startswith(prefix)}
        native_chains[arm] = chain_checks(steps, plan, native=True)
        if not steps:
            global_errors.append("native continuation chain missing: " + arm)
        forecast_chains[arm] = chain_checks(forecasts["trajectories"]["sequence." + arm]["steps"], plan, native=False)
        for step, check in native_chains[arm].items():
            rows[prefix + step]["native_chain"] = check
            rows[prefix + step]["mechanics_errors"].extend(check["errors"])
    aliases = {}
    for alias, source in plan["panel_reuses"].items():
        prediction = forecasts["path_rows"][alias]
        origin = select(forecasts, observations[source]["prediction_path"])
        aliases[alias] = {"observation_row": source, "physical_duplicate": False,
            "forecast_numeric_equal": all(set(prediction[field]) == set(origin[field]) and all(
                np.array_equal(value, origin[field][key]) for key, value in prediction[field].items())
                for field in ("heads", "roots", "stages"))}
        if not aliases[alias]["forecast_numeric_equal"]:
            global_errors.append("reused path forecast differs from sequence source: " + alias)
    if set(receipts) != set(observations):
        global_errors.append("typed transition row inventory differs from observations")
    summary_path = publication / "summary.json"
    summary = (h["read"](summary_path) if summary_path.is_file() else {"counts": {
        "physical_forwards": run["transaction_report"]["work"]["native_calls"],
        "replay_operations": sum(len(value) for value in receipts.values()),
        "acts": sum(len(row["acts"]) for row in observations.values())},
        "origin": "offline raw transaction/receipt counts; optional worker summary absent"})
    work = summary["counts"]
    if work["physical_forwards"] != run["transaction_report"]["work"]["native_calls"]:
        global_errors.append("physical call count differs from public transaction native work")
    if work["physical_forwards"] > plan["budgets"]["max_physical_forwards"]:
        global_errors.append("declared finite physical call budget exceeded")
    native_events = h["read"](run["data"] / "native-events.json")
    if len(native_events) != work["physical_forwards"] or [event["index"] for event in native_events] != list(range(1, len(native_events) + 1)):
        global_errors.append("physical native entry ledger/count differs")
    expected_panels = {f"mlp.{operation}.{arm}" for operation in plan["mlp_programs"]
                       for arm in ("native", "zero_post0", "zero_post1", "zero_post2")}
    expected_panels.update(set(forecasts["path_rows"]) - set(plan["panel_reuses"]))
    missing = sorted(expected_panels - set(observations))
    if missing:
        global_errors.append("planned matched-path/MLP observations missing: " + ",".join(missing))
    if set(plan["panel_reuses"]) & set(observations):
        global_errors.append("reused panel aliases must not add physical observations")
    if len(observations) != work["physical_forwards"]:
        global_errors.append("physical model call count differs from retained native capture count")
    if sum(len(value) for value in receipts.values()) != work["replay_operations"]:
        global_errors.append("typed primitive total differs from worker count")
    if sum(len(row["acts"]) for row in observations.values()) != work["acts"]:
        global_errors.append("typed Act total differs from worker count")
    forecast_errors = {arm: {key: value["errors"] for key, value in chain.items() if value["errors"]}
                       for arm, chain in forecast_chains.items()}
    frozen = frozen_payload_checks(forecasts, payloads, forecast_doc["payloads"], run["tree"])
    payload_errors = {key: value["errors"] for key, value in frozen.items() if value["errors"]}
    prior = prior_attempt_checks(run, forecast_doc)
    global_errors.extend(prior["errors"])
    row_errors = {key: value["mechanics_errors"] for key, value in rows.items() if value["mechanics_errors"]}
    scientific = {"decision_mismatch_rows": [key for key, row in rows.items() if row.get("head_comparison")
                      and any(field["decision_differences"] for field in row["head_comparison"].values())],
                  "different_own_state_root_rows": [key for key, row in rows.items() if row.get("head_comparison")
                      and not all(row["forecast_native_root_joins"].values())],
                  "forecast_horizon_ended_rows": [key for key, row in rows.items() if row.get("forecast_status") == "horizon-ended"]}
    weights = run["tree"].decode(run["seal"]["declarations"]["actual_weight_manifest"])
    closure = {name: {"chains": len(chains), "steps": sum(len(chain) for chain in chains.values()),
                     "closed_steps": sum(not row["errors"] for chain in chains.values() for row in chain.values())}
               for name, chains in (("native", native_chains), ("independent_forecast", forecast_chains))}
    return {"schema": "mdb-state-transition-offline-audit-v1", "source": run["source"],
        "mechanics_status": "instrument-errors" if global_errors or row_errors or payload_errors or any(forecast_errors.values()) else "valid",
        "global_mechanics_errors": global_errors, "row_mechanics_errors": row_errors,
        "forecast_chain_errors": forecast_errors, "native_chains": native_chains, "forecast_chains": forecast_chains,
        "rows": rows, "panel_reuses": aliases, "frozen_payloads": frozen, "payload_mechanics_errors": payload_errors,
        "scientific_observations": scientific,
        "consumer_effects": consumer_effects(observations, forecasts, retained), "own_chain_closure": closure,
        "previous_instrument_attempt": prior,
        "parameter_inventory": {"arrays": len(weights), "scalars": sum(value.size for value in weights.values())},
        "counts": work, "terminal_status": "unassessed",
        "worker_status": "failed; raw observations retained before optional publication" if failed else "succeeded",
        "worker_failure": h["read"](run["data"] / "failure.json") if failed else None,
        "derived_publication": {"summary": summary, "files_sha256": {path.relative_to(publication).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(publication.rglob("*")) if path.is_file() and (path.parent == catalog_path or path in (summary_path, comparison_path))}},
        "trend_status": "all per-row decisions and signed intermediate residuals retained; no scientific numeric gate",
        "model_calls": 0, "equation_evaluations": 0,
        "analysis_sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (Path(__file__).name, "summarize_context_window.py")},
        "limits": ["Host supplies segment names/programs; this does not discover instruction semantics.",
            "Forecast and native chains own separate prior state and may diverge; forecasts are never reconditioned here.",
            "Replay equality uses source-bound worker comparisons plus offline typed receipt validation; no ATen reexecution.",
            "Frozen payload endpoints/casts and consumption are verified; source model equations are not reexecuted.",
            "Trusted callback/module guards and seals are custody instruments, not OS isolation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--failed", action="store_true", help="audit retained complete observations from a failed optional publication")
    parser.add_argument("--derived-dir", type=Path, help="posthoc comparisons/summary/equations outside immutable raw results")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to((args.run_dir / "extracted").resolve()):
        parser.error("offline analysis output must stay outside immutable raw results")
    result = audit(args.run_dir, derived_dir=args.derived_dir, failed=args.failed)
    snapshot = args.output.with_name(args.output.stem + "-source.py")
    source = Path(__file__).read_bytes()
    if snapshot.exists():
        if snapshot.read_bytes() != source:
            parser.error("immutable analysis source snapshot already differs")
    else:
        with snapshot.open("xb") as stream:
            stream.write(source)
    result["analysis_source"] = {"snapshot": snapshot.name, "sha256": hashlib.sha256(source).hexdigest(),
        "numpy_version": np.__version__, "public_metadata_modules": {
            name: hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest() for name in
            ("mdb.forecast_transaction", "mdb.forecast_storage", "mdb.discovery_catalog", "mdb.forward_ports", "mdb.forward_symbols", "mdb.equation_comparison")}}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "mechanics_status": result["mechanics_status"], "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
