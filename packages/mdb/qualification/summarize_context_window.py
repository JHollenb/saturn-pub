"""Audit a context-window/P2-transfer panel from immutable evidence only.

The three prior mixed-window rows remain reused observations. Seven fresh rows
are checked against a new globally sealed raw-input forecast. Natural diagonal
P2 writes and off-diagonal frozen analytical payloads are distinguished; this
is not a bit-exact fixed-payload 2x2 factorial. No model/equation is executed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from mdb.discovery_catalog import StructuralCatalog
from mdb.forecast_transaction import load_forecast_tree, verify_transaction
from mdb.forward_ports import NativePortCatalog
from mdb.forward_symbols import verify_trace

CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"
FRESH = {"old.native", "old.zero", "old.U_swap", "old.U_swap_P2native", "new.zero",
         "old.U_swap_P2other", "new.U_swap_P2other"}
REUSED = {"new.native", "new.U_swap", "new.U_swap_P2native"}
HEADS = {"regs", "flags", "dest", "src", "width", "length", "syscall", "imm",
         "store", "store_addr", "store_value"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("nonfinite JSON token: " + value)
    return json.loads(path.read_text(), parse_constant=invalid)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe artifact path")
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            "missing or escaped artifact: " + str(relative))
    return path


class NumericalTree:
    def __init__(self, transaction_root):
        self.root = transaction_root

    def decode(self, node):
        def containers(value):
            if isinstance(value, Mapping):
                return {key: containers(child) for key, child in value.items()}
            if isinstance(value, tuple):
                return [containers(child) for child in value]
            return value
        return containers(load_forecast_tree(self.root, node))

    def select(self, node, path):
        for component in path:
            require(node["kind"] in {"mapping", "list", "tuple"}, "numerical descriptor path is not a container")
            node = node["items"][component]
        return self.decode(node)


def source_run(root, data_name, *, failed=False, max_archive_bytes=128 << 20, max_member_bytes=64 << 20,
               recover_metadata=False):
    require(type(max_archive_bytes) is int and 0 < max_archive_bytes <= 512 << 20,
            "explicit archive allowance must be positive and at most 512 MiB")
    require(type(max_member_bytes) is int and 0 < max_member_bytes <= 256 << 20,
            "explicit member allowance must be positive and at most 256 MiB")
    root = Path(root).resolve()
    data = root / "extracted" / data_name
    request, report, job = (read(root / name) for name in ("request.json", "report.json", "job.json"))
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    require(sha(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()) == request["request_sha256"]
            == report["request_sha256"], "external request/report seal differs")
    require(report["job_id"] == job["job_id"] and job["state"] == ("failed" if failed else "succeeded"),
            "external job state differs from the declared attempt role")
    for name, expected in request["source_hashes"].items():
        require(sha(file_path(root / "payload", name).read_bytes()) == expected, "request-bound staged source differs: " + name)
    inventory = {path.relative_to(data).as_posix() for path in data.rglob("*") if path.is_file()}
    published_manifest = (data / "files.json").is_file()
    require(published_manifest or failed, "completed raw result lacks a file manifest")
    manifest = (read(data / "files.json") if published_manifest
                else {name: sha(file_path(data, name).read_bytes()) for name in sorted(inventory)})
    for name, expected in manifest.items():
        require(sha(file_path(data, name).read_bytes()) == expected, "raw artifact differs from its result manifest: " + name)
    require(inventory == set(manifest) | ({"files.json"} if published_manifest else set()),
            "raw result inventory differs from manifest")
    marker = "MDB_DISCOVERY_UPGRADES_ARCHIVE_V1 "
    archives = [line[len(marker):].strip() for line in (root / "worker.log").read_text().splitlines()
                if line.startswith(marker)]
    require(len(archives) == 1, "collector log must contain exactly one result archive")
    archive_raw = base64.b64decode(archives[0], validate=True)
    require(len(archive_raw) <= max_archive_bytes and sha(archive_raw) == report["archive_sha256"],
            "result archive bytes differ from collector binding")
    archived = {}
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            relative = Path(member.name)
            require(member.isfile() and relative.parts[0] == data_name
                    and not relative.is_absolute() and ".." not in relative.parts and member.size <= max_member_bytes,
                    "archive contains an unsafe/unexpected member")
            key = Path(*relative.parts[1:]).as_posix()
            require(key not in archived, "duplicate archive member")
            archived[key] = sha(archive.extractfile(member).read())
    require(set(archived) == inventory and all(sha(file_path(data, name).read_bytes()) == digest
                for name, digest in archived.items()), "extracted result differs from retained archive")
    tx = data / "transaction"
    recovered = {}
    if not (data / "transaction-report.json").is_file():
        require(failed and recover_metadata, "missing transaction report requires explicit failed-publication metadata recovery")
        # The raw archive has already been checked. Reconstruct reporting fields
        # from the public sealed transaction without altering any raw file.
        checked = verify_transaction(tx)
        saved_events = [read(path) for path in sorted((tx / "events").glob("*.json"))]
        require(saved_events and checked["phase"] == "closed"
                and any(row["kind"] == "observation_persisted" for row in saved_events)
                and not any(row["kind"] in {"failed", "violation"} for row in saved_events),
                "metadata recovery requires a closed transaction with recorded observation and no transaction failure")
        tx_report = {"schema": "mdb-forecast-transaction-v1", "phase": checked["phase"],
            "seal_sha256": checked["seal_sha256"], "event_head": checked["event_head"],
            "events": saved_events, "work": saved_events[-1]["work"], "failure": None,
            "violations": read(tx / "seal.json")["violations"],
            "frontiers": sorted(path.stem for path in (tx / "frontiers").glob("*.json")),
            "forecasts": sorted(path.stem for path in (tx / "forecasts").glob("*.json")),
            "observations": sorted(path.stem for path in (tx / "observations").glob("*.json"))}
        recovered["transaction_report"] = "derived from verified durable public transaction and phase receipts"
    else:
        tx_report = read(data / "transaction-report.json")
    verification = verify_transaction(tx, expected_seal_sha256=tx_report["seal_sha256"],
                                      expected_event_head=tx_report["event_head"])
    events = [read(path) for path in sorted((tx / "events").glob("*.json"))]
    require(events == tx_report["events"], "reported phase events differ from their public custody chain")
    global_seals = [event for event in events if event["kind"] == "globally_sealed"]
    releases = [event for event in events if event["kind"] == "native_execution_released"]
    require(len(global_seals) == len(releases) == 1 and global_seals[0]["sequence"] < releases[0]["sequence"],
            "global forecast seal/native release ordering differs")
    require(all(event["work"]["native_calls"] == 0 for event in events
                if event["sequence"] <= global_seals[0]["sequence"]), "native execution preceded global raw-input forecast seal")
    seal = read(tx / "seal.json")
    require(seal["pre_release_work"]["native_calls"] == 0 and not seal["violations"],
            "forecast/native exclusion instrument recorded a pre-release native call/violation")
    source = {"job_id": job["job_id"], "request_sha256": request["request_sha256"],
        "request_file_sha256": sha((root / "request.json").read_bytes()),
        "report_file_sha256": sha((root / "report.json").read_bytes()),
        "archive_sha256": report.get("archive_sha256"), "verified_source_hashes": request["source_hashes"],
        "verified_artifact_files": len(manifest),
        "manifest_sha256": sha((data / "files.json").read_bytes()) if published_manifest else None,
        "artifact_inventory_authority": "published manifest plus retained archive" if published_manifest else "failed-attempt retained archive; no completed result manifest",
        "transaction": verification}
    if (data / "plan.json").is_file():
        plan = read(data / "plan.json")
    else:
        require(failed and recover_metadata, "missing raw plan requires explicit failed-publication metadata recovery")
        plan = read(root / "plan.json")
        require(plan == request["config"], "recovered plan differs from request-bound submitted configuration")
        recovered["plan"] = "request-bound submitted configuration; no raw plan was published"
    if recovered:
        source["metadata_recovery"] = recovered
    return {"root": root, "data": data, "request": request, "report": report, "manifest": manifest,
            "transaction": tx, "transaction_report": tx_report, "seal": seal, "events": events,
            "source": source, "tree": NumericalTree(tx), "plan": plan}


def difference(observed, reference):
    observed, reference = np.asarray(observed), np.asarray(reference)
    require(observed.shape == reference.shape, "numerical comparison geometry differs")
    residual = observed.astype(np.float64) - reference.astype(np.float64)
    index = int(np.argmax(np.abs(residual))) if residual.size else None
    return {"shape": list(residual.shape), "elements": int(residual.size),
        "changed_elements": int(np.count_nonzero(residual)), "max_abs": float(np.abs(residual).max()) if residual.size else 0.,
        "l2": float(np.linalg.norm(residual.reshape(-1))), "signed_sum": float(residual.sum()),
        "signed_mean": float(residual.mean()) if residual.size else 0.,
        "minimum_signed": float(residual.min()) if residual.size else 0.,
        "maximum_signed": float(residual.max()) if residual.size else 0.,
        "largest_effect": None if index is None else {"coordinate": [int(value) for value in np.unravel_index(index, residual.shape)],
                                                     "signed": float(residual.reshape(-1)[index])}}


def head_comparison(observed, predicted):
    require(set(observed) == set(predicted) == HEADS, "comparison must retain all eleven heads")
    result = {}
    for name in sorted(HEADS):
        actual, forecast = observed[name], predicted[name]
        left, right = ((actual.argmax(axis=-1), forecast.argmax(axis=-1)) if name in {"dest", "src", "width", "length"}
                       else (actual > 0, forecast > 0))
        changed = np.argwhere(left != right)
        result[name] = {"residual": difference(actual, forecast),
            "decision_rule": "argmax" if name in {"dest", "src", "width", "length"} else "strict-logit>0",
            "decision_differences": int(len(changed)), "different_coordinates": changed.tolist()}
    return result


def decoded(heads):
    return {"register_words": [sum(1 << bit for bit, value in enumerate(word) if value > 0) for word in heads["regs"][0]],
            "flags": [int(value > 0) for value in heads["flags"][0]],
            "length": int(heads["length"][0].argmax()) + 1,
            "store": bool(heads["store"][0] > 0), "syscall": bool(heads["syscall"][0] > 0)}


def hard_changes(observed, baseline):
    left, right = decoded(observed), decoded(baseline)
    changed = {}
    for register, (a, b) in enumerate(zip(left["register_words"], right["register_words"], strict=True)):
        if a != b:
            changed[str(register)] = [bit for bit in range(64) if (a ^ b) & (1 << bit)]
    return {"observed": left, "reference": right, "register_changed_bits": changed,
            "register_changed_bit_count": sum(map(len, changed.values())),
            "flags_changed": [index for index, (a, b) in enumerate(zip(left["flags"], right["flags"], strict=True)) if a != b],
            "length_changed": left["length"] != right["length"], "store_changed": left["store"] != right["store"],
            "syscall_changed": left["syscall"] != right["syscall"]}


def raw_roots(program_hex, before):
    program = bytes.fromhex(program_hex)
    window = program[before["offset"]:before["offset"] + 15].ljust(15, b"\0")
    words = before["register_words"]
    words = [int(word, 16) if isinstance(word, str) else int(word) for word in words]
    return {"window": np.array([[(byte >> bit) & 1 for byte in window for bit in range(8)]], dtype=np.float32),
            "regs": np.array([[[(word >> bit) & 1 for bit in range(64)] for word in words]], dtype=np.float32),
            "flags": np.asarray([before["flags"]], dtype=np.float32)}


def tensor_descriptor(value):
    value = np.asarray(value)
    require(value.dtype == np.float32, "native observed tensor is not FP32")
    return {"shape": list(value.shape), "dtype": "torch.float32", "sha256": sha(value.astype("<f4", copy=False).tobytes())}


def descriptor_join(value, descriptor, message):
    require(all(descriptor.get(key) == expected for key, expected in tensor_descriptor(value).items()), message)


def public_receipt(receipt):
    require(receipt.get("schema") == "saturn-pub-receipt-v1"
            and sha(canonical({key: value for key, value in receipt.items() if key != "fingerprint"})) == receipt["fingerprint"],
            "public typed receipt fingerprint differs")


def expected_stage_ports():
    result = {"z": "module:decoder:return", "input": "module:inp:return",
              "query": "module:query:return", "offset_head": "module:offset_head:return"}
    suffixes = {"M": "mixers.{layer}", "in_projected": "mixers.{layer}.in_proj",
                "selected": "mixers.{layer}.select", "dt_affine": "mixers.{layer}.dt_proj",
                "U": "mixers.{layer}.out_proj", "P2": "post.{layer}",
                "post2_norm": "post.{layer}.0", "post2_affine": "post.{layer}.1",
                "post2_gelu": "post.{layer}.2", "P2_computed": "post.{layer}.3"}
    for layer in range(3):
        result.update({f"layer{layer}." + name: "module:" + module.format(layer=layer) + ":return"
                       for name, module in suffixes.items()})
    result.update({"layer2.S": "module:mixers.2:input", "layer2.h": "module:reg_out:input"})
    return result


def verify_observation(run, row_id, observation, prediction, program_hex, *, capture_relative=None):
    capture_relative = capture_relative or "captures/" + row_id
    trace = read(file_path(run["data"], capture_relative + "/trace.json"))
    verify_trace(trace)
    require(trace["model_identity"] == CHECKPOINT and trace["fingerprint"] == observation["trace_fingerprint"]
            and trace["input_identity"] == observation["input_identity"], "actual trace/input identity differs")
    require(trace["execution"] == {"backend": "cpu", "dtype": "float32", "threads": 1},
            "native numerical execution profile differs")
    ports = NativePortCatalog(trace)
    require(set(trace["outputs"]) == HEADS, "actual native trace omits/adds a head")
    expected_ports = expected_stage_ports()
    require(set(observation["actual_ports"]) <= set(expected_ports), "unexpected actual stage binding")
    for name, binding in observation["actual_ports"].items():
        path = ["args", 0] if name in {"layer2.S", "layer2.h"} else None
        resolved = ports.resolve(expected_ports[name], path=path)
        require(resolved == binding, "recorded actual module port differs: " + name)
        if name in observation["stages"]:
            descriptor_join(observation["stages"][name], trace["values"][resolved["value_id"]],
                            "actual return tensor differs from its authoritative port: " + name)
    roots = raw_roots(program_hex, observation["before"])
    require(set(observation["roots"]) == set(roots), "actual raw root inventory differs")
    for name, value in roots.items():
        require(np.array_equal(value, observation["roots"][name]), "actual root differs from its literal own state: " + name)
        if "input_roots" in observation:
            descriptor_join(value, observation["input_roots"][name], "actual root descriptor differs")
    declared_roots = sorted(sha(canonical(tensor_descriptor(value))) for value in roots.values())
    traced_roots = sorted(sha(canonical({key: value[key] for key in ("shape", "dtype", "sha256")}))
                          for value in trace["values"].values() if value["origin"] == "input")
    require(declared_roots == traced_roots, "traced input roots differ from all three actual serving arrays")
    root_joins = {name: bool(np.array_equal(prediction["roots"][name], value)) for name, value in roots.items()}
    require(all(root_joins.values()) == observation["forecast_root_join"], "declared forecast/raw-input join differs")
    require(decoded(observation["heads"]) == observation["decoded"], "observed decoded state differs from raw heads")
    require(decoded(prediction["heads"]) == prediction["decoded"], "forecast decoded state differs from sealed raw heads")
    require(set(observation["replay_native_exact"]) == HEADS, "native/replay comparison omits a head")
    for name, encoded in trace["outputs"].items():
        require(set(encoded) == {"$value"}, "native head does not have one SSA return")
        descriptor_join(observation["heads"][name], trace["values"][encoded["$value"]],
                        "native head differs from its actual trace output")
    for act in observation.get("acts", []):
        public_receipt(act.get("receipt", act))
    # The final FP32 add is reconstructed from two retained actual returns and
    # checked at its actual reg_out input, not guessed from an inferred scope.
    h_binding = ports.resolve("module:reg_out:input", path=["args", 0])
    h = np.asarray(observation["stages"]["layer2.M"] + observation["stages"]["layer2.P2"], dtype=np.float32)
    descriptor_join(h, trace["values"][h_binding["value_id"]], "actual final carrier+write add does not match reg_out input")
    if "layer2.h" in observation["stages"]:
        require(np.array_equal(h, observation["stages"]["layer2.h"]), "retained final h differs from actual input port")
    comparison = head_comparison(observation["heads"], prediction["heads"])
    stage_comparison = {name: difference(value, prediction["stages"][name])
                        for name, value in observation["stages"].items()}
    return {"trace": trace, "ports": ports, "h": h,
            "proof": {"trace_fingerprint": trace["fingerprint"], "actual_ports": len(observation["actual_ports"]),
                "actual_h_port": h_binding, "literal_root_joins": {name: True for name in roots},
                "forecast_root_joins": root_joins,
                "native_replay_heads": observation["replay_native_exact"],
                "head_comparison": comparison, "stage_comparison": stage_comparison,
                "forecast_decode": prediction["decoded"], "actual_decode": observation["decoded"]}}


def stage(observation, name):
    return np.asarray(observation["stages"][name], dtype=np.float64)


def recipient_transfer(recipient, donor, cross, observations, proofs, predictions, payload, weight):
    before, source, after = (observations[name] for name in (recipient, donor, cross))
    weight = np.asarray(weight, dtype=np.float64)
    require(weight.shape == (8, 48), "full register readout weight geometry differs")
    invariant = {"M": bool(np.array_equal(before["stages"]["layer2.M"], after["stages"]["layer2.M"])),
        "U": bool(np.array_equal(before["stages"]["layer2.U"], after["stages"]["layer2.U"])),
        "z": bool(np.array_equal(before["stages"]["z"], after["stages"]["z"])),
        "before": before["before"] == after["before"],
        "roots": all(np.array_equal(before["roots"][name], after["roots"][name]) for name in ("window", "regs", "flags"))}
    require(np.array_equal(payload, after["stages"]["layer2.P2"]), "cross-write return differs from frozen analytical payload")
    actual_donor = stage(source, "layer2.P2")
    actual_recipient = stage(before, "layer2.P2")
    applied = np.asarray(payload, dtype=np.float64)
    project = lambda values: (values @ weight.T).transpose(0, 2, 1)
    donor_difference = project(actual_donor - actual_recipient)
    prediction_residual = project(applied - actual_donor)
    write_effect = project(applied - actual_recipient)
    carrier_effect = project(stage(after, "layer2.M") - stage(before, "layer2.M"))
    h_effect = project(np.asarray(proofs[cross]["h"], dtype=np.float64) - np.asarray(proofs[recipient]["h"], dtype=np.float64))
    head_effect = np.asarray(after["heads"]["regs"], dtype=np.float64) - np.asarray(before["heads"]["regs"], dtype=np.float64)
    add_rounding = h_effect - write_effect - carrier_effect
    readout_rounding = head_effect - h_effect
    predicted_effect = (np.asarray(predictions[cross]["heads"]["regs"], dtype=np.float64)
                        - np.asarray(predictions[recipient]["heads"]["regs"], dtype=np.float64))
    rows = []
    for register in range(8):
        for bit in range(64):
            coordinate = (0, register, bit)
            rows.append({"register": register, "bit": bit, "head_effect": float(head_effect[coordinate]),
                "forecast_effect": float(predicted_effect[coordinate]),
                "observed_donor_minus_recipient_P2_projection": float(donor_difference[coordinate]),
                "frozen_predicted_donor_minus_observed_donor_P2_projection": float(prediction_residual[coordinate]),
                "actual_carrier_difference_projection": float(carrier_effect[coordinate]),
                "FP32_final_add_rounding_projection": float(add_rounding[coordinate]),
                "FP32_readout_rounding": float(readout_rounding[coordinate]),
                "component_closure_residual": float(head_effect[coordinate] - donor_difference[coordinate]
                    - prediction_residual[coordinate] - carrier_effect[coordinate]
                    - add_rounding[coordinate] - readout_rounding[coordinate])})
    return {"recipient_diagonal": recipient, "actual_donor_diagonal": donor, "transferred_row": cross,
        "recipient_invariance": invariant, "matched_recipient": all(invariant.values()),
        "payload_vs_actual_donor": difference(payload, source["stages"]["layer2.P2"]),
        "actual_donor_vs_recipient": difference(source["stages"]["layer2.P2"], before["stages"]["layer2.P2"]),
        "reglogit_component_rows": rows, "RSI_bit1": rows[6 * 64 + 1],
        "full_bank_flags_and_effects": hard_changes(after["heads"], before["heads"]),
        "all_head_effects": {name: difference(after["heads"][name], before["heads"][name]) for name in sorted(HEADS)},
        "scope": "matched recipient carrier/U/z/raw roots; actual donor P2 is used only for after-release accounting, never to construct the frozen payload",
        "limits": "Natural diagonal P2 is endogenous; off-diagonal payload is analytical FP32. This is not a bit-exact fixed-payload 2x2 factorial."}


def payload_changes(row_id):
    context, arm = row_id.split(".", 1)
    changes = []
    if arm != "native":
        changes.append(("module:post.0:return", "post0_zero"))
    if arm.startswith("U_swap"):
        changes.append(("module:mixers.2.out_proj:return", context + "_U_native"))
    if arm == "U_swap_P2native":
        changes.append(("module:post.2:return", context + "_P2_native"))
    if arm == "U_swap_P2other":
        changes.append(("module:post.2:return", ("new" if context == "old" else "old") + "_P2_U"))
    return changes


def verify_replay(run, observations, proofs, payloads, summary):
    transitions = read(run["data"] / "public-replay-receipts.json")
    by_branch = {name: [] for name in observations}
    for wrapper in transitions:
        require(set(wrapper) == {"branch", "receipt"} and wrapper["branch"] in by_branch,
                "transition branch wrapper differs")
        by_branch[wrapper["branch"]].append(wrapper["receipt"])
    act_rows, fresh_acts = {}, 0
    for row_id, observation in observations.items():
        context = row_id.split(".", 1)[0]
        source_id = context + ".native" if row_id in FRESH else row_id
        source = proofs[source_id]
        trace = source["trace"]
        catalog = StructuralCatalog(trace)
        receipts = by_branch[row_id]
        require(len(receipts) == len(trace["events"]), "branch primitive count differs from complete source replay: " + row_id)
        diagnostics = {"exact": 0, "different": 0, "not_checked": 0}
        for pc, receipt in enumerate(receipts):
            public_receipt(receipt)
            require(receipt["operation"] == "mdb.discovery.native-transition"
                    and receipt["trace_fingerprint"] == trace["fingerprint"]
                    and receipt["program_id"] == catalog.program_id
                    and receipt["operator"] == trace["events"][pc]["operator"]
                    and receipt["before_pc"] == pc and receipt["after_pc"] == pc + 1
                    and receipt["mechanics_status"] == "valid", "public primitive/source join differs: " + row_id)
            exact = receipt.get("exact_outputs")
            diagnostics["exact" if exact is True else "different" if exact is False else "not_checked"] += 1
        expected = payload_changes(row_id)
        require(len(observation["acts"]) == len(expected), "Act inventory differs from declared recipe")
        checked = []
        # Prior rows were replayed from their already-intervened root recipes.
        # Their historical Acts bind the original native trace, not that recipe.
        act_source = proofs[context + ".native"]
        for wrapper, (selector, payload_name) in zip(observation["acts"], expected, strict=True):
            act = wrapper.get("receipt", wrapper)
            public_receipt(act)
            port = act_source["ports"].resolve(selector)
            require(act["operation"] == "mdb.discovery.ssa-act"
                    and act["trace_fingerprint"] == act_source["trace"]["fingerprint"]
                    and act["pc"] == port["available_pc"] and act["value_id"] == port["value_id"]
                    and act["act"]["reads"] == act["act"]["writes"] == [port["value_id"]]
                    and act["mechanics_status"] == "valid", "Act actual-return/source binding differs")
            descriptor_join(payloads[payload_name], act["after"], "Act payload differs from globally frozen bytes")
            parameters = act["act"]["parameters"]["items"]
            require(len(parameters) == 1 and parameters[0]["key"]["value"] == "value",
                    "typed replacement parameter closure differs")
            descriptor_join(payloads[payload_name], parameters[0]["value"], "typed Act parameter payload differs")
            if row_id in FRESH:
                require(wrapper["payload_name"] == payload_name and wrapper["port"] == port,
                        "fresh Act declared native port differs")
                fresh_acts += 1
            checked.append({"fingerprint": act["fingerprint"], "payload_name": payload_name,
                            "source_port": port, "historical": row_id in REUSED})
        act_rows[row_id] = {"transitions": len(receipts), "source_trace": trace["fingerprint"],
                           "source_program": catalog.program_id, "descriptor_diagnostics": diagnostics,
                           "Acts": checked}
    require(len(transitions) == summary["counts"]["replay_operations"]
            and fresh_acts == summary["counts"]["acts"], "reported new work differs from typed receipts")
    return {"new_primitive_receipts": len(transitions), "new_Acts": fresh_acts,
            "historical_Acts_preserved": sum(len(observations[name]["acts"]) for name in REUSED),
            "rows": act_rows,
            "diagnostic_scope": "Exact-output comparisons to an unedited source trace may differ after an Act; independent native/replay output equality is retained per head."}


def verify_prior(run, prior, observations):
    binding = read(run["data"] / "prior-binding.json")
    pins = run["plan"]["prior"]
    require(binding["pins"] == pins == run["request"]["prior_binding"]["pins"], "prior plan/request binding differs")
    require(binding["request_sha256"] == prior["request"]["request_sha256"]
            and binding["summary_sha256"] == sha((prior["data"] / "summary.json").read_bytes())
            and pins["job_id"] == prior["source"]["job_id"]
            and pins["archive_sha256"] == prior["source"]["archive_sha256"]
            and pins["forecast_seal"] == prior["transaction_report"]["seal_sha256"]
            and pins["event_head"] == prior["transaction_report"]["event_head"]
            and binding["archive_verified"] is True, "reused original external custody pins differ")
    for relative, expected in binding["files"].items():
        require(sha(file_path(prior["data"], relative).read_bytes()) == expected
                == sha(file_path(run["data"] / "prior", relative).read_bytes()),
                "copied prior bytes differ from their independently verified original archive: " + relative)
    node = read(prior["transaction"] / "observations/native.json")["value"]
    rows = {}
    for row_id in sorted(REUSED):
        original_id = binding["row_ids"][row_id]
        require(original_id == pins["case_id"] + "." + row_id.split(".", 1)[1] + ".0", "prior row selector differs")
        previous = prior["tree"].select(node, [original_id])
        current = observations[row_id]
        for field in ("heads", "roots", "stages"):
            require(all(np.array_equal(value, current[field][name]) for name, value in previous[field].items()),
                    "reused numeric evidence was changed: " + row_id + "/" + field)
        for field in ("before", "decoded", "acts", "trace_fingerprint", "input_identity", "input_roots"):
            require(previous[field] == current[field], "reused historical metadata/receipt changed: " + field)
        require(current["evidence_role"] == "prior" and current["original_row_id"] == original_id
                and current["prior_binding"] == pins and set(current["prior_recipe_exact"]) == HEADS,
                "reused role/recipe instrument metadata differs")
        rows[row_id] = {"original_row_id": original_id,
                       "all_original_numeric_fields_preserved": True,
                       "original_stage_count": len(previous["stages"]),
                       "extra_stage_scope": "layer2.S/h added by exact recipe replay at actual original native input-port addresses; other stages retain original observed bytes",
                       "recipe_native_exact": current["prior_recipe_exact"]}
    return {"pins": pins, "selected_files": len(binding["files"]), "rows": rows,
            "scope": "Three previously observed mixed-window first-step controls; not fresh prospective outcomes."}, binding


def verify_payloads(run, predictions, binding):
    forecast = read(run["transaction"] / "forecasts/campaign.json")
    declarations = read(run["transaction"] / "declarations.json")
    require(declarations["model_identity"] == CHECKPOINT == run["plan"]["model_identity"]
            == run["request"]["model_identity"], "full model identity differs")
    require(declarations["source_identity"] == run["request"]["source_hashes"]
            and declarations["input_identity"] == run["request"]["config"]["plan_file_sha256"],
            "transaction full source/raw-plan binding differs")
    expected = {"post0_zero", "old_U_native", "new_U_native", "old_P2_native", "new_P2_native", "old_P2_U", "new_P2_U"}
    require(set(forecast["payloads"]) == expected, "frozen payload closure differs")
    payloads, report = {}, {}
    for name, descriptor in forecast["payloads"].items():
        original, value = (run["tree"].decode(descriptor[key]) for key in ("original", "value"))
        require(value.dtype == np.float32 and value.shape == (1, 64, 48)
                and np.array_equal(np.asarray(original, dtype=np.float32), value),
                "payload is not exactly the declared one-time FP32 cast: " + name)
        if name == "post0_zero":
            require(np.count_nonzero(value) == 0, "zero post0 payload differs")
            path = None
        else:
            context, endpoint = name.split("_", 1)
            source_row = context + (".U_swap" if endpoint == "P2_U" else ".native")
            source_stage = "layer2.U" if endpoint == "U_native" else "layer2.P2"
            path = ["rows", source_row, "stages", source_stage]
            require(descriptor["provenance"]["forecast_path"] == path
                    and np.array_equal(original, predictions[source_row]["stages"][source_stage]),
                    "frozen payload does not originate at its sealed independent forecast endpoint")
        payloads[name] = value
        report[name] = {"descriptor": descriptor, "source_forecast_path": path,
                        "FP32_cast_minus_FP64_endpoint": difference(value, original)}
    # Prior payload bytes are independently sourced and interpreted after sealing.
    prior_tree = NumericalTree(run["data"] / "prior/transaction")
    for name, descriptor in binding["selected_payloads"].items():
        current_name = "post0_zero" if name == "post0_zero" else "new_" + name
        require(np.array_equal(prior_tree.decode(descriptor["value"]), payloads[current_name]),
                "source-bound prior intervention payload bytes differ")
        report[current_name]["exact_prior_payload_bytes"] = True
    weights = run["tree"].decode(declarations["actual_weight_manifest"])
    return payloads, report, weights


def effect(left, right, observations, predictions):
    actual_left, actual_right = (observations[name] for name in (left, right))
    predicted_left, predicted_right = (predictions[name] for name in (left, right))
    target = (0, 6, 1)
    actual_delta = float(actual_left["heads"]["regs"][target]) - float(actual_right["heads"]["regs"][target])
    predicted_delta = float(predicted_left["heads"]["regs"][target]) - float(predicted_right["heads"]["regs"][target])
    return {"observed_minus_reference": [left, right],
        "evidence_roles": [actual_left["evidence_role"], actual_right["evidence_role"]],
        "actual_invariance": {"before": actual_left["before"] == actual_right["before"],
            "roots": {name: bool(np.array_equal(actual_left["roots"][name], actual_right["roots"][name]))
                      for name in ("window", "regs", "flags")},
            "stages": {name: bool(np.array_equal(actual_left["stages"][name], actual_right["stages"][name]))
                       for name in ("layer2.M", "layer2.U", "z")}},
        "RSI_bit1": {"observed": float(actual_left["heads"]["regs"][target]),
            "reference": float(actual_right["heads"]["regs"][target]), "effect": actual_delta,
            "forecast_effect": predicted_delta, "effect_forecast_residual": actual_delta - predicted_delta},
        "full_bank_flags_and_effects": hard_changes(actual_left["heads"], actual_right["heads"]),
        "all_head_effects": {name: difference(actual_left["heads"][name], actual_right["heads"][name]) for name in sorted(HEADS)},
        "forecast_all_head_effects": {name: difference(predicted_left["heads"][name], predicted_right["heads"][name]) for name in sorted(HEADS)},
        "actual_stage_effects": {name: difference(actual_left["stages"][name], actual_right["stages"][name])
                                 for name in sorted(actual_left["stages"])}}


def generic_record(run, row_id, observation, proof, payloads):
    context, arm = row_id.split(".", 1)
    trace = proof["trace"]
    ports = proof["ports"]
    program_hex = run["plan"]["contexts"][context]["program_hex"]
    body = {"schema": "mdb-context-window-immutable-record-v1",
        "model_identity": CHECKPOINT, "identity_kind": "actual_native_trace_and_literal_raw_roots",
        "traced_program_id": StructuralCatalog(trace).program_id, "trace_fingerprint": trace["fingerprint"],
        "input_identity": trace["input_identity"],
        "roots": {name: tensor_descriptor(value) for name, value in observation["roots"].items()},
        "applicability": {"program_hex": program_hex, "window_hex": bytes.fromhex(program_hex)[:15].ljust(15, b"\0").hex(),
            "before": observation["before"], "execution": trace["execution"],
            "horizon": "first instruction only; no successors or commits in this assay"},
        "recipe": arm, "interventions": [{"selector": selector, "payload_name": name,
            "payload": tensor_descriptor(payloads[name])} for selector, name in payload_changes(row_id)],
        "observed_native_ports": {selector: ports.resolve(selector) for selector in
            ("module:post.0:return", "module:mixers.2.out_proj:return", "module:post.2:return")},
        "head_descriptors": {name: tensor_descriptor(value) for name, value in observation["heads"].items()},
        "decoded": observation["decoded"], "evidence_role": observation["evidence_role"],
        "source_archive_sha256": run["source"]["archive_sha256"],
        "source_request_sha256": run["request"]["request_sha256"],
        "forecast_seal_sha256": run["transaction_report"]["seal_sha256"],
        "prior_original_row_id": observation.get("original_row_id")}
    return {"record_id": "record:" + sha(canonical(body)), "row_id": row_id, "record": body}


def register_accounting(observation, prediction, proof, weights):
    """Algebra on saved arrays, with every FP32 residual kept explicitly."""
    weight = np.asarray(weights["reg_out.weight"], dtype=np.float64)
    project = lambda value: (np.asarray(value, dtype=np.float64) @ weight.T).transpose(0, 2, 1)
    skip, update, carrier, post = (stage(observation, "layer2." + name) for name in ("S", "U", "M", "P2"))
    terms = {"prior": observation["roots"]["regs"].astype(np.float64) * 8. - 4.,
             "skip": project(skip), "projection": project(update),
             "FP32_carrier_add_residual": project(carrier - skip - update), "post": project(post),
             "FP32_final_add_residual": project(np.asarray(proof["h"], dtype=np.float64) - carrier - post),
             "bias": np.broadcast_to(np.asarray(weights["reg_out.bias"], dtype=np.float64)[None, :, None], (1, 8, 64))}
    total = sum(terms.values())
    terms["FP32_readout_residual"] = observation["heads"]["regs"].astype(np.float64) - total
    forecast_total = sum(prediction["register_terms"].values())
    return {"definition": "register logit = copy prior + Wreg*S + Wreg*U + carrier-add residual + Wreg*P2 + final-add residual + bias + readout residual",
        "all_reader_terms": {name: value.tolist() for name, value in terms.items()},
        "RSI_bit1": {name: float(value[0, 6, 1]) for name, value in terms.items()},
        "actual_closure": difference(sum(terms.values()), observation["heads"]["regs"]),
        "saved_forecast_term_closure": difference(forecast_total, prediction["heads"]["regs"]),
        "scope": "Full supplied weights and saved actual module inputs/returns; algebraic accounting does not establish unit necessity."}


def verify_previous_attempt(root, run, observations, proofs):
    previous = source_run(root, "context-window-result", failed=True)
    require(previous["plan"] == run["plan"], "repaired attempt changed the sealed plan")
    old_forecast = read(previous["transaction"] / "forecasts/campaign.json")
    new_forecast = read(run["transaction"] / "forecasts/campaign.json")
    require(old_forecast["predictions"] == new_forecast["predictions"]
            and old_forecast["payloads"] == new_forecast["payloads"],
            "repaired attempt changed an original sealed forecast/FP32 payload descriptor")
    # Equal descriptors alone do not prove their addressed bytes: the public
    # verifier checked each sealed file independently in both run roots above.
    require(read(previous["transaction"] / "declarations.json")["actual_weight_manifest"]
            == read(run["transaction"] / "declarations.json")["actual_weight_manifest"],
            "repaired attempt changed the full weight declarations")
    old_hypothesis = file_path(previous["data"], "context-prefix-hypothesis.json").read_bytes()
    require(old_hypothesis == file_path(run["data"], "context-prefix-hypothesis.json").read_bytes(),
            "repaired attempt changed the structural hypothesis")
    events = read(previous["data"] / "native-events.json")
    timings = read(previous["data"] / "phase-timings.json")
    starts = [item["row"] for item in timings if item["phase"] == "native_record" and item["event"] == "start"]
    ends = [item["row"] for item in timings if item["phase"] == "native_record" and item["event"] == "end"]
    require(starts == ["old.native", "old.zero"] and ends == ["old.native"]
            and len(events) == previous["transaction_report"]["work"]["native_calls"] == 2,
            "failed-attempt native exposure ledger differs")
    trace = read(previous["data"] / "captures/old.native/trace.json")
    verify_trace(trace)
    require(trace["fingerprint"] == proofs["old.native"]["trace"]["fingerprint"],
            "repeated old.native actual trace identity changed")
    for name, encoded in trace["outputs"].items():
        descriptor_join(observations["old.native"]["heads"][name], trace["values"][encoded["$value"]],
                        "repeated old.native head differs from the preserved completed first attempt")
    receipts = read(previous["data"] / "public-replay-receipts.json")
    branch_pcs = {}
    for wrapper in receipts:
        name, receipt = wrapper["branch"], wrapper["receipt"]
        require(name in REUSED | {"old.native"}, "failed-attempt replay unexpectedly reached a later branch")
        source = proofs[name]["trace"]
        pc = branch_pcs.get(name, 0)
        public_receipt(receipt)
        require(receipt["trace_fingerprint"] == source["fingerprint"]
                and receipt["before_pc"] == pc and receipt["after_pc"] == pc + 1
                and receipt["operator"] == source["events"][pc]["operator"]
                and receipt["mechanics_status"] == "valid", "preserved failed-attempt replay lineage differs")
        branch_pcs[name] = pc + 1
    require(set(branch_pcs) == REUSED | {"old.native"}
            and all(count == len(proofs[name]["trace"]["events"]) for name, count in branch_pcs.items()),
            "failed-attempt completed recipe replay inventory differs")
    changes = {name: {"initial": previous["request"]["source_hashes"].get(name),
                      "repaired": run["request"]["source_hashes"].get(name)}
               for name in set(previous["request"]["source_hashes"]) | set(run["request"]["source_hashes"])
               if previous["request"]["source_hashes"].get(name) != run["request"]["source_hashes"].get(name)}
    return {"source": previous["source"], "failure": read(previous["data"] / "failure.json"),
        "initial_seal_preceded_all_native_exposures": True,
        "all_forecast_and_payload_descriptors_and_addressed_bytes_unchanged": True,
        "completed_native_rows": ends, "partial_native_rows": [name for name in starts if name not in ends],
        "native_root_entries": events, "completed_old_native_byte_exact_on_repeat": True,
        "original_seal_sha256": previous["transaction_report"]["seal_sha256"],
        "repaired_seal_sha256": run["transaction_report"]["seal_sha256"],
        "changed_staged_sources": changes,
        "exposure_scope": "Seven calls completed in the repaired run. old.native was already completed and old.zero partly entered during the failed instrument attempt; do not call all seven first exposures.",
        "repaired_exposure_classification": {name: "repeated completed row" if name == "old.native"
            else "previous partial instrument exposure" if name == "old.zero"
            else "first completed/native row exposure in this assay" for name in sorted(FRESH)}}


def audit(run_dir, prior_run_dir, *, previous_attempt=None):
    run = source_run(run_dir, "context-window-result")
    prior = source_run(prior_run_dir, "sequence-result")
    summary, plan = read(run["data"] / "summary.json"), run["plan"]
    require(summary["job_id"] == run["source"]["job_id"]
            and summary["request_sha256"] == run["request"]["request_sha256"], "summary external binding differs")
    require(set(plan["fresh_arms"]) == set(summary["fresh_arms"]) == FRESH
            and set(plan["prior_arms"]) == set(summary["prior_arms"]) == REUSED,
            "assay fresh/prior inventory differs")
    require(plan["schema"] == "mdb-context-window-plan-v1"
            and plan["register_words"] == [0, 0, 0, 0, 0, 0, 3, 262] and plan["flags"] == [0, 0, 0, 0],
            "declared matched initial architecture differs")
    staged_plan = file_path(run["root"] / "payload", "plan.json")
    require(read(staged_plan) == plan and sha(staged_plan.read_bytes()) == run["request"]["config"]["plan_file_sha256"],
            "raw plan bytes/content differs from request")
    source = read(run["data"] / "source.json")
    require(source["request"] == run["request"], "worker source request differs from external staged request")
    frontier_node = read(run["transaction"] / "frontiers/raw.json")
    require(run["tree"].decode(frontier_node["value"]) == plan, "globally sealed raw frontier differs from plan")
    windows = {context: bytes.fromhex(plan["contexts"][context]["program_hex"])[:15].ljust(15, b"\0")
               for context in ("old", "new")}
    changed_window = [index for index in range(15) if windows["old"][index] != windows["new"][index]]
    require(changed_window == list(range(8, 14))
            and windows["old"][:7] == windows["new"][:7] == bytes.fromhex("4881ef01000000"),
            "trailing-window contrast does not preserve literal first instruction")
    predictions = run["tree"].decode(read(run["transaction"] / "forecasts/campaign.json")["predictions"])["rows"]
    observations = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    require(set(predictions) == set(observations) == FRESH | REUSED, "raw evidence omits/adds an assay row")
    prior_proof, binding = verify_prior(run, prior, observations)
    sealed_binding = read(run["transaction"] / "frontiers/prior_binding.json")
    require(run["tree"].decode(sealed_binding["value"]) == binding, "prior reuse binding was not globally sealed")
    payloads, payload_report, weights = verify_payloads(run, predictions, binding)
    proofs = {}
    for row_id, observation in observations.items():
        context = row_id.split(".", 1)[0]
        require(observation["evidence_role"] == ("fresh" if row_id in FRESH else "prior"), "row evidence role differs")
        require(observation["before"] == {"register_words": plan["register_words"], "flags": plan["flags"], "offset": 0},
                "row before state differs from matched plan")
        relative = None if row_id in FRESH else "prior/captures/" + binding["row_ids"][row_id]
        proof = verify_observation(run, row_id, observation, predictions[row_id],
                                   plan["contexts"][context]["program_hex"], capture_relative=relative)
        require(set(observation["actual_ports"]) == set(observation["stages"]) == set(expected_stage_ports()),
                "actual module-port inventory differs")
        for name, weight in weights.items():
            port = proof["ports"].resolve("parameter:" + name)
            require(not port["absent"], "full declared parameter lacks an actual SSA root")
            descriptor_join(weight, proof["trace"]["values"][port["value_id"]], "observed model parameter differs from full forecast weights: " + name)
        proofs[row_id] = proof
    replay = verify_replay(run, observations, proofs, payloads, summary)
    native_events = read(run["data"] / "native-events.json")
    require(len(native_events) == summary["counts"]["physical_forwards"] == 7
            and summary["counts"]["prior_rows_reused"] == 3
            and run["transaction_report"]["work"]["native_calls"] == 7,
            "fresh physical counts differ from all three instruments")
    require([event["index"] for event in native_events] == list(range(1, 8))
            and all(event["kind"] == "native_root_entry" for event in native_events), "fresh root-call ledger differs")
    timings = read(run["data"] / "phase-timings.json")
    seal_end = [event for event in timings if event["phase"] == "global_seal" and event["event"] == "end"]
    native_start = [event for event in timings if event["phase"] == "native_execution" and event["event"] == "start"]
    require(len(seal_end) == len(native_start) == 1
            and seal_end[0]["physical_forwards"] == 0
            and seal_end[0]["elapsed_seconds"] <= native_start[0]["elapsed_seconds"]
            < native_events[0]["elapsed_seconds"], "fresh native call timing does not follow global seal")
    require(all(event["elapsed_seconds"] >= seal_end[0]["elapsed_seconds"] for event in timings
                if event["phase"] in {"prior_recipe_replay", "native_record", "typed_replay"}),
            "prior numerical reuse or new replay began before the new global seal")
    structural_hypothesis = None
    if "context-prefix-hypothesis.json" in run["manifest"]:
        hypothesis_raw = (run["data"] / "context-prefix-hypothesis.json").read_bytes()
        require(run["request"]["source_hashes"]["context-prefix-hypothesis.json"] == sha(hypothesis_raw),
                "structural prefix hypothesis bytes differ from request")
        hypothesis = read(run["transaction"] / "frontiers/structural_prefix_hypothesis.json")
        require(run["tree"].decode(hypothesis["value"]) == json.loads(hypothesis_raw),
                "structural prefix hypothesis was not globally sealed")
        declared = json.loads(hypothesis_raw)
        scored = {}
        for name, expected in declared["predictions"].items():
            require(name in FRESH and name.startswith("old."), "structural prefix forecast addresses an unexpected row")
            actual_value = float(observations[name]["heads"]["regs"][0, 6, 1])
            sign = int(actual_value > 0) - int(actual_value < 0)
            scored[name] = {"observed": actual_value, "predicted_sign": expected["RSI_bit1_sign"],
                "observed_sign": sign, "sign_matches": sign == expected["RSI_bit1_sign"],
                "approximate_previous_case_value": expected["RSI_bit1_logit_approx"],
                "observed_minus_approximate_previous_case_value": actual_value - expected["RSI_bit1_logit_approx"],
                "complete_raw_input_equation_forecast": float(predictions[name]["heads"]["regs"][0, 6, 1])}
        structural_hypothesis = {"declared": declared, "source_sha256": sha(hypothesis_raw), "rows": scored,
            "sign_matches": sum(row["sign_matches"] for row in scored.values()), "sign_forecasts": len(scored),
            "source_index_integrity_scope": "The request seals the historical index digest as declared; this audit verifies the new hypothesis bytes and seal, and does not reopen that separate historical index.",
            "magnitude_scope": "Historical intervention payloads were conditional-prefix forecasts. New payloads are full raw-input forecasts; no prefix-byte identity or exact intervention magnitude is presumed."}
    transfers = [recipient_transfer("old.U_swap", "new.U_swap", "old.U_swap_P2other", observations, proofs,
                                   predictions, payloads["new_P2_U"], weights["reg_out.weight"]),
                 recipient_transfer("new.U_swap", "old.U_swap", "new.U_swap_P2other", observations, proofs,
                                   predictions, payloads["old_P2_U"], weights["reg_out.weight"])]
    for transfer, donor in zip(transfers, ("new", "old"), strict=True):
        original = predictions[donor + ".U_swap"]["stages"]["layer2.P2"]
        transfer["FP64_predicted_donor_minus_observed_donor_P2"] = difference(original, observations[donor + ".U_swap"]["stages"]["layer2.P2"])
        transfer["FP32_cast_minus_FP64_predicted_donor_P2"] = difference(payloads[donor + "_P2_U"], original)
    contrasts = {
        "matched_context": [effect("new." + arm, "old." + arm, observations, predictions)
                            for arm in ("native", "zero", "U_swap", "U_swap_P2native")],
        "post0_zero": [effect(context + ".zero", context + ".native", observations, predictions) for context in ("old", "new")],
        "same_context_U_restore": [effect(context + ".U_swap", context + ".zero", observations, predictions) for context in ("old", "new")],
        "common_native_P2_repair": [effect(context + ".U_swap_P2native", context + ".U_swap", observations, predictions) for context in ("old", "new")]}
    rows, aggregate = {}, {}
    for role, names in (("fresh", FRESH), ("prior", REUSED)):
        aggregate[role] = {"rows": len(names), "all_head_decision_differences": 0,
                           "decoded_state_mismatches": 0, "head_max_abs": {name: 0. for name in HEADS},
                           "native_replay_mismatch_rows": []}
        for name in sorted(names):
            proof = proofs[name]["proof"]
            mismatch = {head: values["decision_differences"] for head, values in proof["head_comparison"].items()}
            aggregate[role]["all_head_decision_differences"] += sum(mismatch.values())
            aggregate[role]["decoded_state_mismatches"] += int(proof["forecast_decode"] != proof["actual_decode"])
            if not all(proof["native_replay_heads"].values()):
                aggregate[role]["native_replay_mismatch_rows"].append(name)
            for head, values in proof["head_comparison"].items():
                aggregate[role]["head_max_abs"][head] = max(aggregate[role]["head_max_abs"][head], values["residual"]["max_abs"])
            rows[name] = {"evidence_role": role, **proof,
                          "RSI_bit1": {"observed": float(observations[name]["heads"]["regs"][0, 6, 1]),
                                       "forecast": float(predictions[name]["heads"]["regs"][0, 6, 1])},
                          "vs_same_context_native": hard_changes(observations[name]["heads"], observations[name.split(".", 1)[0] + ".native"]["heads"]),
                          "register_accounting": register_accounting(observations[name], predictions[name], proofs[name], weights)}
    native_mismatches = [name for role in aggregate.values() for name in role["native_replay_mismatch_rows"]]
    require(set(summary["instrument_mismatch_rows"]) == set(native_mismatches) & FRESH,
            "worker fresh mismatch summary differs from retained per-head statuses")
    reattempt = (verify_previous_attempt(previous_attempt, run, observations, proofs)
                 if previous_attempt is not None else {"status": "no previous attempt supplied; no first-exposure claim made"})
    return {"schema": "mdb-context-window-publication-v1", "source": run["source"],
        "prior_source": prior["source"], "plan": plan, "raw_summary": summary,
        "verification": {"custody": "verified", "reused_evidence": prior_proof, "public_replay": replay,
                         "full_parameter_arrays": len(weights), "full_parameter_scalars": sum(value.size for value in weights.values()),
                         "frozen_payloads": payload_report, "native_root_calls": native_events},
        "aggregate": aggregate, "rows": rows, "contrasts": contrasts,
        "matched_recipient_P2_transfers": transfers,
        "sealed_structural_prefix_hypothesis": structural_hypothesis,
        "window_contrast": {"old_hex": windows["old"].hex(), "new_hex": windows["new"].hex(),
                            "changed_byte_indices": changed_window, "identical_first_instruction": True,
                            "identical_architectural_beforestate": True},
        "generic_records": [generic_record(run, name, observations[name], proofs[name], payloads) for name in sorted(observations)],
        "phase_timings": timings, "transaction_events": run["events"],
        "previous_instrument_attempt": reattempt,
        "scope": ("Performance reproduction of seven already-tested native cases and three source-bound prior controls; no first-exposure or new discovery claim."
                  if summary.get("execution_role") == "performance-reproduction" else
                  "One fixed SUB input architecture, two literal15-byte windows, seven fresh and three source-bound reused controls. All equations and full weights supplied; no operation-rule discovery or future-instruction semantic claim."),
        "mechanics_status": "valid" if not native_mismatches else "instrument-mismatch",
        "instrument_status": {"custody": "verified", "native_replay_mismatch_rows": native_mismatches},
        "trend_status": "per-row signed observations retained; no scientific acceptance threshold",
        "terminal_status": "unassessed", "new_model_calls_during_analysis": 0,
        "limits": plan["limits"] + ["Trusted Python forecast callback ordering is instrumented, not an OS-level isolation proof.",
                                     "All continuous forecast residuals and per-head native/replay statuses remain observations; no numerical science threshold was imposed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--previous-attempt", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "analysis output already exists; choose a new immutable version")
    snapshot = args.output.with_name(args.output.stem + "-source.py")
    require(not snapshot.exists(), "analysis source snapshot already exists")
    raw_source = Path(__file__).read_bytes()
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open("xb") as stream:
        stream.write(raw_source)
    try:
        result = audit(args.run_dir, args.prior_run, previous_attempt=args.previous_attempt)
        result["analysis_source"] = {"snapshot": snapshot.name, "sha256": sha(raw_source),
            "python_version": sys.version, "numpy_version": np.__version__,
            "public_metadata_modules": {module: sha(Path(sys.modules[module].__file__).read_bytes()) for module in
                ("mdb.forecast_transaction", "mdb.forecast_storage", "mdb.forward_ports", "mdb.forward_symbols", "mdb.discovery_catalog")}}
        with args.output.open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    except Exception as error:
        failure = args.output.with_name(args.output.stem + "-offline-failure.json")
        with failure.open("x") as stream:
            json.dump({"schema": "mdb-context-window-offline-audit-failure-v1", "error": f"{type(error).__name__}: {error}",
                       "analysis_source_sha256": sha(raw_source), "new_model_calls": 0}, stream, indent=2)
            stream.write("\n")
        raise
    print(json.dumps({"output": str(args.output), "output_sha256": sha(args.output.read_bytes()),
                      "aggregate": result["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
