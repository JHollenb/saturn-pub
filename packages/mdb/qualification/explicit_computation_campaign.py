"""Seal complete suffix equations before releasing any native suffix execution.

One finite CPU lease. The only pre-seal model calls are genuine aborted native
prefixes; all intervention bytes and all mathematical predictions are persisted
before full recording, native suffix observation, or suffix replay is permitted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import re
import runpy
import time
from pathlib import Path

CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"
ARMS = ("native", "zero", "U_swap", "S_swap", "both", "sham",
        "U_swap_P2native", "S_swap_P2native")
HEAD_MODULES = ("reg_out", "flag_query", "flags", "width_head", "length", "syscall",
                "imm_out", "store_flag", "store_out")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def support():
    path = Path(__file__).with_name("campaign_support.py")
    if not path.exists():
        path = Path(__file__).with_name("discovery_upgrades.py")
    return runpy.run_path(str(path))


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("explicit computation plan must be an object")
    unsigned = dict(plan)
    claimed = unsigned.pop("plan_sha256", None)
    if not isinstance(claimed, str) or identity(unsigned) != claimed:
        raise ValueError("plan self seal differs")
    if (plan.get("schema") != "mdb-explicit-computation-plan-v1"
            or plan.get("model_identity") != CHECKPOINT or plan.get("backend") != "cpu-fp32"
            or plan.get("cpu_threads") != 1 or tuple(plan.get("arms", ())) != ARMS):
        raise ValueError("plan model, execution contract or arm order differs")
    cases = plan.get("cases")
    if not isinstance(cases, list) or len(cases) != 6:
        raise ValueError("this finite campaign requires six declared cases")
    owners = set()
    for case in cases:
        name = case.get("case_id")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_-]{1,64}", name) or name in owners:
            raise ValueError("case IDs must be safe and unique")
        owners.add(name)
        code = case.get("program_hex")
        if not isinstance(code, str) or not code or len(code) % 2 or any(c not in "0123456789abcdef" for c in code):
            raise ValueError("program must be canonical whole hexadecimal bytes")
        if not isinstance(case.get("register_words"), list) or len(case["register_words"]) != 8:
            raise ValueError("eight raw architectural register words required")
        if not isinstance(case.get("flags"), list) or len(case["flags"]) != 4 or any(
                type(bit) is not int or bit not in (0, 1) for bit in case["flags"]):
            raise ValueError("four raw flag bits required")
    return plan


class Arrays:
    """Lossless content-addressed NPY bytes; deduplication is not a model approximation."""

    def __init__(self, output):
        self.output = output
        (output / "arrays").mkdir(exist_ok=True)

    def put(self, value):
        import numpy as np

        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.ascontiguousarray(value)
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError("array evidence must be finite numeric data")
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        raw = buffer.getvalue()
        relative = "arrays/" + sha(raw) + ".npy"
        path = self.output / relative
        if not path.exists():
            path.write_bytes(raw)
        elif path.read_bytes() != raw:
            raise ValueError("array content identity collision")
        return {"path": relative, "sha256": sha(raw), "shape": list(value.shape), "dtype": str(value.dtype)}

    def get(self, reference):
        import numpy as np

        raw = (self.output / reference["path"]).read_bytes()
        if sha(raw) != reference["sha256"]:
            raise ValueError("sealed array bytes changed")
        value = np.load(io.BytesIO(raw), allow_pickle=False)
        if list(value.shape) != reference["shape"] or str(value.dtype) != reference["dtype"]:
            raise ValueError("array geometry differs from its seal")
        return value

    def mapping(self, values):
        return {name: self.put(value) for name, value in values.items()}


def worker():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    import torch
    from isa_lab.native_forward_symbol_assay import load_r3_mamba_model

    from mdb.discovery_cli import save_capture
    from mdb.discovery_isa import ISADriver
    from mdb.discovery_runtime import ReplayForward
    from mdb.forward_prefix import record_prefix
    from mdb.forward_recording import record_forward, tensor_descriptor

    formulas = runpy.run_path("explicit_suffix_equations.py")
    predict = formulas["predict_suffix"]
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    plan = validate_plan(json.loads(Path("plan.json").read_text()))
    if sha(Path("plan.json").read_bytes()) != config["plan_file_sha256"]:
        raise ValueError("plan file seal differs")
    for name, expected in request["source_hashes"].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError("staged source changed: " + name)
    distribution = importlib.metadata.distribution("isa-model-comparison")
    for name, expected in request["source_hashes"].items():
        if name.endswith(".py") and name.startswith(("isa_lab/", "direct_effect/")):
            if sha(distribution.locate_file(name).read_bytes()) != expected:
                raise ValueError("installed native source changed: " + name)
    formula_manifest = formulas["formula_manifest"]()
    for name, expected in formula_manifest["source_formulas"].items():
        if request["source_hashes"].get(name) != expected:
            raise ValueError("explicit equation/native source identity differs: " + name)
    torch.set_num_threads(1)
    model, model_sources = load_r3_mamba_model(config["checkpoint"])
    model.eval().requires_grad_(False)
    output = Path("explicit-result")
    output.mkdir(exist_ok=False)
    arrays = Arrays(output)
    started = time.monotonic()
    timeline, phase, released = [], "prefix-only", False
    counters = {"physical_attempts": 0, "completed_full_forwards": 0,
                "mixers.2": 0, "post.2": 0,
                "suffix_heads": dict.fromkeys(HEAD_MODULES, 0),
                "prefix_auxiliary_heads": {"dest": 0, "src": 0}}
    ledger = {"prefix_native_attempts": 0, "full_native_captures": 0, "native_branch_audits": 0,
              "own_state_continuations": 0, "replay_native_operations": 0, "acts": 0}

    def event(kind, **fields):
        row = {"sequence": len(timeline), "kind": kind, "phase": phase,
               "elapsed_seconds": time.monotonic() - started, **fields}
        timeline.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        with (output / "timeline.txt").open("a") as stream:
            stream.write(f"{row['sequence']:04d} {phase} {kind} {json.dumps(fields, sort_keys=True)}\n")
        return row

    def attempt(module, args):
        if counters["physical_attempts"] >= config["max_physical_forwards"]:
            raise ValueError("finite physical-forward budget exhausted")
        counters["physical_attempts"] += 1
        event("native_forward_attempt", physical_attempt=counters["physical_attempts"])

    def completed(module, args, value):
        counters["completed_full_forwards"] += 1
        event("native_full_forward_completed")

    def entry(name):
        def observe(module, args):
            if name in HEAD_MODULES or name in ("mixers.2", "post.2"):
                if not released:
                    raise ValueError("native suffix was not released: " + name)
                if name in HEAD_MODULES:
                    counters["suffix_heads"][name] += 1
                else:
                    counters[name] += 1
            else:
                counters["prefix_auxiliary_heads"][name] += 1
            event("native_module_enter", module=name)
        return observe

    hooks = [model.register_forward_pre_hook(attempt), model.register_forward_hook(completed)]
    for name in ("mixers.2", "post.2", *HEAD_MODULES, "dest", "src"):
        hooks.append(model.get_submodule(name).register_forward_pre_hook(entry(name)))
    weights_original = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    weights_identity = {name: tensor_descriptor(value)["sha256"] for name, value in model.state_dict().items()}
    save(output / "weights.json", {"parameter_arrays": formulas["weight_manifest"](weights_original),
                                  "checkpoint_sha256": CHECKPOINT,
                                  "complete_parameter_dictionary": True,
                                  "model_sources": model_sources})
    save(output / "formula.json", formula_manifest)
    save(output / "plan.json", plan)
    # Persist complete uncompressed parameter arrays once; every native and
    # forecast row binds these exact bytes, with no fitted low-rank substitute.
    save(output / "parameter-arrays.json", arrays.mapping(weights_original))
    weights = formulas["prepare_weights"](weights_original)
    prefix_rows, forecasts, errors = [], [], []

    def advance(runtime, pc, receipts):
        while runtime.pc < pc:
            if ledger["replay_native_operations"] >= config["max_replay_operations"]:
                raise ValueError("finite replay primitive budget exhausted")
            receipts.append(runtime.advance_one())
            ledger["replay_native_operations"] += 1

    def fp32(value):
        result = np.asarray(value, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite intervention payload")
        return np.ascontiguousarray(result)

    def decode(heads, driver):
        converted = {key: torch.from_numpy(np.asarray(value).copy()) for key, value in heads.items()}
        return {key: value for key, value in driver.decode(converted).items() if key != "raw_logits"}

    for case in plan["cases"]:
        case_id = case["case_id"]
        driver = ISADriver(case["program_hex"], case["register_words"], case["flags"], max_instructions=2)
        roots = driver.inputs()
        root_arrays = arrays.mapping({"window_bits": roots[0], "reg_bits": roots[1], "flags": roots[2]})
        capture = record_prefix(model, roots, stop_before="mixers.2", observe=("decoder",),
            model_identity=CHECKPOINT, execution={"backend": "cpu", "dtype": "float32", "threads": 1},
            input_identity=identity(root_arrays), metadata={"case_id": case_id}, max_operations=8192)
        ledger["prefix_native_attempts"] += 1
        recipe_path = output / "prefix-recipes" / case_id
        save_capture(capture, recipe_path)
        runtime = ReplayForward(capture)
        source_law = runtime.catalog.resolve("module:native.post.0")
        source_port = runtime.catalog.resolve("module:native.post.0:output")["value_id"]
        receipts = []
        advance(runtime, source_law["exit"], receipts)
        source_cut = runtime.capture()
        zero_payload = torch.zeros_like(runtime.read(source_port))
        native = runtime.fork(source_cut)
        edited = runtime.fork(source_cut)
        act = edited.apply(source_port, zero_payload)
        ledger["acts"] += 1
        native_receipts, edited_receipts = [], []
        advance(native, len(native.trace["events"]), native_receipts)
        advance(edited, len(edited.trace["events"]), edited_receipts)
        n, e = native.outputs(), edited.outputs()
        Sn = n["boundary"]["args"][0].detach().numpy().copy()
        Se = e["boundary"]["args"][0].detach().numpy().copy()
        z = n["observed"]["decoder"][0].detach().numpy().copy()
        if not torch.equal(n["observed"]["decoder"][0], e["observed"]["decoder"][0]):
            raise ValueError("source removal changed an already computed decoder root")
        row = {"case_id": case_id, "scope": case["scope"], "roots": root_arrays,
               "Sn": arrays.put(Sn), "Se": arrays.put(Se), "z": arrays.put(z),
               "source_zero": arrays.put(zero_payload), "trace_fingerprint": capture.trace["fingerprint"],
               "prefix_contract": capture.trace["metadata"].get("prefix_contract"),
               "source_law": source_law, "source_port": source_port,
               "source_cut_manifest": source_cut.manifest(), "source_act": act,
               "root_recipe": str(recipe_path.relative_to(output)),
               "receipt_file": "prefix-receipts/" + case_id + ".json"}
        save(output / row["receipt_file"], {"source_prefix": receipts, "native_suffix_of_prefix": native_receipts,
                                          "edited_suffix_of_prefix": edited_receipts, "act": act,
                                          "native_join_cut": native.capture().manifest(),
                                          "edited_join_cut": edited.capture().manifest()})
        prefix_rows.append(row)
        reg, flags = roots[1].numpy(), roots[2].numpy()
        predicted_n = predict(Sn, z, reg, flags, weights)
        predicted_e = predict(Se, z, reg, flags, weights)
        payloads = {"U_native": arrays.put(fp32(predicted_n["stages"]["U"])),
                    "U_edited": arrays.put(fp32(predicted_e["stages"]["U"])),
                    "P2_native": arrays.put(fp32(predicted_n["stages"]["P2"])),
                    "M_skip_native": arrays.put(fp32(Sn + fp32(predicted_e["stages"]["U"]))),
                    "M_both_native": arrays.put(fp32(Sn + fp32(predicted_n["stages"]["U"])))}
        for arm in ARMS:
            options = {}
            if arm in ("U_swap", "U_swap_P2native"):
                options["projection_override"] = arrays.get(payloads["U_native"])
            if arm in ("S_swap", "S_swap_P2native"):
                options["carrier_override"] = arrays.get(payloads["M_skip_native"])
            if arm == "both":
                options["carrier_override"] = arrays.get(payloads["M_both_native"])
            if arm.endswith("_P2native"):
                options["post_override"] = arrays.get(payloads["P2_native"])
            forecast = (predicted_n if arm == "native" else predicted_e if arm in ("zero", "sham")
                        else predict(Se, z, reg, flags, weights, **options))
            forecasts.append({"case_id": case_id, "arm": arm, "scope": case["scope"],
                "heads": arrays.mapping(forecast["heads"]), "stages": arrays.mapping(forecast["stages"]),
                "register_terms": arrays.mapping(forecast["register_terms"]),
                "decoded": decode(forecast["heads"], driver), "payloads": payloads,
                "intervention_recipe": plan["recipes"][arm], "arithmetic": "NumPy FP64; overrides exact frozen FP32 bytes",
                "future_native_values_used": False})
        event("case_forecasts_complete", case_id=case_id, arms=len(ARMS))
        del runtime, native, edited, capture, predicted_n, predicted_e, forecast
    event("prefixes_complete", cases=len(prefix_rows))
    save(output / "prefixes.json", {"rows": prefix_rows})
    save(output / "forecasts.json", {"rows": forecasts})
    pre_release = copy.deepcopy(counters)
    if (pre_release["completed_full_forwards"] or pre_release["mixers.2"] or pre_release["post.2"]
            or any(pre_release["suffix_heads"].values())):
        raise ValueError("forecast exclusion instrument observed a native future before seal")
    seal_files = {str(path.relative_to(output)): sha(path.read_bytes()) for path in output.rglob("*")
                  if path.is_file() and path.name not in ("timeline.jsonl", "timeline.txt")}
    seal = {"schema": "mdb-explicit-forecast-seal-v1", "plan_sha256": plan["plan_sha256"],
            "pre_release_counters": pre_release, "files": seal_files,
            "forecast_rows": len(forecasts), "prefix_rows": len(prefix_rows),
            "source_sha256": request["source_hashes"], "first_release_sequence": len(timeline) + 1}
    save(output / "forecast-seal.json", seal)
    seal_sha = sha((output / "forecast-seal.json").read_bytes())
    event("forecast_sealed", forecast_seal_sha256=seal_sha, pre_release_counters=pre_release)
    for name, expected in seal_files.items():
        if sha((output / name).read_bytes()) != expected:
            raise ValueError("forecast evidence changed before release: " + name)
    phase, released = "native-suffix-released", True
    release_event = event("native_suffix_released", forecast_seal_sha256=seal_sha)
    observations, joins = [], []

    for case, prefix in zip(plan["cases"], prefix_rows, strict=True):
        case_id = case["case_id"]
        driver = ISADriver(case["program_hex"], case["register_words"], case["flags"], max_instructions=2)
        roots = driver.inputs()
        capture = record_forward(model, roots, model_identity=CHECKPOINT,
            execution={"backend": "cpu", "dtype": "float32", "threads": 1},
            input_identity=identity(prefix["roots"]), metadata={"case_id": case_id}, max_operations=8192)
        ledger["full_native_captures"] += 1
        save(output / "native-traces" / (case_id + ".json"), capture.trace)
        base = ReplayForward(capture)
        source = base.catalog.resolve("module:post.0")
        source_port = base.catalog.resolve("module:post.0:output")["value_id"]
        p2 = base.catalog.resolve("module:post.2")
        projection = base.catalog.resolve("module:mixers.2.out_proj")
        U_port = base.catalog.resolve("module:mixers.2.out_proj:output")["value_id"]
        M_port = base.catalog.resolve("module:post.2:input")["value_id"]
        P_port = base.catalog.resolve("module:post.2:output")["value_id"]
        h_port = base.catalog.resolve("module:reg_out:input")["value_id"]
        z_port = base.catalog.resolve("module:decoder:output")["value_id"]
        mixer2_entry = min(index for index, event_row in enumerate(base.trace["events"])
                           if event_row["module_path"].startswith("mixers.2"))
        base_receipts = []
        advance(base, source["exit"], base_receipts)
        parent = base.capture()
        save(output / "native-prefix-receipts" / (case_id + ".json"), base_receipts)
        for arm in ARMS:
            forecast_index = next(i for i, row in enumerate(forecasts)
                                  if row["case_id"] == case_id and row["arm"] == arm)
            forecast = forecasts[forecast_index]
            payloads = forecast["payloads"]
            runtime = base.fork(parent)
            receipts, acts, actual_stages = [], [], {}
            row = {"case_id": case_id, "arm": arm, "scope": case["scope"],
                   "forecast_row_index": forecast_index, "forecast_seal_sha256": seal_sha,
                   "intervention_recipe": plan["recipes"][arm]}
            try:
                if arm != "native":
                    acts.append(runtime.apply(source_port, torch.from_numpy(arrays.get(prefix["source_zero"]))))
                advance(runtime, mixer2_entry, receipts)
                # The actual native/replay join can only be read after release.
                # It is compared against pre-seal joins, never fed to forecasts.
                next_event = runtime.trace["events"][mixer2_entry]
                S_port = next_event["inputs"][0]
                # First mixer primitive may consume norm parameters first;
                # resolve its activation against the known supplied shape.
                candidates = [key for key in next_event["inputs"]
                              if runtime.trace["values"][key]["shape"] == prefix["Sn"]["shape"]]
                if len(candidates) != 1:
                    raise ValueError("mixer2 entry activation port is ambiguous")
                S_port = candidates[0]
                actual_stages["S"] = runtime.read(S_port)
                actual_stages["z"] = runtime.read(z_port)
                join_expected = arrays.get(prefix["Sn"] if arm == "native" else prefix["Se"])
                join_exact = np.array_equal(actual_stages["S"].numpy(), join_expected)
                joins.append({"case_id": case_id, "arm": arm, "exact_S": join_exact,
                              "exact_z": np.array_equal(actual_stages["z"].numpy(), arrays.get(prefix["z"])),
                              "S_port": S_port, "z_port": z_port, "pc": runtime.pc})
                if not joins[-1]["exact_S"] or not joins[-1]["exact_z"]:
                    errors.append({"case_id": case_id, "arm": arm, "kind": "prefix-join-mismatch",
                                   "observation_retained": True})
                advance(runtime, projection["exit"], receipts)
                if arm in ("U_swap", "U_swap_P2native"):
                    acts.append(runtime.apply(U_port, torch.from_numpy(arrays.get(payloads["U_native"]))))
                elif arm == "sham":
                    acts.append(runtime.apply(U_port, runtime.read(U_port).clone()))
                actual_stages["U"] = runtime.read(U_port)
                advance(runtime, p2["entry"], receipts)
                if arm in ("S_swap", "S_swap_P2native", "both"):
                    key = "M_both_native" if arm == "both" else "M_skip_native"
                    acts.append(runtime.apply(M_port, torch.from_numpy(arrays.get(payloads[key]))))
                actual_stages["M"] = runtime.read(M_port)
                advance(runtime, p2["exit"], receipts)
                if arm.endswith("_P2native"):
                    acts.append(runtime.apply(P_port, torch.from_numpy(arrays.get(payloads["P2_native"]))))
                actual_stages["P2"] = runtime.read(P_port)
                advance(runtime, len(runtime.trace["events"]), receipts)
                actual_stages["h"] = runtime.read(h_port)
                actual_heads = runtime.outputs()
                row.update({"heads": arrays.mapping(actual_heads), "stages": arrays.mapping(actual_stages),
                            "decoded": decode(actual_heads, driver), "status": "observed"})
                audit_stages, audit_hooks = {}, []

                def replace_output(payload):
                    def replace(module, args, result):
                        return torch.from_numpy(arrays.get(payload).copy())
                    return replace

                if arm != "native":
                    audit_hooks.append(model.post[0].register_forward_hook(replace_output(prefix["source_zero"])))
                if arm in ("U_swap", "U_swap_P2native"):
                    audit_hooks.append(model.mixers[2].out_proj.register_forward_hook(replace_output(payloads["U_native"])))
                if arm == "sham":
                    audit_hooks.append(model.mixers[2].out_proj.register_forward_hook(lambda module, args, result: result.clone()))
                if arm in ("S_swap", "S_swap_P2native", "both"):
                    key = "M_both_native" if arm == "both" else "M_skip_native"
                    audit_hooks.append(model.mixers[2].register_forward_hook(replace_output(payloads[key])))
                if arm.endswith("_P2native"):
                    audit_hooks.append(model.post[2].register_forward_hook(replace_output(payloads["P2_native"])))

                def observe_output(name):
                    def observe(module, args, result):
                        audit_stages[name] = result.detach().clone()
                    return observe

                audit_hooks.extend([
                    model.decoder.register_forward_hook(observe_output("z")),
                    model.mixers[2].register_forward_pre_hook(lambda module, args: audit_stages.update(S=args[0].detach().clone())),
                    model.mixers[2].out_proj.register_forward_hook(observe_output("U")),
                    model.mixers[2].register_forward_hook(observe_output("M")),
                    model.post[2].register_forward_hook(observe_output("P2")),
                    model.reg_out.register_forward_pre_hook(lambda module, args: audit_stages.update(h=args[0].detach().clone()))])
                try:
                    with torch.no_grad():
                        audited = model(*roots)
                    ledger["native_branch_audits"] += 1
                finally:
                    for hook in audit_hooks:
                        hook.remove()
                exact_heads = {key: torch.equal(actual_heads[key], value) for key, value in audited.items()}
                exact_stages = {key: torch.equal(actual_stages[key], value) for key, value in audit_stages.items()}
                row["audit"] = {"heads": arrays.mapping(audited), "stages": arrays.mapping(audit_stages),
                                "exact_heads": exact_heads, "exact_stages": exact_stages}
                if arm == "native":
                    row["audit"]["capture_heads_exact"] = {
                        key: torch.equal(value, capture.output[key]) for key, value in actual_heads.items()}
                    row["capture_control_status"] = ("verified" if all(
                        row["audit"]["capture_heads_exact"].values()) else "differs")
                    if row["capture_control_status"] == "differs":
                        errors.append({"case_id": case_id, "arm": arm, "kind": "native-capture-control-mismatch",
                                       "observation_retained": True})
                else:
                    row["capture_control_status"] = "not-applicable"
                row["mechanics_status"] = ("valid" if all(exact_heads.values()) and all(exact_stages.values())
                                           and joins[-1]["exact_S"] and joins[-1]["exact_z"]
                                           and row["capture_control_status"] != "differs" else "mismatch")
                if row["mechanics_status"] == "mismatch":
                    errors.append({"case_id": case_id, "arm": arm, "kind": "native-audit-mismatch",
                                   "observation_retained": True})
            except Exception as error:
                row.update(status="errored", error=f"{type(error).__name__}: {error}")
                errors.append({"case_id": case_id, "arm": arm, "error": row["error"]})
            ledger["acts"] += len(acts)
            receipt_path = "branch-receipts/" + case_id + "-" + arm + ".json"
            save(output / receipt_path, {"receipts": receipts, "acts": acts,
                                        "trace_fingerprint": runtime.trace["fingerprint"],
                                        "final_cut": runtime.capture().manifest(),
                                        "source_cut": parent.manifest()})
            row["replay"] = {"receipt_file": receipt_path, "act_count": len(acts),
                             "native_operations": len(receipts), "trace_fingerprint": runtime.trace["fingerprint"]}
            observations.append(row)
            event("primary_row_retained", case_id=case_id, arm=arm, status=row.get("status"))
        del capture, base, runtime
    # Publish all primary rows before spending time on optional own-state
    # futures. A failed or skipped continuation cannot erase its primary row.
    save(output / "primary-observations.json", {"rows": observations})
    event("all_primary_rows_retained", rows=len(observations))
    case_map = {case["case_id"]: case for case in plan["cases"]}
    for row in observations:
        continuation = {"forecast_claim": "none; conditional future consumes literal predicted architectural state"}
        row["continuation"] = continuation
        if row.get("status") != "observed":
            continuation["status"] = "not-applicable-primary-error"
            continue
        if time.monotonic() - started >= config["continuation_deadline_seconds"]:
            continuation.update(status="not-run-time-budget", primary_retained=True)
            continue
        try:
            case = case_map[row["case_id"]]
            successor = ISADriver(case["program_hex"], case["register_words"], case["flags"], max_instructions=2)
            heads = {key: torch.from_numpy(arrays.get(ref)) for key, ref in row["heads"].items()}
            commit = successor.commit(heads)
            continuation["commit"] = {key: value for key, value in commit.items() if key != "raw_logits"}
            if commit["committed"] and not successor.termination:
                next_roots = successor.inputs()
                continuation["inputs"] = arrays.mapping(dict(zip(("window_bits", "reg_bits", "flags"), next_roots, strict=True)))
                with torch.no_grad():
                    next_heads = model(*next_roots)
                ledger["own_state_continuations"] += 1
                continuation.update(status="observed", heads=arrays.mapping(next_heads), decoded=decode(next_heads, successor))
            else:
                continuation["status"] = "effect-blocked-or-terminated"
        except Exception as error:
            continuation.update(status="errored", error=f"{type(error).__name__}: {error}")
    save(output / "observations.json", {"rows": observations})
    save(output / "prefix-join-checks.json", {"rows": joins})
    after_weights = {name: tensor_descriptor(value)["sha256"] for name, value in model.state_dict().items()}
    if after_weights != weights_identity:
        raise ValueError("shared frozen weights changed")
    for hook in hooks:
        hook.remove()
    save(output / "execution-ledger.json", {**ledger, "counters": counters,
        "pre_release_counters": pre_release, "release_event": release_event,
        "one_outer_lease": True, "additional_scheduler_submissions": 0})
    report = {"schema": "mdb-explicit-computation-campaign-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "plan_sha256": plan["plan_sha256"],
        "checkpoint_sha256": CHECKPOINT, "forecast_seal_sha256": seal_sha,
        "pre_release_counters": pre_release, "release_event": release_event,
        "mechanics_status": "valid" if not errors else "partial-mechanics-errors",
        "instrument_status": "prefix-exclusion-and-native-comparisons-retained",
        "trend_status": "complete-weight-explicit-equation-observations", "terminal_status": "unestablished",
        "case_count": len(prefix_rows), "forecast_rows": len(forecasts), "primary_rows": len(observations),
        "backend": "cpu-fp32", "forecast_arithmetic": "NumPy FP64 sequential scan",
        "parameter_compression": False, "physical_batching": False,
        "physical_model_forwards": counters["physical_attempts"], "completed_full_forwards": counters["completed_full_forwards"],
        "physical_forward_limit": config["max_physical_forwards"], "work": ledger, "errors": errors,
        "prefix_join_checks": joins, "model_weights_unchanged": True,
        "worker_seconds": time.monotonic() - started, "oracle_state_used": False,
        "scope": "conditional suffix account from observed prefix; dest/src are prefix-computed auxiliary heads",
        "file_manifest": {str(path.relative_to(output)): {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
                          for path in output.rglob("*") if path.is_file()}}
    save(output / "report.json", report)
    support()["emit"](output, report)


def stage(run_dir, isa_root, plan_path):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    raw_plan = Path(plan_path).read_bytes()
    plan = validate_plan(json.loads(raw_plan))
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plan.json").write_bytes(raw_plan)
    upstream = runpy.run_path(str(isa_root / "scripts/native_forward_symbols.py"))
    upstream["_worker_environment_defaults"]()
    sources, origins, _ = upstream["source_inventory"]()
    wheel = runpy.run_path(str(isa_root / "scripts/symbol_sources.py"))["build_isa_wheel"](run_dir, sources)
    files = [(path, name) for name, path in sources.items()]
    files += [(wheel, wheel.name), (Path(__file__).with_name("discovery_upgrades.py"), "campaign_support.py"),
              (Path(__file__).with_name("explicit_suffix_equations.py"), "explicit_suffix_equations.py"),
              (run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "path_token_allowlist": ["explicit-result"],
                    "hashed_roots": ["artifacts", "direct_effect", "isa_lab", wheel.name,
                        "explicit_computation_campaign.py", "explicit_suffix_equations.py", "campaign_support.py",
                        "plan.json", "source-manifest.json", "preflight.v1.json"]})
    files.append((preflight, preflight.name))
    config = {"schema": "mdb-explicit-computation-config-v1", "checkpoint": "artifacts/model.pt",
              "plan_file_sha256": sha(raw_plan), "plan_sha256": plan["plan_sha256"],
              "max_physical_forwards": 128, "max_replay_operations": 65536,
              "continuation_deadline_seconds": 210, "backend": "cpu-fp32", "cpu_threads": 1}
    request = {"schema": "mdb-explicit-computation-request-v1", "task_family": "mdb-explicit-computation",
               "model_identity": CHECKPOINT, "needs": {"cpu": True}, "config": config,
               "plan_sha256": plan["plan_sha256"], "execution_environment": intent(),
               "scheduling_contract": one_lease_scheduling_contract(logical_operations=128, logical_branches=48,
                   logical_checkpoint_replays=96, max_local_work_items=128,
                   extra={"max_replay_native_operations": 65536, "max_physical_model_forwards": 128}),
               "selection_policy": "all six declared cases and eight arms; predictions globally sealed before native suffix release",
               "source_hashes": {name: sha(path.read_bytes()) for path, name in files}, "source_origins": origins}
    request["source_hashes"]["explicit_computation_campaign.py"] = sha(Path(__file__).read_bytes())
    request["request_sha256"] = identity(request)
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=Path(__file__), config=config, request=request, additional_files=files)
    command = worker_command("explicit_computation_campaign.py") + ["--worker"]
    index = command.index("python", command.index("uv") + 1)
    command[index:index] = ["--with", "./" + wheel.name]
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, experiment="mdb-explicit-computation",
        task="r3-prefix-sealed-full-weight-suffix-equations", model=None, payload=run_dir / "payload",
        prefer_host="beast", gpu=False, needs=request["needs"], ram_mb=2304, cpu_threads=1,
        timeout_s=300, retry_on_kill=False, preflight=True,
        note="One finite CPU1 R3 full-weight suffix-equation campaign:6 truly aborted prefixes;all48 forecasts/frozenFP32Acts globally sealed withzero suffixentry;then6captures/48nativeaudits/optional48ownstatefutures <=128physical,65k native replayops.2304MiB measured parent exception; no compression/oracle/training/retries. Stopafterallprimaryrows/archive.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir), "plan_sha256": plan["plan_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--isa-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.worker:
        try:
            worker()
        except Exception as error:
            output = Path("explicit-result")
            if output.is_dir():
                request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
                report = {"schema": "mdb-explicit-computation-campaign-v1", "mechanics_status": "errored",
                          "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
                          "error": f"{type(error).__name__}: {error}"}
                save(output / "failure.json", report)
                support()["emit"](output, report)
            raise
    elif args.collect:
        print(json.dumps(support()["collect"](args.run_dir, args.collect,
                                             max_expanded_bytes=512 * 1024 * 1024), sort_keys=True))
    else:
        if args.plan is None or args.run_dir is None or args.isa_root is None:
            parser.error("staging requires --plan, --run-dir and --isa-root")
        print(json.dumps(stage(args.run_dir.resolve(), args.isa_root.resolve(), args.plan.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
