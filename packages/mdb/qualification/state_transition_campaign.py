"""Sealed own-state instruction futures, context-path cuts and matched MLP contrasts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import time
from pathlib import Path

HEADS = ("regs", "flags", "dest", "src", "width", "length", "syscall", "imm", "store", "store_addr", "store_value")
CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def prediction_at(predictions, path):
    try:
        for key in path:
            predictions = predictions[key]
        return predictions
    except KeyError:
        return None


def array_identities(value, path=()):
    result = {}
    if isinstance(value, dict):
        if value.get("kind") == "array":
            result[json.dumps(path)] = {key: value[key] for key in ("sha256", "content_sha256", "dtype", "shape")}
        else:
            for key, child in value.items():
                result.update(array_identities(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(array_identities(child, (*path, index)))
    return result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def support():
    source = Path(__file__).with_name("campaign_support.py")
    return runpy.run_path(str(source if source.exists() else source.with_name("discovery_upgrades.py")))


def recipe(selector, payload, *, direct=False):
    return {"selector": selector, "payload_name": payload,
            "patch_kind": "direct-window-columns" if direct else "replace",
            **({"path": ["args", 0]} if direct else {})}


def patch_direct_columns(value, payload):
    """Replace exactly the immediate2/slots8/pick1 columns, preserving all others."""
    import numpy as np
    import torch

    frozen = torch.from_numpy(np.array(payload, copy=True))
    if tuple(value.shape) != (1, 64, 293) or tuple(frozen.shape) != (1, 64, 11):
        raise ValueError("direct feature patch geometry differs")
    return torch.cat((value[:, :, :10], frozen[:, :, :2], value[:, :, 12:284], frozen[:, :, 2:]), dim=-1)


def plan():
    zero = recipe("module:post.0:return", "zero")
    projection = recipe("module:mixers.2.out_proj:return", "sequence.U_native")
    recipes = {"sequence.native.0": [], "sequence.U_swap.0": [zero, projection],
               "sequence.U_swap_P2native.0": [zero, projection, recipe("module:post.2:return", "sequence.P2_native")]}
    for z, values in (("new", "old"), ("old", "new"), ("new", "new")):
        key = f"path.{z}.{values}"
        edits = ([recipe("module:decoder:return", "path.z_new")] if z == "new" else [])
        if values == "new":
            edits += [recipe("module:inp:input", key + ".direct", direct=True)]
        recipes[key] = edits + [zero, projection]
    for key in ("path.carrier_other", "path.carrier_other_P2own"):
        recipes[key] = [zero, projection, recipe("module:mixers.2:return", "path.M_other")]
        if key.endswith("P2own"):
            recipes[key] += [recipe("module:post.2:return", "path.P2_own")]
    for op in ("or", "and"):
        for arm in ("native", "zero_post0", "zero_post1", "zero_post2"):
            recipes[f"mlp.{op}.{arm}"] = [] if arm == "native" else [recipe(f"module:post.{arm[-1]}:return", "zero")]
    return {"schema": "mdb-state-transition-plan-v1", "model_identity": CHECKPOINT,
        "execution": {"backend": "cpu", "dtype": "float32", "threads": 1},
        "register_words": [0, 0, 0, 0, 0, 0, 3, 262], "flags": [0, 0, 0, 0],
        "arms": ["native", "U_swap", "U_swap_P2native"],
        "segments": [{"id": "repeated_sub", "program_hex": "4881ef010000004881ef01000000", "max_instructions": 2},
                     {"id": "xor", "program_hex": "4831f7", "max_instructions": 1},
                     {"id": "or", "program_hex": "4809f7", "max_instructions": 1},
                     {"id": "and", "program_hex": "4821f7", "max_instructions": 1}],
        "contexts": {"old": "4881ef010000004881ef01000000", "new": "4881ef010000004831f74881c701000000"},
        "mlp_programs": {"or": "4809f7", "and": "4821f7"}, "row_recipes": recipes,
        "panel_reuses": {"path.old.old": "sequence.U_swap.0"},
        "budgets": {"max_physical_forwards": 40, "max_replay_operations": 40000, "timeout_seconds": 300, "retries": 0},
        "forecast_policy": "all arrays, payloads, full own-state futures globally sealed before any native call",
        "segment_policy": "host explicitly supplies each segment; own predicted registers/flags cross boundaries; PC resets to0; no hidden-state persistence",
        "path_policy": "all old roots, post0 zero, one fixed old native U; decoder source x direct byte-value source; same-z direct feature construction",
        "limits": ["Supplied full source equations and weights; no automatic semantic operation discovery or complete ISA claim.",
                   "Segmented inputs are not one contiguous fetched program; only first two SUBs are contiguous.",
                   "Known first-step contrast is replication; new own-state successors and path cuts are new exposures under this plan.",
                   "Predicted stores/syscalls stop before effect commit; no oracle or native-output reconditioning.",
                   "MLP zeroing is a distributed component contrast, not a minimal symbol or feature localization."]}


def worker():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    import numpy as np
    import torch
    from isa_lab.native_forward_symbol_assay import load_r3_mamba_model

    from mdb.discovery_cli import save_capture
    from mdb.discovery_isa import ISADriver
    from mdb.discovery_runtime import ReplayForward
    from mdb.equation_comparison import EquationCatalog, compare_equations
    from mdb.forecast_transaction import ForecastPayload, ForecastTransaction, verify_transaction
    from mdb.forward_ports import NativePortCatalog
    from mdb.forward_recording import record_forward, tensor_descriptor

    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    declared = json.loads(Path("plan.json").read_text())
    if sha(Path("plan.json").read_bytes()) != config["plan_file_sha256"]:
        raise ValueError("plan changed")
    for name, expected in request["source_hashes"].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError("staged source differs: " + name)
    formulas = runpy.run_path("state_transition_equations.py")
    distribution = importlib.metadata.distribution("isa-model-comparison")
    for name, expected in formulas["raw"]["SOURCE_FORMULAS"].items():
        if sha(distribution.locate_file(name).read_bytes()) != expected:
            raise ValueError("native installed source differs: " + name)
    torch.set_num_threads(1)
    model, model_sources = load_r3_mamba_model(config["checkpoint"])
    model.eval().requires_grad_(False)
    output = Path("state-transition-result")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    save(output / "plan.json", declared)
    save(output / "source.json", {"request": request, "model_sources": model_sources})
    counters = {"physical_forwards": 0, "replay_operations": 0, "acts": 0}
    native_events, transitions, reports = [], [], []
    parents, rows = {}, {}

    def enter(module, args):
        counters["physical_forwards"] += 1
        if counters["physical_forwards"] > declared["budgets"]["max_physical_forwards"]:
            raise ValueError("physical forward budget exhausted")
        native_events.append({"kind": "native_root_entry", "index": counters["physical_forwards"],
                              "elapsed_seconds": time.monotonic() - started})

    counter = model.register_forward_pre_hook(enter)

    def predict(context):
        result = formulas["forecast"](declared, context.weights)
        return {"predictions": result["predictions"], "payloads": {name: ForecastPayload(value, dtype="float32",
            provenance={"source": "independent raw-input equations", "recipe": name,
                        "cast": "frozen FP64 endpoint or feature -> explicit FP32 intervention; no observed donor"})
            for name, value in result["payloads"].items()}}

    def advance(runtime, pc, row_id):
        while runtime.pc < pc:
            if counters["replay_operations"] >= declared["budgets"]["max_replay_operations"]:
                raise ValueError("replay operation budget exhausted")
            receipt = runtime.advance_one()["receipt"]
            transitions.append({"row_id": row_id, "receipt": receipt})
            counters["replay_operations"] += 1

    def run_row(row_id, driver, prediction_path, context, *, lineage, replay_source_row=None):
        inputs = driver.inputs()
        before = {"register_words": list(driver.registers), "flags": list(driver.flags), "offset": driver.offset}
        recipes = declared["row_recipes"].get(row_id, [])
        hooks = []
        try:
            for edit in recipes:
                value = context.payloads["campaign"][edit["payload_name"]]
                module_path = edit["selector"][len("module:"):].rsplit(":", 1)[0]
                if edit["patch_kind"] == "direct-window-columns":
                    def prehook(module, args, frozen=value):
                        return (patch_direct_columns(args[0], frozen), *args[1:])
                    hooks.append(model.get_submodule(module_path).register_forward_pre_hook(prehook))
                else:
                    def replace(module, args, old, frozen=value):
                        return torch.from_numpy(np.array(frozen, copy=True))
                    hooks.append(model.get_submodule(module_path).register_forward_hook(replace))
            capture = record_forward(model, inputs, model_identity=declared["model_identity"],
                execution=declared["execution"], input_identity=identity({"program": driver.program.hex(), **before}),
                metadata={"case_id": row_id}, capture_ports=True, max_operations=8192)
        finally:
            for hook in hooks:
                hook.remove()
        manifest = save_capture(capture, output / "captures" / row_id)
        source_id = replay_source_row or row_id
        source = parents[source_id] if source_id != row_id else capture
        if source_id == row_id and (row_id == "sequence.native.0" or row_id.endswith(".native")):
            parents[row_id] = capture
        runtime = ReplayForward(source)
        source_ports = NativePortCatalog(source.trace)
        acts = []
        for edit in recipes:
            port = source_ports.resolve(edit["selector"], **({"path": edit["path"]} if "path" in edit else {}))
            advance(runtime, port["available_pc"], row_id)
            value = torch.from_numpy(np.array(context.payloads["campaign"][edit["payload_name"]], copy=True))
            extra = {}
            if edit["patch_kind"] == "direct-window-columns":
                prior = runtime.read(port["value_id"])
                extra["before_value"] = prior.detach().cpu().numpy().copy()
                value = patch_direct_columns(prior, context.payloads["campaign"][edit["payload_name"]])
                extra["after_value"] = value.detach().cpu().numpy().copy()
            receipt = runtime.apply(port["value_id"], value)
            acts.append({**edit, "port": port, "receipt": receipt, **extra})
            counters["acts"] += 1
        advance(runtime, len(runtime.trace["events"]), row_id)
        parity = {name: bool(torch.equal(runtime.outputs()[name], capture.output[name])) for name in HEADS}
        ports = NativePortCatalog(capture.trace)
        selectors = {"z": ("module:decoder:return", None), "tokens": ("module:inp:input", ["args", 0]),
                     "input": ("module:inp:return", None), "query": ("module:query:return", None),
                     "offset_head": ("module:offset_head:return", None), "layer2.h": ("module:reg_out:input", ["args", 0])}
        for layer in range(3):
            for module, name in ((f"mixers.{layer}", "M"), (f"mixers.{layer}.in_proj", "in_projected"),
                    (f"mixers.{layer}.select", "selected"), (f"mixers.{layer}.dt_proj", "dt_affine"),
                    (f"mixers.{layer}.out_proj", "U"), (f"post.{layer}", "P2"),
                    (f"post.{layer}.0", "post2_norm"), (f"post.{layer}.1", "post2_affine"),
                    (f"post.{layer}.2", "post2_gelu"), (f"post.{layer}.3", "P2_computed")):
                selectors[f"layer{layer}.{name}"] = ("module:" + module + ":return", None)
            selectors[f"layer{layer}.S"] = (f"module:mixers.{layer}:input", ["args", 0])
        actual_ports, stages = {}, {}
        for name, (selector, path) in selectors.items():
            port = ports.resolve(selector, **({"path": path} if path is not None else {}))
            actual_ports[name] = port
            stages[name] = capture.observed_values[port["value_id"]].detach().cpu().numpy().copy()
        native = driver.decode(capture.output)
        decoded = {"register_words": [int(word, 16) for word in native["register_words"]],
                   "flags": native["flags_CF_ZF_SF_OF"], "length": native["length"],
                   "store": native["store"], "syscall": native["syscall"]}
        roots = {name: value.detach().cpu().numpy().copy() for name, value in zip(("window", "regs", "flags"), inputs, strict=True)}
        predicted = prediction_at(context.predictions["campaign"], prediction_path)
        row = {"panel": row_id.split(".")[0], "prediction_path": prediction_path, "before": before,
            "roots": roots, "heads": {name: capture.output[name].detach().cpu().numpy().copy() for name in HEADS},
            "stages": stages, "decoded": decoded, "actual_ports": actual_ports, "acts": acts,
            "lineage": lineage, "replay_source_row": source_id, "replay_native_exact": parity,
            "forecast_root_join": predicted is not None and all(np.array_equal(roots[name], predicted["roots"][name]) for name in roots),
            "forecast_status": "sealed" if predicted is not None else "forecast-horizon-ended",
            "trace_fingerprint": capture.trace["fingerprint"], "input_identity": capture.trace["input_identity"],
            "capture_manifest": manifest, "input_roots": {name: tensor_descriptor(value) for name, value in zip(roots, inputs, strict=True)}}
        rows[row_id] = row
        print("MDB_STATE_TRANSITION_PROGRESS " + json.dumps({"row": row_id, "physical_forwards": counters["physical_forwards"],
              "RSI": decoded["register_words"][6], "RDI": decoded["register_words"][7], "replay_exact": all(parity.values())}), flush=True)
        return capture.output, row

    def execute(context):
        for arm in declared["arms"]:
            words, flags, previous, global_step, stopped = list(declared["register_words"]), list(declared["flags"]), None, 0, False
            for segment in declared["segments"]:
                driver = ISADriver(segment["program_hex"], words, flags, max_instructions=segment["max_instructions"])
                for segment_step in range(segment["max_instructions"]):
                    if driver.termination:
                        break
                    row_id = f"sequence.{arm}.{global_step}"
                    value, row = run_row(row_id, driver, ["trajectories", "sequence." + arm, "steps", str(global_step)], context,
                        lineage={"previous_row": previous, "segment_id": segment["id"], "segment_step": segment_step,
                                 "program_hex": segment["program_hex"], "host_segment_start": segment_step == 0},
                        replay_source_row="sequence.native.0" if global_step == 0 and arm != "native" else None)
                    committed = driver.commit(value)
                    row["driver_commit"] = {key: committed[key] for key in ("committed", "before", "after", "stop_causes", "overshoot") if key in committed}
                    previous, global_step = row_id, global_step + 1
                    if not committed["committed"]:
                        stopped = True
                        break
                words, flags = list(driver.registers), list(driver.flags)
                if stopped:
                    break
        for row_id in ("path.new.old", "path.old.new", "path.new.new", "path.carrier_other", "path.carrier_other_P2own"):
            driver = ISADriver(declared["contexts"]["old"], declared["register_words"], declared["flags"], max_instructions=1)
            run_row(row_id, driver, ["path_rows", row_id], context, lineage={"program_hex": declared["contexts"]["old"]},
                    replay_source_row="sequence.native.0")
        for op, program in declared["mlp_programs"].items():
            for arm in ("native", "zero_post0", "zero_post1", "zero_post2"):
                row_id = f"mlp.{op}.{arm}"
                driver = ISADriver(program, declared["register_words"], declared["flags"], max_instructions=1)
                run_row(row_id, driver, ["mlp_rows", row_id], context, lineage={"program_hex": program},
                        replay_source_row=f"mlp.{op}.native" if arm != "native" else None)
        return rows

    transaction = ForecastTransaction(output / "transaction", model=model, model_identity=CHECKPOINT,
        input_identity=config["plan_file_sha256"], source_identity=request["source_hashes"], frontier={"kind": "raw-input"},
        budgets={"max_native_calls": 40, "max_wall_ms": 240000, "max_bytes": 256 * 1024 * 1024, "max_files": 8192})
    rules = {name: "argmax" if name in ("dest", "src", "width", "length") else "positive" for name in HEADS}

    def analyze(observed):
        predictions = transaction.prediction("campaign")
        catalog = EquationCatalog(output / "equations")
        for row_id, actual in observed.items():
            predicted = prediction_at(predictions, actual["prediction_path"])
            if predicted is None:
                reports.append({"row_id": row_id, "forecast_status": "forecast-horizon-ended", "actual_decode": actual["decoded"]})
                continue
            baseline_id = ("sequence.native." + row_id.rsplit(".", 1)[1] if actual["panel"] == "sequence" else
                           "sequence.U_swap.0" if actual["panel"] == "path" else ".".join(row_id.split(".")[:2]) + ".native")
            baseline = observed.get(baseline_id)
            baseline_prediction = None if baseline is None else prediction_at(predictions, baseline["prediction_path"])
            if baseline_prediction is None:
                baseline = None
            head_report = compare_equations(predicted["heads"], actual["heads"], decision_rules=rules,
                baseline=None if baseline is None else baseline["heads"],
                predicted_baseline=None if baseline is None else baseline_prediction["heads"],
                targets={"regs": [[0, 7, bit] for bit in range(64)] + [[0, 6, 1]]}, components={"regs": predicted["register_terms"]})
            stage_report = compare_equations({name: predicted["stages"][name] for name in actual["stages"]}, actual["stages"])
            horizon = 5 if actual["panel"] == "sequence" else 1
            tested = {"checkpoint": CHECKPOINT, "source_identity": identity(request["source_hashes"]),
                      "execution": declared["execution"], "horizon": horizon, "panel": actual["panel"],
                      "case_id": row_id, "arm": row_id.split(".")[1],
                      "instruction": int(row_id.rsplit(".", 1)[1]) if actual["panel"] == "sequence" else 0,
                      "segment": actual["lineage"].get("segment_id"), "program_hex": actual["lineage"]["program_hex"],
                      "register_words": actual["before"]["register_words"], "flags": actual["before"]["flags"],
                      "byte_offset": actual["before"]["offset"], "input_identity": actual["input_identity"]}
            route = "+".join(edit["selector"] + "." + edit["patch_kind"] for edit in declared["row_recipes"].get(row_id, [])) or "native.forward"
            same_roots = baseline is not None and all(np.array_equal(actual["roots"][name], baseline["roots"][name]) for name in actual["roots"])
            record = catalog.record(law_id="r3.raw-state-transition", route_id=route,
                binding={"model_identity": CHECKPOINT, "source_identity": request["source_hashes"], "execution": declared["execution"],
                         "horizon": horizon, "input_identity": actual["input_identity"], "frontier": "raw-input"},
                conditions=[{"field": key, "op": "eq", "value": tested[key]} for key in tested],
                tested_context=tested, readers=["regs[6,1]", "regs[7,*]", "flags"], report=head_report,
                evidence={"forecast_seal": transaction.report()["seal_sha256"], "capture": actual["trace_fingerprint"],
                          "baseline_row": baseline_id, "forecast_root_join": actual["forecast_root_join"],
                          "contrast_scope": "matched-input" if same_roots else "own-state-trajectory-contrast"})
            reports.append({"row_id": row_id, "baseline_row": baseline_id, "head_report": head_report,
                            "stage_report": stage_report, "equation_record": record["fingerprint"],
                            "contrast_scope": "matched-input" if same_roots else "own-state-trajectory-contrast"})
        return {"rows": reports}

    def retain_previous_attempt():
        previous = config.get("previous_workdir")
        if previous is None:
            return
        root = Path(previous) / "state-transition-result"
        result = {"job_id": config["previous_job_id"], "remote_root": str(root),
                  "scope": "post-new-seal custody/identity check only; no previous arrays enter prediction callbacks"}
        try:
            report = json.loads((root / "transaction-report.json").read_text())
            result["verification"] = verify_transaction(root / "transaction", expected_seal_sha256=report["seal_sha256"],
                expected_event_head=report["event_head"], max_bytes=512 * 1024 * 1024, max_files=8192)
            prior = json.loads((root / "transaction/forecasts/campaign.json").read_text())
            current = json.loads((output / "transaction/forecasts/campaign.json").read_text())
            old_ids, new_ids = array_identities(prior), array_identities(current)
            result["retained_array_occurrences"] = len(new_ids)
            result["changed_retained_array_paths"] = [key for key, value in new_ids.items() if old_ids.get(key) != value]
            result["native_events"] = json.loads((root / "native-events.json").read_text())
            result["work"] = report["work"]
            result["original_files"] = {path.relative_to(root).as_posix(): {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
                for path in sorted(root.rglob("*")) if path.is_file()}
            for name in ("transaction-report.json", "native-events.json", "failure.json", "transaction/seal.json",
                         "transaction/forecasts/campaign.json"):
                (output / ("prior-" + name.replace("/", "-"))).write_bytes((root / name).read_bytes())
            result["status"] = "verified" if not result["changed_retained_array_paths"] and not result["native_events"] else "differs"
        except (OSError, ValueError, RuntimeError) as error:
            result.update(status="evidence-incomplete", error=f"{type(error).__name__}: {error}")
        save(output / "prior-attempt.json", result)

    try:
        with transaction:
            transaction.run_script([{"op": "frontier", "name": "raw", "value": declared},
                {"op": "predict", "name": "campaign", "callback": predict}, {"op": "seal"}])
            retain_previous_attempt()
            transaction.run_script([{"op": "execute", "name": "native", "callback": execute, "analysis": analyze}])
    finally:
        counter.remove()
        save(output / "transaction-report.json", transaction.report())
        save(output / "native-events.json", native_events)
        save(output / "public-replay-receipts.json", transitions)
    save(output / "comparisons.json", {"rows": reports})
    verified = verify_transaction(output / "transaction", expected_seal_sha256=transaction.report()["seal_sha256"],
                                  expected_event_head=transaction.report()["event_head"])
    summary = {"schema": "mdb-state-transition-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "counts": counters, "rows": len(rows),
        "worker_seconds": time.monotonic() - started, "transaction_verification": verified,
        "mechanics_status": "valid" if all(all(row["replay_native_exact"].values()) for row in rows.values()) else "instrument-mismatch",
        "terminal_status": "unassessed", "limits": declared["limits"]}
    save(output / "summary.json", summary)
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path.read_bytes()) for path in sorted(output.rglob("*")) if path.is_file()})
    support()["emit"](output, summary)


def stage(run_dir, isa_root, *, previous_workdir=None, previous_job_id=None):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    declared = plan()
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", declared)
    upstream = runpy.run_path(str(isa_root / "scripts/native_forward_symbols.py"))
    upstream["_worker_environment_defaults"]()
    sources, origins, _ = upstream["source_inventory"]()
    wheel = runpy.run_path(str(isa_root / "scripts/symbol_sources.py"))["build_isa_wheel"](run_dir, sources)
    files = [(path, name) for name, path in sources.items()]
    for name in ("state_transition_equations.py", "raw_isa_equations.py", "explicit_suffix_equations.py"):
        files.append((Path(__file__).with_name(name), name))
    files += [(wheel, wheel.name), (run_dir / "plan.json", "plan.json"),
              (Path(__file__).with_name("discovery_upgrades.py"), "campaign_support.py")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "path_token_allowlist": ["state-transition-result"],
        "hashed_roots": ["artifacts", "direct_effect", "isa_lab", wheel.name, "state_transition_campaign.py",
                         "state_transition_equations.py", "raw_isa_equations.py", "explicit_suffix_equations.py", "campaign_support.py",
                         "plan.json", "source-manifest.json", "preflight.v1.json"]})
    files.append((preflight, preflight.name))
    config = {"schema": "mdb-state-transition-config-v1", "checkpoint": "artifacts/model.pt",
              "plan_file_sha256": sha((run_dir / "plan.json").read_bytes())}
    if previous_workdir is not None:
        config.update(previous_workdir=previous_workdir, previous_job_id=previous_job_id)
    request = {"schema": "mdb-state-transition-request-v1", "task_family": "mdb-state-transition", "model_identity": CHECKPOINT,
        "needs": {"cpu": True}, "config": config, "execution_environment": intent(), "source_origins": origins,
        "source_hashes": {name: sha(path.read_bytes()) for path, name in files},
        "scheduling_contract": one_lease_scheduling_contract(logical_operations=40, logical_branches=28,
            logical_checkpoint_replays=28, max_local_work_items=64,
            extra={"max_replay_native_operations": 40000, "max_physical_model_forwards": 40}),
        "selection_policy": "one fixed presealed finite panel; no outcome-adaptive selection or retries"}
    request["source_hashes"][Path(__file__).name] = sha(Path(__file__).read_bytes())
    request["request_sha256"] = identity(request)
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=Path(__file__), config=config, request=request, additional_files=files)
    command = worker_command(Path(__file__).name) + ["--worker"]
    index = command.index("python", command.index("uv") + 1)
    command[index:index] = ["--with", "./" + wheel.name]
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, experiment="mdb-state-transition",
        task="r3-own-state-segments-context-paths-matched-mlps", model=None, payload=run_dir / "payload",
        prefer_host="beast", gpu=False, needs=request["needs"], ram_mb=2304, cpu_threads=1,
        timeout_s=300, retry_on_kill=False, preflight=True,
        note="CPU1 FP32; <=28planned/40max B1 captures,40k typed replayops. Seal all forecasts first. Unchanged twoSUB then explicit host XOR/OR/AND segments; 4context corners fixedU plus2carrier contrasts; OR/AND x3MLPzeros.2304MiB measured exception(parent1129MiB+bounded256MiB evidence).NoB4 component replay/no oracle/no retries. Stopafterpanel/archive.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--isa-root", type=Path)
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.worker:
        try:
            worker()
        except Exception as error:
            output = Path("state-transition-result")
            if output.is_dir():
                request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
                failed = {"schema": "mdb-state-transition-result-v1", "mechanics_status": "errored", "job_id": os.environ["MRUN_JOB_ID"],
                          "request_sha256": request["request_sha256"], "error": f"{type(error).__name__}: {error}"}
                save(output / "failure.json", failed)
                support()["emit"](output, failed)
            raise
    elif args.collect:
        print(json.dumps(support()["collect"](args.run_dir, args.collect, max_expanded_bytes=512 * 1024 * 1024)))
    else:
        if args.run_dir is None or args.isa_root is None:
            parser.error("staging requires run-dir and isa-root")
        print(json.dumps(stage(args.run_dir.resolve(), args.isa_root.resolve())))


if __name__ == "__main__":
    main()
