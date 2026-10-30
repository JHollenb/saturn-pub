"""Forecast complete raw-input ISA trajectories before any native model call."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import time
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def support():
    source = Path(__file__).with_name("campaign_support.py")
    if not source.exists():
        source = Path(__file__).with_name("discovery_upgrades.py")
    return runpy.run_path(str(source))


def worker():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
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

    formulas = runpy.run_path("raw_isa_equations.py")
    plan_raw = Path("plan.json").read_bytes()
    plan = json.loads(plan_raw)
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if sha(plan_raw) != config["plan_file_sha256"]:
        raise ValueError("plan bytes differ")
    for name, expected in request["source_hashes"].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError("staged source differs: " + name)
    distribution = importlib.metadata.distribution("isa-model-comparison")
    for name, expected in formulas["SOURCE_FORMULAS"].items():
        if sha(distribution.locate_file(name).read_bytes()) != expected:
            raise ValueError("installed ISA source differs: " + name)
    torch.set_num_threads(1)
    model, model_sources = load_r3_mamba_model(config["checkpoint"])
    model.eval().requires_grad_(False)
    if (model.arch != "mamba" or not model.aligned or not model.store or model.snap
            or model.regptr or model.relptr or model.maskkey or len(model.mixers) != 3):
        raise ValueError("native configuration differs from raw reference")
    output = Path("sequence-result")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    save(output / "plan.json", plan)
    save(output / "source.json", {"request": request, "model_sources": model_sources})
    counters = {"physical_forwards": 0, "replay_operations": 0, "acts": 0}
    captures, observed_rows, all_acts, transition_receipts = {}, {}, [], []
    native_events = []

    def enter(module, args):
        counters["physical_forwards"] += 1
        if counters["physical_forwards"] > plan["budgets"]["max_physical_forwards"]:
            raise ValueError("physical budget exhausted")
        native_events.append({"kind": "native_root_entry", "index": counters["physical_forwards"],
                              "elapsed_seconds": time.monotonic() - started})

    count_hook = model.register_forward_pre_hook(enter)
    rules = {name: "argmax" if name in ("dest", "src", "width", "length") else "positive"
             for name in ("regs", "flags", "dest", "src", "width", "length", "syscall", "imm",
                          "store", "store_addr", "store_value")}

    def predict(context):
        weights = formulas["prepare_weights"](context.weights)
        trajectories, payloads = {}, {}
        for case in plan["cases"]:
            for arm in plan["arms"]:
                key = case["case_id"] + "." + arm
                trajectory = formulas["forecast_trajectory"](plan["program_hex"], case["register_words"],
                    case["flags"], weights, arm=arm, instructions=plan["max_instructions"])
                trajectories[key] = trajectory
                for name, value in trajectory["payloads"].items():
                    payloads[key + "." + name] = ForecastPayload(value, dtype="float32", provenance={
                        "source": "full raw-input independent native analytical trajectory",
                        "case": case["case_id"], "arm": arm, "instruction": 0,
                        "cast_schedule": "FP64 computed native endpoint -> FP32 payload; no observed future"})
        return {"predictions": trajectories, "payloads": payloads}

    def advance(runtime, pc):
        while runtime.pc < pc:
            if counters["replay_operations"] >= plan["budgets"]["max_replay_operations"]:
                raise ValueError("replay budget exhausted")
            row = runtime.advance_one()
            transition_receipts.append(row["receipt"])
            counters["replay_operations"] += 1

    def port_value(capture, ports, selector, path=None):
        resolved = ports.resolve(selector, path=path)
        return capture.observed_values[resolved["value_id"]].detach().cpu().numpy().copy()

    def execute(context):
        forecasts = context.predictions["campaign"]
        payloads = context.payloads["campaign"]
        result = {}
        for case in plan["cases"]:
            source = None
            for arm in plan["arms"]:
                key = case["case_id"] + "." + arm
                trajectory = forecasts[key]
                driver = ISADriver(plan["program_hex"], case["register_words"], case["flags"],
                                   max_instructions=plan["max_instructions"])
                for step in range(plan["max_instructions"]):
                    if driver.termination:
                        break
                    words, flags, offset = list(driver.registers), list(driver.flags), driver.offset
                    inputs = driver.inputs()
                    raw = tuple(value.detach().cpu().numpy().copy() for value in inputs)
                    predicted = trajectory["steps"].get(str(step))
                    root_join = predicted is not None and all(np.array_equal(value, predicted["roots"][name])
                        for name, value in zip(("window", "regs", "flags"), raw, strict=True))
                    hooks = []

                    def replace(value):
                        def replacement(module, args, old):
                            return torch.from_numpy(np.array(value, copy=True))
                        return replacement

                    try:
                        if step == 0 and arm != "native":
                            hooks.append(model.post[0].register_forward_hook(replace(payloads[key + ".post0_zero"])))
                            hooks.append(model.mixers[2].out_proj.register_forward_hook(replace(payloads[key + ".U_native"])))
                            if arm == "U_swap_P2native":
                                hooks.append(model.post[2].register_forward_hook(replace(payloads[key + ".P2_native"])))
                        capture = record_forward(model, inputs, model_identity=plan["model_identity"],
                            input_identity=identity({"program": plan["program_hex"], "offset": offset,
                                                     "words": words, "flags": flags}),
                            execution=plan["execution"], capture_ports=True,
                            metadata={"case_id": key, "program_step": step}, max_operations=8192)
                    finally:
                        for hook in hooks:
                            hook.remove()
                    row_id = key + "." + str(step)
                    capture_manifest = save_capture(capture, output / "captures" / row_id)
                    ports = NativePortCatalog(capture.trace)
                    if step == 0 and arm == "native":
                        source = capture
                    runtime = ReplayForward(source if step == 0 and arm != "native" else capture)
                    acts = []
                    if step == 0 and arm != "native":
                        source_ports = NativePortCatalog(source.trace)
                        changes = [("module:post.0:return", "post0_zero"),
                                   ("module:mixers.2.out_proj:return", "U_native")]
                        if arm == "U_swap_P2native":
                            changes.append(("module:post.2:return", "P2_native"))
                        for selector, name in changes:
                            port = source_ports.resolve(selector)
                            advance(runtime, port["available_pc"])
                            act = runtime.apply(port["value_id"], torch.from_numpy(np.array(payloads[key + "." + name], copy=True)))
                            acts.append(act)
                            counters["acts"] += 1
                    advance(runtime, len(runtime.trace["events"]))
                    replay_heads = runtime.outputs()
                    parity = {name: bool(torch.equal(replay_heads[name], capture.output[name])) for name in rules}
                    if not all(parity.values()):
                        raise ValueError("MDB replay/native instrument mismatch: " + row_id)
                    actual_heads = {name: value.detach().cpu().numpy().copy() for name, value in capture.output.items()}
                    native_decode = driver.decode(capture.output)
                    decoded = {"register_words": [int(word, 16) for word in native_decode["register_words"]],
                               "flags": native_decode["flags_CF_ZF_SF_OF"], "length": native_decode["length"],
                               "store": native_decode["store"], "syscall": native_decode["syscall"]}
                    actual_stages = {}
                    references = {}
                    for module, stage in [("decoder", "z"), ("inp", "input"), ("query", "query"),
                                          ("offset_head", "offset_head")]:
                        selector = "module:" + module + ":return"
                        actual_stages[stage] = port_value(capture, ports, selector)
                        references[stage] = ports.resolve(selector)
                    for layer in range(3):
                        module_stages = {f"mixers.{layer}": "M", f"mixers.{layer}.in_proj": "in_projected",
                            f"mixers.{layer}.select": "selected", f"mixers.{layer}.dt_proj": "dt_affine",
                            f"mixers.{layer}.out_proj": "U", f"post.{layer}": "P2",
                            f"post.{layer}.0": "post2_norm", f"post.{layer}.1": "post2_affine",
                            f"post.{layer}.2": "post2_gelu", f"post.{layer}.3": "P2_computed"}
                        for module, stage in module_stages.items():
                            selector = "module:" + module + ":return"
                            name = f"layer{layer}." + stage
                            actual_stages[name] = port_value(capture, ports, selector)
                            references[name] = ports.resolve(selector)
                    # Demonstrate actual bias ports; source root bytes bind the full state dictionary.
                    named_bias = ports.resolve("parameter:mixers.2.dt_proj.bias")
                    result[row_id] = {"heads": actual_heads, "stages": actual_stages,
                        "decoded": decoded, "before": {"register_words": list(words), "flags": list(flags), "offset": offset},
                        "roots": {name: value for name, value in zip(("window", "regs", "flags"), raw, strict=True)},
                        "forecast_root_join": root_join, "replay_native_exact": parity,
                        "trace_fingerprint": capture.trace["fingerprint"], "input_identity": capture.trace["input_identity"],
                        "capture_manifest": capture_manifest,
                        "actual_ports": references, "named_bias_port": named_bias, "acts": acts,
                        "input_roots": {name: tensor_descriptor(value) for name, value in zip(("window", "regs", "flags"), inputs, strict=True)}}
                    observed_rows[row_id] = result[row_id]
                    all_acts.extend(acts)
                    captures[row_id] = capture.trace["fingerprint"]
                    commit = driver.commit(capture.output)
                    result[row_id]["driver_commit"] = {name: commit[name] for name in
                        ("committed", "before", "after", "stop_causes", "overshoot") if name in commit}
                    if not commit["committed"]:
                        result[row_id]["termination"] = "stop-before-effect-commit"
                        break
                    if driver.termination:
                        result[row_id]["termination"] = driver.termination
                        break
        return result

    reports = []
    transaction = ForecastTransaction(output / "transaction", model=model,
        model_identity=plan["model_identity"], input_identity=sha(plan_raw),
        source_identity=request["source_hashes"], frontier=plan["frontier"],
        budgets={"max_native_calls": 24, "max_wall_ms": 240000,
                 "max_bytes": 384 * 1024 * 1024, "max_files": 8192})

    def analyze(observed):
        forecasts = transaction.prediction("campaign")
        catalog = EquationCatalog(output / "equations")
        for row_id, actual in observed.items():
            case, arm, step = row_id.split(".")
            predicted = forecasts[case + "." + arm]["steps"].get(step)
            if predicted is None:
                reports.append({"row_id": row_id, "instrument_status": "forecast-horizon-ended", "actual": actual["decoded"]})
                continue
            parent = observed.get(case + ".native." + step)
            parent_prediction = forecasts[case + ".native"]["steps"].get(step)
            comparison = compare_equations(predicted["heads"], actual["heads"], decision_rules=rules,
                baseline=None if parent is None else parent["heads"],
                predicted_baseline=None if parent_prediction is None else parent_prediction["heads"],
                targets={"regs": [[0, 7, bit] for bit in range(64)]},
                components={"regs": predicted["register_terms"]})
            stage_report = compare_equations({name: predicted["stages"][name] for name in actual["stages"]}, actual["stages"])
            record = catalog.record(law_id="r3.full-raw-state-transition", route_id=arm,
                binding={"model_identity": plan["model_identity"], "execution": plan["execution"],
                         "input_identity": actual["input_identity"], "frontier": "raw-input",
                         "horizon": plan["max_instructions"], "source_identity": request["source_hashes"]},
                report=comparison, conditions=[
                    {"field": "architecture", "op": "eq", "value": "r3-mamba-aligned"},
                    {"field": "checkpoint", "op": "eq", "value": plan["model_identity"]},
                    {"field": "execution", "op": "eq", "value": plan["execution"]},
                    {"field": "source_identity", "op": "eq", "value": identity(request["source_hashes"])}],
                tested_context={"case": case, "arm": arm, "instruction": int(step),
                                "baseline": "same original state trajectory; later roots may differ"},
                readers=["regs:7", "regs:6:1", "flags"],
                evidence={"forecast_seal": transaction.report()["seal_sha256"], "capture": actual["trace_fingerprint"]})
            reports.append({"row_id": row_id, "forecast_root_join": actual["forecast_root_join"],
                "forecast_decode": dict(predicted["decoded"]), "actual_decode": dict(actual["decoded"]),
                "head_report": comparison, "stage_report": stage_report, "equation_record": record["fingerprint"]})
        return {"rows": reports}

    try:
        with transaction:
            transaction.run_script([
                {"op": "frontier", "name": "raw", "value": plan},
                {"op": "predict", "name": "campaign", "callback": predict},
                {"op": "seal"},
                {"op": "execute", "name": "native", "callback": execute, "analysis": analyze},
            ])
    finally:
        count_hook.remove()
        save(output / "transaction-report.json", transaction.report())
        save(output / "native-events.json", native_events)
        save(output / "public-replay-receipts.json", transition_receipts)
    save(output / "comparisons.json", {"rows": reports})
    verified = verify_transaction(output / "transaction", expected_seal_sha256=transaction.report()["seal_sha256"],
                                  expected_event_head=transaction.report()["event_head"])
    summary = {"schema": "mdb-raw-isa-sequence-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "mechanics_status": "valid", "terminal_status": "unassessed",
        "worker_seconds": time.monotonic() - started, "counts": counters, "rows": len(reports),
        "head_decision_mismatches": sum(row.get("head_report", {}).get("decision_mismatches", 0) for row in reports),
        "forecast_root_joins": sum(row.get("forecast_root_join", False) for row in reports),
        "max_head_abs": max((row.get("head_report", {}).get("max_abs", 0) for row in reports), default=0),
        "max_stage_abs": max((row.get("stage_report", {}).get("max_abs", 0) for row in reports), default=0),
        "transaction_verification": verified, "architectural_correctness": "not an oracle-scored ISA accuracy claim",
        "limits": ["two fixed related initial states, one supplied17byteprogram andthree declaredarms",
                   "full source equations and frozen weights supplied; no semantic operation discovery",
                   "each forecast/native trajectory owns its own prior state; mismatches never recondition forecasts"]}
    save(output / "summary.json", summary)
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path.read_bytes())
                               for path in sorted(output.rglob("*")) if path.is_file()})
    support()["emit"](output, summary)


def stage(run_dir, isa_root, plan_path):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    raw_plan = plan_path.read_bytes()
    plan = json.loads(raw_plan)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plan.json").write_bytes(raw_plan)
    upstream = runpy.run_path(str(isa_root / "scripts/native_forward_symbols.py"))
    upstream["_worker_environment_defaults"]()
    sources, origins, _ = upstream["source_inventory"]()
    wheel = runpy.run_path(str(isa_root / "scripts/symbol_sources.py"))["build_isa_wheel"](run_dir, sources)
    files = [(path, name) for name, path in sources.items()]
    files += [(wheel, wheel.name), (Path(__file__).with_name("discovery_upgrades.py"), "campaign_support.py"),
              (Path(__file__).with_name("raw_isa_equations.py"), "raw_isa_equations.py"),
              (Path(__file__).with_name("explicit_suffix_equations.py"), "explicit_suffix_equations.py"),
              (run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "path_token_allowlist": ["sequence-result"],
        "hashed_roots": ["artifacts", "direct_effect", "isa_lab", wheel.name, "raw_isa_sequence_campaign.py",
                         "raw_isa_equations.py", "explicit_suffix_equations.py", "campaign_support.py",
                         "plan.json", "source-manifest.json", "preflight.v1.json"]})
    files.append((preflight, preflight.name))
    config = {"schema": "mdb-raw-isa-sequence-config-v1", "checkpoint": "artifacts/model.pt",
              "plan_file_sha256": sha(raw_plan)}
    request = {"schema": "mdb-raw-isa-sequence-request-v1", "task_family": "mdb-raw-isa-sequence",
        "model_identity": plan["model_identity"], "needs": {"cpu": True}, "config": config,
        "execution_environment": intent(), "source_hashes": {name: sha(path.read_bytes()) for path, name in files},
        "source_origins": origins, "scheduling_contract": one_lease_scheduling_contract(logical_operations=24,
            logical_branches=6, logical_checkpoint_replays=18, max_local_work_items=24,
            extra={"max_replay_native_operations": 20000, "max_physical_model_forwards": 24}),
        "selection_policy": "two fixed related states xthree arms xthree own-state instructions; all forecasts sealed before any nativecall"}
    request["source_hashes"]["raw_isa_sequence_campaign.py"] = sha(Path(__file__).read_bytes())
    request["request_sha256"] = identity(request)
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=Path(__file__), config=config, request=request, additional_files=files)
    command = worker_command("raw_isa_sequence_campaign.py") + ["--worker"]
    index = command.index("python", command.index("uv") + 1)
    command[index:index] = ["--with", "./" + wheel.name]
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, experiment="mdb-raw-isa-sequence",
        task="r3-full-raw-equations-three-instruction-forecast", model=None, payload=run_dir / "payload",
        prefer_host="beast", gpu=False, needs=request["needs"], ram_mb=2304, cpu_threads=1,
        timeout_s=300, retry_on_kill=False, preflight=True,
        note="One CPU1 finite full-weight raw-input forecast:2fixedstates x3arms x3instructions;zero nativecalls beforeglobalseal; <=24physical/20k replayops. SUB/XOR/ADD own-state futures,no oracle/no reconditioning/no retries.2304MiB retained measuredCPUparent exception (prior1034.5MiB peak) plus bounded384MiB evidence. Stopafter18comparisons/archive.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--isa-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.worker:
        try:
            worker()
        except Exception as error:
            output = Path("sequence-result")
            if output.is_dir():
                request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
                report = {"schema": "mdb-raw-isa-sequence-result-v1", "mechanics_status": "errored",
                          "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
                          "error": f"{type(error).__name__}: {error}"}
                save(output / "failure.json", report)
                support()["emit"](output, report)
            raise
    elif args.collect:
        print(json.dumps(support()["collect"](args.run_dir, args.collect, max_expanded_bytes=512 * 1024 * 1024)))
    else:
        if args.run_dir is None or args.isa_root is None or args.plan is None:
            parser.error("staging requires run-dir, isa-root and plan")
        print(json.dumps(stage(args.run_dir.resolve(), args.isa_root.resolve(), args.plan.resolve())))


if __name__ == "__main__":
    main()
