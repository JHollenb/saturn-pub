"""Execute a sealed, prospective route panel in one finite installed CPU lease.

Forecasts are evidence to compare after execution. Only their supplied program
bytes and initial state enter the model driver; expected states and pattern
predictions never enter a forward, Act, or architectural commit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import runpy
import time
from pathlib import Path

CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"
SITES = ("post.0", "post.1", "post.2")
ARMS = ("identity", "post0_zero", "post1_zero", "post2_zero", "projection_native",
        "skip_native", "carrier_native", "projection_sham")
HEADS = {"regs": (1, 8, 64), "flags": (1, 4), "dest": (1, 8), "src": (1, 8),
         "width": (1, 3), "length": (1, 15), "syscall": (1,), "imm": (1, 64),
         "store": (1,), "store_addr": (1, 64), "store_value": (1, 64)}


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


def raw_heads(outputs):
    return {key: value.detach().cpu().tolist() for key, value in outputs.items()}


def projection(decoded):
    return {key: value for key, value in decoded.items() if key != "raw_logits"}


def retain_primary_on_continuation_error(row, action):
    """Attach an optional future without discarding completed primary evidence.

    ``action`` fills the continuation dictionary in place, so an error retains
    any already observed commit, literal successor inputs, or secondary heads.
    """
    continuation = {}
    row["continuation"] = continuation
    try:
        action(continuation)
    except Exception as error:
        continuation.update({"status": "errored", "error": f"{type(error).__name__}: {error}"})
    return row


def validate_plan(plan):
    """Validate the supplied input envelope without evaluating its forecasts."""
    if not isinstance(plan, dict) or not isinstance(plan.get("cases"), list) or len(plan["cases"]) != 6:
        raise ValueError("predictive campaign requires six sealed cases")
    unsealed = dict(plan)
    plan_digest = unsealed.pop("plan_sha256", None)
    if not isinstance(plan_digest, str) or identity(unsealed) != plan_digest:
        raise ValueError("prospective plan self seal differs")
    if (plan.get("schema") != "mdb-predictive-route-plan-v1" or plan.get("model_identity") != CHECKPOINT
            or plan.get("backend") != "cpu-fp32" or plan.get("cpu_threads") != 1):
        raise ValueError("prospective plan model/execution applicability differs")
    owners = set()
    for case in plan["cases"]:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9_-]{1,64}", case_id):
            raise ValueError("case_id must be a safe, explicit identifier")
        if case_id in owners:
            raise ValueError("duplicate forecast case identity")
        owners.add(case_id)
        code = case.get("program_hex")
        if (not isinstance(code, str) or not code or len(code) % 2
                or any(char not in "0123456789abcdef" for char in code)):
            raise ValueError("program bytes must be canonical hexadecimal")
        registers, flags = case.get("register_words"), case.get("flags")
        if not isinstance(registers, list) or len(registers) != 8:
            raise ValueError("each case needs eight raw initial register words")
        if not isinstance(flags, list) or len(flags) != 4 or any(type(v) is not int or v not in (0, 1) for v in flags):
            raise ValueError("each case needs four raw initial flag bits")
        window = bytes.fromhex(code)[:15].ljust(15, b"\0").hex()
        envelope = {"window_hex": window, "register_words": registers, "flags": flags}
        if case.get("window_hex") != window or case.get("input_envelope_sha256") != identity(envelope):
            raise ValueError("case input envelope differs from sealed raw program/state")
    declared = plan.get("arms")
    if declared is not None:
        names = [value.get("name", value.get("arm")) if isinstance(value, dict) else value for value in declared]
        if tuple(names) != ARMS:
            raise ValueError("forecast arm order differs from the implemented route panel")
    return plan


def worker():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    from isa_lab.native_forward_symbol_assay import load_r3_mamba_model

    from mdb.discovery_cli import save_capture
    from mdb.discovery_isa import ISADriver
    from mdb.discovery_runtime import ReplayForward
    from mdb.forward_recording import record_forward, tensor_descriptor

    torch.set_num_threads(1)
    started = time.perf_counter()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    output = Path("predictive-route-result")
    output.mkdir(exist_ok=False)
    if config.get("schema") != "mdb-predictive-route-config-v1" or request.get("model_identity") != CHECKPOINT:
        raise ValueError("predictive campaign config/model binding differs")
    unsealed = dict(request)
    request_digest = unsealed.pop("request_sha256")
    if identity(unsealed) != request_digest:
        raise ValueError("predictive campaign request seal differs")
    for name, expected in request["source_hashes"].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError("staged source/input changed: " + name)
    forecast_raw = Path("forecast.json").read_bytes()
    forecast_digest = sha(forecast_raw)
    if (forecast_digest != request.get("forecast_sha256")
            or forecast_digest != config.get("forecast_sha256")):
        raise ValueError("forecast binding differs from sealed request/config")
    plan = validate_plan(json.loads(forecast_raw))
    if plan["plan_sha256"] != request.get("plan_sha256") or plan["plan_sha256"] != config.get("plan_sha256"):
        raise ValueError("plan self seal differs from request/config")
    installed = importlib.metadata.distribution("isa-model-comparison")
    for name, expected in request["source_hashes"].items():
        if name.endswith(".py") and name.startswith(("isa_lab/", "direct_effect/")):
            if sha(installed.locate_file(name).read_bytes()) != expected:
                raise ValueError("installed ISA source changed: " + name)
    model, model_sources = load_r3_mamba_model(config["checkpoint"])
    model.eval().requires_grad_(False)
    weights_before = {name: tensor_descriptor(value)["sha256"] for name, value in model.state_dict().items()}
    physical_calls, replay_operations = 0, 0
    physical_geometry, phases, receipts, acts, followups, case_records, errors = [], [], [], [], [], [], []
    current_branch = "setup"

    def count_forward(module, args):
        nonlocal physical_calls
        if physical_calls >= config["max_physical_forwards"]:
            raise ValueError("predictive campaign physical-forward allowance exhausted")
        physical_calls += 1
        physical_geometry.append({"attempt": physical_calls, "branch": current_branch,
                                  "batch": int(args[0].shape[0])})
    count_hook = model.register_forward_pre_hook(count_forward)
    # Persist the forecast and verification before ANY model execution. Loading
    # weights and input-envelope validation above perform no native forward.
    (output / "forecast.json").write_bytes(forecast_raw)
    binding = {"schema": "mdb-predictive-route-forecast-binding-v1",
               "forecast_sha256": forecast_digest, "request_sha256": request_digest,
               "plan_sha256": plan["plan_sha256"],
               "model_identity": CHECKPOINT, "case_ids": [case["case_id"] for case in plan["cases"]],
               "arms": list(ARMS), "physical_forwards_at_binding": physical_calls,
               "verification": "sealed-before-first-model-forward",
               "forecast_used_as_model_input": False,
               "serving_inputs": "supplied program bytes and initial registers/flags only"}
    save(output / "forecast-binding.json", binding)
    print("MDB_PREDICTIVE_FORECAST_BOUND " + json.dumps(binding, sort_keys=True), flush=True)

    def measured(name, callback):
        before, begin = physical_calls, time.perf_counter()
        result = callback()
        phases.append({"phase": name, "seconds": time.perf_counter() - begin,
                       "physical_forwards": physical_calls - before})
        (output / "progress.json").write_text(json.dumps({"phases": phases,
            "physical_forwards": physical_calls, "replay_operations": replay_operations}))
        print("MDB_PREDICTIVE_ROUTE_PROGRESS " + json.dumps(phases[-1]), flush=True)
        return result

    def native(driver):
        with torch.no_grad():
            return model(*driver.inputs())

    def advance(runtime, pc):
        nonlocal replay_operations
        while runtime.pc < pc:
            if replay_operations >= config["max_replay_operations"]:
                raise ValueError("predictive campaign replay primitive allowance exhausted")
            replay_operations += 1
            result = runtime.advance_one()
            receipts.append({"branch": current_branch, "receipt": result["receipt"]})

    def apply(runtime, port, payload):
        result = runtime.apply(port, payload)
        acts.append({"branch": current_branch, "receipt": result})
        return result

    def compare(left, right):
        if set(left) != set(HEADS) or set(right) != set(HEADS):
            raise ValueError("comparison must include all eleven native heads")
        return {key: {"exact": torch.equal(left[key], right[key]),
                      "max_abs": float((left[key] - right[key]).abs().max())} for key in HEADS}

    def investigate(case):
        nonlocal current_branch
        case_id = case["case_id"]
        folder = output / "cases" / case_id
        # These are the ONLY fields of the forecast case used to build serving
        # state. Predicted first/second results, reader/sign and pattern IDs are
        # never consulted during model execution or commit.
        driver = ISADriver(case["program_hex"], case["register_words"], case["flags"], max_instructions=2)
        roots = driver.inputs()
        input_digest = identity([tensor_descriptor(value) for value in roots])
        current_branch = case_id + "/native_reference"
        capture = record_forward(model, roots, model_identity=CHECKPOINT,
            input_identity=input_digest, execution={"backend": "cpu", "dtype": "float32", "threads": 1},
            max_operations=8192)
        runtime = ReplayForward(capture)
        laws = {site: runtime.catalog.resolve("module:" + site) for site in SITES}
        proj = runtime.catalog.resolve("module:mixers.2.out_proj:output")["value_id"]
        carrier = runtime.catalog.resolve("module:post.2:input")["value_id"]
        producer = capture.trace["events"][capture.trace["values"][carrier]["producer"]]
        if producer["operator"] != "aten.add.Tensor" or proj not in producer["inputs"]:
            raise ValueError("final mixer producer is not the declared skip/projection add")
        if producer["arguments"].get("kwargs", {}).get("alpha", 1) != 1:
            raise ValueError("final mixer has a nonunit residual coefficient")
        other = [value for value in producer["inputs"] if value != proj]
        if len(other) != 1:
            raise ValueError("final mixer skip operand is ambiguous")
        skip = other[0]
        proj_pc = capture.trace["values"][proj]["producer"] + 1
        carrier_pc = capture.trace["values"][carrier]["producer"] + 1
        if not proj_pc < carrier_pc <= laws["post.2"]["entry"]:
            raise ValueError("final mixer ports are not ordered before the final MLP")
        donor = {}
        handles = [model.mixers[2].register_forward_pre_hook(
            lambda module, args: donor.__setitem__("skip", args[0].detach().clone())),
            model.mixers[2].out_proj.register_forward_hook(
            lambda module, args, value: donor.__setitem__("projection", value.detach().clone())),
            model.mixers[2].register_forward_hook(
            lambda module, args, value: donor.__setitem__("carrier", value.detach().clone()))]
        try:
            native_heads = native(driver)
        finally:
            for handle in handles:
                handle.remove()
        baseline_decode = projection(driver.decode(native_heads))
        capture_check = compare(capture.output, native_heads)
        if not all(row["exact"] for row in capture_check.values()):
            raise ValueError("unchanged capture differs from its independent native control")
        save(folder / "trace.json", capture.trace)
        save_capture(capture, folder / "root-recipe")
        cuts = {}
        for site in SITES:
            advance(runtime, laws[site]["exit"])
            cuts[site] = runtime.capture()
        advance(runtime, len(capture.trace["events"]))
        replay_check = compare(runtime.outputs(), native_heads)
        if not all(row["exact"] for row in replay_check.values()):
            raise ValueError("unchanged scalar replay differs from independent native control")
        current_branch = case_id + "/native_donor"
        donor_runtime = runtime.fork(cuts["post.1"])
        advance(donor_runtime, carrier_pc)
        replay_donor = {"skip": donor_runtime.read(skip), "projection": donor_runtime.read(proj),
                        "carrier": donor_runtime.read(carrier)}
        donor_exact = {key: torch.equal(value, donor[key]) for key, value in replay_donor.items()}
        if not all(donor_exact.values()):
            raise ValueError("live donor operands differ from independent native hook observations")
        donor_cut = donor_runtime.capture().fingerprint
        record = {"case_id": case_id, "forecast_sha256": forecast_digest, "condition": driver.state(),
                  "input_digest": input_digest, "input_roots": [tensor_descriptor(value) for value in roots],
                  "input_values": [value.tolist() for value in roots],
                  "program_id": runtime.catalog.program_id, "trace_fingerprint": capture.trace["fingerprint"],
                  "laws": laws, "ports": {"skip": skip, "projection": proj, "carrier": carrier},
                  "projection_pc": proj_pc, "carrier_pc": carrier_pc, "carrier_producer": producer,
                  "baseline_heads": raw_heads(native_heads), "baseline_decode": baseline_decode,
                  "donor_values": raw_heads(replay_donor),
                  "donor_descriptors": {key: tensor_descriptor(value) for key, value in replay_donor.items()},
                  "donor_cut": donor_cut, "donor_exact": donor_exact,
                  "parent_cuts": {site: cut.manifest() for site, cut in cuts.items()},
                  "native_capture_control": capture_check, "native_replay_control": replay_check}
        save(folder / "case.json", record)
        case_records.append(record)

        def audit(site, arm):
            fresh = {}
            source_module = model.post[int(site.rsplit(".", 1)[1])]
            multiplier = 1. if arm == "identity" else 0.
            hooks = [source_module.register_forward_hook(
                lambda module, args, value: value * multiplier)]
            if arm == "projection_native":
                hooks.append(model.mixers[2].out_proj.register_forward_hook(
                    lambda module, args, value: donor["projection"].clone()))
            elif arm == "projection_sham":
                hooks.append(model.mixers[2].out_proj.register_forward_hook(
                    lambda module, args, value: value.clone()))
            elif arm == "skip_native":
                hooks.append(model.mixers[2].out_proj.register_forward_hook(
                    lambda module, args, value: fresh.__setitem__("projection", value.detach().clone())))
                hooks.append(model.mixers[2].register_forward_hook(
                    lambda module, args, value: donor["skip"] + fresh["projection"]))
            elif arm == "carrier_native":
                hooks.append(model.mixers[2].register_forward_hook(
                    lambda module, args, value: donor["carrier"].clone()))
            try:
                return native(driver)
            finally:
                for hook in hooks:
                    hook.remove()

        for arm in ARMS:
            current_branch = case_id + "/" + arm
            site = {"post1_zero": "post.1", "post2_zero": "post.2"}.get(arm, "post.0")
            try:
                candidate = runtime.fork(cuts[site])
                port = candidate.catalog.resolve("module:" + site + ":output")["value_id"]
                source_value = candidate.read(port)
                source_input = candidate.read(laws[site]["inputs"][0])
                source_payload = source_value * (1. if arm == "identity" else 0.)
                source_receipt = apply(candidate, port, source_payload)
                restoration = {"mode": arm, "source_site": site, "source_value_id": port,
                    "source_parent": cuts[site].fingerprint, "source_act": source_receipt["fingerprint"],
                    "source_before": tensor_descriptor(source_value),
                    "source_payload": tensor_descriptor(source_payload), "donor_case_id": case_id,
                    "donor_live_cut": donor_cut,
                    "donor_source": "same-context independent native hooks and live root-produced replay"}
                restoration_values = {}
                if arm in {"projection_native", "projection_sham"}:
                    advance(candidate, proj_pc)
                    before = candidate.read(proj)
                    payload = replay_donor["projection"] if arm == "projection_native" else before
                    restoration.update({"site": proj, "before": tensor_descriptor(before),
                                        "payload": tensor_descriptor(payload)})
                    restoration_values.update({"before": before.tolist(), "payload": payload.tolist()})
                    restoration["act"] = apply(candidate, proj, payload)["fingerprint"]
                    if arm == "projection_sham":
                        restoration["control_scope"] = "same-value SSA write; not an energy-matched hostile carrier"
                elif arm in {"skip_native", "carrier_native"}:
                    advance(candidate, carrier_pc)
                    before, edited_u = candidate.read(carrier), candidate.read(proj)
                    payload = replay_donor["skip"] + edited_u if arm == "skip_native" else replay_donor["carrier"]
                    restoration.update({"site": carrier, "before": tensor_descriptor(before),
                        "payload": tensor_descriptor(payload), "edited_projection": tensor_descriptor(edited_u),
                        "construction": "native skip + independently edited projection" if arm == "skip_native"
                                        else "native complete carrier; broad closure control"})
                    restoration_values.update({"before": before.tolist(), "payload": payload.tolist(),
                                               "edited_projection": edited_u.tolist()})
                    restoration["act"] = apply(candidate, carrier, payload)["fingerprint"]
                if site != "post.2" and candidate.pc <= carrier_pc:
                    advance(candidate, carrier_pc)
                    upstream = {"skip": candidate.read(skip), "projection": candidate.read(proj),
                                "carrier": candidate.read(carrier)}
                    restoration["retained_upstream_values"] = {
                        key: tensor_descriptor(value) for key, value in upstream.items()}
                    restoration_values["retained_upstream_values"] = raw_heads(upstream)
                advance(candidate, len(capture.trace["events"]))
                heads = candidate.outputs()
                decoded = projection(driver.decode(heads))
                independent = audit(site, arm)
                audit_decoded = projection(driver.decode(independent))
                audit_comparison = compare(heads, independent)
                row = {"case_id": case_id, "arm": arm, "forecast_sha256": forecast_digest,
                    "source_site": site, "input_digest": input_digest,
                    "program_id": runtime.catalog.program_id, "trace_fingerprint": capture.trace["fingerprint"],
                    "heads": raw_heads(heads), "decode": decoded,
                    "native_audit": audit_comparison, "native_audit_heads": raw_heads(independent),
                    "native_audit_decode": audit_decoded,
                    "parent_fingerprint": cuts[site].fingerprint, "restoration": restoration,
                    "restoration_values": restoration_values,
                    "source_values": {"input": source_input.tolist(), "output": source_value.tolist(),
                                      "payload": source_payload.tolist()},
                    "source_input_norm": float(torch.linalg.vector_norm(source_input)),
                    "source_output_norm": float(torch.linalg.vector_norm(source_value)),
                    "source_delta_norm": float(torch.linalg.vector_norm(source_payload - source_value)),
                    "baseline_comparison": compare(heads, native_heads),
                    "mechanics_status": "valid", "instrument_status": "native-audit-exact" if all(
                        value["exact"] for value in audit_comparison.values()) else "native-audit-differs",
                    "evidence_status": "observation", "semantic_name": None}

                def continue_branch(continuation):
                    successor = driver.fork()
                    continuation["before"] = successor.state()
                    committed = successor.commit(heads)
                    continuation["first_commit"] = committed
                    if committed["committed"] and not successor.termination:
                        successor_inputs = successor.inputs()
                        continuation["input_roots"] = [tensor_descriptor(value) for value in successor_inputs]
                        continuation["input_values"] = [value.tolist() for value in successor_inputs]
                        continuation["driver_state"] = successor.state()
                        with torch.no_grad():
                            second = model(*successor_inputs)
                        continuation["second_heads"] = raw_heads(second)
                        continuation["second_decode"] = projection(successor.decode(second))
                        continuation["status"] = "observed-own-predicted-state"
                    else:
                        continuation.update({"status": "not-applicable",
                            "reason": successor.termination or committed.get("stop_causes")})

                retain_primary_on_continuation_error(row, continue_branch)
                if row["continuation"]["status"] == "errored":
                    errors.append({"case_id": case_id, "arm": arm, "branch": current_branch,
                                   "kind": "continuation", "error": row["continuation"]["error"]})
                row["id"] = "variant:" + identity(row)
                followups.append(row)
                save(folder / "branches" / (arm + ".json"), row)
            except Exception as error:
                failure = {"case_id": case_id, "arm": arm, "branch": current_branch,
                           "kind": "mechanics", "error": f"{type(error).__name__}: {error}"}
                errors.append(failure)
                save(folder / "branches" / (arm + "-error.json"), failure)
                # Acts and replay are private to this discarded branch. A fresh
                # sibling remains usable; completed rows are retained unchanged.

    for case in plan["cases"]:
        try:
            measured("case_" + case["case_id"], lambda case=case: investigate(case))
        except Exception as error:
            failure = {"case_id": case["case_id"], "kind": "case-mechanics",
                       "error": f"{type(error).__name__}: {error}"}
            errors.append(failure)
            save(output / "cases" / case["case_id"] / "error.json", failure)
    save(output / "cases.json", case_records)
    save(output / "followups.json", {"rows": followups,
        "interpretation": "prospective generic route predictions; broad carrier restoration is closure, not minimality"})
    save(output / "errors.json", errors)
    with (output / "transition-receipts.jsonl").open("x") as stream:
        for row in receipts:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    save(output / "act-receipts.json", acts)
    weights_after = {name: tensor_descriptor(value)["sha256"] for name, value in model.state_dict().items()}
    if weights_before != weights_after:
        raise ValueError("shared frozen model changed during predictive campaign")
    count_hook.remove()
    save(output / "weight-identity.json", {"before": weights_before, "after": weights_after})
    report = {"schema": "mdb-predictive-route-campaign-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request_digest, "forecast_sha256": forecast_digest,
        "plan_sha256": plan["plan_sha256"],
        "checkpoint_sha256": CHECKPOINT, "forecast_binding": binding,
        "mechanics_status": "valid" if not errors else "partial-mechanics-errors",
        "instrument_status": "per-row-native-comparisons-retained", "trend_status": "prospective-observations",
        "terminal_status": "unestablished", "backend": "cpu-fp32", "cpu_threads": 1,
        "execution_mode": "scalar-native-capture-and-typed-replay", "physical_batching": False,
        "physical_model_forwards": physical_calls, "physical_forward_limit": config["max_physical_forwards"],
        "physical_geometry": physical_geometry, "case_count": len(case_records), "followup_rows": len(followups),
        "arms": list(ARMS), "replay_operations": replay_operations, "act_receipts": len(acts),
        "errors": errors, "model_source_hashes": model_sources, "phases": phases,
        "worker_seconds": time.perf_counter() - started, "oracle_state_used": False,
        "expected_states_used_as_execution_inputs": False,
        "supplied": ["program bytes and initial state", "generic sites and eight arms",
                     "sealed forecasts for later comparison", "same-context actual native donor values"],
        "file_manifest": {str(path.relative_to(output)): {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
                          for path in output.rglob("*") if path.is_file()}}
    save(output / "report.json", report)
    support()["emit"](output, report)


def stage(run_dir, isa_root, forecast):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    run_dir.mkdir(parents=True, exist_ok=False)
    raw = Path(forecast).read_bytes()
    plan = validate_plan(json.loads(raw))
    (run_dir / "forecast.json").write_bytes(raw)
    forecast_path = run_dir / "forecast.json"
    forecast_digest = sha(forecast_path.read_bytes())
    upstream = runpy.run_path(str(isa_root / "scripts/native_forward_symbols.py"))
    upstream["_worker_environment_defaults"]()
    sources, origins, _ = upstream["source_inventory"]()
    origins["forecast.json"] = {"origin": "pre-execution-frozen-prospective-plan",
                                "sha256": forecast_digest, "plan_sha256": plan["plan_sha256"]}
    packager = runpy.run_path(str(isa_root / "scripts/symbol_sources.py"))
    wheel = packager["build_isa_wheel"](run_dir, sources)
    helper = Path(__file__).with_name("discovery_upgrades.py")
    files = [(path, name) for name, path in sources.items()]
    files += [(wheel, wheel.name), (helper, "campaign_support.py"), (forecast_path, "forecast.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": ["predictive-route-result"],
        "hashed_roots": ["artifacts", "direct_effect", "isa_lab", wheel.name,
                         "predictive_route_campaign.py", "campaign_support.py", "forecast.json",
                         "source-manifest.json", "preflight.v1.json"]})
    files.append((preflight, preflight.name))
    config = {"schema": "mdb-predictive-route-config-v1", "checkpoint": "artifacts/model.pt",
              "forecast_sha256": forecast_digest, "max_physical_forwards": 128,
              "plan_sha256": plan["plan_sha256"],
              "max_replay_operations": 65536, "backend": "cpu-fp32", "cpu_threads": 1}
    request = {"schema": "mdb-predictive-route-request-v1", "task_family": "mdb-predictive-route-campaign",
        "model_identity": CHECKPOINT, "needs": {"cpu": True}, "config": config,
        "forecast_sha256": forecast_digest, "plan_sha256": plan["plan_sha256"], "execution_environment": intent(),
        "selection_policy": "all six forecast cases and all eight arms; no response-dependent pruning",
        "scheduling_contract": one_lease_scheduling_contract(logical_operations=128, logical_branches=48,
            logical_checkpoint_replays=96, max_local_work_items=128,
            extra={"max_replay_native_operations": 65536, "max_physical_model_forwards": 128}),
        "source_hashes": {name: sha(path.read_bytes()) for path, name in files}, "source_origins": origins}
    request["source_hashes"]["predictive_route_campaign.py"] = sha(Path(__file__).read_bytes())
    request["request_sha256"] = identity(request)
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=Path(__file__), config=config, request=request, additional_files=files)
    command = worker_command("predictive_route_campaign.py") + ["--worker"]
    index = command.index("python", command.index("uv") + 1)
    command[index:index] = ["--with", "./" + wheel.name]
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, experiment="mdb-predictive-route-campaign",
        task="r3-frozen-route-predictions-six-cases", model=None, payload=run_dir / "payload",
        prefer_host="beast", gpu=False, needs=request["needs"], ram_mb=2304, cpu_threads=1,
        timeout_s=300, retry_on_kill=False, preflight=True,
        note="One finite CPU1 FP32 R3 prospective route campaign: sealed six fixtures/eightarms before firstforward;6captures+6controls+48nativeaudits+48ownstatecontinuations=108<=128physical;65k replayprimitivecap;2304MiB measured prior1631.7MiB plus boundedcutbuffers. Rawinputs only; forecasts never model/commit inputs; no oracle/training/remote/retries; archiveonce.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir), "forecast_sha256": forecast_digest,
            "plan_sha256": plan["plan_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--isa-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--forecast", type=Path)
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.worker:
        try:
            worker()
        except Exception as error:
            output = Path("predictive-route-result")
            if output.is_dir():
                request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
                report = {"schema": "mdb-predictive-route-campaign-v1", "job_id": os.environ["MRUN_JOB_ID"],
                          "request_sha256": request["request_sha256"],
                          "forecast_sha256": request.get("forecast_sha256"), "mechanics_status": "errored",
                          "error": f"{type(error).__name__}: {error}"}
                save(output / "failure.json", report)
                support()["emit"](output, report)
            raise
    elif args.collect:
        print(json.dumps(support()["collect"](args.run_dir, args.collect), sort_keys=True))
    else:
        if args.forecast is None:
            parser.error("--forecast must name an already frozen prospective plan")
        print(json.dumps(stage(args.run_dir.resolve(), args.isa_root.resolve(), args.forecast), sort_keys=True))


if __name__ == "__main__":
    main()
