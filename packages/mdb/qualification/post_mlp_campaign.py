"""Matched three-post-MLP screen and final-mixer operand restoration via mrun.

Model execution imports installed public packages and sealed ISA implementation.
Generic names describe sites and experiments; no oracle or semantic resolver runs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import time
from pathlib import Path

CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"
SITES = ("post.0", "post.1", "post.2")
DOSES = (0., .5, 1., 1.5)
WORDS = (252, 253, 254, 255)
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


def worker():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    from isa_lab.native_forward_symbol_assay import load_r3_mamba_model

    from mdb.discovery_batch import (
        BatchProfile,
        BatchRow,
        FrozenNativeBatch,
        OutputAct,
        compare_screen_rows,
    )
    from mdb.discovery_cli import save_capture
    from mdb.discovery_isa import ISADriver
    from mdb.discovery_runtime import ReplayForward
    from mdb.discovery_storage import save_frontier
    from mdb.forward_recording import record_forward, tensor_descriptor

    torch.set_num_threads(1)
    started = time.perf_counter()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    output = Path("post-mlp-result")
    output.mkdir(exist_ok=False)
    if config.get("schema") != "mdb-post-mlp-config-v1" or request.get("model_identity") != CHECKPOINT:
        raise ValueError("campaign model/config binding differs")
    sealed_request = dict(request)
    request_digest = sealed_request.pop("request_sha256")
    if identity(sealed_request) != request_digest:
        raise ValueError("campaign request seal differs")
    for name, expected in request["source_hashes"].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError("staged input changed: " + name)
    installed_isa = importlib.metadata.distribution("isa-model-comparison")
    for name, expected in request["source_hashes"].items():
        if name.endswith(".py") and name.startswith(("isa_lab/", "direct_effect/")):
            if sha(installed_isa.locate_file(name).read_bytes()) != expected:
                raise ValueError("installed ISA differs: " + name)
    model, model_sources = load_r3_mamba_model(config["checkpoint"])
    model.eval().requires_grad_(False)
    weights_before = {name: tensor_descriptor(value)["sha256"] for name, value in model.state_dict().items()}
    physical_calls = 0
    physical_geometry = []
    def count_forward(module, args):
        nonlocal physical_calls
        if physical_calls >= config["max_physical_forwards"]:
            raise ValueError("campaign physical-forward allowance exhausted")
        physical_calls += 1
        physical_geometry.append({"attempt": physical_calls, "batch": int(args[0].shape[0])})
    count_hook = model.register_forward_pre_hook(count_forward)
    phases = []
    def measured(name, callback):
        before, begin = physical_calls, time.perf_counter()
        result = callback()
        phases.append({"phase": name, "seconds": time.perf_counter()-begin,
                       "physical_forwards": physical_calls-before})
        (output / "progress.json").write_text(json.dumps({"phases": phases, "physical_forwards": physical_calls}))
        print("MDB_POST_MLP_PROGRESS " + json.dumps(phases[-1]), flush=True)
        return result
    def native(driver):
        with torch.no_grad():
            return model(*driver.inputs())
    program = "4881c701000000" * 2
    drivers = [ISADriver(program, [0, 0, 0, 0, 0, 0, 3, word], max_instructions=2) for word in WORDS]
    profile = BatchProfile(CHECKPOINT, "r3-frozen-native-inputs-v1", ((1, 120), (1, 8, 64), (1, 4)),
        HEADS, allowed_modules=SITES, max_batch_size=4, atol=1e-4, rtol=1e-5,
        qualification="three-post-matched-exploration-with-full-scalar-controls")
    screen = FrozenNativeBatch(model, profile, max_physical_calls=64, max_logical_rows=128)
    screen_rows = []
    def screen_panel():
        for site in SITES:
            for dose in DOSES:
                requests = [BatchRow(f"context_{index:02d}", "input_00", f"{site}_dose_{dose}", 0,
                    driver.inputs(), acts=(OutputAct(site, "scale", dose),)) for index, driver in enumerate(drivers)]
                batched = screen.screen(requests)
                scalar = screen.screen(requests, batch_size=1)
                if batched.errors or scalar.errors or len(batched.rows) != 4 or len(scalar.rows) != 4:
                    raise ValueError("required native screen geometry or execution invalid")
                for index, (b, s, driver) in enumerate(zip(batched.rows, scalar.rows, drivers)):
                    comparison = compare_screen_rows(s, b, profile)
                    row = {"site": site, "context": f"context_{index:02d}", "input_rdi": WORDS[index],
                        "dose": dose, "condition": driver.state(), "binding": b.binding,
                        "roots_digest": b.roots_digest, "batch_attempt": b.attempt_id,
                        "scalar_attempt": s.attempt_id, "batch_row": b.physical_row,
                        "raw_heads": raw_heads(b.outputs), "scalar_heads": raw_heads(s.outputs),
                        "batch_decode": projection(driver.decode(b.outputs)),
                        "scalar_decode": projection(driver.decode(s.outputs)),
                        "numeric_comparison": comparison,
                        "minimum_sign_margins": {key: float(value.abs().min()) for key, value in s.outputs.items()
                            if key in {"regs", "flags", "imm", "store", "syscall", "store_addr", "store_value"}},
                        "length_top_two_gap": float(torch.topk(s.outputs["length"][0], 2).values.diff().abs()[0]),
                        "evidence_status": "observation", "semantic_name": None}
                    row["id"] = "variant:" + identity(row)
                    screen_rows.append(row)
    measured("matched_native_B4_and_B1_screen", screen_panel)
    save(output / "screen.json", {"rows": screen_rows, "attempts": screen.attempts,
        "profile": profile.manifest(), "work": screen.work,
        "selection_uses": "scalar native head responses, not expected answers"})

    def scalar_row(site, context, dose):
        return next(r for r in screen_rows if (r["site"], r["context"], r["dose"]) == (site, context, dose))
    selection_scores = []
    for index in range(4):
        context = f"context_{index:02d}"
        deltas = []
        for site in SITES[:2]:
            zero = torch.tensor(scalar_row(site, context, 0.)["scalar_heads"]["regs"])
            base = torch.tensor(scalar_row(site, context, 1.)["scalar_heads"]["regs"])
            deltas.append(zero-base)
        selection_scores.append({"context": context, "input_rdi": WORDS[index],
            "score": float(torch.linalg.vector_norm(deltas[0]-deltas[1]))})
    ranked = sorted(selection_scores, key=lambda row: (-row["score"], row["context"]))
    selected = ["context_03", ranked[0]["context"] if ranked[0]["context"] != "context_03" else "context_00"]
    selection = {"policy": request["selection_policy"], "scores": selection_scores, "contexts": selected,
                 "followup_dose": 0., "discovery_only": True}
    save(output / "selection.json", selection)

    replay_operations = 0
    receipt_rows = []
    act_rows = []
    followups = []
    context_records = []
    current_branch = "setup"
    def advance(runtime, pc):
        nonlocal replay_operations
        while runtime.pc < pc:
            if replay_operations >= config["max_replay_operations"]:
                raise ValueError("campaign replay primitive allowance exhausted")
            row = runtime.advance_one()
            replay_operations += 1
            receipt_rows.append({"branch": current_branch, "receipt": row["receipt"]})
    def apply(runtime, port, value):
        if not runtime.inspect()["values"][port]["writable"]:
            raise ValueError("selected activation is not writable at its producer boundary")
        receipt = runtime.apply(port, value)
        act_rows.append({"branch": current_branch, "receipt": receipt})
        return receipt
    def comparisons(left, right):
        return {key: {"exact": torch.equal(left[key], right[key]),
                      "max_abs": float((left[key]-right[key]).abs().max())} for key in HEADS}

    def investigate_context(context):
        nonlocal current_branch
        index = int(context.split("_")[-1])
        driver = drivers[index]
        roots = driver.inputs()
        input_digest = identity([tensor_descriptor(value) for value in roots])
        capture = record_forward(model, roots, model_identity=CHECKPOINT,
            input_identity=input_digest, execution={"backend": "cpu", "dtype": "float32", "threads": 1},
            max_operations=8192)
        runtime = ReplayForward(capture)
        catalog = runtime.catalog
        laws = {site: catalog.resolve("module:"+site) for site in SITES}
        proj = catalog.resolve("module:mixers.2.out_proj:output")["value_id"]
        carrier = catalog.resolve("module:post.2:input")["value_id"]
        carrier_event = capture.trace["events"][capture.trace["values"][carrier]["producer"]]
        if carrier_event["operator"] != "aten.add.Tensor" or proj not in carrier_event["inputs"]:
            raise ValueError("final mixer is not the declared skip-plus-projection topology")
        if carrier_event["arguments"].get("kwargs", {}).get("alpha", 1) != 1:
            raise ValueError("final-mixer add has nonunit alpha")
        other = [value for value in carrier_event["inputs"] if value != proj]
        if len(other) != 1:
            raise ValueError("final mixer skip operand is ambiguous")
        skip = other[0]
        proj_pc = capture.trace["values"][proj]["producer"] + 1
        carrier_pc = capture.trace["values"][carrier]["producer"] + 1
        if carrier_pc <= proj_pc:
            raise ValueError("carrier does not follow the projected update")
        donor = {}
        hooks = [model.mixers[2].register_forward_pre_hook(
                    lambda module, args: donor.__setitem__("skip", args[0].detach().clone())),
                 model.mixers[2].out_proj.register_forward_hook(
                    lambda module, args, value: donor.__setitem__("projection", value.detach().clone())),
                 model.mixers[2].register_forward_hook(
                    lambda module, args, value: donor.__setitem__("carrier", value.detach().clone()))]
        try:
            native_heads = native(driver)
        finally:
            for hook in hooks:
                hook.remove()
        capture_check = comparisons(capture.output, native_heads)
        if not all(row["exact"] for row in capture_check.values()):
            raise ValueError("required capture and independent native control differ")
        folder = output / "contexts" / context
        save(folder / "trace.json", capture.trace)
        save(folder / "native-heads.json", raw_heads(native_heads))
        save_capture(capture, folder / "root-recipe")
        current_branch = context+"/native_reference"
        cuts = {}
        for site in SITES:
            advance(runtime, laws[site]["exit"])
            cuts[site] = runtime.capture()
            if site == "post.0":
                durable = save_frontier(runtime, output / "selected-store", metadata={
                    "context": context, "live_parent_fingerprint": cuts[site].fingerprint, "site": site})
                save(folder / "durable-parent.json", durable)
        advance(runtime, len(capture.trace["events"]))
        replay_check = comparisons(runtime.outputs(), native_heads)
        if not all(row["exact"] for row in replay_check.values()):
            raise ValueError("required unchanged replay differs from independent native")
        # Obtain donor operands from actual native replay, not recorded future roots.
        native_carrier = runtime.fork(cuts["post.1"])
        current_branch = context+"/native_donor"
        advance(native_carrier, carrier_pc)
        donor_cut = native_carrier.capture()
        replay_donor = {"skip": native_carrier.read(skip), "projection": native_carrier.read(proj),
                        "carrier": native_carrier.read(carrier)}
        donor_check = {key: torch.equal(value, donor[key]) for key, value in replay_donor.items()}
        if not all(donor_check.values()):
            raise ValueError("replayed donor operands differ from independent native hooks")
        save(folder / "carrier-binding.json", {"program_id": catalog.program_id,
            "trace_fingerprint": capture.trace["fingerprint"], "skip": skip, "projection": proj,
            "carrier": carrier, "projection_pc": proj_pc, "carrier_pc": carrier_pc,
            "producer": carrier_event, "native_operand_exact": donor_check,
            "donor_cut": donor_cut.fingerprint,
            "donor_descriptors": {key: tensor_descriptor(value) for key, value in replay_donor.items()},
            "donor_values": raw_heads(replay_donor), "laws": laws})
        context_records.append({"context": context, "input_rdi": WORDS[index], "input_digest": input_digest,
            "program_id": catalog.program_id, "trace_fingerprint": capture.trace["fingerprint"],
            "native_capture_control": capture_check, "native_replay_control": replay_check})

        def audit(site, arm):
            fresh = {}
            source_module = model.post[int(site.split(".")[-1])]
            handles = [source_module.register_forward_hook(lambda module, args, value:
                value * (1. if arm == "identity" else 0.))]
            if arm == "projection_native":
                handles.append(model.mixers[2].out_proj.register_forward_hook(
                    lambda module, args, value: donor["projection"].clone()))
            elif arm == "projection_sham":
                handles.append(model.mixers[2].out_proj.register_forward_hook(
                    lambda module, args, value: value.clone()))
            elif arm == "skip_native":
                handles.append(model.mixers[2].out_proj.register_forward_hook(
                    lambda module, args, value: fresh.__setitem__("projection", value.detach().clone())))
                handles.append(model.mixers[2].register_forward_hook(
                    lambda module, args, value: donor["skip"] + fresh["projection"]))
            elif arm == "carrier_native":
                handles.append(model.mixers[2].register_forward_hook(
                    lambda module, args, value: donor["carrier"].clone()))
            try:
                return native(driver)
            finally:
                for handle in handles:
                    handle.remove()

        for site in SITES:
            port = catalog.resolve("module:"+site+":output")["value_id"]
            source_native = runtime.fork(cuts[site])
            source_value = source_native.read(port)
            activation_input = source_native.read(laws[site]["inputs"][0])
            arms = ("identity", "zero", "projection_native", "skip_native", "carrier_native", "projection_sham") if site != "post.2" else ("identity", "zero")
            for arm in arms:
                current_branch = context+"/"+site+"/"+arm
                candidate = runtime.fork(cuts[site])
                source_receipt = apply(candidate, port, source_value*(1. if arm == "identity" else 0.))
                restoration = {"mode": arm, "source_parent": cuts[site].fingerprint,
                    "source_value_id": port, "source_act": source_receipt["fingerprint"],
                    "donor_context": context, "donor_source": "independent unchanged native and live replay"}
                if arm in {"projection_native", "projection_sham"}:
                    advance(candidate, proj_pc)
                    original = candidate.read(proj)
                    payload = replay_donor["projection"] if arm == "projection_native" else original
                    restoration.update({"site": proj, "before": tensor_descriptor(original),
                        "payload": tensor_descriptor(payload), "donor_live_cut": donor_cut.fingerprint})
                    restoration["act"] = apply(candidate, proj, payload)["fingerprint"]
                elif arm in {"skip_native", "carrier_native"}:
                    advance(candidate, carrier_pc)
                    original = candidate.read(carrier)
                    altered_projection = candidate.read(proj)
                    payload = replay_donor["skip"]+altered_projection if arm == "skip_native" else replay_donor["carrier"]
                    restoration.update({"site": carrier, "before": tensor_descriptor(original),
                        "payload": tensor_descriptor(payload), "edited_projection": tensor_descriptor(altered_projection),
                        "construction": "native skip + edited projected update" if arm == "skip_native" else "native complete carrier",
                        "donor_live_cut": donor_cut.fingerprint})
                    restoration["act"] = apply(candidate, carrier, payload)["fingerprint"]
                if site != "post.2" and candidate.pc <= carrier_pc:
                    advance(candidate, carrier_pc)
                    restoration["retained_upstream_values"] = {"skip": tensor_descriptor(candidate.read(skip)),
                        "projection": tensor_descriptor(candidate.read(proj)), "carrier": tensor_descriptor(candidate.read(carrier))}
                advance(candidate, len(capture.trace["events"]))
                heads = candidate.outputs()
                independent = audit(site, arm)
                audit_result = comparisons(heads, independent)
                successor = driver.fork()
                before = successor.state()
                committed = successor.commit(heads)
                continuation = {"before": before, "first_commit": committed}
                if committed["committed"] and not successor.termination:
                    successor_inputs = successor.inputs()
                    continuation["input_roots"] = [tensor_descriptor(value) for value in successor_inputs]
                    with torch.no_grad():
                        second = model(*successor_inputs)
                    continuation.update({"second_heads": raw_heads(second),
                        "second_decode": projection(successor.decode(second)), "status": "observed-own-predicted-state"})
                else:
                    continuation.update({"status": "not-applicable", "reason": successor.termination or committed.get("stop_causes")})
                row = {"context": context, "source": site, "arm": arm, "input_digest": input_digest,
                    "program_id": catalog.program_id, "trace_fingerprint": capture.trace["fingerprint"],
                    "heads": raw_heads(heads), "decode": projection(driver.decode(heads)),
                    "native_audit": audit_result, "independent_native_heads": raw_heads(independent),
                    "independent_native_decode": projection(driver.decode(independent)),
                    "continuation": continuation,
                    "parent_fingerprint": cuts[site].fingerprint, "restoration": restoration,
                    "source_input_norm": float(torch.linalg.vector_norm(activation_input)),
                    "source_output_norm": float(torch.linalg.vector_norm(source_value)),
                    "source_delta_norm": 0. if arm == "identity" else float(torch.linalg.vector_norm(source_value)),
                    "evidence_status": "observation", "semantic_name": None}
                row["id"] = "variant:"+identity(row)
                followups.append(row)
                save(folder / "branches" / (site+"-"+arm+".json"), row)
    for context in selected:
        measured("scalar_restoration_"+context, lambda context=context: investigate_context(context))
    save(output / "followups.json", {"rows": followups, "interpretation":
        "final-mixer skip/projected-update restoration factorial; broad carrier restoration is closure, not minimality"})
    save(output / "contexts.json", context_records)
    with (output / "transition-receipts.jsonl").open("x") as stream:
        for row in receipt_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
    save(output / "act-receipts.json", act_rows)
    weights_after = {name: tensor_descriptor(value)["sha256"] for name, value in model.state_dict().items()}
    if weights_before != weights_after:
        raise ValueError("frozen model state changed during campaign")
    count_hook.remove()
    save(output / "weight-identity.json", {"before": weights_before, "after": weights_after})
    report = {"schema": "mdb-post-mlp-campaign-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "checkpoint_sha256": CHECKPOINT,
        "mechanics_status": "valid", "instrument_status": "per-row-native-comparisons-retained",
        "trend_status": "exploratory", "terminal_status": "unestablished",
        "backend": "cpu-fp32", "cpu_threads": 1, "physical_model_forwards": physical_calls,
        "physical_geometry": physical_geometry, "physical_forward_limit": config["max_physical_forwards"],
        "screen_rows": len(screen_rows), "followup_rows": len(followups),
        "replay_operations": replay_operations, "act_receipts": len(act_rows),
        "selection": selection, "contexts": context_records, "model_source_hashes": model_sources,
        "phases": phases, "worker_seconds": time.perf_counter()-started,
        "oracle_state_used": False, "supplied": ["byte program and before-state grid", "sites and doses",
            "final-mixer operand restoration geometry", "model-generated same-context native donors"],
        "file_manifest": {str(path.relative_to(output)): {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
                          for path in output.rglob("*") if path.is_file()}}
    save(output / "report.json", report)
    support()["emit"](output, report)


def stage(run_dir, isa_root):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    upstream = runpy.run_path(str(isa_root / "scripts/native_forward_symbols.py"))
    upstream["_worker_environment_defaults"]()
    run_dir.mkdir(parents=True, exist_ok=False)
    sources, origins, _ = upstream["source_inventory"]()
    packager = runpy.run_path(str(isa_root / "scripts/symbol_sources.py"))
    wheel = packager["build_isa_wheel"](run_dir, sources)
    helper = Path(__file__).with_name("discovery_upgrades.py")
    files = [(path, name) for name, path in sources.items()] + [(wheel, wheel.name), (helper, "campaign_support.py")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "path_token_allowlist": ["post-mlp-result"],
        "hashed_roots": ["artifacts", "direct_effect", "isa_lab", wheel.name, "post_mlp_campaign.py",
                         "campaign_support.py", "source-manifest.json", "preflight.v1.json"]})
    files.append((preflight, preflight.name))
    config = {"schema": "mdb-post-mlp-config-v1", "checkpoint": "artifacts/model.pt",
        "max_physical_forwards": 128, "max_replay_operations": 65536, "backend": "cpu-fp32", "cpu_threads": 1}
    request = {"schema": "mdb-post-mlp-request-v1", "task_family": "mdb-post-mlp-campaign",
        "model_identity": CHECKPOINT, "needs": {"cpu": True}, "config": config,
        "execution_environment": intent(),
        "selection_policy": "include context03/RDI255; add maximum L2 disagreement between post0 and post1 zero-minus-identity scalar regs; if maximum is context03 use context00/RDI252",
        "scheduling_contract": one_lease_scheduling_contract(logical_operations=128, logical_branches=28,
            logical_checkpoint_replays=40, extra={"max_replay_native_operations": 65536}),
        "source_hashes": {name: sha(path.read_bytes()) for path, name in files}, "source_origins": origins}
    request["source_hashes"]["post_mlp_campaign.py"] = sha(Path(__file__).read_bytes())
    request["request_sha256"] = identity(request)
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=Path(__file__), config=config, request=request, additional_files=files)
    command = worker_command("post_mlp_campaign.py") + ["--worker"]
    index = command.index("python", command.index("uv")+1)
    command[index:index] = ["--with", "./"+wheel.name]
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, experiment="mdb-post-mlp-campaign",
        task="r3-three-posts-and-final-mixer-restoration", model=None, payload=run_dir / "payload",
        prefer_host="beast", gpu=False, needs=request["needs"], ram_mb=2304, cpu_threads=1,
        timeout_s=300, retry_on_kill=False, preflight=True,
        note="One finite CPU1 R3 campaign:48B4 logicalrows+48B1 controls=60physical;2captures+2controls;28scalarreplaybranches eachfreshnativeaudit+ownstatecontinuation, expected120<=128physical. Finalmixer skip/write factorial,65kprimitivecap.2304MB measured prior1631.7MB+boundedcut buffers; no oracle/training/remote service/retries; archive once.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--isa-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.worker:
        try:
            worker()
        except Exception as error:
            output = Path("post-mlp-result")
            if output.is_dir():
                request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
                report = {"schema": "mdb-post-mlp-campaign-v1", "job_id": os.environ["MRUN_JOB_ID"],
                    "request_sha256": request["request_sha256"], "mechanics_status": "errored",
                    "error": f"{type(error).__name__}: {error}"}
                save(output / "failure.json", report)
                support()["emit"](output, report)
            raise
    elif args.collect:
        print(json.dumps(support()["collect"](args.run_dir, args.collect), sort_keys=True))
    else:
        print(json.dumps(stage(args.run_dir.resolve(), args.isa_root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
