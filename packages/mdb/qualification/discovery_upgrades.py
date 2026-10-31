"""Qualify native batches, durable cuts and remote discovery in one finite lease.

The maintainer supplies an explicit ISA source root for staging. Model execution
uses installed wheels, frozen raw-input R3, CPU FP32, and no oracle imports.
"""
from __future__ import annotations

import argparse
import base64
import cProfile
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import pstats
import runpy
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPORT = "MDB_DISCOVERY_UPGRADES_REPORT_V1 "
ARCHIVE = "MDB_DISCOVERY_UPGRADES_ARCHIVE_V1 "
CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def emit(output, report, *, max_compressed_bytes=64 * 1024 * 1024):
    if type(max_compressed_bytes) is not int or not 0 < max_compressed_bytes <= 128 * 1024 * 1024:
        raise ValueError("max_compressed_bytes must be a positive integer no greater than 128 MiB")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(output.rglob("*")):
            if path.is_symlink():
                raise ValueError("result archive contains a symbolic link")
            if path.is_file():
                archive.add(path, arcname=path.as_posix(), recursive=False)
    raw = buffer.getvalue()
    if len(raw) > max_compressed_bytes:
        raise ValueError("result archive exceeds finite compressed bound")
    report = {**report, "archive_sha256": sha(raw)}
    print(REPORT + base64.b64encode(gzip.compress(json.dumps(report).encode())).decode(), flush=True)
    print(ARCHIVE + base64.b64encode(raw).decode(), flush=True)


def worker():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    from isa_lab.native_forward_symbol_assay import load_r3_mamba_model

    from mdb.discovery import DiscoverySession
    from mdb.discovery_batch import BatchProfile, BatchRow, FrozenNativeBatch, OutputAct
    from mdb.discovery_isa import ISADriver
    from mdb.discovery_remote import attach_discovery_from_environment, serve_discovery
    from mdb.discovery_runtime import ReplayForward
    from mdb.discovery_storage import save_session_frontier
    from mdb.forward_recording import record_forward, tensor_descriptor

    torch.set_num_threads(1)
    started = time.perf_counter()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if os.environ.get("MRUN_SERVER_URL", "").rstrip("/") != request["scheduler_endpoint"]:
        raise ValueError("worker scheduler endpoint differs from sealed request")
    output = Path("upgrade-result")
    output.mkdir(exist_ok=False)
    for name, expected in request["source_hashes"].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError("staged input changed: " + name)
    distribution = importlib.metadata.distribution("isa-model-comparison")
    for name, expected in request["source_hashes"].items():
        if name.endswith(".py") and name.startswith(("isa_lab/", "direct_effect/")):
            if sha(distribution.locate_file(name).read_bytes()) != expected:
                raise ValueError("installed ISA implementation differs: " + name)
    model, source_hashes = load_r3_mamba_model(config["checkpoint"])
    model.requires_grad_(False)
    physical_calls = 0
    def count_forward(module, args):
        nonlocal physical_calls
        if physical_calls >= config["max_physical_forwards"]:
            raise ValueError("qualification physical-forward budget exhausted")
        physical_calls += 1
    counter = model.register_forward_pre_hook(count_forward)
    phases = []
    def measured(name, callback):
        begin, before = time.perf_counter(), physical_calls
        result = callback()
        phases.append({"phase": name, "seconds": time.perf_counter() - begin,
                       "physical_forwards": physical_calls - before})
        (output / "progress.json").write_text(json.dumps({"physical_forwards": physical_calls,
                                                          "completed_phases": phases}))
        return result
    program = "4881c701000000" * 2
    drivers = [ISADriver(program, [0, 0, 0, 0, 0, 0, 3, value], max_instructions=2)
               for value in (252, 253, 254, 255)]
    heads = {"regs": (1, 8, 64), "flags": (1, 4), "dest": (1, 8), "src": (1, 8),
             "width": (1, 3), "length": (1, 15), "syscall": (1,), "imm": (1, 64),
             "store": (1,), "store_addr": (1, 64), "store_value": (1, 64)}
    helper = runpy.run_path("discovery_batch.py")["qualify_native_batches"]
    batch = measured("native_batch_qualification", lambda: helper(model,
        model_identity=CHECKPOINT, program_identity="r3-frozen-native-inputs-v1", drivers=drivers,
        head_shapes=heads, repetitions=2, max_physical_calls=32,
        replacement_driver=ISADriver(program, [0, 0, 0, 0, 0, 0, 3, 511], max_instructions=2)))
    save(output / "batch.json", batch)
    profile = BatchProfile(CHECKPOINT, "r3-frozen-native-inputs-v1",
        ((1, 120), (1, 8, 64), (1, 4)), heads, allowed_modules=("post.1",),
        max_batch_size=4, qualification="exploratory-screen-with-scalar-controls", atol=1e-4, rtol=1e-5)
    screen = FrozenNativeBatch(model, profile, max_physical_calls=16, max_logical_rows=64)
    variants = []
    def screen_variants():
        for dose in (0., .5, 1., 1.5):
            rows = [BatchRow(f"context_{index:02d}", "input_00", f"dose_{dose}", 0,
                            driver.inputs(), acts=(OutputAct("post.1", "scale", dose),))
                    for index, driver in enumerate(drivers)]
            result = screen.screen(rows)
            if result.errors:
                raise ValueError(result.errors)
            for row, driver in zip(result.rows, drivers):
                decoded = driver.decode(row.outputs)
                variants.append({"schema": "mdb-computation-variant-v1", "evidence_status": "observation",
                    "site": "module:post.1", "condition": driver.state(), "dose": dose,
                    "binding": row.binding, "roots_digest": row.roots_digest,
                    "physical_attempt": row.attempt_id, "physical_row": row.physical_row,
                    "heads": {key: tensor_descriptor(value) for key, value in row.outputs.items()},
                    "raw_heads": decoded["raw_logits"],
                    "predicted_register_words": decoded["register_words"], "flags": decoded["flags_CF_ZF_SF_OF"],
                    "predicted_length": decoded["length"], "store": decoded["store"], "syscall": decoded["syscall"],
                    "minimum_sign_margins": {key: float(value.abs().min()) for key, value in row.outputs.items()
                        if key in {"regs", "flags", "imm", "store", "syscall", "store_addr", "store_value"}},
                    "length_top_two_gap": float(torch.topk(row.outputs["length"][0], 2).values.diff().abs()[0]),
                    "proposed_semantic_name": None, "terminal_status": "unestablished"})
    measured("sixteen_variant_screen", screen_variants)
    save(output / "variants.json", {"rows": variants, "work": screen.work, "attempts": screen.attempts,
                                   "interpretation": "four before-states crossed with four supplied MLP doses"})

    captures = []
    session = DiscoverySession({"max_model_forwards": 8, "max_native_operations": 131072,
                                "max_commands": 128, "max_wall_ms": 180000})
    def factory(*roots):
        capture = record_forward(model, roots, model_identity=CHECKPOINT,
            execution={"backend": "cpu", "dtype": "float32", "threads": 1},
            input_identity=sha(json.dumps([tensor_descriptor(v) for v in roots], sort_keys=True).encode()),
            max_operations=8192)
        captures.append(capture)
        if len(captures) == 1:
            save(output / "native-trace.json", capture.trace)
        return ReplayForward(capture)
    profiler = cProfile.Profile()
    profiler.enable()
    measured("capture_and_create_lane", lambda: session.add_lane("research", factory,
             driver=drivers[0], reference_operations=8192, reference_forwards=1))
    stop = measured("replay_to_post1_exit", lambda: session.call("motion",
        {"action": "until", "selector": "module:post.1", "edge": "exit"}))
    law = session.lanes["research"].branch.runtime.catalog.resolve("module:post.1")
    save(output / "post1-stop.json", {"law": law, "response": stop})
    if not stop["ok"] or stop["lane"]["pc"] != law["exit"]:
        raise ValueError("failed to reach actual R3 MLP frontier")
    saved = measured("durable_save", lambda: save_session_frontier(session, "research", output / "store"))
    profiler.disable()
    profiler.dump_stats(str(output / "discovery.pstats"))
    stats = pstats.Stats(profiler)
    costs = [{"file": key[0], "line": key[1], "function": key[2],
              "primitive_calls": value[0], "calls": value[1], "self_seconds": value[2],
              "cumulative_seconds": value[3]} for key, value in stats.stats.items()]
    save(output / "profile.json", {"timing_scope": "profiled capture/create/replay/save; profiler overhead included",
                                   "total_seconds": stats.total_tt,
                                   "functions": sorted(costs, key=lambda row: row["self_seconds"], reverse=True)})
    save(output / "saved-frontier.json", saved)
    actions = output / "resume.mdb"
    actions.write_text("interface mdb-discovery-v1\nstatus\ninspect state\nstep forward\ninspect outputs\nstep instruction\n")
    command = [sys.executable, "-I", "-m", "mdb", "discovery", "resume", str(output / "store"),
               saved["cut_id"], "--script", str(actions), "--output", str(output / "resumed")]
    child = measured("fresh_process_resume_and_suffix", lambda: subprocess.run(command,
                     capture_output=True, text=True, timeout=60))
    save(output / "resume-invocation.json", {"returncode": child.returncode,
                                           "stdout": child.stdout, "stderr": child.stderr})
    if child.returncode:
        raise ValueError("durable CLI resume failed: " + child.stderr[-2000:])
    resumed = json.loads((output / "resumed/result.json").read_text())
    commands = [row for row in resumed["records"] if row.get("schema") == "mdb-discovery-command-record-v1"]
    initial_work = commands[0]["response"]["work"]
    resumed_outputs = next(row["response"]["value"] for row in commands if row.get("command") == "inspect outputs")
    native = captures[0].output
    exact = {key: resumed_outputs[key]["value"] == value.tolist() for key, value in native.items()}
    if len(exact) != 11 or not all(exact.values()):
        raise ValueError("fresh-process suffix does not match native all-head capture")
    resume_delta = resumed["status"]["work"]["replay_native_operations"] - initial_work["replay_native_operations"]
    if resume_delta != len(captures[0].trace["events"]) - law["exit"]:
        raise ValueError("resumed work includes a replayed prefix or missing suffix")
    # A different native execution, outside the replay interpreter, checks that
    # the original capture still matches the unmodified model at the same roots.
    with torch.no_grad():
        unchanged = model(*drivers[0].inputs())
    capture_control = {key: torch.equal(unchanged[key], native[key]) for key in heads}
    if not all(capture_control.values()):
        raise ValueError("native reference capture/control differs")

    worker_control, poller = attach_discovery_from_environment(session)
    print("MDB_DISCOVERY_UPGRADES_READY " + json.dumps({"job_id": os.environ["MRUN_JOB_ID"],
        "pc": session.lanes["research"].branch.runtime.pc, "profile": "record-replay"}), flush=True)
    residency = measured("remote_residency_including_client_wait", lambda: serve_discovery(worker_control, poller,
        max_commands=96, max_wall_ms=90000, scheduling_contract=request["scheduling_contract"]))
    if residency["reason"] != "closed":
        raise ValueError("remote qualification did not close within its finite service allowance")
    branch = session.lanes["research"].branch
    actual = branch.runtime.outputs()
    hook = model.post[1].register_forward_hook(lambda module, args, value: torch.zeros_like(value))
    try:
        with torch.no_grad():
            audited = model(*drivers[0].inputs())
    finally:
        hook.remove()
    remote_audit = {key: {"exact": torch.equal(actual[key], audited[key]),
                           "max_abs": float((actual[key] - audited[key]).abs().max())} for key in heads}
    if not all(row["exact"] for row in remote_audit.values()):
        raise ValueError("remote intervention differs from independent native audit")
    counter.remove()
    save(output / "remote-events.json", session.events)
    report = {"schema": "mdb-discovery-upgrades-qualification-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "checkpoint_sha256": CHECKPOINT,
        "mechanics_status": "valid", "backend": "cpu-fp32", "cpu_threads": 1,
        "physical_model_forwards": physical_calls, "physical_forward_limit": config["max_physical_forwards"],
        "batch": {"physical_batching": True, "work": batch["adapter_work"], "rounds": batch["rounds"],
                  "instrument_status": batch["instrument_status"]}, "variant_rows": len(variants),
        "durable": {"saved": saved, "resumed_initial_pc": law["exit"], "law": law, "prefix_replayed": False,
                    "executed_suffix_primitives": resume_delta, "all_native_heads_exact": exact,
                    "capture_control_exact": capture_control, "final_driver": resumed["status"]["lanes"][0]["driver"]},
        "remote": {"residency": residency, "independent_native_audit": remote_audit},
        "phases": phases, "worker_seconds": time.perf_counter() - started, "source_hashes": source_hashes,
        "oracle_state_used": False, "scientific_status": "exploratory observations; no semantic naming claim"}
    save(output / "report.json", report)
    emit(output, report)


def stage(run_dir, isa_root):
    from mdb.discovery_remote import bounded_debugger_scheduling_contract
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import stage_worker, worker_command
    from mdb.tools import scheduler_api

    upstream = runpy.run_path(str(isa_root / "scripts/native_forward_symbols.py"))
    upstream["_worker_environment_defaults"]()
    run_dir.mkdir(parents=True, exist_ok=False)
    sources, origins, _ = upstream["source_inventory"]()
    packager = runpy.run_path(str(isa_root / "scripts/symbol_sources.py"))
    wheel = packager["build_isa_wheel"](run_dir, sources)
    config = {"schema": "mdb-discovery-upgrades-config-v1", "checkpoint": "artifacts/model.pt",
              "max_physical_forwards": 64, "backend": "cpu-fp32", "cpu_threads": 1}
    files = [(path, name) for name, path in sources.items()]
    files += [(wheel, wheel.name), (Path(__file__).with_name("discovery_batch.py"), "discovery_batch.py")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "path_token_allowlist": ["upgrade-result"],
                     "hashed_roots": ["artifacts", "direct_effect", "isa_lab", wheel.name,
                                      "discovery_upgrades.py", "discovery_batch.py", "source-manifest.json", "preflight.v1.json"]})
    files.append((preflight, preflight.name))
    request = {"schema": "mdb-discovery-upgrades-request-v1", "task_family": "mdb-discovery-upgrades",
               "model_identity": CHECKPOINT, "needs": {"cpu": True, "saturn_debug_credential_v1": True},
               "scheduler_endpoint": scheduler_api()._resolve().rstrip("/"),
               "config": config, "execution_environment": intent(),
               "scheduling_contract": bounded_debugger_scheduling_contract(max_debug_commands=128,
                   max_wall_ms=90000, max_native_operations=131072, max_model_forwards=64, max_branches=16),
               "source_hashes": {name: sha(path.read_bytes()) for path, name in files}, "source_origins": origins}
    request["source_hashes"]["discovery_upgrades.py"] = sha(Path(__file__).read_bytes())
    request["request_sha256"] = sha(json.dumps(request, sort_keys=True).encode())
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=Path(__file__), config=config, request=request, additional_files=files)
    command = worker_command("discovery_upgrades.py") + ["--worker"]
    # The deployed agent supports scoped credentials but predates public URL
    # injection. Seal the endpoint; leave credential injection to mrun.
    command.insert(1, "MRUN_SERVER_URL=" + request["scheduler_endpoint"])
    index = command.index("python", command.index("uv") + 1)
    command[index:index] = ["--with", "./" + wheel.name]
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, experiment="mdb-discovery-upgrades",
        task="r3-batches-durable-remote-profile", model=None, payload=run_dir / "payload", prefer_host="beast",
        gpu=False, needs=request["needs"], ram_mb=2304, cpu_threads=1, timeout_s=300, retry_on_kill=False,
        preflight=True, note="One finite CPU1 R3 debugger qualification: native B4 screens/control/audit<=64 physical forwards;16 observed context/dose rows;CAS paused frontier freshprocess;scoped authenticated remote<=128commands/90s;profilecopy/hash/replay.2304MB above measured1744.5MB parent plus storage/batch buffers; no oracle/training/retries; stopafterclose/archive.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


def collect(run_dir, job_id, *, max_expanded_bytes=256 * 1024 * 1024,
            max_compressed_bytes=64 * 1024 * 1024):
    if type(max_expanded_bytes) is not int or not 0 < max_expanded_bytes <= 1024 * 1024 * 1024:
        raise ValueError("max_expanded_bytes must be a positive integer no greater than 1 GiB")
    if type(max_compressed_bytes) is not int or not 0 < max_compressed_bytes <= 128 * 1024 * 1024:
        raise ValueError("max_compressed_bytes must be a positive integer no greater than 128 MiB")
    from mdb.job_support import decode_marker, terminal_job

    _, job, logs = terminal_job(job_id)
    if job["state"] not in {"succeeded", "failed", "timeout", "cancelled", "lost", "killed_ram", "killed_vram"}:
        return {"job_id": job_id, "state": job["state"], "ready": "MDB_DISCOVERY_UPGRADES_READY " in logs}
    save(run_dir / "job.json", job)
    (run_dir / "worker.log").write_text(logs)
    report = decode_marker(logs, REPORT)
    if report is None:
        return {"job_id": job_id, "state": job["state"], "report": False}
    request = json.loads((run_dir / "request.json").read_text())
    if report["job_id"] != job_id or report["request_sha256"] != request["request_sha256"]:
        raise ValueError("report job/request binding differs")
    lines = [line[len(ARCHIVE):] for line in logs.splitlines() if line.startswith(ARCHIVE)]
    if len(lines) != 1:
        raise ValueError("expected one output archive")
    raw = base64.b64decode(lines[0], validate=True)
    if len(raw) > max_compressed_bytes or sha(raw) != report["archive_sha256"]:
        raise ValueError("archive bytes differ")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 4096 or sum(item.size for item in members) > max_expanded_bytes:
            raise ValueError("archive geometry exceeds finite bound")
        for item in members:
            relative = Path(item.name)
            if not item.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe output archive member")
            path = run_dir / "extracted" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as destination:
                destination.write(archive.extractfile(item).read())
    save(run_dir / "report.json", report)
    return {"job_id": job_id, "state": job["state"], "report": str(run_dir / "report.json")}


def drive(job_id, output, endpoint):
    from mdb.discovery_remote import RemoteDiscoverySession

    output.mkdir(parents=True, exist_ok=False)
    client = RemoteDiscoverySession.connect(job_id, endpoint=endpoint, timeout=20)
    status = client.call("status")
    parent = client.call("checkpoint", {"action": "capture", "name": "retry_parent"})
    first = client.call("motion", {"unit": "op"}, request_id="qualification-one-native-op")
    repeat = client.call("motion", {"unit": "op"}, request_id="qualification-one-native-op")
    try:
        conflict = client.call("motion", {"unit": "law"}, request_id="qualification-one-native-op")
    except Exception as error:
        from mdb.debugger_protocol import DebuggerProtocolError

        if not isinstance(error, DebuggerProtocolError) or "different command content" not in str(error):
            raise
        conflict = {"ok": False, "error": str(error)}
    if first != repeat or conflict.get("ok") is not False:
        raise ValueError("remote at-most-once/content-conflict proof failed")
    restored = client.call("checkpoint", {"action": "restore", "cut": parent["cut"]})
    save(output / "retry-proof.json", {"initial": status, "first": first, "repeat": repeat,
                                      "changed_content": conflict, "restored": restored})
    actions = output / "remote.mdb"
    actions.write_text("""interface mdb-discovery-v1
status
alias mlp_focus = module:post.1
capture remote_parent
fork remote_zero
select remote_zero
call {"operation":"act","payload":{"selector":"mlp_focus:output","value_spec":{"kind":"zeros_like"}}}
step update
step forward
inspect outputs
step instruction
call {"operation":"close"}
""")
    command = [sys.executable, "-I", "-m", "mdb", "discovery", "attach", job_id,
               "--endpoint", endpoint, "--script", str(actions), "--output", str(output / "cli")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    save(output / "invocation.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode:
        raise ValueError("remote CLI failed: " + result.stderr[-2000:])
    return {"job_id": job_id, "ok": True, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--isa-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--collect")
    parser.add_argument("--drive")
    parser.add_argument("--endpoint", default="http://192.168.68.68:9025")
    args = parser.parse_args()
    if args.worker:
        try:
            worker()
        except Exception as error:
            output = Path("upgrade-result")
            if output.is_dir():
                request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
                report = {"schema": "mdb-discovery-upgrades-qualification-v1", "mechanics_status": "errored",
                    "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
                    "error": f"{type(error).__name__}: {error}"}
                save(output / "failure.json", report)
                emit(output, report)
            raise
    elif args.collect:
        print(json.dumps(collect(args.run_dir, args.collect), sort_keys=True))
    elif args.drive:
        print(json.dumps(drive(args.drive, args.run_dir, args.endpoint), sort_keys=True))
    else:
        print(json.dumps(stage(args.run_dir.resolve(), args.isa_root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
