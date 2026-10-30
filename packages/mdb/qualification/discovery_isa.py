"""One finite installed-wheel R3 discovery debugger demonstration via mrun.

Maintainer staging takes an explicit ISA source root. Worker imports resolve
installed distributions, never sibling checkouts. No oracle is imported.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPORT = "MDB_DISCOVERY_REPORT_V1 "
ARCHIVE = "MDB_DISCOVERY_ARCHIVE_V1 "
CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def worker():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import importlib.metadata

    import torch
    from isa_lab.native_forward_symbol_assay import load_r3_mamba_model

    from mdb.discovery import DiscoverySession
    from mdb.discovery_cli import save_capture
    from mdb.discovery_isa import ISADriver
    from mdb.discovery_runtime import ReplayForward
    from mdb.discovery_script import DiscoveryJournal, ScriptRunner
    from mdb.forward_recording import record_forward, tensor_descriptor

    torch.set_num_threads(1)
    started = time.perf_counter()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config.get("schema") != "mdb-discovery-qualification-config-v1":
        raise ValueError("wrong qualification config")
    for name, expected in request["source_hashes"].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError("staged source changed: " + name)
    distribution = importlib.metadata.distribution("isa-model-comparison")
    for name, expected in request["source_hashes"].items():
        if name.endswith(".py") and name.startswith(("isa_lab/", "direct_effect/")):
            if sha(distribution.locate_file(name).read_bytes()) != expected:
                raise ValueError("installed ISA source changed: " + name)
    model, source_hashes = load_r3_mamba_model(config["checkpoint"])
    output = Path("discovery-result")
    output.mkdir(exist_ok=False)
    session = DiscoverySession(budgets={"max_model_forwards": 32, "max_native_operations": 262144,
                                       "max_branches": 16, "max_commands": 512, "max_wall_ms": 180000})
    controls, captures, audits = [], [], []
    physical_forwards = 0
    journal = DiscoveryJournal(output / "journal")
    runner = ScriptRunner(session, journal=journal, max_commands=512)
    model_digest = {key: tensor_descriptor(value)["sha256"] for key, value in model.state_dict().items()}

    def admit_physical_forward():
        nonlocal physical_forwards
        if physical_forwards >= config["max_model_forwards"]:
            raise ValueError("qualification total physical-forward budget exhausted")
        physical_forwards += 1
        (output / "progress.json").write_text(json.dumps({"physical_forwards": physical_forwards,
                                                         "completed_captures": len(captures)}))

    def native_forward(inputs):
        admit_physical_forward()
        with torch.no_grad():
            return model(*inputs)

    def factory(*inputs):
        index = len(captures)
        identity = sha(json.dumps([tensor_descriptor(v) for v in inputs], sort_keys=True).encode())
        admit_physical_forward()
        capture = record_forward(model, inputs, model_identity=CHECKPOINT,
                                 execution={"backend": "cpu", "dtype": "float32", "threads": 1},
                                 input_identity=identity, metadata={"forward_call": index}, max_operations=8192)
        control = native_forward(inputs)
        exact = {key: torch.equal(capture.output[key], control[key]) for key in control}
        if len(exact) != 11 or not all(exact.values()):
            raise ValueError("native capture/control parity differs")
        controls.append(exact)
        captures.append({"input_identity": identity, "trace": capture.trace["fingerprint"],
                         "operations": len(capture.trace["events"]), "input_summaries": [tensor_descriptor(v) for v in inputs]})
        if index == 0:
            save_capture(capture, output / "root-recipe")
        return ReplayForward(capture)

    def commands(lines):
        result = runner.run("interface mdb-discovery-v1\n" + "\n".join(lines))
        if not result["ok"]:
            raise ValueError(result["last_response"])
        return result

    def call(operation, payload=None):
        result = commands(["call " + json.dumps({"operation": operation, "payload": payload or {}}, separators=(",", ":"))])
        records = [r for r in result["records"] if r.get("schema") == "mdb-discovery-command-record-v1"]
        return records[-1]["response"]

    program = "4881c701000000" * 2
    for index, word in enumerate((252, 253, 254, 255)):
        driver = ISADriver(program, [0, 0, 0, 0, 0, 0, 3, word], max_instructions=2)
        session.add_lane(f"lane_{index:02d}", factory, driver=driver,
                         reference_operations=8192, reference_forwards=1)
    commands(["alias mlp_focus = module:post.1", "break mlp_focus exit --id post1_exit"])
    stops = call("motion", {"action": "continue", "all": True, "stop_policy": "lane"})
    save(output / "four-lane-stops.json", stops)
    parked = session.lanes["lane_01"].branch.runtime.pc
    for lane in session.lanes.values():
        if lane.branch.runtime.pc != lane.branch.runtime.catalog.resolve("module:post.1")["exit"]:
            raise ValueError("lane did not stop at discovered MLP exit")
    call("lane", {"action": "select", "lane_id": "lane_00"})
    parent = call("capture", {"name": "mlp_parent"})
    runtime = session.lanes["lane_00"].branch.runtime
    law = runtime.catalog.resolve("module:post.1")
    endpoint = runtime.catalog.resolve("module:post.1:output")["value_id"]
    before = runtime.read(endpoint)
    call("fork", {"name": "zero_post1", "cut": "mlp_parent"})
    call("select", {"name": "zero_post1"})
    call("watch", {"action": "add", "id": "endpoint_changed", "selector": endpoint})
    call("act", {"selector": endpoint, "value": torch.zeros_like(before).tolist(), "sign": "replace"})
    watch_stop = call("motion", {"action": "step", "unit": "update"})
    call("watch", {"action": "disable", "id": "endpoint_changed"})
    call("motion", {"action": "step", "unit": "forward"})
    changed = session.lanes["lane_00"].branch.runtime.outputs()
    # Independent fresh eager execution with one module output hook, not replay.
    audit_inputs = session.lanes["lane_00"].branch.driver.inputs()
    hook = model.post[1].register_forward_hook(lambda module, args, result: torch.zeros_like(result))
    try:
        audited = native_forward(audit_inputs)
    finally:
        hook.remove()
    audit = {key: {"exact": torch.equal(changed[key], audited[key]),
                   "max_abs": float((changed[key] - audited[key]).abs().max())} for key in changed}
    if len(audit) != 11 or not all(row["exact"] for row in audit.values()):
        raise ValueError("edited replay differs from fresh native all-head audit")
    audits.append(audit)
    branch_decode = session.lanes["lane_00"].branch.driver.decode(changed)
    call("motion", {"action": "step", "unit": "instruction"})
    branch_state = session.lanes["lane_00"].branch.driver.state()
    # Restore architectural state and MLP frontier together; rerun a no-op path.
    call("select", {"name": "main"})
    call("restore", {"cut": "mlp_parent"})
    call("break", {"action": "disable", "id": "post1_exit"})
    update = call("motion", {"action": "step", "unit": "update"})
    update_pc = session.lanes["lane_00"].branch.runtime.pc
    if not update["completed_unit"] or update_pc != law["exit"] + 1:
        raise ValueError("update did not complete the actual residual transition")
    call("motion", {"action": "step", "unit": "forward"})
    native_replay = session.lanes["lane_00"].branch.runtime.outputs()
    native = native_forward(session.lanes["lane_00"].branch.driver.inputs())
    root_exact = {key: torch.equal(native_replay[key], native[key]) for key in native}
    branch_vs_native = {key: {"changed": not torch.equal(changed[key], native[key]),
                              "max_abs": float((changed[key] - native[key]).abs().max())} for key in native}
    if not all(root_exact.values()):
        raise ValueError("restored replay differs from fresh unchanged native output")
    count_before_commit = len(captures)
    call("motion", {"action": "step", "unit": "instruction"})
    if len(captures) != count_before_commit:
        raise ValueError("forward-complete instruction commit ran an extra model")
    committed_first = session.lanes["lane_00"].branch.driver.state()
    # A second instruction must receive only this model's own committed state.
    call("break", {"action": "disable", "id": "post1_exit"})
    call("motion", {"action": "step", "unit": "instruction"})
    committed_second = session.lanes["lane_00"].branch.driver.state()
    # Refill one stopped lane as a new immutable epoch, without discarding history.
    replacement = session.replace_input("lane_03", factory,
        driver=ISADriver(program, [0, 0, 0, 0, 0, 0, 3, 511], max_instructions=2),
        expected_revision=session.lanes["lane_03"].revision,
        reference_operations=8192, reference_forwards=1)
    journal.append({"schema": "mdb-discovery-input-replacement-v1", "response": replacement})
    if session.lanes["lane_01"].branch.runtime.pc != parked:
        raise ValueError("advancing/refilling another lane changed a parked lane")
    final_model_digest = {key: tensor_descriptor(value)["sha256"] for key, value in model.state_dict().items()}
    if final_model_digest != model_digest:
        raise ValueError("debugger changed frozen model weights")
    actions = output / "example.mdb"
    actions.write_text("""interface mdb-discovery-v1
alias mlp_focus = module:post.1
break mlp_focus exit --id post1_exit
on-hit post1_exit
    capture at_mlp
    inspect mlp_focus:output
    continue --defer
end
step forward
call {"operation":"break","payload":{"action":"disable","id":"post1_exit"}}
restore at_mlp
fork zero_post1
select zero_post1
call {"operation":"act","payload":{"selector":"mlp_focus:output","value_spec":{"kind":"zeros_like"}}}
step update
step forward
inspect outputs
""")
    cli_command = [sys.executable, "-I", "-m", "mdb", "discovery", "replay",
                   str(output / "root-recipe"), "--script", str(actions),
                   "--output", str(output / "cli-result")]
    cli = subprocess.run(cli_command, capture_output=True, text=True, timeout=60)
    save(output / "cli-invocation.json", {"argv": cli_command, "returncode": cli.returncode,
                                         "stdout": cli.stdout, "stderr": cli.stderr})
    if cli.returncode:
        raise ValueError("installed CLI example failed: " + cli.stderr[-2000:])
    cli_result = json.loads((output / "cli-result/result.json").read_text())
    save(output / "script-records.json", runner.records)
    save(output / "catalog.json", runtime.catalog.sites)
    report = {"schema": "mdb-discovery-qualification-v1", "job_id": os.environ["MRUN_JOB_ID"],
              "checkpoint_sha256": CHECKPOINT, "request_sha256": request["request_sha256"],
              "mechanics_status": "valid", "execution_mode": "record-replay",
              "backend": "cpu-fp32", "cpu_threads": 1, "physical_batching": False,
              "restoration": "live", "source_hashes": source_hashes,
              "model_weights_unchanged": True, "oracle_state_used": False,
              "law": law, "endpoint": endpoint, "update_pc": update_pc,
              "parent": parent, "watch_stop": watch_stop,
              "branch_decode": branch_decode, "branch_state": branch_state,
              "committed_first": committed_first, "committed_second": committed_second,
              "native_controls_all_heads": controls, "restored_native_all_heads": root_exact,
              "edited_fresh_native_all_heads": audits, "captures": captures,
              "branch_vs_native": branch_vs_native,
              "installed_cli": {"returncode": cli.returncode, "commands_executed": cli_result["commands_executed"],
                                "handler_hits": cli_result["handler_hits"], "work": cli_result["status"]["work"]},
              "work": session.work, "additional_untraced_controls": len(controls),
              "additional_full_forward_audits": 2,
              "total_physical_forwards": physical_forwards,
              "total_physical_forward_limit": config["max_model_forwards"],
              "commands_executed": runner.calls, "journal_records": len(journal.verify()),
              "worker_seconds": time.perf_counter() - started,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("mdb-runtime", "saturn-pub", "torch", "isa-model-comparison")},
              "limits": ["capture finishes before replay pause", "resident scalar, no physical batch",
                         "live cuts do not restore across process restart", "no RAM or syscall execution"]}
    save(output / "report.json", report)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.as_posix(), recursive=False)
    raw = buffer.getvalue()
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("qualification archive exceeds finite 32 MiB bound")
    report["archive_sha256"] = sha(raw)
    print(REPORT + base64.b64encode(gzip.compress(json.dumps(report).encode())).decode(), flush=True)
    print(ARCHIVE + base64.b64encode(raw).decode(), flush=True)


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
    config = {"schema": "mdb-discovery-qualification-config-v1", "checkpoint": "artifacts/model.pt",
              "backend": "cpu-fp32", "cpu_threads": 1, "max_model_forwards": 32}
    files = [(path, name) for name, path in sources.items()]
    files.append((wheel, wheel.name))
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
                     "path_token_allowlist": ["discovery-result"],
                     "hashed_roots": ["artifacts", "direct_effect", "isa_lab", wheel.name,
                                      "discovery_isa.py", "source-manifest.json", "preflight.v1.json"]})
    files.append((preflight, preflight.name))
    request = {"schema": "mdb-discovery-qualification-request-v1", "task_family": "mdb-discovery-debugger",
               "model_identity": CHECKPOINT, "needs": {"cpu": True}, "config": config,
               "execution_environment": intent(),
               "scheduling_contract": one_lease_scheduling_contract(
                   model_loads=1, logical_branches=16, logical_operations=64,
                   max_local_work_items=64, logical_checkpoint_replays=64,
                   checkpoint_policy="frozen CPU1 R3; capture-first resident scalar lanes; at most 32 captures plus 32 controls and two full audits; no training/retry"),
               "source_hashes": {name: sha(path.read_bytes()) for path, name in files},
               "source_origins": origins}
    request["source_hashes"]["discovery_isa.py"] = sha(Path(__file__).read_bytes())
    request["request_sha256"] = sha(json.dumps(request, sort_keys=True).encode())
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=Path(__file__), config=config, request=request, additional_files=files)
    command = worker_command("discovery_isa.py") + ["--worker"]
    index = command.index("python", command.index("uv") + 1)
    command[index:index] = ["--with", "./" + wheel.name]
    if "--offline" in command:
        command.remove("--offline")
    identifier = guarded_mrun_launch(command, request=request, experiment="mdb-discovery-debugger",
        task="r3-scalar-discovery-qualification", model=None, payload=run_dir / "payload",
        prefer_host="beast", gpu=False, needs={"cpu": True}, ram_mb=2048, cpu_threads=1,
        timeout_s=300, retry_on_kill=False, preflight=True,
        note="R3 debugger installed-wheel qualification, four resident scalar lanes, max 32 captures plus controls/two native audits; CPU FP32/thread1; 2 GiB measured prior four-parent exception (1242MB); no physical batch, oracle, training or retries. Stop after finite script and sealed archive.")
    save(run_dir / "submission.json", {"job_id": str(identifier)})
    return {"job_id": str(identifier), "run_dir": str(run_dir)}


def collect(run_dir, job_id):
    from mdb.job_support import decode_marker, terminal_job

    _, job, logs = terminal_job(job_id)
    if job["state"] not in {"succeeded", "failed", "cancelled", "timeout", "lost", "killed_ram", "killed_vram"}:
        return {"job_id": job_id, "state": job["state"]}
    save(run_dir / "job.json", job)
    (run_dir / "worker.log").write_text(logs)
    report = decode_marker(logs, REPORT)
    if report is None:
        return {"job_id": job_id, "state": job["state"], "report": False}
    request = json.loads((run_dir / "request.json").read_text())
    if report.get("job_id") != job_id or report.get("request_sha256") != request["request_sha256"]:
        raise ValueError("collected report has a different request/job binding")
    lines = [line[len(ARCHIVE):] for line in logs.splitlines() if line.startswith(ARCHIVE)]
    if len(lines) != 1:
        raise ValueError("expected exactly one result archive")
    raw = base64.b64decode(lines[0], validate=True)
    if len(raw) > 32 * 1024 * 1024 or sha(raw) != report["archive_sha256"]:
        raise ValueError("result archive identity differs")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if sum(member.size for member in members) > 128 * 1024 * 1024:
            raise ValueError("result expanded archive exceeds finite bound")
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError("unsafe result archive member")
            path = run_dir / "extracted" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as destination:
                destination.write(archive.extractfile(member).read())
    save(run_dir / "report.json", report)
    return {"job_id": job_id, "state": job["state"], "report": str(run_dir / "report.json")}


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
            output = Path("discovery-result")
            if output.is_dir():
                request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
                report = {"schema": "mdb-discovery-qualification-v1", "mechanics_status": "errored",
                          "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
                          "error": f"{type(error).__name__}: {error}"}
                save(output / "failure.json", report)
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                    for path in sorted(output.rglob("*")):
                        if path.is_file() and not path.is_symlink():
                            archive.add(path, arcname=path.as_posix(), recursive=False)
                raw = buffer.getvalue()
                if len(raw) <= 32 * 1024 * 1024:
                    report["archive_sha256"] = sha(raw)
                    print(REPORT + base64.b64encode(gzip.compress(json.dumps(report).encode())).decode(), flush=True)
                    print(ARCHIVE + base64.b64encode(raw).decode(), flush=True)
            raise
    elif args.collect:
        print(json.dumps(collect(args.run_dir, args.collect), sort_keys=True))
    else:
        print(json.dumps(stage(args.run_dir.resolve(), args.isa_root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
