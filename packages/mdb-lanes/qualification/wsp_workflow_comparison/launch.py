"""Freeze, run and collect a paired experimental-editor workflow comparison."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import runpy
import secrets
import shutil
import signal
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from protocol import ARMS, COMMON, DOCS, digest, editor_schema, save

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_editor_process(cmd, prompt, directory, timeout):
    """Keep partial output and stop this call's complete process group on timeout."""
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=directory, start_new_session=True) as process:
        timed_out = False
        try:
            stdout, stderr = process.communicate(prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate(timeout=10)
        return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr), timed_out


def initialize(base):
    base.mkdir(parents=True, exist_ok=False)
    kinds = ["amplification", "sign", "healthy"]
    random.SystemRandom().shuffle(kinds)
    specimens = [{"id": "specimen-" + secrets.token_hex(3), "seed": secrets.randbits(31), "kind": kind} for kind in kinds]
    plan = {"schema": "mdb-paired-workflow-pilot-v1", "specimens": specimens,
        "arms": list(ARMS), "rounds": 2, "calls_per_round": 6, "program_seconds": 60,
        "editor": {"model": "gpt-6-astra", "reasoning_effort": "high", "call_timeout_seconds": 180,
                   "parallel_inference_calls": 3, "provider_random_seed": "not exposed by CLI; fresh calls, not fixed-seed replication"},
        "selection": "largest observed discovery objective, native on ties; no evaluation-dependent edits",
        "output_scratch": "/mnt/hdd8tb/mdb/workflow-comparison",
        "limits": {"max_analysis_code_chars": 18000, "max_prompt_bytes": 64000},
        "source_hashes": {p.name: sha(p) for p in HERE.glob("*.py")},
        "scope": "small constructed, untrained, paired calibration; no pretrained or general superiority claim",
        "terminal_status": "not-evaluated"}
    plan["identity"] = digest(plan)
    save(base / "plan.json", plan)
    save(base / "editor-schema.json", editor_schema())
    return {"base": str(base), "plan_identity": plan["identity"], "editor": plan["editor"], "specimens": len(specimens)}


def extracted(base, name):
    root = base / name / "extracted/workflow-result"
    report = json.loads((base / name / "report.json").read_text())
    if sha(root / "files.json") != report["files_manifest_sha256"]:
        raise ValueError("collected source inventory pin differs")
    for filename, fingerprint in json.loads((root / "files.json").read_text()).items():
        if sha(root / filename) != fingerprint:
            raise ValueError("collected source file differs")
    return root


def verified_proposals(base, round_number, source):
    root = base / ("editor-round-" + str(round_number))
    path = root / "proposals.json"
    request = json.loads((source / "request.json").read_text())
    if sha(path) != request["source_hashes"]["proposals.json"]:
        raise ValueError("editor proposals differ from the executed request")
    rows = json.loads(path.read_text())
    for row in rows:
        directory = root / row["id"]
        receipt = row["editor_receipt"]
        if receipt.get("inference_status") == "timeout":
            for filename, field in (("prompt.txt", "prompt_sha256"), ("command.json", "command_sha256"),
                                    ("controller-fallback.json", "fallback_sha256"), ("failure.json", "failure_sha256")):
                if sha(directory / filename) != receipt[field]:
                    raise ValueError("timed-out editor custody differs")
            if json.loads((directory / "controller-fallback.json").read_text()) != row["proposal"] or receipt["usage"]:
                raise ValueError("timed-out editor must retain an explicit controller fallback and unknown usage")
            continue
        for filename, field in (("prompt.txt", "prompt_sha256"), ("answer.json", "answer_sha256"), ("events.jsonl", "events_sha256")):
            if sha(directory / filename) != receipt[field]:
                raise ValueError("editor source bytes differ from their executed receipt")
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines() if line.startswith("{")]
        usage = [event["usage"] for event in events if event.get("type") == "turn.completed"]
        if usage != receipt["usage"] or json.loads((directory / "answer.json").read_text()) != row["proposal"]:
            raise ValueError("editor usage or numerical program differs")
    return rows


def stage(base, name, phase, round_number=0):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    plan = json.loads((base / "plan.json").read_text())
    run = base / name
    run.mkdir(parents=True, exist_ok=False)
    payload = run / "payload"
    payload.mkdir()
    config = {**plan, "phase": phase, "round": round_number}
    files = {"worker.py": HERE / "worker.py", "protocol.py": HERE / "protocol.py",
             "flux_family_custody.py": HERE.parent / "flux_family_custody.py"}
    if phase == "search":
        files["proposals.json"] = base / ("editor-round-" + str(round_number)) / "proposals.json"
        history = {}
        if round_number == 2:
            prior = extracted(base, "search-round-1")
            for row in json.loads((prior / "public-feedback.json").read_text()):
                history[row["trial"]] = [row]
        save(payload / "history.json", history)
        files["history.json"] = payload / "history.json"
    if phase == "evaluate":
        packets = json.loads((extracted(base, "prepare") / "public-packets.json").read_text())
        candidates = {}
        source_pins = {}
        for r in (1, 2):
            source = extracted(base, "search-round-" + str(r))
            verified_proposals(base, r, source)
            source_summary = json.loads((source / "summary.json").read_text())
            source_pins[str(r)] = {"files_sha256": sha(source / "files.json"),
                "job_id": source_summary["job_id"], "request_sha256": source_summary["request_sha256"],
                "proposals_sha256": sha(base / ("editor-round-" + str(r)) / "proposals.json")}
            for row in json.loads((source / "public-feedback.json").read_text()):
                if not row["parameter_drift"]:
                    candidates.setdefault(row["trial"], []).extend(row["probes"])
        selections = []
        for specimen in plan["specimens"]:
            for arm in ARMS:
                tid = specimen["id"] + "-" + arm
                native = {"edits": [], "result": packets[specimen["id"]]["baseline"]}
                best = max([native, *candidates.get(tid, [])], key=lambda row: row["result"]["objective"])
                selections.append({"id": tid, "arm": arm, "specimen": specimen, "selection": best["edits"],
                    "discovery_objective": best["result"]["objective"], "selection_rule": plan["selection"],
                    "discovery_sources": source_pins, "plan_identity": plan["identity"]})
        save(run / "selections.json", selections)
        files["selections.json"] = run / "selections.json"
    if phase == "audit":
        bundle = {"plan": plan,
            "packets": json.loads((extracted(base, "prepare") / "public-packets.json").read_text()),
            "selections": json.loads((base / "evaluate/selections.json").read_text()),
            "evaluation": json.loads((extracted(base, "evaluate") / "evaluation.json").read_text()),
            "discovery": [], "proposals": [], "source_pins": {}}
        for r in (1, 2):
            source = extracted(base, "search-round-" + str(r))
            bundle["discovery"].extend(json.loads((source / "public-feedback.json").read_text()))
            proposals = base / ("editor-round-" + str(r)) / "proposals.json"
            bundle["proposals"].extend(verified_proposals(base, r, source))
            bundle["source_pins"]["round-" + str(r)] = {"files_sha256": sha(source / "files.json"), "proposals_sha256": sha(proposals)}
        bundle["source_pins"]["evaluation_files_sha256"] = sha(extracted(base, "evaluate") / "files.json")
        save(payload / "audit-input.json", bundle)
        files["audit-input.json"] = payload / "audit-input.json"
        files["audit_report.py"] = HERE / "audit_report.py"
    save(payload / "preflight.v1.json", {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": [config["output_scratch"]], "hashed_roots": [*files, "preflight.v1.json"]})
    files["preflight.v1.json"] = payload / "preflight.v1.json"
    wheel_paths = runpy.run_path(str(HERE.parent / "wsp_case_study/launch.py"))["wheels"]()
    wheel_map = {p.name: p for p in wheel_paths}
    request = {"schema": "mdb-workflow-comparison-request-v1", "config": config,
        "execution_environment": {"python": "3.14.6", "uv_link_mode": "symlink", "uv_cache_dir": "/mnt/hdd8tb/mdb/uv-cache", "extra_model_roots": "/mnt/big/llm-models",
            "torch_home": "/mnt/hdd8tb/mdb/torch", "requirements": ["torch==2.13.0", "numpy==2.5.4", "pyyaml==6.0.3", "psutil==7.2.2"]},
        "source_hashes": {n: sha(p) for n, p in files.items()}, "runtime_wheels": {n: sha(p) for n, p in wheel_map.items()},
        "task_family": "analysis", "backend": "hf", "device": "cpu", "dtype": "float64", "needs": {"cpu": True},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_branches=9,
            logical_operations=128, logical_checkpoint_replays=0, max_local_work_items=256,
            extra={"task_family": "analysis", "constructed_model": True, "pretrained_model_loads": 0})}
    request["request_sha256"] = digest(request)
    save(run / "request.json", request)
    shutil.copyfile(__file__, run / "frozen-launcher.py")
    for n, p in {**files, **wheel_map}.items():
        if p.resolve() != (payload / n).resolve():
            shutil.copyfile(p, payload / n)
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    (payload / "saturn_run_config.json").write_bytes(raw)
    save(payload / "saturn_debug_request.json", request)
    packages = {}
    for n, p in wheel_map.items():
        package = "mdb" if n.startswith("mdb_") else "saturn_pub" if n.startswith("saturn_") else "mrun"
        with zipfile.ZipFile(p) as archive:
            packages[package] = {member[len(package)+1:]: hashlib.sha256(archive.read(member)).hexdigest()
                for member in archive.namelist() if member.startswith(package + "/") and not member.endswith("/")}
    save(payload / "payload-seal.json", {"packages": packages, "wheels": request["runtime_wheels"], "files": request["source_hashes"],
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "worker": "worker.py", "worker_sha256": request["source_hashes"]["worker.py"]})
    save(run / "command.json", runpy.run_path(str(HERE.parent / "wsp_case_study/launch.py"))["command"](request))
    return {"run": str(run), "request_sha256": request["request_sha256"]}


def submit(base, name):
    from mdb_lanes.job_support import guarded_mrun_launch
    run = base / name
    if (run / "submission.json").exists():
        raise ValueError("finite lease already submitted")
    request = json.loads((run / "request.json").read_text())
    for n, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / n) != expected:
            raise ValueError("staged payload differs")
    os.environ.setdefault("MRUN_URL", "http://192.168.68.68:9025")
    phase = request["config"]["phase"]
    job = guarded_mrun_launch(json.loads((run / "command.json").read_text()), request=request,
        experiment="mdb-workflow-comparison-" + phase, task="analysis", model=None,
        payload=run / "payload", prefer_host="beast", pin="beast", gpu=False, needs={"cpu": True}, timeout_s=840,
        retry_on_kill=False, preflight=True,
        note="Paired workflow calibration, constructed width4 CPU FP64: " + phase +
             "; 3 specimens x 3 tool workflows, at most6 native calls/row this round," + str(request["config"]["program_seconds"]) + "s numerical program deadline. No autograd, pretrained model, retries or children. One finite CPU lease.")
    result = {"job_id": str(job), "request_sha256": request["request_sha256"]}
    save(run / "submission.json", result)
    return result


def collect(base, name):
    run = base / name
    job = json.loads((run / "submission.json").read_text())["job_id"]
    result = runpy.run_path(str(HERE.parent / "flux_family_custody.py"))["collect"](run, job,
        limits={"max_archive_bytes": 32 << 20, "max_expanded_bytes": 96 << 20, "max_member_bytes": 16 << 20})
    if result.get("collection_status") == "not-terminal":
        return {k: result[k] for k in ("collection_status", "job_id", "state")}
    return result


def infer_one(base, round_number, specimen, arm, packet, history, *, common=COMMON, docs=DOCS):
    plan = json.loads((base / "plan.json").read_text())
    tid = specimen["id"] + "-" + arm
    root = base / ("editor-round-" + str(round_number)) / tid
    root.mkdir(parents=True, exist_ok=False)
    packet = dict(packet)
    packet["numerical_program_seconds"] = plan["program_seconds"]
    if "weights" in packet:
        packet["weights"] = {name.removesuffix(".weight"): value for name, value in packet["weights"].items()}
    prompt = common + "\n" + docs[arm] + "\nROUND " + str(round_number) + "\nPUBLIC PACKET:\n" + json.dumps(packet) + "\nOWN HISTORY:\n" + json.dumps(history)
    if len(prompt.encode()) > plan["limits"]["max_prompt_bytes"]:
        raise ValueError("editor prompt exceeds frozen byte ceiling")
    (root / "prompt.txt").write_text(prompt)
    shutil.copyfile(base / "editor-schema.json", root / "schema.json")
    cmd = ["rtk", "proxy", "codex", "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
        "--skip-git-repo-check", "--model", plan["editor"]["model"], "--json", "--color", "never",
        "--output-schema", str(root / "schema.json"), "--output-last-message", str(root / "answer.json"),
        "-c", 'model_reasoning_effort="' + plan["editor"]["reasoning_effort"] + '"',
        "-c", "project_doc_max_bytes=0", "-c", "skills.include_instructions=false", "-c", "skills.bundled.enabled=false",
        "-c", 'web_search="disabled"']
    for feature in ("shell_tool", "apps", "plugins", "memories", "multi_agent", "browser_use", "computer_use", "image_generation", "hooks", "workspace_dependencies", "sleep_tool"):
        cmd += ["-c", "features." + feature + "=false"]
    cmd += ["-"]
    save(root / "command.json", cmd)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mdb-comparison-editor-") as clean_directory:
        result, timed_out = run_editor_process(cmd, prompt, clean_directory, plan["editor"]["call_timeout_seconds"])
    (root / "events.jsonl").write_text(result.stdout)
    (root / "stderr.txt").write_text(result.stderr)
    if timed_out:
        save(root / "failure.json", {"returncode": result.returncode, "timeout": True,
            "timeout_seconds": plan["editor"]["call_timeout_seconds"], "seconds": time.perf_counter()-started,
            "scope": "provider deadline; partial stdout/stderr retained, process group stopped"})
        raise RuntimeError("editor deadline reached; partial output preserved " + str(root))
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    item_types = {e.get("item", {}).get("type") for e in events if "item" in e}
    unexpected = item_types - {"agent_message", "reasoning"}
    if result.returncode or unexpected or not (root / "answer.json").exists():
        save(root / "failure.json", {"returncode": result.returncode, "unexpected_item_types": sorted(unexpected),
            "seconds": time.perf_counter()-started})
        raise RuntimeError("editor inference failed or invoked a forbidden tool; preserved " + str(root))
    answer = json.loads((root / "answer.json").read_text())
    if set(answer) != set(editor_schema()["required"]):
        raise ValueError("editor response schema differs")
    usages = [e["usage"] for e in events if e.get("type") == "turn.completed"]
    receipt = {"model": plan["editor"]["model"], "reasoning_effort": plan["editor"]["reasoning_effort"],
        "seconds": time.perf_counter()-started, "usage": usages, "prompt_sha256": sha(root / "prompt.txt"),
        "answer_sha256": sha(root / "answer.json"), "events_sha256": sha(root / "events.jsonl"),
        "tool_calls": 0, "source_history_identity": digest(history), "controller_sha256": sha(__file__),
        "protocol_source_sha256": sha(HERE / "protocol.py")}
    save(root / "receipt.json", receipt)
    print("EDITOR_DONE " + tid + " round=" + str(round_number) + " seconds=" + str(round(receipt["seconds"], 2)), flush=True)
    return {"id": tid, "arm": arm, "specimen": specimen, "proposal": answer, "editor_receipt": receipt}


def editors(base, round_number):
    plan = json.loads((base / "plan.json").read_text())
    packets = json.loads((extracted(base, "prepare") / "public-packets.json").read_text())
    history = {}
    if round_number == 2:
        for row in json.loads((extracted(base, "search-round-1") / "public-feedback.json").read_text()):
            history[row["trial"]] = [row]
    root = base / ("editor-round-" + str(round_number))
    root.mkdir(parents=True, exist_ok=False)
    calls = [(specimen, arm) for i, specimen in enumerate(plan["specimens"])
             for arm in (*ARMS[i % 3:], *ARMS[:i % 3])]
    result = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan["editor"]["parallel_inference_calls"]) as pool:
        futures = [pool.submit(infer_one, base, round_number, specimen, arm, packets[specimen["id"]],
            history.get(specimen["id"] + "-" + arm, [])) for specimen, arm in calls]
        for future in concurrent.futures.as_completed(futures):
            result.append(future.result())
    save(root / "proposals.json", sorted(result, key=lambda r: r["id"]))
    return {"round": round_number, "proposals": len(result), "identity": digest(result)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("init", "stage", "submit", "collect", "editors", "editor-smoke"))
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--name", default="prepare")
    p.add_argument("--phase", choices=("prepare", "search", "evaluate", "audit"), default="prepare")
    p.add_argument("--round", type=int, default=0)
    a = p.parse_args()
    base = a.base.resolve()
    if a.action == "editor-smoke":
        result = infer_one(base, 0, {"id": "readiness", "seed": 0, "kind": "healthy"}, "wsp_python",
            {"readiness_only": True, "instruction": "No specimen is supplied. Return analysis_python as print('ready'), mechanism as readiness, forecast as not applicable."}, [])
    else:
        result = initialize(base) if a.action == "init" else stage(base, a.name, a.phase, a.round) if a.action == "stage" else submit(base, a.name) if a.action == "submit" else collect(base, a.name) if a.action == "collect" else editors(base, a.round)
    print(json.dumps(result, sort_keys=True))
