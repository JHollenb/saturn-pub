"""Stage, submit once, and collect a bounded installed-wheel scaffold assay."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def environment():
    os.environ.setdefault("MDB_WORKER_UV_CACHE_DIR", "/mnt/big/llm-models/.uv-cache")
    os.environ.setdefault("MDB_WORKER_TORCH_HOME", "/mnt/big/llm-models/.torch-cache")
    os.environ.setdefault("MDB_WORKER_MODEL_ROOTS", "/mnt/big/llm-models")
    from mdb.execution_environment import intent
    return intent()


def stage(root, helper, mode):
    from mdb import parameter_artifacts
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    root.mkdir(parents=True, exist_ok=False)
    shots = "12+35=47\n481+307=788\n256+139=395\n604+218=822\n"
    alt = "Compute: 23+41=64; 518+266=784; 107+395=502; 732+149=881; "
    four = "12+35=47\n481+307=788\n4812+3079=7891\n2563+1398=3961\n6045+2187=8232\n"
    cases = [
        {"id": "new-three-a", "a": 347, "b": 125, "digits": 3, "prefix": shots},
        {"id": "new-three-b", "a": 582, "b": 206, "digits": 3, "prefix": shots},
        {"id": "alternate-a", "a": 347, "b": 125, "digits": 3, "prefix": alt},
        {"id": "alternate-b", "a": 246, "b": 137, "digits": 3, "prefix": alt},
        {"id": "four-a", "a": 3471, "b": 1256, "digits": 4, "prefix": four},
        {"id": "four-b", "a": 5823, "b": 2061, "digits": 4, "prefix": four},
    ]
    for case in cases:
        case["prompt"] = case.pop("prefix") + f"{case['a']}+{case['b']}="
    pairs = []
    for case, donor_a in ((cases[0], 547), (cases[1], 382), (cases[2], 547), (cases[4], 5471)):
        old = f"{case['a']}+{case['b']}="
        pairs.append({"id": "edit-" + case["id"], "recipient": case["prompt"],
                      "donor": case["prompt"][:-len(old)] + f"{donor_a}+{case['b']}=",
                      "digits": case["digits"], "changed_variable": "a most-significant digit"})
    config = {"model": "qwen2.5-1.5b", "model_path": "/mnt/big/llm-models/Qwen2.5-1.5B",
              "expected_weight_file_sha256": "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796",
              "output_root": "/mnt/big/mdb/qualification/live-scaffold-20261111", "seed": 20261111,
              "cases": cases[:1] if mode == "smoke" else cases,
              "pairs": pairs[:1] if mode == "smoke" else pairs,
              "object_followup": mode == "main", "mode": mode}
    save(root / "plan.json", config)
    worker = Path(__file__).with_name("live_scaffold_worker.py")
    retrieval = Path(parameter_artifacts.__file__)
    preflight = {"schema_version": "mrun-preflight-config-v1",
                 "hashed_roots": [worker.name, "parameter_source.py", "retrieve_weights.py", "preflight.v1.json"],
                 "path_token_allowlist": [config["model_path"], config["output_root"]]}
    save(root / "preflight.v1.json", preflight)
    request = {"schema": "mdb-live-scaffold-request-v1", "model": config["model"],
               "task_family": "forward", "backend": "hf", "device": "cuda", "dtype": "bfloat16",
               "batch_size": 1, "seq_len": 128,
               "backend_options": {"physical_batch_size": 1, "full_prefill_logits": False},
               "needs": {"cuda": True}, "execution_environment": environment(), "config": config,
               "source_hashes": {worker.name: sha(worker), "parameter_source.py": sha(helper),
                                 "retrieve_weights.py": sha(retrieval)},
               "scheduling_contract": one_lease_scheduling_contract(
                   model_loads=1, logical_branches=40, logical_operations=200,
                   logical_checkpoint_replays=8, max_local_work_items=256,
                   checkpoint_policy="One BF16 Qwen resident; scalar native role observations and branch-owned suffixes; one bounded MDB two-receiver discovery; no nested jobs or retries"),
               "workload_geometry": {"model_loads": 1, "physical_batch": 1,
                                     "max_prompt_tokens": 128, "max_hf_forwards": 260,
                                     "max_discovery_layer_tokens": 10000, "output_archive_max_bytes": 128 << 20},
               "resource_basis": "Intent-first placement; prior same-model native-eager qualification measured 3448MB VRAM/2396MB RSS. No hand reservation.",
               "timeout_s": 840,
               "queue_note": "Live time-scaffold exploration: one BF16 Qwen1.5B/eager resident, scalar batch, frozen addition reader roles, <=260 native forwards plus <=10000 MDB layer-token transitions; collect source bindings, signed interventions and own-token futures; stop on one report or 840s. No retries."}
    # This embedded v1 identity deliberately uses the bootstrap's original
    # JSON encoding; the outer mrun request has its own compact identity.
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save(root / "request.json", request)
    stage_worker(root / "payload", worker=worker, config=config, request=request,
                 additional_files=[(helper, "parameter_source.py"), (retrieval, "retrieve_weights.py"),
                                   (root / "preflight.v1.json", "preflight.v1.json")])
    print(json.dumps({"staged": str(root), "mode": mode, "request_sha256": request["request_sha256"]}))


def submit(root):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    if (root / "submission.json").exists():
        raise ValueError("already submitted; inspect the existing job")
    request = json.loads((root / "request.json").read_text())
    environment()
    job = guarded_mrun_launch(worker_command("live_scaffold_worker.py"), request=request,
        model=request["model"], task="forward", experiment="mdb-live-scaffold-20261111",
        payload=root / "payload", prefer_host="beast", gpu=True, needs=request["needs"],
        timeout_s=request["timeout_s"], retry_on_kill=False, preflight=True, detach=True,
        note=request["queue_note"])
    save(root / "submission.json", {"job_id": str(job), "request_sha256": request["request_sha256"]})
    print(json.dumps({"job_id": str(job), "root": str(root)}))


def collect(root):
    from mdb.job_support import guarded_job_receipt, terminal_job
    job_id = json.loads((root / "submission.json").read_text())["job_id"]
    identifier, job, logs = terminal_job(job_id)
    if job.get("state") not in {"succeeded", "failed", "timeout", "killed_ram", "killed_vram", "cancelled", "lost"}:
        print(json.dumps({"job_id": identifier, "state": job.get("state")}))
        return
    (root / "worker.log.gz").write_bytes(gzip.compress(logs.encode(), mtime=0))
    save(root / "job.json", job)
    save(root / "custody.json", guarded_job_receipt(identifier))
    marker = "SCAFFOLD_REPORT_V1 "
    encoded = next((line[len(marker):] for line in logs.splitlines() if line.startswith(marker)), None)
    if encoded is None:
        print(json.dumps({"job_id": identifier, "state": job.get("state"), "report": False,
                          "tail": "\n".join(logs.splitlines()[-12:])}))
        return
    report = json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))
    chunks, count = {}, None
    for line in logs.splitlines():
        if line.startswith("SCAFFOLD_ARCHIVE_V1 "):
            head, chunk = line[len("SCAFFOLD_ARCHIVE_V1 "):].split(" ", 1)
            index, total = map(int, head.split("/"))
            if index in chunks or (count is not None and count != total):
                raise ValueError("duplicate or inconsistent archive chunks")
            count = total
            chunks[index] = base64.b64decode(chunk, validate=True)
    if count is None or sorted(chunks) != list(range(count)):
        raise ValueError("incomplete archive")
    raw = b"".join(chunks[i] for i in range(count))
    if len(raw) != report["archive"]["bytes"] or hashlib.sha256(raw).hexdigest() != report["archive"]["sha256"]:
        raise ValueError("archive byte identity mismatch")
    (root / "result.tar.gz").write_bytes(raw)
    results = root / "results"
    results.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        if sum(p.size for p in archive.getmembers()) > 512 << 20:
            raise ValueError("expanded archive exceeds bound")
        for member in archive.getmembers():
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe archive member")
            target = results / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(archive.extractfile(member).read())
    manifest = json.loads((results / "output-manifest.json").read_text())
    for entry in manifest["files"]:
        path = results / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("retained file differs: " + entry["path"])
    request = json.loads((root / "request.json").read_text())
    if report["request_sha256"] != request["request_sha256"]:
        raise ValueError("report request differs")
    save(root / "report.json", report)
    print(json.dumps({"job_id": identifier, "state": job.get("state"), "results": str(results),
                      "files_verified": len(manifest["files"]), "archive_bytes": len(raw),
                      "mechanics_status": report["mechanics_status"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "submit", "collect"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--parameter-helper", type=Path)
    parser.add_argument("--mode", choices=("smoke", "main"), default="smoke")
    args = parser.parse_args()
    if args.action == "stage":
        if args.parameter_helper is None:
            parser.error("stage requires explicit --parameter-helper source")
        stage(args.root.resolve(), args.parameter_helper.resolve(), args.mode)
    elif args.action == "submit":
        submit(args.root.resolve())
    else:
        collect(args.root.resolve())


if __name__ == "__main__":
    main()
