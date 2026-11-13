"""Stage, admit and collect one finite cache experiment or saved-only reader."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def command(request, worker):
    env = request["execution_environment"]
    cmd = ["/usr/bin/env", "PYTHONUNBUFFERED=1", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=", "CUDA_VISIBLE_DEVICES=", "UV_LINK_MODE=" + env["uv_link_mode"],
        "UV_CACHE_DIR=" + env["uv_cache_dir"], "TORCH_HOME=" + env["torch_home"],
        "LLM_MODELS_EXTRA_ROOT=", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false", "SATURN_RESIDENT_WORKER=1",
        "SATURN_RUN_CONFIG=saturn_run_config.json", "SATURN_DEBUG_REQUEST=saturn_debug_request.json",
        "TMPDIR=/mnt/hdd8tb/mdb/tmp", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4",
        "uv", "run", "--no-project", "--offline", "--python", env["python"]]
    for dependency in [*("./" + name for name in request["runtime_wheels"]), *env["requirements"]]:
        cmd += ["--with", dependency]
    return cmd + ["python", "-B", "-u", "-m", "mdb_lanes.worker_bootstrap", worker]


def stage(workspace, base, run, kind, source):
    from mdb_lanes.job_support import one_lease_scheduling_contract

    run.mkdir(parents=True, exist_ok=False)
    payload = run / "payload"
    payload.mkdir()
    previous = workspace / "artifacts/mdb-qwen-families-20261107/attention/development-03"
    parent_request = read(previous / "request.json")
    environment = parent_request["execution_environment"]
    qualification = workspace / "mdb/qualification"
    worker = "qwen_cache_restoration.py" if kind == "native" else "qwen_cache_restoration_audit.py"
    sources = {worker: qualification / worker}
    if kind == "native":
        sources.update({name: qualification / name for name in (
            "qwen_family_math.py", "qwen_family_campaign.py", "retrieve_qwen_family_weights.py")})
        sources["design.json"] = base / "design-v1.json"
        sources.update({path.name: path for path in (base / "sources").iterdir() if path.is_file()})
        config = dict(schema="mdb-qwen-cache-restoration-v1", model_path="/mnt/big/llm-models/Qwen2.5-0.5B-Instruct",
            output_scratch="/mnt/hdd8tb/mdb/research", max_wall_seconds=180,
            design_sha256=sha(base / "design-v1.json"), scientific_gates=[], retries=False, nested_mrun=False)
        paths = [config["model_path"], config["output_scratch"], read(base / "design-v1.json")["parent_native_root"]]
        extras = dict(physical_forward_census=dict(functional=20, native=20, prefills=0), task_family="recorder")
    else:
        require(source is not None, "native source run required for saved audit")
        publication = authenticate(source, "native")
        config = dict(source=publication["output_root"], manifest_sha256=publication["files_manifest_sha256"],
            output="/mnt/hdd8tb/mdb/research/qwen-cache-audit-" + publication["job_id"], model_forwards=0)
        paths = [config["source"], config["output"], read(base / "design-v1.json")["parent_native_root"]]
        extras = dict(task_family="analysis")
    save(run / "plan.json", config)
    save(payload / "preflight.v1.json", dict(schema_version="mrun-preflight-config-v1",
        hashed_roots=[*sources, "preflight.v1.json"], path_token_allowlist=paths))
    sources["preflight.v1.json"] = payload / "preflight.v1.json"
    request = dict(schema="mdb-qwen-cache-restoration-request-v1", kind=kind, config=config,
        execution_environment=environment, runtime_wheels=parent_request["runtime_wheels"],
        source_hashes={name: sha(path) for name, path in sources.items()}, launcher_sha256=sha(__file__),
        backend="hf" if kind == "native" else "saved-array-reader", device="cpu", dtype="float32", needs=dict(cpu=True),
        batch_size=1, seq_len=32, backend_options=dict(physical_batch_size=1, full_prefill_logits=False),
        scheduling_contract=one_lease_scheduling_contract(model_free=kind != "native", model_loads=int(kind == "native"),
            logical_branches=4 if kind == "native" else 1, logical_operations=40 if kind == "native" else 1,
            logical_checkpoint_replays=4 if kind == "native" else 0, max_local_work_items=64, extra=extras,
            checkpoint_policy="One finite lease; four same-parent cache forks and own-token futures; no retries or children"))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save(run / "request.json", request)
    for name, path in sources.items():
        if path != payload / name:
            shutil.copyfile(path, payload / name)
    for name, expected in request["runtime_wheels"].items():
        origin = previous / "payload" / name
        require(sha(origin) == expected, "pinned public wheel changed")
        shutil.copyfile(origin, payload / name)
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    (payload / "saturn_run_config.json").write_bytes(raw)
    save(payload / "saturn_debug_request.json", request)
    packages = {}
    for name in request["runtime_wheels"]:
        package = "mdb" if name.startswith("mdb_") else "saturn_pub" if name.startswith("saturn_") else "mrun"
        with zipfile.ZipFile(payload / name) as archive:
            packages[package] = {member[len(package)+1:]: hashlib.sha256(archive.read(member)).hexdigest()
                for member in archive.namelist() if member.startswith(package + "/") and not member.endswith("/")}
    save(payload / "payload-seal.json", dict(packages=packages, wheels=request["runtime_wheels"],
        files=request["source_hashes"], config_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        worker=worker, worker_sha256=request["source_hashes"][worker]))
    args = command(request, worker)
    if kind == "audit":
        args += ["--source", config["source"], "--manifest-sha256", config["manifest_sha256"], "--out", config["output"]]
    save(run / "staged-command.json", dict(command=args))
    shutil.copyfile(__file__, run / "frozen-launcher.py")
    return dict(request_sha256=request["request_sha256"], kind=kind, payload_bytes=sum(path.stat().st_size for path in payload.iterdir()))


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    request = read(run / "request.json")
    require(not (run / "submission.json").exists(), "finite request already submitted")
    require(sha(__file__) == request["launcher_sha256"], "frozen launcher differs")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        require(sha(run / "payload" / name) == expected, "staged member changed: " + name)
    review = read(run / "finite-run-review.json")
    require(review["request_sha256"] == request["request_sha256"] and review["blocking_issues"] == [], "exact finite review missing")
    args = read(run / "staged-command.json")["command"]
    save(run / "admission-command.json", dict(command=args))
    native = request["kind"] == "native"
    options = {} if native else dict(ram_mb=2048, cpu_threads=2, est_wall_s=30, disk_gb=1)
    job = guarded_mrun_launch(args, request=request, experiment="mdb-qwen-cache-restoration-" + request["kind"],
        task="recorder" if native else "analysis", model="qwen2.5-0.5b-instruct" if native else None,
        payload=run / "payload", prefer_host="beast", needs=request["needs"], gpu=False, timeout_s=300,
        retry_on_kill=False, preflight=True, **options,
        note="Finite CPU FP32 eager cache K/V factorial: 4 branches x 5 cached choices, 20 functional +20 native calls, no prefills; all predictions seal before native; stop after complete arrays. Saved reader uses zero models.")
    value = dict(job_id=str(job), request_sha256=request["request_sha256"])
    save(run / "submission.json", value)
    return value


def authenticate(run, kind):
    from mrun.client.api import Api

    from mdb_lanes.job_support import request_sha256

    request, jid = read(run / "request.json"), read(run / "submission.json")["job_id"]
    api = Api()
    job = api.json("GET", "/api/jobs/" + jid)
    require(job["state"] == "succeeded" and job["result"]["returncode"] == 0, "exact lease not completed")
    require(job["cmd"] == read(run / "admission-command.json")["command"] and
            job["config"]["saturn_request_sha256"] == request_sha256(request), "command/request differs")
    custody = job["payload_custody"]
    require(custody["required"] and custody["sealed"]["sha256"] == custody["declared"]["sha256"] == custody["executed"]["sha256"], "payload custody differs")
    _, logs, _ = api.request("GET", "/api/jobs/" + jid + "/logs")
    marker = "MDB_QWEN_CACHE_READY " if kind == "native" else "MDB_QWEN_CACHE_AUDIT_READY "
    publications = [json.loads(line[len(marker):]) for line in logs.decode(errors="replace").splitlines() if line.startswith(marker)]
    require(len(publications) == 1, "one complete output publication required")
    publication = publications[0]
    if kind == "native":
        require(publication["job_id"] == jid and publication["request_sha256"] == request["request_sha256"], "publication owner differs")
    for name, value in (("publication.json", publication), ("scheduler-job.json", job)):
        if (run / name).exists():
            require(read(run / name) == value, "recorded authentication differs")
        else:
            save(run / name, value)
    return publication


def collect(run, kind):
    publication = authenticate(run, kind)
    code = '''import hashlib,json
from pathlib import Path
r=Path(ROOT)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if KIND=='native':
 if sha(r/'files.json')!=DIGEST:raise ValueError('manifest differs')
 files=json.loads((r/'files.json').read_text())['files']
else:
 if sha(r/'audit.json')!=DIGEST:raise ValueError('audit differs')
 files={str(p.relative_to(r)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in r.rglob('*') if p.is_file()}
for name,ref in files.items():
 p=r/name
 if p.is_symlink() or not p.resolve().is_relative_to(r) or sha(p)!=ref['sha256'] or p.stat().st_size!=ref['bytes']:raise ValueError('member differs: '+name)
names=[n for n in files if not n.endswith(('.npz','.npy','.npyb'))]
if KIND=='native':names.append('files.json')
selected={n:{'sha256':sha(r/n),'bytes':(r/n).stat().st_size} for n in names}
if sum(ref['bytes'] for ref in selected.values())>32<<20:raise ValueError('metadata exceeds budget')
print(json.dumps({'files':selected,'raw_bytes':sum(ref['bytes'] for ref in files.values()),'raw_members':len(files)}))
'''.replace("ROOT", repr(publication["output_root"])).replace("KIND", repr(kind)).replace("DIGEST", repr(publication.get("files_manifest_sha256", publication.get("audit_sha256"))))

    def remote():
        r = subprocess.run(["ssh", "-T", "beast", "python3", "-"], input=code, text=True, capture_output=True, check=True, timeout=60)
        return json.loads(r.stdout)

    before = remote()
    target = run / "compact"
    target.mkdir(exist_ok=False)
    subprocess.run(["rsync", "-a", "--protect-args", "--files-from=-", "beast:" + publication["output_root"] + "/", str(target) + "/"],
        input="\n".join(before["files"]) + "\n", text=True, capture_output=True, check=True, timeout=60)
    actual = {str(p.relative_to(target)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in target.rglob("*") if p.is_file()}
    require(before == remote() and actual == before["files"], "collection custody differs")
    save(run / "collection.json", dict(kind=kind, publication=publication, inventory=before, raw_retained_on_fleet=True))
    return dict(raw_bytes=before["raw_bytes"], compact_bytes=sum(ref["bytes"] for ref in actual.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "submit", "collect"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--kind", choices=("native", "audit"), default="native")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    value = stage(args.workspace.resolve(), args.base.resolve(), args.run.resolve(), args.kind,
                  args.source.resolve() if args.source else None) if args.action == "stage" else (
                      submit(args.run.resolve()) if args.action == "submit" else collect(args.run.resolve(), args.kind))
    print(json.dumps(value, sort_keys=True))
