"""Stage or explicitly admit the finite Qwen2 circuit-discovery qualification.

The default action only creates a sealed payload.  ``--submit`` is the
explicit admission edge; no model work is performed by this frontend.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WORKER = "worker.py"
MODEL = "qwen2.5-1.5b"
MODEL_PATH = "/mnt/big/llm-models/Qwen2.5-1.5B"
OUTPUT_ROOT = "/mnt/big/mdb/qualification/circuit-discovery-20261106"
EXPERIMENT = "mdb-qwen-circuit-discovery-20261106"
# Historical trained-MDB Qwen receipt: the raw SHA256 of the one-file
# ``model.safetensors`` checkpoint.  Multi-file source-map identity is carried
# separately as ``expected_weight_content_sha256`` when a manifest is staged.
KNOWN_WEIGHT_FILE_SHA256 = "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"
REPORT_MARKER = "MDB_CIRCUIT_DISCOVERY_REPORT_V1 "
ARCHIVE_MARKER = "MDB_CIRCUIT_DISCOVERY_ARCHIVE_V1 "
MAX_EXPANDED_BYTES = 512 << 20
MAX_ARCHIVE_BYTES = 128 << 20
MAX_ARCHIVE_FILES = 65536


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def worker_intent() -> dict[str, Any]:
    """Bind default worker caches/model root to the known Beast filesystem."""
    # The public environment helper preserves explicit maintainer overrides;
    # its generic local default points at the mbp1 home directory, which is
    # not writable after a payload is pulled to Beast.
    os.environ.setdefault("MDB_WORKER_UV_CACHE_DIR", "/mnt/big/llm-models/.uv-cache")
    os.environ.setdefault("MDB_WORKER_TORCH_HOME", "/mnt/big/llm-models/.torch-cache")
    os.environ.setdefault("MDB_WORKER_MODEL_ROOTS", "/mnt/big/llm-models")
    from mdb_lanes.execution_environment import intent

    return intent()


def _apply_worker_intent(environment: dict[str, Any]) -> None:
    """Make command construction use the exact sealed worker environment."""
    os.environ["MDB_WORKER_UV_CACHE_DIR"] = str(environment["uv_cache_dir"])
    os.environ["MDB_WORKER_TORCH_HOME"] = str(environment["torch_home"])
    os.environ["MDB_WORKER_MODEL_ROOTS"] = str(environment["extra_model_roots"])
    os.environ["MDB_WORKER_PYTHON"] = str(environment["python"])


def _save_log(run_dir: Path, logs: str, storage: str = "plain") -> Path:
    if storage == "plain":
        target = run_dir / "worker.log"
        target.write_text(logs)
        return target
    if storage == "gzip":
        target = run_dir / "worker.log.gz"
        target.write_bytes(gzip.compress(logs.encode("utf-8"), mtime=0))
        return target
    raise ValueError("log storage must be plain or gzip")


def _decode_report(logs: str) -> dict[str, Any] | None:
    for line in reversed(logs.splitlines()):
        if line.startswith(REPORT_MARKER):
            raw = base64.b64decode(line[len(REPORT_MARKER):].strip(), validate=True)
            value = json.loads(gzip.decompress(raw))
            if not isinstance(value, dict):
                raise ValueError("circuit discovery report marker is not an object")
            return value
    return None


def _collect_archive(logs: str, custody: dict[str, Any]) -> bytes:
    chunks: dict[int, bytes] = {}
    expected = None
    for line in logs.splitlines():
        if not line.startswith(ARCHIVE_MARKER):
            continue
        header, encoded = line[len(ARCHIVE_MARKER):].split(" ", 1)
        index_text, count_text = header.split("/", 1)
        index, count = int(index_text), int(count_text)
        if count <= 0 or not 0 <= index < count or (expected is not None and expected != count):
            raise ValueError("invalid circuit discovery archive chunk geometry")
        expected = count
        if index in chunks:
            raise ValueError("duplicate circuit discovery archive chunk")
        chunks[index] = base64.b64decode(encoded, validate=True)
    if expected is None or set(chunks) != set(range(expected)):
        raise ValueError("incomplete circuit discovery archive marker stream")
    raw = b"".join(chunks[index] for index in range(expected))
    if len(raw) != int(custody["archive_bytes"]) or len(raw) > MAX_ARCHIVE_BYTES:
        raise ValueError("circuit discovery archive byte count differs")
    if sha(raw) != custody["archive_sha256"]:
        raise ValueError("circuit discovery archive SHA differs")
    return raw


def _verify_extracted(root: Path) -> dict[str, Any]:
    manifest_path = root / "output-manifest.json"
    manifest = read(manifest_path)
    if manifest.get("schema") != "mdb-circuit-discovery-output-manifest-v1":
        raise ValueError("unsupported circuit discovery output manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) > MAX_ARCHIVE_FILES:
        raise ValueError("invalid circuit discovery output manifest file list")
    total = 0
    expected_paths = set()
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in expected_paths:
            raise ValueError("unsafe or duplicate result manifest path")
        path = root / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"result manifest file differs: {relative}")
        if file_sha(path) != entry["sha256"]:
            raise ValueError(f"result manifest SHA differs: {relative}")
        expected_paths.add(relative.as_posix())
        total += path.stat().st_size
    if total != int(manifest.get("expanded_bytes")) or total > MAX_EXPANDED_BYTES:
        raise ValueError("result manifest expanded byte count differs")
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file() and not path.is_symlink()}
    if actual_paths != expected_paths | {"output-manifest.json"}:
        raise ValueError("result archive contains an unexpected or missing file")
    return manifest


def collect(run_dir: Path, job_id: str, *, log_storage: str = "plain") -> dict[str, Any]:
    """Collect the terminal job's bounded raw session archive and verify every file."""
    from mdb_lanes.job_support import terminal_job

    run_dir = run_dir.resolve()
    request = read(run_dir / "request.json")
    terminal_id, job, logs = terminal_job(str(job_id))
    if str(terminal_id) != str(job_id):
        raise ValueError("collector received a retry lineage instead of the admitted job")
    _save_log(run_dir, logs, log_storage)
    report = _decode_report(logs)
    if report is None:
        return {"job_id": str(job_id), "state": job.get("state"), "report": False}
    if report.get("job_id") != str(job_id) or report.get("request_sha256") != request["request_sha256"]:
        raise ValueError("report job/request binding differs")
    custody = report.get("artifact_custody")
    if not isinstance(custody, dict) or custody.get("schema") != "mdb-circuit-discovery-custody-v1":
        raise ValueError("report lacks bounded artifact custody")
    raw = _collect_archive(logs, custody)
    archive_path = run_dir / "result.tar.gz"
    if archive_path.exists():
        raise ValueError("result archive already collected")
    archive_path.write_bytes(raw)
    extracted = run_dir / "results"
    extracted.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_FILES or sum(member.size for member in members) > MAX_EXPANDED_BYTES:
            raise ValueError("result archive expanded geometry exceeds bound")
        for member in members:
            relative = Path(member.name)
            if (not member.isfile() or member.issym() or member.islnk()
                    or relative.is_absolute() or ".." in relative.parts):
                raise ValueError("unsafe result archive member")
            destination = extracted / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("result archive member has no data")
                shutil.copyfileobj(source, stream, length=1 << 20)
    manifest = _verify_extracted(extracted)
    save(run_dir / "report.json", report)
    save(run_dir / "archive-custody.json", {
        "schema": "mdb-qwen-circuit-discovery-collected-custody-v1",
        "job_id": str(job_id), "request_sha256": request["request_sha256"],
        "archive": "result.tar.gz", "archive_sha256": sha(raw),
        "archive_bytes": len(raw), "manifest": "results/output-manifest.json",
        "manifest_sha256": file_sha(extracted / "output-manifest.json"),
        "file_count": manifest["file_count"], "expanded_bytes": manifest["expanded_bytes"],
    })
    save(run_dir / "job.json", job)
    return {"job_id": str(job_id), "state": job.get("state"),
            "report": str(run_dir / "report.json"), "results": str(extracted),
            "archive": str(archive_path)}


def request_identity(request: dict[str, Any]) -> str:
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    return sha(json.dumps(body, sort_keys=True).encode("utf-8"))


def plan() -> dict[str, Any]:
    return {
        "schema": "mdb-qwen-circuit-discovery-plan-v1",
        "model": MODEL,
        "model_path": MODEL_PATH,
        "family": "qwen2",
        "device": "cuda",
        "dtype": "bfloat16",
        "seed": 20261106,
        # This is the prior trained MDB Qwen2.5-1.5B raw model.safetensors
        # SHA.  A supplied parameter manifest additionally binds the canonical
        # multi-file source-map digest below during staging and worker load.
        "expected_weight_file_sha256": KNOWN_WEIGHT_FILE_SHA256,
        "output_root": OUTPUT_ROOT,
        "parameter_manifest": "parameter-artifacts.json",
        "retrieval_script": "retrieve_weights.py",
        "cases": [
            {
                "id": "yen-hidden-bridge-positive",
                "prompt": "The capital of the country whose currency is the yen is",
                "followup": " The country was",
                "observable": [" Tokyo", " Paris"],
            },
            {
                "id": "fresh-contextual-binding",
                "prompt": ("Lena placed the green key in the box. Omar placed the silver coin in the bag. "
                           "The object Omar placed is the"),
                "followup": " The object Lena placed is the",
            },
        ],
        "discovery_policy": {
            "receiver_offsets": "[0, 2 + len(follow_up_ids)] (usually [0,3]); first answer and last supplied follow-up token",
            "layers": "all-by-default",
            "row_search": "all committed history, runtime maximum 256 rows",
            "candidates_per_receiver": 2,
            "scales": [0.0, 0.5],
            "continuation_tokens": "full finite plan length (worker resolves; <=32)",
            "top_k": 5,
            "max_native_operations": 10000,
            "max_capture_bytes": 256 << 20,
            "plan_authority": "caller token IDs for final prompt token and fixed follow-up; native-greedy for generated tokens",
            "site_policy": "no hand-labeled carrier site; passive native read observer selects candidates",
        },
        "limits": [
            "Two short contexts and one fixed seed; trends are exploratory observations.",
            "The known yen case is a positive lead, not a semantic or minimal-circuit certification.",
            "Candidate rankings use the runtime's FP32 explanatory read contribution; native BF16 execution remains authoritative.",
            "The explicit native-eager backend is qualified; any default-backend rounding difference is diagnostic only.",
            "No automatic retries, nested mrun submissions, parameter updates, or hand-labeled sites.",
        ],
    }


def _manifest_identity(path: Path, model_path: str) -> tuple[dict[str, Any], dict[str, str], str, str]:
    from mdb_lanes.model_os_contracts import content_digest
    from mdb_lanes.parameter_artifacts import validate_manifest

    manifest = validate_manifest(read(path))
    defaults = manifest.get("defaults", {})
    checkpoint_root = defaults.get("checkpoint_root")
    if checkpoint_root and Path(checkpoint_root).resolve() != Path(model_path).resolve():
        raise ValueError("parameter manifest checkpoint root differs from Qwen2.5-1.5B fleet path")
    source_hashes: dict[str, str] = {}
    for entry in manifest["artifacts"]:
        source = entry["source"]
        filename, digest = str(source["file"]), str(source["file_sha256"])
        if filename in source_hashes and source_hashes[filename] != digest:
            raise ValueError(f"parameter manifest has conflicting source hash: {filename}")
        source_hashes[filename] = digest
    if not source_hashes:
        raise ValueError("parameter manifest contains no checkpoint source files")
    checkpoint = Path(checkpoint_root or model_path)
    if checkpoint.is_dir():
        for filename, expected in source_hashes.items():
            source = checkpoint / filename
            if not source.is_file() or file_sha(source) != expected:
                raise ValueError(f"checkpoint source does not match parameter manifest: {source}")
        verification = "local-checkpoint-verified"
    else:
        # The frontend commonly runs on mbp1 while the immutable checkpoint
        # lives on Beast.  The manifest remains the source binding; the worker
        # performs the same full source-hash verification under its CUDA lease.
        verification = "deferred-to-beast-worker"
    return manifest, source_hashes, content_digest(source_hashes), verification


def stage(run_dir: Path, *, parameter_manifest: Path | None = None,
          retrieval_script: Path | None = None) -> dict[str, Any]:
    from mdb_lanes import parameter_artifacts
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import runtime_wheel_names, stage_worker

    run_dir = run_dir.resolve()
    retrieval_script = (retrieval_script or Path(parameter_artifacts.__file__)).resolve()
    if not retrieval_script.is_file():
        raise ValueError(f"standalone retrieval source does not exist: {retrieval_script}")
    manifest = None
    source_hashes: dict[str, str] = {}
    checkpoint_verification = "generated-on-worker"
    weight_file_sha = KNOWN_WEIGHT_FILE_SHA256
    weight_content_sha = None
    if parameter_manifest is not None:
        parameter_manifest = parameter_manifest.resolve()
        if not parameter_manifest.is_file():
            raise ValueError(f"parameter manifest does not exist: {parameter_manifest}")
        manifest, source_hashes, weight_content_sha, checkpoint_verification = _manifest_identity(parameter_manifest, MODEL_PATH)
        weight_file_sha = source_hashes.get("model.safetensors")
    run_dir.mkdir(parents=True, exist_ok=False)
    if parameter_manifest is not None:
        shutil.copyfile(parameter_manifest, run_dir / "parameter-artifacts.json")
    shutil.copyfile(retrieval_script, run_dir / "retrieve_weights.py")
    config = plan()
    config.update({"expected_weight_file_sha256": weight_file_sha,
                   "expected_weight_content_sha256": weight_content_sha,
                   "parameter_manifest": "parameter-artifacts.json" if parameter_manifest is not None else "generated-on-worker",
                   "parameter_manifest_sha256": file_sha(run_dir / "parameter-artifacts.json") if parameter_manifest is not None else None,
                   "retrieval_script_sha256": file_sha(run_dir / "retrieve_weights.py")})
    save(run_dir / "plan.json", config)
    binding = {
        "schema": "mdb-qwen-circuit-discovery-source-binding-v1",
        "model": MODEL, "model_path": MODEL_PATH, "family": "qwen2",
        # In the normal mbp1 -> Beast flow the full recipe is generated and
        # sealed by the worker before model load.  A supplied manifest keeps
        # this field as the staged payload name for exact replay.
        "parameter_manifest": ("parameter-artifacts.json" if parameter_manifest is not None
                                else "generated-on-worker"),
        "parameter_manifest_sha256": config["parameter_manifest_sha256"],
        "retrieval_script": "retrieve_weights.py",
        "retrieval_script_sha256": config["retrieval_script_sha256"],
        "checkpoint_source_files_sha256": source_hashes,
        "expected_weight_file_sha256": weight_file_sha,
        "expected_weight_content_sha256": weight_content_sha,
        "artifact_count": len(manifest["artifacts"]) if manifest is not None else None,
        "checkpoint_source_verification": checkpoint_verification,
        "binding_scope": "full registered Qwen2.5-1.5B safetensors checkpoint",
    }
    save(run_dir / "source-binding.json", binding)
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {
        "schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [WORKER, "launcher.py", "plan.json", "source-binding.json",
                         "retrieve_weights.py", "preflight.v1.json"]
                         + (["parameter-artifacts.json"] if parameter_manifest is not None else []),
        "path_token_allowlist": [MODEL_PATH, OUTPUT_ROOT, "parameter-artifacts.json", "retrieve_weights.py"],
    })
    raw = (json.dumps(config, sort_keys=True) + "\n").encode("utf-8")
    extras = [
        (HERE / "launch.py", "launcher.py"),
        (run_dir / "plan.json", "plan.json"),
        (run_dir / "source-binding.json", "source-binding.json"),
        (run_dir / "retrieve_weights.py", "retrieve_weights.py"),
        (preflight, "preflight.v1.json"),
    ]
    if parameter_manifest is not None:
        extras.append((run_dir / "parameter-artifacts.json", "parameter-artifacts.json"))
    resource = {"timeout_s": 900,
                "calibration": ("intent-first mrun planning under strict preflight; prior Qwen2.5-1.5B native-eager "
                                 "cot-register run measured peak VRAM 4792MB and RSS 2130MB; this all-layer observer "
                                 "workload is larger and remains scheduler-admitted rather than hand-reserved")}
    scheduling = one_lease_scheduling_contract(
        model_loads=1, logical_branches=32, logical_operations=32,
        logical_checkpoint_replays=32, max_local_work_items=64,
        checkpoint_policy=("one resident Qwen2.5-1.5B BF16 CUDA native-eager model; two sequential explicit-ID cases; "
                           "one prefill plus 16 bounded logical futures per case (two direct native audits, two observer/parity "
                           "futures, eight candidate-dose futures, four exact-reinstall futures); "
                           "all branches in process, one outer lease, no retries"),
        extra={"logical_futures": 32, "logical_futures_per_case": 16,
               "logical_future_units": "one bounded native suffix or direct native audit",
               "prefills": 2,
               "plan_token_positions_per_case_max": 32,
               "total_plan_token_positions_max": 64,
               "physical_native_suffix_tokens_upper_bound": 1024,
               "physical_native_suffix_token_units": "32 logical futures * 32-token cap",
               "physical_native_prefill_calls": 2,
               "native_discovery_calls": 2},
    )
    request: dict[str, Any] = {
        "schema": "mdb-qwen-circuit-discovery-request-v1",
        "task_family": "forward", "model": MODEL, "backend": "hf",
        "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "seq_len": 64,
        "needs": {"cuda": True}, "execution_environment": worker_intent(),
        "scheduling_contract": scheduling, "config": config,
        "config_bytes_sha256": sha(raw), "source_hashes": {},
        "runtime_wheels": list(runtime_wheel_names()),
        "resource_basis": resource,
        "workload_geometry": {"model_loads": 1, "cases": 2, "prefills": 2,
                              "logical_futures": 32, "logical_futures_per_case": 16,
                              "plan_token_positions_per_case_max": 32,
                              "total_plan_token_positions_max": 64,
                              "physical_native_suffix_tokens_upper_bound": 1024,
                              "physical_native_suffix_token_units": "32 logical futures * 32-token cap",
                              "physical_native_prefill_calls": 2,
                              "receiver_offsets_per_case": 2,
                              "scales_per_candidate": 2, "candidates_per_receiver": 2,
                              "max_native_operations_per_discovery": 10000,
                              "max_capture_bytes_per_discovery": 256 << 20,
                              "parameter_manifest": config["parameter_manifest"]},
        "selection_policy": "fixed prompts and fixed follow-up text are tokenized into explicit IDs before native discovery; all raw rows retained",
        "queue_note": ("Finite CUDA Qwen2.5-1.5B native-eager circuit discovery: one resident BF16 load, two short cases, "
                       "two receivers and two candidates per receiver, scales 0/.5, own-history suffix and exact reinstall; "
                       "strict preflight, one Beast lease, no retries or nested jobs."),
    }
    source_pairs = [(HERE / WORKER, WORKER), *extras]
    request["source_hashes"] = {name: file_sha(path) for path, name in source_pairs}
    request["request_sha256"] = request_identity(request)
    save(run_dir / "request.json", request)
    payload = run_dir / "payload"
    stage_worker(payload, worker=HERE / WORKER, config=config, request=request,
                 config_bytes=raw, additional_files=extras)
    save(run_dir / "staged-command.json", {"submitted": False,
                                            "request_sha256": request["request_sha256"],
                                            "payload": str(payload)})
    return {"submitted": False, "run_dir": str(run_dir),
            "request_sha256": request["request_sha256"],
            "parameter_manifest_sha256": config["parameter_manifest_sha256"],
            "expected_weight_file_sha256": weight_file_sha,
            "expected_weight_content_sha256": weight_content_sha,
            "runtime_wheels": request["runtime_wheels"]}


def submit(run_dir: Path) -> dict[str, Any]:
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command

    run_dir = run_dir.resolve()
    if (run_dir / "submission.json").exists():
        raise ValueError("qualification has already been admitted")
    request = read(run_dir / "request.json")
    if request_identity(request) != request.get("request_sha256"):
        raise ValueError("staged request identity changed")
    payload = run_dir / "payload"
    seal = read(payload / "payload-seal.json")
    for name, digest in request["source_hashes"].items():
        if seal.get("files", {}).get(name) != digest or file_sha(payload / name) != digest:
            raise ValueError(f"staged payload identity changed: {name}")
    _apply_worker_intent(request["execution_environment"])
    job = guarded_mrun_launch(
        worker_command(WORKER), request=request, model=MODEL, task="forward",
        experiment=EXPERIMENT, payload=payload, prefer_host="beast", gpu=True,
        needs=request["needs"], timeout_s=request["resource_basis"]["timeout_s"],
        retry_on_kill=False, preflight=True, detach=True,
        note=request["queue_note"][:500])
    save(run_dir / "submission.json", {"job_id": str(job),
                                        "request_sha256": request["request_sha256"],
                                        "retry_on_kill": False})
    return {"submitted": True, "job_id": str(job), "run_dir": str(run_dir),
            "request_sha256": request["request_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--parameter-manifest", type=Path,
                        help="full Qwen2.5-1.5B parameter-artifacts.json")
    parser.add_argument("--manifest-from-run", type=Path,
                        help="recover qwen-parameter-artifacts.json from an existing payload/run")
    parser.add_argument("--retrieval-script", type=Path,
                        help="standalone retrieval source; defaults to installed public mdb helper")
    parser.add_argument("--submit", action="store_true", help="stage, then explicitly admit the job")
    parser.add_argument("--submit-staged", action="store_true", help="admit an already staged run")
    parser.add_argument("--collect", metavar="JOB_ID",
                        help="collect and verify the terminal job's bounded raw session archive")
    args = parser.parse_args()
    selected_actions = sum(bool(value) for value in (args.submit, args.submit_staged, args.collect))
    if selected_actions > 1:
        parser.error("choose only one of --submit, --submit-staged or --collect")
    if args.collect:
        result = collect(args.run_dir, args.collect)
    elif args.submit_staged:
        result = submit(args.run_dir)
    else:
        manifest = args.parameter_manifest
        if manifest is not None and args.manifest_from_run is not None:
            parser.error("choose --parameter-manifest or --manifest-from-run")
        if manifest is None and args.manifest_from_run is not None:
            source_run = args.manifest_from_run.resolve()
            candidates = (source_run / "payload/qwen-parameter-artifacts.json",
                          source_run / "payload/parameter-artifacts.json",
                          source_run / "qwen-parameter-artifacts.json",
                          source_run / "parameter-artifacts.json")
            manifest = next((path for path in candidates if path.is_file()), None)
            if manifest is None:
                parser.error("existing run has no recoverable qwen parameter-artifacts.json")
        if manifest is None:
            import os
            selected = os.environ.get("MDB_QWEN_PARAMETER_MANIFEST")
            manifest = Path(selected) if selected else None
        result = stage(args.run_dir, parameter_manifest=manifest,
                       retrieval_script=args.retrieval_script)
        if args.submit:
            result = {**result, **submit(args.run_dir)}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
