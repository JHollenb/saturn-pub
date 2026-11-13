"""Stage, admit, or collect the finite Qwen receiver-free read-weight assay.

The frontend is intentionally model-free.  The default action creates a sealed
workspace payload; ``--submit`` is the explicit mrun admission edge and
``--collect`` only reads the terminal job and verifies its bounded archive.
The data-plane worker is supplied by the parent task as ``worker.py``.  The
previous read-composition and circuit-discovery workers are staged under
explicit support names so helper code remains available without importing a
sibling checkout at runtime.
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
WORKSPACE_ROOT = HERE.parents[4]
ARTIFACT_ROOT = WORKSPACE_ROOT / "artifacts"
DEFAULT_RUN_DIR = ARTIFACT_ROOT / "mdb-read-weight-transport-20261107" / "run-01"
WORKER = "worker.py"
COMPOSITION_SUPPORT_TARGET = "composition_support.py"
QUALIFICATION_SUPPORT_TARGET = "qualification_support.py"
MODEL = "qwen2.5-1.5b"
MODEL_PATH = "/mnt/big/llm-models/Qwen2.5-1.5B"
OUTPUT_ROOT = "/mnt/big/mdb/qualification/read-weight-transport-20261107"
EXPERIMENT = "mdb-qwen-read-weight-transport-20261107"
KNOWN_WEIGHT_FILE_SHA256 = (
    "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"
)
RECOVERED_RESULTS = (
    WORKSPACE_ROOT
    / "artifacts/mdb-circuit-discovery-20261106/trained-recovered-job-760379f49115/results"
)
DEFAULT_PARAMETER_MANIFEST = RECOVERED_RESULTS / "parameter-artifacts.json"
DEFAULT_RETRIEVAL_SCRIPT = RECOVERED_RESULTS / "retrieve_weights.py"
DEFAULT_COMPOSITION_SUPPORT_SOURCE = HERE.parent / "read_composition_20261107" / "worker.py"
DEFAULT_QUALIFICATION_SUPPORT_SOURCE = HERE.parent / "circuit_discovery_20261106" / "worker.py"

REPORT_MARKER = "MDB_READ_WEIGHT_TRANSPORT_REPORT_V1 "
ARCHIVE_MARKER = "MDB_READ_WEIGHT_TRANSPORT_ARCHIVE_V1 "
MAX_EXPANDED_BYTES = 2 << 30
MAX_ARCHIVE_BYTES = 512 << 20
MAX_ARCHIVE_FILES = 262144


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
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc


def _workspace_artifact(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    try:
        candidate.relative_to(ARTIFACT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"staging and collection must remain under workspace artifacts: {candidate}"
        ) from exc
    return candidate


def worker_intent() -> dict[str, Any]:
    """Resolve the public intent-first environment for the Beast worker."""

    os.environ.setdefault("MDB_WORKER_UV_CACHE_DIR", "/mnt/big/llm-models/.uv-cache")
    os.environ.setdefault("MDB_WORKER_TORCH_HOME", "/mnt/big/llm-models/.torch-cache")
    os.environ.setdefault("MDB_WORKER_MODEL_ROOTS", "/mnt/big/llm-models")
    from mdb_lanes.execution_environment import intent

    return intent()


def _apply_worker_intent(environment: dict[str, Any]) -> None:
    os.environ["MDB_WORKER_UV_CACHE_DIR"] = str(environment["uv_cache_dir"])
    os.environ["MDB_WORKER_TORCH_HOME"] = str(environment["torch_home"])
    os.environ["MDB_WORKER_MODEL_ROOTS"] = str(environment["extra_model_roots"])
    os.environ["MDB_WORKER_PYTHON"] = str(environment["python"])


def _decode_report(logs: str) -> dict[str, Any] | None:
    for line in reversed(logs.splitlines()):
        if not line.startswith(REPORT_MARKER):
            continue
        try:
            raw = base64.b64decode(line[len(REPORT_MARKER) :].strip(), validate=True)
            value = json.loads(gzip.decompress(raw))
        except (ValueError, TypeError, OSError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
            raise ValueError("read-weight-transport report marker is malformed") from exc
        if not isinstance(value, dict):
            raise ValueError("read-weight-transport report marker is not an object")
        return value
    return None


def _collect_archive(logs: str, custody: dict[str, Any]) -> bytes:
    chunks: dict[int, bytes] = {}
    expected: int | None = None
    for line in logs.splitlines():
        if not line.startswith(ARCHIVE_MARKER):
            continue
        try:
            header, encoded = line[len(ARCHIVE_MARKER) :].split(" ", 1)
            index_text, count_text = header.split("/", 1)
            index, count = int(index_text), int(count_text)
            chunk = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("read-weight-transport archive marker is malformed") from exc
        if count <= 0 or not 0 <= index < count:
            raise ValueError("read-weight-transport archive chunk geometry is invalid")
        if expected is not None and expected != count:
            raise ValueError("read-weight-transport archive chunk counts disagree")
        if index in chunks:
            raise ValueError("duplicate read-weight-transport archive chunk")
        expected = count
        chunks[index] = chunk
    if expected is None or set(chunks) != set(range(expected)):
        raise ValueError("read-weight-transport archive marker stream is incomplete")
    raw = b"".join(chunks[index] for index in range(expected))
    if len(raw) != int(custody.get("archive_bytes", -1)):
        raise ValueError("read-weight-transport archive byte count differs")
    if len(raw) > MAX_ARCHIVE_BYTES or sha(raw) != custody.get("archive_sha256"):
        raise ValueError("read-weight-transport archive SHA or size differs")
    return raw


def _verify_extracted(root: Path) -> dict[str, Any]:
    manifest_path = root / "output-manifest.json"
    manifest = read(manifest_path)
    if manifest.get("schema") not in {
        "mdb-read-weight-transport-output-manifest-v1",
        # The composition and discovery helpers may emit their own bounded
        # support inventories; the worker is responsible for identifying them
        # in the report while this collector preserves their custody.
        "mdb-read-composition-output-manifest-v1",
        # The support worker is intentionally reused from the prior
        # qualification.  Its raw inventory has the same fields and custody
        # semantics, but retains the historical schema name.
        "mdb-circuit-discovery-output-manifest-v1",
    }:
        raise ValueError("unsupported read-weight-transport output manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) > MAX_ARCHIVE_FILES:
        raise ValueError("read-weight-transport output manifest file list is invalid")
    expected_paths: set[str] = set()
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("read-weight-transport output manifest entry is invalid")
        relative = Path(entry["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in expected_paths
        ):
            raise ValueError("unsafe or duplicate read-weight-transport result path")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(entry["bytes"])
            or file_sha(path) != entry["sha256"]
        ):
            raise ValueError(f"read-weight-transport result differs: {relative}")
        expected_paths.add(relative.as_posix())
        total += path.stat().st_size
    if total != int(manifest.get("expanded_bytes", -1)) or total > MAX_EXPANDED_BYTES:
        raise ValueError("read-weight-transport expanded byte count differs")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != expected_paths | {"output-manifest.json"}:
        raise ValueError("read-weight-transport result has unexpected or missing files")
    return manifest


def collect(run_dir: Path, job_id: str, *, log_storage: str = "plain") -> dict[str, Any]:
    """Collect one direct terminal job and verify its bounded raw archive."""

    from mdb_lanes.job_support import terminal_job

    run_dir = _workspace_artifact(run_dir)
    request = read(run_dir / "request.json")
    terminal_id, job, logs = terminal_job(str(job_id))
    if str(terminal_id) != str(job_id):
        raise ValueError("collector received a retry lineage instead of the admitted job")
    if log_storage == "gzip":
        (run_dir / "worker.log.gz").write_bytes(gzip.compress(logs.encode(), mtime=0))
    elif log_storage == "plain":
        (run_dir / "worker.log").write_text(logs, encoding="utf-8")
    else:
        raise ValueError("log storage must be plain or gzip")
    report = _decode_report(logs)
    if report is None:
        save(run_dir / "job.json", job)
        return {"job_id": str(job_id), "state": job.get("state"), "report": False}
    if report.get("schema") != "mdb-read-weight-transport-report-v1":
        raise ValueError("read-weight-transport report schema differs")
    if (
        report.get("job_id") != str(job_id)
        or report.get("request_sha256") != request.get("request_sha256")
    ):
        raise ValueError("read-weight-transport report/job/request binding differs")
    custody = report.get("artifact_custody")
    if not isinstance(custody, dict) or custody.get("schema") not in {
        "mdb-read-weight-transport-custody-v1",
        # Partial support-worker reports retain the prior custody schema while
        # carrying the same archive/hash/manifest fields.
        "mdb-read-composition-custody-v1",
        "mdb-circuit-discovery-custody-v1",
    }:
        raise ValueError("read-weight-transport report lacks bounded artifact custody")
    raw = _collect_archive(logs, custody)
    archive_path = run_dir / "result.tar.gz"
    if archive_path.exists():
        raise ValueError("result archive already collected")
    archive_path.write_bytes(raw)
    extracted = run_dir / "results"
    extracted.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_FILES or sum(m.size for m in members) > MAX_EXPANDED_BYTES:
            raise ValueError("read-weight-transport archive exceeds custody bounds")
        for member in members:
            relative = Path(member.name)
            if (
                not member.isfile()
                or member.issym()
                or member.islnk()
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                raise ValueError("unsafe read-weight-transport archive member")
            destination = extracted / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("read-weight-transport archive member has no data")
                shutil.copyfileobj(source, stream, length=1 << 20)
    manifest = _verify_extracted(extracted)
    save(run_dir / "report.json", report)
    save(
        run_dir / "archive-custody.json",
        {
            "schema": "mdb-read-weight-transport-collected-custody-v1",
            "job_id": str(job_id),
            "request_sha256": request["request_sha256"],
            "archive": "result.tar.gz",
            "archive_sha256": sha(raw),
            "archive_bytes": len(raw),
            "manifest": "results/output-manifest.json",
            "manifest_sha256": file_sha(extracted / "output-manifest.json"),
            "file_count": manifest["file_count"],
            "expanded_bytes": manifest["expanded_bytes"],
        },
    )
    save(run_dir / "job.json", job)
    return {
        "job_id": str(job_id),
        "state": job.get("state"),
        "report": str(run_dir / "report.json"),
        "results": str(extracted),
        "archive": str(archive_path),
    }


def request_identity(request: dict[str, Any]) -> str:
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    # Keep this identical to the support worker's public request binding.  The
    # mrun seam separately computes its compact canonical launch fingerprint.
    return sha(json.dumps(body, sort_keys=True, allow_nan=False).encode())


def _fresh_cases() -> list[dict[str, Any]]:
    """Return the four fixed fresh contexts used by the transport worker.

    This is deliberately pure and torch-free.  The worker validates the exact
    dictionaries from its own constructor before loading the model, so a
    staged plan can be inspected without importing a model framework.
    """

    rows: list[dict[str, Any]] = []
    for a, b, x, y in (("Mira", "Daniel", "key", "coin"),
                        ("Zoe", "Liam", "book", "cup")):
        for previous in (a, b):
            rows.append({
                "id": f"{a.lower()}-{previous.lower()}",
                "owners": [a, b],
                "objects": [x, y],
                "previous_owner": previous,
                "prompt": (
                    f"{a} placed the {x} in the box. {b} placed the {y} in the bag. "
                    f"The object {previous} placed is the"
                ),
                "followups": [
                    f" The object {a} placed is the",
                    f" The object {b} placed is the",
                ],
                "previous_current_owner_relation": "same" if previous == a else "other",
            })
    return rows


def plan() -> dict[str, Any]:
    """Return the fixed finite assay plan consumed by the data-plane worker.

    The compiler is intentionally a direction-only, baseline-00 construction.
    Native A-receiver observations are post-seal measurements and are not
    exposed to the compiler.  The plan therefore does not claim closure of an
    A-modified output or a serving speedup.
    """

    return {
        "schema": "mdb-read-weight-transport-plan-v1",
        "model": MODEL,
        "model_path": MODEL_PATH,
        "public_module": "mdb_lanes.read_weight_transport",
        "family": "qwen2",
        "device": "cuda",
        "dtype": "bfloat16",
        "attention_backend": "native-eager",
        "seed": 20261107,
        "output_root": OUTPUT_ROOT,
        "parameter_manifest": "parameter-artifacts.json",
        "retrieval_script": "retrieve_weights.py",
        "max_logical_branches": 100,
        "max_logical_operations": 200,
        "max_native_calls": 300,
        "max_native_token_transitions": 300,
        "physical_batch_size": 1,
        "prefill_calls": 4,
        "continuation_tokens": 2,
        "doses": [0.0, 0.25, 0.5, 1.0],
        "custody": {
            "max_expanded_bytes": MAX_EXPANDED_BYTES,
            "max_archive_bytes": MAX_ARCHIVE_BYTES,
            "max_files": MAX_ARCHIVE_FILES,
        },
        "weight_transport": {
            "sites": {
                "A": {
                    "layer": 19,
                    "kv_head": 0,
                    "row_role": "preceding-answer-first-generated",
                },
                "B": {
                    "layer": 22,
                    "kv_head": 0,
                    "row_role": "current-followup-copula",
                },
            },
            "doses": [0.0, 0.25, 0.5, 1.0],
            "compiler": {
                "kind": "smooth-fp32-jvp",
                "source": "baseline-00-layer-19",
                "receiver": "layer-22",
                "logical_linears": 19,
                "tangent": True,
                "dtype": "float32",
                "invocations": 8,
                "local_source_o19_projections_per_receiver": 4,
                "dense_fp32_control_maps_after_weight_seal_per_receiver": 19,
                "extra_mlp22_forecast_linears": 12,
                "prediction_scope": "direction-only-per-receiver",
            },
            "baseline_visibility": {
                "allowed": [
                    "full unedited native-00 tape through layer 22, including q/k/scores22 and history layers 20/21",
                    "declared A source write and dose",
                    "public Qwen2.5-1.5B parameter artifacts",
                ],
                "forbidden_until_seal": [
                    "native A receiver q/k/weights/attention",
                    "AB receiver outputs",
                    "held-out receiver outcomes",
                ],
            },
            "native_scoring": "after-predictions-sealed",
            "query_key_control": "saved native 00 and A receiver observations remain diagnostics",
            "history_source": "transport-prediction branch end cut",
            "semantic_cot_qualified": False,
            "baseline_a_output_closure": False,
            "speed_claim": False,
        },
        "cases": _fresh_cases(),
        "limits": [
            "Four genuinely fresh lexical-binding contexts: Mira/Daniel key/coin and Zoe/Liam book/cup, each previous owner.",
            "A and B are explicit physical addresses; role labels locate rows and do not assert semantic content.",
            "The smooth FP32 JVP reads the full unedited native-00 tape through layer 22 before native A-receiver scoring.",
            "All four dose points are retained; no effect-size threshold is a mechanics gate.",
            "Two native continuation tokens are generated per branch; generated tokens remain branch-owned.",
            "The second receiver is reached from the transport-prediction branch end cut, not an AB-world shortcut.",
            "One resident CUDA BF16 model and one finite outer lease; no retries or nested submissions.",
            "This assay establishes only a baseline-only direction compilation per receiver; it does not claim A-output closure or speed.",
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
        verification = "deferred-to-beast-worker"
    return manifest, source_hashes, content_digest(source_hashes), verification


def stage(
    run_dir: Path,
    *,
    parameter_manifest: Path | None = None,
    retrieval_script: Path | None = None,
    composition_support_source: Path | None = None,
    qualification_support_source: Path | None = None,
) -> dict[str, Any]:
    """Create a sealed payload in the workspace artifact tree."""

    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import runtime_wheel_names, stage_worker

    run_dir = _workspace_artifact(run_dir)
    if run_dir.exists():
        raise ValueError(f"staging destination already exists: {run_dir}")
    worker_path = HERE / WORKER
    if not worker_path.is_file():
        raise ValueError(f"parent worker source is not present: {worker_path}")
    composition_support = (
        composition_support_source or DEFAULT_COMPOSITION_SUPPORT_SOURCE
    ).resolve()
    qualification_support = (
        qualification_support_source or DEFAULT_QUALIFICATION_SUPPORT_SOURCE
    ).resolve()
    if not composition_support.is_file():
        raise ValueError(
            f"explicit composition support source does not exist: {composition_support}"
        )
    if not qualification_support.is_file():
        raise ValueError(
            f"explicit qualification support source does not exist: {qualification_support}"
        )
    parameter_manifest = (parameter_manifest or DEFAULT_PARAMETER_MANIFEST).resolve()
    retrieval_script = (retrieval_script or DEFAULT_RETRIEVAL_SCRIPT).resolve()
    if not parameter_manifest.is_file():
        raise ValueError(f"parameter manifest does not exist: {parameter_manifest}")
    if not retrieval_script.is_file():
        raise ValueError(f"standalone retrieval source does not exist: {retrieval_script}")

    manifest, source_hashes, weight_content_sha, checkpoint_verification = _manifest_identity(
        parameter_manifest, MODEL_PATH
    )
    weight_file_sha = source_hashes.get("model.safetensors", KNOWN_WEIGHT_FILE_SHA256)
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(parameter_manifest, run_dir / "parameter-artifacts.json")
    shutil.copyfile(retrieval_script, run_dir / "retrieve_weights.py")
    config = plan()
    config.update(
        {
            "expected_weight_file_sha256": weight_file_sha,
            "expected_weight_content_sha256": weight_content_sha,
            "parameter_manifest_sha256": file_sha(run_dir / "parameter-artifacts.json"),
            "retrieval_script_sha256": file_sha(run_dir / "retrieve_weights.py"),
            "composition_support_sha256": file_sha(composition_support),
            "qualification_support_sha256": file_sha(qualification_support),
        }
    )
    save(run_dir / "plan.json", config)
    binding = {
        "schema": "mdb-read-weight-transport-source-binding-v1",
        "model": MODEL,
        "model_path": MODEL_PATH,
        "family": "qwen2",
        "parameter_manifest": "parameter-artifacts.json",
        "parameter_manifest_sha256": config["parameter_manifest_sha256"],
        "retrieval_script": "retrieve_weights.py",
        "retrieval_script_sha256": config["retrieval_script_sha256"],
        "checkpoint_source_files_sha256": source_hashes,
        "expected_weight_file_sha256": weight_file_sha,
        "expected_weight_content_sha256": weight_content_sha,
        "artifact_count": len(manifest["artifacts"]),
        "checkpoint_source_verification": checkpoint_verification,
        "support_sources": {
            COMPOSITION_SUPPORT_TARGET: config["composition_support_sha256"],
            QUALIFICATION_SUPPORT_TARGET: config["qualification_support_sha256"],
        },
        "binding_scope": "full registered Qwen2.5-1.5B safetensors checkpoint",
    }
    save(run_dir / "source-binding.json", binding)
    preflight = run_dir / "preflight.v1.json"
    save(
        preflight,
        {
            "schema_version": "mrun-preflight-config-v1",
            "hashed_roots": [
                WORKER,
                "launcher.py",
                COMPOSITION_SUPPORT_TARGET,
                QUALIFICATION_SUPPORT_TARGET,
                "plan.json",
                "source-binding.json",
                "retrieve_weights.py",
                "parameter-artifacts.json",
                "preflight.v1.json",
            ],
            "path_token_allowlist": [
                MODEL_PATH,
                OUTPUT_ROOT,
                "parameter-artifacts.json",
                "retrieve_weights.py",
                COMPOSITION_SUPPORT_TARGET,
                QUALIFICATION_SUPPORT_TARGET,
            ],
        },
    )
    raw = (json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n").encode()
    extras = [
        (HERE / "launch.py", "launcher.py"),
        (composition_support, COMPOSITION_SUPPORT_TARGET),
        (qualification_support, QUALIFICATION_SUPPORT_TARGET),
        (run_dir / "plan.json", "plan.json"),
        (run_dir / "source-binding.json", "source-binding.json"),
        (run_dir / "retrieve_weights.py", "retrieve_weights.py"),
        (run_dir / "parameter-artifacts.json", "parameter-artifacts.json"),
        (preflight, "preflight.v1.json"),
    ]
    scheduling = one_lease_scheduling_contract(
        model_loads=1,
        logical_branches=100,
        logical_operations=200,
        logical_checkpoint_replays=0,
        max_local_work_items=200,
        checkpoint_policy=(
            "one resident Qwen2.5-1.5B BF16 CUDA native-eager model; four genuinely fresh "
            "lexical-binding cases; baseline-only JVP predictions are sealed before native "
            "A-receiver scoring; "
            "one outer lease, no retries or nested submissions"
        ),
        extra={
            "cases": 4,
            "prefills": 4,
            "physical_batch_size": 1,
            "continuation_tokens": 2,
            "doses": [0.0, 0.25, 0.5, 1.0],
            "max_native_token_transitions": 300,
            "max_native_calls": 300,
            "physical_native_prefill_calls": 4,
            "physical_native_token_transitions_upper_bound": 300,
            "logical_future_units": "one bounded smooth FP32 JVP compilation per receiver; scalar evaluations per dose",
            "compiler_invocations": 8,
            "direction_compiler_invocations": 8,
        },
    )
    resource_basis = {
        "timeout_s": 900,
        "calibration": (
            "intent-first CUDA planning under strict preflight; one resident model, physical batch one; "
            "no hand-guessed RAM/VRAM reservation"
        ),
    }
    request: dict[str, Any] = {
        "schema": "mdb-read-weight-transport-request-v1",
        "task_family": "forward",
        "model": MODEL,
        "backend": "hf",
        "device": "cuda",
        "dtype": "bfloat16",
        "batch_size": 1,
        "seq_len": 128,
        "needs": {"cuda": True},
        "execution_environment": worker_intent(),
        "scheduling_contract": scheduling,
        "config": config,
        "config_bytes_sha256": sha(raw),
        "source_hashes": {},
        "runtime_wheels": list(runtime_wheel_names()),
        "resource_basis": resource_basis,
        "workload_geometry": {
            "model_loads": 1,
            "cases": 4,
            "prefills": 4,
            "physical_batch_size": 1,
            "max_logical_branches": 100,
            "max_logical_operations": 200,
            "max_native_calls": 300,
            "max_native_token_transitions": 300,
            "continuation_tokens": 2,
            "doses": [0.0, 0.25, 0.5, 1.0],
            "sites": {"source": [19, 0], "receiver": [22, 0]},
            "compiler": "smooth-fp32-jvp",
            "logical_linears": 19,
            "tangent": True,
            "native_scoring": "after-predictions-sealed",
            "controls": ["00", "dose-0", "sham", "reinstall", "stale", "wrong-sign", "query-only"],
            "compiler_invocations": 8,
            "max_expanded_bytes": MAX_EXPANDED_BYTES,
            "max_archive_bytes": MAX_ARCHIVE_BYTES,
            "max_archive_files": MAX_ARCHIVE_FILES,
            "parameter_manifest": "parameter-artifacts.json",
        },
        "selection_policy": (
            "fixed prompt/follow-up text is tokenized into explicit IDs before execution; "
            "source/receiver addresses are bound to observed cuts and hashes; generated tokens remain native-owned"
        ),
        "queue_note": (
            "Finite CUDA Qwen2.5-1.5B native-eager read-weight transport assay: one resident BF16 model, "
            "four genuinely fresh lexical-binding contexts, baseline-only smooth FP32 JVP from layer 19 "
            "to layer 22 across four doses, predictions sealed before native receiver scoring, strict "
            "preflight, one Beast lease, no retries or nested jobs."
        ),
    }
    source_pairs = [(worker_path, WORKER), *extras]
    request["source_hashes"] = {name: file_sha(path) for path, name in source_pairs}
    request["request_sha256"] = request_identity(request)
    save(run_dir / "request.json", request)
    payload = run_dir / "payload"
    stage_worker(
        payload,
        worker=worker_path,
        config=config,
        request=request,
        config_bytes=raw,
        additional_files=extras,
    )
    save(
        run_dir / "staged-command.json",
        {
            "submitted": False,
            "request_sha256": request["request_sha256"],
            "payload": str(payload),
            "parameter_manifest": str(parameter_manifest),
            "retrieval_script": str(retrieval_script),
            "composition_support_source": str(composition_support),
            "qualification_support_source": str(qualification_support),
        },
    )
    return {
        "submitted": False,
        "run_dir": str(run_dir),
        "request_sha256": request["request_sha256"],
        "parameter_manifest_sha256": config["parameter_manifest_sha256"],
        "retrieval_script_sha256": config["retrieval_script_sha256"],
        "composition_support_sha256": config["composition_support_sha256"],
        "qualification_support_sha256": config["qualification_support_sha256"],
        "expected_weight_file_sha256": weight_file_sha,
        "expected_weight_content_sha256": weight_content_sha,
        "runtime_wheels": request["runtime_wheels"],
    }


def submit(run_dir: Path) -> dict[str, Any]:
    """Submit one already sealed payload through the guarded public seam."""

    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command

    run_dir = _workspace_artifact(run_dir)
    if (run_dir / "submission.json").exists():
        raise ValueError("read-weight-transport assay has already been admitted")
    request = read(run_dir / "request.json")
    if request_identity(request) != request.get("request_sha256"):
        raise ValueError("staged read-weight-transport request identity changed")
    payload = run_dir / "payload"
    seal = read(payload / "payload-seal.json")
    for name, digest in request["source_hashes"].items():
        if seal.get("files", {}).get(name) != digest or file_sha(payload / name) != digest:
            raise ValueError(f"staged read-weight-transport payload identity changed: {name}")
    _apply_worker_intent(request["execution_environment"])
    job_id = guarded_mrun_launch(
        worker_command(WORKER),
        request=request,
        model=MODEL,
        task="forward",
        experiment=EXPERIMENT,
        payload=payload,
        prefer_host="beast",
        gpu=True,
        needs=request["needs"],
        timeout_s=request["resource_basis"]["timeout_s"],
        retry_on_kill=False,
        preflight=True,
        detach=True,
        note=request["queue_note"][:500],
    )
    save(
        run_dir / "submission.json",
        {"job_id": str(job_id), "request_sha256": request["request_sha256"], "retry_on_kill": False},
    )
    return {
        "submitted": True,
        "job_id": str(job_id),
        "run_dir": str(run_dir),
        "request_sha256": request["request_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=DEFAULT_RUN_DIR)
    parser.add_argument("--parameter-manifest", type=Path, default=None)
    parser.add_argument("--retrieval-script", type=Path, default=None)
    parser.add_argument("--composition-support-source", type=Path, default=None)
    parser.add_argument("--qualification-support-source", type=Path, default=None)
    parser.add_argument("--submit", action="store_true", help="stage if needed, then explicitly admit")
    parser.add_argument("--collect", metavar="JOB_ID", help="collect and verify a direct terminal job")
    parser.add_argument("--gzip-log", action="store_true", help="store collected worker logs compressed")
    args = parser.parse_args()
    if args.submit and args.collect:
        parser.error("choose only one of --submit or --collect")
    if args.collect:
        result = collect(args.run_dir, args.collect, log_storage="gzip" if args.gzip_log else "plain")
    elif args.submit:
        run_dir = _workspace_artifact(args.run_dir)
        result = submit(run_dir) if run_dir.exists() else stage(
            run_dir,
            parameter_manifest=args.parameter_manifest,
            retrieval_script=args.retrieval_script,
            composition_support_source=args.composition_support_source,
            qualification_support_source=args.qualification_support_source,
        )
        if not result.get("submitted"):
            result = {**result, **submit(run_dir)}
    else:
        result = stage(
            args.run_dir,
            parameter_manifest=args.parameter_manifest,
            retrieval_script=args.retrieval_script,
            composition_support_source=args.composition_support_source,
            qualification_support_source=args.qualification_support_source,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
