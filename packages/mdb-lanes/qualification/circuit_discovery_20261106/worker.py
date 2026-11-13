"""Run two finite trained-Qwen native circuit-discovery qualification cases.

The worker is intentionally a data-plane program.  It loads one registered
Qwen2.5-1.5B resident through the public ``mdb``/``mrun`` boundary, opens a
journal-mode BenchSession with the explicit native-eager backend, and keeps
all discovery branches inside that one lease.  No scheduler or subprocess is
used from the worker.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import struct
import tarfile
import time
from pathlib import Path
from typing import Any

MODEL = "qwen2.5-1.5b"
MODEL_PATH = "/mnt/big/llm-models/Qwen2.5-1.5B"
REPORT_SCHEMA = "mdb-qwen-circuit-discovery-report-v1"
# This is the historical raw SHA256 of the one-file Qwen checkpoint.  Keep the
# raw file binding separate from the canonical digest of the safetensors map:
# the former is the published trained-MDB identity, while the latter is the
# portable multi-file identity used by parameter-artifact provenance.
KNOWN_WEIGHT_FILE_SHA256 = "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"
REPORT_MARKER = "MDB_CIRCUIT_DISCOVERY_REPORT_V1 "
ARCHIVE_MARKER = "MDB_CIRCUIT_DISCOVERY_ARCHIVE_V1 "
MAX_EXPANDED_BYTES = 512 << 20
MAX_ARCHIVE_BYTES = 128 << 20
ARCHIVE_CHUNK_BYTES = 512 << 10
# The first trained assay produced 10,896 files / 171,104,411 bytes. Source-frame
# receipts count toward custody even when their tensors share payload bytes.
MAX_ARCHIVE_FILES = 65536


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise RuntimeError(f"truncated safetensors header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 64 << 20:
            raise RuntimeError(f"safetensors header exceeds bound: {path}")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise RuntimeError(f"truncated safetensors header payload: {path}")
    value = json.loads(raw_header)
    if not isinstance(value, dict):
        raise RuntimeError(f"safetensors header is not an object: {path}")
    return value


def _native_dtype(dtype: str) -> tuple[str, str, int]:
    # parameter_artifacts intentionally admits only lossless native words.
    values = {"BF16": ("BF16", "<u2", 2), "F32": ("F32", "<f4", 4),
              "F64": ("F64", "<f8", 8), "I64": ("I64", "<i8", 8)}
    try:
        return values[dtype]
    except KeyError as exc:
        raise RuntimeError(f"checkpoint tensor dtype is not an admitted native artifact: {dtype}") from exc


def _npy_prefix(dtype: str, shape: list[int]) -> bytes:
    import numpy as np

    _native, descr, _width = _native_dtype(dtype)
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {
        "descr": descr, "fortran_order": False, "shape": tuple(shape)})
    return stream.getvalue()


def _payload_hash(path: Path, start: int, stop: int, prefix: bytes) -> tuple[str, str]:
    digest = hashlib.sha256()
    artifact = hashlib.sha256()
    artifact.update(prefix)
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = stop - start
        while remaining:
            block = stream.read(min(8 << 20, remaining))
            if not block:
                raise RuntimeError(f"truncated safetensors payload: {path}:{start}:{stop}")
            digest.update(block)
            artifact.update(block)
            remaining -= len(block)
    return digest.hexdigest(), artifact.hexdigest()


def _write_parameter_manifest(checkpoint: Path, output: Path) -> dict[str, Any]:
    """Create a full native parameter recipe before loading the model.

    Only safetensors headers, immutable source hashes, and per-tensor payload
    hashes are retained.  No tensor copy is written.  ``retrieve_weights.py``
    remains the standalone public retriever for later exact materialization.
    """
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no safetensors found under {checkpoint}")
    artifacts: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for file_index, path in enumerate(files):
        file_hash = _file_sha(path)
        source_hashes[path.name] = file_hash
        header = _safetensors_header(path)
        with path.open("rb") as stream:
            raw_length = struct.unpack("<Q", stream.read(8))[0]
        payload_base = 8 + raw_length
        for tensor_index, (tensor_name, entry) in enumerate(sorted(header.items())):
            if tensor_name == "__metadata__":
                continue
            if not isinstance(entry, dict):
                raise RuntimeError(f"malformed safetensors tensor entry: {tensor_name}")
            dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
            if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                raise RuntimeError(f"malformed safetensors tensor geometry: {tensor_name}")
            native, _descr, width = _native_dtype(dtype)
            if any(type(dimension) is not int or dimension < 0 for dimension in shape):
                raise RuntimeError(f"invalid tensor shape: {tensor_name}")
            elements = 1
            for dimension in shape:
                elements *= dimension
            start, stop = (int(offsets[0]), int(offsets[1]))
            if start < 0 or stop < start or stop - start != elements * width:
                raise RuntimeError(f"invalid tensor data offsets: {tensor_name}")
            absolute_start, absolute_stop = payload_base + start, payload_base + stop
            if absolute_stop > path.stat().st_size:
                raise RuntimeError(f"tensor payload escapes source file: {tensor_name}")
            prefix = _npy_prefix(native, [int(dimension) for dimension in shape])
            payload_sha, artifact_sha = _payload_hash(path, absolute_start, absolute_stop, prefix)
            safe_file = re.sub(r"[^A-Za-z0-9_.-]", "_", path.name)
            safe_tensor = re.sub(r"[^A-Za-z0-9_.-]", "_", tensor_name)
            artifact_path = f"parameters/{file_index:03d}-{safe_file}/{tensor_index:04d}-{safe_tensor}.npy"
            artifacts.append({"path": artifact_path, "bytes": len(prefix) + stop - start,
                              "sha256": artifact_sha,
                              "npy_prefix_b64": base64.b64encode(prefix).decode("ascii"),
                              "source": {"file": path.name, "file_sha256": file_hash,
                                         "tensor": tensor_name, "dtype": native,
                                         "shape": [int(dimension) for dimension in shape],
                                         "data_offsets": [start, stop],
                                         "payload_sha256": payload_sha}})
    manifest = {"schema": "mdb-parameter-artifacts-v1", "defaults": {
        "artifact_root": ".", "checkpoint_root": str(checkpoint.resolve()),
        "python": "python3"}, "artifacts": artifacts,
        "provenance": {"generator": "circuit-discovery worker", "source_files_sha256": source_hashes,
        "weight_files_sha256": source_hashes,
        "weight_content_sha256": __import__("mdb_lanes.model_os_contracts", fromlist=["content_digest"]).content_digest(source_hashes),
                        "tensor_payloads_only": True}}
    _write(output, manifest)
    return manifest


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_output_manifest(root: Path) -> dict[str, Any]:
    """Inventory raw session custody without retaining pretrained weight copies."""
    files = []
    expanded_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            if path.is_symlink():
                raise RuntimeError(f"result contains a symbolic link: {path}")
            continue
        if path.name in {"report.json", "output-manifest.json", "result.tar.gz"}:
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        expanded_bytes += size
        if len(files) >= MAX_ARCHIVE_FILES or expanded_bytes > MAX_EXPANDED_BYTES:
            raise RuntimeError("raw result custody exceeds finite archive geometry")
        files.append({"path": relative, "bytes": size, "sha256": _file_sha(path)})
    manifest = {"schema": "mdb-circuit-discovery-output-manifest-v1",
                "scope": "raw native session closure, report inputs, parameter recipe and standalone retriever",
                "files": files, "expanded_bytes": expanded_bytes,
                "file_count": len(files), "pretrained_weight_copies": False}
    _write(root / "output-manifest.json", manifest)
    return manifest


def _archive_results(root: Path) -> tuple[Path, str, int]:
    """Archive raw outputs for direct mrun collection with a bounded marker stream."""
    archive_path = root / "result.tar.gz"
    with tarfile.open(archive_path, mode="w:gz", compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"result archive contains a symbolic link: {path}")
            if not path.is_file() or path == archive_path or path.name == "report.json":
                continue
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    size = archive_path.stat().st_size
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise RuntimeError(f"result archive exceeds bounded custody size: {size} bytes")
    return archive_path, _file_sha(archive_path), size


def _emit_results(root: Path, report: dict[str, Any]) -> None:
    archive_path = root / "result.tar.gz"
    encoded = gzip.compress(json.dumps(report, sort_keys=True, allow_nan=False).encode("utf-8"), mtime=0)
    print(REPORT_MARKER + base64.b64encode(encoded).decode("ascii"), flush=True)
    with archive_path.open("rb") as stream:
        chunks = []
        while True:
            chunk = stream.read(ARCHIVE_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(base64.b64encode(chunk).decode("ascii"))
    for index, chunk in enumerate(chunks):
        print(f"{ARCHIVE_MARKER}{index}/{len(chunks)} {chunk}", flush=True)
    print("MDB_CIRCUIT_DISCOVERY_READY " + json.dumps({
        "job_id": report["job_id"], "result_root": str(root),
        "manifest": str(root / "output-manifest.json"), "archive": str(archive_path),
        "archive_sha256": report["artifact_custody"]["archive_sha256"],
        "archive_bytes": report["artifact_custody"]["archive_bytes"],
    }, sort_keys=True), flush=True)


def _request_identity(request: dict[str, Any]) -> str:
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    return _sha(json.dumps(body, sort_keys=True).encode("utf-8"))


def _safe(value: Any) -> Any:
    """Make a bounded report copy without leaking live tensors or Paths."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        try:
            from mdb_lanes.bench import tensor_digest

            return {"tensor_sha256": tensor_digest(value), "shape": list(value.shape),
                    "dtype": str(value.dtype)}
        except Exception:
            return {"tensor": type(value).__name__, "shape": list(getattr(value, "shape", ())),
                    "dtype": str(getattr(value, "dtype", "unknown"))}
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except Exception:
            pass
    return repr(value)


def _single_token(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list):
        ids = list(ids)
    result = [int(token) for token in ids]
    if not result:
        raise ValueError(f"text encoded to no token IDs: {text!r}")
    return result


def _margin_observable(tokenizer: Any, left: str, right: str) -> dict[str, Any] | None:
    left_ids, right_ids = _single_token(tokenizer, left), _single_token(tokenizer, right)
    if len(left_ids) == len(right_ids) == 1 and left_ids[0] != right_ids[0]:
        return {"kind": "token_margin", "token_id": left_ids[0], "against_token_id": right_ids[0]}
    return None


def _case_spec(case: dict[str, Any], tokenizer: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_ids = _single_token(tokenizer, str(case["prompt"]))
    followup_ids = _single_token(tokenizer, str(case["followup"]))
    if len(prompt_ids) < 2:
        raise ValueError(f"{case['id']} prompt needs at least two tokens")
    plan = [
        {"kind": "input", "token_ids": [prompt_ids[-1]]},
        {"kind": "generate", "count": 2},
        {"kind": "input", "token_ids": followup_ids},
        {"kind": "generate", "count": 2},
    ]
    transitions = 1 + 2 + len(followup_ids) + 2
    if transitions > 32:
        raise ValueError(f"{case['id']} discovery plan exceeds 32 token transitions")
    # Offset 0 is the first answer decision.  The later receiver is the last
    # caller-supplied follow-up token, after the two own-history tokens.
    receiver_offsets = [0, 2 + len(followup_ids)]
    spec: dict[str, Any] = {
        "name": str(case["id"]),
        "plan": plan,
        "receiver_offsets": receiver_offsets,
        # The API ranks at most four per receiver.  Two keeps both receiver
        # contexts visible without letting one global top list dominate.
        "candidates_per_receiver": 2,
        "scales": [0.0, 0.5],
        # Cover the complete finite suffix so the first receiver reaches the
        # fixed follow-up and its own native answer.  The runtime still caps
        # this at 32 transitions during validation.
        "continuation_tokens": transitions,
        "top_k": 5,
        "max_native_operations": 10000,
        "max_capture_bytes": 256 << 20,
    }
    if case.get("observable"):
        spec["observable"] = _margin_observable(tokenizer, *case["observable"])
        if spec["observable"] is None:
            spec.pop("observable")
    return spec, {"prompt_ids": prompt_ids, "followup_ids": followup_ids,
                  "receiver_offsets": receiver_offsets, "transition_count": transitions}


def _native_plan(bench: Any, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for segment in plan:
        kind = segment["kind"]
        if kind == "input":
            instructions = [(int(token), "caller-supplied") for token in segment["token_ids"]]
        elif kind == "generate":
            instructions = [(None, "native-greedy") for _ in range(int(segment["count"]))]
        else:
            raise ValueError(f"unknown native plan segment: {kind!r}")
        for supplied, authority in instructions:
            if supplied is not None:
                bench.stage_input(supplied)
            result = bench.step()
            rows.append({"offset": offset, "input_authority": authority,
                         "consumed_token_id": int(result["token_id"]),
                         "next_token_id": int(result["next_token_id"]),
                         "cut_id": result["cut"], "receipt_ref": result["receipt_ref"],
                         "bundle_ref": result.get("bundle_ref"),
                         "logits_sha256": result["logits_sha256"]})
            offset += 1
    return rows


def _signature(rows: list[dict[str, Any]]) -> list[tuple[int, str]]:
    return [(int(row["consumed_token_id"]), str(row["logits_sha256"])) for row in rows]


def _report_contract(result: dict[str, Any], *, expected_offsets: list[int],
                     reinstall_exact: bool) -> dict[str, Any]:
    report = result.get("report")
    if not isinstance(report, dict) or report.get("schema") != "saturn-bench-circuit-discovery-v1":
        raise RuntimeError("discover_circuit returned no qualified report")
    baseline = report.get("baseline")
    receivers = report.get("receivers")
    tests = report.get("tests")
    graph = report.get("graph")
    observer_control = report.get("observer_control", {})
    parity = observer_control.get("effect", {}) if isinstance(observer_control, dict) else {}
    if not isinstance(baseline, dict) or not isinstance(baseline.get("trajectory"), list):
        raise RuntimeError("discover_circuit report lacks baseline trajectory")
    if not isinstance(receivers, list) or not receivers:
        raise RuntimeError("discover_circuit report lacks per-receiver passive readmaps")
    if not isinstance(tests, list) or not tests:
        raise RuntimeError("discover_circuit report lacks intervention tests")
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise RuntimeError("discover_circuit report lacks evidence graph")
    if not parity.get("closure_exact") or not all(row.get("same_logits") for row in parity.get("steps", [])):
        raise RuntimeError("no-op observer parity did not preserve native logits")
    observed_offsets = {int(row.get("offset")) for row in receivers}
    if observed_offsets != set(expected_offsets):
        raise RuntimeError(f"receiver offsets differ: {observed_offsets!r} != {set(expected_offsets)!r}")
    candidate_count = 0
    reinstall_count = 0
    local_prediction_count = 0
    own_history_suffix = False
    for receiver in receivers:
        if not isinstance(receiver.get("coverage"), list) or not receiver["coverage"]:
            raise RuntimeError("receiver has no passive native readmap coverage")
        candidates = receiver.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("receiver has no selected candidate")
        candidate_count += len(candidates)
        for candidate in candidates:
            reinstall = candidate.get("reinstall", {})
            effect = reinstall.get("effect", {}) if isinstance(reinstall, dict) else {}
            if effect.get("closure_exact") and all(row.get("same_logits") for row in effect.get("steps", [])):
                reinstall_count += 1
    for test in tests:
        local_prediction = test.get("local_prediction")
        if not isinstance(local_prediction, dict) or not local_prediction.get("query_and_keys_exact"):
            raise RuntimeError("candidate test lacks exact first-receiver query/key route audit")
        local_prediction_count += 1
        trajectory = test.get("candidate", {}).get("trajectory", [])
        receiver_offset = int(test.get("receiver_offset", -1))
        if any(int(row.get("offset", -1)) > receiver_offset
               and row.get("input_authority") == "native-greedy" for row in trajectory):
            own_history_suffix = True
    if not own_history_suffix:
        raise RuntimeError("discover_circuit report lacks an own-history native suffix")
    if candidate_count != reinstall_count or not reinstall_exact:
        raise RuntimeError("exact reinstall audit did not close every selected candidate")
    return {"status": "valid", "no_op_observer_parity": True,
            "selected_candidates": candidate_count, "exact_reinstall_candidates": reinstall_count,
            "local_write_predictions": local_prediction_count,
            "own_history_native_suffix": True, "evidence_graph": {"nodes": len(graph["nodes"]),
                                                                     "edges": len(graph["edges"])}}


def _parameter_identity(config: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    from mdb_lanes.model_os_contracts import content_digest
    from mdb_lanes.parameter_artifacts import validate_manifest

    manifest = validate_manifest(_read(manifest_path))
    defaults = manifest.get("defaults", {})
    checkpoint_root = str(defaults.get("checkpoint_root", config["model_path"]))
    if Path(checkpoint_root).resolve() != Path(config["model_path"]).resolve():
        raise RuntimeError("parameter manifest checkpoint root differs from the qualified model path")
    source_hashes: dict[str, str] = {}
    for entry in manifest["artifacts"]:
        source = entry["source"]
        filename, expected = str(source["file"]), str(source["file_sha256"])
        if filename in source_hashes and source_hashes[filename] != expected:
            raise RuntimeError(f"parameter manifest has conflicting source hash: {filename}")
        source_hashes[filename] = expected
    for filename, expected in source_hashes.items():
        path = Path(checkpoint_root) / filename
        if not path.is_file() or _file_sha(path) != expected:
            raise RuntimeError(f"parameter source differs from manifest: {path}")
    manifest_sha = _sha(manifest_path.read_bytes())
    expected_manifest_sha = config.get("parameter_manifest_sha256")
    if expected_manifest_sha and manifest_sha != expected_manifest_sha:
        raise RuntimeError("parameter-artifacts.json bytes differ from staged identity")
    content_sha = content_digest(source_hashes)
    expected_content = config.get("expected_weight_content_sha256")
    if expected_content and content_sha != expected_content:
        raise RuntimeError("parameter-artifact source map differs from the sealed identity")
    return {"schema": manifest.get("schema"), "manifest_sha256": manifest_sha,
            "checkpoint_root": checkpoint_root, "source_files_sha256": source_hashes,
            "weight_content_sha256": content_sha,
            "artifact_count": len(manifest["artifacts"]),
            "scope": "all checkpoint safetensors required; source hashes are immutable bindings"}


def _load_model(config: dict[str, Any], parameter: dict[str, Any]):
    import torch
    from mrun.models import find_safetensors

    from mdb_lanes.ar.model import load_ar_model
    from mdb_lanes.model_os_contracts import content_digest

    if not torch.cuda.is_available():
        raise RuntimeError("circuit-discovery qualification requires CUDA")
    torch.manual_seed(int(config.get("seed", 20261106)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    spec, model, tokenizer = load_ar_model(config["model"], device="cuda",
                                           torch_dtype=torch.bfloat16,
                                           attn_implementation="eager")
    if spec.family != "qwen2" or spec.registry_name != MODEL:
        raise RuntimeError(f"qualified worker requires Qwen2 registry model, got {spec.to_dict()!r}")
    model.eval().requires_grad_(False)
    files: dict[str, str] = {}
    for path in find_safetensors(spec.registry_name):
        files[Path(path).name] = _file_sha(Path(path))
    if not files:
        raise RuntimeError("registered Qwen checkpoint has no safetensors")
    if set(files) != set(parameter["source_files_sha256"]):
        raise RuntimeError("parameter manifest is not a full checkpoint source binding")
    if files != parameter["source_files_sha256"]:
        raise RuntimeError("loaded Qwen safetensor bytes differ from parameter artifacts")
    weight_content_sha = content_digest(files)
    expected_file = config.get("expected_weight_file_sha256")
    if expected_file:
        if files.get("model.safetensors") != expected_file:
            raise RuntimeError("Qwen model.safetensors differs from the trained-MDB identity")
    expected_content = config.get("expected_weight_content_sha256")
    if expected_content and weight_content_sha != expected_content:
        raise RuntimeError("Qwen safetensors map differs from the staged parameter identity")
    # BenchSession's model identity historically uses the raw single-file
    # checkpoint hash for this Qwen registry entry.  Preserve that field while
    # retaining the canonical map digest in the report.
    weights_sha = files.get("model.safetensors") or weight_content_sha
    return spec, model, tokenizer, weights_sha, files, weight_content_sha


def _run_case(config: dict[str, Any], case: dict[str, Any], *, model: Any, tokenizer: Any,
              model_spec: Any, weights_sha: str, root: Path) -> dict[str, Any]:
    from mdb_lanes.bench import BenchSession

    spec, resolved = _case_spec(case, tokenizer)
    case_root = root / f"case-{case['id']}"
    case_root.mkdir(parents=True, exist_ok=False)
    bench = BenchSession.from_resident(
        model, tokenizer, root=case_root, model_id=model_spec.model_id,
        revision=model_spec.revision, model_sha256=weights_sha,
        session_id=case_root.name, lane="virtual", attention_backend="native-eager",
        resident_kv_bytes=256 << 20, continuation_capacity=64,
        publication_mode="journal")
    started = time.monotonic()
    try:
        parent_cut = bench.prefill(resolved["prompt_ids"][:-1])
        baseline = _native_plan(bench, spec["plan"])
        bench.restore(parent_cut)
        discovered = bench.discover_circuit(spec)
        bench.restore(parent_cut)
        reinstall = _native_plan(bench, spec["plan"])
        reinstall_exact = _signature(reinstall) == _signature(baseline)
        audit = _report_contract(discovered, expected_offsets=resolved["receiver_offsets"],
                                 reinstall_exact=reinstall_exact)
        report = discovered["report"]
        result = {
            "id": case["id"], "prompt": case["prompt"], "followup": case["followup"],
            "resolved": resolved, "spec": spec, "parent_cut": parent_cut,
            "baseline_trajectory": baseline, "discover_result": _safe(discovered),
            "discover_report": _safe(report), "reinstall_trajectory": reinstall,
            "instrumentation_audit": audit,
            "native_backend": {"requested": "native-eager", "actual": bench.vm.runtime.backend,
                                "default_backend_is_diagnostic_only": True},
            "journal": {"session_id": bench.session_id, "identity": _safe(bench.identity),
                         "events": _safe(bench.events), "cursor_after_reinstall": _safe(bench.cursor())},
            "elapsed_seconds": time.monotonic() - started,
        }
        return result
    finally:
        bench.close()


def _emit_partial_custody(root: Path, request: dict[str, Any], config: dict[str, Any],
                          cases: list[dict[str, Any]], failure: BaseException,
                          *, parameter: dict[str, Any] | None = None,
                          model_spec: Any | None = None,
                          weight_files: dict[str, str] | None = None,
                          weight_content_sha: str | None = None) -> None:
    """Seal completed cases and source recipes when a later step fails.

    The worker still re-raises the original exception so mrun records a failed
    job.  The compact report/archive lets the collector retain completed raw
    rows and the exact failure boundary instead of losing an earlier case to a
    later optional panel or model-side exception.
    """
    try:
        output_manifest = _write_output_manifest(root)
        archive_path, archive_sha256, archive_bytes = _archive_results(root)
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA, "job_id": os.environ.get("MRUN_JOB_ID", ""),
            "request_sha256": request.get("request_sha256"),
            "source_hashes": request.get("source_hashes", {}),
            "model": ({"registry": model_spec.to_dict(), "model_path": config["model_path"],
                       "family": model_spec.family, "weights_sha256": (weight_files or {}).get("model.safetensors"),
                       "safetensors_sha256": weight_files or {}, "weight_content_sha256": weight_content_sha,
                       "dtype": "bfloat16"} if model_spec is not None else None),
            "parameter_identity": parameter,
            "cases": cases,
            "discovery_policy": config.get("discovery_policy", {}),
            "mechanics_status": "incomplete", "instrument_status": "partial-or-not-run",
            "trend_status": "observations-retained", "terminal_status": "not-assessed",
            "outer_semantic_cot_qualified": False,
            "partial_failure": {"type": type(failure).__name__, "message": str(failure)},
            "limits": config.get("limits", []),
            "artifact_custody": {
                "schema": "mdb-circuit-discovery-custody-v1",
                "output_root": str(root), "manifest": "output-manifest.json",
                "manifest_sha256": _file_sha(root / "output-manifest.json"),
                "archive": archive_path.name, "archive_sha256": archive_sha256,
                "archive_bytes": archive_bytes, "expanded_bytes": output_manifest["expanded_bytes"],
                "file_count": output_manifest["file_count"], "pretrained_weight_copies": False,
                "partial": True,
                "collection": "launch.py --collect JOB_ID; chunked archive markers are direct custody bytes",
            },
        }
        _write(root / "report.json", report)
        _emit_results(root, report)
    except Exception as custody_failure:
        # Never hide the original model/evidence failure behind a secondary
        # custody error; mrun's raw receipt remains authoritative.
        print("MDB_CIRCUIT_DISCOVERY_PARTIAL_CUSTODY_ERROR " + json.dumps({
            "type": type(custody_failure).__name__, "message": str(custody_failure)}, sort_keys=True), flush=True)


def main() -> None:
    raw_config = Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()
    request = _read(Path(os.environ["SATURN_DEBUG_REQUEST"]))
    if _sha(raw_config) != request.get("config_bytes_sha256"):
        raise RuntimeError("sealed configuration bytes changed")
    config = json.loads(raw_config)
    if request.get("config") != config or request.get("request_sha256") != _request_identity(request):
        raise RuntimeError("request/config identity changed")
    if config.get("schema") != "mdb-qwen-circuit-discovery-plan-v1":
        raise RuntimeError("unexpected circuit-discovery plan schema")
    if config.get("model") != MODEL or config.get("family") != "qwen2" or config.get("model_path") != MODEL_PATH:
        raise RuntimeError("worker is bound to the Qwen2.5-1.5B known fleet checkpoint")
    job_id = os.environ.get("MRUN_JOB_ID", "")
    if not re.fullmatch(r"job-[0-9a-f]{12}", job_id):
        raise RuntimeError("mrun job identity is invalid")
    retrieval_path = Path(config["retrieval_script"])
    output_root = Path(config["output_root"])
    root = output_root / job_id
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(retrieval_path, root / "retrieve_weights.py")
    if config.get("parameter_manifest") == "generated-on-worker":
        # Generate and seal the immutable recipe before any model/tokenizer
        # load.  This is the normal path when mbp1 has no Beast checkpoint.
        manifest_path = root / "parameter-artifacts.json"
        _write_parameter_manifest(Path(config["model_path"]), manifest_path)
    else:
        manifest_path = Path(config["parameter_manifest"])
        shutil.copyfile(manifest_path, root / "parameter-artifacts.json")
        manifest_path = root / "parameter-artifacts.json"
    cases = []
    parameter = None
    spec = None
    model = None
    tokenizer = None
    weights_sha = None
    weight_files = None
    weight_content_sha = None
    started = time.monotonic()
    try:
        parameter = _parameter_identity(config, manifest_path)
        spec, model, tokenizer, weights_sha, weight_files, weight_content_sha = _load_model(config, parameter)
        for case in config["cases"]:
            case_result = _run_case(config, case, model=model, tokenizer=tokenizer,
                                    model_spec=spec, weights_sha=weights_sha, root=root)
            cases.append(case_result)
            # Persist every completed case before entering the next one.  A
            # later failure therefore retains all earlier native evidence.
            _write(root / f"case-{case['id']}.json", case_result)
    except BaseException as failure:
        _emit_partial_custody(root, request, config, cases, failure,
                              parameter=parameter, model_spec=spec,
                              weight_files=weight_files,
                              weight_content_sha=weight_content_sha)
        raise
    finally:
        if model is not None:
            del model
            import torch
            torch.cuda.empty_cache()
    assert spec is not None and weights_sha is not None and weight_files is not None
    report = {
        "schema": REPORT_SCHEMA, "job_id": job_id, "request_sha256": request["request_sha256"],
        "source_hashes": request.get("source_hashes", {}),
        "model": {"registry": spec.to_dict(), "model_path": config["model_path"],
                  "family": spec.family, "weights_sha256": weights_sha,
                  "safetensors_sha256": weight_files,
                  "weight_content_sha256": weight_content_sha, "dtype": "bfloat16"},
        "parameter_identity": parameter, "cases": cases,
        "discovery_policy": config["discovery_policy"],
        "mechanics_status": "valid", "instrument_status": "native-passive-observer-audited",
        "trend_status": "observations-retained", "terminal_status": "not-assessed",
        "outer_semantic_cot_qualified": False,
        "limits": config["limits"], "elapsed_seconds": time.monotonic() - started,
    }
    output_manifest = _write_output_manifest(root)
    archive_path, archive_sha256, archive_bytes = _archive_results(root)
    report["artifact_custody"] = {
        "schema": "mdb-circuit-discovery-custody-v1",
        "output_root": str(root), "manifest": "output-manifest.json",
        "manifest_sha256": _file_sha(root / "output-manifest.json"),
        "archive": archive_path.name, "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes, "expanded_bytes": output_manifest["expanded_bytes"],
        "file_count": output_manifest["file_count"],
        "pretrained_weight_copies": False,
        "collection": "launch.py --collect JOB_ID; chunked archive markers are direct custody bytes",
    }
    _write(root / "report.json", report)
    _emit_results(root, report)


if __name__ == "__main__":
    main()
