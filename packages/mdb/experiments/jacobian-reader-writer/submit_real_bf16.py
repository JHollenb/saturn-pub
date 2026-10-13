#!/usr/bin/env python3
"""Admit and collect the sealed real-model BF16 Jacobian reader/writer assay."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from mdb.execution_environment import intent  # noqa: E402
from mdb.job_support import (  # noqa: E402
    guarded_job_receipt,
    guarded_mrun_launch,
    one_lease_scheduling_contract,
    terminal_job,
)

RESULTS = HERE / "results"
WHEEL_NAME = "mdb_runtime-0.1.0-py3-none-any.whl"
WHEEL = HERE / "installed-wheel/final-qualified-1120c5299ffe/wheel" / WHEEL_NAME
WHEEL_SHA256 = "1120c5299ffe096a76db84ceaa71cd44348db8f03a46e6bf17e72e433b96ea2e"
RESIDENT_PYTHON = "/mnt/big/llm-models/.uv-cache/archive-v0/Z8hg7fARI9lZbMS7/bin/python"
REPORT_MARKER = "MDB_JSPACE_BF16_REPORT_BASE64 "
BUNDLE_CHUNK_MARKER = "MDB_JSPACE_BF16_BUNDLE_CHUNK "
BUNDLE_SHA_MARKER = "MDB_JSPACE_BF16_BUNDLE_SHA256 "
BUNDLE_COUNT_MARKER = "MDB_JSPACE_BF16_BUNDLE_CHUNKS "
TERMINAL = {"succeeded", "failed", "killed_ram", "killed_vram", "timeout", "cancelled", "lost"}
MEASURED_ADMISSION_FACTOR = 1.25
JOB_ID_PATTERN = re.compile(r"job-[a-z0-9]+")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return sha256_bytes(raw)


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable evidence conflict: {path}")
        return
    path.write_bytes(data)


def source_files() -> dict[str, Path]:
    sources = {
        "bootstrap_real_bf16.py": HERE / "bootstrap_real_bf16.py",
        "real_bf16_worker.py": HERE / "real_bf16_worker.py",
        "protocol.json": HERE / "protocol.json",
        WHEEL_NAME: WHEEL,
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise RuntimeError("missing sealed assay source(s): " + ", ".join(missing))
    if sha256_file(WHEEL) != WHEEL_SHA256:
        raise RuntimeError("qualified MDB wheel SHA-256 mismatch")
    return sources


def measured_admission_exception(
    job: str,
    hashes: dict[str, str],
    request: dict[str, Any],
    resource_equivalent_change: str | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Derive a resource exception from a retained, same-workload VRAM kill."""
    if JOB_ID_PATTERN.fullmatch(job) is None:
        raise RuntimeError("measured-admission job ID is invalid")
    record_path = RESULTS / job / "job-record.json"
    if not record_path.is_file():
        raise RuntimeError(f"retained terminal record is missing: {record_path}")
    record_raw = record_path.read_bytes()
    record = json.loads(record_raw)
    if record.get("state") != "killed_vram":
        raise RuntimeError("measured admission requires a retained killed_vram job")
    if record.get("experiment") != request["experiment"]:
        raise RuntimeError("measured-admission experiment identity differs")
    old_config = record.get("config")
    if not isinstance(old_config, dict):
        raise RuntimeError("measured-admission record lacks its submitted config")
    for field in ("model", "backend", "device", "dtype", "workload_geometry"):
        if old_config.get(field) != request[field]:
            raise RuntimeError(f"measured-admission workload differs at {field}")
    old_hashes = old_config.get("source_sha256")
    identity_names = ("real_bf16_worker.py", "protocol.json", WHEEL_NAME)
    if not isinstance(old_hashes, dict):
        raise RuntimeError("measured-admission record lacks source identities")
    if resource_equivalent_change is None:
        if any(old_hashes.get(name) != hashes[name] for name in identity_names):
            raise RuntimeError(
                "measured admission is restricted to the exact worker/protocol/wheel"
            )
        equivalent_change = None
    else:
        reason = resource_equivalent_change.strip()
        if not reason:
            raise RuntimeError("resource-equivalent change reason must be nonempty")
        if any(old_hashes.get(name) != hashes[name] for name in ("protocol.json", WHEEL_NAME)):
            raise RuntimeError(
                "resource-equivalent change permits only the worker SHA-256 to differ"
            )
        old_worker = old_hashes.get("real_bf16_worker.py")
        new_worker = hashes["real_bf16_worker.py"]
        if not isinstance(old_worker, str) or old_worker == new_worker:
            raise RuntimeError(
                "resource-equivalent change requires a changed worker SHA-256"
            )
        equivalent_change = {
            "reason": reason,
            "old_worker_sha256": old_worker,
            "new_worker_sha256": new_worker,
            "scope": (
                "resource-history transfer across an audited worker-only harness fix; "
                "the payload is not bit-identical"
            ),
        }

    result = record.get("result")
    reservation = record.get("reservation")
    plan = record.get("plan")
    if not all(isinstance(value, dict) for value in (result, reservation, plan)):
        raise RuntimeError("measured-admission record lacks result, reservation, or plan")
    peak_vram = float(result.get("peak_vram_mb", math.nan))
    peak_rss = float(result.get("peak_rss_mb", math.nan))
    prior_vram = float(reservation.get("vram_mb", math.nan))
    planned_ram = float(plan.get("ram_limit_mb", math.nan))
    if not all(
        math.isfinite(value) and value > 0
        for value in (peak_vram, peak_rss, prior_vram, planned_ram)
    ):
        raise RuntimeError("measured-admission record has invalid resource observations")
    failure = result.get("failure")
    resources = failure.get("resources") if isinstance(failure, dict) else None
    failure_ceiling = (
        float(resources.get("vram_ceiling_mb", math.nan))
        if isinstance(resources, dict)
        else math.nan
    )
    if not math.isfinite(failure_ceiling) or failure_ceiling <= 0:
        raise RuntimeError("measured-admission record lacks the VRAM failure ceiling")

    overrides = {
        "vram_mb": max(prior_vram, peak_vram) * MEASURED_ADMISSION_FACTOR,
        "ram_mb": max(planned_ram, peak_rss * MEASURED_ADMISSION_FACTOR),
    }
    basis = {
        "schema": "mdb-measured-admission-exception-v1",
        "basis_job_id": job,
        "basis_state": record["state"],
        "job_record_sha256": sha256_bytes(record_raw),
        "factor": MEASURED_ADMISSION_FACTOR,
        "same_workload_sha256": {name: hashes[name] for name in identity_names},
        "observed_peak_vram_mb": peak_vram,
        "observed_peak_rss_mb": peak_rss,
        "prior_reservation_vram_mb": prior_vram,
        "prior_plan_ram_limit_mb": planned_ram,
        "failure_vram_ceiling_mb": failure_ceiling,
        "requested_vram_mb": overrides["vram_mb"],
        "requested_ram_mb": overrides["ram_mb"],
        "censoring": (
            "resource-killed run: observed peaks and failure ceiling are floors; "
            "the full-run peak is unknown"
        ),
    }
    if equivalent_change is not None:
        basis["resource_equivalent_change"] = equivalent_change
    return basis, overrides


def specification(
    measured_admission_from: str | None = None,
    resource_equivalent_change: str | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, Path], bytes, dict[str, float]]:
    sources = source_files()
    protocol_raw = sources["protocol.json"].read_bytes()
    json.loads(protocol_raw)
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    contract = one_lease_scheduling_contract(
        model_loads=1,
        logical_branches=32,
        logical_operations=256,
        logical_checkpoint_replays=64,
        max_local_work_items=256,
        checkpoint_policy=(
            "one resident frozen Qwen checkpoint; BF16 corpus fit, exact cuts, residual Acts, "
            "native continuations, KV mediation, export, and verification remain in process; "
            "no retries or nested submissions"
        ),
        extra={
            "physical_backward_invocations": 576,
            "physical_assay_token_operation_ceiling": 1024,
            "worker_deadline_seconds": 840,
        },
    )
    request: dict[str, Any] = {
        "schema": "mdb-jspace-real-bf16-request-v1",
        "experiment": "mdb-jspace-real-bf16",
        "task_family": "forward",
        "model": "Qwen/Qwen2.5-1.5B",
        "backend": "hf",
        "attention_backend": "native-eager",
        "instrument_backend": "mdb-native-eager",
        "device": "cuda",
        "dtype": "bfloat16",
        "batch_size": 8,
        "seq_len": 32,
        "needs": {"cuda": True},
        "backward_graph": True,
        "autograd": "explicit frozen-suffix BF16 VJPs; zero parameter updates and no optimizer",
        "execution_environment": intent(),
        "resident_python": RESIDENT_PYTHON,
        "source_sha256": hashes,
        "protocol_sha256": sha256_bytes(protocol_raw),
        "qualified_wheel": {"name": WHEEL_NAME, "sha256": WHEEL_SHA256},
        "workload_geometry": {
            "model_loads": 1,
            "fit_prompt_count": 3,
            "fit_max_length": 32,
            "vjp_chunk": 8,
            "backward_invocations": 576,
            "residual_width": 1536,
            "source_block": 11,
            "completed_layer_boundary": 12,
            "target_block": 27,
            "max_assay_token_operations": 1024,
            "max_worker_seconds": 840,
        },
        "scheduling_contract": contract,
        "scheduler": {
            "strict_preflight": True,
            "retry_on_kill": False,
            "reservation_policy": "intent-first; no manual RAM or VRAM reservation",
        },
        "scientific_scope": {
            "parameters_frozen": True,
            "parameter_updates": 0,
            "fit": "BF16 frozen-suffix VJPs, FP32 corpus pooling, BF16 sealed transport",
            "claim_stage": "exploratory-mechanics-and-trend",
        },
    }
    reservation_overrides: dict[str, float] = {}
    if resource_equivalent_change is not None and measured_admission_from is None:
        raise RuntimeError(
            "resource-equivalent change requires --measured-admission-from"
        )
    if measured_admission_from is not None:
        basis, reservation_overrides = measured_admission_exception(
            measured_admission_from, hashes, request, resource_equivalent_change
        )
        request["measured_admission_exception"] = basis
        request["scheduler"]["reservation_policy"] = (
            "measured same-workload resource-kill exception; no automatic retry"
        )
    request["request_fingerprint"] = canonical_sha256(request)
    command = [
        "/usr/bin/env",
        "PYTHONUNBUFFERED=1",
        "PYTHONNOUSERSITE=1",
        "PYTHONPATH=",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "SATURN_RESIDENT_WORKER=1",
        RESIDENT_PYTHON,
        "-I",
        "-u",
        "bootstrap_real_bf16.py",
    ]
    return request, command, sources, protocol_raw, reservation_overrides


def stage_payload(destination: Path, sources: dict[str, Path]) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name, source in sources.items():
        target = destination / name
        shutil.copyfile(source, target)
        hashes[name] = sha256_file(target)
    write_once(destination / "payload-manifest.json", json_bytes(hashes))
    return {**hashes, "payload-manifest.json": sha256_file(destination / "payload-manifest.json")}


def preserve_custody(staging: Path, request: dict[str, Any], payload_hashes: dict[str, str]) -> Path:
    custody_id = canonical_sha256(
        {"request": request, "payload_sha256": payload_hashes}
    )[:20]
    custody = RESULTS / "source-custody" / custody_id
    for relative, expected in payload_hashes.items():
        data = (staging / relative).read_bytes()
        if sha256_bytes(data) != expected:
            raise RuntimeError(f"staged payload changed before custody: {relative}")
        write_once(custody / "payload" / relative, data)
    write_once(custody / "submitted-request.json", json_bytes(request))
    write_once(custody / "payload-manifest.json", json_bytes(payload_hashes))
    return custody


def submit(
    measured_admission_from: str | None = None,
    resource_equivalent_change: str | None = None,
) -> str:
    request, command, sources, _, reservation_overrides = specification(
        measured_admission_from, resource_equivalent_change
    )
    with tempfile.TemporaryDirectory(prefix="mdb-jspace-bf16-") as temporary:
        staging = Path(temporary) / "payload"
        payload_hashes = stage_payload(staging, sources)
        custody = preserve_custody(staging, request, payload_hashes)
        job = guarded_mrun_launch(
            command,
            request=request,
            experiment=request["experiment"],
            task="forward",
            model=request["model"],
            payload=staging,
            prefer_host="beast",
            gpu=True,
            needs={"cuda": True},
            mrun_config=request,
            timeout_s=900,
            detach=True,
            retry_on_kill=False,
            preflight=True,
            **reservation_overrides,
            note=(
                "Real MDB J-space BF16 proof: one frozen Qwen2.5-1.5B load; three corpus "
                "fit graphs, 576 chunked VJPs; reader parity, signed residual Acts, native "
                "fixed-query continuations and successor-KV reinstall; 840s worker stop; "
                "one lease, no nested jobs or retries; reservation is intent-first unless "
                "the request carries a same-workload measured resource-kill exception"
            ),
        )
    write_once(custody / f"{job}.json", json_bytes({"job_id": job, "custody": custody.name}))
    print(json.dumps({"job": job, "custody": str(custody)}, sort_keys=True))
    return job


def safe_extract_bundle(raw: bytes, destination: Path) -> dict[str, str]:
    if len(raw) > 512 << 20:
        raise RuntimeError("compressed result bundle exceeds 512 MiB")
    hashes: dict[str, str] = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if member.isdir():
                continue
            if (
                not member.isfile()
                or relative.is_absolute()
                or relative == Path(".")
                or ".." in relative.parts
                or relative.as_posix() in hashes
            ):
                raise RuntimeError("result bundle contains an unsafe or duplicate member")
            total += member.size
            if total > 2 << 30:
                raise RuntimeError("expanded result bundle exceeds 2 GiB")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError("result bundle member could not be read")
            data = handle.read()
            if len(data) != member.size:
                raise RuntimeError("result bundle member is truncated")
            hashes[relative.as_posix()] = sha256_bytes(data)
            write_once(destination / relative, data)
    required = {"output/report.json", "output/protocol.json"}
    if not required.issubset(hashes):
        raise RuntimeError("result bundle is missing output/report.json or output/protocol.json")
    if not any(name.startswith("lens/") or name.startswith("output/lens/") for name in hashes):
        raise RuntimeError("result bundle is missing the lens artifact tree")
    if not any(
        name.startswith("bench-export/") or name.startswith("output/bench-export/")
        for name in hashes
    ):
        raise RuntimeError("result bundle is missing the Bench export tree")
    return hashes


def verify_result_identity(
    report: dict[str, Any], artifacts: Path, custody: Path
) -> dict[str, str]:
    submitted_hashes = json.loads((custody / "payload-manifest.json").read_text())
    expected = {
        "worker_sha256": submitted_hashes.get("real_bf16_worker.py"),
        "protocol_sha256": submitted_hashes.get("protocol.json"),
        "wheel_sha256": submitted_hashes.get(WHEEL_NAME),
    }
    if any(not isinstance(value, str) or len(value) != 64 for value in expected.values()):
        raise RuntimeError("submitted payload custody lacks required source identities")
    if expected["wheel_sha256"] != WHEEL_SHA256:
        raise RuntimeError("submitted wheel differs from the qualified wheel identity")
    for field, digest in expected.items():
        if report.get(field) != digest:
            raise RuntimeError(f"report identity differs from submitted payload at {field}")
    profile = report.get("profile")
    if not isinstance(profile, dict) or profile.get("application_lane") != "cuda-bfloat16-eager":
        raise RuntimeError("report did not execute in the declared CUDA BF16 eager lane")
    if (
        profile.get("dtype") != "torch.bfloat16"
        or profile.get("bench_lane") != "native"
        or profile.get("attention_backend") != "native-eager"
    ):
        raise RuntimeError("report profile differs from the declared native BF16 Bench profile")
    submitted_protocol = json.loads((custody / "payload/protocol.json").read_text())
    bundled_protocol = json.loads((artifacts / "output/protocol.json").read_text())
    if bundled_protocol != submitted_protocol:
        raise RuntimeError("bundled output/protocol.json differs from the submitted protocol")
    if submitted_protocol.get("wheel_sha256") != WHEEL_SHA256:
        raise RuntimeError("submitted protocol does not bind the qualified MDB wheel")
    return expected


def verify_guarded_receipt(
    guarded: dict[str, Any], job: str, state: str, request: dict[str, Any]
) -> None:
    if guarded.get("schema") != "mrun.guarded-job-receipt.v2":
        raise RuntimeError("guarded mrun receipt schema is invalid")
    if guarded.get("job_id") != job or guarded.get("state") != state:
        raise RuntimeError("guarded mrun receipt differs from the requested terminal job")
    submitted = guarded.get("submitted_request")
    if not isinstance(submitted, dict):
        raise RuntimeError("guarded mrun receipt lacks the submitted request")
    if guarded.get("submitted_request_sha256") != canonical_sha256(submitted):
        raise RuntimeError("guarded mrun submitted-request seal is invalid")
    config = submitted.get("config")
    if not isinstance(config, dict) or config.get("saturn_request_sha256") != canonical_sha256(
        request
    ):
        raise RuntimeError("guarded mrun receipt is not bound to the submitted MDB request")
    if submitted.get("experiment") != request["experiment"]:
        raise RuntimeError("guarded mrun experiment identity changed")
    if state == "succeeded":
        custody = guarded.get("payload_custody")
        if not isinstance(custody, dict) or custody.get("required") is not True:
            raise RuntimeError("guarded mrun receipt lacks required payload custody")
        identities = [custody.get(name) for name in ("declared", "sealed", "executed")]
        if any(not isinstance(value, dict) for value in identities) or not (
            identities[0] == identities[1] == identities[2]
        ):
            raise RuntimeError("guarded mrun declared, sealed, and executed payloads differ")


def decode_markers(
    logs: str,
) -> tuple[dict[str, Any] | None, bytes | None, str | None, int]:
    reports: list[dict[str, Any]] = []
    chunks: dict[int, str] = {}
    declared_hashes: list[str] = []
    declared_counts: list[int] = []
    for line in logs.splitlines():
        if line.startswith(REPORT_MARKER):
            encoded = line[len(REPORT_MARKER) :].strip()
            reports.append(json.loads(base64.b64decode(encoded, validate=True)))
        elif line.startswith(BUNDLE_CHUNK_MARKER):
            fields = line[len(BUNDLE_CHUNK_MARKER) :].split(maxsplit=1)
            if len(fields) != 2:
                raise RuntimeError("malformed bundle chunk marker")
            index = int(fields[0])
            if index < 0 or index in chunks:
                raise RuntimeError("invalid or duplicate bundle chunk index")
            encoded = fields[1].strip()
            if not encoded or any(character.isspace() for character in encoded):
                raise RuntimeError("bundle chunk contains invalid whitespace")
            chunks[index] = encoded
        elif line.startswith(BUNDLE_SHA_MARKER):
            declared_hashes.append(line[len(BUNDLE_SHA_MARKER) :].strip())
        elif line.startswith(BUNDLE_COUNT_MARKER):
            declared_counts.append(int(line[len(BUNDLE_COUNT_MARKER) :].strip()))
    if len(reports) > 1 or len(declared_hashes) > 1 or len(declared_counts) > 1:
        raise RuntimeError("duplicate report, bundle SHA, or bundle count marker")
    if chunks:
        count = declared_counts[0] if declared_counts else None
        if count is None or count < 1 or sorted(chunks) != list(range(count)):
            raise RuntimeError("bundle chunks are incomplete")
        bundle = base64.b64decode(
            "".join(chunks[index] for index in range(count)), validate=True
        )
    else:
        bundle = None
    declared = declared_hashes[0] if declared_hashes else None
    if bundle is not None:
        if declared is None or len(declared) != 64 or sha256_bytes(bundle) != declared:
            raise RuntimeError("bundle SHA-256 marker is missing or mismatched")
    elif declared is not None or declared_counts:
        raise RuntimeError("bundle completion marker was emitted without bundle chunks")
    return (reports[0] if reports else None), bundle, declared, len(chunks)


def collect(job: str) -> int:
    if not job:
        raise ValueError("collect requires --job")
    collected_job, record, logs = terminal_job(job, wait=False)
    if collected_job != job:
        raise RuntimeError("direct collection returned a different scheduler job")
    state = str(record.get("state"))
    if state not in TERMINAL:
        print(json.dumps({"job": job, "state": state, "collection": "deferred"}, sort_keys=True))
        return 0
    output = RESULTS / job
    write_once(output / "job-record.json", json_bytes(record))
    write_once(output / "job.log", logs.encode())
    custody_bindings = list((RESULTS / "source-custody").glob(f"*/{job}.json"))
    if len(custody_bindings) != 1:
        raise RuntimeError("collection requires exactly one source-custody binding")
    custody = custody_bindings[0].parent
    submitted_request = json.loads((custody / "submitted-request.json").read_text())
    guarded = guarded_job_receipt(job)
    verify_guarded_receipt(guarded, job, state, submitted_request)
    write_once(output / "submitted-request.json", (custody / "submitted-request.json").read_bytes())
    write_once(output / "job-receipt.json", json_bytes(guarded))
    write_once(output / "payload-manifest.json", (custody / "payload-manifest.json").read_bytes())
    for source in (custody / "payload").iterdir():
        if source.is_file():
            write_once(output / "submitted-payload" / source.name, source.read_bytes())
    report, bundle, bundle_sha, bundle_chunks = decode_markers(logs)
    artifact_hashes: dict[str, str] = {}
    if bundle is not None:
        write_once(output / "result-bundle.tar.gz", bundle)
        artifact_hashes = safe_extract_bundle(bundle, output / "artifacts")
        write_once(output / "artifact-manifest.json", json_bytes(artifact_hashes))
    if report is not None:
        report_raw = json_bytes(report)
        if bundle is not None:
            bundled_report = json.loads((output / "artifacts/output/report.json").read_text())
            if bundled_report != report:
                raise RuntimeError("report marker differs from bundled output/report.json")
            result_identity = verify_result_identity(report, output / "artifacts", custody)
        else:
            result_identity = {}
        write_once(output / "report.json", report_raw)
    else:
        result_identity = {}
    receipt = {
        "schema": "mdb-jspace-real-bf16-collection-v1",
        "job_id": job,
        "state": state,
        "source_custody": str(custody),
        "scheduler_record_sha256": sha256_bytes(json_bytes(record)),
        "guarded_job_receipt_sha256": sha256_bytes(json_bytes(guarded)),
        "log_sha256": sha256_bytes(logs.encode()),
        "report_present": report is not None,
        "bundle_sha256": bundle_sha,
        "bundle_chunks": bundle_chunks,
        "artifact_count": len(artifact_hashes),
        "artifact_hashes": artifact_hashes,
        "result_identity": result_identity,
    }
    write_once(output / "collection-receipt.json", json_bytes(receipt))
    complete = report is not None and bundle is not None and bool(artifact_hashes)
    if state == "succeeded" and not complete:
        raise RuntimeError("successful scheduler job has incomplete declared result custody")
    print(
        json.dumps(
            {"job": job, "state": state, "complete": complete, "output": str(output)},
            sort_keys=True,
        )
    )
    return 0 if state == "succeeded" and complete else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "submit", "collect"))
    parser.add_argument("--job")
    parser.add_argument("--measured-admission-from", metavar="JOB_ID")
    parser.add_argument("--resource-equivalent-change", metavar="REASON")
    args = parser.parse_args()
    if args.action == "collect":
        if not args.job:
            parser.error("collect requires --job JOB_ID")
        if args.measured_admission_from:
            parser.error("--measured-admission-from is valid only for preflight or submit")
        if args.resource_equivalent_change:
            parser.error("--resource-equivalent-change is valid only for preflight or submit")
        return collect(args.job)
    if args.job:
        parser.error("--job is valid only for collect")
    request, command, sources, _, reservation_overrides = specification(
        args.measured_admission_from, args.resource_equivalent_change
    )
    if args.action == "preflight":
        print(
            json.dumps(
                {
                    "kind": "payload-and-intent-preview; submit performs engine strict preflight",
                    "request": request,
                    "command": command,
                    "payload": {name: sha256_file(path) for name, path in sources.items()},
                    "reservation_override": reservation_overrides or None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    submit(args.measured_admission_from, args.resource_equivalent_change)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
