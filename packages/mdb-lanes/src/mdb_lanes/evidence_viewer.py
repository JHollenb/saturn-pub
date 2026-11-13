"""Read-only computation records, branch futures and consumer comparisons.

Exports portable offline HTML. The FLUX bridge adapter consumes existing saved
evidence; it never loads weights, executes a model or promotes a curated record
to a qualified equation or writable symbol binding. Other temporal evidence uses
MDB's existing temporal causal viewer through the same command.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

VIEW_SCHEMA = "mdb-computation-evidence-view-v1"
BRIDGE_SCHEMA = "mdb-explorer-binding-bridge-v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_MEDIA_BYTES = 40 * 1024 * 1024
ARMS = ("whole_Q", "residual_only", "write_only", "both_RW")
METHODS = ("native", "hold", "candidate")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceViewError(ValueError):
    """Saved evidence cannot be displayed with its declared identity."""


def _require(value: Any, message: str) -> None:
    if not value:
        raise EvidenceViewError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError) as error:
        raise EvidenceViewError("view metadata must be finite JSON") from error


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        _require(path.stat().st_size <= MAX_JSON_BYTES, f"JSON input exceeds bound: {path.name}")
        result = json.loads(path.read_bytes(), parse_constant=reject_constant)
        _canonical(result)
        return result
    except (OSError, ValueError, RecursionError) as error:
        raise EvidenceViewError(f"cannot read bounded finite JSON {path}: {error}") from error


def _check_file(path: Path, expected: Mapping[str, Any]) -> bytes:
    raw = path.read_bytes()
    _require(_sha(raw) == expected.get("sha256"), f"artifact hash differs: {path.name}")
    if "bytes" in expected:
        _require(len(raw) == expected["bytes"], f"artifact byte count differs: {path.name}")
    return raw


def _collection(root: Path, report: Mapping[str, Any], report_path: Path) -> dict[str, Any] | None:
    path = root.parent / "collection.json"
    if not path.is_file():
        return None
    value = _read(path)
    _require(isinstance(value, dict) and value.get("schema") == "mdb-explorer-binding-bridge-collection-v1",
             "unsupported adjacent collection; pass the actual compact evidence directory")
    _require(value.get("requestSHA256") == report.get("requestSHA256"), "collection/report request differs")
    expected = value.get("compactInventory", {}).get("files", {}).get(report_path.name)
    _require(isinstance(expected, dict), "report absent from collection inventory")
    _check_file(report_path, expected)
    return value


def _sidecar(root: Path, name: str, collection: Mapping[str, Any] | None) -> dict[str, Any] | None:
    path = root / name
    if not path.is_file():
        return None
    if collection:
        expected = collection["compactInventory"]["files"].get(name)
        _require(isinstance(expected, dict), f"sidecar absent from collection inventory: {name}")
        _check_file(path, expected)
    value = _read(path)
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _media(root: Path, item: Mapping[str, Any], source_root: str,
           collection: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve only the declared root mapping; never search by basename."""
    declared = item.get("path")
    _require(isinstance(declared, str) and declared, "image requires a declared path")
    path = PurePosixPath(declared)
    if path.is_absolute():
        try:
            member = path.relative_to(PurePosixPath(source_root))
        except ValueError:
            return {"status": "unavailable", "reason": "image is outside the declared source root", "path": declared}
    else:
        member = path
    _require(".." not in member.parts and member.parts, "unsafe image member")
    local = root / str(member)
    _require(local.resolve().is_relative_to(root.resolve()), "image escapes evidence directory")
    if not local.is_file():
        return {"status": "unavailable", "reason": "image is not retained in this local bundle", "path": declared}
    _require(local.stat().st_size <= MAX_IMAGE_BYTES, "image exceeds portable export bound")
    raw = _check_file(local, item)
    if collection:
        expected = collection["compactInventory"]["files"].get(str(member))
        _require(isinstance(expected, dict), "image absent from collection inventory")
        _check_file(local, expected)
    _require(raw.startswith(b"\x89PNG\r\n\x1a\n"), "only retained PNG images may be embedded")
    return {"status": "verified", "path": declared, "member": str(member), "sha256": _sha(raw),
            "bytes": len(raw), "data_url": "data:image/png;base64," + base64.b64encode(raw).decode("ascii")}


def load_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load curated records as supplied observations, without semantic promotion."""
    result, seen = [], set()
    for raw_path in paths:
        path = Path(raw_path)
        _require(path.stat().st_size <= MAX_JSON_BYTES, "record input exceeds bound")
        raw = path.read_bytes()
        if path.suffix == ".jsonl":
            try:
                rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            except (ValueError, RecursionError) as error:
                raise EvidenceViewError("invalid record JSONL") from error
        else:
            value = _read(path)
            rows = value if isinstance(value, list) else [value]
        _require(len(result) + len(rows) <= 2048, "too many computation records")
        for row in rows:
            _canonical(row)
            _require(isinstance(row, dict) and row.get("schema_version") == "known-computations.v1",
                     "records require known-computations.v1 curated observations")
            _require(isinstance(row.get("id"), str) and row["id"], "record requires an ID")
            _require(isinstance(row.get("results", []), list), "record results must be a list")
            _require(row.get("model") is None or isinstance(row["model"], str), "record model label must be text")
            _require(row["id"] not in seen, "duplicate computation record ID")
            seen.add(row["id"])
            result.append({"record": row, "source": str(path), "source_sha256": _sha(raw)})
    return result


def _metrics(rows: Any, *, cases: set[str], kind: str) -> list[dict[str, Any]]:
    _require(isinstance(rows, list), f"{kind} comparisons must be a list")
    result, seen = [], set()
    for row in rows:
        _require(isinstance(row, dict), "comparison must be an object")
        if kind == "endpoint":
            _require(row.get("case") in cases, "endpoint has an unknown case")
            identity = (row.get("case"), row.get("method"), row.get("port"))
        else:
            identity = (row.get("context"), row.get("method"), row.get("port"), row.get("arm"))
        _require(identity not in seen, f"duplicate {kind} comparison")
        seen.add(identity)
        _require(row.get("method") in {"candidate", "hold"}, "comparison method is unsupported")
        _require(isinstance(row.get("port"), str), "comparison requires a port")
        for name in ("rms", "referenceRms", "candidateRms", "relativeRms", "cosine"):
            value = row.get(name)
            _require(value is None or (type(value) in (int, float) and math.isfinite(value)),
                     f"{name} must be finite or unknown")
        result.append(dict(row))
    return result


def bridge_bundle(report_path: str | Path, *, records: Sequence[Mapping[str, Any]] = (),
                  source_root: str | None = None) -> dict[str, Any]:
    """Adapt the measured eight-case campaign without reopening model tensors."""
    path = Path(report_path)
    if path.is_dir():
        path = path / "report.json"
    root = path.parent
    report = _read(path)
    _require(isinstance(report, dict) and report.get("schema") == BRIDGE_SCHEMA, "unsupported future report schema")
    _require(bool(_SHA.fullmatch(str(report.get("requestSHA256", "")))), "report request identity is absent")
    collection = _collection(root, report, path)
    bridge = _sidecar(root, "bridge.json", collection)
    source_root = source_root or (collection or {}).get("remoteRoot") or str(root.resolve())
    cases = []
    _require(isinstance(report.get("accounts"), list), "report requires accounts")
    case_ids = sorted({row.get("case") for row in report["accounts"] if isinstance(row, dict) and isinstance(row.get("case"), str)})
    for case_id in case_ids:
        context, separator, arm = case_id.rpartition("/")
        _require(separator and arm in ARMS, "bridge case requires its declared context/arm")
        current = (bridge or {}).get("cases", {}).get(case_id, {})
        input_cut = current.get("input_cut_after_conditioning_reset", {}).get("record", {})
        previous = current.get("previous_velocity", {})
        cases.append({"id": case_id, "context": context, "arm": arm,
            "input_cut": input_cut.get("cut"), "previous_cut": previous.get("cut", {}).get("record", {}).get("cut"),
            "model_identity": input_cut.get("model_identity"),
            "inputs": {"Z3": current.get("input_z3"), "E": current.get("conditioning"), "v2": previous.get("descriptor")},
            "execution_point": input_cut.get("execution_point"), "previous_velocity_role": previous.get("role")})
    _require(cases, "future report has no cases")
    models = {case["model_identity"] for case in cases if case["model_identity"]}
    _require(len(models) <= 1, "mixed model identities require separate views")
    endpoints = _metrics(report["accounts"], cases=set(case_ids), kind="endpoint")
    responses = _metrics(report.get("signedResponses", []), cases=set(case_ids), kind="response")
    interactions = _metrics(report.get("interactions", []), cases=set(case_ids), kind="interaction")
    for row in responses + interactions:
        _require(any(case["context"] == row.get("context") for case in cases), "contrast has unknown context")
    for row in responses:
        _require(row.get("arm") in ARMS[1:] and row.get("orientation") == "arm-minus-whole_Q", "unsupported response orientation")
    for row in interactions:
        _require(row.get("orientation") == "RW-R-W+Q", "unsupported interaction orientation")
    futures, total_media = [], 0
    for case in cases:
        for method in METHODS:
            key = case["id"] + "/" + method
            descriptor = report.get("images", {}).get(key)
            image = (_media(root, descriptor, source_root, collection) if isinstance(descriptor, Mapping)
                     else {"status": "unavailable", "reason": "no image was retained for this future"})
            total_media += image.get("bytes", 0)
            _require(total_media <= MAX_MEDIA_BYTES, "portable image total exceeds bound")
            futures.append({"id": key, "case_id": case["id"], "method": method, "image": image,
                "field_refs": report.get("fields", {}).get(key, {}),
                "metrics": {row["port"]: row for row in endpoints if row["case"] == case["id"] and row["method"] == method}})
    model = next(iter(models), None)
    linked_records = []
    for source in records:
        row = dict(source["record"])
        declared_cases = set()
        for group in row.get("results", []):
            if isinstance(group, dict):
                occurrences = group.get("endpoint_rows", [])
                _require(isinstance(occurrences, list), "record endpoint occurrences must be a list")
                for endpoint in occurrences:
                    if isinstance(endpoint, dict) and "case" in endpoint:
                        _require(isinstance(endpoint["case"], str), "record case ID must be text")
                        declared_cases.add(endpoint["case"])
        same_request = row.get("request_sha256") == report["requestSHA256"]
        same_model = model is not None and row.get("model_identity") == model
        linked = sorted(declared_cases & set(case_ids)) if same_request and same_model else []
        linked_records.append({**dict(source), "linked_cases": linked,
            "link_status": "exact-run-and-model" if linked else "unlinked",
            "link_reason": ("Explicit occurrence case IDs, request digest and model identity match." if linked else
                            "No explicit matching occurrences under this request and model; shared names do not imply applicability."),
            "authority": "curated observation; no writable symbol binding or equation qualification granted"})
    model_names = {row["record"].get("model") for row in linked_records if row["linked_cases"] and row["record"].get("model")}
    input_dtypes = {case["inputs"]["Z3"].get("dtype") for case in cases if isinstance(case["inputs"]["Z3"], dict)}
    input_devices = {case["inputs"]["Z3"].get("device") for case in cases if isinstance(case["inputs"]["Z3"], dict)}
    transaction = {"status": "unavailable", "reason": "transaction files are not local"}
    if (root / "forecast-transaction/seal.json").is_file():
        from .forecast_transaction import verify_transaction

        try:
            verified = verify_transaction(root / "forecast-transaction", expected_seal_sha256=report.get("sealSHA256"),
                expected_event_head=report.get("transaction", {}).get("event_head"))
        except (ValueError, OSError) as error:
            raise EvidenceViewError(f"local transaction verification refused: {error}") from error
        transaction = {"status": "verified", "scope": "local transaction files against report seal/event head", "result": verified}
    release = _sidecar(root, "release.json", collection)
    body = {"schema": VIEW_SCHEMA, "binding": {"model_identity": model,
        "model_name": next(iter(model_names)) if len(model_names) == 1 else None,
        "observed_input_dtype": next(iter(input_dtypes)) if len(input_dtypes) == 1 else None,
        "observed_input_device": next(iter(input_devices)) if len(input_devices) == 1 else None,
        "request_sha256": report["requestSHA256"],
        "job_id": (collection or {}).get("jobID"), "reference_job": (bridge or {}).get("source", {}).get("job_id"),
        "population": (bridge or {}).get("intended_use", "Existing saved development evidence; no held-out claim supplied"),
        "numerical_profile": report.get("numericalProfile", {})},
        "cases": cases, "futures": futures, "records": linked_records,
        "comparisons": {"endpoint": endpoints, "response": responses, "interaction": interactions},
        "methods": [{"id": "native", "label": "Native reference", "role": "previously measured"},
                    {"id": "hold", "label": "Hold own v2", "role": "counterfactual continuation"},
                    {"id": "candidate", "label": "Frozen predictor", "role": "approximate prediction"}],
        "pipeline": [{"id": "inputs", "label": "Observed state", "port": None},
                     {"id": "velocity3", "label": "Velocity proposal", "port": "velocity3"},
                     {"id": "latent4", "label": "Scheduler", "port": "latent4"},
                     {"id": "decodedRaw", "label": "VAE", "port": "decodedRaw"},
                     {"id": "pngRGB", "label": "Image", "port": "pngRGB"}],
        "evidence": {"report": {"sha256": _sha(path.read_bytes()), "collection_hash_checked": collection is not None},
            "images": {"verified": sum(future["image"]["status"] == "verified" for future in futures), "bytes": total_media},
            "transaction": transaction, "seal_sha256": report.get("sealSHA256"),
            "preseal_future_file_opens": (release or {}).get("futureOpensBeforeRelease"),
            "producer_native_witnesses": report.get("nativeControls", []),
            "external_files": {"status": "collector-reported" if collection else "unknown",
                "files": (collection or {}).get("compactInventory", {}).get("externalPredictionFilesVerified"),
                "scope": "External tensor files are not rehashed by this image/metadata viewer."},
            "mechanics_status": report.get("mechanics_status"), "instrument_status": report.get("instrument_status"),
            "terminal_status": report.get("terminal_status"), "record_authority": "Supplied curated records; source references are displayed, not independently certified."},
        "work": report.get("work", {}), "limitations": report.get("limitations", []),
        "provenance": {"source_report": str(path), "source_schema": BRIDGE_SCHEMA, "model_executions": 0,
                       "live_controls": False, "read_only": True}}
    body["content_sha256"] = _sha(_canonical(body))
    return body


def render_html(bundle: Mapping[str, Any], *, title: str = "MDB computation explorer") -> str:
    """Render a validated, sealed metadata view without network dependencies."""
    _require(bundle.get("schema") == VIEW_SCHEMA, "expected normalized computation view")
    body = dict(bundle)
    expected = body.pop("content_sha256", None)
    _require(expected == _sha(_canonical(body)), "normalized view digest differs")
    for key in ("binding", "comparisons", "evidence", "work", "provenance"):
        _require(isinstance(body.get(key), dict), f"normalized view requires {key} object")
    for key in ("cases", "futures", "records", "methods", "pipeline", "limitations"):
        _require(isinstance(body.get(key), list), f"normalized view requires {key} list")
    case_ids = [case.get("id") for case in body["cases"] if isinstance(case, dict)]
    _require(len(case_ids) == len(body["cases"]) and all(isinstance(value, str) for value in case_ids)
             and len(set(case_ids)) == len(case_ids), "normalized case IDs must be unique text")
    for future in body["futures"]:
        _require(isinstance(future, dict) and future.get("case_id") in case_ids, "normalized future has unknown case")
    for row in body["records"]:
        _require(isinstance(row, dict) and isinstance(row.get("record"), dict)
                 and isinstance(row.get("linked_cases"), list) and set(row["linked_cases"]) <= set(case_ids),
                 "normalized record links require known cases")
    for future in body.get("futures", []):
        media = future.get("image", {})
        if media.get("status") == "verified":
            encoded = media.get("data_url", "")
            _require(encoded.startswith("data:image/png;base64,"), "unsafe embedded media")
            try:
                raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
            except ValueError as error:
                raise EvidenceViewError("invalid embedded PNG") from error
            _require(len(raw) <= MAX_IMAGE_BYTES and raw.startswith(b"\x89PNG\r\n\x1a\n")
                     and _sha(raw) == media.get("sha256"), "embedded PNG identity differs")
    from .temporal_causal_viewer import _script_json

    template = files("mdb_lanes").joinpath("assets/computation_viewer.html").read_text(encoding="utf-8")
    substitutions = {"__TITLE__": html.escape(title, quote=True), "__VIEW_DATA__": _script_json(dict(bundle))}
    return re.sub(r"__TITLE__|__VIEW_DATA__", lambda match: substitutions[match.group()], template)


def export_view(inputs: Sequence[str | Path], output: str | Path, *, records: Sequence[str | Path] = (),
                source_root: str | None = None, title: str = "MDB computation explorer",
                json_output: str | Path | None = None) -> dict[str, Any]:
    """Route saved image futures or existing temporal receipts to the right view."""
    _require(bool(inputs), "provide an evidence report")
    paths = [Path(path) / "report.json" if Path(path).is_dir() else Path(path) for path in inputs]
    first = _read(paths[0])
    if isinstance(first, dict) and first.get("schema") == VIEW_SCHEMA:
        _require(len(paths) == 1 and not records and not source_root, "normalized views are already bound; export them separately")
        bundle = first
        rendered = render_html(bundle, title=title)
    elif isinstance(first, dict) and first.get("schema") == BRIDGE_SCHEMA:
        _require(len(paths) == 1, "view separate bridge runs separately; numerical contracts are not pooled")
        bundle = bridge_bundle(paths[0], records=load_records(records), source_root=source_root)
        rendered = render_html(bundle, title=title)
    else:
        _require(not records and not source_root, "record/media joins currently require a supported future report")
        from .temporal_causal_viewer import normalize_evidence
        from .temporal_causal_viewer import render_html as temporal_html

        bundle = normalize_evidence([_read(path) for path in paths], sources=[str(path) for path in paths])
        rendered = temporal_html(bundle, title=title)
    destination = Path(output)
    _require(destination.resolve() not in {path.resolve() for path in paths}, "output cannot replace source evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("x", encoding="utf-8") as stream:
            json.dump(bundle, stream, sort_keys=True, indent=2, allow_nan=False)
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore saved MDB computations, branches and evidence in offline HTML")
    parser.add_argument("inputs", nargs="+", help="future report/compact directory, or temporal causal evidence JSON")
    parser.add_argument("-o", "--output", required=True, help="new portable HTML file")
    parser.add_argument("--records", action="append", default=[], help="curated computation JSONL/JSON; repeatable")
    parser.add_argument("--source-root", help="explicit original root for relocating image paths; no basename search")
    parser.add_argument("--json-output", help="optional new normalized view JSON")
    parser.add_argument("--title", default="MDB computation explorer")
    args = parser.parse_args(argv)
    try:
        bundle = export_view(args.inputs, args.output, records=args.records, source_root=args.source_root,
                             title=args.title, json_output=args.json_output)
    except (EvidenceViewError, OSError, ValueError) as error:
        parser.exit(2, f"mdb view: {error}\n")
    print(json.dumps({"output": str(args.output), "schema": bundle["schema"], "content_sha256": bundle["content_sha256"],
        "cases": len(bundle.get("cases", [])), "futures": len(bundle.get("futures", [])),
        "records": len(bundle.get("records", [])), "model_executions": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
