"""Post-job evidence pipeline: WATCH -> CLAIMS -> XREF as one standard verb.

``saturn-evidence run <dir>`` sweeps a finished job or experiment directory
and runs the three evidence inversions over it, fail-soft per step with a
step-status report:

1. WATCH — the standard spec pack (:mod:`saturn.evidence_specs`) is adapted
   to every ``*.jsonl`` stream under the directory and evaluated; fires are
   written as hash-chained ledgers under ``index/watch/<experiment>/<job>/
   <stream>.fires.jsonl`` (never inside ``results/``) and each ledger is
   verified.  Ledger writes are deterministic overwrite-or-skip: when a
   rerun produces fires identical to a verified existing ledger (chain
   fields aside) the ledger bytes are left untouched; otherwise it is
   atomically replaced; zero fires means no ledger (a stale one is removed).
2. CLAIMS — every receipt-like ``*.json`` (top-level ``gates`` dict) is
   ingested into an auto registry (default ``index/claims-auto.jsonl``,
   NEVER the curated ``index/claims-registry.jsonl``) with declared-refs
   harvest.  Identity is the deterministic path-derived ``claim_id``: a
   receipt whose bytes match the current row's recorded receipt fingerprint
   is skipped, so re-ingest appends nothing.
3. XREF — an incremental :func:`saturn.xref_index.build_index` sweep.  The
   xref API scopes by root, but stored paths are root-relative and full
   builds purge unseen paths, so a scoped build against the shared database
   would evict every other experiment: the pipeline therefore always builds
   over the full xref root (default ``results/``) into the shared database,
   relying on the builder's mtime/size skip for incrementality.  A
   pre-existing fire ledger that fails chain verification is surfaced as a
   broken watch step even when the current stream is re-evaluated and a
   replacement ledger is written; recomputation must not silently repair
   evidence without reporting the prior integrity failure.

``saturn-evidence verify <dir>`` re-verifies the fire ledgers for the
directory's streams and runs the claims registry's read-only drift check.

Exit codes (both commands): 0 ok, 3 findings fired (watch fires on run,
claim drift on verify), 2 broken (refused input, step error, or ledger/chain
integrity failure).

Honest claim boundary: mechanics only.  The pipeline indexes and re-checks
receipts that already exist; it makes no scientific claim about any model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import types
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EVIDENCE_PIPELINE_SCHEMA = "saturn-evidence-pipeline-v1"
from .paths import data_root
REPO_ROOT = data_root()
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_WATCH_ROOT = REPO_ROOT / "index" / "watch"
DEFAULT_AUTO_REGISTRY = REPO_ROOT / "index" / "claims-auto.jsonl"
DEFAULT_XREF_DB = REPO_ROOT / "index" / "xref.sqlite"
MAX_RECEIPT_BYTES = 32 * 1024 * 1024
MAX_STREAM_BYTES = 256 * 1024 * 1024

EXIT_OK = 0
EXIT_BROKEN = 2
EXIT_FINDINGS = 3


def _load_saturn_module(name: str) -> Any:
    """Load MDB's dependency-free evidence modules without a synthetic package."""
    return importlib.import_module("mdb." + name)


receipt_watch = _load_saturn_module("receipt_watch")
claims_registry = _load_saturn_module("claims_registry")
evidence_citations = _load_saturn_module("evidence_citations")
xref_index = _load_saturn_module("xref_index")
evidence_specs = _load_saturn_module("evidence_specs")

_CHAIN_FIELDS = ("segment_id", "prev_sha256", "sha256")


class EvidencePipelineError(ValueError):
    """Raised when the pipeline refuses an input it cannot safely process."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def stream_coordinates(
    stream: str | Path, results_root: str | Path = RESULTS_ROOT
) -> tuple[str, str, str]:
    """Map a stream path to ``(experiment, job, ledger basename)``.

    Streams under ``results_root`` use their tree coordinates; nested extra
    directories are folded into the ledger basename with ``__``, and a short
    sha256 digest of the folded relative path is appended (mirroring
    :func:`claim_id_for_receipt`) so distinct streams never collide — plain
    ``__`` folding alone maps ``a/b.jsonl`` and ``a__b.jsonl`` to one ledger,
    and a zero-fire collider would then delete the fires the other stream
    just wrote.  Streams outside ``results_root`` fall back to their parent
    directory names.  Deterministic by construction.
    """

    stream = Path(stream).resolve()
    try:
        parts = stream.relative_to(Path(results_root).resolve()).parts
    except ValueError:
        experiment = stream.parent.parent.name or "_"
        job = stream.parent.name or "_"
        remainder: tuple[str, ...] = (stream.name,)
    else:
        if len(parts) >= 3:
            experiment, job, remainder = parts[0], parts[1], parts[2:]
        elif len(parts) == 2:
            experiment, job, remainder = parts[0], "_", parts[1:]
        else:
            experiment, job, remainder = "_", "_", parts
    digest = hashlib.sha256("/".join(remainder).encode("utf-8")).hexdigest()[:8]
    base = "__".join(remainder)
    if base.endswith(".jsonl"):
        base = base[: -len(".jsonl")]
    return experiment, job, f"{base}-{digest}.fires.jsonl"


def ledger_path_for_stream(
    stream: str | Path,
    watch_root: str | Path = DEFAULT_WATCH_ROOT,
    results_root: str | Path = RESULTS_ROOT,
) -> Path:
    experiment, job, basename = stream_coordinates(stream, results_root)
    return Path(watch_root) / experiment / job / basename


def _fire_bodies(rows: Sequence[Any]) -> list[dict[str, Any]]:
    """Strip per-run chain fields so fire content compares across runs."""

    bodies = []
    for row in rows:
        payload = row.to_json_dict() if hasattr(row, "to_json_dict") else dict(row)
        for name in _CHAIN_FIELDS:
            payload.pop(name, None)
        bodies.append(payload)
    return bodies


def write_fire_ledger(fires: Sequence[Any], ledger: str | Path) -> str:
    """Deterministic overwrite-or-skip ledger write; returns the action taken.

    ``"unchanged"``: an existing ledger verifies and holds the same fire
    bodies (chain fields aside) — bytes left untouched.  ``"written"``: the
    ledger was atomically replaced.  ``"rewritten_unverified_existing"``: an
    existing ledger FAILED hash-chain verification (tampered or corrupted)
    and was replaced — the caller must surface this, silently healing a
    detected integrity failure would destroy the only signal of tampering.
    ``"no_fires"`` / ``"removed_stale"``: zero fires leave no ledger behind
    (a stale one is deleted).
    """

    ledger = Path(ledger)
    if not fires:
        if ledger.is_file():
            ledger.unlink()
            return "removed_stale"
        return "no_fires"
    new_bodies = _fire_bodies(fires)
    existing_unverified = False
    if ledger.is_file():
        if receipt_watch.verify_fire_ledger(ledger):
            old_rows = [
                record
                for record in receipt_watch.iter_jsonl_records(ledger)
                if "_malformed" not in record
            ]
            if _canonical(_fire_bodies(old_rows)) == _canonical(new_bodies):
                return "unchanged"
        else:
            existing_unverified = True
    ledger.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger.with_name(ledger.name + ".tmp")
    tmp.unlink(missing_ok=True)
    receipt_watch.write_fires(fires, tmp)
    os.replace(tmp, ledger)
    return "rewritten_unverified_existing" if existing_unverified else "written"


def _iter_streams(target: Path) -> list[Path]:
    return sorted(
        path
        for path in target.rglob("*.jsonl")
        if path.is_file() and not path.name.endswith(".fires.jsonl")
    )


def _classify_missing_ledger(stream: Path) -> tuple[str, int | None]:
    """Explain why a stream legitimately has no fire ledger, if it does.

    ``write_fire_ledger`` intentionally removes ledgers for streams with no
    fires. Verification must therefore replay the read-only watch decision
    before calling an absent ledger an integrity finding. Oversized streams
    and streams with no adapted predicates are also deliberate no-ledger
    outcomes from :func:`run_watch_step`.
    """

    if stream.stat().st_size > MAX_STREAM_BYTES:
        return "skipped_too_large", None
    specs = evidence_specs.specs_for_stream(
        stream, sample_limit=evidence_specs.DEFAULT_SAMPLE_RECORDS
    )
    if not specs:
        return "no_predicates", 0
    service = receipt_watch.ReceiptWatchService(specs)
    _, stats = service.eval_stream(receipt_watch.iter_jsonl_records(stream))
    fires = int(stats["fires"])
    return ("no_fires" if fires == 0 else "fires_present"), fires


def run_watch_step(
    target: Path,
    *,
    watch_root: Path,
    results_root: Path,
    sample_limit: int,
) -> dict[str, Any]:
    """Adapt + evaluate the standard pack over every stream; write ledgers."""

    totals = {
        "streams": 0,
        "streams_with_predicates": 0,
        "records": 0,
        "fires": 0,
        "ledgers_written": 0,
        "ledgers_unchanged": 0,
        "ledgers_verified": 0,
        "ledgers_broken": 0,
        "streams_too_large": 0,
        "stream_errors": 0,
    }
    per_stream: list[dict[str, Any]] = []
    for stream in _iter_streams(target):
        totals["streams"] += 1
        entry: dict[str, Any] = {
            "stream": _display_path(stream),
            "ledger": None,
            "ledger_status": None,
            "ledger_verified": None,
            "predicates": 0,
            "records": 0,
            "fires": 0,
            "error": None,
        }
        try:
            if stream.stat().st_size > MAX_STREAM_BYTES:
                totals["streams_too_large"] += 1
                entry["ledger_status"] = "skipped_too_large"
                per_stream.append(entry)
                continue
            specs = evidence_specs.specs_for_stream(stream, sample_limit=sample_limit)
            entry["predicates"] = len(specs)
            ledger = ledger_path_for_stream(stream, watch_root, results_root)
            entry["ledger"] = _display_path(ledger)
            # ``write_fire_ledger`` may remove a stale ledger when this replay
            # currently produces no fires.  Remember an invalid prior ledger
            # before that repair so a rerun cannot silently turn an integrity
            # failure into a clean result.
            existing_ledger_unverified = ledger.is_file() and not receipt_watch.verify_fire_ledger(
                ledger
            )
            preserved_unverified = (
                _preserve_unverified_ledger(ledger)
                if existing_ledger_unverified
                else None
            )
            if not specs:
                entry["ledger_status"] = write_fire_ledger([], ledger)
                if existing_ledger_unverified:
                    entry["ledger_integrity"] = "preexisting_unverified"
                    entry["preserved_unverified_ledger"] = _display_path(
                        preserved_unverified
                    )
                    totals["ledgers_broken"] += 1
                per_stream.append(entry)
                continue
            totals["streams_with_predicates"] += 1
            service = receipt_watch.ReceiptWatchService(specs)
            fires, stats = service.eval_stream(receipt_watch.iter_jsonl_records(stream))
            entry["records"] = stats["records"]
            entry["fires"] = stats["fires"]
            totals["records"] += stats["records"]
            totals["fires"] += stats["fires"]
            # ``write_fire_ledger`` deliberately repairs an unverified ledger
            # so the current watch result remains available.  Preserve the
            # integrity signal at the caller: otherwise a rerun would turn a
            # tampered historical ledger into an apparently clean run.
            entry["ledger_status"] = write_fire_ledger(fires, ledger)
            if existing_ledger_unverified:
                entry["ledger_integrity"] = "preexisting_unverified"
                entry["preserved_unverified_ledger"] = _display_path(
                    preserved_unverified
                )
                totals["ledgers_broken"] += 1
            if entry["ledger_status"] == "written":
                totals["ledgers_written"] += 1
            elif entry["ledger_status"] == "unchanged":
                totals["ledgers_unchanged"] += 1
            if fires:
                verified = bool(receipt_watch.verify_fire_ledger(ledger))
                entry["ledger_verified"] = verified
                totals["ledgers_verified" if verified else "ledgers_broken"] += 1
        except Exception as exc:  # noqa: BLE001 - fail-soft per stream
            totals["stream_errors"] += 1
            entry["error"] = f"{type(exc).__name__}: {exc}"
        per_stream.append(entry)
    return {"status": "ok", **totals, "per_stream": per_stream}


def _preserve_unverified_ledger(ledger: Path) -> Path:
    """Keep immutable bytes from a corrupt ledger before availability repair."""

    raw = ledger.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    preserved = ledger.with_name(f"{ledger.name}.unverified-{digest[:16]}")
    if preserved.is_file():
        if preserved.read_bytes() != raw:
            raise EvidencePipelineError(
                f"unverified-ledger custody collision: {preserved}"
            )
        return preserved
    temporary = preserved.with_name(f".{preserved.name}.{os.getpid()}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, preserved)
    return preserved


def claim_id_for_receipt(receipt: str | Path, results_root: str | Path = RESULTS_ROOT) -> str:
    """Deterministic kebab claim id for a receipt path (auto namespace).

    Derived from the receipt's path relative to ``results_root`` (absolute
    when outside it) plus an 8-hex path digest, so the id is stable across
    reruns and unique even after slug truncation.
    """

    resolved = Path(receipt).resolve()
    try:
        rel = resolved.relative_to(Path(results_root).resolve()).as_posix()
    except ValueError:
        rel = resolved.as_posix()
    digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:8]
    stem = rel[: -len(".json")] if rel.endswith(".json") else rel
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[-80:].strip("-")
    return f"auto-{slug}-{digest}" if slug else f"auto-{digest}"


def _recorded_receipt_sha(row: Any) -> str | None:
    for entry in row.fingerprint.entries:
        if entry.get("kind") == "file" and entry.get("name") == "receipt":
            return entry.get("sha256")
    return None


def run_claims_step(
    target: Path,
    *,
    registry_path: Path,
    results_root: Path,
) -> dict[str, Any]:
    """Ingest every receipt-like JSON (top-level gates dict) non-duplicating.

    Claim identity is :func:`claim_id_for_receipt`; a receipt whose current
    bytes match the recorded receipt fingerprint of the current row for that
    id is skipped, so re-running the pipeline appends no duplicate rows.
    Changed receipt bytes append a fresh row (that is the registry's
    intended update path).
    """

    registry = claims_registry.ClaimsRegistry(registry_path)
    current = registry.current() if registry_path.is_file() else {}
    counters = {
        "receipts_found": 0,
        "receipts_accepted": 0,
        "ingested": 0,
        "skipped_unchanged": 0,
        "skipped_too_large": 0,
        "skipped_invalid": 0,
    }
    ingested: list[dict[str, Any]] = []
    skipped_invalid: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in sorted(target.rglob("*.json")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > MAX_RECEIPT_BYTES:
                counters["skipped_too_large"] += 1
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue  # not a parseable receipt; xref still sees it
            if not isinstance(payload, dict) or not isinstance(payload.get("gates"), dict):
                continue
            counters["receipts_found"] += 1
            claim_id = claim_id_for_receipt(path, results_root)
            row = current.get(claim_id)
            if row is not None and _recorded_receipt_sha(row) == _file_sha256(path):
                counters["skipped_unchanged"] += 1
                counters["receipts_accepted"] += 1
                continue
            try:
                appended = claims_registry.ingest_receipt(
                    registry,
                    path,
                    claim_id,
                    f"auto-ingested gates receipt {_display_path(path)}",
                    "measured",
                    harvest_declared=True,
                )
            except claims_registry.ClaimsRegistryError as exc:
                # A receipt that parses but cannot be ingested (e.g. NaN in its
                # gates dict) is an uningestable receipt, not a pipeline failure:
                # count it like an unparseable one rather than escalating the whole
                # sweep to broken (exit 2). Reserve ``errors`` for step-level faults.
                counters["skipped_invalid"] += 1
                skipped_invalid.append(
                    {"receipt": _display_path(path), "reason": f"{type(exc).__name__}: {exc}"}
                )
                continue
            current[claim_id] = appended
            counters["ingested"] += 1
            counters["receipts_accepted"] += 1
            ingested.append({"claim_id": claim_id, "receipt": _display_path(path)})
        except Exception as exc:  # noqa: BLE001 - fail-soft per receipt
            errors.append(
                {"receipt": _display_path(path), "error": f"{type(exc).__name__}: {exc}"}
            )
    chain_ok = registry.verify_chain() if registry.path.is_file() else True
    return {
        "status": "ok",
        **counters,
        "ingested_claims": ingested,
        "skipped_invalid_receipts": skipped_invalid,
        "errors": errors,
        "chain_ok": chain_ok,
        "registry": str(registry.path),
    }


def run_xref_step(*, xref_root: Path, xref_db: Path) -> dict[str, Any]:
    """One incremental xref sweep over ``xref_root`` into ``xref_db``."""

    report = xref_index.build_index(xref_root, xref_db)
    return {
        "status": "ok",
        "root": str(xref_root),
        "db": str(xref_db),
        **asdict(report),
    }


@dataclass
class PipelineReport:
    """Step-status report for one ``run`` or ``verify`` sweep."""

    command: str
    target: str
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    findings: int = 0
    broken: bool = False
    instrument_status: str = "not-assessed"

    @property
    def exit_code(self) -> int:
        if self.broken:
            return EXIT_BROKEN
        return EXIT_FINDINGS if self.findings else EXIT_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_PIPELINE_SCHEMA,
            "pack": evidence_specs.EVIDENCE_PACK_SCHEMA,
            "command": self.command,
            "target": self.target,
            "steps": {name: dict(step) for name, step in self.steps.items()},
            "findings": self.findings,
            "broken": self.broken,
            "instrument_status": self.instrument_status,
            "exit_code": self.exit_code,
        }


def _target_scope(streams: int, receipts: int, invalid_receipts: int = 0) -> str:
    """Classify which target-local evidence surfaces actually existed."""

    if streams and receipts:
        return "target-streams-and-receipts"
    if streams:
        return "target-streams-only"
    if receipts:
        return "target-receipts-only"
    if invalid_receipts:
        return "evidence-incomplete-invalid-receipts"
    return "evidence-incomplete-empty-target"


def _iter_receipts(target: Path) -> list[Path]:
    """Return parseable, bounded, target-local gate receipts."""

    receipts: list[Path] = []
    for path in sorted(target.rglob("*.json")):
        if not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("gates"), dict):
            receipts.append(path)
    return receipts


def _check_run_guards(
    target: Path, watch_root: Path, registry_path: Path, results_root: Path
) -> None:
    if not target.is_dir():
        raise EvidencePipelineError(f"target is not a directory: {target}")
    watch_resolved = watch_root.resolve()
    for forbidden_root in (results_root.resolve(), target.resolve()):
        if watch_resolved == forbidden_root or forbidden_root in watch_resolved.parents:
            raise EvidencePipelineError(
                f"refusing watch root inside results/target tree: {watch_root}"
            )
    registry_resolved = registry_path.resolve()
    curated = claims_registry.DEFAULT_REGISTRY_PATH.resolve()
    if registry_resolved == curated:
        raise EvidencePipelineError(
            f"refusing to auto-ingest into the curated registry: {curated}"
        )
    results_resolved = results_root.resolve()
    if registry_resolved == results_resolved or results_resolved in registry_resolved.parents:
        raise EvidencePipelineError(
            f"refusing registry path inside the results tree: {registry_path}"
        )


def run_pipeline(
    target: str | Path,
    *,
    watch_root: str | Path = DEFAULT_WATCH_ROOT,
    registry_path: str | Path = DEFAULT_AUTO_REGISTRY,
    xref_db: str | Path = DEFAULT_XREF_DB,
    xref_root: str | Path | None = None,
    results_root: str | Path = RESULTS_ROOT,
    sample_limit: int = evidence_specs.DEFAULT_SAMPLE_RECORDS,
    skip_xref: bool = False,
) -> PipelineReport:
    """WATCH -> CLAIMS -> XREF over one directory, fail-soft per step."""

    target = Path(target)
    watch_root = Path(watch_root)
    registry_path = Path(registry_path)
    results_root = Path(results_root)
    xref_root = Path(xref_root) if xref_root is not None else results_root
    _check_run_guards(target, watch_root, registry_path, results_root)

    report = PipelineReport(command="run", target=str(target.resolve()))
    try:
        report.steps["watch"] = run_watch_step(
            target,
            watch_root=watch_root,
            results_root=results_root,
            sample_limit=sample_limit,
        )
    except Exception as exc:  # noqa: BLE001 - fail-soft per step
        report.steps["watch"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    try:
        report.steps["claims"] = run_claims_step(
            target, registry_path=registry_path, results_root=results_root
        )
    except Exception as exc:  # noqa: BLE001 - fail-soft per step
        report.steps["claims"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if skip_xref:
        report.steps["xref"] = {"status": "skipped"}
    else:
        try:
            report.steps["xref"] = run_xref_step(xref_root=xref_root, xref_db=xref_db)
        except Exception as exc:  # noqa: BLE001 - fail-soft per step
            report.steps["xref"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    watch = report.steps["watch"]
    claims = report.steps["claims"]
    target_streams = int(watch.get("streams", 0))
    target_receipts = int(claims.get("receipts_accepted", 0))
    invalid_receipts = int(claims.get("skipped_invalid", 0))
    report.instrument_status = _target_scope(
        target_streams, target_receipts, invalid_receipts
    )
    report.steps["target_scope"] = {
        "status": "ok" if target_streams or target_receipts else "incomplete",
        "instrument_status": report.instrument_status,
        "streams": target_streams,
        "receipts": target_receipts,
        "invalid_receipts": invalid_receipts,
    }
    report.findings = int(watch.get("fires", 0))
    report.broken = (
        any(step.get("status") == "error" for step in report.steps.values())
        or int(watch.get("ledgers_broken", 0)) > 0
        or int(watch.get("stream_errors", 0)) > 0
        or bool(claims.get("errors"))
        or claims.get("chain_ok") is False
        or (target_streams == 0 and target_receipts == 0)
    )
    return report


def verify_pipeline(
    target: str | Path,
    *,
    watch_root: str | Path = DEFAULT_WATCH_ROOT,
    registry_path: str | Path = DEFAULT_AUTO_REGISTRY,
    results_root: str | Path = RESULTS_ROOT,
) -> PipelineReport:
    """Re-verify fire ledgers for the directory's streams + claims drift check.

    Pure read: ledgers are re-hashed with
    :func:`receipt_watch.verify_fire_ledger` and the claims registry is swept
    with ``check_only=True`` (nothing is ever appended).  Ledger paths are
    recomputed from the streams currently present, so orphan ledgers of
    deleted streams are not covered. Because zero-fire streams intentionally
    have no ledger, an absent ledger is classified by a read-only replay and
    is only counted as missing when the stream currently produces fires (or
    the replay itself fails).
    """

    target = Path(target)
    watch_root = Path(watch_root)
    registry_path = Path(registry_path)
    results_root = Path(results_root)
    if not target.is_dir():
        raise EvidencePipelineError(f"target is not a directory: {target}")

    report = PipelineReport(command="verify", target=str(target.resolve()))
    ledgers: list[dict[str, Any]] = []
    checked = missing = missing_expected = broken_count = 0
    for stream in _iter_streams(target):
        ledger = ledger_path_for_stream(stream, watch_root, results_root)
        if not ledger.is_file():
            try:
                reason, fires = _classify_missing_ledger(stream)
            except Exception as exc:  # noqa: BLE001 - surface as a finding
                missing += 1
                ledgers.append(
                    {
                        "ledger": _display_path(ledger),
                        "verified": False,
                        "expected_missing": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if fires in (None, 0):
                missing_expected += 1
                ledgers.append(
                    {
                        "ledger": _display_path(ledger),
                        "verified": None,
                        "expected_missing": True,
                        "reason": reason,
                        "fires": fires,
                    }
                )
            else:
                missing += 1
                ledgers.append(
                    {
                        "ledger": _display_path(ledger),
                        "verified": False,
                        "expected_missing": False,
                        "reason": reason,
                        "fires": fires,
                    }
                )
            continue
        checked += 1
        ok = bool(receipt_watch.verify_fire_ledger(ledger))
        if not ok:
            broken_count += 1
        ledgers.append({"ledger": _display_path(ledger), "verified": ok})
    report.steps["watch"] = {
        "status": "ok",
        "ledgers_checked": checked,
        "ledgers_missing": missing,
        "ledgers_missing_expected": missing_expected,
        "ledgers_broken": broken_count,
        "ledgers": ledgers,
    }

    receipt_paths = _iter_receipts(target)

    if registry_path.is_file():
        registry = claims_registry.ClaimsRegistry(registry_path)
        try:
            claims_report = claims_registry.verify(registry, check_only=True)
            report.steps["claims"] = {"status": "ok", **claims_report.to_dict()}
            current = registry.current()
            target_claims = []
            missing_claims = 0
            drifted_claims = 0
            for receipt in receipt_paths:
                claim_id = claim_id_for_receipt(receipt, results_root)
                row = current.get(claim_id)
                if row is None:
                    missing_claims += 1
                    state = "missing"
                elif _recorded_receipt_sha(row) != _file_sha256(receipt):
                    drifted_claims += 1
                    state = "drifted"
                else:
                    state = "current"
                target_claims.append(
                    {
                        "claim_id": claim_id,
                        "receipt": _display_path(receipt),
                        "state": state,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - fail-soft per step
            report.steps["claims"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            target_claims = []
            missing_claims = len(receipt_paths)
            drifted_claims = 0
    else:
        report.steps["claims"] = {"status": "skipped", "registry": str(registry_path)}
        target_claims = []
        missing_claims = len(receipt_paths)
        drifted_claims = 0

    stream_count = len(_iter_streams(target))
    receipt_count = len(receipt_paths)
    report.instrument_status = _target_scope(stream_count, receipt_count)
    report.steps["target_scope"] = {
        "status": (
            "incomplete"
            if (stream_count == 0 and receipt_count == 0) or missing_claims
            else "ok"
        ),
        "instrument_status": report.instrument_status,
        "streams": stream_count,
        "receipts": receipt_count,
        "claims_current": receipt_count - missing_claims - drifted_claims,
        "claims_missing": missing_claims,
        "claims_drifted": drifted_claims,
        "claims": target_claims,
    }

    claims = report.steps["claims"]
    # The shared registry may contain unrelated claims.  Preserve its global
    # drift report, but let only this target's receipts determine this target's
    # finding count.
    report.findings = drifted_claims + missing
    report.broken = (
        broken_count > 0
        or claims.get("status") == "error"
        or claims.get("chain_ok") is False
        or missing_claims > 0
        or (stream_count == 0 and receipt_count == 0)
    )
    return report


def _print_human(report: PipelineReport) -> None:
    for name, step in report.steps.items():
        summary = " ".join(
            f"{key}={value}"
            for key, value in sorted(step.items())
            if isinstance(value, (int, bool, str)) and key != "status"
        )
        print(f"{name}: status={step.get('status')} {summary}")
    label = {EXIT_OK: "ok", EXIT_FINDINGS: "findings fired", EXIT_BROKEN: "broken"}
    print(f"exit {report.exit_code} ({label[report.exit_code]})")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saturn-evidence",
        description="Standard post-job evidence pipeline: WATCH -> CLAIMS -> XREF.",
        epilog=(
            "Exit codes: 0 ok; 3 findings fired (watch fires on run, claim drift on "
            "verify); 2 broken (refused input, step error, or ledger/chain integrity "
            "failure)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("target", type=Path, help="job or experiment results directory")
    common.add_argument(
        "--watch-root",
        type=Path,
        default=DEFAULT_WATCH_ROOT,
        help=f"fire-ledger root, never inside results/ (default: {DEFAULT_WATCH_ROOT})",
    )
    common.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_AUTO_REGISTRY,
        help=(
            f"auto claims ledger (default: {DEFAULT_AUTO_REGISTRY}; the curated "
            "claims-registry.jsonl is refused)"
        ),
    )
    common.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help=f"results tree root used for coordinates (default: {RESULTS_ROOT})",
    )
    common.add_argument("--json", action="store_true", dest="as_json")

    run = sub.add_parser(
        "run", parents=[common], help="watch + ingest claims + xref over one directory"
    )
    run.add_argument(
        "--xref-db", type=Path, default=DEFAULT_XREF_DB, help=f"default: {DEFAULT_XREF_DB}"
    )
    run.add_argument(
        "--xref-root",
        type=Path,
        default=None,
        help="xref sweep root (default: --results-root; see module docstring)",
    )
    run.add_argument(
        "--sample-limit", type=int, default=evidence_specs.DEFAULT_SAMPLE_RECORDS
    )
    run.add_argument("--skip-xref", action="store_true", help="skip the xref build step")

    sub.add_parser(
        "verify",
        parents=[common],
        help="re-verify fire ledgers + read-only claims drift check",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            report = run_pipeline(
                args.target,
                watch_root=args.watch_root,
                registry_path=args.registry,
                xref_db=args.xref_db,
                xref_root=args.xref_root,
                results_root=args.results_root,
                sample_limit=args.sample_limit,
                skip_xref=args.skip_xref,
            )
        else:
            report = verify_pipeline(
                args.target,
                watch_root=args.watch_root,
                registry_path=args.registry,
                results_root=args.results_root,
            )
    except (EvidencePipelineError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_BROKEN
    if args.as_json:
        print(json.dumps(report.to_dict(), sort_keys=True, allow_nan=False))
    else:
        _print_human(report)
    return report.exit_code


__all__ = [
    "DEFAULT_AUTO_REGISTRY",
    "DEFAULT_WATCH_ROOT",
    "DEFAULT_XREF_DB",
    "EVIDENCE_PIPELINE_SCHEMA",
    "EXIT_BROKEN",
    "EXIT_FINDINGS",
    "EXIT_OK",
    "MAX_RECEIPT_BYTES",
    "MAX_STREAM_BYTES",
    "REPO_ROOT",
    "RESULTS_ROOT",
    "EvidencePipelineError",
    "PipelineReport",
    "claim_id_for_receipt",
    "ledger_path_for_stream",
    "main",
    "run_claims_step",
    "run_pipeline",
    "run_watch_step",
    "run_xref_step",
    "stream_coordinates",
    "verify_pipeline",
    "write_fire_ledger",
]


if __name__ == "__main__":
    raise SystemExit(main())
