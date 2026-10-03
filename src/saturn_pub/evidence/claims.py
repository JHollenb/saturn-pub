"""Executable claim registry: typed claims that carry their own evidence identity.

Each claim row records the receipt it came from, an optional replay recipe
(worker, config hash, submit command, expected gates), and a dependency
fingerprint over the exact file bytes the claim rests on. :func:`verify`
recomputes those fingerprints and auto-downgrades drifted claims to ``stale``,
so a claim whose inputs changed on disk stops presenting itself as current
evidence. The ledger is append-only JSONL and hash-chained: every row carries
``prev_sha256`` and its own ``sha256`` over canonical JSON of the record minus
the ``sha256`` field, guarded by a sidecar head pointer against tail truncation.

Declared evidence references: ``ingest_receipt(..., harvest_declared=True)``
walks the receipt JSON for sibling-key pairs such as ``{"path": ...,
"sha256": ...}`` or ``{"file": ..., "content_sha256": ...}`` and records them as
``declared`` fingerprint entries carrying the original path string, the declared
digest, and a hash-semantics tag: ``sha256`` next to a file path means raw file
bytes; ``content_sha256``/``config_sha256`` mean a canonical-JSON seal computed
over the payload either with or without its own embedded ``content_sha256``
field (writers that self-seal exclude it, so verification accepts either); any
other ``*sha256`` sibling key is tagged ``unknown`` and never verified -- the
module does not guess hash recipes.

Declared paths are resolved fail-closed against false bindings. Candidates are
the declared absolute path as-is, or -- for relative paths -- the receipt's own
directory (``job-dir``) and an optional caller-supplied ``base_root``, each
normalized and rejected if ``..`` traversal escapes its base. A candidate binds
only when its recomputed digest equals the declared sha (``match_basis:
"digest"``), or -- for an existing absolute path -- by path identity
(``match_basis: "path"``) so a drifted absolute source stays drift-detectable.
For a bare artifact name the receipt directory also permits a recursive
descendant search, but only when exactly one descendant digest-matches.

Unresolvable paths are kept with ``resolution: "missing"`` and :func:`verify`
reports them under their own ``unresolvable`` counter instead of ever marking
the claim stale (a dead reference is not drift -- drift means a resolvable
fingerprint changed).

Honest claim boundary: this is mechanics-only infrastructure. It keeps
bookkeeping about receipts that already exist; nothing here makes, strengthens,
or validates a scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._canonical import canonical_json_bytes, sanitize_non_finite, sha256_hex

try:  # POSIX advisory locking serializes concurrent appends; optional elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

CLAIMS_REGISTRY_SCHEMA = "saturn-pub-claims-registry-v1"
CLAIM_STATUSES = frozenset({"measured", "asserted", "refuted", "qualified", "stale"})
GATE_FLOAT_REL_TOL = 1e-9
DECLARED_SEMANTICS = frozenset({"raw-bytes", "canonical-json-seal", "unknown"})
DECLARED_RESOLUTIONS = frozenset({"absolute", "job-dir", "base-root", "missing"})
DECLARED_MATCH_BASES = frozenset({"digest", "path"})
_SEAL_KEY = "content_sha256"

_CLAIM_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ENTRY_KINDS = frozenset({"file", "literal", "declared"})
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DECLARED_PATH_KEYS = ("path", "file")
_DECLARED_SHA_SEMANTICS = {
    "sha256": "raw-bytes",
    "content_sha256": "canonical-json-seal",
    "config_sha256": "canonical-json-seal",
}


class ClaimsRegistryError(ValueError):
    """Raised when a claim row, receipt, or ledger line cannot be trusted."""


def _canonical(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _canonical_seal_digests(payload: Any) -> list[str]:
    """Digests a ``canonical-json-seal`` declared sha may legitimately match.

    Self-sealing writers compute the seal over the body EXCLUDING the embedded
    ``content_sha256`` field; externally-sealed files carry no embedded seal at
    all. Both digests are returned (seal-excluded first when one is embedded)
    and agreement with either counts as a match.
    """

    sanitized = sanitize_non_finite(payload)
    digests = []
    if isinstance(sanitized, Mapping) and _SEAL_KEY in sanitized:
        body = {key: value for key, value in sanitized.items() if key != _SEAL_KEY}
        digests.append(sha256_hex(_canonical(body)))
    digests.append(sha256_hex(_canonical(sanitized)))
    return digests


def _file_sha256(path: Path) -> str | None:
    """Hash file contents; ``None`` records a missing file (that is drift)."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file_reference(path: str | Path) -> Path:
    """Resolve a dependency path; relative paths anchor to the process CWD."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def _now_iso(now: str | datetime | None = None) -> str:
    if now is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(now, datetime):
        return now.isoformat(timespec="seconds")
    return str(now)


def _validate_claim_id(claim_id: str) -> None:
    if not _CLAIM_ID_RE.fullmatch(str(claim_id)):
        raise ClaimsRegistryError(f"claim_id must be a kebab-case slug: {claim_id!r}")


def _check_finite_gates(value: Any, receipt_path: Path) -> None:
    """Fail closed on non-finite gate values (NaN/Infinity survive json.loads)."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ClaimsRegistryError(
            f"receipt gates contain a non-finite value ({value!r}): {receipt_path}"
        )
    if isinstance(value, Mapping):
        for item in value.values():
            _check_finite_gates(item, receipt_path)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_finite_gates(item, receipt_path)


def _validate_status(status: str) -> None:
    if status not in CLAIM_STATUSES:
        raise ClaimsRegistryError(f"status must be one of {sorted(CLAIM_STATUSES)}: {status!r}")


def _validate_declared_entry(entry: Mapping[str, Any]) -> None:
    """Fail closed on a malformed declared entry (built or read from a ledger)."""

    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise ClaimsRegistryError(f"declared entry needs a non-empty path string: {path!r}")
    sha256 = entry.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_HEX_RE.fullmatch(sha256):
        raise ClaimsRegistryError(f"declared entry sha256 must be 64 hex chars: {sha256!r}")
    if entry.get("semantics") not in DECLARED_SEMANTICS:
        raise ClaimsRegistryError(
            f"declared entry semantics must be one of {sorted(DECLARED_SEMANTICS)}: "
            f"{entry.get('semantics')!r}"
        )
    resolution = entry.get("resolution")
    if resolution not in DECLARED_RESOLUTIONS:
        raise ClaimsRegistryError(
            f"declared entry resolution must be one of {sorted(DECLARED_RESOLUTIONS)}: "
            f"{resolution!r}"
        )
    resolved_path = entry.get("resolved_path")
    match_basis = entry.get("match_basis")
    if resolution == "missing":
        if resolved_path is not None:
            raise ClaimsRegistryError(
                "declared entry with resolution 'missing' must not carry a resolved_path: "
                f"{resolved_path!r}"
            )
        if match_basis is not None:
            raise ClaimsRegistryError(
                "declared entry with resolution 'missing' must not carry a match_basis: "
                f"{match_basis!r}"
            )
    else:
        if not isinstance(resolved_path, str) or not resolved_path:
            raise ClaimsRegistryError(
                f"declared entry with resolution {resolution!r} needs a resolved_path: "
                f"{resolved_path!r}"
            )
        if match_basis is not None and match_basis not in DECLARED_MATCH_BASES:
            raise ClaimsRegistryError(
                f"declared entry match_basis must be one of {sorted(DECLARED_MATCH_BASES)}: "
                f"{match_basis!r}"
            )


@dataclass
class DependencyFingerprint:
    """Content hashes of the files (and pinned literals) a claim depends on.

    ``file`` entries are recomputable from disk and can drift; ``literal``
    entries pin a digest directly and never drift on their own. ``declared``
    entries carry a (path, sha256) pair harvested from inside a receipt: a
    resolvable one is recomputed per its ``semantics`` tag and can drift or go
    missing; one that never resolved at ingest (``resolution: "missing"``) is
    counted by :meth:`unresolvable` and never drifts; one tagged ``semantics:
    "unknown"`` is never verified.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add_file(self, path: str | Path, *, name: str | None = None) -> dict[str, Any]:
        path = Path(path)
        resolved = _resolve_file_reference(path)
        entry = {
            "kind": "file",
            "name": str(name or path.name),
            "path": str(path),
            "sha256": _file_sha256(resolved),
        }
        self.entries.append(entry)
        return entry

    def add_literal(self, name: str, sha256: str) -> dict[str, Any]:
        entry = {"kind": "literal", "name": str(name), "sha256": str(sha256)}
        self.entries.append(entry)
        return entry

    def add_declared(
        self,
        *,
        path: str,
        sha256: str,
        semantics: str,
        resolution: str,
        resolved_path: str | None = None,
        match_basis: str | None = None,
    ) -> dict[str, Any]:
        entry = {
            "kind": "declared",
            "name": str(path),
            "path": str(path),
            "resolved_path": str(resolved_path) if resolved_path is not None else None,
            "resolution": str(resolution),
            "match_basis": str(match_basis) if match_basis is not None else None,
            "sha256": str(sha256),
            "semantics": str(semantics),
        }
        _validate_declared_entry(entry)
        self.entries.append(entry)
        return entry

    @staticmethod
    def _declared_observed(entry: Mapping[str, Any]) -> tuple[list[str] | None, str | None]:
        """Recompute a resolvable declared entry: (observed_shas, error_reason)."""

        target = Path(entry["resolved_path"])
        if entry.get("semantics") == "raw-bytes":
            observed = _file_sha256(target)
            return ([observed] if observed is not None else None), None
        if not target.is_file():
            return None, None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "unparseable"
        try:
            return _canonical_seal_digests(payload), None
        except (TypeError, ValueError):
            return None, "uncanonicalizable"

    def drift(self) -> list[str]:
        """Recompute recomputable entries; describe every one that disagrees."""

        reasons = []
        for entry in self.entries:
            kind = entry.get("kind")
            if kind == "declared":
                if entry.get("resolution") == "missing" or entry.get("semantics") == "unknown":
                    continue
                path = entry.get("path")
                resolved = entry.get("resolved_path")
                recorded = entry.get("sha256")
                observed, error = self._declared_observed(entry)
                if error is not None:
                    reasons.append(f"declared {path} ({resolved}) {error}")
                elif observed is None:
                    reasons.append(f"declared {path} ({resolved}) missing")
                elif recorded not in observed:
                    reasons.append(
                        f"declared {path} ({resolved}) changed {recorded[:12]}->{observed[0][:12]}"
                    )
                continue
            if kind != "file":
                continue
            path = entry.get("path")
            name = entry.get("name") or path
            recorded = entry.get("sha256")
            observed = _file_sha256(_resolve_file_reference(path)) if path else None
            if recorded == observed and recorded is not None:
                continue
            if observed is None:
                reasons.append(f"{name} ({path}) missing")
            elif recorded is None:
                reasons.append(f"{name} ({path}) appeared {observed[:12]}")
            else:
                reasons.append(f"{name} ({path}) changed {recorded[:12]}->{observed[:12]}")
        return sorted(reasons)

    def unresolvable(self) -> int:
        """Count declared entries whose path never resolved at ingest."""

        return sum(
            1
            for entry in self.entries
            if entry.get("kind") == "declared" and entry.get("resolution") == "missing"
        )

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [dict(entry) for entry in self.entries]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DependencyFingerprint:
        entries = []
        for entry in payload.get("entries") or []:
            if not isinstance(entry, Mapping):
                raise ClaimsRegistryError("fingerprint entries must be objects")
            if entry.get("kind") not in _ENTRY_KINDS:
                raise ClaimsRegistryError(f"unknown fingerprint entry kind: {entry.get('kind')!r}")
            if entry.get("kind") == "declared":
                _validate_declared_entry(entry)
            entries.append(dict(entry))
        return cls(entries=entries)


def fingerprint_paths(paths: Sequence[str | Path]) -> DependencyFingerprint:
    """Fingerprint file contents; a missing file is recorded with sha256=None."""

    fingerprint = DependencyFingerprint()
    for path in paths:
        fingerprint.add_file(path)
    return fingerprint


def _declared_digest_matches(candidate: Path, sha256: str, semantics: str) -> bool:
    """True when ``candidate``'s recomputed digest (per semantics) equals the sha.

    ``unknown`` semantics can never digest-match (guessing the recipe is exactly
    the false-positive class this module refuses).
    """

    if semantics == "raw-bytes":
        return _file_sha256(candidate) == sha256
    if semantics == "canonical-json-seal":
        if not candidate.is_file():
            return False
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return sha256 in _canonical_seal_digests(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return False
    return False


def _resolve_declared_path(
    declared: str, job_dir: Path, base_root: Path | None, sha256: str, semantics: str
) -> tuple[str, str | None, str | None]:
    """Resolve a declared path to ``(resolution, resolved_path, match_basis)``.

    Fail-closed against false bindings. Absolute paths are tried as-is only;
    relative paths are tried against the receipt's directory then an optional
    ``base_root``. Every candidate is lexically normalized and a relative path
    whose ``..`` traversal escapes its base is never a candidate. A candidate
    binds only when its recomputed digest equals the declared sha
    (``match_basis: "digest"``, at any candidate root), or -- for an existing
    absolute path -- by path identity (``match_basis: "path"``). A bare artifact
    name may additionally resolve to a unique digest-matching descendant of the
    receipt directory. Anything else -> ``("missing", None, None)``.
    """

    declared_path = Path(declared)
    if declared_path.is_absolute():
        candidate = Path(os.path.normpath(str(declared_path)))
        if not candidate.is_file():
            return "missing", None, None
        if _declared_digest_matches(candidate, sha256, semantics):
            return "absolute", str(candidate), "digest"
        return "absolute", str(candidate), "path"

    bases = [("job-dir", job_dir)]
    if base_root is not None:
        bases.append(("base-root", base_root))
    contained: list[tuple[str, Path]] = []
    for resolution, base in bases:
        base = Path(os.path.normpath(str(base)))
        candidate = Path(os.path.normpath(str(base / declared_path)))
        if candidate != base and base not in candidate.parents:
            continue  # traversal escaped its base: never a candidate
        if candidate.is_file():
            contained.append((resolution, candidate))
    for resolution, candidate in contained:
        if _declared_digest_matches(candidate, sha256, semantics):
            return resolution, str(candidate), "digest"

    # A bare artifact name may live below the receipt directory (for example
    # ``depth.png`` under ``job/scene/`` while the receipt is at ``job/``). This
    # is safe only when the digest identifies exactly one descendant.
    normalized_declared = Path(os.path.normpath(str(declared_path)))
    if len(normalized_declared.parts) == 1 and normalized_declared.name not in {".", ".."}:
        matches: dict[str, Path] = {}
        try:
            job_root = job_dir.resolve()
            for candidate in job_root.rglob(normalized_declared.name):
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if resolved == job_root or job_root not in resolved.parents:
                    continue
                if resolved.is_file() and _declared_digest_matches(resolved, sha256, semantics):
                    matches[str(resolved)] = resolved
        except OSError:
            matches = {}
        if len(matches) == 1:
            resolved = next(iter(matches.values()))
            return "job-dir", str(resolved), "digest"

    return "missing", None, None


def harvest_declared_refs(
    receipt: Mapping[str, Any],
    receipt_path: str | Path,
    *,
    base_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Walk a receipt for declared (path, sha256) sibling-key evidence refs.

    A declared ref is any JSON object carrying one of the path keys
    ``path``/``file`` (string value) plus at least one sibling ``*sha256`` key
    whose value is a 64-hex digest. Semantics come from the sha key name only:
    ``sha256`` -> ``raw-bytes``, ``content_sha256``/``config_sha256`` ->
    ``canonical-json-seal``, anything else -> ``unknown``. Refs are deduplicated
    on (path, sha256, semantics) in first-seen order. Each ref is resolved with
    the fail-closed rules of :func:`_resolve_declared_path` against the receipt
    directory and the optional ``base_root``.
    """

    receipt_path = Path(receipt_path)
    job_dir = receipt_path.resolve().parent
    root = Path(base_root) if base_root is not None else None
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            declared = next(
                (
                    node[key]
                    for key in _DECLARED_PATH_KEYS
                    if isinstance(node.get(key), str) and node[key]
                ),
                None,
            )
            if declared is not None:
                for key, value in node.items():
                    if key in _DECLARED_PATH_KEYS or not key.endswith("sha256"):
                        continue
                    if not isinstance(value, str) or not _SHA256_HEX_RE.fullmatch(value):
                        continue
                    semantics = _DECLARED_SHA_SEMANTICS.get(key, "unknown")
                    dedup_key = (declared, value.lower(), semantics)
                    if dedup_key in seen:
                        continue
                    resolution, resolved, basis = _resolve_declared_path(
                        declared, job_dir, root, value.lower(), semantics
                    )
                    seen[dedup_key] = {
                        "kind": "declared",
                        "name": declared,
                        "path": declared,
                        "resolved_path": resolved,
                        "resolution": resolution,
                        "match_basis": basis,
                        "sha256": value.lower(),
                        "semantics": semantics,
                    }
            for value in node.values():
                visit(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value)

    visit(receipt)
    return list(seen.values())


@dataclass
class ReplayRecipe:
    """How to re-earn a claim: worker, config identity, submit command, gates."""

    worker: str
    config_sha256: str | None = None
    submit_command: str | None = None
    expected_gates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker": self.worker,
            "config_sha256": self.config_sha256,
            "submit_command": self.submit_command,
            "expected_gates": dict(self.expected_gates),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReplayRecipe:
        return cls(
            worker=str(payload.get("worker") or "unknown"),
            config_sha256=payload.get("config_sha256"),
            submit_command=payload.get("submit_command"),
            expected_gates=dict(payload.get("expected_gates") or {}),
        )


@dataclass
class ClaimRow:
    """One ledger row: a typed claim plus the evidence identity it rests on."""

    claim_id: str
    text: str
    status: str
    receipt_path: str
    receipt_content_sha256: str | None = None
    gates: dict[str, Any] = field(default_factory=dict)
    recipe: ReplayRecipe | None = None
    fingerprint: DependencyFingerprint = field(default_factory=DependencyFingerprint)
    created_at: str = ""
    updated_at: str = ""
    prev_sha256: str | None = None
    sha256: str | None = None
    stale_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CLAIMS_REGISTRY_SCHEMA,
            "claim_id": self.claim_id,
            "text": self.text,
            "status": self.status,
            "receipt_path": self.receipt_path,
            "receipt_content_sha256": self.receipt_content_sha256,
            "gates": dict(self.gates),
            "recipe": self.recipe.to_dict() if self.recipe is not None else None,
            "fingerprint": self.fingerprint.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "prev_sha256": self.prev_sha256,
            "sha256": self.sha256,
            "stale_reason": self.stale_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClaimRow:
        recipe = payload.get("recipe")
        fingerprint = payload.get("fingerprint")
        return cls(
            claim_id=str(payload.get("claim_id") or ""),
            text=str(payload.get("text") or ""),
            status=str(payload.get("status") or ""),
            receipt_path=str(payload.get("receipt_path") or ""),
            receipt_content_sha256=payload.get("receipt_content_sha256"),
            gates=dict(payload.get("gates") or {}),
            recipe=ReplayRecipe.from_dict(recipe) if isinstance(recipe, Mapping) else None,
            fingerprint=(
                DependencyFingerprint.from_dict(fingerprint)
                if isinstance(fingerprint, Mapping)
                else DependencyFingerprint()
            ),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            prev_sha256=payload.get("prev_sha256"),
            sha256=payload.get("sha256"),
            stale_reason=payload.get("stale_reason"),
        )


class ClaimsRegistry:
    """Append-only hash-chained JSONL ledger; current view = latest row per claim."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def head_path(self) -> Path:
        """Sidecar head pointer guarding against silent tail truncation."""

        return self.path.with_name(self.path.name + ".head")

    @property
    def lock_path(self) -> Path:
        """Sidecar lock file serializing read-tail+append+head-write."""

        return self.path.with_name(self.path.name + ".lock")

    def _write_head(self, tail_sha256: str, rows: int) -> None:
        payload = json.dumps(
            {"tail_sha256": tail_sha256, "rows": rows}, sort_keys=True, separators=(",", ":")
        )
        tmp = self.head_path.with_name(self.head_path.name + ".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp, self.head_path)

    def _raw_rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ClaimsRegistryError(
                    f"ledger line {lineno} is not valid JSON: {self.path}"
                ) from exc
            if not isinstance(payload, dict):
                raise ClaimsRegistryError(f"ledger line {lineno} is not an object: {self.path}")
            rows.append(payload)
        return rows

    def rows(self) -> list[ClaimRow]:
        return [ClaimRow.from_dict(payload) for payload in self._raw_rows()]

    def append(self, row: ClaimRow) -> ClaimRow:
        """Chain ``row`` onto the ledger tail, sealing prev_sha256 and sha256."""

        _validate_claim_id(row.claim_id)
        _validate_status(row.status)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = self.lock_path.open("a", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            raw = self._raw_rows()
            row.prev_sha256 = raw[-1].get("sha256") if raw else None
            payload = row.to_dict()
            payload.pop("sha256")
            row.sha256 = sha256_hex(_canonical(payload))
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(row.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                )
            self._write_head(row.sha256, len(raw) + 1)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        return row

    def current(self) -> dict[str, ClaimRow]:
        latest: dict[str, ClaimRow] = {}
        for row in self.rows():
            latest[row.claim_id] = row
        return latest

    def history(self, claim_id: str) -> list[ClaimRow]:
        return [row for row in self.rows() if row.claim_id == claim_id]

    def verify_chain(self) -> bool:
        """Recompute every row hash and prev link over the whole file.

        When the sidecar head pointer exists it must agree with the observed
        tail hash and row count, which catches silent truncation of trailing
        rows. A missing head file marks a legacy ledger and falls back to
        chain-only verification.
        """

        prev: str | None = None
        count = 0
        for payload in self._raw_rows():
            record = dict(payload)
            stored = record.pop("sha256", None)
            if stored is None:
                return False
            if record.get("prev_sha256") != prev:
                return False
            if sha256_hex(_canonical(record)) != stored:
                return False
            prev = stored
            count += 1
        if self.head_path.is_file():
            try:
                head = json.loads(self.head_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return False
            if not isinstance(head, dict):
                return False
            if head.get("tail_sha256") != prev or head.get("rows") != count:
                return False
        return True


def ingest_receipt(
    registry: ClaimsRegistry,
    receipt_path: str | Path,
    claim_id: str,
    text: str,
    status: str = "measured",
    *,
    dependency_paths: Sequence[str | Path] | None = None,
    worker: str | None = None,
    submit_command: str | None = None,
    harvest_declared: bool = False,
    base_root: str | Path | None = None,
    now: str | datetime | None = None,
) -> ClaimRow:
    """Parse one receipt JSON into a chained claim row and append it.

    Captures ``config_sha256``, ``content_sha256``, and ``gates`` when the
    receipt exposes them, and fingerprints the receipt file plus any declared
    dependency paths. With ``harvest_declared=True`` the receipt body is also
    walked for declared (path, sha256) evidence refs (see
    :func:`harvest_declared_refs`); ``base_root`` adds a second resolution root
    for relative declared paths.
    """

    receipt_path = Path(receipt_path)
    if not receipt_path.is_file():
        raise ClaimsRegistryError(f"receipt does not exist: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClaimsRegistryError(f"receipt is not valid JSON: {receipt_path}") from exc
    if not isinstance(receipt, Mapping):
        raise ClaimsRegistryError(f"receipt root must be an object: {receipt_path}")

    raw_gates = receipt.get("gates")
    gates = dict(raw_gates) if isinstance(raw_gates, Mapping) else {}
    _check_finite_gates(gates, receipt_path)
    content_sha256 = receipt.get("content_sha256")
    content_sha256 = str(content_sha256) if content_sha256 is not None else None
    config_sha256 = receipt.get("config_sha256")
    config_sha256 = str(config_sha256) if config_sha256 is not None else None

    fingerprint = DependencyFingerprint()
    fingerprint.add_file(receipt_path, name="receipt")
    if content_sha256 is not None:
        fingerprint.add_literal("receipt_content_sha256", content_sha256)
    for dependency in dependency_paths or []:
        fingerprint.add_file(dependency)
    if harvest_declared:
        for entry in harvest_declared_refs(receipt, receipt_path, base_root=base_root):
            fingerprint.add_declared(
                path=entry["path"],
                sha256=entry["sha256"],
                semantics=entry["semantics"],
                resolution=entry["resolution"],
                resolved_path=entry["resolved_path"],
                match_basis=entry["match_basis"],
            )

    recipe = None
    if worker is not None or submit_command is not None:
        recipe = ReplayRecipe(
            worker=str(worker) if worker is not None else "unknown",
            config_sha256=config_sha256,
            submit_command=submit_command,
            expected_gates=dict(gates),
        )

    stamp = _now_iso(now)
    row = ClaimRow(
        claim_id=claim_id,
        text=str(text),
        status=status,
        receipt_path=str(receipt_path),
        receipt_content_sha256=content_sha256,
        gates=gates,
        recipe=recipe,
        fingerprint=fingerprint,
        created_at=stamp,
        updated_at=stamp,
    )
    return registry.append(row)


@dataclass
class VerifyReport:
    """Outcome of one drift sweep over the registry's current claims."""

    checked: int = 0
    still_valid: int = 0
    newly_stale: int = 0
    restored: int = 0
    unresolvable: int = 0
    chain_ok: bool = True
    newly_stale_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "still_valid": self.still_valid,
            "newly_stale": self.newly_stale,
            "restored": self.restored,
            "unresolvable": self.unresolvable,
            "chain_ok": self.chain_ok,
            "newly_stale_claims": list(self.newly_stale_claims),
        }


def verify(
    registry: ClaimsRegistry,
    *,
    check_only: bool = False,
    now: str | datetime | None = None,
) -> VerifyReport:
    """Recompute every current claim's fingerprint; downgrade drifted claims.

    Idempotent: a claim already stale for the same reason set is checked but not
    re-staled. Stale rows keep the original (expected) fingerprint so later
    restoration stays detectable: a currently-stale claim whose recomputed
    fingerprint matches the original expectation again is counted as
    ``restored`` (not auto-un-staled, not counted in ``still_valid``).

    Declared refs unresolvable at ingest are totalled in ``unresolvable`` and
    never contribute to drift. With ``check_only=True`` nothing is appended.
    """

    report = VerifyReport(chain_ok=registry.verify_chain())
    stamp = _now_iso(now)
    for claim_id, claim in sorted(registry.current().items()):
        report.checked += 1
        report.unresolvable += claim.fingerprint.unresolvable()
        reasons = claim.fingerprint.drift()
        if not reasons:
            if claim.status == "stale":
                report.restored += 1
            else:
                report.still_valid += 1
            continue
        reason_text = "; ".join(reasons)
        if claim.status == "stale" and claim.stale_reason == reason_text:
            continue
        report.newly_stale += 1
        report.newly_stale_claims.append(claim_id)
        if check_only:
            continue
        registry.append(
            ClaimRow(
                claim_id=claim.claim_id,
                text=claim.text,
                status="stale",
                receipt_path=claim.receipt_path,
                receipt_content_sha256=claim.receipt_content_sha256,
                gates=dict(claim.gates),
                recipe=claim.recipe,
                fingerprint=claim.fingerprint,
                created_at=claim.created_at,
                updated_at=stamp,
                stale_reason=reason_text,
            )
        )
    return report


def _gate_value_matches(expected: Any, observed: Any) -> bool:
    if isinstance(expected, bool) or isinstance(observed, bool):
        return isinstance(expected, bool) and isinstance(observed, bool) and expected == observed
    if expected == observed:
        return True
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        try:
            return math.isclose(float(expected), float(observed), rel_tol=GATE_FLOAT_REL_TOL)
        except (OverflowError, ValueError):
            return bool(expected == observed)
    return bool(expected == observed)


def gates_match(claim: ClaimRow, receipt_dict: Mapping[str, Any]) -> tuple[bool, list[dict]]:
    """Compare a claim's expected gates against a fresh receipt's gates.

    Bools and strings compare exactly; floats compare with relative tolerance
    ``1e-9``. Uses the recipe's expected gates when a recipe is present,
    otherwise the gate snapshot captured at ingest.
    """

    expected = claim.gates
    if claim.recipe is not None and claim.recipe.expected_gates:
        expected = claim.recipe.expected_gates
    raw_observed = receipt_dict.get("gates")
    observed = dict(raw_observed) if isinstance(raw_observed, Mapping) else {}
    mismatches = []
    for name in sorted(expected):
        if name not in observed:
            mismatches.append(
                {"gate": name, "expected": expected[name], "observed": None, "reason": "missing"}
            )
        elif not _gate_value_matches(expected[name], observed[name]):
            mismatches.append(
                {
                    "gate": name,
                    "expected": expected[name],
                    "observed": observed[name],
                    "reason": "value",
                }
            )
    return (not mismatches, mismatches)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="claims ledger path (append-only JSONL)",
    )
    parser = argparse.ArgumentParser(
        description="Executable claim registry with drift-aware verification"
    )
    sub = parser.add_subparsers(dest="claims_command", required=True)

    ingest = sub.add_parser("ingest", parents=[common], help="ingest one receipt as a claim row")
    ingest.add_argument("receipt", type=Path)
    ingest.add_argument("--claim-id", required=True)
    ingest.add_argument("--text", required=True)
    ingest.add_argument("--status", default="measured", choices=sorted(CLAIM_STATUSES))
    ingest.add_argument(
        "--dep",
        dest="dependency_paths",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help="dependency file fingerprinted with the receipt (repeatable)",
    )
    ingest.add_argument("--worker")
    ingest.add_argument("--submit-command")
    ingest.add_argument("--base-root", type=Path, default=None, help="second declared-ref root")
    ingest.add_argument(
        "--harvest-declared",
        action="store_true",
        help=(
            "walk the receipt JSON for declared (path, sha256) sibling-key evidence refs "
            "and record them as 'declared' fingerprint entries with hash semantics tags"
        ),
    )

    list_parser = sub.add_parser("list", parents=[common], help="list current claims")
    list_parser.add_argument("--status", choices=sorted(CLAIM_STATUSES))
    list_parser.add_argument("--json", action="store_true")

    verify_parser = sub.add_parser(
        "verify",
        parents=[common],
        help="recompute fingerprints; downgrade drifted claims",
        description=(
            "Recompute every current claim's dependency fingerprint. Default mode appends "
            "a stale row for each newly drifted claim; exit 0 when the chain verifies, "
            "1 when it is broken. With --check nothing is ever written: exit 0 = all "
            "claims valid, 3 = newly-stale/drift detected, 2 = hash chain broken."
        ),
    )
    verify_parser.add_argument("--json", action="store_true")
    verify_parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "read-only CI mode: never appends to the ledger; "
            "exit 0 all valid, 3 newly-stale-or-drift detected, 2 chain broken"
        ),
    )

    history = sub.add_parser("history", parents=[common], help="show every row for one claim")
    history.add_argument("claim_id")
    history.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = ClaimsRegistry(args.registry)
    if args.claims_command == "ingest":
        row = ingest_receipt(
            registry,
            args.receipt,
            args.claim_id,
            args.text,
            args.status,
            dependency_paths=args.dependency_paths,
            worker=args.worker,
            submit_command=args.submit_command,
            harvest_declared=args.harvest_declared,
            base_root=args.base_root,
        )
        declared = [entry for entry in row.fingerprint.entries if entry.get("kind") == "declared"]
        print(
            json.dumps(
                {
                    "claim_id": row.claim_id,
                    "status": row.status,
                    "sha256": row.sha256,
                    "registry": str(registry.path),
                    "declared_refs": len(declared),
                    "declared_unresolvable": sum(
                        1 for entry in declared if entry["resolution"] == "missing"
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.claims_command == "list":
        rows = sorted(registry.current().values(), key=lambda row: row.claim_id)
        if args.status:
            rows = [row for row in rows if row.status == args.status]
        if args.json:
            print(json.dumps({row.claim_id: row.to_dict() for row in rows}, sort_keys=True))
        else:
            for row in rows:
                print(f"{row.claim_id}\t{row.status}\t{row.updated_at}\t{row.text}")
        return 0
    if args.claims_command == "verify":
        report = verify(registry, check_only=args.check)
        if args.json:
            print(json.dumps(report.to_dict(), sort_keys=True))
        else:
            print(
                f"checked={report.checked} still_valid={report.still_valid} "
                f"newly_stale={report.newly_stale} restored={report.restored} "
                f"unresolvable={report.unresolvable} chain_ok={report.chain_ok}"
            )
        if args.check:
            if not report.chain_ok:
                return 2
            return 3 if report.newly_stale else 0
        return 0 if report.chain_ok else 1
    if args.claims_command == "history":
        rows = registry.history(args.claim_id)
        if args.json:
            print(json.dumps([row.to_dict() for row in rows], sort_keys=True))
        else:
            for row in rows:
                print(f"{row.updated_at}\t{row.status}\t{row.sha256}\t{row.stale_reason or ''}")
        return 0
    raise ClaimsRegistryError(f"unknown command: {args.claims_command!r}")


__all__ = [
    "CLAIMS_REGISTRY_SCHEMA",
    "CLAIM_STATUSES",
    "DECLARED_MATCH_BASES",
    "DECLARED_RESOLUTIONS",
    "DECLARED_SEMANTICS",
    "ClaimRow",
    "ClaimsRegistry",
    "ClaimsRegistryError",
    "DependencyFingerprint",
    "ReplayRecipe",
    "VerifyReport",
    "fingerprint_paths",
    "gates_match",
    "harvest_declared_refs",
    "ingest_receipt",
    "verify",
]


if __name__ == "__main__":
    raise SystemExit(main())
