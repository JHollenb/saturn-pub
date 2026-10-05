"""Watchpoint predicates over streams of tensor-free receipt records.

Saturn runs already emit small JSON receipts (gates, metrics, environment).
This module evaluates declarative watch predicates over such record streams
and, on fire, emits a hash-chained cut-request record so a later scheduler
hop can turn the fire into a durable StateCut at the offending step.

Boundary: this is the local predicate/ledger core only.  mrun scheduler
``/api/watch`` integration and durable mid-run StateCut publication are
follow-up work (request handle ``mrun-watch-agent-breakpoints`` in
``saturn/request.jsonl``).  This is new mechanics-only infrastructure; it
makes no scientific claim about any model.

Mechanics that matter to callers:

* Field access is a dotted path (``metrics.ce``); any missing segment or
  non-dict traversal yields a sentinel and the record simply does not fire
  (counted under ``missing_field``, which also covers values that cannot be
  coerced for a numeric operator).
* Non-finite policy: the ``non_finite`` op fires when the observed value is
  ``nan``/``+inf``/``-inf`` (its threshold is ignored and may be null).  For
  every numeric op (``gt``/``lt``/``ge``/``le``/``abs_delta_gt``/
  ``rel_delta_gt``) a non-finite observed value counts as ``missing_field``
  (unevaluable): it never fires and never enters the delta history, so a
  later real jump is compared against the last *finite* value.
* Delta operators compare the current value to the value observed ``window``
  *observations* earlier for the same predicate; records missing the field do
  not advance the delta history.
* Fire records are hash-chained per segment: each service instance mints one
  ``segment_id`` (uuid4 hex) and each fire carries ``prev_sha256`` and its own
  ``sha256`` over canonical JSON (``sort_keys=True``,
  ``separators=(",", ":")``) of the record minus its own ``sha256`` field.
  The chain links ``prev_sha256`` within the same segment only; the first
  fire of a segment has ``prev_sha256`` ``None``.  The ledger is append-only
  and multi-writer-tolerant across segments: verification detects in-place
  edits and reordering within a segment but NOT truncation of a segment tail
  or deletion of whole segments — an external head pointer is the caller's
  job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import uuid
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RECEIPT_WATCH_SCHEMA = "saturn-receipt-watch-v1"
SUPPORTED_OPS = frozenset(
    {"gt", "lt", "ge", "le", "eq", "ne", "abs_delta_gt", "rel_delta_gt", "regex", "non_finite"}
)
_DELTA_OPS = frozenset({"abs_delta_gt", "rel_delta_gt"})
_NUMERIC_OPS = frozenset({"gt", "lt", "ge", "le"}) | _DELTA_OPS

MISSING = object()
"""Sentinel returned by :func:`lookup_dotted` when a path cannot be resolved."""


class ReceiptWatchError(ValueError):
    """Raised when a watch predicate specification cannot be interpreted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sanitize_non_finite(value: Any) -> Any:
    """Recursively replace non-finite floats with strict-JSON-safe strings.

    ``nan`` -> ``"NaN"``, ``inf`` -> ``"Infinity"``, ``-inf`` -> ``"-Infinity"``.
    Fire-record payloads (``observed``/``reference`` included) pass through
    this before hashing and before ledger serialization so ledger lines stay
    strict JSON.
    """

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {key: _sanitize_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_non_finite(item) for item in value]
    return value


def _split_lines(text: str) -> list[str]:
    """Split on ``\\n`` only, stripping one trailing ``\\r`` per line.

    Never ``str.splitlines()``: U+2028/U+2029/U+0085 inside JSON strings are
    line breaks to ``splitlines`` and would shatter records.
    """

    return [raw[:-1] if raw.endswith("\r") else raw for raw in text.split("\n")]


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def fire_sha256(fire_dict: Mapping[str, Any]) -> str:
    """Hash a fire record over canonical JSON of everything but ``sha256``."""

    body = {key: value for key, value in fire_dict.items() if key != "sha256"}
    return _sha256(body)


def lookup_dotted(record: Any, path: str) -> Any:
    """Resolve a dotted path; return :data:`MISSING` on any failed segment."""

    value = record
    for segment in str(path).split("."):
        if not isinstance(value, Mapping) or segment not in value:
            return MISSING
        value = value[segment]
    return value


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _values_equal(observed: Any, threshold: Any) -> bool:
    left, right = _as_float(observed), _as_float(threshold)
    if left is not None and right is not None:
        return left == right
    return str(observed) == str(threshold)


@dataclass
class WatchPredicateSpec:
    """One declarative watchpoint over a receipt-record field.

    ``cooldown`` counts every evaluated record for that predicate, evaluable
    or not: records with the field missing or unevaluable still tick the
    cooldown down.  The ``non_finite`` op ignores ``threshold`` (it may be
    null) and fires when the observed value is ``nan``/``+inf``/``-inf``.
    """

    predicate_id: str
    field: str
    op: str
    threshold: float | str | None
    window: int = 1
    cooldown: int = 0
    fire_once: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.predicate_id).strip():
            raise ReceiptWatchError("predicate_id cannot be empty")
        if not str(self.field).strip():
            raise ReceiptWatchError(f"{self.predicate_id}: field cannot be empty")
        if self.op not in SUPPORTED_OPS:
            raise ReceiptWatchError(
                f"{self.predicate_id}: unsupported op {self.op!r} "
                f"(expected one of {sorted(SUPPORTED_OPS)})"
            )
        self.window = int(self.window)
        self.cooldown = int(self.cooldown)
        if self.window < 1:
            raise ReceiptWatchError(f"{self.predicate_id}: window must be >= 1")
        if self.cooldown < 0:
            raise ReceiptWatchError(f"{self.predicate_id}: cooldown must be >= 0")
        if self.op in _NUMERIC_OPS and _as_float(self.threshold) is None:
            raise ReceiptWatchError(
                f"{self.predicate_id}: op {self.op!r} needs a numeric threshold, "
                f"got {self.threshold!r}"
            )
        if self.op == "regex":
            try:
                re.compile(str(self.threshold))
            except re.error as exc:
                raise ReceiptWatchError(
                    f"{self.predicate_id}: invalid regex threshold: {exc}"
                ) from exc

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> WatchPredicateSpec:
        if not isinstance(payload, Mapping):
            raise ReceiptWatchError(f"predicate spec must be an object, got {type(payload)}")
        known = {name for name in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ReceiptWatchError(f"predicate spec has unknown keys: {sorted(unknown)}")
        missing = {"predicate_id", "field", "op", "threshold"} - set(payload)
        if payload.get("op") == "non_finite":
            missing.discard("threshold")  # threshold is ignored and may be absent/null
        if missing:
            raise ReceiptWatchError(f"predicate spec is missing keys: {sorted(missing)}")
        data = dict(payload)
        data.setdefault("threshold", None)
        return cls(**data)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "field": self.field,
            "op": self.op,
            "threshold": self.threshold,
            "window": self.window,
            "cooldown": self.cooldown,
            "fire_once": self.fire_once,
            "description": self.description,
        }

    @property
    def reason(self) -> str:
        return self.description or (
            f"{self.predicate_id}: {self.field} {self.op} {self.threshold}"
        )


def load_specs(path: str | Path) -> list[WatchPredicateSpec]:
    """Load a JSON list of predicate specs; raise on any malformed entry."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ReceiptWatchError(f"specs file must contain a JSON list: {path}")
    return [WatchPredicateSpec.from_json_dict(item) for item in payload]


@dataclass
class WatchFire:
    """One hash-chained cut-request record emitted when a predicate fires.

    ``segment_id`` is one uuid4 hex minted per :class:`ReceiptWatchService`
    instance; the ``prev_sha256`` chain links fires within the same segment
    only.
    """

    predicate_id: str
    record_index: int
    field: str
    observed: Any
    reference: float | str | None
    threshold: float | str | None
    record_sha256: str
    cut_request: dict[str, Any]
    segment_id: str
    prev_sha256: str | None
    sha256: str

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> WatchFire:
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "record_index": self.record_index,
            "field": self.field,
            "observed": self.observed,
            "reference": self.reference,
            "threshold": self.threshold,
            "record_sha256": self.record_sha256,
            "cut_request": dict(self.cut_request),
            "segment_id": self.segment_id,
            "prev_sha256": self.prev_sha256,
            "sha256": self.sha256,
        }


def _cut_request(spec: WatchPredicateSpec, record: Mapping[str, Any]) -> dict[str, Any]:
    step: Any = None
    for key in ("step", "cycle", "index"):
        if key in record:
            step = record[key]
            break
    return {
        "job_id": record.get("job_id"),
        "step": step,
        "reason": spec.reason,
        "requested": "statecut",
    }


@dataclass
class _PredicateState:
    history: deque[float]
    cooldown_remaining: int = 0
    fired: bool = False


class ReceiptWatchService:
    """Evaluate watch predicates over receipt records, one record at a time."""

    def __init__(self, specs: Sequence[WatchPredicateSpec]) -> None:
        self.specs = list(specs)
        seen: set[str] = set()
        for spec in self.specs:
            if spec.predicate_id in seen:
                raise ReceiptWatchError(f"duplicate predicate_id: {spec.predicate_id}")
            seen.add(spec.predicate_id)
        self._state = {
            spec.predicate_id: _PredicateState(history=deque(maxlen=spec.window + 1))
            for spec in self.specs
        }
        self._stats = {
            "records": 0,
            "fires": 0,
            "missing_field": 0,
            "suppressed_cooldown": 0,
            "suppressed_fire_once": 0,
        }
        self._segment_id = uuid.uuid4().hex
        self._prev_sha256: str | None = None

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _condition(
        self, spec: WatchPredicateSpec, observed: Any, state: _PredicateState
    ) -> tuple[bool, float | None, bool]:
        """Return (fired, reference, evaluable) for one present observation."""

        if spec.op == "regex":
            return bool(re.search(str(spec.threshold), str(observed))), None, True
        if spec.op in ("eq", "ne"):
            equal = _values_equal(observed, spec.threshold)
            return (equal if spec.op == "eq" else not equal), None, True
        value = _as_float(observed)
        if value is None:
            return False, None, False
        if spec.op == "non_finite":
            return not math.isfinite(value), None, True
        if not math.isfinite(value):
            # Non-finite observed under a numeric op: unevaluable, and it must
            # not enter the delta history (a later jump compares against the
            # last finite value).
            return False, None, False
        threshold = _as_float(spec.threshold)
        assert threshold is not None  # enforced by __post_init__
        if spec.op in _DELTA_OPS:
            state.history.append(value)
            if len(state.history) < spec.window + 1:
                return False, None, True
            reference = state.history[0]
            delta = abs(value - reference)
            if spec.op == "rel_delta_gt":
                if reference == 0.0:
                    delta = math.inf if delta > 0.0 else 0.0
                else:
                    delta = delta / abs(reference)
            return delta > threshold, reference, True
        comparisons = {
            "gt": value > threshold,
            "lt": value < threshold,
            "ge": value >= threshold,
            "le": value <= threshold,
        }
        return comparisons[spec.op], None, True

    def evaluate(self, record: dict, record_index: int | None = None) -> list[WatchFire]:
        """Evaluate every predicate against one record; return new fires."""

        index = self._stats["records"] if record_index is None else int(record_index)
        self._stats["records"] += 1
        fires: list[WatchFire] = []
        record_sha: str | None = None
        for spec in self.specs:
            state = self._state[spec.predicate_id]
            in_cooldown = state.cooldown_remaining > 0
            if in_cooldown:
                state.cooldown_remaining -= 1
            observed = lookup_dotted(record, spec.field)
            if observed is MISSING:
                self._stats["missing_field"] += 1
                continue
            fired, reference, evaluable = self._condition(spec, observed, state)
            if not evaluable:
                self._stats["missing_field"] += 1
                continue
            if not fired:
                continue
            if spec.fire_once and state.fired:
                self._stats["suppressed_fire_once"] += 1
                continue
            if in_cooldown:
                self._stats["suppressed_cooldown"] += 1
                continue
            state.fired = True
            state.cooldown_remaining = spec.cooldown
            if record_sha is None:
                record_sha = _sha256(record)
            body = {
                "predicate_id": spec.predicate_id,
                "record_index": index,
                "field": spec.field,
                "observed": _sanitize_non_finite(observed),
                "reference": _sanitize_non_finite(reference),
                "threshold": spec.threshold,
                "record_sha256": record_sha,
                "cut_request": _sanitize_non_finite(_cut_request(spec, record)),
                "segment_id": self._segment_id,
                "prev_sha256": self._prev_sha256,
            }
            sha = fire_sha256(body)
            self._prev_sha256 = sha
            self._stats["fires"] += 1
            fires.append(WatchFire(**body, sha256=sha))
        return fires

    def eval_stream(
        self, records: Iterable[dict]
    ) -> tuple[list[WatchFire], dict[str, int]]:
        """Evaluate an iterable of records; return (fires, cumulative stats)."""

        fires: list[WatchFire] = []
        for record in records:
            fires.extend(self.evaluate(record))
        return fires, self.stats


def _parse_line(raw: str) -> dict | None:
    """Parse one JSONL line; blank -> None, malformed -> ``{"_malformed": raw}``."""

    if not raw.strip():
        return None
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return {"_malformed": raw}
    if not isinstance(record, dict):
        return {"_malformed": raw}
    return record


def iter_jsonl_records(path: str | Path) -> Iterator[dict]:
    """One-shot read of a JSONL file using the shared line conventions.

    Bytes are decoded with ``errors="replace"`` and split on ``\\n`` only,
    exactly as :func:`tail_jsonl` does, so one-shot and follow mode behave
    identically on the same bytes and invalid UTF-8 cannot crash the reader.
    """

    text = Path(path).read_bytes().decode("utf-8", errors="replace")
    for raw in _split_lines(text):
        record = _parse_line(raw)
        if record is not None:
            yield record


def tail_jsonl(
    path: str | Path,
    *,
    poll_interval: float = 0.5,
    max_iterations: int | None = None,
    from_start: bool = True,
) -> Iterator[dict]:
    """Yield records appended to a JSONL file, polling for growth.

    Only complete lines (ending in a newline, or the final line of a chunk
    that itself ends in a newline) are consumed; a partial final line stays
    buffered in the file and is retried on the next poll.  When
    ``max_iterations`` is not ``None``, the generator stops after that many
    consecutive polls without consumable growth, so tests terminate.
    """

    path = Path(path)
    if from_start:
        offset = 0
    else:
        try:
            offset = path.stat().st_size
        except OSError:
            offset = 0
    idle_polls = 0
    while True:
        try:
            size = path.stat().st_size
        except OSError:
            size = offset
        consumed = b""
        if size > offset:
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
            if chunk.endswith(b"\n"):
                consumed = chunk
            else:
                newline = chunk.rfind(b"\n")
                consumed = chunk[: newline + 1] if newline >= 0 else b""
        if consumed:
            idle_polls = 0
            offset += len(consumed)
            for raw in _split_lines(consumed.decode("utf-8", errors="replace")):
                record = _parse_line(raw)
                if record is not None:
                    yield record
        else:
            idle_polls += 1
            if max_iterations is not None and idle_polls >= max_iterations:
                return
        time.sleep(poll_interval)


def write_fires(fires: Sequence[WatchFire | Mapping[str, Any]], path: str | Path) -> int:
    """Append fire records to a JSONL ledger; return the number written.

    Non-finite floats (``observed``/``reference`` included) are sanitized to
    the ``"NaN"``/``"Infinity"``/``"-Infinity"`` strings and every line is
    serialized with ``allow_nan=False``: ledger lines are strict JSON.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for fire in fires:
        payload = fire.to_json_dict() if isinstance(fire, WatchFire) else dict(fire)
        lines.append(json.dumps(_sanitize_non_finite(payload), sort_keys=True, allow_nan=False))
    if lines:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    return len(lines)


def verify_fire_ledger(path: str | Path) -> bool:
    """Re-hash a fire ledger; True iff every record and chain link verifies.

    Rows are grouped by ``segment_id`` (one uuid4 hex per service instance)
    and each segment's ``prev_sha256`` chain is verified independently: the
    first row of a segment has ``prev_sha256`` ``None``, and every later row
    of that segment must link the previous row's hash.  The ledger is
    append-only and multi-writer-tolerant across segments: verification
    detects in-place edits and reordering within a segment but NOT truncation
    of a segment tail or deletion of whole segments — an external head
    pointer is the caller's job.
    """

    prev_by_segment: dict[Any, str] = {}
    try:
        raw_bytes = Path(path).read_bytes()
    except OSError:
        return False
    for raw in _split_lines(raw_bytes.decode("utf-8", errors="replace")):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if not isinstance(record, dict) or not record.get("sha256"):
            return False
        if fire_sha256(record) != record["sha256"]:
            return False
        segment = record.get("segment_id")
        stored_prev = record.get("prev_sha256")
        if segment not in prev_by_segment:
            if stored_prev is not None:
                return False
        elif stored_prev != prev_by_segment[segment]:
            return False
        prev_by_segment[segment] = record["sha256"]
    return True


def _format_fire(fire: WatchFire) -> str:
    return (
        f"FIRE predicate={fire.predicate_id} index={fire.record_index} "
        f"field={fire.field} observed={fire.observed!r} reference={fire.reference!r} "
        f"threshold={fire.threshold!r} sha256={fire.sha256[:12]}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate watchpoint predicates over a JSONL receipt stream"
    )
    parser.add_argument("--specs", required=True, type=Path, help="JSON list of predicate specs")
    parser.add_argument("--stream", required=True, type=Path, help="JSONL receipt stream")
    parser.add_argument("--follow", action="store_true", help="poll the stream for appends")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--ledger", type=Path, default=None, help="append fires to this JSONL")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        specs = load_specs(args.specs)
        service = ReceiptWatchService(specs)
    except (OSError, ValueError) as exc:
        print(f"unreadable specs: {exc}", file=sys.stderr)
        return 2
    fires: list[WatchFire] = []
    try:
        if args.follow:
            source: Iterable[dict] = tail_jsonl(
                args.stream,
                poll_interval=args.poll_interval,
                max_iterations=args.max_iterations,
            )
            for record in source:
                for fire in service.evaluate(record):
                    fires.append(fire)
                    if args.ledger:
                        write_fires([fire], args.ledger)
                    if not args.as_json:
                        print(_format_fire(fire))
            stats = service.stats
        else:
            fires, stats = service.eval_stream(iter_jsonl_records(args.stream))
            if args.ledger:
                write_fires(fires, args.ledger)
            if not args.as_json:
                for fire in fires:
                    print(_format_fire(fire))
    except OSError as exc:
        # Fail-open on stream content: report and still exit 0.
        print(f"stream unreadable: {exc}", file=sys.stderr)
        stats = service.stats
    if args.as_json:
        print(
            json.dumps(
                {
                    "schema": RECEIPT_WATCH_SCHEMA,
                    "fires": [fire.to_json_dict() for fire in fires],
                    "stats": stats,
                },
                sort_keys=True,
            )
        )
    else:
        print(
            "stats "
            + " ".join(f"{key}={value}" for key, value in sorted(stats.items()))
        )
    return 0


__all__ = [
    "MISSING",
    "RECEIPT_WATCH_SCHEMA",
    "SUPPORTED_OPS",
    "ReceiptWatchError",
    "ReceiptWatchService",
    "WatchFire",
    "WatchPredicateSpec",
    "fire_sha256",
    "iter_jsonl_records",
    "load_specs",
    "lookup_dotted",
    "main",
    "tail_jsonl",
    "verify_fire_ledger",
    "write_fires",
]


if __name__ == "__main__":
    raise SystemExit(main())
