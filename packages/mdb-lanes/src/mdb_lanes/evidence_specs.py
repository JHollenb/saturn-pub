"""Standard evidence watch-spec pack, adapted to each stream's numeric fields.

:mod:`saturn.receipt_watch` evaluates declarative predicates but ships none.
This module is the shipped default pack: :data:`STANDARD_SPEC_PACK` is the
rule data (operator, threshold, field-selection recipe per rule) and
:func:`build_specs` / :func:`specs_for_stream` adapt those rules to the
numeric fields actually present in a stream, so ``saturn-evidence run`` can
watch any results stream without a hand-written spec file.

Field detection samples the first ``sample_limit`` records of a JSONL stream
and walks each record flat plus exactly one nesting level (``metrics.rms``);
a field counts as numeric when any sampled record holds an ``int``/``float``
(bools excluded, non-finite floats included — a field already ``NaN`` must
still receive its ``non_finite`` predicate).  Keys containing ``.`` and
empty/whitespace-only keys are skipped: they cannot be addressed by
:func:`receipt_watch.lookup_dotted` (an empty key would otherwise build a
spec that ``WatchPredicateSpec`` rejects, aborting the whole stream).

Every built spec carries provenance: its ``description`` starts with the pack
version tag :data:`EVIDENCE_PACK_SCHEMA`, which flows into each fire record's
``cut_request.reason``.  Honest claim boundary: these are mechanics-only
default thresholds; a fire marks a record worth a look, not a scientific
finding.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

EVIDENCE_PACK_SCHEMA = "saturn-evidence-pack-v1"
DEFAULT_SAMPLE_RECORDS = 64


from . import receipt_watch


class EvidenceSpecError(ValueError):
    """Raised when a pack rule or stream sample cannot be interpreted."""


STANDARD_SPEC_PACK: tuple[dict[str, Any], ...] = (
    {
        "rule": "non_finite",
        "select": {"kind": "all_numeric"},
        "op": "non_finite",
        "threshold": None,
        "window": 1,
        "cooldown": 0,
    },
    {
        "rule": "rms_extreme",
        "select": {"kind": "exact", "field": "metrics.rms"},
        "op": "gt",
        "threshold": 100.0,
        "window": 1,
        "cooldown": 0,
    },
    {
        "rule": "margin_tie",
        "select": {"kind": "exact", "field": "metrics.margin"},
        "op": "le",
        "threshold": 0.0,
        "window": 1,
        "cooldown": 0,
    },
    {
        "rule": "ans_collapse",
        "select": {"kind": "leaf_exact", "leaf": "answer_logit_delta_vs_native"},
        "op": "lt",
        "threshold": -20.0,
        "window": 1,
        "cooldown": 0,
    },
    {
        "rule": "ent_spike",
        "select": {"kind": "leaf_contains", "needle": "entropy"},
        "op": "rel_delta_gt",
        "threshold": 3.0,
        "window": 1,
        "cooldown": 8,
    },
    {
        "rule": "throughput_cliff",
        "select": {"kind": "leaf_suffix", "suffix": "tokens_per_second"},
        "op": "rel_delta_gt",
        "threshold": 3.0,
        "window": 1,
        "cooldown": 8,
    },
)

_SELECT_KEYS = {
    "all_numeric": (),
    "exact": ("field",),
    "leaf_exact": ("leaf",),
    "leaf_contains": ("needle",),
    "leaf_suffix": ("suffix",),
}
_RULE_REQUIRED = ("rule", "select", "op", "threshold", "window", "cooldown")


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def detect_numeric_fields(
    records: Iterable[Mapping[str, Any]],
    sample_limit: int = DEFAULT_SAMPLE_RECORDS,
) -> list[str]:
    """Sample records and return sorted dotted paths of numeric fields.

    Walks flat keys plus exactly one nesting level.  A field qualifies when
    any sampled record holds a non-bool ``int``/``float`` there (non-finite
    floats included).  Keys containing ``.`` and empty/whitespace-only keys
    are skipped (unaddressable by dotted lookup, and an empty field name
    would make ``WatchPredicateSpec`` reject the whole adapted pack).  Only
    the first ``sample_limit`` records are examined.
    """

    if int(sample_limit) < 1:
        raise EvidenceSpecError(f"sample_limit must be >= 1, got {sample_limit!r}")
    fields: set[str] = set()
    for index, record in enumerate(records):
        if index >= int(sample_limit):
            break
        if not isinstance(record, Mapping):
            continue
        for key, value in record.items():
            if not isinstance(key, str) or not key.strip() or "." in key:
                continue
            if _is_numeric(value):
                fields.add(key)
            elif isinstance(value, Mapping):
                for inner_key, inner_value in value.items():
                    if not isinstance(inner_key, str) or not inner_key.strip():
                        continue
                    if "." in inner_key:
                        continue
                    if _is_numeric(inner_value):
                        fields.add(f"{key}.{inner_key}")
    return sorted(fields)


def _validate_rule(rule: Mapping[str, Any]) -> None:
    missing = [key for key in _RULE_REQUIRED if key not in rule]
    if missing:
        raise EvidenceSpecError(f"pack rule is missing keys {missing}: {rule!r}")
    select = rule["select"]
    if not isinstance(select, Mapping) or "kind" not in select:
        raise EvidenceSpecError(f"pack rule select must be an object with a kind: {select!r}")
    kind = select["kind"]
    if kind not in _SELECT_KEYS:
        raise EvidenceSpecError(
            f"unknown select kind {kind!r} (expected one of {sorted(_SELECT_KEYS)})"
        )
    for key in _SELECT_KEYS[kind]:
        if not isinstance(select.get(key), str) or not select[key]:
            raise EvidenceSpecError(f"select kind {kind!r} needs a non-empty {key!r}: {select!r}")


def _matches(select: Mapping[str, Any], field: str) -> bool:
    leaf = field.rsplit(".", 1)[-1]
    kind = select["kind"]
    if kind == "all_numeric":
        return True
    if kind == "exact":
        return field == select["field"]
    if kind == "leaf_exact":
        return leaf == select["leaf"]
    if kind == "leaf_contains":
        return select["needle"] in leaf
    if kind == "leaf_suffix":
        return leaf.endswith(select["suffix"])
    raise EvidenceSpecError(f"unknown select kind: {kind!r}")  # pragma: no cover


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def build_specs(
    fields: Sequence[str],
    pack: Sequence[Mapping[str, Any]] = STANDARD_SPEC_PACK,
) -> list[Any]:
    """Adapt pack rules to detected numeric fields; return WatchPredicateSpecs.

    Deterministic: rules in pack order, fields in sorted order within a rule.
    Predicate ids are ``<rule>-<field slug>`` with a numeric suffix on slug
    collisions, so the built list is always constructible by
    :class:`receipt_watch.ReceiptWatchService`.
    """

    ordered_fields = sorted(str(field) for field in fields)
    used_ids: set[str] = set()
    specs: list[Any] = []
    for rule in pack:
        _validate_rule(rule)
        for field in ordered_fields:
            if not _matches(rule["select"], field):
                continue
            base = f"{_slug(rule['rule'])}-{_slug(field)}".strip("-")
            predicate_id = base or _slug(rule["rule"])
            suffix = 2
            while predicate_id in used_ids:
                predicate_id = f"{base}-{suffix}"
                suffix += 1
            used_ids.add(predicate_id)
            description = (
                f"[{EVIDENCE_PACK_SCHEMA}] {rule['rule']}: {field} "
                f"{rule['op']} {rule['threshold']}"
            )
            specs.append(
                receipt_watch.WatchPredicateSpec(
                    predicate_id=predicate_id,
                    field=field,
                    op=str(rule["op"]),
                    threshold=rule["threshold"],
                    window=int(rule["window"]),
                    cooldown=int(rule["cooldown"]),
                    description=description,
                )
            )
    return specs


def _sample_stream(path: Path, sample_limit: int) -> Iterator[dict[str, Any]]:
    """Lazily parse up to ``sample_limit`` records from a JSONL stream.

    Blank and malformed lines are skipped (they carry no fields); reading
    stops after ``sample_limit`` parsed records so detection never loads a
    large stream into memory.
    """

    parsed = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            if parsed >= sample_limit:
                break
            text = raw_line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                parsed += 1
                yield record


def specs_for_stream(
    path: str | Path,
    *,
    sample_limit: int = DEFAULT_SAMPLE_RECORDS,
    pack: Sequence[Mapping[str, Any]] = STANDARD_SPEC_PACK,
) -> list[Any]:
    """Detect a stream's numeric fields and build the adapted standard pack."""

    stream = Path(path)
    fields = detect_numeric_fields(_sample_stream(stream, int(sample_limit)), sample_limit)
    return build_specs(fields, pack)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the standard evidence watch-spec pack for one JSONL stream. "
            "Exit codes: 0 specs built (possibly zero), 2 unreadable stream."
        )
    )
    parser.add_argument("stream", type=Path, help="JSONL receipt stream to adapt the pack to")
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_RECORDS)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        specs = specs_for_stream(args.stream, sample_limit=args.sample_limit)
    except (OSError, ValueError) as exc:
        print(f"unreadable stream: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(
            json.dumps(
                {
                    "pack": EVIDENCE_PACK_SCHEMA,
                    "stream": str(args.stream),
                    "specs": [spec.to_json_dict() for spec in specs],
                },
                sort_keys=True,
            )
        )
    else:
        for spec in specs:
            print(f"{spec.predicate_id}\t{spec.field}\t{spec.op}\t{spec.threshold}")
    return 0


__all__ = [
    "DEFAULT_SAMPLE_RECORDS",
    "EVIDENCE_PACK_SCHEMA",
    "STANDARD_SPEC_PACK",
    "EvidenceSpecError",
    "build_specs",
    "detect_numeric_fields",
    "main",
    "specs_for_stream",
]


if __name__ == "__main__":
    raise SystemExit(main())
