"""Source-qualified observations and explicitly proposed response families.

Only sealed metadata is imported. Effects, continuations and counterexamples
are caller-supplied measurements with provenance; nothing executes a model or
induces a rule. Exact applicability is separate from family comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from .equation_comparison import EquationCatalog, applicability

_SHA = re.compile(r"[0-9a-f]{64}")
OBSERVATION = "mdb-computation-atlas-observation-v1"
FAMILY = "mdb-computation-atlas-family-v1"
_BINDINGS = ("model_identity", "source_identity", "execution", "horizon")
_PREDICATE_NAMES = {"model_identity": ("model_identity", "checkpoint"),
                    "source_identity": ("source_identity",), "execution": ("execution",), "horizon": ("horizon",)}


class ComputationAtlasError(ValueError):
    """Malformed, unqualified, ambiguous or changed metadata."""


def _require(condition, message):
    if not condition:
        raise ComputationAtlasError(message)


def _json(value):
    def keys(item, depth=0):
        _require(depth <= 64, "atlas metadata nesting exceeds 64 levels")
        if isinstance(item, dict):
            _require(all(isinstance(key, str) and key for key in item), "metadata keys must be nonempty text")
            for child in item.values():
                keys(child, depth + 1)
        elif isinstance(item, (list, tuple)):
            for child in item:
                keys(child, depth + 1)
    keys(value)
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError, OverflowError) as error:
        raise ComputationAtlasError("atlas values must be finite JSON metadata") from error


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sealed(value, schema):
    value = _json(value)
    _require(isinstance(value, dict) and value.get("schema") == schema, "unsupported sealed metadata schema")
    identifier = value.get("fingerprint")
    _require(isinstance(identifier, str) and _SHA.fullmatch(identifier)
             and _digest({key: item for key, item in value.items() if key != "fingerprint"}) == identifier,
             "sealed metadata fingerprint differs")
    return value


def _references(values, *, required=True):
    _require(isinstance(values, list) and (values or not required), "explicit artifact provenance is required")
    for value in values:
        _require(isinstance(value, dict) and isinstance(value.get("path"), str) and value["path"].strip()
                 and isinstance(value.get("sha256"), str) and _SHA.fullmatch(value["sha256"]),
                 "artifact provenance needs a path and SHA256")
    return values


def _qualified(record):
    record = _sealed(record, "mdb-equation-record-v1")
    _sealed(record["report"], "mdb-equation-comparison-v1")
    binding, conditions = record["binding"], record["applicability_conditions"]
    for name in _BINDINGS:
        _require(name in binding and binding[name] not in (None, "", {}), "missing exact " + name + " binding")
        representations = [binding[name]]
        if name == "source_identity":
            representations.extend((_digest(binding[name]), hashlib.sha256(
                json.dumps(binding[name], sort_keys=True, allow_nan=False).encode()).hexdigest()))
        _require(any(condition.get("field") in _PREDICATE_NAMES[name] and condition.get("op") == "eq"
                     and condition.get("value") in representations for condition in conditions),
                 "applicability requires exact " + name)
    sources = binding["source_identity"]
    _require((isinstance(sources, str) and _SHA.fullmatch(sources)) or (isinstance(sources, dict) and sources
             and all(isinstance(key, str) and key and isinstance(value, str) and _SHA.fullmatch(value)
                     for key, value in sources.items())), "source_identity must be an explicit SHA256 identity or source map")
    _require(isinstance(binding["execution"], dict) and binding["execution"].get("backend"), "execution backend is required")
    _require(record.get("law_id") and record.get("route_id") and record.get("readers"),
             "generic law, route and explicit consumer IDs are required")
    _require(applicability(conditions, record["tested_context"])["status"] == "matches",
             "tested context must satisfy all supplied applicability predicates")
    return record


def _effects(values, consumers):
    _require(isinstance(values, list), "signed_effects must be a list")
    identities = set()
    for value in values:
        _require(isinstance(value, dict) and value.get("consumer_id") in consumers, "effect consumer is not declared")
        for name in ("contrast_id", "units"):
            _require(isinstance(value.get(name), str) and value[name].strip(), "effect needs " + name)
        number = value.get("value")
        try:
            finite = type(number) in (int, float) and math.isfinite(number)
        except OverflowError:
            finite = False
        _require(finite, "signed effect must be finite and numeric")
        _references(value.get("provenance"))
        key = value["consumer_id"], value["contrast_id"]
        _require(key not in identities, "ambiguous duplicate consumer/contrast effect")
        identities.add(key)
    return values


class ComputationAtlas:
    """An append-only metadata atlas backed by public Saturn Receipt CAS.

    ``ingest`` accepts only explicitly selected EquationCatalog IDs. Supplements
    are mappings keyed by those IDs. Returned ``fingerprint`` values address
    independent immutable atlas receipts. External artifact references are
    retained declarations; they are never fetched or executed.
    """

    def __init__(self, root):
        self.root = Path(root)
        _require(not self.root.is_symlink(), "atlas root must not be a symbolic link")

    def _save(self, **body):
        from saturn_pub import Receipt
        from saturn_pub.store import LocalStore

        receipt = Receipt.make(**_json(body))
        LocalStore(self.root).receipt(receipt)
        return receipt.to_dict()

    def ingest(self, catalog_root, record_ids, *, provenance, signed_effects=None,
               continuation=None, counterexamples=None):
        catalog_root = Path(catalog_root)
        _require(catalog_root.is_dir() and not catalog_root.is_symlink(), "explicit source catalog is required")
        ids = list(record_ids)
        _require(0 < len(ids) <= 256 and all(isinstance(value, str) and _SHA.fullmatch(value) for value in ids)
                 and len(ids) == len(set(ids)), "select 1..256 unique equation SHA256 IDs")
        provenance = _json(provenance)
        _require(isinstance(provenance, dict) and isinstance(provenance.get("run_id"), str)
                 and provenance["run_id"].strip(), "provenance needs an explicit run_id")
        _references(provenance.get("artifacts"))
        supplements = [_json(value or {}) for value in (signed_effects, continuation, counterexamples)]
        _require(all(isinstance(value, dict) and not set(value) - set(ids) for value in supplements),
                 "supplements must refer only to selected equation records")
        catalog, pending = EquationCatalog(catalog_root), []
        for identifier in ids:
            path = catalog_root / (identifier + ".json")
            _require(not path.is_symlink(), "source equation record must not be a symbolic link")
            _require(path.stat().st_size <= 8 * 1024 * 1024, "source equation record exceeds 8MiB")
            record = _qualified(catalog.show(identifier))
            effects = _effects(supplements[0].get(identifier, []), record["readers"])
            future = supplements[1].get(identifier, {"status": "not-measured", "provenance": []})
            _require(isinstance(future, dict) and isinstance(future.get("status"), str) and future["status"],
                     "continuation needs an explicit status")
            _references(future.get("provenance"), required=future["status"] != "not-measured")
            exceptions = supplements[2].get(identifier, [])
            _require(isinstance(exceptions, list), "counterexamples must remain a list")
            for item in exceptions:
                _require(isinstance(item, dict) and item.get("claim_id") and item.get("status"),
                         "counterexample needs claim_id and status")
                _references(item.get("provenance"))
            pending.append({"operation": "mdb.atlas.import", "status": "observed", "atlas_schema": OBSERVATION,
                "equation": record, "source_record": {"fingerprint": identifier,
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                "law_id": record["law_id"], "route_id": record["route_id"], "consumer_ids": record["readers"],
                "provenance": provenance, "signed_effects": effects, "continuation": future,
                "counterexamples": exceptions, "rule_status": "observation; no validated rule",
                "reference_scope": "Equation snapshot/seals verified; external artifact references are supplied, not fetched."})
        return [self._save(**value) for value in pending]

    def show(self, identifier):
        _require(isinstance(identifier, str) and _SHA.fullmatch(identifier), "atlas ID must be a SHA256")
        path = self.root / "receipts" / (identifier + ".json")
        _require(not path.is_symlink(), "atlas receipt must not be a symbolic link")
        value = _sealed(json.loads(path.read_text()), "saturn-pub-receipt-v1")
        _require(value["fingerprint"] == identifier and value.get("atlas_schema") in (OBSERVATION, FAMILY), "atlas receipt identity differs")
        if value["atlas_schema"] == OBSERVATION:
            _qualified(value["equation"])
            _require(value["source_record"]["fingerprint"] == value["equation"]["fingerprint"], "source equation identity differs")
            _require(value["law_id"] == value["equation"]["law_id"] and value["route_id"] == value["equation"]["route_id"]
                     and value["consumer_ids"] == value["equation"]["readers"], "atlas generic IDs differ from source record")
            _effects(value["signed_effects"], value["consumer_ids"])
            _require(value["rule_status"] == "observation; no validated rule", "observation cannot promote itself to a rule")
        else:
            _require(value["rule_status"] == "not-validated" and value["hypothesis_status"] in
                     ("family-candidate", "observation", "evidence-incomplete"), "family receipt cannot assert rule validation")
        return value

    def find(self, *, law_id=None, route_id=None, consumer_id=None, model_identity=None, applicable_to=None):
        results = []
        for path in sorted((self.root / "receipts").glob("*.json")):
            record = self.show(path.stem)
            if record["atlas_schema"] != OBSERVATION:
                continue
            if law_id is not None and record["law_id"] != law_id or route_id is not None and record["route_id"] != route_id:
                continue
            if consumer_id is not None and consumer_id not in record["consumer_ids"]:
                continue
            if model_identity is not None and record["equation"]["binding"]["model_identity"] != model_identity:
                continue
            if applicable_to is not None:
                result = applicability(record["equation"]["applicability_conditions"], applicable_to)
                if result["status"] != "matches":
                    continue
                record["applicability_query"] = result
            results.append(record)
        return results

    def compare(self, record_ids, *, consumer_id, contrast_id):
        ids = list(record_ids)
        _require(0 < len(ids) <= 256, "compare 1..256 explicitly selected observations")
        _require(isinstance(consumer_id, str) and consumer_id.strip() and isinstance(contrast_id, str) and contrast_id.strip(),
                 "explicit consumer and contrast IDs are required")
        records = [self.show(identifier) for identifier in ids]
        _require(records and all(value["atlas_schema"] == OBSERVATION for value in records), "select observations for comparison")
        _require(len({value["source_record"]["fingerprint"] for value in records}) == len(records),
                 "copied versions of one equation observation cannot count twice")
        _require(len({(value["law_id"], value["route_id"]) for value in records}) == 1,
                 "family comparison requires the same generic law and route")
        _require(all(consumer_id in value["consumer_ids"] for value in records), "consumer is not declared in every record")
        rows, cohorts, units = [], {}, set()
        for record in records:
            binding = {name: record["equation"]["binding"][name] for name in _BINDINGS}
            cohort = _digest(binding)
            cohorts[cohort] = binding
            effects = [value for value in record["signed_effects"] if value["consumer_id"] == consumer_id and value["contrast_id"] == contrast_id]
            effect = effects[0] if effects else None
            if effect is not None:
                units.add(effect["units"])
            rows.append({"atlas_id": record["fingerprint"], "equation_id": record["source_record"]["fingerprint"],
                "run_id": record["provenance"]["run_id"], "cohort_id": cohort,
                "tested_context": record["equation"]["tested_context"], "effect": effect,
                "measurement_status": "observed" if effect is not None else "not-measured",
                "continuation": record["continuation"], "counterexamples": record["counterexamples"]})
        _require(len(units) <= 1, "effect units differ; no numerical pooling is defined")
        by_cohort = defaultdict(list)
        for row in rows:
            by_cohort[row["cohort_id"]].append(row)
        summaries = {}
        for cohort, members in by_cohort.items():
            values = [row["effect"]["value"] for row in members if row["effect"] is not None]
            summaries[cohort] = {"binding": cohorts[cohort], "rows": len(members), "measured": len(values),
                "positive": sum(value > 0 for value in values), "negative": sum(value < 0 for value in values),
                "zero": sum(value == 0 for value in values), "signed_values": values,
                "minimum": min(values) if values else None, "maximum": max(values) if values else None,
                "mean": sum(values) / len(values) if values else None}
        measured = sum(row["effect"] is not None for row in rows)
        return self._save(operation="mdb.atlas.compare", status="observed", atlas_schema=FAMILY,
            law_id=records[0]["law_id"], route_id=records[0]["route_id"], consumer_id=consumer_id, contrast_id=contrast_id,
            hypothesis_status="family-candidate" if measured >= 2 else "observation" if measured else "evidence-incomplete",
            rule_status="not-validated", measured_rows=measured, selected_rows=len(rows),
            distinct_run_labels=len({row["run_id"] for row in rows}), units=next(iter(units), None),
            cohorts=summaries, rows=rows, cross_cohort_pooling="not performed; source/backend/horizon identities remain separate",
            independence="not established; distinct records/run labels may reuse underlying observations",
            authority="Caller-selected response comparison; no automatic grouping, rule induction, applicability expansion or causal certification.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    subs = parser.add_subparsers(dest="command", required=True)
    ingest = subs.add_parser("ingest")
    ingest.add_argument("--catalog", type=Path, required=True)
    ingest.add_argument("--metadata", type=Path, required=True)
    ingest.add_argument("identifiers", nargs="+")
    show = subs.add_parser("show")
    show.add_argument("identifier")
    find = subs.add_parser("find")
    for option in ("law-id", "route-id", "consumer-id", "model-identity"):
        find.add_argument("--" + option)
    find.add_argument("--applicable-to", type=Path)
    compare = subs.add_parser("compare")
    compare.add_argument("--consumer-id", required=True)
    compare.add_argument("--contrast-id", required=True)
    compare.add_argument("identifiers", nargs="+")
    args = parser.parse_args(argv)
    atlas = ComputationAtlas(args.atlas)
    if args.command == "ingest":
        result = atlas.ingest(args.catalog, args.identifiers, **json.loads(args.metadata.read_text()))
    elif args.command == "show":
        result = atlas.show(args.identifier)
    elif args.command == "find":
        result = atlas.find(law_id=args.law_id, route_id=args.route_id, consumer_id=args.consumer_id,
            model_identity=args.model_identity, applicable_to=json.loads(args.applicable_to.read_text()) if args.applicable_to else None)
    else:
        result = atlas.compare(args.identifiers, consumer_id=args.consumer_id, contrast_id=args.contrast_id)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = ["ComputationAtlas", "ComputationAtlasError"]

if __name__ == "__main__":
    raise SystemExit(main())
