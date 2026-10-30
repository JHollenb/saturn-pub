"""Measured equation residuals and immutable, searchable computation records.

Importing the catalogue needs no model framework. Numerical comparisons import
NumPy lazily. Applicability predicates are supplied conditions, never discovered
semantic rules; no numerical tolerance assigns a scientific pass/fail status.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .circuit_explorer import _copy_json, _seal, _sealed

REPORT_SCHEMA = "mdb-equation-comparison-v1"
RECORD_SCHEMA = "mdb-equation-record-v1"


def _array(value, label):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(label + " requires finite real numerical arrays")
    return value.astype(np.float64, copy=True)


def _summary(value, *, include_residuals=False):
    import numpy as np

    flat = value.reshape(-1)
    index = int(np.argmax(np.abs(flat))) if flat.size else None
    result = {"shape": list(value.shape), "elements": int(value.size),
              "nonzero": int(np.count_nonzero(value)),
              "max_abs": float(np.max(np.abs(flat))) if flat.size else 0.,
              "l2": float(np.linalg.norm(flat)),
              "signed_sum": float(flat.sum()),
              "signed_mean": float(flat.mean()) if flat.size else 0.,
              "minimum_signed": float(flat.min()) if flat.size else 0.,
              "maximum_signed": float(flat.max()) if flat.size else 0.,
              "largest": None if index is None else {
                  "coordinate": [int(x) for x in np.unravel_index(index, value.shape)],
                  "signed": float(flat[index])}}
    if include_residuals:
        result["signed_values"] = value.tolist()
    return result


def _decisions(predicted, observed, rule):
    import numpy as np

    if rule == "none":
        return {"rule": rule, "status": "not-requested"}
    if rule == "positive":
        left, right = predicted > 0, observed > 0
    elif rule == "argmax":
        if predicted.ndim == 0 or predicted.shape[-1] == 0:
            raise ValueError("argmax requires a nonempty final axis")
        left, right = predicted.argmax(axis=-1), observed.argmax(axis=-1)
    else:
        raise ValueError("decision rule must be positive, argmax or none")
    different = np.asarray(left != right)
    return {"rule": rule, "decisions": int(different.size),
            "different": int(np.count_nonzero(different)),
            "different_coordinates": np.argwhere(different).tolist(),
            "predicted": np.asarray(left).astype(int).tolist(),
            "observed": np.asarray(right).astype(int).tolist()}


def _effects(value, baseline, rule, mask, include_residuals):
    import numpy as np

    delta = value - baseline
    result = {"signed_delta": _summary(delta, include_residuals=include_residuals),
              "decision_changes": _decisions(baseline, value, rule)}
    if mask is not None:
        result["target"] = _summary(delta[mask], include_residuals=include_residuals)
        result["collateral"] = _summary(delta[~mask], include_residuals=include_residuals)
        if rule == "positive":
            changed = (value > 0) != (baseline > 0)
            result["target_decision_changes"] = int(np.count_nonzero(changed & mask))
            result["collateral_decision_changes"] = int(np.count_nonzero(changed & ~mask))
    return result


def compare_equations(predicted, observed, *, decision_rules=None, baseline=None,
                      predicted_baseline=None, targets=None, components=None,
                      observed_components=None, include_residuals=False,
                      max_elements=1_000_000):
    """Compare matching named arrays; signs are always observed minus predicted.

    ``baseline`` is the observed comparison branch. ``predicted_baseline`` is
    separate and never substituted for it. ``targets`` maps a head to explicit
    full tensor coordinates; all other coordinates are collateral. Omitting
    targets leaves that classification unassigned. Components map head -> named
    additive terms and measure closure independently of forecast accuracy.
    """
    import numpy as np

    if (not isinstance(predicted, Mapping) or not isinstance(observed, Mapping)
            or not predicted or set(predicted) != set(observed)):
        raise ValueError("predicted and observed must have the same nonempty field set")
    rules = dict(decision_rules or {})
    if set(rules) - set(predicted) or (targets is not None and set(targets) - set(predicted)):
        raise ValueError("unknown decision or target field")
    for supplied in [baseline, predicted_baseline, components, observed_components]:
        if supplied is not None and set(supplied) - set(predicted):
            raise ValueError("unknown baseline or component field")
    if type(max_elements) is not int or max_elements < 1:
        raise ValueError("max_elements must be a positive integer")
    fields, total = {}, 0
    for name in sorted(predicted):
        left, right = _array(predicted[name], name), _array(observed[name], name)
        if left.shape != right.shape:
            raise ValueError("forecast/observation shape mismatch: " + name)
        total += left.size
        if total > max_elements:
            raise ValueError("comparison exceeds declared element budget")
        rule = rules.get(name, "none")
        mask = None
        if targets is not None:
            mask = np.zeros(left.shape, dtype=bool)
            for coordinate in targets.get(name, []):
                if (not isinstance(coordinate, (list, tuple)) or len(coordinate) != left.ndim
                        or any(type(i) is not int or i < 0 or i >= size
                               for i, size in zip(coordinate, left.shape, strict=True))):
                    raise ValueError("invalid target coordinate: " + name)
                mask[tuple(coordinate)] = True
        field = {"shape": list(left.shape),
                 "residual": _summary(right - left, include_residuals=include_residuals),
                 "decisions": _decisions(left, right, rule)}
        if mask is not None:
            field["target_coordinates"] = np.argwhere(mask).tolist()
        for key, supplied, value in [("observed_effect", baseline, right),
                                     ("predicted_effect", predicted_baseline, left)]:
            if supplied is not None and name in supplied:
                parent = _array(supplied[name], name + " baseline")
                if parent.shape != value.shape:
                    raise ValueError("baseline shape mismatch: " + name)
                field[key] = _effects(value, parent, rule, mask, include_residuals)
        for key, supplied, value in [("prediction_component_closure", components, left),
                                     ("observation_component_closure", observed_components, right)]:
            if supplied is not None and name in supplied:
                terms = supplied[name]
                if not isinstance(terms, Mapping) or not terms:
                    raise ValueError("component closure needs named terms")
                summed = np.zeros_like(value)
                for label, term in terms.items():
                    term = _array(term, name + "." + label)
                    if term.shape != value.shape:
                        raise ValueError("component shape mismatch: " + name)
                    summed += term
                field[key] = {"terms": sorted(terms),
                              "residual": _summary(value - summed, include_residuals=include_residuals)}
        fields[name] = field
    return _seal({"fields": fields, "residual_direction": "observed-minus-predicted",
                  "target_scope": "unassigned" if targets is None else "explicit-tensor-coordinates",
                  "mechanics_status": "valid", "terminal_status": "unassessed",
                  "max_abs": max(row["residual"]["max_abs"] for row in fields.values()),
                  "decision_mismatches": sum(row["decisions"].get("different", 0) for row in fields.values()),
                  "decision_fields": sum(row["decisions"]["rule"] != "none" for row in fields.values())},
                 REPORT_SCHEMA, "equation comparison")


def _conditions(conditions):
    result = _copy_json(conditions, "applicability conditions")
    if not isinstance(result, list):
        raise ValueError("conditions must be a list")
    for item in result:
        if not isinstance(item, dict) or not isinstance(item.get("field"), str) or not item["field"]:
            raise ValueError("condition needs a context field")
        op = item.get("op")
        expected = {"eq": {"field", "op", "value"}, "in": {"field", "op", "values"},
                    "range": {"field", "op", "minimum", "maximum"}}.get(op)
        if expected is None or set(item) != expected:
            raise ValueError("conditions support exact eq, in and inclusive range predicates")
        if op == "in" and (not isinstance(item["values"], list) or not item["values"]):
            raise ValueError("membership requires nonempty values")
        if op == "range" and (any(type(item[key]) not in (int, float) for key in ["minimum", "maximum"])
                              or item["minimum"] > item["maximum"]):
            raise ValueError("invalid applicability range")
    return result


def applicability(conditions, context):
    """Evaluate supplied selectors; matching is not empirical validation."""
    conditions = _conditions(conditions)
    context = _copy_json(context)
    if not isinstance(context, dict):
        raise ValueError("applicability context must be an object")
    results = []
    for item in conditions:
        value = context
        for part in item["field"].split("."):
            if not isinstance(value, dict) or part not in value:
                results.append({"condition": item, "status": "unknown"})
                break
            value = value[part]
        else:
            op = item["op"]
            match = value == item["value"] if op == "eq" else value in item["values"] if op == "in" else (
                type(value) in (int, float) and item["minimum"] <= value <= item["maximum"])
            results.append({"condition": item, "status": "matches" if match else "does-not-match"})
    status = ("unspecified" if not results else "does-not-match" if any(r["status"] == "does-not-match" for r in results)
              else "unknown" if any(r["status"] == "unknown" for r in results) else "matches")
    return {"status": status, "conditions": results, "authority": "supplied-predicate-only; tested scope is separate"}


class EquationCatalog:
    """Immutable JSON evidence records; source tensors remain in their custody store."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record(self, *, law_id, binding, report, conditions=(), tested_context=None,
               route_id=None, alias=None, readers=(), evidence=None):
        if not isinstance(law_id, str) or not law_id.strip():
            raise ValueError("generic law_id must be nonempty")
        for name, value in [("route_id", route_id), ("alias", alias)]:
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(name + " must be nonempty text")
        binding = _copy_json(binding, "equation binding")
        required = {"model_identity", "execution", "input_identity", "frontier", "horizon", "source_identity"}
        if not isinstance(binding, dict) or required - set(binding):
            raise ValueError("equation binding needs model/execution/input/frontier/horizon/source identities")
        if any(value is None or value == "" or value == {} for value in binding.values()):
            raise ValueError("equation binding contains an empty identity")
        report = _sealed(report, REPORT_SCHEMA, "equation comparison")
        if not isinstance(readers, (list, tuple)) or any(not isinstance(r, str) or not r for r in readers):
            raise ValueError("readers must be explicit text selectors")
        record = _seal({"law_id": law_id, "route_id": route_id, "alias": alias,
                        "binding": binding, "report": report, "readers": list(readers),
                        "applicability_conditions": _conditions(list(conditions)),
                        "tested_context": _copy_json(tested_context or {}),
                        "evidence": _copy_json(evidence or {}),
                        "applicability_authority": "supplied; no generalization inferred",
                        "observation_unit": "one caller-bound comparison, not independent tasks"},
                       RECORD_SCHEMA, "equation record")
        path = self.root / (record["fingerprint"] + ".json")
        raw = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
        handle, temporary = tempfile.mkstemp(prefix=".equation-", dir=self.root)
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_text() != raw:
                    raise ValueError("immutable equation record bytes changed")
            descriptor = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            os.unlink(temporary)
        return record

    def show(self, identifier):
        if not isinstance(identifier, str) or len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("equation record ID must be a SHA256 fingerprint")
        record = _sealed(json.loads((self.root / (identifier + ".json")).read_text()), RECORD_SCHEMA, "equation record")
        if record["fingerprint"] != identifier:
            raise ValueError("equation record filename identity differs")
        _sealed(record["report"], REPORT_SCHEMA, "equation comparison")
        _conditions(record["applicability_conditions"])
        return record

    def find(self, *, law_id=None, route_id=None, model_identity=None, reader=None,
             alias=None, text=None, applicable_to=None, tested_context=None):
        matches = []
        for path in sorted(self.root.glob("*.json")):
            record = self.show(path.stem)
            if any(value is not None and record[key] != value for key, value in
                   [("law_id", law_id), ("route_id", route_id), ("alias", alias)]):
                continue
            if model_identity is not None and record["binding"]["model_identity"] != model_identity:
                continue
            if reader is not None and reader not in record["readers"]:
                continue
            if text is not None and text.casefold() not in json.dumps(record, sort_keys=True).casefold():
                continue
            if tested_context is not None and any(record["tested_context"].get(k) != v for k, v in tested_context.items()):
                continue
            if applicable_to is not None:
                result = applicability(record["applicability_conditions"], applicable_to)
                if result["status"] != "matches":
                    continue
                record = {**record, "applicability_query": result}
            matches.append(record)
        return matches


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mdb equations", description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    compare = subs.add_parser("compare")
    compare.add_argument("predicted", type=Path)
    compare.add_argument("observed", type=Path)
    compare.add_argument("--rules", type=Path)
    compare.add_argument("--baseline", type=Path)
    compare.add_argument("--targets", type=Path)
    compare.add_argument("--residuals", action="store_true")
    for command in ["list", "find", "show"]:
        sub = subs.add_parser(command)
        sub.add_argument("--catalog", type=Path, required=True)
        if command == "show":
            sub.add_argument("identifier")
        if command == "find":
            for name in ["law-id", "route-id", "model-identity", "reader", "alias", "text"]:
                sub.add_argument("--" + name)
            sub.add_argument("--applicable-to", type=Path)
    args = parser.parse_args(argv)
    read = lambda path: json.loads(path.read_text()) if path else None
    if args.command == "compare":
        result = compare_equations(read(args.predicted), read(args.observed), decision_rules=read(args.rules),
                                   baseline=read(args.baseline), targets=read(args.targets), include_residuals=args.residuals)
    else:
        catalog = EquationCatalog(args.catalog)
        if args.command == "show":
            result = catalog.show(args.identifier)
        else:
            query = {key: getattr(args, key) for key in ["law_id", "route_id", "model_identity", "reader", "alias", "text"]} if args.command == "find" else {}
            if args.command == "find":
                query["applicable_to"] = read(args.applicable_to)
            result = catalog.find(**query)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = ["EquationCatalog", "applicability", "compare_equations"]
