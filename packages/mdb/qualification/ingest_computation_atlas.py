"""Import explicitly selected audit scalars into an atlas; no model/equations.

The selection binds exact audit bytes and JSON pointers. It supplies contrast,
consumer, unit and claim labels; this tool never guesses them from prose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from mdb.computation_atlas import ComputationAtlas


def pointer(value, path):
    if path == "":
        return value
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("selection needs an explicit JSON pointer")
    for component in path[1:].split("/"):
        if re.search(r"~(?![01])", component):
            raise ValueError("invalid JSON pointer escape")
        component = component.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", component):
                raise ValueError("JSON pointer array index must be canonical")
            component = int(component)
        value = value[component]
    return value


def ingest(atlas, catalog, audit_path, selection_path):
    raw, selection_raw = audit_path.read_bytes(), selection_path.read_bytes()
    audit, selection = json.loads(raw), json.loads(selection_raw)
    if hashlib.sha256(raw).hexdigest() != selection["audit_sha256"]:
        raise ValueError("selection audit SHA256 differs")
    audit_ref = {"path": str(audit_path), "sha256": hashlib.sha256(raw).hexdigest()}
    selection_ref = {"path": str(selection_path), "sha256": hashlib.sha256(selection_raw).hexdigest()}
    effects, continuation, counterexamples, identifiers = {}, {}, {}, []
    for record in selection["records"]:
        identifier = record["equation_id"]
        identifiers.append(identifier)
        effects[identifier] = [{"consumer_id": item["consumer_id"], "contrast_id": item["contrast_id"],
            "units": item["units"], "value": pointer(audit, item["value_pointer"]),
            "provenance": [{**audit_ref, "json_pointer": item["value_pointer"]}, selection_ref]}
            for item in record.get("effects", [])]
        if "continuation" in record:
            item = record["continuation"]
            continuation[identifier] = {"status": item["status"], "value": pointer(audit, item["pointer"]),
                "provenance": [{**audit_ref, "json_pointer": item["pointer"]}, selection_ref]}
        counterexamples[identifier] = [{"claim_id": item["claim_id"], "status": item["status"],
            "value": pointer(audit, item["pointer"]),
            "provenance": [{**audit_ref, "json_pointer": item["pointer"]}, selection_ref]}
            for item in record.get("counterexamples", [])]
    return ComputationAtlas(atlas).ingest(catalog, identifiers,
        provenance={"run_id": selection["run_id"], "artifacts": [audit_ref, selection_ref]},
        signed_effects=effects, continuation=continuation, counterexamples=counterexamples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("atlas", "catalog", "audit", "selection"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    records = ingest(args.atlas, args.catalog, args.audit, args.selection)
    print(json.dumps({"records": [row["fingerprint"] for row in records], "model_calls": 0,
                      "equation_evaluations": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
