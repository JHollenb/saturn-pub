"""Materialize a source-qualified cross-run atlas from saved audit metadata.

Scalar subtraction and immutable array decoding are posthoc accounting only;
no source/model equation or native model is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from collections.abc import Mapping
from pathlib import Path

from mdb.computation_atlas import ComputationAtlas
from mdb.equation_comparison import EquationCatalog
from mdb.forecast_transaction import load_forecast_tree, verify_transaction

EXPECTED_AUDIT = "99d3ba8249f56b39de2c3e5668fba0e434332cdb00f33dca54ffbf42bd96ddbe"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text())


def reference(path, pointer=None):
    parts = path.resolve().parts
    value = {"path": "/".join(parts[parts.index("artifacts"):]) if "artifacts" in parts else str(path),
             "sha256": sha(path.read_bytes())}
    if pointer is not None:
        value["json_pointer"] = pointer
    return value


def save(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(raw)


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def values(run, name, audit):
    tx = run / "extracted" / name / "transaction"
    pins = audit["source"]["transaction"]
    verification = verify_transaction(tx, expected_seal_sha256=pins["seal_sha256"], expected_event_head=pins["event_head"])
    metadata = read(tx / "observations/native.json")["value"]["items"]
    selected = {key: {"kind": "mapping", "items": {field: row["items"][field] for field in ("heads", "roots", "before", "decoded")}}
                for key, row in metadata.items()}
    return load_forecast_tree(tx, {"kind": "mapping", "items": selected}), verification


def materialize(root, prior_run, helper_source):
    root, prior_run = root.resolve(), prior_run.resolve()
    index_path, atlas_root = root / "atlas-index.json", root / "atlas"
    require(not index_path.exists() and not list((atlas_root / "receipts").glob("*.json")), "atlas publication already exists")
    new_run = root / "run-02"
    new_path, old_path = new_run / "analysis-v1.json", prior_run / "analysis-v1.json"
    require(sha(new_path.read_bytes()) == EXPECTED_AUDIT, "independent audit differs from the externally supplied SHA256")
    new, old = read(new_path), read(old_path)
    require(new["mechanics_status"] == "valid", "new source audit is not mechanically valid")
    new_values, new_verified = values(new_run, "state-transition-result", new)
    old_values, old_verified = values(prior_run, "context-window-result", old)
    comparison_path, oracle_path = new_run / "derived/comparisons.json", new_run / "derived/isa-oracle-score.json"
    comparisons, oracle = read(comparison_path)["rows"], read(oracle_path)
    require(oracle["source_observation_sha256"] == sha((new_run / "extracted/state-transition-result/transaction/observations/native.json").read_bytes()),
            "posthoc oracle score addresses other observations")
    new_catalog, old_catalog = EquationCatalog(new_run / "derived/equations"), EquationCatalog(prior_run / "equations-source-qualified")
    old_records = old_catalog.find()
    require(len(comparisons) == len(new_values) == 28 and len(old_records) == 10, "source record geometry differs")
    selections = root / "atlas-selections"
    normalized = {"schema": "mdb-state-transition-atlas-selected-audit-v1", "rows": {},
        "sources": {"new_audit": reference(new_path), "prior_audit": reference(old_path),
                    "comparisons": reference(comparison_path), "posthoc_oracle": reference(oracle_path)},
        "scope": "Posthoc saved-scalar accounting. No equations/model calls. Own-state trajectory contrasts are not matched-input interventions."}
    plans = {}
    for label, catalog, records, audit, raw, audit_path in (
            ("new", new_catalog, comparisons, new, new_values, new_path),
            ("prior", old_catalog, old_records, old, old_values, old_path)):
        selection_rows = []
        for row in records:
            if label == "new":
                row_id, equation_id = row["row_id"], row["equation_record"]
                equation = catalog.show(equation_id)
                require(equation["report"] == row["head_report"], "derived equation report join differs")
                measured = audit["consumer_effects"][row_id]
                baseline, scope = measured["baseline_row"], measured["contrast_scope"]
            else:
                equation, equation_id = row, row["fingerprint"]
                row_id, baseline = row["evidence"]["row_id"], row["evidence"]["native_baseline_row"]
                scope = "matched-input"
            actual, parent = raw[row_id], raw[baseline]
            same = all(actual["roots"][key].shape == parent["roots"][key].shape
                       and actual["roots"][key].dtype == parent["roots"][key].dtype
                       and actual["roots"][key].tobytes() == parent["roots"][key].tobytes() for key in actual["roots"])
            require(scope != "matched-input" or same, "matched-input effect has divergent serving roots")
            value = float(actual["heads"]["regs"][0, 6, 1]) - float(parent["heads"]["regs"][0, 6, 1])
            if label == "new":
                require(value == measured["RSI_bit1"]["effect_vs_baseline"] and same == measured["same_serving_roots"], "audited scalar/root contrast join differs")
            selected_id = label + "." + row_id
            selected = {"row_id": row_id, "baseline_row": baseline, "effect": value, "contrast_scope": scope,
                "same_serving_roots": same, "actual_decode": plain(actual["decoded"]), "baseline_decode": plain(parent["decoded"]),
                "equation_id": equation_id, "source_audit": reference(audit_path), "counterexamples": []}
            entry = {"equation_id": equation_id, "row_id": row_id, "effects": [{"consumer_id": "regs[6,1]",
                "contrast_id": "observed-minus-baseline/" + scope, "units": "logit", "value_pointer": "/rows/" + selected_id + "/effect"}]}
            if label == "new" and row_id.startswith("sequence."):
                _, arm, step = row_id.split(".")
                selected["continuation"] = {"from_instruction": int(step), "arm": arm,
                    "steps": {key: content for key, content in audit["native_chains"][arm].items() if int(key) >= int(step)},
                    "scope": "Actual own-state lineage with explicit host segment boundaries; no new forecast inferred."}
                entry["continuation"] = {"status": "observed-own-state-chain", "pointer": "/rows/" + selected_id + "/continuation"}
            if label == "new" and row_id.startswith("mlp."):
                native_id = "mlp." + row_id.split(".")[1] + ".native"
                score = oracle["rows"][native_id]
                require(score["register_mismatches"] or score["flag_mismatches"] or not score["length_matches"], "expected native oracle counterexample is absent")
                selected["counterexamples"].append({"claim_id": "native-baseline-implements-supplied-ISA-op", "status": "posthoc-oracle-mismatch",
                    "native_row": native_id, "score": score, "source": reference(oracle_path),
                    "scope": "Oracle supplied after collection; never a forecast/model/Act input."})
            if label == "new" and row_id == "mlp.and.zero_post2":
                require(parent["decoded"]["register_words"][7] == 261 and actual["decoded"]["register_words"][7] == 263, "late-MLP counterexample changed")
                selected["counterexamples"].append({"claim_id": "late-MLP-zero-preserves-RDI", "status": "counterexample-observed",
                    "native_RDI": 261, "zero_RDI": 263, "scope": "Failure of a working cross-context generalization, not a previously validated rule."})
            entry["counterexamples"] = [{"claim_id": item["claim_id"], "status": item["status"],
                "pointer": f"/rows/{selected_id}/counterexamples/{index}"} for index, item in enumerate(selected["counterexamples"])]
            normalized["rows"][selected_id] = selected
            selection_rows.append(entry)
        plans[label] = {"run_id": audit["source"]["job_id"], "records": selection_rows}
    audit_file = selections / "selected-effects-v1.json"
    save(audit_file, normalized)
    helper = runpy.run_path(str(helper_source.resolve()))
    atlas, imported, selection_refs = ComputationAtlas(atlas_root), {}, {}
    for label, catalog in (("new", new_catalog), ("prior", old_catalog)):
        plan = {**plans[label], "audit_sha256": sha(audit_file.read_bytes())}
        selection_path = selections / (label + "-selection-v1.json")
        save(selection_path, plan)
        rows = helper["ingest"](atlas_root, catalog.root, audit_file, selection_path)
        imported[label] = {entry["row_id"]: row["fingerprint"] for entry, row in zip(plan["records"], rows, strict=True)}
        selection_refs[label] = reference(selection_path)
    matches = atlas.find(consumer_id="regs[6,1]")
    require(len(matches) == 38, "cross-run atlas consumer query must retain all 38 records")
    for row in matches:
        context = row["equation"]["tested_context"]
        require(len(atlas.find(applicable_to=context)) == 1, "exact applicability context is ambiguous")
        changed = {**context, "source_identity": "0" * 64}
        require(not atlas.find(applicable_to=changed), "altered source still matches")
    families = []
    for layer in range(3):
        ids = [imported["new"][f"mlp.{op}.zero_post{layer}"] for op in ("or", "and")]
        candidate = atlas.compare(ids, consumer_id="regs[6,1]", contrast_id="observed-minus-baseline/matched-input")
        require(candidate["hypothesis_status"] == "family-candidate", "expected candidate comparison missing")
        families.append({"id": candidate["fingerprint"], "law_id": candidate["law_id"], "route_id": candidate["route_id"],
            "row_ids": [f"mlp.{op}.zero_post{layer}" for op in ("or", "and")],
            "effects": [row["effect"]["value"] for row in candidate["rows"]], "rule_status": candidate["rule_status"]})
    source_raw, helper_raw = Path(__file__).read_bytes(), helper_source.read_bytes()
    for name, content in (("publisher", source_raw), ("ingester", helper_raw)):
        snapshot = selections / (name + "-source-" + sha(content)[:16] + ".py")
        with snapshot.open("xb") as stream:
            stream.write(content)
    index = {"schema": "mdb-state-transition-cross-run-atlas-index-v1", "atlas_path": str(atlas_root.relative_to(root.parent.parent)),
        "records": imported, "counts": {"new_observations": 28, "prior_observations": 10, "consumer_matches": 38, "family_candidates": 3},
        "queries": {"consumer_id": "regs[6,1]", "exact_context_unique": 38, "altered_source_matches": 0},
        "family_candidates": families, "selections": selection_refs, "selected_effects": reference(audit_file),
        "sources": normalized["sources"], "source_hashes": {"publisher": sha(source_raw), "ingester": sha(helper_raw)},
        "verified_transactions": {"new": new_verified, "prior": old_verified},
        "counterexamples": {key: row["counterexamples"] for key, row in normalized["rows"].items() if row["counterexamples"]},
        "continuation_scope": "15 sequence observations retain audited native own-state chains and host segment boundaries; other imported records make no continuation claim.",
        "comparison_scope": "Three caller-selected OR/AND contrasts on the same structural zero-MLP routes. No cross-run law-ID aliasing, automatic grouping, learned rule or validation promotion.",
        "effect_scope": "Posthoc actual RSI-bit1 logit minus explicit baseline. Matched-input and divergent-own-state-trajectory contrasts have different contrast IDs.",
        "new_model_calls": 0, "equation_evaluations": 0, "terminal_status": "unassessed"}
    save(index_path, index)
    return {"index": str(index_path), "sha256": sha(index_path.read_bytes()), "counts": index["counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "prior-run", "ingest-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.root, args.prior_run, args.ingest_source), sort_keys=True))


if __name__ == "__main__":
    main()
