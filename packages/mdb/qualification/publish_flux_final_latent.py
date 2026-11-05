"""Curate accepted saved final-latent evidence with public equation/atlas APIs.

This publishes source-bound numerical observations; it executes no predictor,
model, replay or family induction. Native VAE outcomes belong to the complete
final-Z4 program, without causal attribution to an individual catalogued field.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from mdb.computation_atlas import ComputationAtlas
from mdb.equation_comparison import EquationCatalog, applicability, compare_equations
from mdb.forecast_transaction import load_forecast_tree, verify_transaction
from mdb.tensor_archive import load_tensor_archive

HERE = Path(__file__).resolve().parent
MAX_ELEMENTS = 4_194_304
OUTPUT_ROLES = ("final_noise", "final_Z4")
REFERENCES = {"native": "native", "matched_native_suffix": "bridge.predicted"}
FIELD_NAMES = tuple(role+"."+reference for role in OUTPUT_ROLES for reference in REFERENCES)
COUNTS = {"model_loads":1,"prompt_encodings":1,"noise_draws":1,"adapter_steps":134,"acts":3,
    "candidate_calls":0,"receipt_emissions":137,"images":4,"vae_calls":4,"physical_batch":1}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def member(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name,
        "noncanonical evidence path")
    target = Path(root) / path
    require(not any(parent.is_symlink() for parent in (target, *target.parents)), "symlink evidence path")
    return target


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def audit_authority(run, audit, request, *, current_auditor=None):
    """Accept a changed reader only through an explicit source-bound receipt."""
    original = request["independent_math_sources"]["analyze_flux_final_latent.py"]
    frozen = Path(run) / "independent-math-freeze/analyze_flux_final_latent.py"
    current = Path(current_auditor) if current_auditor is not None else HERE / "analyze_flux_final_latent.py"
    actual = sha(current)
    require(sha(frozen) == original and audit["auditor_source_sha256"] == actual
        and audit["independent_math_sources"] == request["independent_math_sources"],
        "accepted auditor/frozen mathematical source differs")
    if actual == original:
        require(not audit.get("audit_correction"), "unchanged auditor has an unexpected correction authority")
        return {"status": "pre-outcome-auditor", "preoutcome_auditor_sha256": original,
            "accepted_auditor_sha256": actual, "correction": None}
    correction = audit.get("audit_correction")
    require(isinstance(correction, dict) and audit.get("preoutcome_auditor_sha256") == original,
        "changed auditor requires explicit offline correction receipt")
    path = Path(correction["path"])
    require(path.is_absolute() and not path.is_symlink() and sha(path) == correction["sha256"],
        "offline correction receipt file/hash differs")
    record = read(path)
    require(record == correction["record"] and record["schema"] == "mdb-flux-final-latent-audit-correction-v1"
        and record["request_sha256"] == request["request_sha256"] == audit["request_sha256"]
        and record["job_id"] == audit["job_id"]
        and record["original_auditor_sha256"] == original
        and record["corrected_auditor_sha256"] == actual
        and type(record["model_calls"]) is int and record["model_calls"] == 0
        and isinstance(record.get("reason"), str) and record["reason"].strip(),
        "offline correction receipt does not bind the actual request/job/source repair")
    return {"status": "explicit-offline-instrument-correction", "preoutcome_auditor_sha256": original,
        "accepted_auditor_sha256": actual, "correction": correction,
        "scope": "Post-outcome saved-evidence reader repair; original pre-outcome source and frozen numerical math retained. Producer, sealed forecast and native experiment are unchanged; no model rerun."}


def curate(output, *, fields, binding, tested_context, provenance, endpoints):
    """Four whole-suffix observations from two sealed fields and one history."""
    output = Path(output)
    require(set(fields) == set(FIELD_NAMES), "final-latent composition comparison inventory differs")
    output.mkdir(parents=True, exist_ok=False)
    catalog, atlas = EquationCatalog(output/"equations"),ComputationAtlas(output/"atlas")
    records,effects,futures,exceptions,metadata,query_contexts = [],{},{},{},[],[]
    for name in FIELD_NAMES:
        role,reference_kind = name.split(".")
        field,reference = fields[name],REFERENCES[reference_kind]
        endpoint = endpoints[reference_kind]
        require(endpoint["candidate_branch"] == "full.predicted" and endpoint["reference_branch"] == reference,
            "whole-program VAE future mapping differs")
        context = {**tested_context,"horizon":binding["horizon"],"frontier":binding["frontier"],
            "output_role":role,"reference_kind":reference_kind,"reference_cut":field["provenance"]["native_cut"]}
        conditions = [{"field":key,"op":"eq","value":context[key]} for key in
            ("model_identity","source_identity","execution","horizon","input_cut","seed","prompt","frontier",
             "output_role","reference_kind","reference_cut")]
        report = compare_equations({name:field["predicted"]},{name:field["observed"]},
            decision_rules={name:"none"},include_residuals=False,max_elements=MAX_ELEMENTS)
        record = catalog.record(law_id="flux.denoising.own-state-suffix",route_id="flux.final."+name,
            binding=binding,report=report,conditions=conditions,tested_context=context,
            readers=["flux.terminal.Z4","flux.native.VAE"],
            evidence={"output_role":role,"reference_kind":reference_kind,"array_provenance":field["provenance"],
                "artifacts":provenance["artifacts"],
                "composition":"Whole twenty-single/readout/Euler suffix from own predicted joint4 I/T, project context and Z3; prospective44-stage horizon from observed pre6.",
                "independence":"One seed107 native history and one own-state forecast. Four dependent comparisons reuse two sealed fields and one matched native suffix; no independent replication or family induction.",
                "decision_scope":"All supplied tensor discrepancies retained; no image-quality or semantic gate.",
                "intervention_scope":"Complete finalZ4 program admission. No isolated noise, readout or Euler intervention; noise-linked VAE future belongs to the complete finalZ4 program."})
        identifier = record["fingerprint"]
        records.append(identifier)
        effects[identifier] = [{"consumer_id":consumer,"contrast_id":"full.predicted-minus-"+reference,
            "value":value,"units":units,"provenance":provenance["artifacts"],
            "scope":"Complete terminalZ4 admission followed by nativeVAE only; no isolated effect of this named field or numerical component. RGB is the actual VAE response to latent differences, not a predicted RGB field or VAE kernel error."}
            for consumer,value,units in (("flux.terminal.Z4",endpoint["latent"]["signed_mean"],"latent-value"),
                ("flux.native.VAE",endpoint["rgb"]["signed_mean"],"uint8-channel-value"))]
        futures[identifier] = {"status":"observed-native-VAE-only","branch":"full.predicted","reference_branch":reference,
            "denoiser_calls_after_installation":0,"measurements":endpoint,"provenance":provenance["artifacts"],
            "scope":"Actual nativeVAE consumes complete sealed predictedZ4. Native/sham/full rows have no denoiser continuation; the bridge alone executes22 native transitions before its VAE. No independently predicted RGB or isolated noise intervention."}
        residual = report["fields"][name]["residual"]
        exceptions[identifier] = ([{"claim_id":"exact-native-CUDA-equality","status":"observed-counterexample",
            "measurements":residual,"provenance":provenance["artifacts"],
            "scope":"Whole-composition numerical discrepancy; no numerical discovery gate or isolated component attribution."}]
            if residual["nonzero"] else [])
        metadata.append({"equation_id":identifier,"law_id":"flux.denoising.own-state-suffix","output_role":role,
            "reference_kind":reference_kind,"report":report,"array_provenance":field["provenance"],
            "candidate_branch":"full.predicted","reference_branch":reference})
        query_contexts.append({"context":context,"conditions":conditions,"equation_id":identifier})
    imported = atlas.ingest(output/"equations",records,provenance=provenance,
        signed_effects=effects,continuation=futures,counterexamples=exceptions)
    query_rows = []
    for item in query_contexts:
        context,conditions = item["context"],item["conditions"]
        changed = copy.deepcopy(context)
        changed["seed"] += 1
        missing = copy.deepcopy(context)
        del missing["input_cut"]
        row = {"equation_id":item["equation_id"],"exact_context_matches":len(atlas.find(applicable_to=context)),
            "changed_seed_matches":len(atlas.find(applicable_to=changed)),
            "missing_input_cut_matches":len(atlas.find(applicable_to=missing)),
            "missing_input_cut_applicability":applicability(conditions,missing)["status"]}
        require(row["exact_context_matches"] == 1 and row["changed_seed_matches"] == 0
            and row["missing_input_cut_matches"] == 0 and row["missing_input_cut_applicability"] == "unknown",
            "public occurrence applicability query differs")
        query_rows.append(row)
    queries = {key:sum(row[key] for row in query_rows) for key in
        ("exact_context_matches","changed_seed_matches","missing_input_cut_matches")}
    queries.update(missing_input_cut_applicability="unknown",occurrences=query_rows,
        authority="Sum of four fully specified occurrence queries, one match each. Supplied predicates do not establish generalization.")
    return {"records":metadata,"atlas_observation_ids":[item["fingerprint"] for item in imported],"queries":queries,
        "query_contexts":query_contexts,"curated_records":4,"distinct_named_field_comparisons":4,"native_histories":1,
        "distinct_sealed_output_fields":2,"family_candidates_created":0,"rule_induction":"not performed"}


def difference(left, right):
    require(left.shape == right.shape, "endpoint shape differs")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {"signed_mean": float(delta.mean()), "mae": float(np.abs(delta).mean()),
        "rms": float(np.sqrt(np.mean(delta * delta))), "l2": float(np.linalg.norm(delta.ravel())),
        "max_abs": float(np.abs(delta).max()), "changed_value_count": int(np.count_nonzero(delta)),
        "sample_count": int(delta.size)}


def publish(run_dir, audit_path, output, *, historical_references=()):
    from PIL import Image

    run, audit_path, output = Path(run_dir).resolve(), Path(audit_path).resolve(), Path(output).resolve()
    require(not output.exists(), "refuse to overwrite publication")
    request, job, collection, audit = [read(path) for path in
        (run / "request.json", run / "job.json", run / "collection.json", audit_path)]
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    request_sha = hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(request_sha == request["request_sha256"] == audit["request_sha256"] == collection["request_sha256"]
        and job["state"] == "succeeded" and job["result"]["returncode"] == 0
        and job["job_id"] == audit["job_id"] == collection["job_id"] and audit["mechanics_status"] == "valid"
        and audit["models_executed"] is False, "accepted audit/job/request differs")
    authority = audit_authority(run, audit, request)
    require(sha(run / "collection.json") == audit["collection"]["sha256"], "accepted audit/collection binding differs")
    for name, digest in request["source_hashes"].items():
        require(sha(member(run / "payload", name)) == digest, "producer payload source differs")
    for name, digest in request["independent_math_sources"].items():
        require(sha(member(run / "independent-math-freeze", name)) == digest, "frozen independent source differs")
    require(all(sha(member(run, name)) == digest for name, digest in collection["bindings"].items()), "collection metadata binding differs")
    inventory = read(run / "collection-local-inventory.json")
    require(inventory == read(run / "collection-remote-before.json") == read(run / "collection-remote-after.json"),
        "authenticated before/after/local inventories differ")
    data = run / "extracted/flux-final-latent-result"

    def bound(name):
        path = member(data, name)
        require(sha(path) == inventory["files"][name]["sha256"], "producer metadata changed")
        return path

    manifest = read(bound("remote-manifest.json"))
    require(manifest["files"] == {key: value for key, value in inventory["files"].items() if key != "remote-manifest.json"}
        and manifest["job_id"] == job["job_id"] and manifest["request_sha256"] == request_sha
        and sha(data / "remote-manifest.json") == collection["manifest_sha256"], "producer manifest differs")
    source, summary, plan, binding, segment = [read(bound(name)) for name in
        ("source.json", "summary.json", "plan.json", "binding.json", "native-segment.json")]
    require(plan == request["config"] and source["context"] == plan["context"] and plan["context"]["seed"] == 107
        and source["source_hashes"] == request["source_hashes"]
        and source["request_sha256"] == summary["request_sha256"] == request_sha
        and source["job_id"] == summary["job_id"] == job["job_id"] and summary["counts"] == audit["counts"] == COUNTS
        and sha(data / "summary.json") == collection["summary_sha256"], "producer scientific metadata differs")
    tx, seal = data / "transaction", summary["seal"]["seal_sha256"]
    require(seal == audit["seal"]["seal_sha256"] == collection["seal_sha256"], "audit/producer/collection seal differs")
    verification = verify_transaction(tx, expected_seal_sha256=seal)
    require(verification["phase"] == "closed", "source numerical transaction is not complete")
    encoded = read(bound("transaction/forecasts/final_latent.json"))["predictions"]["items"]["final_step"]["items"]
    prediction = {role:load_forecast_tree(tx,encoded[stage]["items"][field])
        for role,stage,field in (("final_noise","readout","noise"),("final_Z4","commit","latent"))}
    rows = {name:read(bound("rows/"+name+".json")) for name in ("native","sham","bridge.predicted","full.predicted")}
    require(len(segment["boundaries"]) == len(segment["receipts"]) == 44
        and len(rows["bridge.predicted"]["bridge_boundaries"]) == len(rows["bridge.predicted"]["bridge_records"]) == 22,
        "complete native/bridge trajectory census differs")
    for name,row in rows.items():
        terminal = row["terminal_final"]
        require(terminal["clock"] == {"step":4,"phase":"project","cursor":0}
            and all(terminal["closure"][slot] is None for slot in ("context","image","text","noise_prediction"))
            and row["final_cut"] == terminal["cut"] and row["continuation_receipts"] == []
            and len(row["bridge_receipts"]) == (22 if name == "bridge.predicted" else 0),
            "terminal reset or zero-denoiser future contract differs")
    require(rows["native"]["final_cut"] == segment["native_final"]["cut"]
        and rows["bridge.predicted"]["bridge_records"][20]["after"] == rows["bridge.predicted"]["bridge_boundaries"][20],
        "selected native noise/terminal lineage differs")
    fields,endpoints = {},{}
    for reference_kind,reference_name in REFERENCES.items():
        selected = ({"final_noise":segment["boundaries"][42],"final_Z4":segment["native_final"]}
            if reference_name == "native" else
            {"final_noise":rows[reference_name]["bridge_boundaries"][20],"final_Z4":rows[reference_name]["terminal_final"]})
        for role,slot,stage in (("final_noise","noise_prediction","readout"),("final_Z4","latent","commit")):
            native = selected[role]
            expected_clock = (3,"scheduler") if role == "final_noise" else (4,"project")
            require((native["point"]["logical_step"],native["point"]["phase"]) == expected_clock,
                "selected noise/latent field lifetime differs")
            record = native["closure"][slot]
            observed = load_tensor_archive(data,record)
            predicted = prediction[role]
            require(predicted.dtype == observed.dtype == np.float32 and predicted.shape == observed.shape,
                "full final output shape/dtype differs")
            fields[role+"."+reference_kind] = {"predicted":predicted,"observed":observed,
                "provenance":{"sealed_array":encoded[stage]["items"]["noise" if role == "final_noise" else "latent"],
                    "native_array":record,"native_cut":native["cut"],"native_frontier":native["point"],
                    "reference_branch":reference_name,"candidate_branch":"full.predicted",
                    "matched_suffix_input_cut":rows["bridge.predicted"]["installed"]["cut"]}}
        left = load_tensor_archive(data,rows["full.predicted"]["final_latent"])
        right = load_tensor_archive(data,rows[reference_name]["final_latent"])
        with Image.open(bound(rows["full.predicted"]["image"]["path"])) as image:
            pixels_left = np.array(image)
        with Image.open(bound(rows[reference_name]["image"]["path"])) as image:
            pixels_right = np.array(image)
        endpoints[reference_kind] = {"candidate_branch":"full.predicted","reference_branch":reference_name,
            "latent":difference(left,right),"rgb":difference(pixels_left,pixels_right)}

    def reference(path):
        return {"path": os.path.relpath(path, output), "sha256": sha(path)}

    artifacts = [reference(path) for path in (audit_path, run / "request.json", run / "collection.json",
        run / "collection-local-inventory.json", data / "remote-manifest.json", data / "source.json",
        data / "summary.json", data / "native-segment.json", data / "binding.json", tx / "seal.json")]
    execution = {"backend": "diffusers", "native_execution": source["execution"], "forecast_backend": "NumPy"}
    source_identity = {**{"producer:" + name: value for name, value in request["source_hashes"].items()},
        **{"independent:" + name: value for name, value in request["independent_math_sources"].items()},
        **{"native:" + name: value for name, value in source["implementation_sources"].items()}}
    equation_binding = {"model_identity": source["model_identity"], "execution": execution,
        "input_identity": {"cut": binding["input_cut"], "request_sha256": request_sha},
        "frontier": {"input_cut": binding["input_cut"], "point": binding["point"]},
        "horizon": "Observed step2/pre6 through44 own-state transitions:14single/readout/Euler/project3/joint0..4/20single/readout/EulerZ4; nativeVAE only after complete finalZ4 admission",
        "source_identity": source_identity}
    context = {"model_identity": source["model_identity"], "source_identity": source_identity, "execution": execution,
        "input_cut": binding["input_cut"], "seed": plan["context"]["seed"], "prompt": plan["context"]["prompt"]}
    result = curate(output, fields=fields, binding=equation_binding, tested_context=context,
        provenance={"run_id": job["job_id"], "artifacts": artifacts}, endpoints=endpoints)
    result.update(schema="mdb-flux-final-latent-equation-publication-v1", mechanics_status="valid",
        job_id=job["job_id"], request_sha256=request_sha, model_calls=0, prediction_calls=0,
        publisher_source_sha256=sha(Path(__file__)), accepted_audit=reference(audit_path), accepted_auditor_authority=authority,
        reference_bases={"publication": ".", "run": os.path.relpath(run, output),
            "native_arrays":os.path.relpath(data,output),"sealed_arrays":os.path.relpath(tx,output)}, counts=audit["counts"],
        forecast_seal=summary["seal"], transaction_verification=verification,
        historical_references=[reference(Path(path).resolve()) for path in historical_references],
        historical_scope="References only; no pooling, independent replication or automatic family induction.",
        custody_scope="Accepted full saved-evidence audit and authenticated complete inventories bound. Selected metadata/arrays rechecked without repeating the full audit.",
        scope="Four dependent final-noise/Z4 composition comparisons from two sealed fields and one seed107 native history, through public EquationCatalog/ComputationAtlas only. No semantic or universal claim.")
    write(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--historical-reference", action="append", type=Path, default=[])
    args = parser.parse_args()
    report = publish(args.run_dir, args.audit, args.output, historical_references=args.historical_reference)
    print(json.dumps({key: report[key] for key in ("job_id", "mechanics_status", "curated_records", "queries")}))
