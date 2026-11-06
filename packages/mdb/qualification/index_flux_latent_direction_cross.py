"""Curate six exact-context terminal-latent equations from saved evidence.

No model, predictor recurrence, native VAE or job is executed. The mathematical
84-transition program ends at a latent; actual VAE PNGs are downstream evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

from mdb.computation_atlas import ComputationAtlas
from mdb.equation_comparison import EquationCatalog, applicability, compare_equations

HERE = Path(__file__).resolve().parent
PUBLISHER_SHA256 = "d0484413bc5fa86262ce07d442b84c65f48152d4654844034590e73cebfef8ca"
ANALYZER_SHA256 = "51797f1587c755d6b2790268025fe8d4a776e8383f6b9fe10bf6962b9756dda4"
CELLS = ("c1-o0", "c0-o1")
BRANCHES = tuple(c + suffix for c in CELLS for suffix in (".sham", ".d-c1-o0", ".d-c0-o1"))
LAW = "flux.denoising.own-state-suffix"
ROUTE = "flux.three-pass.terminal-latent"
SCOPE = ("Six dependent occurrences using two known receivers and two reused incoming-error directions. "
    "The low own-direction input was already known. Effective BF16 directions differ across receivers. "
    "Whole three-pass numerical compositions, not semantic rules or isolated component effects. "
    "Native VAE consumes terminal latents; no mathematical VAE prediction, family induction or rule promotion.")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def pinned(name, expected):
    path = HERE / name
    require(sha(path) == expected, "qualified saved-only source differs: " + name)
    return runpy.run_path(str(path))


def curate(output, *, contexts, arrays, occurrences, provenance):
    """Public catalogue/atlas only. ``arrays(branch)`` supplies saved terminal P/N."""
    require(set(contexts) == set(occurrences) == set(BRANCHES), "six occurrence census differs")
    output = Path(output)
    require(not output.exists(), "fresh index output required")
    output.mkdir(parents=True)
    catalog, atlas = EquationCatalog(output / "equations"), ComputationAtlas(output / "atlas")
    records, effects, futures, exceptions, summaries, query_contexts = [], {}, {}, {}, [], []
    for branch in BRANCHES:
        context, occurrence = contexts[branch], occurrences[branch]
        require(context["branch"] == branch and context["horizon"] == 84
            and context["clock"] == {"step": 1, "phase": "project", "cursor": 0}, "occurrence frontier differs")
        predicted, observed = arrays(branch)
        predicted, observed = np.asarray(predicted), np.asarray(observed)
        require(predicted.shape == observed.shape and predicted.size > 0, "terminal shape differs")
        comparison = compare_equations({"terminal_latent": predicted}, {"terminal_latent": observed},
            decision_rules={"terminal_latent": "none"}, max_elements=8_000_000)
        delta = predicted.astype(np.float64) - observed.astype(np.float64)
        endpoint = occurrence["endpoint"]["latent"]
        require(np.isclose(float(delta.mean()), endpoint["signed_mean"], rtol=1e-12, atol=1e-12)
            and np.isclose(float(np.sqrt(np.mean(delta * delta))), endpoint["rms"], rtol=1e-12, atol=1e-12),
            "terminal saved arrays differ from accepted endpoint account")
        conditions = [{"field": key, "op": "eq", "value": value} for key, value in context.items()]
        binding = {key: context[key] for key in ("model_identity", "execution", "source_identity", "horizon")}
        binding.update(input_identity=identity(context["installed_input"]), frontier={"input_cut": context["input_cut"],
            "receiver": context["receiver"], "clock": context["clock"]})
        record = catalog.record(law_id=LAW, route_id=ROUTE, binding=binding, report=comparison,
            conditions=conditions, tested_context=context, readers=["flux.terminal.latent", "flux.native.VAE"],
            evidence={"branch": branch, "scope": SCOPE, "occurrence": occurrence,
                "artifacts": provenance["artifacts"], "residual_direction": "native-minus-sealed mathematical latent",
                "signed_effect_direction": "sealed mathematical latent minus native latent; signed mean in latent-value units",
                "consumer_scope": "Actual predicted and native PNGs are native VAE responses; the equation does not predict RGB.",
                "quality_policy": "No numerical tolerance or image quality gate; all discrepancies/counterexamples retained."})
        identifier = record["fingerprint"]
        records.append(identifier)
        effects[identifier] = [{"consumer_id": consumer, "contrast_id": contrast,
            "value": occurrence["endpoint"][field]["signed_mean"], "units": units,
            "sign_orientation": "predicted-branch minus native-branch; signed coordinate mean",
            "scope": "Latent program residual" if field == "latent" else "Observed native-VAE consumer comparison, not a mathematical RGB equation",
            "provenance": provenance["artifacts"]} for consumer, field, units, contrast in
            (("flux.terminal.latent", "latent", "latent-value", "sealed-terminal-latent-minus-native-terminal-latent"),
             ("flux.native.VAE", "RGB", "uint8-channel-value", "native-VAE-of-sealed-latent-minus-native-VAE-of-native-latent"))]
        if occurrence.get("native_changed_minus_sham") is not None:
            effects[identifier].extend({"consumer_id": consumer,
                "contrast_id": "native-changed-minus-same-receiver-sham", "value": occurrence["native_changed_minus_sham"][field]["signed_mean"],
                "units": units, "sign_orientation": "native changed minus native same-receiver sham; signed coordinate mean",
                "provenance": provenance["artifacts"]} for consumer, field, units in
                (("flux.terminal.latent", "latent", "latent-value"), ("flux.native.VAE", "RGB", "uint8-channel-value")))
        futures[identifier] = {"status": "observed-native-VAE-consumer", "provenance": provenance["artifacts"],
            "PNG_sources": occurrence["PNG_sources"], "native_denoiser_suffix_steps": 84,
            "mathematical_program_steps": 84, "predicted_PNG_denoiser_calls": 0,
            "scope": "Predicted terminal decoded by actual native VAE before native targets; native terminal follows its own84 transitions.",
            "endpoint": occurrence["endpoint"]}
        exceptions[identifier] = ([{"claim_id": "exact-native-terminal-equality", "status": "observed-counterexample",
            "measurements": comparison["fields"]["terminal_latent"]["residual"], "provenance": provenance["artifacts"]}]
            if comparison["fields"]["terminal_latent"]["residual"]["nonzero"] else [])
        summaries.append({"branch": branch, "equation_id": identifier, "law_id": LAW, "route_id": ROUTE,
            "comparison": comparison, "occurrence": occurrence})
        query_contexts.append({"branch": branch, "context": context, "conditions": conditions, "equation_id": identifier})
        del predicted, observed, delta
    imported = atlas.ingest(output / "equations", records, provenance=provenance,
        signed_effects=effects, continuation=futures, counterexamples=exceptions)
    queries = []
    for item in query_contexts:
        context = item["context"]
        results = {"branch": item["branch"], "equation_id": item["equation_id"],
            "exact_matches": len(atlas.find(applicable_to=context))}
        for field, replacement in (("installed_input", {"changed": True}), ("receiver", "different-receiver"),
                ("clock", {"step": 2, "phase": "project", "cursor": 0})):
            changed = copy.deepcopy(context)
            changed[field] = replacement
            results["changed_" + field + "_matches"] = len(atlas.find(applicable_to=changed))
        missing = copy.deepcopy(context)
        del missing["input_cut"]
        results.update(missing_input_matches=len(atlas.find(applicable_to=missing)),
            missing_input_applicability=applicability(item["conditions"], missing)["status"])
        require(results["exact_matches"] == 1 and all(results[key] == 0 for key in
            ("changed_installed_input_matches", "changed_receiver_matches", "changed_clock_matches", "missing_input_matches"))
            and results["missing_input_applicability"] == "unknown", "public exact-context query mechanics differ")
        queries.append(results)
    result = {"schema": "mdb-latent-direction-equation-index-v1", "records": summaries,
        "atlas_observation_ids": [r["fingerprint"] for r in imported], "queries": queries,
        "curated_records": 6, "exact_context_matches": sum(row["exact_matches"] for row in queries),
        "distinct_laws": 1, "distinct_routes": 1, "family_candidates_created": 0, "rule_induction": "not performed",
        "scope": SCOPE, "query_contexts": query_contexts, "provenance": provenance}
    save(output / "result.json", result)
    return result


def index(run, *, math_run, parent, analysis, output):
    run, math_run, parent, analysis, output = [Path(v).resolve() for v in (run, math_run, parent, analysis, output)]
    require(not output.exists(), "fresh index output required")
    publisher = pinned("publish_flux_latent_direction_cross.py", PUBLISHER_SHA256)
    analyzer = pinned("analyze_flux_latent_direction_cross.py", ANALYZER_SHA256)
    p, h, q = publisher["helpers"]()
    native = publisher["native_producer"](h, run)
    cpu = publisher["verify_math"](h, q, math_run)
    original = p["producer"](h, parent, "flux-prompt-factorial-result", "mdb-prompt-factorial-ssh-collection-v1",
        "collection-job.json", expected_runtime_wheels=native["request"]["runtime_wheels"])
    report = read(analysis / "analysis.json")
    receipt = publisher["verify_analysis"](h, analysis, report, native, cpu)
    require(native["request"]["config"]["math_job_id"] == cpu["job"]["job_id"]
        and native["request"]["config"]["math_cohort_sha256"] == cpu["summary"]["cohort_seal_sha256"]
        and cpu["source"]["parent_job_id"] == original["job"]["job_id"]
        and cpu["source"]["parent_request_sha256"] == original["request"]["request_sha256"], "original/native/CPU source join differs")
    _, png_refs = publisher["load_images"](h, native, report)
    numeric = analyzer["numeric"]()
    conditions = {v["id"]: v for v in original["request"]["config"]["conditions"]}
    source_identity = {"indexer": sha(__file__), "analyzer": ANALYZER_SHA256,
        "math_request": cpu["request"]["request_sha256"], "native_request": native["request"]["request_sha256"],
        "original_request": original["request"]["request_sha256"]}
    programs, native_finals, contexts, occurrences = {}, {}, {}, {}
    artifacts = [{"path": str(directory / name), "sha256": sha(directory / name)} for directory, names in
        ((run, ("request.json", "collection.json", "job.json")), (math_run, ("request.json", "collection.json", "result.json")),
         (parent, ("request.json", "collection.json")), (analysis, ("analysis.json", "analysis-receipt.json"))) for name in names]
    for branch in BRANCHES:
        spec = cpu["inputs"]["branches"][branch]
        receiver = spec["receiver"]
        origin = cpu["inputs"]["receivers"][receiver]
        ref = cpu["summary"]["programs"][branch]
        program = read(h["file_path"](cpu["data"], ref["path"]))
        require(program["branch_id"] == branch and program["chain_length"] == len(program["transitions"]) == 84
            and program["inputs"] == spec, "sealed84 program/input identity differs")
        programs[branch] = program["terminal_latent"]
        with (native["data"] / "native" / (branch + ".jsonl")).open() as stream:
            rows = [json.loads(line) for line in stream]
        require(len(rows) == 84 and rows[-1]["logical_step"] == 3 and rows[-1]["local_index"] == 27, "native terminal cursor differs")
        native_finals[branch] = rows[-1]["outputs"]["latent"]
        require(native_finals[branch]["archive"] == native["summary"]["rows"][branch]["final_latent"], "native final consumer array differs")
        contexts[branch] = {"branch": branch, "receiver": receiver, "direction": spec["direction"] or "sham",
            "model_identity": native["source"]["model_identity"], "source_identity": source_identity,
            "execution": {"backend": "numpy-reference-versus-native-CUDA", "native": native["source"]["execution"],
                "math_plan": cpu["request"]["config"], "runtime_wheels": native["request"]["runtime_wheels"]}, "horizon": 84,
            "input_cut": native["summary"]["rows"][branch]["entry"]["cut"], "receiving_cut": origin["cut"]["cut"],
            "clock": {"step": 1, "phase": "project", "cursor": 0}, "installed_input": spec["input_descriptor"],
            "conditioning": {key: origin["origin"]["inputs"][key] for key in ("conditioning", "img_ids", "txt_ids", "timesteps", "sigmas")},
            "prompt_context": conditions[receiver]}
        occurrences[branch] = {"program": ref, "cohort_seal_sha256": cpu["summary"]["cohort_seal_sha256"],
            "terminal_arrays": {"sealed": programs[branch], "native": native_finals[branch]}, "PNG_sources": png_refs[branch],
            "endpoint": report["endpoints"][branch], "response_accounts": report["responses"].get(branch, []),
            "RGB_response": report["RGB"].get(branch), "effective_input": spec,
            "native_changed_minus_sham": ({"latent": next(row for row in report["responses"][branch]
                if row["position"] == 83)["fields"]["latent"]["native_response"],
                "RGB": report["RGB"][branch]["native_response"]} if branch in report["responses"] else None),
            "expectations": report["expectations"], "input_construction_audit": report["input_construction_audit"],
            "consumer_join_audit": report["consumer_join_audit"], "all_signed_factorial_accounts": report["factorial"],
            "all_signed_RGB_factorial_accounts": report["RGB_factorial"]}
        del rows, program
    provenance = {"run_id": native["job"]["job_id"], "math_job_id": cpu["job"]["job_id"],
        "artifacts": artifacts, "analysis_receipt": receipt, **publisher["native_provenance"](native),
        "reader_custody_sources": native["reader_custody_sources"]}
    result = curate(output, contexts=contexts, occurrences=occurrences, provenance=provenance,
        arrays=lambda branch: (numeric["load_field"](cpu["data"], programs[branch]),
            numeric["load_field"](native["data"], native_finals[branch])))
    save(output / "receipt.json", {"schema": "mdb-latent-direction-index-receipt-v1", "source_sha256": sha(__file__),
        "publisher_source_sha256": PUBLISHER_SHA256, "analyzer_source_sha256": ANALYZER_SHA256,
        **publisher["native_provenance"](native), "reader_custody_sources": native["reader_custody_sources"],
        "inputs": artifacts, "outputs": {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*")) if path.is_file()},
        "model_calls": 0, "mathematical_recurrence_calls": 0, "native_VAE_calls": 0,
        "records": result["curated_records"], "exact_context_matches": result["exact_context_matches"]})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "math-run", "parent", "analysis", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    result = index(args.run, math_run=args.math_run, parent=args.parent, analysis=args.analysis, output=args.out)
    print(json.dumps({"records": result["curated_records"], "exact_context_matches": result["exact_context_matches"]}))
