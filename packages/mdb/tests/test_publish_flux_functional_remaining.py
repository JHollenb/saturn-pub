"""Synthetic metadata/PNG publisher checks; no model or recurrence execution."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest
from PIL import Image

P = runpy.run_path(str(Path(__file__).parents[1] / "qualification/publish_flux_functional_remaining.py"))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def metric(value=0.):
    return dict(rms=value, relative_rms=value, native_rms=1., predicted_rms=1. + value,
                signed_mean=-value, native_signed_mean=-1., predicted_signed_mean=-1. - value,
                bitwise_exact=value == 0, sample_count=8, fixture=True)


def pack(root):
    root.mkdir()
    write = lambda name, value: (root / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    provenance = dict(source_job=P["PRIOR_JOB"], source_request_sha256=P["PRIOR_REQUEST"],
        native_manifest_sha256=P["PRIOR_MANIFEST"], source_forecast_seal_sha256=P["PRIOR_SEAL"])
    native_request, native_job = digest("synthetic native request"), "job-synthetic-publisher-fixture"
    source = dict(job_id=native_job, request_sha256=native_request, worker_sha256=P["WORKER_SHA256"],
                  source_provenance=provenance, input_records={}, reference_records={})
    native_plan = dict(request_sha256=native_request, source_provenance=provenance, inputs={}, references={}, height=8, width=8)
    for key in ("transformer_config", "scheduler_config", "binding", "parameter_manifest", "retrieval_script", "vae_config", "vae_parameter_manifest"):
        name = key + ".json"
        write(name, dict(synthetic_fixture=key))
        native_plan[key], native_plan[key + "_sha256"] = name, P["sha"](root / name)
    native_summary = dict(job_id=native_job, request_sha256=native_request)
    before = []
    for step in (2, 3):
        before.extend([dict(step=step, phase="project", cursor=0), *[dict(step=step, phase="joint", cursor=i) for i in range(5)],
            *[dict(step=step, phase="single", cursor=i) for i in range(20)], dict(step=step, phase="readout", cursor=20), dict(step=step, phase="scheduler", cursor=20)])
    audit = dict(schema="mdb-functional-remaining-audit-v1", mechanics_status="authenticated-full-state-saved-evidence-audit",
        auditor_sha256=P["AUDITOR_SHA256"], worker_sha256=P["WORKER_SHA256"], native_request_sha256=native_request,
        job=native_job, source_model_identity="fixture-original-model", verification_adapter_identity="fixture-adapter",
        forecast_horizon_per_case=56, dependent_cases=4, source_provenance=provenance,
        counts={lane: dict(transitions=224) for lane in ("forecast", "native")},
        totals=dict(vae_calls=8, predicted_decoder_calls=4, native_reference_decoder_calls=4), decoder_counts=dict(fixture=True),
        transactions={name: dict(verified=True, phase="closed") for name in ("input", "latent-forecast", "forecast")},
        boundaries={}, signed_finite_responses=[], consumers={}, observed_rgb_finite_responses={},
        parameter_restoration=dict(fixture=True), first_covered_bitwise_mismatch=dict.fromkeys(P["CASES"]), known_control="step2/project known control")
    states, consumers, native_files = {}, {}, {}
    for case_index, case in enumerate(P["CASES"]):
        source["input_records"][case] = dict(cut=digest(case + "prior prediction"))
        source["reference_records"][case] = dict(cut=digest(case + "prior native"))
        native_plan["inputs"][case], native_plan["references"][case] = source["input_records"][case], source["reference_records"][case]
        states[case] = {}
        for lane, source_key in (("forecast", "input_records"), ("native", "reference_records")):
            parent = source[source_key][case]["cut"]
            states[case][lane] = []
            for number in range(56):
                identifier = digest(case + lane + str(number))
                states[case][lane].append(dict(cut=identifier, parent=parent, point=dict(logical_step=4 if number == 55 else before[number]["step"], phase="project" if number == 55 else "fixture", local_clock=0)))
                parent = identifier
        audit["boundaries"][case] = [dict(number=i, before=point, fields={field: metric(.25 if case == "HH" and i == 3 else 0.) for field in ("image", "text", "latent")}, primitive_witnesses={}) for i, point in enumerate(before)]
        audit["consumers"][case], consumers[case] = {}, {}
        for origin, prefix in P["ORIGINS"].items():
            directory = root / prefix / case
            directory.mkdir(parents=True)
            path = directory / "image.png"
            Image.new("RGB", (8, 8), (20 + case_index * 40, 45, 60 if origin == "predicted" else 65)).save(path)
            relative = path.relative_to(root).as_posix()
            lane = "forecast" if origin == "predicted" else "native"
            record = dict(origin=origin, input_cut=states[case][lane][-1]["cut"], independent_decoder_prediction=False,
                native_future_transitions=0 if origin == "predicted" else 224,
                png=dict(path=relative, sha256=P["sha"](path), bytes=path.stat().st_size))
            write(prefix + "/" + case + "/consumer.json", record)
            native_files[relative] = dict(sha256=record["png"]["sha256"], bytes=record["png"]["bytes"])
            consumers[case][origin] = record
            audit["consumers"][case][origin] = dict(record=record, pre_decoder_input_join="synthetic fixture only")
        audit["consumers"][case]["comparison"] = {field: metric(.125 if case == "HH" else 0.) for field in ("decoder_input", "actual_decoder_input", "decoded", "pixels")}
    audit["first_covered_bitwise_mismatch"]["HH"] = dict(number=3, before=before[3], fields=["image"], primitive_ports=[])
    for number, point in enumerate(before):
        audit["signed_finite_responses"].append(dict(number=number, before=point, responses={case: {field: metric(.25 if case == "HH" and number == 3 else 0.) for field in ("image", "text", "latent")} for case in P["CASES"][1:]}))
    audit["observed_rgb_finite_responses"] = {case: metric(.125 if case == "HH" else 0.) for case in P["CASES"][1:]}
    write("source-files.json", dict(files=native_files, job_id=native_job, request_sha256=native_request))
    audit["native_manifest_sha256"] = P["sha"](root / "source-files.json")
    plan = dict(native_manifest_sha256=audit["native_manifest_sha256"], native_request_sha256=native_request, source_job=native_job)
    request = dict(config=plan, source_hashes={"audit_flux_functional_remaining.py": P["AUDITOR_SHA256"]}, schema="synthetic-only")
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    summary = dict(source_job=native_job, job_id="job-synthetic-audit-fixture", request_sha256=request["request_sha256"])
    for name, value in (("audit.json", audit), ("request.json", request), ("plan.json", plan), ("summary.json", summary),
            ("source-source.json", source), ("source-plan.json", native_plan), ("source-summary.json", native_summary),
            ("state-index.json", states), ("consumer-index.json", consumers),
            ("external-evidence.json", dict(manifests=dict(native={k: v["sha256"] for k, v in native_files.items()})))):
        write(name, value)
    (root / "retrieve_evidence.py").write_text("# inert synthetic fixture\n")
    write("files.json", {p.relative_to(root).as_posix(): P["sha"](p) for p in root.rglob("*") if p.is_file()})
    return dict(expected_manifest=P["sha"](root / "files.json"), expected_request=native_request, expected_job=native_job)


def test_saved_publication_keeps_misses_signed_values_and_all_original_pngs(tmp_path):
    root, output = tmp_path / "pack", tmp_path / "publication"
    pins = pack(root)
    result = P["publish"](root, output, **pins)
    publication = P["read"](output / "publication.json")
    assert publication["transitions_per_case"] == 56 and len(publication["boundaries"]["HH"]) == 56
    assert publication["first_covered_bitwise_mismatch"]["HH"]["number"] == 3
    assert publication["signed_finite_responses"][3]["responses"]["HH"]["image"]["native_signed_mean"] == -1.
    assert publication["lineage"]["stitched_lineage_transitions"] == 84 and publication["lineage"]["single84_stage_prospective_forecast"] is False
    assert result["new_model_calls"] == result["new_numerical_recurrence"] == 0
    assert (output / "state-index.json").read_bytes() == (root / "state-index.json").read_bytes()
    for case in P["CASES"]:
        for origin, prefix in P["ORIGINS"].items():
            assert (output / "images" / origin / (case + ".png")).read_bytes() == (root / prefix / case / "image.png").read_bytes()
    for name in ("forecast-errors.png", "response-trajectories.png", "decoder-errors.png", "image-contact-sheet.png"):
        with Image.open(output / name) as image:
            assert image.width > 500 and image.height > 400


def test_reinventoried_png_cannot_replace_audited_consumer_bytes(tmp_path):
    root = tmp_path / "pack"
    pins = pack(root)
    path = root / "predicted-decoder/HH/image.png"
    Image.new("RGB", (8, 8), "white").save(path)
    manifest = P["read"](root / "files.json")
    manifest[path.relative_to(root).as_posix()] = P["sha"](path)
    (root / "files.json").write_text(json.dumps(manifest))
    pins["expected_manifest"] = P["sha"](root / "files.json")
    with pytest.raises(ValueError, match="PNG custody"):
        P["publish"](root, tmp_path / "out", **pins)
    assert not (tmp_path / "out").exists()


def test_external_job_and_collection_request_pins_are_enforced(tmp_path):
    root = tmp_path / "pack"
    pins = pack(root)
    wrong = copy.deepcopy(pins)
    wrong["expected_job"] = "job-other"
    with pytest.raises(ValueError, match="request/job"):
        P["authenticate"](root, **wrong)
    receipt = tmp_path / "collection.json"
    receipt.write_text(json.dumps(dict(files_manifest_sha256=pins["expected_manifest"], job_id="job-synthetic-audit-fixture", request_sha256="wrong")))
    with pytest.raises(ValueError, match="collection receipt"):
        P["publish"](root, tmp_path / "out", collection_receipt=receipt, **pins)
