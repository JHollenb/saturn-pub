"""Synthetic compact metadata/PNG publication checks; no model or raw arrays."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

PATH = Path(__file__).parents[1] / "qualification/publish_flux_functional_decoder.py"
P = runpy.run_path(str(PATH))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def metric(value=0.):
    return dict(rms=value, relative_rms=value, native_rms=1., predicted_rms=1. + value,
        signed_mean=-value, native_signed_mean=-1., predicted_signed_mean=-1. - value,
        maxabs=value, bitwise_exact=value == 0, sample_count=192, fixture=True)


def pack(root, *, recovered=False):
    root.mkdir()
    def write(name, value):
        (root / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    provenance = dict(source_job=P["PRIOR_JOB"], source_request_sha256=P["PRIOR_REQUEST"], native_manifest_sha256=P["PRIOR_MANIFEST"], source_forecast_seal_sha256=P["PRIOR_SEAL"])
    job, request = "job-synthetic-decoder-publisher", digest("synthetic producer request")
    helpers = {"flux_functional_decoder.py": P["DECODER_SHA256"], "flux_functional_vae.py": P["VAE_SHA256"]}
    native_plan = dict(request_sha256=request, inputs={case: dict(cut=cut) for case, cut in P["INPUT_CUTS"].items()},
        device="cuda", width=8, height=8, source_provenance=provenance, helper_source_sha256=helpers)
    for key in ("vae_config", "parameter_manifest", "retrieval_script"):
        name = key + ".json"
        write(name, dict(block_out_channels=[4, 4], synthetic_fixture=key))
        native_plan[key], native_plan[key + "_sha256"] = name, P["sha"](root / name)
    frontend = dict(input_state_device="cpu", unpack_device="cpu", bn_statistics_device="cpu", bn_sqrt_device="cpu", bn_multiply_device="cpu", bn_add_device="cpu", unpatch_device="cpu", decoder_argument_device="cuda", transfer_boundary="After CPU unpatch, before post_quant_conv; applies to forecast, native and diagnostic lanes.")
    contract = dict(frontend=frontend, replication=True, dtype="torch.bfloat16", inference_mode=True, autocast=False)
    seal = dict(seal_sha256=digest("synthetic original pretarget cohort seal"))
    totals = dict(functional_decode_calls=4, native_decode_calls=4, diagnostic_prefix_calls=2, forecast_module_entries=0, denoiser_transitions=0)
    counts = {lane: dict(synthetic_only=True) for lane in ("forecast", "native", "diagnostic")}
    source = dict(job_id=job, request_sha256=request, worker_sha256=P["WORKER_SHA256"], helper_source_sha256=helpers, source_provenance=provenance,
        input_records={case: dict(cut=cut) for case, cut in P["INPUT_CUTS"].items()}, forecast_seal=seal)
    native_summary = dict(job_id=job, request_sha256=request, forecast_seal=seal, totals=totals, counts=counts)
    names = P["macro_names"](dict(block_out_channels=[4, 4]))
    macros, native_inventory = {lane: {} for lane in ("forecast", "native")}, {}
    witnesses, diagnostics, first_micro = {}, {}, {}
    events = [dict(kind="forecast_case_completed", case=case) for case in P["CASES"]] + [dict(kind="cohort_globally_sealed", seal=seal, native_decode_calls=0)]
    for i, case in enumerate(P["CASES"]):
        for lane in ("forecast", "native"):
            parent = P["INPUT_CUTS"][case]
            macros[lane][case] = {}
            for name in names:
                alias = "post_quant_conv" if name == "decoder_input" else name
                descriptor = dict(sha256=digest(case + lane + alias), shape=[1, 8, 8, 3] if name == "pixels" else [1, 3, 8, 8], dtype="torch.uint8" if name == "pixels" else "torch.bfloat16", stride=[192, 24, 3, 1] if name == "pixels" else [192, 64, 8, 1])
                cut = digest(case + lane + name + "cut")
                macros[lane][case][name] = dict(cut=cut, parent=parent, name=name, descriptor=descriptor, original_source_cut=P["INPUT_CUTS"][case])
                parent = cut
            directory = root / (lane + "-images")
            directory.mkdir(exist_ok=True)
            path = directory / (case + ".png")
            Image.new("RGB", (8, 8), (20 + 30 * i, 45, 60 if lane == "forecast" else 65)).save(path)
            native_inventory[path.relative_to(root).as_posix()] = dict(sha256=P["sha"](path), bytes=path.stat().st_size)
            native_inventory[lane + "-micro/" + case + ".json"] = dict(sha256=digest(case + lane + "original micro file"), bytes=1234)
        events.extend([dict(kind="native_decode_entry", case=case, actual_before_post_quant_descriptor=macros["native"][case]["latent_unpatched"]["descriptor"]),
            dict(kind="native_decoder_entry", case=case, actual_after_post_quant_descriptor=macros["native"][case]["decoder_input"]["descriptor"]),
            dict(kind="native_case_completed", case=case, native_decode_calls=i + 1)])
        miss = case in ("HH", "HL")
        field = dict(full_descriptor_exact=not miss, layout_exact=True, witness_bitwise_exact=not miss, witness=metric(.25 if miss else 0.))
        witnesses[case] = [dict(ordinal=0, id="post_quant_conv.0", name="post_quant_conv", kind="conv2d", fields={"input.0": copy.deepcopy(field), "output": field})]
        first_micro[case] = dict(ordinal=0, id="post_quant_conv.0", field="output") if miss else None
        if miss:
            original = dict(descriptor=copy.deepcopy(macros["forecast"][case]["post_quant_conv"]["descriptor"]), indices=[0, 1], values=[.25, -.25], full_values_retained=False)
            status = "hash-matched-reconstruction" if case == "HH" else "replay-differs-from-original-sealed-hash"
            diagnostics[case] = dict(status=status, prefix_calls=1, original_sealed=original, replay_descriptor=copy.deepcopy(original["descriptor"]), unchanged_input=P["INPUT_CUTS"][case], helper_sha256=P["DECODER_SHA256"],
                native_full=macros["native"][case]["post_quant_conv"], retention_stage="post-observation prefix reconstruction, not pretarget full micro retention")
            if case == "HH":
                diagnostics[case]["forecast_full"] = macros["forecast"][case]["post_quant_conv"]
                diagnostics[case]["independently_reduced_full_pair"] = metric(.25)
            else:
                diagnostics[case]["replay_descriptor"]["sha256"] = digest("deliberate replay hash miss")
        else:
            diagnostics[case] = dict(status="not-needed", prefix_calls=0)
    for case in ("HH", "HL"):
        events.extend([dict(kind="diagnostic_prefix_started", case=case, native_decode_calls=4), dict(kind="diagnostic_prefix_completed", case=case, native_decode_calls=4, status=diagnostics[case]["status"])])
    audit = dict(schema="mdb-functional-decoder-audit-v1", mechanics_status="authenticated-full-macro-saved-evidence-audit", auditor_sha256=P["AUDITOR_SHA256"], worker_sha256=P["WORKER_SHA256"], new_model_calls=0, new_reference_forward_calls=0,
        job=job, native_request_sha256=request, source_model_identity="synthetic-only-model", source_provenance=provenance, input_records=source["input_records"], native_contract=contract, counts=counts, totals=totals,
        transactions={name: dict(verified=True, phase="closed", seal_sha256=seal["seal_sha256"]) for name in ("input", "forecast")}, macro_records=macros,
        macro_comparisons={case: {name: metric(.25 if case == "HH" and name == "decoded" else 0.) for name in names} for case in P["CASES"]},
        signed_pair_responses={case: {name: metric(.25 if case == "HH" and name == "decoded" else 0.) for name in names} for case in P["CASES"][1:]},
        primitive_witnesses=witnesses, first_macro_difference={case: "decoded" if case == "HH" else None for case in P["CASES"]}, first_micro_descriptor_difference=first_micro,
        diagnostic_reconstruction=diagnostics, diagnostic_instrument=dict(primary_evidence_preserved=True, module_entry_attempts=0),
        coverage=dict(cases=4, full_macros_per_case=len(names), full_micro_arrays=False), parameter_restoration=dict(synthetic_fixture_only=True))
    copied = {"plan.json": native_plan, "source.json": source, "summary.json": native_summary, "native-contract.json": contract, "comparisons.json": dict(synthetic_fixture=True), "signed-responses.json": audit["signed_pair_responses"], "forecast-macros.json": macros["forecast"], "native-macros.json": macros["native"]}
    if recovered:
        source["producer_status"] = native_summary["producer_status"] = "failed"
        contract["metadata_origin"] = "Derived after original reporting failure, not original runtime export"
    for name, value in copied.items():
        write("source-" + name, value)
    (root / "source-timeline.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    for name in [*copied, "timeline.jsonl"]:
        path = root / ("source-" + name)
        native_inventory[name] = dict(sha256=P["sha"](path), bytes=path.stat().st_size)
    if recovered:
        for name in ("source.json", "summary.json", "native-contract.json"):
            native_inventory.pop(name)
    write("source-files.json", dict(job_id=job, request_sha256=request, files=native_inventory, **(dict(original_job_state="failed") if recovered else {})))
    audit["native_manifest_sha256"] = P["sha"](root / "source-files.json")
    plan = dict(native_manifest_sha256=audit["native_manifest_sha256"], native_request_sha256=request, source_job=job)
    audit_request = dict(config=plan, source_hashes={"audit_flux_functional_decoder.py": P["AUDITOR_SHA256"]}, synthetic_fixture_only=True)
    if recovered:
        audit_request["source_hashes"]["recover_flux_functional_decoder.py"] = P["RECOVERY_SHA256"]
    audit_request["request_sha256"] = hashlib.sha256(json.dumps(audit_request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    summary = dict(source_job=job, job_id="job-synthetic-decoder-audit", request_sha256=audit_request["request_sha256"])
    if recovered:
        packet, metadata = root / "failed-source-authentication", root / "recovery-metadata"
        packet.mkdir()
        metadata.mkdir()
        executed = digest("synthetic executed failed payload")
        write("failed-source-authentication/job.json", dict(job_id=job, state="failed", result=dict(returncode=1, executed_payload=dict(sha256=executed)), payload_custody={name: dict(sha256=executed) for name in ("declared", "sealed", "executed")}))
        (packet / "worker.log").write_text("synthetic failure after saved decoder evidence\n")
        auth = dict(job_id=job, original_job_state="failed", request_sha256=request, job_sha256=P["sha"](packet / "job.json"), log_sha256=P["sha"](packet / "worker.log"), executed_payload_sha256=executed)
        write("failed-source-authentication/authentication.json", auth)
        derived = {}
        for name in ("source.json", "summary.json", "native-contract.json"):
            data = (root / ("source-" + name)).read_bytes()
            (metadata / name).write_bytes(data)
            derived[name] = P["sha"](metadata / name)
        recovery = dict(original_job_state="failed", final_resident_parameter_check="unestablished; original check after failing catalog was not executed",
            failed_authentication_sha256=P["sha"](packet / "authentication.json"), failed_job_sha256=auth["job_sha256"], failed_log_sha256=auth["log_sha256"],
            retained_manifest_sha256=audit["native_manifest_sha256"], derived_files=derived, failed_executed_payload_sha256=executed)
        write("recovery-metadata/recovery.json", recovery)
        audit.update(recovery=recovery, producer_status="failed", mechanics_status="authenticated-recovered-full-macro-saved-evidence-audit", final_resident_parameter_check=recovery["final_resident_parameter_check"], declared_profile_coverage="Declared source/config; final runtime export absent")
        summary["original_producer_status"] = "failed"
    for name, value in (("audit.json", audit), ("request.json", audit_request), ("plan.json", plan), ("summary.json", summary), ("macro-index.json", macros), ("external-evidence.json", dict(manifests=dict(native={name: row["sha256"] for name, row in native_inventory.items()})))):
        write(name, value)
    (root / "retrieve_evidence.py").write_text("# inert saved-only synthetic fixture\n")
    write("files.json", {p.relative_to(root).as_posix(): P["sha"](p) for p in root.rglob("*") if p.is_file()})
    return dict(expected_manifest=P["sha"](root / "files.json"), expected_request=request, expected_job=job)


def reinventory(root):
    manifest = {p.relative_to(root).as_posix(): P["sha"](p) for p in root.rglob("*") if p.is_file() and p.name != "files.json"}
    (root / "files.json").write_text(json.dumps(manifest))
    return P["sha"](root / "files.json")


def test_publication_keeps_all_misses_signed_metrics_replay_proofs_and_original_pngs(tmp_path):
    root, output = tmp_path / "pack", tmp_path / "publication"
    pins = pack(root)
    receipt = P["publish"](root, output, **pins)
    publication = P["read"](output / "publication.json")
    assert len(publication["macro_retention_clock"]) == 10
    assert publication["macro_comparisons"]["HH"]["decoded"]["rms"] == .25
    assert publication["signed_pair_responses"]["HH"]["decoded"]["native_signed_mean"] == -1.
    assert publication["first_macro_difference"]["HH"] == "decoded"
    assert len(publication["all_micro_misses"]["HH"]) == len(publication["all_micro_misses"]["HL"]) == 1
    assert publication["diagnostic_reconstruction"]["HH"]["status"] == "hash-matched-reconstruction"
    assert publication["diagnostic_reconstruction"]["HL"]["status"] == "replay-differs-from-original-sealed-hash"
    assert "forecast_full" not in publication["diagnostic_reconstruction"]["HL"]
    assert publication["lineage"]["prior_denoiser_transitions"] == 84 and publication["lineage"]["additional_denoiser_transitions"] == 0
    assert publication["lineage"]["macro_indices_are_physical_transitions"] is False
    assert publication["native_contract"]["frontend"]["bn_multiply_device"] == "cpu"
    assert receipt["new_model_calls"] == receipt["new_numerical_recurrence"] == 0
    for lane in ("forecast", "native"):
        for case in P["CASES"]:
            assert (output / "images" / lane / (case + ".png")).read_bytes() == (root / (lane + "-images") / (case + ".png")).read_bytes()
    for name in ("macro-errors.png", "signed-responses.png", "rgb-pixel-errors.png", "image-contact-sheet.png"):
        with Image.open(output / name) as image:
            assert image.width > 500 and image.height > 400


def test_reinventoried_png_cannot_replace_original_native_bytes(tmp_path):
    root = tmp_path / "pack"
    pins = pack(root)
    Image.new("RGB", (8, 8), "white").save(root / "forecast-images/HH.png")
    pins["expected_manifest"] = reinventory(root)
    with pytest.raises(ValueError, match="original PNG custody"):
        P["publish"](root, tmp_path / "out", **pins)
    assert not (tmp_path / "out").exists()


def test_reconstruction_hash_and_external_identity_pins(tmp_path):
    root = tmp_path / "pack"
    pins = pack(root)
    with pytest.raises(ValueError, match="native request/job"):
        P["authenticate"](root, **{**pins, "expected_job": "job-other"})
    audit = P["read"](root / "audit.json")
    audit["diagnostic_reconstruction"]["HH"]["replay_descriptor"]["sha256"] = digest("wrong reconstructed tensor")
    (root / "audit.json").write_text(json.dumps(audit))
    pins["expected_manifest"] = reinventory(root)
    with pytest.raises(ValueError, match="sealed hash proof"):
        P["publish"](root, tmp_path / "out", **pins)
    assert not (tmp_path / "out").exists()


def test_saved_only_cli_and_collection_receipt(tmp_path):
    root, output = tmp_path / "pack", tmp_path / "publication"
    pins = pack(root)
    receipt = tmp_path / "collection.json"
    summary = P["read"](root / "summary.json")
    receipt.write_text(json.dumps(dict(files_manifest_sha256=pins["expected_manifest"], job_id=summary["job_id"], request_sha256=summary["request_sha256"])))
    args = [sys.executable, "-I", str(PATH), "--pack", str(root), "--out", str(output), "--collection-receipt", str(receipt)]
    for key, value in pins.items():
        args.extend(["--" + key.replace("_", "-"), value])
    run = subprocess.run(args, cwd=tmp_path, env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")}, text=True, capture_output=True, check=True)
    assert json.loads(run.stdout)["new_model_calls"] == 0
    assert P["read"](output / "publication-receipt.json")["collection_receipt"]["sha256"] == P["sha"](receipt)


def test_recovered_failed_source_is_separate_from_successful_saved_only_audit(tmp_path):
    root, output = tmp_path / "pack", tmp_path / "publication"
    pins = pack(root, recovered=True)
    P["publish"](root, output, **pins)
    publication = P["read"](output / "publication.json")
    assert publication["producer_status"] == "failed" and publication["saved_only_audit_status"] == "successful recovery audit"
    assert publication["final_resident_parameter_check"].startswith("unestablished")
    assert P["read"](output / "failed-source-authentication/job.json")["state"] == "failed"
    assert "failed after all primary calls" in publication["scope"]
    assert publication["lineage"]["additional_denoiser_transitions"] == 0
