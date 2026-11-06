"""Saved-only paired publisher fixtures; no arrays, models, or native calls."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1] / "qualification"
P = runpy.run_path(str(HERE / "publish_flux_raw_mask_pair.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def metric(value, *, signed=0.0, sample_count=4, relative=True):
    return {
        "shape": [2, 2],
        "sample_count": sample_count,
        "orientation": "functional-minus-native",
        "rms": value,
        "relative_rms": None if not relative else value / 2,
        "native_rms": 2.0,
        "predicted_rms": 2.0 + value,
        "signed_mean": signed,
        "native_signed_mean": 0.0,
        "predicted_signed_mean": signed,
        "dot": 1.0,
        "cosine": 1.0,
        "changed_values": 0 if value == 0 else 4,
        "bitwise_exact": value == 0,
        "maxabs": value,
        "p95": value,
        "p99": value,
    }


def decoder_account(value):
    row = metric(value, signed=-value / 2)
    terms = {
        "functional_minus_same_forecast_latent_native_VAE": metric(value / 2),
        "same_forecast_latent_native_VAE_minus_native_latent_native_VAE": metric(value / 3),
        "whole_functional_minus_native": row,
    }
    return {
        "terms": terms,
        "common_native_rms": 2.0,
        "common_native_relative_rms": {"decoder_component": value / 2, "inherited_latent_consumer_effect": value / 3, "whole": value},
        "squared_norms": {"decoder_component": value, "inherited_latent_consumer_effect": value / 2, "whole": value * 2},
        "cross_dot": value / 4,
        "signed_cross_term": value / 2,
        "cross_cosine": 0.25,
        "coordinate_identity_exact": True,
        "energy_identity_residual": 0.0,
    }


def make_pack(root):
    root = Path(root)
    root.mkdir()
    native_request = "n" * 64
    native_job = "native-pair-job"
    audit_request = "a" * 64
    runtime = {"schema": "runtime-contract-v1", "dtype": "torch.bfloat16", "selected_kernel": "unprofiled"}
    write(root / "source-runtime-contract.json", runtime)
    source = {
        "schema": "mdb-flux-raw-mask-pair-source-v1",
        "job_id": native_job,
        "request_sha256": native_request,
        "worker_sha256": P["PRODUCER_WORKER_SHA256"],
        "helper_source_sha256": {"flux_raw_to_image.py": P["PRODUCER_PRIOR_SHA256"]},
        "source_files": {},
        "tokenizer_files": {},
        "runtime_contract_sha256": P["sha"](root / "source-runtime-contract.json"),
        "forecast_seal": {"seal_sha256": "s" * 64},
    }
    plan = {"native_request_sha256": native_request, "native_manifest_sha256": "m" * 64, "source_job": native_job, "scope": "fixture"}
    request = {
        "schema": "mdb-saved-flux-audit-request-v1",
        "config": plan,
        "source_hashes": {
            "audit_flux_raw_mask_pair.py": P["AUDITOR_SHA256"],
            **P["AUDIT_SOURCE_PINS"],
        },
    }
    request["request_sha256"] = P["request_digest"](request)
    audit_request = request["request_sha256"]
    summary = {"schema": "mdb-raw-mask-pair-compact-v1", "job_id": "audit-pair-job", "source_job": native_job,
               "request_sha256": audit_request, "model_calls": 0, "mechanics_status": "verified-saved-pair"}
    cases = {}
    for case, count, padded in (("short", 480, 32), ("long", 650, 0)):
        parser = {
            "prompt_utf8_sha256": (case[0] * 64),
            "uncropped_token_count": count,
            "producer_uncropped_token_count_recorded": 2 if case == "long" else 1,
            "producer_uncropped_count_matches_actual": False,
            "mask": {"attended_rows": 512 - padded, "padded_rows": padded, "regime": "partially-padded" if padded else "all-attended",
                     "expected_regime": "partially-padded" if padded else "all-attended", "expected_regime_observed": True},
            "dispatch": {"backend": "sdpa", "gqa_policy": "repeat_kv" if padded else "enable_gqa", "source": "fixture"},
        }
        qwen_fields = {}
        for name, value in (("hidden.0", 0.0), ("hidden.1", 0.2), ("conditioning", 0.4)):
            qwen_fields[name] = {"full": metric(value, signed=value / 10),
                                "attended_query_rows": {"rows": 512 - padded, "comparison": metric(value / 2) if padded != 512 else None},
                                "padded_query_rows": {"rows": padded, "comparison": None if not padded else metric(value / 3)}}
        boundaries = [{"number": index, "clock": {"step": index, "phase": "single", "cursor": 0},
                       "fields": {"latent": metric(0.1 * (index + 1), signed=(-1) ** index * 0.01)}} for index in range(3)]
        cases[case] = {
            "parser": parser,
            "qwen": {"fields": qwen_fields},
            "denoiser": {"boundaries": boundaries},
            "decoder": {"fields": {"decoded": decoder_account(0.0 if case == "short" else 0.2), "pixels": decoder_account(0.1)}},
        }
    # Case-facing rows use hidden.N; signed reductions use the producer's
    # compact numeric keys N.  Keep the fixture source-faithful so the
    # publisher tests the explicit translation rather than a guessed schema.
    signed_qwen = {name: {"full": metric(0.1, signed=-0.01)} for name in ("0", "1", "conditioning")}
    signed_denoiser = [{"number": -1, "clock": {"step": -1}, "fields": {"latent": {"full": metric(0.05, signed=-0.01)}}}]
    signed_denoiser.extend({"number": index, "clock": {"step": index}, "fields": {"latent": {"full": metric(0.1, signed=0.02)}}} for index in range(3))
    signed = {
        "qwen": signed_qwen,
        "denoiser": signed_denoiser,
        "decoder": {"decoded": {"full": metric(0.0), "same_latent_conditional_response_error": metric(0.2), "inherited_native_consumer_response_error": metric(0.1)},
                     "pixels": {"full": metric(0.1), "same_latent_conditional_response_error": metric(0.2), "inherited_native_consumer_response_error": metric(0.1)}},
    }
    audit = {
        "schema": "mdb-raw-mask-pair-saved-audit-v1", "job_id": native_job, "request_sha256": native_request,
        "worker_sha256": P["PRODUCER_WORKER_SHA256"], "source": source, "config": {"schema": "mdb-flux-raw-mask-pair-plan-v1", "seed": 277, "steps": 4, "height": 512, "width": 512, "dtype": "bfloat16", "attention_dispatch": {"backend": "sdpa"}, "max_length": 512},
        "runtime": runtime, "cases": cases, "signed_responses": signed,
        "coverage": {"original_pngs": 6, "retained_rows": 512}, "counts": {"functional_transitions": 224, "native_transitions": 224},
        "native_contract": {}, "chronology": {}, "mechanics_status": "authenticated-saved-paired-program", "trend_status": "observed", "terminal_status": "unassessed",
        "new_neural_forward_calls": 0, "new_learned_recurrence_calls": 0, "scope": "fixture",
        "sdpa": {lane: {case: {"calls": 2, "all_argument_profiles_match_declared_regime": True, "selected_kernel": "unprofiled"} for case in P["CASES"]} for lane in ("forecast", "native")},
    }
    write(root / "audit.json", audit)
    write(root / "request.json", request)
    write(root / "plan.json", plan)
    write(root / "summary.json", summary)
    write(root / "source-source.json", source)
    write(root / "source-plan.json", plan)
    write(root / "source-request.json", {"request_sha256": native_request})
    write(root / "source-summary.json", {"job_id": native_job, "request_sha256": native_request})
    write(root / "source-files.json", {"schema": "mdb-image-tail-files-v1", "job_id": native_job, "request_sha256": native_request, "files": {}})
    write(root / "external-evidence.json", {"manifests": {"native": {}}})
    (root / "retrieve_evidence.py").write_text("# fixture\n")
    for index, (case, lane) in enumerate(P["PNG_NAMES"]):
        Image.new("RGB", (8, 8), (index * 20, 20, 120)).save(root / P["PNG_NAMES"][(case, lane)], format="PNG")
    entries = {path.relative_to(root).as_posix(): P["sha"](path) for path in root.rglob("*") if path.is_file()}
    (root / "files.json").write_text(json.dumps(entries, sort_keys=True, indent=2) + "\n")
    manifest_sha = P["sha"](root / "files.json")
    collection = root.parent / "collection.json"
    write(collection, {"schema": "mdb-compact-collection-v1", "files_manifest_sha256": manifest_sha,
                       "summary_sha256": P["sha"](root / "summary.json"), "job_id": "audit-pair-job", "request_sha256": audit_request,
                       "source_job_id": native_job, "source_request_sha256": native_request})
    return root, collection, manifest_sha, native_request, native_job


def test_pair_publisher_authenticates_and_publishes_all_views(tmp_path):
    root, collection, manifest, request, job = make_pack(tmp_path / "pack")
    qualification = tmp_path / "audit-qualification.json"
    write(qualification, {"schema": "mdb-reader-abi-qualification-v2", "returncode": 0,
                          "sources": {"saturn-pub/packages/mdb/qualification/audit_flux_raw_mask_pair.py": P["AUDITOR_SHA256"]}})
    result = P["publish"](root, tmp_path / "publication", expected_manifest=manifest, expected_request=request,
                           expected_job=job, collection_receipt=collection, qualification=qualification)
    output = tmp_path / "publication"
    assert result["new_model_calls"] == 0
    assert len(list((output / "images").glob("*.png"))) == 6
    assert len(list(output.glob("*.png"))) == 6
    assert len(list(output.glob("*-traces.png"))) == 2
    assert (output / "signed-response-trajectories.png").is_file()
    assert (output / "endpoint-decomposition.png").is_file()
    assert (output / "mask-sdpa-context.json").is_file()
    publication = json.loads((output / "publication.json").read_text())
    assert publication["claims"]["isolated_padding_causal_claim"] is False
    assert publication["mask_sdpa"][0]["producer_uncropped_token_count_recorded"] == 1
    assert publication["signed_response_totals"]["qwen"][0]["sample_count"] == 4
    assert publication["signed_response_totals"]["qwen"][0]["field"] == "hidden.0"
    assert publication["signed_response_totals"]["qwen"][0]["signed_field"] == "0"
    assert publication["signed_response_totals"]["denoiser"][0]["number"] == -1
    assert publication["endpoint_decomposition"]["short"]["decoded"]["squared_norms"]["whole"] == 0


def test_pair_publisher_refuses_wrong_lineage_or_raw_member(tmp_path):
    root, collection, manifest, request, job = make_pack(tmp_path / "pack")
    with pytest.raises(ValueError, match="expected native request"):
        P["publish"](root, tmp_path / "wrong-request", expected_manifest=manifest, expected_request="x" * 64, expected_job=job, collection_receipt=collection)
    bad = root / "evil.npy"
    bad.write_bytes(b"raw")
    entries = json.loads((root / "files.json").read_text())
    entries[bad.name] = hashlib.sha256(bad.read_bytes()).hexdigest()
    (root / "files.json").write_text(json.dumps(entries, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="compact manifest differs|raw or malformed"):
        P["publish"](root, tmp_path / "raw-member", expected_manifest=P["sha"](root / "files.json"), expected_request=request, expected_job=job, collection_receipt=collection)


def test_aggregate_preserves_zero_and_signed_totals():
    assert P["_aggregate"]([metric(0.0, signed=0.0)]) == {
        "field_count": 1, "sample_count": 4, "rms": 0.0, "signed_total": 0.0, "signed_mean": 0.0, "relative_rms": 0.0,
    }
    result = P["_aggregate"]([metric(1.0, signed=-0.5, sample_count=2), metric(2.0, signed=0.25, sample_count=2)])
    assert result["signed_total"] == -0.5
    assert result["sample_count"] == 4
