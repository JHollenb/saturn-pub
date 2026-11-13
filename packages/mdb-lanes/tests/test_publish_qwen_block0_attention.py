"""Saved-only block-0 attention publisher tests; no arrays, models, or CUDA."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
P = runpy.run_path(str(HERE / "qualification/publish_qwen_block0_attention.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def metric(value, *, signed=0.0):
    return {
        "shape": [1, 2],
        "elements": 2,
        "orientation": "left-minus-right",
        "rms": value,
        "reference_rms": 2.0,
        "relative_rms": value / 2 if value else 0.0,
        "signed_mean": signed,
        "cosine": 1.0 if value == 0 else 0.5,
        "changed_values": 0 if value == 0 else 1,
        "bitwise_exact": value == 0,
    }


def sdpa_rows():
    rows = []
    labels = ["functional.short", "functional.long", *P["ARMS"], "native.short", "native.long"]
    for ordinal, label in enumerate(labels):
        rows.append({
            "ordinal": ordinal,
            "label": label,
            "is_causal": ordinal in {1, 3, 4, 6},
            "enable_gqa": ordinal in {1, 4, 6},
            "scale": 0.5,
            "dropout_p": 0.0,
            "mask": None if ordinal in {1, 3, 4, 6} else {"shape": [1, 1, 2, 2]},
            "query": {"shape": [1, 2, 2, 2]},
            "key": {"shape": [1, 2, 2, 2]},
            "value": {"shape": [1, 2, 2, 2]},
            "output": {"shape": [1, 2, 2, 2]},
            "profiler": {"status": "observed", "events": ["aten::scaled_dot_product_attention", f"kernel-{ordinal}"]},
        })
    return rows


def make_pack(root):
    root = Path(root)
    root.mkdir()
    native_job = "native-block0-job"
    native_request = "n" * 64
    source_plan = {
        "schema": "mdb-qwen-block0-attention-plan-v1",
        "qwen_config": "qwen-config.json",
        "qwen_config_sha256": "",
        "qwen_manifest": "qwen-parameter-artifacts.json",
        "qwen_manifest_sha256": "",
        "retrieval_script": "retrieve_qwen_weights.py",
        "retrieval_script_sha256": P["RETRIEVER_SHA256"],
        "request_sha256": native_request,
    }
    (root / "qwen-config.json").write_text("{}\n")
    (root / "qwen-parameter-artifacts.json").write_text("{}\n")
    (root / "retrieve_qwen_weights.py").write_text("# synthetic metadata-only retriever\n")
    retriever_hash = P["sha"](root / "retrieve_qwen_weights.py")
    source_plan["retrieval_script_sha256"] = retriever_hash
    source_plan["qwen_config_sha256"] = P["sha"](root / "qwen-config.json")
    source_plan["qwen_manifest_sha256"] = P["sha"](root / "qwen-parameter-artifacts.json")
    source = {
        "schema": "mdb-qwen-block0-attention-source-v1",
        "job_id": native_job,
        "request_sha256": native_request,
        "worker_sha256": P["WORKER_SHA256"],
        "helper_sources": P["HELPER_PINS"],
        "source_provenance": {},
    }
    plan = {"native_request_sha256": native_request, "native_manifest_sha256": "m" * 64,
            "source_job": native_job, "shared_prefix_rows": 2, "fixed_bank_case": "long", "schema": "fixture-plan"}
    request = {"schema": "mdb-saved-flux-audit-request-v1", "config": plan,
               "source_hashes": {"audit_qwen_block0_attention.py": P["AUDITOR_SHA256"]}}
    request["request_sha256"] = P["request_digest"]({**request, "request_sha256": ""})
    # The native request is carried by the saved audit; this is the compact CPU request.
    pair = {lane: {port: metric(0.0 if port == "H0" else 0.1) for port in P["PORTS"]} for lane in ("functional", "native")}
    matched = {case: {port: metric(0.01) for port in P["PORTS"]} for case in P["CASES"]}
    replay = {case: {lane: {port: metric(0.02) for port in ("H0", "H1")} for lane in ("forecast", "native")} for case in P["CASES"]}
    triangle = {}
    for index, port in enumerate(P["TRIANGLE_PORTS"]):
        triangle[port] = {
            arm: {"full": metric(0.1 + index / 100), "shared_prefix": metric(0.01 + index / 100)}
            for arm in ("B-minus-A", "C-minus-B", "C-minus-A")
        }
        triangle[port]["coordinate_identity"] = True
        triangle[port]["prefix_telescope"] = {
            "first_energy": 0.1, "second_energy": 0.2, "sum_energy": 0.3,
            "signed_cross_term": 0.0, "energy_identity_residual": 0.0,
        }
    summary = {"schema": "mdb-qwen-block0-attention-compact-v1", "job_id": "audit-cpu-job",
               "source_job": native_job, "request_sha256": request["request_sha256"], "model_calls": 0, "records": 2}
    audit = {
        "schema": "mdb-qwen-block0-attention-saved-audit-v1", "job_id": native_job, "request_sha256": native_request,
        "raw_manifest_sha256": "", "source": source, "mechanics_status": "authenticated-saved-matched-operands",
        "terminal_status": "unassessed", "new_model_calls": 0, "new_tokenizer_calls": 0,
        "new_cuda_calls": 0, "new_attention_calls": 0,
        "actual": {"native_block_calls": 2, "native_rotary_calls": 2, "sdpa_calls": 7, "source_parameter_reads": 12},
        "sdpa": sdpa_rows(), "pair_comparisons": pair, "matched_functional_vs_native": matched,
        "historical_replay": replay, "original_cross_case": {"forecast": {}, "native": {}},
        "triangle_accounts": triangle, "port_provenance": {"native": {}, "functional": {}, "triangle": "fixture"},
        "capture_layout": {"short": {}, "long": {}}, "source_masks": {"short": {}, "long": {}},
        "hypotheses_observations": {"first_retained_prefix_difference_in_declared_port_order": "q_raw"},
        "per_token_head": {},
    }
    write(root / "source-plan.json", source_plan)
    write(root / "source-source.json", source)
    write(root / "source-request.json", {"request_sha256": native_request})
    write(root / "source-summary.json", {"job_id": native_job, "request_sha256": native_request, "trend_status": "observed-uninterpreted"})
    source_files = {"schema": "mdb-image-tail-files-v1", "job_id": native_job, "request_sha256": native_request, "files": {}}
    write(root / "source-files.json", source_files)
    audit["raw_manifest_sha256"] = P["sha"](root / "source-files.json")
    plan["native_manifest_sha256"] = audit["raw_manifest_sha256"]
    request["config"] = plan
    request["request_sha256"] = P["request_digest"](request)
    summary["request_sha256"] = request["request_sha256"]
    write(root / "audit.json", audit)
    write(root / "request.json", request)
    write(root / "plan.json", plan)
    write(root / "summary.json", summary)
    write(root / "per-token-head.json", {"fixture": True})
    write(root / "catalog/provenance.json", {"records": 0, "queries": [], "automatic_family_induction": False})
    write(root / "external-evidence.json", {"manifests": {"native": {}}})
    write(root / "source-runtime-contract.json", {"device": "cuda", "profiling": "passive"})
    manifest = {path.relative_to(root).as_posix(): P["sha"](path) for path in root.rglob("*") if path.is_file()}
    write(root / "files.json", manifest)
    compact_manifest = P["sha"](root / "files.json")
    collection = root.parent / "collection.json"
    write(collection, {"files_manifest_sha256": compact_manifest, "raw_tensor_arrays_collected": False})
    qualification = root.parent / "qualification.json"
    write(qualification, {"schema": "fixture", "status": "passed", "sources": {"audit_qwen_block0_attention.py": P["AUDITOR_SHA256"]}})
    return root, compact_manifest, native_request, native_job, qualification, collection, retriever_hash


def test_publisher_authenticates_and_keeps_large_record_single_copy(tmp_path):
    root, manifest, request, job, qualification, collection, retriever_hash = make_pack(tmp_path / "pack")
    P["publish"].__globals__["RETRIEVER_SHA256"] = retriever_hash
    result = P["publish"](root, tmp_path / "publication", expected_manifest=manifest, expected_request=request,
                           expected_job=job, qualification=qualification, collection_receipt=collection)
    output = tmp_path / "publication"
    publication = json.loads((output / "publication.json").read_text())
    assert result["new_model_calls"] == result["new_cuda_calls"] == 0
    assert publication["claims"]["independent_attention_kernel"] is False
    assert publication["cross_case"]["first_retained_prefix_difference"] == "q_raw"
    assert publication["sdpa_scope"]["kernel_claim"].startswith("No independent")
    assert (output / "compact-evidence/per-token-head.json").is_file()
    assert not (output / "audit-data.json").exists()
    assert len(list(output.glob("*.png"))) == 5
    assert len(publication["sdpa"]) == 7


def test_publisher_refuses_wrong_lineage_and_raw_member(tmp_path):
    root, manifest, request, job, qualification, collection, retriever_hash = make_pack(tmp_path / "pack")
    P["publish"].__globals__["RETRIEVER_SHA256"] = retriever_hash
    with pytest.raises(ValueError, match="expected native job/request"):
        P["publish"](root, tmp_path / "wrong", expected_manifest=manifest, expected_request="x" * 64,
                      expected_job=job, qualification=qualification, collection_receipt=collection)
    raw = root / "bad.npy"
    raw.write_bytes(b"raw")
    entries = json.loads((root / "files.json").read_text())
    entries[raw.name] = hashlib.sha256(raw.read_bytes()).hexdigest()
    write(root / "files.json", entries)
    with pytest.raises(ValueError, match="raw or malformed"):
        P["publish"](root, tmp_path / "raw", expected_manifest=P["sha"](root / "files.json"), expected_request=request,
                      expected_job=job, qualification=qualification, collection_receipt=collection)


def test_dispatch_excerpt_is_short_but_keeps_observed_runtime_identity():
    assert P["_dispatch_excerpt"](["fmha_cutlassF_bf16_aligned_64x128_rf_sm80(...) "]) == "efficient / CUTLASS"
    assert P["_dispatch_excerpt"](["void pytorch_flash::flash_fwd_kernel<...>(...)"]) == "Flash / flash_fwd"
    assert P["_dispatch_excerpt"]([]) == "unavailable /\nunsupported"


def test_unavailable_profiler_is_a_diagnostic_and_does_not_block_publication(tmp_path):
    root, manifest, request, job, qualification, collection, retriever_hash = make_pack(tmp_path / "pack")
    P["publish"].__globals__["RETRIEVER_SHA256"] = retriever_hash
    audit_path = root / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["sdpa"][0]["profiler"] = {"status": "unavailable", "events": None, "reason": "fixture fallback"}
    write(audit_path, audit)
    entries = {
        path.relative_to(root).as_posix(): P["sha"](path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "files.json"
    }
    write(root / "files.json", entries)
    compact_manifest = P["sha"](root / "files.json")
    result = P["publish"](root, tmp_path / "publication", expected_manifest=compact_manifest,
                           expected_request=request, expected_job=job, qualification=qualification)
    assert result["new_model_calls"] == 0
    publication = json.loads((tmp_path / "publication/publication.json").read_text())
    assert publication["sdpa"][0]["profiler"]["status"] == "unavailable"
    assert publication["sdpa_scope"]["kernel_claim"].startswith("No independent")
