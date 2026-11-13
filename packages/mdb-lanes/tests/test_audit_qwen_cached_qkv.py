"""Saved actual tiny panel, exact public schema, hostile custody; no models."""
from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
A = runpy.run_path(str(HERE / "audit_qwen_cached_qkv.py"))
SAVED = HERE / "saved-fixture"
QCONFIG = dict(head_dim=4, num_attention_heads=4, num_key_value_heads=2)


def source_values():
    records = A["read"](SAVED / "source-frontier.json")["source_arrays"]
    return {name: A["base"]()["load_array"](HERE / "synthetic-parent", record) for name, record in records.items()}


def accounts(root=SAVED):
    return A["numerical_accounts"](root, source_values(), A["read"](root / "protocol.json"))


def tx(root=SAVED, inventory=None):
    source = A["read"](root / "source.json")
    return A["transaction"](root, A["read"](root / "protocol.json"), A["read"](root / "summary.json"), inventory or A["read"](root / "files.json"),
        dict(source["helper_source_sha256"], qwen_native=A["row"]()["NATIVE_SHA256"], sdpa_native=A["row"]()["SDPA_SHA256"]), 64 << 20)


def test_actual_saved_writer_dynamic_windows_three_channels_and_max_elements(monkeypatch):
    import mdb_lanes.equation_comparison as api

    original, budgets = api.compare_equations, []
    def compare(*args, **kwargs):
        budgets.append(kwargs["max_elements"])
        return original(*args, **kwargs)
    monkeypatch.setattr(api, "compare_equations", compare)
    report, equations = accounts()
    protocol = A["read"](SAVED / "protocol.json")
    assert len(budgets) == 6 and set(budgets) == {272}
    assert set(report["arms"]) == {arm["id"] for arm in protocol["arms"]}
    for arm in protocol["arms"]:
        result = report["arms"][arm["id"]]
        assert result["support"]["closed_rows"] == arm["structural_support_closed_rows"]
        assert len(equations[arm["id"]]) == len(result["positioned_QKV"]) == 3
        assert result["coverage"]["overlapping_endpoint_windows"] == 12*len(arm["windows"])
        for fields in result["channels"].values():
            assert set(fields) == set(A["PORTS"])
            assert all(len(field["response_error"]["windows"]) == len(arm["windows"]) for field in fields.values())
    assert "torch" not in sys.modules and "transformers" not in sys.modules


def test_actual_dense_restoration_and_all_seven_live_QKV_head_context_joins():
    refs = A["read"](SAVED / "source-frontier.json")["source_arrays"]
    layouts = A["layout_accounts"](SAVED, refs)
    assert len(layouts["arrays"]) == 13
    assert all(layouts["arrays"][name]["loaded_logical"]["stride"] != layouts["arrays"][name]["restored"]["stride"] for name in A["QKV"])
    census = A["operand_accounts"](SAVED, A["read"](SAVED / "protocol.json"), A["read"](SAVED / "summary.json"), QCONFIG)
    assert len(census["actual_SDPA_calls"]) == 7 and len(census["QKV_layouts"]) == 21
    assert len(census["native_entries"]) == 3 and census["observed"]["native_module_entries"] == 45


def test_actual_global_transaction_full_inventory_and_future_exclusion(tmp_path):
    from mdb_lanes.forecast_transaction import ForecastTransactionError

    assert tx()["verified"]["events"] == 7 and tx()["native_targets_at_seal"] == 0
    wrapper = A["row"]()["wrapper"](SAVED, A["sha"](SAVED / "files.json"), "1"*64)
    assert len(wrapper["files"]) == 137
    root = tmp_path / "raw"; shutil.copytree(SAVED, root)
    inventory = A["read"](root / "files.json")
    inventory["files"]["future-unsealed.npy"] = dict(bytes=1, sha256="0"*64)
    with pytest.raises(ValueError, match="complete pretarget"):
        tx(root, inventory)
    (root / "transaction/seal.json").write_text("{}")
    with pytest.raises(ForecastTransactionError, match="global seal differs"):
        tx(root)


def test_hostile_actual_operand_layout_and_signed_commitment_refuse(tmp_path):
    root = tmp_path / "raw"; shutil.copytree(SAVED, root)
    path = root / "sdpa.json"; value = A["read"](path)
    value["calls"][0]["operands"][0]["stride"] = [1, 1, 1, 1]
    path.unlink(); A["save"](path, value)
    with pytest.raises(ValueError, match="actual consumed operand"):
        A["operand_accounts"](root, A["read"](root / "protocol.json"), A["read"](root / "summary.json"), QCONFIG)
    path = root / "forecast-responses.json"; value = A["read"](path)
    first = next(iter(value["arms"].values()))
    first["cached"]["head"]["sha256"] = "0"*64
    path.unlink(); A["save"](path, value)
    with pytest.raises(ValueError, match="full pretarget signed response"):
        accounts(root)


def test_pointwise_window_misses_and_numeric_bit_signedzero_are_retained():
    right = np.zeros((1, 8, 2), np.float32); left = right.copy()
    left[0, 7, 0] = .5; left[0, 0, 0] = -0.
    result = A["windowed"](left, right, [[0, 3], [3, 5], [5, 8]])
    assert result["whole"]["changed_values"] == 1 and result["whole"]["signed_zero"]["defects"] == 1
    assert result["windows"][0]["comparison"]["rms"] == 0 and not result["windows"][0]["comparison"]["bitwise_exact"]
    assert result["windows"][-1]["comparison"]["changed_values"] == 1
    with pytest.raises(ValueError, match="overlapping or nondense"):
        A["dense_layout"]([1, 4, 8, 4], [128, 1, 1, 1])


def test_actual_parent_serialized_metadata_source13_and_layout_descriptors():
    source, protocol = A["read"](SAVED / "source.json"), A["read"](SAVED / "protocol.json")
    metadata = {name: Path(record["path"]) for name, record in source["metadata"].items()}
    values, parent = A["inherited_accounts"](SAVED, dict(input_root=protocol["parent_raw_root"]), metadata, protocol)
    assert set(values) == set(source_values()) and parent["arrays"] == 13
    assert all(A["same"](values[name], source_values()[name]) for name in values)


def test_production_context_builder_real_installed_atlas_schema_and_retrieval_aliases(tmp_path):
    from mdb_lanes.computation_atlas import ComputationAtlas, ComputationAtlasError

    protocol, summary = A["read"](SAVED / "protocol.json"), A["read"](SAVED / "summary.json")
    report, equations = accounts(); report["job_id"] = "tiny-saved-only"
    report["census"] = A["operand_accounts"](SAVED, protocol, summary, QCONFIG)
    contexts = A["catalog_contexts"](dict(qwen_parameter_manifest_sha256="1"*64, attention_dispatch=dict(backend="sdpa", gqa_policy="enable_gqa")),
        dict(source_hashes={"synthetic": "2"*64}), dict(runtime=A["read"](SAVED / "runtime-contract.json")), protocol, report)
    good = tmp_path / "good"; A["save"](good / "audit.json", report)
    A["catalog_accounts"](good, report, equations, contexts)
    ids = A["read"](good / "atlas/provenance.json")["observations"]
    assert len(ids) == 2 and all(ComputationAtlas(good / "atlas").show(identifier)["equation"]["binding"]["execution"]["backend"] == "cpu" for identifier in ids)
    for context in contexts.values():
        del context["execution"]["backend"]
    bad = tmp_path / "bad"; A["save"](bad / "audit.json", report)
    with pytest.raises(ComputationAtlasError, match="execution backend is required"):
        A["catalog_accounts"](bad, report, equations, contexts)
    external = A["row"]()["external_evidence"](SAVED, A["read"](SAVED / "files.json"))
    assert set(external["roots"]) == set(external["manifests"]) == {"native", "reader"} and len(external["manifests"]["reader"]) == 13


def test_cli_worker_standard_identity_and_bound_marker_without_science(tmp_path):
    source = HERE / "audit_qwen_cached_qkv.py"
    result = subprocess.run([sys.executable, "-I", "-B", str(source), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--expected-manifest" in result.stdout
    raw = tmp_path / "raw"; raw.mkdir(); (raw / "metadata").mkdir()
    names = ("plan.json", "declared-config.json", "request.json", "source.json", "summary.json", "runtime-contract.json", "files.json", "source-frontier.json", "forecast-responses.json",
        "forecast.json", "native.json", "layout-restoration.json", "parameter-identity-before.json", "parameter-identity-after.json", "native-contract.json", "native-sdpa.json", "sdpa.json")
    for name in names:
        A["save"](raw / name, dict(input_root="synthetic-parent", source_arrays={}) if name == "source-frontier.json" else {})
    (raw / "timeline.jsonl").write_text("")
    config = dict(native_root=str(raw), native_manifest_sha256="3"*64, native_request_sha256="4"*64, output_scratch=str(tmp_path / "output"))
    request = dict(config=config, source_hashes={str(source): A["sha"](source)})
    request["request_sha256"] = A["base"]()["identity"](request)
    A["save"](tmp_path / "config.json", config); A["save"](tmp_path / "request.json", request)
    script = tmp_path / "controlled.py"
    script.write_text("import runpy\na=runpy.run_path("+repr(str(source))+")\n"
        "def audit(root,output,**kwargs):\n output.mkdir(parents=True); a['save'](output/'audit.json',{'scope':'model-free callback'}); return {'job_id':'job-fake','arms':{'a':{},'b':{}}},{'files':{}}\n"
        "a['worker'].__globals__['audit']=audit\na['worker']()\n")
    env = dict(os.environ, SATURN_DEBUG_REQUEST=str(tmp_path / "request.json"), SATURN_RUN_CONFIG=str(tmp_path / "config.json"), MRUN_JOB_ID="job-tiny-reader", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-I", "-B", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    marker = json.loads(result.stdout.split("MDB_DIRECTORY_READY ")[1])
    assert marker["request_sha256"] == request["request_sha256"] and A["sha"](Path(marker["output_root"])/"files.json") == marker["files_manifest_sha256"]
    request["source_hashes"][str(source)] = "0"*64; request["request_sha256"] = A["base"]()["identity"](request)
    (tmp_path / "request.json").write_text(json.dumps(request)); env["MRUN_JOB_ID"] = "job-denied"
    denied = subprocess.run([sys.executable, "-I", "-B", str(script)], env=env, capture_output=True, text=True)
    assert denied.returncode != 0 and "source differs" in denied.stderr and not (tmp_path / "output/job-denied").exists()
