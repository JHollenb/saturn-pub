"""Actual tiny writer serialization; saved-only reductions and hostile joins."""
from __future__ import annotations

import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
A = runpy.run_path(str(HERE / "audit_qwen_error_consumer.py"))
SAVED = HERE / "saved-fixture"
QCONFIG = dict(head_dim=4, num_attention_heads=4, num_key_value_heads=2)


def protocol(root=SAVED):
    return A["read"](root / "protocol.json")


def inherited():
    value = protocol()
    records = {**value["world_sources"], **value["shared_context_sources"]}
    return {name: A["base"]()["load_array"](Path(value["roots"]["world" if name in A["WORLDS"] else "context"]["root"]), record) for name, record in records.items()}


def tx(root=SAVED, inventory=None):
    return A["transaction"](root, protocol(root), A["read"](root / "summary.json"), inventory or A["read"](root / "files.json"),
        A["read"](root / "source.json")["helper_source_sha256"], 64 << 20)


def test_actual_writer_full_endpoints_responses_all_windows_explicit_budget(monkeypatch):
    import mdb.equation_comparison as api

    compare, budgets = api.compare_equations, []
    def instrument(*args, **kwargs):
        budgets.append(kwargs["max_elements"])
        return compare(*args, **kwargs)
    monkeypatch.setattr(api, "compare_equations", instrument)
    result, equations = A["numerical_accounts"](SAVED, inherited(), protocol())
    assert budgets == [320, 320] and set(equations) == set(A["WORLDS"])
    assert result["coverage"] == dict(primary_endpoint_fields=10, primary_response_fields=5, dependent_endpoint_windows=20, dependent_response_windows=10)
    assert result["input_difference"]["windows"][0]["comparison"]["changed_values"] == 0
    assert result["input_difference"]["windows"][1]["comparison"]["changed_values"] > 0
    for world in result["worlds"].values():
        assert world["first_numeric_difference"] is None and all(field["whole"]["rms"] == 0 for field in world["fields"].values())
    assert all(field["error"]["whole"]["rms"] == 0 for field in result["responses"].values())
    assert "torch" not in sys.modules and "transformers" not in sys.modules


def test_actual_four_calls_two_native_inputs_context_and_no_hidden_baseline():
    summary = A["read"](SAVED / "summary.json")
    result = A["operand_accounts"](SAVED, inherited(), protocol(), summary, QCONFIG)
    assert len(result["actual_SDPA_calls"]) == 4 and len(result["native_entries"]) == 2
    assert result["observed"]["native_module_entries"] == 30 and result["observed"]["native_authorizations"] == 2
    refs = A["read"](SAVED / "source-frontier.json")
    allrefs = {**{key: value["input_reference"] for key, value in refs["worlds"].items()}, **refs["shared_context"]}
    assert len(A["saved"]()["layout_accounts"](SAVED, allrefs)["arrays"]) == 7


def test_actual_closed_transaction_and_complete_pretarget_inventory(tmp_path):
    from mdb.forecast_transaction import ForecastTransactionError

    assert tx()["verified"]["events"] == 7 and tx()["native_targets_at_seal"] == 0
    wrapper = A["row"]()["wrapper"](SAVED, A["sha"](SAVED / "files.json"), "1"*64)
    assert len(wrapper["files"]) > 80
    inventory = A["read"](SAVED / "files.json")
    inventory["files"]["unsealed-future.npy"] = dict(bytes=1, sha256="0"*64)
    with pytest.raises(ValueError, match="complete pretarget"):
        tx(inventory=inventory)
    raw = tmp_path / "raw"; shutil.copytree(SAVED, raw)
    (raw / "transaction/seal.json").write_text("{}")
    with pytest.raises(ForecastTransactionError, match="global seal differs"):
        tx(raw)


def test_hostile_world_input_layout_context_and_response_commitments(tmp_path):
    raw = tmp_path / "raw"; shutil.copytree(SAVED, raw)
    path = raw / "native.json"; value = A["read"](path)
    value["worlds"]["cheap-input"]["actual_input_descriptor"] = value["worlds"]["full-input"]["actual_input_descriptor"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="actual consumed operand"):
        A["numerical_accounts"](raw, inherited(), protocol())
    shutil.copyfile(SAVED / "native.json", path)
    path = raw / "sdpa.json"; value = A["read"](path); value["calls"][0]["operands"][0]["stride"] = [1, 1, 1, 1]; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="actual consumed operand"):
        A["operand_accounts"](raw, inherited(), protocol(), A["read"](raw / "summary.json"), QCONFIG)
    path = raw / "forecast-responses.json"; value = A["read"](path); value["fields"]["head"]["sha256"] = "0"*64; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="complete pretarget signed response"):
        A["numerical_accounts"](raw, inherited(), protocol())


def test_misses_signedzero_negative_cross_terms_and_finite_defect_are_observations():
    reference = np.zeros((1, 8, 2), np.float32); prediction = reference.copy()
    prediction[0, 0, 0] = -0.; prediction[0, 7, 0] = .5
    report = A["windowed"](prediction, reference, [[0, 4], [4, 8]])
    assert report["whole"]["changed_values"] == 1 and report["whole"]["signed_zero"]["defects"] == 1
    assert report["windows"][0]["comparison"]["rms"] == 0 and not report["windows"][0]["comparison"]["bitwise_exact"]
    h = np.ones((1, 8, 2)); projection = 2*h; r = 3*h+.25; m = -2*h; y = r+m-.125
    values = dict(projection=projection, R=r, M=m, Hnext=y)
    account = A["residual_accounts"](h, values, [[0, 4], [4, 8]])["whole"]
    assert account["R_M"]["signed_cross_term"] < 0
    assert account["residual"]["terms"]["finite_addition_defect"]["signed_mean"] == .25
    assert account["output"]["terms"]["finite_addition_defect"]["signed_mean"] == -.125
    assert account["output"]["vector_identity_residual"]["max_abs"] == 0
    assert A["vector_pair"](h, np.zeros_like(h))["cosine"] is None


def test_configured_runtime_profile_and_forecast_input_layout_are_integrity(tmp_path):
    flags = dict(cuda_allow_bf16_reduced_precision_reduction=True, cuda_allow_tf32=False, cudnn_allow_tf32=False,
        cudnn_benchmark=False, cudnn_deterministic=True, deterministic_algorithms=True, float32_matmul_precision="highest", num_threads=4)
    assert "not independently profiled" in A["runtime_profile"](dict(flags=flags))["authority"]
    with pytest.raises(ValueError, match="runtime primitive policy"):
        A["runtime_profile"](dict(flags=dict(flags, cuda_allow_tf32=True)))
    raw = tmp_path / "raw"; shutil.copytree(SAVED, raw)
    path = raw / "forecast.json"; value = A["read"](path)
    value["worlds"]["cheap-input"]["consumed_input_descriptor"]["stride"] = [1, 1, 1]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="actual consumed operand"):
        A["numerical_accounts"](raw, inherited(), protocol())


def test_actual_audit_core_and_production_catalog_context_installed_atlas(tmp_path, monkeypatch):
    """Real saved Tx/census/numeric/catalog paths; outer custody/recipe service are explicit synthetic substitutes."""
    from mdb.computation_atlas import ComputationAtlas

    g = A["audit"].__globals__; summary = A["read"](SAVED / "summary.json"); source = A["read"](SAVED / "source.json")
    config = dict(forecast_transaction_bytes=64 << 20, qwen_parameter_manifest_sha256="2"*64, attention_dispatch=dict(backend="sdpa", gqa_policy="enable_gqa"))
    source = dict(source, source_hashes={"synthetic": "3"*64})
    source_identity = dict(source["helper_source_sha256"], qwen_native=A["row"]()["NATIVE_SHA256"], sdpa_native=A["row"]()["SDPA_SHA256"])
    # Actual tiny producer used helper-only source identity; keep the exact tested transaction identity.
    original_transaction = A["transaction"]
    monkeypatch.setitem(g, "transaction", lambda root, p, s, i, _source, budget: original_transaction(root, p, s, i, source["helper_source_sha256"], budget))
    monkeypatch.setitem(g, "authenticate", lambda *_args: (config, source, summary, A["read"](SAVED / "files.json"), {}, protocol()))
    monkeypatch.setitem(g, "inherited_accounts", lambda *_args: (inherited(), dict(scope="synthetic already-custodied source fixture")))
    monkeypatch.setitem(g, "source_accounts", lambda *_args: (dict(runtime=A["read"](SAVED / "runtime-contract.json")) if (SAVED / "runtime-contract.json").exists() else dict(runtime=dict(flags={}, scope="synthetic CPU")), QCONFIG))
    out = tmp_path / "audit"
    report, _ = A["audit"](SAVED, out, expected_manifest=A["sha"](SAVED / "files.json"), expected_request="1"*64)
    assert source_identity and report["coverage"]["primary_endpoint_fields"] == 10
    records = A["read"](out / "atlas/provenance.json")["observations"]
    assert len(records) == 2 and all(ComputationAtlas(out / "atlas").show(value)["equation"]["binding"]["execution"]["backend"] == "cpu" for value in records)
    for alias in ("world", "context"):
        external = A["external_evidence"](SAVED, A["read"](SAVED / "files.json"), alias)
        assert set(external["roots"]) == set(external["manifests"]) == {"native", "reader"}
        assert len(external["manifests"]["reader"]) == (2 if alias == "world" else 5)


def test_actual_request_envelope_and_member_corruption_refusal(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    config = dict(output_scratch=str(tmp_path / "out"), output_name="qwen-error-consumer-result")
    request = dict(config=config, source_hashes={"reader.py": "2"*64}); request["request_sha256"] = A["base"]()["identity"](request)
    job, identity = "job-synthetic-envelope", request["request_sha256"]
    source = dict(source_hashes=request["source_hashes"], job_id=job, request_sha256=identity)
    for name, value in {"request.json": request, "declared-config.json": config, "source.json": source, "summary.json": dict(job_id=job, request_sha256=identity),
        "plan.json": dict(config, output=str(Path(config["output_scratch"])/job/config["output_name"]), request_source_hashes=request["source_hashes"], request_sha256=identity, job_id=job)}.items():
        A["save"](raw / name, value)
    files = {path.name: dict(bytes=path.stat().st_size, sha256=A["sha"](path)) for path in raw.iterdir()}
    A["save"](raw / "files.json", dict(schema="mdb-image-tail-files-v1", job_id=job, request_sha256=identity, files=files, expanded_bytes=sum(value["bytes"] for value in files.values())))
    inventory = A["row"]()["wrapper"](raw, A["sha"](raw / "files.json"), identity)
    assert A["row"]()["request_envelope"](raw, inventory, identity)[0]["job_id"] == job
    (raw / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="retained member checksum"):
        A["row"]()["wrapper"](raw, A["sha"](raw / "files.json"), identity)


def test_cli_help_no_model_import_and_production_budget_geometry():
    result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "audit_qwen_error_consumer.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--expected-manifest" in result.stdout
    assert 512*32*128 + 4*512*2560 == 7340032
    assert "torch" not in sys.modules and "transformers" not in sys.modules


def test_worker_actual_marker_and_compact_metadata_only_pack(tmp_path, monkeypatch, capsys):
    g = A["worker"].__globals__; raw = tmp_path / "raw"; raw.mkdir(); (raw / "metadata").mkdir()
    frontier = dict(roots={alias: dict(root="synthetic-"+alias) for alias in ("world", "context")},
        worlds={name: dict(input_reference=dict(path=name+".npy", sha256="1"*64)) for name in A["WORLDS"]}, shared_context={})
    for name in A["COPY_FILES"]:
        if name == "timeline.jsonl":
            (raw / name).write_text("")
        else:
            A["save"](raw / name, frontier if name == "source-frontier.json" else {})
    config = dict(native_root=str(raw), native_manifest_sha256="2"*64, native_request_sha256="3"*64, output_scratch=str(tmp_path / "output"))
    source = HERE / "audit_qwen_error_consumer.py"
    request = dict(config=config, source_hashes={str(source): A["sha"](source)}); request["request_sha256"] = A["base"]()["identity"](request)
    A["save"](tmp_path / "request.json", request); A["save"](tmp_path / "config.json", config)
    for name, value in dict(SATURN_DEBUG_REQUEST=str(tmp_path/"request.json"), SATURN_RUN_CONFIG=str(tmp_path/"config.json"), MRUN_JOB_ID="job-tiny-reader", CUDA_VISIBLE_DEVICES="").items():
        monkeypatch.setenv(name, value)
    def fake_audit(_root, output, **_kwargs):
        output.mkdir(parents=True); A["save"](output/"audit.json", dict(scope="model-free transport callback")); return dict(job_id="job-source"), dict(files={})
    monkeypatch.setitem(g, "audit", fake_audit)
    A["worker"]()
    marker = json.loads(capsys.readouterr().out.split("MDB_DIRECTORY_READY ")[1]); output = Path(marker["output_root"])
    assert marker["request_sha256"] == request["request_sha256"] and A["sha"](output/"files.json") == marker["files_manifest_sha256"]
    assert not list(output.rglob("*.npy")) and (output/"external-evidence-context.json").exists()
    request["source_hashes"][str(source)] = "0"*64; request["request_sha256"] = A["base"]()["identity"](request)
    (tmp_path/"request.json").write_text(json.dumps(request))
    with pytest.raises(ValueError, match="reader source differs"):
        A["worker"]()
