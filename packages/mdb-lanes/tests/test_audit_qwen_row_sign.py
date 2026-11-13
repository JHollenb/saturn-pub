"""Actual saved writer/Tx accounts; synthetic custody harnesses are separate."""
from __future__ import annotations

import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
A = runpy.run_path(str(HERE / "audit_qwen_row_sign.py"))
SAVED = HERE / "saved-fixture"


def saved_values(root=SAVED):
    frontier = A["read"](root / "source-frontier.json")
    protocol = dict(fixed_split=3, source_arrays={name: {key: value for key, value in row.items() if key not in {"inherited", "root"}}
        for name, row in frontier["source_arrays"].items()})
    values = {name: A["base"]()["load_array"](HERE / "synthetic-parent", row) for name, row in protocol["source_arrays"].items()}
    return values, protocol


def tiny_inventory(root):
    return dict(files={path.relative_to(root).as_posix(): dict(bytes=path.stat().st_size, sha256=A["sha"](path))
        for path in root.rglob("*") if path.is_file() and path.name != "files.json"})


def transaction(root=SAVED, inventory=None):
    return A["transaction"](root, A["read"](root / "summary.json"), inventory or A["read"](root / "files.json"), max_bytes=64 << 20,
        source_identity=dict(qwen_native=A["NATIVE_SHA256"], sdpa_native=A["SDPA_SHA256"]))


def toy_arrays():
    positive = {lane: {name: np.ones((1, 8, 2, 2) if name == "head" else (1, 8, 4), np.float32)
                       for name in (*A["PORTS"], "gate", "up")} for lane in ("functional", "native")}
    positive["native"]["M"] *= np.float32(1.5)
    arms = {}
    for arm in A["ARMS"]:
        signs = np.full(8, -1, np.float32)
        if arm == "prefix-negative":
            signs[4:] = 1
        permitted = np.tri(8, dtype=bool)
        closed = permitted.any(-1) & ~(permitted & (signs[:, None] != signs[None, :])).any(-1)
        lanes = {lane: {name: value * signs.reshape(1, -1, *([1] * (value.ndim - 2)))
                       for name, value in positive["native" if lane == "native" else "functional"].items()}
                 for lane in ("shortcut", "functional", "native")}
        # Deliberate mixed-window disagreement remains a complete observation.
        lanes["functional"]["Hnext"][:, 6, 0] += np.float32(.25)
        hashes = {lane: {name: A["descriptor"](bank[name], positive["functional"][name]) for name in A["PORTS"]}
                  for lane, bank in lanes.items() if lane != "native"}
        arms[arm] = dict(signs=signs, permitted=permitted, support_closed=closed, lanes=lanes, response_hashes=hashes)
    return positive, arms


def test_all_rows_retained_windows_support_and_three_error_channels():
    positive, arms = toy_arrays()
    result, equations = A["array_accounts"](positive, arms, split=4)
    assert len(equations) == 2 and all(len(row) == 3 for row in equations.values())
    assert result["arms"]["all-negative"]["support"]["closed_rows"] == 8
    mixed = result["arms"]["prefix-negative"]["support"]["windows"]
    assert mixed["prefix"]["classification"] == "closed" and mixed["suffix"]["classification"] == "mixed-or-empty"
    for arm in A["ARMS"]:
        fields = result["arms"][arm]["channels"]["shortcut-minus-full-functional"]["fields"]
        assert fields["Hnext"]["endpoint"]["whole"]["changed_values"] == 1
        assert fields["Hnext"]["endpoint"]["windows"]["prefix"]["changed_values"] == 0
        assert fields["Hnext"]["endpoint"]["windows"]["suffix"]["changed_values"] == 1
        assert all(row["endpoint"]["whole"]["shape"][1] == 8 for row in fields.values())
    # Distinct native positive baseline makes M response error differ from its
    # endpoint error; substituting functional plus into native would hide it.
    fields = result["arms"]["all-negative"]["channels"]["shortcut-minus-native"]["fields"]
    assert fields["M"]["endpoint"]["whole"]["maxabs"] == .5
    assert fields["M"]["response_error"]["whole"]["maxabs"] == 1.


def test_signed_commitment_corruption_and_wrong_intervention_refuse():
    positive, arms = toy_arrays()
    arms["prefix-negative"]["response_hashes"]["shortcut"]["M"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pretarget signed response hash"):
        A["array_accounts"](positive, arms, split=4)
    positive, arms = toy_arrays()
    arms["prefix-negative"]["signs"][0] = 1
    with pytest.raises(ValueError, match="fixed row-sign"):
        A["array_accounts"](positive, arms, split=4)


def test_structural_empty_support_is_diagnostic_not_prediction_selector():
    signs = np.array([-1., -1., 1., 1.], np.float32)
    permitted = np.tri(4, dtype=bool)
    permitted[0] = False
    closed = np.array([False, True, False, False])
    result = A["support_account"](signs, permitted, closed, split=2)
    assert result["empty_rows"] == 1 and result["rows"] == 4 and result["closed_rows"] == 1


def test_row_signed_zero_defect_is_distinct_from_numeric_equality():
    positive = np.zeros((1, 4, 2), np.float32)
    signs = np.array([-1., -1., 1., 1.], np.float32)
    edited = positive.copy()
    report = A["premise"](positive, edited, signs, split=2)
    assert report["whole"]["changed_values"] == 0
    assert report["whole"]["bitwise_exact"] is False
    assert report["signed_zero"]["prefix"]["defects"] == 4 and report["signed_zero"]["suffix"]["defects"] == 0


def test_actual_writer_arrays_context_and_native_census_without_framework_imports():
    # This is the producer's actual persisted tiny native Qwen class fixture,
    # not handwritten descriptors or a rerun of that neural fixture.
    values, protocol = saved_values()
    result, equations = A["saved_accounts"](SAVED, values, protocol)
    native = A["native_accounts"](SAVED, A["read"](SAVED / "summary.json"), dict(head_dim=4, num_attention_heads=4, num_key_value_heads=2))
    assert native["actual_component_entries"] == 3 and native["actual_context_tensor_joins"] == 9
    assert native["observed"]["native_module_entries"] == 45
    assert result["head_alignment"]["permutation"] == [0, 2, 1, 3]
    assert result["arms"]["prefix-negative"]["support"]["closed_rows"] == 3
    assert len(equations) == 2 and all(len(row) == 3 for row in equations.values())
    assert "torch" not in sys.modules and "transformers" not in sys.modules


def test_actual_writer_public_frontier_seal_and_complete_inventory():
    result = transaction()
    assert result["verified"]["phase"] == "closed" and result["verified"]["events"] == 7
    assert result["native_targets_at_seal"] == 0 and result["inherited_arrays"] == 13
    assert [row["native_module_entries"] for row in result["chronology"][-2:]] == [30, 45]
    wrapper = A["wrapper"](SAVED, A["sha"](SAVED / "files.json"), "1" * 64)
    assert wrapper["job_id"] == "job-tiny-row-sign" and len(wrapper["files"]) == 114


def test_actual_writer_hostile_native_future_inclusion_and_missing_source(tmp_path):
    from mdb_lanes.forecast_transaction import ForecastTransactionError

    root = tmp_path / "raw"
    shutil.copytree(SAVED, root)
    inventory = tiny_inventory(root)
    inventory["files"]["undeclared-future.npy"] = dict(bytes=1, sha256="0" * 64)
    with pytest.raises(ValueError, match="complete pretarget frontier"):
        transaction(root, inventory)
    inventory = tiny_inventory(root)
    del inventory["files"]["context/cos.npy"]
    with pytest.raises(ValueError, match="complete pretarget frontier"):
        transaction(root, inventory)
    (root / "transaction/seal.json").write_text("{}")
    with pytest.raises(ForecastTransactionError, match="global seal differs"):
        transaction(root)


def test_actual_writer_wrong_consumed_context_and_presealed_source_refuse(tmp_path):
    root = tmp_path / "raw"
    shutil.copytree(SAVED, root)
    path = root / "native-contract.json"
    value = A["read"](path)
    value["block_entries"][1]["consumed_context"]["cos"]["content_sha256"] = "0" * 64
    path.unlink(); A["save"](path, value)
    with pytest.raises(ValueError, match="actual consumed context"):
        A["native_accounts"](root, A["read"](root / "summary.json"), dict(head_dim=4, num_attention_heads=4, num_key_value_heads=2))
    values, protocol = saved_values(root)
    protocol["source_arrays"]["head"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="inherited source reference"):
        A["saved_accounts"](root, values, protocol)


def test_catalog_atlas_distinguishes_actual_consumer_effect_from_errors(tmp_path):
    from mdb_lanes.computation_atlas import ComputationAtlas, ComputationAtlasError

    values, protocol = saved_values()
    report, equations = A["saved_accounts"](SAVED, values, protocol)
    report["census"] = A["native_accounts"](SAVED, A["read"](SAVED / "summary.json"), dict(head_dim=4, num_attention_heads=4, num_key_value_heads=2))
    report["job_id"] = "tiny-saved-only"
    plan = dict(qwen_parameter_manifest_sha256="1" * 64, attention_dispatch=dict(backend="sdpa", gqa_policy="enable_gqa"), horizon=19, fixed_split=3)
    contexts = A["catalog_contexts"](plan, dict(source_hashes={"synthetic_source": "2" * 64}),
        dict(runtime=A["read"](SAVED / "runtime-contract.json")), protocol, report)
    assert all(row["execution"]["backend"] == "cpu" and row["execution"]["operand_device"] == "cpu" for row in contexts.values())
    good = tmp_path / "good"
    A["save"](good / "audit.json", report)
    A["catalog_accounts"](good, report, equations, contexts)
    records = A["read"](good / "catalog/provenance.json")["records"]
    ids = A["read"](good / "atlas/provenance.json")["observations"]
    assert len(records) == len(ids) == 2
    atlas = ComputationAtlas(good / "atlas")
    for identifier in ids:
        observation = atlas.show(identifier)
        assert observation["equation"]["binding"]["execution"]["backend"] == "cpu"
        assert {row["contrast_id"] for row in observation["signed_effects"]} == {"native-edited-minus-native-baseline",
            "shortcut-minus-native-prediction-error", "shortcut-response-minus-native-response-error"}
    # Reproduce the exact installed schema rejection that the prior handmade
    # context fixture failed to expose; production uses this same builder now.
    bad_contexts = {arm: {**row, "execution": {key: value for key, value in row["execution"].items() if key != "backend"}} for arm, row in contexts.items()}
    bad = tmp_path / "missing-backend"
    A["save"](bad / "audit.json", report)
    with pytest.raises(ComputationAtlasError, match="execution backend is required"):
        A["catalog_accounts"](bad, report, equations, bad_contexts)


def test_inherited_alias_matches_actual_frozen_retriever_cli_without_network(tmp_path):
    manifest = A["external_evidence"](SAVED, A["read"](SAVED / "files.json"))
    assert set(manifest["roots"]) == set(manifest["manifests"]) == {"native", "reader"}
    assert len(manifest["manifests"]["reader"]) == 13 and "inherited" in manifest["alias_roles"]["reader"]
    path = next(iter(manifest["manifests"]["reader"]))
    destination = tmp_path / "retrieved"
    target = destination / path
    target.parent.mkdir(parents=True)
    shutil.copyfile(HERE / "synthetic-parent" / path, target)
    A["save"](tmp_path / "external-evidence.json", manifest)
    result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "retrieve_reader_evidence.py"), "reader", path, str(destination),
        "--manifest", str(tmp_path / "external-evidence.json")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(target) and A["sha"](target) == manifest["manifests"]["reader"][path]


def test_standard_request_envelope_and_changed_config_are_separate_model_free_custody(tmp_path):
    config = dict(output_scratch="/not-executed", output_name="result")
    request = dict(config=config, source_hashes={"not-executed.py": "4" * 64})
    request["request_sha256"] = A["base"]()["identity"](request)
    job, identity = "job-model-free-envelope", request["request_sha256"]
    for name, value in {"declared-config.json": config, "request.json": request,
        "plan.json": dict(config, output=f"/not-executed/{job}/result", request_source_hashes=request["source_hashes"], request_sha256=identity, job_id=job),
        "source.json": dict(job_id=job, request_sha256=identity, source_hashes=request["source_hashes"]),
        "summary.json": dict(job_id=job, request_sha256=identity)}.items():
        A["save"](tmp_path / name, value)
    assert A["request_envelope"](tmp_path, dict(job_id=job), identity)[0]["job_id"] == job
    path = tmp_path / "declared-config.json"
    path.unlink(); A["save"](path, dict(config, changed=True))
    with pytest.raises(ValueError, match="standard executed request/config"):
        A["request_envelope"](tmp_path, dict(job_id=job), identity)


def test_cli_help_and_worker_request_marker_model_free(tmp_path):
    import json
    import os

    help_result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "audit_qwen_row_sign.py"), "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0 and "expected-manifest" in help_result.stdout
    raw = tmp_path / "raw"
    raw.mkdir(); (raw / "metadata").mkdir()
    names = ("plan.json", "declared-config.json", "request.json", "source.json", "summary.json", "runtime-contract.json", "files.json", "source-frontier.json", "forecast-responses.json",
        "forecast.json", "shortcut.json", "native.json", "parameter-identity-before.json", "parameter-identity-after.json", "native-contract.json", "native-sdpa.json")
    for name in names:
        value = dict(input_root="synthetic-parent", source_arrays={}) if name == "source-frontier.json" else {}
        A["save"](raw / name, value)
    (raw / "timeline.jsonl").write_text("")
    config = dict(native_root=str(raw), native_manifest_sha256="5" * 64, native_request_sha256="6" * 64, output_scratch=str(tmp_path / "output"))
    source = HERE / "audit_qwen_row_sign.py"
    request = dict(config=config, source_hashes={str(source): A["sha"](source)})
    request["request_sha256"] = A["base"]()["identity"](request)
    A["save"](tmp_path / "request.json", request); A["save"](tmp_path / "config.json", config)
    script = tmp_path / "controlled-worker.py"
    script.write_text("import runpy,pathlib\na=runpy.run_path(" + repr(str(source)) + ")\n"
        "def audit(root,output,**kwargs):\n output.mkdir(parents=True); a['save'](output/'audit.json',{'scope':'model-free worker mechanics'}); return {'job_id':'job-fake-science'}, {'files':{}}\n"
        "a['worker'].__globals__['audit']=audit\na['worker']()\n")
    env = dict(os.environ, SATURN_DEBUG_REQUEST=str(tmp_path / "request.json"), SATURN_RUN_CONFIG=str(tmp_path / "config.json"), MRUN_JOB_ID="job-tiny-reader", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-I", "-B", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    marker = json.loads(result.stdout.split("MDB_DIRECTORY_READY ")[1])
    output = Path(marker["output_root"])
    assert marker["job_id"] == "job-tiny-reader" and marker["request_sha256"] == request["request_sha256"]
    assert A["sha"](output / "files.json") == marker["files_manifest_sha256"]
    assert A["sha"](output / "summary.json") == marker["summary_sha256"]
    request["source_hashes"][str(source)] = "0" * 64
    request["request_sha256"] = A["base"]()["identity"](request)
    (tmp_path / "request.json").write_text(json.dumps(request))
    env["MRUN_JOB_ID"] = "job-denied-reader"
    denied = subprocess.run([sys.executable, "-I", "-B", str(script)], env=env, capture_output=True, text=True)
    assert denied.returncode != 0 and "source differs" in denied.stderr and not (tmp_path / "output/job-denied-reader").exists()
