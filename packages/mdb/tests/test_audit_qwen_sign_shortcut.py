"""Saved-only sign-shortcut accounts and actual-writer serialization mechanics."""
from __future__ import annotations

import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
AUDIT = runpy.run_path(str(HERE / "audit_qwen_sign_shortcut.py"))
SAVED = HERE / "saved-fixture"


def banks(size=12):
    positive = {
        lane: {name: np.full((1, 1, size), index + 1, np.float32) for index, name in enumerate(AUDIT["HEADS"])}
        for lane in ("functional", "native")
    }
    positive["native"]["M"] = positive["native"]["M"] + np.float32(0.5)
    negative = {
        lane: {name: -value.copy() for name, value in positive["native" if lane == "native" else "functional"].items()}
        for lane in ("shortcut", "functional", "native")
    }
    negative["shortcut"]["product"][0, 0, 0] += np.float32(0.25)
    negative["functional"]["Hnext"][0, 0, -1] -= np.float32(0.5)
    seal = {lane: {name: AUDIT["signed_descriptor"](negative[lane][name], positive["functional"][name])
                   for name in AUDIT["HEADS"]} for lane in ("functional", "shortcut")}
    return negative, positive, seal


def test_three_endpoint_and_own_positive_response_errors_retain_misses():
    negative, positive, seal = banks()
    result, equations = AUDIT["compare_values"](negative, positive, seal)
    assert len(result["comparisons"]) == len(equations) == 3
    shortcut = result["comparisons"]["shortcut-minus-native-negative"]["fields"]
    assert shortcut["product"]["endpoint"]["changed_values"] == 1
    # Endpoint offset is .5, while independently owned plus baselines add a
    # second .5 to the signed edit-response discrepancy. Cross-installing the
    # functional plus ports would silently erase that distinction.
    assert shortcut["M"]["endpoint"]["maxabs"] == pytest.approx(0.5)
    assert shortcut["M"]["signed_edit_response_error"]["maxabs"] == pytest.approx(1.)
    assert "own-positive" in shortcut["M"]["left_response"]["orientation"]
    assert result["comparisons"]["shortcut-minus-full-functional-negative"]["first_endpoint_bit_discrepancy"] == "product"
    assert shortcut["Hnext"]["endpoint"]["bitwise_exact"]


def test_complete_signed_commitment_tamper_refuses_before_catalog():
    negative, positive, seal = banks()
    seal["shortcut"]["M"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="signed response hash"):
        AUDIT["compare_values"](negative, positive, seal)


def test_real_mdb_catalog_budget_above_default_million():
    from mdb.equation_comparison import compare_equations

    negative, positive, seal = banks(350_000)
    declared = []

    def observed_public_call(*args, **kwargs):
        declared.append(kwargs["max_elements"])
        return compare_equations(*args, **kwargs)

    result, reports = AUDIT["compare_values"](negative, positive, seal, compare_core=observed_public_call)
    assert declared == [1_050_000] * 3
    assert result["comparison_max_elements"] == 1_050_000
    assert all(report["schema"] == "mdb-equation-comparison-v1" for report in reports.values())


def test_oddness_distinguishes_numerical_and_signed_zero_bits():
    positive = np.array([0., -0., 1., -2.], np.float32)
    negative = np.array([0., -0., -1., 2.], np.float32)
    report = AUDIT["oddness"](positive, negative)
    assert report["numerical_changed"] == 0
    assert report["bit_changed_elements"] == report["zero_signbit_defects"] == 2
    assert report["status"] == "observation-only"


def test_one_rule_three_conditional_catalog_and_atlas_observations(tmp_path):
    negative, positive, seal = banks()
    tail, equations = AUDIT["compare_values"](negative, positive, seal)
    cases = AUDIT["CASE_IDS"]
    audit = dict(job_id="tiny-saved-only", cases={case: dict(tail=tail) for case in cases})
    AUDIT["save"](tmp_path / "audit.json", audit)
    contexts = {case: dict(model_identity="1" * 64, execution=dict(backend="cpu", dtype="bfloat16"),
        input_identity=AUDIT["digest"](case), frontier="own-functional-positive-RGU", horizon=19,
        source_identity={"worker": "2" * 64}, case=case) for case in cases}
    summary = AUDIT["catalog_accounts"](tmp_path, audit, {case: equations for case in cases}, contexts)
    assert summary["records"] == summary["atlas_observations"] == 3
    provenance = AUDIT["read"](tmp_path / "catalog/provenance.json")
    assert all(row["exact"] == 1 and row["wrong_input"] == 0 for row in provenance["queries"])
    assert provenance["automatic_family_induction"] is False


def test_actual_writer_saved_three_context_case_accounts_without_models():
    summary = AUDIT["read"](SAVED / "summary.json")
    responses = AUDIT["read"](SAVED / "forecast-responses.json")
    measured = {}
    for case in summary["cases"]:
        measured[case], reports = AUDIT["case_accounts"](SAVED, case, edit_index=2, horizon=3, native_layers=5,
            sealed=responses["cases"][case])
        assert len(reports) == 3
        assert measured[case]["coverage"]["primary_prefix_fields"] == 4
        assert measured[case]["native"]["block_entries"] == measured[case]["native"]["SDPA"] == 6
        assert measured[case]["native"]["genuine_qwen_act"] is False
    assert measured["tiny-long"]["source"]["actual_padded_rows"] == 0
    assert measured["tiny-objects"]["source"]["actual_padded_rows"] == 2
    assert sum(row["native"]["module_entries"] for row in measured.values()) == summary["counts"]["native_module_entries"]


def fixture_inventory(root):
    return dict(files={path.relative_to(root).as_posix(): dict(bytes=path.stat().st_size, sha256=AUDIT["sha"](path))
                       for path in root.rglob("*") if path.is_file()})


def test_actual_public_transaction_full_external_inventory_and_zero_native_seal():
    summary = AUDIT["read"](SAVED / "summary.json")
    result, prediction = AUDIT["transaction"](SAVED, dict(forecast_transaction_bytes=64 << 20, edit_index=2, horizon=3), {}, summary,
        fixture_inventory(SAVED), case_ids=summary["cases"], source_identity={"synthetic_fixture": "1" * 64})
    assert result["native_targets_at_seal"] == 0 and result["verified"]["phase"] == "closed"
    assert len(prediction["response_hashes"]) == 3


def test_hostile_unsealed_extra_and_wrong_native_consumer_are_refused(tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(SAVED, root)
    summary = AUDIT["read"](root / "summary.json")
    (root / "undeclared-future.json").write_text("{}")
    with pytest.raises(ValueError, match="complete pretarget file frontier"):
        AUDIT["transaction"](root, dict(forecast_transaction_bytes=64 << 20, edit_index=2, horizon=3), {}, summary,
            fixture_inventory(root), case_ids=summary["cases"], source_identity={"synthetic_fixture": "1" * 64})
    case = summary["cases"][0]
    native_path = root / "cases" / case / "native.json"
    native = AUDIT["read"](native_path)
    native["consumed"][0]["hidden"]["tensor"] = native["common"]["2"]
    native_path.unlink()
    AUDIT["save"](native_path, native)
    with pytest.raises(ValueError, match="actual native suffix consumed"):
        AUDIT["case_accounts"](root, case, edit_index=2, horizon=3, native_layers=5,
            sealed=AUDIT["read"](root / "forecast-responses.json")["cases"][case])


def envelope(root):
    """Model-free custody specimen, separate from the actual tiny neural data."""
    import mdb.parameter_artifacts as api

    root.mkdir()
    config = dict(schema="mdb-qwen-sign-shortcut-campaign-v1", output_scratch="/not-executed", output_name="result",
        selection_protocol_sha256=AUDIT["PROTOCOL_SHA256"], edit_index=18, horizon=19, selected=[9, 18, 19],
        qwen_rope_buffer_policy=dict(init_device="cpu", init_dtype="float32", placement_dtype="float32", source="original native CPU initializer; device-only placement"),
        attention_dispatch=dict(backend="sdpa", gqa_policy="repeat_kv"), helper_source_sha256=AUDIT["PRODUCER_HELPERS"])
    for key in AUDIT["PATH_KEYS"]:
        path = root / (key + ".json")
        if key == "retrieval_script":
            shutil.copyfile(api.__file__, path)
        else:
            path.write_text("{}\n")
        config[key], config[key + "_sha256"] = str(path), AUDIT["sha"](path)
    shutil.copyfile(HERE / "protocol.json", root / "protocol.json")
    hashes = {**AUDIT["PRODUCER_HELPERS"], "qwen_sign_shortcut_campaign.py": AUDIT["WORKER_SHA256"]}
    request = dict(config=config, source_hashes=hashes, task="synthetic-envelope-only")
    request["request_sha256"] = AUDIT["identity"](request)
    job = "job-tiny-envelope"
    source = dict(job_id=job, request_sha256=request["request_sha256"], source_hashes=hashes,
        helper_source_sha256=AUDIT["PRODUCER_HELPERS"], parameter_manifest_sha256=config["qwen_parameter_manifest_sha256"])
    plan = dict(config, output="/not-executed/" + job + "/result", request_source_hashes=hashes,
        request_sha256=request["request_sha256"], job_id=job)
    for name, value in (("request.json", request), ("plan.json", plan), ("source.json", source), ("declared-config.json", config),
                        ("summary.json", dict(schema=config["schema"], job_id=job, request_sha256=request["request_sha256"]))):
        AUDIT["save"](root / name, value)

    def finalize():
        path = root / "files.json"
        path.unlink(missing_ok=True)
        files = fixture_inventory(root)["files"]
        AUDIT["save"](path, dict(schema="mdb-image-tail-files-v1", job_id=job, request_sha256=request["request_sha256"], files=files,
            expanded_bytes=sum(row["bytes"] for row in files.values())))
        return AUDIT["sha"](path)

    return request, finalize


def test_standard_default_separator_request_and_bound_source_envelope(tmp_path):
    root = tmp_path / "envelope"
    request, finalize = envelope(root)
    expected = finalize()
    plan, source, _summary, _files = AUDIT["authenticate"](root, expected, request["request_sha256"])
    assert plan["job_id"] == source["job_id"] == "job-tiny-envelope"
    source["source_hashes"] = dict(source["source_hashes"], qwen_sign_shortcut_campaign_py="0" * 64)
    (root / "source.json").unlink()
    AUDIT["save"](root / "source.json", source)
    with pytest.raises(ValueError, match="source job/request/source joins"):
        AUDIT["authenticate"](root, finalize(), request["request_sha256"])


def test_changed_request_config_refuses_even_with_fresh_wrapper(tmp_path):
    root = tmp_path / "envelope"
    request, finalize = envelope(root)
    request["config"]["selected"] = [9, 18, 20]
    (root / "request.json").unlink()
    AUDIT["save"](root / "request.json", request)
    with pytest.raises(ValueError, match="standard executed request/config"):
        AUDIT["authenticate"](root, finalize(), request["request_sha256"])


def test_cli_exposes_saved_only_arguments_without_model_imports():
    result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "audit_qwen_sign_shortcut.py"), "--help"],
        check=True, capture_output=True, text=True)
    assert "--expected-manifest" in result.stdout and "--expected-request" in result.stdout


def test_actual_saved_accounts_run_with_neural_framework_imports_blocked():
    code = """import runpy,sys
from pathlib import Path
for name in ('torch','transformers','tokenizers'):
    sys.modules[name] = None
h = runpy.run_path('audit_qwen_sign_shortcut.py')
r = Path('saved-fixture')
s = h['read'](r/'summary.json')
sealed = h['read'](r/'forecast-responses.json')['cases']
for case in s['cases']:
    h['case_accounts'](r,case,edit_index=2,horizon=3,native_layers=5,sealed=sealed[case])
print('saved-account-frameworks-blocked')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=HERE, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "saved-account-frameworks-blocked"


def test_actual_shortcut_four_key_profile_joins_extended_campaign_runtime():
    # The four-key operand profile is actual producer serialization. The four
    # additional keys reproduce frozen83f9 runtime_contract's metadata schema;
    # this is not a claim of an observed CUDA runtime in the tiny fixture.
    row = AUDIT["read"](SAVED / "cases/tiny-objects/shortcut.json")
    execution = row["profile"]["execution"]
    runtime = dict(execution, cudnn_allow_tf32=False, cudnn_benchmark=False, cudnn_deterministic=True, num_threads=4)
    assert len(execution) == 4 and len(runtime) == 8
    report = AUDIT["execution_accounts"](execution, runtime)
    assert report["shortcut_bound"] == execution and len(report["campaign_additional"]) == 4
    wrong = dict(runtime, cuda_allow_tf32=not runtime["cuda_allow_tf32"])
    with pytest.raises(ValueError, match="primitive execution policy"):
        AUDIT["execution_accounts"](execution, wrong)
