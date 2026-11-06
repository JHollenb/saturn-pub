"""Actual tiny installed Qwen3 cohort, serializer and pretarget frontier checks."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

W = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_sign_shortcut_campaign.py"))


def fixture(root):
    root = Path(root)
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng():
        torch.manual_seed(824)
        config = Qwen3Config(vocab_size=23, hidden_size=6, intermediate_size=12, num_hidden_layers=5,
            num_attention_heads=4, num_key_value_heads=2, head_dim=2, max_position_embeddings=16,
            rope_theta=1_000_000, attention_bias=False, attention_dropout=0., pad_token_id=0,
            use_cache=False, tie_word_embeddings=True)
        model = Qwen3ForCausalLM(config).eval().to(torch.bfloat16).requires_grad_(False)
        model.config._attn_implementation = "sdpa"
        model.model.rotary_emb = type(model.model.rotary_emb)(config=config, device=torch.device("cpu"))
    cases = [dict(id="tiny-objects", input_ids=torch.tensor([[1, 3, 5, 0, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0, 0]]), metadata=dict(synthetic=True, intended_regime="padded")),
             dict(id="tiny-procedure", input_ids=torch.tensor([[2, 4, 6, 8, 0]]), attention_mask=torch.tensor([[1, 1, 1, 1, 0]]), metadata=dict(synthetic=True, intended_regime="padded")),
             dict(id="tiny-long", input_ids=torch.tensor([[7, 8, 9, 10, 11]]), attention_mask=torch.ones(1, 5, dtype=torch.int64), metadata=dict(synthetic=True, intended_regime="all-attended"))]
    helpers = W["helpers"]()
    kwargs = dict(graph=helpers["flux_functional_qwen_prefix.py"], history=helpers["qwen_h18_state_history.py"],
        shortcut=helpers["qwen_sign_shortcut.py"], edit_index=2, horizon=3, selected=(1, 2, 3),
        source_identity={"synthetic_fixture": "1" * 64})
    return root, model, cases, kwargs


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root, model, cases, kwargs = fixture(tmp_path_factory.mktemp("shortcut") / "result")
    before = W["parameter_signature"](model)
    result = W["execute_cases"](root, model, cases, **kwargs)
    assert before == W["parameter_signature"](model)
    return root, result


def test_actual_three_context_cohort_global_order_and_physical_counts(completed):
    root, result = completed
    assert result["counts"]["functional_reference_blocks"] == 12
    assert result["counts"]["native_blocks"] == 18
    assert result["counts"]["shortcut_down_projections"] == 3
    assert result["counts"]["native_authorizations"] == 6
    assert result["counts"]["forecast_module_attempts"] == 0
    events = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    seal = next(row["number"] for row in events if row["kind"] == "cohort_globally_sealed")
    assert len([row for row in events[:seal] if row["kind"] == "functional_case_completed"]) == 3
    assert all(row["native_module_entries"] == 0 for row in events[:seal + 1])
    assert min(row["number"] for row in events if row["kind"] == "native_component_authorized") > seal
    for case in result["cases"]:
        first = next(row["number"] for row in events if row["kind"] == "positive_only_shortcut_completed" and row["case"] == case)
        negative = next(row["number"] for row in events if row["kind"] == "functional_negative_reference_started" and row["case"] == case)
        assert first < negative < seal
    assert result["transaction"]["verified"] is True and result["transaction"]["phase"] == "closed"
    assert result["counts"]["functional_observed_SDPA"] == 12
    assert result["counts"]["native_observed_SDPA"] == 18
    assert result["counts"]["shortcut_observed_SDPA"] == 0
    assert result["counts"]["shortcut_observed_linear"] == 3


def test_actual_serialized_states_endpoints_responses_and_dispatch(completed):
    root, result = completed
    responses = W["read"](root / "forecast-responses.json")
    comparisons = W["read"](root / "comparisons.json")
    for case in result["cases"]:
        f, n, s = [W["read"](root / "cases" / case / name) for name in ("forecast.json", "native.json", "shortcut.json")]
        assert set(f["common"]) == {"0", "1", "2", "3"} and set(f["minus"]) == {"2", "3"}
        assert set(n["reference_only"]) == {"4", "5"}
        assert s["frontier"]["only_positive_basis"] == ["R", "gate", "up"]
        assert s["frontier"]["before_first_minus_callback"] is True
        for key in ("product", "M", "Hnext"):
            assert W["signed_hash"](root, s["ports"][key], f["ports"]["plus"][key]) == responses["cases"][case]["shortcut"][key]
            assert comparisons[case]["endpoint"]["shortcut-minus-native"][key]["absolute_rms"] == 0
            assert comparisons[case]["signed_response"]["shortcut-minus-native"][key]["error_rms"] == 0
        assert n["consumed"][0]["index"] == 2
        np.testing.assert_array_equal(W["load_array"](root, n["consumed"][0]["hidden"]["tensor"]), -W["load_array"](root, n["common"]["2"]))
        assert n["counts"]["SDPA"] == 6 and n["baseline_logits_shape"] == [1, 5, 23]
        padded = case != "tiny-long"
        assert f["profile"]["attention_dispatch"]["gqa_policy"] == ("repeat_kv" if padded else "enable_gqa")
        assert (n["source_context"]["attention_mask"] is not None) == padded


def test_sealed_external_corruption_refuses_before_any_native_entry(tmp_path):
    root, model, cases, kwargs = fixture(tmp_path / "corrupt")
    called = []
    handle = model.register_forward_pre_hook(lambda *_args: called.append(True))

    def corrupt(out, records):
        selected = next(key for key in records if key.endswith("shortcut/product.npy"))
        with (out / selected).open("ab") as stream:
            stream.write(b"corrupt")

    try:
        with pytest.raises(ValueError, match="sealed external file changed"):
            W["execute_cases"](root, model, cases, **kwargs, before_execute=corrupt)
        assert called == []
    finally:
        handle.remove()
    assert torch.nn.functional.scaled_dot_product_attention is kwargs["history"]["F"].scaled_dot_product_attention


def test_native_module_attempt_in_functional_callback_is_refused_and_hooks_restore(tmp_path):
    root, model, cases, kwargs = fixture(tmp_path / "guard")
    original = kwargs["history"]["functional_state_history"]
    hooks = [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()]

    def hostile(*args, **kw):
        model(input_ids=cases[0]["input_ids"], attention_mask=cases[0]["attention_mask"])
        return original(*args, **kw)

    kwargs["history"] = dict(kwargs["history"], functional_state_history=hostile)
    with pytest.raises(RuntimeError, match="native learned entry forbidden"):
        W["execute_cases"](root, model, cases, **kwargs)
    assert hooks == [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()]


def test_tensor_serializer_preserves_bf16_signed_zero_stride_and_integer_ids(tmp_path):
    value = torch.tensor([[[0., -0., 1.], [2., 3., 4.]]], dtype=torch.bfloat16).transpose(1, 2)
    row = W["tensor_record"](tmp_path, "bf16.npy", value)
    decoded = W["load_array"](tmp_path, row)
    assert row["stride"] == list(value.stride()) and row["stored_dtype"] == "uint16"
    assert np.signbit(decoded[0, 1, 0])
    ids = torch.tensor([[2**54 + 7, 0]], dtype=torch.int64)
    saved = W["tensor_record"](tmp_path, "ids.npy", ids)
    np.testing.assert_array_equal(W["load_array"](tmp_path, saved), ids.numpy())


def test_wrong_premise_remains_nonzero_observation_not_gate(completed, tmp_path):
    root, result = completed
    case = result["cases"][0]
    s = W["read"](root / "cases" / case / "shortcut.json")
    actual = W["load_array"](root, s["ports"]["Hnext"])
    wrong = torch.from_numpy(actual.copy()).to(torch.bfloat16)
    wrong[..., 0] += .5
    predicted = W["tensor_record"](tmp_path, "prediction.npy", torch.from_numpy(actual).to(torch.bfloat16))
    record = W["tensor_record"](tmp_path, "wrong-premise-reference.npy", wrong)
    measured = W["compare"](tmp_path, predicted, record)
    assert measured["absolute_rms"] > 0 and measured["changed_numeric"] > 0
    assert result["mechanics_status"] == "completed"


def test_filtered_config_and_exact_production_count_budget():
    base = {key: "metadata/" + key for key in W["PATH_KEYS"]}
    base.update({key + "_sha256": "1" * 64 for key in W["PATH_KEYS"]})
    base.update({key: "2" * 64 for key in W["RUNTIME_KEYS"]})
    base.update(parameter_root="/checkpoint", tokenizer_path="/checkpoint/tokenizer", flux_config="DROP", vae_config="DROP")
    base["retrieval_script_sha256"] = W["RETRIEVER_SHA256"]
    config = W["make_config"](base)
    counts = W["validate_config"](config)
    assert not any(key.startswith(("flux_", "vae_")) for key in config)
    assert counts["functional_reference_blocks"] == 60 and counts["native_blocks"] == 111
    assert W["helpers"]()["qwen_sign_shortcut.py"]["comparison_element_budget"](
        dict(product=[1, 512, 9728], M=[1, 512, 2560], Hnext=[1, 512, 2560])) == 7_602_176
    wrong = dict(config, selection_protocol_sha256="0" * 64)
    with pytest.raises(ValueError, match="selection differs"):
        W["validate_config"](wrong)
    assert hashlib.sha256(W["canonical"](config)).hexdigest()


def test_actual_protocol_serializer_names_and_literal_byte_custody(tmp_path):
    selection, output = tmp_path / "selection", tmp_path / "output"
    selection.mkdir()
    output.mkdir()
    rows = []
    for index, text in enumerate(("one", "two", "three")):
        path = selection / f"raw-{index}.txt"
        path.write_bytes(text.encode())
        rows.append(dict(case_id=f"case-{index}", raw_file=path.name, raw_text=text,
            raw_bytes=len(text.encode()), raw_sha256=W["sha"](path), intended_mask_regime="padded"))
    protocol = selection / "protocol.json"
    W["save"](protocol, dict(cases=rows))
    actual = W["selected_raw_inputs"](protocol, output, expected_sha256=W["sha"](protocol))
    assert actual == rows
    assert (output / "protocol.json").read_bytes() == protocol.read_bytes()
    assert [row["case_id"] for row in actual] == ["case-0", "case-1", "case-2"]
    (selection / rows[0]["raw_file"]).write_bytes(b"bad")
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    with pytest.raises(ValueError, match="literal predeclared raw text"):
        W["selected_raw_inputs"](protocol, hostile, expected_sha256=W["sha"](protocol))


def test_request_bound_worker_uses_real_finalizer_directory_marker_and_experimental_identity(tmp_path, monkeypatch, capfd):
    base = {key: "metadata/" + key for key in W["PATH_KEYS"]}
    base.update({key + "_sha256": "1" * 64 for key in W["PATH_KEYS"]})
    base.update({key: "2" * 64 for key in W["RUNTIME_KEYS"]})
    base.update(parameter_root="/checkpoint", tokenizer_path="/checkpoint/tokenizer")
    base["retrieval_script_sha256"] = W["RETRIEVER_SHA256"]
    config = W["make_config"](base)
    config["output_scratch"] = str(tmp_path / "output")
    source = tmp_path / "stub-source.py"
    source.write_bytes(b"# fixture-only executed source identity\n")
    config_path = tmp_path / "config.json"
    W["save"](config_path, config)
    request = dict(schema="synthetic-request", config=config, source_hashes={source.name: W["sha"](source)})
    request["request_sha256"] = W["request_identity"](request)
    request_path = tmp_path / "request.json"
    W["save"](request_path, request)
    monkeypatch.setenv("SATURN_DEBUG_REQUEST", str(request_path))
    monkeypatch.setenv("MRUN_JOB_ID", "job-tiny-bootstrap")
    monkeypatch.setenv("MRUN_REQUEST_SHA256", "DO-NOT-USE-NORMALIZED-SCHEDULER-FINGERPRINT")
    observed = []

    def fake_science(derived):
        observed.append(derived)
        output = Path(derived["output"])
        output.mkdir(parents=True)
        W["save"](output / "source.json", dict(job_id=derived["job_id"], request_sha256=derived["request_sha256"]))
        W["save"](output / "summary.json", dict(job_id=derived["job_id"], request_sha256=derived["request_sha256"], scope="model-free worker envelope fixture only"))

    monkeypatch.setitem(W["worker"].__globals__, "run", fake_science)
    ready = W["worker"](config_path)
    assert len(observed) == 1
    assert ready["job_id"] == "job-tiny-bootstrap" and ready["request_sha256"] == request["request_sha256"]
    output = Path(observed[0]["output"])
    files = W["read"](output / "files.json")
    assert files["schema"] == "mdb-image-tail-files-v1"
    assert files["job_id"] == ready["job_id"] and files["request_sha256"] == request["request_sha256"]
    assert W["read"](output / "summary.json")["request_sha256"] == request["request_sha256"]
    assert "MDB_DIRECTORY_READY " in capfd.readouterr().out
    assert W["read"](output / "declared-config.json") == request["config"]
    request["config"] = dict(config, num_threads=3)
    request_path.write_bytes(W["canonical"](request))
    with pytest.raises(ValueError, match="canonical experimental request"):
        W["worker"](config_path)
    W["worker"].__globals__["faulthandler"].cancel_dump_traceback_later()


def test_runtime_sources_match_original_loader_module_paths_without_original_assets():
    import inspect

    import jinja2.environment
    import jinja2.sandbox
    import tokenizers
    import tokenizers.tokenizers
    import transformers.models.qwen3.modeling_qwen3

    modules = dict(qwen_native_source_sha256=transformers.models.qwen3.modeling_qwen3,
        tokenizers_wrapper_source_sha256=tokenizers, tokenizers_native_source_sha256=tokenizers.tokenizers,
        jinja_environment_source_sha256=jinja2.environment, jinja_sandbox_source_sha256=jinja2.sandbox)
    config = {key: W["sha"](inspect.getfile(module)) for key, module in modules.items()}
    config["tokenizers_native_source_sha256"] = "runtime-bound"
    actual = W["runtime_contract"](config)
    assert actual["sources"]["tokenizers_wrapper_source_sha256"]["path"] == inspect.getfile(tokenizers)
    assert actual["sources"]["tokenizers_native_source_sha256"]["binding"] == "runtime-observed-on-CUDA-host"
    assert actual["sources"]["qwen_native_source_sha256"]["sha256"] == config["qwen_native_source_sha256"]
