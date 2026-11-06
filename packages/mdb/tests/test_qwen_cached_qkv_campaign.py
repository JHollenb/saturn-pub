"""Tiny installed Qwen CPU, authentic saved source layouts and actual panel ABI."""
from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

PRODUCER = Path(__file__).parents[1] / "qualification/qwen_cached_qkv_campaign.py"


def prepare(root):
    m = runpy.run_path(str(PRODUCER)); c = m["C"]; row = m["ROW"]
    h = row["helpers"](); graph, history = h["flux_functional_qwen_prefix.py"], h["qwen_h18_state_history.py"]
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision("highest")
    with torch.random.fork_rng():
        torch.manual_seed(326)
        config = Qwen3Config(vocab_size=23, hidden_size=6, intermediate_size=12, num_hidden_layers=19,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=16,
            rope_theta=1_000_000, attention_bias=False, attention_dropout=0., use_cache=False)
        config._attn_implementation = "sdpa"
        layer = Qwen3DecoderLayer(config, layer_idx=18).to(torch.bfloat16).eval().requires_grad_(False)
        hidden = torch.randn(1, 8, 6, dtype=torch.bfloat16)
        hidden[0, -1] = 0; hidden[0, -1, 0] = -0.
    ids, mask = torch.arange(8, dtype=torch.long).unsqueeze(0), torch.ones(1, 8, dtype=torch.long)
    dispatch = graph["AttentionDispatch"].from_mapping(dict(backend="sdpa", gqa_policy="enable_gqa", source=row["NATIVE_SHA256"]))
    profile = graph["_native_profile"](config, 19)
    context = history["_context"](graph, ids, mask, profile, dispatch, row["ROPE_POLICY"], torch.bfloat16)
    positive = {}
    def observe(_lane, name, value):
        if name in m["SOURCE_KEYS"]:
            positive[name] = value.detach().clone(memory_format=torch.preserve_format)
    with torch.inference_mode(), history["_no_modules"]():
        history["_block"](graph, hidden, {"model.layers.18." + name: value for name, value in layer.named_parameters()},
            profile, 18, context, dispatch, observe, "separate-parent-preparation")
    rotary = Qwen3RotaryEmbedding(config=config, device=torch.device("cpu"))
    positive.update(input_ids=ids, attention_mask=mask, position_ids=context[2])
    positive.update({"rope_buffer:model.rotary_emb." + name: getattr(rotary, name).detach().clone() for name in ("inv_freq", "original_inv_freq")})
    parent = root / "synthetic-parent"; parent.mkdir()
    arrays = {key: c["tensor_record"](parent, "arrays/" + key.replace(":", "-") + ".npy", value) for key, value in positive.items()}
    for name in ("q", "k", "v"):
        assert arrays[name]["stride"][1] == 4 and arrays[name]["stride"][2] > 4
    metadata_names = dict(qwen_config="qwen-config.json", parent_parameter_manifest="qwen-parameter-artifacts.json",
        qwen_source_binding="qwen-source-binding.json", retrieval_script="retrieve_qwen_weights.py", parent_source="source.json", parent_runtime="runtime.json",
        parent_forecast="cases/synthetic/forecast.json")
    config_paths, parent_files = dict(input_root=str(parent)), {}
    for key, name in metadata_names.items():
        payload = dict(synthetic=True, name=name)
        if key == "parent_forecast":
            payload = dict(ports=dict(plus={key: arrays[key] for key in ("h", "q", "k", "v", *c["PORTS"])}), position_ids=arrays["position_ids"])
        c["save"](parent / name, payload); config_paths[key] = str(parent / name)
        parent_files[name] = dict(bytes=(parent / name).stat().st_size, sha256=c["sha"](parent / name))
    external = {**parent_files, **{item["path"]: dict(bytes=item["bytes"], sha256=item["sha256"]) for item in arrays.values()}}
    request = dict(config=dict(synthetic=True), source_hashes={})
    request["request_sha256"] = row["BASE"]["request_identity"](request)
    c["save"](parent / "request.json", request)
    c["save"](parent / "files.json", dict(job_id="job-synthetic-parent", request_sha256=request["request_sha256"], files=external))
    c["save"](parent / "seal.json", dict(schema="mdb-global-forecast-seal-v1", violations=[], pre_release_work=dict(native_calls=0), declarations=dict(frontier=dict(external_files=external))))
    config_paths.update(parent_request=str(parent / "request.json"), parent_files=str(parent / "files.json"), parent_seal=str(parent / "seal.json"))
    signs = {"last-row-negative": [1] * 7 + [-1], "alternating64": [1, 1, -1, -1, 1, 1, -1, -1]}
    arms = []
    for name, sign in signs.items():
        s = np.asarray(sign)
        cross = np.tri(8, dtype=bool) & (s[:, None] != s[None, :])
        arms.append(dict(id=name, row_signs=sign, windows=[[0, 7], [7, 8]] if name == "last-row-negative" else [[0, 2], [2, 4], [4, 6], [6, 8]],
            structural_support_closed_rows=int((~cross.any(-1)).sum()), permitted_cross_sign_edges=int(cross.sum())))
    protocol = dict(schema="mdb-qwen-cached-qkv-selection-v1", edit_index=18, horizon=19, sequence_rows=8, source_case="synthetic", source_arrays=arrays, arms=arms,
        parent_raw_root=str(parent), parent_job="job-synthetic-parent", parent_request_sha256=request["request_sha256"],
        parent_raw_manifest_sha256=c["sha"](config_paths["parent_files"]), parent_global_seal_sha256=c["sha"](config_paths["parent_seal"]), physical_plan=m["PHYSICAL_PLAN"])
    output = root / m["RESULT"]; output.mkdir()
    sources, references, layouts, verify = c["inherited_sources"](config_paths, protocol, torch.device("cpu"))
    c["save"](output / "layout-restoration.json", layouts); c["save"](output / "protocol.json", protocol)
    c["save"](output / "plan.json", dict(schema=m["SCHEMA"], qualification="tiny synthetic CPU", input_root=str(parent), sequence_rows=8))
    c["save"](output / "source.json", dict(synthetic_component=True, helper_source_sha256=m["HELPERS"], frontier=c["FRONTIER_SCOPE"],
        parameter_payloads=c["identities"](layer, 18)["payloads"], metadata={key: dict(path=value, sha256=c["sha"](value)) for key, value in config_paths.items() if key != "input_root"}))
    c["save"](output / "runtime-contract.json", row["runtime_contract"](dict(qwen_native_source_sha256=row["NATIVE_SHA256"], sdpa_source_sha256=row["SDPA_SHA256"])))
    return m, c, layer, sources, references, protocol, verify, output


def write_fixture(root, *, before_execute=None, alternate_callbacks=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    m, c, layer, sources, references, protocol, verify, output = prepare(root)
    callbacks, expected = m["callbacks"](), m["contract"]()
    aliases = dict(cached="cached_attention_downstream_evaluations", functional="full_functional_edited_blocks")
    if alternate_callbacks:
        callbacks = dict(candidate=callbacks["cached"], direct=callbacks["functional"])
        rename = dict(cached_SDPA="candidate_SDPA", cached_linear="candidate_linear", cached_attention_downstream_evaluations="candidate_updates",
            functional_SDPA="direct_SDPA", functional_linear="direct_linear", full_functional_edited_blocks="direct_blocks")
        expected = {rename.get(key, key): value for key, value in expected.items()}
        aliases = dict(candidate="candidate_updates", direct="direct_blocks")
    summary = c["execute_panel"](output, layer, sources, references, protocol, callbacks=callbacks, expected_counts=expected, evaluation_counts=aliases,
        source_identity={**m["HELPERS"], "qwen_native": m["ROW"]["NATIVE_SHA256"], "sdpa_native": m["ROW"]["SDPA_SHA256"]},
        source_metadata=dict(input_root=protocol["parent_raw_root"], source_case="synthetic", **{key: protocol[key] for key in ("parent_job", "parent_request_sha256", "parent_raw_manifest_sha256", "parent_global_seal_sha256")}),
        verify_sources=verify, before_execute=before_execute, job_id="job-tiny-cached-qkv", request_sha256="1" * 64, result_schema=m["SCHEMA"])
    c["save"](output / "request.json", dict(qualification="tiny synthetic CPU", request_sha256="1" * 64))
    c["save"](output / "declared-config.json", dict(schema=m["SCHEMA"], qualification="tiny synthetic CPU"))
    receipt = m["ROW"]["helpers"]()["flux_image_tail_microscope.py"]["finalize"](output, job_id="job-tiny-cached-qkv", request_sha256="1" * 64,
        max_expanded_bytes=2 << 30, max_member_bytes=64 << 20)
    c["save"](root / "qualification.json", dict(qualification="tiny installed native Qwen CPU; synthetic separate parent; no original assets/GPU/scheduler",
        sources={name: c["sha"](PRODUCER.with_name(name)) for name in ("qwen_component_experiment.py", "qwen_cached_qkv.py", "qwen_cached_qkv_campaign.py")},
        tests={Path(name).name: c["sha"](Path(__file__).with_name(name)) for name in ("test_qwen_component_experiment.py", "test_qwen_cached_qkv_campaign.py")},
        summary=summary, receipt=receipt))
    return m, c, output, receipt


def test_actual_panel_layouts_signed_zeros_census_and_global_seal(tmp_path):
    m, c, output, receipt = write_fixture(tmp_path)
    summary = c["read"](output / "summary.json")
    assert all(summary["counts"][key] == value for key, value in m["contract"]().items())
    layouts = c["read"](output / "layout-restoration.json")["arrays"]
    for name in ("q", "k", "v"):
        assert layouts[name]["loaded_logical"]["stride"] != layouts[name]["restored"]["stride"]
        assert layouts[name]["value_bit_exact"]
    forecast, native = (c["read"](output / name) for name in ("forecast.json", "native.json"))
    assert native["head_to_canonical"] == [0, 2, 1, 3]
    calls = c["read"](output / "sdpa.json")["calls"]
    assert len(calls) == 7 and sum(row["lane"] == "native" for row in calls) == 3
    for call in calls:
        assert call["attention_mask"] is None and call["is_causal"] and call["enable_gqa"] and call["dropout_p"] == 0
        for port, operand in zip(call["QKV"], call["operands"]):
            assert port["content_sha256"] == operand["content_sha256"] and port["stride"] == operand["stride"]
            assert (output / port["path"]).is_file()
    for arm, row in forecast["arms"].items():
        assert row["windows"] and set(row["lanes"]) == {"cached", "functional"}
        for name in ("q", "k", "v"):
            record = row["lanes"]["cached"]["ports"][name]
            assert record["stride"] == forecast["positive"][name]["stride"]
        # Exact real signed-zero inputs survive the stored/restored source path.
        q = m["ROW"]["BASE"]["load_array"](output, row["lanes"]["cached"]["ports"]["q"])
        assert np.count_nonzero((q[:, :, -1] == 0) & np.signbit(q[:, :, -1])) > 0
    events = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    seal = next(row["number"] for row in events if row["kind"] == "cohort_globally_sealed")
    assert all(row["number"] > seal for row in events if row["kind"] == "native_component_authorized")
    assert receipt["files_manifest_sha256"] == c["sha"](output / "files.json")
    assert len(c["read"](output / "source-frontier.json")["source_arrays"]) == 13


def test_named_callbacks_and_declared_windows_are_reusable(tmp_path):
    _m, c, output, _receipt = write_fixture(tmp_path, alternate_callbacks=True)
    arms = c["read"](output / "forecast.json")["arms"]
    assert all(set(arm["lanes"]) == {"candidate", "direct"} for arm in arms.values())
    assert len(arms["last-row-negative"]["windows"]) == 2 and len(arms["alternating64"]["windows"]) == 4
    summary = c["read"](output / "summary.json")
    assert summary["counts"]["candidate_updates"] == summary["counts"]["direct_blocks"] == 2


def test_corrupted_sealed_forecast_prevents_native_targets(tmp_path):
    def corrupt(output, _sealed):
        (output / "forecast.json").write_text("{}\n")
    with pytest.raises(ValueError, match="sealed external file changed"):
        write_fixture(tmp_path, before_execute=corrupt)
    assert "native_component_authorized" not in (tmp_path / "qwen-cached-qkv-result/timeline.jsonl").read_text()


def test_original_source_file_drift_prevents_native_targets(tmp_path):
    def corrupt(output, _sealed):
        source = output.parent / "synthetic-parent/arrays/q.npy"
        source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="inherited source file changed"):
        write_fixture(tmp_path, before_execute=corrupt)


def test_illegal_forecast_module_entry_is_denied(tmp_path):
    m, c, layer, sources, references, protocol, verify, output = prepare(tmp_path)
    def illegal(*_args):
        layer(sources["h"])
    with pytest.raises(RuntimeError, match="native module forward forbidden"):
        c["execute_panel"](output, layer, sources, references, protocol, callbacks=dict(illegal=illegal), expected_counts=m["contract"](),
            source_identity=dict(synthetic="1" * 64), verify_sources=verify)
    assert not (output / "native.json").exists()


def test_parent_forecast_layout_descriptor_is_authoritative(tmp_path):
    _m, c, _layer, _sources, _refs, protocol, _verify, output = prepare(tmp_path)
    parent = output.parent / "synthetic-parent"
    protocol["source_arrays"]["q"]["stride"] = [128, 32, 4, 1]
    paths = dict(input_root=str(parent), parent_files=str(parent / "files.json"), parent_request=str(parent / "request.json"), parent_seal=str(parent / "seal.json"),
        parent_forecast=str(parent / "cases/synthetic/forecast.json"), parent_source=str(parent / "source.json"), parent_runtime=str(parent / "runtime.json"),
        parent_parameter_manifest=str(parent / "qwen-parameter-artifacts.json"), qwen_config=str(parent / "qwen-config.json"),
        qwen_source_binding=str(parent / "qwen-source-binding.json"), retrieval_script=str(parent / "retrieve_qwen_weights.py"))
    with pytest.raises(ValueError, match="source/layout descriptor differs"):
        c["inherited_sources"](paths, protocol, torch.device("cpu"))
