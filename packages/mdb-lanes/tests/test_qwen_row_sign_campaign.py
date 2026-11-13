"""Installed-package CPU mechanics for the actual conditional component writer."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

PRODUCER = Path(__file__).parents[1] / "qualification/qwen_row_sign_campaign.py"


def fixture(root):
    m = runpy.run_path(str(PRODUCER))
    h = m["helpers"]()
    graph, history = h["flux_functional_qwen_prefix.py"], h["qwen_h18_state_history.py"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    with torch.random.fork_rng():
        torch.manual_seed(319)
        config = Qwen3Config(vocab_size=23, hidden_size=6, intermediate_size=12, num_hidden_layers=19,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=16,
            rope_theta=1_000_000, attention_bias=False, attention_dropout=0., use_cache=False)
        config._attn_implementation = "sdpa"
        layer = Qwen3DecoderLayer(config, layer_idx=18).to(torch.bfloat16).eval().requires_grad_(False)
        hidden = torch.randn(1, 6, 6, dtype=torch.bfloat16)
    ids = torch.arange(6, dtype=torch.long).unsqueeze(0)
    mask = torch.ones_like(ids)
    dispatch = graph["AttentionDispatch"].from_mapping(dict(backend="sdpa", gqa_policy="enable_gqa", source=m["NATIVE_SHA256"]))
    profile = graph["_native_profile"](config, 19)
    context = history["_context"](graph, ids, mask, profile, dispatch, m["ROPE_POLICY"], torch.bfloat16)
    ports = {}
    def observe(_lane, name, value):
        if name in {"h", "head", "R", "gate", "up", "product", "M", "Hnext"}:
            ports[name] = value.detach().clone()
    parameters = {"model.layers.18." + name: value for name, value in layer.named_parameters()}
    with torch.inference_mode(), history["_no_modules"]():
        history["_block"](graph, hidden, parameters, profile, 18, context, dispatch, observe, "parent-positive")
    # Parent preparation is separate from the new experiment's physical census.
    rotary = Qwen3RotaryEmbedding(config=config, device=torch.device("cpu"))
    sources = dict(ports, input_ids=ids, attention_mask=mask, position_ids=context[2])
    sources.update({"rope_buffer:model.rotary_emb." + name: getattr(rotary, name).detach().clone()
        for name in ("inv_freq", "original_inv_freq")})
    parent = root / "synthetic-parent"
    parent.mkdir()
    references = {key: dict(m["tensor_record"](parent, "arrays/" + key.replace(":", "-") + ".npy", value), inherited=True, root="input_root")
        for key, value in sources.items()}
    output = root / "qwen-row-sign-result"
    output.mkdir()
    m["save"](output / "plan.json", dict(schema=m["SCHEMA"], qualification="tiny synthetic CPU; no original parameters", source_rows=6, fixed_split=3))
    m["save"](output / "source.json", dict(synthetic_component=True, frontier=m["FRONTIER_SCOPE"], helper_source_sha256=m["HELPERS"], parameter_payloads=m["identities"](layer, 18)["payloads"]))
    m["save"](output / "runtime-contract.json", m["runtime_contract"](dict(qwen_native_source_sha256=m["NATIVE_SHA256"], sdpa_source_sha256=m["SDPA_SHA256"])))
    return m, layer, sources, references, output, parent


def write_fixture(root, *, before_execute=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    m, layer, sources, references, output, parent = fixture(root)
    result = m["execute_component"](output, layer, sources, references, split=3, source_metadata=dict(input_root=str(parent), source_case="synthetic-parent"),
        source_identity=dict(qwen_native=m["NATIVE_SHA256"], sdpa_native=m["SDPA_SHA256"]), before_execute=before_execute,
        job_id="job-tiny-row-sign", request_sha256="1" * 64)
    m["save"](output / "request.json", dict(qualification="tiny synthetic CPU", request_sha256="1" * 64))
    m["save"](output / "declared-config.json", dict(schema=m["SCHEMA"], qualification="tiny synthetic CPU"))
    receipt = m["helpers"]()["flux_image_tail_microscope.py"]["finalize"](output, job_id="job-tiny-row-sign", request_sha256="1" * 64,
        max_expanded_bytes=2 << 30, max_member_bytes=64 << 20)
    m["save"](root / "qualification.json", dict(qualification="tiny installed native component CPU mechanics; no original parameters/GPU/scheduler",
        producer_sha256=m["sha"](PRODUCER), test_sha256=m["sha"](__file__), receipt=receipt, summary=result))
    return m, output, receipt


def test_actual_writer_counts_global_seal_and_component_context(tmp_path):
    m, output, receipt = write_fixture(tmp_path)
    summary, native = (m["read"](output / name) for name in ("summary.json", "native.json"))
    expected = dict(functional_blocks=2, shortcut_down_projections=2, native_blocks=3, functional_SDPA=2,
        shortcut_SDPA=0, native_SDPA=3, functional_linear=14, shortcut_linear=2, native_linear=21,
        native_authorizations=3, forecast_module_attempts=0)
    assert all(summary["counts"][key] == value for key, value in expected.items())
    assert native["head_to_canonical"] == [0, 2, 1, 3]
    assert native["baseline"]["ports"]["head"]["shape"] == [1, 4, 6, 4]
    events = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    seal = next(row["number"] for row in events if row["kind"] == "cohort_globally_sealed")
    assert all(row["number"] > seal for row in events if row["kind"] == "native_component_authorized")
    assert summary["frontier"] == m["FRONTIER_SCOPE"]
    assert not any((output / row["path"]).exists() for row in m["read"](output / "source-frontier.json")["source_arrays"].values())
    entries = m["read"](output / "native-contract.json")["block_entries"]
    for row in entries:
        assert row["context"] == native["context"]
        for name in ("position_ids", "cos", "sin"):
            assert row["consumed_context"][name]["content_sha256"] == native["context"][name]["content_sha256"]
    calls = m["read"](output / "native-sdpa.json")["calls"]
    assert len(calls) == 3 and all(row["attention_mask"] is None and row["is_causal"] and row["enable_gqa"] and row["dropout_p"] == 0 for row in calls)
    assert receipt["files_manifest_sha256"] == m["sha"](output / "files.json")
    responses = m["read"](output / "forecast-responses.json")
    assert all(set(lanes) == {"shortcut", "functional"} and all(set(ports) == set(m["PORTS"]) for ports in lanes.values()) for lanes in responses["arms"].values())
    assert m["read"](output / "shortcut.json")["arms"]["prefix-negative"]["support"] == dict(closed_rows=3, mixed_or_empty_rows=3, total_rows=6, empty_rows=0, use="diagnostic only; all complete output rows retained")


def test_changed_pretarget_file_prevents_native_component(tmp_path):
    def tamper(output, _sealed):
        (output / "forecast.json").write_text("{}\n")
    with pytest.raises(ValueError, match="sealed external file changed"):
        write_fixture(tmp_path, before_execute=tamper)
    events = (tmp_path / "qwen-row-sign-result/timeline.jsonl").read_text()
    assert "native_component_authorized" not in events


def test_forecast_module_entry_is_denied(tmp_path, monkeypatch):
    m, layer, sources, references, output, parent = fixture(tmp_path)
    history = m["helpers"]()["qwen_h18_state_history.py"]
    def illegal(*_args, **_kwargs):
        return layer(sources["h"])
    monkeypatch.setitem(history, "_block", illegal)
    with pytest.raises(RuntimeError, match="native module forward forbidden"):
        m["execute_component"](output, layer, sources, references, split=3, source_metadata=dict(input_root=str(parent)))
    assert not (output / "native.json").exists()


def test_inherited_loader_authenticates_exact_thirteen_and_stable_files(tmp_path, monkeypatch):
    import numpy as np

    m, _layer, sources, references, _output, parent = fixture(tmp_path)
    config, files = dict(input_root=str(parent)), {}
    metadata_names = {"parent_source": "source.json", "parent_runtime": "runtime.json", "parent_parameter_manifest": "qwen-parameter-artifacts.json",
        "qwen_config": "qwen-config.json", "qwen_source_binding": "qwen-source-binding.json", "retrieval_script": "retrieve_qwen_weights.py"}
    for key, name in metadata_names.items():
        path = parent / name
        m["save"](path, dict(synthetic=True, name=name))
        config[key] = str(path)
        files[name] = dict(bytes=path.stat().st_size, sha256=m["sha"](path))
    selected = {key: {k: value for k, value in row.items() if k not in ("inherited", "root")} for key, row in references.items()}
    arrays = {row["path"]: dict(bytes=row["bytes"], sha256=row["sha256"]) for row in selected.values()}
    files.update(arrays)
    request = dict(config=dict(synthetic=True), source_hashes={})
    request["request_sha256"] = m["BASE"]["request_identity"](request)
    m["save"](parent / "request.json", request)
    m["save"](parent / "files.json", dict(job_id="job-synthetic-parent", request_sha256=request["request_sha256"], files=files))
    m["save"](parent / "seal.json", dict(schema="mdb-global-forecast-seal-v1", violations=[], pre_release_work=dict(native_calls=0),
        declarations=dict(frontier=dict(external_files=arrays))))
    config.update(parent_request=str(parent / "request.json"), parent_files=str(parent / "files.json"), parent_seal=str(parent / "seal.json"))
    protocol = dict(parent_raw_root=str(parent), parent_raw_manifest_sha256=m["sha"](config["parent_files"]),
        parent_global_seal_sha256=m["sha"](config["parent_seal"]), parent_request_sha256=request["request_sha256"],
        parent_job="job-synthetic-parent", source_arrays=selected)
    reads, original_load = [], np.load
    def load(path, **kwargs):
        reads.append(Path(path).relative_to(parent).as_posix())
        return original_load(path, **kwargs)
    monkeypatch.setattr(np, "load", load)
    loaded, inherited, verify = m["inherited_sources"](config, protocol, torch.device("cpu"))
    assert len(reads) == 13 and set(reads) == set(arrays) and inherited == references
    assert all(torch.equal(value, sources[key]) for key, value in loaded.items())
    verify()
    path = parent / references["h"]["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="inherited source changed"):
        verify()
