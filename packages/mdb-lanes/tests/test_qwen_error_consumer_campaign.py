"""Actual tiny two-world Qwen writer, public transaction and source mechanics."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

PRODUCER = Path(__file__).parents[1] / "qualification/qwen_error_consumer_campaign.py"


def prepare(root):
    m = runpy.run_path(str(PRODUCER)); c = m["C"]
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision("highest")
    with torch.random.fork_rng():
        torch.manual_seed(481)
        config = Qwen3Config(vocab_size=23, hidden_size=6, intermediate_size=12, num_hidden_layers=20,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=16,
            rope_theta=1_000_000, attention_bias=False, attention_dropout=0., use_cache=False)
        config._attn_implementation = "sdpa"
        layer = Qwen3DecoderLayer(config, layer_idx=19).to(torch.bfloat16).eval().requires_grad_(False)
        full = torch.randn(1, 8, 6, dtype=torch.bfloat16)
        cheap = full.clone(); cheap[:, 4:] += torch.tensor(.5, dtype=torch.bfloat16)
        full[0, 0, 0] = cheap[0, 0, 0] = -0.
    rotary = Qwen3RotaryEmbedding(config=config, device=torch.device("cpu"))
    ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
    context = dict(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=ids.clone(),
        **{"rope_buffer:model.rotary_emb." + name: getattr(rotary, name).clone() for name in ("inv_freq", "original_inv_freq")})
    protocol = dict(component_index=19, horizon=20, sequence_rows=8, world_order=["full-input", "cheap-input"], baseline_world="full-input",
        response=dict(left_world="cheap-input", right_world="full-input", orientation="cheap-input-minus-full-input", ports=list(c["PORTS"])),
        windows=[[0, 4], [4, 8]], physical_plan=c["contract"](), roots={}, shared_context_sources={}, world_sources={})
    config_paths = {}
    for alias in ("world", "context"):
        parent = root / ("synthetic-" + alias); parent.mkdir()
        config_paths[alias + "_root"] = str(parent)
        if alias == "world":
            for name, value in (("full-input", full), ("cheap-input", cheap)):
                lane = "forecast" if name == "full-input" else "shortcut"
                record = c["tensor_record"](parent, lane + "/prefix-negative/Hnext.npy", value)
                protocol["world_sources"][name] = record
                c["save"](parent / (lane + ".json"), dict(arms={"prefix-negative": dict(ports=dict(Hnext=record))}))
                config_paths["world_" + lane] = str(parent / (lane + ".json"))
        else:
            protocol["shared_context_sources"] = {name: c["tensor_record"](parent, "arrays/" + name.replace(":", "-") + ".npy", value) for name, value in context.items()}
        external = c["inventory"](parent)
        request = dict(config=dict(synthetic=True, alias=alias), source_hashes={})
        request["request_sha256"] = c["ROW"]["BASE"]["request_identity"](request)
        c["save"](parent / "request.json", request)
        c["save"](parent / "files.json", dict(job_id="job-synthetic-" + alias, request_sha256=request["request_sha256"], files=external))
        c["save"](parent / "seal.json", dict(schema="mdb-global-forecast-seal-v1", violations=[], pre_release_work=dict(native_calls=0), declarations=dict(frontier=dict(external_files=external))))
        config_paths.update({alias + "_" + key: str(parent / (key + ".json")) for key in ("files", "seal", "request")})
        protocol["roots"][alias] = dict(root=str(parent), job_id="job-synthetic-" + alias, request_sha256=request["request_sha256"],
            raw_manifest_sha256=c["sha"](parent / "files.json"), global_seal_sha256=c["sha"](parent / "seal.json"))
    output = root / m["RESULT"]; output.mkdir()
    tensors, references, layouts, verify = c["load_sources"](config_paths, protocol, torch.device("cpu"))
    c["save"](output / "layout-restoration.json", layouts); c["save"](output / "protocol.json", protocol)
    c["save"](output / "plan.json", dict(qualification="synthetic CPU component only; no original assets", roots=protocol["roots"]))
    c["save"](output / "source.json", dict(synthetic=True, helper_source_sha256=m["HELPERS"], parameter_payloads=c["identities"](layer, 19)["payloads"]))
    return m, c, layer, tensors, references, protocol, verify, output, config_paths


def write_fixture(root, **kwargs):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    m, c, layer, tensors, references, protocol, verify, output, paths = prepare(root)
    summary = c["execute_worlds"](output, layer, {name: tensors[name] for name in protocol["world_order"]},
        {name: tensors[name] for name in m["CONTEXT_KEYS"]}, protocol, references=references, source_identity=m["HELPERS"],
        source_metadata=dict(roots=protocol["roots"]), verify_sources=verify, job_id="job-tiny-worlds", request_sha256="1" * 64, **kwargs)
    receipt = c["ROW"]["helpers"]()["flux_image_tail_microscope.py"]["finalize"](output, job_id="job-tiny-worlds", request_sha256="1" * 64,
        max_expanded_bytes=2 << 30, max_member_bytes=64 << 20)
    c["save"](root / "fixture.json", dict(scope="actual installed native tiny Qwen CPU, synthetic two-source inventories/envelopes; no original model or production custody claim",
        sources={name: c["sha"](PRODUCER.with_name(name)) for name in ("qwen_component_experiment_v2.py", "qwen_error_consumer_campaign.py")}, summary=summary, receipt=receipt))
    return m, c, output, receipt, paths


def test_actual_two_world_ports_census_responses_and_seal(tmp_path):
    m, c, output, receipt, _paths = write_fixture(tmp_path)
    summary = c["read"](output / "summary.json")
    assert summary["counts"] == c["contract"]() and summary["counts"]["native_module_entries"] == 30
    forecasts, natives = (c["read"](output / name)["worlds"] for name in ("forecast.json", "native.json"))
    load = c["ROW"]["BASE"]["load_array"]
    responses = c["read"](output / "forecast-responses.json")
    for name in c["PORTS"]:
        for world in ("full-input", "cheap-input"):
            f = load(output, forecasts[world]["ports"][name]); n = load(output, natives[world]["ports"][name])
            if name == "head":
                n = n.transpose(0, 2, 1, 3)
            assert np.array_equal(f, n)
        delta = load(output, forecasts["cheap-input"]["ports"][name]).astype(np.float64) - load(output, forecasts["full-input"]["ports"][name]).astype(np.float64)
        assert responses["fields"][name]["sha256"] == hashlib.sha256(delta.astype("<f8").tobytes()).hexdigest()
        assert np.count_nonzero(delta[:, :4]) == 0
    assert np.count_nonzero(delta[:, 4:]) > 0
    calls = c["read"](output / "sdpa.json")["calls"]
    assert [row["lane"] for row in calls] == ["functional", "functional", "native", "native"]
    events = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    sealed = next(row["number"] for row in events if row["kind"] == "cohort_globally_sealed")
    assert all(row["number"] > sealed for row in events if row["kind"].startswith("native_"))
    assert c["read"](output / "native-contract.json")["projection_provenance"] == "actual passive o_proj output hook"
    assert set(c["read"](output / "addition-diagnostics.json")) == {"functional", "native"}
    assert receipt["files_manifest_sha256"] == c["sha"](output / "files.json")
    assert m["PROTOCOL_SHA256"] == "4e62bc3fa8adcd059ac12f775effa0010752e3c5e0cca2a2ebc4702ecef7f847"


def test_tampered_sealed_response_denies_both_targets_and_releases_hooks(tmp_path):
    m, c, layer, tensors, refs, protocol, verify, output, _paths = prepare(tmp_path)
    linear, sdpa = F.linear, F.scaled_dot_product_attention
    def corrupt(output, _files):
        (output / "forecast-responses.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed external file changed"):
        c["execute_worlds"](output, layer, {name: tensors[name] for name in protocol["world_order"]},
            {name: tensors[name] for name in m["CONTEXT_KEYS"]}, protocol, references=refs, source_identity={}, verify_sources=verify, before_execute=corrupt)
    assert F.linear is linear and F.scaled_dot_product_attention is sdpa
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in layer.modules())
    assert "native_component_authorized" not in (output / "timeline.jsonl").read_text()


def test_inherited_file_drift_denies_targets(tmp_path):
    def corrupt(output, _files):
        path = output.parent / "synthetic-world/shortcut/prefix-negative/Hnext.npy"
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="inherited source file changed"):
        write_fixture(tmp_path, before_execute=corrupt)


def test_module_forecast_forbidden_and_cleanup(tmp_path):
    m, c, layer, tensors, refs, protocol, verify, output, _paths = prepare(tmp_path)
    def illegal(hidden, _capture):
        layer(hidden)
    with pytest.raises(RuntimeError, match="native module forward forbidden"):
        c["execute_worlds"](output, layer, {name: tensors[name] for name in protocol["world_order"]},
            {name: tensors[name] for name in m["CONTEXT_KEYS"]}, protocol, references=refs, source_identity={}, verify_sources=verify, functional=illegal)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in layer.modules())


def test_world_descriptor_layout_must_match_original_writer(tmp_path):
    _m, c, _layer, _tensors, _refs, protocol, _verify, _output, paths = prepare(tmp_path)
    protocol["world_sources"]["cheap-input"]["stride"] = [48, 1, 8]
    with pytest.raises(ValueError, match="source/layout differs"):
        c["load_sources"](paths, protocol, torch.device("cpu"))


def test_parent_native_future_cannot_enter_whitelist(tmp_path):
    _m, c, _layer, _tensors, _refs, protocol, _verify, _output, paths = prepare(tmp_path)
    seal = c["read"](paths["world_seal"])
    del seal["declarations"]["frontier"]["external_files"][protocol["world_sources"]["cheap-input"]["path"]]
    Path(paths["world_seal"]).write_bytes(c["canonical"](seal))
    protocol["roots"]["world"]["global_seal_sha256"] = c["sha"](paths["world_seal"])
    with pytest.raises(ValueError, match="source missing from pretarget"):
        c["load_sources"](paths, protocol, torch.device("cpu"))


def test_adapter_fixed_protocol_and_original_recipe_identity():
    m = runpy.run_path(str(PRODUCER))
    base = {key: key + ".json" for key in m["PATH_KEYS"]}
    base.update({key + "_sha256": "1" * 64 for key in m["PATH_KEYS"]})
    base.update({key: "/synthetic/" + key for key in m["ROOT_KEYS"]})
    base.update(selection_protocol_sha256=m["PROTOCOL_SHA256"], qwen_parameter_manifest_sha256=m["SUBSET_SHA256"], retrieval_script_sha256=m["ROW"]["BASE"]["RETRIEVER_SHA256"])
    config = m["make_config"](base)
    assert m["validate_config"](config) == m["C"]["contract"]()
    config["component_index"] = 18
    with pytest.raises(ValueError, match="fixed configuration differs"):
        m["validate_config"](config)


def test_actual_target_cannot_consume_wrong_world(tmp_path):
    m, c, layer, tensors, refs, protocol, verify, output, _paths = prepare(tmp_path)
    # A hostile external hook tries to replace the first target's argument.
    hostile = layer.register_forward_pre_hook(lambda _module, _args: (tensors["cheap-input"],))
    try:
        with pytest.raises(ValueError, match="actual native input world differs"):
            c["execute_worlds"](output, layer, {name: tensors[name] for name in protocol["world_order"]},
                {name: tensors[name] for name in m["CONTEXT_KEYS"]}, protocol, references=refs, source_identity={}, verify_sources=verify)
    finally:
        hostile.remove()
    assert not (output / "native.json").exists()
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in layer.modules())


def test_request_bound_worker_finalizer_and_drift_refusal(tmp_path, monkeypatch, capfd):
    """Real worker/identity/finalizer; only science and local destination are stubbed."""
    m = runpy.run_path(str(PRODUCER)); c = m["C"]; worker = m["worker"]; g = worker.__globals__
    original_make = g["make_config"]
    def local_make(base):
        return dict(original_make(base), output_scratch=str(tmp_path / "output"))
    monkeypatch.setitem(g, "make_config", local_make)
    base = {key: key + ".json" for key in m["PATH_KEYS"]}
    base.update({key + "_sha256": "1" * 64 for key in m["PATH_KEYS"]})
    base.update({key: "/synthetic/" + key for key in m["ROOT_KEYS"]})
    base.update(selection_protocol_sha256=m["PROTOCOL_SHA256"], qwen_parameter_manifest_sha256=m["SUBSET_SHA256"], retrieval_script_sha256=m["ROW"]["BASE"]["RETRIEVER_SHA256"])
    config = local_make(base); config_path = tmp_path / "config.json"; c["save"](config_path, config)
    source = tmp_path / PRODUCER.name; source.write_bytes(PRODUCER.read_bytes())
    request = dict(config=config, source_hashes={source.name: c["sha"](source)})
    request["request_sha256"] = c["ROW"]["BASE"]["request_identity"](request)
    request_path = tmp_path / "request.json"; c["save"](request_path, request)
    monkeypatch.setenv("SATURN_DEBUG_REQUEST", str(request_path)); monkeypatch.setenv("MRUN_JOB_ID", "job-tiny-bootstrap")
    calls = []
    def fake_science(derived):
        calls.append(derived)
        output = Path(derived["output"]); output.mkdir(parents=True)
        c["save"](output / "summary.json", dict(job_id=derived["job_id"], request_sha256=derived["request_sha256"], fixture=True))
    monkeypatch.setitem(g, "run", fake_science)
    receipt = worker(config_path)
    assert len(calls) == 1 and receipt["job_id"] == "job-tiny-bootstrap"
    assert receipt["request_sha256"] == request["request_sha256"]
    marker = next(line for line in capfd.readouterr().out.splitlines() if line.startswith("MDB_DIRECTORY_READY "))
    assert json.loads(marker.removeprefix("MDB_DIRECTORY_READY ")) == receipt
    source.write_bytes(source.read_bytes() + b"\n# altered synthetic source copy\n")
    try:
        with pytest.raises(ValueError, match="executed source differs"):
            worker(config_path)
        assert len(calls) == 1
        Path(config_path).write_bytes(c["canonical"](dict(config, horizon=19)))
        with pytest.raises(ValueError, match="fixed configuration differs"):
            worker(config_path)
        assert len(calls) == 1
    finally:
        g["faulthandler"].cancel_dump_traceback_later()
