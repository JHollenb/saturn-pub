"""Installed tiny random-Qwen mechanics; no pretrained checkpoint or service."""
from __future__ import annotations

import ast
import copy
import importlib.metadata
import inspect
import json
import os
import runpy
import shutil
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

C = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_spectral_readout_campaign.py"))


def tiny_model():
    torch.manual_seed(17); torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=31, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval().requires_grad_(False)


def test_public_config_is_definitions_only_and_fixed_to_source_scope():
    config = C["make_config"]({})
    counts = C["validate_config"](config)
    assert counts["component_readout_targets"] == 100 and counts["ordinary_prefill_adjudications"] == 16
    assert counts["autonomous_cached_calls_max"] == 128 and counts["native_physical_entries_max"] == 280
    assert config["consumer_strings"] == [" red", " blue"] and config["continuation_tokens"] == 8
    assert len(config["prompts"]) == 4 and len(C["helpers"]()) == 3
    config["doses"] = [0., 1.]
    with pytest.raises(ValueError, match="fixed spectral contract"):
        C["validate_config"](config)


def test_original_default_separator_request_identity_and_source_tamper(tmp_path):
    source = Path(__file__).parents[1] / "qualification"
    names = ["qwen_spectral_readout_campaign.py", *C["HELPERS"]]
    for name in names:
        shutil.copyfile(source / name, tmp_path / name)
    config = C["make_config"]({})
    request = dict(config=config, source_hashes={name: C["sha"](tmp_path / name) for name in names}, fixture="metadata-only")
    import hashlib

    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert request["request_sha256"] != C["digest"]({key: value for key, value in request.items() if key != "request_sha256"})
    C["authenticate_request"](request, config, tmp_path)
    changed = copy.deepcopy(request); changed["fixture"] = "altered"
    with pytest.raises(ValueError, match="request/config identity differs"):
        C["authenticate_request"](changed, config, tmp_path)
    (tmp_path / names[0]).write_bytes(b"altered source")
    with pytest.raises(ValueError, match="request source changed"):
        C["authenticate_request"](request, config, tmp_path)


def test_actual_prefix_keeps_complete_native_KV_and_private_future_history():
    model = tiny_model(); ids = torch.tensor([[2, 4, 6]])
    positions = torch.arange(3).unsqueeze(0)
    mask = torch.triu(torch.full((1, 1, 3, 3), torch.finfo(torch.float32).min), diagonal=1)
    cache = DynamicCache(config=model.config)
    called = []
    handles = [module.register_forward_pre_hook(lambda *_args, name=name: called.append(name))
        for name, module in (("postnorm", model.model.layers[-1].post_attention_layernorm), ("mlp", model.model.layers[-1].mlp), ("norm", model.model.norm), ("head", model.lm_head))]
    try:
        capture = C["prefix_with_cache"](model, ids, mask, positions, cache, identity="tiny-random", execution=dict(backend="cpu", dtype="float32"), maximum=2048)
    finally:
        for handle in handles:
            handle.remove()
    assert called == [] and capture.output["boundary"]["args"][0].shape == (1, 3, 8)
    assert cache.get_seq_length() == 3 and len(C["cache_arrays"](cache)) == 2
    parent = C["cache_snapshot"](cache)
    with torch.no_grad():
        ordinary = model(ids, attention_mask=mask, position_ids=positions, past_key_values=DynamicCache(config=model.config), use_cache=True, return_dict=True)
    assert C["cache_snapshot"](ordinary.past_key_values) == parent
    private = copy.deepcopy(ordinary.past_key_values)
    assert all(a.data_ptr() != b.data_ptr() for left, right in zip(C["cache_arrays"](private), C["cache_arrays"](cache), strict=True) for a, b in zip(left, right, strict=True))
    token = int(ordinary.logits[0, -1].argmax())
    with torch.no_grad():
        future = model(torch.tensor([[token]]), past_key_values=private, use_cache=True, return_dict=True)
    assert future.past_key_values.get_seq_length() == 4 and C["cache_snapshot"](cache) == parent


def test_actual_tiny_global_seal_component_adjudication_and_autonomous_futures(tmp_path):
    model = tiny_model()
    settings = dict(seed=37, lanczos_steps=8, max_prompt_tokens=8, max_prefix_ops=2048, doses=[0., 1.], continuation_tokens=2,
        max_native_entries=64, max_wall_seconds=60, transaction_bytes=16 << 20)
    result = C["campaign"](model, [dict(id="tiny", text="untrained fixture only", ids=torch.tensor([[2, 4, 6]]))], [7, 11], tmp_path / "result",
        model_identity=dict(name="tiny-random-Qwen2-no-pretrained"), source_identity=dict(fixture="new-source-only-tiny"), settings=settings)
    counts = result["summary"]["counts"]
    assert counts["prefixes"] == counts["native_MLP_bases"] == counts["ordinary_shams"] == 1
    assert counts["component_readouts"] == counts["ordinary_prefills"] == 4
    assert counts["parity_targets"] == 6 and counts["cached_futures"] == 8 and counts["physical_entries"] == 25
    assert counts["denied_preseal_entries"] == 0 and result["summary"]["transaction"]["verified"]
    predicted = result["predictions"]["prompts"]["tiny"]
    observed = result["observations"]["prompts"]["tiny"]
    assert len(predicted["selection"]["rows"]) == 4 and "0,0" in predicted["selection"]["rows"]
    assert observed["sham_vs_baseline"]["max_abs"] == 0
    for name, row in observed["ordinary"].items():
        assert row["cache_exact_vs_observed_prefix"] and row["earlier_rows_exact"]
        assert row["component_vs_ordinary"]["max_abs"] < 1e-6
        if name != "sham":
            assert len(row["continuation"]) == 2 and row["continuation"][0]["consumed_token"] == row["greedy_token"]
            assert row["continuation"][1]["consumed_token"] == row["continuation"][0]["greedy_token"]
    assert not result["predictions"]["continuation_forecast"]
    report = C["publish_accounts"](result, tmp_path / "result", execution=dict(backend="cpu", dtype="float32", attention="eager"),
        source_identity=dict(fixture=C["digest"]("source-only tiny")), run_id="tiny-no-pretrained")
    assert len(report["rows"]) == len(report["atlas_ids"]) == 4
    assert report["coverage"]["independent_contexts"] == 1
    receipt = C["helpers"]()["flux_image_tail_microscope.py"]["finalize"](tmp_path / "result", job_id="tiny-mechanics-only", request_sha256=C["digest"]("tiny"))
    assert receipt["file_count"] > 20 and not (tmp_path / "result/parameters").exists()
    manifest = json.loads((tmp_path / "result/files.json").read_text())
    assert manifest["files"]["summary.json"]["sha256"] == receipt["summary_sha256"]


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_exact_original_recipes_restore_tied_head_without_persistent_bank(tmp_path, source_dtype):
    import numpy as np
    from safetensors.torch import save_file

    import mdb_lanes.parameter_artifacts as artifacts

    model = tiny_model()
    parameters = dict(model.named_parameters())
    selected = {name: value for name, value in parameters.items() if name == "model.embed_tokens.weight" or name == "model.norm.weight" or name.startswith("model.layers.1.post_attention_layernorm.") or name.startswith("model.layers.1.mlp.")}
    with torch.no_grad():
        for value in selected.values():
            value.copy_(value.to(source_dtype).float())
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    file = checkpoint / "model.safetensors"
    save_file({name: value.detach().to(source_dtype).contiguous() for name, value in selected.items()}, file)
    inventory = {file.name: dict(bytes=file.stat().st_size, sha256=C["sha"](file))}
    result = tmp_path / "recipes"; result.mkdir()
    recipes = C["helpers"]()["qwen_spectral_parameter_recipes.py"]
    bindings = recipes["prepare"](model, checkpoint, inventory, result, inspect.getfile(artifacts))
    assert not list(result.rglob("*.npy")) and bindings["head"]["original_tensor"] == "model.embed_tokens.weight"
    assert "lm_head.weight" in bindings["head"]["actual_shared_parameter_aliases"]
    manifest = json.loads((result / "parameter-artifacts.json").read_text())
    restored = tmp_path / "restored"
    receipt = artifacts.restore(manifest, restored, checkpoint)
    assert receipt["checkpoint_files_verified"] == 1 and receipt["model_calls"] == receipt["conversions"] == 0
    for name, binding in bindings.items():
        value = np.load(restored / ("parameters/"+name+".npy"), allow_pickle=False)
        if source_dtype == torch.bfloat16:
            value = (value.astype(np.uint32) << 16).view(np.float32)
        assert value.tobytes() == model.get_parameter(binding["native_name"]).detach().numpy().tobytes()
    with torch.no_grad():
        model.model.norm.weight[0] += .25
    corrupted = tmp_path / "bad-recipes"; corrupted.mkdir()
    with pytest.raises(ValueError, match="does not reproduce"):
        recipes["prepare"](model, checkpoint, inventory, corrupted, inspect.getfile(artifacts))


def test_original_loader_inventory_and_fixed_single_token_consumers_refuse_corruption(tmp_path):
    path = tmp_path / "model.safetensors"; path.write_bytes(b"synthetic-header-only")
    expected = {path.name: dict(bytes=path.stat().st_size, sha256=C["sha"](path))}
    actual, signatures = C["checkpoint_inventory"](tmp_path, expected)
    assert actual == expected and signatures[path.name][2] == len(b"synthetic-header-only")
    path.write_bytes(b"changed-header-bytes!")
    with pytest.raises(ValueError, match="checkpoint identity differs"):
        C["checkpoint_inventory"](tmp_path, expected)

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [1, 2] if text == " red" else [3]

    with pytest.raises(ValueError, match="must be single tokens"):
        C["tokenize"](Tokenizer(), C["make_config"]({}))


def test_worker_version_metadata_crosses_strict_catalog_without_model_calls(tmp_path, monkeypatch):
    """Evaluate the real worker's profile expression against actual TorchVersion."""
    import numpy as np
    from torch.torch_version import TorchVersion

    from mdb_lanes.forecast_transaction import load_forecast_tree

    fixture = os.environ.get("MDB_QWEN_SPECTRAL_TINY_FIXTURE")
    if not fixture:
        pytest.skip("Explicit retained tiny evidence required; this regression executes no model")
    source = Path(fixture); tx = source / "transaction"
    result = dict(summary=json.loads((source / "summary.json").read_text()),
        predictions=load_forecast_tree(tx, json.loads((tx / "forecasts/cohort.json").read_text())["predictions"]),
        observations=load_forecast_tree(tx, json.loads((tx / "observations/native.json").read_text())["value"]))
    native_version = TorchVersion(str(torch.__version__)); monkeypatch.setattr(torch, "__version__", native_version)

    def denied_forward(*_args, **_kwargs):
        raise AssertionError("Metadata regression must not call a model")

    monkeypatch.setattr(torch.nn.Module, "__call__", denied_forward)
    # Isolate the actual worker expression so this does not construct a model,
    # mock its architecture or duplicate the profile's version conversion.
    worker = ast.parse(inspect.getsource(C["worker"]))
    expression = next(node.value for node in ast.walk(worker) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "execution" for target in node.targets))
    execution = eval(compile(ast.Expression(expression), "worker_execution_metadata", "eval"), dict(torch=torch, np=np, importlib=importlib))
    assert type(native_version) is TorchVersion and type(execution["torch"]) is str and execution["torch"] == str(native_version)

    def account_root(name):
        path = tmp_path / name; (path / "transaction").mkdir(parents=True)
        shutil.copyfile(source / "summary.json", path / "summary.json")
        shutil.copyfile(tx / "seal.json", path / "transaction/seal.json")
        return path

    with pytest.raises(ValueError, match="binding.execution.torch contains unsupported"):
        C["publish_accounts"](result, account_root("unconverted-version"), execution=dict(execution, torch=native_version), source_identity=dict(fixture="1"*64), run_id="metadata-regression")
    output = account_root("plain-version")
    records = C["publish_accounts"](result, output, execution=execution, source_identity=dict(fixture="1"*64), run_id="metadata-regression")
    assert len(records["rows"]) == len(records["atlas_ids"]) == 4
    saved = json.loads(next((output / "catalog").glob("*.json")).read_text())
    assert saved["binding"]["execution"]["torch"] == str(native_version)
