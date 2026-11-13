"""Generated Qwen/tokenizer end-to-end qualification; no pretrained assets."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from mdb_lanes.computation_protocol import SPEC_SCHEMA, compile_protocol
from mdb_lanes.forecast_transaction import verify_transaction

QUALIFICATION = Path(__file__).resolve().parents[1] / "qualification"
WORKER = runpy.run_path(str(QUALIFICATION / "qwen_family_campaign.py"))
RETRIEVER = runpy.run_path(str(QUALIFICATION / "retrieve_qwen_family_weights.py"))
AUDIT = runpy.run_path(str(QUALIFICATION / "qwen_family_audit.py"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def generated_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("generated-qwen13")
    torch.set_num_threads(1)
    torch.manual_seed(4919)
    config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=13, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=64, tie_word_embeddings=True,
                        rope_parameters={"rope_type": "default", "rope_theta": 500000.})
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    model.save_pretrained(root, safe_serialization=True)
    token = Tokenizer(BPE(unk_token="[UNK]"))
    token.pre_tokenizer, token.decoder = ByteLevel(add_prefix_space=False), ByteLevelDecoder()
    token.train_from_iterator(["alpha red beta blue Read A Answer", "alpha red beta blue Read B Answer",
                               " red", " blue"] * 8,
                              trainer=BpeTrainer(vocab_size=128, min_frequency=1, special_tokens=["[UNK]", "[PAD]"]))
    token.add_tokens(["unused" + str(index) for index in range(128 - token.get_vocab_size())])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=token, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.save_pretrained(root)
    loaded = AutoTokenizer.from_pretrained(root, local_files_only=True)
    assert len(loaded.encode(" red", add_special_tokens=False)) == 1
    assert len(loaded.encode(" blue", add_special_tokens=False)) == 1
    assert len(loaded.encode("alpha red beta blue Read A Answer", add_special_tokens=False)) == len(
        loaded.encode("alpha red beta blue Read B Answer", add_special_tokens=False))
    return root


def plan(family, weights_sha):
    if family == "attention":
        ports = {"q": {"shape": [1, 4, 1, 8], "dtype": "torch.float32", "stage": "positioned final query"},
                 **{name: {"shape": [1, 2, "T", 8], "dtype": "torch.float32", "stage": "visible positioned history"}
                    for name in ("k", "v")}}
        factors = [{"id": "route", "ports": ["q", "k"]}, {"id": "value", "ports": ["v"]}]
        nodes = [{"id": "result", "op": "sdpa", "inputs": ["q", "k", "v"]}]
    else:
        ports = {name: {"shape": [1, 1, 64], "dtype": "torch.float32", "stage": "final raw factor"} for name in ("G", "U")}
        factors = [{"id": "gate", "ports": ["G"]}, {"id": "value", "ports": ["U"]}]
        nodes = [{"id": "activated", "op": "silu", "inputs": ["G"]},
                 {"id": "result", "op": "multiply", "inputs": ["activated", "U"]}]
    return compile_protocol({"schema": SPEC_SCHEMA, "family_id": "synthetic-qwen13-" + family,
        "model_identity": weights_sha, "endpoint_adapter": "qwen2-eager-original-final-row-v1",
        "ports": ports, "factors": factors, "nodes": nodes, "output": "result", "max_contexts": 3,
        "source_policy": "generated random fixture and literal IDs; no pretrained assets",
        "receiver": {"residual": "Q", "selected_layer_cache": "Q", "prefix_rows": "Q"},
        "clock": {"layer": 12, "token": "final-prompt-token"},
        "continuation": {"consumer": "original tiny own-token/KV suffix", "transitions": 2,
                         "execution_kind": "autoregressive", "vae_decodes": 0},
        "metric": {"observable": "first-readout logit(answer_Q)-logit(answer_P)", "units": "logits",
                   "direction": "minimize", "reference": "supplied synthetic labels", "scope": "fixture mechanics"}})


def config(checkpoint, output, family):
    file_sha = sha(checkpoint / "model.safetensors")
    weights_sha = hashlib.sha256(json.dumps({"model.safetensors": file_sha}, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    return {"schema": WORKER["SCHEMA"], "family": family, "phase": "development",
            "model_path": str(checkpoint), "weights_sha256": weights_sha, "checkpoint_file_sha256": file_sha,
            "compiled_protocol": plan(family, weights_sha),
            "contexts": [{"id": "generated-pair", "seed": 4919,
                          "prompts": {"P": "alpha red beta blue Read A Answer", "Q": "alpha red beta blue Read B Answer"},
                          "answer_labels": {"P": " red", "Q": " blue"}}],
            "tokens": 2, "max_prompt_tokens": 16, "layer": 12, "cpu_threads": 1,
            "dtype": "float32", "device": "cpu", "output_scratch": str(output), "max_wall_seconds": 120,
            "scientific_gates": [], "retry": False, "nested_mrun": False,
            "design_sha256": "d" * 64, "effect_rule": None}


@pytest.mark.parametrize("family", ["attention", "mlp"])
def test_actual_six_role_worker_seal_own_futures_and_saved_audit(generated_checkpoint, tmp_path, monkeypatch, family):
    settings = config(generated_checkpoint, tmp_path, family)
    request = {"schema": "mdb-qwen-family-request-v1", "config": settings,
               "source_hashes": {name: sha(QUALIFICATION / name) for name in (
                   "qwen_family_campaign.py", "qwen_family_math.py", "retrieve_qwen_family_weights.py")}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    monkeypatch.setenv("MRUN_JOB_ID", "generated-" + family)
    publication = WORKER["run"](settings, request)
    root = Path(publication["output_root"])
    assert sha(root / "files.json") == publication["files_manifest_sha256"]
    summary = json.loads((root / "summary.json").read_text())
    assert summary["futures"] == 6 and summary["tokens_per_future"] == 2
    assert summary["counts"]["native_calls"] == summary["counts"]["functional_calls"] == 12
    assert summary["counts"]["denied_entries"] == 0
    assert summary["counts"]["graph_calls"] == 4
    geometry = settings["compiled_protocol"]["execution_geometry"]["per_context_per_lane"]
    assert geometry["physical_forwards"] == summary["counts"]["native_calls"]
    assert geometry["prefill_forwards"] == geometry["cached_forwards"] == 6
    assert geometry["VAE_decodes"] == geometry["common_parent_Acts"] == 0
    assert summary["counts"]["native_hybrid_kernels"] == (4 if family == "attention" else 0)
    events = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    seal = next(index for index, row in enumerate(events) if row["kind"] == "global_forecast_sealed")
    assert len([row for row in events if row["kind"] == "global_forecast_sealed"]) == 1
    assert all(row["counts"]["native_calls"] == 0 for row in events[:seal + 1])
    assert len([row for row in events[:seal] if row["kind"] == "forecast_future_complete"]) == 6
    assert len([row for row in events[seal + 1:] if row["kind"] == "native_future_complete"]) == 6
    verified = verify_transaction(root / "transaction", expected_seal_sha256=events[seal]["seal_sha256"])
    assert verified
    inventory = json.loads((root / "files.json").read_text())["files"]
    for name, reference in inventory.items():
        assert sha(root / name) == reference["sha256"] and (root / name).stat().st_size == reference["bytes"]
    indices = {lane: json.loads((root / filename).read_text()) for lane, filename in (
        ("forecast", "forecast-index.json"), ("native", "native-index.json"))}
    prompt_length = len(json.loads((root / "inputs.json").read_text())["generated-pair"]["Q"][0])
    assert set(indices["forecast"]) == set(indices["native"])
    for lane, index in indices.items():
        for row in index.values():
            first, second = row["steps"]
            assert sha(root / first["path"]) == first["sha256"]
            assert sha(root / second["path"]) == second["sha256"]
            with np.load(root / first["path"], allow_pickle=False) as before, np.load(root / second["path"], allow_pickle=False) as after:
                assert after["input_ids"].tolist() == [[int(before["token"][0])]]
                assert len([name for name in before.files if name.startswith("k") and name[1:].isdigit()]) == 13
                for layer in range(13):
                    assert before[f"k{layer:02d}"].shape == (1, 2, prompt_length, 8)
                    assert after[f"k{layer:02d}"].shape == (1, 2, prompt_length + 1, 8)
    with monkeypatch.context() as local:
        def forbidden(*_args, **_kwargs):
            pytest.fail("saved-only audit entered a native module")
        local.setattr(torch.nn.Module, "_call_impl", forbidden)
        report = AUDIT["audit"](root, publication["files_manifest_sha256"], tmp_path / "audit")
    assert report["mechanics_status"] == "verified" and report["model_forwards"] == 0
    assert report["token_predictions"] == 12 and len(report["cases"]) == 6
    assert report["frozen_effect_evaluation"] is None
    assert report["max_logit_error"] < 2e-6 and report["max_cache_error"] < 2e-6
    assert report["token_matches"] == 12
    assert len(report["selected_layer_port_comparisons"]) == 6
    assert set(report["equation_reports"]) == set(report["cases"])
    assert all(value["schema"] == "mdb-equation-comparison-v1" for value in report["equation_reports"].values())
    tampered = tmp_path / "tampered-index"
    shutil.copytree(root, tampered)
    index_path = tampered / "forecast-index.json"
    index = json.loads(index_path.read_text())
    index[next(iter(index))]["text"] += " changed"
    index_path.write_text(json.dumps(index, sort_keys=True) + "\n")
    altered_inventory = json.loads((tampered / "files.json").read_text())
    altered_inventory["files"]["forecast-index.json"] = {"sha256": sha(index_path), "bytes": index_path.stat().st_size}
    (tampered / "files.json").write_text(json.dumps(altered_inventory, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="sealed case census"):
        AUDIT["audit"](tampered, sha(tampered / "files.json"), tmp_path / "tampered-audit")


def test_validate_rejects_invalid_profile_and_prospective_rule(generated_checkpoint, tmp_path):
    settings = config(generated_checkpoint, tmp_path, "attention")
    assert WORKER["validate"](settings)["candidate_menu"] == ["value", "route"]
    for name, value in (("weights_sha256", "0" * 64), ("layer", 11), ("tokens", 3)):
        corrupt = copy.deepcopy(settings)
        corrupt[name] = value
        with pytest.raises(ValueError, match="compiled program"):
            WORKER["validate"](corrupt)
    settings["device"] = "cuda"
    with pytest.raises(ValueError, match="CPU FP32"):
        WORKER["validate"](settings)
    settings["device"], settings["phase"] = "cpu", "fresh"
    settings["effect_rule"] = {"plan_sha256": settings["compiled_protocol"]["plan_sha256"], "fresh_context_metadata": {}}
    with pytest.raises(ValueError, match="prospective rule"):
        WORKER["validate"](settings)


def test_original_recipe_restoration_and_tamper_rejection(generated_checkpoint, tmp_path):
    model = Qwen2ForCausalLM.from_pretrained(generated_checkpoint, local_files_only=True,
                                           dtype=torch.float32, attn_implementation="eager")
    output = tmp_path / "recipes"
    output.mkdir()
    manifest = WORKER["prepare_parameters"](generated_checkpoint, model, output)
    assert manifest["persistent_parameter_exports"] is False
    assert manifest["aliases"] == {"lm_head.weight": "model.embed_tokens.weight"}
    restored = tmp_path / "restored.safetensors"
    RETRIEVER["retrieve"](manifest, generated_checkpoint, restored)
    original, rebuilt = load_file(generated_checkpoint / "model.safetensors"), load_file(restored)
    assert set(original) == set(rebuilt)
    assert all(torch.equal(original[name], rebuilt[name]) for name in original)
    corrupt = copy.deepcopy(manifest)
    name = next(iter(corrupt["tensors"]))
    corrupt["tensors"][name]["archive_bytes_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tensor bytes differ"):
        RETRIEVER["retrieve"](corrupt, generated_checkpoint)
    corrupt = copy.deepcopy(manifest)
    corrupt["checkpoint_files"]["model.safetensors"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint file differs"):
        RETRIEVER["retrieve"](corrupt, generated_checkpoint)
