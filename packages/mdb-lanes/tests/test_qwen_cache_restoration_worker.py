"""Complete generated-checkpoint worker transaction; no pretrained assets."""
from __future__ import annotations

import hashlib
import inspect
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2

from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction

QUAL = Path(__file__).resolve().parents[1] / "qualification"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_whole_worker_publishes_sealed_four_corner_own_futures(tmp_path, monkeypatch):
    payload, checkpoint, setup = (tmp_path / name for name in ("payload", "checkpoint", "setup"))
    for directory in (payload, checkpoint, setup):
        directory.mkdir()
    for name in ("qwen_cache_restoration.py", "qwen_family_campaign.py", "qwen_family_math.py",
                 "retrieve_qwen_family_weights.py"):
        shutil.copyfile(QUAL / name, payload / name)
    worker = runpy.run_path(str(payload / "qwen_cache_restoration.py"))
    torch.set_num_threads(1)
    torch.manual_seed(1903)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=14, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True, attention_dropout=0.,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.})
    config._attn_implementation = "eager"
    original = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    original.save_pretrained(checkpoint, safe_serialization=True)
    token = Tokenizer(WordLevel({"[UNK]": 0, **{"word" + str(index): index for index in range(1, 64)}},
                                unk_token="[UNK]"))
    token.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=token, unk_token="[UNK]").save_pretrained(checkpoint)
    model = Qwen2ForCausalLM.from_pretrained(checkpoint, dtype=torch.float32, attn_implementation="eager",
                                            local_files_only=True).eval().requires_grad_(False)
    prepare = runpy.run_path(str(payload / "qwen_family_campaign.py"))["prepare_parameters"]
    manifest = prepare(checkpoint, model, setup)
    worker["save"](payload / "parent-parameter-artifacts.json", manifest)
    worker["save"](payload / "parent-files.json", {"files": {}})
    pending_ids = torch.tensor([[3, 11, 27]])
    with torch.inference_mode():
        prefill = model(pending_ids, use_cache=True, logits_to_keep=1)
    pending = int(prefill.logits[0, -1].argmax())
    q = tuple((layer.keys.clone(), layer.values.clone()) for layer in prefill.past_key_values.layers)
    e = tuple(tuple(value.clone() for value in pair) for pair in q)
    e[13][0][..., -1, :] += .25
    e[13][1][..., -1, :] -= .5
    for side, cache in (("q", q), ("e", e)):
        values = dict(input_ids=pending_ids.numpy(), token=np.array([pending], dtype=np.int64),
                      logits=prefill.logits.detach().numpy())
        for layer, pair in enumerate(cache):
            for kind, value in zip("kv", pair, strict=True):
                values[f"{kind}{layer:02d}"] = value.numpy()
        np.savez(payload / ("parent-" + side + ".npz"), **values)
    file_sha = sha(checkpoint / "model.safetensors")
    weights_sha = hashlib.sha256(worker["canonical"]({"model.safetensors": file_sha})).hexdigest()
    runtime = dict(native_source_sha256=sha(inspect.getfile(modeling_qwen2)),
        math_source_sha256=sha(payload / "qwen_family_math.py"), torch=str(torch.__version__),
        transformers=str(transformers.__version__), dtype="torch.float32", device="cpu", attention="eager",
        checkpoint_sha256=file_sha, weights_identity_sha256=weights_sha)
    worker["save"](payload / "parent-runtime.json", runtime)
    source_files = {path.name: dict(bytes=path.stat().st_size, sha256=sha(path)) for path in payload.glob("parent-*")}
    design = dict(new_choices_per_corner=5, new_forward_census=dict(functional=20, native=20, prefills=0, model_loads=1),
        pending_token=pending, cache_length=3, changed_layer_start=13, layers=14,
        checkpoint_file_sha256=file_sha, weights_sha256=weights_sha, source_files=source_files,
        source_job="generated-fixture")
    worker["save"](payload / "design.json", design)
    settings = dict(design_sha256=sha(payload / "design.json"), output_scratch=str(tmp_path / "outputs"),
                    model_path=str(checkpoint), max_wall_seconds=120)
    request = dict(request_sha256="7" * 64, source_hashes={name: sha(payload / name) for name in (
        "qwen_cache_restoration.py", "qwen_family_campaign.py", "qwen_family_math.py")})
    monkeypatch.setenv("MRUN_JOB_ID", "generated-whole-cache-worker")
    publication = worker["run"](settings, request)
    root = Path(publication["output_root"])
    assert sha(root / "files.json") == publication["files_manifest_sha256"]
    verified = verify_transaction(root / "transaction")
    assert verified["verified"]
    counts = json.loads((root / "counts.json").read_text())
    assert counts == dict(functional_calls=20, native_calls=20, denied_entries=0)
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    seal_index = next(index for index, row in enumerate(timeline) if row["kind"] == "global_forecast_sealed")
    assert timeline[seal_index]["counts"] == dict(functional_calls=20, native_calls=0, denied_entries=0)
    assert len([row for row in timeline[:seal_index] if row["kind"] == "forecast_future_complete"]) == 4
    indices = {lane: json.loads((root / (lane + "-index.json")).read_text()) for lane in ("forecast", "native")}
    prediction = json.loads((root / "transaction/forecasts/all_futures.json").read_text())
    sealed = load_forecast_tree(root / "transaction", prediction["predictions"])
    assert set(sealed) == set(indices["forecast"]) == {"QQ", "EQ", "QE", "EE"}
    for corner in sealed:
        assert len(sealed[corner]["steps"]) == 5
        for lane, index in indices.items():
            incoming = pending
            previous = {f"{kind}{layer:02d}": pair[component].numpy()
                        for layer, pair in enumerate(q) for component, kind in enumerate("kv")}
            for step, record in enumerate(index[corner]["steps"]):
                assert sha(root / record["path"]) == record["sha256"]
                with np.load(root / record["path"], allow_pickle=False) as archive:
                    row = {name: archive[name] for name in archive.files}
                assert row["input_ids"].tolist() == [[incoming]]
                incoming = int(row["logits"][0, -1].argmax())
                assert row["token"].tolist() == [incoming] and record["token_id"] == incoming
                assert len([name for name in row if name[:1] in "ar" and name[1:].isdigit()]) == 28
                for field in (f"{kind}{layer:02d}" for layer in range(14) for kind in "kv"):
                    assert row[field].shape[-2] == 4 + step
                    if step:
                        assert row[field][..., :-1, :].tobytes() == previous[field].tobytes()
                previous = row
            assert row["k00"].shape[-2] == 8
        for forecast_record, native_record in zip(indices["forecast"][corner]["steps"], indices["native"][corner]["steps"], strict=True):
            with np.load(root / forecast_record["path"], allow_pickle=False) as forecast, np.load(
                    root / native_record["path"], allow_pickle=False) as native:
                assert set(forecast.files) == set(native.files)
                assert all(forecast[name].tobytes() == native[name].tobytes() for name in forecast.files)
    inventory = json.loads((root / "files.json").read_text())["files"]
    assert set(inventory) == {path.relative_to(root).as_posix() for path in root.rglob("*")
                              if path.is_file() and path.name != "files.json"}
