import json
import shutil
import torch
import pytest
from diffusers import (StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler)
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from mdb.bench import BenchSession, BenchError, tensor_digest


@pytest.fixture
def pipeline(tmp_path):
    pytest.importorskip("mrun", reason="image execution requires the optional private mrun adapter")
    torch.manual_seed(39)
    vocab = tmp_path/"vocab.json"; merges = tmp_path/"merges.txt"
    vocab.write_text(json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2, "b</w>": 3}))
    merges.write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(vocab_file=str(vocab), merges_file=str(merges), model_max_length=8)
    config = CLIPTextConfig(vocab_size=4, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8,
        projection_dim=8, bos_token_id=0, eos_token_id=1, pad_token_id=1)
    vae = AutoencoderKL(block_out_channels=(16, 32), in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",)*2, up_block_types=("UpDecoderBlock2D",)*2,
        latent_channels=4, norm_num_groups=4).eval()
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), block_out_channels=(16, 32),
        layers_per_block=1, cross_attention_dim=32, attention_head_dim=4, norm_num_groups=4,
        addition_embed_type="text_time", addition_time_embed_dim=4, projection_class_embeddings_input_dim=32).eval()
    pipe = StableDiffusionXLPipeline(vae=vae, unet=unet, scheduler=EulerDiscreteScheduler(num_train_timesteps=32),
        text_encoder=CLIPTextModel(config).eval(), text_encoder_2=CLIPTextModelWithProjection(config).eval(),
        tokenizer=tokenizer, tokenizer_2=tokenizer, add_watermarker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def test_finite_diffusion_example(pipeline, tmp_path):
    from pathlib import Path
    from mdb.bench_cli import run_commands
    commands = json.loads((Path(__file__).resolve().parents[1] /
                           "examples/diffusion-commands.json").read_text())
    with BenchSession.from_resident(pipeline, root=tmp_path / "session") as session:
        session.prepare("a", seed=17, height=16, width=16, num_inference_steps=2)
        rows = run_commands(session, commands)
        assert session.cursor()["phase"] == "complete"
        assert all(row["session_id"] == session.session_id for row in rows)
        assert rows[-1]["cut_id"] == session.branch.cut


@pytest.mark.parametrize("guidance", [1.0, 2.0])
def test_full_sdxl_native_rgb_partial_cut_restart(pipeline, tmp_path, guidance):
    with torch.inference_mode():
        stock = pipeline("a", generator=torch.Generator().manual_seed(7), height=16, width=16,
                         num_inference_steps=2, guidance_scale=guidance, output_type="np").images
    b = BenchSession.from_resident(pipeline, root=tmp_path/"session")
    from mdb.bench_adapters import CONTRACTS
    assert CONTRACTS["sdxl"].qualify(b.capabilities())["bench_status"] == "implemented"
    parent = b.prepare("a", seed=7, height=16, width=16, num_inference_steps=2, guidance_scale=guidance)
    b.step("block")
    cut = b.checkpoint("down-block")
    assert "down_block_res_samples" in b.state["execution"]["frame"]["locals"]
    b.fork("sibling")
    edit = b.preview_edit("residual", 0, mode="add"); b.abort_edit(edit["act_id"])
    b.generate(); expected = tensor_digest(b.resolve_value("image"))
    assert torch.equal(b.resolve_value("image"), torch.from_numpy(stock))
    b.checkout("main"); b.generate()
    assert tensor_digest(b.resolve_value("image")) == expected
    b.restore(cut); b.export(tmp_path/"export"); b.close(); shutil.rmtree(tmp_path/"session")
    fresh = BenchSession.resume(pipeline, root=tmp_path/"export")
    fresh.generate(); assert tensor_digest(fresh.resolve_value("image")) == expected
    fresh.restore(parent); fresh.step("denoise", 2)
    with pytest.raises(BenchError): fresh.resolve_value("image")
    fresh.step("vae")
    assert fresh.cursor()["phase"] == "decoded"
    with pytest.raises(BenchError): fresh.resolve_value("image")
    fresh.step("commit")
    assert tensor_digest(fresh.resolve_value("image")) == expected
    from mdb.bench_verify import verify
    assert verify(tmp_path/"export")["mechanics_status"] == "valid"
    fresh.close()


def test_diffusion_debugger_and_publication_rollback(pipeline, tmp_path, monkeypatch):
    import runpy
    from pathlib import Path
    monkeypatch.setenv("MRUN_JOB_ID", "tiny-image-debugger")
    from debugger_fixture import debugger
    b = BenchSession.from_resident(pipeline, root=tmp_path/"session")
    parent = b.prepare("a", seed=7, height=16, width=16, num_inference_steps=2)
    assert debugger(b)["same_python_identity"]
    b.checkout("main"); b.restore(parent)
    native = b.adapter.scheduler_capture()
    publish = b._publish
    def fail(): raise OSError("injected diffusion publication failure")
    monkeypatch.setattr(b, "_publish", fail)
    with pytest.raises(OSError): b.step()
    assert b.cursor()["step"] == 0
    assert b.model.scheduler._step_index == native["_step_index"]
    monkeypatch.setattr(b, "_publish", publish)
    bad = b.artifacts.get(parent)
    bad["state"]["phase"] = "complete"
    ref = b.artifacts.put(bad)
    with pytest.raises(BenchError): b.restore(ref)
    assert b.cursor()["step"] == 0
    discovery = b.discover({"name": "image-response", "sites": [0, 1],
        "contrasts": [{"name": "attenuate", "mode": "scale", "value": 0.9}],
        "observable": {"kind": "latent_mean"}, "continuation_steps": 1})
    assert b.replay_observation(discovery["report"]["observations"][0]["evidence_ref"])["exact"]
    b.step("denoise")
    with pytest.raises(BenchError): b.bind_symbol(discovery["report"])
    b.close()


def test_stochastic_scheduler_keeps_live_rng_and_conditioner_edit(pipeline, tmp_path):
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    with torch.inference_mode():
        stock = pipeline("a", generator=torch.Generator().manual_seed(11), height=16, width=16,
                         num_inference_steps=3, guidance_scale=1.0, output_type="np").images
    b = BenchSession.from_resident(pipeline, root=tmp_path/"stochastic")
    parent = b.prepare("a", seed=11, height=16, width=16, num_inference_steps=3)
    original = b.resolve_value("conditioner").clone()
    edit = b.preview_edit("conditioner", 0.5, mode="scale"); b.commit_edit(edit["act_id"])
    b.step("block")
    assert torch.equal(b.state["execution"]["frame"]["locals"]["encoder_hidden_states"], original*0.5)
    b.restore(parent); b.step("denoise")
    partial = b.checkpoint("stochastic-step-one")
    b.generate(); expected = tensor_digest(b.resolve_value("image"))
    assert torch.equal(b.resolve_value("image"), torch.from_numpy(stock))
    b.restore(partial); b.generate()
    assert tensor_digest(b.resolve_value("image")) == expected


def test_held_consumer_uses_same_native_adapter_and_replays(pipeline, tmp_path):
    b = BenchSession.from_resident(pipeline, root=tmp_path/"held-consumers")
    parent = b.prepare("a", seed=7, height=16, width=16, num_inference_steps=2)
    result = b.discover({"protocol": "native-consumer-v1", "name": "rgb-response", "sites": [0, 1],
        "contrasts": [{"name": "attenuate", "mode": "scale", "value": 0.9}],
        "observable": {"kind": "image_mean"}, "held_contexts": [{"prompt": "b", "seed": 9}],
        "max_native_operations": 200})
    assert result["quality_status"] == "measured-native-response"
    assert b.artifacts.get(parent)["cursor"] == b.cursor()
    row = next(r for r in result["report"]["observations"] if r["phase"] == "heldout")
    assert b.replay_observation(row["evidence_ref"])["exact"]
    b.close()
    b.close()
