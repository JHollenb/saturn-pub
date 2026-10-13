import shutil
import pytest
import torch
from diffusers import (Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler)
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from mdb.bench import BenchSession, tensor_digest


def test_tiny_native_flux_rgb_and_block_resume(tmp_path):
    pytest.importorskip("mrun", reason="image execution requires the optional private mrun adapter")
    torch.manual_seed(43)
    backend = Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "a": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
        chat_template="{{ messages[0]['content'] }}")
    encoder = Qwen3ForCausalLM(Qwen3Config(vocab_size=3, hidden_size=8, intermediate_size=16,
        num_hidden_layers=28, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        max_position_embeddings=1024)).eval()
    denoiser = Flux2Transformer2DModel(in_channels=16, out_channels=16, num_layers=2,
        num_single_layers=2, attention_head_dim=8, num_attention_heads=2, joint_attention_dim=24,
        timestep_guidance_channels=16, axes_dims_rope=(2, 2, 2, 2)).eval()
    vae = AutoencoderKLFlux2(block_out_channels=(16, 32), latent_channels=4, norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",)*2, up_block_types=("UpDecoderBlock2D",)*2,
        layers_per_block=1).eval()
    pipe = Flux2KleinPipeline(transformer=denoiser, vae=vae, text_encoder=encoder, tokenizer=tokenizer,
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=32), is_distilled=True)
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        stock = pipe(prompt="a", generator=torch.Generator().manual_seed(3), height=16, width=16,
            num_inference_steps=2, guidance_scale=1.0, output_type="np").images
    b = BenchSession.from_resident(pipe, root=tmp_path/"session")
    parent = b.prepare("a", seed=3, height=16, width=16, num_inference_steps=2)
    b.step("block"); cut = b.checkpoint("double-stream")
    assert "encoder_hidden_states" in b.state["execution"]["frame"]["locals"]
    edit = b.preview_edit("residual", 1, mode="scale"); b.commit_edit(edit["act_id"])
    b.generate(); expected = tensor_digest(b.resolve_value("image"))
    assert torch.equal(b.resolve_value("image"), torch.from_numpy(stock))
    b.restore(cut); b.export(tmp_path/"export"); b.close(); shutil.rmtree(tmp_path/"session")
    fresh = BenchSession.resume(pipe, root=tmp_path/"export")
    fresh.generate(); assert tensor_digest(fresh.resolve_value("image")) == expected
    fresh.restore(parent); fresh.generate()
    assert tensor_digest(fresh.resolve_value("image")) == expected
    fresh.close()
    # BF16 encoders may own FP32 rotary buffers; closing must preserve their
    # exact bytes rather than round-trip them through the parameter dtype.
    encoder.to(dtype=torch.bfloat16)
    original = torch.tensor([0.123456789], dtype=torch.float32)
    encoder.register_buffer("bench_owner_probe", original.clone())
    owner = BenchSession.from_resident(pipe, root=tmp_path/"owner")
    owner.close()
    assert owner.adapter.phase.resident_phase == "none"
    assert encoder.bench_owner_probe.dtype == torch.float32
    assert torch.equal(encoder.bench_owner_probe, original)
