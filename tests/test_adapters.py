import subprocess
import sys

import pytest
import torch
from diffusers import DDIMScheduler

from saturn_pub import Act
from saturn_pub.adapters.diffusion import DiffusionAdapter
from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.store import LocalStore


@pytest.fixture
def qwen():
    return QwenAdapter.tiny()


def test_qwen_native_multi_token_parity_and_layer_restore(qwen):
    session = qwen.session([5, 7, 11])
    for _ in range(3):
        tokens = session.read("tokens")[0].tolist()
        expected = qwen.native_logits(tokens)
        session.continue_(2)
        cut = session.capture()
        branch = session.fork(cut)
        qwen.generate(branch)
        torch.testing.assert_close(branch.read("logits"), expected, rtol=1e-5, atol=1e-7)
        qwen.generate(session)
        assert session.compare(branch)["equal_payload"]
        session.restore(cut)
        qwen.generate(session)
        torch.testing.assert_close(session.read("logits"), expected, rtol=1e-5, atol=1e-7)


def test_qwen_intervention_and_exact_uninstall(qwen):
    session = qwen.session([5, 7, 11])
    session.continue_(qwen.layers + 1)
    parent = session.capture()
    native, candidate = session.fork(parent), session.fork(parent)
    candidate.apply(Act.zero("hidden"))
    qwen.generate(native)
    qwen.generate(candidate)
    assert not torch.equal(native.read("logits"), candidate.read("logits"))
    candidate.restore(parent)
    qwen.generate(candidate)
    assert native.compare(candidate)["equal_payload"]
    with pytest.raises(ValueError, match="shape"):
        session.apply(Act.replace("hidden", torch.zeros(1)))
    with pytest.raises(ValueError, match="non-finite|invalid hidden"):
        session.apply(Act.replace("hidden", torch.full_like(session.read("hidden"), torch.nan)))


def test_qwen_durable_layer_cut_and_blob_corruption(qwen, tmp_path):
    session = qwen.session([5, 7, 11])
    session.continue_(2)
    store = LocalStore(tmp_path)
    identifier = store.save(session.capture())
    restored = store.load(identifier)
    session.restore(restored)
    code = (
        "import torch, sys; torch.set_num_threads(1); "
        "from saturn_pub.adapters.qwen import QwenAdapter; "
        "from saturn_pub import Session; from saturn_pub.store import LocalStore; "
        "a=QwenAdapter.tiny(); c=LocalStore(sys.argv[1]).load(sys.argv[2]); "
        "s=Session(a,c.payload); s.restore(c); a.generate(s); "
        "assert int(s.read('tokens')[0,-1])==56"
    )
    subprocess.run([sys.executable, "-c", code, str(tmp_path), identifier], check=True)
    with pytest.raises(ValueError, match="incompatible"):
        QwenAdapter.tiny(seed=8).session([5]).restore(restored)
    blob = next((tmp_path / "blobs").iterdir())
    blob.write_bytes(blob.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="digest"):
        store.load(identifier)


def test_diffusion_matches_native_steps_and_restores(tmp_path):
    adapter = DiffusionAdapter.tiny()
    session = adapter.session(steps=4)
    latent = session.read("latent")
    scheduler = DDIMScheduler.from_config(adapter.scheduler_config)
    scheduler.set_timesteps(4)
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            noise = adapter.model(latent, timestep).sample
            latent = scheduler.step(noise, timestep, latent, eta=0.0).prev_sample
    session.continue_()
    parent = session.capture()
    store = LocalStore(tmp_path)
    restored = store.load(store.save(parent))
    session.continue_(3)
    assert torch.equal(session.read("latent"), latent)
    image = adapter.decode(session)
    assert image.shape == (1, 3, 8, 8)
    session.restore(restored)
    session.continue_(3)
    assert torch.equal(session.read("latent"), latent)
    with pytest.raises(ValueError, match="complete"):
        session.continue_()


def test_conditional_diffusion_native_vae_and_shape_contract():
    from diffusers import AutoencoderKL, UNet2DConditionModel

    with torch.random.fork_rng():
        torch.manual_seed(7)
        unet = UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(8, 16),
            norm_num_groups=4,
            cross_attention_dim=8,
            attention_head_dim=2,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        )
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            block_out_channels=(8,),
            latent_channels=4,
            norm_num_groups=4,
            sample_size=8,
        )
    adapter = DiffusionAdapter(
        unet, DDIMScheduler(num_train_timesteps=32), decoder=vae, scaling_factor=0.18215
    )
    conditioning = torch.ones(1, 3, 8)
    session = adapter.session(steps=2, conditioning=conditioning)
    parent = session.capture()
    branch = session.fork(parent)
    branch.apply(Act.zero("conditioning"))
    session.continue_(2)
    branch.continue_(2)
    assert not torch.equal(session.read("latent"), branch.read("latent"))
    with torch.inference_mode():
        expected = (vae.decode(session.read("latent") / 0.18215).sample / 2 + 0.5).clamp(0, 1)
    assert torch.equal(adapter.decode(session), expected)
    with pytest.raises(ValueError, match="conditioning"):
        adapter.session(steps=2, conditioning=torch.ones(1, 3, 7))
    with pytest.raises(ValueError, match="conditioning"):
        adapter.session(steps=2)
