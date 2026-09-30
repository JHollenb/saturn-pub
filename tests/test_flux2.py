# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import FlowMatchEulerDiscreteScheduler

from saturn_pub import Act
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore


def fixture(dtype=torch.float32):
    adapter = Flux2KleinAdapter.tiny()
    if dtype != torch.float32:
        adapter = Flux2KleinAdapter(adapter.model.to(dtype), FlowMatchEulerDiscreteScheduler())
    generator = torch.Generator().manual_seed(11)
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(2)
    inputs = {
        # Native pipelines pack channels-first latents as a transposed view.
        "latent": torch.randn(1, 8, 4, generator=generator).to(dtype).transpose(1, 2),
        "conditioning": torch.randn(1, 3, 24, generator=generator).to(dtype),
        "img_ids": torch.zeros(1, 4, 4),
        "txt_ids": torch.zeros(1, 3, 4),
        "timesteps": scheduler.timesteps,
        "sigmas": scheduler.sigmas,
    }
    return adapter, inputs, scheduler


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_flow_parity_and_block_replay(tmp_path, dtype):
    adapter, inputs, scheduler = fixture(dtype)
    session = adapter.session(**inputs)
    with torch.inference_mode():
        native = inputs["latent"]
        for timestep in scheduler.timesteps:
            noise = adapter.model(
                hidden_states=native,
                encoder_hidden_states=inputs["conditioning"],
                timestep=timestep.expand(1).to(dtype) / 1000,
                img_ids=inputs["img_ids"],
                txt_ids=inputs["txt_ids"],
                return_dict=False,
            )[0]
            native = scheduler.step(noise, timestep, native, return_dict=False)[0]
    session.continue_(4)  # projection and three joint blocks
    assert session.inspect().boundary == "diffusion-step:0/after:joint.2"
    parent = session.capture()
    store = LocalStore(tmp_path)
    loaded = store.load(store.save(parent))
    replay = session.fork(loaded)
    adapter.finish(session)
    adapter.finish(replay)
    assert torch.equal(session.read("latent"), native)
    assert session.compare(replay)["equal_payload"]
    candidate = session.fork(parent)
    candidate.apply(Act.zero("text"))
    adapter.finish(candidate)
    assert not torch.equal(candidate.read("latent"), native)
    assert session.restore(parent).to_dict()["verified_exact"]


def test_streamed_matches_resident_and_preserves_resident_contract():
    resident, inputs_r, _ = fixture()
    streamed = Flux2KleinAdapter.tiny(residency="streamed", device="cpu")
    _, inputs_s, _ = fixture()
    # Resident execution keeps the historical contract verbatim (no residency field),
    # so existing 4B cuts and receipts are unaffected; streamed declares its mode.
    assert "residency" not in resident.execution
    assert streamed.execution["residency"]["mode"] == "streamed"
    left = resident.session(**inputs_r)
    right = streamed.session(**inputs_s)
    resident.finish(left)
    streamed.finish(right)
    assert torch.equal(left.read("latent"), right.read("latent"))
    streamed.validate_execution()  # host weights never moved; frozen guard holds


def test_refuses_klein_kv_pipeline_with_reason():
    class Flux2KleinKVPipeline:  # stand-in: the real class needs 9B-KV weights
        pass

    with pytest.raises(ValueError, match="distinct trajectory ABI"):
        Flux2KleinAdapter.from_pipeline(Flux2KleinKVPipeline())


def test_wrong_boundary_write_and_incomplete_closure():
    adapter, inputs, _ = fixture()
    session = adapter.session(**inputs)
    session.continue_(4)
    parent = session.capture()
    with pytest.raises(ValueError, match="invalidate"):
        session.apply(Act.zero("conditioning"))
    assert session.capture().fingerprint == parent.fingerprint
    with pytest.raises(ValueError, match="shape/dtype"):
        session.apply(Act.replace("text", torch.zeros(1, 3, 16)))
    bad = parent.payload
    bad["context"].pop("temb")
    with pytest.raises(ValueError, match="incomplete"):
        adapter.validate(bad)
