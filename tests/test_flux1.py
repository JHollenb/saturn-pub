# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import FlowMatchEulerDiscreteScheduler

from saturn_pub import Act
from saturn_pub.adapters.flux1 import Flux1Adapter
from saturn_pub.store import LocalStore


def fixture(dtype=torch.float32, *, guidance=False, residency="resident", device=None, seed=5):
    adapter = Flux1Adapter.tiny(guidance=guidance, residency=residency, device=device)
    if dtype != torch.float32:
        adapter = Flux1Adapter(
            adapter.model.to(dtype),
            FlowMatchEulerDiscreteScheduler(),
            granularity="block",
            residency=residency,
            device=device,
        )
    generator = torch.Generator().manual_seed(seed)
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(2)
    inputs = {
        # Native pipelines pack channels-first latents as a transposed view.
        "latent": torch.randn(1, 8, 5, generator=generator).to(dtype).transpose(1, 2),
        "conditioning": torch.randn(1, 3, 24, generator=generator).to(dtype),
        "pooled": torch.randn(1, 16, generator=generator).to(dtype),
        "img_ids": torch.zeros(1, 5, 3),
        "txt_ids": torch.zeros(1, 3, 3),
        "timesteps": scheduler.timesteps,
        "sigmas": scheduler.sigmas,
        "guidance": (3.5 if guidance else None),
    }
    return adapter, inputs, scheduler


def _native_trajectory(adapter, inputs, scheduler, dtype):
    native = inputs["latent"]
    kwargs = {}
    if inputs["guidance"] is not None:
        kwargs["guidance"] = torch.full([1], inputs["guidance"], dtype=torch.float32)
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            noise = adapter.model(
                hidden_states=native,
                encoder_hidden_states=inputs["conditioning"],
                pooled_projections=inputs["pooled"],
                timestep=timestep.expand(1).to(dtype) / 1000,
                img_ids=inputs["img_ids"],
                txt_ids=inputs["txt_ids"],
                return_dict=False,
                **kwargs,
            )[0]
            native = scheduler.step(noise, timestep, native, return_dict=False)[0]
    return native


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("guidance", [False, True])
def test_native_flow_parity_and_block_replay(tmp_path, dtype, guidance):
    adapter, inputs, scheduler = fixture(dtype, guidance=guidance)
    native = _native_trajectory(adapter, inputs, scheduler, dtype)
    session = adapter.session(**inputs)
    session.continue_(3)  # projection and two joint blocks
    assert session.inspect().boundary == "diffusion-step:0/after:joint.1"
    parent = session.capture()
    store = LocalStore(tmp_path)
    loaded = store.load(store.save(parent))
    replay = session.fork(loaded)
    adapter.finish(session)
    adapter.finish(replay)
    assert torch.equal(session.read("latent"), native)
    assert session.compare(replay)["equal_payload"]
    # Same-parent fork isolation: a zeroed route diverges from the native suffix.
    candidate = session.fork(parent)
    candidate.apply(Act.zero("text"))
    adapter.finish(candidate)
    assert not torch.equal(candidate.read("latent"), native)
    assert session.restore(parent).to_dict()["verified_exact"]


@pytest.mark.parametrize("guidance", [False, True])
def test_streamed_matches_resident_and_declares_residency(guidance):
    resident, inputs_r, _ = fixture(guidance=guidance)
    streamed, inputs_s, _ = fixture(guidance=guidance, residency="streamed", device="cpu")
    assert resident.execution["residency"]["mode"] == "resident"
    assert streamed.execution["residency"]["mode"] == "streamed"
    # Residency is declared in the execution contract, so cuts do not cross modes.
    assert resident.execution != streamed.execution
    left = resident.session(**inputs_r)
    right = streamed.session(**inputs_s)
    resident.finish(left)
    streamed.finish(right)
    # CPU-only equality exercises the streaming call path; cross-device bit-identity
    # is validated on GPU (RTX 4080). Same seed gives both adapters identical weights.
    assert torch.equal(left.read("latent"), right.read("latent"))
    # Host weights never moved, so the frozen-model guard still holds after streaming.
    streamed.validate_execution()


def test_fresh_process_replay_is_exact(tmp_path):
    adapter, inputs, scheduler = fixture()
    native = _native_trajectory(adapter, inputs, scheduler, torch.float32)
    session = adapter.session(**inputs)
    session.continue_(2)  # mid-trajectory cut
    cut = session.capture()
    digest = LocalStore(tmp_path).save(cut)
    # A fresh adapter/store (new-process stand-in) hydrates the cut and continues.
    fresh_adapter, _, _ = fixture()
    reloaded = LocalStore(tmp_path).load(digest)
    from saturn_pub import Session

    resumed = Session(fresh_adapter, reloaded.payload, parent=reloaded.parent)
    resumed.restore(reloaded)
    fresh_adapter.finish(resumed)
    assert torch.equal(resumed.read("latent"), native)


def test_guidance_contract_and_boundary_refusals():
    schnell, inputs, _ = fixture(guidance=False)
    with pytest.raises(ValueError, match="no guidance embedding"):
        schnell.session(**{**inputs, "guidance": 3.5})
    dev, dev_inputs, _ = fixture(guidance=True)
    with pytest.raises(ValueError, match="requires a guidance scale"):
        dev.session(**{**dev_inputs, "guidance": None})
    session = schnell.session(**inputs)
    session.continue_(3)
    parent = session.capture()
    with pytest.raises(ValueError, match="invalidate"):
        session.apply(Act.zero("conditioning"))
    assert session.capture().fingerprint == parent.fingerprint
    with pytest.raises(ValueError, match="shape/dtype"):
        session.apply(Act.replace("text", torch.zeros(1, 3, 16)))
    bad = parent.payload
    bad["context"].pop("temb")
    with pytest.raises(ValueError, match="incomplete"):
        schnell.validate(bad)
