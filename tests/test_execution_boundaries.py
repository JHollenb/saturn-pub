import subprocess
import sys

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file

from saturn_pub import Act, Session
from saturn_pub.adapters.diffusion import DiffusionAdapter
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.experimental.memory import SharedQwenAdapter
from saturn_pub.experimental.phase import PhaseLaw, PhaseOriginAdapter
from saturn_pub.store import LocalStore


def _flux_fixture(*, granularity="operation"):
    adapter = Flux2KleinAdapter.tiny(granularity=granularity)
    generator = torch.Generator().manual_seed(11)
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(1)
    inputs = {
        "latent": torch.randn(1, 8, 4, generator=generator).transpose(1, 2),
        "conditioning": torch.randn(1, 3, 24, generator=generator),
        "img_ids": torch.zeros(1, 4, 4),
        "txt_ids": torch.zeros(1, 3, 4),
        "timesteps": scheduler.timesteps,
        "sigmas": scheduler.sigmas,
    }
    return adapter, inputs, scheduler


def test_qwen_operation_boundaries_restore_and_native_parity(tmp_path):
    adapter = QwenAdapter.tiny(granularity="operation")
    session = adapter.session([5, 7, 11])
    expected = adapter.native_logits([5, 7, 11])
    phases = []
    cuts = []
    while session.read("tokens").shape[1] == 3:
        cut = session.capture()
        phases.append(cut.execution_point["phase"])
        cuts.append(cut)
        assert set(cut.payload) == {slot["name"] for slot in cut.surface["slots"]}
        session.continue_()
    assert phases == ["embed", "layer", "layer", "normalization", "readout", "sample", "commit"]
    torch.testing.assert_close(session.read("logits"), expected, rtol=1e-5, atol=1e-7)

    store = LocalStore(tmp_path)
    saved = store.load(store.save(cuts[3]))
    replay = Session(adapter, saved.payload)
    replay.restore(saved)
    adapter.generate(replay)
    assert replay.compare(session)["equal_payload"]
    identifiers = [store.save(cut) for cut in cuts]
    final_identifier = store.save(session.capture())
    code = (
        "import sys, torch; torch.set_num_threads(int(sys.argv[1])); "
        "from saturn_pub import Session; "
        "from saturn_pub.adapters.qwen import QwenAdapter; "
        "from saturn_pub.store import LocalStore; from saturn_pub.values import describe; "
        "store=LocalStore(sys.argv[2]); expected=store.load(sys.argv[3]); "
        "adapter=QwenAdapter.tiny(granularity='operation'); "
        "cuts=[store.load(value) for value in sys.argv[4:]]; "
        "sessions=[Session(adapter, cut.payload) for cut in cuts]; "
        "[item.restore(cut) for item,cut in zip(sessions,cuts)]; "
        "[adapter.generate(item) for item in sessions]; "
        "assert all(describe(item.capture().payload)==describe(expected.payload) for item in sessions)"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(torch.get_num_threads()),
            str(tmp_path),
            final_identifier,
            *identifiers,
        ],
        check=True,
    )


def test_diffusion_operation_boundary_atomicity_and_refusals(tmp_path):
    adapter = DiffusionAdapter.tiny(granularity="operation")
    session = adapter.session(steps=1)
    parent = session.capture()
    session.continue_()
    assert session.capture().execution_point["phase"] == "scheduler"
    prediction_cut = session.capture()
    session.continue_()
    completed = session.capture()
    replay = session.fork(prediction_cut)
    replay.continue_()
    assert replay.compare(session)["equal_payload"]

    untouched = session.fork(parent)
    before = untouched.capture(retain=False).fingerprint
    with pytest.raises(ValueError, match="complete"):
        untouched.continue_(3)
    assert untouched.capture(retain=False).fingerprint == before
    root = Session(adapter, parent.payload)
    root.restore(parent)
    root_before = root.capture(retain=False).fingerprint
    failed_child = root.fork()
    with pytest.raises(ValueError, match="complete"):
        failed_child.continue_(3)
    assert root.capture(retain=False).fingerprint == root_before

    scheduler_branch = session.fork(prediction_cut)
    with pytest.raises(ValueError, match="read-only"):
        scheduler_branch.apply(Act.zero("latent"))
    malformed = completed.payload
    malformed["step"] = True
    with pytest.raises(ValueError, match="cursor"):
        adapter.validate(malformed)
    store = LocalStore(tmp_path)
    weight_path = tmp_path / "diffusion.safetensors"
    save_file(
        {name: value.detach().contiguous() for name, value in adapter.model.state_dict().items()},
        weight_path,
    )
    identifiers = [store.save(parent), store.save(prediction_cut)]
    final_identifier = store.save(completed)
    code = (
        "import sys, torch; torch.set_num_threads(int(sys.argv[1])); "
        "from saturn_pub import Session; "
        "from diffusers import DDIMScheduler; "
        "from safetensors.torch import load_file; "
        "from saturn_pub.adapters.diffusion import DiffusionAdapter; "
        "from saturn_pub.store import LocalStore; from saturn_pub.values import describe; "
        "store=LocalStore(sys.argv[2]); expected=store.load(sys.argv[3]); "
        "base=DiffusionAdapter.tiny(); base.model.load_state_dict(load_file(sys.argv[4])); "
        "adapter=DiffusionAdapter(base.model, DDIMScheduler(num_train_timesteps=32, "
        "clip_sample=False), granularity='operation'); "
        "cuts=[store.load(value) for value in sys.argv[5:]]; "
        "sessions=[Session(adapter, cut.payload) for cut in cuts]; "
        "[item.restore(cut) for item,cut in zip(sessions,cuts)]; "
        "[item.continue_(2 if item.capture().execution_point['phase']=='denoiser' else 1) "
        "for item in sessions]; "
        "assert all(describe(item.capture().payload)==describe(expected.payload) for item in sessions)"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(torch.get_num_threads()),
            str(tmp_path),
            final_identifier,
            str(weight_path),
            *identifiers,
        ],
        check=True,
    )


def test_flux_operation_readout_and_scheduler_are_distinct_native_steps(tmp_path):
    adapter, inputs, scheduler = _flux_fixture()
    session = adapter.session(**inputs)
    with torch.inference_mode():
        native_noise = adapter.model(
            hidden_states=inputs["latent"],
            encoder_hidden_states=inputs["conditioning"],
            timestep=scheduler.timesteps[0].expand(1) / 1000,
            img_ids=inputs["img_ids"],
            txt_ids=inputs["txt_ids"],
            return_dict=False,
        )[0]
        native = scheduler.step(
            native_noise, scheduler.timesteps[0], inputs["latent"], return_dict=False
        )[0]

    boundaries = []
    new_boundary_cuts = []
    while session._state["step"] == 0:
        cut = session.capture()
        boundaries.append(cut.execution_point["phase"])
        if cut.execution_point["phase"] in {"readout", "scheduler"}:
            new_boundary_cuts.append(cut)
        session.continue_()
    assert boundaries == [
        "project",
        "joint",
        "joint",
        "joint",
        "single",
        "single",
        "readout",
        "scheduler",
    ]
    assert torch.equal(session.read("latent"), native)
    store = LocalStore(tmp_path)
    weight_path = tmp_path / "flux.safetensors"
    save_file(
        {name: value.detach().contiguous() for name, value in adapter.model.state_dict().items()},
        weight_path,
    )
    identifiers = [store.save(cut) for cut in new_boundary_cuts]
    final_identifier = store.save(session.capture())
    code = (
        "import sys, torch; torch.set_num_threads(int(sys.argv[1])); "
        "from saturn_pub import Session; "
        "from diffusers import FlowMatchEulerDiscreteScheduler; "
        "from safetensors.torch import load_file; "
        "from saturn_pub.adapters.flux2 import Flux2KleinAdapter; "
        "from saturn_pub.store import LocalStore; from saturn_pub.values import describe; "
        "store=LocalStore(sys.argv[2]); expected=store.load(sys.argv[3]); "
        "base=Flux2KleinAdapter.tiny(); base.model.load_state_dict(load_file(sys.argv[4])); "
        "adapter=Flux2KleinAdapter(base.model, FlowMatchEulerDiscreteScheduler(), "
        "granularity='operation'); cuts=[store.load(value) for value in sys.argv[5:]]; "
        "sessions=[Session(adapter, cut.payload) for cut in cuts]; "
        "[item.restore(cut) for item,cut in zip(sessions,cuts)]; "
        "[adapter.finish(item) for item in sessions]; "
        "assert all(describe(item.capture().payload)==describe(expected.payload) for item in sessions)"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(torch.get_num_threads()),
            str(tmp_path),
            final_identifier,
            str(weight_path),
            *identifiers,
        ],
        check=True,
    )


def test_adapter_freeze_guard_requires_rewrap():
    adapter = QwenAdapter.tiny()
    session = adapter.session([5, 7, 11])
    with torch.no_grad():
        next(adapter.model.parameters()).add_(1)
    with pytest.raises(ValueError, match="rewrap"):
        session.continue_()


def test_adapter_freeze_guard_refuses_same_version_parameter_replacement():
    adapter = QwenAdapter.tiny()
    session = adapter.session([5, 7, 11])
    original = adapter.model.model.embed_tokens.weight
    replacement = torch.nn.Parameter(original.detach().clone())
    with torch.no_grad():
        while replacement._version < original._version:
            replacement.add_(0)
    assert replacement._version == original._version
    adapter.model.model.embed_tokens.weight = replacement
    with pytest.raises(ValueError, match="parameters or buffers changed"):
        session.capture()


def test_live_numeric_environment_drift_is_refused_before_capture():
    original = torch.are_deterministic_algorithms_enabled()
    adapter = QwenAdapter.tiny()
    session = adapter.session([5, 7, 11])
    try:
        torch.use_deterministic_algorithms(not original)
        with pytest.raises(ValueError, match="numerical environment drifted"):
            session.capture()
    finally:
        torch.use_deterministic_algorithms(original)
    session.capture()


def test_operation_macro_steps_are_atomic_at_token_and_denoise_boundaries():
    qwen = QwenAdapter.tiny(granularity="operation")
    token_session = qwen.session([5, 7, 11])
    token_parent = token_session.capture().fingerprint
    with pytest.raises(ValueError, match="macro-step limit"):
        token_session.step(granularity="token", max_steps=6)
    assert token_session.capture().fingerprint == token_parent
    token_session.step(granularity="token", max_steps=7)
    assert token_session.read("tokens").shape[1] == 4

    diffusion = DiffusionAdapter.tiny(granularity="operation")
    denoise_session = diffusion.session(steps=1)
    denoise_parent = denoise_session.capture().fingerprint
    with pytest.raises(ValueError, match="macro-step limit"):
        denoise_session.step(granularity="denoise", max_steps=1)
    assert denoise_session.capture().fingerprint == denoise_parent
    denoise_session.step(granularity="denoise", max_steps=2)
    assert denoise_session.capture().execution_point["logical_step"] == 1


def test_durable_shared_history_cut_and_bounded_capacity(tmp_path):
    source = QwenAdapter.tiny()
    adapter = SharedQwenAdapter(source, capacity=2)
    session = adapter.session([5, 7], 11)
    initial_accounting = adapter.accounting(session._state)
    assert initial_accounting["durable_ancestor_dedup_eligible"]
    assert not initial_accounting["live_session_ancestor_shared"]
    assert not initial_accounting["reduced_attention_work_claim"]
    assert initial_accounting["attention_rows"] == 2
    expected = source.native_logits([5, 7, 11])
    session.continue_()
    assert adapter.accounting(session._state)["attention_rows"] == 3
    torch.testing.assert_close(session.read("logits"), expected, rtol=1e-5, atol=1e-7)
    cut = session.capture()
    store = LocalStore(tmp_path)
    cut_identifier = store.save(cut)
    loaded = store.load(cut_identifier)
    replay = Session(adapter, loaded.payload)
    replay.restore(loaded)
    next_token = int(session.read("tokens")[0, -1])
    expected_next = source.native_logits([5, 7, 11, next_token])
    replay.continue_()
    torch.testing.assert_close(replay.read("logits"), expected_next, rtol=1e-5, atol=1e-7)
    final_identifier = store.save(replay.capture())
    code = (
        "import sys, torch; torch.set_num_threads(int(sys.argv[1])); "
        "from saturn_pub import Session; from saturn_pub.adapters.qwen import QwenAdapter; "
        "from saturn_pub.experimental.memory import SharedQwenAdapter; "
        "from saturn_pub.store import LocalStore; from saturn_pub.values import describe; "
        "store=LocalStore(sys.argv[2]); cut=store.load(sys.argv[3]); "
        "expected=store.load(sys.argv[4]); source=QwenAdapter.tiny(); "
        "adapter=SharedQwenAdapter(source,capacity=2); session=Session(adapter,cut.payload); "
        "session.restore(cut); session.continue_(); "
        "assert describe(session.capture().payload)==describe(expected.payload)"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(torch.get_num_threads()),
            str(tmp_path),
            cut_identifier,
            final_identifier,
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="capacity"):
        replay.continue_()


def test_phase_origin_cut_moves_metadata_without_rewriting_keys(tmp_path):
    rng = np.random.default_rng(7)
    adapter = PhaseOriginAdapter(PhaseLaw.from_inv_freq([1.0, 0.01]), shift=512)
    session = adapter.session(rng.normal(size=(3, 2, 2)), rng.normal(size=(2, 2)))
    before = session.read("local_keys").numpy().tobytes()
    session.continue_()
    first_scores = session.read("scores")
    assert session.read("local_keys").numpy().tobytes() == before
    cut = session.capture()
    store = LocalStore(tmp_path)
    loaded = store.load(store.save(cut))
    replay = Session(adapter, loaded.payload)
    replay.restore(loaded)
    replay.continue_()
    assert replay.read("local_keys").numpy().tobytes() == before
    assert torch.equal(replay.read("scores"), first_scores)
    final_identifier = store.save(replay.capture())
    code = (
        "import sys; from saturn_pub import Session; "
        "from saturn_pub.experimental.phase import PhaseLaw,PhaseOriginAdapter; "
        "from saturn_pub.store import LocalStore; from saturn_pub.values import describe; "
        "store=LocalStore(sys.argv[1]); cut=store.load(sys.argv[2]); "
        "expected=store.load(sys.argv[3]); "
        "adapter=PhaseOriginAdapter(PhaseLaw.from_inv_freq([1.0,0.01]),shift=512); "
        "session=Session(adapter,cut.payload); session.restore(cut); session.continue_(); "
        "assert describe(session.capture().payload)==describe(expected.payload)"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), store.save(cut), final_identifier],
        check=True,
    )
