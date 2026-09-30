"""Fast interop check: Saturn's layer:L boundary == the HF residual-stream input.

Skipped unless the ``interop`` extra (sae_lens + transformer_lens) is installed. The
libraries only gate the suite; the assertions run on a tiny random GPT-2 with a synthetic
SAE-like direction so they stay fast and download nothing. Building a TransformerLens
``HookedTransformer`` from a random tiny config is awkward, so this exercises the same
numerical contract the example relies on -- Saturn's carrier equals ``resid_pre[L]`` and a
carrier ablation matches the identical intervention applied through an HF forward hook --
against HF's own kernels (which TransformerLens ``no_processing`` is built to match).
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("sae_lens")
pytest.importorskip("transformer_lens")

import torch  # noqa: E402

from saturn_pub import Session  # noqa: E402
from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from saturn_pub.store import LocalStore  # noqa: E402

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "interop_saelens.py"
_spec = importlib.util.spec_from_file_location("interop_saelens", _EXAMPLE)
interop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(interop)

PROMPT = [5, 7, 11, 3]
LAYER = 1
TOL = {"rtol": 1e-4, "atol": 1e-5}


@pytest.fixture
def adapter():
    return DecoderAdapter.tiny("gpt2")


def _hf_resid_pre(adapter, tokens, layer):
    # hidden_states[i] is the residual stream entering block i (resid_pre[i]).
    with torch.inference_mode():
        out = adapter.model(tokens, output_hidden_states=True)
    return out.hidden_states[layer][:, -1]


def _synthetic_direction(adapter, seed=0):
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(adapter.hidden_size, generator=generator)
    return direction / direction.norm()


def test_saturn_boundary_equals_hf_resid_pre(adapter):
    tokens = torch.tensor([PROMPT])
    session = interop.saturn_step_to_layer(adapter, PROMPT, LAYER)
    # Saturn boundary "layer:L" is the address of TransformerLens blocks.L.hook_resid_pre.
    assert session.inspect().boundary == f"layer:{LAYER}"
    assert interop.hook_name(LAYER) == f"blocks.{LAYER}.hook_resid_pre"
    torch.testing.assert_close(
        session.read("hidden")[0, 0], _hf_resid_pre(adapter, tokens, LAYER)[0], **TOL
    )


def test_carrier_ablation_matches_hf_forward_hook(adapter):
    tokens = torch.tensor([PROMPT])
    direction = _synthetic_direction(adapter)
    activation = 6.0
    session = interop.saturn_step_to_layer(adapter, PROMPT, LAYER)
    parent = session.capture()
    native, candidate = session.fork(parent), session.fork(parent)

    carrier_before = candidate.read("hidden").clone()
    candidate.apply(interop.ablation_act(direction, activation))
    expected_carrier = carrier_before - activation * direction.reshape(1, 1, -1)
    torch.testing.assert_close(candidate.read("hidden"), expected_carrier, **TOL)

    adapter.generate(native, 1)
    adapter.generate(candidate, 1)
    native_logits = native.read("logits")[0]
    candidate_logits = candidate.read("logits")[0]
    assert not torch.allclose(native_logits, candidate_logits)  # a real intervention

    block = adapter._layer_modules[LAYER]

    def pre(module, args, kwargs):
        if args:
            hidden = args[0].clone()
            hidden[:, -1, :] = hidden[:, -1, :] - activation * direction
            return (hidden, *args[1:]), kwargs
        hidden = kwargs["hidden_states"].clone()
        hidden[:, -1, :] = hidden[:, -1, :] - activation * direction
        return args, {**kwargs, "hidden_states": hidden}

    handle = block.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        with torch.inference_mode():
            hf_ablated = adapter.model(tokens).logits[:, -1]
    finally:
        handle.remove()
    torch.testing.assert_close(candidate_logits, hf_ablated[0], **TOL)

    # Native branch (no hook) still matches the clean HF forward.
    torch.testing.assert_close(native_logits, adapter.native_logits(PROMPT)[0], **TOL)


def test_ablated_cut_replays_from_store(adapter, tmp_path):
    direction = _synthetic_direction(adapter)
    session = interop.saturn_step_to_layer(adapter, PROMPT, LAYER)
    parent = session.capture()
    candidate = session.fork(parent)
    candidate.apply(interop.ablation_act(direction, 6.0))
    candidate_cut = candidate.capture()

    in_process = session.fork(candidate_cut)
    adapter.generate(in_process, 3)
    expected = in_process.read("tokens")[0].tolist()

    store = LocalStore(tmp_path)
    identifier = store.save(candidate_cut)
    reloaded = Session.from_cut(adapter, store.load(identifier))
    adapter.generate(reloaded, 3)
    assert reloaded.read("tokens")[0].tolist() == expected
