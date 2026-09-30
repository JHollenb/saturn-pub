"""Parity, replay, isolation, custody, and refusal tests for the native AR families.

Every fixture is a tiny random-config model; no checkpoints are downloaded. Real-weight
validation lives in ``experiments/family_validation`` and runs on a scheduler.
"""

import json
import subprocess
import sys

import pytest
import torch

from saturn_pub import Act
from saturn_pub.adapters import load, supported_families
from saturn_pub.adapters.decoder import SUPPORTED_FAMILIES, DecoderAdapter
from saturn_pub.adapters.mamba import MambaAdapter
from saturn_pub.store import LocalStore

PROMPT = [5, 7, 11, 3]
# Tiny fp32 native kernels agree with the full forward to a few ulps; name the envelope.
TOL = {"rtol": 1e-4, "atol": 1e-5}


def _native_greedy(model, prompt, new_tokens):
    tokens = torch.tensor([prompt])
    with torch.inference_mode():
        out = model.generate(
            tokens,
            attention_mask=torch.ones_like(tokens),
            max_new_tokens=new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    return out[0].tolist()


@pytest.fixture(params=SUPPORTED_FAMILIES)
def decoder(request):
    return DecoderAdapter.tiny(request.param)


def test_decoder_stepped_matches_native_logits(decoder):
    session = decoder.session(PROMPT)
    session.continue_(decoder.layers + 2)  # embed + layers + fused readout/commit
    torch.testing.assert_close(session.read("logits"), decoder.native_logits(PROMPT), **TOL)


def test_decoder_greedy_matches_native_generate(decoder):
    expected = _native_greedy(decoder.model, PROMPT, 4)
    session = decoder.session(PROMPT)
    decoder.generate(session, tokens=4)
    assert session.read("tokens")[0].tolist() == expected


def test_decoder_fork_isolation_intervention_and_replay(decoder):
    session = decoder.session(PROMPT)
    session.continue_(decoder.layers + 1)  # complete stack, hidden still resident
    parent = session.capture()
    native, candidate = session.fork(parent), session.fork(parent)
    candidate.apply(Act.zero("hidden"))
    decoder.generate(native)
    decoder.generate(candidate)
    assert not torch.equal(native.read("logits"), candidate.read("logits"))
    replay_a, replay_b = session.fork(parent), session.fork(parent)
    decoder.generate(replay_a, 3)
    decoder.generate(replay_b, 3)
    assert replay_a.read("tokens")[0].tolist() == replay_b.read("tokens")[0].tolist()
    assert replay_a.compare(replay_b)["equal_payload"]


def test_decoder_write_guards(decoder):
    session = decoder.session(PROMPT)
    session.continue_(decoder.layers + 1)
    with pytest.raises(ValueError, match="shape"):
        session.apply(Act.replace("hidden", torch.zeros(1)))
    with pytest.raises(ValueError, match="non-finite|invalid hidden"):
        session.apply(Act.replace("hidden", torch.full_like(session.read("hidden"), torch.nan)))


def test_decoder_operation_granularity_exposes_phases_and_matches_native():
    adapter = DecoderAdapter.tiny("qwen2", granularity="operation")
    session = adapter.session(PROMPT)
    phases = []
    for _ in range(adapter.layers + 6):
        phases.append(session.inspect().execution_point["phase"])
        session.continue_()
        if session.read("tokens").shape[1] > len(PROMPT):
            break
    assert {"embed", "layer", "normalization", "readout", "sample", "commit"}.issubset(set(phases))
    assert session.read("logits") is not None
    torch.testing.assert_close(session.read("logits"), adapter.native_logits(PROMPT), **TOL)


# --- Mamba -----------------------------------------------------------------------------


def test_mamba_stepped_matches_native_logits():
    adapter = MambaAdapter.tiny()
    single = adapter.session([5])
    single.continue_(adapter.layers + 2)
    torch.testing.assert_close(single.read("logits"), adapter.native_logits([5]), **TOL)
    session = adapter.session(PROMPT)
    session.continue_(adapter.layers + 2)
    torch.testing.assert_close(session.read("logits"), adapter.native_logits(PROMPT), **TOL)


def test_mamba_greedy_matches_native_generate():
    adapter = MambaAdapter.tiny()
    expected = _native_greedy(adapter.model, PROMPT, 5)
    session = adapter.session(PROMPT)
    adapter.generate(session, tokens=5)
    assert session.read("tokens")[0].tolist() == expected


def test_mamba_fork_isolation_and_replay():
    adapter = MambaAdapter.tiny()
    session = adapter.session(PROMPT)
    session.continue_(adapter.layers + 1)
    parent = session.capture()
    native, candidate = session.fork(parent), session.fork(parent)
    candidate.apply(Act.zero("hidden"))
    adapter.generate(native)
    adapter.generate(candidate)
    assert not torch.equal(native.read("logits"), candidate.read("logits"))
    replay_a, replay_b = session.fork(parent), session.fork(parent)
    adapter.generate(replay_a, 3)
    adapter.generate(replay_b, 3)
    assert replay_a.read("tokens")[0].tolist() == replay_b.read("tokens")[0].tolist()


# --- durable custody across a fresh interpreter ----------------------------------------

_FRESH = """
import json, sys, torch
torch.set_num_threads(1)
from saturn_pub import Session
from saturn_pub.store import LocalStore
from saturn_pub.adapters.decoder import DecoderAdapter
from saturn_pub.adapters.mamba import MambaAdapter
root, manifest = sys.argv[1], json.loads(sys.argv[2])
store = LocalStore(root)
for family, entry in manifest.items():
    identifier, expected = entry
    adapter = MambaAdapter.tiny() if family == 'mamba' else DecoderAdapter.tiny(family)
    cut = store.load(identifier)
    session = Session(adapter, cut.payload)
    session.restore(cut)
    adapter.generate(session, 2)
    got = int(session.read('tokens')[0, -1])
    assert got == expected, (family, got, expected)
print('FRESH_OK', len(manifest))
"""


def test_all_families_capture_save_and_fresh_process_continuation(tmp_path):
    store = LocalStore(tmp_path)
    manifest = {}
    for family in (*SUPPORTED_FAMILIES, "mamba"):
        adapter = MambaAdapter.tiny() if family == "mamba" else DecoderAdapter.tiny(family)
        session = adapter.session(PROMPT)
        session.continue_(2)  # mid-layer cut
        cut = session.capture()
        identifier = store.save(cut)
        # identical continuation must be reproducible in-process before we trust the store
        reloaded = store.load(identifier)
        assert reloaded.fingerprint == identifier
        branch = session.fork(cut)
        adapter.generate(branch, 2)
        manifest[family] = [identifier, int(branch.read("tokens")[0, -1])]
    result = subprocess.run(
        [sys.executable, "-c", _FRESH, str(tmp_path), json.dumps(manifest)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().endswith(str(len(manifest)))


def test_store_rejects_incompatible_adapter(tmp_path):
    adapter = DecoderAdapter.tiny("llama")
    session = adapter.session(PROMPT)
    session.continue_(2)
    store = LocalStore(tmp_path)
    restored = store.load(store.save(session.capture()))
    with pytest.raises(ValueError, match="incompatible"):
        DecoderAdapter.tiny("llama", granularity="operation").session([5]).restore(restored)
    with pytest.raises(ValueError, match="incompatible"):
        DecoderAdapter.tiny("gpt2").session([5]).restore(restored)


# --- dispatch --------------------------------------------------------------------------


def test_load_dispatches_by_model_type():
    llama = DecoderAdapter.tiny("llama").model
    assert isinstance(load(llama), DecoderAdapter)
    mamba = MambaAdapter.tiny().model
    assert isinstance(load(mamba), MambaAdapter)
    assert set(SUPPORTED_FAMILIES) | {"mamba"} == set(supported_families())


# --- fail-closed refusals --------------------------------------------------------------


def _model(family, **overrides):
    from saturn_pub.adapters.decoder import families

    spec = families()[family]
    config = spec.config_cls(**{**dict(spec.tiny_config), **overrides})
    return spec.model_cls(config)


def test_refuses_unregistered_family():
    with pytest.raises(ValueError, match="unknown family|supported"):
        DecoderAdapter.tiny("bert")
    with pytest.raises(ValueError, match="unregistered|supported"):
        load(object())


def test_refuses_non_eager_attention():
    model = _model("qwen2", attn_implementation="sdpa")
    with pytest.raises(ValueError, match="eager"):
        DecoderAdapter(model)


def test_refuses_gpt2_wrong_activation():
    model = _model("gpt2", activation_function="relu")
    with pytest.raises(ValueError, match="gelu_new"):
        DecoderAdapter(model)


def test_refuses_llama_pretraining_tp():
    model = _model("llama", pretraining_tp=2)
    with pytest.raises(ValueError, match="pretraining_tp"):
        DecoderAdapter(model)


def test_refuses_gpt_neox_tied_embeddings():
    model = _model("gpt_neox", tie_word_embeddings=True)
    with pytest.raises(ValueError, match="untied"):
        DecoderAdapter(model)


def test_refuses_phi_qk_layernorm():
    model = _model("phi", qk_layernorm=True)
    with pytest.raises(ValueError, match="qk_layernorm"):
        DecoderAdapter(model)


def test_refuses_gemma_attention_bias():
    model = _model("gemma", attention_bias=True)
    with pytest.raises(ValueError, match="biasless"):
        DecoderAdapter(model)


def test_refuses_mixtral_invalid_topk():
    model = _model("mixtral", num_experts_per_tok=4, num_local_experts=4)
    with pytest.raises(ValueError, match="top-k"):
        DecoderAdapter(model)


def test_refuses_mistral_context_beyond_sliding_window():
    model = _model("mistral", sliding_window=3)
    adapter = DecoderAdapter(model)
    with pytest.raises(ValueError, match="sliding-window"):
        adapter.session([5, 7, 11, 3, 9])


def test_refuses_mamba_wrong_activation():
    from transformers import MambaConfig, MambaForCausalLM

    config = MambaConfig(
        vocab_size=64,
        hidden_size=32,
        state_size=16,
        num_hidden_layers=2,
        conv_kernel=4,
        expand=2,
        time_step_rank=4,
        hidden_act="relu",
    )
    with pytest.raises(ValueError, match="SiLU"):
        MambaAdapter(MambaForCausalLM(config))
