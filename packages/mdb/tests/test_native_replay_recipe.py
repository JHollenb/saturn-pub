from copy import deepcopy
from types import SimpleNamespace
import pytest
from mdb.native_replay_recipe import NativeReplay, SCHEMA, seal, validate


def recipe():
    return seal({"schema": SCHEMA, "model_identity": "model", "execution": {"profile": "exact"},
        "roots": [{"id": "r", "token_ids": [1, 2]}],
        "events": [{"id": "first", "session": "r", "kind": "continue", "steps": 1}]})


def test_payloads_and_stale_or_forward_references_are_refused():
    original = recipe()
    tampered = deepcopy(original)
    tampered["roots"][0]["token_ids"][0] = 9
    with pytest.raises(ValueError, match="seal"):
        validate(tampered)
    for modify in (
        lambda v: v["roots"][0].update(activations=[1]),
        lambda v: v["events"][0].update(kind="restore", source="future"),
        lambda v: v["events"][0].update(steps=True),
    ):
        body = deepcopy(original)
        body.pop("fingerprint")
        modify(body)
        with pytest.raises(ValueError):
            validate(seal(body))


def test_profile_checked_before_creating_native_sessions():
    with pytest.raises(ValueError, match="incompatible"):
        NativeReplay(recipe(), SimpleNamespace(model_identity="other", execution={"profile": "exact"}))


def test_tiny_native_replay_materializes_from_tokens_and_original_parameters():
    torch = pytest.importorskip("torch")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from saturn_pub import StateAddress
    from mdb.prompt_qwen import PromptQwenAdapter
    torch.manual_seed(19)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    adapter = PromptQwenAdapter(model)
    native = adapter.session([1, 2])
    native.continue_(5)  # embed, two layers, norm, readout
    target = native.capture(retain=False)
    body = {"schema": SCHEMA, "model_identity": adapter.model_identity, "execution": dict(adapter.execution),
        "roots": [{"id": "r", "token_ids": [1, 2]}],
        "events": [{"id": "readout", "session": "r", "kind": "continue", "steps": 5,
                    "expected_cut": target.fingerprint}]}
    replay = NativeReplay(seal(body), adapter).run()
    assert torch.equal(replay.materialize("readout", StateAddress("logits")), native.read("logits"))
    assert replay.cuts["readout"].fingerprint == target.fingerprint
    with pytest.raises(ValueError, match="fresh"):
        replay.run()
