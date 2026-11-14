"""DecoderAdapter.session(prompt, prefill=K) (mdb M26/M35): K prompt rows in one batched call,
the rest stepped one row at a time with the prompt's tokens forced at ``sample``.

A default session (all rows but the last) has no ``prompt`` slot, so its fingerprints are those
of the session before the option existed; a stepped session carries the prompt as a slot, so
its cuts differ from the default's and a replay binds the procedure.
"""

import pytest

torch = pytest.importorskip("torch")

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402

PROMPT = [1, 5, 9, 3, 7, 2]


def _past_prompt(session, count):
    """Continue until the prompt's last row has committed its sampled token (logits read)."""
    for _ in range(10_000):
        state = session.capture(retain=False).payload
        if state["tokens"].shape[1] == count + 1:
            return state
        session.continue_()
    raise AssertionError("never committed the last prompt row")


@pytest.mark.parametrize("granularity", ["layer", "operation"])
def test_stepped_prompt_rows_force_the_prompt_tokens(granularity):
    adapter = DecoderAdapter.tiny("qwen2", granularity=granularity)
    default = _past_prompt(adapter.session(PROMPT), len(PROMPT))
    for prefill in (0, 2, len(PROMPT) - 2):
        session = adapter.session(PROMPT, prefill=prefill)
        root = session.capture(retain=False).payload
        assert root["tokens"].tolist() == [PROMPT[: prefill + 1]]
        assert root["prompt"].tolist() == [PROMPT]
        assert all(k.shape[-2] == prefill for k in root["keys"])
        state = _past_prompt(session, len(PROMPT))
        assert state["tokens"][:, :-1].tolist() == [PROMPT]   # every forced token committed
        assert torch.equal(state["tokens"], default["tokens"])
        assert all(k.shape[-2] == len(PROMPT) for k in state["keys"])
        # the same values up to rounding: a batched and a stepped row need not agree bitwise
        # (measured: this fp32 tiny model, prefill 0 and 2, max |diff| 3e-8; M5: Qwen2.5-1.5B)
        torch.testing.assert_close(state["logits"], default["logits"], rtol=0, atol=1e-5)


def test_default_session_has_no_prompt_slot_and_unchanged_fingerprints():
    adapter = DecoderAdapter.tiny("qwen2", granularity="operation")
    plain = adapter.session(PROMPT)
    explicit = adapter.session(PROMPT, prefill=len(PROMPT) - 1)
    assert "prompt" not in plain.capture(retain=False).payload
    assert plain.capture(retain=False).fingerprint == explicit.capture(retain=False).fingerprint
    stepped = adapter.session(PROMPT, prefill=0)
    assert stepped.capture(retain=False).fingerprint != plain.capture(retain=False).fingerprint
    with pytest.raises(ValueError, match="prompt"):
        adapter.read(plain.capture(retain=False).payload, "prompt")
    for bad in (-1, len(PROMPT), True, 1.0):
        with pytest.raises(ValueError, match="prefill must be an integer"):
            adapter.session(PROMPT, prefill=bad)


def test_a_stepped_state_whose_tokens_leave_the_prompt_is_invalid():
    adapter = DecoderAdapter.tiny("qwen2", granularity="operation")
    state = dict(adapter.session(PROMPT, prefill=1).capture(retain=False).payload)
    state["tokens"] = torch.tensor([[1, 6]], dtype=torch.long)
    with pytest.raises(ValueError, match="prompt"):
        adapter.validate_state(state)
