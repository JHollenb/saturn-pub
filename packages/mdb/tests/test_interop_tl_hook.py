"""``names.tl_hook`` (SC7): TL-layout tensors reach the callback in mdb layout and come back in
TL layout. Shape tests need only torch; the bridge test needs transformer_lens 4.x and a cached
Qwen tokenizer (as ``test_interop.py``)."""
from __future__ import annotations

import warnings

import pytest

from mdb.interop import names

torch = pytest.importorskip("torch")

B, T, H, DH = 2, 5, 3, 4


def test_heads_flat_hook_sees_mdb_layout_and_returns_tl_layout():
    x = torch.arange(B * T * H * DH, dtype=torch.float32).reshape(B, T, H, DH)    # TL hook_z
    seen = {}

    def fn(m, hook):
        seen["shape"], seen["hook"] = tuple(m.shape), hook
        m = m.clone()
        m[0, -1, 1 * DH:2 * DH] = 0                      # mdb-flat: head 1 of the last row
        return m

    out = names.tl_hook("L3.A", fn)(x, "H")
    assert seen == {"shape": (B, T, H * DH), "hook": "H"}
    want = x.clone()
    want[0, -1, 1] = 0                                   # the same patch on the 4-D TL tensor
    assert torch.equal(out, want)
    # the mdb view is the TL tensor flattened over (head, head_dim), head-major
    got = {}
    names.tl_hook("v", lambda m, h: got.setdefault("m", m.clone()))(x, None)
    assert torch.equal(got["m"], x.reshape(B, T, H * DH))


def test_identity_none_and_every_layout_round_trip():
    x = torch.randn(B, T, H, DH)
    assert torch.equal(names.tl_hook("L0.A", lambda m, h: m)(x, None), x)
    assert names.tl_hook("L0.A", lambda m, h: None)(x, None) is None
    seq = torch.randn(B, T, 7)
    assert torch.equal(names.tl_hook("L1.o", lambda m, h: m)(seq, None), seq)
    kt = torch.randn(B, H, T, DH)                        # TL q_norm: (batch, head, pos, d_head)
    seen = {}
    out = names.tl_hook("qn", lambda m, h: seen.setdefault("s", m.shape) and m)(kt, None)
    assert tuple(seen["s"]) == (B, T, H, DH) and torch.equal(out, kt)


def test_wrong_rank_and_wrong_return_shape_refuse():
    hook = names.tl_hook("L0.A", lambda m, h: m)
    with pytest.raises(ValueError, match="4-D"):
        hook(torch.zeros(B, T, H * DH), None)            # the mdb-flat tensor is not a TL hook_z
    with pytest.raises(ValueError, match="expected the mdb layout shape"):
        names.tl_hook("L0.A", lambda m, h: m[:, :, :-1])(torch.zeros(B, T, H, DH), None)
    with pytest.raises(names.NoName):
        names.tl_hook("L0.sg", lambda m, h: m)           # no TL hook for silu(g)


def test_bridge_head_patch_equals_manual_4d_patch():
    pytest.importorskip("transformer_lens")
    from transformers import AutoTokenizer

    from mdb.engine.models import tiny_model
    from mdb.interop import tl

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("no cached Qwen tokenizer")
    warnings.filterwarnings("ignore")
    model = tiny_model("qwen2", "fp32", layers=2)
    bridge = tl.bridge(model, tok)
    ids = torch.tensor([[1, 5, 9, 3, 7, 2]])
    hook_name = names.to_tl("L1.A")
    dh = int(model.config.hidden_size // model.config.num_attention_heads)

    def manual(x, hook):                                 # TL 4-D indexing
        x = x.clone()
        x[0, -1, 1] = 0
        return x

    def mdb_flat(m, hook):                               # the same patch in mdb layout
        m = m.clone()
        m[0, -1, dh:2 * dh] = 0
        return m

    with torch.no_grad():
        base = bridge.run_with_hooks(ids, fwd_hooks=[])
        a = bridge.run_with_hooks(ids, fwd_hooks=[(hook_name, manual)])
        b = bridge.run_with_hooks(ids, fwd_hooks=[(hook_name, names.tl_hook("L1.A", mdb_flat))])
    assert torch.equal(a, b) and not torch.equal(a, base)
