"""Attention port mechanics on tiny CPU tensors (no pretrained model)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mdb_lanes.attention_partition import dense_attention
from mdb_lanes.attention_port import (AttentionPort, FluxBackend, QwenBackend, class_flow, cut, floor_arms, full_probs,
                                invisible_below, kv_group, kv_groups, observe, paired, sham)

H, S, D, T = 3, 5, 4, 2  # heads, rows, head width, "text" rows (rows T: are "image")


def sdpa(q, k, v, attn_mask=None, **kwargs):
    """Stand-in native kernel with the diffusers [B, S, H, D] layout."""
    out = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), attn_mask=attn_mask)
    return out.permute(0, 2, 1, 3)


class FakeBlock(torch.nn.Module):
    """One transformer block: one native attention call through the patched module attribute."""

    def __init__(self, module, heads, calls=1):
        super().__init__()
        self.attn, self.module, self.calls = SimpleNamespace(heads=heads), module, calls

    def forward(self, hidden_states=None, encoder_hidden_states=None, qkv=None):
        out = None
        for _ in range(self.calls):
            out = self.module.dispatch_attention_fn(*qkv)
        return out


class FakeTransformer(torch.nn.Module):
    """Blocks run in order like diffusers / saturn_pub (joint blocks get encoder_hidden_states)."""

    def __init__(self, module, blocks, heads=H, text_rows=T, calls=1):
        super().__init__()
        self.text_rows = text_rows
        self.transformer_blocks = torch.nn.ModuleList([FakeBlock(module, heads, calls) for _ in range(blocks[0])])
        self.single_transformer_blocks = torch.nn.ModuleList([FakeBlock(module, heads, calls)
                                                              for _ in range(blocks[1])])

    def forward(self, q, k, v, text_rows=None):
        enc = torch.zeros(q.shape[0], self.text_rows if text_rows is None else text_rows, 1)
        outs = [b(encoder_hidden_states=enc, qkv=(q, k, v)) for b in self.transformer_blocks]
        outs += [b(encoder_hidden_states=None, qkv=(q, k, v)) for b in self.single_transformer_blocks]
        return outs


def flux_stub(blocks=(1, 0), text_rows=T, heads=H, calls=1):
    module = SimpleNamespace(dispatch_attention_fn=sdpa)
    transformer = FakeTransformer(module, blocks, heads, text_rows, calls)
    backend = FluxBackend(transformer, text_rows=text_rows, module=module)
    backend.fake = transformer
    return backend, module


def qkv(dtype=torch.float32, seed=0, s=S, h=H, d=D):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, s, h, d, generator=g).to(dtype) for _ in range(3)]


def run(port, cuts, inputs, forwards=1, observers=(), **kw):
    """Run ``forwards`` transformer forwards; returns every attention output (one if only one)."""
    backend = port.backend
    module = backend._module
    outs = []
    with port.apply(*cuts, observers=observers, forwards=forwards, **kw):
        for _ in range(forwards):
            outs.extend(backend.fake(*inputs))
    assert module.dispatch_attention_fn is sdpa
    return outs if len(outs) > 1 else outs[0]


ROWS = {"text": list(range(T)), "image": list(range(T, S))}


# ------------------------------------------------------------------ mapping

@pytest.mark.parametrize("n_heads,n_kv", [(12, 2), (14, 2), (16, 4), (8, 8), (28, 4), (6, 1)])
def test_kv_group_matches_hf_repeat_kv(n_heads, n_kv):
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    key = torch.randn(1, n_kv, 3, 2)
    repeated = repeat_kv(key, n_heads // n_kv)
    assert kv_groups(n_heads, n_kv) == tuple(h // (n_heads // n_kv) for h in range(n_heads))
    for h in range(n_heads):
        assert torch.equal(repeated[:, h], key[:, kv_group(h, n_heads, n_kv)])


def test_kv_group_rejects_bad_geometry():
    for args in ((0, 6, 4), (6, 6, 2), (-1, 6, 2), (0, 6, 0), (True, 6, 2), (np.bool_(True), 6, 2), (1.0, 6, 2)):
        with pytest.raises(ValueError):
            kv_group(*args)


def test_kv_group_accepts_numpy_integers():
    assert kv_group(np.int64(5), np.int32(6), np.int64(2)) == 1
    assert type(kv_group(np.int64(5), 6, 2)) is int
    assert kv_groups(np.int64(6), np.int64(3)) == (0, 0, 1, 1, 2, 2)


# ----------------------------------------------------------- hand computed

def hand(q, k, v, head, receiver, removed, renormalize):
    qh, kh, vh = (x[0, :, head].double().numpy() for x in (q, k, v))
    s = qh[receiver] @ kh.T / math.sqrt(q.shape[-1])
    p = np.exp(s - s.max())
    p /= p.sum()
    keep = np.ones(k.shape[1], bool)
    keep[removed] = False
    if renormalize:
        return (p * keep / (p * keep).sum()) @ vh, p
    return (p * keep) @ vh, p


@pytest.mark.parametrize("zero_method", ["delta", "recompute"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_cut_equals_hand_result_and_dense_reference(renormalize, zero_method):
    q, k, v = qkv()
    backend, _ = flux_stub()
    port = AttentionPort(backend, rows=ROWS, zero_method=zero_method)
    c = cut([4], [0, 2], heads=[1], renormalize=renormalize)
    out = run(port, [c], (q, k, v))
    expected, p = hand(q, k, v, 1, 4, [0, 2], renormalize)
    np.testing.assert_allclose(out[0, 4, 1].double().numpy(), expected, rtol=1e-5, atol=1e-6)
    # Independent reference: the renormalised cut is dense attention over the kept sources;
    # the zero cut is that times the kept mass.
    keep = [1, 3, 4]
    qh, kh, vh = (x.permute(0, 2, 1, 3).double().numpy() for x in (q, k, v))
    dense = dense_attention(qh[:, 1:2, 4:5], kh[:, 1:2, keep], vh[:, 1:2, keep])[0, 0, 0]
    reference = dense if renormalize else dense * p[keep].sum()
    np.testing.assert_allclose(out[0, 4, 1].double().numpy(), reference, rtol=1e-5, atol=1e-6)
    # Untouched rows of the cut head equal the dense native result.
    full = dense_attention(qh, kh, vh)
    np.testing.assert_allclose(out[0, :4, 1].double().numpy(), full[0, 1, :4], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("zero_method", ["delta", "recompute"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_flux_cut_with_several_receivers_and_heads_matches_float64_oracle(renormalize, zero_method):
    s, h = 7, 2
    q, k, v = qkv(seed=3, s=s, h=h)
    backend, _ = flux_stub(heads=h)
    port = AttentionPort(backend, zero_method=zero_method)
    receivers, sources = [1, 3, 4, 6], [0, 2, 5]
    out = run(port, [cut(receivers, [sources], heads=None, renormalize=renormalize)], (q, k, v))
    for head in range(h):
        for r in range(s):
            expected, _ = hand(q, k, v, head, r, sources if r in receivers else [], renormalize)
            np.testing.assert_allclose(out[0, r, head].double().numpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("zero_method", ["delta", "recompute"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_per_head_cut_leaves_other_heads_and_rows_bit_identical(renormalize, zero_method):
    q, k, v = qkv(torch.bfloat16)
    native = sdpa(q, k, v)
    backend, _ = flux_stub()
    port = AttentionPort(backend, rows=ROWS, zero_method=zero_method)
    out = run(port, [cut(["image"], ["text"], heads=[2], renormalize=renormalize)], (q, k, v))
    assert torch.equal(out[:, :, :2], native[:, :, :2])        # heads 0 and 1 untouched
    assert torch.equal(out[:, :T, 2], native[:, :T, 2])        # text receivers of head 2 untouched
    assert not torch.equal(out[:, T:, 2], native[:, T:, 2])    # image receivers of head 2 changed


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_delta_sham_is_identity_by_construction(dtype):
    """A delta sham removes nothing, so it returns native exactly; it is NOT a floor (see floor_arms)."""
    q, k, v = qkv(dtype)
    native = sdpa(q, k, v)
    backend, _ = flux_stub(blocks=(1, 1))
    port = AttentionPort(backend, rows=ROWS)
    c = cut(["image", [0]], [[1]], heads=[(0, 1), (1, 2)], renormalize=False)
    for s in (sham(c), sham(renormalize=False)):
        outs = run(port, [s], (q, k, v), forwards=2)
        assert all(torch.equal(o, native) for o in outs)
    assert sham(c).sham and not c.sham and sham(c).heads == c.heads
    assert paired(c) == (c, sham(c))


@pytest.mark.parametrize("renormalize,zero_method", [(True, "delta"), (False, "recompute")])
def test_fp32_recompute_sham_has_only_a_rounding_floor(renormalize, zero_method):
    q, k, v = qkv(torch.bfloat16)
    native = sdpa(q, k, v)
    backend, _ = flux_stub()
    port = AttentionPort(backend, rows=ROWS, zero_method=zero_method)
    out = run(port, [sham(renormalize=renormalize)], (q, k, v))
    torch.testing.assert_close(out, native, atol=2e-2, rtol=2e-2)  # bf16 rounding only


def test_floor_arms_give_the_recompute_sham_and_the_recompute_twin():
    z = cut(["image"], ["text"], heads=[(0, 1)], steps=[0], renormalize=False)
    arms = floor_arms(z)
    assert set(arms) == {"sham_recompute", "cut_recompute"}
    assert arms["sham_recompute"] == sham(z, zero_method="recompute")
    assert arms["sham_recompute"].sham and arms["sham_recompute"].zero_method == "recompute"
    assert arms["cut_recompute"] == cut(["image"], ["text"], heads=[(0, 1)], steps=[0], renormalize=False,
                                        zero_method="recompute")
    m = cut(["image"], ["text"], heads=[(0, 1)], renormalize=True)
    assert floor_arms(m) == {"sham": sham(m)}
    with pytest.raises(ValueError):
        floor_arms(sham(z))
    # The arms run through the port and the per-cut zero_method overrides the port's.
    q, k, v = qkv(torch.bfloat16)
    native = sdpa(q, k, v)
    backend, _ = flux_stub()
    port = AttentionPort(backend, rows=ROWS, zero_method="delta")
    delta = run(port, [cut(["image"], ["text"], heads=[1], renormalize=False)], (q, k, v))
    arms = floor_arms(cut(["image"], ["text"], heads=[1], renormalize=False))
    rec = run(port, [arms["cut_recompute"]], (q, k, v))
    rec_port = run(AttentionPort(backend, rows=ROWS, zero_method="recompute"),
                   [cut(["image"], ["text"], heads=[1], renormalize=False)], (q, k, v))
    assert torch.equal(rec, rec_port)
    assert not torch.equal(rec, delta)          # the arithmetic choice alone moves the cut
    sham_rec = run(port, [arms["sham_recompute"]], (q, k, v))
    torch.testing.assert_close(sham_rec, native, atol=2e-2, rtol=2e-2)
    assert port.apply(arms["cut_recompute"]).metadata()["cuts"][0]["zero_method"] == "recompute"


def test_bf16_delta_and_recompute_against_float64_cut():
    """Documented tolerances (ATTENTION-PORT.md): recompute is within one bf16 rounding of the exact
    cut; delta is within the native kernel's own error on that row plus one bf16 rounding."""
    s, h, d, t = 64, 2, 32, 16
    q, k, v = qkv(torch.bfloat16, seed=1, s=s, h=h, d=d)
    k[0, :4, 0] = (q[0, 40, 0].float() * 0.6).to(torch.bfloat16)   # receiver 40 of head 0: mass on 0..3
    backend, _ = flux_stub(heads=h, text_rows=t)
    qh, kh, vh = (x[0].permute(1, 0, 2).double() for x in (q, k, v))
    p = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(d), -1)
    exact_native = (p @ vh).permute(1, 0, 2)[None]
    native_err = (sdpa(q, k, v).double() - exact_native).abs()
    receivers = torch.arange(t, s)
    for removed in (torch.arange(8, 12), torch.arange(0, 4), torch.arange(0, t)):
        pc = p.clone()
        pc[:, receivers[:, None], removed[None, :]] = 0
        exact = (pc @ vh).permute(1, 0, 2)[None]
        outs = {}
        for zm in ("delta", "recompute"):
            port = AttentionPort(backend, zero_method=zm)
            outs[zm] = run(port, [cut(receivers.tolist(), [removed.tolist()], renormalize=False)], (q, k, v))
            err = (outs[zm].double() - exact)[:, t:].abs()
            if zm == "recompute":
                bound = 2.0 ** -8 * exact[:, t:].abs() + 1e-5
            else:
                bound = native_err[:, t:] + 2.0 ** -8 * exact[:, t:].abs() / (1 - 2.0 ** -8) + 1e-5
            assert bool((err <= bound).all()), (zm, float((err / bound).max()))
        assert not torch.equal(outs["delta"], outs["recompute"])


@pytest.mark.parametrize("zero_method", ["delta", "recompute"])
def test_zero_cut_removing_every_source_writes_exact_zero(zero_method):
    q, k, v = qkv(torch.bfloat16, seed=1, s=16, h=2, d=8)
    native = sdpa(q, k, v)
    backend, _ = flux_stub(heads=2)
    port = AttentionPort(backend, zero_method=zero_method)
    out = run(port, [cut([7, 9], ["all"], heads=[0], renormalize=False)], (q, k, v))
    assert torch.count_nonzero(out[0, [7, 9], 0]) == 0
    others = [r for r in range(16) if r not in (7, 9)]
    assert torch.equal(out[0, others], native[0, others]) and torch.equal(out[0, :, 1], native[0, :, 1])


def test_composite_cut_honours_each_specs_heads():
    q, k, v = qkv()
    native = sdpa(q, k, v)
    backend, _ = flux_stub()
    port = AttentionPort(backend, rows=ROWS)
    a = cut([4], [0], heads=[0], renormalize=False)
    b = cut([3], [1], heads=[1], renormalize=False)
    both = run(port, [a, b], (q, k, v))
    only_a = run(port, [a], (q, k, v))
    only_b = run(port, [b], (q, k, v))
    assert torch.equal(both[:, :, 0], only_a[:, :, 0])
    assert torch.equal(both[:, :, 1], only_b[:, :, 1])
    assert torch.equal(both[:, :, 2], native[:, :, 2])
    # The 2026-11-11 `both()` bug would have applied b's cut to head 0 as well.
    assert torch.equal(both[:, 3, 0], native[:, 3, 0])
    assert torch.equal(both[:, 4, 1], native[:, 4, 1])
    # Overlapping specs in one head remove the union of their reads.
    c = cut([4], [1], heads=[0], renormalize=False)
    union = run(port, [a, c], (q, k, v))
    expected, _ = hand(q, k, v, 0, 4, [0, 1], False)
    np.testing.assert_allclose(union[0, 4, 0].double().numpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("zero_method", ["delta", "recompute"])
def test_mixed_cut_types_in_one_head_need_disjoint_receivers(zero_method):
    q, k, v = qkv()
    backend, _ = flux_stub()
    port = AttentionPort(backend, rows=ROWS, zero_method=zero_method)
    m = cut([3], [0], heads=[0], renormalize=True)
    z = cut([4], [1], heads=[0], renormalize=False)
    mixed = run(port, [m, z], (q, k, v))
    assert torch.equal(mixed[0, 3], run(port, [m], (q, k, v))[0, 3])
    assert torch.equal(mixed[0, 4], run(port, [z], (q, k, v))[0, 4])
    for r, src, ren in ((3, [0], True), (4, [1], False)):
        expected, _ = hand(q, k, v, 0, r, src, ren)
        np.testing.assert_allclose(mixed[0, r, 0].double().numpy(), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="same \\(head, receiver\\) cell"):
        run(port, [cut([4], [0], heads=[0], renormalize=False), cut([4], [1], heads=[0], renormalize=True)],
            (q, k, v))
    other = "recompute" if zero_method == "delta" else "delta"
    with pytest.raises(ValueError, match="zero methods"):
        run(port, [z, cut([4], [2], heads=[0], renormalize=False, zero_method=other)], (q, k, v))


def test_blocks_steps_and_block_head_pairs_select_calls():
    q, k, v = qkv()
    native = sdpa(q, k, v)
    backend, _ = flux_stub(blocks=(1, 1))  # two calls per step
    port = AttentionPort(backend, rows=ROWS)
    outs = run(port, [cut(["image"], ["text"], heads=[(1, 0)], steps=[1], renormalize=False)], (q, k, v), forwards=2)
    changed = [not torch.equal(o, native) for o in outs]
    assert changed == [False, False, False, True]  # (step 1, block 1) only
    outs = run(port, [cut(["image"], ["text"], blocks=[0], renormalize=True)], (q, k, v), forwards=2)
    assert [not torch.equal(o, native) for o in outs] == [True, False, True, False]


def test_receiver_left_without_source_is_refused():
    q, k, v = qkv()
    backend, _ = flux_stub()
    port = AttentionPort(backend, rows=ROWS)
    with pytest.raises(ValueError, match="no visible source"):
        run(port, [cut([4], ["text", "image"], renormalize=True)], (q, k, v))
    out = run(port, [cut([4], ["text", "image"], renormalize=False)], (q, k, v))
    assert torch.count_nonzero(out[0, 4]) == 0
    zero_port = AttentionPort(backend, rows=ROWS, empty="zero")
    out = run(zero_port, [cut([4], ["text", "image"], heads=[1], renormalize=True)], (q, k, v))
    assert torch.count_nonzero(out[0, 4, 1]) == 0
    assert zero_port.emptied == [dict(step=0, block=0, head=1, receiver_query_row=4)]
    assert zero_port.apply().metadata()["empty"] == "zero"


def test_legacy_precision_reproduces_the_read_graph_arithmetic():
    q, k, v = qkv(torch.bfloat16)
    backend, _ = flux_stub()
    heads = [[0, 1]]
    receivers, sources = [T, T + 1, T + 2, 0], [1]
    m = torch.ones(S, S, dtype=torch.bool)
    m[torch.tensor(receivers)[:, None], torch.tensor(sources)[None, :]] = False
    mh = torch.ones(1, H, S, S, dtype=torch.bool)
    mh[0, [1]] = m
    qh, kh, vh = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
    w = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(D), dim=-1) * mh
    old_zero = (w @ vh).permute(0, 2, 1, 3).to(q.dtype)
    old_mask = sdpa(q, k, v, attn_mask=mh)
    port = AttentionPort(backend, rows=ROWS, precision="legacy")
    spec = (["image", [0]], [sources])
    assert torch.equal(run(port, [cut(*spec, heads=heads, renormalize=False)], (q, k, v)), old_zero)
    assert torch.equal(run(port, [cut(*spec, heads=heads, renormalize=True)], (q, k, v)), old_mask)
    # Old-style shams: every head of every block in FP32, and an all-true boolean mask.
    old_sham = (torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(D), dim=-1) @ vh).permute(0, 2, 1, 3).to(q.dtype)
    assert torch.equal(run(port, [sham(renormalize=False)], (q, k, v)), old_sham)
    assert torch.equal(run(port, [sham(renormalize=True)], (q, k, v)),
                       sdpa(q, k, v, attn_mask=torch.ones(1, 1, S, S, dtype=torch.bool)))


def test_legacy_precision_still_runs_observers_on_cut_calls():
    q, k, v = qkv(torch.bfloat16)
    backend, _ = flux_stub(blocks=(1, 1))
    c = cut([4], [0], heads=[(0, 1)], renormalize=False)
    records = {}
    for precision in ("native", "legacy"):
        port = AttentionPort(backend, rows=ROWS, precision=precision)
        probs = observe(full_probs("float32"))
        run(port, [c], (q, k, v), observers=[probs])
        assert sorted(probs.records) == [(0, 0), (0, 1)]
        records[precision] = probs.records
    for key in records["native"]:
        np.testing.assert_array_equal(records["legacy"][key], records["native"][key])


# ------------------------------------------------------- FLUX call placement

def test_flux_requires_forwards_and_checks_the_count():
    q, k, v = qkv()
    backend, _ = flux_stub(blocks=(1, 1))
    port = AttentionPort(backend, rows=ROWS)
    c = cut([4], [0], heads=[0], renormalize=False)
    with pytest.raises(ValueError, match="forwards=N"):
        with port.apply(c):
            pass
    assert backend._module.dispatch_attention_fn is sdpa
    # Classifier-free guidance as separate forwards: 2 steps x 2 passes against forwards=2.
    with pytest.raises(ValueError, match="exceeds forwards=2"):
        with port.apply(c, forwards=2):
            for _ in range(4):
                backend.fake(q, k, v)
    assert backend._module.dispatch_attention_fn is sdpa
    # A second image in the same apply is the same refusal (several images per apply: unsupported).
    with pytest.raises(ValueError, match="exceeds forwards=1"):
        with port.apply(c, forwards=1):
            backend.fake(q, k, v)
            backend.fake(q, k, v)
    with pytest.raises(ValueError, match="expected 3 model forwards, observed 2"):
        with port.apply(c, forwards=3):
            backend.fake(q, k, v)
            backend.fake(q, k, v)
    # Counters reset per apply: the same port gives the same placement twice.
    native = sdpa(q, k, v)
    for _ in range(2):
        outs = run(port, [cut([4], [0], heads=[0], steps=[1], blocks=[1], renormalize=False)], (q, k, v), forwards=2)
        assert [not torch.equal(o, native) for o in outs] == [False, False, False, True]
        assert port.calls == 4 and port.forward_count == 2


def test_flux_refuses_partial_out_of_order_or_multi_call_blocks():
    q, k, v = qkv()
    backend, module = flux_stub(blocks=(1, 1))
    port = AttentionPort(backend, rows=ROWS)
    c = cut([4], [0], heads=[0], renormalize=False)
    with pytest.raises(ValueError, match="stopped before block 1"):
        with port.apply(c, forwards=1):
            backend.fake.transformer_blocks[0](encoder_hidden_states=torch.zeros(1, T, 1), qkv=(q, k, v))
    with pytest.raises(ValueError, match="block 1 ran where block 0 was expected"):
        with port.apply(c, forwards=1):
            backend.fake.single_transformer_blocks[0](qkv=(q, k, v))
    with pytest.raises(ValueError, match="outside a transformer block"):
        with port.apply(c, forwards=1):
            module.dispatch_attention_fn(q, k, v)
    assert module.dispatch_attention_fn is sdpa
    two, _ = flux_stub(blocks=(1, 1), calls=2)   # e.g. reference-image extract mode: 2 calls per block
    with pytest.raises(ValueError, match="second attention call"):
        run(AttentionPort(two, rows=ROWS), [c], (q, k, v))


def test_flux_refuses_sq_ne_sk_even_on_calls_without_a_plan():
    t, r, n = 3, 2, 3                                     # keys [txt, ref, img], queries [txt, img]
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, t + n, H, D, generator=g)
    k, v = (torch.randn(1, t + r + n, H, D, generator=g) for _ in range(2))
    backend, _ = flux_stub(blocks=(1, 1), text_rows=t)
    port = AttentionPort(backend)
    with pytest.raises(ValueError, match="query rows against"):
        run(port, [cut([0], [t + r], heads=[0], blocks=[1], renormalize=False)], (q, k, v))  # block 0 has no plan
    with pytest.raises(ValueError, match="query rows against"):
        run(port, [], (q, k, v))                                                            # passthrough too


def test_flux_text_rows_is_required_and_checked():
    module = SimpleNamespace(dispatch_attention_fn=sdpa)
    transformer = FakeTransformer(module, (1, 1))
    with pytest.raises(TypeError):
        FluxBackend(transformer)
    backend = FluxBackend(transformer, text_rows=T + 1, module=module)
    backend.fake = transformer
    q, k, v = qkv()
    with pytest.raises(ValueError, match="text_rows=3 but block 0 received 2 text rows"):
        run(AttentionPort(backend), [cut([4], [0], heads=[0], renormalize=False)], (q, k, v))


@pytest.mark.parametrize("backend_kind", ["flux", "qwen"])
def test_batch_larger_than_one_is_refused(backend_kind):
    c = cut([4], [0], heads=[0], renormalize=False)
    if backend_kind == "flux":
        q, k, v = (torch.cat([x, x]) for x in qkv())
        backend, _ = flux_stub()
        with pytest.raises(ValueError, match="batch size 2"):
            run(AttentionPort(backend), [c], (q, k, v))
    else:
        model = tiny_qwen()
        with torch.no_grad(), pytest.raises(ValueError, match="batch size 2"):
            with AttentionPort(QwenBackend(model)).apply(c):
                model(torch.tensor([[3, 5, 7, 11, 13]] * 2))


def test_cuts_or_observers_that_never_run_are_refused():
    q, k, v = qkv()
    backend, _ = flux_stub(blocks=(1, 1))
    port = AttentionPort(backend, rows=ROWS)
    with pytest.raises(ValueError, match="no attention call was intercepted"):
        with port.apply(cut([4], [0], renormalize=False), forwards=1):
            pass
    with pytest.raises(ValueError, match="never matched"):
        run(port, [cut([4], [0], steps=[5], renormalize=False)], (q, k, v))
    with pytest.raises(ValueError, match="never matched"):
        run(port, [], (q, k, v), observers=[observe(full_probs(), blocks=[9])])


# ------------------------------------------------------------------ observe

def test_observe_records_reductions_without_changing_output():
    q, k, v = qkv(torch.bfloat16)
    native = sdpa(q, k, v)
    backend, _ = flux_stub(blocks=(1, 1))
    port = AttentionPort(backend, rows=ROWS)
    classes = np.zeros((S, 2), np.float32)
    classes[:T, 0] = classes[T:, 1] = 1
    probs, flow = observe(full_probs("float32")), observe(class_flow(classes))
    outs = run(port, [], (q, k, v), forwards=2, observers=[probs, flow])
    assert all(torch.equal(o, native) for o in outs)
    assert sorted(probs.records) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    qh, kh = (x.permute(0, 2, 1, 3).float() for x in (q, k))
    p = torch.softmax(qh @ kh.transpose(-1, -2) * D ** -0.5, dim=-1)[0].numpy()
    np.testing.assert_allclose(probs.records[(1, 0)], p, rtol=1e-6, atol=1e-7)
    expected = np.einsum("rs,hsc->hrc", (classes / classes.sum(0)).T, p @ classes)
    np.testing.assert_allclose(flow.records[(0, 1)], expected, rtol=1e-5, atol=1e-6)
    assert flow.stack().shape == (2, 2, H, 2, 2)
    np.testing.assert_allclose(flow.stack().sum(-1), 1.0, rtol=1e-5)
    meta = port.apply(cut(["image"], ["text"], heads=[(0, 1)], renormalize=False), observers=[flow]).metadata()
    assert meta["cuts"][0]["heads"] == [[0, 1]] and meta["observers"][0]["name"] == "class_flow_p"


# --------------------------------------------------------------------- qwen

def tiny_qwen(dtype=torch.float32, impl="eager"):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
    config._attn_implementation = impl
    return Qwen2ForCausalLM(config).eval().to(dtype)


def capture_layer0(model, ids, attention_mask=None):
    """Native layer-0 q/k/v through the plain observer (independent of the port's arithmetic)."""
    from mdb_lanes.attention_observer import AttentionObserver
    with torch.no_grad(), AttentionObserver(model, module=_modeling(model)) as obs:
        model(ids, attention_mask=attention_mask)
    row = obs.rows[0]
    return row["query"], row["key"], row["value"]


def _modeling(model):
    import sys
    return sys.modules[type(model.model.layers[0].self_attn).__module__]


def head_outputs(model, ids, port, *cuts, attention_mask=None):
    """Layer-0 attention output per head ([S, H, D]) as fed to o_proj, with the port's cuts applied."""
    got = {}
    hook = model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(
        lambda m, args: got.setdefault("x", args[0].detach().clone()))
    try:
        with torch.no_grad(), port.apply(*cuts):
            model(ids, attention_mask=attention_mask)
    finally:
        hook.remove()
    n_heads = model.config.num_attention_heads
    return got["x"][0].view(ids.shape[1], n_heads, -1)


def oracle(query, key, value, head, n_kv_groups, visible, removed, renormalize):
    """float64 attention of one head with the native visibility and a cut, for every receiver."""
    qh = query[0, head].double()
    kh, vh = key[0, head // n_kv_groups].double(), value[0, head // n_kv_groups].double()
    s = qh @ kh.T / math.sqrt(qh.shape[-1])
    s = s.masked_fill(~visible, float("-inf"))
    p = torch.softmax(s, -1)
    keep = visible & ~removed
    if renormalize:
        w = torch.softmax(s.masked_fill(~keep, float("-inf")), -1)
        w = torch.nan_to_num(w, nan=0.0)
    else:
        w = p * keep
    return w @ vh


def test_qwen_backend_observe_sham_and_cut():
    model = tiny_qwen()
    ids = torch.tensor([[3, 5, 7, 11, 13]])
    with torch.no_grad():
        native = model(ids, output_attentions=True)
        port = AttentionPort(QwenBackend(model))
        probs = observe(full_probs("float32"))
        with port.apply(observers=[probs], forwards=1):
            observed = model(ids, output_attentions=True)
        assert torch.equal(observed.logits, native.logits)
        assert sorted(probs.records) == [(0, 0), (0, 1)] and port.calls == 2
        np.testing.assert_allclose(probs.records[(0, 1)], native.attentions[1][0].numpy(), rtol=1e-5, atol=1e-6)
        with port.apply(sham(renormalize=False)):
            shammed = model(ids)
        assert torch.equal(shammed.logits, native.logits)
        c = cut([4], [1], heads=[(0, 3)], renormalize=False)
        with port.apply(c):
            edited = model(ids, output_attentions=True)
        a, n = edited.attentions[0][0], native.attentions[0][0]
        assert torch.equal(a[:3], n[:3]) and torch.equal(a[3, :4], n[3, :4])
        assert a[3, 4, 1] == 0
        torch.testing.assert_close(a[3, 4, [0, 2, 3, 4]], n[3, 4, [0, 2, 3, 4]])
        assert torch.equal(edited.logits[:, :4], native.logits[:, :4])  # causal: earlier rows untouched
        assert not torch.equal(edited.logits[:, 4], native.logits[:, 4])
        with port.apply(cut(["all"], [1], heads=[(0, 3)], renormalize=True)):
            renorm = model(ids, output_attentions=True)
        w = renorm.attentions[0][0][3]
        assert torch.equal(w.triu(1), torch.zeros_like(w))        # nothing lands on masked (future) sources
        assert bool((w[1:, 1] == 0).all())
        torch.testing.assert_close(w.sum(-1), torch.ones(5))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen_renormalising_cut_never_reads_masked_future_sources(dtype):
    """Audit finding 1: the HF eager causal mask is finite (finfo.min), not -inf."""
    model = tiny_qwen(dtype)
    ids = torch.tensor([[3, 5, 7, 11, 13]])
    assert invisible_below(dtype) >= -1e30
    port = AttentionPort(QwenBackend(model))
    # Receiver 2 loses every source it can see (0, 1, 2): refused by default.
    with torch.no_grad(), pytest.raises(ValueError, match="no visible source"):
        with port.apply(cut([2], [0, 1, 2], heads=[(0, 0)], renormalize=True)):
            model(ids)
    # The BOS-sink cut: every receiver loses source 0, so receiver 0 has nothing left.
    with torch.no_grad(), pytest.raises(ValueError, match="no visible source"):
        with port.apply(cut(["all"], [0], heads=[(0, 0)], renormalize=True)):
            model(ids)
    zero_port = AttentionPort(QwenBackend(model), empty="zero")
    with torch.no_grad(), zero_port.apply(cut(["all"], [0], heads=[(0, 0)], renormalize=True)):
        out = model(ids, output_attentions=True)
    w = out.attentions[0][0, 0].float()
    assert torch.equal(w.triu(1), torch.zeros_like(w))
    assert torch.count_nonzero(w[0]) == 0 and bool((w[:, 0] == 0).all())
    torch.testing.assert_close(w[1:].sum(-1), torch.ones(4), atol=1e-2, rtol=0)
    assert zero_port.emptied == [dict(step=0, block=0, head=0, receiver_query_row=0)]
    x = head_outputs(model, ids, zero_port, cut(["all"], [0], heads=[(0, 0)], renormalize=True))
    assert torch.count_nonzero(x[0, 0]) == 0
    # A zero cut removing every visible source writes exact zero (no future reads either).
    for zm in ("delta", "recompute"):
        x = head_outputs(model, ids, AttentionPort(QwenBackend(model), zero_method=zm),
                         cut([2], [0, 1, 2], heads=[(0, 0)], renormalize=False))
        assert torch.count_nonzero(x[2, 0]) == 0


@pytest.mark.parametrize("zero_method", ["delta", "recompute"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_qwen_cut_values_match_float64_oracle(renormalize, zero_method):
    model = tiny_qwen()
    ids = torch.tensor([[3, 5, 7, 11, 13]])
    query, key, value = capture_layer0(model, ids)
    groups = model.config.num_attention_heads // model.config.num_key_value_heads
    port = AttentionPort(QwenBackend(model), zero_method=zero_method)
    receivers, sources, head = [2, 3, 4], [1, 2], 3
    x = head_outputs(model, ids, port, cut(receivers, [sources], heads=[(0, head)], renormalize=renormalize))
    visible = torch.ones(5, 5, dtype=torch.bool).tril()
    removed = torch.zeros(5, 5, dtype=torch.bool)
    removed[torch.tensor(receivers)[:, None], torch.tensor(sources)[None, :]] = True
    expected = oracle(query, key, value, head, groups, visible, removed, renormalize)
    np.testing.assert_allclose(x[:, head].double().numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)
    native = oracle(query, key, value, 0, groups, visible, torch.zeros_like(removed), False)
    np.testing.assert_allclose(x[:, 0].double().numpy(), native.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("renormalize", [False, True])
def test_qwen_cut_with_padding_attention_mask_matches_oracle(renormalize):
    """A cut combined with an existing (left padding) attention mask."""
    model = tiny_qwen()
    ids, mask = torch.tensor([[0, 5, 7, 11, 13]]), torch.tensor([[0, 1, 1, 1, 1]])
    query, key, value = capture_layer0(model, ids, mask)
    groups = model.config.num_attention_heads // model.config.num_key_value_heads
    visible = torch.ones(5, 5, dtype=torch.bool).tril()
    visible[:, 0] = False                                   # padding source never visible
    removed = torch.zeros(5, 5, dtype=torch.bool)
    removed[[3, 4], 1] = True
    port = AttentionPort(QwenBackend(model))
    with torch.no_grad():
        native = model(ids, attention_mask=mask, output_attentions=True)
    x = head_outputs(model, ids, port, cut([3, 4], [1], heads=[(0, 2)], renormalize=renormalize),
                     attention_mask=mask)
    expected = oracle(query, key, value, 2, groups, visible, removed, renormalize)
    np.testing.assert_allclose(x[1:, 2].double().numpy(), expected[1:].numpy(), rtol=1e-5, atol=1e-6)
    with torch.no_grad(), port.apply(cut(["all"], [1], heads=[(0, 2)], renormalize=renormalize)):
        if renormalize:
            with pytest.raises(ValueError, match="no visible source"):   # row 1 sees only source 1
                model(ids, attention_mask=mask)
        else:
            out = model(ids, attention_mask=mask, output_attentions=True)
    if not renormalize:
        w, n = out.attentions[0][0, 2], native.attentions[0][0, 2]
        assert torch.equal(w[0], n[0])           # padding receiver sees nothing natively: left native
        assert bool((w[1:, 0] == 0).all()) and torch.equal(w.triu(1)[1:], torch.zeros_like(w[1:]))
    zero_port = AttentionPort(QwenBackend(model), empty="zero")
    with torch.no_grad(), zero_port.apply(cut(["all"], [1], heads=[(0, 2)], renormalize=True)):
        out = model(ids, attention_mask=mask, output_attentions=True)
    w = out.attentions[0][0, 2]
    assert torch.count_nonzero(w[1]) == 0 and bool((w[1:, :2] == 0).all())
    assert torch.equal(w[0], native.attentions[0][0, 2, 0])
    assert zero_port.emptied == [dict(step=0, block=0, head=2, receiver_query_row=1)]


def test_qwen_refuses_non_eager_attention_and_unintercepted_runs():
    model = tiny_qwen(impl="sdpa")
    ids = torch.tensor([[3, 5, 7, 11, 13]])
    port = AttentionPort(QwenBackend(model))
    with pytest.raises(ValueError, match="needs eager attention"):
        with port.apply(cut([4], [0, 1, 2, 3], renormalize=False)):
            model(ids)
    assert not getattr(_modeling(model).eager_attention_forward, "_mdb_attention_observer", False)
    # Eager model, but the patch lands on a module the model never calls: zero calls -> refused at exit.
    eager = tiny_qwen()
    stray = SimpleNamespace(eager_attention_forward=_modeling(eager).eager_attention_forward)
    port = AttentionPort(QwenBackend(eager, module=stray))
    with torch.no_grad(), pytest.raises(ValueError, match="no attention call was intercepted"):
        with port.apply(cut([4], [0], renormalize=False)):
            eager(ids)


def test_qwen_kv_cached_decode_and_forward_count():
    model = tiny_qwen()
    ids, nxt = torch.tensor([[3, 5, 7, 11, 13]]), torch.tensor([[17]])
    port = AttentionPort(QwenBackend(model))
    c = cut([5], [0], heads=[(0, 0)], steps=[1], renormalize=False)
    with torch.no_grad():
        nat = model(ids, use_cache=True)
        nat_dec = model(nxt, past_key_values=nat.past_key_values, use_cache=True).logits
        with port.apply(c, forwards=2):
            pre = model(ids, use_cache=True)
            dec = model(nxt, past_key_values=pre.past_key_values, use_cache=True).logits
        ref = None
        with port.apply(cut([5], [0], heads=[(0, 0)], renormalize=False)):
            ref = model(torch.cat([ids, nxt], 1)).logits[:, -1]
    assert torch.equal(pre.logits, nat.logits) and not torch.equal(dec, nat_dec)
    torch.testing.assert_close(dec[:, -1], ref, rtol=1e-5, atol=1e-6)
    with torch.no_grad(), pytest.raises(ValueError, match="expected 3 model forwards, observed 1"):
        with port.apply(cut([4], [0], renormalize=False), forwards=3):
            model(ids)


def test_qwen3_backend_patches_its_own_modeling_module():
    """Qwen3 defines its own eager_attention_forward (transformers 5.x); the backend must patch it."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import transformers.models.qwen2.modeling_qwen2 as mq2
    import transformers.models.qwen3.modeling_qwen3 as mq3
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=32)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    assert _modeling(model) is mq3 and mq3.eager_attention_forward is not mq2.eager_attention_forward
    ids = torch.tensor([[3, 5, 7, 11, 13]])
    query, key, value = capture_layer0(model, ids)
    port = AttentionPort(QwenBackend(model))
    with torch.no_grad():
        native = model(ids).logits
        with port.apply(sham(renormalize=False)):
            assert torch.equal(model(ids).logits, native) and port.calls == 2
    for renormalize in (False, True):
        x = head_outputs(model, ids, port, cut([3, 4], [0, 2], heads=[(0, 1)], renormalize=renormalize))
        removed = torch.zeros(5, 5, dtype=torch.bool)
        removed[3:, [0, 2]] = True
        expected = oracle(query, key, value, 1, 2, torch.ones(5, 5, dtype=torch.bool).tril(), removed, renormalize)
        np.testing.assert_allclose(x[:, 1].double().numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)


def test_qwen_backend_shares_the_observer_nesting_guard():
    from mdb_lanes.attention_observer import AttentionObserver
    model = tiny_qwen()
    port = AttentionPort(QwenBackend(model))
    with port.apply():
        with pytest.raises(ValueError, match="nested"):
            with AttentionObserver(model):
                pass
    with AttentionObserver(model):
        with pytest.raises(ValueError, match="nested"):
            with port.apply():
                pass
