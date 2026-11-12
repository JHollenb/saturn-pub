"""Attention port mechanics on tiny CPU tensors (no pretrained model)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mdb.attention_partition import dense_attention
from mdb.attention_port import (AttentionPort, FluxBackend, QwenBackend, class_flow, cut, full_probs,
                                kv_group, kv_groups, observe, paired, sham)

H, S, D, T = 3, 5, 4, 2  # heads, rows, head width, "text" rows (rows T: are "image")


def sdpa(q, k, v, attn_mask=None, **kwargs):
    """Stand-in native kernel with the diffusers [B, S, H, D] layout."""
    out = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), attn_mask=attn_mask)
    return out.permute(0, 2, 1, 3)


def flux_stub(blocks=(1, 1)):
    attn = SimpleNamespace(heads=H)
    transformer = SimpleNamespace(transformer_blocks=[SimpleNamespace(attn=attn)] * blocks[0],
                                  single_transformer_blocks=[SimpleNamespace(attn=attn)] * blocks[1])
    module = SimpleNamespace(dispatch_attention_fn=sdpa)
    return FluxBackend(transformer, text_rows=T, module=module), module


def qkv(dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, S, H, D, generator=g).to(dtype) for _ in range(3)]


def run(port, cuts, inputs, calls=1, observers=()):
    backend = port.backend
    module = backend._module
    outs = []
    with port.apply(*cuts, observers=observers):
        for _ in range(calls):
            outs.append(module.dispatch_attention_fn(*inputs))
    assert module.dispatch_attention_fn is sdpa
    return outs if calls > 1 else outs[0]


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
    for args in ((0, 6, 4), (6, 6, 2), (-1, 6, 2), (0, 6, 0), (True, 6, 2)):
        with pytest.raises(ValueError):
            kv_group(*args)


# ----------------------------------------------------------- hand computed

def hand(q, k, v, head, receiver, removed, renormalize):
    qh, kh, vh = (x[0, :, head].double().numpy() for x in (q, k, v))
    s = qh[receiver] @ kh.T / math.sqrt(D)
    p = np.exp(s - s.max())
    p /= p.sum()
    keep = np.ones(S, bool)
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
def test_sham_is_identity_for_delta_zero_cuts(dtype):
    q, k, v = qkv(dtype)
    native = sdpa(q, k, v)
    backend, _ = flux_stub()
    port = AttentionPort(backend, rows=ROWS)
    c = cut(["image", [0]], [[1]], heads=[(0, 1), (1, 2)], renormalize=False)
    for s in (sham(c), sham(renormalize=False)):
        outs = run(port, [s], (q, k, v), calls=2)
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
    with pytest.raises(ValueError, match="both types"):
        run(port, [a, cut([4], [1], heads=[0], renormalize=True)], (q, k, v))


def test_blocks_steps_and_block_head_pairs_select_calls():
    q, k, v = qkv()
    native = sdpa(q, k, v)
    backend, _ = flux_stub(blocks=(1, 1))  # two calls per step
    port = AttentionPort(backend, rows=ROWS)
    outs = run(port, [cut(["image"], ["text"], heads=[(1, 0)], steps=[1], renormalize=False)], (q, k, v), calls=4)
    changed = [not torch.equal(o, native) for o in outs]
    assert changed == [False, False, False, True]  # (step 1, block 1) only
    outs = run(port, [cut(["image"], ["text"], blocks=[0], renormalize=True)], (q, k, v), calls=4)
    assert [not torch.equal(o, native) for o in outs] == [True, False, True, False]


def test_receiver_left_without_source_is_refused():
    q, k, v = qkv()
    backend, _ = flux_stub()
    port = AttentionPort(backend, rows=ROWS)
    with pytest.raises(ValueError, match="no source"):
        run(port, [cut([4], ["text", "image"], renormalize=True)], (q, k, v))
    out = run(port, [cut([4], ["text", "image"], renormalize=False)], (q, k, v))
    assert torch.count_nonzero(out[0, 4].abs() > 1e-6) == 0


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


# ------------------------------------------------------------------ observe

def test_observe_records_reductions_without_changing_output():
    q, k, v = qkv(torch.bfloat16)
    native = sdpa(q, k, v)
    backend, _ = flux_stub(blocks=(1, 1))
    port = AttentionPort(backend, rows=ROWS)
    classes = np.zeros((S, 2), np.float32)
    classes[:T, 0] = classes[T:, 1] = 1
    probs, flow = observe(full_probs("float32")), observe(class_flow(classes))
    outs = run(port, [], (q, k, v), calls=4, observers=[probs, flow])
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

def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval()


def test_qwen_backend_observe_sham_and_cut():
    model = tiny_qwen()
    ids = torch.tensor([[3, 5, 7, 11, 13]])
    with torch.no_grad():
        native = model(ids, output_attentions=True)
        port = AttentionPort(QwenBackend(model))
        probs = observe(full_probs("float32"))
        with port.apply(observers=[probs]):
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
        with port.apply(cut([4], [1], heads=[(0, 3)], renormalize=True)):
            renorm = model(ids, output_attentions=True)
        row = renorm.attentions[0][0][3, 4]
        assert row[1] == 0 and abs(float(row.sum()) - 1) < 1e-5


def test_qwen_backend_shares_the_observer_nesting_guard():
    from mdb.attention_observer import AttentionObserver
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
