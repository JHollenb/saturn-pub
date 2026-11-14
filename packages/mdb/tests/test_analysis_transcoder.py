"""Transcoders as a member of the decomposition family (gap audit P3) on tiny Qwen3 (fp32) and
Qwen2 (bf16), with a random mwhanna-layout transcoder written to safetensors.

The set loads ``layer_<L>.safetensors``; encode equals exp3's hand ``encode`` (float32 selection,
float64 activations from the native bits); the MLP-write binding partitions the native ``m``
exactly (closure at float64 roundoff, the error the remainder); ``component_table`` shares add up
to the write's direct share; removing a feature with ``edited`` and ``Run.set`` at ``m`` moves the
margin natively; ``FeatureContext`` reads transcoder features at ``L<l>.n``; SAELens's own
``Transcoder`` built from the basis encodes and decodes the same."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")

from mdb.analysis import components, features, sae  # noqa: E402
from mdb.analysis.finite import f64  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

P = [1, 5, 9, 3, 7]
LAYER = 1


@pytest.fixture(scope="module", params=[("qwen3", "fp32"), ("qwen2", "bf16")],
                ids=lambda c: "-".join(c))
def setup(request, tmp_path_factory):
    dbg = open_tiny(*request.param)
    d, d_sae = dbg.adapter.hidden_size, 4 * dbg.adapter.hidden_size
    g = torch.Generator().manual_seed(7)
    state = {"W_enc": torch.randn(d_sae, d, generator=g) / d ** 0.5,
             "b_enc": torch.randn(d_sae, generator=g) * 0.1,
             "W_dec": torch.randn(d_sae, d, generator=g) / d_sae ** 0.5,
             "b_dec": torch.randn(d, generator=g) * 0.01}
    path = tmp_path_factory.mktemp("tc")
    safetensors.save_file({k: v.to(torch.bfloat16) for k, v in state.items()},
                          str(path / f"layer_{LAYER}.safetensors"))
    tset = sae.open_transcoders("tiny-tc", str(path))
    run = dbg.run(P, name="tc")
    run.until("t4.logits")
    return dbg, tset, run


def _hand_encode(tc, n):
    """exp3_hop3.py encode (on CPU, the GPU bf16 matmul replaced by the float32 one)."""
    pre = (tc.W_enc.float() @ n.float()) + tc.b_enc.float()
    ids = (pre > 0).nonzero().reshape(-1)
    rows = tc.W_enc.index_select(0, ids).double()
    acts = torch.relu(rows @ n.double() + tc.b_enc.index_select(0, ids).double())
    keep = acts > 0
    return sorted(ids[keep].tolist()), acts[keep]


def test_set_and_encode(setup):
    dbg, tset, run = setup
    assert tset.layers == (LAYER,) and tset.for_site(LAYER, "n") is not None
    tc = tset.get(LAYER)
    assert tc.covers(LAYER, "n") and not tc.covers(LAYER, "h")
    n = run.read(f"t4.L{LAYER}.n").reshape(-1)
    enc = tc.encode(n)
    ids, acts = _hand_encode(tc, n)
    assert sorted(enc["ids"]) == ids
    order = sorted(range(len(enc["ids"])), key=lambda i: enc["ids"][i])
    assert torch.equal(enc["acts"][order], acts)
    desc = tc.describe()
    assert desc["saelens"]["hook_name"] == f"blocks.{LAYER}.mlp.hook_in"
    assert tc.node(4, ids[0])["id"] == f"{LAYER}_{ids[0]}_4"


def test_binding_partitions_the_write(setup):
    dbg, tset, run = setup
    tc = tset.get(LAYER)
    cb = components.transcoder_binding(run, "mlp", f"t4.L{LAYER}.m", tc)
    assert cb.cell == f"t4.L{LAYER}.m" and cb.site == "mlp-write"
    assert torch.equal(cb.native, f64(run.read(f"t4.L{LAYER}.m")).reshape(-1).cpu())
    close = cb.closure()
    assert close["within_f64"] and close["rounding_norm"]["value"] == 0.0
    table = components.component_table(run, cb, Margin(4, 3, 7))
    assert table["partition"]["holds"]
    assert table["mass"]["state"] == "not applicable"
    names = [r["component"] for r in table["components"]]
    assert names[-3:] == ["other_features", "bdec", "error"]
    with pytest.raises(Refusal):
        components.transcoder_binding(run, "x", "t4.L0.m", tc)          # wrong layer
    with pytest.raises(Refusal):
        components.transcoder_binding(run, "x", "cache.L1.v[kv0,row=2]", tc)


def test_feature_removal_through_branches(setup):
    dbg, tset, run = setup
    tc = tset.get(LAYER)
    cb = components.transcoder_binding(run, "mlp", f"t4.L{LAYER}.m", tc)
    top = cb.components[0].name
    value = cb.edited(remove=(top,)).reshape(1, 1, -1).to(dbg.adapter.dtype)
    other = dbg.run(P, name="tc-branch")
    br = dbg.branches(other, Margin(4, 3, 7), at=f"t4.L{LAYER}.h")
    from mdb.analysis.branches import edit

    br.add("remove", [edit(f"t4.L{LAYER}.m", value, origin={"kind": "transcoder",
                                                            "removed": top})])
    row = br.run()[0]
    assert row["edits"][0]["origin"]["removed"] == top
    hand = dbg.run(P, name="tc-hand")
    hand.until(f"t4.L{LAYER}.h")
    hand.set(f"t4.L{LAYER}.m", value)
    hand.until("t4.logits")
    lg = f64(hand.read("t4.logits")).reshape(-1)
    assert row["M"]["value"] == float(lg[3] - lg[7])


def test_feature_context_and_saelens(setup):
    dbg, tset, run = setup
    tc = tset.get(LAYER)
    acts = features.activations(run, tc, 4, LAYER)
    enc = tc.encode(run.read(f"t4.L{LAYER}.n"))
    assert acts == {int(f): float(a) for f, a in zip(enc["ids"], enc["acts"].tolist())}
    pytest.importorskip("sae_lens")
    lens = tc.to_saelens()
    n = run.read(f"t4.L{LAYER}.n").reshape(1, -1).float()
    feats = lens.encode(n)[0]
    assert sorted(int(i) for i in (feats > 0).nonzero().reshape(-1)) == sorted(enc["ids"])
    dec = tc.decompose(run.read(f"t4.L{LAYER}.n"), run.read(f"t4.L{LAYER}.m"))
    assert torch.allclose(lens.decode(feats[None])[0].double(), dec.recon, atol=1e-4)
