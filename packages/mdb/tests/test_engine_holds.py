"""Holds (gap audit P4) on tiny Qwen2 (bf16) and Qwen3 (fp32, 4 layers).

A hold against the run itself is bit-identical to the free run; a held arm equals exp4's
hand-built arm (p, then norm scales, set at every later layer by hand) bit for bit; the hold edits
are ordinary labelled edits that survive checkpoint/restore; and on fp32 the held arm's finite
difference follows nnsight's frozen-pattern, frozen-norm gradient (the attribution-graph
linearisation) while the free arm need not."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.engine import holds  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops.debugger import attach, open_tiny  # noqa: E402

P = [1, 5, 9, 3, 7]
LAST = len(P) - 1


@pytest.fixture(scope="module", params=[("qwen2", "bf16", 2), ("qwen3", "fp32", 4)],
                ids=lambda c: f"{c[0]}-{c[1]}")
def dbg(request):
    family, dtype, layers = request.param
    return attach(tiny_model(family, dtype, layers=layers))


def _base(dbg):
    base = dbg.run(P, name="base")
    base.until(f"t{LAST}.logits")
    return base


def test_self_hold_is_bit_identical(dbg):
    base = _base(dbg)
    run = dbg.run(P, name="held")
    run.hold("t@last.L*.p", source=base)
    run.hold("t@last.L*.norm_scale", source=base, name="norms")
    run.until(f"t{LAST}.logits")
    assert torch.equal(run.read(f"t{LAST}.logits"), base.read(f"t{LAST}.logits"))
    fired = [e for e in run.committed_edits() if e.get("origin", {}).get("kind") == "hold"]
    assert len(fired) == 3 * dbg.adapter.layers                 # p, a, n per layer
    assert all(e["realised_exact"] for e in fired)
    assert {h["name"] for h in run.holds} == {"hold1", "norms"}
    assert holds.final_held_logits(run, LAST, base).shape[-1] == dbg.adapter.vocab_size


def _hand_arm(dbg, base, x, delta, norms):
    """exp4_brackets.py arm1/arm2, verbatim in mdb calls (the code holds replace)."""
    model, NL = dbg.adapter.model, dbg.adapter.layers
    ref = {}
    for x2 in range(x + 1, NL):
        L_ = model.model.layers[x2]
        h = base.read(f"t{LAST}.L{x2}.h").reshape(-1)
        r = base.read(f"t{LAST}.L{x2}.r").reshape(-1)
        ref[x2] = {"p": base.read(f"t{LAST}.L{x2}.p").clone(),
                   "ra": holds.rms_scale(h, L_.input_layernorm.variance_epsilon),
                   "rn": holds.rms_scale(r, L_.post_attention_layernorm.variance_epsilon)}
    run = dbg.run(P, name="hand")
    run.until(f"t{LAST}.L{x}.h")
    run.set(f"t{LAST}.L{x}.m", delta, "add")
    for x2 in range(x + 1, NL):
        run.until(f"t{LAST}.L{x2}.h")
        run.set(f"t{LAST}.L{x2}.p", ref[x2]["p"])
        if norms:
            L2 = model.model.layers[x2]
            h = run.read(f"t{LAST}.L{x2}.h")
            run.set(f"t{LAST}.L{x2}.a", L2.input_layernorm.weight
                    * (h.float() * ref[x2]["ra"]).to(h.dtype))
            run.until(f"t{LAST}.L{x2}.r")
            r = run.read(f"t{LAST}.L{x2}.r")
            run.set(f"t{LAST}.L{x2}.n", L2.post_attention_layernorm.weight
                    * (r.float() * ref[x2]["rn"]).to(r.dtype))
    run.until(f"t{LAST}.logits")
    return run


def _held_arm(dbg, base, x, delta, norms):
    run = dbg.run(P, name="holds")
    run.hold(f"t@last.L>{x}.p", source=base)
    if norms:
        run.hold(f"t@last.L>{x}.norm_scale", source=base)
    run.until(f"t{LAST}.L{x}.h")
    run.set(f"t{LAST}.L{x}.m", delta, "add")
    run.until(f"t{LAST}.logits")
    return run


@pytest.mark.parametrize("norms", [False, True], ids=["arm1-p", "arm2-p+norms"])
def test_hold_reproduces_exp4_hand_arms(dbg, norms):
    base = _base(dbg)
    d = dbg.adapter.hidden_size
    delta = (torch.arange(d, dtype=torch.float32) / d - 0.5).reshape(1, 1, d).to(dbg.adapter.dtype)
    hand = _hand_arm(dbg, base, 0, delta, norms)
    held = _held_arm(dbg, base, 0, delta, norms)
    last = dbg.adapter.layers - 1
    assert torch.equal(hand.read(f"t{LAST}.L{last}.h_out"), held.read(f"t{LAST}.L{last}.h_out"))
    assert torch.equal(hand.read(f"t{LAST}.logits"), held.read(f"t{LAST}.logits"))
    free = dbg.run(P, name="free")
    free.until(f"t{LAST}.L0.h")
    free.set(f"t{LAST}.L0.m", delta, "add")
    free.until(f"t{LAST}.logits")
    if dbg.adapter.dtype == torch.float32:      # the hold changes something downstream
        assert not torch.equal(free.read(f"t{LAST}.L{last}.h_out"),
                               held.read(f"t{LAST}.L{last}.h_out"))


def test_hold_edits_survive_restore_and_release(dbg):
    base = _base(dbg)
    run = dbg.run(P, name="ck")
    run.hold("t@last.L*.norm_scale", source=base)
    run.until(f"t{LAST}.L0.m")
    staged = [e for e in run.edits if e.get("origin", {}).get("kind") == "hold"]
    assert [e["port"] for e in staged] == ["a", "n"]
    ck = run.checkpoint("mid")
    run.until(f"t{LAST}.logits")
    first = run.read(f"t{LAST}.logits")
    run.restore(ck)
    run.until(f"t{LAST}.logits")
    assert torch.equal(run.read(f"t{LAST}.logits"), first)
    assert run.release() == ["hold1"] and run.holds == []
    run.restore(ck)
    assert [e["port"] for e in run.edits] == ["a", "n"]        # the checkpoint's staged edits


def test_cell_hold_and_refusals(dbg):
    base = _base(dbg)
    other = dbg.run([2, 8, 4, 6, 3], name="other")
    other.until(f"t{LAST}.logits")
    run = dbg.run(P, name="cell")
    run.until(f"t{LAST}.L1.h")
    out = run.hold("cache.L0.v[kv0,row=1]", source=other)
    assert out["kind"] == "cell" and out["cell"]["origin"]["kind"] == "hold"
    assert run.cells[0]["origin"]["address"] == "cache.L0.v[kv0,row=1]"
    with pytest.raises(ValueError):
        run.hold("t@last.L*.kr", source=base)                  # not editable
    with pytest.raises(ValueError):
        run.hold("t@last.L*.p")                                # no source
    with pytest.raises(ValueError):
        run.hold("L9.p", source=base)                          # no such layer


def test_held_arm_follows_nnsight_frozen_gradient():
    """check_frozen_grad.py's comparison with mdb holds as the frozen forward: on fp32 the held
    arm's finite difference matches grad . delta (nnsight, q/k detached, norm scales frozen)
    within 2e-3 at a 1e-5 move (the free arm does not: the gap is the re-routing and norm terms).
    The frozen scales must be per row (a last-row scale broadcast to every row moves the other
    rows' keys and the gradient by a few percent: measured while writing this test)."""
    nnsight = pytest.importorskip("nnsight")
    from mdb.interop import nn

    model = tiny_model("qwen3", "fp32", layers=4)
    dbg = attach(model)
    base = _base(dbg)
    layer, NL, a, b = 1, 4, 3, 7
    nm = nn.wrap(model)
    with torch.no_grad(), nm.trace(torch.tensor([P])):      # per-row scales, as exp4 saved them
        saved = nnsight.save({})
        for x2 in range(layer + 1, NL):
            L2 = nm.model.layers[x2]
            saved[("a", x2)] = L2.input_layernorm.input.save()
            saved[("n", x2)] = L2.post_attention_layernorm.input.save()
        saved[("F", 0)] = nm.model.norm.input.save()
    sc = {"a": {}, "n": {}}
    for x2 in range(layer + 1, NL):
        L2 = model.model.layers[x2]
        sc["a"][x2] = (holds.rms_scale(saved[("a", x2)], L2.input_layernorm.variance_epsilon)
                       * L2.input_layernorm.weight.float())
        sc["n"][x2] = (holds.rms_scale(saved[("n", x2)], L2.post_attention_layernorm.variance_epsilon)
                       * L2.post_attention_layernorm.weight.float())
    sc["F"] = (holds.rms_scale(saved[("F", 0)], model.model.norm.variance_epsilon)
               * model.model.norm.weight.float())
    nm = nn.wrap(model)
    with nn._Frozen(model), torch.enable_grad(), nm.trace(torch.tensor([P])):
        nm.model.embed_tokens.output.requires_grad_(True)
        mout = nm.model.layers[layer].mlp.output
        for x in range(layer + 1, NL):
            L_ = nm.model.layers[x]
            L_.input_layernorm.output = L_.input_layernorm.input * sc["a"][x]
            L_.self_attn.q_proj.output = L_.self_attn.q_proj.output.detach()
            L_.self_attn.k_proj.output = L_.self_attn.k_proj.output.detach()
            L_.post_attention_layernorm.output = L_.post_attention_layernorm.input * sc["n"][x]
        nm.model.norm.output = nm.model.norm.input * sc["F"]
        lg = nm.lm_head.output[0, -1].float()
        M = lg[a] - lg[b]
        g = nnsight.save([])
        with M.backward():
            g.append(mout.grad[0, -1].detach().double())
    grad = g[0]
    M0 = float(base.read(f"t{LAST}.logits").reshape(-1)[a] - base.read(f"t{LAST}.logits").reshape(-1)[b])
    ratios, free_ratios = [], []
    for seed in range(3):
        d = torch.randn(32, generator=torch.Generator().manual_seed(seed)) * 1e-5
        held = _held_arm(dbg, base, layer, d.reshape(1, 1, -1), True)
        lg_h = holds.final_held_logits(held, LAST, base).reshape(-1).double()
        linear = float(grad @ d.double())
        ratios.append((float(lg_h[a] - lg_h[b]) - M0) / linear)
        free = dbg.run(P, name="free")
        free.until(f"t{LAST}.L{layer}.h")
        free.set(f"t{LAST}.L{layer}.m", d.reshape(1, 1, -1), "add")
        free.until(f"t{LAST}.logits")
        lg_f = free.read(f"t{LAST}.logits").reshape(-1).double()
        free_ratios.append((float(lg_f[a] - lg_f[b]) - M0) / linear)
    # measured 2026-11-14: held 1.00001/1.00018/0.99998, free 0.603/-1.851/1.004 (eps 1e-5)
    assert all(abs(r - 1) < 2e-3 for r in ratios), ratios
    assert max(abs(r - 1) for r in free_ratios) > 0.1, free_ratios
