"""mdb.machines.x86.mlp: the gated-MLP dispatch as sites, on the hand-assembled binary of test_machines_x86."""
import pytest

torch = pytest.importorskip("torch")
tm = pytest.importorskip("test_machines_x86")       # sets MDB_TORCHX86_SRC, skips without torchx86

from mdb.machines.x86 import mlp  # noqa: E402
from mdb.machines.x86.lanes import Lanes  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    return tm.binary()


@pytest.fixture(scope="module")
def inputs():
    return [tm.inp(5, 7), *tm.donors(5)]


@pytest.fixture(scope="module")
def hard(tiny, inputs):
    return mlp.capture(tiny, inputs)


def test_hard_equals_plain(tiny, inputs, hard):
    plain = Lanes(tiny, inputs).run(record="pc")
    assert torch.equal(plain.pc, hard.pc)
    assert plain.y == hard.result.y and torch.equal(plain.T, hard.result.T)
    assert plain.y[0] == tm.expected(5, 7).to_bytes(8, "little")


def test_blind_attribution_names_the_class(tiny, hard):
    cls = mlp.decoded_class(tiny, hard.pc)[hard.live]
    r = mlp.attribute_capture(hard)
    anybit = hard.down[hard.live].bool().any(-1)
    for key in ("named_contribution", "named_ablation"):
        named = r[key]
        assert torch.equal(named[anybit], cls[anybit])        # every step with an output bit on
        assert bool((named[~anybit] == -1).all())             # an all-zero output names nothing
    assert int((hard.gate[hard.live] != 0).sum(-1).max()) == 1


def test_gate_side_names_the_zero_valued_selected_unit(tiny, hard):
    """Run 6: every step whose output is all zero names nothing from f; the gate names it."""
    cls = mlp.decoded_class(tiny, hard.pc)[hard.live]
    r = mlp.attribute_capture(hard)
    anybit = hard.down[hard.live].bool().any(-1)
    sel = r["named_selected"]
    assert bool((~anybit).any())                               # the fixture has a zero output
    assert torch.equal(sel[~anybit], cls[~anybit])             # named by the gate
    nonzero_f = (hard.gate[hard.live][:, :, None] * hard.up[hard.live].float()).abs().sum((1, 2)) > 0
    assert bool((sel[nonzero_f] == -1).all())                  # a valued selection is not "zero"


def test_soft_gate(tiny, inputs, hard):
    taus = torch.tensor([1.0, 0.001]).repeat_interleave(len(inputs))[:, None]
    soft = mlp.capture(tiny, inputs * 2, mode="soft", tau=taus)
    n = len(inputs)
    assert torch.equal(soft.pc[n:, :hard.pc.shape[1]], hard.pc)          # tau = 0.001 is exact
    up = soft.up[:n][soft.live[:n]].float()
    cls = mlp.decoded_class(tiny, soft.pc[:n])[soft.live[:n]]
    exact = (soft.down[:n][soft.live[:n]] == up[torch.arange(len(cls)), cls].to(torch.uint8)).all(-1)
    assert float(exact.float().mean()) < 1.0                            # tau = 1 blurs the output


def test_usage(tiny, hard):
    u = mlp.usage(hard, mlp.decoded_class(tiny, hard.pc), [lp.head for lp in tiny.loops])
    assert all(x["units_on_per_step_mean"] == 1.0 for x in u)
    assert all(x["union_units"] == x["union_classes_decoded"] for x in u)
    assert u[0]["first_loop_iteration_end_tick"] > 0


def test_sites_and_whatif(tiny):
    run = mlp.MlpRun(tiny, tm.inp(5, 7))
    names = run.names
    t = next(i for i in range(run.T) if int(mlp.decoded_class(tiny, run.cap_.pc[0, i:i + 1])) == names.index("add"))
    assert run.read(f"i{t}.mlp.gate[add]")["value"] == 1.0
    assert run.read(f"i{t}.mlp.f[sub]")["nonzero_bits"] == 0
    assert run.read(f"i{t}.mlp.down")["value"] == run.read(f"i{t}.mlp.up[add]")["value"]
    assert run.attribute(t)["named_ablation"] == "add"
    assert "value" in run.print(f"i{t}.mlp.down")
    off = run.whatif(f"i{t}.mlp.gate[add]", 0)
    assert off["down_prediction"] == "confirmed" and off["total_effect"] == "effect"
    both = run.whatif(f"i{t}.mlp.gate[sub]", 1)
    assert both["down_prediction"] == "confirmed"
    with pytest.raises(ValueError):
        run.read(f"i{t}.mlp.down[add]")


def test_topk_sae_and_purity():
    g = torch.Generator().manual_seed(0)
    K, n = 3, 300
    cls = torch.arange(n) % K
    X = torch.zeros(n, K * mlp.NB)
    for i in range(n):
        X[i, cls[i] * mlp.NB:(cls[i] * mlp.NB) + 8] = (torch.rand(8, generator=g) > 0.5).float() + 0.0
        X[i, cls[i] * mlp.NB + 128] = 1.0
    p = mlp.train_topk_sae(X, [torch.arange(n), torch.arange(0, n, K)], width=4 * K, k=2, steps=300, batch=64)
    assert p["W_enc"].shape == (2, 4 * K, K * mlp.NB) and p["W_dec"].shape == (2, K * mlp.NB, 4 * K)
    a = mlp.sae_encode(p, 0, X)
    assert int((a > 0).sum(1).max()) <= 2
    feats = mlp.purity(a, cls, K)
    live = [f for f in feats if f["fires"]]
    assert live and all(0 < f["purity"] <= 1 for f in live)
    vs = mlp.value_split(a, X, cls, feats, K)
    assert vs["pure_features"] >= 1
    a1 = mlp.sae_encode(p, 1, X[cls == 0])
    assert all(f["purity"] == 1.0 for f in mlp.purity(a1, cls[cls == 0], K) if f["fires"])
