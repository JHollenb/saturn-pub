"""Donor counts on nulls (``analysis.donors``) and ``path-diverged`` with the paired effect
(``analysis.paths``), binary ground truth lessons 7 and 8.

A self-patch (the donor carries the base's own value) is the identity intervention behind most
single-donor nulls on the binaries: it must read ``no effect (1 donor, not informative ...)``,
never a bare ``no effect``. Several donors pool to ``effect`` when any one has an effect, with
``n_informative`` counting the donors whose value differed."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.analysis import donors, paths  # noqa: E402
from mdb.analysis.compare import Side, compare  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.analysis.liveness import liveness  # noqa: E402
from mdb.analysis.propagate import Edit, whatif  # noqa: E402
from mdb.analysis.reads import readout, reads  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops import dispatch  # noqa: E402

MAIN, CLEAN, OTHER = [1, 5, 9, 3, 6, 2], [1, 7, 9, 3, 6, 2], [1, 8, 9, 3, 6, 2]
M = Margin(5, 4, 9)


@pytest.fixture(scope="module")
def dbg():
    return mdb.attach(tiny_model("qwen2", "bf16", layers=4))


def _run(dbg, tokens, name=None):
    run = dbg.run(tokens, prefill=0) if name is None else dbg.run(tokens, name=name, prefill=0)
    run.until("t5.logits")
    return run


def test_labels_and_pooling_rules():
    assert donors.label("no effect") == "no effect (1 donor)"
    assert donors.label("no effect", 1, 0).startswith("no effect (1 donor, not informative")
    assert donors.label("no effect", 16, 12) == "no effect (16 donors, 12 informative)"
    assert donors.label("effect") == "effect"
    rows = [{"verdict": "no effect", "informative": False},
            {"verdict": "effect", "informative": True},
            {"verdict": "no effect", "informative": True}]
    pooled = donors.pooled(rows)
    assert (pooled["verdict"], pooled["n_donors"], pooled["n_informative"], pooled["n_effect"]) \
        == ("effect", 3, 2, 1)
    assert pooled["label"] == "effect (1 of 3 donors, 2 informative)"
    null = donors.pooled(rows[:1] + rows[2:])
    assert null["label"] == "no effect (2 donors, 1 informative)"
    assert donors.informative(torch.ones(3), torch.ones(3)) is False
    assert donors.informative(torch.ones(3), torch.tensor([1.0, 1.0, 2.0])) is True


def test_a_self_patch_is_a_one_donor_uninformative_null(dbg):
    run = _run(dbg, MAIN)
    own = run.read("t1.L2.h_out").clone()
    effect = whatif(run, Edit("t1.L2.h_out", own), M).report["effect"]
    assert effect["verdict"] == "no effect"
    assert effect["donors"]["n"]["value"] == 1 and effect["donors"]["informative"]["value"] == 0
    assert effect["label"].startswith("no effect (1 donor, not informative")


def test_multi_donor_whatif_pools_and_counts_informative_donors(dbg):
    run = _run(dbg, MAIN)
    clean = _run(dbg, CLEAN)
    values = [run.ref("t1.L2.h_out"), clean.ref("t1.L2.h_out")]
    out = donors.whatif(run, "t1.L2.h_out", values, M)
    assert out["n_donors"] == 2 and out["n_informative"] == 1
    assert [d["informative"] for d in out["per_donor"]] == [False, True]
    assert out["per_donor"][0]["verdict"] == "no effect"
    assert out["verdict"] == ("effect" if out["per_donor"][1]["verdict"] == "effect"
                              else "no effect")
    assert run.where == "t5.logits"


def test_compare_stamps_each_side_with_its_informativeness(dbg):
    run = _run(dbg, MAIN)
    clean = _run(dbg, CLEAN)
    a = Side("self", [Edit("t1.L1.h_out", run.read("t1.L1.h_out").clone())])
    b = Side("clean", [Edit("t1.L1.h_out", clean.read("t1.L1.h_out").clone())])
    verdicts = compare(run, a, b, M)["report"]["verdicts"]
    assert verdicts["A"]["verdict"] == "no effect"
    assert verdicts["A"]["donors"]["informative"]["value"] == 0
    assert verdicts["B"]["donors"]["informative"]["value"] == 1
    pooled = donors.compare(run, lambda d: (a, Side("d", [Edit("t1.L1.h_out", d)])),
                            [run.read("t1.L1.h_out").clone(), clean.read("t1.L1.h_out").clone()],
                            M)
    assert pooled["columns"]["B"]["n_donors"] == 2
    assert pooled["columns"]["B"]["n_informative"] == 1


def test_liveness_resample_donors_identity_donor_is_exactly_null(dbg):
    run = _run(dbg, MAIN)
    same, other = _run(dbg, MAIN), _run(dbg, OTHER)
    cell = SimpleNamespace(row=1, ports=None, layers=None, kv=None)
    binding = SimpleNamespace(name="x", cells=[cell])
    out = liveness(run, [binding], readout(run, M), donors=[same, other])["x"]
    assert out["donors"]["n"]["value"] == 1 and "label" in out
    pooled = out["donors_resample"]
    assert pooled["n_donors"] == 2 and pooled["n_informative"] == 1
    assert pooled["per_donor"][0]["delta"]["value"] == 0.0
    assert pooled["per_donor"][0]["verdict"] == "no effect"


def test_reads_donors_pool_per_read(dbg):
    run = _run(dbg, MAIN)
    same, other = _run(dbg, MAIN), _run(dbg, OTHER)
    found = reads(run, "t1", consumer=M, top=2, donors=[same, other]).report
    for entry in found["reads"]:
        pooled = entry["causal"]["donors"]
        assert pooled["n_donors"] == 2
        assert pooled["per_donor"][0]["informative"] is False
        assert pooled["per_donor"][0]["verdict"] == "no effect"


def test_sweep_over_several_origins_counts_overturned_nulls(dbg):
    ctx = dbg.ops()
    dispatch(ctx, "start", {"prompt": CLEAN, "name": "clean", "prefill": 0})
    dispatch(ctx, "until", {"target": "t5.logits"})
    dispatch(ctx, "start", {"prompt": MAIN, "name": "same", "prefill": 0})
    dispatch(ctx, "until", {"target": "t5.logits"})
    dispatch(ctx, "start", {"prompt": MAIN, "prefill": 0})
    dispatch(ctx, "until", {"target": "t5.logits"})
    one = dispatch(ctx, "sweep", {"port": "h", "origin": "same", "consumer": "t5.M[4,9]",
                                  "rows": "0:3"})
    assert one["nulls"]["count"] == one["cells"]["count"]
    assert one["nulls"]["informative"]["value"] == 0
    both = dispatch(ctx, "sweep", {"port": "h", "origin": "same,clean",
                                   "consumer": "t5.M[4,9]", "rows": "0:3"})
    d = {k: v["value"] if isinstance(v, dict) and "value" in v else v
         for k, v in both["donors"].items()}
    assert d["n_donors"] == 2 and len(both["records"]) == 2
    # every "same" cell is a null; those the clean origin changes are overturned
    assert d["single_origin_nulls_overturned"] == d["cells_effect"] > 0
    assert d["n_informative"][0][0] == 0           # row 0: the same token in all runs


def test_argmax_rows_ties_and_divergence():
    p = torch.tensor([[0.5, 0.5, 0.0], [0.1, 0.7, 0.2]])
    assert paths.argmax_rows(p).tolist() == [-1, 1]
    out = paths.diverged([1, 2, -1], [1, 0, 2], ["a", "b", "c"])
    assert out["diverged"] and out["moved"] == 1 and out["ties"] == 1 and out["first"] == "b"
    assert not paths.diverged([1, 2], [1, 2])["diverged"]


def test_paired_effects_self_patch_and_large_edit(dbg):
    run = _run(dbg, MAIN)
    own = run.read("t1.L1.h_out").clone()
    same = paths.paired(run, [("t1.L1.h_out", own, "replace", None)], M)
    assert same["verdict"] == "no effect" and not same["path"]["diverged"]
    assert same["total"]["dM"]["value"] == same["frozen"]["dM"]["value"] == 0.0
    big = paths.paired(run, [("t1.L1.h_out", 40.0, "add", None)], M)
    assert big["path"]["selections"] == 5 * 2 * dbg.adapter.num_heads  # rows 1..5 x L2..3
    if big["path"]["diverged"]:
        assert big["verdict"] == "path-diverged" and big["path"]["first"].startswith("t")
        assert big["frozen_minus_total"]["value"] != 0.0
    assert run.where == "t5.logits"
