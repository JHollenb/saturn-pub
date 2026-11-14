"""Pools (gap audit P6) on tiny Qwen3 (fp32) and Qwen2 (bf16).

Collection binds @roles per item and keeps each item's origin; the leave-one-out mean never holds
the item (the TP1 bug cannot happen) and equals the hand mean exactly; same-label means and donor
rules exclude the item; the matched-norm random has the item's norm; the sealed record verifies
and a saved pool reloads bit-exactly; a pool value fed to a branch arm keeps the pool pointer."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis.branches import edit  # noqa: E402
from mdb.analysis.finite import f64  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.analysis.pools import Pool, bind, collect  # noqa: E402
from mdb.core.records import verify_record  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

ITEMS = [{"id": f"i{k}", "ids": [1 + k, 5 + k, 9, 3 + k, 7], "roles": {"carrier": 2},
          "country": "AB"[k % 2], "correct": k != 5} for k in range(6)]


@pytest.fixture(scope="module", params=[("qwen3", "fp32"), ("qwen2", "bf16")],
                ids=lambda c: "-".join(c))
def dbg(request):
    return open_tiny(*request.param)


@pytest.fixture(scope="module")
def pool(dbg):
    return dbg.pool(ITEMS, "cache.L1.v[kv0,row=@carrier]", labels=("country",), name="cv",
                    where=lambda item, run: item["correct"])


def test_collect_binds_roles_and_filters(dbg, pool):
    assert bind("t@last.L1.h", ITEMS[0]) == "t4.L1.h"
    assert bind("cache.L1.v[kv0,row=@carrier]", ITEMS[0]) == "cache.L1.v[kv0,row=2]"
    assert len(pool) == 5 and pool.skipped == ["i5"]
    run = dbg.run(ITEMS[1]["ids"], name="check")
    run.until("t4.L0.h")
    assert torch.equal(pool.value("i1"), run.read_cache(1, "v")[0:1, 0:1, 2:3].cpu())
    assert pool.items[1]["origin"]["address"] == "cache.L1.v[kv0,row=2]"
    ports = collect(dbg, ITEMS[:2], "t@last.L1.h", name="h")
    assert ports.values.shape[1:] == (1, 1, dbg.adapter.hidden_size)


def test_leave_one_out_and_label_rules(pool):
    loo = pool.mean_for("i0")
    assert loo.pointer["excluded"] == "i0" and "i0" not in loo.pointer["items"]
    assert torch.equal(loo.value, f64(pool.values[1:]).mean(0))
    same = pool.class_mean_for("i0", by="country")
    assert same.pointer["items"] == ["i2", "i4"]
    assert torch.equal(same.value, f64(pool.values[[2, 4]]).mean(0))
    assert pool.donor("i0", rule="other:country").pointer["donor"] == "i1"
    assert pool.donor("i0", rule="same:country").pointer["donor"] == "i2"
    assert pool.donor("i0", rule="any", order=["i0", "i3"]).pointer["donor"] == "i3"
    rnd = pool.random("i0", seed=3)
    assert abs(float(rnd.value.norm()) - float(f64(pool.value("i0")).norm())) < 1e-12
    assert pool.mean().pointer["excluded"] is None and pool.mean().pointer["n"] == 5
    with pytest.raises(Refusal):
        pool.donor("i0", rule="nearest")


def test_seal_save_load(pool, tmp_path):
    record = pool.seal()
    verify_record(record)
    assert record["body"]["n"] == 5 and record["body"]["custody"]["schema"] == "mdb-custody-v1"
    assert pool.held_payload()["ids"] == pool.ids
    back = Pool.load(pool.save(tmp_path / "pool.json"))
    assert torch.equal(back.values, pool.values) and back.record["id"] == record["id"]
    assert pool.mean_for("i0").pointer["record"] == record["id"]


def test_pool_value_in_a_branch(dbg, pool):
    item = ITEMS[0]
    run = dbg.run(item["ids"], name="pb")
    br = dbg.branches(run, Margin(4, 3, 7), at="t4.L0.h")
    br.add("loo", [edit("cache.L1.v[kv0,row=2]", pool.mean_for(item))])
    br.add_null("self", "self", like="loo")
    rows = {r["name"]: r for r in br.run()}
    origin = rows["loo"]["cells"][0]["origin"]
    assert origin["kind"] == "pool" and origin["excluded"] == "i0"
    assert rows["self"]["dM"]["value"] == 0.0


def test_collect_with_prefill_split_is_stepped(dbg):
    """``prefill=`` collects each cell on the driver's own split: the value equals a run of
    ``dbg.run(ids, prefill=K)`` read at the same row, the origin records K, and the pool's kernel
    path is ``mdb-stepped`` (the default pool reads the batched prefill)."""
    items = ITEMS[:3]
    stepped = dbg.pool(items, "cache.L1.v[kv0,row=@carrier]", name="cv-stepped",
                       prefill=lambda item: 1)
    assert stepped.kernel_path == "mdb-stepped"
    assert all(it["origin"]["prefill"] == 1 for it in stepped.items)
    for item in items:
        run = dbg.run(item["ids"], name="check-stepped", prefill=1)
        run.until("t4.logits")
        assert torch.equal(stepped.value(item["id"]),
                           run.read_cache(1, "v")[0:1, 0:1, 2:3].cpu())
    batched = dbg.pool(items, "cache.L1.v[kv0,row=@carrier]", name="cv-batched")
    assert batched.kernel_path == "mdb-prefill-batched"
    late = dbg.pool(items, "cache.L1.v[kv0,row=@carrier]", name="cv-late",
                    prefill=lambda item: 3)        # the cell row 2 is prefilled: not stepped
    assert late.kernel_path == "mdb-prefill-batched"
