"""Python selectors and cross-run origins on Run.set (gap audit P5) on tiny Qwen2/Qwen3.

``Run.set`` with a selector equals the hand-built mask (exp5/exp6 pattern); a ``ValueRef`` from
another run patches exactly the selected block and records the origin pointer; cache-cell targets
become ``set_cell`` with rows/kv/d from the selector; a ref of the wrong block shape is refused;
the REPL's ``set`` and the Python path stage the same edit."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.engine import refs  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

A = [1, 5, 9, 3, 7]
B = [2, 8, 4, 6, 3]


@pytest.fixture(scope="module", params=[("qwen2", "bf16"), ("qwen3", "fp32")],
                ids=lambda c: "-".join(c))
def dbg(request):
    return open_tiny(*request.param)


def test_selector_mask_equals_hand_mask(dbg):
    run = dbg.run(A, name="sel")
    run.until("t4.L1.h")
    ck = run.checkpoint()
    run.set("t@last.L1.p[h1,src=0:3]", 0.0)
    run.until("t4.logits")
    by_selector = run.read("t4.logits")
    edit = run.committed_edits()[-1]
    assert edit["selection"] == "h1,src=0:3" and edit["mask_count"] == 3
    run.restore(ck)
    mask = torch.zeros(run.port_shape("p"), dtype=torch.bool)
    mask[0, 1, 0, 0:3] = True
    run.set("t4.L1.p", 0.0, mask=mask)
    run.until("t4.logits")
    assert torch.equal(run.read("t4.logits"), by_selector)


def test_ref_patches_block_with_origin(dbg):
    hd = dbg.adapter.head_dim
    donor = dbg.run(B, name="donor")
    donor.until("t4.logits")
    where = donor.where
    ref = donor.ref("t4.L1.A[h1]")
    assert donor.where == where                              # reading moved nothing
    assert ref.shape == (1, 1, hd) and ref.pointer["origin"] == "donor"
    assert ref.pointer["address"] == "t4.L1.A[h1]"
    run = dbg.run(A, name="recipient")
    run.until("t4.L1.h")
    manifest = run.set("L1.A[h1]", ref)
    assert manifest["origin"]["sha256"] == ref.pointer["sha256"]
    run.until("t4.L1.A")
    got, want = run.read("t4.L1.A"), donor.read("t4.L1.A")
    assert torch.equal(got[..., hd:2 * hd], want[..., hd:2 * hd])
    assert torch.equal(got[..., :hd], run.original("t4.L1.A")[..., :hd])
    run.abort()
    with pytest.raises(ValueError):
        run.set("L1.A[h0]", donor.ref("t4.L1.A"))          # whole port into one head: refused


def test_cache_cell_target_and_origin(dbg):
    donor = dbg.run(B, name="cdonor")
    donor.until("t4.logits")
    run = dbg.run(A, name="crec")
    run.until("t4.L1.h")
    run.set("cache.L0.v[kv0,row=2]", donor.ref("cache.L0.v[kv0,row=2]"))
    cell = run.cells[0]
    assert cell["cell"]["rows"] == [2] and cell["cell"]["kv"] == [0]
    assert cell["origin"]["address"] == "cache.L0.v[kv0,row=2]"
    assert torch.equal(run.read_cache(0, "v")[0, 0, 2], donor.read_cache(0, "v")[0, 0, 2])
    with pytest.raises(ValueError):
        run.set("cache.L0.v[kv0]", 0.0)                     # rows are required


def test_repl_and_python_stage_the_same_edit(dbg):
    from mdb.ops import dispatch

    donor = dbg.run(B, name="rd")
    donor.until("t4.logits")
    run = dbg.run(A, name="rr")
    run.until("t4.L1.h")
    ctx = dbg.ops(run)
    dispatch(ctx, "set", {"target": "t4.L1.A[h1]", "value": "rd:t4.L1.A[h1]"})
    repl = run.edits[-1]
    run.abort()
    run.set("t4.L1.A[h1]", donor.ref("t4.L1.A[h1]"))
    py = run.edits[-1]
    for key in ("port", "op", "selection", "mask_sha256", "mask_count"):
        assert repl[key] == py[key]
    assert repl["value"]["sha256"] == py["value"]["sha256"]
    assert repl["origin"]["sha256"] == py["origin"]["sha256"]


def test_split_and_resolve(dbg):
    run = dbg.run(A, name="parse")
    assert refs.split("t4.L1.p[h0]") == ("t4.L1.p", "h0")
    assert refs.resolve(run, "t@last.L1.p[h0]")["address"] == "t4.L1.p[h0]"
    with pytest.raises(ValueError):
        refs.resolve(run, "t@carrier.L1.p")
