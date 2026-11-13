"""State-cell edits (``Run.set_cell``, ``engine.edits.CellEdit``): native realisation against an
unhooked reference, the read boundary, rewind and durability, labels and interrupts.

The reference is the plain Hugging Face incremental program (``model(..., past_key_values=)``)
with the cache tensor rewritten by hand between token steps, never through mdb code.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import saturn_pub  # noqa: E402
from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from saturn_pub.core import Session  # noqa: E402
from test_engine_ports import PROMPT, build_model  # noqa: E402

import mdb  # noqa: E402
from mdb.engine import Run  # noqa: E402
from mdb.engine.edits import CellEdit  # noqa: E402
from mdb.engine.ports import tensor_digest  # noqa: E402

CASES = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32")]
ROW = len(PROMPT) - 1  # 6: the first stepped row
KV, ROWS, DIMS = [1], [2, 4], [0, 3, 5, 15]  # a selector subset of (2 kv heads, head_dim 16)


@pytest.fixture(scope="module", params=CASES, ids=lambda case: "-".join(case))
def model(request):
    return build_model(*request.param)


@pytest.fixture(scope="module")
def adapter(model):
    return DecoderAdapter(model, granularity="operation")


def _value(seed=7):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, len(KV), len(ROWS), len(DIMS)), generator=generator)


# ops: (op, value, dose); the reference rewrites the cache by hand
OPS = {
    "replace": ("replace", _value(), None),
    "add": ("add", 0.375, None),
    "scale": ("scale", 1.5, None),
    "mix": ("mix", _value(9), 0.5),
}


def _hand(op, value, dose, rows=ROWS):
    """The cache rewrite of one cell edit, written out by hand (no mdb code)."""

    def rewrite(cache, layer, tensor):
        slot = cache.layers[layer]
        old = slot.keys if tensor == "k" else slot.values
        index = (0, torch.tensor(KV).view(-1, 1, 1), torch.tensor(rows).view(1, -1, 1),
                 torch.tensor(DIMS).view(1, 1, -1))
        new = old.clone()
        block = old[index]
        if op == "replace":
            new[index] = value[0].to(old.dtype)
        elif op == "add":
            new[index] = block + torch.tensor(value).to(old.dtype)
        elif op == "scale":
            new[index] = block * torch.tensor(value).to(old.dtype)
        else:
            x = block.double()  # float64 from the native bits, rounded once
            new[index] = (x + dose * (value[0].double() - x)).to(old.dtype)
        if tensor == "k":
            slot.keys = new
        else:
            slot.values = new

    return rewrite


def reference(model, before=(), after=()):
    """Logits of rows 6 and 7 of the plain incremental program, with ``before`` cache rewrites
    applied before row 6 runs and ``after`` ones between rows 6 and 7."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(torch.tensor([PROMPT[:-1]]), past_key_values=cache, use_cache=True)
        for rewrite in before:
            rewrite(cache)
        six = model(torch.tensor([[PROMPT[-1]]]), past_key_values=cache,
                    use_cache=True).logits[0, -1]
        for rewrite in after:
            rewrite(cache)
        seven = model(torch.tensor([[int(six.argmax())]]), past_key_values=cache,
                      use_cache=True).logits[0, -1]
    return six, seven


def _logits(run):
    run.until(f"t{ROW}.logits")
    six = run.read().reshape(-1)
    run.until(f"t{ROW + 1}.logits")
    return six, run.read().reshape(-1)


# (cursor, cell layer, tensor, timing, when the reference rewrites: before/after row 6)
PLACES = [
    (f"t{ROW}.L1.qr", 1, "k", "before", "before"),   # kr (the k read boundary) is still ahead
    (f"t{ROW}.L1.kr", 1, "k", "after", "after"),     # the current row's k read happened
    (f"t{ROW}.L1.s", 1, "v", "before", "before"),    # p (the v read boundary) is still ahead
    (f"t{ROW}.L1.p", 1, "v", "after", "after"),      # A was computed with p: v read happened
    (f"t{ROW}.L1.h", 0, "v", "before", "after"),     # an earlier layer: row 6 already read it
    (f"t{ROW}.L1.h", 2, "k", "before", "before"),    # a later layer: row 6 reads it after
    (f"t{ROW}.logits", 1, "v", "before", "after"),   # carried to the next layer transition
    (f"t{ROW}.emb", 1, "k", "before", "before"),
]


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("place", PLACES, ids=lambda p: f"{p[0]}-L{p[1]}{p[2]}")
def test_cell_edit_equals_the_unhooked_program_with_a_hand_rewritten_cache(model, adapter,
                                                                            place, op):
    where, layer, tensor, timing, when = place
    name, value, dose = OPS[op]
    run = Run(adapter, PROMPT)
    run.until(where)
    manifest = run.set_cell(layer, tensor, value, name, rows=ROWS, kv=KV, d=DIMS, dose=dose)
    assert manifest["cell"]["timing"] == timing and manifest["port"] == f"cache.L{layer}.{tensor}"
    six, seven = _logits(run)
    rewrite = _hand(name, value, dose)
    hand = [lambda cache: rewrite(cache, layer, tensor)]
    expected = reference(model, before=hand if when == "before" else (),
                         after=hand if when == "after" else ())
    assert torch.equal(six, expected[0]) and torch.equal(seven, expected[1])
    base = reference(model)
    assert not torch.equal(seven, base[1])  # the edit reached the next row
    if when == "after":
        assert torch.equal(six, base[0])  # reads already made are not recomputed
    fired = run.committed_cells()
    assert len(fired) == 1 and fired[0]["realised_timing"] == timing


def test_row_written_by_the_pending_layer_and_unwritten_rows_are_refused(adapter):
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L1.q")
    with pytest.raises(ValueError, match="written by the pending layer"):
        run.set_cell(1, "v", 0.0, rows=[ROW])
    with pytest.raises(ValueError, match="not written yet"):
        run.set_cell(2, "v", 0.0, rows=[ROW])
    run.set_cell(0, "v", 0.0, rows=[ROW])  # layer 0 of row 6 is committed: a cell now
    with pytest.raises(ValueError, match="names no element"):
        run.set_cell(0, "v", 0.0, rows=[])
    with pytest.raises(ValueError, match="mix needs a dose"):
        run.set_cell(0, "v", 0.0, "mix", rows=[1])
    with pytest.raises(ValueError, match="block shape"):
        run.set_cell(0, "v", torch.zeros(1, 2, 3), rows=[1])
    assert len(run.cells) == 1


def test_mix_one_is_an_exact_copy_and_mix_half_is_one_rounded_fp64_interpolation(adapter):
    value = _value(3)
    runs = {}
    for label, op, dose in (("replace", "replace", None), ("mix1", "mix", 1.0),
                            ("mix0.5", "mix", 0.5)):
        run = Run(adapter, PROMPT)
        run.until(f"t{ROW}.L0.q")
        before = run.read_cache(1, "v").clone()
        run.set_cell(1, "v", value, op, rows=ROWS, kv=KV, d=DIMS, dose=dose)
        runs[label] = (run, before, run.read_cache(1, "v"))
    index = (0, torch.tensor(KV).view(-1, 1, 1), torch.tensor(ROWS).view(1, -1, 1),
             torch.tensor(DIMS).view(1, 1, -1))
    assert torch.equal(runs["replace"][2], runs["mix1"][2])
    run, before, after = runs["mix0.5"]
    x = before[index].double()
    hand = (x + 0.5 * (value[0].double() - x)).to(before.dtype)
    assert torch.equal(after[index], hand)
    rest = torch.ones(before.shape, dtype=torch.bool)
    rest[index] = False
    assert torch.equal(after[rest], before[rest])  # the rest stays bit-identical
    six = {label: _logits(item[0]) for label, item in runs.items()}
    assert torch.equal(six["replace"][1], six["mix1"][1])


# --- rewind, checkpoints, durability -------------------------------------------------------


def test_checkpoint_edit_step_restore_and_reverse_restage_the_cell(adapter):
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L0.q")
    clean = run.checkpoint("clean")
    run.set_cell(1, "v", 0.25, "add", rows=[1, 3])
    staged = run.checkpoint("staged")
    run.until(f"t{ROW}.L1.o")
    assert run.cells == [] and len(run.committed_cells()) == 1
    edited = run.checkpoint("edited")
    out = run.read().clone()
    run.restore(clean)
    assert run.cells == [] and run.committed_cells() == []
    assert run.read_cache(1, "v").equal(Run(adapter, PROMPT).read_cache(1, "v"))
    run.restore(staged)
    assert [c["cell"]["rows"] for c in run.cells] == [[1, 3]]
    run.restore(edited)
    assert torch.equal(run.read(), out) and run.checkpoint("edited") == edited
    # back to the boundary after the transition that realised it: still committed there
    run.reverse_step("layer")
    assert run.where == f"t{ROW}.L0.h_out" and run.cells == []
    assert len(run.committed_cells()) == 1
    # back before that transition (the cell was staged at L0.q, in it): staged again
    run.reverse_step("layer")
    assert run.where == f"t{ROW}.emb" and len(run.cells) == 1
    assert run.committed_cells() == []
    run.reverse_step("token")
    assert run.cells == []  # before it was staged
    # forward again: the same cut as the edited run
    run.until(f"t{ROW}.L0.q")
    run.set_cell(1, "v", 0.25, "add", rows=[1, 3])
    run.until(f"t{ROW}.L1.o")
    assert run.checkpoint("edited")["cut"] == edited["cut"]


def test_restore_replays_cells_from_the_root_with_retained_rows(adapter):
    run = Run(adapter, PROMPT, retain_rows=0)
    run.until(f"t{ROW}.L1.h")
    run.set_cell(0, "k", 0.5, "scale", rows=[0, ROW])
    run.until(f"t{ROW + 2}.L0.h")
    checkpoint = run.checkpoint()
    other = Run(adapter, PROMPT)  # a fresh run replays the path, cells included
    other.restore(checkpoint)
    assert other.checkpoint() == {**checkpoint, "name": None}
    assert len(other.committed_cells()) == 1
    assert torch.equal(other.read_cache(0, "k"), run.read_cache(0, "k"))


_CHILD = """
import json, sys
sys.path.insert(0, sys.argv[2])
import torch
args = json.loads(sys.argv[1])
scope = args["scope"]  # a cut binds the numeric scope: match the parent's
torch.set_num_threads(scope["torch_num_threads"])
torch.use_deterministic_algorithms(scope["deterministic_algorithms"])
torch.set_float32_matmul_precision(scope["float32_matmul_precision"])
torch.backends.cuda.matmul.allow_tf32 = scope["tf32_matmul"]
import mdb
from test_engine_ports import build_model
dbg = mdb.attach(build_model(args["family"], args["dtype"]), project=args["project"])
run = dbg.restart("after-cell")
cells = run.committed_cells()
print(json.dumps({"where": run.where, "cut": run.checkpoint()["cut"],
                  "origin": cells[0].get("origin"), "value": cells[0]["value"]}))
"""


def test_durable_checkpoint_after_a_cross_run_cell_edit_restarts_in_a_fresh_process(
        model, tmp_path):
    family, dtype = model.config.model_type, str(model.dtype).removeprefix("torch.")
    dtype = {"bfloat16": "bf16", "float32": "fp32"}[dtype]
    dbg = mdb.attach(model, project=tmp_path / "project")
    donor = dbg.run([3, 17, 9, 29, 11, 40, 8], name="donor")
    donor.until(f"t{ROW}.logits")
    run = dbg.run(PROMPT)
    value, pointer = dbg.cell_value("donor:cache.L1.v[kv1,row=2]", run=run)
    run.set_cell(1, "v", value, rows=[2], kv=[1], origin=pointer)
    run.until(f"t{ROW + 1}.L0.h")
    stored = dbg.checkpoint(run, "after-cell")
    paths = [str(Path(module.__file__).parents[1]) for module in (saturn_pub, mdb)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    scope = run.checkpoint()["numeric_scope"]
    payload = json.dumps({"family": family, "dtype": dtype, "project": str(tmp_path / "project"),
                          "scope": scope})
    result = subprocess.run([sys.executable, "-c", _CHILD, payload, str(Path(__file__).parent)],
                            env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    found = json.loads(result.stdout.strip().splitlines()[-1])
    assert found["where"] == run.where and found["cut"] == stored["cut"] == run.checkpoint()["cut"]
    assert found["origin"] == pointer and found["value"]["sha256"] == tensor_digest(value)


# --- labels ----------------------------------------------------------------------------


def test_upstream_edits_list_the_cell_only_for_values_computed_after_it(adapter):
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L1.p")
    run.set_cell(1, "v", 0.0, rows=[1])  # after the read: realised at L1's exit
    assert run.read_info()["upstream_edits"] == []  # p was computed before
    run.until(f"t{ROW}.L1.o")
    assert run.read_info()["upstream_edits"] == []  # this row's A did not read it
    run.until(f"t{ROW}.L2.q")
    ups = run.read_info()["upstream_edits"]
    assert [(u["port"], u["cell"], u["staged"]) for u in ups] == [
        ("cache.L1.v", "cache.L1.v", False)]
    assert run.read_info(f"t{ROW}.L1.o")["upstream_edits"] == []  # recomputed read: still not
    run2 = Run(adapter, PROMPT)
    run2.until(f"t{ROW}.L1.q")
    run2.set_cell(1, "k", 0.0, rows=[1], selection="row=1")
    ups = run2.read_info().get("upstream_edits")
    assert [(u["cell"], u["staged"]) for u in ups] == [("cache.L1.k[row=1]", True)]


def test_cache_cell_reads_are_labelled_by_the_cell_edits_that_wrote_them(adapter):
    from mdb.core import address as addresses
    from mdb.engine import breakpoints as bp

    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L0.q")
    head_dim = run.adapter.head_dim

    def label(text):
        cell = addresses.parse(text)
        full = run.read_cache(cell.layer, cell.tensor)
        chosen = bp.select(full, bp.CELL_AXES, cell.selector, head_dim=head_dim)
        return bp.cell_provenance(run, cell.layer, chosen, cell.tensor)

    assert label("cache.L1.v[row=2]")["authority"] == "measured"
    run.set_cell(1, "v", torch.ones(1, 2, 1, 16), rows=[2])
    assert label("cache.L1.v[row=2]")["authority"] == "supplied"
    assert label("cache.L1.v[row=3]")["authority"] == "measured"
    run.set_cell(1, "v", 0.5, "add", rows=[2], kv=[0])
    assert label("cache.L1.v[kv1,row=2]")["authority"] == "supplied"
    assert label("cache.L1.v[row=2]")["authority"] == "derived-exact"
    found = label("cache.L1.v[row=2:4]")
    assert found["authority"] == "derived-exact" and found["mixed"]["untouched"] == 32
    run.until(f"t{ROW}.L1.h")  # committed: the label holds
    assert label("cache.L1.v[kv1,row=2]")["authority"] == "supplied"


# --- interrupts -----------------------------------------------------------------------------


class _Boom(BaseException):
    """A stand-in for KeyboardInterrupt."""


def _in_sync(run):
    return run.session.capture(retain=False).fingerprint == run._trail[-1].fingerprint


def test_an_interrupt_inside_the_native_cell_write_leaves_nothing_half_done(adapter, monkeypatch):
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L0.q")
    run.set_cell(1, "k", 0.5, "add", rows=[2])
    where, cells = run.where, run.cells
    original = CellEdit.realise
    calls = {"n": 0}

    def realise(self, cache):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Boom
        return original(self, cache)

    monkeypatch.setattr(CellEdit, "realise", realise)
    with pytest.raises(_Boom):
        run.commit()
    monkeypatch.undo()
    assert _in_sync(run) and run.where == where and run.cells == cells
    assert run.committed_cells() == []
    run.until(f"t{ROW}.logits")
    twin = Run(adapter, PROMPT)
    twin.until(f"t{ROW}.L0.q")
    twin.set_cell(1, "k", 0.5, "add", rows=[2])
    twin.until(f"t{ROW}.logits")
    assert torch.equal(run.read(), twin.read()) and run.checkpoint() == twin.checkpoint()


def test_set_cell_is_atomic_when_its_probe_is_interrupted(adapter, monkeypatch):
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L1.q")
    before = run.checkpoint()

    def boom(self, controller):
        raise _Boom

    monkeypatch.setattr(Session, "probe", boom)
    with pytest.raises(_Boom):
        run.set_cell(1, "k", 0.5, rows=[2])
    monkeypatch.undo()
    assert run.cells == [] and run.checkpoint() == before


def test_an_interrupt_after_a_session_restore_keeps_committed_cells(adapter, monkeypatch):
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L0.q")
    run.set_cell(1, "v", 0.5, "scale", rows=[1])
    run.until(f"t{ROW}.L1.q")
    run.until(f"t{ROW}.L2.m")
    where, committed = run.where, run.committed_cells()
    original = Session.restore
    fired = {"n": 0}

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not fired["n"]:
            fired["n"] = 1
            raise _Boom
        return result

    monkeypatch.setattr(Session, "restore", wrapped)
    with pytest.raises(_Boom):
        run.reverse_step("layer", 3)
    monkeypatch.undo()
    assert _in_sync(run) and run.committed_cells() == committed
    assert run.where in (where, f"t{ROW}.L1.h_out")  # the first step may have completed


def test_a_measurement_interrupted_in_its_restore_still_restores(adapter, monkeypatch):
    from mdb.analysis import compare
    from mdb.analysis.forecast import Logit
    from mdb.analysis.propagate import Edit, state_digest

    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L0.q")
    before = state_digest(run)
    edit = Edit("cache.L1.v[row=1]", 0.5, "add", cell={"layer": 1, "tensor": "v", "rows": [1],
                                                       "kv": None, "d": None})
    original = Session.restore
    fired = {"n": 0}

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not fired["n"]:
            fired["n"] = 1
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(Session, "restore", wrapped)
    with pytest.raises(KeyboardInterrupt):
        compare.measure(run, [edit], Logit(ROW, 3))
    monkeypatch.undo()
    assert fired["n"] == 1 and state_digest(run) == before and _in_sync(run)
