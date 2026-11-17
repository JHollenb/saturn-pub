"""Checkpoints as content-addressed tensor pointers (#517, ``mdb.engine.checkpoint``) on tiny
Qwen2 (bf16) and Qwen3 (fp32), CPU.

Every verb on a run rehydrated from a checkpoint gives what it gives on the source run, in the
same process and in a fresh one (file and project record id); a cut inside a row continues bit
for bit at every port phase; Branches, Pool and each run from a checkpoint; checkpoints move
between the source run and a reloaded one (both ways, and across an earlier checkpoint of the
same lineage) and are refused off the lineage; durable checkpoints live in the project store
and are addressed by record id; tampering, another model and another dtype are refused (a
replica only when asked); legacy v1 core files still load."""

import copy
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import saturn_pub  # noqa: E402

import mdb  # noqa: E402
from mdb.analysis.branches import edit  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.engine import checkpoint as states  # noqa: E402
from mdb.engine import core_file  # noqa: E402
from mdb.engine.models import cast_weights, tiny_model  # noqa: E402
from mdb.engine.ports import tensor_digest  # noqa: E402
from mdb.ops.debugger import attach, open_tiny  # noqa: E402

P = [1, 5, 9, 3, 7, 2, 11]
LAST = len(P) - 1
A, B = 4, 9
CASES = [("qwen2", "bf16"), ("qwen3", "fp32")]
# the cut of the battery: inside row 4 (layer 0 committed, cursor at L1.p), after an edit at t3
CUT = "t4.L1.p"


@pytest.fixture(scope="module", params=CASES, ids=lambda c: "-".join(c))
def case(request):
    return request.param


@pytest.fixture(scope="module")
def dbg(case):
    return open_tiny(*case)


def _source(dbg, name="src"):
    run = dbg.run(P, name=name, prefill=2)
    run.until("t3.L0.h")
    run.set("t3.L0.m", 0.25, "add")
    run.until(CUT)
    return run


@pytest.fixture(scope="module")
def saved(dbg, tmp_path_factory):
    path = tmp_path_factory.mktemp("ck") / "cut.pt"
    record = _source(dbg, "saver").dump(path)
    return path, record


# ------------------------------------------------------------------ the verb battery

# record ids differ by custody (the script); a circuit's ledger number by what the debugger
# built before (the battery runs twice on one debugger)
_LABEL = re.compile(r", from checkpoint [0-9a-f]{8}…, path \w+")
_IDS = [(_LABEL, ""),       # a rehydrated run's values say where they came from (H1)
        (re.compile(r"record [0-9a-f]{16}"), "record <id>"),
        (re.compile(r"^circuit \d+ ", re.M), "circuit N "),
        (re.compile(r"\b[0-9a-f]{16,64}\b"), "<hex>")]


def _san(x):
    if isinstance(x, torch.Tensor):
        return tensor_digest(x)[:16]
    if isinstance(x, dict):
        return {str(k): _san(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_san(v) for v in x]
    if isinstance(x, str):
        for rx, rep in _IDS:
            x = rx.sub(rep, x)
        return x
    return x


def _shell(dbg, run, *lines):
    from mdb.repl.shell import Shell

    buf = io.StringIO()
    sh = Shell(dbg.ops(run), out=buf)
    for line in lines:
        buf.write(f">> {line}\n")
        sh.execute(line)
    return buf.getvalue()


def battery(dbg, make):
    """Every verb from the cut ``make(name)`` returns (``make`` builds a fresh run each time)."""
    out = {}
    run = make("v1")
    out["where"] = run.where
    out["steps"] = [run.step() for _ in range(3)]
    run.until(f"t{LAST}.logits")
    out["logits"] = run.read()
    run = make("v2")
    run.set("L1.m", 0.5, "add")
    run.until(f"t{LAST}.logits")
    out["edit"] = [run.read(), run.read("t4.L1.m")]
    run = make("v3")
    run.set("cache.L0.v[kv0,row=1]", 0.0)
    run.until(f"t{LAST}.logits")
    out["cell"] = run.read()
    out["forecast"] = _shell(dbg, make("v4"), f"until t{LAST}.L1.p", f"forecast M[{A},{B}]",
                             f"until t{LAST}.logits", "verify")
    out["whatif"] = _shell(dbg, make("v5"), f"until t{LAST}.L0.h",
                           f"whatif L1.A[h1] *= 0 --consumer M[{A},{B}]", "where")
    out["symbols"] = _shell(dbg, make("v6"), f"until t{LAST}.logits",
                            f"symbols consumer=M[{A},{B}] limit=3")
    out["reads"] = _shell(dbg, make("v7"), f"until t{LAST}.logits",
                          f"reads t5 --consumer M[{A},{B}] top=3")
    out["readout"] = _shell(dbg, make("v8"), f"until t{LAST}.logits", "print logits top=3",
                            f"explain M[{A},{B}]")
    out["circuit"] = _shell(dbg, make("v9"), f"until t{LAST}.logits",
                            f"circuit M[{A},{B}] top=3 depth=1")
    run = make("v10")
    br = dbg.branches(run, Margin(LAST, A, B), at=f"t{LAST}.L1.h")
    br.add("zero-m", [edit(f"t{LAST}.L1.m", 0.0, "scale")])
    br.add("add-m", [edit(f"t{LAST}.L1.m", 0.25, "add")])
    br.add_null("self", "self", like="zero-m")
    out["branches"] = [[r["name"], r["dM"]] for r in br.run()]
    run = make("v11")
    run.until(f"t{LAST}.L1.p")
    point = run.checkpoint("pre")
    run.set(f"t{LAST}.L1.m", 0.0, "scale")
    run.until(f"t{LAST}.logits")
    edited = run.read()
    out["restore"] = [edited, run.restore(point), (run.until(f"t{LAST}.logits"), run.read())[1]]
    base = dbg.run(P, name="base", prefill=2)
    base.until(f"t{LAST}.logits")
    run = make("v12")
    run.hold("t@last.L>0.p", source=base)
    run.until(f"t{LAST}.logits")
    out["hold"] = [run.holds, run.read()]
    dbg.runs.clear()
    return _san(out)


def test_every_verb_on_a_rehydrated_run_matches_the_source(dbg, saved):
    path, _ = saved
    source = battery(dbg, lambda name: _source(dbg, name))
    loaded = battery(dbg, lambda name: dbg.run_from(path, name=name))
    assert loaded == source
    shown = _shell(dbg, dbg.run_from(path, name="lab"), f"until t{LAST}.L0.h", "print L0.h top=1")
    assert len(_LABEL.findall(shown)) >= 1      # ... and only that differs


# ------------------------------------------------------------------ provenance (audit H1, M1)

def _reseal(path, out, mutate):
    """Rewrite a checkpoint file's document and seal a new record for it (ckaudit probe1): the
    seal gives integrity, not authenticity; anyone can do this."""
    from mdb.core.records import seal_record

    data = torch.load(path, weights_only=True)
    rec = json.loads(data["record"])
    doc = json.loads(data["document"])
    mutate(doc)
    text = states._canonical(doc)
    body = dict(rec["body"])
    body["document"] = {"state": "file", "bytes": len(text.encode()),
                        "sha256": states._sha_text(text)}
    body["recipe"] = dict(body["recipe"], sha256=states._sha_text(states._canonical(
        doc["recipe"])), path_edits=len(doc["recipe"]["path_edits"]),
        staged=len(doc["recipe"]["edits"]))
    body["edited"] = bool(doc["recipe"]["path_edits"] or doc["recipe"]["edits"])
    body["lineage"] = doc["lineage"]
    new = seal_record(session_id=rec["session_id"], kind=rec["kind"],
                      model_identity=rec["model_identity"], execution=rec["execution"],
                      authority=rec["authority"], body=body)
    data["record"] = json.dumps(new)
    data["document"] = text
    torch.save(data, out)
    return new


def _strip(doc):
    for m in doc["ancestry"] + doc["marks"]:
        m["edits"], m["cells"], m["fired"], m["cell_fired"] = [], [], [], []
    doc["ancestry"] = []
    doc["recipe"]["path_edits"] = []
    doc["edit_tensors"] = {}


def test_a_stripped_and_resealed_history_is_labelled_unverified_until_verify(dbg, tmp_path):
    """Audit H1 (ckaudit probe1 A): a file whose edit history was stripped and resealed loads
    (the state is genuine), so every value and record of a run rooted at it says ``from
    checkpoint <id>, path unverified``; ``fresh`` inherits it; ``verify`` raises on the forgery
    and clears the label on the honest file."""
    src = _source(dbg, "h1")
    honest = src.dump(tmp_path / "a.pt", history="row")
    forged = _reseal(tmp_path / "a.pt", tmp_path / "forged.pt", _strip)
    assert forged["id"] != honest["id"]              # the id binds edits and lineage
    run = dbg.run_from(tmp_path / "forged.pt", name="h1f")
    assert run._trail[-1].fingerprint == src._trail[-1].fingerprint and not run.edited
    assert run.from_checkpoint == {"checkpoint": forged["id"], "path": "unverified"}
    assert run.fresh().from_checkpoint == run.from_checkpoint
    shown = _shell(dbg, run, f"until t{LAST}.L0.h", "print L0.h top=1")
    assert f"[measured, from checkpoint {forged['id'][:8]}…, path unverified]" in shown
    # records sealed about the run carry it (custody and procedure)
    from mdb.core import custody

    sink = custody.records_to(tmp_path / "records.jsonl")
    try:
        _shell(dbg, dbg.run_from(tmp_path / "forged.pt", name="h1g"), f"until t{LAST}.L1.p",
               f"forecast M[{A},{B}]")
    finally:
        sink.close()
    sealed = [r for r in sink.records() if r["kind"] == "forecast"]
    assert sealed and all(r["body"]["custody"]["from_checkpoint"]["path"] == "unverified"
                          and r["body"]["procedure"]["from_checkpoint"]["checkpoint"] ==
                          forged["id"] for r in sealed)
    with pytest.raises(states.CheckpointUnverified, match="did not verify"):
        run.verify()
    assert run.from_checkpoint["path"] == "unverified"
    with pytest.raises(states.CheckpointUnverified):
        dbg.run_from(tmp_path / "forged.pt", verify=True)
    ok = dbg.run_from(tmp_path / "a.pt", name="h1h")
    assert ok.verify()["ok"] and ok.from_checkpoint["path"] == "verified"
    assert ok.fresh().from_checkpoint["path"] == "verified"
    shown = _shell(dbg, ok, f"until t{LAST}.L0.h", "print L0.h top=1")
    assert "path verified]" in shown
    assert dbg.run_from(tmp_path / "a.pt", verify=True).from_checkpoint["path"] == "verified"


def test_verify_raises_unless_report_and_a_foreign_root_needs_root(dbg, tmp_path):
    """Audit M1: ``verify=True`` raises on failure (``run_from``, ``Checkpoint.verify``), a
    lineage whose root is another checkpoint raises unless ``root=`` is given, and the soft
    result is opt-in (``verify="report"``)."""
    src = _source(dbg, "m1")
    _reseal(src.dump(tmp_path / "a.pt") and tmp_path / "a.pt", tmp_path / "f.pt", _strip)
    report = dbg.run_from(tmp_path / "f.pt", verify="report").core["verified"]
    assert report["ok"] is False and report["state"] == "resolved"
    other = attach(tiny_model(*_case_of(dbg), seed=7))       # same shape, other weights
    with pytest.raises(states.CheckpointUnverified, match="source's model"):
        other.load_checkpoint(tmp_path / "a.pt").verify(other.adapter)
    replica = other.run_from(tmp_path / "a.pt", replica=True, name="m1r")
    replica.until(f"t{LAST}.L0.h")
    replica.dump(tmp_path / "b.pt")                     # rooted at a checkpoint, not a prompt
    second = states.load_file(tmp_path / "b.pt")
    assert second.lineage["root"] == "checkpoint"
    with pytest.raises(states.CheckpointUnverified, match="pass root="):
        second.verify(other.adapter)
    with pytest.raises(states.CheckpointUnverified, match="pass root="):
        other.run_from(tmp_path / "b.pt", verify=True)
    assert second.verify(other.adapter, report=True)["state"] == "unresolved"
    assert second.verify(other.adapter, root=replica)["ok"]
    run = other.run_from(tmp_path / "b.pt", verify=True, root=replica)
    assert run.from_checkpoint["path"] == "verified"


def _case_of(dbg):
    return ("qwen2", "bf16") if "bfloat16" in str(dbg.adapter.dtype) else ("qwen3", "fp32")


_CHILD = """
import json, sys
import torch
args = json.loads(sys.argv[1])
scope = args["scope"]   # a cut binds the numeric scope: match the parent's
torch.set_num_threads(scope["torch_num_threads"])
torch.use_deterministic_algorithms(scope["deterministic_algorithms"])
torch.set_float32_matmul_precision(scope["float32_matmul_precision"])
sys.path.insert(0, args["tests"])
import test_engine_checkpoint as T
from mdb.engine.models import tiny_model
import mdb
# the battery runs on a debugger with no project (a project stores records and blobs, which
# changes the printed lines); the record id is resolved through the project's debugger
dbg = mdb.attach(tiny_model(*args["case"]))
durable = mdb.attach(tiny_model(*args["case"]), project=args["project"])
out = {"file": T.battery(dbg, lambda name: dbg.run_from(args["path"], name=name))}
by_id = durable.load_checkpoint(args["record"][:12])
out["record"] = T.battery(dbg, lambda name: dbg.run_from(by_id, name=name))
run = durable.restart("cut")
out["restart"] = [run.where, run.core["exact"]]
print(json.dumps(out))
"""


def test_every_verb_in_a_fresh_process(dbg, case, saved, tmp_path):
    path, _ = saved
    project = tmp_path / "proj"
    pdbg = attach(tiny_model(*case), project=project)
    info = pdbg.checkpoint(_source(pdbg, "p"), "cut")
    source = battery(dbg, lambda name: _source(dbg, name))
    paths = [str(Path(m.__file__).parents[1]) for m in (saturn_pub, mdb)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    payload = json.dumps({"case": list(case), "path": str(path), "project": str(project),
                          "record": info["state"], "tests": str(Path(__file__).parent),
                          "scope": dbg.adapter.execution["numeric_scope"]})
    result = subprocess.run([sys.executable, "-c", _CHILD, payload], env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    child = json.loads(result.stdout.strip().splitlines()[-1])
    assert child["file"] == source
    assert child["record"] == source
    assert child["restart"][:2] == [CUT, True]


# ------------------------------------------------------------------ cuts inside a row

STOPS = ["t4.start", "t4.emb", "t4.L0.p", "t4.L0.m", "t4.L0.h_out", "t4.L1.sg", "t4.fn",
         "t4.logits", "t4.sample"]


@pytest.mark.parametrize("stop", STOPS)
def test_a_cut_at_every_port_phase_continues_bit_for_bit(dbg, stop, tmp_path):
    src = _source(dbg, "phase")
    if src.parse(stop) < src.parse(src.where):
        src.rewind(stop)
    else:
        src.until(stop)
    if src.pending_layer is not None and src._port is not None:
        src.set(f"L{src.pending_layer}.h_out", 0.125, "add")       # staged at the cursor
    record = src.dump(tmp_path / "c.pt")
    assert record["body"]["where"] == src.where
    run = dbg.run_from(tmp_path / "c.pt", name="phase-r")
    assert run.where == src.where and run.edits == src.edits
    assert run._trail[-1].fingerprint == src._trail[-1].fingerprint
    src.until(f"t{LAST}.logits")
    run.until(f"t{LAST}.logits")
    assert torch.equal(run.read(), src.read())


def test_a_staged_cell_edit_at_the_cursor_is_kept(dbg, tmp_path):
    src = _source(dbg, "cellsrc")
    src.set("cache.L1.k[kv0,row=2]", 0.0)
    src.dump(tmp_path / "cell.pt")
    run = dbg.run_from(tmp_path / "cell.pt", name="cell-r")
    assert run.cells == src.cells
    src.until(f"t{LAST}.logits")
    run.until(f"t{LAST}.logits")
    assert torch.equal(run.read(), src.read())


def test_a_cut_before_an_edit_is_refused_never_truncated(dbg, tmp_path):
    src = _source(dbg, "trunc")            # edit at t3.L0, cursor in t4
    with pytest.raises(ValueError, match="after t3.start"):
        src.dump(tmp_path / "x.pt", row=3)
    src.dump(tmp_path / "x.pt", row=4)      # no edit after t4.start: fine
    src.set("L1.m", 1.0, "add")
    with pytest.raises(ValueError, match="staged"):
        src.dump(tmp_path / "y.pt", row=4)


def test_history_sets_how_far_back_the_reloaded_run_rewinds(dbg, tmp_path):
    src = _source(dbg, "hist")
    for history, earliest in (("cursor", "t4.L0.h_out"), ("row", "t4.start"),
                              ("root", "t2.start")):
        src.dump(tmp_path / f"{history}.pt", history=history)
        run = dbg.run_from(tmp_path / f"{history}.pt", name=f"h-{history}")
        assert run.root == earliest
        with pytest.raises(ValueError, match="rehydrated from a checkpoint"):
            run.rewind(f"t{run.first_row - 1}.L0.m" if run.first_row > 0 else "t0.start")
    run = dbg.run_from(tmp_path / "root.pt", name="h-root2")
    run.rewind("t2.L0.m")                   # before the t3 edit: the source's own trail
    clean = dbg.run(P, name="clean", prefill=2)
    clean.until("t3.L0.h")
    assert torch.equal(run.read("t2.L0.m"), clean.read("t2.L0.m"))
    run.until("t3.L0.h")
    run.set("t3.L0.m", 0.25, "add")
    run.until(f"t{LAST}.logits")
    src.until(f"t{LAST}.logits")
    assert torch.equal(run.read(), src.read())


# ------------------------------------------------------------------ branches, pool, each

def test_branches_pool_and_each_from_a_checkpoint(dbg, saved):
    path, record = saved
    src, run = _source(dbg, "bp-src"), dbg.run_from(path, name="bp-r")
    rows = []
    for r in (src, run):
        br = dbg.branches(r, Margin(LAST, A, B), at=f"t{LAST}.L1.h")
        br.add("zero", [edit(f"t{LAST}.L1.m", 0.0, "scale")])
        br.add_null("rnd", "random", like="zero", seed=0)
        rows.append([[x["name"], x["dM"], x.get("effect_floors")] for x in br.run()])
        assert torch.equal(br.reference.read(f"t{LAST}.logits"), br.base_logits)
    assert rows[0] == rows[1]
    assert br.reference.lineage["root"] == "prompt" and br.reference._origin is not None
    # pool: a port and a cache cell, from the file and from the loaded object
    src.until(f"t{LAST}.L1.h")
    pool = dbg.pool([{"id": "a", "core": str(path)},
                     {"id": "b", "from": dbg.load_checkpoint(path)}], f"t{LAST}.L1.h")
    assert torch.equal(pool.values[0], src.read(f"t{LAST}.L1.h"))
    assert torch.equal(pool.values[0], pool.values[1])
    assert pool.items[0]["origin"]["checkpoint"] == record["id"]
    cells = dbg.pool([{"id": "a", "core": str(path)}], "cache.L1.v[kv0,row=3]")
    assert torch.equal(cells.values[0], src.read_cache(1, "v")[:, :1, 3:4])
    # each: the item key (core / from) and the Python argument agree with a start of the prompt
    line = f"whatif L1.A[h1] *= 0 --consumer M[{A},{B}]"
    by_key = dbg.each([{"name": "x", "core": str(path)}], line).results
    by_arg = dbg.each([{"name": "x"}], line, core=str(path)).results
    plain = dbg.each([{"name": "x", "tokens": P}], line, prefill=2).results
    assert _san(by_key) == _san(by_arg)
    assert _san(by_key)["x"]["prediction"] != _san(plain)["x"]["prediction"]  # the t3 edit
    with pytest.raises(Exception, match="core or from"):
        dbg.each([{"name": "x", "core": str(path), "from": str(path)}], line)


def test_a_set_level_core_is_only_for_items_without_their_own_prompt(dbg, saved):
    """Audit M3 (ckaudit probe3): ``each(core=X)`` used to rehydrate an item that brought its own
    tokens (and refused one whose tokens differed). It applies only to items with no prompt or
    tokens; a pair donor is never rehydrated from the set's core."""
    from mdb.ops.sets import Bound, _item

    path, _ = saved
    line = f"whatif L1.A[h1] *= 0 --consumer M[{A},{B}]"
    plain = dbg.each([{"name": "x", "tokens": P}], line, prefill=2).results
    own = dbg.each([{"name": "x", "tokens": P}], line, prefill=2, core=str(path)).results
    assert _san(own) == _san(plain)
    other = [1, 2, 3, 4, 5, 6, 7]
    alone = dbg.each([{"name": "x", "tokens": other}], line).results
    mixed = dbg.each([{"name": "x", "tokens": other}], line, core=str(path))
    assert _san(mixed.results) == _san(alone)
    assert Bound(dbg, _item({"name": "y", "tokens": P}, "y"), core=str(path)).core is None
    assert Bound(dbg, _item({"name": "y", "core": str(path)}, "y")).core == str(path)


# ------------------------------------------------------------------ lineage

def test_checkpoints_move_between_source_and_reloaded_runs(dbg, saved, tmp_path):
    path, _ = saved
    src, run = _source(dbg, "lin-src"), dbg.run_from(path, name="lin-r")
    assert run._path_manifest(0) == src._path_manifest(len(src._trail) - 1)[:1]
    src.until(f"t{LAST}.L0.p")
    src.set("L0.u", 0.5, "add")
    later = src.checkpoint("later")
    run.restore(later)                                        # source -> reloaded
    assert run.where == src.where and run.edits == src.edits
    run.until(f"t{LAST}.logits")
    mine = run.checkpoint("mine")
    src.rewind("t5.L0.h")
    src.restore(mine)                                         # reloaded -> source
    assert torch.equal(src.read(), run.read())
    # a run reloaded from an EARLIER checkpoint of the same lineage restores the later one
    early = _source(dbg, "lin-early")
    early.rewind("t4.start")
    early.dump(tmp_path / "early.pt", history="cursor")
    from_early = dbg.run_from(tmp_path / "early.pt", name="lin-e")
    from_early.restore(mine)
    assert torch.equal(from_early.read(), run.read())
    # refused: before the root, and another path before the root (not this lineage)
    before = _source(dbg, "lin-y")
    before.rewind("t3.L0.h")
    with pytest.raises(ValueError, match="before this run's root"):
        from_early.restore(before.checkpoint())
    other = dbg.run(P, name="lin-o", prefill=2)
    other.until("t5.start")
    with pytest.raises(ValueError, match="another path before this run's root"):
        from_early.restore(other.checkpoint())


def test_a_fingerprint_binds_content_and_ancestry(dbg, saved, tmp_path):
    path, _ = saved
    src = _source(dbg, "fp")
    run = dbg.run_from(path, name="fp-r")
    assert [m.fingerprint for m in run._trail] == [
        m.fingerprint for m in src._trail[-len(run._trail):]]
    verified = dbg.load_checkpoint(path).verify(dbg.adapter)
    assert verified["ok"] and verified["reached"] == src._trail[-1].fingerprint
    # a forged parent (same tensors, other claimed ancestry) is refused at rehydration
    data = torch.load(path, weights_only=True)
    doc = json.loads(data["document"])
    doc["marks"][-1]["cut"]["parent"] = "0" * 64
    data["document"] = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    torch.save(data, tmp_path / "forged.pt")
    with pytest.raises(ValueError, match="document differs"):
        states.load_file(tmp_path / "forged.pt")
    loaded = states.load_file(path)
    loaded.doc["marks"][-1]["cut"]["parent"] = "0" * 64       # past the seal, in memory
    with pytest.raises(ValueError, match="broken chain"):    # L1: the links agree
        loaded.run(dbg.adapter)
    loaded = states.load_file(path)
    loaded.doc["marks"][0]["cut"]["parent"] = "0" * 64        # the root: no link to check
    with pytest.raises(ValueError, match="differs from its fingerprint"):
        loaded.run(dbg.adapter)


# ------------------------------------------------------------------ project store

def test_project_store_addressing_and_durable_checkpoints(case, tmp_path, monkeypatch):
    root = tmp_path / "proj"
    dbg = attach(tiny_model(*case), project=root)
    src = _source(dbg)
    info = dbg.checkpoint(src, "cut")
    durable = src.checkpoint("again", durable=dbg.project)["durable"]
    src.until(f"t{LAST}.logits")
    want = src.read()
    other = attach(tiny_model(*case), project=root)            # another debugger, same project
    for ref in (info["state"], info["state"][:10], durable["record"]):
        run = other.run_from(ref, name="pr")
        assert run.where == CUT and run.core["record"].startswith(ref[:10])
        run.until(f"t{LAST}.logits")
        assert torch.equal(run.read(), want)
    restarted = other.restart("cut", verify=True)
    assert restarted.core["exact"] and restarted.core["verified"]["ok"]
    replayed = other.restart("cut", replay=True)
    assert replayed._trail[-1].fingerprint == restarted._trail[-1].fingerprint
    body = states.find_record(other.project, info["state"])["body"]
    shas = states._unpacked(body["tensors"]["sha256"])
    assert len(shas) == body["tensors"]["count"] > 0
    assert all(other.project.store.has_tensor(p["ref"])
               for p in states.load(info["state"], project=other.project).doc["parts"])
    with pytest.raises(Exception, match="record id"):
        other.run_from("zz-not-an-id")
    with pytest.raises(Exception, match="ambiguous|no checkpoint record"):
        other.run_from("0000000000")
    # owner decision: no whole native cut beside the state (K/V is stored once); a value from
    # the checkpoint by name rehydrates the state
    assert other.project.cut_ids(dbg.session_id) == []
    here = other.run(P, name="here", prefill=2)
    _, pointer = other.cell_value("cut:cache.L0.v[kv0,row=1]", run=here)
    assert pointer["via"] == "checkpoint"
    # L6: a file named like the record id prefix is refused as ambiguous, not silently read
    monkeypatch.chdir(tmp_path)
    (tmp_path / info["state"][:10]).write_bytes(b"x")
    with pytest.raises(Exception, match="names both a file and checkpoint record"):
        other.run_from(info["state"][:10])
    # M1: restart(verify=True) raises when the path does not verify
    monkeypatch.setattr(states.Checkpoint, "_verify", lambda self, adapter, root: {
        "ok": False, "state": "resolved", "checkpoint": self.id, "expected": "a" * 64,
        "reached": "b" * 64})
    with pytest.raises(states.CheckpointUnverified, match="did not verify"):
        other.restart("cut", verify=True)
    assert other.restart("cut", verify="report").core["verified"]["ok"] is False


# ------------------------------------------------------------------ refusals

def test_tamper_model_and_dtype_refusals(case, dbg, saved, tmp_path):
    path, _ = saved
    data = torch.load(path, weights_only=True)
    key = sorted(data["tensors"])[0]
    data["tensors"][key] = data["tensors"][key].clone()
    data["tensors"][key].view(-1)[0] += 1
    torch.save(data, tmp_path / "t.pt")
    with pytest.raises(ValueError, match="differs from its sha256"):
        states.load_file(tmp_path / "t.pt")
    other = attach(tiny_model(case[0], case[1], seed=7))       # same shape, other weights
    with pytest.raises(ValueError, match="another model"):
        other.run_from(path)
    replica = other.run_from(path, replica=True)
    assert replica.core["replica"] and replica.lineage["root"] == "checkpoint"
    with pytest.raises(ValueError, match="another model shape"):
        open_tiny("llama", "fp32").run_from(path)
    model = tiny_model(case[0], case[1])
    wide = attach(cast_weights(copy.deepcopy(model), torch.float64
                               if case[1] == "fp32" else torch.float32))
    with pytest.raises(ValueError, match="another (model|execution)"):   # the identity binds
        wide.run_from(path)                                            # the dtype too
    run = wide.run_from(path, replica=True)
    assert run.core["replica"] and run.where == CUT
    # L2: nothing captured -> the result and the label say there was no null check
    assert run.core_check["ok"] is None and "no null check" in run.core_check["reason"]
    assert run.from_checkpoint["replica"] and run.from_checkpoint["null_check"] is None
    # saved at an earlier boundary after running on, the logits the source computed are captured
    # and a replica is null-checked against them
    src = _source(dbg, "cap")
    src.until(f"t{LAST}.logits")
    record = src.dump(tmp_path / "cap.pt", at="t4.start")
    assert record["body"]["captured"] == [f"t{LAST}.logits"]
    check = wide.run_from(tmp_path / "cap.pt", replica=True).core_check
    assert check["same_dtype_and_identity"] is False and "argmax_equal" in check["checks"][
        f"t{LAST}.logits"]
    exact = dbg.run_from(tmp_path / "cap.pt").core_check
    assert exact["ok"] and exact["checks"][f"t{LAST}.logits"]["max_abs_diff"] == 0.0
    # a store blob changed on disk is refused when the checkpoint is used
    root = tmp_path / "proj"
    pdbg = attach(tiny_model(*case), project=root)
    info = pdbg.checkpoint(_source(pdbg, "tp"), "cut")
    part = states.load(info["state"], project=pdbg.project).doc["parts"][0]
    blob = root / "store" / "tensors" / f"{part['ref']}.safetensors"
    blob.write_bytes(blob.read_bytes()[:-1] + b"\1")
    with pytest.raises(ValueError, match="corrupt|differs"):
        attach(tiny_model(*case), project=root).run_from(info["state"])


def test_legacy_v1_core_files_still_load(dbg, tmp_path):
    src = dbg.run(P, name="v1src", prefill=2)
    src.until(f"t{LAST}.logits")
    core_file.dump_v1(src, tmp_path / "v1.pt", row=4)
    loaded = core_file.load(tmp_path / "v1.pt")
    assert isinstance(loaded, core_file.CoreFile)
    run = dbg.run_from(tmp_path / "v1.pt", name="v1")
    assert run.where == "t4.start" and run.core_check["ok"]
    assert run.lineage["root"] == "checkpoint"                # no parent recorded: foreign
    run.until(f"t{LAST}.logits")
    assert torch.equal(run.read(), src.read())
    br = dbg.branches(run, Margin(LAST, A, B), at=f"t{LAST}.L1.h")   # fresh() from the v1 root
    br.add("zero", [edit(f"t{LAST}.L1.m", 0.0, "scale")])
    assert len(br.run()) == 1


def test_an_active_hold_is_rearmed_and_continues_bit_for_bit(dbg, tmp_path):
    """Audit M2 (ckaudit probe1 C): a run holding ``t@last.L>0.p`` at a clean source, saved and
    reloaded, continued 0.0182 away from the source (the hold was not re-armed). The checkpoint
    keeps each hold with the source values it reads from the cut's row on: the reloaded run
    holds the same ports, fires the same hold edits and continues bit for bit; ``holds="drop"``
    is recorded and continues free."""
    base = dbg.run(P, name="m2base", prefill=2)
    base.until(f"t{LAST}.logits")
    src = _source(dbg, "m2")
    src.hold("t@last.L>0.p", source=base)
    record = src.dump(tmp_path / "h.pt")
    assert record["body"]["holds"] == [{"name": "hold1", "address": "t@last.L>0.p",
                                        "dropped": False}]
    run = dbg.run_from(tmp_path / "h.pt", name="m2r")
    assert run.holds == src.holds and run.core["holds"] == ["hold1"]
    for r in (src, run):
        r.until(f"t{LAST}.logits")
    assert torch.equal(run.read(), src.read())
    assert run.checkpoint()["cut"] == src.checkpoint()["cut"]
    assert [e["origin"] for e in run.committed_edits() if e.get("origin", {}).get("kind") ==
            "hold"] == [e["origin"] for e in src.committed_edits()
                        if e.get("origin", {}).get("kind") == "hold"]
    # a norm-scale hold too
    src = _source(dbg, "m2n")
    src.hold("t@last.L*.norm_scale", source=base)
    src.dump(tmp_path / "n.pt")
    run = dbg.run_from(tmp_path / "n.pt", name="m2nr")
    for r in (src, run):
        r.until(f"t{LAST}.logits")
    assert torch.equal(run.read(), src.read())
    # dropped: recorded, and the reloaded run continues free (as a clean-hold-free source)
    src = _source(dbg, "m2d")
    free = _source(dbg, "m2f")
    src.hold("t@last.L>0.p", source=base)
    assert src.dump(tmp_path / "d.pt", holds="drop")["body"]["holds"][0]["dropped"]
    run = dbg.run_from(tmp_path / "d.pt", name="m2dr")
    assert run.holds == [] and run.core["holds_dropped"] == ["hold1"]
    for r in (free, run):
        r.until(f"t{LAST}.logits")
    assert torch.equal(run.read(), free.read())
    with pytest.raises(ValueError, match="holds='drop'"):
        src2 = _source(dbg, "m2e")
        src2.hold("t@last.L>0.p", source=base)
        src2.dump(tmp_path / "e.pt", at="t4.start")


def test_the_repl_starts_from_a_checkpoint(dbg, saved, tmp_path):
    """Owner decision: ``start from=ID|PATH`` (REPL) rehydrates a checkpoint and shows where it
    came from; ``verify=true`` replays it first and labels the path verified."""
    from mdb.repl.shell import Shell

    path, record = saved
    src = _source(dbg, "repl-src")
    src.until(f"t{LAST}.L0.h")
    buf = io.StringIO()
    sh = Shell(dbg.ops(None), out=buf)
    for line in (f"start from={path} name=ck", f"until t{LAST}.L0.h", "print L0.h top=1"):
        sh.execute(line)
    out = buf.getvalue()
    assert f"from checkpoint {record['id'][:8]}…, path unverified (record {record['id'][:16]}" \
        in out and "[measured, under edit t3.L0.m, from checkpoint" in out
    assert torch.equal(dbg.runs["ck"].read(f"t{LAST}.L0.h"), src.read(f"t{LAST}.L0.h"))
    buf.truncate(0)
    sh.execute(f"start from={path} verify=true")
    assert "path verified" in buf.getvalue() and "reaches the checkpoint [measured]" in \
        buf.getvalue()
    buf.truncate(0)
    sh.execute(f"start from={path} 1 2 3")
    assert "give no prompt" in buf.getvalue()


def test_records_sealed_from_python_on_a_checkpoint_run_carry_custody(dbg, saved, tmp_path):
    """Records sealed through the Python API (no op dispatch) about a run rooted at a
    checkpoint carry ``custody.from_checkpoint`` (the sealer names the run): Branches.seal,
    each, Pool, forecast, whatif, trace, sweep; a run of the prompt adds nothing."""
    from mdb.analysis import forecast
    from mdb.analysis.propagate import Edit, whatif
    from mdb.analysis.rows import row_table
    from mdb.core import custody
    from mdb.ops.sweep import sweep

    path, record = saved
    want = {"checkpoint": record["id"], "path": "unverified"}
    sink = custody.records_to(tmp_path / "r.jsonl")
    try:
        run = dbg.run_from(path, name="cus")
        run.until(f"t{LAST}.L1.p")
        forecast.seal(run, Margin(LAST, A, B))
        whatif(run, Edit(f"t{LAST}.L1.m", 0.5, "add"), Margin(LAST, A, B))
        run.until(f"t{LAST}.logits")
        row_table(run, {"a": [0, 1, 2], "b": [3, 4, 5]}, letters=[A, B], X="A",
                  R="B").seal(session_id="s1", model_identity=run.adapter.model_identity,
                              execution=dict(run.adapter.execution))   # no run= given
        br = dbg.branches(dbg.run_from(path, name="cus-b"), Margin(LAST, A, B),
                          at=f"t{LAST}.L1.h")
        br.add("zero", [edit(f"t{LAST}.L1.m", 0.0, "scale")])
        br.run()
        br.seal()
        dbg.pool([{"id": "a", "core": str(path)}], f"t{LAST}.L1.h").seal()
        out = dbg.each([{"name": "x"}], f"whatif L1.A[h1] *= 0 --consumer M[{A},{B}]",
                       core=str(path))
        assert out.summary["items"][0]["state"] == "resolved", out.summary
        dbg.runs["clean"] = dbg.run(P, name="clean", prefill=2)
        dbg.runs["clean"].until(f"t{LAST}.logits")
        sw = dbg.run_from(path, name="cus-s")
        sw.until(f"t{LAST}.logits")
        sweep(dbg.ops(sw), "h", "5", "0:2", "clean", f"M[{A},{B}]", 3)
        plain = dbg.run(P, name="plain", prefill=2)
        plain.until(f"t{LAST}.L1.p")
        forecast.seal(plain, Margin(LAST, A, B))
    finally:
        sink.close()
    sealed = list(sink.records())
    by_kind = {}
    for r in sealed:
        by_kind.setdefault(r["kind"], []).append(r["body"].get("custody", {}).get(
            "from_checkpoint"))
    for kind in ("forecast", "whatif", "trace", "branches", "sweep"):
        assert want in by_kind[kind], (kind, by_kind[kind])
    assert by_kind["pool"] == [want]
    assert by_kind["whatif"].count(want) == 2             # whatif by Python and by each
    assert by_kind["forecast"] == [want, None]            # the prompt's run: nothing added
