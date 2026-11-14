"""M8: a value read under edits carries an occurrence pointer addressed ``@cut+edit...`` (the
boundary cut it is computed from and the reference of each edit applied after it) with the
stored recipe that reproduces it. ``mdb.resolve`` returns it bitwise, in this process and in a
fresh one opened on the project directory, or refuses; a tampered edit is refused, never
answered with another (or the unedited) value."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import saturn_pub  # noqa: E402

import mdb  # noqa: E402
from mdb.core import index as I  # noqa: E402
from mdb.core import recipe as recipes  # noqa: E402
from mdb.core.address import from_json  # noqa: E402
from mdb.core.resolve import resolve  # noqa: E402
from mdb.engine import Run  # noqa: E402
from mdb.engine.ports import tensor_digest  # noqa: E402
from mdb.ops import dispatch  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT, DONOR = [1, 5, 9, 3], [1, 7, 9, 3]
CASES = [("qwen2", "bf16"), ("qwen3", "fp32")]
CELL = "cache.L0.v[kv0,row=1]"


@pytest.fixture(params=CASES, ids=lambda case: "-".join(case))
def case(request, tmp_path):
    family, dtype = request.param
    root = tmp_path / "project"
    dbg = open_tiny(family, dtype, project=root)
    return {"family": family, "dtype": dtype, "root": root, "dbg": dbg, "ctx": dbg.ops()}


def _printed(ctx, address):
    shown = dispatch(ctx, "print", {"address": address})
    assert "pointer" in shown, shown.get("pointer_omitted")
    return shown["pointer"], ctx.run.read(shown["address"])


def _unedited(dbg, address, *, until):
    run = Run(dbg.adapter, PROMPT)
    run.until(until)
    return run.read(address)


def _port_edit(case):
    """A supplied tensor (``add``) and a scalar (``scale``) staged at t3.L1.m: the tensor is
    held only in memory until the pointer stores it."""
    ctx = case["ctx"]
    dispatch(ctx, "start", {"prompt": PROMPT})
    dispatch(ctx, "until", {"target": "L1.p"})
    shape = ctx.run.port_shape("m")
    ctx.run.set("L1.m", torch.linspace(-1, 1, shape[-1]).reshape(shape), "add")
    dispatch(ctx, "set", {"target": "L1.m", "value": "0.5", "op": "scale"})
    dispatch(ctx, "until", {"target": "L1.m"})  # the pending layer: its port cache holds m
    return _printed(ctx, "t3.L1.m")


def _cell_edit(case):
    """A cache cell whose value comes from another named run, read through attention at L0
    after layer 0 was committed (the edit is one of that transition's committed edits)."""
    ctx = case["ctx"]
    dispatch(ctx, "start", {"prompt": DONOR, "name": "boiling"})
    dispatch(ctx, "until", {"target": "t3.logits"})
    dispatch(ctx, "start", {"prompt": PROMPT})
    dispatch(ctx, "set", {"target": CELL, "value": f"boiling:{CELL}"})
    dispatch(ctx, "until", {"target": "L1.q"})
    return _printed(ctx, "t3.L0.o")


def _check_same_process(case, pointer, value, *, until, tier):
    origin = from_json(pointer["address"], check_ports=False).origin
    assert origin.cut == pointer["cut"] and origin.edits  # @cut+edit...
    assert pointer["recipe"] is not None and pointer["event"] is not None
    assert pointer["value"]["sha256"] == tensor_digest(value)
    assert not torch.equal(_unedited(case["dbg"], pointer_target(pointer), until=until), value)
    for tiers in (("live", "blob", "cut", "replay"), ("replay",)):
        result = resolve(pointer, session=case["dbg"], tiers=tiers)
        assert result.state == "resolved", result.reason
        assert torch.equal(result.value, value) and result.value.dtype == value.dtype
    assert resolve(pointer, session=case["dbg"]).tier == tier


def pointer_target(pointer):
    from dataclasses import replace

    from mdb.core.address import format_address

    return format_address(replace(from_json(pointer["address"], check_ports=False), origin=None))


def test_a_value_read_under_a_port_edit_resolves_bitwise(case):
    pointer, value = _port_edit(case)
    _check_same_process(case, pointer, value, until="t3.L1.m", tier="live")
    origin = from_json(pointer["address"], check_ports=False).origin
    assert len(origin.edits) == 2  # the tensor add, then the scalar scale, in order
    recipe = recipes.load(case["dbg"].project, pointer["recipe"])
    stored = recipes.tensor_pointers(recipe)  # the supplied tensor operand, now a blob
    assert len(stored) == 1 and case["dbg"].project.store.has_tensor(stored[0]["ref"])
    assert stored[0]["sha256"] in case["ctx"].run.edit_tensors()
    assert len(json.dumps(recipe)) <= 64 * 1024


def test_a_value_read_under_a_cell_edit_from_another_run_resolves_bitwise(case):
    pointer, value = _cell_edit(case)
    # layer 0 is committed: no port cache holds o, so it is recomputed
    _check_same_process(case, pointer, value, until="t3.L1.q", tier="replay")


_CHILD = """
import json, sys
import torch
args = json.loads(sys.argv[1])
scope = args["scope"]   # a cut binds the numeric scope: match the parent's before attaching
torch.set_num_threads(scope["torch_num_threads"])
torch.use_deterministic_algorithms(scope["deterministic_algorithms"])
torch.set_float32_matmul_precision(scope["float32_matmul_precision"])
torch.backends.cuda.matmul.allow_tf32 = scope["tf32_matmul"]
from mdb.core.resolve import resolve
from mdb.ops.debugger import open_tiny
pointers = json.load(open("pointers.json"))
unattached = [resolve(p, project=".").state for p in pointers]
dbg = open_tiny(args["family"], args["dtype"], project=".")
for n, pointer in enumerate(pointers):
    result = resolve(pointer, session=dbg)
    assert result.state == "resolved", result.reason
    torch.save(result.value, f"resolved-{n}.pt")
print(json.dumps({"unattached": unattached}))
"""


def test_both_resolve_bitwise_in_a_fresh_process_from_the_project_directory(case):
    values = []
    pointers = []
    for read in (_port_edit, _cell_edit):
        pointer, value = read(case)
        pointers.append(pointer)
        values.append(value)
    root = Path(case["root"])
    (root / "pointers.json").write_text(json.dumps(pointers))
    paths = [str(Path(module.__file__).parents[1]) for module in (saturn_pub, mdb)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    payload = json.dumps({"family": case["family"], "dtype": case["dtype"],
                          "scope": case["dbg"].adapter.execution["numeric_scope"]})
    result = subprocess.run([sys.executable, "-c", _CHILD, payload], env=env, cwd=root,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    child = json.loads(result.stdout.strip().splitlines()[-1])
    assert child["unattached"] == ["unresolved", "unresolved"]  # no model: nothing computed
    for n, value in enumerate(values):
        resolved = torch.load(root / f"resolved-{n}.pt")
        assert torch.equal(resolved, value) and resolved.dtype == value.dtype


def _retarget(project, pointer, change):
    """The pointer's recipe with one edit manifest changed, re-sealed and stored (a valid
    recipe of other edits), and the pointer redirected to it with no value digest (so only
    the origin's edit references can catch it)."""
    recipe = recipes.load(project, pointer["recipe"])
    entry = recipe["staged"] if recipe["target"]["transitions"] == pointer["event"] else next(
        t["edits"] for t in recipe["transitions"] if t["transition"] == pointer["event"])
    change(entry)
    forged = recipes.seal(recipe)
    sid = project.sessions()[0]["session_id"]
    recipes.save(project, sid, forged)
    value = {**pointer["value"], "sha256": None}
    return {**pointer, "recipe": forged["fingerprint"], "value": value}


def test_a_tampered_edit_manifest_is_refused(case):
    pointer, value = _port_edit(case)
    project, dbg = case["dbg"].project, case["dbg"]

    def scale_more(edits):
        edits[-1]["value"]["value"]["value"] = 0.25

    forged = _retarget(project, pointer, scale_more)
    for tiers in (("live", "blob", "cut", "replay"), ("replay",), ("blob",)):
        result = resolve(forged, session=dbg, tiers=tiers)
        assert result.state == "refused" and result.value is None
        assert "the edits differ" in result.reason
    dropped = _retarget(project, pointer, lambda edits: edits.pop())
    assert resolve(dropped, session=dbg).state == "refused"
    # an origin naming another edit reference (the recipe is untouched)
    address = json.loads(json.dumps(pointer["address"]))
    address["origin"]["edits"][0] = "0" * 64
    result = resolve({**pointer, "address": address}, session=dbg)
    assert result.state == "refused" and "the edits differ" in result.reason
    # a stored recipe row edited in place fails its seal
    with I._connect(project) as db:
        body = json.loads(db.execute("SELECT body FROM recipes WHERE id=?",
                                     (pointer["recipe"],)).fetchone()[0])
        body["staged"][-1]["op"] = "add"
        db.execute("UPDATE recipes SET body=? WHERE id=?", (json.dumps(body), pointer["recipe"]))
    result = resolve(pointer, session=dbg)
    assert result.state == "refused" and result.value is None and "seal" in result.reason
    assert value is not None


def test_without_a_project_the_pointer_names_its_edits_and_resolution_refuses(case, tmp_path):
    dbg = open_tiny(case["family"], case["dtype"])
    ctx = dbg.ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    dispatch(ctx, "until", {"target": "L1.p"})
    dispatch(ctx, "set", {"target": "L1.m", "value": "0.5", "op": "scale"})
    dispatch(ctx, "until", {"target": "L1.m"})
    pointer, _ = _printed(ctx, "t3.L1.m")
    assert pointer["recipe"] is None and from_json(pointer["address"]).origin.edits
    result = resolve(pointer, project=case["root"])
    assert result.state == "refused" and "its edits cannot be reproduced" in result.reason


def test_seek_by_an_edited_pointer_returns_to_the_edited_value(case):
    pointer, value = _port_edit(case)
    ctx = case["dbg"].ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    dispatch(ctx, "seek", {"target": json.dumps(pointer)})
    assert torch.equal(ctx.run.read("t3.L1.m"), value)
