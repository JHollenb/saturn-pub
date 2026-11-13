import json
import random
import subprocess
import sys

import pytest

from mdb.core.pointers import check_bounded
from mdb.ops import (
    INTEGER_KEYS,
    BadArgs,
    Context,
    Refused,
    UnknownOp,
    describe,
    dispatch,
    registry,
)

SLOTS = {"forecast", "verify", "whatif", "explain", "reads", "symbols", "circuit", "compare",
         "seek", "export"}
GDB_VERBS = {"run", "start", "where", "frame", "step", "next", "finish", "until", "advance",
             "continue", "break", "tbreak", "delete", "watch", "display", "undisplay", "print",
             "x", "info", "list", "backtrace", "up", "down", "set", "commit", "abort",
             "checkpoint", "restart", "reverse-step", "reverse-next", "reverse-finish",
             "reverse-continue", "checkpoints"}


def test_catalogue_has_every_verb_with_a_schema_and_round_trips_json():
    ops = registry()
    assert GDB_VERBS | SLOTS <= set(ops)
    catalogue = describe()
    assert json.loads(json.dumps(catalogue)) == catalogue
    check_bounded(catalogue, "catalogue")
    for item in catalogue["ops"]:
        assert set(item) >= {"name", "summary", "effect", "needs_live", "args"}
        assert item["effect"] in ("read", "navigate", "mutate", "native")
        for arg in item["args"]:
            assert set(arg) >= {"name", "type", "required"}
    slots = {item["name"]: item for item in catalogue["ops"] if "refused_until" in item}
    assert set(slots) == SLOTS


def test_unknown_bad_args_and_refusals_without_a_model():
    ctx = Context()
    with pytest.raises(UnknownOp):
        dispatch(ctx, "frobnicate", {})
    with pytest.raises(BadArgs, match="unknown argument"):
        dispatch(ctx, "step", {"count": 2})
    with pytest.raises(BadArgs, match="integer"):
        dispatch(ctx, "step", {"n": "2"})
    with pytest.raises(BadArgs, match=">= 1"):
        dispatch(ctx, "step", {"n": 0})
    with pytest.raises(BadArgs, match="one of"):
        dispatch(ctx, "step", {"unit": "row"})
    with pytest.raises(BadArgs, match="missing argument"):
        dispatch(ctx, "until", {})
    with pytest.raises(BadArgs, match="JSON object"):
        dispatch(ctx, "where", [1])
    with pytest.raises(Refused, match="needs a live run"):
        dispatch(ctx, "where", {})
    with pytest.raises(Refused, match="no model attached"):
        dispatch(ctx, "start", {"prompt": [1, 2]})
    for name in SLOTS:
        with pytest.raises(Refused, match="not built yet: (A5|A6|B1)"):
            dispatch(ctx, name, {"anything": 1})
    assert dispatch(ctx, "checkpoints", {}) == {
        "op": "checkpoints", "checkpoints": [], "total": 0, "offset": 0, "limit": 32,
        "state": "resolved"}


def test_registry_imports_and_describes_without_torch():
    code = (
        "import sys, json, mdb, mdb.cli, mdb.ops, mdb.repl\n"
        "json.dumps(mdb.ops.describe())\n"
        "assert 'torch' not in sys.modules, 'torch loaded'\n"
        "assert 'transformers' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "ok"


# --------------------------------------------------------------------------- live fuzz

def _unlabelled(node, path="", key=None, labelled=False):
    """Numeric leaves that are neither inside an authority-labelled mapping nor integers under
    an allowlisted key."""
    bad = []
    if isinstance(node, dict):
        has = "authority" in node
        for k, value in node.items():
            bad += _unlabelled(value, f"{path}.{k}", k, has)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            bad += _unlabelled(value, f"{path}[{i}]", key, labelled)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        if labelled:
            return []
        if type(node) is int and key in INTEGER_KEYS:
            return []
        bad.append(path)
    return bad


def test_unlabelled_walker_catches_bare_numbers():
    assert _unlabelled({"stats": {"norm": 1.5}}) == [".stats.norm"]
    assert _unlabelled({"norm": {"value": 1.5, "authority": "measured"}}) == []
    assert _unlabelled({"shape": [1, 2]}) == []
    assert _unlabelled({"shape": [1.5]}) == [".shape[0]"]


@pytest.fixture(scope="module", params=[("qwen2", "bf16"), ("qwen3", "fp32")],
                ids=lambda c: "-".join(c))
def debugger(request):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mdb.ops.debugger import open_tiny

    return open_tiny(*request.param)


def _fuzz_args(ctx, name):
    run = ctx.run
    row, index = run.parse(run.where)
    stops = run.stops()
    ahead = f"t{row}.{stops[index + 1]}" if index + 1 < len(stops) else f"t{row + 1}.start"
    return {
        "where": {}, "frame": {"address": run.where}, "step": {"n": 1}, "next": {},
        "finish": {}, "until": {"target": ahead}, "advance": {"target": ahead},
        "continue": {"budget": 1}, "print": {"top": 3, "slice": 4}, "x": {"count": 8},
        "info": {"subject": "edits"}, "list": {}, "backtrace": {}, "up": {"n": 1},
        "down": {"n": 1}, "checkpoints": {}, "restart": {"name": "fz"},
        "reverse-step": {"n": 1}, "reverse-next": {}, "reverse-finish": {},
        "reverse-continue": {},
    }[name]


def test_fuzz_read_and_navigate_ops_return_bounded_labelled_results(debugger):
    ctx = debugger.ops()
    rng = random.Random(1234)
    names = sorted(name for name, item in registry().items()
                   if item.effect in ("read", "navigate") and item.milestone is None)
    dispatch(ctx, "break", {"pattern": "L*.p"})
    dispatch(ctx, "watch", {"address": "L*.h[d=0]"})
    stops = None
    seen_numbers = 0
    for position in range(20):
        dispatch(ctx, "start", {"prompt": [1, 5, 9, 3]})
        if stops is None:
            stops = ctx.run.stops()
            dispatch(ctx, "display", {"address": "L0.a[d=0:4]"})
        row = 3 + rng.randrange(2)
        target = f"t{row}.{stops[rng.randrange(len(stops))]}"
        if ctx.run.parse(target) > ctx.run.parse(ctx.run.where):
            dispatch(ctx, "until", {"target": target})
        dispatch(ctx, "checkpoint", {"name": "fz"})
        for name in rng.sample(names, len(names)):
            if ctx.run.where == f"t{len(ctx.run.tokens) - 1}.start" and name.startswith(
                    "reverse"):
                continue
            try:
                result = dispatch(ctx, name, _fuzz_args(ctx, name))
            except Refused as error:
                # Values that do not exist at this cursor are refused, never invented.
                assert any(word in str(error) for word in (
                    "not been computed", "not ahead", "no frame", "no consumer",
                    "cannot reverse", "needs a layer")), (name, ctx.run.where, str(error))
                continue
            assert check_bounded(result, name) == result
            assert _unlabelled(result) == [], (name, ctx.run.where, _unlabelled(result))
            seen_numbers += json.dumps(result).count('"authority"')
    assert seen_numbers > 100


def test_dispatch_refuses_an_oversize_result_instead_of_returning_it(monkeypatch):
    import importlib

    registry = importlib.import_module("mdb.ops.registry")
    import dataclasses

    found = registry._REGISTRY["where"]
    big = dataclasses.replace(found, fn=lambda ctx, **kw: {"stuff": list(range(500))})
    monkeypatch.setitem(registry._REGISTRY, "where", big)
    with pytest.raises(Refused, match="record bound|bound"):
        dispatch(type("C", (), {"run": object()})(), "where", {})
