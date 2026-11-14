"""The prefill split is part of the procedure (M26) and stepped prompt rows are stops (M35).

``Run(prefill=K)`` / ``start prefill=K``: K prompt rows in one batched call, the rest stepped one
row at a time through the same transitions as generated rows, their tokens forced. On tiny
Qwen2 fp32 a batched and a stepped row differ at the last bits (measured), so a replay must
use the recorded split; the checkpoint, the recipe and ``where``/``info runs`` carry it.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.analysis.propagate import state_digest  # noqa: E402
from mdb.engine import Run  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops import Refused, dispatch  # noqa: E402
from mdb.ops.predict import session_digest  # noqa: E402

PROMPT = [1, 5, 9, 3, 7, 2]
LAST = f"t{len(PROMPT) - 1}.logits"
CLEAN8, CORRUPT8 = [1, 7, 9, 3, 6, 2, 4, 8], [1, 5, 9, 3, 6, 2, 4, 8]   # row 1 differs


@pytest.fixture(scope="module")
def fp32():
    return mdb.attach(tiny_model("qwen2", "fp32"))


@pytest.fixture(scope="module")
def deep():
    return mdb.attach(tiny_model("qwen2", "bf16", layers=4))


def _logits(run, where=LAST):
    run.until(where)
    return run.read(where)


def test_stepped_rows_force_the_prompt_and_differ_from_batched_at_the_last_bits(fp32):
    default = Run(fp32.adapter, PROMPT)
    assert default.prefill == len(PROMPT) - 1 and default.first_row == len(PROMPT) - 1
    want = _logits(default)
    stepped = Run(fp32.adapter, PROMPT, prefill=0)
    assert stepped.first_row == 0 and stepped.root == "t0.start"
    got = _logits(stepped)
    assert stepped.tokens == PROMPT
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    assert not torch.equal(got, want)   # measured on this model: the split changes bits
    assert torch.equal(_logits(Run(fp32.adapter, PROMPT, prefill=0)), got)   # deterministic
    assert stepped.procedure["text"] == ("prefill=0: no row batched, rows 0..5 stepped "
                                         "(tokens forced)")
    with pytest.raises(ValueError, match="prefill"):
        Run(fp32.adapter, PROMPT, prefill=len(PROMPT))


def test_checkpoint_and_restore_carry_the_split_bitwise(fp32):
    stepped = Run(fp32.adapter, PROMPT, prefill=2)
    stepped.until("t3.L1.h")
    stepped.set("t3.L1.h", torch.full(stepped.port_shape("h"), 0.25), "add")
    want = _logits(stepped)
    checkpoint = stepped.checkpoint()
    assert checkpoint["prefill"] == 2
    other = Run(fp32.adapter, PROMPT)          # a default run adopts the checkpoint's split
    other.hold({sha: t for sha, t in stepped.edit_tensors().items()})
    other.restore(checkpoint)
    assert other.prefill == 2 and other.checkpoint()["cut"] == checkpoint["cut"]
    assert torch.equal(other.read(LAST), want)
    default = Run(fp32.adapter, PROMPT)
    default.until(LAST)
    assert default.checkpoint()["cut"] != Run(fp32.adapter, PROMPT, prefill=0).checkpoint()["cut"]


def test_a_durable_recipe_replays_the_split(tmp_path):
    first = mdb.attach(tiny_model("qwen2", "fp32"), project=tmp_path / "p")
    ctx = first.ops()
    dispatch(ctx, "start", {"prompt": PROMPT, "prefill": 1})
    dispatch(ctx, "until", {"target": LAST})
    want = ctx.run.read(LAST)
    made = dispatch(ctx, "checkpoint", {"name": "pre"})
    assert "durable" in made
    second = mdb.attach(tiny_model("qwen2", "fp32"), project=tmp_path / "p")
    later = second.ops(None)
    dispatch(later, "restart", {"name": "pre"})
    assert later.run.prefill == 1 and torch.equal(later.run.read(LAST), want)
    assert not torch.equal(want, _logits(Run(second.adapter, PROMPT)))


def test_where_info_runs_and_start_show_the_split(deep):
    ctx = deep.ops()
    started = dispatch(ctx, "start", {"prompt": CORRUPT8, "prefill": 3})
    assert started["prefill"]["rows"] == 3 and not started["prefill"]["default"]
    assert started["where"] == "t3.L0.h"
    where = dispatch(ctx, "where", {})
    assert where["prefill"]["text"].startswith("prefill=3: rows 0..2 in one batched call")
    runs = dispatch(ctx, "info", {"subject": "runs"})
    assert any(item.get("prefill", {}).get("rows") == 3 for item in runs["runs"])
    assert dispatch(ctx, "start", {"prompt": CORRUPT8})["prefill"]["default"]
    with pytest.raises(Refused, match="prefill=8 is past the prompt"):
        dispatch(ctx, "start", {"prompt": CORRUPT8, "prefill": 8})


def test_ports_at_a_stepped_earlier_row_are_stops(deep):
    ctx = deep.ops()
    dispatch(ctx, "start", {"prompt": CLEAN8, "name": "clean", "prefill": 0})
    dispatch(ctx, "until", {"target": "t7.logits"})
    dispatch(ctx, "start", {"prompt": CORRUPT8, "prefill": 5})
    assert dispatch(ctx, "until", {"target": "t5.L3.h"})["where"] == "t5.L3.h"
    clean = deep.runs["clean"]
    set_ = dispatch(ctx, "set", {"target": "t5.L3.A[h1]", "value": "clean:t5.L3.A[h1]"})
    assert set_["source"] == "clean:t5.L3.A[h1]"
    dispatch(ctx, "until", {"target": "t6.L0.h"})
    printed = dispatch(ctx, "print", {"address": "t5.L3.m"})
    assert printed["shape"] == [1, 1, deep.adapter.hidden_size]
    before = session_digest(ctx)
    result = dispatch(ctx, "whatif", {"target": "t5.L3.h", "value": "clean:t5.L3.h",
                                      "consumer": "t7.M[4,9]"})
    assert result["session"]["unchanged"] and session_digest(ctx) == before
    assert result["measured"]["delta"]["authority"] == "measured"
    assert result["edit"]["origin"]["origin"] == "clean" and clean.where == "t7.logits"


def test_ports_at_a_prefilled_row_refuse_naming_the_split(deep):
    ctx = deep.ops()
    dispatch(ctx, "start", {"prompt": CLEAN8, "name": "clean", "prefill": 0})
    dispatch(ctx, "until", {"target": "t7.logits"})
    dispatch(ctx, "start", {"prompt": CORRUPT8})                 # default: rows 0..6 prefilled
    dispatch(ctx, "until", {"target": "t7.logits"})
    for name, args in (("until", {"target": "t5.L3.h"}),
                       ("print", {"address": "t5.L3.m"}),
                       ("set", {"target": "t5.L3.A[h1]", "value": "clean:t5.L3.A[h1]"}),
                       ("whatif", {"target": "t5.L3.h", "value": "clean:t5.L3.h",
                                   "consumer": "t7.M[4,9]"})):
        with pytest.raises(Refused, match=r"prefill=K") as caught:
            dispatch(ctx, name, args)
        assert "K <= 5" in str(caught.value), (name, str(caught.value))


@pytest.mark.parametrize("prefill", [0, 1])
def test_patching_every_stepped_row_of_one_layer_is_the_clean_run(deep, prefill):
    """Corrupt differs from clean at row 1. The full residual ``h`` of layer 2 patched from
    clean at every stepped row: every value of layer 2 on is the clean one, so the final
    logits are clean bit for bit (both runs use the same split). Leaving row 1 out does not."""
    last = len(CLEAN8) - 1
    ctx = deep.ops()
    dispatch(ctx, "start", {"prompt": CLEAN8, "name": "clean", "prefill": prefill})
    dispatch(ctx, "until", {"target": f"t{last}.logits"})
    want = deep.runs["clean"].read(f"t{last}.logits")

    def patched(skip=()):
        dispatch(ctx, "start", {"prompt": CORRUPT8, "prefill": prefill})
        for row in range(prefill, last + 1):
            if row in skip:
                continue
            dispatch(ctx, "until", {"target": f"t{row}.L2.h"})
            dispatch(ctx, "set", {"target": f"t{row}.L2.h", "value": f"clean:t{row}.L2.h"})
        dispatch(ctx, "until", {"target": f"t{last}.logits"})
        return ctx.run.read(f"t{last}.logits")

    assert torch.equal(patched(), want)
    assert not torch.equal(patched(skip=(1,)), want)


def test_stepping_a_long_prompt_holds_little_memory():
    """Held bytes (measured) of a 12-layer tiny model on a 24-token prompt at the last logits:
    stepping every row keeps one trail mark per transition, each sharing the cache rows."""
    dbg = mdb.attach(tiny_model("qwen2", "bf16", layers=12))
    prompt = [(7 * i + 3) % 32 for i in range(24)]
    held = {}
    for prefill in (23, 0):
        run = Run(dbg.adapter, prompt, prefill=prefill)
        run.until("t23.logits")
        held[prefill] = run.held_bytes()
        before = dict(run.held_bytes())
        digest = state_digest(run)
        run.checkpoint(pin=False)
        assert run.held_bytes() == before and state_digest(run) == digest
    assert held[0]["marks"] > held[23]["marks"]
    assert held[0]["total"] < 64 * 1024, held
    assert held[0]["edits"] == held[23]["edits"] == 0
