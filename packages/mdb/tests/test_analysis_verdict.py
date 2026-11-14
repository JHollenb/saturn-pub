"""Effect verdicts from the P20 accounting against the emulated change (M51): the native change
against the emulated one within one dtype spacing at the consumer (supplied, the constant verify
uses); the exact change is reported beside it; no worst-case rounding bound decides a verdict or
a result's state (M3)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import verdict as _verdict  # noqa: E402
from mdb.analysis.compare import Side, compare  # noqa: E402
from mdb.analysis.forecast import Logit, Margin  # noqa: E402
from mdb.analysis.propagate import Edit, whatif  # noqa: E402
from mdb.analysis.verdict import EFFECT, NO_EFFECT, decide, spacing  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3]
CONFIGS = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32"),
           ("llama", "bf16"), ("llama", "fp32"), ("mistral", "bf16"), ("mistral", "fp32")]


def T(value):
    return torch.tensor(value, dtype=torch.float64)


def test_spacing_and_the_rule_on_numbers():
    assert spacing(20.0, torch.bfloat16) == 0.125 and spacing(16.0, torch.bfloat16) == 0.125
    assert spacing(15.9, torch.bfloat16) == 0.0625
    bf16 = torch.bfloat16
    # the hot/cold reads of M51: native 0.25 = emulated 0.25 at |M| near 20 (spacing 0.125)
    read = decide(T(0.25), T(0.25), 20.0, bf16, exact=T(0.33))
    assert (read["verdict"], read["state"]) == (EFFECT, "resolved")
    assert read["floor"]["value"] == 0.125 and read["floor"]["authority"] == "supplied"
    assert read["floor_spacings"]["value"] == _verdict.FLOOR_SPACINGS == 1.0
    assert read["residual"]["value"] == 0.0 and read["residual"]["authority"] == "measured"
    assert read["exact"]["authority"] == read["emulated"]["authority"] == "derived-exact"
    assert read["rounding_effect"]["value"] == pytest.approx(-0.08)
    # one spacing apart is within the floor; more is not, and says by how much
    assert decide(T(0.375), T(0.25), 20.0, bf16)["verdict"] == EFFECT
    apart = decide(T(0.5), T(0.25), 20.0, bf16)
    assert (apart["verdict"], apart["state"]) == ("unresolved", "unresolved")
    assert "2 spacings" in apart["reason"]
    # a finding of where they part decides beyond the floor
    found = {"verdict": "not confirmed", "state": "resolved", "reason": "t3.L0.sg: outside"}
    assert decide(T(0.5), T(0.25), 20.0, bf16, parting=found)["verdict"] == "not confirmed"
    # no change in the dtype: no effect, with the exact change the dtype cannot show
    quiet = decide(T(0.0), T(0.0), 20.0, bf16, exact=T(0.033))
    assert (quiet["verdict"], quiet["state"]) == (NO_EFFECT, "resolved")
    assert "no change at bfloat16 resolution" in quiet["note"] and "0.033" in quiet["note"]
    # a native change that is rounding only keeps the word and says so
    noise = decide(T(0.125), T(0.125), 20.0, bf16, exact=T(0.0))
    assert noise["verdict"] == EFFECT and "rounding" in noise["note"]
    # no emulated chain: native only, edited against base
    native = decide(T(0.125), None, 20.0, bf16, why_native_only="no chain")
    assert native["verdict"] == EFFECT and native["basis"] == "native only: no chain"
    assert decide(T(0.0), None, 20.0, bf16)["verdict"] == NO_EFFECT
    vector = decide(T([0.0, -2.0]), T([0.0, -2.0]), 20.0, bf16)
    assert vector["index"] == 1 and vector["verdict"] == EFFECT


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _at(debugger, cursor="t3.L0.h"):
    run = debugger.run(PROMPT)
    run.until(cursor)
    return run


def _check_rule(report):
    """The result's state is the verdict's, and the verdict follows the rule on its numbers:
    native against emulated within one spacing; effect iff native is not zero."""
    effect = report["effect"]
    assert report["state"] == effect["state"]
    n, e = effect["native"]["value"], effect["emulated"]["value"]
    assert effect["floor"]["value"] == effect["spacing"]["value"]          # 1 spacing
    assert n == e and effect["residual"]["value"] == 0.0, effect         # measured: bit-exact
    assert effect["verdict"] == (EFFECT if n != 0 else NO_EFFECT), effect
    return effect


@pytest.mark.parametrize("address,value,op,consumer", [
    ("t3.L0.h_out", 4.0, "add", Margin(3, 4, 9)),           # port edit, one layer upstream
    ("t3.L0.v", 3.0, "add", Margin(3, 4, 9)),               # two layers upstream
    ("t3.L1.m", -2.0, "add", Logit(3, 7)),                  # last layer
])
def test_large_port_edits_resolve_as_effects(debugger, address, value, op, consumer):
    report = whatif(_at(debugger), Edit(address, value, op), consumer).report
    effect = _check_rule(report)
    assert (effect["verdict"], report["state"]) == (EFFECT, "resolved")


def test_a_large_state_cell_edit_resolves_as_an_effect(debugger):
    edit = Edit("cache.L0.v[kv0,row=1]", 8.0, "add",
                cell={"layer": 0, "tensor": "v", "rows": [1], "kv": (0,), "d": None})
    report = whatif(_at(debugger), edit, Margin(3, 4, 9)).report
    assert _check_rule(report)["verdict"] == EFFECT and report["state"] == "resolved"


def test_a_no_op_edit_is_no_effect(debugger):
    report = whatif(_at(debugger), Edit("t3.L1.m", 1.0, "scale"), Margin(3, 4, 9)).report
    effect = _check_rule(report)
    # native: bit-identical; exact (with the written operand, read back): the float64 chain
    # through the native m minus the float64 base chain, i.e. rounding of m alone
    assert effect["native"]["value"] == 0.0
    assert abs(effect["exact"]["value"]) <= effect["floor"]["value"]
    assert (effect["verdict"], report["state"]) == (NO_EFFECT, "resolved")


def test_a_sub_spacing_edit_follows_the_rule(debugger):
    # below half a step of m in the model dtype: the write rounds back to the same value
    tiny = 1e-7 if debugger.run(PROMPT).adapter.dtype == torch.bfloat16 else 1e-14
    report = whatif(_at(debugger), Edit("t3.L1.m", tiny, "add"), Margin(3, 4, 9)).report
    effect = _check_rule(report)
    assert effect["native"]["value"] == 0.0
    assert (effect["verdict"], report["state"]) == (NO_EFFECT, "resolved")
    assert "resolution" in effect["note"]


def test_compare_verdicts_are_native_only(debugger):
    run = _at(debugger)
    result = compare(run, Side("A", edits=[Edit("t3.L0.h_out", 4.0, "add")]),
                     Side("B", edits=[Edit("t3.L1.m", 1.0, "scale")]), Margin(3, 4, 9))
    verdicts = result["report"]["verdicts"]
    assert verdicts["A"]["verdict"] == EFFECT and verdicts["B"]["verdict"] == NO_EFFECT
    assert verdicts["A"]["basis"].startswith("native only")
    assert "exact" not in verdicts["A"]
