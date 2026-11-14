"""Effect verdicts from the P20 accounting (native = exact + measured residual) against the floor
``max(|residual|, 2 x dtype spacing at the consumer)``; no worst-case rounding bound decides a
verdict or a result's state (M3)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

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
    big = decide(T(3.75), T(3.69), T(0.06), 20.0, bf16)          # the Qwen2.5 L20 case
    assert (big["verdict"], big["state"]) == (EFFECT, "resolved")
    assert big["floor"]["value"] == 0.25 and big["floor"]["authority"] == "measured"
    assert big["floor"]["proof"] == "P20" and big["exact"]["authority"] == "derived-exact"
    # exact has the other sign. With residual = native - exact (P20) a scalar sign
    # disagreement has |residual| > |native|, so it never clears the floor and lands in the
    # second branch; the first branch's sign test is reached with an inconsistent residual
    flipped = decide(T(0.5), T(-0.1), T(0.05), 20.0, bf16)
    assert flipped["state"] == "unresolved" and "other sign" in flipped["reason"]
    consistent = decide(T(0.5), T(-0.1), T(0.6), 20.0, bf16)
    assert consistent["verdict"] == NO_EFFECT and consistent["floor"]["value"] == 0.6
    small_exact = decide(T(0.5), T(0.05), T(0.45), 20.0, bf16)    # |exact| <= |residual|
    assert small_exact["state"] == "unresolved" and "|exact" in small_exact["reason"]
    quiet = decide(T(0.125), T(0.01), T(0.115), 20.0, bf16)       # both within the floor
    assert (quiet["verdict"], quiet["state"]) == (NO_EFFECT, "resolved")
    hidden = decide(T(0.3), T(0.6), T(-0.3), 20.0, bf16)          # native in, exact above floor
    assert hidden["state"] == "unresolved" and "within the floor" in hidden["reason"]
    native = decide(T(0.5), None, None, 20.0, bf16, "no chain")
    assert native["verdict"] == EFFECT and native["floor"]["value"] == 0.25
    assert native["basis"] == "native only: no chain"
    assert decide(T(0.25), None, None, 20.0, bf16)["verdict"] == NO_EFFECT
    vector = decide(T([0.01, -2.0]), T([0.0, -1.9]), T([0.01, -0.1]), 20.0, bf16)
    assert vector["index"] == 1 and vector["verdict"] == EFFECT


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _at(debugger, cursor="t3.L0.h"):
    run = debugger.run(PROMPT)
    run.until(cursor)
    return run


def _check_rule(report):
    """The result's state is the verdict's, and the verdict follows the rule on its numbers."""
    effect = report["effect"]
    assert report["state"] == effect["state"]
    n, e, r = (effect[k]["value"] for k in ("native", "exact", "residual"))
    floor = effect["floor"]["value"]
    assert floor == max(abs(r), 2 * effect["spacing"]["value"])
    if abs(n) > floor:
        want = EFFECT if (e != 0 and (e > 0) == (n > 0) and abs(e) > abs(r)) else "unresolved"
    else:
        want = NO_EFFECT if abs(e) <= floor else "unresolved"
    assert effect["verdict"] == want, effect
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


def test_a_no_op_edit_is_no_effect_above_the_noise_floor(debugger):
    report = whatif(_at(debugger), Edit("t3.L1.m", 1.0, "scale"), Margin(3, 4, 9)).report
    effect = _check_rule(report)
    # native: bit-identical; exact (with the written operand, read back): the float64 chain
    # through the native m minus the float64 base chain, i.e. rounding of m alone
    assert effect["native"]["value"] == 0.0
    assert abs(effect["exact"]["value"]) <= effect["floor"]["value"]
    assert (effect["verdict"], report["state"]) == (NO_EFFECT, "resolved")


def test_a_sub_floor_edit_follows_the_rule(debugger):
    # below half a step of m in the model dtype: the write rounds back to the same value
    tiny = 1e-7 if debugger.run(PROMPT).adapter.dtype == torch.bfloat16 else 1e-14
    report = whatif(_at(debugger), Edit("t3.L1.m", tiny, "add"), Margin(3, 4, 9)).report
    effect = _check_rule(report)
    assert effect["native"]["value"] == 0.0
    assert (effect["verdict"], report["state"]) == (NO_EFFECT, "resolved")


def test_compare_verdicts_are_native_only_against_the_spacing_floor(debugger):
    run = _at(debugger)
    result = compare(run, Side("A", edits=[Edit("t3.L0.h_out", 4.0, "add")]),
                     Side("B", edits=[Edit("t3.L1.m", 1.0, "scale")]), Margin(3, 4, 9))
    verdicts = result["report"]["verdicts"]
    assert verdicts["A"]["verdict"] == EFFECT and verdicts["B"]["verdict"] == NO_EFFECT
    assert verdicts["A"]["basis"].startswith("native only")
    assert "exact" not in verdicts["A"]
