"""Matmul ``bound``s state their accumulation premise and read it from the run's numeric scope
(M1): float32 on CPU and on CUDA with ``allow_{bf16,fp16}_reduced_precision_reduction`` False;
reduced precision (every partial sum rounded to the model dtype) when the flag is True on a CUDA
16-bit run, where an n-term sum with ``n u >= 1`` has no bound and the bound is refused."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import circuits, explain  # noqa: E402
from mdb.analysis.accumulation import (  # noqa: E402
    PREMISE_FAILS,
    U32,
    Accumulation,
    accumulation,
)
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.analysis.propagate import Edit, whatif  # noqa: E402
from mdb.analysis.reads import head_mix, reads  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3]


def _adapter(device, dtype, **scope):
    return SimpleNamespace(dtype=dtype, execution={"device": {"type": device, "index": 0},
                                                   "numeric_scope": scope})


def test_accumulation_follows_device_dtype_and_recorded_flags():
    bf16, fp16 = torch.bfloat16, torch.float16
    cpu = accumulation(_adapter("cpu", bf16, bf16_reduced_precision_reduction=True))
    assert cpu.u == U32 and not cpu.reduced and cpu.gamma(1536) is not None
    off = accumulation(_adapter("cuda", bf16, bf16_reduced_precision_reduction=False))
    assert off.u == U32 and "= False (numeric scope)" in off.premise
    on = accumulation(_adapter("cuda", bf16, bf16_reduced_precision_reduction=True))
    assert on.reduced and on.u == 2.0 ** -8 and "True (numeric scope)" in on.premise
    assert on.gamma(255) is not None and on.gamma(256) is None and on.gamma(1536) is None
    assert on.refusal(1536).startswith(PREMISE_FAILS)
    half = accumulation(_adapter("cuda", fp16, fp16_reduced_precision_reduction=[True, True]))
    assert half.u == 2.0 ** -11 and half.gamma(2047) is not None and half.gamma(2048) is None
    tuple_off = accumulation(_adapter("cuda", fp16,
                                      fp16_reduced_precision_reduction=[False, False]))
    assert tuple_off.u == U32
    tf32 = accumulation(_adapter("cuda", torch.float32, tf32_matmul=True,
                                 float32_matmul_precision="highest"))
    assert tf32.gamma(8) is None and tf32.refusal(8).startswith(PREMISE_FAILS)
    fp32 = accumulation(_adapter("cuda", torch.float32, tf32_matmul=False,
                                 float32_matmul_precision="highest"))
    assert fp32.u == U32
    # a scope without the field falls back to the live torch global, and says so
    live = accumulation(_adapter("cuda", bf16))
    assert "(torch global)" in live.premise


def _cuda_view(run):
    """The run's adapter as the analysis would see it on CUDA: same dtype and recorded numeric
    scope, device ``cuda`` (the native calls stay on the CPU, whose execution is validated)."""
    execution = dict(run.adapter.execution, device={"type": "cuda", "index": 0})
    return SimpleNamespace(dtype=run.adapter.dtype, execution=execution)


@pytest.fixture(scope="module")
def debugger():
    return open_tiny("qwen2", "bf16")


def test_whatif_bound_takes_the_reduced_precision_premise_under_cuda(debugger, monkeypatch):
    from mdb.analysis import forecast

    run = debugger.run(PROMPT)
    run.until("t3.L0.h")
    flag = run.adapter.execution["numeric_scope"]["bf16_reduced_precision_reduction"]
    assert flag is True                     # the torch default, recorded in the scope
    edit, consumer = Edit("t3.L1.m", 1.5, "scale"), Margin(3, 4, 9)
    cpu = whatif(run, edit, consumer).report
    monkeypatch.setattr(forecast, "accumulation", lambda owner: accumulation(_cuda_view(owner)))
    cuda = whatif(run, edit, consumer).report
    # the consumer bound (measured 144 at fp32) is wider than |exact| 3e-4: the note and the
    # number kept (`bound_value`), the premises still stated
    fp32_bound = cpu["prediction"]["exact"]["bound_value"]["value"]
    reduced = cuda["prediction"]["exact"]
    assert reduced["bound_value"]["value"] > fp32_bound    # u = 2^-8 instead of 2^-24
    assert reduced["bound_informative"] is False and "bound" not in reduced
    assert any("cuBLAS may round partial sums to bf16" in p for p in reduced["premises"])
    assert not any("cuBLAS" in p for p in cpu["prediction"]["exact"]["premises"])
    assert cuda["measured"]["delta"] == cpu["measured"]["delta"]     # nothing native changed


@pytest.fixture
def no_gamma(monkeypatch):
    """Every analysis module sees an accumulation with no ``gamma_n`` for any sum (``u = 1``):
    the tiny models are too narrow (n u < 1 at bf16) to reach the refusal on their own."""
    from mdb.analysis import accumulation as module
    from mdb.analysis import forecast
    from mdb.analysis import reads as reads_module

    coarse = Accumulation(1.0, "test premise: partial sums at u = 1", reduced=True)
    for target in (module, forecast, reads_module, circuits):
        monkeypatch.setattr(target, "accumulation", lambda *a, **k: coarse)
    return coarse


def test_failed_accumulation_premise_refuses_every_matmul_bound(debugger, no_gamma):
    run = debugger.run(PROMPT)
    run.until("t3.L0.h")
    report = whatif(run, Edit("t3.L1.m", 1.5, "scale"), Margin(3, 4, 9)).report
    exact = report["prediction"]["exact"]
    assert "bound" not in exact and f"refused: {PREMISE_FAILS}" in exact["bound_note"], exact
    rounding = next(t for t in report["split"]["terms"] if t["name"] == "rounding effect")
    assert "bound" not in rounding and "refused" in rounding["bound_note"]
    assert report["state"] == report["effect"]["state"]      # the bound does not set the state
    run.until("t3.logits")
    checks = head_mix(run, 3, 0).checks()
    assert checks["head_write"]["bound"] is None and checks["value_mix"]["bound"] is None
    assert checks["head_write"]["refused"].startswith(PREMISE_FAILS)
    summary = reads(run, "t1", causal=False).report["checks"]
    assert summary["head_write"]["state"] == "refused"
    assert summary["value_mix"]["state"] == "refused"
    causal = reads(run, "t1", consumer=Margin(3, 4, 9), top=1).report["reads"][0]["causal"]
    assert "rounding_bound" not in causal and "floor" in causal["effect"]
    body = circuits.build(run, Margin(3, 4, 9), top=2, depth=3).record["body"]
    assert body["remainders_within"] is None
    out = explain.explain_margin(run, 4, 9)
    head = next(r for r in out["remainders"] if r["name"] == "head")
    assert head["bound"]["state"] == "refused" and out["all_within_rounding"] is None


def test_symbols_renders_a_refused_remainder_bound(debugger, no_gamma):
    from mdb.ops import dispatch
    from mdb.repl.render import render

    run = debugger.run(PROMPT)
    run.until("t3.logits")
    result = dispatch(debugger.ops(run), "symbols", {"consumer": "M[4,9]"})
    remainder = result["unbound"]["parts"]["remainder"]
    assert remainder["within"] is None and remainder["bound"]["state"] == "refused"
    assert "bound refused" in render(result)


# --- M50: the accumulation premise is measured per run, not assumed from the flag -------------

def _measured(dev, premise="fp32-then-round", by_type=None):
    return {"premise": premise, "device": "cuda", "dtype": str(torch.bfloat16), "batch": 1,
            "rows": [1, 16], "stages": 42, "max_deviation_milli": int(dev * 1000),
            "max_ulp_index": 3,
            "by_type_milli": by_type or {"gate_proj": int(dev * 1000), "q_proj": 0}}


def test_measured_fp32_then_round_resolves_a_long_sum_the_flag_premise_refused():
    scope = dict(bf16_reduced_precision_reduction=True, accumulation=_measured(1.0))
    acc = accumulation(_adapter("cuda", torch.bfloat16, **scope))
    assert acc.u == U32 and "measured on this device" in acc.premise
    assert "max deviation 1 eps sum|terms|" in acc.premise and "gate_proj" in acc.premise
    exact = U32 * 1536 / (1 - U32 * 1536)
    assert acc.gamma(1536) is not None and acc.gamma(1536) > exact     # widened by 1 ulp
    assert acc.gamma(1536) < exact + 2.0 ** -7 * 1.01     # d = 1 eps sum|terms|
    clean = accumulation(_adapter("cuda", torch.bfloat16, bf16_reduced_precision_reduction=True,
                                  accumulation=_measured(0.0)))
    assert clean.gamma(1536) == pytest.approx(exact)                   # no deviation, no widening
    old = accumulation(_adapter("cuda", torch.bfloat16, bf16_reduced_precision_reduction=True))
    assert old.gamma(1536) is None                                     # nothing measured: refused


def test_measurement_that_does_not_support_fp32_then_round_stays_refused():
    scope = dict(bf16_reduced_precision_reduction=True,
                 accumulation=_measured(7.0, premise="unsupported"))
    acc = accumulation(_adapter("cuda", torch.bfloat16, **scope))
    assert acc.gamma(1536) is None and acc.refusal(1536).startswith(PREMISE_FAILS)
    assert "does not support fp32-then-round" in acc.refusal(1536)
    assert "max deviation 7 eps" in acc.refusal(1536)


def test_measure_accumulation_records_per_stage_type_and_flags_a_bad_measurement(monkeypatch):
    from saturn_pub import numerics
    from mdb.engine.models import tiny_model

    model = tiny_model("qwen2", dtype="bf16")
    record = numerics.measure_accumulation(model.model.layers, model.lm_head)
    assert record["premise"] == "fp32-then-round" and record["batch"] == 1
    assert record["max_deviation_milli"] == max(record["by_type_milli"].values())
    assert record["max_deviation_milli"] <= 2000 and "max_ulp_index" in record
    assert {"q_proj", "gate_proj", "down_proj", "lm_head"} <= set(record["by_type_milli"])
    assert set(record["by_type_ulp_index"]) == set(record["by_type_milli"])
    monkeypatch.setattr(numerics, "_stage_deviation", lambda *a: (9, 9.0))
    bad = numerics.measure_accumulation(model.model.layers, model.lm_head)
    assert bad["premise"] == "unsupported" and bad["max_deviation_milli"] == 9000


def test_adapter_records_the_measurement_in_its_numeric_scope(monkeypatch):
    import saturn_pub.adapters.decoder as decoder
    from mdb.engine.models import tiny_model

    monkeypatch.setattr(decoder, "needs_accumulation_measurement", lambda *a: True)
    adapter = decoder.DecoderAdapter(tiny_model("qwen2", dtype="bf16"), granularity="operation")
    record = adapter.execution["numeric_scope"]["accumulation"]
    assert record["premise"] == "fp32-then-round" and record["max_deviation_milli"] == 0
    from mdb.core import records
    records._scan_claims(adapter.execution, "execution", (), records._PROFILE)   # no bare float
    adapter.validate_execution()          # the extra field is not a drift-checked live flag
    monkeypatch.undo()
    cpu = decoder.DecoderAdapter(tiny_model("qwen2", dtype="bf16"), granularity="operation")
    assert "accumulation" not in cpu.execution["numeric_scope"]


def test_whatif_bound_prints_the_measured_premise(debugger, monkeypatch):
    from mdb.analysis import forecast

    run = debugger.run(PROMPT)
    run.until("t3.L0.h")

    def view(owner):
        scope = dict(owner.adapter.execution["numeric_scope"], accumulation=_measured(1.0))
        execution = dict(owner.adapter.execution, device={"type": "cuda", "index": 0},
                         numeric_scope=scope)
        return accumulation(SimpleNamespace(dtype=owner.adapter.dtype, execution=execution))

    monkeypatch.setattr(forecast, "accumulation", view)
    exact = whatif(run, Edit("t3.L1.m", 1.5, "scale"), Margin(3, 4, 9)).report["prediction"]["exact"]
    assert "bound" in exact or exact["bound_value"]["value"] > 0     # a bound, or the note on it
    assert any("measured on this device" in p for p in exact["premises"])


def test_truncated_layer_types_names_the_expected_length():
    from saturn_pub.adapters.decoder import DecoderAdapter
    from mdb.engine.models import tiny_model

    model = tiny_model("qwen2", dtype="bf16", layers=2)
    model.config.layer_types = ["full_attention"] * 5
    with pytest.raises(ValueError, match=r"layer_types has 5 entries.*2 decoder layers.*exactly 2"):
        DecoderAdapter(model, granularity="operation")
