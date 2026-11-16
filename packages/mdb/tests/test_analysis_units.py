"""Gate-side attribution (``analysis.units``, run 6 lesson): selected but zero-valued units."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.analysis import units  # noqa: E402
from mdb.analysis.explain import explain_margin  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402

PROMPT = [1, 5, 9, 3, 6, 2]


def test_selected_zero_rule():
    gate = torch.tensor([[1.0, 0.9, 0.1, 0.0]])
    share = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    mask = units.selected_zero(gate, share, gate_floor=0.25, zero_tol=0.01, relative=True)
    assert mask.tolist() == [[False, True, False, False]]     # gate on, contributes nothing


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_explain_lists_selected_zero_units_and_shares_sum_to_the_mlp_term(dtype):
    dbg = mdb.attach(tiny_model("qwen3", dtype, layers=3))
    run = dbg.run(PROMPT, prefill=0)
    run.until("t5.logits")
    # silence u at one strongly gated unit of L1: selected, zero-valued
    sg = run.read("t5.L1.sg").reshape(-1).float()
    unit = int(sg.abs().argmax())
    run2 = dbg.run(PROMPT, prefill=0)
    run2.until("t5.L1.h")
    mask = torch.zeros(sg.numel(), dtype=torch.bool)
    mask[unit] = True
    run2.set("t5.L1.u", 0.0, "replace", mask=mask.reshape(1, 1, -1))
    run2.until("t5.logits")
    out = explain_margin(run2, 4, 9, units=8)["mlp_units"]
    names = [u["unit"] for u in out["selected_zero"]]
    assert f"L1.f[i={unit}]" in names
    assert all(abs(u["contribution"]["value"]) == 0.0 for u in out["selected_zero"]
               if u["unit"] == f"L1.f[i={unit}]")
    plain = explain_margin(run, 4, 9, units=4)
    terms = {t["name"]: t["value"]["value"] for t in plain["terms"] if "value" in t}
    per_unit = plain["mlp_units"]["top_contribution"]
    assert len(per_unit) == 4 and per_unit[0]["contribution"]["authority"] == "derived-exact"
    # the units' shares of a layer sum to its mlp term (f = sg * u in float64 vs native m)
    for row in plain["mlp_units"]["per_layer"]:
        tol = 1e-4 if dtype == "fp32" else 0.05
        assert row["sum"]["value"] == pytest.approx(terms[f"L{row['layer']}.mlp"], abs=tol)
