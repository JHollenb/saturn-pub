"""Subspace bindings (DAS entry 11b) on tiny Qwen2 (bf16, V bias) and Qwen3 (fp32).

The basis is orthonormal and its authority/provenance enforced; ``swap`` with the full basis is the
source and with a zero-overlap basis the base; the readout directions are the symbol table's cell
direction (``symbols._cell_direction``) differenced; a trained-shape pyvene rotation becomes a
``fitted`` binding whose swap equals pyvene's own intervention; an mdb-branch interchange with the
full basis equals the whole-cell interchange; the symbol table reports a bound subspace with its
mass and an in-subspace consumer share that adds up with the complement's."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import subspace, symbols  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3, 7]
SOURCE = [2, 8, 4, 6, 3]


@pytest.fixture(scope="module", params=[("qwen2", "bf16"), ("qwen3", "fp32")],
                ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _value(debugger, ids, layer, kv, row):
    run = debugger.run(ids)
    run.until(f"t{len(ids) - 1}.logits")
    return subspace.f64(run.read_cache(layer, "v", staged=False)[0, kv, row]).cpu()


def test_basis_rules_and_algebra(debugger):
    dh = debugger.adapter.head_dim
    g = torch.Generator().manual_seed(0)
    raw = torch.randn(dh, 3, generator=g, dtype=torch.float64)
    sb = symbols.Binding.subspace("c", 1, 0, raw, authority="supplied")
    q = sb.basis
    assert sb.k == 3 and sb.kind == "subspace" and sb.cells == []
    assert torch.allclose(q.T @ q, torch.eye(3, dtype=torch.float64), atol=1e-12)
    with pytest.raises(Refusal):
        subspace.subspace_binding("c", 1, 0, raw, authority="fitted")          # no provenance
    with pytest.raises(Refusal):
        subspace.subspace_binding("c", 1, 0, raw, authority="guessed")
    full = subspace.subspace_binding("f", 1, 0, torch.eye(dh, dtype=torch.float64))
    a, b = torch.randn(dh, generator=g, dtype=torch.float64), torch.randn(dh, generator=g, dtype=torch.float64)
    assert torch.allclose(full.swap(a, b), b, atol=1e-12)
    comp = subspace.orthonormal(torch.eye(dh, dtype=torch.float64) - q @ q.T)
    assert comp.shape[1] == dh - 3
    other = subspace.subspace_binding("o", 1, 0, comp)
    assert torch.allclose(other.swap(a, a + q @ torch.ones(3, dtype=torch.float64)), a, atol=1e-12)
    assert max(sb.angles(sb)) < 1e-5 and min(sb.angles(comp[:, :3])) > 89.999
    m = sb.mass(a)["value"] + other.mass(a)["value"]
    assert abs(m - 1.0) < 1e-12
    back = subspace.SubspaceBinding.from_json(sb.to_json())
    assert back.digest() == sb.digest() and back.authority == "supplied"
    bad = sb.to_json()
    bad["digest"] = "0" * 64
    with pytest.raises(Refusal):
        subspace.SubspaceBinding.from_json(bad)


def test_readout_directions_are_the_symbol_tables_cell_direction(debugger):
    ad = debugger.adapter
    run = debugger.run(PROMPT)
    run.until("t4.logits")
    rows = subspace.readout_directions(ad, 1, 0, [3, 9])
    want = symbols._cell_direction(run, symbols.CellTarget(row=2, layer=1, kv=0, a=3, b=9))
    assert torch.allclose(rows[0] - rows[1], want, atol=1e-10)
    sb = subspace.readout_subspace(ad, "countries", 1, 0, [3, 9, 11, 12, 20], 2, set_name="t")
    assert sb.authority == "derived-exact" and sb.k == 2
    assert sb.provenance["n_tokens"] == 5 and 0 < sb.provenance["explained"] <= 1


def test_class_mean_subspace_is_fitted_with_provenance(debugger):
    dh = debugger.adapter.head_dim
    g = torch.Generator().manual_seed(1)
    vals = {c: [torch.randn(dh, generator=g) for _ in range(3)] for c in "abcd"}
    sb = subspace.class_mean_subspace("m", 1, 0, vals, 2, provenance={"items": "x"})
    assert sb.authority == "fitted" and sb.provenance["classes"] == list("abcd") and sb.k == 2


def test_from_pyvene_swap_equals_pyvene(debugger):
    pv = pytest.importorskip("pyvene")
    dh = debugger.adapter.head_dim
    torch.manual_seed(0)
    iv = pv.LowRankRotatedSpaceIntervention(embed_dim=dh, low_rank_dimension=3)
    sb = subspace.from_pyvene(iv, "das3", 1, 0, provenance={"steps": 0})
    assert sb.authority == "fitted" and sb.provenance["tool"] == "pyvene"
    base, src = torch.randn(2, dh), torch.randn(2, dh)
    with torch.no_grad():
        out = iv(base, src)
    for i in range(2):
        assert torch.allclose(sb.swap(base[i], src[i]).float(), out[i], atol=1e-5)


def test_interchange_full_basis_is_whole_cell_and_table_reports(debugger):
    ad = debugger.adapter
    layer, kv, row = 1, 0, 2
    src = _value(debugger, SOURCE, layer, kv, row)
    full = subspace.subspace_binding("full", layer, kv, torch.eye(ad.head_dim, dtype=torch.float64))
    margin = Margin(4, 4, 9)
    out = subspace.interchange(debugger, full, PROMPT, row, src, margin, floor=0.5)
    assert out["M_floors"]["value"] == out["M"]["value"] / 0.5
    assert out["M"]["value"] == out["M_whole"]["value"]
    assert out["M0"]["authority"] == "measured"
    run = debugger.run(PROMPT)
    run.until("t4.logits")
    lg = subspace.f64(run.read("t4.logits")).reshape(-1)
    # prefill=row vs a fully stepped run: same values up to the prefill kernel path (fp32 ~1e-8)
    assert abs(out["M0"]["value"] - float(lg[4] - lg[9])) < 1e-5
    assert abs(out["mass_of_delta"]["value"] - 1.0) < 1e-12
    recs = [{"M0": out["M0"], "floor": 1e-9, "M": out["M"]}]
    assert subspace.iia(recs)["margin1"]["eligible"] in (0, 1)
    # symbol table: a bound subspace and its complement add up to the cell's direct share
    g = torch.Generator().manual_seed(2)
    q = subspace.orthonormal(torch.randn(ad.head_dim, 2, generator=g, dtype=torch.float64))
    part = subspace.subspace_binding("part", layer, kv, q, provenance={"t": 1}, authority="supplied")
    rest = subspace.subspace_binding("rest", layer, kv,
                                     torch.eye(ad.head_dim, dtype=torch.float64) - q @ q.T)
    part.attach("iia", {"flips": 1, "eligible": 2})
    rep = symbols.table(run, consumer=margin, subspaces=[part.bound(row), rest.bound(row)]).report
    rows = rep["subspaces"]
    assert [r["name"] for r in rows] == ["part", "rest"]
    assert rows[0]["labels"]["iia"]["source_authority"] == "measured"
    assert rows[0]["labels"]["iia"]["result"]["flips"]["value"] == 1
    total = rows[0]["subspace_to_consumer"]["value"] + rows[1]["subspace_to_consumer"]["value"]
    assert abs(total - rows[0]["cell_to_consumer"]["value"]) < 1e-9
    assert abs(rows[0]["mass_native"]["value"] + rows[1]["mass_native"]["value"] - 1) < 1e-9
    with pytest.raises(Refusal):
        subspace.table_row(run, part, margin)                                  # template, no row


def test_decompose_is_a_component_binding_that_closes(debugger):
    from mdb.analysis import components

    ad = debugger.adapter
    run = debugger.run(PROMPT)
    run.until("t4.logits")
    g = torch.Generator().manual_seed(4)
    sb = subspace.subspace_binding("s", 1, 0, torch.randn(ad.head_dim, 2, generator=g,
                                                          dtype=torch.float64)).bound(2)
    cb = sb.decompose(run)
    assert [c.name for c in cb.components] == ["subspace", "complement"]
    assert cb.closure()["max_gap"]["value"] < 1e-12
    tab = components.component_table(run, cb, Margin(4, 4, 9))
    parts = [float(r["to_consumer"]["value"]) for r in tab["components"]]
    assert abs(sum(parts) - tab["cell_to_consumer"]["value"]) < 1e-9
    row = subspace.table_row(run, sb, Margin(4, 4, 9))
    assert abs(parts[0] - row["subspace_to_consumer"]["value"]) < 1e-12
