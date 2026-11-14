"""SAE feature bases on tiny Qwen2/Qwen3 with random SAEs: the identity x = b_dec + D a + e closes,
the error row is native minus reconstruction, the direct-effect split closes on the native margin,
a feature edit equals adding the decoder direction, and every op result is labelled."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import sae  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.analysis.reads import readout  # noqa: E402
from mdb.ops import Refused, dispatch  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

from test_ops_predict import unlabelled  # noqa: E402

PROMPT = [1, 5, 9, 3]
K, F = 6, 64


@pytest.fixture(scope="module", params=[("qwen2", "bf16"), ("qwen3", "fp32")],
                ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


@pytest.fixture(scope="module")
def sae_dir(tmp_path_factory, debugger):
    d = tmp_path_factory.mktemp("sae")
    width = int(debugger.model.config.hidden_size)
    g = torch.Generator().manual_seed(7)
    for layer in range(debugger.adapter.layers):
        torch.save({"W_enc": torch.randn(F, width, generator=g),
                    "b_enc": torch.randn(F, generator=g) * 0.1,
                    "W_dec": torch.randn(width, F, generator=g) / 8,
                    "b_dec": torch.randn(width, generator=g) * 0.1}, d / f"layer{layer}.sae.pt")
    (d / "config.json").write_text(json.dumps({"k": K, "hook_point": "resid_post"}))
    return d


def _ctx(debugger, sae_dir, where="t3.logits"):
    run = debugger.run(PROMPT)
    run.until(where)
    ctx = debugger.ops(run)
    dispatch(ctx, "sae", {"action": "load", "name": "s", "path": str(sae_dir)})
    return ctx


def test_identity_and_error_row(debugger, sae_dir):
    found = sae.open_set("s", str(sae_dir))
    basis = found.get(0)
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    x = run.read("t3.L0.h_out")
    dec = basis.decompose(x)
    x64 = x.reshape(-1).to(torch.float64)
    # error is native minus reconstruction; the identity closes to float64 roundoff
    assert torch.equal(dec.error, x64 - dec.recon)
    assert float((dec.bias + dec.parts.sum(1) + dec.error - x64).abs().max()) < 1e-12
    # the active set is the published rule: top-k of relu(W_enc x + b_enc) in float32
    pre = torch.relu(basis.W_enc @ x.reshape(-1).float() + basis.b_enc)
    want = set(torch.topk(pre, K).indices[torch.topk(pre, K).values > 0].tolist())
    assert set(dec.ids) == want
    # activations are float64 from the native bits
    ref = torch.relu(basis.W_enc.double()[dec.ids] @ x64 + basis.b_enc.double()[dec.ids])
    assert torch.allclose(dec.acts, ref, rtol=0, atol=1e-12)
    # decoder roundtrip: decode of the activations equals the reconstruction
    assert torch.allclose(basis.bias() + basis.directions(dec.ids) @ dec.acts, dec.recon,
                          rtol=0, atol=1e-12)


def test_margin_split_closes_exactly(debugger, sae_dir):
    found = sae.open_set("s", str(sae_dir))
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    reader = readout(run, Margin(3, 4, 9))
    for layer in range(debugger.adapter.layers):
        for port in ("h_out", "h"):
            basis = found.for_site(layer, port)
            if basis is None:
                continue
            s = sae.margin_split(run, reader, basis, layer, port)
            feats = sum(e for _, _, e in s["features"])
            assert abs(s["direct"] - (feats + s["bias"] + s["error"])) < 1e-12
            later = sum(v for _, _, v in s["downstream"])
            assert abs(s["native"] - (s["direct"] + later + s["head_bias"] + s["remainder"])) \
                < 1e-12
            # the remainder is rounding only (small against the margin's size)
            assert abs(s["remainder"]) < 0.05


def test_feature_edit_equals_decoder_direction_add(debugger, sae_dir):
    ctx = _ctx(debugger, sae_dir, "t3.L0.h")
    basis = sae.open_set("s", str(sae_dir)).get(0)
    result = dispatch(ctx, "set", {"target": "L0.h_out", "value": "feature[5]*2", "op": "add"})
    assert result["source"].startswith("feature[5]*2")
    ctx.run.until("t3.logits")
    ref = debugger.run(PROMPT)
    ref.set("t3.L0.h_out", (2.0 * basis.direction(5)).reshape(1, 1, -1), "add")
    ref.until("t3.logits")
    assert torch.equal(ctx.run.read("t3.logits"), ref.read("t3.logits"))


def test_ops_labelled_closure_and_error_node(debugger, sae_dir):
    ctx = _ctx(debugger, sae_dir)
    out = dispatch(ctx, "features", {"address": "L0.h_out", "consumer": "M[4,9]", "lens": 2,
                                     "circuit": True, "match": "L0.f[i=3]"})
    assert unlabelled(out) == []
    assert "error" in out and "effect" in out["error"]
    assert abs(out["accounting"]["closure"]["value"]) < 1e-12
    shown = dispatch(ctx, "circuit", {"target": "show", "id": out["circuit"]["id"]})
    kinds = []
    stack = [shown["record"]["body"]["root"]]
    while stack:
        node = stack.pop()
        kinds.append(node["kind"])
        stack += node.get("children", [])
    assert "error" in kinds and "feature" in kinds
    profile = dispatch(ctx, "features", {"address": "L*.h_out", "consumer": "M[4,9]"})
    assert unlabelled(profile) == []
    assert profile["count"] == debugger.adapter.layers
    assert all(abs(r["closure"]["value"]) < 1e-12 for r in profile["layers"])
    assert unlabelled(dispatch(ctx, "sae", {"action": "list"})) == []


def test_whatif_error_splice_and_coef_ablation(debugger, sae_dir):
    ctx = _ctx(debugger, sae_dir)
    basis = sae.open_set("s", str(sae_dir)).get(0)
    dec = basis.decompose(ctx.run.read("t3.L0.h_out"))
    f = dec.ids[0]
    out = dispatch(ctx, "whatif", {"target": "L0.h_out", "value": f"feature[{f},coef]*-1",
                                   "op": "add", "consumer": "M[4,9]"})
    assert unlabelled(out) == []
    splice = dispatch(ctx, "whatif", {"target": "L0.h_out", "value": "error*-1", "op": "add",
                                      "consumer": "M[4,9]"})
    # native measurement equals a fresh run with h_out replaced by bf16(x - e)
    ref = debugger.run(PROMPT)
    ref.until("t3.L0.h")
    ref.set("t3.L0.h_out", (-dec.error).reshape(1, 1, -1), "add")
    ref.until("t3.logits")
    logits = ref.read("t3.logits").reshape(-1)
    edited = float(logits[4].double() - logits[9].double())
    assert splice["measured"]["edited"]["value"] == edited


def test_cross_run_terms_and_refusals(debugger, sae_dir):
    ctx = _ctx(debugger, sae_dir)
    dispatch(ctx, "run", {"name": "b", "prompt": [2, 6, 10, 4]})
    debugger.runs["b"].until("t3.logits")
    dispatch(ctx, "frame", {"address": "main:"})
    out = dispatch(ctx, "whatif", {"target": "L0.h_out", "value": "recon@b+error@b",
                                   "op": "replace", "consumer": "M[4,9]"})
    assert unlabelled(out) == []
    with pytest.raises(Refused):
        dispatch(ctx, "whatif", {"target": "L0.h_out", "value": "recon@nope", "op": "add",
                                 "consumer": "M[4,9]"})
    with pytest.raises(Refused):
        dispatch(ctx, "features", {"address": "L0.q", "consumer": "M[4,9]"})
    # the whole other-run residual equals its native value bit for bit after the cast
    found = sae.open_set("s", str(sae_dir)).get(0)
    x_b = debugger.runs["b"].read("t3.L0.h_out")
    value = sae.edit_vector(found, sae.parse_terms("recon@b+error@b"), {"b": x_b})
    assert torch.equal(value.to(x_b.dtype), x_b.reshape(-1))


def test_parse_terms():
    terms = sae.parse_terms("feature[12,coef]*-1+2*feature[3]-error+recon@rome*0.5")
    assert terms == [("feature", 12, True, -1.0, None), ("feature", 3, False, 2.0, None),
                     ("error", None, False, -1.0, None), ("recon", None, False, 0.5, "rome")]
    assert sae.is_expression("feature[1]") and not sae.is_expression("L0.h")
    # an origin ends at the next sign: run names take no '-' (regression: job-702270785e57)
    assert sae.parse_terms("recon@rome-recon") == [("recon", None, False, 1.0, "rome"),
                                                   ("recon", None, False, -1.0, None)]
    assert sae.parse_terms("feature[7,coef]@rome-feature[7,coef]") == [
        ("feature", 7, True, 1.0, "rome"), ("feature", 7, True, -1.0, None)]
    with pytest.raises(sae.SaeError):
        sae.parse_terms("feature[x]")


def test_feature_trace_writers_sum_to_probe(debugger, sae_dir):
    ctx = _ctx(debugger, sae_dir)
    basis = sae.open_set("s", str(sae_dir)).get(0)
    dec = basis.decompose(ctx.run.read("t3.L0.h_out"))
    out = dispatch(ctx, "feature", {"feature": dec.ids[0], "layer": 0, "top": 2})
    assert unlabelled(out) == []
    assert out["active_at_site"] is True
    p0, w0 = out["probe"][0], out["writers"][0]
    # r = h + o and h_out = r + m up to one rounding of each native add
    tol = 0.05 * max(1.0, abs(p0["h_out"]["value"]))
    assert abs(p0["r"]["value"] - p0["h"]["value"] - w0["attn"]["value"]) < tol
    assert abs(p0["h_out"]["value"] - p0["r"]["value"] - w0["mlp"]["value"]) < tol
