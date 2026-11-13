"""P19 margin explanation and per-port reconstruction on tiny GQA Qwen2/Qwen3 models.

Run with ``-s`` to print the measured maxima (closure in float64, eta against native).
"""

import json
import math
import os
import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from test_analysis_finite import (  # noqa: E402
    AUTHORITY, CASES, DTYPES, ETA_REL, PROMPT, STATE, is_labelled, tiny, unlabelled_numbers,
)

from mdb.analysis import explain, finite  # noqa: E402
from mdb.engine.ports import PortEdit, program_for  # noqa: E402
from mdb.engine.run import Run  # noqa: E402

PAIRS = [(4, 9), (0, 47), (13, 21)]
SUMMARY: dict = {}


def note(case, key, value):
    entry = SUMMARY.setdefault(key, {})
    entry[case] = max(entry.get(case, 0.0), float(value))


@pytest.fixture(scope="module", autouse=True)
def report():
    yield
    lines = ["", "analysis-explain measured maxima"]
    for key in sorted(SUMMARY):
        for case, value in sorted(SUMMARY[key].items()):
            lines.append(f"  {key:44s} {case:12s} {value:.3e}")
    print("\n".join(lines))
    path = os.environ.get("MDB_ANALYSIS_SUMMARY")
    if path:
        with open(path, "a") as handle:
            handle.write(json.dumps({"module": "explain", "summary": SUMMARY}) + "\n")


@pytest.fixture(scope="module", params=CASES, ids=lambda c: "-".join(c))
def bundle(request):
    family, dtype = request.param
    torch.set_num_threads(1)
    adapter = DecoderAdapter(tiny(family, dtype), granularity="operation")
    return {"case": "-".join(request.param), "dtype": dtype, "adapter": adapter,
            "base": explain.capture_row(adapter, PROMPT)}


def walk(node):
    """Bare-number paths of an explain output (edit records included: no exemption)."""
    return unlabelled_numbers(node)


def labelled(value, authority, proof=None):
    assert value["authority"] == authority, value
    if proof is not None:
        assert value["proof"] == proof
    return value["value"]


def unnormalised(out):
    rho = out["rho_F"]["value"]
    for t in out["terms"]:
        assert abs(t["value"]["value"] - rho * t["projection"]["value"]) <= 1e-15 * (
            abs(rho * t["projection"]["value"]) + 1)
    return {t["name"]: t["projection"]["value"] for t in out["terms"]}


def test_explain_margin_closes_on_the_native_margin(bundle):
    case, base = bundle["case"], bundle["base"]
    logits = finite.f64(base.logits).reshape(-1)
    for a, b in PAIRS:
        out = explain.explain_margin(base, a, b)
        native = labelled(out["native_margin"], "measured", "P19")
        assert native == float(logits[a] - logits[b])
        for term in out["terms"]:
            labelled(term["value"], "derived-exact", "P19")
        kinds = {r["name"]: r["value"]["authority"] for r in out["remainders"]}
        assert kinds == {"residual_rounding": "measured", "unlisted": "derived-exact",
                         "final_norm": "measured", "head": "measured"}
        assert len(out["terms"]) == 1 + 2 * bundle["adapter"].layers
        closure = abs(labelled(out["closure"], "derived-exact"))
        assert closure <= 1e-9 * max(abs(native), max(abs(t["value"]["value"])
                                                      for t in out["terms"]))
        eta = labelled(out["eta"], "measured")
        scale = float(logits.abs().max())
        assert abs(eta) <= ETA_REL[bundle["dtype"]] * scale
        note(case, "P19 explain closure abs", closure)
        note(case, "P19 explain eta/max|logit|", abs(eta) / scale)
        for r in out["remainders"]:
            note(case, f"P19 remainder {r['name']} abs", abs(r["value"]["value"]))
        assert abs(out["remainders"][1]["value"]["value"]) <= 1e-12 * scale


def test_explain_margin_terms_follow_a_layer_edit(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    a, b = PAIRS[0]
    before = explain.explain_margin(base, a, b)
    flipped = explain.capture_row(adapter, PROMPT, edits={1: [PortEdit("o", -1.0, "scale")]})
    doubled = explain.capture_row(adapter, PROMPT, edits={2: [PortEdit("m", 2.0, "scale")]})
    u0 = unnormalised(before)
    for caps, layer, port, factor in ((flipped, 1, "attn", -1.0), (doubled, 2, "mlp", 2.0)):
        out = explain.explain_margin(caps, a, b)
        assert abs(out["closure"]["value"]) <= 1e-9 * max(abs(t["value"]["value"])
                                                          for t in out["terms"])
        u1 = unnormalised(out)
        edited = f"L{layer}.{port}"
        assert u1[edited] == factor * u0[edited]
        upstream = ["embedding"] + [f"L{i}.{k}" for i in range(layer) for k in ("attn", "mlp")]
        if port == "mlp":
            upstream.append(f"L{layer}.attn")
        for name in upstream:
            assert u1[name] == u0[name], name
        assert out["native_margin"]["value"] != before["native_margin"]["value"]
        note(case, f"P19 edited {edited} margin shift",
             abs(out["native_margin"]["value"] - before["native_margin"]["value"]))


def test_explain_margin_first_order_column_is_labelled_local(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    out = explain.explain_margin(base, *PAIRS[0], first_order=True)
    column = out["first_order"]
    assert column["authority"] == "derived-local"
    assert [r["name"] for r in column["rows"]] == [t["name"] for t in out["terms"]]
    for row in column["rows"]:
        assert row["value"]["authority"] == "derived-local"
    scale = float(finite.f64(base.logits).abs().max())
    for primal in column["consumer_primal_minus_native"]:
        assert abs(labelled(primal, "measured")) <= ETA_REL[bundle["dtype"]] * scale
        note(case, "consumer_gradient primal - native", abs(primal["value"]))
    edited = explain.capture_row(adapter, PROMPT, edits={1: [PortEdit("o", -1.0, "scale")]})
    refused = explain.explain_margin(edited, *PAIRS[0], first_order=True)["first_order"]
    assert refused["state"] == "refused" and "unedited" in refused["reason"]
    assert refused["authority"] is None and "rows" not in refused


def test_explain_port_reconstructs_every_port(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    program = program_for(adapter)
    for layer in range(adapter.layers):
        for port in program.names:
            info = explain.explain_port(base, layer, port)
            eta = labelled(info["eta_rel"], "derived-exact" if port == "s" else "measured",
                           info["proof"])
            assert torch.isfinite(info["result"].eta).all()
            assert eta <= ETA_REL[bundle["dtype"]], (layer, port, eta)
            assert info["proof"] in finite.IDENTITIES
            assert info["function"] in finite.IDENTITIES[info["proof"]]["functions"]
            assert info["operator"] == program.spec(port).operator
            note(case, f"port {port} eta_rel", eta)
    assert explain.explain_port(base, 1, "h")["eta_max_abs"]["value"] == 0.0


def test_run_capture_matches_capture_row_with_and_without_edits(bundle):
    adapter, base = bundle["adapter"], bundle["base"]
    run = Run(adapter, PROMPT)
    run.until("logits")
    caps = explain.captures_from_run(run)
    for layer in range(adapter.layers):
        for port, value in base.ports[layer].items():
            assert torch.equal(caps.ports[layer][port], value), (layer, port)
    for left, right in ((caps.values, base.values), (caps.keys, base.keys)):
        assert all(torch.equal(x, y) for x, y in zip(left, right))
    assert torch.equal(caps.logits, base.logits) and torch.equal(caps.normed, base.normed)
    assert explain.explain_margin(run, 4, 9) == explain.explain_margin(base, 4, 9)
    edited_run = Run(adapter, PROMPT)
    edited_run.until("L1.o")
    edited_run.set("L1.o", -1.0, "scale")
    edited_run.until("logits")
    expected = explain.capture_row(adapter, PROMPT, edits={1: [PortEdit("o", -1.0, "scale")]})
    got = explain.explain_margin(edited_run, 4, 9)
    want = explain.explain_margin(expected, 4, 9)
    assert got["terms"] == want["terms"] and got["native_margin"] == want["native_margin"]
    assert got["edits"] and got["edits"][1][0]["op"] == "scale"


def test_p19_readout_winner_certificate_and_pairwise_bound(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    final = adapter.backbone.norm
    weight = adapter.model.lm_head.weight
    h = finite.f64(base.ports[-1]["h_out"]).reshape(-1)
    res = finite.readout_logits(h, weight, final.weight, final.variance_epsilon,
                                native=base.logits.reshape(-1))
    assert res.eta_rel <= ETA_REL[bundle["dtype"]]
    note(case, "P19 readout_logits eta_rel", res.eta_rel)
    cert = finite.winner_certificate(h, weight, final.weight, final.variance_epsilon,
                                    bias_free=True)
    radius = float(cert.total)
    assert cert.authority == "bound" and cert.extras["unique"] and radius > 0
    generator = torch.Generator().manual_seed(19)
    agree = 0
    for scale in (0.5, 0.99, 2.0, 5.0, 20.0):
        for _ in range(10):
            delta = torch.randn(h.shape, generator=generator, dtype=torch.float64)
            delta = delta / delta.norm() * radius * scale
            out = finite.winner_certificate(h, weight, final.weight, final.variance_epsilon,
                                            delta=delta, bias_free=True).extras
            retained = out["argmax_after"] == cert.extras["winner"]
            assert out["condition_holds"] == retained
            if scale < 1:
                assert retained
            agree += 1
    note(case, "P19 certificate radius", radius)
    a, b = PAIRS[0]
    dy = torch.randn(h.shape, generator=generator, dtype=torch.float64)
    bound = finite.margin_error_bound(weight, a, b, dy)
    assert abs(bound.extras["actual"]) <= float(bound.total)
    assert agree == 50


# --------------------------------------------------------------------------- shared helpers


def forward64(model, tokens):
    """Float64 forward written from the weights only: embedding -> layers -> (final norm, head).

    Returns the final residual of the last row. Independent of mdb, saturn and HF modelling code:
    RMS norm, optional Qwen3 per-head q/k norm, half-split RoPE, causal GQA attention with
    contiguous kv groups, SwiGLU. Weights are the model's (bf16 weights are exact in float64).
    """
    cfg = model.config
    sd = {k: v.detach().to(torch.float64) for k, v in model.state_dict().items()}
    heads, kv = cfg.num_attention_heads, cfg.num_key_value_heads
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    eps, theta = cfg.rms_norm_eps, cfg.rope_parameters["rope_theta"]
    t = len(tokens)

    def rms(x, w):
        return w * x / torch.sqrt(eps + (x * x).mean(-1, keepdim=True))

    h = sd["model.embed_tokens.weight"][torch.tensor(list(tokens))]
    inv = theta ** (-2.0 * torch.arange(dh // 2, dtype=torch.float64) / dh)
    ang = torch.arange(t, dtype=torch.float64)[:, None] * inv[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]

    def rope(x):
        x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)
    for layer in range(cfg.num_hidden_layers):
        pre = f"model.layers.{layer}."
        a = rms(h, sd[pre + "input_layernorm.weight"])

        def proj(name, x, pre=pre):
            y = x @ sd[pre + f"self_attn.{name}_proj.weight"].T
            key = pre + f"self_attn.{name}_proj.bias"
            return y + sd[key] if key in sd else y

        q = proj("q", a).reshape(t, heads, dh)
        k = proj("k", a).reshape(t, kv, dh)
        v = proj("v", a).reshape(t, kv, dh)
        if pre + "self_attn.q_norm.weight" in sd:
            q = rms(q, sd[pre + "self_attn.q_norm.weight"])
            k = rms(k, sd[pre + "self_attn.k_norm.weight"])
        q, k = rope(q), rope(k)
        k, v = k.repeat_interleave(heads // kv, dim=1), v.repeat_interleave(heads // kv, dim=1)
        scores = torch.einsum("thd,shd->hts", q, k) / dh**0.5
        p = torch.softmax(scores.masked_fill(mask, float("-inf")), -1)
        ctx = torch.einsum("hts,shd->thd", p, v).reshape(t, heads * dh)
        r = h + proj("o", ctx)
        n = rms(r, sd[pre + "post_attention_layernorm.weight"])
        g = n @ sd[pre + "mlp.gate_proj.weight"].T
        u = n @ sd[pre + "mlp.up_proj.weight"].T
        h = r + (g * torch.sigmoid(g) * u) @ sd[pre + "mlp.down_proj.weight"].T
    return h[-1]


def ulp(x, dtype):
    """Spacing of ``dtype`` at ``x`` (written here, not imported): eps_dtype 2^floor(log2 |x|)."""
    x = abs(float(x))
    return 0.0 if x == 0 else float(torch.finfo(dtype).eps) * 2.0 ** math.floor(math.log2(x))


def edit_vector(caps, port, seed, size=0.5):
    g = torch.Generator().manual_seed(seed)
    ref = caps.ports[1][port]
    return (torch.randn(ref.shape, generator=g) * size).to(ref.dtype)


def margin_rho_c(adapter, caps, a, b):
    """rho^F, c = Gamma^F (W_a - W_b) from the final residual of ``caps``, written independently."""
    model = adapter.model
    gamma = model.model.norm.weight.detach().double()
    weight = model.lm_head.weight.detach().double()  # upcast first: bf16/fp32 differences round
    diff = weight[a] - weight[b]
    h = caps.ports[-1]["h_out"].detach().double().reshape(-1)
    rho = float((model.config.rms_norm_eps + (h * h).mean()) ** -0.5)
    return rho, gamma * diff


# --------------------------------------------------------------------------- fix 1


def test_h_and_h_out_edits_are_listed_supplied_terms_not_rounding(bundle):
    case, adapter, base, dtype = bundle["case"], bundle["adapter"], bundle["base"], bundle["dtype"]
    a, b = PAIRS[0]
    clean = explain.explain_margin(base, a, b)
    rr0 = labelled(clean["remainders"][0]["value"], "measured")
    assert [t["name"] for t in clean["terms"]] == ["embedding"] + [
        f"L{i}.{k}" for i in range(3) for k in ("attn", "mlp")]
    sizes = {}
    for layer_index, port in ((1, "h"), (2, "h_out"), (0, "h")):
        vector = edit_vector(base, port, 3 + layer_index)
        caps = explain.capture_row(adapter, PROMPT,
                                   edits={layer_index: [PortEdit(port, vector, "add")]})
        out = explain.explain_margin(caps, a, b)
        name = f"L{layer_index}.edit_{port}"
        edit = {t["name"]: t for t in out["terms"]}[name]
        # an add operand combined with the native value is derived-exact; the op is shown
        assert edit["value"]["authority"] == "derived-exact" and edit["op"] == "add"
        assert edit["projection"]["authority"] == "derived-exact"
        assert edit["port"] == f"L{layer_index}.{port}" and edit["layer"] == layer_index
        # independent: rho^F c^T (edited - original), original from the unedited row (upstream
        # of the edit nothing differs, so the pre-edit value equals the unedited capture)
        rho, c = margin_rho_c(adapter, caps, a, b)
        delta = finite.f64(caps.ports[layer_index][port]) - finite.f64(
            base.ports[layer_index][port])
        want = rho * float(c @ delta.reshape(-1))
        got = edit["value"]["value"]
        assert abs(got - want) <= 1e-9 * abs(want) + 1e-12, (name, got, want)
        supplied = rho * float(c @ finite.f64(vector).reshape(-1))
        # the supplied vector as realised by the port dtype (fp32: 1e-4 of the term, bf16 rounds
        # the cast and the add to 8 bits)
        assert abs(got - supplied) <= (1e-4 if dtype == "fp32" else 0.5) * abs(supplied)
        assert abs(got) > 1e-3  # a real edit, not a no-op
        sizes[name] = abs(got)
        # the remainder is the unedited rounding, with a state and a scale
        rr = out["remainders"][0]
        assert rr["name"] == "residual_rounding"
        rounding = labelled(rr["value"], "measured")
        assert rr["within_rounding"], (name, rounding, rr["scale"])
        if dtype == "fp32":
            assert abs(rounding) <= 1e-5, (name, rounding)
        assert labelled(rr["edits_removed"], "derived-exact") != 0.0 or layer_index == 0
        # embedding at layer 0 is the pre-edit h_0: its scale-free projection c^T h_0 is unchanged
        if layer_index == 0:
            emb = next(t for t in out["terms"] if t["name"] == "embedding")
            assert emb["projection"]["value"] == clean["terms"][0]["projection"]["value"]
        assert abs(out["closure"]["value"]) <= 1e-9 * max(
            abs(t["value"]["value"]) for t in out["terms"])
        assert out["native_margin"]["value"] != clean["native_margin"]["value"]
        assert out["all_within_rounding"] is True
        assert not walk(out), walk(out)
        note(case, f"P19 edit term {name}", abs(got))
        note(case, f"P19 residual_rounding with {name}", abs(rounding))
        note(case, f"P19 residual_rounding scale with {name}", rr["scale"]["value"])
    # both edits in one row
    both = explain.capture_row(adapter, PROMPT, edits={
        1: [PortEdit("h", edit_vector(base, "h", 4), "add")],
        2: [PortEdit("h_out", edit_vector(base, "h_out", 5), "add")]})
    out = explain.explain_margin(both, a, b)
    assert {"L1.edit_h", "L2.edit_h_out"} <= {t["name"] for t in out["terms"]}
    assert out["remainders"][0]["within_rounding"] and out["all_within_rounding"]
    # an o-scale edit stays an ordinary component edit (no edit_ term)
    flipped = explain.capture_row(adapter, PROMPT, edits={1: [PortEdit("o", -1.0, "scale")]})
    names = [t["name"] for t in explain.explain_margin(flipped, a, b)["terms"]]
    assert not any("edit_" in n for n in names)


# --------------------------------------------------------------------------- fix 2


def test_remainders_within_rounding_vs_an_independent_float64_forward(bundle):
    case, adapter, base, dtype = bundle["case"], bundle["adapter"], bundle["base"], bundle["dtype"]
    model, tdtype = adapter.model, DTYPES[dtype]
    h64 = forward64(model, PROMPT)
    h_nat = finite.f64(base.ports[-1]["h_out"]).reshape(-1)
    # the test's own forward reproduces the native model to the dtype's rounding
    assert float((h64 - h_nat).abs().max()) <= ETA_REL[dtype] * float(h_nat.abs().max())
    note(case, "independent f64 forward vs native h_L rel",
         float((h64 - h_nat).abs().max()) / float(h_nat.abs().max()))
    weight = finite.f64(model.lm_head.weight)
    gamma = finite.f64(model.model.norm.weight)
    eps = model.config.rms_norm_eps
    x_nat = finite.f64(base.normed).reshape(-1)
    logits = finite.f64(base.logits).reshape(-1)
    emb = finite.f64(model.model.embed_tokens.weight)[PROMPT[-1]]
    o_m = sum(finite.f64(p["o"]).reshape(-1) + finite.f64(p["m"]).reshape(-1) for p in base.ports)
    rho = float((eps + (h_nat * h_nat).mean()) ** -0.5)
    eps_dt = float(torch.finfo(tdtype).eps)

    def half_ulp(z):
        z = finite.f64(z).reshape(-1).abs()
        return torch.where(z > 0, 0.5 * eps_dt * torch.exp2(torch.floor(torch.log2(
            z.clamp_min(1e-300)))), torch.zeros_like(z))

    stream = sum((half_ulp(p["r"]) + half_ulp(p["h_out"]) for p in base.ports),
                 torch.zeros(weight.shape[1], dtype=torch.float64))
    sixteen = tdtype in (torch.bfloat16, torch.float16)
    unit = 2.0 ** -24 if sixteen else eps_dt / 2
    d = weight.shape[1]
    gamma_d = d * unit / (1 - d * unit)
    for a, b in PAIRS:
        out = explain.explain_margin(base, a, b)
        diff = weight[a] - weight[b]
        c = gamma * diff
        # remainders re-derived from the weights, natives and the independent forward
        want = {
            "final_norm": float(diff @ x_nat - diff @ (rho * gamma * h_nat)),
            "head": float(logits[a] - logits[b] - diff @ x_nat),
            "residual_rounding": rho * float(c @ (h_nat - emb - o_m)),
        }
        scale = {
            "final_norm": float(diff.abs().sum()) * ulp(x_nat.abs().max(), tdtype),
            # per-coordinate true bound: half ulp of every residual add result, weighted by |c|
            "residual_rounding": rho * float(c.abs() @ stream),
            "head": sum(gamma_d * float((weight[j] * x_nat).abs().sum())
                        + (0.5 * ulp(logits[j].abs(), tdtype) if sixteen else 0.0)
                        for j in (a, b)),
        }
        ks = {"final_norm": 2, "head": 1, "residual_rounding": 1}
        got = {r["name"]: r for r in out["remainders"]}
        for name in want:
            value = got[name]["value"]["value"]
            assert abs(value - want[name]) <= 1e-9 * scale[name], (name, value, want[name])
            assert abs(got[name]["scale"]["value"] - scale[name]) <= 1e-9 * scale[name]
            assert got[name]["scale"]["authority"] == (
                "derived-exact" if name == "final_norm" else "bound")
            assert got[name]["k"] == ks[name] and got[name]["within_rounding"] is True
            assert got[name]["heuristic"] is (name == "final_norm")
            assert got[name]["scale_formula"]
            # the stated multiple of the bound, on the independently recomputed numbers
            assert abs(want[name]) <= ks[name] * scale[name], (name, want[name], scale[name])
            note(case, f"P19 {name} / scale", abs(want[name]) / scale[name])
        assert out["remainders"][0]["premises"] and out["remainders"][3]["premises"]
        assert out["all_within_rounding"]
        ref64 = float(diff @ (rho * gamma * h_nat))
        assert abs(out["float64_margin"]["value"] - ref64) <= 1e-12 * (abs(ref64) + 1)
        # whole-model check against the independent forward's margin (rounding of every stage)
        m64 = float(diff @ (gamma * h64 / math.sqrt(eps + float((h64 * h64).mean()))))
        assert abs(out["native_margin"]["value"] - m64) <= ETA_REL[dtype] * float(logits.abs().max())


# --------------------------------------------------------------------------- fix 3 / 5 / 8 / 9


def test_explain_outputs_are_all_labelled_numbers_with_states(bundle):
    adapter, base = bundle["adapter"], bundle["base"]
    out = explain.explain_margin(base, *PAIRS[0], first_order=True)
    assert out["state"] == "resolved" and out["reason"] is None
    assert not walk(out), walk(out)
    assert is_labelled(out["listed_sum"]) and out["listed_sum"]["authority"] == "derived-exact"
    program = program_for(adapter)
    for layer in range(adapter.layers):
        for port in program.names:
            info = explain.explain_port(base, layer, port)
            view = {k: v for k, v in info.items() if k not in ("reconstruction", "native", "result")}
            assert not unlabelled_numbers(view), (layer, port, unlabelled_numbers(view))
            assert info["state"] == "resolved" and info["reason"] is None
            assert info["state"] in STATE
            summary = info["result"].summary()
            assert not unlabelled_numbers(summary), (layer, port, unlabelled_numbers(summary))
            assert summary["authority"] in AUTHORITY


def test_explain_port_s_is_labelled_against_the_replica(bundle):
    adapter, base = bundle["adapter"], bundle["base"]
    program = program_for(adapter)
    for layer in range(adapter.layers):
        for port in program.names:
            info = explain.explain_port(base, layer, port)
            if port == "s":
                assert info["against"] == "replica"
                assert info["eta_rel"]["authority"] == "derived-exact"
                assert info["eta_max_abs"]["authority"] == "derived-exact"
            else:
                assert info["against"] == "native", port
                assert info["eta_rel"]["authority"] == "measured", port


def lm_head_with_bias(family, dtype, seed=5):
    model = tiny(family, dtype, seed)
    head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=True)
    g = torch.Generator().manual_seed(77)
    with torch.no_grad():
        head.weight.copy_(model.lm_head.weight)
        head.bias.copy_(torch.randn(head.bias.shape, generator=g))
    model.lm_head = head.to(DTYPES[dtype])
    return model


@pytest.mark.parametrize("family,dtype", [("qwen2", "fp32"), ("qwen3", "bf16")])
def test_head_bias_is_a_listed_term(family, dtype):
    torch.set_num_threads(1)
    adapter = DecoderAdapter(lm_head_with_bias(family, dtype), granularity="operation")
    caps = explain.capture_row(adapter, PROMPT)
    bias = adapter.model.lm_head.bias.detach().double()
    for a, b in PAIRS:
        out = explain.explain_margin(caps, a, b)
        term = {t["name"]: t for t in out["terms"]}["head_bias"]
        assert term["value"]["authority"] == "derived-exact"
        assert term["value"]["value"] == float(bias[a] - bias[b]) and abs(term["value"]["value"]) > 1e-3
        assert abs(out["closure"]["value"]) <= 1e-9 * max(abs(t["value"]["value"])
                                                          for t in out["terms"])
        head = {r["name"]: r for r in out["remainders"]}["head"]
        assert head["within_rounding"], head
        assert out["all_within_rounding"]
        assert not walk(out)
    # without the bias term the head remainder would absorb it
    plain = explain.explain_margin(explain.capture_row(
        DecoderAdapter(tiny(family, dtype), granularity="operation"), PROMPT), *PAIRS[0])
    assert "head_bias" not in {t["name"] for t in plain["terms"]}


def test_unmodelled_logit_softcap_and_rope_are_refused(bundle):
    adapter, base = bundle["adapter"], bundle["base"]
    config = adapter.model.config
    for name, value in (("final_logit_softcapping", 30.0), ("logit_scale", 0.5),
                        ("logits_scaling", 2.0)):
        had = hasattr(config, name)
        old = getattr(config, name, None)
        setattr(config, name, value)
        try:
            out = explain.explain_margin(base, *PAIRS[0])
        finally:
            setattr(config, name, old) if had else delattr(config, name)
        assert out["state"] == "refused" and name in out["reason"], out
        assert "terms" not in out and not walk(out)
    assert explain.explain_margin(base, *PAIRS[0])["state"] == "resolved"
    params = dict(config.rope_parameters)
    flavours = [({**params, "rope_type": "linear", "factor": 2.0}, "rope type"),
                ({**params, "rope_type": "yarn", "factor": 4.0}, "rope type"),
                ({**params, "partial_rotary_factor": 0.5}, "partial_rotary_factor")]
    try:
        for new, text in flavours:
            config.rope_parameters = new
            for port in ("qr", "kr"):
                info = explain.explain_port(base, 1, port)
                assert info["state"] == "refused" and text in info["reason"], info
                assert "eta_rel" not in info and info["proof"] == "P04"
            assert explain.explain_port(base, 1, "s")["state"] == "resolved"
        config.rope_parameters = params
        config.partial_rotary_factor = 0.5
        assert explain.explain_port(base, 1, "qr")["state"] == "refused"
    finally:
        config.rope_parameters = params
        if hasattr(config, "partial_rotary_factor"):
            del config.partial_rotary_factor
    assert explain.explain_port(base, 1, "qr")["state"] == "resolved"


# --------------------------------------------------------------------------- fix 6


def test_run_edits_are_keyed_by_row_and_layer(bundle):
    adapter, base = bundle["adapter"], bundle["base"]
    a, b = PAIRS[0]
    clean = Run(adapter, PROMPT)
    clean.until("t8.logits")
    clean_caps = explain.captures_from_run(clean)
    assert clean_caps.edits == {} and clean_caps.edits_history == {}
    clean_out = explain.explain_margin(clean_caps, a, b)
    assert "edits_history" not in clean_out

    def run_with(edits, recompute=False):
        run = Run(adapter, PROMPT)
        for target, value, op in edits:
            run.until(target)
            run.set(target, value, op)
        run.until("t8.logits")
        return run, explain.captures_from_run(run, recompute_originals=recompute)

    # edit at t6 only: history, not this row
    _, caps6 = run_with([("t6.L1.o", -1.0, "scale")])
    assert caps6.position == 8 and caps6.edits == {} and caps6.edits_this_row == {}
    assert list(caps6.edits_history) == [(6, 1)]
    out6 = explain.explain_margin(caps6, a, b)
    assert out6["edits"] == {} and out6["edits_history"]["count"] == 1
    assert out6["edits_history"]["entries"][0]["row"] == 6
    assert [t["name"] for t in out6["terms"]] == [t["name"] for t in clean_out["terms"]]
    assert not any("edit_" in t["name"] for t in out6["terms"])
    assert not walk(out6)
    fo = explain.explain_margin(caps6, a, b, first_order=True)["first_order"]
    assert fo["state"] == "refused" and re.search(r"unedited.*earlier rows", fo["reason"])
    # t6 and t7 at the same layer: two history entries, never merged into this row
    _, caps67 = run_with([("t6.L1.o", -1.0, "scale"), ("t7.L1.o", 0.5, "scale")])
    assert caps67.edits == {} and sorted(caps67.edits_history) == [(6, 1), (7, 1)]
    out67 = explain.explain_margin(caps67, a, b)
    assert out67["edits_history"]["count"] == 2 and out67["edits"] == {}
    fo = explain.explain_margin(caps67, a, b, first_order=True)["first_order"]
    assert fo["state"] == "refused" and "earlier rows" in fo["reason"]
    # an edit in the explained row: this-row edits, no history
    _, caps8 = run_with([("t8.L1.o", -1.0, "scale")])
    assert list(caps8.edits) == [1] and list(caps8.edits_this_row) == [(8, 1)]
    assert caps8.edits_history == {}
    out8 = explain.explain_margin(caps8, a, b)
    assert out8["edits"][1][0]["op"] == "scale" and "edits_history" not in out8
    fo = explain.explain_margin(caps8, a, b, first_order=True)["first_order"]
    assert fo["state"] == "refused" and re.search(r"unedited.*this row", fo["reason"])
    # both: this row and history
    _, caps_both = run_with([("t6.L1.o", -1.0, "scale"), ("t8.L2.m", 2.0, "scale")])
    out_both = explain.explain_margin(caps_both, a, b)
    assert list(out_both["edits"]) == [2] and out_both["edits_history"]["count"] == 1
    fo = explain.explain_margin(caps_both, a, b, first_order=True)["first_order"]
    assert fo["state"] == "refused" and re.search(r"this row.*earlier rows", fo["reason"])
    # an h edit in the explained row through a Run: without recompute the term is unresolved
    vector = edit_vector(clean_caps, "h", 11)
    _, caps_nr = run_with([("t8.L1.h", vector, "add")])
    assert caps_nr.originals_missing == [(1, "h")] and caps_nr.originals[1] == {}
    out_nr = explain.explain_margin(caps_nr, a, b)
    reason = "pre-edit value not retained; pass recompute_originals=True"
    assert out_nr["state"] == "unresolved" and out_nr["reason"] == reason
    shell = {t["name"]: t for t in out_nr["terms"]}["L1.edit_h"]
    assert shell["state"] == "unresolved" and shell["reason"] == reason and "value" not in shell
    for key in ("listed_sum", "closure", "closure_rel"):
        assert out_nr[key] == {"state": "unresolved", "reason": reason}
    rr_nr = out_nr["remainders"][0]
    assert rr_nr["state"] == "unresolved" and "value" not in rr_nr  # never folded into rounding
    assert not walk(out_nr)
    # with recompute (opt-in) it is booked like capture_row, and the run is left untouched
    run_h, caps_h = run_with([("t8.L1.h", vector, "add")], recompute=True)
    assert caps_h.originals_source[(1, "h")] == "Run.original(recompute=True)"
    assert caps_h.originals[1]["h"].shape == caps_h.ports[1]["h"].shape
    out_h = explain.explain_margin(caps_h, a, b)
    edit = {t["name"]: t for t in out_h["terms"]}["L1.edit_h"]
    assert edit["value"]["authority"] == "derived-exact" and edit["op"] == "add"
    rho, c = margin_rho_c(adapter, caps_h, a, b)
    delta = finite.f64(caps_h.ports[1]["h"]) - finite.f64(clean_caps.ports[1]["h"])
    want = rho * float(c @ delta.reshape(-1))
    assert abs(edit["value"]["value"] - want) <= 1e-9 * abs(want) + 1e-12
    rr = out_h["remainders"][0]
    assert rr["within_rounding"] and abs(rr["value"]["value"]) <= max(
        1e-5 if bundle["dtype"] == "fp32" else 1.0, 0.0)
    # the same row through capture_row gives the same terms
    direct = explain.capture_row(adapter, PROMPT + [caps_h.tokens[-1]] if False else list(
        caps_h.tokens), edits={1: [PortEdit("h", vector, "add")]})
    assert {t["name"] for t in explain.explain_margin(direct, a, b)["terms"]} == {
        t["name"] for t in out_h["terms"]}


# --------------------------------------------------------------------------- fix 10

FD_TOLERANCE = 5e-4  # measured worst |first_order - native FD|: 2.2e-4 (audit kit), 1.6e-4 (here)


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_first_order_column_matches_native_central_finite_differences(family):
    """lambda^T xi vs d/de M(port *= 1 + e) at e = 0 on the native model (fixed past KV).

    Richardson on central differences with e = 5e-3 and 1e-2 in fp32. A pullback with the adjoint
    one layer off, or the wrong layer's MLP, moves rows by O(0.1-1); the tolerance sits at about
    twice the measured worst difference.
    """
    torch.set_num_threads(1)
    adapter = DecoderAdapter(tiny(family, "fp32"), granularity="operation")
    base = explain.capture_row(adapter, PROMPT)
    a, b = PAIRS[0]
    out = explain.explain_margin(base, a, b, first_order=True)
    rows = {r["name"]: r["value"]["value"] for r in out["first_order"]["rows"]}

    def margin(layer, port, e):
        caps = explain.capture_row(adapter, PROMPT, edits={layer: [PortEdit(port, 1.0 + e,
                                                                            "scale")]})
        lg = caps.logits.reshape(-1).double()
        return float(lg[a] - lg[b])

    def deriv(layer, port):
        def cd(e):
            return (margin(layer, port, e) - margin(layer, port, -e)) / (2 * e)

        return (4 * cd(5e-3) - cd(1e-2)) / 3

    worst = 0.0
    sites = [("embedding", 0, "h")]
    for layer in range(adapter.layers):
        sites += [(f"L{layer}.attn", layer, "o"), (f"L{layer}.mlp", layer, "m")]
    for name, layer, port in sites:
        diff = abs(rows[name] - deriv(layer, port))
        assert diff <= FD_TOLERANCE, (family, name, rows[name], diff)
        assert abs(rows[name]) > 1e-3  # not a vacuous zero column
        worst = max(worst, diff)
    note(f"{family}-fp32", "first-order vs native FD abs diff", worst)


def test_winner_certificate_is_tight_along_the_worst_direction(bundle):
    adapter, base = bundle["adapter"], bundle["base"]
    final = adapter.backbone.norm
    weight = finite.f64(adapter.model.lm_head.weight)
    gamma = finite.f64(final.weight)
    eps = final.variance_epsilon
    h = finite.f64(base.ports[-1]["h_out"]).reshape(-1)
    cert = finite.winner_certificate(h, adapter.model.lm_head.weight, final.weight, eps, bias_free=True)
    radius = float(cert.total)
    # independent radius: min over competitors of margin / (rho ||Gamma (W_a - W_b)||)
    rho = float((eps + (h * h).mean()) ** -0.5)
    logits = rho * (gamma * h) @ weight.T
    win = int(logits.argmax())
    assert cert.extras["winner"] == win and cert.state == "resolved"
    others = [i for i in range(weight.shape[0]) if i != win]
    ratios = [(float(logits[win] - logits[i]) / (rho * float((gamma * (weight[win] - weight[i])).norm()))
               , i) for i in others]
    want, worst = min(ratios)
    assert abs(radius - want) <= 1e-12 * want
    c = gamma * (weight[win] - weight[worst])
    direction = -c / c.norm()  # moves h straight toward the worst competitor
    inside = finite.winner_certificate(h, adapter.model.lm_head.weight, final.weight, eps,
                                       delta=0.999 * radius * direction, bias_free=True).extras
    beyond = finite.winner_certificate(h, adapter.model.lm_head.weight, final.weight, eps,
                                       delta=1.001 * radius * direction, bias_free=True).extras
    assert inside["argmax_after"] == win and inside["condition_holds"] is True
    assert beyond["argmax_after"] != win and beyond["condition_holds"] is False
    # the competitor that takes over is the worst one when only that boundary is crossed
    assert beyond["argmax_after"] == worst
    note(bundle["case"], "P19 certificate tight radius", radius)


def test_readout_margin_gradient_matches_autograd(bundle):
    adapter, base = bundle["adapter"], bundle["base"]
    final = adapter.backbone.norm
    weight = finite.f64(adapter.model.lm_head.weight)
    gamma, eps = finite.f64(final.weight), float(final.variance_epsilon)
    h = finite.f64(base.ports[-1]["h_out"]).reshape(-1)
    a, b = PAIRS[0]
    res = finite.readout_margin(h, weight, gamma, eps, a, b)

    def margin(x):
        normed = gamma * x / torch.sqrt(eps + (x * x).mean(-1, keepdim=True))
        return (weight[a] - weight[b]) @ normed

    auto = torch.autograd.functional.jacobian(margin, h)
    err = float((res.extras["gradient"] - auto).abs().max() / auto.abs().max())
    assert err <= 1e-12, err
    assert float(auto.abs().max()) > 1e-3
    # a transposed Jacobian would give a different vector: the check is not vacuous
    wrong = finite.rms_jacobian(h, gamma, eps).extras["jacobian"] @ (weight[a] - weight[b])
    assert float((wrong - auto).abs().max() / auto.abs().max()) > 1e-3
    note(bundle["case"], "P19 readout gradient vs autograd rel", err)


def test_explain_rope_rotate_against_native_qr_and_kr(bundle):
    adapter, base, dtype = bundle["adapter"], bundle["base"], bundle["dtype"]
    for layer in range(adapter.layers):
        for port in ("qr", "kr"):
            info = explain.explain_port(base, layer, port)
            assert info["proof"] == "P04" and info["function"] == "rope_rotate"
            assert info["against"] == "native"
            assert info["eta_rel"]["value"] <= ETA_REL[dtype]
            assert torch.isfinite(info["result"].eta).all()


# --------------------------------------------------------------------------- public engine API


def _run_to(adapter, edits, recompute=True):
    """Run to t8.logits with edits ``(target_to_reach, edit_target, value, op, kwargs)``."""
    run = Run(adapter, PROMPT)
    for reach, target, value, op, kwargs in edits:
        run.until(reach)
        run.set(target, value, op, **kwargs)
    run.until("t8.logits")
    return run, explain.captures_from_run(run, recompute_originals=recompute)


def test_masked_h_edit_is_one_term_and_unselected_elements_contribute_zero(bundle):
    adapter, dtype = bundle["adapter"], bundle["dtype"]
    a, b = PAIRS[0]
    clean = Run(adapter, PROMPT)
    clean.until("t8.logits")
    clean_caps = explain.captures_from_run(clean)
    ref = clean_caps.ports[1]["h"]
    mask = torch.zeros(ref.shape, dtype=torch.bool)
    mask.reshape(-1)[3:11] = True
    vector = edit_vector(clean_caps, "h", 21)
    run, caps = _run_to(adapter, [("t8.L1.h", "t8.L1.h", vector, "add",
                                   {"mask": mask, "selection": "dims 3-10"})])
    state_before = run.where
    delta = finite.f64(caps.ports[1]["h"]) - finite.f64(clean_caps.ports[1]["h"])
    assert (delta[~mask] == 0).all() and (delta[mask] != 0).any()  # unselected bit-identical
    assert caps.originals_source[(1, "h")] == "Run.original(recompute=True)"
    assert torch.equal(caps.originals[1]["h"], clean_caps.ports[1]["h"])
    out = explain.explain_margin(caps, a, b)
    edit = {t["name"]: t for t in out["terms"]}["L1.edit_h"]
    rho, c = margin_rho_c(adapter, caps, a, b)
    want = rho * float(c @ delta.reshape(-1))
    assert abs(edit["value"]["value"] - want) <= 1e-9 * abs(want) + 1e-12
    only_selected = rho * float(c[mask.reshape(-1)] @ delta.reshape(-1)[mask.reshape(-1)])
    assert abs(edit["value"]["value"] - only_selected) <= 1e-9 * abs(want) + 1e-12
    # the edit record is labelled and keeps its structure; the masked count is a structural int
    record = out["edits"][1][0]
    assert record["mask_count"] == 8 and record["selection"] == "dims 3-10"
    assert record["changed_fraction"]["authority"] == "measured"
    assert record["cast_max_abs_error"]["authority"] == "measured"
    assert record["value"]["shape"] and record["op"] == "add"
    assert not walk(out) and run.where == state_before
    assert out["remainders"][0]["within_rounding"]


def test_edit_term_label_follows_the_op_replace_supplied_add_scale_derived(bundle):
    adapter, dtype = bundle["adapter"], bundle["dtype"]
    a, b = PAIRS[0]
    clean = Run(adapter, PROMPT)
    clean.until("t8.logits")
    clean_caps = explain.captures_from_run(clean)
    vector = edit_vector(clean_caps, "h", 31)
    replacement = clean_caps.ports[1]["h"] + vector
    scale = 1.25
    out_replacement = clean_caps.ports[2]["h_out"] + vector
    cases = [("replace", replacement, out_replacement, "supplied"),
             ("add", vector, vector, "derived-exact"), ("scale", scale, scale, "derived-exact")]
    for op, value, out_value, authority in cases:
        # h at layer 1 (original via Run.original) and h_out at layer 2 (r + m native add)
        run, caps = _run_to(adapter, [("t8.L1.h", "t8.L1.h", value, op, {}),
                                      ("t8.L2.m", "t8.L2.h_out", out_value, op, {})])
        out = explain.explain_margin(caps, a, b)
        terms = {t["name"]: t for t in out["terms"]}
        for name in ("L1.edit_h", "L2.edit_h_out"):
            assert terms[name]["value"]["authority"] == authority, (op, name)
            assert terms[name]["projection"]["authority"] == authority
            assert terms[name]["op"] == op
            assert abs(terms[name]["value"]["value"]) > 1e-4
        assert out["state"] == "resolved" and out["remainders"][0]["within_rounding"]
        assert abs(out["closure"]["value"]) <= 1e-9 * max(abs(t["value"]["value"])
                                                          for t in out["terms"])
        assert out["listed_sum"]["authority"] == "derived-exact"
        assert out["listed_sum"]["includes_edits"] == ["L1.edit_h", "L2.edit_h_out"]
        assert ("includes_supplied" in out["listed_sum"]) == (op == "replace")
        # the h_out original is the native add r + m of the captured r and m
        assert torch.equal(caps.originals[2]["h_out"],
                           caps.ports[2]["r"] + caps.ports[2]["m"])
        assert caps.originals_source[(2, "h_out")].startswith("native add r + m")
        assert not walk(out)
    # an h_out edit of a committed layer needs the opt-in too
    _, caps_nr = _run_to(adapter, [("t8.L2.m", "t8.L2.h_out", vector, "add", {})],
                         recompute=False)
    assert caps_nr.originals_missing == [(2, "h_out")]
    out_nr = explain.explain_margin(caps_nr, a, b)
    assert out_nr["state"] == "unresolved" and "recompute_originals=True" in out_nr["reason"]


def test_explain_port_declares_p04_only_for_rope_ports(bundle):
    for layer in range(bundle["adapter"].layers):
        for port in ("qr", "kr"):
            info = explain.explain_port(bundle["base"], layer, port)
            assert info["proofs_declared"] == ["P04"], info["proofs_declared"]


def test_analysis_has_no_private_access_to_a_run_object():
    """AST: no ``run._x`` / ``getattr(run, "_x")`` anywhere in ``mdb/analysis``."""
    import ast
    import pathlib

    import mdb.analysis as analysis

    root = pathlib.Path(analysis.__file__).parent
    offences = []

    def chain_root(node):
        while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
            node = node.value if not isinstance(node, ast.Call) else node.func
        return node.id if isinstance(node, ast.Name) else None

    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith("_") \
                    and not node.attr.startswith("__") and chain_root(node.value) in (
                        "run", "source"):
                offences.append(f"{path.name}:{node.lineno} .{node.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in ("getattr", "setattr", "hasattr") and len(node.args) >= 2 \
                    and isinstance(node.args[0], ast.Name) and node.args[0].id == "run" \
                    and isinstance(node.args[1], ast.Constant) \
                    and str(node.args[1].value).startswith("_"):
                offences.append(f"{path.name}:{node.lineno} {node.func.id}(run, ...)")
    assert not offences, offences
    # the checker bites: a synthetic offence is found
    probe = ast.parse("def f(run):\n    return run._trail[0]\n")
    assert any(isinstance(n, ast.Attribute) and n.attr == "_trail" and chain_root(n.value) == "run"
               for n in ast.walk(probe))


@pytest.mark.skip(reason=(
    "bf16 first-order vs native FD is not discriminating: measured worst |first_order - FD| "
    "(Richardson, tiny seed-5 models, margin (4, 9)) is 1.28 / 0.60 at step 0.1 and 0.24 / 0.37 "
    "at step 0.25 (qwen2 / qwen3), against row magnitudes up to 3.2. The margin's bf16 ulp is "
    "~1.6e-2 and the step must grow past the linear range to beat it, so a tolerance that "
    "passes the correct column (>= 0.4) also passes mutants that move rows by 0.5-1. "
    "Kept fp32-only; script: scratchpad/fix-analysis/fd_bf16.py"))
def test_first_order_column_vs_native_fd_bf16():
    raise AssertionError("skipped: see reason")


# --------------------------------------------------------------------------- analysis round 3


def _edit_value(ref, op, masked, seed):
    g = torch.Generator().manual_seed(seed)
    if op == "scale":
        return 1.7
    noise = (0.3 * torch.randn(ref.shape, generator=g)).to(ref.dtype)
    return ref + noise if op == "replace" else noise


def test_margin_terms_take_labels_and_markers_from_the_value_they_read(bundle):
    """Fix 2: each edited term's authority/op/edit marker follows read_info of the port it reads;
    a native term carries none; closure is unchanged by the labelling."""
    adapter, base = bundle["adapter"], bundle["base"]
    a, b = PAIRS[0]
    pos = base.position
    for port, term_name in (("o", "L1.attn"), ("m", "L1.mlp"), ("r", "L1.attn")):
        for op in ("replace", "add", "scale"):
            for masked in (False, True):
                ref = base.ports[1][port]
                mask = None
                if masked:
                    mask = torch.zeros(ref.shape, dtype=torch.bool)
                    mask.reshape(-1)[::2] = True
                edit = PortEdit(port, _edit_value(ref, op, masked, 3), op, mask=mask)
                caps = explain.capture_row(adapter, PROMPT, edits={1: [edit]})
                out = explain.explain_margin(caps, a, b)
                terms = {t["name"]: t for t in out["terms"]}
                term = terms[term_name]
                tag = (port, op, masked)
                if port == "r":
                    assert term["kind"] in ("realisation",) and term["edit"] == [f"t{pos}.L1.r"]
                    # an exactly realised r is a derived-exact o; an inexact one is not supplied
                    assert term["value"]["authority"] in ("derived-exact", "measured"), tag
                else:
                    want = "supplied" if op == "replace" and not masked else "derived-exact"
                    assert term["value"]["authority"] == want, tag
                    assert term["projection"]["authority"] == want, tag
                    assert term["kind"] == "edit" and term["edit"] == [f"t{pos}.L1.{port}"], tag
                    assert term["ops"] == [op] and term["op"] == op, tag
                # every other layer's terms stay native: no marker
                for name, other in terms.items():
                    if name != term_name and name.startswith("L") and "edit_" not in name:
                        assert "edit" not in other, (tag, name)
                listed = out["listed_sum"]
                assert listed["authority"] == "derived-exact" and term_name in listed[
                    "includes_edits"], tag
                assert ("includes_supplied" in listed) == (term["value"]["authority"] == "supplied")
                assert abs(out["closure"]["value"]) <= 1e-14, (tag, out["closure"])
                assert not walk(out)
    # a Run-built capture reads the same labels through Run.read_info
    run = Run(adapter, PROMPT)
    run.until("t8.L1.o")
    run.set("t8.L1.o", -1.0, "scale")
    run.until("t8.L2.m")
    run.set("t8.L2.m", torch.zeros(base.ports[2]["m"].shape, dtype=base.ports[2]["m"].dtype),
            "replace")
    run.until("t8.logits")
    caps = explain.captures_from_run(run)
    assert caps.infos and caps.infos[1]["o"]["ops"] == ["scale"]
    terms = {t["name"]: t for t in explain.explain_margin(caps, a, b)["terms"]}
    assert terms["L1.attn"]["value"]["authority"] == "derived-exact"
    assert terms["L1.attn"]["edit"] == ["t8.L1.o"] and terms["L1.attn"]["op"] == "scale"
    assert terms["L2.mlp"]["value"]["authority"] == "supplied"
    assert terms["L2.mlp"]["edit"] == ["t8.L2.m"]
    out = explain.explain_margin(caps, a, b)
    assert out["listed_sum"]["includes_supplied"] == ["L2.mlp"]
    assert out["listed_sum"]["includes_edits"] == ["L1.attn", "L2.mlp"]


def test_explain_port_never_books_an_edit_as_a_rounding_remainder(bundle):
    """Fix 1: an edit's displacement is not eta; the native original keeps its own eta."""
    adapter, base = bundle["adapter"], bundle["base"]
    scale = float(finite.f64(base.logits).abs().max())
    cases = [("o", PortEdit("o", base.ports[1]["o"] * 3.0 + 1.0, "replace"), "supplied", "edit"),
             ("m", PortEdit("m", 2.0, "scale"), "derived-exact", "edit"),
             ("h_out", PortEdit("h_out", edit_vector(base, "h_out", 5), "add"), "derived-exact",
              "edit"),
             ("o", PortEdit("r", edit_vector(base, "r", 6), "replace"), None, "realisation"),
             ("r", PortEdit("r", edit_vector(base, "r", 6), "replace"), None, "realisation")]
    for port, edit, authority, kind in cases:
        caps = explain.capture_row(adapter, PROMPT, edits={1: [edit]})
        info = explain.explain_port(caps, 1, port)
        tag = (port, edit.port, edit.op)
        assert info["state"] == "resolved" and info["value_kind"] == kind, tag
        assert "eta_max_abs" not in info and "eta_rel" not in info, tag
        assert "eta_max_abs" not in info["result"].summary(), tag
        if authority is not None:
            assert info["authority"] == authority, tag
        assert info["authority"] in AUTHORITY
        assert info["edit"]["ops"] == [edit.op] and info["edit"]["by"] == edit.port
        delta = labelled(info["edit_delta_max_abs"], "measured")
        original = labelled(info["original"]["eta_max_abs"], "measured")
        assert original <= ETA_REL[bundle["dtype"]] * max(scale, 1.0) * 10, tag
        if port == "r":
            # r is realised THROUGH o, so r = h + o_realised: displacement from its own operands
            # is rounding-sized; the edit shows up as o (checked above) and in the original
            assert delta <= ETA_REL[bundle["dtype"]] * max(scale, 1.0) * 10, tag
        else:
            assert delta > 0.1 and original < delta / 5, tag  # displacement, not a remainder
        assert not walk({k: v for k, v in info.items()
                         if k not in ("reconstruction", "native", "result")})
    # native value downstream of an edit: eta as before, plus under_edits
    caps = explain.capture_row(adapter, PROMPT, edits={0: [PortEdit("o", -1.0, "scale")]})
    info = explain.explain_port(caps, 1, "a")
    assert info["value_kind"] == "native" and "eta_max_abs" in info
    assert info["eta_max_abs"]["authority"] == "measured"
    assert [(e["layer"], e["port"], e["op"]) for e in info["under_edits"]] == [(0, "o", "scale")]
    assert explain.explain_port(base, 1, "a")["under_edits"] == []


def test_explain_port_replica_labels_agree_and_a_failed_check_refuses(bundle):
    """Fix 5 and the replica rules: one label for s everywhere; failed check -> refused;
    check not run -> resolved, derived-exact, unchecked (replica_equal None, said in a note)."""
    import dataclasses

    base = bundle["base"]
    info = explain.explain_port(base, 1, "s")
    top = info["eta_max_abs"]
    inner = info["result"].summary()["eta_max_abs"]
    assert top["authority"] == inner["authority"] == "derived-exact"
    assert info["against"] == "replica" and info["value_kind"] == "replica"
    assert any("replica" in n and "derived-exact" in n for n in info["notes"])
    assert explain.explain_port(base, 1, "p")["eta_max_abs"]["authority"] == "measured"
    unchecked = [dict(c) for c in base.checks]
    unchecked[1].pop("s", None)
    got = explain.explain_port(dataclasses.replace(base, checks=unchecked), 1, "s")
    assert got["state"] == "resolved" and got["replica_equal"] is None
    assert got["replica_checked"] is False and any("not run" in n for n in got["notes"])
    assert got["eta_max_abs"]["authority"] == "derived-exact"
    failed = [dict(c) for c in base.checks]
    failed[1]["s"] = {"equal": False, "max_abs_diff": 0.25}
    got = explain.explain_port(dataclasses.replace(base, checks=failed), 1, "s")
    assert got["state"] == "refused" and "replica check failed" in got["reason"]
    assert "eta_max_abs" not in got and got["against"] == "replica"


def _perturbed(base, layer, port, vec):
    """Copy of ``base`` whose captured ``L{layer}.{port}`` is shifted by ``vec`` (a wrong term)."""
    import dataclasses

    ports = [dict(p) for p in base.ports]
    ports[layer][port] = (finite.f64(ports[layer][port]).reshape(-1) + vec).reshape(
        ports[layer][port].shape).to(base.ports[layer][port].dtype)
    return dataclasses.replace(base, ports=ports)


def test_within_rounding_flags_a_wrong_term_in_fp32_and_bf16(bundle):
    """Fix 3: a term wrong by 6x the margin fails the residual bound (it could not before)."""
    adapter, base, dtype = bundle["adapter"], bundle["base"], bundle["dtype"]
    a, b = PAIRS[0]
    clean = explain.explain_margin(base, a, b)
    assert clean["all_within_rounding"] is True
    margin = abs(clean["native_margin"]["value"])
    rho, c = margin_rho_c(adapter, base, a, b)
    for layer in (0, 1, 2):
        wrong = 6.0 * margin
        bad = _perturbed(base, layer, "o", c / float(c @ c) * (wrong / rho))
        out = explain.explain_margin(bad, a, b)
        term = {t["name"]: t for t in out["terms"]}[f"L{layer}.attn"]
        assert abs(term["value"]["value"] - {t["name"]: t for t in clean["terms"]}[
            f"L{layer}.attn"]["value"]["value"]) == pytest.approx(wrong, rel=1e-2)
        rr = out["remainders"][0]
        assert rr["within_rounding"] is False and out["all_within_rounding"] is False
        assert abs(rr["value"]["value"]) > 5 * margin
        assert rr["k"] == 1 and rr["scale"]["authority"] == "bound" and not rr["heuristic"]
        assert any("round-to-nearest" in p for p in rr["premises"])
        assert any(str(base.ports[0]["h"].dtype) in p for p in rr["premises"])
        assert abs(out["closure"]["value"]) <= 1e-12 * max(1.0, margin)  # closure unaffected
    # the bound is per-coordinate and tight: far below the old global-ulp * 2L envelope
    rr = clean["remainders"][0]
    old = (float(rho) * float(c.abs().sum()) * float(torch.finfo(base.logits.dtype).eps)
           * 2.0 ** math.floor(math.log2(max(float(finite.f64(p[k]).abs().max())
                                              for p in base.ports for k in ("h", "r", "h_out"))))
           * 2 * adapter.layers)
    assert rr["scale"]["value"] < old
    note(bundle["case"], "P19 residual bound / old k*scale", rr["scale"]["value"] / old)
    # the dtype comes from the residual ports, not from the logits
    other = torch.bfloat16 if dtype == "fp32" else torch.float32
    import dataclasses
    mixed = dataclasses.replace(base, logits=base.logits.to(other))
    assert explain.explain_margin(mixed, a, b)["remainders"][0]["scale"]["value"] == rr[
        "scale"]["value"]


def test_within_rounding_still_catches_a_wrong_term_with_a_massive_activation():
    """One large embedding coordinate (400) inflates the stream max; the per-coordinate bound
    keeps catching a 6x-margin error (the global-ulp bound let +2.5 logits through)."""
    from test_analysis_finite import tiny

    for dtype in ("bf16", "fp32"):
        model = tiny("qwen2", dtype)
        with torch.no_grad():
            model.model.embed_tokens.weight[:, 0] = 400.0
        adapter = DecoderAdapter(model, granularity="operation")
        base = explain.capture_row(adapter, PROMPT)
        a, b = PAIRS[0]
        clean = explain.explain_margin(base, a, b)
        assert clean["all_within_rounding"] is True
        rho, c = margin_rho_c(adapter, base, a, b)
        margin = abs(clean["native_margin"]["value"])
        bad = _perturbed(base, 1, "o", c / float(c @ c) * (6.0 * margin / rho))
        out = explain.explain_margin(bad, a, b)
        assert out["remainders"][0]["within_rounding"] is False, dtype
        assert out["all_within_rounding"] is False, dtype


def test_fp32_wide_head_has_no_false_alarm_across_pairs():
    import transformers

    torch.manual_seed(3)
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=1536, intermediate_size=128,
                                   num_hidden_layers=1, num_attention_heads=12,
                                   num_key_value_heads=2, attn_implementation="eager",
                                   tie_word_embeddings=False)
    wide = transformers.Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():
        wide.model.embed_tokens.weight.normal_()
        wide.lm_head.weight.normal_(0, 1.0)
    base = explain.capture_row(DecoderAdapter(wide, granularity="operation"), PROMPT)
    worst = 0.0
    for a in range(0, 64, 2):
        out = explain.explain_margin(base, a, a + 1)
        head = {r["name"]: r for r in out["remainders"]}["head"]
        assert head["within_rounding"] is True and head["k"] == 1 and not head["heuristic"]
        assert head["scale"]["authority"] == "bound"
        assert out["remainders"][0]["within_rounding"] is True
        worst = max(worst, abs(head["value"]["value"]) / head["scale"]["value"])
    assert worst < 1.0
    note("wide-fp32", "P19 head |remainder| / bound d=1536", worst)


def test_head_bound_with_a_bias_adds_the_bias_rounding():
    from test_analysis_finite import tiny

    model = tiny("qwen2", "fp32")
    head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=True)
    with torch.no_grad():
        head.weight.copy_(model.lm_head.weight)
        head.bias.copy_(torch.randn(model.config.vocab_size))
    model.lm_head = head
    base = explain.capture_row(DecoderAdapter(model, granularity="operation"), PROMPT)
    out = explain.explain_margin(base, *PAIRS[0])
    head_rem = {r["name"]: r for r in out["remainders"]}["head"]
    assert head_rem["within_rounding"] is True
    assert any("bias" in p for p in head_rem["premises"])
    assert {t["name"]: t for t in out["terms"]}["head_bias"]["value"]["authority"] == "derived-exact"


def test_first_order_column_on_an_edited_row_is_refused_not_raised(bundle):
    adapter = bundle["adapter"]
    caps = explain.capture_row(adapter, PROMPT, edits={1: [PortEdit("m", 2.0, "scale")]})
    out = explain.explain_margin(caps, *PAIRS[0], first_order=True)
    column = out["first_order"]
    assert column["state"] == "refused" and column["authority"] is None
    assert "this row" in column["reason"] and "rows" not in column
    assert out["state"] == "resolved" and not walk(out)
