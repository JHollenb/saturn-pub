"""Proof-series identities on captured ports of tiny GQA Qwen2/Qwen3 models.

Closure (real arithmetic) is asserted at <= 1e-9 relative. The native remainder eta is measured:
only finiteness and the dtype's magnitude are asserted. Run with ``-s`` to print the per-identity
maxima; set ``MDB_ANALYSIS_SUMMARY=<path>`` to also write them as JSON.
"""

import json
import math
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402

from mdb.analysis import finite  # noqa: E402
from mdb.analysis.explain import capture_row  # noqa: E402
from mdb.engine.ports import PortEdit  # noqa: E402

CASES = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32")]
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
ETA_REL = {"bf16": 5e-2, "fp32": 1e-4}
CLOSURE = 1e-9
PROMPT = [3, 17, 5, 29, 11, 8, 21]
OTHER = [9, 2, 33, 14, 27, 6, 40]
ONE_SOURCE = [3, 17, 40, 29, 11, 8, 21]
SUMMARY: dict = {}


def tiny(family, dtype, seed=5):
    """Random 3-layer GQA model (4 query heads, 2 KV groups) with non-trivial gains and biases."""
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    common = dict(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                  attn_implementation="eager")
    if family == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(**common))
    else:
        model = Qwen3ForCausalLM(Qwen3Config(head_dim=8, **common))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1 + 0.3 * torch.randn_like(param))
            elif name.endswith("bias"):
                param.copy_(0.5 * torch.randn_like(param))
            elif "embed" in name:
                param.copy_(torch.randn_like(param))
            else:
                param.copy_(0.3 * torch.randn_like(param))
    return model.eval().to(DTYPES[dtype])


def record(case, res, label=None):
    entry = SUMMARY.setdefault(f"{res.proof} {label or res.name}", {}).setdefault(
        case, {"closure_rel": 0.0, "eta_rel": 0.0, "eta_abs": 0.0, "eta_rel_change": 0.0,
               "n": 0})
    entry["n"] += 1
    if res.reference is not None:
        entry["closure_rel"] = max(entry["closure_rel"], res.closure_rel)
    if res.native is not None:
        entry["eta_rel"] = max(entry["eta_rel"], res.eta_rel)
        entry["eta_abs"] = max(entry["eta_abs"], res.eta_max_abs)
        entry["eta_rel_change"] = max(entry["eta_rel_change"], res.eta_rel_to_change)
    return res


def exact(case, res, label=None, tol=CLOSURE):
    record(case, res, label)
    assert res.closure_rel <= tol, (res.name, res.closure_rel)
    if res.native is not None:
        dtype = case.split("-")[1]
        assert torch.isfinite(res.eta).all()
        assert res.eta_rel <= ETA_REL[dtype], (res.name, res.eta_rel)
    return res


def quadratic(errors, case=None, key=None):
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(3.0 < r < 5.0 for r in ratios), (errors, ratios)
    if case is not None:
        measured(case, f"{key} fd ratio min", min(ratios))
        measured(case, f"{key} fd ratio max", max(ratios))
    return ratios


MEASURED: dict = {}


def measured(case, key, value):
    entry = MEASURED.setdefault(key, {})
    value = float(value)
    if key.endswith("min"):
        entry[case] = min(entry.get(case, value), value)
    else:
        entry[case] = max(entry.get(case, value), value)
    return value


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


@pytest.fixture(scope="module", autouse=True)
def report():
    yield
    lines = ["", "analysis-finite measured maxima (closure_rel float64 | eta_rel native)"]
    for key in sorted(SUMMARY):
        for case, v in sorted(SUMMARY[key].items()):
            lines.append(f"  {key:42s} {case:12s} closure {v['closure_rel']:.2e} "
                         f"eta_rel {v['eta_rel']:.2e} eta_abs {v['eta_abs']:.2e} "
                         f"eta/change {v['eta_rel_change']:.2e} n={v['n']}")
    lines.append("analysis-finite measured comparisons (autograd / FD / absolute closures)")
    for key in sorted(MEASURED):
        for case, value in sorted(MEASURED[key].items()):
            lines.append(f"  {key:42s} {case:12s} {value:.3e}")
    print("\n".join(lines))
    path = os.environ.get("MDB_ANALYSIS_SUMMARY")
    if path:
        with open(path, "a") as handle:
            handle.write(json.dumps({"module": "finite", "summary": SUMMARY,
                                     "measured": MEASURED}) + "\n")


@pytest.fixture(scope="module", params=CASES, ids=lambda c: "-".join(c))
def bundle(request):
    family, dtype = request.param
    torch.set_num_threads(1)
    adapter = DecoderAdapter(tiny(family, dtype), granularity="operation")
    base = capture_row(adapter, PROMPT)
    generator = torch.Generator().manual_seed(11)
    h1 = base.ports[1]["h"]
    noise = (torch.randn(h1.shape, generator=generator) * h1.float().std()).to(h1.dtype)
    return {
        "case": "-".join(request.param),
        "adapter": adapter,
        "base": base,
        "other": capture_row(adapter, OTHER),
        "one": capture_row(adapter, ONE_SOURCE),
        "edited": capture_row(adapter, PROMPT, edits={1: [PortEdit("h", noise, "add")]}),
    }


def vec(x):
    return finite.f64(x).reshape(-1)


# Structural integer keys a result may carry as bare integers (indices, counts, layer, head, k).
INT_KEYS = frozenset({"layer", "head", "k", "count", "position", "a", "b", "row", "index",
                      "source", "rank", "token", "transition", "mask_count", "shape", "mask_shape"})
LABEL_KEYS = {"value", "authority", "proof", "unit", "pointer", "includes_edits",
              "includes_supplied"}
AUTHORITY = {"measured", "derived-exact", "derived-local", "bound", "supplied"}
STATE = {"resolved", "unresolved", "refused"}


def is_labelled(node):
    return (isinstance(node, dict) and "value" in node and "authority" in node
            and set(node) <= LABEL_KEYS and node["authority"] in AUTHORITY
            and isinstance(node["value"], (int, float)) and not isinstance(node["value"], bool)
            and math.isfinite(node["value"]))


def is_inspection_only(node):
    """A number shown but not claimed: ``{value, authority: None, inspection_only: True}``."""
    return (isinstance(node, dict) and node.get("authority", 0) is None
            and node.get("inspection_only") is True and "value" in node
            and set(node) <= LABEL_KEYS | {"inspection_only"}
            and isinstance(node["value"], (int, float)) and not isinstance(node["value"], bool)
            and math.isfinite(node["value"]))


def labelled_inside(node, path="", skip=("premises",)):
    """Paths of labelled numbers (authority in the vocabulary) outside the ``skip`` keys."""
    if is_labelled(node):
        return [path]
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k not in skip:
                out += labelled_inside(v, f"{path}.{k}", skip)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            out += labelled_inside(v, f"{path}[{i}]", skip)
    return out


def unlabelled_numbers(node, path="", key=None):
    """Paths of numeric leaves that are neither a labelled number nor an allow-listed integer."""
    if is_labelled(node) or is_inspection_only(node):
        return []
    if isinstance(node, bool) or node is None or isinstance(node, str):
        return []
    if isinstance(node, (int, float)):
        if isinstance(node, int) and key in INT_KEYS:
            return []
        return [f"{path} = {node!r}"]
    if isinstance(node, dict):
        out = []
        for k, v in node.items():
            out += unlabelled_numbers(v, f"{path}.{k}", k)
        return out
    if isinstance(node, (list, tuple)):
        out = []
        for i, v in enumerate(node):
            out += unlabelled_numbers(v, f"{path}[{i}]", key)
        return out
    if isinstance(node, torch.Tensor):
        return [f"{path} is a tensor"]
    return []


def layer(adapter, index):
    return adapter._layer_modules[index]


def head_operands(adapter, caps, index, head):
    """Positioned query, keys, values of one head at one layer (kappa-mapped)."""
    kappa = finite.gqa_kappa(adapter.num_heads, adapter.num_kv_heads)
    ports = caps.ports[index]
    return (ports["qr"][0, head, 0], ports["kr"][0, kappa[head]],
            caps.values[index][0, kappa[head]])


def test_finite_changes_are_finite_not_small(bundle):
    base, other, edited = bundle["base"], bundle["other"], bundle["edited"]
    for index in range(3):
        for port in ("r", "A", "f"):
            x0 = vec(base.ports[index][port])
            for caps in (other, edited) if index >= 1 else (other,):
                dx = vec(caps.ports[index][port]) - x0
                assert dx.norm() > 0.3 * x0.norm(), (index, port)
    p0, p1 = vec(base.ports[1]["p"]), vec(other.ports[1]["p"])
    assert (p1 - p0).abs().max() > 0.1


def test_p02_rms_forward_components_and_radial(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    for index in range(3):
        mod, ports = layer(adapter, index), base.ports[index]
        for src, dst, norm in (("h", "a", mod.input_layernorm),
                               ("r", "n", mod.post_attention_layernorm)):
            res = finite.rms_norm(vec(ports[src]), norm.weight, norm.variance_epsilon,
                                  native=vec(ports[dst]))
            record(case, res)
            assert res.eta_rel <= ETA_REL[case.split("-")[1]]
    mod = layer(adapter, 2)
    comps = {"emb": vec(base.ports[0]["h"])}
    for index in range(2):
        comps[f"o{index}"] = vec(base.ports[index]["o"])
        comps[f"m{index}"] = vec(base.ports[index]["m"])
    res = finite.rms_norm(vec(base.ports[2]["h"]), mod.input_layernorm.weight,
                          mod.input_layernorm.variance_epsilon, components=comps,
                          native=vec(base.ports[2]["a"]))
    exact(case, res)
    x = vec(base.ports[1]["r"])
    norm = layer(adapter, 1).post_attention_layernorm
    jac = finite.rms_jacobian(x, norm.weight, norm.variance_epsilon)
    exact(case, jac, "rms_radial")
    replica = torch.autograd.functional.jacobian(
        lambda y: finite.rms_norm(y, norm.weight, norm.variance_epsilon).total, x)
    assert measured(case, "P02 jacobian vs autograd rel",
                    rel(jac.extras["jacobian"], replica)) <= 1e-12


def test_p03_rms_finite_change(bundle):
    case, adapter = bundle["case"], bundle["adapter"]
    pairs = [(bundle["base"], bundle["other"], i) for i in range(3)]
    pairs += [(bundle["base"], bundle["edited"], i) for i in (1, 2)]
    for c0, c1, index in pairs:
        mod = layer(adapter, index)
        for src, dst, norm in (("h", "a", mod.input_layernorm),
                               ("r", "n", mod.post_attention_layernorm)):
            res = finite.rms_finite_change(vec(c0.ports[index][src]), vec(c1.ports[index][src]),
                                           norm.weight, norm.variance_epsilon,
                                           native0=vec(c0.ports[index][dst]),
                                           native1=vec(c1.ports[index][dst]))
            exact(case, res)
            assert res.extras["drho_rationalized_minus_direct"] <= 1e-12
            two = (res.extras["two_term_total"] - res.reference).abs().max()
            assert two <= CLOSURE * res.reference.abs().max()
            assert res.terms["drho_x0"].abs().max() > 1e-3


def test_p04_affine_gqa_and_route_average(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    kappa = finite.gqa_kappa(adapter.num_heads, adapter.num_kv_heads)
    assert kappa == [0, 0, 1, 1]
    for index in range(3):
        attn, ports = layer(adapter, index).self_attn, base.ports[index]
        for name in ("q", "k", "v"):
            proj = getattr(attn, f"{name}_proj")
            res = finite.affine(vec(ports["a"]), proj.weight, proj.bias, native=vec(ports[name]))
            record(case, res, f"affine_{name}")
            assert res.eta_rel <= ETA_REL[case.split("-")[1]]
        res = finite.head_route_average_drop(ports["p"][0, :, 0], base.values[index][0],
                                             attn.o_proj.weight, kappa)
        exact(case, res)
        assert res.total.abs().max() > 1e-3


def test_p04_rope_rotation_relative_and_score_pairs(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    config = adapter.model.config
    theta = config.rope_parameters["rope_theta"]
    d_h, t = adapter.head_dim, base.position
    pre = "qn" if base.family == "qwen3" else "q"
    for index in range(3):
        ports = base.ports[index]
        qbar = finite.f64(ports[pre]).reshape(adapter.num_heads, d_h)
        kbar = finite.f64(ports[pre.replace("q", "k")]).reshape(adapter.num_kv_heads, d_h)
        for label, rot in (("rope_rotate_q", finite.rope_rotate(qbar, t, theta,
                                                              native=ports["qr"][0, :, 0])),
                           ("rope_rotate_k", finite.rope_rotate(kbar, t, theta,
                                                              native=ports["kr"][0, :, t]))):
            record(case, rot, label)
            assert rot.proof == "P04"
            # asserted against the native qr/kr captures, not only recorded
            assert rot.eta_rel <= ETA_REL[case.split("-")[1]], (label, index, rot.eta_rel)
            measured(case, f"P04 {label} eta_rel", rot.eta_rel)
    for j in (0, 3, 6):
        rel_res = finite.rope_relative(t, j, theta, d_h)
        exact(case, rel_res, tol=1e-12)
        assert rel_res.proof == "P04"
    prefix = [capture_row(adapter, PROMPT[: j + 1]) for j in range(len(PROMPT))]
    kappa = finite.gqa_kappa(adapter.num_heads, adapter.num_kv_heads)
    attn = layer(adapter, 0).self_attn
    for index in (0, 2):
        for head in range(adapter.num_heads):
            qbar = finite.f64(base.ports[index][pre]).reshape(adapter.num_heads, d_h)[head]
            kbar = torch.stack([
                finite.f64(prefix[j].ports[index][pre.replace("q", "k")])
                .reshape(adapter.num_kv_heads, d_h)[kappa[head]] for j in range(len(PROMPT))])
            res = finite.rope_score_pairs(qbar, kbar, t, range(len(PROMPT)), theta,
                                          scaling=attn.scaling,
                                          native=base.ports[index]["s"][0, head, 0])
            exact(case, res)


def test_p05_contrasts_anchor_and_query_invariance(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    scaling = layer(adapter, 1).self_attn.scaling
    generator = torch.Generator().manual_seed(4)
    for head in range(adapter.num_heads):
        q, keys, values = head_operands(adapter, base, 1, head)
        exact(case, finite.softmax_contrasts(q, keys, 0, scaling=scaling,
                                             native_probs=base.ports[1]["p"][0, head, 0]))
        for j0 in (3, 6):
            res = exact(case, finite.softmax_contrasts(
                q, keys, j0, scaling=scaling, native_probs=base.ports[1]["p"][0, head, 0]),
                f"softmax_contrasts_j0_{j0}")
            # independent: log-ratio of the float64 softmax of the scores
            s64 = scaling * finite.f64(keys) @ finite.f64(q)
            logp = s64 - torch.logsumexp(s64, -1)
            assert (res.total - (logp - logp[j0])).abs().max() <= 1e-12
            assert res.total[j0] == 0 and res.total.abs().max() > 0.01
        p = torch.softmax(scaling * finite.f64(keys) @ finite.f64(q), -1)
        exact(case, finite.anchored_transport(p, values, 2))
        # native (rounded) p: sum p != 1, the premise is reported and the term closes the identity
        native_p = finite.anchored_transport(base.ports[1]["p"][0, head, 0], values, 2)
        record(case, native_p, "anchored_transport_native_p")
        assert native_p.closure_rel <= 1e-12, native_p.closure_rel
        pnat = finite.f64(base.ports[1]["p"][0, head, 0])
        defect = float(1 - pnat.sum())
        pm = native_p.premises["mass_defect"]
        assert pm["value"]["authority"] == "measured" and pm["value"]["value"] == abs(defect)
        assert native_p.state == "resolved"
        want = -defect * finite.f64(values)[2]
        assert (native_p.terms["premise_remainder"] - want).abs().max() <= 1e-15
        measured(case, "P05 anchored mass defect abs", abs(defect))
        measured(case, "P05 anchored closure_rel native p", native_p.closure_rel)
        direction = torch.randn(adapter.head_dim, generator=generator, dtype=torch.float64)
        # a random direction is NOT orthogonal to the contrasts: refused as given, not projected
        given = finite.query_shift_invariance(q, keys, direction, scaling=scaling,
                                              doses=(1.0, 10.0, 100.0))
        assert given.state == "refused" and given.extras["direction"] == "as_given"
        assert "removed_norm=" in given.reason and "tolerance" in given.reason
        assert given.extras["removed_norm"] > 0.01
        assert given.premises["edit_orthogonal_to_contrasts"]["holds"] is False
        out = given.summary()
        assert out["authority"] is None and out["direction"] == "as_given"
        # the projected question is offered only through project=True and says so
        res = finite.query_shift_invariance(q, keys, direction, scaling=scaling,
                                            doses=(1.0, 10.0, 100.0), project=True)
        record(case, res)
        measured(case, "P05 query_shift max |dp| abs", res.closure)
        assert res.state == "resolved" and res.summary()["direction"] == "projected"
        assert res.extras["perp_norm"] > 0.1 and res.closure <= 1e-12
        removed = res.premises["removed_norm"]
        assert removed["holds"] is True and removed["value"]["authority"] == "measured"
        assert removed["value"]["value"] == pytest.approx(given.extras["removed_norm"])
        # a direction already orthogonal to the contrasts passes as given, no projection
        again = finite.query_shift_invariance(q, keys, torch.tensor(res.extras["perp_norm"]) * 0
                                              + _perp(keys, direction), scaling=scaling)
        assert again.state == "resolved" and again.extras["direction"] == "as_given"
        assert again.closure <= 1e-12


def _perp(keys, direction, j0=0):
    """The part of ``direction`` orthogonal to every contrast ``k_j - k_{j0}`` (float64)."""
    k = finite.f64(keys)
    contrasts = torch.cat([k[:j0], k[j0 + 1:]]) - k[j0]
    coef = torch.linalg.lstsq(contrasts.T, direction.reshape(-1, 1)).solution
    return direction - (contrasts.T @ coef).reshape(-1)


def test_p06_attention_differential_vs_autograd_and_fd(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    scaling = layer(adapter, 1).self_attn.scaling
    generator = torch.Generator().manual_seed(6)
    for head in range(adapter.num_heads):
        q, keys, values = map(finite.f64, head_operands(adapter, base, 1, head))
        dirs = [torch.randn(x.shape, generator=generator, dtype=torch.float64) * x.std()
                for x in (q, keys, values)]

        def attn(qq, kk, vv):
            return torch.softmax(scaling * kk @ qq, -1) @ vv

        lam = torch.randn(values.shape[-1], generator=generator, dtype=torch.float64)
        res = finite.attention_differential(q, keys, values, scaling=scaling, dquery=dirs[0],
                                            dkeys=dirs[1], dvalues=dirs[2], adjoint=lam)
        _, jvp = torch.autograd.functional.jvp(attn, (q, keys, values), tuple(dirs))
        assert measured(case, "P06 dA vs autograd jvp rel", rel(res.total, jvp)) <= 1e-12
        base_value = attn(q, keys, values)
        errors = [float((attn(q + e * dirs[0], keys + e * dirs[1], values + e * dirs[2])
                         - base_value - e * res.total).norm()) for e in (1e-2, 5e-3, 2.5e-3)]
        quadratic(errors, case, "P06")
        grads = torch.autograd.functional.vjp(attn, (q, keys, values), lam)[1]
        for name, auto in zip(("grad_q", "grad_k", "grad_v"), grads):
            assert measured(case, "P06 adjoints vs autograd rel",
                            rel(res.extras[name], auto)) <= 1e-12
        s = scaling * keys @ q
        ds = torch.autograd.functional.vjp(lambda ss: torch.softmax(ss, -1) @ values, s, lam)[1]
        assert measured(case, "P06 adjoints vs autograd rel", rel(res.extras["dM_ds"], ds)) <= 1e-12


def test_p07_finite_score_and_transport_split(bundle):
    case, adapter = bundle["case"], bundle["adapter"]
    kappa = finite.gqa_kappa(adapter.num_heads, adapter.num_kv_heads)
    scaling = layer(adapter, 0).self_attn.scaling
    pairs = [(bundle["base"], bundle["other"], i) for i in range(3)]
    pairs += [(bundle["base"], bundle["edited"], i) for i in (1, 2)]
    for c0, c1, index in pairs:
        p0, p1 = c0.ports[index], c1.ports[index]
        res = finite.score_finite_change(p0["qr"][0, :, 0], p1["qr"][0, :, 0],
                                         p0["kr"][0][kappa], p1["kr"][0][kappa], scaling=scaling,
                                         native0=p0["s"][0, :, 0], native1=p1["s"][0, :, 0])
        exact(case, res)
        heads = adapter.num_heads
        res = finite.transport_finite_split(
            p0["p"][0, :, 0], p1["p"][0, :, 0], c0.values[index][0][kappa],
            c1.values[index][0][kappa], native0=p0["A"].reshape(heads, -1),
            native1=p1["A"].reshape(heads, -1))
        exact(case, res)
        assert min(res.terms[k].abs().max() for k in res.terms) > 1e-3


def test_p08_exact_reweighting(bundle):
    case, base, other = bundle["case"], bundle["base"], bundle["other"]
    for index in range(3):
        s0, s1 = base.ports[index]["s"][0, :, 0], other.ports[index]["s"][0, :, 0]
        res = finite.softmax_reweighting(s0, finite.f64(s1) - finite.f64(s0),
                                         native1=other.ports[index]["p"][0, :, 0])
        exact(case, res)


def _one_source(adapter, c0, c1, index, head):
    kappa = finite.gqa_kappa(adapter.num_heads, adapter.num_kv_heads)
    p0, p1 = c0.ports[index], c1.ports[index]
    d_h = adapter.head_dim
    return finite.one_source_response(
        p0["s"][0, head, 0], p1["s"][0, head, 0], c0.values[index][0, kappa[head]],
        c1.values[index][0, kappa[head]], 2, query0=p0["qr"][0, head, 0],
        query1=p1["qr"][0, head, 0], native0=p0["A"].reshape(-1, d_h)[head],
        native1=p1["A"].reshape(-1, d_h)[head])


def test_p08_one_source_closed_form_inside_its_assumptions(bundle):
    case, adapter = bundle["case"], bundle["adapter"]
    for head in range(adapter.num_heads):
        res = _one_source(adapter, bundle["base"], bundle["one"], 0, head)
        assert res.extras["assumptions"]["holds"], res.extras["assumptions"]
        exact(case, res)
        assert abs(res.extras["delta"]) > 0.05


def test_p08_one_source_closed_form_fails_outside_its_assumptions(bundle):
    case, adapter = bundle["case"], bundle["adapter"]
    failures = []
    for head in range(adapter.num_heads):
        res = _one_source(adapter, bundle["base"], bundle["one"], 1, head)
        checks = res.extras["assumptions"]
        assert not checks["holds"]
        assert checks["query_fixed"] is False and not checks["other_scores_fixed"]
        record(case, res, "one_source_response_violated")
        failures.append(res.closure_rel)
        # outside its premises the result is refused, not resolved
        assert res.state == "refused" and res.reason.startswith("premise ")
        assert res.reason.endswith(" violated")
        assert res.premises["query_fixed"]["holds"] is False
        out = res.summary()
        assert out["state"] == "refused" and out["reason"] == res.reason
        assert out["authority"] is None
        assert is_inspection_only(out["closure_max_abs"]) and is_inspection_only(
            out["closure_rel"])
    assert min(failures) > 1e-3, failures


def test_p08_one_source_premises_when_only_values_differ():
    """Scores and query fixed everywhere; another source's VALUE differs: premise must fail."""
    g = torch.Generator().manual_seed(8)
    s0 = torch.randn(6, generator=g, dtype=torch.float64)
    s1 = s0.clone()
    s1[2] += 0.7
    v0 = torch.randn(6, 5, generator=g, dtype=torch.float64)
    v1 = v0.clone()
    v1[2] += 0.1  # the changed source: allowed
    q = torch.randn(5, generator=g, dtype=torch.float64)
    ok = finite.one_source_assumptions(s0, s1, v0, v1, 2, query0=q, query1=q)
    assert ok["holds"] and ok["other_values_fixed"]
    v2 = v1.clone()
    v2[4] += 1e-3  # an OTHER source's value changes
    bad = finite.one_source_assumptions(s0, s1, v0, v2, 2, query0=q, query1=q)
    assert bad["other_scores_fixed"] and bad["visible_set_fixed"] and bad["query_fixed"] is True
    assert bad["other_values_fixed"] is False and bad["holds"] is False
    assert bad["other_value_max_abs_change"] == pytest.approx(1e-3)
    res = finite.one_source_response(s0, s1, v0, v2, 2, query0=q, query1=q)
    assert res.state == "refused" and res.reason == "premise other_values_fixed violated"
    prem = res.premises["other_values_fixed"]
    assert prem["holds"] is False and prem["value"]["authority"] == "measured"
    assert prem["tolerance"]["value"] == 0.0 and prem["tolerance"]["authority"] == "supplied"
    assert res.premises["other_scores_fixed"]["holds"] is True
    assert res.closure_rel > 1e-6
    ok_res = finite.one_source_response(s0, s1, v0, v1, 2, query0=q, query1=q)
    assert ok_res.state == "resolved" and ok_res.closure_rel <= 1e-12
    assert all(p["holds"] for p in ok_res.premises.values())
    summary = ok_res.summary()
    assert summary["closure_max_abs"]["authority"] == "derived-exact"


def test_t_kind_identity_with_a_failed_premise_is_refused():
    """winner_certificate on a tied head: premise unique_winner fails, the closure is not exact."""
    w = torch.zeros(3, 4, dtype=torch.float64)
    w[0] = 1.0
    w[1] = 1.0  # exact tie between the two best rows
    w[2] = -1.0
    h = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)
    ones = torch.ones(4, dtype=torch.float64)
    res = finite.winner_certificate(h, w, ones, 1e-6, bias_free=True)
    assert res.state == "refused" and res.reason == "premise unique_winner violated"
    assert res.premises["unique_winner"]["holds"] is False
    out = res.summary()
    assert out["state"] == "refused" and out["reason"] == "premise unique_winner violated"
    assert out["authority"] is None and is_inspection_only(out["total_max_abs"])
    assert not unlabelled_numbers(out)
    # the unique case resolves
    w[1] = 0.5
    ok = finite.winner_certificate(h, w, ones, 1e-6, bias_free=True)
    assert ok.state == "resolved" and ok.summary()["total_max_abs"]["authority"] == "bound"
    assert ok.premises["bias_free_or_supplied"]["holds"] is True


def test_winner_certificate_bias_premise_is_checked_not_assumed():
    g = torch.Generator().manual_seed(11)
    w = torch.randn(6, 8, generator=g, dtype=torch.float64)
    h = torch.randn(8, generator=g, dtype=torch.float64)
    gamma = torch.ones(8, dtype=torch.float64)
    plain = (h / torch.sqrt(1e-6 + (h * h).mean())) @ w.T
    order = plain.argsort(descending=True)
    # neither a bias nor bias_free: the premise is not silently assumed
    bare = finite.winner_certificate(h, w, gamma, 1e-6)
    assert bare.state == "refused" and "bias_free_or_supplied" in bare.reason
    assert bare.premises["bias_free_or_supplied"]["holds"] is False
    assert bare.summary()["authority"] is None
    free = finite.winner_certificate(h, w, gamma, 1e-6, bias_free=True)
    assert free.state == "resolved" and free.extras["winner"] == int(order[0])
    # a supplied bias enters the margins exactly: here it flips the winner and the radius
    bias = torch.zeros(6, dtype=torch.float64)
    bias[order[1]] = float(plain[order[0]] - plain[order[1]]) + 1.0
    cert = finite.winner_certificate(h, w, gamma, 1e-6, bias=bias, delta=0.0 * h)
    assert cert.state == "resolved" and cert.extras["winner"] == int(order[1])
    assert cert.extras["winner"] == int((plain + bias).argmax())
    assert cert.extras["argmax_after"] == cert.extras["winner"]
    assert cert.premises["bias_free_or_supplied"]["holds"] is True
    # radius is the exact min margin / (rho ||c||) with the bias in the margins
    rho = 1.0 / torch.sqrt(1e-6 + (h * h).mean())
    full = rho * h @ w.T + bias
    a = int(full.argmax())
    others = [i for i in range(6) if i != a]
    want = min(float((full[a] - full[i]) / (rho * (w[a] - w[i]).norm())) for i in others)
    assert float(cert.total) == pytest.approx(want, rel=1e-12)
    assert float(free.total) != pytest.approx(want, rel=1e-6)


def test_refused_and_unresolved_summaries_carry_no_labelled_number():
    """Under refused/unresolved: top-level authority None, every term/total/closure inspection-only,
    only premises keep labels."""
    g = torch.Generator().manual_seed(5)
    r = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64)  # noqa: E731
    refused = finite.one_source_response(r(5), r(5), r(5, 3), r(5, 3), 1)
    unresolved = finite.IdentityResult("P01", "toy", "A", "derived-exact",
                                       {"t": torch.tensor([float("nan")]), "u": torch.ones(2)},
                                       torch.ones(2), reference=torch.ones(2) * 2)
    for res, state in ((refused, "refused"), (unresolved, "unresolved")):
        out = res.summary()
        assert out["state"] == state and out["authority"] is None
        assert not labelled_inside(out), labelled_inside(out)
        assert out["terms_max_abs"] or state == "unresolved"
        for entry in list(out["terms_max_abs"].values()) + [out.get("total_max_abs"),
                                                            out.get("closure_max_abs"),
                                                            out.get("closure_rel")]:
            if entry is not None:
                assert is_inspection_only(entry), entry
    assert any(is_labelled(p["value"]) for p in refused.summary()["premises"].values())


def test_p09_zero_and_mask_cuts_match_native_p_edits(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    kappa = finite.gqa_kappa(adapter.num_heads, adapter.num_kv_heads)
    index, cut = 1, [0, 3]
    p = base.ports[index]["p"]
    zero = p.clone()
    zero[..., cut] = 0
    p64 = finite.f64(zero)
    mask = (p64 / p64.sum(-1, keepdim=True)).to(p.dtype)
    runs = {name: capture_row(adapter, PROMPT, edits={index: [PortEdit("p", value)]})
            for name, value in (("zero", zero), ("mask", mask))}
    heads = adapter.num_heads
    natives = dict(native_base=base.ports[index]["A"].reshape(heads, -1),
                   native_zero=runs["zero"].ports[index]["A"].reshape(heads, -1),
                   native_mask=runs["mask"].ports[index]["A"].reshape(heads, -1))
    values = base.values[index][0][kappa]
    probs = torch.softmax(finite.f64(base.ports[index]["s"][0, :, 0]), -1)
    results = finite.cut_sources(probs, values, cut, **natives)
    for res in results.values():
        exact(case, res)
        assert res.total.abs().max() > 1e-3
    assert (results["zero"].total - results["mask"].total).abs().max() > 1e-3
    rounded = finite.cut_sources(p[0, :, 0], values, cut, **natives)
    record(case, rounded["zero"], "cut_zero_native_p")
    assert rounded["zero"].closure_rel <= CLOSURE
    mask = record(case, rounded["mask"], "cut_mask_native_p_sum_not_one")
    # the rounded p has sum p != 1; the premise is reported and premise_remainder closes it
    defect = mask.extras["mass_defect"].abs().max()
    assert defect > 0
    assert mask.closure_rel <= 1e-12, mask.closure_rel
    assert mask.state == "resolved"
    pm = mask.premises["mass_defect"]
    assert pm["value"]["authority"] == "measured" and pm["value"]["value"] == float(defect)
    assert pm["holds"] is False
    assert mask.terms["premise_remainder"].abs().max() > 0
    measured(case, "P09 mask native-p closure_rel", mask.closure_rel)
    measured(case, "P09 mask native-p mass defect abs", defect)
    assert mask.summary()["terms_max_abs"]["premise_remainder"]["authority"] == "derived-exact"
    # the chapter form alone (without premise_remainder) misses by exactly that term
    bare = sum(v for k, v in mask.terms.items() if k != "premise_remainder")
    assert (bare + mask.terms["premise_remainder"] - mask.reference).abs().max() <= 1e-12 * (
        1 + mask.reference.abs().max())


def test_p09_mask_closes_on_native_p_of_both_dtypes():
    """Closure <= 1e-12 relative on rounded bf16 and fp32 p (sum p = 1 - eps), cut at several sets."""
    g = torch.Generator().manual_seed(9)
    for dtype in (torch.bfloat16, torch.float32):
        p = torch.softmax(torch.randn(4, 9, generator=g) * 2, -1).to(dtype)
        v = torch.randn(4, 9, 6, generator=g, dtype=torch.float64)
        for cut in ([0], [1, 4], [2, 3, 8]):
            res = finite.cut_sources(p, v, cut)["mask"]
            assert res.closure_rel <= 1e-12, (dtype, cut, res.closure_rel)
            assert res.state == "resolved"
            defect = res.premises["mass_defect"]
            assert defect["value"]["authority"] == "measured"
        a = finite.f64(p).sum(-1)
        assert dtype == torch.float32 or (1 - a).abs().max() > 1e-4  # bf16 does miss


def test_summary_unresolved_not_raising_on_nonfinite():
    """bf16 p underflow (log 0 in contrasts) and mu = 1 (all mass cut) are unresolved, not errors."""
    q = torch.tensor([1.0, 0.5], dtype=torch.float64)
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float64)
    p_native = torch.tensor([0.9, 0.0, 0.1], dtype=torch.bfloat16)  # underflowed to exactly 0
    res = finite.softmax_contrasts(q, keys, 0, scaling=1.0, native_probs=p_native)
    out = res.summary()
    assert out["state"] == "unresolved"
    assert "eta_max_abs" in out["reason"] or "eta_rel" in out["reason"]
    assert "eta_max_abs" not in out and not unlabelled_numbers(out)
    # the reason names the quantity; the finite numbers are still reported
    assert out["authority"] is None and is_inspection_only(out["closure_max_abs"])
    p = torch.tensor([0.5, 0.5], dtype=torch.float64)
    v = torch.eye(2, 3, dtype=torch.float64)
    cut = finite.cut_sources(p, v, [0, 1])["mask"]  # mu = 1
    out = cut.summary()
    assert out["state"] == "unresolved" and "not finite" in out["reason"]
    assert any(name in out["reason"] for name in ("total_max_abs", "terms_max_abs"))
    assert cut.premises["mu_lt_1"]["holds"] is False
    assert not unlabelled_numbers(out)


def test_p10_p11_swiglu_operator_and_quadratic_form(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    for index in range(3):
        mlp, ports = layer(adapter, index).mlp, base.ports[index]
        norm = layer(adapter, index).post_attention_layernorm
        out = finite.swiglu_forward(vec(ports["n"]), mlp.gate_proj.weight, mlp.up_proj.weight,
                                    mlp.down_proj.weight,
                                    native={k: vec(ports[k]) for k in ("g", "sg", "u", "f", "m")})
        for port, res in out.items():
            record(case, res, f"swiglu_{port}")
            assert res.eta_rel <= ETA_REL[case.split("-")[1]] * 4
        exact(case, finite.mlp_state_operator(vec(ports["r"]), mlp.gate_proj.weight,
                                              mlp.up_proj.weight, mlp.down_proj.weight,
                                              norm.weight, norm.variance_epsilon,
                                              native_m=vec(ports["m"])))
        for neuron in (0, 7, 33):
            res = finite.neuron_quadratic_form(mlp.gate_proj.weight[neuron],
                                               mlp.up_proj.weight[neuron], vec(ports["n"]),
                                               native_f=vec(ports["f"])[neuron])
            record(case, res)
            assert res.closure_rel <= CLOSURE
            assert res.extras["eig_closure"] <= 1e-12 * res.extras["eig_closed"].abs().max()
            assert res.extras["rank"] <= 2


def test_p12_p13_read_plane_and_raw_hessian_vs_autograd(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    mlp, norm = layer(adapter, 1).mlp, layer(adapter, 1).post_attention_layernorm
    gamma, eps = finite.f64(norm.weight), norm.variance_epsilon
    n, r = vec(base.ports[1]["n"]), vec(base.ports[1]["r"])
    for neuron in (0, 7, 33):
        g, u = finite.f64(mlp.gate_proj.weight[neuron]), finite.f64(mlp.up_proj.weight[neuron])

        def f_n(x):
            return finite.silu(g @ x) * (u @ x)

        res = finite.neuron_read_plane(g, u, n)
        grad = torch.autograd.functional.jacobian(f_n, n)
        hess = torch.autograd.functional.hessian(f_n, n)
        assert measured(case, "P12 gradient vs autograd rel", rel(res.total, grad)) <= 1e-12
        assert measured(case, "P12 hessian vs autograd rel",
                        rel(res.extras["hessian"], hess)) <= 1e-12
        assert res.extras["plane_residual"] <= 1e-12 * hess.abs().max()

        def f_r(x):
            nn = finite.rms_norm(x, gamma, eps).total
            return finite.silu(g @ nn) * (u @ nn)

        raw = finite.neuron_raw_hessian(g, u, gamma, eps, r)
        hess_r = torch.autograd.functional.hessian(f_r, r)
        assert measured(case, "P13 raw hessian vs autograd rel", rel(raw.total, hess_r)) <= 1e-10
        assert raw.extras["chain_rank"] <= 3


def test_p14_mlp_jacobian_and_pullback_vs_autograd_and_fd(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    generator = torch.Generator().manual_seed(14)
    for index in range(3):
        mlp, norm = layer(adapter, index).mlp, layer(adapter, index).post_attention_layernorm
        weights = [finite.f64(w) for w in (mlp.gate_proj.weight, mlp.up_proj.weight,
                                           mlp.down_proj.weight, norm.weight)]
        eps = norm.variance_epsilon
        r = vec(base.ports[index]["r"])

        def write(x):
            n = finite.rms_norm(x, weights[3], eps).total
            return weights[2] @ (finite.silu(weights[0] @ n) * (weights[1] @ n))

        res = finite.mlp_jacobian(r, *weights[:3], weights[3], eps)
        auto = torch.autograd.functional.jacobian(write, r)
        assert measured(case, "P14 J_M vs autograd rel", rel(res.total, auto)) <= 1e-12
        xi = torch.randn(r.shape, generator=generator, dtype=torch.float64) * r.std()
        errors = [float((write(r + e * xi) - write(r) - e * res.total @ xi).norm())
                  for e in (1e-2, 5e-3, 2.5e-3)]
        quadratic(errors, case, "P14")
        lam = torch.randn(r.shape, generator=generator, dtype=torch.float64)
        pull = finite.mlp_pullback(lam, r, *weights[:3], weights[3], eps)
        auto_pull = torch.autograd.functional.vjp(lambda x: x + write(x), r, lam)[1]
        assert measured(case, "P14 pullback vs autograd rel", rel(pull.total, auto_pull)) <= 1e-12
        assert (pull.extras["neuron_multiplier"] - weights[2].T @ lam).abs().max() == 0


def test_p15_finite_gate_up_split_slope_pairs_and_down_edit(bundle):
    case, adapter = bundle["case"], bundle["adapter"]
    pairs = [(bundle["base"], bundle["other"], i) for i in range(3)]
    pairs += [(bundle["base"], bundle["edited"], i) for i in (1, 2)]
    for c0, c1, index in pairs:
        p0, p1 = c0.ports[index], c1.ports[index]
        res = finite.swiglu_finite_split(vec(p0["g"]), vec(p1["g"]), vec(p0["u"]), vec(p1["u"]),
                                         native0=vec(p0["f"]), native1=vec(p1["f"]))
        exact(case, res)
        assert res.extras["symmetric_closure"] <= CLOSURE * res.reference.abs().max()
        errors = []
        for k in (1, 2, 4, 8, 16):
            slope = finite.integral_mean_slope(vec(p0["g"]), vec(p1["g"]), k)
            record(case, slope, f"integral_mean_slope_k{k}")
            errors.append(slope.closure_rel)
        assert errors[-1] <= 1e-9 and errors[-1] < errors[0]
    base = bundle["base"]
    mlp = layer(adapter, 1).mlp
    comps = {"emb": vec(base.ports[0]["h"]), "o0": vec(base.ports[0]["o"]),
             "m0": vec(base.ports[0]["m"]), "o1": vec(base.ports[1]["o"])}
    norm = layer(adapter, 1).post_attention_layernorm
    rho = finite.rms_norm(vec(base.ports[1]["r"]), norm.weight, norm.variance_epsilon)
    n_comps = {k: rho.extras["rho"] * finite.f64(norm.weight) * v for k, v in comps.items()}
    n_comps["rounding"] = vec(base.ports[1]["n"]) - sum(n_comps.values())
    for neuron in (0, 7, 33):
        res = finite.swiglu_source_pairs(mlp.gate_proj.weight[neuron], mlp.up_proj.weight[neuron],
                                         n_comps, native_f=vec(base.ports[1]["f"])[neuron])
        exact(case, res)
    generator = torch.Generator().manual_seed(15)
    d0 = finite.f64(mlp.down_proj.weight)
    d1 = d0 + torch.randn(d0.shape, generator=generator, dtype=torch.float64) * d0.std()
    exact(case, finite.down_finite_split(d0, d1, vec(base.ports[1]["f"]),
                                         vec(bundle["other"].ports[1]["f"])))


def test_p20_prediction_error_split_and_energy_weighting(bundle):
    case, adapter, base, other = bundle["case"], bundle["adapter"], bundle["base"], bundle["other"]
    mlp, norm = layer(adapter, 1).mlp, layer(adapter, 1).post_attention_layernorm
    weights = [finite.f64(w) for w in (mlp.gate_proj.weight, mlp.up_proj.weight,
                                       mlp.down_proj.weight, norm.weight)]
    eps = norm.variance_epsilon

    def block(x, keep=None):
        n = finite.rms_norm(x, weights[3], eps).total
        f = finite.silu(weights[0] @ n) * (weights[1] @ n)
        if keep is not None:
            f = f * keep
        return x + weights[2] @ f

    keep = (torch.arange(weights[0].shape[0]) % 2 == 0).double()
    res = finite.prediction_error_split(block, lambda x: block(x, keep), vec(base.ports[1]["r"]),
                                        vec(other.ports[1]["r"]))
    exact(case, res)
    target = torch.stack([vec(base.ports[i]["h_out"]) for i in range(3)] + [torch.zeros(32)])
    pred = torch.stack([vec(other.ports[i]["h_out"]) for i in range(3)] + [torch.ones(32)])
    res = finite.energy_weighted_error(pred, target)
    exact(case, res)
    assert res.terms["zero_rows"] > 0


def test_p27_second_order_vs_autograd_and_fd(bundle):
    case, adapter, base = bundle["case"], bundle["adapter"], bundle["base"]
    mlp, norm = layer(adapter, 1).mlp, layer(adapter, 1).post_attention_layernorm
    gw, uw, dw, gamma = (finite.f64(w) for w in (mlp.gate_proj.weight, mlp.up_proj.weight,
                                                  mlp.down_proj.weight, norm.weight))
    eps = norm.variance_epsilon
    r0 = vec(base.ports[1]["r"])
    xi1 = vec(bundle["other"].ports[1]["r"]) - r0
    xi2 = vec(base.ports[1]["o"])
    final = adapter.backbone.norm
    w_out = finite.f64(adapter.model.lm_head.weight)

    def gu(ab):
        n = finite.rms_norm(r0 + ab[0] * xi1 + ab[1] * xi2, gamma, eps).total
        return torch.stack([gw @ n, uw @ n])

    def f_of(ab):
        g, u = gu(ab)
        return finite.silu(g) * u

    def consumer(f):
        h = r0 + dw @ f
        x = finite.rms_norm(h, final.weight, final.variance_epsilon).total
        return (w_out[4] - w_out[9]) @ x

    zero = torch.zeros(2, dtype=torch.float64)
    jac = torch.func.jacfwd(gu)(zero)
    hess = torch.func.jacfwd(torch.func.jacrev(gu))(zero)
    g, u = gu(zero)
    res = finite.swiglu_second_order(g, u, jac[0, :, 0], jac[0, :, 1], jac[1, :, 0], jac[1, :, 1],
                                     hess[0, :, 0, 1], hess[1, :, 0, 1])
    auto = torch.func.jacfwd(torch.func.jacrev(f_of))(zero)[:, 0, 1]
    assert measured(case, "P27 f_ab vs autograd rel", rel(res.total, auto)) <= 1e-10
    assert res.terms["gate_trajectory"].abs().max() > 1e-6
    assert res.terms["up_trajectory"].abs().max() > 1e-6

    def mixed_fd(fn, e):
        return (fn(torch.tensor([e, e], dtype=torch.float64)) - fn(torch.tensor(
            [e, -e], dtype=torch.float64)) - fn(torch.tensor([-e, e], dtype=torch.float64))
            + fn(torch.tensor([-e, -e], dtype=torch.float64))) / (4 * e * e)

    quadratic([float((mixed_fd(f_of, e) - res.total).norm()) for e in (4e-2, 2e-2, 1e-2)],
              case, "P27 f_ab")
    f0 = f_of(zero)
    jf = torch.func.jacfwd(f_of)(zero)
    grad_h = torch.func.grad(consumer)(f0)
    hess_h = torch.func.hessian(consumer)(f0)
    m_ab = finite.consumer_second_order(grad_h, hess_h, jf[:, 0], jf[:, 1], res.total)
    auto_m = torch.func.jacfwd(torch.func.jacrev(lambda ab: consumer(f_of(ab))))(zero)[0, 1]
    assert measured(case, "P27 M_ab vs autograd rel", rel(m_ab.total, auto_m)) <= 1e-10
    measured(case, "P27 consumer curvature / M_ab",
             abs(float(m_ab.terms["consumer_curvature"])) / abs(float(auto_m)))
    assert abs(float(m_ab.terms["consumer_curvature"])) > 1e-6 * abs(float(auto_m))
    quadratic([abs(float(mixed_fd(lambda ab: consumer(f_of(ab)), e) - m_ab.total))
               for e in (4e-2, 2e-2, 1e-2)], case, "P27 M_ab")


def test_p28_path_quadrature_parseval_and_rectangle(bundle):
    case, adapter, base, other = bundle["case"], bundle["adapter"], bundle["base"], bundle["other"]
    final = adapter.backbone.norm
    w_out = finite.f64(adapter.model.lm_head.weight)

    def margin(h):
        x = finite.rms_norm(h, final.weight, final.variance_epsilon).total
        return (w_out[4] - w_out[9]) @ x

    grad = torch.func.grad(margin)
    x0, x1 = vec(base.ports[2]["h_out"]), vec(other.ports[2]["h_out"])
    errors = []
    for k in (1, 2, 4, 8, 16):
        res = finite.path_integral(grad, x0, x1, k, endpoint_fn=margin)
        record(case, res, f"path_integral_k{k}")
        errors.append(res.closure_rel)
    assert errors[-1] <= 1e-9 and errors[-1] < errors[0]
    q, keys, values = head_operands(adapter, base, 1, 0)
    _, _, values1 = head_operands(adapter, other, 1, 0)
    scaling = layer(adapter, 1).self_attn.scaling
    lam = torch.ones(adapter.head_dim, dtype=torch.float64)
    adj = finite.attention_differential(q, keys, values, scaling=scaling, adjoint=lam)
    exact(case, finite.parseval_split(adj.extras["grad_v"],
                                      finite.f64(values1) - finite.f64(values), axis=0))
    a = -vec(base.ports[2]["m"])
    b = -vec(base.ports[2]["o"])
    rects = [finite.rectangle(margin, x0, a, b, hess_fn=torch.func.hessian(margin), k=k)
             for k in (2, 8)]
    for k, res in zip((2, 8), rects):
        record(case, res, f"rectangle_k{k}")
    assert rects[1].closure_rel < rects[0].closure_rel and rects[1].closure_rel <= 1e-6


def test_p29_mobius_and_walsh_hadamard():
    two = {(): -4.750, ("a",): -0.875, ("b",): -3.875, ("a", "b"): 0.375}
    inter = finite.mobius(two)
    assert inter == {frozenset(): -4.75, frozenset("a"): 3.875, frozenset("b"): 0.875,
                     frozenset("ab"): 0.375}
    wht = finite.walsh_hadamard(two)
    assert wht == {frozenset(): -2.28125, frozenset("a"): -2.03125, frozenset("b"): -0.53125,
                   frozenset("ab"): 0.09375}
    coef = {(): 1, ("a",): 2, ("b",): 3, ("c",): 5, ("a", "b"): 7, ("a", "c"): 11,
            ("b", "c"): 13, ("a", "b", "c"): 17}
    table = finite.mobius_inverse(coef)
    assert table[frozenset("abc")] == 59 and table[frozenset("bc")] == 22
    assert table[frozenset("a")] == 3 and table[frozenset()] == 1
    assert finite.mobius(table) == {frozenset(k): v for k, v in coef.items()}
    wht3 = finite.walsh_hadamard(table)
    assert wht3[frozenset()] == 15.875 and wht3[frozenset("a")] == -7.625
    assert wht3[frozenset("abc")] == -2.125
    generator = torch.Generator().manual_seed(29)
    items = ("x", "y", "z", "w")
    cube = {frozenset(c): float(torch.randn(1, generator=generator))
            for r in range(5) for c in __import__("itertools").combinations(items, r)}
    back = finite.mobius_inverse(finite.mobius(cube))
    assert max(abs(back[k] - cube[k]) for k in cube) <= 1e-12
    back = finite.walsh_hadamard_inverse(finite.walsh_hadamard(cube))
    assert max(abs(back[k] - cube[k]) for k in cube) <= 1e-12
    partial = {k: v for k, v in two.items() if k != ("b",)}
    assert frozenset("ab") not in finite.mobius(partial)
    with pytest.raises(ValueError, match="complete cube"):
        finite.walsh_hadamard(partial)


def test_summary_carries_the_results_own_authority_and_structured_approximation(bundle):
    case, base, other = bundle["case"], bundle["base"], bundle["other"]
    g0, g1 = vec(base.ports[1]["g"]), vec(other.ports[1]["g"])
    slope = finite.integral_mean_slope(g0, g1, 3)
    out = slope.summary()
    assert out["authority"] == "derived-local"
    assert out["approximation"] == {"rule": "gauss-legendre", "k": 3}
    for key in ("closure_max_abs", "closure_rel", "total_max_abs"):
        assert out[key]["authority"] == "derived-local", key
    assert out["state"] == "resolved" and out["reason"] is None
    assert not unlabelled_numbers(out), unlabelled_numbers(out)
    final = bundle["adapter"].backbone.norm
    w_out = finite.f64(bundle["adapter"].model.lm_head.weight)

    def margin(h):
        x = finite.rms_norm(h, final.weight, final.variance_epsilon).total
        return (w_out[4] - w_out[9]) @ x

    x0, x1 = vec(base.ports[2]["h_out"]), vec(other.ports[2]["h_out"])
    path = finite.path_integral(torch.func.grad(margin), x0, x1, 4, endpoint_fn=margin)
    po = path.summary()
    assert po["approximation"] == {"rule": "gauss-legendre", "k": 4}
    assert po["closure_max_abs"]["authority"] == "derived-local"
    assert all(v["authority"] == "derived-local" for v in po["terms_max_abs"].values())
    a, b = -vec(base.ports[2]["m"]), -vec(base.ports[2]["o"])
    rect = finite.rectangle(margin, x0, a, b, hess_fn=torch.func.hessian(margin), k=4)
    ro = rect.summary()
    # exact four-corner total, derived-local closure against the quadrature reference
    assert ro["authority"] == "derived-exact" and ro["approximation"]["k"] == 4
    assert ro["total_max_abs"]["authority"] == "derived-exact"
    assert ro["closure_rel"]["authority"] == "derived-local"
    assert ro["closure_max_abs"]["authority"] == "derived-local"
    plain = finite.rectangle(margin, x0, a, b)  # no quadrature reference
    assert plain.summary()["authority"] == "derived-exact" and "approximation" not in plain.summary()
    # a forward identity keeps derived-exact
    rms = finite.rms_finite_change(vec(base.ports[1]["h"]), vec(other.ports[1]["h"]),
                                   layer(bundle["adapter"], 1).input_layernorm.weight, 1e-6)
    assert rms.summary()["closure_max_abs"]["authority"] == "derived-exact"
    assert rms.summary()["terms_max_abs"]["rho0_dx"]["authority"] == "derived-exact"


def test_walker_flags_bare_numbers():
    assert unlabelled_numbers({"x": 1.5, "y": {"value": 1.0, "authority": "measured"},
                               "layer": 3, "n": 4, "z": [2.0]}) == [".x = 1.5", ".n = 4",
                                                                    ".z[0] = 2.0"]
    assert unlabelled_numbers({"v": {"value": 1.0, "authority": "guess"}})


def test_every_summary_number_is_a_labelled_number(bundle):
    """Walk the summary of every result kind exercised here: no bare floats outside the allowlist."""
    adapter, base, other, one = bundle["adapter"], bundle["base"], bundle["other"], bundle["one"]
    kappa = finite.gqa_kappa(adapter.num_heads, adapter.num_kv_heads)
    scaling = layer(adapter, 1).self_attn.scaling
    q, keys, values = head_operands(adapter, base, 1, 0)
    p64 = torch.softmax(scaling * finite.f64(keys) @ finite.f64(q), -1)
    results = [
        _one_source(adapter, base, one, 0, 0), _one_source(adapter, base, one, 1, 1),
        finite.softmax_contrasts(q, keys, 3, scaling=scaling,
                                 native_probs=base.ports[1]["p"][0, 0, 0]),
        finite.anchored_transport(base.ports[1]["p"][0, 0, 0], values, 2),
        finite.query_shift_invariance(q, keys, _perp(keys, torch.ones(adapter.head_dim,
                                                                         dtype=torch.float64)),
                                      scaling=scaling),
        finite.attention_differential(q, keys, values, scaling=scaling),
        *finite.cut_sources(p64, finite.f64(values), [0, 3]).values(),
        finite.integral_mean_slope(vec(base.ports[1]["g"]), vec(other.ports[1]["g"]), 2),
        finite.rms_jacobian(vec(base.ports[1]["r"]), layer(adapter, 1).post_attention_layernorm.weight,
                            1e-6),
    ]
    for res in results:
        out = res.summary()
        assert out["state"] in STATE and (out["state"] == "resolved") == (out["reason"] is None)
        assert (out["authority"] is None) == (out["state"] != "resolved")
        assert out["authority"] is None or out["authority"] in AUTHORITY
        bad = unlabelled_numbers(out)
        assert not bad, (res.name, bad)
        for entry in out["premises"].values():
            assert isinstance(entry["holds"], bool)
            assert is_labelled(entry["value"]) and (entry["tolerance"] is None
                                                    or is_labelled(entry["tolerance"]))
    assert kappa == [0, 0, 1, 1]


def test_catalogue_p04_owns_the_rope_identities_and_p26_is_not_implemented():
    p04, p26 = finite.IDENTITIES["P04"], finite.IDENTITIES["P26"]
    for name in ("rope_rotate", "rope_relative", "rope_score_pairs"):
        assert name in p04["functions"] and name not in p26["functions"]
    assert p04["title"] == "Affine Q/K/V, GQA, positional pair expansion"
    assert p26["title"] == "RoPE score kernel's two-frequency transform and off-grid caveat"
    rect = finite.IDENTITIES["P28"]["functions"]["rectangle"]
    assert rect["authority"] == "derived-exact" and rect["closure_authority"] == "derived-local"
    assert p26["status"] == "not implemented" and not p26["functions"]
    assert p04["status"] == "implemented"
    listing = {e["proof"]: e for e in finite.list_identities()}
    assert listing["P26"]["status"] == "not implemented" and listing["P26"]["functions"] == []
    assert {"rope_rotate", "rope_relative", "rope_score_pairs"} <= {
        f["function"] for f in listing["P04"]["functions"]}


def test_num_fallback_validates_like_records_num(monkeypatch):
    from mdb.core import records

    monkeypatch.setattr(finite, "_RECORD_NUM", [None])  # force the local fallback
    cases = [(3, "supplied"), (2.5, "measured"), (True, "measured"), (float("nan"), "measured"),
             (float("inf"), "measured"), ("1", "measured"), (1.0, "guess")]
    for value, authority in cases:
        try:
            want = records.num(value, authority, proof="P19")
        except ValueError:
            with pytest.raises(ValueError):
                finite.num(value, authority, "P19")
        else:
            assert finite.num(value, authority, "P19") == want
            assert type(finite.num(value, authority)["value"]) is type(want["value"])
    assert tuple(records.AUTHORITIES) == finite.AUTHORITIES or set(records.AUTHORITIES) == set(
        finite.AUTHORITIES)


def test_import_is_torch_free_and_catalogue_listable():
    code = ("import sys, json, mdb.analysis as a; assert 'torch' not in sys.modules; "
            "json.dumps(a.list_identities()); assert 'torch' not in sys.modules; "
            "print(len(a.IDENTITIES))")
    env = dict(os.environ)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == len(finite.IDENTITIES)
    for proof, entry in finite.IDENTITIES.items():
        assert entry["kind"]
        assert entry["functions"] or entry["status"] == "not implemented", proof
        for item in entry["functions"].values():
            assert item["authority"] in {"derived-exact", "derived-local", "bound"}
            assert callable(item["callable"])
