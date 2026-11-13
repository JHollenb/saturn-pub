"""Saved-only custody, MDB state/port joins and independent attention accounting."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
from mdb_lanes.forecast_transaction import verify_transaction, load_forecast_tree
from saturn_pub.store import LocalStore


def read(path):
    return json.loads(Path(path).read_text())


def tree(value):
    if isinstance(value, Mapping):
        return {k: tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tree(v) for v in value]
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def metric(value):
    a = np.asarray(value, np.float64)
    return dict(rms=float(np.sqrt(np.mean(a*a))), l2=float(np.linalg.norm(a)),
                max_abs=float(np.max(np.abs(a))), signed_mean=float(a.mean()), scalars=int(a.size))


def error(actual, expected):
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    d = a-e
    result = metric(d)
    result.update(exact=bool(np.array_equal(a, e)), changed_scalars=int(np.count_nonzero(d)),
                  relative_rms=result["rms"]/max(metric(e)["rms"], 1e-300))
    return result


def bits(value):
    a = np.asarray(value, np.float32)
    u = a.view(np.uint32)
    if np.any(u & np.uint32(65535)):
        raise ValueError("native port is not exact widened BF16")
    return hashlib.sha256((u >> 16).astype("<u2").tobytes()).hexdigest()


def B(value):
    a = np.asarray(value, np.float32)
    u = a.view(np.uint32)
    return ((u + np.uint32(32767) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def reference(q, k, v):
    """Independent all-key softmax contraction, no staged dense() call."""
    q, k, v = [np.asarray(a, np.float64) for a in (q, k, v)]
    logits = np.einsum("bhqd,bhkd->bhqk", q, k, optimize=False)/np.sqrt(q.shape[-1])
    exp = np.exp(logits-logits.max(-1, keepdims=True))
    p = exp/exp.sum(-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", p, v, optimize=False), p


def audit(collection, output):
    here = Path(__file__).resolve().parent
    limits = {"max_member_bytes": 160 * 1024 * 1024}
    custody = runpy.run_path(str(here / "flux_family_custody.py"))["verify"](collection, limits)
    assert read(collection / "collection-limits.json")["overrides"] == limits
    data = collection / "extracted/binding-minimal-result"
    config, summary, rows = [read(data / n) for n in ("plan.json", "summary.json", "rows.json")]
    request = read(collection / "request.json")
    assert config == request["config"] and summary == read(collection / "report.json")["report"]
    assert summary["request_sha256"] == request["request_sha256"] == custody["request_sha256"]
    assert summary["counts"] == config["expected_counts"]
    for name, fingerprint in request["source_hashes"].items():
        assert sha(collection / "payload" / name) == fingerprint
    transaction = data / "transaction"
    verified = verify_transaction(transaction, expected_seal_sha256=summary["seal"]["seal_sha256"])
    assert verified["verified"] and verified["phase"] == "closed"
    frozen = tree(load_forecast_tree(transaction, read(transaction / "forecasts/raw_state.json")["predictions"]))
    observed = tree(load_forecast_tree(transaction, read(transaction / "observations/native.json")["value"]))
    census = {k: v if k in {"component_loads", "numpy_block_forecasts", "numpy_corner_forecasts"} else 0
              for k, v in config["expected_counts"].items()}
    assert frozen["counts_at_seal"] == read(data / "seal-census.json")["counts"] == census
    assert frozen["hypotheses"] == config["hypotheses"]
    assert observed["rows"] == rows and observed["counts"] == summary["counts"]
    source = read(data / "native-source.json")
    recipes = {r["source"]["tensor"].removeprefix("transformer_blocks.1."): r["source"]
               for r in read(data / "parameter-artifacts.json")["artifacts"]}
    assert set(source["parameters"]) == set(recipes) and len(recipes) == 16
    for name, descriptor in source["parameters"].items():
        assert descriptor["sha256"] == recipes[name]["payload_sha256"]
        assert descriptor["shape"] == recipes[name]["shape"] and descriptor["dtype"] == "torch.bfloat16"
    public_receipt = runpy.run_path(str(here / "summarize_context_window.py"))["public_receipt"]
    receipts = read(data / "receipts.json")
    assert len(receipts) == 3 and [r["receipt"] for r in rows] == receipts
    # Use the public typed reader to validate input/output state boundaries.
    store = LocalStore(data / "store")
    input_arrays = {n: np.load(collection / "payload" / (n + ".npy"), allow_pickle=False) for n in config["input_files"]}
    inputs = dict(base=input_arrays["text_base"], changed=input_arrays["text_changed"],
                  midpoint=B((input_arrays["text_base"].astype(np.float64)+input_arrays["text_changed"])/2))
    def fields(fp):
        m = read(data / "store/cuts" / (fp + ".json"))["manifest"]
        return {i["key"]["value"]: i["value"] for i in m["payload"]["items"]}
    native, comparisons, joins, equations = {}, {}, {}, {}
    expected_ports = {
        "q_raw": "attn.to_q:return[selected-rows]", "image_k_raw": "attn.to_k:return[selected-rows]",
        "text_k_raw": "attn.add_k_proj:return[selected-rows]", "q_norm": "attn.norm_q:return[selected-rows]",
        "image_k_norm": "attn.norm_k:return[selected-rows]", "text_k_norm": "attn.norm_added_k:return[selected-rows]",
        "attention": "attn:return[selected-rows]", "projected": "ff.linear_in:return[selected-rows]",
        "activated": "ff.act_fn.gate_fn:return[selected-rows]", "product": "ff.act_fn:return[selected-rows]",
        "mlp": "ff:return[selected-rows]", "mlp_input": "ff:input[selected-image-rows]",
        "norm_image": "attn:input.hidden_states[selected-image-rows]",
        "norm_text": "attn:input.encoder_hidden_states[selected-text-rows]",
        "q": "native.dispatch_attention_fn:input.q[positioned,image-queries]",
        "k": "native.dispatch_attention_fn:input.k[positioned,all-keys]",
        "v": "native.dispatch_attention_fn:input.v[all-values]",
        "head": "native.dispatch_attention_fn:return[image-queries]",
        "image_exit": "native.block:return.image[selected-image-rows]"}
    for name, row in zip(config["cases"], rows, strict=True):
        assert row["id"] == name
        receipt = row["receipt"]
        public_receipt(receipt)
        assert receipt == read(data / "store/receipts" / (receipt["fingerprint"] + ".json"))
        assert receipt["operation"] == "continue" and receipt["steps"] == 1
        assert receipt["parent"] == row["parent"] and receipt["result"] == row["result"]
        before, after = store.load(row["parent"]), store.load(row["result"])
        assert before.model_identity == after.model_identity == config["model"]
        assert before.execution["device"] == after.execution["device"] == "cpu"
        assert before.payload["advanced"] == 0 and after.payload["advanced"] == 1
        assert before.execution_point["index"] == 1 and after.execution_point["index"] == 2
        for field in ("image", "text", "img_mod", "txt_mod", "cos", "sin"):
            expected = inputs[name] if field == "text" else input_arrays[field]
            actual = before.payload[field].float().numpy()
            assert np.array_equal(actual, expected)
        pre, post = fields(row["parent"]), fields(row["result"])
        assert all(pre[k] == post[k] for k in pre if k not in {"image", "text", "advanced"})
        path = data / row["archive"]["path"]
        assert sha(path) == row["archive"]["sha256"]
        with np.load(path, allow_pickle=False) as archive:
            native[name] = {k: archive[k] for k in archive.files}
        assert set(native[name]) == set(frozen["blocks"][name]) == set(row["ports"]) == set(expected_ports)
        for field, value in native[name].items():
            port = row["ports"][field]
            assert port["port"] == expected_ports[field]
            assert port["descriptor"]["sha256"] == bits(value)
            assert port["descriptor"]["shape"] == list(value.shape) and port["descriptor"]["dtype"] == "torch.bfloat16"
            assert np.isfinite(value).all()
        actual_exit = after.payload["image"][:, config["indices"]].float().numpy()
        assert np.array_equal(actual_exit, native[name]["image_exit"])
        comparisons[name] = {k: error(v, frozen["blocks"][name][k]) for k, v in native[name].items()}
        joins[name] = dict(ports=len(row["ports"]), input_state_exact=True, exit_state_exact=True,
                           unchanged_context=True, receipt=receipt["fingerprint"])
        mod = np.split(input_arrays["img_mod"].reshape(1, 1, -1), 6, axis=-1)
        incoming = input_arrays["image"][:, config["indices"]]
        U = B(incoming+B(mod[2]*native[name]["attention"]))
        rebuilt = B(U+B(mod[5]*native[name]["mlp"]))
        equations[name] = error(native[name]["image_exit"], rebuilt)
        del before, after
    corner_path = data / observed["corners"]["path"]
    assert sha(corner_path) == observed["corners"]["sha256"]
    with np.load(corner_path, allow_pickle=False) as z:
        corners = {k: z[k] for k in z.files}
    corner_results = {}
    for name in config["corners"]:
        kn = "changed" if name[1] == "1" else "base"
        vn = "changed" if name[3] == "1" else "base"
        independent, _ = reference(frozen["blocks"]["base"]["q"], frozen["blocks"][kn]["k"], frozen["blocks"][vn]["v"])
        corner_results[name] = dict(native_vs_sealed=error(corners[name], frozen["corners"][name]),
                                   sealed_vs_independent=error(frozen["corners"][name], B(independent)))
    base, changed = native["base"], native["changed"]
    out0, p0 = reference(base["q"], base["k"], base["v"])
    out1, p1 = reference(base["q"], changed["k"], changed["v"])
    dp, dv = p1-p0, changed["v"].astype(np.float64)-base["v"]
    contract = lambda p, v: np.einsum("bhqk,bhkd->bhqd", p, np.asarray(v, np.float64), optimize=False)
    parts = dict(route_text=contract(dp[..., :512], base["v"][..., :512, :]),
                 route_image=contract(dp[..., 512:], base["v"][..., 512:, :]),
                 value=contract(p0, dv), interaction=contract(dp, dv))
    target = out1-out0
    closure = sum(parts.values())-target
    metrics = read(data / "metrics.json")
    with np.load(data / "routing-components.npz", allow_pickle=False) as z:
        independent_joins = {k: error(parts[k], z[k]) for k in parts}
        independent_joins["delta"] = error(target, z["delta"])
    structural = {name: dict(q=error(v["q"], base["q"]),
        image_k=error(v["k"][..., 512:, :], base["k"][..., 512:, :]),
        image_v=error(v["v"][..., 512:, :], base["v"][..., 512:, :])) for name, v in native.items()}
    text_mass0, text_mass1 = p0[..., :512].sum(-1), p1[..., :512].sum(-1)
    result = dict(schema="mdb-flux-attention-state-law-audit-v1", mechanics_status="verified",
        instrument_status="applicable typed standalone-component state/port observations; no ATen replay",
        trend_status="per-stage errors and routing/value contributions retained", terminal_status="unassessed",
        job_id=summary["job_id"], request_sha256=request["request_sha256"], custody=custody, transaction=verified,
        counts=summary["counts"], state_port_joins=joins, parameter_recipe_joins=16,
        errors=comparisons, actual_port_residual_equations=equations, structural=structural, corners=corner_results,
        routing_components={k: metric(v) for k, v in parts.items()}, routing_delta=metric(target), routing_closure=metric(closure),
        independently_rebuilt_component_joins=independent_joins,
        observed_query_changed=not structural["changed"]["q"]["exact"],
        text_read_mass=dict(base_mean=float(text_mass0.mean()), changed_mean=float(text_mass1.mean()),
            delta=metric(text_mass1-text_mass0), increased=int(np.count_nonzero(text_mass1 > text_mass0)),
            decreased=int(np.count_nonzero(text_mass1 < text_mass0)), query_heads=int(text_mass0.size)),
        native_head_rounding=dict(base=error(base["head"], out0), changed=error(changed["head"], out1)),
        worker_seconds=summary["worker_seconds"], new_model_calls_in_audit=0,
        scope="32image queries read all1536keys/values. Raw full block entry observed; midpoint T newly tested. Forecast does not consume native intermediates. Text exits/full next-block closure not predicted. Algebra splits are post-hoc numeric explanations, not semantic labels.",
        audit_sources={p.name: sha(p) for p in (Path(__file__), here / "flux_family_custody.py", here / "summarize_context_window.py")})
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return dict(mechanics_status=result["mechanics_status"],
        errors={n: {k: v[k] for k in ("q", "k", "v", "head", "attention", "mlp_input", "mlp", "image_exit")} for n, v in comparisons.items()},
        routing_components=result["routing_components"], routing_delta=result["routing_delta"], routing_closure=result["routing_closure"],
        text_read_mass=result["text_read_mass"], structural=structural,
        corners=corner_results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("collection", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.collection.resolve(), args.output.resolve()), sort_keys=True))
