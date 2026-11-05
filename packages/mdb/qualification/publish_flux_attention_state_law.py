"""Index audited attention computations and explain saved routing contrasts.

Post-hoc analysis only. No model calls, fitted parameters or semantic selectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
from mdb.equation_comparison import EquationCatalog, compare_equations
from mdb.forecast_transaction import load_forecast_tree


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(root, collection, output):
    helper_path = root / "audit-sources-v2/analyze_flux_attention_state_law.py"
    helper = runpy.run_path(str(helper_path))
    metric, error, B, reference = [helper[k] for k in ("metric", "error", "B", "reference")]
    audit = read(root / "audit-v1.json")
    assert audit["mechanics_status"] == "verified"
    assert audit["audit_sources"][helper_path.name] == sha(helper_path)
    data = collection / "extracted/binding-minimal-result"
    config, summary, rows = [read(data / n) for n in ("plan.json", "summary.json", "rows.json")]
    assert summary["request_sha256"] == audit["request_sha256"]
    tx = data / "transaction"
    predicted = helper["tree"](load_forecast_tree(tx, read(tx / "forecasts/raw_state.json")["predictions"]))
    native = {}
    for row in rows:
        path = data / row["archive"]["path"]
        assert sha(path) == row["archive"]["sha256"]
        with np.load(path, allow_pickle=False) as z:
            native[row["id"]] = {n: z[n] for n in z.files}
    output.mkdir(exist_ok=False, parents=True)
    catalog = EquationCatalog(output / "equations")
    records, queries = [], []
    evidence_paths = [root / "audit-v1.json", data / "transaction/seal.json", data / "native-source.json",
                      data / "parameter-artifacts.json", collection / "collection-limits.json"]
    refs = [dict(path=str(p), sha256=sha(p)) for p in evidence_paths]

    def index(name, law, route, context, forecast, observed, readers, **evidence):
        report = compare_equations(forecast, observed, max_elements=20_000_000)
        record = catalog.record(law_id=law, route_id=route, alias="cmp.flux.attention."+name,
            binding=dict(model_identity=config["model"], execution=context["execution"],
                input_identity=context["input_identity"], source_identity=audit["request_sha256"],
                horizon=context["horizon"], frontier=context["frontier"]),
            report=report, conditions=[dict(field=k, op="eq", value=v) for k, v in context.items()],
            tested_context=context, readers=readers,
            evidence=dict(artifacts=refs, seal=summary["seal"], new_fitted_parameters=0,
                terminal_status="unassessed", semantic_labels="unassigned", **evidence))
        assert len(catalog.find(applicable_to=context)) == 1
        altered = {**context, "execution": {**context["execution"], "backend": "cuda"}}
        assert not catalog.find(applicable_to=altered)
        altered = {**context, "clock": {**context["clock"], "block": 2}}
        assert not catalog.find(applicable_to=altered)
        records.append(record["fingerprint"])
        queries.append(dict(case=name, record=record["fingerprint"], tested_context=context))

    execution = dict(backend="cpu", device="cpu", dtype="bfloat16", threads=4,
        geometry="native1536queries/all1536keys;forecast32imagequeries/all1536keys")
    for name, row in zip(config["cases"], rows, strict=True):
        context = dict(model_identity=config["model"], checkpoint=config["checkpoint_sha256"],
            execution=execution, input_identity=row["parent"], case=name,
            clock=dict(step=2, phase="joint", block=1), image_rows=config["indices"],
            text_witness_rows=config["text_indices"], horizon="selected image exit of original joint1 block",
            frontier=dict(kind="observed raw block-entry image/text/context", component_parent=row["parent"]))
        index(name, "law.flux.joint-image-state.v1", "route.raw-state.qkv.softmax.image-mlp.residual.v1",
            context, predicted["blocks"][name], native[name], ["native.block:return.image[selected-image-rows]"],
            state_status="new whole-block internal midpoint input" if name == "midpoint" else "previously observed source endpoint",
            complete_text_exit_forecast=False, next_block_closure=False, port_trace="port observations; no ATen replay")

    with np.load(data / "native-corners.npz", allow_pickle=False) as z:
        corners = {n: z[n] for n in z.files}
    for name in config["corners"]:
        kn, vn = ("changed" if name[1] == "1" else "base"), ("changed" if name[3] == "1" else "base")
        context = dict(model_identity=config["model"], checkpoint=config["checkpoint_sha256"],
            execution={**execution, "geometry": "native32queries/all1536keys supplied predicted QKV"},
            input_identity=summary["seal"]["seal_sha256"]+":"+name, case=name,
            clock=dict(step=2, phase="joint", block=1), horizon="selected supplied-operand SDPA head return",
            frontier=dict(kind="sealed predicted QKV", q="base", k=kn, v=vn))
        index(name, "law.attention.softmax-value.v1", "route.positioned-qkv.softmax-value.v1",
            context, {"head": predicted["corners"][name]}, {"head": corners[name]},
            ["torch.nn.functional.scaled_dot_product_attention:return"], original_block_continuation=False)

    # A small endpoint error can conceal a much larger error on the difference.
    effects = {}
    for name in ("changed", "midpoint"):
        effects[name] = {}
        for field in ("q", "k", "v", "head", "attention", "mlp_input", "mlp", "image_exit"):
            actual = native[name][field].astype(np.float64)-native["base"][field]
            forecast = predicted["blocks"][name][field].astype(np.float64)-predicted["blocks"]["base"][field]
            e = error(actual, forecast)
            magnitude = metric(actual)["rms"]
            e["error_relative_to_observed_change"] = e["rms"]/magnitude if magnitude else None
            effects[name][field] = dict(observed=metric(actual), predicted=metric(forecast), error=e)

    base, changed = native["base"], native["changed"]
    out0, p0 = reference(base["q"], base["k"], base["v"])
    out1, p1 = reference(base["q"], changed["k"], changed["v"])
    assert audit["observed_query_changed"] is False
    assert audit["structural"]["changed"]["image_k"]["exact"] and audit["structural"]["changed"]["image_v"]["exact"]
    with np.load(data / "routing-components.npz", allow_pickle=False) as z:
        parts = {k: z[k] for k in ("route_text", "route_image", "value", "interaction")}
    delta = out1-out0
    t0, t1 = p0[..., :512].sum(-1), p1[..., :512].sum(-1)
    scale = (1-t1)/(1-t0)
    original_image_read = np.einsum("bhqk,bhkd->bhqd", p0[..., 512:], base["v"][..., 512:, :].astype(np.float64), optimize=False)
    image_reweighting = (scale[..., None]-1)*original_image_read
    ratio = error(parts["route_image"], image_reweighting)
    def cosine(a, b):
        den = np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.vdot(a, b)/den) if den else None
    heads = []
    for h in range(24):
        d = delta[:, h]
        terms = {k: v[:, h] for k, v in parts.items()}
        heads.append(dict(head=h, delta=metric(d), contributions={k: metric(v) for k, v in terms.items()},
            routing_value_cosine=cosine(terms["route_text"]+terms["route_image"], terms["value"]),
            text_mass_base=float(t0[:, h].mean()), text_mass_changed=float(t1[:, h].mean()),
            increased_queries=int(np.count_nonzero(t1[:, h] > t0[:, h])),
            decreased_queries=int(np.count_nonzero(t1[:, h] < t0[:, h]))))
    routing = dict(unchanged_image_normalizer_identity=ratio, per_head=heads,
        routing_value_cosine=cosine(parts["route_text"]+parts["route_image"], parts["value"]),
        additive_without_interaction=metric(parts["route_text"]+parts["route_image"]+parts["value"]),
        interaction_additive_cosine=cosine(parts["interaction"], parts["route_text"]+parts["route_image"]+parts["value"]),
        native_heads_vs_rounded_independent=dict(base=error(base["head"], B(out0)), changed=error(changed["head"], B(out1))),
        interpretation="Post-hoc original observed QKV accounting. Read probabilities are not semantic importance. Component norms can cancel; no causal percentages.")
    result = dict(schema="mdb-flux-attention-state-law-publication-v1", equation_occurrences=len(records),
        selected_equation_ids=records, exact_context_queries=len(records), changed_backend_abstentions=len(records),
        changed_clock_abstentions=len(records), queries=queries, effects=effects, routing=routing,
        source_job=audit["job_id"], source_request=audit["request_sha256"], new_model_calls=0,
        qualification_sources={p.name: sha(p) for p in (Path(__file__), helper_path)}, artifacts=refs)
    with (output / "provenance.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for row in queries:
        with (output / ("context-"+row["case"]+".json")).open("x") as stream:
            json.dump(row["tested_context"], stream, indent=2, sort_keys=True)
            stream.write("\n")
    return dict(equation_occurrences=len(records), effects=effects, normalizer_identity=ratio,
                routing_value_cosine=routing["routing_value_cosine"], new_model_calls=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("collection", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    print(json.dumps(publish(a.root.resolve(), a.collection.resolve(), a.output.resolve()), sort_keys=True))
