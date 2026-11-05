"""Index audited MLP computation records and exercise native MDB math views."""
import argparse
import json
from pathlib import Path
import runpy

from mdb.equation_comparison import EquationCatalog, compare_equations
from mdb.forecast_transaction import load_forecast_tree
from mdb.forward_algebra import acquire_symbol_candidates
from mdb.forward_ports import NativePortCatalog
from mdb.forward_readable import ForwardReadableProgram
from mdb.forward_symbols import ForwardSymbolTable


def publish(root, output):
    here = Path(__file__).resolve().parent
    helper = runpy.run_path(str(here / "flux_mlp_state_law.py"))
    decode = runpy.run_path(str(here / "analyze_flux_mlp_state_law.py"))["json_tree"]
    read, sha, save = helper["read"], helper["sha"], helper["save"]
    audit = read(root / "audit-v1.json")
    assert audit["mechanics_status"] == "verified"
    data = root / "collection-01/extracted/binding-minimal-result"
    config, summary = read(data / "plan.json"), read(data / "summary.json")
    tx = data / "transaction"
    predicted = decode(load_forecast_tree(tx, read(tx / "forecasts/stage_outputs.json")["predictions"]))["fields"]
    actual = decode(load_forecast_tree(tx, read(tx / "observations/native.json")["value"]))["fields"]
    output.mkdir(exist_ok=False, parents=True)
    catalog = EquationCatalog(output / "equations")
    ids, queries = [], []
    for name in config["cases"]:
        trace = read(data / "traces" / (name + ".json"))
        report = compare_equations(predicted[name], actual[name], max_elements=2_000_000)
        context = dict(model_identity=trace["model_identity"], execution=trace["execution"],
            input_identity=trace["input_identity"], source_identity=sha(data / "request.json"),
            horizon="isolated image-MLP return", case=name,
            native_input_shape=[1, 32, 3072], selected_source_rows=config["selected_rows"])
        refs = [dict(path=str(p), sha256=sha(p)) for p in (
            root / "audit-v1.json", root / "state-chain-v1.json", data / "traces" / (name + ".json"),
            data / "parameter-artifacts.json", data / "transaction/seal.json")]
        record = catalog.record(law_id="law.flux.image-swiglu.v1",
            route_id="route.image-normalized-row.gated-product.output-projection.v1", alias="cmp.flux.mlp."+name,
            binding=dict(model_identity=context["model_identity"], execution=context["execution"],
                input_identity=context["input_identity"], source_identity=context["source_identity"],
                horizon=context["horizon"], frontier=dict(kind="observed-normalized-MLP-input", source_job=config["source_job"])),
            report=report, conditions=[dict(field=k, op="eq", value=v) for k, v in context.items()],
            tested_context=context, readers=["native.module.image-MLP.return"],
            evidence=dict(artifacts=refs, seal=summary["seal"], checkpoint_parameters="exact named original tensors; no fitted coefficients",
                status="calibration endpoint" if name in ("base", "changed") else "new local native input after seal",
                continuation="Module output observed; no new full-model/image continuation."))
        identifier = record["fingerprint"]
        ids.append(identifier)
        assert len(catalog.find(applicable_to=context)) == 1
        assert not catalog.find(applicable_to={**context, "execution": {**context["execution"], "backend": "cuda"}})
        queries.append(dict(case=name, record=identifier, tested_context=context))
    midpoint = read(data / "traces/midpoint.json")
    readable = ForwardReadableProgram(midpoint)
    table, ports = ForwardSymbolTable(midpoint), NativePortCatalog(midpoint)
    algebra = acquire_symbol_candidates(readable)
    save(output / "algebra.json", algebra)
    (output / "readable.txt").write_text(readable.render() + "\n")
    selected = {p: ports.resolve(p) for p in (
        "module:linear_in:return", "module:act_fn:return", "module:linear_out:return",
        "parameter:linear_in.weight", "parameter:linear_out.weight")}
    product = selected["module:act_fn:return"]["value_id"]
    save(output / "native-walkthrough.json", dict(trace_fingerprint=midpoint["fingerprint"],
        ports=selected, product_ancestry=table.inspect(product), operations=table.list_steps(),
        algebra_graph_candidates=len(algebra["graph_candidates"]),
        algebra_scope="Structural identities are built-in templates, not discoveries from this run. No semantic labels inferred."))
    result = dict(schema="mdb-flux-mlp-state-law-publication-v1", equation_occurrences=len(ids),
        selected_equation_ids=ids, exact_context_queries=5, changed_backend_abstentions=5,
        queries=queries, native_operations_per_trace=len(table.list_steps()),
        automatic_algebra_graph_candidates=len(algebra["graph_candidates"]),
        native_parameter_ports=2, new_model_calls=0,
        qualification_sources={p.name: sha(p) for p in (Path(__file__), here / "flux_mlp_state_law.py", here / "analyze_flux_mlp_state_law.py")})
    save(output / "provenance.json", result)
    return {k: result[k] for k in ("equation_occurrences", "exact_context_queries", "changed_backend_abstentions", "native_operations_per_trace", "automatic_algebra_graph_candidates", "new_model_calls")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.root.resolve(), args.output.resolve()), sort_keys=True))
