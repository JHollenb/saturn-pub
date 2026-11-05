"""Independent saved-array/trace audit of the isolated MLP computation study."""
import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
from mdb.forward_ports import NativePortCatalog
from mdb.forecast_transaction import verify_transaction, load_forecast_tree


def read(path):
    return json.loads(Path(path).read_text())


def json_tree(value):
    if isinstance(value, Mapping):
        return {k: json_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_tree(v) for v in value]
    return value


def descriptor(value):
    a = np.asarray(value, np.float32)
    bits = a.view(np.uint32)
    if np.any(bits & np.uint32(65535)):
        raise ValueError("expected exact widened native BF16 bits")
    raw = (bits >> 16).astype("<u2").tobytes()
    return dict(shape=list(a.shape), dtype="torch.bfloat16", sha256=hashlib.sha256(raw).hexdigest())


def error(actual, predicted):
    a, p = np.asarray(actual, np.float64), np.asarray(predicted, np.float64)
    d = a-p
    rms = float(np.sqrt(np.mean(d*d)))
    return dict(exact=bool(np.array_equal(a, p)), changed_scalars=int(np.count_nonzero(d)),
                scalars=int(d.size), rms=rms, max_abs=float(np.max(np.abs(d))),
                relative_rms=rms/max(float(np.sqrt(np.mean(p*p))), 1e-300))


def analyze(collection, output):
    here = Path(__file__).resolve().parent
    custody = runpy.run_path(str(here / "flux_family_custody.py"))["verify"](collection)
    data = collection / "extracted/binding-minimal-result"
    summary, config = [read(data / n) for n in ("summary.json", "plan.json")]
    request = read(collection / "request.json")
    assert summary == read(collection / "report.json")["report"] and config == request["config"]
    assert summary["request_sha256"] == request["request_sha256"] == custody["request_sha256"]
    assert summary["counts"] == config["expected_counts"]
    txroot = data / "transaction"
    tx = verify_transaction(txroot, expected_seal_sha256=summary["seal"]["seal_sha256"])
    assert tx["verified"] and tx["phase"] == "closed"
    forecast = json_tree(load_forecast_tree(txroot, read(txroot / "forecasts/stage_outputs.json")["predictions"]))
    observed = json_tree(load_forecast_tree(txroot, read(txroot / "observations/native.json")["value"]))
    before = {**config["expected_counts"], "native_mlp_calls": 0, "native_linear_calls": 0, "native_swiglu_calls": 0}
    assert forecast["counts_at_seal"] == before == read(data / "seal-census.json")["counts"]
    assert forecast["hypotheses"] == config["hypotheses"]
    assert observed["counts"] == summary["counts"]
    inputs = {name: np.load(data / (name + ".npy"), allow_pickle=False) for name in ("base", "changed")}
    # Independent input construction, preserving FP64 blend -> FP32 -> BF16 RNE.
    middle = ((inputs["base"].astype(np.float64) + inputs["changed"])/2).astype(np.float32)
    bits = middle.view(np.uint32)
    middle = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    inputs["midpoint"] = middle
    inputs["permuted_midpoint"] = middle[:, ::-1].copy()
    inputs["single_row"] = inputs["base"].copy()
    inputs["single_row"][:, 0] = inputs["changed"][:, 0]
    manifest = read(data / "parameter-artifacts.json")
    recipes = {r["source"]["tensor"].removeprefix("transformer_blocks.1.ff."): r["source"] for r in manifest["artifacts"]}
    assert set(recipes) == {"linear_in.weight", "linear_out.weight"}
    joins, comparisons, operations = {}, {}, {}
    for name in config["cases"]:
        trace = read(data / "traces" / (name + ".json"))
        catalog = NativePortCatalog(trace)
        assert trace["execution"] == dict(backend="cpu", device="cpu", dtype="bfloat16", threads=4)
        assert trace["metadata"]["case_id"] == name
        binding = catalog.resolve("module:<root>:input", path=["args", 0])
        expected = descriptor(inputs[name])
        assert all(trace["values"][binding["value_id"]][k] == v for k, v in expected.items())
        native = observed["fields"][name]
        recorded_ports = read(data / "ports" / (name + ".json"))
        for field, path in (("projected", "linear_in"), ("product", "act_fn"), ("output", "linear_out")):
            port = catalog.resolve("module:"+path+":return")
            assert port == recorded_ports[field]
            d = descriptor(native[field])
            assert all(trace["values"][port["value_id"]][k] == v for k, v in d.items())
        silu = [e for e in trace["events"] if e["operator"] == "aten.silu.default"]
        assert len(silu) == 1 and len(silu[0]["outputs"]) == 1
        d = descriptor(native["activated"])
        assert all(trace["values"][silu[0]["outputs"][0]][k] == v for k, v in d.items())
        parameters = read(data / "parameters" / (name + ".json"))
        for parameter, recipe in recipes.items():
            port = catalog.resolve("parameter:"+parameter)
            assert port == parameters[parameter]
            value = trace["values"][port["value_id"]]
            assert value["sha256"] == recipe["payload_sha256"] and value["shape"] == recipe["shape"] and value["dtype"] == "torch.bfloat16"
        joins[name] = dict(input=True, module_returns=3, silu_operation=1, named_parameters=2)
        comparisons[name] = {k: error(v, forecast["fields"][name][k]) for k, v in native.items()}
        operations[name] = [dict(pc=e["sequence"], operator=e["operator"], module=e["module_path"],
                                 inputs=e["inputs"], outputs=e["outputs"]) for e in trace["events"]]
    actual = observed["fields"]
    locality = dict(permutation=error(actual["permuted_midpoint"]["output"], actual["midpoint"]["output"][:, ::-1]),
        untouched_rows=error(actual["single_row"]["output"][:, 1:], actual["base"]["output"][:, 1:]),
        selected_row=error(actual["single_row"]["output"][:, :1], actual["changed"]["output"][:, :1]))
    # Exact supplied-product identity, independent of any fitted model or matrix.
    s0, s1 = [actual[n]["activated"].astype(np.float64) for n in ("base", "changed")]
    v0, v1 = [np.split(actual[n]["projected"], 2, axis=-1)[1].astype(np.float64) for n in ("base", "changed")]
    h0, h1 = [actual[n]["product"].astype(np.float64) for n in ("base", "changed")]
    components = [s0*(v1-v0), (s1-s0)*v0, (s1-s0)*(v1-v0), (h1-s1*v1)-(h0-s0*v0)]
    closure = float(np.max(np.abs(sum(components)-(h1-h0))))
    recorded = read(data / "metrics.json")
    for name, fields in comparisons.items():
        for key, value in fields.items():
            assert all(recorded["stage_errors"][name][key][k] == v for k, v in value.items())
    for key, value in locality.items():
        assert all(recorded[key][k] == v for k, v in value.items())
    assert closure == recorded["product_difference_closure_max_abs"]
    curve = actual["midpoint"]["output"].astype(np.float64)-(actual["base"]["output"].astype(np.float64)+actual["changed"]["output"])/2
    pcurve = forecast["fields"]["midpoint"]["output"].astype(np.float64)-(forecast["fields"]["base"]["output"].astype(np.float64)+forecast["fields"]["changed"]["output"])/2
    result = dict(schema="mdb-flux-mlp-state-law-audit-v1", mechanics_status="verified", instrument_status="applicable CPU BF16 native-port trace",
        trend_status="per-stage numerical and structural observations", terminal_status="unassessed",
        job_id=summary["job_id"], request_sha256=request["request_sha256"], counts=summary["counts"], custody=custody,
        transaction=tx, native_port_joins=joins, comparisons=comparisons, locality=locality,
        exact_product_difference_closure_max_abs=closure,
        midpoint_curvature=dict(actual_rms=float(np.sqrt(np.mean(curve*curve))),
            predicted_rms=float(np.sqrt(np.mean(pcurve*pcurve))), residual=error(curve, pcurve)),
        projected_component_metrics=recorded["projected_components"],
        projected_component_scope="Producer-calculated checkpoint Wout projection; named weight identity independently joined, projection summary not independently recomputed by this saved-only audit.",
        operation_sequences=operations, worker_seconds=summary["worker_seconds"], new_model_calls_in_audit=0,
        scope="32selected normalized image-MLP input rows. Base/changed calibration endpoints previously observed on CUDA; midpoint/permutation/single-row native CPU targets newly executed after seal. No upstream attention, full model or image forecast.")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return dict(mechanics_status=result["mechanics_status"], counts=result["counts"],
                outputs={n: v["output"] for n, v in comparisons.items()}, locality=locality,
                curvature=result["midpoint_curvature"], components=recorded["projected_components"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("collection", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    print(json.dumps(analyze(args.collection.resolve(), args.output.resolve()), sort_keys=True))
