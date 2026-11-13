"""Saved-only dependency trace from attention write through the image MLP.

No checkpoint loading, native execution or new independent attention forecast.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
from mdb_lanes.forecast_transaction import load_forecast_tree


def trace(source, study, output):
    here = Path(__file__).resolve().parent
    source_helper = runpy.run_path(str(here / "analyze_flux_context_transfer.py"))
    mlp = runpy.run_path(str(here / "flux_mlp_state_law.py"))
    read, sha = mlp["read"], mlp["sha"]
    assert read(source / "audit-v2.json")["mechanics_status"] == "verified"
    assert read(study / "audit-v1.json")["mechanics_status"] == "verified"
    original = source / "collection-01/extracted/binding-minimal-result"
    data = study / "collection-01/extracted/binding-minimal-result"
    rows = {r["id"]: r for r in read(original / "rows.json")}
    config = read(data / "plan.json")
    ix = config["selected_rows"]
    root = data / "transaction"
    decode = runpy.run_path(str(here / "analyze_flux_mlp_state_law.py"))["json_tree"]
    actual = decode(load_forecast_tree(root, read(root / "observations/native.json")["value"]))["fields"]
    predicted = decode(load_forecast_tree(root, read(root / "forecasts/stage_outputs.json")["predictions"]))["fields"]
    B, difference, stats = mlp["quantize"], mlp["difference"], mlp["statistics"]
    reader = source_helper["read_slot"]
    by = {}
    for name, old in (("base", "control"), ("changed", "restore-I")):
        w = rows[old]["writer"]
        R = reader(original / "store", w["entry"], "image")[:, ix]
        context = source_helper["load_cut"](original / "store", w["entry"]).payload["context"]
        mod = context["img_mod"].float().numpy().reshape(1, 1, -1)
        shiftA, scaleA, gateA, shiftM, scaleM, gateM = np.split(mod, 6, axis=-1)
        ports = {}
        for key, record in w["fields"].items():
            p = original / record["path"]
            assert sha(p) == record["sha256"]
            value = np.load(p, allow_pickle=False)
            ports[key] = value[:, ix] if value.shape[1] == 1024 else value
        observed_x = np.load(data / (name + ".npy"), allow_pickle=False)
        assert np.array_equal(observed_x, ports["mlp_input"])
        U = B(R + B(gateA * ports["attention_native"]))
        wide = U.astype(np.float64)
        centered = wide-wide.mean(-1, keepdims=True)
        normalized = B(centered / np.sqrt(np.mean(centered*centered, -1, keepdims=True) + 1e-6))
        x = B(B(normalized * B(1 + scaleM)) + shiftM)
        image_exit = reader(original / "store", w["exit"], "image")[:, ix]
        measured_composition = B(U + B(gateM * ports["mlp_native"]))
        predicted_mlp_composition = B(U + B(gateM * predicted[name]["output"]))
        by[name] = dict(native_writer_entry=w["entry"], native_writer_exit=w["exit"],
            normalized_input_reconstruction=difference(x, observed_x),
            CPU_MLP_vs_original_CUDA_MLP=difference(actual[name]["output"], ports["mlp_native"]),
            numpy_forecast_vs_original_CUDA_MLP=difference(predicted[name]["output"], ports["mlp_native"]),
            observed_port_residual_equation=difference(measured_composition, image_exit),
            conditional_predicted_MLP_residual_equation=difference(predicted_mlp_composition, image_exit))
    s0, s1 = [actual[n]["activated"].astype(np.float64) for n in ("base", "changed")]
    g0, g1 = [np.split(actual[n]["projected"], 2, axis=-1)[0].astype(np.float64) for n in ("base", "changed")]
    v0, v1 = [np.split(actual[n]["projected"], 2, axis=-1)[1].astype(np.float64) for n in ("base", "changed")]
    h0, h1 = [actual[n]["product"].astype(np.float64) for n in ("base", "changed")]
    components = dict(value=s0*(v1-v0), gate=(s1-s0)*v0, interaction=(s1-s0)*(v1-v0),
                      product_rounding=(h1-s1*v1)-(h0-s0*v0))
    # Smooth SwiGLU core diagnostic at the midpoint of observed projections.
    # This excludes BF16 rounding and avoids attributing quantized curvature
    # wholly to nonlinear activation. It is not a complete input-path forecast.
    smooth0, smooth1 = mlp["silu"](g0)*v0, mlp["silu"](g1)*v1
    smooth_mid = mlp["silu"]((g0+g1)/2)*((v0+v1)/2)
    core_curvature = smooth_mid-(smooth0+smooth1)/2
    current = {}
    base_fp = rows["control"]["writer"]["entry"]
    high_fp = rows["restore-I"]["writer"]["entry"]
    for field in ("image", "text", "conditioning"):
        low = reader(original / "store", base_fp, field)
        high = reader(original / "store", high_fp, field)
        delta = high.astype(np.float64)-low
        current[field] = dict(**difference(high, low), rows_changed=int(np.count_nonzero(np.any(delta != 0, axis=-1))))
    gate_observation = dict(projected_gate_scalars=int(g0.size), signs_changed=int(np.count_nonzero((g0 > 0) != (g1 > 0))),
        projected_gate_values_changed=int(np.count_nonzero(g0 != g1)),
        activated_gate_values_changed=int(np.count_nonzero(s0 != s1)),
        negative_activated_values_base=int(np.count_nonzero(s0 < 0)),
        semantics="SiLU gates are continuous and may be negative; not binary branches or named object concepts")
    result = dict(schema="mdb-flux-mlp-state-chain-v1", mechanics_status="verified saved-source joins",
        instrument_status="conditional arithmetic; exact native operands explicitly identified", terminal_status="unassessed",
        source_entry_contrast=current, selected_rows=ix, branches=by,
        hidden_product_components={k: stats(v) for k, v in components.items()},
        hidden_product_closure_max_abs=float(np.max(np.abs(sum(components.values())-(h1-h0)))),
        gate_observation=gate_observation,
        smooth_projected_core_curvature=dict(curvature=stats(core_curvature), endpoint_delta=stats(smooth1-smooth0),
            scope="Unrounded SiLU-product core at observed projected endpoints; no full-model or upstream prediction"),
        remaining_gap="Positioned Q/K/V and routing-versus-value decomposition of the changed attention return. Attention returns remain observed operands in this chain.",
        new_model_calls=0, source_files=[dict(path=str(p), sha256=sha(p)) for p in (
            source / "audit-v2.json", study / "audit-v1.json", original / "rows.json", data / "plan.json")],
        qualification_sources={p.name: sha(p) for p in (Path(__file__), here / "flux_mlp_state_law.py", here / "analyze_flux_context_transfer.py")})
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(branches=by, gate_observation=gate_observation,
                smooth_projected_core_curvature=result["smooth_projected_core_curvature"],
                hidden_product_components=result["hidden_product_components"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("source", "study", "output"):
        p.add_argument(n, type=Path)
    args = p.parse_args()
    print(json.dumps(trace(args.source.resolve(), args.study.resolve(), args.output.resolve()), sort_keys=True))
