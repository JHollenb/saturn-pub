"""Saved-only custody, equations, native-future and fixed-parent cost audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def measure(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = a - b
    energy = float(np.sum(b * b))
    return {"mae": float(np.mean(np.abs(d))), "l2": float(np.sqrt(np.sum(d * d))),
        "rms": float(np.sqrt(np.mean(d * d))), "max_abs": float(np.max(np.abs(d))),
        "relative_rms": float(np.sqrt(np.sum(d * d) / energy)) if energy else None,
        "exact_values": int(np.count_nonzero(d == 0)), "values": int(d.size)}


def analyze(run, destination):
    from mdb.forecast_transaction import verify_transaction
    from mdb.tensor_archive import load_tensor_archive

    here = Path(__file__).resolve().parent
    NumericalTree = runpy.run_path(str(here / "summarize_context_window.py"))["NumericalTree"]
    data = run / "extracted/predicted-future-result"
    custody = runpy.run_path(str(here / "flux_family_custody.py"))["verify"](run)
    summary, request, binding, forecasts, rows, pool = (read(data / name) for name in
        ("summary.json", "request.json", "binding.json", "forecasts.json", "rows.json", "dose-pool.json"))
    source = read(data / "source.json")
    assert summary == read(run / "report.json")["report"]
    assert request == read(run / "request.json") and request["config"] == read(data / "plan.json")
    assert summary["request_sha256"] == request["request_sha256"] == custody["request_sha256"]
    assert hashlib.sha256(canonical(source["checkpoint_files"])).hexdigest() == request["config"]["checkpoint_sha256"]
    for name, digest in request["source_hashes"].items():
        assert sha(run / "payload" / name) == digest
    transaction = verify_transaction(data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    assert transaction["verified"] and transaction["phase"] == "closed"
    tree = NumericalTree(data / "transaction")
    declaration = tree.decode(read(data / "transaction/forecasts/fields.json")["predictions"])
    assert declaration["fields"] == forecasts["fields"]
    assert declaration["semantic_predictions"] == request["config"]["semantic_predictions"]
    assert binding["counts_before_forecast"]["source_blocks"] == binding["counts_before_forecast"]["images"] == 0
    typed = runpy.run_path(str(here / "analyze_flux_mlp_forecast.py"))
    routes = runpy.run_path(str(here / "analyze_flux_concept_routes.py"))
    store = typed["audit_store"](data / "store")
    def slot(cut, name):
        return routes["slot_array"](data / "store", cut, name, typed)
    def fields(cut):
        return {item["key"]["value"]: item["value"] for item in store["cuts"][cut]["payload"]["items"]}
    def arr(record):
        return load_tensor_archive(data, record)
    all_receipts = read(data / "receipts.json")
    public = runpy.run_path(str(here / "summarize_context_window.py"))["public_receipt"]
    by_receipt = {}
    for receipt in all_receipts:
        public(receipt)
        by_receipt[receipt["fingerprint"]] = receipt
    assert len(all_receipts) == summary["counts"]["receipts"] == 528
    assert sum(r["operation"] == "continue" for r in all_receipts) == 523
    assert sum(r["operation"] == "apply" for r in all_receipts) == 5
    native_errors = read(data / "field-errors.json")
    full_field_errors, predictions = {}, {}
    for role in ("base", "color", "shape", "joint"):
        inp = binding["inputs"][role]
        assert np.array_equal(arr(inp["image"]), slot(inp["parent"], "image"))
        prediction, native = arr(forecasts["fields"][role]), arr(native_errors[role]["native_field"])
        full_field_errors[role] = measure(prediction, native)
        for key in ("values", "exact_values", "max_abs", "l2", "rms", "relative_rms"):
            assert np.isclose(full_field_errors[role][key], native_errors[role][key], rtol=1e-12, atol=1e-12)
        predictions[role] = prediction
    # Recompute a predeclared sparse arithmetic witness from actual entry and
    # weights with an independent equation implementation. No current outputs.
    entry = runpy.run_path(str(here / "analyze_flux_single_mlp.py"))
    weights = arr(binding["weight"])
    bias = None if binding["bias"] is None else arr(binding["bias"])
    neurons = list(range(0, 9216, 288))
    weight_ids = neurons + [9216 + j for j in neurons]
    sparse = {}
    for role, inp in binding["inputs"].items():
        rebuilt = entry["entry_equations"](arr(inp["image"]), arr(inp["mod"]), weights[weight_ids],
            None if bias is None else bias[weight_ids], eps=binding["eps"],
            queries=[0,31,32,255,511,512,767,1023], neurons=neurons, hidden_width=3072,
            mlp_width=9216, quantize=typed["quantize_bf16"])["bf16_reference"]["activation"]
        sampled = predictions[role][:, [0,31,32,255,511,512,767,1023]][:, :, neurons]
        sparse[role] = measure(sampled, rebuilt)
    additive = typed["quantize_bf16"]((predictions["color"].astype(np.float64) + predictions["shape"] - predictions["base"]).astype(np.float32))
    assert np.array_equal(additive, arr(forecasts["fields"]["additive"]))
    images, latents = {}, {}
    for row in rows:
        assert sha(data / row["png"]) == row["png_sha256"]
        with Image.open(data / row["png"]) as image:
            assert image.mode == "RGB" and image.size == (512,512)
            images[row["id"]] = np.asarray(image).copy()
        latents[row["id"]] = arr(row["final_latent"])
        assert np.array_equal(latents[row["id"]], slot(row["final_cut"], "latent"))
        assert len(row["path"]) == 15
        previous = row["act"]["post_act"] if row["act"] else row["path"][0]["parent"]
        for receipt in row["path"]:
            assert receipt == by_receipt[receipt["fingerprint"]] and receipt["parent"] == previous
            previous = receipt["result"]
        assert previous == row["final_cut"]
        if row["act"]:
            a = row["act"]
            assert a["receipt"] == by_receipt[a["receipt_id"]] and a["act"]["writes"] == ["image"]
            before, after = fields(a["parent"]), fields(a["post_act"])
            assert before.keys() == after.keys()
            assert all(before[k] == after[k] for k in before if k != "image")
    assert np.array_equal(images["sham"], images["base"])
    assert np.array_equal(latents["sham"], latents["base"])
    checks = read(data / "consumer-checks.json")
    assert len(checks) == len(pool["rows"]) == 17
    assert all(row["image_exact"] and row["text_exact"] for row in checks.values())
    pairs = [("transfer.predicted", "transfer.measured"), ("transfer.measured", "base"),
        ("transfer.additive", "transfer.predicted"), ("transfer.roll", "transfer.predicted"),
        ("color", "base"), ("shape", "base"), ("joint", "base")]
    comparisons = {a + "__" + b: {"rgb": measure(images[a], images[b]), "latent": measure(latents[a], latents[b])} for a,b in pairs}
    full, fast = [sum(pool["times"][key]) for key in ("full_block", "cached_projection")]
    predictor = sum(forecasts["prediction_seconds"].values())
    report = {"schema": "mdb-predicted-future-audit-v1", "job_id": summary["job_id"],
        "request_sha256": request["request_sha256"], "mechanics_status": "verified", "instrument_status": "applicable",
        "trend_status": "retain-signed-and-visual-results", "terminal_status": "unassessed",
        "custody": custody, "transaction": transaction, "counts": summary["counts"], "store": store["counts"],
        "full_field_errors": full_field_errors, "independent_sparse_equations": sparse,
        "comparisons": comparisons, "cached_consumer_exact_endpoints": 17,
        "cost": {"prediction_seconds_all4": predictor, "prediction_seconds_joint": forecasts["prediction_seconds"]["joint"],
            "full_instrumented_seconds_17": full, "cached_checked_seconds_17": fast,
            "instrumented_endpoint_ratio": full / fast, "all4_prediction_plus_cached_seconds": predictor + fast,
            "joint_prediction_plus_cached_seconds": forecasts["prediction_seconds"]["joint"] + fast,
            "scope": "All17 reference blocks are executed validation cost; no physical blocks avoided in that validation panel. Native baseline uses capture/finite/footprint checks. Cost ratio is not a production kernel or full-generation speedup."},
        "worker_seconds": summary["worker_seconds"],
        "coverage": "All sealed field/source/native hashes and full errors verified; independent entry arithmetic covers8queries32neurons per role, not full-field recomputation. Native consumer equality is source-bound direct execution, not independent CUDA-kernel tracing. Visual semantics reviewed separately."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    sheet = Image.new("RGB", (3*512,3*542), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x, y = (index%3)*512, (index//3)*542
        draw.text((x+8,y+8), row["id"], fill="black")
        sheet.paste(Image.fromarray(images[row["id"]]), (x,y+30))
    sheet.save(destination.parent / "native-futures.png")
    return {"audit": str(destination), "counts": summary["counts"], "cost": report["cost"], "comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run.resolve(), args.output.resolve()), sort_keys=True))
