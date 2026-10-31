"""Audit a native R/A/M factorial from saved ports, cuts and actual RGB futures."""
from __future__ import annotations

import argparse
import itertools
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def analyze(root):
    here = Path(__file__).resolve().parent
    h = runpy.run_path(str(here / "summarize_context_window.py"))
    typed = runpy.run_path(str(here / "analyze_flux_mlp_forecast.py"))
    routes = runpy.run_path(str(here / "analyze_flux_concept_routes.py"))
    require, read, sha = (h[key] for key in ("require", "read", "sha"))
    run = h["source_run"](root, "flux-merge-result", max_archive_bytes=256 << 20, max_member_bytes=128 << 20)
    data, plan, tree = run["data"], run["plan"], run["tree"]
    source, summary = (read(data / name) for name in ("source.json", "summary.json"))
    observed = tree.decode(read(run["transaction"] / "observations/native.json")["value"])
    hypotheses = tree.decode(read(run["transaction"] / "forecasts/merge_hypotheses.json")["predictions"])
    require(hypotheses == plan["hypotheses"] and plan == run["request"]["config"], "sealed plan/hypothesis mismatch")
    require(observed["source_sha256"] == sha((data / "source.json").read_bytes()) and observed["counts"] == summary["counts"], "source/work mismatch")
    store = typed["audit_store"](data / "typed-store")
    require(source["input_cuts"] == plan["input_cuts"] and source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"], "source input/checkpoint mismatch")
    slot = lambda cut, name: routes["slot_array"](data / "typed-store", cut, name, typed)

    def fields(cut):
        return {item["key"]["value"]: item["value"] for item in store["cuts"][cut]["payload"]["items"]}

    def arrays(record):
        path = data / record["path"]
        require(sha(path.read_bytes()) == record["sha256"], "port archive digest mismatch")
        with np.load(path, allow_pickle=False) as loaded:
            result = {key: loaded[key] for key in loaded.files}
        require(set(result) == set(record["descriptors"]), "port inventory mismatch")
        for name, value in result.items():
            require(typed["matches"](value, record["descriptors"][name], bf16=True), "native BF16 port descriptor mismatch")
        return result

    ports = {name: arrays(record) for name, record in source["ports"].items()}
    text_rows, attention_width = source["text_rows"], source["attention_width"]
    gates_equal = np.array_equal(ports["base"]["gate"], ports["changed"]["gate"])
    require(gates_equal, "unmatched time gate")
    native_merge_checks = {}
    for name in ("base", "changed"):
        before, after = source["input_cuts"][name], source["post_cuts"][name]
        require(np.array_equal(ports[name]["residual"], np.concatenate([slot(before, "text"), slot(before, "image")], axis=1)), "native residual port differs from source cut")
        merged = typed["quantize_bf16"](ports[name]["residual"] + typed["quantize_bf16"](ports[name]["gate"] * ports[name]["projection"]))
        actual = np.concatenate([slot(after, "text"), slot(after, "image")], axis=1)
        native_merge_checks[name] = bool(np.array_equal(merged, actual))
        h["public_receipt"](source["prefix_receipts"][name])
    rows = {row["id"]: row for row in observed["rows"]}
    masks = ["".join(bits) for bits in itertools.product("01", repeat=3)]
    require(set(rows) == {"base", "changed", "copy.sham", *("merge." + mask for mask in masks)}
            and len(rows) == plan["max_images"] == summary["counts"]["images"], "row inventory mismatch")
    pixels, latents, diagnostics, recipe_checks = {}, {}, {}, {}
    for name, row in rows.items():
        require(read(data / "rows" / (name + ".json")) == row, "row observation mismatch")
        image_path = data / row["image"]["path"]
        require(sha(image_path.read_bytes()) == row["image"]["sha256"], "PNG digest mismatch")
        pixels[name] = np.asarray(Image.open(image_path).convert("RGB"))
        require(pixels[name].shape == (512, 512, 3), "RGB geometry mismatch")
        diagnostics[name] = typed["image_diagnostics"](pixels[name])
        latents[name] = {}
        for saved in row["schedulers"]:
            path = data / saved["path"]
            value = np.load(path, allow_pickle=False)
            require(sha(path.read_bytes()) == saved["sha256"] and typed["matches"](value, saved["descriptor"], bf16=True), "scheduler descriptor mismatch")
            latents[name][saved["step"]] = value
        require(set(latents[name]) == {1, 2, 3, 4}, "incomplete continuation")
        if row["kind"] == "reference":
            require(row["parent"] == source["post_cuts"][name], "reference parent mismatch")
            continue
        require(row["parent"] == source["post_cuts"]["changed"], "intervention parent mismatch")
        before_fields, post_fields = fields(row["parent"]), fields(row["post_act"])
        require(all(post_fields[key] == value for key, value in before_fields.items() if key != "image"), "undeclared slot changed")
        h["public_receipt"](row["receipt"])
        receipt = read(data / "typed-store/receipts" / (row["receipt"]["fingerprint"] + ".json"))
        require(receipt == row["receipt"] and row["act"]["writes"] == ["image"], "Act receipt mismatch")
        actual_image = slot(row["post_act"], "image")
        if row["kind"] == "copy":
            require(np.array_equal(actual_image, slot(row["parent"], "image")), "copy sham changed image")
            continue
        mask = row["mask_R_A_M"]
        residual = ports["changed"]["residual"].copy()
        operands = ports["changed"]["operands"].copy()
        if mask[0] == "1":
            residual[:, text_rows:] = ports["base"]["residual"][:, text_rows:]
        if mask[1] == "1":
            operands[:, text_rows:, :attention_width] = ports["base"]["operands"][:, text_rows:, :attention_width]
        if mask[2] == "1":
            operands[:, text_rows:, attention_width:] = ports["base"]["operands"][:, text_rows:, attention_width:]
        require(typed["matches"](operands, row["operands_descriptor"], bf16=True), "operand recipe mismatch")
        projected = arrays(row["projection"])["value"]
        result = typed["quantize_bf16"](residual + typed["quantize_bf16"](ports["changed"]["gate"] * projected))
        require(np.array_equal(result[:, text_rows:], actual_image), "BF16 merge residual equation differs from installed image")
        recipe_checks[name] = {"operands_and_residual_equation_verified": True,
            "text_rows_exact_changed": bool(np.array_equal(result[:, :text_rows], slot(source["post_cuts"]["changed"], "text")))}
    receipts = {}
    for path in (data / "typed-store/receipts").glob("*.json"):
        receipt = read(path)
        h["public_receipt"](receipt)
        require(path.stem == receipt["fingerprint"] and receipt["model_identity"] == source["model_identity"], "receipt identity mismatch")
        receipts[path.stem] = receipt
    emissions = [json.loads(line) for line in (data / "receipt-emissions.jsonl").read_text().splitlines()]
    require([row["sequence"] for row in emissions] == list(range(len(emissions))) and len(emissions) == summary["counts"]["receipt_emissions"], "receipt emission sequence mismatch")
    emitted = [receipts[row["receipt"]] for row in emissions]
    require(sum(row.get("steps", 0) for row in emitted if row["operation"] == "continue") == summary["counts"]["adapter_steps"]
            and sum(row["operation"] != "continue" for row in emitted) == summary["counts"]["acts"], "emitted native work census mismatch")
    compare = routes["comparison"]
    comparisons = {name: {"rgb": compare(pixels[name], pixels["base"], pixels["changed"]),
        "scheduler_trajectory": {str(step): compare(value, latents["base"][step], latents["changed"][step]) for step, value in latents[name].items()}} for name in rows}
    return {"schema": "mdb-flux-merge-audit-v1", "source": run["source"], "counts": summary["counts"],
        "selected_block": plan["block"], "model_identity": source["model_identity"], "mechanics_status": "valid", "terminal_status": "unassessed",
        "native_merge_equation_checks": native_merge_checks, "recipe_checks": recipe_checks,
        "typed_store": store["counts"], "unique_receipts": len(receipts), "receipt_emissions": len(emissions),
        "comparisons": comparisons, "image_diagnostics": diagnostics, "sealed_hypotheses": hypotheses,
        "scope": plan["scope"], "auditor_source_sha256": sha(Path(__file__).read_bytes()),
        "audit_helpers": {name: sha((here / name).read_bytes()) for name in ("summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py")},
        "instrument_limitations": "Projection outputs are retained actual shared-kernel calculations, not recomputed offline from the external weight matrix. Offline audit verifies native port hashes, operand selection, BF16 residual equations, typed state closure and full numerical/RGB futures. No independent mathematical forecast of edited images."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "native_merge_equations": result["native_merge_equation_checks"], "counts": result["counts"]}))
