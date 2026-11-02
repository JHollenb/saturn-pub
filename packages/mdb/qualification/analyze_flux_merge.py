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
    run = h["source_run"](root, "flux-merge-result", max_archive_bytes=512 << 20, max_member_bytes=128 << 20)
    data, plan, tree = run["data"], run["plan"], run["tree"]
    source, summary = (read(data / name) for name in ("source.json", "summary.json"))
    observed = tree.decode(read(run["transaction"] / "observations/native.json")["value"])
    hypotheses = tree.decode(read(run["transaction"] / "forecasts/merge_hypotheses.json")["predictions"])
    require(hypotheses == plan["hypotheses"] and plan == run["request"]["config"], "sealed plan/hypothesis mismatch")
    require(observed["source_sha256"] == sha((data / "source.json").read_bytes()) and observed["counts"] == summary["counts"], "source/work mismatch")
    store = typed["audit_store"](data / "typed-store")
    require(source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"], "source checkpoint mismatch")
    if not plan.get("fresh_context"):
        require(source["input_cuts"] == plan["input_cuts"], "source input mismatch")
    else:
        initial = source["initial"]
        require(initial["prompts"] == plan["prompts"] and initial["seed"] == plan["seed"], "fresh inputs differ")
        require(h["sha"]((root / "payload/campaign-plan.json").read_bytes()) == plan["campaign_sha256"], "frozen campaign differs")
    slot = lambda cut, name: routes["slot_array"](data / "typed-store", cut, name, typed)

    def fields(cut):
        return {item["key"]["value"]: item["value"] for item in store["cuts"][cut]["payload"]["items"]}

    if plan.get("fresh_context"):
        initial = source["initial"]
        before_fields, after_fields = fields(initial["base_projected"]), fields(initial["changed_projected"])
        require(all(after_fields[key] == value for key, value in before_fields.items() if key != "text"), "initial text contrast changed undeclared closure")
        require(initial["text_act"]["writes"] == ["text"], "initial contrast not text-only")
        h["public_receipt"](initial["text_receipt"])
    for name, cut in source["input_cuts"].items():
        point = typed["unpack"](store["cuts"][cut]["execution_point"])
        require(all(point[key] == value for key, value in plan["selector"].items()), "wrong selected source boundary")
    left, right = (fields(source["input_cuts"][name]) for name in ("base", "changed"))
    require(all(right[key] == value for key, value in left.items() if key not in {"text", "image"}), "unmatched source closure")

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
        require(native_merge_checks[name], "native BF16 merge equation does not close")
        h["public_receipt"](source["prefix_receipts"][name])
    rows = {row["id"]: row for row in observed["rows"]}
    masks = ["".join(bits) for bits in itertools.product("01", repeat=3)]
    attention_masks = [mask for mask in masks if mask not in {"000", "111"}] if plan.get("attention_sources") else []
    require(set(rows) == {"base", "changed", "copy.sham", *("merge." + mask for mask in masks), *("attention." + mask for mask in attention_masks)}
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
        residual = ports["changed"]["residual"].copy()
        operands = ports["changed"]["operands"].copy()
        if row["kind"] == "attention":
            selected = arrays(row["positioned_qkv"])
            qkv = {name: arrays(source["ports"][name]["positioned_qkv"]) for name in ("base", "changed")}
            mask = row["mask_Q_TKV_IKV"]
            selected_ports = row["positioned_qkv"]["selected_ports"]
            require(typed["matches"](qkv["base" if mask[0] == "1" else "changed"]["q"], selected_ports["q"], bf16=True), "wrong query source")
            for key in ("k", "v"):
                expected = qkv["changed"][key].copy()
                if mask[1] == "1":
                    expected[:, :text_rows] = qkv["base"][key][:, :text_rows]
                if mask[2] == "1":
                    expected[:, text_rows:] = qkv["base"][key][:, text_rows:]
                require(typed["matches"](expected, selected_ports[key], bf16=True), "wrong K/V source selection")
            residual[:, text_rows:] = ports["base"]["residual"][:, text_rows:]
            operands[:, text_rows:, :attention_width] = selected["attention"][:, text_rows:]
        else:
            mask = row["mask_R_A_M"]
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
    attention_joins = {}
    if plan.get("attention_sources"):
        attention_joins = read(data / "attention-joins.json")
        require(set(attention_joins) == {"000", "111"}, "incomplete attention self joins")
        for mask, join in attention_joins.items():
            selected = arrays(join["ports"])
            source_name = "changed" if mask == "000" else "base"
            require(np.array_equal(selected["attention"], ports[source_name]["operands"][:, :, :attention_width]), "native attention self reconstruction differs")
            target_row = rows[join["equivalent_row"]]
            require(typed["matches"](slot(target_row["post_act"], "image"), join["image_descriptor"], bf16=True), "attention corner does not join selected merge image")
            require(join["projection_descriptor"]["sha256"] == target_row["projection"]["descriptors"]["value"]["sha256"], "attention corner projection join differs")
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
        "native_merge_equation_checks": native_merge_checks, "recipe_checks": recipe_checks, "attention_joins": attention_joins,
        "typed_store": store["counts"], "unique_receipts": len(receipts), "receipt_emissions": len(emissions),
        "comparisons": comparisons, "image_diagnostics": diagnostics, "sealed_hypotheses": hypotheses,
        "scope": plan["scope"], "auditor_source_sha256": sha(Path(__file__).read_bytes()),
        "audit_helpers": {name: sha((here / name).read_bytes()) for name in ("summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py")},
        "instrument_limitations": "Projection and attention outputs are retained shared-kernel calculations, not recomputed offline from external weights or SDPA. Offline audit verifies port hashes, Q/K/V source selection, exact native attention self joins, operand selection, BF16 residual equations, typed closure and full numerical/RGB futures. No independent mathematical forecast of edited images."}


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
