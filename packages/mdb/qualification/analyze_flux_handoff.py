"""Independently audit typed restoration closures and full retained futures."""
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def analyze(root):
    here = Path(__file__).resolve().parent
    h = runpy.run_path(str(here / "summarize_context_window.py"))
    typed = runpy.run_path(str(here / "analyze_flux_mlp_forecast.py"))
    compare = runpy.run_path(str(here / "analyze_flux_concept_routes.py"))["comparison"]
    require, read, sha = (h[key] for key in ("require", "read", "sha"))
    run = h["source_run"](root, "flux-handoff-result", max_archive_bytes=256 << 20, max_member_bytes=128 << 20)
    data, tree, plan = run["data"], run["tree"], run["plan"]
    source, summary, frontiers = (read(data / name) for name in ("source.json", "summary.json", "frontiers.json"))
    require(plan == run["request"]["config"] and source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"], "plan/checkpoint mismatch")
    observed = tree.decode(read(run["transaction"] / "observations/native.json")["value"])
    hypotheses = tree.decode(read(run["transaction"] / "forecasts/handoff_hypotheses.json")["predictions"])
    require(hypotheses == plan["hypotheses"] and observed["source_sha256"] == sha((data / "source.json").read_bytes()), "sealed hypothesis/source mismatch")
    require(observed["counts"] == summary["counts"] and summary["counts"]["prompt_encodings"] == 0, "work count mismatch")
    store = typed["audit_store"](data / "typed-store")
    require(source["input_cuts"] == plan["input_cuts"], "source input mismatch")

    def fields(cut_id):
        payload = store["cuts"][cut_id]["payload"]
        return {item["key"]["value"]: item["value"] for item in payload["items"]}

    initial, donor, changed = (
        fields(source["base"]), fields(source["input_cuts"]["relation"]), fields(source["changed"]))
    require(changed == {**initial, "text": donor["text"]}, "initial changed branch is not text-only")
    require(initial["text"] != donor["text"], "source text contrast empty")
    require(all(initial[slot] == donor[slot] for slot in initial if slot not in {"text", "conditioning"}), "input nuisance contrast")
    for boundary in plan["boundaries"]:
        for ref in ("base", "changed"):
            cut_id = frontiers[ref][boundary["id"]]
            point = store["cuts"][cut_id]["execution_point"]
            # Execution points in the StateCut manifest have typed descriptors.
            raw = read(data / "typed-store/cuts" / (cut_id + ".json"))["execution_point"]
            require(all(raw[key] == value for key, value in boundary["selector"].items()), "breakpoint selector mismatch")
            require(point is not None and fields(cut_id)["conditioning"] == initial["conditioning"], "stored conditioning changed")
    rows = {row["id"]: row for row in observed["rows"]}
    expected = {"base", "changed", "late.sham", "late.restore_both"} | {
        b["id"] + ".restore_" + slot for b in plan["boundaries"] for slot in ("text", "image")}
    require(set(rows) == expected and len(rows) == plan["max_images"] == summary["counts"]["images"], "row inventory mismatch")
    arrays, pixels, diagnostics = {}, {}, {}
    act_count = 1
    for name, row in rows.items():
        require(read(data / "rows" / (name + ".json")) == row, "row observation mismatch")
        require(sha((data / row["image"]["path"]).read_bytes()) == row["image"]["sha256"], "PNG digest mismatch")
        pixels[name] = np.asarray(Image.open(data / row["image"]["path"]).convert("RGB"))
        require(pixels[name].shape == (512, 512, 3), "RGB geometry mismatch")
        diagnostics[name] = typed["image_diagnostics"](pixels[name])
        arrays[name] = {}
        for saved in row["schedulers"]:
            path = data / saved["path"]
            values = np.load(path, allow_pickle=False)
            require(sha(path.read_bytes()) == saved["sha256"] and typed["matches"](values, saved["descriptor"], bf16=True), "scheduler array descriptor mismatch")
            arrays[name][saved["step"]] = values
        require(set(arrays[name]) == {1, 2, 3, 4}, "incomplete scheduler future")
        if row["kind"] == "reference":
            require(row["parent"] == source[name], "reference parent mismatch")
            continue
        boundary = row["boundary"]
        require(row["parent"] == frontiers["changed"][boundary] and row["donor"] == frontiers["base"][boundary], "wrong restoration lineage")
        before, baseline, post = (fields(row[key]) for key in ("parent", "donor", "post_act"))
        expected_slots = dict(before)
        for slot in row["slots"]:
            expected_slots[slot] = before[slot] if row["kind"] == "sham" else baseline[slot]
        require(post == expected_slots and row["undeclared_slots_exact"], "restoration changed undeclared state")
        require(len(row["acts"]) == len(row["slots"]), "Act count mismatch")
        for slot, action in zip(row["slots"], row["acts"], strict=True):
            receipt = read(data / "typed-store/receipts" / (action["receipt_id"] + ".json"))
            h["public_receipt"](receipt)
            require(receipt == action["receipt"] and action["act"]["writes"] == [slot], "Act/receipt mismatch")
        act_count += len(row["acts"])
    # Verify every retained native receipt, not only intervention receipts.
    operations = 0
    for path in (data / "typed-store/receipts").glob("*.json"):
        receipt = read(path)
        h["public_receipt"](receipt)
        require(receipt["fingerprint"] == path.stem and receipt["model_identity"] == source["model_identity"], "receipt identity mismatch")
        if receipt["operation"] == "continue":
            operations += receipt["steps"]
    require(operations == summary["counts"]["adapter_steps"] and act_count == summary["counts"]["acts"], "receipt/work census mismatch")
    comparisons = {name: {"rgb": compare(pixels[name], pixels["base"], pixels["changed"]),
        "scheduler_trajectory": {str(step): compare(value, arrays["base"][step], arrays["changed"][step])
            for step, value in arrays[name].items()}} for name in rows}
    controls = {"sham_changed_exact": comparisons["late.sham"]["rgb"]["exact_target"] and all(
        row["exact_target"] for row in comparisons["late.sham"]["scheduler_trajectory"].values()),
        "restore_both_base_exact": comparisons["late.restore_both"]["rgb"]["exact_base"] and all(
        row["exact_base"] for row in comparisons["late.restore_both"]["scheduler_trajectory"].values())}
    helpers = ("summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py")
    return {"schema": "mdb-flux-handoff-audit-v1", "source": run["source"], "counts": summary["counts"],
        "mechanics_status": "valid", "terminal_status": "unassessed", "typed_store": {key: value for key, value in store.items() if key != "cuts"},
        "sealed_hypotheses": hypotheses, "control_observations": controls,
        "comparisons": comparisons, "image_diagnostics": diagnostics, "frontiers": frontiers,
        "auditor_source_sha256": sha(Path(__file__).read_bytes()),
        "audit_helpers": {name: sha((here / name).read_bytes()) for name in helpers},
        "scope": plan["scope"], "equations_or_models_executed": False,
        "interpretation_limits": "Whole-field interventions, known seed31 contrast; no minimal writer, object masks, numerical image prediction or general semantic rule claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "controls": result["control_observations"], "counts": result["counts"]}))
