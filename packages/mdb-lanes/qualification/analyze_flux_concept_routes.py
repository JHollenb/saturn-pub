"""Audit retained carrier swaps and describe their effects without model execution."""
from __future__ import annotations

import argparse
import itertools
import json
import runpy
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def require(value, message):
    if not value:
        raise ValueError(message)


def comparison(actual, base, target):
    """Signed target projection is descriptive, not an object/semantic score."""
    actual, base, target = (np.asarray(value, dtype=np.float64) for value in (actual, base, target))
    require(actual.shape == base.shape == target.shape and all(np.isfinite(x).all() for x in (actual, base, target)),
            "incompatible or nonfinite comparison")
    delta, direction = actual - base, target - base
    energy = float(np.square(direction).sum())
    return {"exact_base": bool(np.array_equal(actual, base)), "exact_target": bool(np.array_equal(actual, target)),
        "base_mae": float(np.abs(delta).mean()), "target_mae": float(np.abs(actual - target).mean()),
        "base_l2": float(np.linalg.norm(delta.ravel())), "target_l2": float(np.linalg.norm((actual - target).ravel())),
        "signed_mean": float(delta.mean()), "target_projection": float(np.sum(delta * direction) / energy) if energy else None,
        "scope": "full numerical arrays; target projection is not a semantic correctness score"}


def slot_array(store, cut_id, slot, typed):
    """Read one verified typed CAS slot without Torch or pickle."""
    cut = json.loads((store / "cuts" / (cut_id + ".json")).read_text())
    page = json.loads((store / "pages" / (cut["slots"][slot] + ".json")).read_text())
    encoded, descriptor = page["encoded"], page["value"]
    require(encoded["kind"] == "tensor", "expected typed tensor slot")
    path = store / "blobs" / (page["tensor_blob"] + ".safetensors")
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        item = header[encoded["name"]]
        require(item["dtype"] in {"BF16", "F32", "I64"}, "unsupported typed slot dtype")
        start, stop = item["data_offsets"]
        stream.seek(8 + size + start)
        raw = stream.read(stop - start)
    if item["dtype"] == "BF16":
        values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        values = np.frombuffer(raw, dtype={"F32": "<f4", "I64": "<i8"}[item["dtype"]])
    values = values.reshape(item["shape"])
    require(typed["matches"](values, descriptor, bf16=item["dtype"] == "BF16"), "slot raw descriptor differs")
    return values


def analyze(root):
    directory = Path(__file__).parent
    h = runpy.run_path(str(directory / "summarize_context_window.py"))
    typed = runpy.run_path(str(directory / "analyze_flux_mlp_forecast.py"))
    run = h["source_run"](root, "flux-concept-result", max_archive_bytes=512 << 20, max_member_bytes=256 << 20)
    data, plan, tree = run["data"], run["plan"], run["tree"]
    source = json.loads((data / "source.json").read_text())
    summary = json.loads((data / "summary.json").read_text())
    require(plan == run["request"]["config"] and source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"], "plan/checkpoint differs")
    observed = tree.decode(h["read"](run["transaction"] / "observations/native.json")["value"])
    forecast = tree.decode(h["read"](run["transaction"] / "forecasts/route_hypotheses.json")["predictions"])
    require(forecast == plan["hypotheses"], "sealed hypotheses differ")
    require(observed["source_sha256"] == h["sha"]((data / "source.json").read_bytes()), "observed source differs")
    store = typed["audit_store"](data / "typed-store")
    rows = {row["id"]: row for row in observed["rows"]}
    expected = {row["id"] + ".native" for row in plan["prompts"]}
    expected |= {f"step{step}.{donor}.{arm}" for step in plan["cut_steps"] for donor in ("relation", "color", "shape") for arm in plan["arms"]}
    expected |= {f"step{step}.base.sham" for step in plan["cut_steps"]}
    require(set(rows) == expected and observed["counts"] == summary["counts"] and len(rows) == plan["max_images"], "planned row/count inventory differs")
    pixels, latents, images = {}, {}, {}
    for name, row in rows.items():
        require(json.loads((data / "rows" / (name + ".json")).read_text()) == row, "row differs from durable observation")
        require(row["image"]["sha256"] == h["sha"]((data / row["image"]["path"]).read_bytes()), "image differs")
        pixels[name] = np.asarray(Image.open(data / row["image"]["path"]).convert("RGB"))
        require(pixels[name].shape == (512, 512, 3), "unexpected RGB geometry")
        images[name] = typed["image_diagnostics"](pixels[name])
        latents[name] = {}
        for saved in row["schedulers"]:
            path = data / saved["path"]
            values = np.load(path, allow_pickle=False)
            require(saved["sha256"] == h["sha"](path.read_bytes()) and typed["matches"](values, saved["descriptor"], bf16=True), "saved scheduler array differs")
            latents[name][saved["step"]] = values
        require(max(latents[name]) == 4, "missing final continuation")
        if row["kind"] != "native":
            require(row["undeclared_slots_exact"] and len(row["acts"]) == 2, "invalid declared slot closure")
            require(row["parent"] == source["cuts"]["base"][str(row["step"])]["parent"]
                    and row["recipient_projected"] == source["cuts"]["base"][str(row["step"])]["projected"]
                    and row["donor_projected"] == source["cuts"][row["donor"]][str(row["step"])]["projected"], "wrong parent or donor")
            for action in row["acts"]:
                receipt = data / "typed-store" / "receipts" / (action["receipt_id"] + ".json")
                require(receipt.is_file(), "missing Act receipt")
                body = json.loads(receipt.read_text())
                unsigned = {key: value for key, value in body.items() if key != "fingerprint"}
                require(body == action["receipt"] and body["fingerprint"] == action["receipt_id"]
                        == h["sha"](h["canonical"](unsigned)), "mismatched Act receipt")
            # Independently compare every post-Act payload slot to the recipient
            # or donor typed descriptors, beyond the worker's boolean statement.
            current = store["cuts"][row["post_act"]]["payload"]
            before = store["cuts"][row["recipient_projected"]]["payload"]
            donor = store["cuts"][row["donor_projected"]]["payload"]
            fields = lambda value: {item["key"]["value"]: item["value"] for item in value["items"]}
            actual, baseline, donor_fields = fields(current), fields(before), fields(donor)
            for slot in baseline:
                swapped = (row["kind"] == "both" and slot in {"text", "conditioning"}) or slot == row["kind"]
                origin = donor_fields if swapped else baseline
                require(actual[slot] == origin[slot], "typed post-Act slot differs: " + slot)
    for prompt in plan["prompts"]:
        name = prompt["id"]
        latents[name + ".native"][1] = slot_array(data / "typed-store", source["cuts"][name]["1"]["parent"], "latent", typed)
    comparisons, control_findings = {}, []
    for name, row in rows.items():
        if row["kind"] == "native":
            continue
        donor = row["donor"] + ".native"
        comparisons[name] = {"rgb": comparison(pixels[name], pixels["base.native"], pixels[donor]),
            "scheduler_trajectory": {str(step): comparison(values, latents["base.native"][step], latents[donor][step])
                for step, values in latents[name].items()}}
        if row["kind"] == "sham" and not (comparisons[name]["rgb"]["exact_base"] and all(item["exact_base"] for item in comparisons[name]["scheduler_trajectory"].values())):
            control_findings.append({"row": name, "finding": "same-parent sham differs"})
        if row["kind"] == "both" and row["step"] == 0 and not (comparisons[name]["rgb"]["exact_target"] and all(item["exact_target"] for item in comparisons[name]["scheduler_trajectory"].values())):
            control_findings.append({"row": name, "finding": "coherent step0 donor differs"})
    mlp, deltas = {}, {}
    with np.load(data / source["mlp"]["base"]["path"], allow_pickle=False) as saved:
        base_ports = dict(saved)
    for name, record in source["mlp"].items():
        path = data / record["path"]
        require(h["sha"](path.read_bytes()) == record["sha256"], "MLP file differs")
        with np.load(path, allow_pickle=False) as saved:
            arrays = dict(saved)
        require(all(typed["matches"](arrays[key], desc, bf16=True) for key, desc in record["ports"].items()), "MLP native port descriptor differs")
        require(np.array_equal(typed["quantize_bf16"](arrays["r"] + arrays["write"]), arrays["post_image"]), "MLP residual equation differs")
        metrics = {}
        for slot, values in arrays.items():
            delta = values.astype(np.float64) - base_ports[slot]
            reference = float(np.linalg.norm(base_ports[slot].astype(np.float64).ravel()))
            metrics[slot] = {"l2_vs_base": float(np.linalg.norm(delta.ravel())), "relative_l2_vs_base": float(np.linalg.norm(delta.ravel()) / reference) if reference else None,
                "max_abs_vs_base": float(np.abs(delta).max()), "exact_base": bool(np.array_equal(values, base_ports[slot]))}
        mlp[name] = metrics
        if name != "base":
            deltas[name] = arrays["write"].astype(np.float64) - base_ports["write"]
            energy = np.square(deltas[name]).sum(-1)[0]
            mlp[name]["write"]["spatial_energy"] = energy.tolist()
            ordered = np.sort(energy)[::-1]
            total = float(energy.sum())
            mlp[name]["write"]["top_10pct_spatial_energy_fraction"] = float(ordered[:int(np.ceil(.1 * len(ordered)))].sum() / total) if total else None
    cosine = {}
    for a, b in itertools.combinations(deltas, 2):
        left, right = deltas[a].ravel(), deltas[b].ravel()
        norm = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosine[a + "/" + b] = float(np.dot(left, right) / norm) if norm else None
    ids = slot_array(data / "typed-store", source["cuts"]["base"]["0"]["projected"], "img_ids", typed)
    return {"schema": "mdb-flux-concept-routes-audit-v1", "source": run["source"], "mechanics_status": "valid",
        "auditor_source_sha256": h["sha"](Path(__file__).read_bytes()),
        "audit_helpers": {path.name: h["sha"](path.read_bytes()) for path in
            (directory / "summarize_context_window.py", directory / "analyze_flux_mlp_forecast.py")},
        "instrument_status": "control-findings" if control_findings else "shams-and-coherent-donor-exact",
        "control_findings": control_findings, "terminal_status": "unassessed", "plan": plan,
        "store_counts": store["counts"], "counts": summary["counts"], "sealed_hypotheses": forecast,
        "comparisons": comparisons, "image_diagnostics": images, "mlp_contrasts": mlp, "write_contrast_cosines": cosine,
        "spatial_grid": {"axes": ["T", "H", "W", "L"], "ids": ids[0].tolist(), "authority": "native image position IDs; supplied spatial lattice"},
        "generic_records": [
            {"id": "flux.route.C", "structural_name": "stored-conditioning-reread", "reads": "conditioning", "consumer": "context_embedder at each project boundary", "semantic_alias": None},
            {"id": "flux.route.T", "structural_name": "within-step-working-text", "reads": "text", "consumer": "joint/single attention and MLPs until scheduler clears working state", "semantic_alias": None},
            {"id": "flux.write.J0FF", "structural_name": "joint0-time-gated-image-ff-write", "reads": "post-attention normalized/modulated image", "consumer": "residual image and remaining native denoiser", "semantic_alias": None}],
        "limits": ["One fresh seed; supplied scene grammar and contrast labels; no calibrated shape evaluator or learned object masks.",
            "Whole-field conditioning/text interventions do not isolate a particular concept or minimal writer.",
            "RGB and latent target projections are numerical descriptors, not semantic accuracy scores.",
            "Prefixes through step1 projection were observed; altered futures were not independently forecast mathematically."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "instrument_status": result["instrument_status"], "counts": result["counts"]}))


if __name__ == "__main__":
    main()
