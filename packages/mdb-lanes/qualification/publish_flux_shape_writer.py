"""Publish every native shape-writer factorial row without resampling pixels.

This is saved-evidence arithmetic and rendering. No model, native replay,
external weights, attention or diffusion equations are evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import runpy
from pathlib import Path, PurePosixPath

MASKS = tuple("".join(bits) for bits in itertools.product("01", repeat=3))
ROWS = ("base", "changed", "copy.sham", *("merge." + mask for mask in MASKS))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def factorial_effects(values):
    """Raw signed finite differences on p, with explicit coefficients/weights."""
    require(set(values) == set(MASKS) and all(value is None or type(value) in {int, float}
            and math.isfinite(value) for value in values.values()), "invalid factorial scalar inventory")
    raw, averaged = {}, {}
    for selected in MASKS:
        active = [i for i, bit in enumerate(selected) if bit == "1"]
        label = "".join("RAM"[i] for i in active) or "intercept"
        raw_weights, average_weights = {}, {}
        for mask in MASKS:
            if all(mask[i] == "0" for i in range(3) if i not in active):
                raw_weights[mask] = (-1) ** sum(mask[i] == "0" for i in active)
            average_weights[mask] = (-1) ** sum(mask[i] == "0" for i in active) / 2 ** (3 - len(active))
        calculate = lambda weights: None if any(values[mask] is None for mask in weights) else sum(weights[mask] * values[mask] for mask in weights)
        raw[label] = {"value": calculate(raw_weights), "weights_on_mask_p": raw_weights}
        averaged[label if active else "grand_mean"] = {"value": calculate(average_weights), "weights_on_mask_p": average_weights}
    return {"baseline_referenced": raw, "averaged_finite_differences": averaged,
            "scope": "baseline coefficients use p000 and inclusion-exclusion at other factors=0; averaged differences average over complementary factors; no rescaling into semantic operations or shape scores"}


def conditional_surfaces(values):
    result = {}
    difference = lambda terms: None if any(values[mask] is None for mask in terms) else sum(values[mask] * weight for mask, weight in terms.items())
    for residual in "01":
        p = {mask[1:]: values[mask] for mask in MASKS if mask[0] == residual}
        result["R=" + residual] = {"p_by_A_M": p,
            "A_at_M0": difference({residual + "10": 1, residual + "00": -1}),
            "A_at_M1": difference({residual + "11": 1, residual + "01": -1}),
            "M_at_A0": difference({residual + "01": 1, residual + "00": -1}),
            "M_at_A1": difference({residual + "11": 1, residual + "10": -1}),
            "A_M_interaction": difference({residual + "11": 1, residual + "10": -1, residual + "01": -1, residual + "00": 1}),
            "scope": "Native recipient receiving residual; A/M ingredient contrasts." if residual == "0" else
                     "Copied base receiving residual may already encode shape; separate state condition, with no monotonic-bound assumption."}
    return result


def compose_sheet(images, p_values):
    """Paste original RGB pixels, with all annotation outside each image tile."""
    from PIL import Image, ImageDraw, ImageFont

    require(set(images) == set(p_values) == set(ROWS), "incomplete contact sheet")
    require(all(image.mode == "RGB" and image.size == (512, 512) for image in images.values()), "requires original 512px RGB images")
    header, band, width = 72, 88, 512
    sheet = Image.new("RGB", (4 * width, header + 3 * (width + band)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((12, 8), "Native shape-writer futures | R/A/M donor bits | common base conditioning from step2", fill="black", font=font)
    draw.text((12, 36), "Final latent p: base=0, changed=1. Numerical geometry, not a semantic shape score.", fill="black", font=font)
    cells = [(0, index, name) for index, name in enumerate(ROWS[:3])]
    cells += [(1 + index // 4, index % 4, "merge." + mask) for index, mask in enumerate(MASKS)]
    tiles = {}
    for row, col, name in cells:
        x, y = col * width, header + row * (width + band)
        sheet.paste(images[name], (x, y))
        box = [x, y, x + width, y + width]
        require(sheet.crop(box).tobytes() == images[name].tobytes(), "contact sheet changed original pixels")
        value = p_values[name]
        label = {"base": "B | early-history reference", "changed": "D | shape-step1 reference", "copy.sham": "D | image copy sham"}.get(name, "R/A/M bits " + name.removeprefix("merge."))
        draw.text((x + 10, y + width + 8), name + " | " + label, fill="black", font=font)
        draw.text((x + 10, y + width + 40), "final p=" + ("undefined" if value is None else f"{value:.9g}"), fill="black", font=font)
        tiles[name] = {"image_box": box, "label_band": [x, y + width, x + width, y + width + band],
                       "pixel_sha256": hashlib.sha256(images[name].tobytes()).hexdigest()}
    return sheet, tiles


def publish(run_dir, audit_path, output):
    from PIL import Image

    run_dir, audit_path, output = (Path(value).resolve() for value in (run_dir, audit_path, output))
    require(not output.exists(), "publication output already exists")
    helper_path = Path(__file__).with_name("summarize_context_window.py")
    helper = runpy.run_path(str(helper_path))
    run = helper["source_run"](run_dir, "flux-merge-result", max_archive_bytes=512 << 20, max_member_bytes=128 << 20)
    audit, plan, data = read(audit_path), run["plan"], run["data"]
    require(audit["mechanics_status"] == "valid" and audit["source"] == run["source"]
            and audit["counts"]["images"] == plan["max_images"] == 11
            and plan["panel"] == audit["panel"] == "shape-step1-writer"
            and audit["selected_block"] == plan["block"] == 4 and audit["scheduler_steps"] == [2, 3, 4], "audit/source/panel differs")
    require(audit["auditor_source_sha256"] == sha(Path(__file__).with_name("analyze_flux_merge.py"))
            and audit["audit_helpers"][helper_path.name] == sha(helper_path), "audit computation/custody source differs")
    observed = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    rows = {row["id"]: row for row in observed["rows"]}
    require(set(rows) == set(ROWS) == set(audit["comparisons"]), "row inventory differs")
    images, p_values, row_index = {}, {}, []
    for name in ROWS:
        row = rows[name]
        require(read(data / "rows" / (name + ".json")) == row, "durable/native row differs")
        relative = PurePosixPath(row["image"]["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() == row["image"]["path"], "unsafe image path")
        path = data / relative
        require(path.is_file() and not path.is_symlink() and sha(path) == row["image"]["sha256"], "native PNG differs")
        with Image.open(path) as opened:
            require(opened.mode == "RGB" and opened.size == (512, 512), "native RGB ABI differs")
            images[name] = opened.copy()
        p_values[name] = audit["comparisons"][name]["scheduler_trajectory"]["4"]["target_projection"]
        row_index.append({"id": name, "kind": row["kind"], "mask_R_A_M": row.get("mask_R_A_M"),
            "parent": row["parent"], "post_act": row.get("post_act"), "final_cut": row["final_cut"],
            "row_record": {"path": str((data / "rows" / (name + ".json")).relative_to(run_dir)), "sha256": sha(data / "rows" / (name + ".json"))},
            "native_image": {"path": str(path.relative_to(run_dir)), "sha256": sha(path)}, "final_p": p_values[name],
            "comparisons": audit["comparisons"][name], "conditioning_reset": audit["conditioning_resets"][name],
            "native_path": audit["native_paths"][name]})
    effects = factorial_effects({mask: p_values["merge." + mask] for mask in MASKS})
    mask_values = {mask: p_values["merge." + mask] for mask in MASKS}
    ranks = {}
    for key, value in (("H-RA-rank", None if any(mask_values[mask] is None for mask in ("110", "100", "010")) else mask_values["110"] - min(mask_values["100"], mask_values["010"])),
                       ("H-A-vs-M", None if any(mask_values[mask] is None for mask in ("010", "001")) else mask_values["010"] - mask_values["001"])):
        ranks[key] = {"sealed_hypothesis": plan["hypotheses"][key], "contrast_value": value,
                      "contrast": "p110-min(p100,p010)" if key == "H-RA-rank" else "p010-p001", "expected_sign": "negative",
                      "observation": "undefined-contrast" if value is None else "direction-match" if value < 0 else "prediction-miss",
                      "scope": "Numerical rank only; not a shape label, scientific gate or rule-promotion decision."}
    controls = {"local": audit["control_observations"], "historical_PNG": {
        name: {"actual_sha256": rows[name]["image"]["sha256"], "declared_historical_sha256": expected,
               "exact": rows[name]["image"]["sha256"] == expected,
               "historical_role": "coarse after_single4.restore_image" if name == "merge.111" else "coarse native reference " + name}
        for name, expected in plan["historical_pngs"].items()}}
    sheet, tiles = compose_sheet(images, p_values)
    output.mkdir(parents=True, exist_ok=False)
    sheet.save(output / "shape-step1-writer.png")
    with Image.open(output / "shape-step1-writer.png") as saved:
        require(all(saved.crop(tile["image_box"]).tobytes() == images[name].tobytes() for name, tile in tiles.items()), "saved contact sheet pixels differ")
    result = {"schema": "mdb-flux-shape-writer-publication-v1", "mechanics_status": "valid", "terminal_status": "unassessed",
        "source": run["source"], "reference_base": os.path.relpath(run_dir, output), "audit": {"path": os.path.relpath(audit_path, output), "sha256": sha(audit_path)},
        "publisher_source_sha256": sha(__file__), "custody_helper_sha256": sha(helper_path), "plan": plan,
        "counts": audit["counts"], "row_index": row_index, "factorial_effects": effects,
        "receiving_residual_surfaces": conditional_surfaces(mask_values), "rank_observations": ranks, "control_observations": controls,
        "native_merge_equation_checks": audit["native_merge_equation_checks"], "recipe_checks": audit["recipe_checks"],
        "figure": {"path": "shape-step1-writer.png", "sha256": sha(output / "shape-step1-writer.png"), "tiles": tiles,
                   "pixel_policy": "original RGB bytes pasted at512px with labels in separate bands; no resampling, recoloring or generated imagery"},
        "sealed_hypotheses": audit["sealed_hypotheses"], "visual_inspection": "pending separate assistant inspection; no automatic shape labels",
        "new_model_calls": 0, "model_or_native_suffix_equations_executed": False,
        "limitations": ["Eleven dependent native futures share one observed shape contrast; prior coarse outcomes informed selection and controls.",
            "p and factorial coefficients describe signed final-latent geometry, not semantic operations, sphere scores or independent image forecasts.",
            "Source ports and shared native projection outputs were observed; the publisher does not reevaluate weights, attention or diffusion.",
            "All masks and failed controls remain in the publication. Undefined p leaves affected coefficients undefined.",
            "Full image state is edited; there is no discovered object mask, first-writer localization or minimal semantic circuit claim."]}
    save(output / "result.json", result)
    lines = ["# Native shape-writer factorial", "", "Every original native row is retained. p uses base=0 and changed=1; it is numerical geometry, not shape correctness.", "",
             "| Row | Kind | R A M donor mask | Final p |", "|---|---|---|---:|"]
    for row in row_index:
        value = row["final_p"]
        lines.append(f"| {row['id']} | {row['kind']} | {row['mask_R_A_M'] or 'reference/copy'} | {'undefined' if value is None else format(value, '.9g')} |")
    lines += ["", "## Raw signed factorial coefficients", "", "Baseline-referenced inclusion-exclusion on the eight p values. Negative values point baseward on this axis; interactions are downstream numerical responses, not named semantic operations.", "", "| Term | Coefficient |", "|---|---:|"]
    for term in ("intercept", "R", "A", "M", "RA", "RM", "AM", "RAM"):
        value = effects["baseline_referenced"][term]["value"]
        lines.append(f"| {term} | {'undefined' if value is None else format(value, '.9g')} |")
    lines += ["", "R=0 A/M contrasts retain the native receiving residual. R=1 copies an already-encoded base receiving residual and is a separate state condition, not a monotonic bound. Both surfaces, averaged finite differences and all numerical prediction misses are retained in result.json.",
              "", "## Historical PNG observations", "", "| Row | Historical role | Byte exact |", "|---|---|---|"]
    for name, control in controls["historical_PNG"].items():
        lines.append(f"| {name} | {control['historical_role']} | {control['exact']} |")
    lines += ["", "Visual interpretation is pending separate inspection; numerical p does not classify spheres, cubes or mixed shapes.", "", "![Eleven original native futures](shape-step1-writer.png)", ""]
    with (output / "result.md").open("x") as stream:
        stream.write("\n".join(lines))
    with (output / "publisher-source.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = publish(args.run_dir, args.audit, args.output)
    print(json.dumps({"mechanics_status": result["mechanics_status"], "rows": len(result["row_index"]), "new_model_calls": 0}))
