"""Publish two native panels with distinct continuation policies and metrics.

Only saved evidence, scalar differences and unchanged original RGB tiles are
used. No model, replay, attention or native suffix equations execute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
from pathlib import Path

DONOR = ("base", "changed", "copy.sham", "merge.000", "merge.001", "merge.010", "merge.011")
SELF = ("self.native", "self.11", "self.10", "self.01", "self.00", "self.1h", "self.h1")
GAINS = {"self.11": [1., 1.], "self.10": [1., 0.], "self.01": [0., 1.],
         "self.00": [0., 0.], "self.1h": [1., .5], "self.h1": [.5, 1.]}


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


def numeric_contrasts(p, drift):
    require(set(p) == {"000", "001", "010", "011"} and set(drift) == set(SELF), "contrast row inventory differs")
    require(all(value is None or type(value) in {int, float} and math.isfinite(value) for value in p.values())
            and all(type(value) in {int, float} and math.isfinite(value) for value in drift.values()), "nonfinite contrast")
    difference = lambda weights: None if any(p[key] is None for key in weights) else sum(p[key] * weight for key, weight in weights.items())
    return {"donor_R0": {"A_at_M0": difference({"010": 1, "000": -1}),
        "M_at_A0": difference({"001": 1, "000": -1}), "A_at_M1": difference({"011": 1, "001": -1}),
        "M_at_A1": difference({"011": 1, "010": -1}),
        "A_M_interaction": difference({"011": 1, "010": -1, "001": -1, "000": 1})},
        "rank_values": {"H-adjacent-A-vs-M": difference({"010": 1, "001": -1}),
            "H-adjacent-AM": None if any(p[key] is None for key in ("011", "010", "001")) else p["011"] - min(p["010"], p["001"]),
            "H-self-drift": drift["self.01"] - drift["self.10"]},
        "scope": "Donor quantities use common-base-C p only; self contrast uses final latent L2 from matched own-C self.native only."}


def compose_panel(order, images, metrics, *, title, metric_label, labels):
    from PIL import Image, ImageDraw, ImageFont

    require(len(order) == len(set(order)) == 7 and all(name in images and name in metrics for name in order), "panel inventory differs")
    require(all(images[name].mode == "RGB" and images[name].size == (512, 512) for name in order), "requires original512px RGB tiles")
    width, band, header = 512, 88, 72
    sheet = Image.new("RGB", (3 * width, header + 3 * (width + band)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((12, 8), title, fill="black", font=font)
    draw.text((12, 36), metric_label + " | numerical diagnostic, not a shape score", fill="black", font=font)
    tiles = {}
    for index, name in enumerate(order):
        x, y = index % 3 * width, header + index // 3 * (width + band)
        box = [x, y, x + width, y + width]
        sheet.paste(images[name], (x, y))
        require(sheet.crop(box).tobytes() == images[name].tobytes(), "panel changed native pixels")
        value = metrics[name]
        draw.text((x + 10, y + width + 8), name + " | " + labels[name], fill="black", font=font)
        draw.text((x + 10, y + width + 40), metric_label + "=" + ("undefined" if value is None else f"{value:.8g}"), fill="black", font=font)
        tiles[name] = {"image_box": box, "label_band": [x, y + width, x + width, y + width + band],
                       "pixel_sha256": hashlib.sha256(images[name].tobytes()).hexdigest()}
    return sheet, tiles


def publish(run_dir, audit_path, output):
    from PIL import Image

    run_dir, audit_path, output = (Path(path).resolve() for path in (run_dir, audit_path, output))
    require(not output.exists(), "publication output already exists")
    here = Path(__file__).resolve().parent
    helper_path = here / "summarize_context_window.py"
    run = runpy.run_path(str(helper_path))["source_run"](run_dir, "flux-merge-result", max_archive_bytes=512 << 20, max_member_bytes=128 << 20)
    data, plan, audit = run["data"], run["plan"], read(audit_path)
    require(audit["mechanics_status"] == "valid" and audit["source"] == run["source"]
            and audit["panel"] == plan["panel"] == "shape-step1-recipient-controls"
            and audit["selected_block"] == plan["block"] == 5 and audit["counts"]["images"] == plan["max_images"] == 14,
            "completed recipient-panel audit/source differs")
    require(audit["auditor_source_sha256"] == sha(here / "analyze_flux_merge.py")
            and audit["audit_helpers"][helper_path.name] == sha(helper_path), "auditor/custody source differs")
    observed = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    rows = {row["id"]: row for row in observed["rows"]}
    require(set(rows) == set(audit["comparisons"]) == set(DONOR + SELF)
            and set(audit["recipient_comparisons"]) == set(SELF), "completed row inventory differs")
    source = read(data / "source.json")
    images, metrics, row_index = {}, {}, []

    def artifact(path):
        return {"path": str(path.relative_to(run_dir)), "sha256": sha(path)}

    def verified_record(record):
        path = data / record["path"]
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(data.resolve())
                and sha(path) == record["sha256"], "retained array/projection hash differs")
        return {**record, **artifact(path)}

    def cut(fingerprint):
        path = data / "typed-store/cuts" / (fingerprint + ".json")
        require(read(path)["fingerprint"] == fingerprint, "retained cut identity differs")
        return {"fingerprint": fingerprint, **artifact(path)}

    for name in DONOR + SELF:
        row = rows[name]
        require(read(data / "rows" / (name + ".json")) == row, "durable native row differs")
        self_row = name in SELF
        policy = "recipient-own-C" if self_row else "common-base-reset"
        require(row["continuation_policy"] == policy and len(row["continuation_acts"]) == (0 if self_row else 1), "continuation policy/reset differs")
        if name in GAINS:
            require(row["recipe_mode"] == "recipient-only-scale" and row["gains"] == GAINS[name], "own-operand gains differ")
        path = data / row["image"]["path"]
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(data.resolve())
                and sha(path) == row["image"]["sha256"], "original PNG differs")
        with Image.open(path) as opened:
            require(opened.mode == "RGB" and opened.size == (512, 512), "native image ABI differs")
            images[name] = opened.copy()
        comparison = audit["recipient_comparisons"][name] if self_row else audit["comparisons"][name]
        final = comparison["scheduler_trajectory"]["4"]
        require(set(comparison["scheduler_trajectory"]) == {"2", "3", "4"}, "scheduler horizon differs")
        if self_row:
            require(final["target_projection"] is None, "self metric must not use unmatched global p")
        metrics[name] = final["base_l2"] if self_row else final["target_projection"]
        continuation = []
        for action in row["continuation_acts"]:
            continuation.append({"parent": cut(action["parent"]), "post_act": cut(action["post_act"]),
                "receipt": artifact(data / "typed-store/receipts" / (action["receipt_id"] + ".json")),
                "receipt_id": action["receipt_id"], "point": action["point"], "source_cut": action["source_cut"]})
        record = {"id": name, "kind": row["kind"], "continuation_policy": policy, "recipe_mode": row["recipe_mode"],
            "gains_A_M": None if name not in GAINS else dict(zip(("A", "M"), GAINS[name], strict=True)),
            "mask_R_A_M": row.get("mask_R_A_M"), "row_record": artifact(data / "rows" / (name + ".json")),
            "image": artifact(path), "parent": cut(row["parent"]), "post_act": None if "post_act" not in row else cut(row["post_act"]),
            "final_cut": cut(row["final_cut"]), "conditioning_at_start": row["conditioning_at_start"],
            "conditioning_source_cut": row["conditioning_source_cut"], "continuation_acts": continuation,
            "scheduler_arrays": [verified_record(saved) for saved in row["schedulers"]],
            "headline_metric": {"name": "final_latent_L2_from_self_native" if self_row else "final_latent_p_base0_changed1", "value": metrics[name]},
            "matched_comparisons": comparison, "native_path": audit["native_paths"][name]}
        if "projection" in row:
            record["native_projection"] = verified_record(row["projection"])
        if "receipt" in row:
            record["image_Act_receipt"] = {"fingerprint": row["receipt"]["fingerprint"],
                **artifact(data / "typed-store/receipts" / (row["receipt"]["fingerprint"] + ".json"))}
        row_index.append(record)
    p = {mask: metrics["merge." + mask] for mask in ("000", "001", "010", "011")}
    contrasts = numeric_contrasts(p, {name: metrics[name] for name in SELF})
    rank_observations = {}
    for key, value in contrasts["rank_values"].items():
        positive = key == "H-self-drift"
        rank_observations[key] = {"sealed_hypothesis": plan["hypotheses"][key], "contrast_value": value,
            "expected_sign": "positive" if positive else "negative", "observation": "undefined-contrast" if value is None else
                "direction-match" if (value > 0 if positive else value < 0) else "prediction-miss",
            "scope": "Numerical diagnostic only; preserve every miss, not a semantic decision or scientific filter."}
    controls = {"local": audit["control_observations"], "historical_PNG": {
        name: {"actual_sha256": rows[name]["image"]["sha256"], "declared_sha256": expected,
               "exact": rows[name]["image"]["sha256"] == expected}
        for name, expected in plan["historical_pngs"].items()}}
    labels = {name: "native reference" if name in {"base", "changed"} else "image copy sham" if name == "copy.sham" else
              "own native future" if name == "self.native" else "A,M gains " + ",".join(map(str, GAINS[name])) if name in GAINS else
              "donor A,M bits " + name.removeprefix("merge.")[1:] for name in DONOR + SELF}
    panels = {}
    output.mkdir(parents=True, exist_ok=False)
    for key, order, title, metric_label in (("donor", DONOR, "Single5 donor A/M | R=recipient | common base C from step2", "p(base0,changed1)"),
                                         ("recipient", SELF, "Single5 recipient-only A/M gains | own future C | no reset", "L2 from self.native")):
        sheet, tiles = compose_panel(order, images, metrics, title=title, metric_label=metric_label, labels=labels)
        path = output / ("recipient-controls-" + key + ".png")
        sheet.save(path)
        with Image.open(path) as saved:
            require(all(saved.crop(tiles[name]["image_box"]).tobytes() == images[name].tobytes() for name in order), "saved panel altered original pixels")
        panels[key] = {"path": path.name, "sha256": sha(path), "tiles": tiles,
                       "continuation_policy": "common-base-reset" if key == "donor" else "recipient-own-C", "metric": metric_label}
    result = {"schema": "mdb-flux-recipient-controls-publication-v1", "mechanics_status": "valid", "terminal_status": "unassessed",
        "reference_base": os.path.relpath(run_dir, output), "source": run["source"], "plan": plan, "counts": audit["counts"],
        "audit": {"path": os.path.relpath(audit_path, output), "sha256": sha(audit_path), "auditor_source_sha256": audit["auditor_source_sha256"]},
        "publisher_source_sha256": sha(__file__), "custody_helper_sha256": sha(helper_path), "rows": row_index, "contrasts": contrasts,
        "rank_observations": rank_observations, "control_observations": controls, "panels": panels,
        "source_ports": {name: verified_record(record) for name, record in source["ports"].items()},
        "source_cut_ids": {key: source[key] for key in ("origin_cuts", "input_cuts", "post_cuts")},
        "source_policies": {key: source[key] for key in ("recipient_controls", "continuation_policies")},
        "source_record": artifact(data / "source.json"), "seal": artifact(run["transaction"] / "seal.json"),
        "global_unmatched_reference_geometry": {name: audit["comparisons"][name] for name in SELF},
        "sealed_hypotheses": audit["sealed_hypotheses"], "visual_inspection": "pending separate arm-blinded inspection; no automatic labels",
        "pixel_policy": "original RGB bytes pasted at512px; labels in separate bands; no resampling or image synthesis",
        "new_model_calls": 0, "model_or_native_suffix_equations_executed": False,
        "limitations": ["Fourteen dependent futures from two observed histories at one occurrence/context, not independent input trials.",
            "The panels have different future C policies. Self headline metrics use matched own-C self.native; global donor p is not their causal score.",
            "p and drift are numerical geometry, not shape/ordering correctness. Undefined p and every prediction miss remain visible.",
            "Recipient-only recipes have no donor tensor; native operands, residual, gate, full projection/bias and post-block text are retained as declared.",
            "Zeroing an operand does not skip its native production or remove projection bias. These are conditional ingredient tests, not minimal-writer proof.",
            "No independent numerical RGB forecast, raw-prompt forecast, universal semantic operation or automatic family induction is claimed."]}
    save(output / "result.json", result)
    lines = ["# Adjacent donor and recipient-only controls", "", "Two continuation policies and two numerical metrics are reported separately. Original pixels are unchanged; visual labels remain pending.", ""]
    for title, order, label in (("Common-base-C donor panel", DONOR, "Final p"), ("Own-C recipient-only panel", SELF, "Final latent L2 from self.native")):
        lines += ["## " + title, "", "| Row | Recipe | " + label + " |", "|---|---|---:|"]
        for name in order:
            value = metrics[name]
            lines.append(f"| {name} | {labels[name]} | {'undefined' if value is None else format(value, '.9g')} |")
        lines.append("")
    lines += ["All donor conditional differences, matched self distances, numerical prediction misses, reset/native paths and original hashes are retained in result.json. Global donor-axis geometry for self rows is retained only as explicitly unmatched context.", "",
              "![Common-source donor panel](recipient-controls-donor.png)", "", "![Own-source recipient panel](recipient-controls-recipient.png)", ""]
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
    print(json.dumps({"mechanics_status": result["mechanics_status"], "rows": len(result["rows"]), "new_model_calls": 0}))
