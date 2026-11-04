"""Publish audited joint-self native futures and scheduler-latent previews.

Saved evidence only: no native model, replay, attention or suffix computation.
Image tiles retain their original 512px RGB values; labels occupy separate bands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
from pathlib import Path

ROWS = ("native", "sham", "Ahalf.recompute", "Azero.recompute", "Ahalf.hold", "Azero.hold",
        "Mhalf", "Mzero", "bothzero", "text.rollback", "Azero.recompute.textrollback", "Azero.hold.textrollback")
PREVIEW_ROWS = ("native", "Azero.recompute", "Azero.hold", "Mzero")


def require(condition, message):
    if not condition:
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


def compose_grid(order, images, annotations, *, title):
    """Return a sheet and crop records; never scale or draw on source pixels."""
    from PIL import Image, ImageDraw, ImageFont

    names = [name for row in order for name in row]
    require(bool(order) and all(len(row) == len(order[0]) for row in order)
            and len(names) == len(set(names)) and set(names) == set(images) == set(annotations), "grid inventory differs")
    require(all(image.mode == "RGB" and image.size == (512, 512) for image in images.values()), "requires original512px RGB")
    width, band, header = 512, 88, 72
    sheet = Image.new("RGB", (len(order[0]) * width, header + len(order) * (width + band)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((12, 8), title, fill="black", font=font)
    draw.text((12, 37), "Original native pixels | distances are numerical, not shape scores", fill="black", font=font)
    tiles = {}
    for row_index, row in enumerate(order):
        for column, name in enumerate(row):
            x, y = column * width, header + row_index * (width + band)
            box = [x, y, x + width, y + width]
            sheet.paste(images[name], (x, y))
            require(sheet.crop(box).tobytes() == images[name].tobytes(), "grid altered native pixels")
            draw.text((x + 8, y + width + 8), name, fill="black", font=font)
            draw.text((x + 8, y + width + 39), annotations[name], fill="black", font=font)
            tiles[name] = {"image_box": box, "label_band": [x, y + width, x + width, y + width + band],
                           "pixel_sha256": hashlib.sha256(images[name].tobytes()).hexdigest()}
    return sheet, tiles


def numeric_ranks(final_native_l2):
    """Retain both tentative rank signs without assigning scientific validity."""
    required = ("Azero.recompute", "Mzero", "Ahalf.recompute", "Ahalf.hold")
    require(all(type(final_native_l2.get(name)) in {float, int} and math.isfinite(final_native_l2[name])
                and final_native_l2[name] >= 0 for name in required), "invalid matched final drift")
    values = {"H-A-vs-M": (final_native_l2["Azero.recompute"] - final_native_l2["Mzero"], "positive"),
              "H-compensation": (final_native_l2["Ahalf.recompute"] - final_native_l2["Ahalf.hold"], "negative")}
    return {key: {"value": value, "expected_sign": sign,
                  "observation": "direction-match" if (value > 0 if sign == "positive" else value < 0) else "prediction-miss",
                  "scope": "Tentative numerical rank only; zero, misses and heterogeneous observations are retained."}
            for key, (value, sign) in values.items()}


def publish(run_dir, audit_path, output):
    from PIL import Image

    run_dir, audit_path, output = (Path(path).resolve() for path in (run_dir, audit_path, output))
    require(not output.exists(), "publication output already exists")
    here = Path(__file__).resolve().parent
    helper = here / "summarize_context_window.py"
    run = runpy.run_path(str(helper))["source_run"](run_dir, "flux-joint-self-result", max_archive_bytes=512 << 20,
                                                     max_member_bytes=128 << 20)
    audit, data, plan = read(audit_path), Path(run["data"]).resolve(), run["plan"]
    require(audit["mechanics_status"] == "valid" and audit["source"] == run["source"], "completed audit/source differs")
    require(audit["auditor_source_sha256"] == sha(here / "analyze_flux_joint_self.py")
            and audit["audit_helpers"][helper.name] == sha(helper), "auditor/custody source differs")
    value = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    rows = {row["id"]: row for row in value["rows"]}
    require(set(rows) == set(audit["comparisons"]) == set(ROWS)
            and audit["counts"]["final_images"] == 12 and audit["counts"]["images"] == 20
            and audit["counts"]["preview_images"] == 8
            and plan["selector"] == {"logical_step": 1, "phase": "joint", "index": 4}, "twelve-future inventory differs")
    rollback_rows = ("Azero.recompute.textrollback", "Azero.hold.textrollback")
    require(set(audit["rollback_comparisons"]) == set(rollback_rows), "matched rollback inventory differs")
    source = read(data / "source.json")

    def artifact(path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run_dir), "artifact escapes run")
        return {"path": str(path.resolve().relative_to(run_dir)), "sha256": sha(path)}

    def verified(record):
        path = data / record["path"]
        binding = artifact(path)
        require(path.resolve().is_relative_to(data.resolve()) and binding["sha256"] == record["sha256"], "retained array/PNG differs")
        return {**record, **binding}

    def normalize_records(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                return verified(value)
            return {key: normalize_records(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize_records(item) for item in value]
        return value

    def cut(fingerprint):
        path = data / "typed-store/cuts" / (fingerprint + ".json")
        require(read(path)["fingerprint"] == fingerprint, "cut identity differs")
        return {"fingerprint": fingerprint, **artifact(path)}

    def png(record):
        bound = verified(record)
        with Image.open(run_dir / bound["path"]) as image:
            require(image.mode == "RGB" and image.size == (512, 512), "original RGB ABI differs")
            return image.copy(), bound

    def trajectory(comparison):
        require(set(comparison["scheduler_trajectory"]) == {"2", "3", "4"}, "retained scheduler horizon differs")
        for metric in comparison["scheduler_trajectory"].values():
            require(type(metric["base_l2"]) in {float, int} and math.isfinite(metric["base_l2"])
                    and metric["base_l2"] >= 0, "invalid matched scheduler distance")
        return comparison

    row_records, images, annotations, native_drift = [], {}, {}, {}
    time_images, time_annotations = {}, {}
    for name in ROWS:
        row = rows[name]
        require(row["continuation_policy"] == "own-C-throughout-no-reset", "own conditioning policy differs")
        row_path = data / "rows" / (name + ".json")
        require(read(row_path) == row, "durable native row differs")
        native_comparison = trajectory(audit["comparisons"][name])
        matched_comparison = trajectory(audit["rollback_comparisons"][name]) if name in rollback_rows else native_comparison
        baseline = "text.rollback" if name in rollback_rows else "native"
        native_drift[name] = native_comparison["scheduler_trajectory"]["4"]["base_l2"]
        final_l2 = matched_comparison["scheduler_trajectory"]["4"]["base_l2"]
        images[name], image_record = png(row["image"])
        annotations[name] = f"Z4 L2={final_l2:.7g} vs {baseline}"
        schedulers = row["schedulers"]
        require({record["step"] for record in schedulers} == {2, 3, 4} and len(schedulers) == 3,
                "scheduler file inventory differs")
        scheduler_records = [{**record, "latent": verified(record["latent"]), "cut": cut(record["cut"])} for record in schedulers]
        acts = []
        for action in row["acts"]:
            receipt_id = action["receipt_id"]
            receipt = data / "typed-store/receipts" / (receipt_id + ".json")
            require(read(receipt) == action["receipt"], "durable Act receipt differs")
            acts.append({**action, "parent": cut(action["parent"]), "post_act": cut(action["post_act"]),
                         "receipt_file": artifact(receipt)})
        previews = row.get("previews", {})
        require(set(previews) == ({"2", "3"} if name in PREVIEW_ROWS else set()), "preview inventory differs")
        preview_records = {}
        for step, record in previews.items():
            key = name + "@Z" + step
            time_images[key], bound = png(record)
            require(record["cut"] == next(item["cut"] for item in schedulers if str(item["step"]) == step),
                    "preview source scheduler cut differs")
            preview_records[step] = {**bound, "cut": cut(record["cut"])}
            time_annotations[key] = f"live latent | L2={native_comparison['scheduler_trajectory'][step]['base_l2']:.7g} vs native"
        if name in PREVIEW_ROWS:
            key = name + "@Z4"
            time_images[key] = images[name]
            time_annotations[key] = f"final native RGB | L2={native_drift[name]:.7g} vs native"
        row_records.append({"id": name, "spec": {key: row[key] for key in
            ("attention_gain", "mlp_gain", "mlp_policy", "rollback_text")}, "continuation_policy": row["continuation_policy"],
            "final_conditioning": row["final_conditioning"], "row_record": artifact(row_path),
            "parent": cut(row["parent"]), "native_post": cut(row["native_post"]), "final_cut": cut(row["final_cut"]),
            "candidate": normalize_records(row["candidate"]), "acts": acts, "image": image_record,
            "schedulers": scheduler_records, "previews": preview_records, "headline_baseline": baseline,
            "matched_comparisons": matched_comparison, "common_native_comparisons": native_comparison,
            "native_path": audit["native_paths"][name]})
    require(len(time_images) == 12 and sum(len(row["previews"]) for row in row_records) == 8, "twenty-PNG inventory differs")
    parity = value["preview_helper_final_parity"]
    require(parity == read(data / "preview-helper-final-parity.json") and parity["pixel_exact"] is True
            and parity["adapter_pixel_sha256"] == parity["helper_pixel_sha256"]
            == hashlib.sha256(images["native"].tobytes()).hexdigest(), "preview helper/final native pixel binding differs")
    ranks = numeric_ranks(native_drift)
    for name, rank in ranks.items():
        rank["sealed_hypothesis"] = audit["sealed_hypotheses"][name]
    output.mkdir(parents=True, exist_ok=False)
    figures = {}
    orders = {"finals": tuple(tuple(ROWS[index:index + 3]) for index in range(0, 12, 3)),
              "timegrid": tuple(tuple(name + "@Z" + step for step in ("2", "3", "4")) for name in PREVIEW_ROWS)}
    for kind, title, originals, labels in (
            ("finals", "Joint4 own-return interventions | own conditioning | twelve futures", images, annotations),
            ("timegrid", "Joint4 selected native branches | live Z2/Z3 previews, final Z4", time_images, time_annotations)):
        sheet, tiles = compose_grid(orders[kind], originals, labels, title=title)
        path = output / ("joint-self-" + kind + ".png")
        sheet.save(path)
        with Image.open(path) as saved:
            require(all(saved.crop(tiles[key]["image_box"]).tobytes() == image.tobytes() for key, image in originals.items()),
                    "saved figure altered original pixels")
        figures[kind] = {"path": path.name, "sha256": sha(path), "tiles": tiles}
    result = {"schema": "mdb-flux-joint-self-publication-v1", "mechanics_status": "valid", "terminal_status": "unassessed",
        "reference_base": os.path.relpath(run_dir, output), "source": run["source"], "plan": plan, "counts": audit["counts"],
        "future_count": 12, "original_PNG_count": 20, "new_model_calls": 0, "model_or_suffix_equations_executed": False,
        "audit": {"path": os.path.relpath(audit_path, output), "sha256": sha(audit_path),
                  "auditor_source_sha256": audit["auditor_source_sha256"]},
        "publisher_source_sha256": sha(__file__), "custody_helper_sha256": sha(helper),
        "source_record": artifact(data / "source.json"), "observed_source": normalize_records(source),
        "preview_helper_final_parity": {"record": value["preview_helper_final_parity"],
                                         "file": artifact(data / "preview-helper-final-parity.json")},
        "seal": artifact(run["transaction"] / "seal.json"), "rows": row_records, "rank_observations": ranks,
        "sealed_hypotheses": audit["sealed_hypotheses"], "control_observations": audit["control_observations"], "figures": figures,
        "pixel_policy": "Original512px RGB bytes pasted without resampling or synthesis; annotations outside image tiles.",
        "visual_inspection": "pending separate inspection; no automatic semantic labels",
        "limitations": ["Twelve dependent futures from one freshly encoded prompt and seed, not independent input trials.",
            "Observed prefix and native same-parent ports precede seal; no independent raw-prompt-to-RGB prediction is claimed.",
            "Own stored conditioning is shared throughout; rollback changes working text only and uses its matched text.rollback baseline.",
            "Recompute-versus-hold is a conditional receiving-state test, not semantic mediation percentage or minimal-writer proof.",
            "Preview PNGs decode live noised scheduler latents and do not feed the future;20 PNGs represent12 futures.",
            "All numeric ranks, misses, ties and full row comparisons remain visible; distances are not shape scores."]}
    save(output / "result.json", result)
    lines = ["# Joint4 own-return and text-state controls", "", "Twelve native futures; eight extra live-latent preview PNGs. Own conditioning remains fixed.", "",
             "| Row | Matched reference | Final latent L2 |", "|---|---|---:|"]
    for row in row_records:
        lines.append(f"| {row['id']} | {row['headline_baseline']} | {row['matched_comparisons']['scheduler_trajectory']['4']['base_l2']:.9g} |")
    lines += ["", "The two rollback attention arms use text.rollback as their matched reference. Common-native distances are also retained explicitly.", "",
              "![Twelve final native images](joint-self-finals.png)", "", "![Four native branches across saved latent clocks](joint-self-timegrid.png)", "",
              "Labels are outside unchanged512px tiles. Live Z2/Z3 previews are not completed diffusion outputs. Visual observations remain separate.", ""]
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
    print(json.dumps({"mechanics_status": result["mechanics_status"], "futures": result["future_count"],
                      "original_PNGs": result["original_PNG_count"]}))
