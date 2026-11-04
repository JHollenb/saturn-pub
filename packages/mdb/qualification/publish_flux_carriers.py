"""Publish saved carrier futures with original pixels and matched rescue metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def compose_panel(order, images, annotations, *, columns, title):
    from PIL import Image, ImageDraw, ImageFont

    require(len(order) == len(set(order)) and set(order) == set(images) == set(annotations)
            and type(columns) is int and columns > 0 and bool(order), "panel inventory differs")
    require(all(image.mode == "RGB" and image.size == (512, 512) for image in images.values()), "requires original512px RGB")
    width, header, band = 512, 72, 88
    sheet = Image.new("RGB", (columns * width, header + math.ceil(len(order) / columns) * (width + band)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((12, 8), title, fill="black", font=font)
    draw.text((12, 37), "Original native pixels | distances are numerical, not semantic scores", fill="black", font=font)
    tiles = {}
    for index, name in enumerate(order):
        x, y = index % columns * width, header + index // columns * (width + band)
        box = [x, y, x + width, y + width]
        sheet.paste(images[name], (x, y))
        require(sheet.crop(box).tobytes() == images[name].tobytes(), "panel altered native pixels")
        draw.text((x + 8, y + width + 8), name, fill="black", font=font)
        draw.text((x + 8, y + width + 40), annotations[name], fill="black", font=font)
        tiles[name] = {"image_box": box, "label_band": [x, y + width, x + width, y + width + band],
                       "pixel_sha256": hashlib.sha256(images[name].tobytes()).hexdigest()}
    return sheet, tiles


def publish(run_dir, audit_path, output):
    from PIL import Image

    run_dir, audit_path, output = (Path(path).resolve() for path in (run_dir, audit_path, output))
    require(not output.exists(), "publication output already exists")
    here = Path(__file__).resolve().parent
    helper = here / "summarize_context_window.py"
    run = runpy.run_path(str(helper))["source_run"](run_dir, "flux-carriers-result", max_archive_bytes=512 << 20,
                                                     max_member_bytes=128 << 20)
    audit, data, plan = read(audit_path), Path(run["data"]).resolve(), run["plan"]
    require(audit["mechanics_status"] == "valid" and audit["source"] == run["source"]
            and audit["auditor_source_sha256"] == sha(here / "analyze_flux_carriers.py")
            and audit["audit_helpers"][helper.name] == sha(helper), "source-bound independent audit differs")
    require(audit["instrument_status"] in {"native-replay-qualified", "replay-unresolved"}, "unknown instrument qualification")
    value = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    rows = {row["id"]: row for row in value["rows"]}
    order = [row["id"] for row in plan["branches"]]
    require(len(rows) == len(order) == 11 and set(rows) == set(order) == set(audit["comparisons"])
            and audit["counts"]["final_images"] == 11 and audit["counts"]["images"] == 15
            and audit["counts"]["preview_images"] == 4, "eleven-future/fifteen-PNG inventory differs")
    source = read(data / "source.json")

    def artifact(path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run_dir), "artifact escapes run")
        return {"path": str(path.resolve().relative_to(run_dir)), "sha256": sha(path)}

    def verified(record):
        path = data / record["path"]
        binding = artifact(path)
        require(path.resolve().is_relative_to(data) and binding["sha256"] == record["sha256"], "retained array/image digest differs")
        return {**record, **binding}

    def records(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                return verified(value)
            return {key: records(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [records(item) for item in value]
        return value

    def cut(fingerprint):
        path = data / "typed-store/cuts" / (fingerprint + ".json")
        require(read(path)["fingerprint"] == fingerprint, "retained cut identity differs")
        return {"fingerprint": fingerprint, **artifact(path)}

    def png(record):
        bound = verified(record)
        with Image.open(run_dir / bound["path"]) as image:
            require(image.mode == "RGB" and image.size == (512, 512), "native image ABI differs")
            return image.copy(), bound

    def comparison(value):
        require(set(value["scheduler_trajectory"]) == {"3", "4"}, "comparison horizon differs")
        require(all(type(item["base_l2"]) in {float, int} and math.isfinite(item["base_l2"]) and item["base_l2"] >= 0
                    for item in value["scheduler_trajectory"].values()), "invalid saved numerical comparison")
        return value

    row_records, images, labels, time_images, time_labels = [], {}, {}, {}, {}
    for name in order:
        row = rows[name]
        require(read(data / "rows" / (name + ".json")) == row, "durable row differs")
        native = comparison(audit["comparisons"][name])
        rescue = row["rescue_from"]
        matched = comparison(audit["rescue_comparisons"][name]) if rescue else native
        baseline = rescue or "native"
        if rescue:
            require(matched["baseline"] == rescue and row["parent"] == row["shared_prefix"]["cut"]
                    and row["shared_prefix"]["row_id"] == rescue, "rescue matched baseline differs")
        images[name], image = png(row["image"])
        labels[name] = f"Z4 L2={matched['scheduler_trajectory']['4']['base_l2']:.7g} vs {baseline}"
        actions = []
        for action in row["acts"]:
            receipt = data / "typed-store/receipts" / (action["receipt_id"] + ".json")
            require(read(receipt) == action["receipt"], "durable Act receipt differs")
            actions.append({key: action[key] for key in ("slot", "point", "act", "receipt_id", "undeclared_slots_exact")} |
                           {"parent": cut(action["parent"]), "post_act": cut(action["post_act"]), "receipt_file": artifact(receipt)})
        schedulers = [{**item, "cut": cut(item["cut"]), "latent": verified(item["latent"])} for item in row["schedulers"]]
        require(len(schedulers) == 2 and {item["step"] for item in schedulers} == {3, 4}, "scheduler inventory differs")
        projects = []
        for item in row["project_reads"]:
            receipt = data / "typed-store/receipts" / (item["receipt"]["fingerprint"] + ".json")
            require(read(receipt) == item["receipt"], "durable native projection receipt differs")
            projects.append({"clock": item["logical_step"], "pre_cut": cut(item["pre_cut"]), "post_cut": cut(item["post_cut"]),
                "receipt_file": artifact(receipt), "ports": {key: verified(item[key]) for key in ("conditioning", "latent", "text", "image")}})
        previews = {}
        require(set(row["previews"]) == ({"3"} if name in plan["preview_rows"] else set()), "preview inventory differs")
        for clock, item in row["previews"].items():
            key = name + "@Z" + clock
            time_images[key], bound = png(item)
            previews[clock] = {**bound, "cut": cut(item["cut"])}
            require(item["cut"] == next(record["cut"] for record in row["schedulers"] if str(record["step"]) == clock),
                    "preview source scheduler cut differs")
            time_labels[key] = f"live noised latent | L2={native['scheduler_trajectory'][clock]['base_l2']:.7g} vs native"
        if name in plan["preview_rows"]:
            key = name + "@Z4"
            time_images[key] = images[name]
            time_labels[key] = f"final RGB | L2={native['scheduler_trajectory']['4']['base_l2']:.7g} vs native"
        row_records.append({"id": name, "alpha": row["alpha"], "beta": row["beta"], "rescue_from": rescue,
            "parent": cut(row["parent"]), "final_cut": cut(row["final_cut"]), "shared_prefix": row.get("shared_prefix"),
            "row_record": artifact(data / "rows" / (name + ".json")), "recipe": records(row["recipe"]),
            "acts": actions, "project_reads": projects, "schedulers": schedulers, "previews": previews,
            "image": image, "continuation_policy": row["continuation_policy"], "final_conditioning": row["final_conditioning"],
            "expected_conditioning": row["expected_conditioning"], "headline_baseline": baseline,
            "matched_comparisons": matched, "common_native_comparisons": native, "native_path": audit["native_paths"][name]})
    require(len(time_images) == 8 and sum(len(row["previews"]) for row in row_records) == 4, "unique PNG/preview inventory differs")
    parity_path = data / "preview-helper-final-parity.json"
    require(read(parity_path) == value["preview_helper_final_parity"], "VAE helper parity observation differs")
    output.mkdir(parents=True, exist_ok=False)
    figures = {}
    for kind, names, originals, panel_labels, columns, title in (
            ("finals", order, images, labels, 3, "Z2 carried-latent / future-C controls | eleven native futures"),
            ("timegrid", [name + "@Z" + clock for name in plan["preview_rows"] for clock in ("3", "4")],
             time_images, time_labels, 2, "Z3 live previews / Z4 final native RGB | four selected histories")):
        sheet, tiles = compose_panel(names, originals, panel_labels, columns=columns, title=title)
        path = output / ("context-carriers-" + kind + ".png")
        sheet.save(path)
        with Image.open(path) as saved:
            require(all(saved.crop(tiles[key]["image_box"]).tobytes() == image.tobytes() for key, image in originals.items()),
                    "saved panel altered original pixels")
        figures[kind] = {"path": path.name, "sha256": sha(path), "tiles": tiles}
    result = {"schema": "mdb-flux-carriers-publication-v1", "mechanics_status": "valid", "instrument_status": audit["instrument_status"],
        "replay_qualification": audit["replay_qualification"], "terminal_status": "unassessed", "future_count": 11, "original_PNG_count": 15,
        "reference_base": os.path.relpath(run_dir, output), "source": run["source"], "plan": plan, "counts": audit["counts"],
        "audit": {"path": os.path.relpath(audit_path, output), "sha256": sha(audit_path), "auditor_source_sha256": audit["auditor_source_sha256"]},
        "publisher_source_sha256": sha(__file__), "custody_helper_sha256": sha(helper), "rows": row_records, "figures": figures,
        "source_record": artifact(data / "source.json"), "seal": artifact(run["transaction"] / "seal.json"),
        "source_cuts": {key: cut(source[key]) for key in ("initial_cut", "parent_cut")},
        "source_arrays": {key: verified(source[key]) for key in ("initial_latent", "parent_latent", "own_conditioning", "anchor", "residual")},
        "source_selection": {"reference_base": "payload", "pins": source["source_selection"]},
        "preview_helper_final_parity": {"record": value["preview_helper_final_parity"], "file": artifact(parity_path)},
        "control_observations": audit["control_observations"], "sealed_hypotheses": audit["sealed_hypotheses"],
        "visual_inspection": "pending separate componentwise inspection; no aggregate semantic gate or inferred labels",
        "new_model_calls": 0, "model_or_projection_weights_or_suffix_equations_executed": False,
        "pixel_policy": "Original512px RGB bytes pasted unchanged; annotations outside image tiles; no resampling or synthesis.",
        "limitations": ["One observed native prompt/seed history, dependent futures and two shared edited prefixes; no independent trial count.",
            "The sigma-scaled own-noise anchor is derived, not observed noise inside Z2 or proven memory erasure.",
            "C0 is an artificial input; model priors, learned bias, positions, time and surviving state remain.",
            "Actual C/latent to T/image projection reads are native observations, not independent projection-weight evaluation.",
            "Rescues compare against their exact zero-C history as well as native; no alternate baseline prefix is borrowed.",
            "Replay controls qualify the named exact-replay claim; false controls retain all rows and metrics as replay-unresolved.",
            "No numeric rank prediction or semantic-quality gate; componentwise visual hypotheses require separate observations."]}
    save(output / "result.json", result)
    lines = ["# Carried latent and future conditioning", "", "Eleven native futures, four extra live-Z3 previews. Instrument: " + audit["instrument_status"], "",
             "| Row | alpha | beta before rescue | Matched reference | Final latent L2 |", "|---|---:|---:|---|---:|"]
    for row in row_records:
        lines.append(f"| {row['id']} | {row['alpha']} | {row['beta']} | {row['headline_baseline']} | {row['matched_comparisons']['scheduler_trajectory']['4']['base_l2']:.9g} |")
    lines += ["", "Rescue rows use their own actual zero-C Z3 history as matched baseline; common-native metrics remain separate. Componentwise visual findings are pending.", "",
              "![Eleven final native images](context-carriers-finals.png)", "", "![Z3 live previews and Z4 finals](context-carriers-timegrid.png)", ""]
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
    print(json.dumps({key: result[key] for key in ("mechanics_status", "instrument_status", "future_count", "original_PNG_count")}))
