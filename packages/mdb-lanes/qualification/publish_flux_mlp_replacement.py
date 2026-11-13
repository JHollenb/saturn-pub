"""Publish audited MLP replacement evidence and unaltered native image tiles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def publish(run, audit_path, output):
    from PIL import Image, ImageDraw, ImageFont

    audit = read(audit_path)
    request, envelope = read(run / "request.json"), read(run / "report.json")
    if (audit["mechanics_status"] != "valid" or audit["counts"] != envelope["report"]["counts"]
            or audit["source"]["request_sha256"] != request["request_sha256"]
            or audit["source"]["archive_sha256"] != envelope["archive_sha256"]
            or audit["auditor_source_sha256"] != sha(Path(__file__).with_name("analyze_flux_mlp_replacement.py"))):
        raise ValueError("publication audit/request/producer binding differs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    order = [row["id"] for row in request["config"]["branches"]]
    rows = {row["id"]: row for row in audit["rows"]}
    if len(order) != 9 or set(order) != set(rows):
        raise ValueError("publication nine-row inventory differs")
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=13)
    band, margin = 62, 12
    sheet = Image.new("RGB", (3 * 512 + 4 * margin, 3 * (512 + band) + 4 * margin), "#e9edf1")
    draw = ImageDraw.Draw(sheet)
    published = []
    for index, name in enumerate(order):
        row = rows[name]
        original = run / row["image"]["path"]
        if original.is_symlink() or not original.resolve().is_relative_to(run.resolve()) or sha(original) != row["image"]["sha256"]:
            raise ValueError("native image path/hash changed")
        target = output / "images" / (name + ".png")
        target.write_bytes(original.read_bytes())
        with Image.open(original) as image:
            if image.size != (512, 512) or image.mode != "RGB":
                raise ValueError("native image geometry differs")
            x, y = margin + (index % 3) * (512 + margin), margin + (index // 3) * (512 + band + margin)
            sheet.paste(image, (x, y + band))
            if sheet.crop((x, y + band, x + 512, y + band + 512)).tobytes() != image.tobytes():
                raise ValueError("native image tile changed during layout")
        zero = row["comparisons"]["zero"]
        draw.text((x + 5, y + 3), name, fill="#111827", font=font)
        draw.text((x + 5, y + 27), f"from low: latent L2 {zero['latent']['base_l2']:.6g}; RGB MAE {zero['rgb']['base_mae']:.6g}", fill="#374151", font=small)
        published.append({**row, "image": {"path": "images/" + target.name, "sha256": sha(target)},
            "original_image": row["image"], "tile": {"x": x, "y": y + band, "width": 512, "height": 512}})
    sheet.save(output / "native-futures.png")
    result = {"schema": "mdb-flux-mlp-replacement-publication-v1", "audit": {"path": str(audit_path.resolve()), "sha256": sha(audit_path)},
        **{key: audit[key] for key in ("source", "counts", "panel_counts", "seal", "mechanics_status", "instrument_status", "trend_status", "terminal_status",
            "matched_pairs", "error_effect_diagnostics", "entry_samples", "candidate_checks", "control_observations", "limits")},
        "context": request["config"]["context"], "model_identity": envelope["report"]["model_identity"],
        "checkpoint_sha256": request["config"]["expected_checkpoint_sha256"],
        "full_mlp_forecast": audit["full_mlp_forecast"], "rows": published,
        "reference_scopes": {"images": "publication-relative exact copies", "original_image": "run-relative",
            "row_record_and_final_latent": "run-relative", "nested_arrays": audit["nested_audit_tensor_reference_base"]},
        "producer": {"path": Path(__file__).name, "sha256": sha(Path(__file__))},
        "native_image_layout": "Nine unchanged512x512 original RGB tiles; labels occupy separate62-pixel bands. Pixel copies verified; no resize or color change."}
    save(output / "result.json", result)
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    return {"output": str(output), "rows": len(rows), "result_sha256": sha(output / "result.json"), "new_model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir.resolve(), args.audit.resolve(), args.output.resolve())))
