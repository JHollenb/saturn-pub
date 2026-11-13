"""Publish all audited chain futures with unchanged native image tiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

    audit, request = read(audit_path), read(run / "request.json")
    if (run / "recovery.json").exists():
        recovered = read(run / "recovery.json")
        summary = read(run / audit["nested_audit_tensor_reference_base"] / "summary.json")
        custody_matches = (audit["source"]["collection_mode"] == "authenticated-ssh-manifest-recovery"
            and audit["source"]["recovery_sha256"] == sha(run / "recovery.json")
            and audit["source"]["archive_sha256"] is None and recovered["archive_sha256"] is None
            and audit["execution_status"]["job_state"] == recovered["scheduler_status"] == "failed"
            and audit["source"]["request_sha256"] == recovered["request_sha256"])
    else:
        envelope = read(run / "report.json")
        summary = envelope["report"]
        custody_matches = audit["source"]["archive_sha256"] == envelope["archive_sha256"]
    if (audit["mechanics_status"] != "valid" or audit["counts"] != summary["counts"] or not custody_matches
            or audit["source"]["request_sha256"] != request["request_sha256"]
            or audit["auditor_source_sha256"] != sha(Path(__file__).with_name("analyze_flux_transition_chain.py"))):
        raise ValueError("chain publication audit/request/source binding differs")
    order = [row["id"] for row in request["config"]["branches"]]
    rows = {row["id"]: row for row in audit["rows"]}
    indices = request["config"]["block_indices"]
    expected = 7 if indices == [6, 7] else 9 if indices == [6, 7, 8] else None
    if expected is None or len(order) != expected or len(rows) != expected or set(order) != set(rows):
        raise ValueError("chain publication must contain all declared native futures")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    columns, margin, band = (4 if expected == 7 else 3), 12, 66
    sheet = Image.new("RGB", (columns * (512 + margin) + margin,
        math.ceil(len(order) / columns) * (512 + band + margin) + margin), "#e9edf1")
    draw = ImageDraw.Draw(sheet)
    font, small = ImageFont.load_default(size=17), ImageFont.load_default(size=13)
    exported = []
    for index, name in enumerate(order):
        row = rows[name]
        original = run / row["image"]["path"]
        if original.is_symlink() or not original.resolve().is_relative_to(run.resolve()) or sha(original) != row["image"]["sha256"]:
            raise ValueError("chain native image path/hash differs")
        target = output / "images" / (name + ".png")
        target.write_bytes(original.read_bytes())
        x, y = margin + (index % columns) * (512 + margin), margin + (index // columns) * (512 + band + margin)
        with Image.open(original) as image:
            if image.size != (512, 512) or image.mode != "RGB":
                raise ValueError("native chain image geometry/mode differs")
            sheet.paste(image, (x, y + band))
            if sheet.crop((x, y + band, x + 512, y + band + 512)).tobytes() != image.tobytes():
                raise ValueError("native chain image changed during layout")
        comparison = audit["comparisons"][name]
        draw.text((x + 5, y + 3), name, fill="#111827", font=font)
        draw.text((x + 5, y + 27), "vs " + comparison["matched_reference"], fill="#374151", font=small)
        draw.text((x + 5, y + 45), f"latent L2 {comparison['latent']['base_l2']:.6g}; RGB MAE {comparison['RGB']['base_mae']:.6g}", fill="#374151", font=small)
        exported.append({**row, "image": {"path": "images/" + target.name, "sha256": sha(target)},
            "original_image": row["image"], "tile": {"x": x, "y": y + band, "width": 512, "height": 512}})
    sheet.save(output / "native-futures.png")
    result = {"schema": "mdb-flux-transition-chain-publication-v1", "audit": {"path": str(audit_path), "sha256": sha(audit_path)},
        **{key: audit[key] for key in ("source", "counts", "panel_counts", "seal", "mechanics_status", "instrument_status", "terminal_status",
            "model_identity", "checkpoint_sha256", "full_chain_forecast", "control_observations", "comparisons",
            "continuation_trajectory", "timing", "physical_census", "execution_status", "limits")},
        "context": request["config"]["context"], "rows": exported,
        "nested_audit_tensor_reference_base": audit["nested_audit_tensor_reference_base"],
        "producer": {"path": Path(__file__).name, "sha256": sha(Path(__file__))},
        "native_image_layout": f"All {expected} unchanged512x512 RGB tiles, separately annotated66-pixel bands. Unused cells stay blank. No resizing or color modification."}
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
