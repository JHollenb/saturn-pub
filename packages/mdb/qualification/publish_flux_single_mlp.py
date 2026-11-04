"""Publish the early MLP forecast with nine unchanged native image futures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def publish(run_dir, audit_path, output):
    from PIL import Image

    from mdb import tensor_archive

    here = Path(__file__).resolve().parent
    run_dir, audit_path, output = [Path(value).resolve() for value in (run_dir, audit_path, output)]
    require(not output.exists(), "MLP publication already exists")
    verified = runpy.run_path(str(here / "flux_family_custody.py"))["verify"](run_dir)
    audit = read(audit_path)
    require(audit["mechanics_status"] == "valid" and all(audit["source"][key] == verified[key]
        for key in ("job_id", "request_sha256", "archive_sha256")), "MLP audit/custody source differs")
    require(audit["auditor_source_sha256"] == sha(here / "analyze_flux_single_mlp.py"), "MLP auditor changed")
    require(audit["tensor_archive_source_sha256"] == sha(tensor_archive.__file__), "MLP audit reader changed")
    for name, digest in audit["audit_helpers"].items():
        require(Path(name).name == name and sha(here / name) == digest, "MLP audit helper changed")
    data = run_dir / "extracted/flux-single-mlp-result"
    plan, source = read(data / "plan.json"), read(data / "source.json")
    panel = audit["panel"]
    order = [row["id"] for row in plan["branches"]]
    require(len(order) == len(set(order)) == 9 and [row["id"] for row in panel["rows"]] == order, "MLP nine-row order differs")
    images, labels = {}, {}
    for row in panel["rows"]:
        for key in ("image", "final_latent", "row_record"):
            path = run_dir / row[key]["path"]
            require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run_dir)
                and sha(path) == row[key]["sha256"], "MLP row escaped or changed")
        with Image.open(run_dir / row["image"]["path"]) as image:
            require(image.mode == "RGB" and image.size == (512, 512), "MLP native image geometry differs")
            images[row["id"]] = image.copy()
        metrics = row["comparisons"]
        labels[row["id"]] = f"Latent L2 low={metrics['zero']['latent']['base_l2']:.6g}; high={metrics['text']['latent']['base_l2']:.6g}"
    compose = runpy.run_path(str(here / "publish_flux_carriers.py"))["compose_panel"]
    figure, tiles = compose(order, images, labels, columns=3,
        title="Fresh seed67 | early MLP forecast and nine native futures")
    output.mkdir(parents=True, exist_ok=False)
    figure_path = output / "single-mlp-finals.png"
    figure.save(figure_path)
    with Image.open(figure_path) as saved:
        for name, image in images.items():
            require(saved.crop(tiles[name]["image_box"]).tobytes() == image.tobytes(), "MLP panel changed native pixels")
    def artifact(path):
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run_dir), "MLP source escaped")
        return {"path": path.relative_to(run_dir).as_posix(), "sha256": sha(path)}
    result = {**audit, "schema": "mdb-flux-single-mlp-publication-v1", "plan": plan,
        "context": source["context"], "model_identity": source["model_identity"], "checkpoint_sha256": source["checkpoint_sha256"],
        "reference_base": os.path.relpath(run_dir, output), "future_count": 9, "original_PNG_count": 9, "new_model_calls": 0,
        "audit": {"path": os.path.relpath(audit_path, output), "sha256": sha(audit_path)},
        "source_record": artifact(data / "source.json"), "producer_manifest": artifact(data / "files.json"),
        "entry_seal": artifact(data / "mlp-entry-transaction/seal.json"),
        "structural_seal": artifact(data / "panel/transaction/seal.json"),
        "projection_seal": artifact(data / "panel/projection-transaction/seal.json"),
        "figure": {"path": figure_path.name, "sha256": sha(figure_path), "tiles": tiles},
        "publisher_source_sha256": sha(__file__), "panel_helper_sha256": sha(here / "publish_flux_carriers.py"),
        "pixel_policy": "Nine original native 512px images pasted without resampling; labels outside scientific pixels",
        "visual_inspection": "Separate arm-blind source-bound observations; numerical distance does not certify semantics"}
    save(output / "result.json", result)
    (output / "result.md").write_text("# Predicting MLP features before observing them\n\nOne fresh seed; all nine native futures retained.\n\n![Native outputs](single-mlp-finals.png)\n")
    (output / "publisher-source.py").write_bytes(Path(__file__).read_bytes())
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = publish(args.run_dir, args.audit, args.output)
    print(json.dumps({key: result[key] for key in ("mechanics_status", "instrument_status", "future_count")}))
