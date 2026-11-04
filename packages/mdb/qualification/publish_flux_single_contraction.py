"""Publish nine native contraction futures with unchanged scientific pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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

    here = Path(__file__).resolve().parent
    run_dir, audit_path, output = [Path(value).resolve() for value in (run_dir, audit_path, output)]
    require(not output.exists(), "publication already exists")
    verified = runpy.run_path(str(here / "flux_family_custody.py"))["verify"](run_dir)
    audit = read(audit_path)
    require(audit["mechanics_status"] == "valid" and all(audit["source"][key] == verified[key]
        for key in ("job_id", "request_sha256", "archive_sha256")), "audit/custody source differs")
    require(audit["auditor_source_sha256"] == sha(here / "analyze_flux_single_contraction.py"), "auditor changed")
    for name, digest in audit["audit_helpers"].items():
        require(Path(name).name == name and sha(here / name) == digest, "audit helper changed")
    data = run_dir / "extracted/flux-single-contraction-result"
    plan, source = read(data / "plan.json"), read(data / "source.json")
    order = [row["id"] for row in plan["branches"]]
    require(len(order) == len(set(order)) == 9 and [row["id"] for row in audit["rows"]] == order, "nine-row order differs")
    images, labels = {}, {}
    for row in audit["rows"]:
        for key in ("image", "final_latent", "row_record"):
            bound = run_dir / row[key]["path"]
            require(bound.is_file() and not bound.is_symlink() and bound.resolve().is_relative_to(run_dir)
                    and sha(bound) == row[key]["sha256"], "row artifact escaped or changed")
        with Image.open(run_dir / row["image"]["path"]) as image:
            require(image.mode == "RGB" and image.size == (512, 512), "native image geometry differs")
            images[row["id"]] = image.copy()
        metrics = row["comparisons"]
        labels[row["id"]] = f"Latent L2 low={metrics['zero']['latent']['base_l2']:.6g}; high={metrics['text']['latent']['base_l2']:.6g}"
    compose = runpy.run_path(str(here / "publish_flux_carriers.py"))["compose_panel"]
    panel, tiles = compose(order, images, labels, columns=3,
        title="Single6 contraction | attention, MLP and residual | nine native futures")
    output.mkdir(parents=True, exist_ok=False)
    figure = output / "single-contraction-finals.png"
    panel.save(figure)
    with Image.open(figure) as saved:
        for name, image in images.items():
            require(saved.crop(tiles[name]["image_box"]).tobytes() == image.tobytes(), "grid altered scientific pixels")

    def artifact(path):
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run_dir), "source artifact escaped")
        return {"path": path.relative_to(run_dir).as_posix(), "sha256": sha(path)}

    result = {**audit, "schema": "mdb-flux-single-contraction-publication-v1", "plan": plan,
        "source_roles": source["source_roles"], "model_identity": source["model_identity"],
        "checkpoint_sha256": source["checkpoint_sha256"], "reference_base": os.path.relpath(run_dir, output),
        "nested_audit_tensor_reference_base": data.relative_to(run_dir).as_posix(),
        "future_count": 9, "original_PNG_count": 9, "new_model_calls": 0,
        "audit": {"path": os.path.relpath(audit_path, output), "sha256": sha(audit_path)},
        "source_record": artifact(data / "source.json"), "producer_manifest": artifact(data / "files.json"),
        "seal": artifact(data / "transaction/seal.json"), "numeric_seal": artifact(data / "projection-transaction/seal.json"),
        "figure": {"path": figure.name, "sha256": sha(figure), "tiles": tiles},
        "publisher_source_sha256": sha(__file__), "panel_helper_sha256": sha(here / "publish_flux_carriers.py"),
        "visual_inspection": "Separate source-bound observations required; metrics do not certify semantics",
        "pixel_policy": "Nine original native 512px images pasted without resampling; annotations outside scientific pixels"}
    save(output / "result.json", result)
    (output / "result.md").write_text("# Single6 contraction\n\nAll nine native futures retained.\n\n![Native outputs](single-contraction-finals.png)\n")
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
