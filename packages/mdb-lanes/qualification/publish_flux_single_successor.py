"""Publish the single1 assay, preserving nine original native RGB files."""
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
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def publish(run_dir, audit_path, output):
    from PIL import Image

    here = Path(__file__).resolve().parent
    run_dir, audit_path, output = (Path(value).resolve() for value in (run_dir, audit_path, output))
    require(not output.exists(), "publication already exists")
    verified = runpy.run_path(str(here / "flux_family_custody.py"))["verify"](run_dir)
    audit = read(audit_path)
    require(audit["mechanics_status"] == "valid" and all(audit["source"][key] == verified[key]
        for key in ("job_id", "request_sha256", "archive_sha256")), "audit/custody binding differs")
    require(audit["auditor_source_sha256"] == sha(here / "analyze_flux_single_successor.py"), "auditor source changed")
    for name, digest in audit["audit_helpers"].items():
        require(Path(name).name == name and sha(here / name) == digest, "audit helper changed")
    data = run_dir / "extracted/flux-single-successor-result"
    plan, source = read(data / "plan.json"), read(data / "source.json")
    order = [row["id"] for row in plan["branches"]]
    require(len(order) == len(set(order)) == 9 and set(audit["comparisons"]) == set(order), "nine-row inventory differs")
    common = runpy.run_path(str(here / "publish_flux_carriers.py"))

    def artifact(path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run_dir), "artifact escapes run")
        return {"path": path.resolve().relative_to(run_dir).as_posix(), "sha256": sha(path)}

    def bind(record):
        path = data / record["path"]
        require(path.resolve().is_relative_to(data), "nested artifact escapes data")
        value = artifact(path)
        require(value["sha256"] == record["sha256"], "artifact digest differs")
        return {**record, **value}

    rows, images, labels = [], {}, {}
    for row_id in order:
        path = data / "rows" / (row_id + ".json")
        row = read(path)
        require(row["id"] == row_id, "durable row ID differs")
        image = bind(row["image"])
        with Image.open(run_dir / image["path"]) as native:
            require(native.mode == "RGB" and native.size == (512, 512), "native image ABI differs")
            images[row_id] = native.copy()
        comparisons = audit["comparisons"][row_id]
        labels[row_id] = f"L2 low={comparisons['zero']['latent']['base_l2']:.6g}; high={comparisons['text']['latent']['base_l2']:.6g}"
        rows.append({"id": row_id, "image": image, "final_latent": bind(row["final_latent"]),
            "row_record": artifact(path), "parent": row["parent"], "final_cut": row["final_cut"],
            "acts": row["acts"], "comparisons": comparisons, "native_path": audit["native_paths"][row_id]})
    output.mkdir(parents=True, exist_ok=False)
    panel, tiles = common["compose_panel"](order, images, labels, columns=3,
        title="Single1 reader | equal input image, different working text | nine native futures")
    figure = output / "single-successor-finals.png"
    panel.save(figure)
    with Image.open(figure) as saved:
        require(all(saved.crop(tiles[name]["image_box"]).tobytes() == original.tobytes()
            for name, original in images.items()), "figure altered native pixels")
    result = {"schema": "mdb-flux-single-successor-publication-v1", "source": audit["source"],
        "mechanics_status": audit["mechanics_status"], "instrument_status": audit["instrument_status"],
        "execution_status": audit["execution_status"], "terminal_status": "unassessed",
        "reference_base": os.path.relpath(run_dir, output), "nested_audit_tensor_reference_base": data.relative_to(run_dir).as_posix(),
        "plan": plan, "counts": audit["counts"], "panel_counts": audit["panel_counts"], "rows": rows,
        "future_count": 9, "original_PNG_count": 9, "source_roles": source["source_roles"],
        "model_identity": source["model_identity"], "checkpoint_sha256": source["checkpoint_sha256"],
        "figure": {"path": figure.name, "sha256": sha(figure), "tiles": tiles},
        "audit": {"path": os.path.relpath(audit_path, output), "sha256": sha(audit_path)},
        "source_record": artifact(data / "source.json"), "producer_manifest": artifact(data / "files.json"),
        "seal": artifact(data / "transaction/seal.json"), "numeric_seal": artifact(data / "attention-transaction/seal.json"),
        "numerical_attention_forecast": audit["numerical_attention_forecast"], "trajectory": audit["trajectory"],
        "text_direction": audit["text_direction"], "candidate_checks": audit["candidate_checks"],
        "control_observations": audit["control_observations"], "image_KV_interactions": audit["image_KV_interactions"],
        "prior_endpoint_controls": audit["prior_endpoint_controls"],
        "publisher_source_sha256": sha(__file__), "panel_helper_sha256": sha(here / "publish_flux_carriers.py"),
        "visual_inspection": "separate source-bound observations required", "new_model_calls": 0,
        "pixel_policy": "All nine original512px native images pasted unchanged; annotation bands outside scientific pixels."}
    save(output / "result.json", result)
    (output / "result.md").write_text("# Single1 successor reader\n\nAll nine native futures retained; numerical distances are not semantic scores.\n\n![Original native images](single-successor-finals.png)\n")
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
