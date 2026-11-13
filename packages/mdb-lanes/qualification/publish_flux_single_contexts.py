"""Publish native images and source-bound numerical evidence.

Bound row/source/image references are relative to the run directory. Nested
numerical-attention evidence is preserved verbatim from its per-context audit;
its tensor paths are relative to ``numerical_attention_reference_base``.
"""
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
    require(audit["schema"] == "mdb-flux-single-contexts-audit-v1"
            and audit["mechanics_status"] == "valid", "fresh-context audit is not mechanically valid")
    require(all(audit["source"][key] == verified[key] for key in
                ("job_id", "request_sha256", "archive_sha256")), "parent audit/custody identity differs")
    require(audit["auditor_source_sha256"] == sha(here / "analyze_flux_single_contexts.py"),
            "auditor source changed after assessment")
    for name, digest in audit["audit_helpers"].items():
        require(Path(name).name == name and sha(here / name) == digest,
                "audit helper source changed after assessment")
    data = run_dir / "extracted" / read(run_dir / "report.json")["archive_root"]
    plan, source = read(data / "plan.json"), read(data / "source.json")
    order = [row["id"] for row in plan["branches"]]
    require(len(order) == 9 and len(set(order)) == 9
            and set(audit["contexts"]) == {case["id"] for case in plan["contexts"]}, "context/row inventory differs")
    common = runpy.run_path(str(here / "publish_flux_carriers.py"))

    def artifact(path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run_dir),
                "artifact escapes run")
        return {"path": path.resolve().relative_to(run_dir).as_posix(), "sha256": sha(path)}

    def bind(base, record):
        path = base / record["path"]
        require(path.resolve().is_relative_to(base), "record escapes context")
        bound = artifact(path)
        require(bound["sha256"] == record["sha256"], "record digest differs")
        return {**record, **bound}

    output.mkdir(parents=True, exist_ok=False)
    published = {}
    for context in plan["contexts"]:
        name = context["id"]
        case = data / "contexts" / name
        assessed = audit["contexts"][name]
        require(set(assessed["comparisons"]) == set(order), "audited row inventory differs")
        rows, images, labels = [], {}, {}
        for row_id in order:
            path = case / "rows" / (row_id + ".json")
            row = read(path)
            require(row["id"] == row_id, "durable row ID differs")
            image = bind(case, row["image"])
            with Image.open(run_dir / image["path"]) as native:
                require(native.mode == "RGB" and native.size == (512, 512), "native image ABI differs")
                images[row_id] = native.copy()
            comparisons = assessed["comparisons"][row_id]
            labels[row_id] = f"L2 zero={comparisons['zero']['latent']['base_l2']:.6g}; text={comparisons['text']['latent']['base_l2']:.6g}"
            rows.append({"id": row_id, "context_id": name, "image": image,
                "final_latent": bind(case, row["final_latent"]), "row_record": artifact(path),
                "parent": row["parent"], "final_cut": row["final_cut"], "comparisons": comparisons,
                "native_path": assessed["native_paths"][row_id], "acts": row["acts"]})
        panel, tiles = common["compose_panel"](order, images, labels, columns=3,
            title=f"{name} | same K / V protocol | nine native futures")
        figure = output / (name + ".png")
        panel.save(figure)
        with Image.open(figure) as saved:
            require(all(saved.crop(tiles[row_id]["image_box"]).tobytes() == native.tobytes()
                        for row_id, native in images.items()), "figure altered native pixels")
        published[name] = {"context": context, "rows": rows,
            "numerical_attention_reference_base": case.relative_to(run_dir).as_posix(),
            "figure": {"path": figure.name, "sha256": sha(figure), "tiles": tiles},
            "source_record": artifact(case / "source.json"), "builder_record": artifact(data / "builders" / name / "builder.json"),
            "seal": artifact(case / "transaction/seal.json"),
            "numeric_seal": artifact(case / "attention-transaction/seal.json"),
            "numerical_attention_forecast": assessed["numerical_attention_forecast"],
            "text_direction": assessed["text_direction"], "candidate_checks": assessed["candidate_checks"],
            "control_observations": assessed["control_observations"],
            "image_KV_interactions": assessed["image_KV_interactions"]}
    require(sum(len(case["rows"]) for case in published.values()) == 27
            and audit["counts"]["images"] == 27, "complete final-image census differs")
    result = {"schema": "mdb-flux-single-contexts-publication-v1", "source": audit["source"],
        "mechanics_status": audit["mechanics_status"], "instrument_status": audit["instrument_status"],
        "execution_status": audit["execution_status"], "terminal_status": "unassessed",
        "reference_base": os.path.relpath(run_dir, output), "plan": plan, "counts": audit["counts"],
        "contexts": published, "future_count": 27, "original_PNG_count": 27,
        "cross_context_summary": audit["cross_context_summary"],
        "model_identity": source["model_identity"], "checkpoint_sha256": source["checkpoint_sha256"],
        "audit": {"path": os.path.relpath(audit_path, output), "sha256": sha(audit_path)},
        "producer_manifest": artifact(data / "files.json"), "source_record": artifact(data / "source.json"),
        "publisher_source_sha256": sha(__file__), "panel_helper_sha256": sha(here / "publish_flux_carriers.py"),
        "visual_inspection": "pending separate source-bound observations", "new_model_calls": 0,
        "pixel_policy": "All27 original512px native images pasted unchanged; annotation bands outside scientific pixels."}
    save(output / "result.json", result)
    lines = ["# Fresh-context single0 replication", "", "All27 native futures retained. Numerical distances are not semantic scores.", ""]
    for name, case in published.items():
        lines.extend([case["context"]["prompt"], "", f"Seed {case['context']['seed']}", "",
                      f"![All nine original native images for {name}]({case['figure']['path']})", ""])
    (output / "result.md").write_text("\n".join(lines))
    (output / "publisher-source.py").write_bytes(Path(__file__).read_bytes())
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path)
        for path in sorted(output.rglob("*")) if path.is_file()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = publish(args.run_dir, args.audit, args.output)
    print(json.dumps({key: result[key] for key in ("mechanics_status", "instrument_status", "future_count")}))
