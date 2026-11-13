"""Publish all single-block K/V observations with unchanged original pixels."""
from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path


def execution_notice(status):
    if status["job_state"] == "failed":
        return ("External scheduler job: failed during REPORT serialization. The unchanged completed scientific outputs "
                "were audited separately; this is an offline assessment, not a rewritten producer-job success.")
    return "External scheduler job: succeeded. Scientific mechanics and interpretation are assessed separately offline."


def publish(run_dir, audit_path, output, *, failed=False):
    from PIL import Image

    here = Path(__file__).resolve().parent
    common = runpy.run_path(str(here / "publish_flux_carriers.py"))
    require, read, sha, save = (common[name] for name in ("require", "read", "sha", "save"))
    run_dir, audit_path, output = (Path(value).resolve() for value in (run_dir, audit_path, output))
    require(not output.exists(), "publication output already exists")
    helper = here / "summarize_context_window.py"
    run = runpy.run_path(str(here / "analyze_flux_single_kv.py"))["source_run"](
        runpy.run_path(str(helper)), run_dir, failed=failed)
    audit, data, plan = read(audit_path), run["data"], run["plan"]
    require(audit["mechanics_status"] == "valid" and audit["source"] == run["source"]
            and audit["auditor_source_sha256"] == sha(here / "analyze_flux_single_kv.py")
            and audit["audit_helpers"][helper.name] == sha(helper)
            and audit["execution_status"] == run["execution_status"], "independent audit source binding differs")
    require(audit["instrument_status"] in {"native-replay-qualified", "replay-unresolved"}, "unknown instrument status")
    observed = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    rows = {row["id"]: row for row in observed["rows"]}
    order = [row["id"] for row in plan["branches"]]
    require(len(rows) == len(order) == 9 and set(rows) == set(order) == set(audit["comparisons"])
            and audit["counts"]["images"] == audit["counts"]["vae_calls"] == 9, "nine future/PNG inventory differs")

    def artifact(path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run_dir), "artifact escapes run")
        return {"path": path.resolve().relative_to(run_dir).as_posix(), "sha256": sha(path)}

    def bind(record):
        result = artifact(data / record["path"])
        require(result["sha256"] == record["sha256"], "saved artifact hash differs")
        return {**record, **result}

    def records(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                return bind(value)
            return {key: records(item) for key, item in value.items()}
        return [records(item) for item in value] if isinstance(value, list) else value

    def cut(value):
        result = artifact(data / "typed-store/cuts" / (value + ".json"))
        require(read(run_dir / result["path"])["fingerprint"] == value, "cut fingerprint differs")
        return {"fingerprint": value, **result}

    images, labels, published = {}, {}, []
    for name in order:
        row = rows[name]
        require(read(data / "rows" / (name + ".json")) == row, "durable row differs")
        image = bind(row["image"])
        with Image.open(run_dir / image["path"]) as value:
            require(value.mode == "RGB" and value.size == (512, 512), "native image ABI differs")
            images[name] = value.copy()
        comparisons = audit["comparisons"][name]
        labels[name] = f"Z4 L2 zero={comparisons['zero']['latent']['base_l2']:.6g}, text={comparisons['text']['latent']['base_l2']:.6g}"
        acts = []
        for action in row["acts"]:
            receipt = artifact(data / "typed-store/receipts" / (action["receipt_id"] + ".json"))
            require(read(run_dir / receipt["path"]) == action["receipt"], "durable Act receipt differs")
            acts.append({key: action[key] for key in ("slot", "point", "act", "receipt_id", "undeclared_slots_exact")} |
                {"parent": cut(action["parent"]), "post_act": cut(action["post_act"]), "receipt_file": receipt})
        published.append({"id": name, "image": image, "previews": {}, "row_record": artifact(data / "rows" / (name + ".json")),
            "parent": cut(row["parent"]), "final_cut": cut(row["final_cut"]), "acts": acts,
            "final_latent": bind(row["final_latent"]), "comparisons": comparisons,
            "native_path": audit["native_paths"][name], "declared_recipe": records({key: value for key, value in row.items()
                if key not in {"acts", "image", "final_latent", "final_cut", "parent"}})})
    output.mkdir(parents=True, exist_ok=False)
    panel, tiles = common["compose_panel"](order, images, labels, columns=3,
        title="Single-block K / V / KV selections | nine native futures")
    figure = output / "single-kv-finals.png"
    panel.save(figure)
    with Image.open(figure) as saved:
        require(all(saved.crop(tiles[name]["image_box"]).tobytes() == image.tobytes() for name, image in images.items()),
                "saved panel changed scientific pixels")
    result = {"schema": "mdb-flux-single-kv-publication-v1", "mechanics_status": "valid",
        "instrument_status": audit["instrument_status"], "replay_qualification": audit["replay_qualification"],
        "execution_status": audit["execution_status"],
        "terminal_status": "unassessed", "reference_base": os.path.relpath(run_dir, output), "source": run["source"],
        "plan": plan, "counts": audit["counts"], "future_count": 9, "original_PNG_count": 9, "rows": published,
        "figures": {"finals": {"path": figure.name, "sha256": sha(figure), "tiles": tiles}},
        "source_record": artifact(data / "source.json"), "prefix_record": artifact(data / "prefixes.json"),
        "candidate_record": artifact(data / "candidates.json"), "seal": artifact(run["transaction"] / "seal.json"),
        "audit": {"path": os.path.relpath(audit_path, output), "sha256": sha(audit_path),
                  "auditor_source_sha256": audit["auditor_source_sha256"]},
        "publisher_source_sha256": sha(__file__), "custody_helper_sha256": sha(helper),
        "panel_helper_sha256": sha(here / "publish_flux_carriers.py"),
        "control_observations": audit["control_observations"], "candidate_checks": audit["candidate_checks"],
        "image_KV_interactions": audit["image_KV_interactions"],
        "text_propagation_differentials": audit["text_propagation_differentials"],
        "numerical_attention_forecast": records(audit["numerical_attention_forecast"]),
        "attention_forecast_record": artifact(data / "attention-forecast.json"),
        "attention_transaction_records": {name: artifact(data / "attention-transaction" / name) for name in
            ("seal.json", "frontiers/source_binding.json", "forecasts/sampled_attention.json", "observations/edited_candidates.json")},
        "paired_source_checks": audit["paired_source_checks"], "sealed_hypotheses": audit["sealed_hypotheses"],
        "visual_inspection": "pending separate source-bound observations; no inferred shape labels",
        "new_model_calls": 0, "models_or_attention_kernels_executed": False,
        "pixel_policy": "Nine original512px RGB files pasted unchanged; separate annotation bands; no resampling or synthesis.",
        "limits": audit["limits"]}
    save(output / "result.json", result)
    lines = ["# Single-block text K/V routes", "", execution_notice(audit["execution_status"]), "",
             "Instrument: " + audit["instrument_status"], "",
             "| Row | Final latent L2 from zero | Final latent L2 from text |", "|---|---:|---:|"]
    for row in published:
        lines.append(f"| {row['id']} | {row['comparisons']['zero']['latent']['base_l2']:.9g} | {row['comparisons']['text']['latent']['base_l2']:.9g} |")
    lines += ["", "All rows retained. Numerical geometry is not a semantic score; visible outcomes require separate assessment.",
              "", "Image K/V interaction is computed directly as KV-K-V+zero on full saved latent/RGB vectors, not by subtracting distances.",
              "Z4 latent interaction L2: " + str(audit["image_KV_interactions"]["Z4_latent"]["l2"]),
              "Final RGB interaction L2: " + str(audit["image_KV_interactions"]["final_RGB"]["l2"]),
              "", "![All nine original native final images](single-kv-finals.png)", ""]
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
    parser.add_argument("--failed", action="store_true", help="Preserve a failed external publication job status")
    args = parser.parse_args()
    result = publish(args.run_dir, args.audit, args.output, failed=args.failed)
    print(json.dumps({key: result[key] for key in ("mechanics_status", "instrument_status", "execution_status", "future_count")}))
