"""Planning-only recipe/source selection from authenticated local JSON metadata.

No checkpoint, tensor array, model, scheduler or remote source is opened.
This does not freeze a scientific protocol or authorize execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "mdb-qwen-row-sign-20261106/compact-collection-01/extracted/binding-minimal-result"
MANIFEST_SHA256 = "d5060dd8137bf34568d3d9b0fda805c5979133b0c4a4538e23579196cb5d3374"
RAW_MANIFEST_SHA256 = "d36f790376a85ad995cf20f01cff89fb6e04d282fe1290f677ba348ec98f4c8f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    if Path(path).suffix != ".json":
        raise ValueError("metadata JSON only")
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")


def select(parent, output):
    manifest_path = parent / "source-metadata/parent_parameter_manifest.json"
    if sha(manifest_path) != MANIFEST_SHA256 or sha(parent / "source-files.json") != RAW_MANIFEST_SHA256:
        raise ValueError("authenticated original metadata differs")
    manifest, files = read(manifest_path), read(parent / "source-files.json")
    if manifest["schema"] != "mdb-parameter-artifacts-v1" or len(manifest["artifacts"]) != 398:
        raise ValueError("complete original Qwen manifest required")
    rows = [row for row in manifest["artifacts"] if row["source"]["tensor"].startswith("model.layers.19.")]
    payload_bytes = sum(row["source"]["data_offsets"][1]-row["source"]["data_offsets"][0] for row in rows)
    if len(rows) != 11 or len({row["source"]["tensor"] for row in rows}) != 11 or payload_bytes != 201861632 or any(row["source"]["dtype"] != "BF16" for row in rows):
        raise ValueError("exact original block19 recipe set differs")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "qwen-block19-parameter-artifacts.json"
    write(target, dict(schema=manifest["schema"], defaults=manifest["defaults"], artifacts=rows,
        provenance=dict(status="planning-only metadata subset; science not frozen", parent_manifest_sha256=MANIFEST_SHA256,
            selected_parameter_prefix="model.layers.19.", source_parameter_bytes_read=0, array_bytes_read=0)))
    forecast, shortcut, summary, frontier = (read(parent / name) for name in ("source-forecast.json", "source-shortcut.json", "source-summary.json", "source-source-frontier.json"))
    worlds = {"full-input": forecast["arms"]["prefix-negative"]["ports"]["Hnext"], "cheap-input": shortcut["arms"]["prefix-negative"]["ports"]["Hnext"]}
    for record in worlds.values():
        if files["files"][record["path"]] != dict(bytes=record["bytes"], sha256=record["sha256"]):
            raise ValueError("world descriptor not in authenticated turn14 inventory")
    selected_files = {}
    for row in rows:
        source = row["source"]
        signature = next(value for name, value in manifest["provenance"]["source_signatures"].items() if Path(name).name == source["file"])
        if signature["before"] != signature["after"]:
            raise ValueError("original metadata does not show unchanged checkpoint source")
        selected_files[source["file"]] = dict(bytes=signature["before"][2], sha256=source["file_sha256"], header_bytes=source["header_bytes"], header_sha256=source["header_sha256"])
    context = {name: record for name, record in frontier["source_arrays"].items() if name in {"input_ids", "attention_mask", "position_ids"} or name.startswith("rope_buffer:")}
    write(output / "selection-note.json", dict(status="design proposal only; requires root direction before scientific freeze", selector_sha256=sha(__file__),
        selected_manifest_sha256=sha(target), original_manifest_sha256=MANIFEST_SHA256, selected_parameter_count=len(rows), parameter_payload_bytes=payload_bytes,
        virtual_original_NPY_archive_bytes=sum(row["bytes"] for row in rows), checkpoint_files=selected_files,
        checkpoint_wholefile_bytes_to_verify=sum(row["bytes"] for row in selected_files.values()), world_sources=worlds, shared_context_sources=context,
        selected_array_archive_bytes=sum(row["bytes"] for row in worlds.values())+sum(row["bytes"] for row in context.values()),
        turn14=dict(job_id=summary["job_id"], request_sha256=summary["request_sha256"], raw_manifest_sha256=RAW_MANIFEST_SHA256, forecast_seal_sha256=summary["forecast_seal"]),
        raw_array_bytes_read=0, parameter_bytes_read=0, model_calls=0, scheduler_calls=0))
    print(json.dumps(dict(selected_manifest_sha256=sha(target), selection_note_sha256=sha(output / "selection-note.json"), parameter_payload_bytes=payload_bytes, checkpoint_files=selected_files), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, default=HERE / "planning-v1")
    args = parser.parse_args(); select(args.parent, args.output)
