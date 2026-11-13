"""Audit four sealed image futures using retained evidence, with no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def audit(run_dir):
    q = Path(__file__).parent
    old = runpy.run_path(str(q / "analyze_flux_trajectory_forecast.py"))
    h, typed = old["helpers"]()
    require = old["require"]
    run = h["source_run"](run_dir, "flux-branch-forecast-result", max_archive_bytes=256 << 20, max_member_bytes=64 << 20)
    data, tx, tree = run["data"], run["transaction"], run["tree"]
    source, summary = (h["read"](data / name) for name in ("source.json", "summary.json"))
    actual = tree.decode(h["read"](tx / "observations/native.json")["value"])
    config = run["plan"]
    require(source["request"] == run["request"] and config == run["request"]["config"], "source/plan differs")
    require(h["sha"](h["canonical"](source["checkpoint_files"])) == config["expected_checkpoint_sha256"], "checkpoint differs")
    weights = h["read"](data / "weights-before-prediction.json")
    for stage in ("at-release", "after-native"):
        require(h["read"](data / ("weights-" + stage + ".json")) == weights, "weights changed")
    declarations = h["read"](tx / "declarations.json")
    require(declarations["input_identity"] == source["parents"] and declarations["source_identity"] == run["request"]["source_hashes"]
            and declarations["model_identity"] == source["model_identity"], "transaction identity differs")
    stored = typed["audit_store"](data / "typed-store")
    carrier_sources = h["read"](data / "carrier-sources.json")
    for name in config["branch_ids"]:
        require(typed["cut_id"](carrier_sources[name]) == config["input_cuts"][name], "source carrier manifest differs")
        if name == "base":
            require(carrier_sources[name] == stored["cuts"][config["input_cuts"][name]], "base source cut differs")
        else:
            record_path = Path(run_dir) / "payload/carriers" / (name + ".json")
            require(h["sha"](record_path.read_bytes()) == run["request"]["source_hashes"]["carriers/" + name + ".json"], "staged carrier source differs")
            record = h["read"](record_path)
            require(record["cut"]["manifest"] == carrier_sources[name] and h["sha"](h["canonical"](record["page"])) == record["cut"]["slots"]["text"]
                    and record["page"]["value"] == {r["key"]["value"]: r["value"] for r in carrier_sources[name]["payload"]["items"]}["text"], "source carrier page join differs")
    timeline = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines()]
    require([r["sequence"] for r in timeline] == list(range(len(timeline))), "timeline sequence differs")
    sealed = [r for r in timeline if r["kind"] == "all_forecasts_sealed"]
    require(len(sealed) == 1 and sealed[0]["seal"] == run["transaction_report"]["seal_sha256"], "global seal differs")
    require([r["branch"] for r in timeline if r["kind"] == "forecast_complete"] == config["branch_ids"], "forecast inventory differs")
    require([r["branch"] for r in timeline if r["kind"] == "native_branch_start"] == config["branch_ids"], "native inventory differs")
    require(all(r["sequence"] < sealed[0]["sequence"] for r in timeline if r["kind"] == "forecast_complete")
            and all(r["sequence"] > sealed[0]["sequence"] for r in timeline if r["kind"] == "native_branch_start"), "RGB observation preceded global seal")
    require(all(sealed[0]["counts"][n] == 0 for n in ("prediction_module_entries", "native_module_entries", "selected_block_entries", "vae_decoder_entries", "native_transitions")), "suffix entered before global seal")
    events = old["expected_events"](4, 5, 20)
    reports, initial = {}, {}

    def fields(manifest):
        return {row["key"]["value"]: row["value"] for row in manifest["payload"]["items"]}

    for name in config["branch_ids"]:
        first = tree.decode(h["read"](tx / f"frontiers/{name}.json")["value"])
        manifest = first["manifest"]
        initial[name] = manifest
        require(typed["cut_id"](manifest) == source["parents"][name]
                and stored["cuts"][source["parents"][name]] == manifest, "typed branch frontier differs")
        old["join_packed_state"](first["state"], manifest["payload"])
        require(first["weights_digest"] == h["sha"](h["canonical"](weights)), "frontier weights differ")
        require(manifest["model_identity"] == source["model_identity"]["adapter"] and manifest["execution"] == source["execution"], "parent contract differs")
        base_fields = fields(initial["base"])
        branch_fields = fields(manifest)
        require({k: v for k, v in branch_fields.items() if k != "text"} == {k: v for k, v in base_fields.items() if k != "text"}, "branch changed undeclared state")
        supplied = carrier_sources[name]
        require(branch_fields["text"] == fields(supplied)["text"], "branch text differs from declared observed source")
        require(base_fields["latent"] != fields(supplied)["latent"], "latent was not fresh")
        if name != "base":
            receipt = source["acts"][name]["receipt"]
            h["public_receipt"](receipt)
            require(receipt["parent"] == source["base_parent"] and receipt["result"] == source["parents"][name], "Act lineage differs")
        prediction = tree.decode(h["read"](tx / f"forecasts/{name}.json")["predictions"])
        observed = actual["branches"][name]
        require(prediction["own_state"] and prediction["native_refreshes"] == 0, "prediction authority differs")
        require(len(prediction["boundaries"]) == len(observed["boundaries"]) == len(events), "truncated branch")
        rows, previous = [], source["parents"][name]
        for event, p, a in zip(events, prediction["boundaries"], observed["boundaries"], strict=True):
            require(p["event"] == event and p["tag"] == a["tag"] == old["event_tag"](event)
                    and set(p["arrays"]) == set(a["arrays"]) == old["expected_arrays"](event["phase"]), "boundary order differs")
            receipt = a["receipt"]
            h["public_receipt"](receipt)
            require(h["read"](data / "typed-store/receipts" / (receipt["fingerprint"] + ".json")) == receipt
                    and receipt["parent"] == previous and receipt["operation"] == "continue" and receipt["steps"] == 1
                    and receipt["model_identity"] == source["model_identity"]["adapter"] and receipt["execution"] == source["execution"], "native receipt chain differs")
            require(len(receipt["transitions"]) == 1, "native transition count differs")
            transition = receipt["transitions"][0]
            require(old["_point_matches"](transition["before"], event) and old["_point_matches"](transition["after"], event, after=True), "native clock differs")
            previous = receipt["result"]
            rows.append({"tag": p["tag"], "arrays": {key: old["inspect_saved_array"](p["arrays"][key], a["arrays"][key], config["witness_values"]) for key in p["arrays"]}})
        require(set(prediction["vae"]) == set(observed["vae"]) == {"decoder_input", "decoded"}, "VAE inventory differs")
        vae = {key: old["inspect_saved_array"](prediction["vae"][key], observed["vae"][key], config["witness_values"]) for key in prediction["vae"]}
        pixel_error = old["compare"](observed["pixels"], prediction["pixels"])
        require(old["metrics_match"](observed["pixel_error"], pixel_error)
                and old["metrics_match"](summary["pixel_errors"][name], pixel_error), "saved pixel metrics differ")
        images = {}
        for kind, values in (("forecast", prediction), ("native", observed)):
            image = data / (name + "." + kind + ".png")
            pixels = values["pixels"]
            require(pixels.dtype == np.uint8 and pixels.shape == (512, 512, 3), "pixel ABI differs")
            require(np.array_equal(np.asarray(Image.open(image)), pixels) and h["sha"](image.read_bytes()) == values[kind + "_png_sha256"], "PNG differs from retained pixels")
            require(np.array_equal(pixels, old["pixels_from_decoded"](values["vae"]["decoded"]["full_values"], values["vae"]["decoded"]["descriptor"]["dtype"])), "pixels differ from decoder/cast policy")
            images[kind] = {"path": str(image.relative_to(Path(run_dir).resolve())), "sha256": h["sha"](image.read_bytes())}
        reports[name] = {"boundaries": rows, "vae": vae, "pixel_error": pixel_error, "images": images}
    counts = summary["counts"]
    require(counts == actual["counts"] and counts["predicted_transitions"] == counts["native_transitions"] == 444
            and counts["selected_block_entries"] == 400 and counts["vae_decoder_entries"] == 4
            and counts["prompt_encodings"] == 0 and counts["prefix_projections"] == 1 and counts["acts"] == 3
            and counts["images"] == 8, "work accounting differs")
    return {"schema": "mdb-flux-branch-forecast-audit-v1", "mechanics_status": "valid", "terminal_status": "unassessed",
        "source": run["source"], "model_identity": source["model_identity"], "execution": source["execution"], "counts": counts,
        "seed": config["seed"], "branch_ids": config["branch_ids"], "branches": reports, "pre_release_counters": sealed[0]["counts"],
        "typed_store": stored["counts"], "own_state": True, "native_refreshes": 0, "RGB_forecast": True,
        "new_model_calls": 0, "equation_module_evaluations": 0, "scope": config["scope"], "limits": config["limits"],
        "audit_sources": {name: h["sha"]((q / name).read_bytes()) for name in
            ("analyze_flux_branch_forecasts.py", "analyze_flux_trajectory_forecast.py", "summarize_context_window.py", "analyze_flux_mlp_forecast.py")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"mechanics_status": result["mechanics_status"], "pixel_errors": {k: v["pixel_error"] for k, v in result["branches"].items()},
                      "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "new_model_calls": 0}))
