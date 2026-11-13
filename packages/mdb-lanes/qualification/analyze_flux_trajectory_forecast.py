"""Audit saved own-state Klein trajectory and RGB forecasts without execution.

Full retained endpoints and deterministic intermediate witnesses are recomputed.
Other full-array metrics came from immutable worker RAM and are identified as
such. Native/forecast differences are observations, not scientific hard gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import runpy
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def helpers():
    directory = Path(__file__).parent
    return (runpy.run_path(str(directory / "summarize_context_window.py")),
            runpy.run_path(str(directory / "analyze_flux_mlp_forecast.py")))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    require(actual.shape == predicted.shape and actual.size > 0 and np.isfinite(actual).all()
            and np.isfinite(predicted).all(), "incompatible/nonfinite saved arrays")
    residual = actual - predicted
    norm, native_norm = float(np.linalg.norm(residual.ravel())), float(np.linalg.norm(actual.ravel()))
    index = np.unravel_index(int(np.argmax(np.abs(residual))), residual.shape)
    return {"shape": list(actual.shape), "elements": int(actual.size), "changed_elements": int(np.count_nonzero(residual)),
        "exact": bool(np.array_equal(actual, predicted)), "max_abs": float(np.abs(residual).max()),
        "mean_abs": float(np.abs(residual).mean()), "l2": norm, "native_l2": native_norm,
        "relative_l2": norm / native_norm if native_norm else None, "signed_mean": float(residual.mean()),
        "largest_effect": {"coordinate": [int(item) for item in index], "native_minus_forecast": float(residual[index])}}


def metrics_match(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(metrics_match(actual[key], item) for key, item in expected.items())
    if type(expected) is float:
        return type(actual) in {int, float} and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14)
    return type(actual) is type(expected) and actual == expected


def expected_events(steps, joint_layers, single_layers):
    """Independent finite schedule, starting after the observed step-zero project."""
    require(all(type(item) is int and item > 0 for item in (steps, joint_layers, single_layers)), "invalid schedule geometry")
    points = []
    for step in range(steps):
        if step:
            points.append((step, "project", 0))
        points.extend((step, "joint", index) for index in range(joint_layers))
        points.extend((step, "single", index) for index in range(single_layers))
        points.extend(((step, "readout", single_layers), (step, "scheduler", single_layers)))
    afters = points[1:] + [(steps, "project", 0)]
    return [{"step": step, "phase": phase, "index": index, "next_step": after[0], "next_phase": after[1], "next_index": after[2]}
            for (step, phase, index), after in zip(points, afters, strict=True)]


def event_tag(event):
    return f"step{event['step']}.{event['phase']}.{event['index']}"


def expected_arrays(phase):
    if phase == "project":
        return {"image", "text", "context.temb", "context.img_mod", "context.txt_mod", "context.single_mod"}
    return {"image", "text"} if phase in {"joint", "single"} else {"noise_prediction"} if phase == "readout" else {"latent"}


def join_packed_state(packed, described):
    """Join widened serialized tensors to typed native raw-byte descriptors."""
    _, typed = helpers()
    kind = described["kind"]
    if kind == "tensor":
        require(isinstance(packed, dict) and set(packed) == {"array", "tensor_dtype"}
                and packed["tensor_dtype"] == described["dtype"], "initial tensor dtype binding differs")
        require(typed["matches"](packed["array"], described, bf16=described["dtype"] == "torch.bfloat16"), "initial typed tensor bytes differ")
    elif kind == "mapping":
        items = {item["key"]["value"]: item["value"] for item in described["items"]}
        require(isinstance(packed, dict) and set(packed) == set(items), "initial mapping inventory differs")
        for name, item in items.items():
            join_packed_state(packed[name], item)
    elif kind in {"tuple", "list"}:
        require(isinstance(packed, (tuple, list)) and len(packed) == len(described["items"]), "initial sequence differs")
        for value, item in zip(packed, described["items"], strict=True):
            join_packed_state(value, item)
    else:
        require(type(packed) is type(described["value"]) and packed == described["value"], "initial typed scalar differs")


def _validate_worker_metric(metric, shape):
    require(metric["shape"] == shape and metric["elements"] == math.prod(shape)
            and type(metric["changed_elements"]) is int and 0 <= metric["changed_elements"] <= metric["elements"], "worker metric geometry/count differs")
    for name in ("max_abs", "mean_abs", "l2", "native_l2"):
        require(type(metric[name]) in {int, float} and math.isfinite(metric[name]) and metric[name] >= 0, "invalid worker metric")
    require(type(metric["exact"]) is bool and metric["exact"] == (metric["changed_elements"] == 0), "worker exact/count metadata differs")
    require(not metric["exact"] or metric["max_abs"] == metric["l2"] == metric["mean_abs"] == 0, "worker exact/error metadata differs")


def inspect_saved_array(predicted, actual, witness_values):
    """Recompute retained evidence; never use a residual threshold for validity."""
    _, typed = helpers()
    descriptor = predicted["descriptor"]
    shape = descriptor["shape"]
    require(shape == actual["descriptor"]["shape"] and descriptor["dtype"] == actual["descriptor"]["dtype"], "boundary array geometry/dtype differs")
    size = math.prod(shape)
    require(size > 0 and type(witness_values) is int and witness_values > 0, "invalid witness budget")
    positions = np.unique(np.linspace(0, size - 1, min(size, witness_values), dtype=np.int64))
    for row in (predicted, actual):
        require(row["witness_indices"].dtype == np.int64 and np.array_equal(row["witness_indices"], positions)
                and row["witness_values"].shape == positions.shape and row["witness_values"].dtype == np.float32,
                "deterministic witness positions/value ABI differs")
    _validate_worker_metric(actual["full_error"], shape)
    report = {"descriptor": descriptor, "native_descriptor": actual["descriptor"],
        "witness": compare(actual["witness_values"], predicted["witness_values"]),
        "worker_full_error": actual["full_error"],
        "native_forecast_raw_descriptor_sha_exact": actual["descriptor"]["sha256"] == descriptor["sha256"]}
    require(actual["full_error"]["max_abs"] + 1e-14 >= report["witness"]["max_abs"]
            and (not actual["full_error"]["exact"] or report["witness"]["exact"]), "witness contradicts full-array worker metric")
    if descriptor["dtype"] == "torch.bfloat16":
        for row in (predicted, actual):
            typed["descriptor"](row["witness_values"], bf16=True)
    full_retained = predicted["full_retained"]
    require(type(full_retained) is bool and ("full_values" in predicted) == full_retained
            and ("full_values" in actual) == full_retained, "full-array retention metadata differs")
    if full_retained:
        for row in (predicted, actual):
            values = row["full_values"]
            require(values.dtype == np.float32 and typed["matches"](values, row["descriptor"], bf16=descriptor["dtype"] == "torch.bfloat16")
                    and np.array_equal(values.ravel()[positions], row["witness_values"]), "retained array descriptor/witness join differs")
        require(hashlib.sha256(predicted["full_values"].tobytes()).hexdigest() == predicted["float_array_sha256"], "sealed full FP32-expanded bytes differ")
        recomputed = compare(actual["full_values"], predicted["full_values"])
        require(metrics_match(actual["full_error"], recomputed), "worker aggregate differs from retained full arrays")
        report.update(full_recomputed=recomputed, offline_scope="full arrays and witnesses")
    else:
        report["offline_scope"] = "witness values recomputed; full-array metrics are worker RAM observations, not offline recomputation"
    return report


def pixels_from_decoded(values, dtype):
    """Offline cast/postprocessing check; no VAE or predictor execution."""
    _, typed = helpers()
    require(values.dtype == np.float32 and values.ndim == 4 and values.shape[:2] == (1, 3), "decoded pixel geometry/dtype differs")
    require(dtype in {"torch.float32", "torch.bfloat16"}, "unsupported native pixel cast")
    first = values * np.float32(.5)
    if dtype == "torch.bfloat16":
        first = typed["quantize_bf16"](first)
    normalized = first + np.float32(.5)
    if dtype == "torch.bfloat16":
        normalized = typed["quantize_bf16"](normalized)
    return (np.clip(normalized, 0, 1).transpose(0, 2, 3, 1)[0] * np.float32(255)).round().astype(np.uint8)


def _point_matches(point, event, *, after=False):
    step, phase, index = ((event["next_step"], event["next_phase"], event["next_index"]) if after
                          else (event["step"], event["phase"], event["index"]))
    return (point["logical_step"], point["phase"], point["local_clock"], point["index"], point["edge"]) == (
        step, phase, index, index if phase in {"joint", "single"} else None, "before")


def recovered_summary(run, source, predicted, actual):
    """Derive a separate report from a verified failed-publication archive."""
    h, _ = helpers()
    failure = h["read"](run["data"] / "failure.json")
    require(failure["job_id"] == run["source"]["job_id"] and failure["request_sha256"] == run["request"]["request_sha256"]
            and failure["error"] == "TypeError: Object of type mappingproxy is not JSON serializable", "failure is not the declared post-execution publication failure")
    plan = run["plan"]
    return {"schema": "mdb-flux-trajectory-offline-recovered-summary-v1", "job_id": failure["job_id"],
        "request_sha256": failure["request_sha256"], "scheduler_status": "failed", "publication_status": "failed-summary-serialization",
        "execution_status": "completed-observation-retained; subject to independent audit", "original_failure": failure,
        "counts": actual["counts"], "pixel_error": actual["pixel_error"], "model_identity": source["model_identity"],
        "arithmetic": source["arithmetic"], "limits": plan["limits"], "scope": plan["scope"],
        "own_state": predicted["own_state"], "native_refreshes": predicted["native_refreshes"], "RGB_forecast": predicted["RGB_forecast"],
        "worker_seconds": None, "timing_scope": "worker summary was not published; no completion time inferred",
        "authority": "offline derivation from archive-verified transaction observations; raw failure and partial summary unchanged",
        "new_model_calls": 0, "equation_module_evaluations": 0}


def audit(run_dir, *, failed=False):
    from PIL import Image

    h, typed = helpers()
    run = h["source_run"](run_dir, "flux-trajectory-result", failed=failed, recover_metadata=failed,
                          max_archive_bytes=128 << 20, max_member_bytes=64 << 20)
    data, tx, tree = run["data"], run["transaction"], run["tree"]
    source = h["read"](data / "source.json")
    predicted = tree.decode(h["read"](tx / "forecasts/trajectory.json")["predictions"])
    actual = tree.decode(h["read"](tx / "observations/native.json")["value"])
    summary = recovered_summary(run, source, predicted, actual) if failed else h["read"](data / "summary.json")
    plan = run["plan"]
    require(source["request"] == run["request"] and plan == run["request"]["config"], "source/plan request differs")
    require(h["sha"](h["canonical"](source["checkpoint_files"])) == plan["expected_checkpoint_sha256"]
            == source["model_identity"]["checkpoint_sha256"], "checkpoint inventory differs")
    require(source["arithmetic"] == summary["arithmetic"] == plan["arithmetic"] and source["observed_scope"] == plan["observed"]
            and source["RAM_only_intermediates"], "observed/shared-kernel/retention scope differs")
    weights = h["read"](data / "weights-before-prediction.json")
    require(weights and all(name.startswith(("transformer.", "vae.")) and len(value["sha256"]) == 64 for name, value in weights.items()), "weight/buffer inventory differs")
    for stage in ("at-release", "after-native"):
        require(h["read"](data / ("weights-" + stage + ".json")) == weights, "native parameter/buffer bytes changed: " + stage)
    declarations = h["read"](tx / "declarations.json")
    initial = tree.decode(h["read"](tx / "frontiers/initial.json")["value"])
    parent = initial["manifest"]
    require(typed["cut_id"](parent) == declarations["input_identity"] == source["parent_fingerprint"]
            and declarations["model_identity"] == source["model_identity"] and declarations["source_identity"] == run["request"]["source_hashes"]
            and tree.decode(declarations["actual_weight_manifest"]) == {}
            and initial["weights_digest"] == h["sha"](h["canonical"](weights)), "transaction initial/weight/source binding differs")
    stored = typed["audit_store"](data / "typed-store")
    require(stored["cuts"].get(source["parent_fingerprint"]) == parent, "typed initial parent differs from CAS")
    join_packed_state(initial["state"], parent["payload"])
    require(parent["model_identity"] == source["model_identity"]["adapter"] and parent["execution"] == source["execution"], "typed parent model/execution differs")
    state = initial["state"]
    require(state["step"] == state["cursor"] == 0 and state["phase"] == "joint", "initial parent is not before step-zero joint0")
    timeline = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines()]
    require([row["sequence"] for row in timeline] == list(range(len(timeline))), "worker timeline is not contiguous")
    captured = [row for row in timeline if row["kind"] == "frontier_captured"]
    sealed = [row for row in timeline if row["kind"] == "trajectory_sealed"]
    require(len(captured) == len(sealed) == 1 and captured[0]["sequence"] < sealed[0]["sequence"]
            and sealed[0]["seal"] == run["transaction_report"]["seal_sha256"]
            and captured[0]["parent"] == source["parent_fingerprint"], "frontier/seal phase binding differs")
    for counts in (initial["native_entry_counts"], captured[0]["counts"], sealed[0]["counts"]):
        require(all(counts[name] == 0 for name in ("prediction_module_entries", "native_module_entries", "selected_block_entries", "vae_decoder_entries", "native_transitions")), "native suffix module entered before global seal")
    require(all(row["sequence"] < sealed[0]["sequence"] for row in timeline if row["kind"] == "predicted_scheduler")
            and all(row["sequence"] > sealed[0]["sequence"] for row in timeline if row["kind"] == "native_scheduler"), "scheduler observations cross global forecast seal")
    require(predicted["own_state"] and predicted["native_refreshes"] == 0 and predicted["RGB_forecast"]
            and summary["own_state"] and summary["native_refreshes"] == 0, "forecast authority differs")

    def layer_count(kind):
        indices = {int(match[1]) for name in weights if (match := re.match(r"transformer\." + kind + r"\.(\d+)\.", name))}
        require(indices == set(range(len(indices))) and indices, "noncontiguous model block inventory")
        return len(indices)

    steps = len(state["timesteps"]["array"])
    events = expected_events(steps, layer_count("transformer_blocks"), layer_count("single_transformer_blocks"))
    require(len(events) == plan["max_transitions"] == plan["max_native_transitions"]
            and len(predicted["boundaries"]) == len(actual["boundaries"]) == len(events), "planned boundary inventory is truncated or extended")
    reports, previous = [], source["parent_fingerprint"]
    full_count = witness_count = 0
    for event, forecast, native in zip(events, predicted["boundaries"], actual["boundaries"], strict=True):
        tag = event_tag(event)
        require(forecast["event"] == event and forecast["tag"] == native["tag"] == tag
                and set(forecast["arrays"]) == set(native["arrays"]) == expected_arrays(event["phase"]), "ordered boundary/array inventory differs")
        receipt = native["receipt"]
        h["public_receipt"](receipt)
        require(h["read"](data / "typed-store/receipts" / (receipt["fingerprint"] + ".json")) == receipt
                and receipt["operation"] == "continue" and receipt["steps"] == len(receipt["transitions"]) == 1
                and receipt["parent"] == previous and receipt["model_identity"] == source["model_identity"]["adapter"]
                and receipt["execution"] == source["execution"], "native receipt/source chain differs")
        transition = receipt["transitions"][0]
        require(_point_matches(transition["before"], event) and _point_matches(transition["after"], event, after=True)
                and receipt["execution_point"] == transition["after"], "native receipt and sealed schedule differ")
        previous = receipt["result"]
        arrays = {}
        for name in sorted(forecast["arrays"]):
            require(forecast["arrays"][name]["full_retained"] == (event["phase"] in {"readout", "scheduler"}), "boundary retention differs from declared endpoint policy")
            arrays[name] = inspect_saved_array(forecast["arrays"][name], native["arrays"][name], plan["witness_values"])
            full_count += int("full_recomputed" in arrays[name])
            witness_count += 1
        reports.append({"tag": tag, "event": event, "receipt_fingerprint": receipt["fingerprint"], "arrays": arrays})
    require(set(predicted["vae"]) == set(actual["vae"]) == {"decoder_input", "decoded"}, "VAE observation inventory differs")
    vae = {name: inspect_saved_array(predicted["vae"][name], actual["vae"][name], plan["witness_values"]) for name in predicted["vae"]}
    require(not predicted["vae"]["decoder_input"]["full_retained"] and predicted["vae"]["decoded"]["full_retained"], "VAE retention differs")
    last = predicted["boundaries"][-1]
    require(last["event"]["phase"] == "scheduler" and np.array_equal(predicted["final_latent"], last["arrays"]["latent"]["full_values"]), "forecast final latent differs from own last scheduler state")
    pixel_error = compare(actual["pixels"], predicted["pixels"])
    require(metrics_match(actual["pixel_error"], pixel_error) and metrics_match(summary["pixel_error"], pixel_error), "pixel aggregate differs from retained pixels")
    images = {}
    require(source["image_processor_config"]["do_normalize"] is True, "native pixel normalization contract differs")
    for name, value, expected_sha in (("forecast", predicted["pixels"], predicted["forecast_png_sha256"]),
                                      ("native", actual["pixels"], actual["native_png_sha256"])):
        require(value.dtype == np.uint8 and value.shape == (plan["height"], plan["width"], 3), "RGB pixel ABI differs")
        retained = predicted if name == "forecast" else actual
        require(np.array_equal(value, pixels_from_decoded(retained["vae"]["decoded"]["full_values"], retained["vae"]["decoded"]["descriptor"]["dtype"])), "pixels do not follow retained decoded tensor/native cast policy")
        path = data / (name + ".png")
        with Image.open(path) as image:
            pixels = np.asarray(image)
        require(h["sha"](path.read_bytes()) == expected_sha and np.array_equal(pixels, value), "PNG differs from sealed/observed pixels")
        images[name] = {"path": str(path.relative_to(Path(run_dir).resolve())), "png_sha256": expected_sha,
                        "pixel_sha256": h["sha"](value.tobytes()), "shape": list(value.shape)}
    counts = summary["counts"]
    require(actual["counts"] == counts and counts["predicted_transitions"] == counts["native_transitions"] == len(events)
            and counts["prediction_module_entries"] == 0 and counts["vae_decoder_entries"] == 1
            and counts["selected_block_entries"] == steps * (layer_count("transformer_blocks") + layer_count("single_transformer_blocks")), "native/functional work accounting differs")
    helper_paths = [Path(__file__).with_name(name) for name in ("summarize_context_window.py", "analyze_flux_mlp_forecast.py")]
    refs = {name: {"path": str(path.relative_to(Path(run_dir).resolve())), "sha256": h["sha"](path.read_bytes())} for name, path in {
        "source": data / "source.json", "partial_summary" if failed else "summary": data / "summary.json", "timeline": data / "timeline.jsonl",
        **({"failure": data / "failure.json"} if failed else {}),
        **{"weights_" + stage: data / ("weights-" + stage + ".json") for stage in ("before-prediction", "at-release", "after-native")}}.items()}
    return {"schema": "mdb-flux-trajectory-forecast-audit-v1", "mechanics_status": "valid", "source": run["source"],
        "model_identity": source["model_identity"], "execution": source["execution"], "versions": source["versions"],
        "implementation_sources": source["implementation_sources"], "vae_config": source["vae_config"], "image_processor_config": source["image_processor_config"],
        "prompt": plan["prompt"], "seed": plan["seed"], "counts": counts, "parent_fingerprint": source["parent_fingerprint"],
        "typed_store": stored["counts"], "pre_release_counters": sealed[0]["counts"], "arithmetic": source["arithmetic"],
        "low_level_kernels": source["low_level_kernels"], "observed_scope": source["observed_scope"], "boundaries": reports,
        "vae": vae, "pixel_error": pixel_error, "images": images, "terminal_status": "unassessed",
        "recomputed": {"boundary_witness_arrays": witness_count, "VAE_witness_arrays": len(vae), "full_boundary_arrays": full_count,
                       "full_VAE_arrays": sum("full_recomputed" in item for item in vae.values()), "full_RGB_arrays": 1},
        "scope": plan["scope"], "own_state": True, "native_refreshes": 0, "RGB_forecast": True,
        "limits": [plan["limits"], "No model, functional predictor, native replay or equation module was executed by this audit.",
            "One observed initial native frontier is supplied; functional code computes later states and contexts from its own previous values.",
            "Separately wired equations share low-level Torch/CUDA primitives with the native reference; this is not independent-kernel replication.",
            "Intermediate full-array aggregates are source-bound worker RAM observations; offline recomputation uses saved witnesses and retained endpoints.",
            "Numerical mismatches remain observations. RGB equality does not establish an object program, semantic circuit or independent prompt-level replication."],
        "audit_sources": {path.name: h["sha"](path.read_bytes()) for path in helper_paths}, "references": refs,
        "scheduler_status": "failed" if failed else "succeeded", "execution_status": "completed-observation-independently-audited",
        "publication_status": "failed-summary-serialization" if failed else "worker-published",
        "recovery": {"summary": summary, "transaction_report": run["transaction_report"]} if failed else None,
        "new_model_calls": 0, "equation_module_evaluations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--failed", action="store_true", help="audit completed observations in a failed publication attempt")
    parser.add_argument("--derived-dir", type=Path, help="new recovery output directory outside immutable raw data")
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    require(args.failed or args.derived_dir is None, "--derived-dir requires --failed")
    derived = (args.derived_dir or args.run_dir / "derived").resolve() if args.failed else None
    run_root = args.run_dir.resolve()
    if derived is not None:
        require(derived.is_relative_to(run_root) and not derived.is_relative_to(run_root / "extracted") and not derived.exists(), "recovery directory must be a new directory within the run, outside raw data")
    result = audit(args.run_dir, failed=args.failed)
    generated = {}
    if derived is not None:
        derived.mkdir(parents=True, exist_ok=False)
        for name, value in (("summary.json", result["recovery"]["summary"]), ("transaction-report.json", result["recovery"]["transaction_report"])):
            path = derived / name
            with path.open("x") as stream:
                json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            generated[name] = {"path": str(path.relative_to(run_root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        result["references"].update({"recovered_" + name.removesuffix(".json").replace("-", "_"): ref for name, ref in generated.items()})
        result["recovery"]["receipt_path"] = str((derived / "recovery-receipt.json").relative_to(run_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.with_name(args.output.stem + "-source.py")
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    result["auditor_source_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    if derived is not None:
        for name, path in (("audit", args.output.resolve()), ("auditor_source", snapshot.resolve())):
            generated[name] = {"path": str(path.relative_to(run_root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        receipt = {"schema": "mdb-flux-trajectory-publication-recovery-v1", "source": result["source"],
            "scheduler_status": "failed", "publication_status": result["publication_status"], "mechanics_status": result["mechanics_status"],
            "raw_references": result["references"], "generated": generated,
            "new_model_calls": 0, "equation_module_evaluations": 0, "raw_files_modified": False}
        with (derived / "recovery-receipt.json").open("x") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"mechanics_status": result["mechanics_status"], "boundaries": len(result["boundaries"]),
                      "pixel_exact": result["pixel_error"]["exact"], "model_calls": 0}))


if __name__ == "__main__":
    main()
