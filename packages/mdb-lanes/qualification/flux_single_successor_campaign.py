"""Imported same-image/different-working-text single1 conditional K/V assay."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import runpy
import time
from collections.abc import Mapping
from pathlib import Path

INPUT_CUTS = {"zero": "a8b22c7d571e2a1dd07d84879e9ec5f6b6c2d5d186dfde830d74b6ff7c0b7570",
    "text": "9b6b1925e426a3559a30930567f808845711828aea3165409bcfc8c0d50f0792"}
QUERIES = [0, 31, 32, 255, 511, 512, 767, 1023]


def json_value(value):
    if isinstance(value, Mapping):
        return {key: json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(child) for child in value]
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(json_value(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_config(config):
    if (config.get("single_index") != 1 or config.get("capture_reference_trajectory") is not True
            or config.get("selector") != {"logical_step": 3, "phase": "single", "index": 1}
            or config.get("input_cuts") != INPUT_CUTS or config.get("input_store") != "input-store"
            or config.get("source_selection") != "source-selection.json" or config.get("height") != 512
            or config.get("width") != 512 or config.get("steps") != 4 or config.get("attention_query_indices") != QUERIES):
        raise ValueError("successor scientific site/source/trajectory/query geometry differs")
    recipes = [("zero", "zero", "zero"), ("text", "text", "text"), ("sham", "zero", "zero"),
        ("K.image", "K", None), ("V.image", "V", None), ("KV.image", "KV", None),
        ("KV.both", "KV", "KV"), ("text.text", None, "text"), ("coherent", "text", "text")]
    expected = [{"id": name, "kind": "reference" if name in {"zero", "text"} else "edited",
        "image_source": image, "text_source": text} for name, image, text in recipes]
    if config.get("branches") != expected:
        raise ValueError("successor nine-arm recipes differ")
    for name, lo, hi in (("max_adapter_steps", 182, 182), ("max_acts", 10, 10), ("max_candidate_calls", 5, 5),
            ("max_images", 9, 9), ("max_vae_calls", 9, 9), ("cpu_threads", 1, 16),
            ("max_wall_seconds", 1, 600), ("archive_compression_level", 1, 9)):
        if type(config.get(name)) is not int or not lo <= config[name] <= hi:
            raise ValueError("invalid finite successor budget: " + name)


def emit(output, report, config):
    runpy.run_path("flux_family_custody.py")["emit"](output, json_value(report),
        limits=config.get("transport_limits"), compression_level=config["archive_compression_level"])


def manifest(output):
    path = output / "files.json"
    if path.exists():
        path.unlink()
    save(path, {item.relative_to(output).as_posix(): sha(item) for item in sorted(output.rglob("*")) if item.is_file()})


def validate_implementation_sources(prior, current):
    """Bind actual native code across installation paths, allowing encoder extras."""
    def basenames(sources):
        result = {}
        for path, digest in sources.items():
            name = Path(path).name
            if name in result:
                raise ValueError("ambiguous native implementation basename: " + name)
            result[name] = digest
        return result
    old, new = basenames(prior), basenames(current)
    if not new or any(old.get(name) != digest for name, digest in new.items()):
        raise ValueError("native implementation source differs from imported history")
    return new


def verify_selection(config):
    """Check staged source/custody pins without importing a model framework."""
    selection = json.loads(Path(config["source_selection"]).read_text())
    if (selection["schema"] != "mdb-flux-single-successor-selection-v1" or selection["prior_job_id"] != "job-b87315cd3534"
            or selection["context_id"] != "p1-s53" or selection["input_cuts"] != config["input_cuts"]):
        raise ValueError("successor selected source occurrence differs")
    if any(config.get(key) != selection[key] for key in ("expected_pngs", "expected_final_latents")):
        raise ValueError("successor declared reference controls differ from selection")
    records = {name: selection[name] for name in ("source_record", "prefix_record", "audit_record", "prior_request", "prior_manifest", "prior_report")}
    loaded = {}
    for name, record in records.items():
        path = Path(record["path"])
        if path.is_symlink() or not path.resolve().is_relative_to(Path.cwd()) or sha(path) != record["sha256"]:
            raise ValueError("successor selected metadata path/digest differs: " + name)
        loaded[name] = json.loads(path.read_text())
    prior, audit, request, report = (loaded[name] for name in ("source_record", "audit_record", "prior_request", "prior_report"))
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or request["request_sha256"] != selection["request_sha256"] or prior["request_sha256"] != selection["request_sha256"]
            or audit["mechanics_status"] != "valid" or audit["source"]["job_id"] != selection["prior_job_id"]
            or audit["source"]["request_sha256"] != selection["request_sha256"]
            or audit["source"]["archive_sha256"] != selection["archive_sha256"]
            or report["job_id"] != selection["prior_job_id"] or report["archive_sha256"] != selection["archive_sha256"]
            or report["request_sha256"] != selection["request_sha256"]
            or report["files_manifest_sha256"] != records["prior_manifest"]["sha256"]
            or prior["context_id"] != selection["context_id"]):
        raise ValueError("successor prior request/audit/transport/source binding differs")
    input_root = Path(config["input_store"]).resolve()
    for path, digest in selection["files"].items():
        item = Path(path)
        if item.is_symlink() or not item.resolve().is_relative_to(input_root) or sha(item) != digest:
            raise ValueError("successor imported CAS path/hash differs")
    root_prefix = "contexts/" + selection["context_id"] + "/"
    for key, metadata in ((root_prefix + "source.json", records["source_record"]),
            (root_prefix + "prefixes.json", records["prefix_record"])):
        if loaded["prior_manifest"][key] != metadata["sha256"]:
            raise ValueError("successor metadata differs from prior producer manifest")
    for name, expected_id in (("zero", "KV.image"), ("text", "coherent")):
        record = selection["prior_rows"][name]
        if sha(record["path"]) != record["sha256"] or loaded["prior_manifest"][root_prefix + "rows/" + expected_id + ".json"] != record["sha256"]:
            raise ValueError("successor source row digest/manifest differs")
        row = json.loads(Path(record["path"]).read_text())
        expected_slots = ["image"] if name == "zero" else ["image", "text"]
        if (row["id"] != expected_id or [act["slot"] for act in row["acts"]] != expected_slots
                or row["acts"][-1]["post_act"] != selection["input_cuts"][name]
                or row["parent"] != loaded["prefix_record"]["prefixes"]["zero"]["post_single0"]
                or row["image"]["sha256"] != selection["expected_pngs"][name]
                or row["final_latent"]["descriptor"] != selection["expected_final_latents"][name]):
            raise ValueError("successor low/high source row/Act/reference occurrence differs")
    return selection, prior


def worker():
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if request["config"] != config or any(sha(path) != digest for path, digest in request["source_hashes"].items()):
        raise ValueError("successor source/request bytes differ")
    request = {**request, "job_id": os.environ["MRUN_JOB_ID"]}
    output = Path("flux-single-successor-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    save(output / "plan-frozen.json", {"job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "plan_sha256": sha(output / "plan.json"), "model_loads": 0, "adapter_steps": 0, "candidate_calls": 0,
        "scope": "Imported observed same-I/different-T single1 parents; no new encoding/noise/raw-prompt forecast"})
    selection, prior = verify_selection(config)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    started = time.monotonic()
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("successor panel requires admitted CUDA")
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"] or prior["checkpoint_sha256"] != identity(inventory):
        raise ValueError("successor checkpoint identity differs")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True,
        text_encoder=None, tokenizer=None)
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("requires deterministic five-joint/twenty-single distilled Klein")
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False).to("cuda")
    pipe.vae.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    if adapter.model_identity != prior["model_identity"] or dict(adapter.execution) != prior["execution"]:
        raise ValueError("successor imported native model/execution identity differs")
    implementations = {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in
        (type(pipe), type(pipe.transformer), type(pipe.transformer.single_transformer_blocks[1]),
         type(pipe.transformer.single_transformer_blocks[1].attn.processor), type(pipe.vae), type(pipe.scheduler), type(adapter))}
    native_source_binding = validate_implementation_sources(prior["implementation_sources"], implementations)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "saturn-pub", "mdb-runtime")}
    if versions != prior["versions"]:
        raise ValueError("successor installed native/package versions differ from source history")
    inputs = LocalStore(config["input_store"])
    cuts = {name: inputs.load(identifier, device="cuda") for name, identifier in config["input_cuts"].items()}
    engine = runpy.run_path("flux_single_contexts.py")
    metadata = {"job_id": request["job_id"], "context_id": selection["context_id"],
        "checkpoint_sha256": identity(inventory), "execution": dict(adapter.execution), "source_selection": selection,
        "source_roles": {"zero": "low: prior KV.image post-single0 I/T, C0", "text": "high: prior coherent post-single0 I/T, C0"},
        "prior_implementation_sources": prior["implementation_sources"],
        "implementation_sources": implementations, "native_source_binding": native_source_binding, "versions": versions,
        "outer_preseal_counts": {"model_loads": 1, "prompt_encodings": 0, "noise_draws": 0, "adapter_steps": 0, "candidate_calls": 0},
        "timing_scope": "max_wall_seconds bounds panel work after hydration/loading; outer lease additionally bounds load and export. worker_seconds starts before checkpoint hashing/loading.",
        "prior_exposure": "Both reference suffixes and RGBs were already executed in the predecessor. Main predictions precede this assay's new site-specific captures; no globally first-execution claim.",
        "trajectory_policy": "Reference zero/text only: full I/T/latent arrays while present at every native exit; selected complete cuts after single1/2/9, pre-readout, post-readout scheduler and final. Actual noise_prediction retained from the post-readout cut; no absent slots invented"}
    summary = engine["run_panel"](pipe=pipe, adapter=adapter, cuts=cuts, output=output, config=config, request=request,
        source_metadata=metadata, output_prepared=True)
    replay = {}
    for name in ("zero", "text"):
        row = json.loads((output / "rows" / (name + ".json")).read_text())
        replay[name] = {"png_equals_prior": row["image"]["sha256"] == selection["expected_pngs"][name],
            "final_latent_equals_prior": row["final_latent"]["descriptor"] == selection["expected_final_latents"][name]}
    save(output / "replay-controls.json", replay)
    outer = {**summary, "schema": "mdb-flux-single-successor-result-v1", "counts": {**summary["counts"], "model_loads": 1},
        "worker_seconds": time.monotonic() - started, "replay_controls": replay,
        "panel_summary_sha256": sha(output / "summary.json"), "mechanics_status": "pending-independent-audit"}
    save(output / "outer-summary.json", outer)
    print("MDB_FLUX_SINGLE_SUCCESSOR_ARCHIVE_START " + json.dumps({"raw_output_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "counts": outer["counts"]}), flush=True)
    manifest(output)
    emit(output, outer, config)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-single-successor-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-flux-single-successor-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            save(output / "failure.json", failure)
            manifest(output)
            emit(output, failure, request["config"])
        raise
