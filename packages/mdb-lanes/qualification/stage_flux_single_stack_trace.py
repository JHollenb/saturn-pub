"""Stage one source-bound, finite native single-stack localization lease."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_single_stack_trace.py"
HELPERS = ("flux_image_tail_microscope.py", "flux_source_receiver.py")
ROLES = {"sham": "c1-o0.sham", "HH": "c1-o0.d-c1-o0", "HL": "c1-o0.d-c0-o1"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def basis():
    return runpy.run_path(str(HERE / "stage_flux_image_tail_microscope.py"))


def request_identity(request):
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()


def select_inputs(source, inventory):
    """Bind real pre-single0 and pre-readout cuts, not regenerated prefixes."""
    inputs, files = {}, {}

    def admit(name):
        path = source / name
        record = inventory[name]
        if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(source.resolve())
                or sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]):
            raise ValueError("recovered source differs: " + name)
        files[name] = {"sha256": record["sha256"], "bytes": record["bytes"]}

    for role, branch in ROLES.items():
        ledger = "native/" + branch + ".jsonl"
        admit(ledger)
        rows = [json.loads(line) for line in (source / ledger).read_text().splitlines()]
        if len(rows) != 84:
            raise ValueError("original source ledger length differs")
        for row_index, phase, index, cursor, operation in ((5, "single", 0, 0, "single_block.0"),
                                                         (25, "readout", None, 20, "norm_and_projection")):
            row = rows[row_index]
            capture = row["capture"]
            expected = dict(edge="before", family="flux2-klein", index=index, local_clock=cursor,
                            logical_step=1, phase=phase, operator=phase, next_operation=operation)
            if (row["logical_step"] != 1 or row["local_index"] != row_index
                    or not capture["retained_full_cut"] or capture["point"] != expected):
                raise ValueError("saved source boundary differs")
            name = "typed-store/cuts/" + capture["cut"] + ".json"
            admit(name)
            cut = read(source / name)
            if cut["fingerprint"] != capture["cut"] or cut["execution_point"] != expected:
                raise ValueError("saved cut identity/clock differs")
            for page in cut["slots"].values():
                admit("typed-store/pages/" + page + ".json")
            for blob in cut["physical_blobs"]:
                admit("typed-store/blobs/" + blob + ".safetensors")
        inputs[role] = dict(store="typed-store", cut=rows[5]["capture"]["cut"], branch=branch,
            ledger=ledger, entry_row=5, exit_row=25,
            historical_image_descriptor=rows[25]["outputs"]["image"]["descriptor"],
            historical_text_descriptor=rows[25]["outputs"]["text"]["descriptor"])
    for name in ("source.json", "parameter-artifacts.json", "retrieve_weights.py"):
        admit(name)
    return inputs, files


def command(request):
    from mdb_lanes.payload import worker_command
    basis()["bind"](request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in (("tmpdir", "TMPDIR"), ("cuda_cache_path", "CUDA_CACHE_PATH"),
                      ("torchinductor_cache_dir", "TORCHINDUCTOR_CACHE_DIR")):
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def validate_payload(run, request):
    if request_identity(request) != request["request_sha256"] or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("single-stack request/launcher changed")
    if sha(HERE / "stage_flux_image_tail_microscope.py") != request["basis_launcher_sha256"]:
        raise ValueError("qualified environment binder changed")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("single-stack staged payload changed: " + name)


def stage(run, campaign):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    run, campaign = Path(run).resolve(), Path(campaign).resolve()
    native = campaign / "latent-direction-native-01"
    source = native / "extracted/latent-direction-native-result"
    collection, original = read(native / "collection.json"), read(native / "request.json")
    if (collection["job_id"] != "job-46a52ffae499" or collection["scheduler_status"] != "failed"
            or collection["native_execution_status"] != "failed-post-exec-finalization"
            or not collection["scientific_work_completed"] or collection["model_calls_during_collection"] != 0
            or collection["collection_mode"] != "authenticated-ssh-manifest-recovery"
            or collection["manifest_origin"] != "collector-derived-source-manifest"
            or collection["request_sha256"] != original["request_sha256"]):
        raise ValueError("accepted failed-finalizer source recovery required")
    for name, digest in collection["bindings"].items():
        if sha(native / name) != digest:
            raise ValueError("source recovery binding changed: " + name)
    job = read(native / "job.json")
    if (job["job_id"] != collection["job_id"] or job["state"] != "failed"
            or job["payload_custody"]["executed"]["sha256"] != collection["executed_payload_sha256"]):
        raise ValueError("actual failed job/custody differs")
    inputs, files = select_inputs(source, read(native / "collection-local-inventory.json")["files"])
    binding = campaign / "run-01/extracted/flux-prompt-factorial-result/binding.json"
    if sha(binding) != read(binding.parent / "files.json")["binding.json"]:
        raise ValueError("original parameter binding differs")
    basis()["bind"](original["execution_environment"])
    module = Path(importlib.metadata.distribution("diffusers").locate_file("diffusers/models/transformers/transformer_flux2.py"))
    config = dict(schema="mdb-single-stack-plan-v1", input_root=collection["remote_path"], inputs=inputs, input_files=files,
        binding="source-binding.json", binding_sha256=sha(binding), source_binding="source-binding.json", source_binding_sha256=sha(binding),
        parameter_manifest="parameter-artifacts.json", parameter_manifest_sha256=sha(source / "parameter-artifacts.json"),
        retrieval_script="retrieve_weights.py", retrieval_script_sha256=sha(source / "retrieve_weights.py"),
        parameter_root=original["config"]["model_path"], expected_model_identity=read(source / "source.json")["model_identity"],
        native_source_sha256=sha(module), helper_source_sha256={name: sha(HERE / name) for name in HELPERS},
        module_config=dict(dim=3072, num_attention_heads=24, attention_head_dim=128, mlp_ratio=3.0, eps=1e-6, bias=False),
        source_reduction_policy=True, device="cuda", cpu_threads=4, max_wall_seconds=600,
        output_scratch="/mnt/big/mdb/research", output_name="binding-minimal-result", directory_custody=True,
        source_provenance=dict(collection, collection_receipt_sha256=sha(native / "collection.json"),
            source_inventory_sha256=sha(native / "collection-local-inventory.json"),
            source_job_record_sha256=sha(native / "job.json"), original_binding_sha256=sha(binding)),
        predeclared_expectations=dict(replay="Compare all three complete native pre-readout image/text exits with original row25 descriptors; retain misses.",
            exploratory="Report every signed response/increment at every boundary. Location and concentration of HH-selective growth are unknown; no numerical quality or dominance gate.",
            selection="Three fixed known branches, not new contexts. Reporting partitions early0–6,middle7–13,late14–19 are frozen before trace."),
        scope="Native single-stack localization from actual saved step1/pre-single0 sham/HH/HL cuts, all own-state fields preserved. One original bank at a time,60 full native single calls and63 complete StateCuts, no Session/Act receipts, prefix regeneration, independent prediction, full model, encoder, noise, suffix, scheduler or VAE. Follow-up writer choice is exploratory.")
    runpy.run_path(str(HERE / WORKER))["validate_config"](config)
    run.mkdir(parents=True, exist_ok=False)
    for name, path in (("source-binding.json", binding), ("parameter-artifacts.json", source / "parameter-artifacts.json"),
                       ("retrieve_weights.py", source / "retrieve_weights.py")):
        shutil.copyfile(path, run / name)
    save(run / "plan.json", config)
    extras = [(HERE / name, name) for name in HELPERS]
    extras += [(run / name, name) for name in ("source-binding.json", "parameter-artifacts.json", "retrieve_weights.py")]
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [WORKER, *[name for _, name in extras], preflight.name],
        "path_token_allowlist": [config["input_root"], config["parameter_root"], config["output_scratch"], config["output_name"]]})
    extras.append((preflight, preflight.name))
    request = dict(schema="mdb-single-stack-request-v1", config=config,
        execution_environment=original["execution_environment"], scheduler_url=original["scheduler_url"],
        model="flux2-klein-4b-distilled", task_family="analysis", backend="torch-components", device="cuda", dtype="bfloat16",
        needs={"cuda": True, "host": "beast"}, runtime_wheels=original["runtime_wheels"],
        source_hashes={name: sha(path) for path, name in [(HERE / WORKER, WORKER), *extras]},
        launcher_sha256=sha(__file__), basis_launcher_sha256=sha(HERE / "stage_flux_image_tail_microscope.py"),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=60,
            logical_branches=3, logical_checkpoint_replays=3, max_local_work_items=64,
            extra={"task_family": "analysis", "native_single_calls": 60, "original_banks_hydrated": 20},
            checkpoint_policy="One finite component CUDA lease,20 original banks each serving three sequential B1 branches; all21 boundaries retained per branch; no retries or nested jobs."),
        resource_exception=dict(ram_mb=8192, vram_mb=3072, cpu_threads=4, disk_gb=3, est_wall_s=240, timeout_s=900,
            calibration="Measured original joint component jobs peakRSS3468–4149MB/VRAM1558–1576MB. Single bank245MB BF16 versus joint491MB; retain three B1x1024image/512text own-state branches, one bank at a time and bounded FP64 response arrays. One streamed7.75GB checkpoint hash plus80 named parameter payload reads, about594MB full states. Declared8192/3072MB measured-envelope headroom; mrun may clamp from history. Scientific600s/outer900s, no speedup claim."))
    request["request_sha256"] = request_identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, additional_files=extras)
    validate_payload(run, request)
    save(run / "staged-command.json", {"command": command(request), "request_sha256": request["request_sha256"]})
    return dict(request_sha256=request["request_sha256"], selected_input_files=len(files), submitted=False)


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_payload(run, request)
    if (run / "submission.json").exists():
        raise ValueError("single-stack run already submitted")
    review = read(run / "finite-run-review-v1.json")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("request-bound finite-run review required")
    args = command(request)
    save(run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"], review_sha256=sha(run / "finite-run-review-v1.json")))
    resource = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-single-stack", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resource[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True,
        note="HH localization: saved actual pre-single0 sham/HH/HL,20 single banks hydrated one at a time,3 sequential B1x1024image/512text calls each=60fullsingle/60SDPA.63 complete typed boundary cuts, own context/residuals, no fullmodel/encode/noise/suffix/scheduler/VAE or fabricatedAct/continue receipts. Original169recipes; rawstatesBeast, compact audit/retrieval only.240sestimate/600sscience/900souter,no retry; trace selects exploratory next writer.")
    save(run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and args.campaign is None:
        parser.error("source campaign is required")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.campaign)))
