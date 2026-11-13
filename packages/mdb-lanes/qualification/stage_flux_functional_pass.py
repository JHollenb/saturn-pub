"""Stage four source-bound, sealed own-state FLUX passes in one finite lease."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import runpy
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_functional_pass.py"
HELPERS = ("flux_functional_denoiser.py", "flux_image_tail_microscope.py", "flux_source_receiver.py")
SOURCE_CUTS = {
    "sham": "20314bc02c4521544ea29e9c07eeb976a0df30265192d2e99569bd9b7b2ac28c",
    "HH": "63c6e947de2f281904f9b1609c18e64ce82ab53461aae43142e9a7cb0194a257",
    "HL": "163d209e86423d74c7183e64aef99e13597d75d566f8ce8b7a6c10abaec9e0c8",
}
BRANCHES = {"sham": "c1-o0.sham", "HH": "c1-o0.d-c1-o0", "HL": "c1-o0.d-c0-o1"}
QUEUE_NOTE = (
    "Own-state FLUX pass:4B1 cases(project+5joint+20single+readout+Euler),112functional"
    " transitions globally sealed before112native;100SDPA/436linear each lane.8observed"
    " next-project consumers. Original169weights resident once; no encode/noise/VAE/"
    " intermediate refresh.3historical controls+fixed3/8newlatent. Raw<=6GiBremote;"
    " compact audit only. Measured fulltransformer envelope RAM14GiB/VRAM14GiB,"
    " CPU4;600s estimate/1200s science/1500s outer,no retry."
)


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
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def select_closure(source, inventory, cuts, *, step):
    inputs, files = {}, {}

    def admit(name):
        path, record = source / name, inventory[name]
        if (path.is_symlink() or not path.resolve().is_relative_to(source.resolve())
                or sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]):
            raise ValueError("recovered source differs: " + name)
        files[name] = dict(sha256=record["sha256"], bytes=record["bytes"])

    identities = set()
    for role, fingerprint in cuts.items():
        name = "typed-store/cuts/" + fingerprint + ".json"
        admit(name)
        cut = read(source / name)
        point = cut["execution_point"]
        if (cut["fingerprint"] != fingerprint or (point["logical_step"], point["phase"], point["local_clock"])
                != (step, "project", 0) or point["next_operation"] != "native_input_projection"):
            raise ValueError("declared complete pre-project cut differs")
        identities.add(cut["manifest"]["model_identity"])
        for page in cut["slots"].values():
            admit("typed-store/pages/" + page + ".json")
        for blob in cut["physical_blobs"]:
            admit("typed-store/blobs/" + blob + ".safetensors")
        inputs[role] = dict(store="typed-store", cut=fingerprint)
    if len(identities) != 1:
        raise ValueError("source model identities differ")
    return inputs, files, identities.pop()


def verify_remote(root, records):
    script = '''import hashlib,json
from pathlib import Path
root=Path(%r);records=json.loads(%r)
for name,row in records.items():
 p=root/name
 assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
 assert p.stat().st_size==row['bytes']
 h=hashlib.sha256()
 with p.open('rb') as s:
  for b in iter(lambda:s.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==row['sha256'],name
print(json.dumps(dict(verified=len(records),bytes=sum(r['bytes'] for r in records.values()))))
''' % (str(root), json.dumps(records, sort_keys=True))
    return json.loads(subprocess.run(["ssh", "-o", "BatchMode=yes", "beast", "python3", "-I", "-"],
                                    input=script, text=True, capture_output=True, check=True, timeout=90).stdout)


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
    if sha(__file__) != request["launcher_sha256"] or request_identity(request) != request["request_sha256"]:
        raise ValueError("functional-pass request/launcher changed")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("functional-pass staged payload differs: " + name)


def stage(run, campaign, planning):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    run, campaign, planning = (Path(path).resolve() for path in (run, campaign, planning))
    native = campaign / "latent-direction-native-01"
    source = native / "extracted/latent-direction-native-result"
    collection, original, job = (read(native / name) for name in ("collection.json", "request.json", "job.json"))
    if (sha(native / "collection.json") != "e5bfbf3e63d29f714f80258ebab1c821c3ced9f2d36dc60622e14d1376621cf8"
            or original["request_sha256"] != collection["request_sha256"]
            or job["job_id"] != collection["job_id"] or job["state"] != collection["scheduler_status"]
            or collection["job_id"] != "job-46a52ffae499" or collection["scheduler_status"] != "failed"
            or not collection["scientific_work_completed"] or collection["manifest_origin"] != "collector-derived-source-manifest"
            or job["payload_custody"]["executed"]["sha256"] != collection["executed_payload_sha256"]):
        raise ValueError("original failed-exporter recovery provenance differs")
    for name in ("request.json", "job.json", "collection-local-inventory.json"):
        if sha(native / name) != collection["bindings"][name]:
            raise ValueError("recovery receipt binding changed: " + name)
    inventory = read(native / "collection-local-inventory.json")["files"]
    historical_cuts = {}
    for role, branch in BRANCHES.items():
        name = "native/" + branch + ".jsonl"
        if sha(source / name) != inventory[name]["sha256"]:
            raise ValueError("historical ledger changed")
        rows = [json.loads(line) for line in (source / name).read_text().splitlines()]
        if len(rows) != 84 or rows[27]["capture"]["point"]["logical_step"] != 2:
            raise ValueError("historical source ledger geometry differs")
        historical_cuts[role] = rows[27]["capture"]["cut"]
    inputs, input_files, identity = select_closure(source, inventory, SOURCE_CUTS, step=1)
    historical, historical_files, historical_identity = select_closure(source, inventory, historical_cuts, step=2)
    if identity != historical_identity:
        raise ValueError("historical control model differs")
    remote = collection["remote_path"]
    selection = dict(inputs=inputs, input_files=input_files, historical=historical, historical_files=historical_files,
                     remote_verification=verify_remote(remote, {**input_files, **historical_files}),
                     historical_policy="Verification-only; worker reads after whole cohort seal and all four native target passes.")
    run.mkdir(parents=True, exist_ok=False)
    save(run / "source-selection.json", selection)
    binding = campaign / "run-01/extracted/flux-prompt-factorial-result/binding.json"
    copies = {"binding.json": binding, "parameter-artifacts.json": source / "parameter-artifacts.json",
              "retrieve_weights.py": source / "retrieve_weights.py",
              "transformer-config.json": planning / "transformer-config.json", "scheduler-config.json": planning / "scheduler-config.json"}
    if sha(binding) != "cdb09e98a1bddbe3111f7422fb979c46b36469531da39eb9ffadabc97f723f23":
        raise ValueError("original full169 binding differs")
    for name, path in copies.items():
        shutil.copyfile(path, run / name)
    basis()["bind"](original["execution_environment"])
    native_source = Path(importlib.metadata.distribution("diffusers").locate_file("diffusers/models/transformers/transformer_flux2.py"))
    config = dict(schema="mdb-flux-functional-pass-plan-v1", input_root=remote, inputs=inputs, input_files=input_files,
        historical_root=remote, historical=historical, historical_files=historical_files, expected_source_model_identity=identity,
        parameter_root=original["config"]["model_path"], binding="binding.json", binding_sha256=sha(run / "binding.json"),
        parameter_manifest="parameter-artifacts.json", parameter_manifest_sha256=sha(run / "parameter-artifacts.json"),
        retrieval_script="retrieve_weights.py", retrieval_script_sha256=sha(run / "retrieve_weights.py"),
        transformer_config="transformer-config.json", transformer_config_sha256=sha(run / "transformer-config.json"),
        scheduler_config="scheduler-config.json", scheduler_config_sha256=sha(run / "scheduler-config.json"),
        native_source_sha256=sha(native_source), helper_source_sha256={name: sha(HERE / name) for name in HELPERS},
        device="cuda", source_reduction_policy=True, cpu_threads=4, max_wall_seconds=1200, witness_values=16,
        transport_limits=dict(max_expanded_bytes=6 << 30, max_member_bytes=128 << 20),
        output_scratch="/mnt/big/mdb/research", output_name="functional-pass-result", directory_custody=True,
        source_provenance=dict(collection, collection_receipt_sha256=sha(native / "collection.json"),
            source_inventory_sha256=sha(native / "collection-local-inventory.json"), source_job_record_sha256=sha(native / "job.json"),
            original_binding_sha256=sha(binding)),
        scope="Four B1 own-state original-weight functional passes from complete step1/project inputs, including fixed3/8 own-history variant. All112 transitions sealed before112 native adapter targets. No observed intermediate refresh. Eight observed native next-project consumers are distinct predicted/reference branches. Historical three controls verification-only. Shared original primitive kernels, not independent CUDA kernels, speedup or raw-prompt-to-image claim.")
    local_config = dict(config, input_root=str(source))
    for key in ("binding", "parameter_manifest", "retrieval_script", "transformer_config", "scheduler_config"):
        local_config[key] = str(run / config[key])
    declared = runpy.run_path(str(HERE / WORKER))["validate_config"](local_config)
    save(run / "plan.json", config)
    save(run / "count-contract.json", declared)
    extras = [(HERE / name, name) for name in HELPERS] + [(run / name, name) for name in copies]
    preflight = run / "preflight.v1.json"
    save(preflight, dict(schema_version="mrun-preflight-config-v1", hashed_roots=[WORKER, *[name for _, name in extras], preflight.name],
        path_token_allowlist=[remote, config["parameter_root"], config["output_scratch"], config["output_name"]]))
    extras.append((preflight, preflight.name))
    request = dict(schema="mdb-flux-functional-pass-request-v1", config=config,
        execution_environment=original["execution_environment"], scheduler_url=original["scheduler_url"],
        model="flux2-klein-4b-distilled", task_family="analysis", backend="torch-native-functional", device="cuda", dtype="bfloat16",
        flux_mode="capture", capture_sites=True, height=512, width=512, steps=1,
        workload_geometry=dict(model_loads=1, peak_physical_batch=1, image_tokens=1024, text_tokens=512,
            functional_passes=4, native_passes=4, diffusion_steps_per_pass=1, observed_project_consumers=8,
            output_bytes_bound=6 << 30),
        needs={"cuda": True, "host": "beast"}, runtime_wheels=original["runtime_wheels"],
        source_hashes={name: sha(path) for path, name in [(HERE / WORKER, WORKER), *extras]}, launcher_sha256=sha(__file__),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=232, logical_branches=4,
            logical_checkpoint_replays=4, max_local_work_items=240,
            extra={"task_family": "analysis", "functional_transitions": 112, "native_transitions": 112, "observed_consumers": 8},
            checkpoint_policy="One finite resident-transformer CUDA lease; four own-state forecasts globally sealed before four native target passes, then historical checks and eight native next-project consumers. No retries or nested submissions."),
        resource_exception=dict(ram_mb=14336, vram_mb=14336, cpu_threads=4, disk_gb=8, est_wall_s=600, timeout_s=1500,
            calibration="Measured full native transformer campaign job-4350531b1dcd RSS8509.1MB/VRAM12936MB, elapsed1033.69s with independent CPU arithmetic,112-step horizon and VAE. This has no encoder/VAE, one7.751GB original transformer resident onGPU, sequentialB1 image1024/text512. Two112-transition lanes plus8projects;169temporary recipe loads, streaming descriptors and fulltypedstates<=6GiB remote. RAM14GiB/VRAM14GiB measured-envelope exception; no claim fastest.600s estimate,1200s science/1500s outer."))
    request["request_sha256"] = request_identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, additional_files=extras)
    validate_payload(run, request)
    save(run / "staged-command.json", dict(command=command(request), request_sha256=request["request_sha256"]))
    return dict(request_sha256=request["request_sha256"], input_files=len(input_files), historical_files=len(historical_files), submitted=False)


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    if len(QUEUE_NOTE) > 500:
        raise ValueError("scheduler queue note exceeds its500-character ABI")
    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_payload(run, request)
    if (run / "submission.json").exists():
        raise ValueError("functional-pass run already submitted")
    review = read(run / "finite-run-review-v1.json")
    if (review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != []
            or review.get("review_status") != "ready-for-finite-run"):
        raise ValueError("request-bound finite-run review required")
    args = command(request)
    save(run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"], review_sha256=sha(run / "finite-run-review-v1.json")))
    resource = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-functional-pass", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resource[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True, note=QUEUE_NOTE)
    save(run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--planning", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and (args.campaign is None or args.planning is None):
        parser.error("original factorial campaign and pinned native config planning directory required")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.campaign, args.planning)))
