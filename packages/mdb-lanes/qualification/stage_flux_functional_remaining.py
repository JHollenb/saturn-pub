"""Stage two remaining own-state FLUX passes and eight real VAE consumers."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_functional_remaining.py"
WORKER_SHA256 = "14d3f7b1dbd2d9d241229ace5b3ba17769441b1de811ae3a735dd61d0ed289c4"
HELPERS = ("flux_functional_denoiser.py", "flux_image_tail_microscope.py", "flux_source_receiver.py")
SOURCE_JOB = "job-027b22769047"
SOURCE_REQUEST = "5fbc74012eac18efc2e9a0887e9bea179739c3b7a6b6b841d4d8923344e54806"
SOURCE_MANIFEST = "6441e7ed43835dbe98ad82a1af97ac9c958f2ace05d0a34a5df3d8e84a884bc4"
SOURCE_SEAL = "23e9d7242cd255253526c1d0b15732148832d0eeb9469f0914ee1c3f20bab67f"
SOURCE_WORKER = "f693a2a66449cf35f048dc66450085ce2e7e1df354f197218040f5dafb422b26"
AUDIT_SHA256 = "993e476c3e34b7c7f7d86c3c686520476a1fe7f3b0beaacbce52d372a13f3fec"
COMPACT_MANIFEST = "11405cc007cd1b4f9ced54f69b6196f5242d718cd4116d8d803ff5dfabb47d32"
REMOTE = "/mnt/big/mdb/research/job-027b22769047/functional-pass-result"
QUEUE_NOTE = (
    "FLUX remaining:4 unchanged predictedS2 inputs,2 own-state passes/case;224functional"
    "+224native transitions,200SDPA/872linear each lane. All forecasts+4 real VAE"
    " consumers seal before referenceS2 reads/native futures;4 reference decodes after."
    " One original transformer+decoder resident,B1,no encode/noise/refresh. Raw<=6GiB"
    " remote,compact audit only. RAM8GiB/VRAM13000MiB CPU4;600s estimate/1200s science/"
    " 1500s outer,no retry."
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


def bind(environment):
    from mdb_lanes.execution_environment import intent

    for key, name in {"uv_cache_dir": "MDB_WORKER_UV_CACHE_DIR", "torch_home": "MDB_WORKER_TORCH_HOME",
                      "extra_model_roots": "MDB_WORKER_MODEL_ROOTS", "python": "MDB_WORKER_PYTHON",
                      "uv_link_mode": "MDB_WORKER_UV_LINK_MODE"}.items():
        os.environ[name] = environment[key]
    if intent() != {key: environment[key] for key in intent()}:
        raise ValueError("installed execution requirements differ from qualified cohort")
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"


def request_identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def remote_closures(expected):
    """Hash complete selected closures on Beast; return metadata, never arrays."""
    script = r'''import hashlib,json
from pathlib import Path
root=Path(ROOT);expected=EXPECTED
manifest=root/'files.json';raw=manifest.read_bytes()
assert hashlib.sha256(raw).hexdigest()==MANIFEST
inventory=json.loads(raw)['files'];verified={};result={};identities=set()
def admit(name):
 if name in verified:return
 p=root/name;r=inventory[name];before=p.stat()
 assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
 assert before.st_size==r['bytes']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==r['sha256'],name
 after=p.stat()
 assert (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
 verified[name]=dict(sha256=r['sha256'],bytes=r['bytes'])
for lane,store in [('forecast','prediction-store'),('native','native-store')]:
 selected={};files={};metadata={}
 for case,fingerprint in expected.items():
  name=store+'/cuts/'+fingerprint[lane]+'.json';admit(name)
  cut=json.loads((root/name).read_text());point=cut['execution_point'];m=cut['manifest'];component=m['execution']['component_pass']
  assert cut['fingerprint']==fingerprint[lane]
  assert (point['logical_step'],point['phase'],point['local_clock'],point['next_operation'])==(2,'project',0,'native_input_projection')
  assert (component['id'],component['lane'],component['case'])==('mdb-flux-functional-pass-v1',lane,case)
  identities.add(m['model_identity']);names={name}
  names.update(store+'/pages/'+v+'.json' for v in cut['slots'].values())
  names.update(store+'/blobs/'+v+'.safetensors' for v in cut['physical_blobs'])
  for path in names:admit(path);files[path]=verified[path]
  selected[case]=dict(store=store,cut=fingerprint[lane])
  metadata[case]=dict(cut=cut['fingerprint'],point=point,parent=cut['logical_parent'],component_provenance=component)
 result[lane]=dict(inputs=selected,files=files,metadata=metadata)
assert len(identities)==1
result.update(model_identity=identities.pop(),verified_files=len(verified),verified_bytes=sum(r['bytes'] for r in verified.values()),tensor_payloads_returned=0,remote_mutations=0)
print(json.dumps(result,sort_keys=True))
'''
    script = "ROOT=" + repr(REMOTE) + "\nEXPECTED=" + repr(expected) + "\nMANIFEST=" + repr(SOURCE_MANIFEST) + "\n" + script
    completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "beast", "python3", "-I", "-"],
                               input=script, text=True, capture_output=True, check=True, timeout=120)
    return json.loads(completed.stdout)


def command(request):
    from mdb_lanes.payload import worker_command

    bind(request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in (("tmpdir", "TMPDIR"), ("cuda_cache_path", "CUDA_CACHE_PATH"),
                      ("torchinductor_cache_dir", "TORCHINDUCTOR_CACHE_DIR")):
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def validate_payload(run, request):
    if sha(__file__) != request["launcher_sha256"] or request_identity(request) != request["request_sha256"]:
        raise ValueError("functional-remaining request/launcher changed")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("functional-remaining staged payload differs: " + name)


def source_job_config(job, original):
    config = job["config"]
    extra = {"mrun_guarded_request_sha256", "retry_on_kill", "saturn_request_sha256"}
    return (set(config) == set(original) | extra
            and {key: value for key, value in config.items() if key not in extra} == original
            and config["retry_on_kill"] is False and config == job["submitted_request"]["config"]
            and job["admission"]["idempotency_key"] == "mrun:" + config["mrun_guarded_request_sha256"])


def authenticate(campaign):
    native = campaign / "run-02"
    collection = campaign / "compact-collection-01"
    compact = collection / "extracted/binding-minimal-result"
    original, job = read(native / "request.json"), read(native / "job.json")
    audit_request, audit_job = read(collection / "request.json"), read(collection / "job.json")
    report, custody = read(collection / "report.json"), read(collection / "archive-custody.json")
    if (original["request_sha256"] != SOURCE_REQUEST or request_identity(original) != SOURCE_REQUEST
            or original["source_hashes"]["flux_functional_pass.py"] != SOURCE_WORKER
            or job["job_id"] != SOURCE_JOB or job["state"] != "succeeded" or job["result"]["returncode"] != 0
            or not source_job_config(job, original)):
        raise ValueError("accepted producer job/request binding differs")
    for record in (job, audit_job):
        seals = record["payload_custody"]
        if not (seals["declared"]["sha256"] == seals["sealed"]["sha256"] == seals["executed"]["sha256"] == record["result"]["executed_payload"]["sha256"]):
            raise ValueError("source executed-payload custody differs")
    if (audit_job["job_id"] != "job-40faf1e39119" or audit_job["state"] != "succeeded"
            or audit_job["result"]["returncode"] != 0 or not source_job_config(audit_job, audit_request)
            or audit_request["request_sha256"] != report["request_sha256"]
            or report["job_id"] != audit_job["job_id"] or report["files_manifest_sha256"] != COMPACT_MANIFEST
            or not custody["complete"] or custody["job_id"] != audit_job["job_id"]
            or sha(collection / "archive.tar.gz") != custody["archive_sha256"] or custody["archive_sha256"] != report["archive_sha256"]):
        raise ValueError("authenticated compact audit collection differs")
    if sha(compact / "files.json") != COMPACT_MANIFEST:
        raise ValueError("compact manifest differs")
    for name, digest in read(compact / "files.json").items():
        path = compact / name
        if path.is_symlink() or not path.resolve().is_relative_to(compact.resolve()) or sha(path) != digest:
            raise ValueError("compact metadata differs: " + name)
    audit, source = read(compact / "audit.json"), read(compact / "source-source.json")
    inventory = read(compact / "source-files.json")
    if (sha(compact / "audit.json") != AUDIT_SHA256 or sha(compact / "source-files.json") != SOURCE_MANIFEST
            or inventory["job_id"] != SOURCE_JOB or inventory["request_sha256"] != SOURCE_REQUEST
            or audit["mechanics_status"] != "authenticated-full-state-saved-evidence-audit"
            or audit["native_manifest_sha256"] != SOURCE_MANIFEST or audit["native_request_sha256"] != SOURCE_REQUEST
            or source["job_id"] != SOURCE_JOB or source["request_sha256"] != SOURCE_REQUEST
            or source["worker_sha256"] != SOURCE_WORKER or source["forecast_seal"]["seal_sha256"] != SOURCE_SEAL
            or not audit["forecast_verification"]["verified"] or audit["forecast_verification"]["seal_sha256"] != SOURCE_SEAL):
        raise ValueError("accepted source/audit identity differs")
    index = read(compact / "state-index.json")
    if set(index) != {"sham", "HH", "HL", "new-3over8"}:
        raise ValueError("four fixed prior histories required")
    expected = {}
    for case, lanes in index.items():
        if set(lanes) != {"forecast", "native"} or any(len(rows) != 28 for rows in lanes.values()):
            raise ValueError("accepted prior pass geometry differs")
        expected[case] = {lane: rows[-1]["cut"] for lane, rows in lanes.items()}
    provenance = dict(source_job=SOURCE_JOB, source_request_sha256=SOURCE_REQUEST, native_manifest_sha256=SOURCE_MANIFEST,
        source_forecast_seal_sha256=SOURCE_SEAL, source_worker_sha256=SOURCE_WORKER,
        source_executed_payload_sha256=job["payload_custody"]["executed"]["sha256"], source_job_record_sha256=sha(native / "job.json"),
        audit_job=audit_job["job_id"], audit_request_sha256=audit_request["request_sha256"], audit_sha256=AUDIT_SHA256,
        compact_files_manifest_sha256=COMPACT_MANIFEST, compact_archive_sha256=report["archive_sha256"],
        compact_report_sha256=sha(collection / "report.json"), compact_archive_custody_sha256=sha(collection / "archive-custody.json"),
        audit_executed_payload_sha256=audit_job["payload_custody"]["executed"]["sha256"], state_index_sha256=sha(compact / "state-index.json"))
    return original, compact, expected, provenance


def validate_installed(original, planning, compact):
    from mdb_lanes import parameter_artifacts
    from mdb_lanes.payload import build_runtime_wheel, runtime_distributions

    if sha(HERE / WORKER) != WORKER_SHA256:
        raise ValueError("qualified remaining worker differs")
    for name in HELPERS:
        if sha(HERE / name) != original["config"]["helper_source_sha256"][name]:
            raise ValueError("qualified frozen helper differs")
    checks = {"native_source_sha256": ("diffusers", "diffusers/models/transformers/transformer_flux2.py")}
    decoder_paths = {"autoencoder_kl_flux2.py": "diffusers/models/autoencoders/autoencoder_kl_flux2.py",
        "vae.py": "diffusers/models/autoencoders/vae.py", "resnet.py": "diffusers/models/resnet.py",
        "attention_processor.py": "diffusers/models/attention_processor.py", "image_processor.py": "diffusers/image_processor.py",
        "flux2_image_processor.py": "diffusers/pipelines/flux2/image_processor.py", "pipeline_flux2_klein.py": "diffusers/pipelines/flux2/pipeline_flux2_klein.py"}
    decoder = read(planning / "decoder-source-sha256.json")
    paths = {name: ("diffusers", relative) for name, relative in decoder_paths.items()}
    values = {}
    for name, (distribution, relative) in {**checks, **paths}.items():
        path = Path(importlib.metadata.distribution(distribution).locate_file(relative)).resolve()
        if path.is_relative_to(HERE.parents[3]) or sha(path) != (decoder[name] if name in decoder else original["config"][name]):
            raise ValueError("installed original native source differs: " + name)
        values[name] = dict(path=str(path), sha256=sha(path))
    contract = read(compact / "source-native-contract.json")["component_source_sha256"]
    component_paths = {"flux2.py": ("saturn-pub", "saturn_pub/adapters/flux2.py"),
        "scheduling_flow_match_euler_discrete.py": ("diffusers", "diffusers/schedulers/scheduling_flow_match_euler_discrete.py"),
        "normalization.py": ("diffusers", "diffusers/models/normalization.py"), "embeddings.py": ("diffusers", "diffusers/models/embeddings.py"),
        "attention_dispatch.py": ("diffusers", "diffusers/models/attention_dispatch.py")}
    for name, (distribution, relative) in component_paths.items():
        path = Path(importlib.metadata.distribution(distribution).locate_file(relative)).resolve()
        if sha(path) != contract[name]:
            raise ValueError("installed component source differs: " + name)
        values[name] = dict(path=str(path), sha256=sha(path))
    manifests = {}
    for name, path in (("transformer", compact / "parameter-artifacts.json"), ("vae", planning / "vae-parameter-artifacts.json")):
        manifest = parameter_artifacts.validate_manifest(read(path))
        if len(manifest["artifacts"]) != (169 if name == "transformer" else 142):
            raise ValueError("exact original recipe count differs: " + name)
        manifests[name] = sha(path)
    if (sha(compact / "retrieve_weights.py") != "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
            or sha(parameter_artifacts.__file__) != sha(compact / "retrieve_weights.py")
            or manifests["vae"] != "beb6b6ac566aa30adea28861fa1a8e92153d3c6627989d8df35f89d193e3cae9"
            or sha(planning / "vae-config.json") != "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948"):
        raise ValueError("qualified standalone retriever/VAE metadata differs")
    wheel_hashes = {}
    with tempfile.TemporaryDirectory(prefix="mdb-remaining-wheel-check-") as temporary:
        for package, distribution, version in runtime_distributions(require_installed=True):
            wheel = build_runtime_wheel(temporary, package=package, distribution=distribution, version=version)
            wheel_hashes[wheel.name] = sha(wheel)
    if wheel_hashes != original["runtime_wheels"]:
        raise ValueError("installed public wheel cohort differs from accepted producer")
    return dict(native_sources=values, runtime_wheels=wheel_hashes, manifests=manifests, model_calls=0, original_retriever_sha256=sha(parameter_artifacts.__file__))


def stage(run, campaign, planning):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    run, campaign, planning = (Path(path).resolve() for path in (run, campaign, planning))
    original, compact, expected, provenance = authenticate(campaign)
    bind(original["execution_environment"])
    installed = validate_installed(original, planning, compact)
    selection = remote_closures(expected)
    inputs, input_files = selection["forecast"]["inputs"], selection["forecast"]["files"]
    references, reference_files = selection["native"]["inputs"], selection["native"]["files"]
    if any(not name.startswith("prediction-store/") for name in input_files) or any(not name.startswith("native-store/") for name in reference_files):
        raise ValueError("source allowlists cross execution lanes")
    run.mkdir(parents=True, exist_ok=False)
    save(run / "source-selection.json", selection)
    save(run / "installed-cohort-validation.json", installed)
    copies = {"binding.json": compact / "binding.json", "parameter-artifacts.json": compact / "parameter-artifacts.json",
              "retrieve_weights.py": compact / "retrieve_weights.py", "transformer-config.json": compact / "transformer-config.json",
              "scheduler-config.json": compact / "scheduler-config.json", "vae-config.json": planning / "vae-config.json",
              "vae-parameter-artifacts.json": planning / "vae-parameter-artifacts.json"}
    for name, path in copies.items():
        shutil.copyfile(path, run / name)
    old = original["config"]
    config = dict(schema="mdb-flux-functional-remaining-plan-v1", input_root=REMOTE, inputs=inputs, input_files=input_files,
        reference_root=REMOTE, references=references, reference_files=reference_files, expected_source_model_identity=selection["model_identity"],
        parameter_root=old["parameter_root"], binding="binding.json", binding_sha256=sha(run / "binding.json"),
        parameter_manifest="parameter-artifacts.json", parameter_manifest_sha256=sha(run / "parameter-artifacts.json"),
        retrieval_script="retrieve_weights.py", retrieval_script_sha256=sha(run / "retrieve_weights.py"),
        transformer_config="transformer-config.json", transformer_config_sha256=sha(run / "transformer-config.json"),
        scheduler_config="scheduler-config.json", scheduler_config_sha256=sha(run / "scheduler-config.json"),
        vae_config="vae-config.json", vae_config_sha256=sha(run / "vae-config.json"),
        vae_parameter_manifest="vae-parameter-artifacts.json", vae_parameter_manifest_sha256=sha(run / "vae-parameter-artifacts.json"),
        vae_parameter_root="/mnt/big/llm-models/FLUX.2-klein-4B/vae", decoder_source_sha256=read(planning / "decoder-source-sha256.json"),
        native_source_sha256=old["native_source_sha256"], helper_source_sha256=old["helper_source_sha256"],
        device="cuda", source_reduction_policy=True, cpu_threads=4, max_wall_seconds=1200, witness_values=16, height=512, width=512,
        transport_limits=dict(max_expanded_bytes=6 << 30, max_member_bytes=128 << 20), output_scratch="/mnt/big/mdb/research",
        output_name="functional-remaining-result", directory_custody=True,
        source_provenance=dict(provenance, vae_manifest_custody_sha256=sha(planning / "vae-manifest-custody-v1.json"),
            decoder_recipe_generator_sha256=sha(planning / "build_vae_manifest.py"), reference_read_policy="After global image+latent seal only; separate prior native-reference S2 history.",
            first_future_occurrence="step2/joint0 especially fixed new numeric case; step2/project known prior consumer control."),
        scope="Four unchanged archived PREDICTED S2 commits; two successive own-state passes/case;224functional transitions and4real VAE consumers globally seal before4separate prior native-reference input reads,224native transitions and4reference VAE consumers. No inner refresh, new prompt, input rederivation, independent VAE/kernel or speed claim. Preserve all misses and signed responses.")
    local_config = dict(config)
    for key in ("binding", "parameter_manifest", "retrieval_script", "transformer_config", "scheduler_config", "vae_config", "vae_parameter_manifest"):
        local_config[key] = str(run / config[key])
    worker = runpy.run_path(str(HERE / WORKER))
    worker["helpers"](local_config)
    declared = worker["contract"](local_config)
    if declared["forecast"]["transitions"] != 224 or declared["native"]["transitions"] != 224:
        raise ValueError("fixed remaining geometry differs")
    save(run / "plan.json", config)
    save(run / "count-contract.json", declared)
    extras = [(HERE / name, name) for name in HELPERS] + [(run / name, name) for name in copies]
    preflight = run / "preflight.v1.json"
    save(preflight, dict(schema_version="mrun-preflight-config-v1", hashed_roots=[WORKER, *[name for _, name in extras], preflight.name],
        path_token_allowlist=[REMOTE, config["parameter_root"], config["vae_parameter_root"], config["output_scratch"], config["output_name"]]))
    extras.append((preflight, preflight.name))
    request = dict(schema="mdb-flux-functional-remaining-request-v1", config=config,
        execution_environment=original["execution_environment"], scheduler_url=original["scheduler_url"],
        model="flux2-klein-4b-distilled", task_family="analysis", backend="torch-native-functional", device="cuda", dtype="bfloat16",
        flux_mode="capture", capture_sites=True, height=512, width=512, steps=2,
        workload_geometry=dict(model_loads=1, decoder_loads=1, peak_physical_batch=1, image_tokens=1024, text_tokens=512,
            functional_branches=4, native_branches=4, diffusion_steps_per_branch=2, functional_transitions=224, native_transitions=224,
            functional_sdpa=200, native_sdpa=200, functional_linear=872, native_linear=872, real_decoder_consumers=8, physical_component_calls=456, output_bytes_bound=6 << 30),
        needs={"cuda": True, "host": "beast"}, runtime_wheels=original["runtime_wheels"],
        source_hashes={name: sha(path) for path, name in [(HERE / WORKER, WORKER), *extras]}, launcher_sha256=sha(__file__),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=232, logical_branches=4, logical_checkpoint_replays=4, max_local_work_items=256,
            extra={"task_family": "analysis", "functional_transitions": 224, "native_transitions": 224, "real_decoder_consumers": 8, "decoder_loads": 1, "physical_component_calls": 456, "logical_operation_unit": "224 matched forecast/native boundary comparison sites plus8real decoder consumers; physical calls remain456"},
            checkpoint_policy="One finite resident-transformer plus decoder CUDA lease. Four own-state futures and four real predicted-latent decoder outputs globally seal before reference input reads and all native future targets; four reference decodes last. No retry/nested/per-case submissions."),
        resource_exception=dict(ram_mb=8192, vram_mb=13000, cpu_threads=4, disk_gb=8, est_wall_s=600, timeout_s=1500,
            calibration="Measured same original transformer/typed-state/witness worker job-027b22769047:RSS2078.7MB,VRAM8320MB,209.77s for112+112transitions+8projects. This doubles denoiser horizon with sequentialB1 and streams states; no all-GPU cut retention. Adds99.24MB original decoder payload, no unused encoder/quant weights. Heavier original VAE campaign job-4350531b1dcd peaked12936MBVRAM. RAM8192MB/VRAM13000MB allow phase workspaces/streamed fullstate export; raw<=6GiB remote,disk8GB.600s estimate/1200s science/1500s outer,no speed claim."))
    if len(QUEUE_NOTE) > 500:
        raise ValueError("finite queue note exceeds scheduler ABI")
    request["request_sha256"] = request_identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, additional_files=extras)
    validate_payload(run, request)
    save(run / "staged-command.json", dict(command=command(request), request_sha256=request["request_sha256"]))
    return dict(request_sha256=request["request_sha256"], input_files=len(input_files), reference_files=len(reference_files), submitted=False)



def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    if len(QUEUE_NOTE) > 500:
        raise ValueError("scheduler queue note exceeds its500-character ABI")
    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_payload(run, request)
    if (run / "submission.json").exists():
        raise ValueError("functional-remaining run already submitted")
    review = read(run / "finite-run-review-v1.json")
    if (review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != []
            or review.get("review_status") != "ready-for-finite-run"):
        raise ValueError("request-bound finite-run review required")
    args = command(request)
    save(run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"], review_sha256=sha(run / "finite-run-review-v1.json")))
    resource = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-functional-remaining", task="analysis", model=None,
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
        parser.error("accepted functional-pass campaign and remaining planning directory required")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.campaign, args.planning)))
