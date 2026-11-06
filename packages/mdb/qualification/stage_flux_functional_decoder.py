"""Authenticate and stage one four-endpoint decoder replication; never auto-submit."""
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
WORKER = "flux_functional_decoder_campaign.py"
WORKER_SHA256 = "58525f9f856c2dd6e56ecf4a2a0d1489196e5f33e3c1cbef3f89c8bcdf517300"
HELPERS = {"flux_functional_decoder.py": "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f",
    "flux_functional_vae.py": "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61",
    "flux_source_receiver.py": "77c0c717dd82afe7c039a3179bf0f34e2f44d77ed3fd14fb0c87893946c11dc2",
    "flux_image_tail_microscope.py": "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926"}
SOURCE_JOB = "job-f7d249b6d3bb"
SOURCE_REQUEST = "00d08d3eb058fa50656b0f3deb23238e9ad38a69a2266751462b80ee2ddce93a"
SOURCE_MANIFEST = "c6baff2a5aae8c03651399226f526794e7c9df3535748f365f101586bbfab41d"
SOURCE_SEAL = "46a1266921696c0b927d1d9394f515565dd00e8077eb2e6d68e1ebc1403ba19e"
SOURCE_WORKER = "71e2a82f47f3a29cc8195589ee20018f74e880cc6696fa47ebc1e04e78aa993e"
AUDIT_SHA256 = "52bd05dc066f1295db32f9baa36986662417b74f1fdbad487d9d19378512d0a9"
AUDITOR_SHA256 = "5027875c03f9ceb3157b4b9d9632366ff5c9b1c4beb429b8b73d79aad0db6b08"
COMPACT_MANIFEST = "61e92fc1540bd7ffb2209ed603a786f6513114d70c1eb10261c324f410c7e9b7"
REMOTE = "/mnt/big/mdb/research/job-f7d249b6d3bb/functional-remaining-result"
QUEUE_NOTE = ("Decoder replication:4 archived predictedS4 inputs;4 functional decodes/full macro states+micro hashes seal before4 same-latent native decodes. B1,142 original recipes,no transformer/encode/noise. At most1 hash-matched diagnostic prefix/case afteralltargets,no extra native. Macro2.35GB;raw<=6GiB,32MiB chunks. RAM8GiB/VRAM10GiB CPU4,600s estimate/1200s worker/1500s outer;one lease,no retry.")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def request_identity(request):
    return hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()


def bind(environment):
    from mdb.execution_environment import intent

    for key, name in {"uv_cache_dir": "MDB_WORKER_UV_CACHE_DIR", "torch_home": "MDB_WORKER_TORCH_HOME", "extra_model_roots": "MDB_WORKER_MODEL_ROOTS", "python": "MDB_WORKER_PYTHON", "uv_link_mode": "MDB_WORKER_UV_LINK_MODE"}.items():
        os.environ[name] = environment[key]
    if intent() != {key: environment[key] for key in intent()}:
        raise ValueError("installed execution requirements differ from qualified cohort")
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"


def source_job_config(job, original):
    config = job["config"]
    extra = {"mrun_guarded_request_sha256", "retry_on_kill", "saturn_request_sha256"}
    return (set(config) == set(original) | extra and {k: v for k, v in config.items() if k not in extra} == original
        and config["retry_on_kill"] is False and config == job["submitted_request"]["config"]
        and job["admission"]["idempotency_key"] == "mrun:" + config["mrun_guarded_request_sha256"])


def authenticate(campaign):
    producer, collection = campaign / "run-02", campaign / "compact-collection-01"
    compact = collection / "extracted/binding-minimal-result"
    original, job = read(producer / "request.json"), read(producer / "job.json")
    audit_request, audit_job = read(collection / "request.json"), read(collection / "job.json")
    report, custody = read(collection / "report.json"), read(collection / "archive-custody.json")
    if (original["request_sha256"] != SOURCE_REQUEST or request_identity(original) != SOURCE_REQUEST
        or original["source_hashes"]["flux_functional_remaining.py"] != SOURCE_WORKER or not source_job_config(job, original)
        or job["job_id"] != SOURCE_JOB or job["state"] != "succeeded" or job["result"]["returncode"] != 0):
        raise ValueError("accepted parent producer differs")
    for record in (job, audit_job):
        c = record["payload_custody"]
        if not c["declared"]["sha256"] == c["sealed"]["sha256"] == c["executed"]["sha256"] == record["result"]["executed_payload"]["sha256"]:
            raise ValueError("parent executed payload custody differs")
    if (audit_job["job_id"] != "job-453a53b4ec8c" or audit_job["state"] != "succeeded" or audit_job["result"]["returncode"] != 0
        or not source_job_config(audit_job, audit_request) or request_identity(audit_request) != audit_request["request_sha256"]
        or report["request_sha256"] != audit_request["request_sha256"] or report["job_id"] != audit_job["job_id"]
        or report["files_manifest_sha256"] != COMPACT_MANIFEST or not custody["complete"] or custody["job_id"] != audit_job["job_id"]
        or sha(collection / "archive.tar.gz") != custody["archive_sha256"] or custody["archive_sha256"] != report["archive_sha256"]):
        raise ValueError("authenticated parent compact collection differs")
    if sha(compact / "files.json") != COMPACT_MANIFEST:
        raise ValueError("parent compact manifest differs")
    for name, digest in read(compact / "files.json").items():
        p = compact / name
        if p.is_symlink() or not p.resolve().is_relative_to(compact.resolve()) or sha(p) != digest:
            raise ValueError("parent compact member differs: " + name)
    audit, source = read(compact / "audit.json"), read(compact / "source-source.json")
    inventory = read(compact / "source-files.json")
    if (sha(compact / "audit.json") != AUDIT_SHA256 or sha(compact / "source-files.json") != SOURCE_MANIFEST
        or inventory["job_id"] != SOURCE_JOB or inventory["request_sha256"] != SOURCE_REQUEST
        or audit["mechanics_status"] != "authenticated-full-state-saved-evidence-audit" or audit["auditor_sha256"] != AUDITOR_SHA256
        or audit["native_manifest_sha256"] != SOURCE_MANIFEST or audit["native_request_sha256"] != SOURCE_REQUEST
        or source["job_id"] != SOURCE_JOB or source["request_sha256"] != SOURCE_REQUEST or source["worker_sha256"] != SOURCE_WORKER
        or source["forecast_seal"]["seal_sha256"] != SOURCE_SEAL or not audit["transactions"]["forecast"]["verified"]
        or audit["transactions"]["forecast"]["seal_sha256"] != SOURCE_SEAL):
        raise ValueError("accepted parent source/audit binding differs")
    index = read(compact / "state-index.json")
    if set(index) != {"sham", "HH", "HL", "new-3over8"} or any(len(v["forecast"]) != 56 for v in index.values()):
        raise ValueError("fixed fourcase parent horizon differs")
    expected = {case: rows["forecast"][-1]["cut"] for case, rows in index.items()}
    provenance = dict(source_job=SOURCE_JOB, source_request_sha256=SOURCE_REQUEST, native_manifest_sha256=SOURCE_MANIFEST,
        source_forecast_seal_sha256=SOURCE_SEAL, source_worker_sha256=SOURCE_WORKER, source_executed_payload_sha256=job["payload_custody"]["executed"]["sha256"],
        source_job_record_sha256=sha(producer / "job.json"), audit_job=audit_job["job_id"], audit_request_sha256=audit_request["request_sha256"],
        accepted_auditor_sha256=AUDITOR_SHA256, audit_sha256=AUDIT_SHA256, compact_files_manifest_sha256=COMPACT_MANIFEST,
        compact_archive_sha256=report["archive_sha256"], compact_report_sha256=sha(collection / "report.json"),
        compact_archive_custody_sha256=sha(collection / "archive-custody.json"), state_index_sha256=sha(compact / "state-index.json"),
        audit_executed_payload_sha256=audit_job["payload_custody"]["executed"]["sha256"])
    return original, compact, expected, provenance


def remote_closures(expected):
    """Authenticate four predicted cut closures remotely, returning metadata only."""
    script = r'''import hashlib,json
from pathlib import Path
root=Path(ROOT);inventory=json.loads((root/'files.json').read_text());assert hashlib.sha256((root/'files.json').read_bytes()).hexdigest()==MANIFEST
verified={};inputs={};metadata={};identities=set()
def admit(name):
 if name in verified:return
 p=root/name;r=inventory['files'][name];s=p.stat();assert p.resolve().is_relative_to(root.resolve()) and not p.is_symlink() and s.st_size==r['bytes']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==r['sha256'],name
 t=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)
 verified[name]=dict(sha256=r['sha256'],bytes=r['bytes'])
for case,fingerprint in EXPECTED.items():
 name='prediction-store/cuts/'+fingerprint+'.json';admit(name);cut=json.loads((root/name).read_text());point=cut['execution_point'];m=cut['manifest'];component=m['execution']['component_pass']
 assert cut['fingerprint']==fingerprint and (point['logical_step'],point['phase'],point['local_clock'])==(4,'project',0)
 assert (component['id'],component['lane'],component['case'])==('mdb-flux-functional-remaining-v1','forecast',case)
 names={name};names.update('prediction-store/pages/'+v+'.json' for v in cut['slots'].values());names.update('prediction-store/blobs/'+v+'.safetensors' for v in cut['physical_blobs'])
 for path in names:admit(path)
 identities.add(m['model_identity']);inputs[case]=dict(store='prediction-store',cut=fingerprint);metadata[case]=dict(cut=fingerprint,parent=cut['logical_parent'],point=point,component_provenance=component)
assert len(identities)==1
print(json.dumps(dict(inputs=inputs,input_files=verified,metadata=metadata,model_identity=identities.pop(),verified_bytes=sum(r['bytes'] for r in verified.values()),tensor_payloads_returned=0,remote_mutations=0),sort_keys=True))
'''
    script = "ROOT=" + repr(REMOTE) + "\nMANIFEST=" + repr(SOURCE_MANIFEST) + "\nEXPECTED=" + repr(expected) + "\n" + script
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "beast", "python3", "-I", "-"], input=script, text=True, capture_output=True, check=True, timeout=120)
    return json.loads(result.stdout)


def validate_installed(original, compact):
    from mdb import parameter_artifacts
    from mdb.payload import build_runtime_wheel, runtime_distributions

    if sha(HERE / WORKER) != WORKER_SHA256 or any(sha(HERE / name) != digest for name, digest in HELPERS.items()):
        raise ValueError("frozen worker/helper source differs")
    graph = runpy.run_path(str(HERE / "flux_functional_decoder.py"))
    native = graph["source_contract"]()
    old = original["config"]
    upstream = {name: old["decoder_source_sha256"][name] for name in ("pipeline_flux2_klein.py", "flux2_image_processor.py")}
    for name, relative in {"pipeline_flux2_klein.py": "diffusers/pipelines/flux2/pipeline_flux2_klein.py", "flux2_image_processor.py": "diffusers/pipelines/flux2/image_processor.py"}.items():
        p = Path(importlib.metadata.distribution("diffusers").locate_file(relative)).resolve()
        if p.is_relative_to(HERE.parents[3]) or sha(p) != upstream[name]:
            raise ValueError("installed original upstream source differs")
    manifest = parameter_artifacts.validate_manifest(read(compact / "vae-parameter-artifacts.json"))
    if (len(manifest["artifacts"]) != 142 or sha(compact / "vae-parameter-artifacts.json") != "beb6b6ac566aa30adea28861fa1a8e92153d3c6627989d8df35f89d193e3cae9"
        or sha(compact / "vae-config.json") != "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948"
        or sha(compact / "retrieve_weights.py") != "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
        or sha(parameter_artifacts.__file__) != sha(compact / "retrieve_weights.py")):
        raise ValueError("original142 recipes/config/retriever differ")
    wheels = {}
    with tempfile.TemporaryDirectory(prefix="mdb-decoder-wheel-check-") as temporary:
        for package, distribution, version in runtime_distributions(require_installed=True):
            p = build_runtime_wheel(temporary, package=package, distribution=distribution, version=version)
            wheels[p.name] = sha(p)
    if wheels != original["runtime_wheels"]:
        raise ValueError("installed public wheels differ from accepted parent")
    return dict(native_contract=native, upstream_source_sha256=upstream, runtime_wheels=wheels, model_calls=0)


def command(request):
    from mdb.payload import worker_command

    bind(request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in (("tmpdir", "TMPDIR"), ("cuda_cache_path", "CUDA_CACHE_PATH"), ("torchinductor_cache_dir", "TORCHINDUCTOR_CACHE_DIR")):
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def validate_payload(run, request):
    if sha(__file__) != request["launcher_sha256"] or request_identity(request) != request["request_sha256"]:
        raise ValueError("decoder request/launcher changed")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("decoder staged payload differs: " + name)


def stage(run, campaign):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    run, campaign = Path(run).resolve(), Path(campaign).resolve()
    original, compact, expected, provenance = authenticate(campaign)
    bind(original["execution_environment"])
    installed = validate_installed(original, compact)
    selection = remote_closures(expected)
    if any(not name.startswith("prediction-store/") for name in selection["input_files"]):
        raise ValueError("only predicted finalS4 closure allowed")
    run.mkdir(parents=True, exist_ok=False)
    save(run / "source-selection.json", selection)
    save(run / "installed-cohort-validation.json", installed)
    copies = {"vae-config.json": compact / "vae-config.json", "parameter-artifacts.json": compact / "vae-parameter-artifacts.json", "retrieve_weights.py": compact / "retrieve_weights.py"}
    for name, p in copies.items():
        shutil.copyfile(p, run / name)
    config = dict(schema="mdb-functional-decoder-plan-v1", input_root=REMOTE, inputs=selection["inputs"], input_files=selection["input_files"],
        expected_source_model_identity=selection["model_identity"], parameter_root=original["config"]["vae_parameter_root"],
        helper_source_sha256=HELPERS, upstream_source_sha256=installed["upstream_source_sha256"],
        vae_config="vae-config.json", vae_config_sha256=sha(run / "vae-config.json"), parameter_manifest="parameter-artifacts.json", parameter_manifest_sha256=sha(run / "parameter-artifacts.json"),
        retrieval_script="retrieve_weights.py", retrieval_script_sha256=sha(run / "retrieve_weights.py"), device="cuda", source_reduction_policy=True,
        cpu_threads=4, max_wall_seconds=1200, witness_values=16, chunk_bytes=32 << 20, height=512, width=512,
        transport_limits=dict(max_expanded_bytes=6 << 30, max_member_bytes=128 << 20), output_scratch="/mnt/big/mdb/research", output_name="functional-decoder-result", directory_custody=True,
        source_provenance=provenance, scope="Replication of four already-observed decoder endpoints from unchanged sealed predicted finalS4 cuts. Explicit original shared Torch primitives; full macro arrays and primitive hashes/fixed witnesses. Four forecasts globally seal before four same-latent native decodes. Optional first-micro hash-matched prefix reconstruction only after all targets; no independent kernel or pretarget full micro retention claim.")
    local = dict(config, vae_config=str(run / "vae-config.json"))
    declared = runpy.run_path(str(HERE / WORKER))["geometry"](local)
    save(run / "plan.json", config)
    save(run / "geometry.json", declared)
    extras = [(HERE / name, name) for name in HELPERS] + [(run / name, name) for name in copies]
    preflight = run / "preflight.v1.json"
    save(preflight, dict(schema_version="mrun-preflight-config-v1", hashed_roots=[WORKER, *[name for _, name in extras], preflight.name],
        path_token_allowlist=[REMOTE, config["parameter_root"], config["output_scratch"], config["output_name"]]))
    extras.append((preflight, preflight.name))
    request = dict(schema="mdb-functional-decoder-request-v1", config=config, execution_environment=original["execution_environment"], scheduler_url=original["scheduler_url"],
        model="flux2-klein-4b-distilled", task_family="analysis", backend="torch-native-functional-decoder", device="cuda", dtype="bfloat16", flux_mode="capture", capture_sites=True, height=512, width=512, steps=1,
        workload_geometry=dict(model_loads=1, decoder_loads=1, transformer_loads=0, peak_physical_batch=1, functional_decodes=4, native_decodes=4,
            conditional_functional_prefixes_bound=4, original_recipe_entries=142, full_macro_bytes=declared["eight_lane_macro_bytes"], output_bytes_bound=6 << 30),
        needs={"cuda": True, "host": "beast"}, runtime_wheels=original["runtime_wheels"], source_hashes={name: sha(p) for p, name in [(HERE / WORKER, WORKER), *extras]}, launcher_sha256=sha(__file__),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=8, logical_branches=4, logical_checkpoint_replays=4, max_local_work_items=256,
            extra=dict(task_family="analysis", decoder_loads=1, transformer_loads=0, logical_operation_unit="four sealed complete functional decodes plus four matched native decoder consumers", conditional_functional_prefixes_bound=4),
            checkpoint_policy="One finite CUDA B1 decoder-only lease. All four forecasts seal before any native target; at most one unchanged diagnostic prefix/case only after all native targets. No retry/nested/per-case submissions."),
        resource_exception=dict(ram_mb=8192, vram_mb=10240, cpu_threads=4, disk_gb=8, est_wall_s=600, timeout_s=1500,
            calibration="Accepted parent job-f7d249b6d3bb measured RSS2453.5MiB/VRAM9222MiB/482.2s for224+224 denoiser transitions and8 native VAE consumers, with7.75GB transformer plus99.24MB decoder resident. This removes transformer/denoising, adds original functional decoder and full macro/hash export; no direct VAE-only workspace measurement claimed.8GiB RAM/10GiB VRAM conservatively bound streaming snapshots/copies. Static macro2,344,615,936B plus bounded conditional diagnostic tensors fit6GiB raw; no allmicro arrays.600s estimate/1200s worker/1500s outer; no speed claim."))
    if len(QUEUE_NOTE) > 500:
        raise ValueError("finite queue note exceeds scheduler ABI")
    request["request_sha256"] = request_identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, additional_files=extras)
    validate_payload(run, request)
    save(run / "staged-command.json", dict(command=command(request), request_sha256=request["request_sha256"]))
    return dict(request_sha256=request["request_sha256"], input_files=len(selection["input_files"]), submitted=False)


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_payload(run, request)
    if (run / "submission.json").exists():
        raise ValueError("decoder run already submitted")
    review = read(run / "finite-run-review-v1.json")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("request-bound finite review required")
    if len(QUEUE_NOTE) > 500:
        raise ValueError("finite queue note exceeds scheduler ABI")
    args = command(request)
    save(run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"], review_sha256=sha(run / "finite-run-review-v1.json")))
    resource = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-functional-decoder", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{k: resource[k] for k in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")}, retry_on_kill=False, preflight=True, note=QUEUE_NOTE)
    save(run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and args.campaign is None:
        parser.error("accepted functional-remaining campaign required")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.campaign)))
