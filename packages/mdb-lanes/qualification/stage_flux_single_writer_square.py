"""Stage a source-pinned paired H/SwiGLU restoration and one-block continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_single_writer_square.py"
HELPERS = ("flux_image_tail_microscope.py", "flux_source_receiver.py", "flux_single_stack_trace.py")
QUEUE_NOTE = ("Paired single18 HxMLP:3source calls;8 native-to_out forecasts sealed before8 full targets,"
              " then8 single19 consumers=19blocks/19SDPA/46linear. HH/HL ownR/mod, full-token concat"
              " substitution, real projection returns. B1 image1024/text512,2 original banks,"
              " zero fullmodel/prefix/encode/noise/scheduler/VAE.19cuts~900MBremote,compact audit only."
              " RAM8GB/VRAM3GB;180s estimate/600s science/900s outer,no retry.")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def basis():
    return runpy.run_path(str(HERE / "stage_flux_image_tail_microscope.py"))


def request_identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def select_inputs(root, compact):
    """Read-only remote verification of all selected typed-cut members."""
    index, manifest = read(compact / "cut-index.json"), read(compact / "source-files.json")
    expected = {f"boundaries/{role}/{point:02d}.json": index[str(point)][role]
                for role in ("sham", "HH", "HL") for point in (18, 19, 20)}
    script = '''import json,hashlib
from pathlib import Path
root=Path(%r); expected=json.loads(%r); manifest=json.loads((root/'files.json').read_text())
files={}; boundaries={}
def admit(name):
 p=root/name; r=manifest['files'][name]
 assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
 assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 files[name]=r
for name,row in expected.items():
 admit(name); actual=json.loads((root/name).read_text()); assert actual==row; boundaries[name]=actual
 cutname='typed-store/cuts/'+actual['cut']+'.json';admit(cutname);cut=json.loads((root/cutname).read_text())
 assert cut['fingerprint']==actual['cut'] and cut['execution_point']==actual['point']
 for page in cut['slots'].values():admit('typed-store/pages/'+page+'.json')
 for blob in cut['physical_blobs']:admit('typed-store/blobs/'+blob+'.safetensors')
for name in ('plan.json','source.json','summary.json','native-contract.json','binding.json','source-binding.json','parameter-artifacts.json','retrieve_weights.py'):admit(name)
print(json.dumps(dict(files=files,boundaries=boundaries,manifest_sha256=hashlib.sha256((root/'files.json').read_bytes()).hexdigest())))
''' % (str(root), json.dumps(expected, sort_keys=True))
    actual = json.loads(subprocess.run(["ssh", "-o", "BatchMode=yes", "beast", "python3", "-I", "-"],
        input=script, text=True, capture_output=True, check=True, timeout=90).stdout)
    if actual["manifest_sha256"] != sha(compact / "source-files.json") or actual["boundaries"] != expected:
        raise ValueError("native selected source/compact joins differ")
    if any(manifest["files"][name] != record for name, record in actual["files"].items()):
        raise ValueError("selected native inventory differs")
    inputs = {role: dict(store="typed-store", cut=index["18"][role]["cut"],
        entry_metadata=f"boundaries/{role}/18.json", exit_metadata=f"boundaries/{role}/19.json", final_metadata=f"boundaries/{role}/20.json")
        for role in ("sham", "HH", "HL")}
    return inputs, actual


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
        raise ValueError("writer request/launcher changed")
    if sha(HERE / "stage_flux_single_stack_trace.py") != request["source_launcher_sha256"]:
        raise ValueError("qualified trace launcher changed")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("writer staged payload differs: " + name)


def stage(run, source_run, compact):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    run, source_run, compact = (Path(path).resolve() for path in (run, source_run, compact))
    original, job = read(source_run / "request.json"), read(source_run / "job.json")
    runpy.run_path(str(HERE / "stage_flux_single_stack_trace.py"))["validate_payload"](source_run, original)
    collector = HERE / "collect_flux_initial_noise.py"
    if sha(collector) != "e59cb6e3c1f4f0b03f0163f4db8544952def504090a3fdded8e61f530880b9ad":
        raise ValueError("qualified actual-job validator differs")
    runpy.run_path(str(collector))["validate_job"](job, job["job_id"], original)
    if job["job_id"] != "job-a6c2b6aae21e" or job["state"] != "succeeded":
        raise ValueError("succeeded native single-stack source required")
    for name, digest in read(compact / "files.json").items():
        if sha(compact / name) != digest:
            raise ValueError("trace compact source changed: " + name)
    audit = read(compact / "audit.json")
    if (audit["job"] != job["job_id"] or audit["native_request_sha256"] != original["request_sha256"]
            or audit["native_manifest_sha256"] != sha(compact / "source-files.json")):
        raise ValueError("trace audit/native identity differs")
    order = audit["localization"]["HH_selective_signed_increment_ordering"]["image"]
    if order[0]["after_block"] != 18:
        raise ValueError("declared exploratory selection no longer matches source evidence")
    basis()["bind"](original["execution_environment"])
    root = Path(original["config"]["output_scratch"]) / job["job_id"] / "binding-minimal-result"
    inputs, selection = select_inputs(root, compact)
    base = original["config"]
    config = {key: base[key] for key in ("module_config", "parameter_root", "expected_model_identity", "native_source_sha256", "source_reduction_policy")}
    config.update(schema="mdb-single-writer-square-plan-v1", selected_block=18, suffix_block=19, block_count=20,
        inputs=inputs, input_root=str(root), input_files=selection["files"], device="cuda", cpu_threads=4,
        binding="binding.json", binding_sha256=sha(compact / "binding.json"),
        source_binding="source-binding.json", source_binding_sha256=sha(compact / "source-binding.json"),
        parameter_manifest="parameter-artifacts.json", parameter_manifest_sha256=sha(compact / "parameter-artifacts.json"),
        retrieval_script="retrieve_weights.py", retrieval_script_sha256=sha(compact / "retrieve_weights.py"),
        helper_source_sha256={name: sha(HERE / name) for name in HELPERS},
        source_provenance=dict(job_id=job["job_id"], scheduler_status=job["state"], request_sha256=original["request_sha256"],
            manifest_sha256=audit["native_manifest_sha256"], source_job_record_sha256=sha(source_run / "job.json"),
            executed_payload_sha256=job["payload_custody"]["executed"]["sha256"],
            source_audit_sha256=sha(compact / "audit.json"), source_compact_manifest_sha256=sha(compact / "files.json"),
            inherited_recovered_source=base["source_provenance"]),
        selection=dict(kind="exploratory-then-frozen", criterion="Largest HH-minus-HL signed increment in absolute image-response RMS, not maximal relative or intrinsic gain.",
            selected=order[0], source_audit_sha256=sha(compact / "audit.json"),
            countertrend="HH image-relative response falls across18, final HL text RMS exceeds HH."),
        predeclared_expectations=dict(joint_restoration="Restoring both full-token H/M to sham reduces HH-minus-HL excess absolute image growth at local and final boundaries; retain misses.",
            numerical="Eight complete local I/T forecasts from observed H/M/R/gate using native to_out and BF16 multiply/add; seal before all mixed target entries. Retain numerical misses.",
            controls="Three source calls replay their own trace exits; own/own11 local and final corners replay HH/HL trace. 00 keeps own residual and is not historical sham.",
            exploratory="H/M conditional roles, tensor interaction, relative response, text collateral and one-block continuation effects; no component winner assumed."),
        max_wall_seconds=600, output_scratch="/mnt/big/mdb/research", output_name="binding-minimal-result", directory_custody=True,
        scope="Paired own/sham full-token attention×activated-SwiGLU square at HH and HL step1/single18 recipients. Three observed source component calls, eight sealed original-primitive local forecasts, eight full native targets with declared pre-to_out concat substitutions, then eight actual single19 consumers. Own residual/modulation/context retained; no full model/prefix/encode/noise/scheduler/VAE, independent-kernel or final numerical-forecast claim.")
    runpy.run_path(str(HERE / WORKER))["validate_config"](config)
    run.mkdir(parents=True, exist_ok=False)
    save(run / "source-selection.json", selection)
    copies = ("binding.json", "source-binding.json", "parameter-artifacts.json", "retrieve_weights.py")
    for name in copies:
        shutil.copyfile(compact / name, run / name)
    save(run / "plan.json", config)
    extras = [(HERE / name, name) for name in HELPERS] + [(run / name, name) for name in copies]
    preflight = run / "preflight.v1.json"
    save(preflight, dict(schema_version="mrun-preflight-config-v1", hashed_roots=[WORKER, *[name for _, name in extras], preflight.name],
        path_token_allowlist=[config["input_root"], config["parameter_root"], config["output_scratch"], config["output_name"]]))
    extras.append((preflight, preflight.name))
    request = dict(schema="mdb-single-writer-square-request-v1", config=config,
        execution_environment=original["execution_environment"], scheduler_url=original["scheduler_url"],
        model="flux2-klein-4b-distilled", task_family="analysis", backend="torch-native-component-primitives", device="cuda", dtype="bfloat16",
        needs={"cuda": True, "host": "beast"}, runtime_wheels=original["runtime_wheels"],
        source_hashes={name: sha(path) for path, name in [(HERE / WORKER, WORKER), *extras]},
        launcher_sha256=sha(__file__), source_launcher_sha256=sha(HERE / "stage_flux_single_stack_trace.py"),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=27,
            logical_branches=8, logical_checkpoint_replays=3, max_local_work_items=32,
            extra={"task_family": "analysis", "source_blocks": 3, "mixed_blocks": 8, "suffix_blocks": 8, "forecast_to_out": 8},
            checkpoint_policy="One finite CUDA lease, original18/19banks; three source calls, eight local forecasts globally sealed before eight mixed targets, eight one-block native continuations; no retries or nested jobs."),
        resource_exception=dict(ram_mb=8192, vram_mb=3072, cpu_threads=4, disk_gb=4, est_wall_s=180, timeout_s=900,
            calibration="Tracejob-a6c2b6aae21e:60singles,608MBstates,59.47selapsed,RSS2308.6MB/VRAM1088MB. This has19singles+8outputlinears, two245MBoriginalbanks insequence, threeobservedH/M/Rfrontiers~283MB, forecasts151MB, dedupactualBF16operands~264MB,16typedoutputs151MB plus3observedsourcecuts<=28.3MB (19cuts,~850–900MBraw). One7.75GBhash+8tensorreads. RepeatedpublicsealverificationaddsIO; declared8192/3072MB retains measured-component headroom, notspeedup.180sestimate/600sscience/900souter."))
    request["request_sha256"] = request_identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, additional_files=extras)
    validate_payload(run, request)
    save(run / "staged-command.json", dict(command=command(request), request_sha256=request["request_sha256"]))
    return dict(request_sha256=request["request_sha256"], source_files=len(selection["files"]), submitted=False)


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    if len(QUEUE_NOTE) > 500:
        raise ValueError("scheduler queue note exceeds its 500-character ABI")
    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_payload(run, request)
    if (run / "submission.json").exists():
        raise ValueError("writer run already submitted")
    review = read(run / "finite-run-review-v1.json")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("request-bound finite-run review required")
    args = command(request)
    save(run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"], review_sha256=sha(run / "finite-run-review-v1.json")))
    resource = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-single-writer-square", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resource[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True,
        note=QUEUE_NOTE)
    save(run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-compact", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and (args.source_run is None or args.source_compact is None):
        parser.error("completed trace run and compact source required")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.source_run, args.source_compact)))
