"""Stage initial-noise112 and matched56 forecasts under one finite CUDA lease."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import runpy
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execution_intent():
    from mdb.execution_environment import intent

    result = intent()
    for key, variable, fallback in (
        ("uv_cache_dir", "MDB_WORKER_UV_CACHE_DIR", "/mnt/hdd8tb/mdb/uv-cache"),
        ("torch_home", "MDB_WORKER_TORCH_HOME", "/mnt/hdd8tb/mdb/torch"),
        ("extra_model_roots", "MDB_WORKER_MODEL_ROOTS", "/mnt/big/llm-models"),
    ):
        result[key] = os.environ.get(variable, fallback)
    result.update(tmpdir="/mnt/hdd8tb/mdb/tmp", cuda_cache_path="/mnt/hdd8tb/mdb/cuda-cache",
        torchinductor_cache_dir="/mnt/hdd8tb/mdb/torchinductor-cache")
    if any(not result[k].startswith("/") or result[k].startswith("/Users/")
           for k in ("uv_cache_dir", "torch_home")):
        raise ValueError("Linux CUDA cache paths must be worker-native and absolute")
    return result


def sealed_worker_command(request):
    """Bind actual launch arguments to staged intent despite ambient defaults."""
    from mdb.payload import worker_command

    environment = request["execution_environment"]
    if any(not environment[k].startswith("/") or environment[k].startswith("/Users/")
           for k in ("uv_cache_dir", "torch_home")):
        raise ValueError("Linux CUDA cache paths must be worker-native and absolute")
    command = worker_command("flux_initial_noise_campaign.py")
    if command[0] != "/usr/bin/env":
        raise ValueError("public worker environment entrypoint differs")
    for variable, key in (("TMPDIR", "tmpdir"), ("CUDA_CACHE_PATH", "cuda_cache_path"),
                          ("TORCHINDUCTOR_CACHE_DIR", "torchinductor_cache_dir")):
        value = environment[key]
        if not value.startswith("/mnt/hdd8tb/mdb/") or ".." in Path(value).parts:
            raise ValueError("worker temporary/cache paths must use the admitted HDD scratch root")
        command.insert(1, variable + "=" + value)
    for variable, key in (("UV_LINK_MODE", "uv_link_mode"), ("UV_CACHE_DIR", "uv_cache_dir"),
                          ("TORCH_HOME", "torch_home"), ("LLM_MODELS_EXTRA_ROOT", "extra_model_roots")):
        positions = [i for i, value in enumerate(command) if value.startswith(variable + "=")]
        if len(positions) != 1:
            raise ValueError("public worker environment ABI differs: " + variable)
        command[positions[0]] = variable + "=" + environment[key]
    command[command.index("--python") + 1] = environment["python"]
    requirements = [command[i + 1] for i, value in enumerate(command) if value == "--with"]
    if [value for value in requirements if not value.startswith("./")] != environment["requirements"]:
        raise ValueError("worker dependency requirements differ from staged intent")
    return command



PRIOR_WORKER_SHA256 = "3922894ac1a47b4c92fae1ea1cd2585825332fa23da3c6513a0da4afebe40774"
TRACKED_ANALYSIS_REF = "44a5215"
PRIOR_HELPERS = ("flux_two_pass_campaign.py", "flux_final_latent_campaign.py", "flux_joint_tail_campaign.py", "flux_joint_chain_campaign.py", "flux_reentry_campaign.py", "flux_reentry_reference.py",
    "flux_step_boundary_campaign.py", "flux_step_boundary_reference.py",
    "flux_transition_reference.py", "flux_mlp_replacement.py", "flux_attention_replacement_campaign.py",
    "flux_single_contexts.py")
CORRECTED_ANALYZERS = ()
NEW_ANALYZERS = ("analyze_flux_initial_noise.py", "collect_flux_initial_noise.py")


def scheduler_endpoint(request=None):
    expected = "http://192.168.68.68:9025" if request is None else request["scheduler_url"]
    endpoint = os.environ.setdefault("MRUN_URL", expected)
    if not endpoint.startswith(("http://", "https://")) or (request is not None and endpoint != expected):
        raise ValueError("scheduler endpoint differs from sealed request")
    return endpoint


def closure_module_sha256():
    import inspect

    import mdb.flux_forecast_closure as closure

    return sha(Path(inspect.getfile(closure)))


def joint_closure_module_sha256():
    import inspect

    import mdb.flux_joint_forecast_closure as closure

    return sha(Path(inspect.getfile(closure)))


def tail_closure_module_sha256():
    import inspect

    import mdb.flux_joint_tail_closure as closure

    return sha(Path(inspect.getfile(closure)))


def final_closure_module_sha256():
    import inspect

    import mdb.flux_final_latent_closure as closure

    return sha(Path(inspect.getfile(closure)))


def derived_terminal_module_sha256():
    import inspect

    import mdb.flux_derived_terminal as derivation

    return sha(Path(inspect.getfile(derivation)))


def prior_sources(prior_run):
    """Read immutable qualified payload, never live legacy scientific helpers."""
    request = read(prior_run / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request.get("request_sha256"):
        raise ValueError("qualified prior request seal differs")
    expected = request["source_hashes"]
    result = {}
    for name in PRIOR_HELPERS:
        path = prior_run / "payload" / name
        if name not in expected or sha(path) != expected[name]:
            raise ValueError("qualified prior scientific helper differs: " + name)
        result[name] = path
    if expected["flux_two_pass_campaign.py"] != PRIOR_WORKER_SHA256:
        raise ValueError("prior native project/readout profile differs")
    return result, request


def independent_sources(here, prior_run):
    """Preserve qualified old math; explicitly override the corrected old reader."""
    import subprocess

    request = read(prior_run / "request.json")
    result, authorities = {}, {}
    for name, expected in request["independent_math_sources"].items():
        path = prior_run / "independent-math-freeze" / name
        if sha(path) != expected:
            raise ValueError("prior independent mathematical source differs: " + name)
        result[name] = path.read_bytes()
        authorities[name] = {"authority": "prior request-pinned independent-math-freeze", "sha256": expected}
    for name in CORRECTED_ANALYZERS:
        raw = subprocess.run(["git", "show", TRACKED_ANALYSIS_REF + ":packages/mdb/qualification/" + name],
            cwd=here.parents[2], check=True, capture_output=True).stdout
        result[name] = raw
        authorities[name] = {"authority": "corrected reader, tracked Git " + TRACKED_ANALYSIS_REF,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "original_preoutcome_sha256": request["independent_math_sources"][name]}
    for name in NEW_ANALYZERS:
        result[name] = (here / name).read_bytes()
        authorities[name] = {"authority": "new pre-outcome saved-only analysis/custody source",
            "sha256": hashlib.sha256(result[name]).hexdigest()}
    return result, authorities



def parameter_artifacts_module_sha256():
    import inspect
    from mdb import parameter_artifacts
    return sha(Path(inspect.getfile(parameter_artifacts)))


def storage_plan(config):
    """Explicit complete-field bounds; checkpoint constants are never exported."""
    import numpy as np

    def member(shape, dtype):
        dtype = np.dtype(dtype)
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header,
            {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": shape})
        return {"shape": list(shape), "dtype": str(dtype), "bytes": header.tell()+math.prod(shape)*dtype.itemsize}
    predicted = member((1,1536,3072), "<f4")
    native = member((1,1536,3072), "<u2")
    full, matched = 104*predicted["bytes"], 52*predicted["bytes"]
    allowance = 256 << 20
    limits = config["transport_limits"]
    if (full+allowance > config["full_transaction_max_bytes"]
            or matched+allowance > config["matched_transaction_max_bytes"]
            or config["full_transaction_max_bytes"] > limits["max_member_bytes"]
            or full+matched+112*native["bytes"]+2*allowance > limits["max_expanded_bytes"]):
        raise ValueError("initial-noise complete-field framing exceeds custody bounds")
    return {"schema":"mdb-flux-initial-noise-storage-plan-v1","predicted_register":predicted,
        "native_register":native,"full_output_registers":104,"matched_output_registers":52,
        "full_prediction_bytes_lower_bound":full,"matched_prediction_bytes_lower_bound":matched,
        "native_trajectory_bytes_lower_bound":112*native["bytes"],"parameter_archives_persisted":0,
        "parameter_recipe_count":169,"checkpoint_parameter_bytes_reused":7751089152,
        "framing_and_context_allowance_per_forecast":allowance,
        "full_transaction_max_bytes":config["full_transaction_max_bytes"],
        "matched_transaction_max_bytes":config["matched_transaction_max_bytes"],**limits,
        "scope":"Stream one checkpoint parameter bank at a time. Full field arrays and typed states retained. Lower bounds are sizing diagnostics; actual inventories govern custody."}


def stage(run_dir, *, prior_run=None):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    here = Path(__file__).resolve().parent
    prior_run = Path(prior_run) if prior_run is not None else here.parents[3]/"artifacts/mdb-flux-two-pass-20261105/run-01"
    old, prior_request = prior_sources(prior_run)
    independent, authorities = independent_sources(here, prior_run)
    worker = here/"flux_initial_noise_campaign.py"
    engine = runpy.run_path(str(worker))
    c = engine["contract"]({})
    config = {"schema":"mdb-flux-initial-noise-plan-v1","model":"flux2-klein-4b-distilled",
        "model_path":"/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256":"dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "expected_closure_module_sha256":closure_module_sha256(),
        "expected_joint_closure_module_sha256":joint_closure_module_sha256(),
        "expected_tail_closure_module_sha256":tail_closure_module_sha256(),
        "expected_final_closure_module_sha256":final_closure_module_sha256(),
        "expected_derived_terminal_module_sha256":derived_terminal_module_sha256(),
        "expected_parameter_artifacts_module_sha256":parameter_artifacts_module_sha256(),
        "context":c["context"],"height":512,"width":512,"steps":4,"block_indices":c["block_indices"],
        "logical_steps":c["logical_steps"],"matched_logical_steps":c["matched_logical_steps"],
        "selector":{"logical_step":0,"phase":"project","index":None},
        "matched_selector":{"logical_step":2,"phase":"project","index":None},
        "cpu_threads":4,"query_chunk":64,"projection_chunk":256,"neuron_chunk":256,"retain_fields":False,
        "sample_indices":engine["SAMPLE_INDICES"],"image_sample_indices":engine["IMAGE_SAMPLE_INDICES"],
        "branches":c["branches"],"branch_execution_order":c["order"],"observation_plan":c["observation_plan"],
        "array_storage":"bf16-bits","archive_integer_ids":"native-int64","parameter_storage":"checkpoint-recipe-v1",
        "parameter_checkpoint_file":"transformer/diffusion_pytorch_model.safetensors",
        **{"max_"+key:c["counts"][key] for key in ("adapter_steps","acts","candidate_calls","images","vae_calls")},
        "max_wall_seconds":1800,"full_transaction_max_bytes":3<<30,"matched_transaction_max_bytes":1536<<20,
        "transport_method":"planned-authenticated-ssh-manifest",
        "transport_limits":{"max_expanded_bytes":10<<30,"max_member_bytes":3<<30},
        "hypotheses":{
            "H-full":"Actual packed Z0, encoded conditioning, IDs and exact four-step schedule feed112 own-state mathematical transitions. Seal the complete trajectory and save its predicted PNG before any native denoiser entry.",
            "H-matched":"After56 native transitions, freeze actual Z2 and independently predict56 remaining transitions. Seal and save matched PNG while logicalstep2/3 native occurrences remain zero; prior native history remains visible.",
            "H-order":"Four native VAE images,112 native transitions,3 typed Acts and115 execution receipts. Prediction roots remain independent. Source-derived terminal shells supply no native future values.",
            "H-errors":"Preserve all168 mathematical transitions and112 native boundaries, absolute and native-normalized errors, signed endpoint replacement contrasts and cross terms. No accuracy cutoff or semantic gate."},
        "terminal_status":"unassessed",
        "scope":"One fresh seed113 512px prompt/history, actualZ0 through four-pass final latent.112native/3Acts/115receipts/4VAE,168 independent math transitions,one load/encode/noise. Native prompt encoding and VAE remain outside mathematical prediction; no universal semantics, speedup or raw-prompt-to-RGB claim."}
    engine["validate_config"](config)
    config["storage_plan"] = storage_plan(config)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True,exist_ok=False)
    save(run_dir/"plan.json",config)
    extras = [(path,name) for name,path in old.items()]
    extras.append((run_dir/"plan.json","plan.json"))
    preflight = run_dir/"preflight.v1.json"
    save(preflight,{"schema_version":"mrun-preflight-config-v1",
        "hashed_roots":[worker.name,*old,"plan.json",preflight.name],
        "path_token_allowlist":["flux-initial-noise-result",config["model_path"]]})
    extras.append((preflight,preflight.name))
    raw = (json.dumps(config,sort_keys=True)+"\n").encode()
    request = {"schema":"mdb-flux-initial-noise-request-v1","config":config,"scheduler_url":scheduler_endpoint(),
        "launcher_source_sha256":sha(Path(__file__)),"task_family":"forward","model":config["model"],
        "backend":"diffusers","device":"cuda","dtype":"bfloat16","batch_size":1,"height":512,"width":512,
        "needs":{"cuda":True},"execution_environment":execution_intent(),
        "config_bytes_sha256":hashlib.sha256(raw).hexdigest(),
        "source_hashes":{name:sha(path) for path,name in [(worker,worker.name),*extras]},
        "source_authorities":{"prior_request_sha256":prior_request["request_sha256"],
            "prior_worker_sha256":PRIOR_WORKER_SHA256,"legacy_sources":{name:sha(path) for name,path in old.items()}},
        "independent_math_sources":{name:hashlib.sha256(value).hexdigest() for name,value in independent.items()},
        "independent_source_authorities":authorities,
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_operations=4,logical_branches=4,
            logical_checkpoint_replays=4,max_local_work_items=64,
            checkpoint_policy="One Z0 history. Full112 seal/PNG before native; native56; matched56 seal/PNG before later56; finishsame native reference andterminalsham. No per-row jobs, retries or hidden future state."),
        "resource_exception":{"ram_mb":14336,"vram_mb":13000,"disk_gb":12,
            "calibration_job":"job-691aa1b12089","measured_peak_rss_mb":8444.6,"measured_peak_vram_mb":9748.0,
            "reason":"Prior56 mathematical transitions cost350.8s;168 predict~1052s plus load/bind/seals/native/collection. Measured RSS8444.6MB/VRAM9748MB. Newcomplete112+matched56 outputs add~2GiB pluspacking versusprior; streamonebank.14336MBRAM/13000MBVRAM,CPU4,disk12GB,10GiB actualoutputbound. Expected1500s,worker1800/outer2100,no retries. Independent long-run code/admission review required."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run_dir/"request.json",request)
    stage_worker(run_dir/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    freeze = run_dir/"independent-math-freeze"
    freeze.mkdir()
    for name,content in independent.items():
        with (freeze/name).open("xb") as stream:
            stream.write(content)
    return {"submitted":False,"run_dir":str(run_dir),"request_sha256":request["request_sha256"]}


def submit_staged(run_dir):
    from mdb.job_support import guarded_mrun_launch

    run_dir = Path(run_dir)
    if (run_dir/"submission.json").exists():
        raise ValueError("finite assay has already been admitted")
    request = read(run_dir/"request.json")
    unsigned = {key:value for key,value in request.items() if key!="request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest()!=request["request_sha256"]
            or sha(Path(__file__))!=request["launcher_source_sha256"]
            or any(sha(run_dir/"payload"/name)!=digest for name,digest in request["source_hashes"].items())
            or any(sha(run_dir/"independent-math-freeze"/name)!=digest for name,digest in request["independent_math_sources"].items())
            or read(run_dir/"plan.json")!=request["config"]):
        raise ValueError("staged source/plan/independent math/launcher changed")
    review_path = run_dir.parent/"long-run-review-final.json"
    review = read(review_path)
    if (review.get("request_sha256")!=request["request_sha256"]
            or review.get("blocking_issues")!=[]
            or review.get("review_status")!="ready-for-finite-run"):
        raise ValueError("source-bound independent long-run review required")
    scheduler_endpoint(request)
    config,resources = request["config"],request["resource_exception"]
    note = ("Klein4B CUDA BF16 B1 512 seed113:actualZ0 full112 mathseal/PNG BEFORE native. "
        "Native56 then matched56 seal/PNG BEFORE later56; same native/sham.112native/3Acts/115receipts/4VAE,168math. "
        "CPU4 RAM14336 VRAM13000 disk12GB; expected1500/worker1800/outer2100s,no retries. "
        "10GiB SSHbound,checkpointrecipes,no weightcopies. Keep numeric misses; stopmechanics/census breach.")
    if len(note)>500:
        raise ValueError("queue note too long")
    command = sealed_worker_command(request)
    save(run_dir/"admission-command.json",{"command":command,"request_sha256":request["request_sha256"],
        "launcher_source_sha256":sha(Path(__file__)),"long_run_review_sha256":sha(review_path)})
    job = guarded_mrun_launch(command,request=request,model=config["model"],task="forward",
        experiment="mdb-flux-initial-noise",payload=run_dir/"payload",prefer_host="beast",gpu=True,needs=request["needs"],
        ram_mb=resources["ram_mb"],vram_mb=resources["vram_mb"],disk_gb=resources["disk_gb"],cpu_threads=4,
        est_wall_s=1500,timeout_s=2100,retry_on_kill=False,preflight=True,note=note)
    save(run_dir/"submission.json",{"job_id":str(job)})
    return {"job_id":str(job),"run_dir":str(run_dir)}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    parser.add_argument("--prior-run",type=Path)
    parser.add_argument("--submit-staged",action="store_true")
    args = parser.parse_args()
    print(json.dumps(submit_staged(args.run_dir.resolve()) if args.submit_staged
        else stage(args.run_dir.resolve(),prior_run=args.prior_run)))
