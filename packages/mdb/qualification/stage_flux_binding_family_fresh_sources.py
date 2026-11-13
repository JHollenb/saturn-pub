"""Stage one frozen-rule-bound prefix-only source lease; root owns submission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_binding_family_fresh_sources.py"
PATH_KEYS = ("qwen_config", "qwen_manifest", "qwen_source_binding", "qwen_retrieval_script", "qwen_recipe_helper",
             "flux_config", "flux_scheduler_config", "flux_manifest", "flux_retrieval_script", "flux_recipe_helper",
             "vae_config", "vae_manifest", "vae_retrieval_script", "vae_recipe_helper")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def collision_check(metadata_root, selected):
    """Only request/plan metadata; never inspect an outcome or array payload."""
    metadata_root = Path(metadata_root).resolve()
    search = subprocess.run(["rtk", "proxy", "rg", "--files", "-g", "request.json", "-g", "plan.json", str(metadata_root)],
                            capture_output=True, text=True)
    if search.returncode not in (0, 1):
        raise ValueError("metadata collision inventory failed: " + search.stderr)
    paths = search.stdout.splitlines()
    wanted = {(row["seed"], row["prompts"]["base"], row["prompts"]["source"]): row["id"] for row in selected}
    checked, collisions = {}, []

    for name in paths:
        path = Path(name)
        if (path.is_symlink() or not path.is_file() or path.stat().st_size > 16 << 20
                or any(part.startswith(("qualification", "attempt-", "writer-attempt-", "prefix-attempt-")) for part in path.relative_to(metadata_root).parts)):
            continue
        try:
            value = read(path)
        except (ValueError, UnicodeError):
            continue
        checked[path.relative_to(metadata_root).as_posix()] = sha(path)
        config = value.get("config", value) if isinstance(value, dict) else {}
        for row in config.get("contexts", []):
            if not isinstance(row, dict) or type(row.get("seed")) is not int or not isinstance(row.get("prompts"), dict):
                continue
            prompts = row["prompts"]
            if not isinstance(prompts.get("base"), str) or not isinstance(prompts.get("source"), str):
                continue
            key = (row["seed"], prompts["base"], prompts["source"])
            if key in wanted:
                collisions.append(dict(context=wanted[key], path=str(path), prior_id=row.get("id"), prior_source_policy=config.get("source_policy")))
    return dict(schema="mdb-fresh-prompt-seed-collision-check-v1", metadata_root=str(metadata_root),
        contexts=selected, checked_metadata=checked, collisions=collisions,
        read_scope="Executed/staged contexts in request.json and plan.json only; no outputs/arrays/images, rule proposal lists or model-free qualification fixtures",
        collision_policy="Exact base+swapped-source+seed is treated as collision regardless of old source policy")


def command(request):
    env = request["execution_environment"]
    variables = dict(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", PYTHONPATH="", UV_LINK_MODE=env["uv_link_mode"],
        UV_CACHE_DIR=env["uv_cache_dir"], TORCH_HOME=env["torch_home"], LLM_MODELS_EXTRA_ROOT=env["extra_model_roots"],
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFUSERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        SATURN_RESIDENT_WORKER="1", SATURN_RUN_CONFIG="saturn_run_config.json", SATURN_DEBUG_REQUEST="saturn_debug_request.json",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    for key, name in dict(tmpdir="TMPDIR", cuda_cache_path="CUDA_CACHE_PATH", torchinductor_cache_dir="TORCHINDUCTOR_CACHE_DIR").items():
        if env.get(key):
            variables[name] = env[key]
    args = ["/usr/bin/env", *[key + "=" + value for key, value in variables.items()],
            "uv", "run", "--no-project", "--offline", "--python", env["python"]]
    for requirement in (["./" + name for name in request["runtime_wheels"]] + env["requirements"]):
        args.extend(("--with", requirement))
    return [*args, "python", "-B", "-u", "-m", "mdb.worker_bootstrap", WORKER, "--config", "saturn_run_config.json"]


def scheduling_contract():
    from mdb.job_support import one_lease_scheduling_contract

    return one_lease_scheduling_contract(model_loads=1, logical_branches=6,
        logical_operations=42, logical_checkpoint_replays=12, max_local_work_items=128,
        checkpoint_policy="One finite source-only CUDA B1 lease; public prefix forks/Acts/rewinds, zero decoder or step2joint0 output, no nested jobs or retries",
        extra=dict(logical_operation_unit="One prompt encode, complete 33-operation natural prefix, complete 23-operation image-primed prefix, or projection",
            per_context_task_census=dict(prompt_encodes=2, natural_prefixes=2, image_primed_prefixes=1, projections=2),
            physical_native_transitions=546, physical_native_blocks=474))


def validate(run, request):
    run = Path(run)
    if identity(request) != request["request_sha256"] or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("fresh source request/launcher differs")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if Path(name).name != name or sha(run / "payload" / name) != expected:
            raise ValueError("fresh staged bytes changed: " + name)
    worker = runpy.run_path(str(run / "payload" / WORKER))
    config = dict(request["config"], family_rule_path=str(run / "payload/family-rule.json"),
                  original_request_path=str(run / "payload/original-generalization-request.json"))
    worker["validate_config"](config)
    if read(run / "payload/freshness-check.json")["collisions"]:
        raise ValueError("fresh prompt+seed collision unresolved")
    seal = read(run / "payload/payload-seal.json")
    if seal["wheels"] != request["runtime_wheels"] or seal["files"] != request["source_hashes"]:
        raise ValueError("fresh runtime payload seal differs")


def stage(run, parent_run, family_rule, family_rule_sha256, original_request, metadata_root):
    run, parent_run = Path(run).resolve(), Path(parent_run).resolve()
    worker = runpy.run_path(str(HERE / WORKER))
    if sha(family_rule) != family_rule_sha256:
        raise ValueError("exact frozen family-rule SHA required before staging")
    rule, original, parent = read(family_rule), read(original_request), read(parent_run / "request.json")
    if identity(parent) != parent["request_sha256"]:
        raise ValueError("parent request identity differs")
    parent_seal = read(parent_run / "payload/payload-seal.json")
    for name, expected in {**parent["source_hashes"], **parent["runtime_wheels"]}.items():
        if sha(parent_run / "payload" / name) != expected:
            raise ValueError("qualified parent bytes differ: " + name)
    if parent_seal["wheels"] != parent["runtime_wheels"]:
        raise ValueError("parent installed-wheel cohort differs")
    config = worker["make_config"](parent["config"], rule, family_rule_sha256, original)
    local = dict(config, family_rule_path=str(Path(family_rule).resolve()), original_request_path=str(Path(original_request).resolve()))
    counts = worker["validate_config"](local)
    collisions = collision_check(metadata_root, worker["contexts"]())
    if collisions["collisions"]:
        raise ValueError("fresh exact prompt+seed collision: " + json.dumps(collisions["collisions"], sort_keys=True))
    sources = {WORKER: HERE / WORKER, "family-rule.json": Path(family_rule).resolve(),
               "original-generalization-request.json": Path(original_request).resolve()}
    for name, expected in worker["HELPERS"].items():
        path = HERE / name
        if sha(path) != expected:
            raise ValueError("frozen prefix helper differs: " + name)
        sources[name] = path
    for key in PATH_KEYS:
        name = config[key]
        if Path(name).name != name:
            raise ValueError("flat original metadata paths required")
        path = parent_run / "payload" / name
        if sha(path) != config[key + "_sha256"]:
            raise ValueError("original metadata differs: " + key)
        sources[name] = path
    run.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan.json", config), ("count-contract.json", counts), ("freshness-check.json", collisions)):
        save(run / name, value)
        sources[name] = run / name
    save(run / "preflight.v1.json", dict(schema="mrun-preflight-config-v1", hashed_roots=list(sources),
        path_token_allowlist=[config["model_path"], config["output_scratch"], worker["RESULT"]],
        frozen_family_rule_before_source_generation=True, source_stop="step2/joint0 INPUT", no_decoder_or_images=True))
    sources["preflight.v1.json"] = run / "preflight.v1.json"
    request = dict(schema="mdb-fresh-binding-prefix-request-v1", worker_script=WORKER, config=config,
        model="flux2-klein-4b-distilled", backend="torch-native-fresh-prefix", device="cuda", dtype="bfloat16",
        task_family="recorder", needs=dict(cuda=True, host="beast"), execution_environment=parent["execution_environment"],
        scheduler_url=parent.get("scheduler_url", "http://192.168.68.68:9025"), runtime_wheels=parent["runtime_wheels"],
        source_hashes={name: sha(path) for name, path in sources.items()}, launcher_sha256=sha(__file__),
        parent_request_sha256=parent["request_sha256"], original_request_sha256=original["request_sha256"],
        family_rule_sha256=family_rule_sha256, workload_geometry=dict(counts, batch_size=1, image_tokens=1024, text_tokens=512),
        resource_exception=dict(ram_mb=12288, vram_mb=10240, cpu_threads=4, disk_gb=12, est_wall_s=240, timeout_s=1200,
            calibration_job="job-cffb90d8d322", measured_peak_rss_mb=6761.2, measured_peak_vram_mb=9210.,
            reason="Same original full-pipeline loader; sequentially release encoder before transformer. Parent16encodes/6008ops/88VAEs426.12s; now12encodes/546ops/0VAEs. Reserved12GiB/10GiB envelope;900sworker/1200souter; no retry."),
        scheduling_contract=scheduling_contract())
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    payload = run / "payload"
    payload.mkdir()
    for name, path in sources.items():
        shutil.copyfile(path, payload / name)
    for name in request["runtime_wheels"]:
        shutil.copyfile(parent_run / "payload" / name, payload / name)
    raw = (json.dumps(config, sort_keys=True, allow_nan=False) + "\n").encode()
    (payload / "saturn_run_config.json").write_bytes(raw)
    (payload / "saturn_debug_request.json").write_text(json.dumps(request, sort_keys=True, allow_nan=False) + "\n")
    save(payload / "payload-seal.json", dict(packages=parent_seal["packages"], wheels=request["runtime_wheels"],
        files=request["source_hashes"], config_bytes_sha256=hashlib.sha256(raw).hexdigest(), worker=WORKER,
        worker_sha256=request["source_hashes"][WORKER]))
    validate(run, request)
    save(run / "staged-command.json", dict(command=command(request), request_sha256=request["request_sha256"], submitted=False))
    return dict(request_sha256=request["request_sha256"], counts=counts, submitted=False)


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    request = read(run / "request.json")
    validate(run, request)
    if (run / "submission.json").exists():
        raise ValueError("finite fresh prefix already submitted")
    review = read(run / "finite-run-review-v1.json")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("request-bound finite-run review required")
    resource = request["resource_exception"]
    os.environ.setdefault("MRUN_URL", request["scheduler_url"])
    job = guarded_mrun_launch(command(request), request=request, model=request["model"], task="recorder",
        experiment="mdb-binding-family-fresh-prefix", payload=run / "payload", gpu=True,
        needs=request["needs"], prefer_host="beast", timeout_s=resource["timeout_s"], preflight=True, retry_on_kill=False,
        **{key: resource[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s")},
        note="Frozen-rule fresh sources only:6pairedcontexts/12encodes/546nativeops/474blocks; stopstep2joint0INPUT;0joint0targets/0images. OneCUDA B1 lease, sequential encoder release, strictpreflight, no retries.")
    save(run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--family-rule", type=Path)
    parser.add_argument("--family-rule-sha256")
    parser.add_argument("--original-request", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    result = submit(args.run) if args.submit else stage(args.run, args.parent_run, args.family_rule,
        args.family_rule_sha256, args.original_request, args.metadata_root)
    print(json.dumps(result, sort_keys=True))
