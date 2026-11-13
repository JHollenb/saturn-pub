"""Pin public assets, stage a finite installed worker, submit and collect directly."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
LOGS = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
BASE = "2fc06364715b967f1860aea9cf38778875588b17"
EDITED = "d1671a178705bcab7644806d23c535517f35723b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def fetch(base):
    source = base / "sources"
    source.mkdir(parents=True, exist_ok=False)
    urls = {"edit.npz": f"https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{LOGS}/exp8_math/runs/p50c/edit/edit.npz",
        "edit-manifest.json": f"https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{LOGS}/exp8_math/runs/p50c/edit/manifest.json",
        "pinned-config.json": f"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/{BASE}/config.json",
        "pinned-tokenizer-config.json": f"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/{BASE}/tokenizer_config.json",
        "pinned-chat-template.jinja": f"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/{BASE}/chat_template.jinja"}
    for name, url in urls.items():
        raw = urllib.request.urlopen(url, timeout=60).read()
        with (source / name).open("xb") as stream:
            stream.write(raw)
    pins = {"schema": "mdb-wsp-public-source-pins-v1", "logs_revision": LOGS,
        "base": {"repo": "Qwen/Qwen3.5-0.8B", "revision": BASE,
            "filename": "model.safetensors-00001-of-00001.safetensors",
            "tokenizer_sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "file_sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"},
        "edited": {"repo": "RikiyaT/Qwen3.5-0.8B-weight-space-programming", "revision": EDITED,
            "file_sha256": "fd1e42acdd118ac1b300d37500fd2885e75f5a09de3a33010d802bf7f737b553"},
        "files": {name: {"url": url, "sha256": sha(source / name)} for name, url in urls.items()}}
    save(source / "pins.json", pins)
    return pins


def wheels():
    result = list((WORKSPACE / "artifacts/mdb-wsp-20261110/installed-03/wheels").glob("*.whl"))
    result.append(WORKSPACE / "artifacts/mdb-parallel-relation-family-20261107/campaign-01/payload/mrun_pub-0.1.2-py3-none-any.whl")
    if len(result) != 3 or any(not p.is_file() for p in result):
        raise ValueError("three pinned installed runtime wheels required")
    return result


def command(request):
    environment = request["execution_environment"]
    cmd = ["/usr/bin/env", "PYTHONUNBUFFERED=1", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=",
        "UV_LINK_MODE=symlink", "UV_CACHE_DIR=" + environment["uv_cache_dir"], "TORCH_HOME=" + environment["torch_home"],
        "LLM_MODELS_EXTRA_ROOT=/mnt/big/llm-models", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false", "SATURN_RESIDENT_WORKER=1", "SATURN_RUN_CONFIG=saturn_run_config.json",
        "SATURN_DEBUG_REQUEST=saturn_debug_request.json", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4",
        "uv", "run", "--no-project", "--offline", "--python", environment["python"]]
    for requirement in [*("./" + name for name in request["runtime_wheels"]), *environment["requirements"]]:
        cmd += ["--with", requirement]
    return cmd + ["python", "-B", "-u", "-m", "mdb.worker_bootstrap", "worker.py"]


def stage(base, name, phase, vram_mb=None, measurement_job=None, measurement_peak_mb=2014, sampling="greedy", native_run="smoke-03"):
    from mdb.job_support import one_lease_scheduling_contract

    run = base / name
    run.mkdir(parents=True, exist_ok=False)
    model_free = phase in {"saved-audit", "recover"}
    cpu_only = phase == "audit" or model_free
    pins = json.loads((base / "sources/pins.json").read_text())
    for filename, reference in pins["files"].items():
        if sha(base / "sources" / filename) != reference["sha256"]:
            raise ValueError("pinned public source asset differs")
    config = {"schema": "mdb-wsp-case-study-plan-v1", "phase": phase, "pins": pins,
        "tokenizer_sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "model_path": "/mnt/big/llm-models/Qwen3.5-0.8B", "output_scratch": "/mnt/hdd8tb/mdb/wsp-case-study",
        "scientific_gates": [], "retry": False, "nested_mrun": False,
        "batch_size": 1, "max_prompt_tokens": 256, "max_new_tokens": 64,
        "device": "cpu" if cpu_only else "cuda", "dtype": "bfloat16",
        "contexts": [
            {"id": "power-sum", "prompt": "What is 2^10 + 24? Please reason step by step and put your final answer within \\boxed{}.", "answer": "1048"},
            {"id": "linear-equation", "prompt": "Solve 3x + 7 = 34. Please reason step by step and put your final answer within \\boxed{}.", "answer": "9"},
            {"id": "short-instruction", "prompt": "Write exactly the word READY and nothing else.", "answer": "READY"}],
        "split": "discovery and numerical qualification only; no held-out evidence",
        "generation_policy": "greedy, thinking on; short horizon, not the published stochastic 32768-token benchmark"}
    config["sampling"] = {"kind": "greedy", "seed": 0} if sampling == "greedy" else {
        "kind": "think35", "seed": 0, "temperature": 0.6, "top_k": 20, "top_p": 0.95, "presence_penalty": 1.5}
    if phase == "discovery":
        config["max_new_tokens"] = 2048
        config["contexts"] = [
            {"id": "pencil-inventory", "prompt": "A store has 37 boxes with 12 pencils in each box. It sells 89 pencils. How many pencils remain? Please reason step by step and put your final answer within \\boxed{}.", "answer": "355"},
            {"id": "rectangle-area", "prompt": "A rectangle has perimeter 46 centimeters. Its length is 5 centimeters greater than its width. What is its area in square centimeters? Please reason step by step and put your final answer within \\boxed{}.", "answer": "126"},
            {"id": "literal-instruction", "prompt": "Reply with exactly the word azure and nothing else.", "answer": "azure"}]
        config["generation_policy"] = "greedy, thinking on, <=2048 own-token steps; exploratory completion/repetition panel, not the published stochastic benchmark"
        config["split"] = "fresh authored exploratory contexts, frozen before new target outcomes; no terminal holdout claim"
        if sampling == "think35":
            config["split"] = "paired sampling-policy follow-up on exploratory contexts; greedy outcomes already available; no held-out claim"
            config["generation_policy"] = "temperature 0.6, top-k 20, top-p 0.95, generated-token presence penalty 1.5; thinking on; seeds 0/1/2; <=2048 tokens; HF eager rather than published vLLM backend"
    if phase == "recover":
        prior = base / native_run
        native_job = json.loads((prior / "submission.json").read_text())["job_id"]
        if json.loads((prior / "job.json").read_text())["state"] != "failed":
            raise ValueError("recovery requires a retained failed job record")
        config["recovery_job"] = native_job
        config["recovery_request_sha256"] = json.loads((prior / "request.json").read_text())["request_sha256"]
        config["recovery_root"] = config["output_scratch"] + "/" + native_job + "/wsp-result"
    if phase == "saved-audit":
        static = base / "audit-02/extracted/wsp-result"
        smoke = base / native_run / "extracted/wsp-result"
        native_job = json.loads((base / native_run / "submission.json").read_text())["job_id"]
        collected = json.loads((base / native_run / "report.json").read_text())
        if collected["job_id"] != native_job or collected["files_manifest_sha256"] != sha(smoke / "files.json"):
            raise ValueError("saved native run differs from its collected custody")
        verification = json.loads((smoke / "wsp-verification.json").read_text())
        config["saved_native_run"] = native_run
        config["saved_audit_pins"] = {"audit_files_sha256": sha(static / "files.json"),
            "native_files_sha256": sha(smoke / "files.json"),
            "native_outcome_manifest": verification["outcome_manifest"],
            "native_forecast_seal": verification["forecast"]["seal_sha256"]}
        config["saved_audit_roots"] = {"audit": "/mnt/hdd8tb/mdb/wsp-case-study/job-e0a441ea99d3/wsp-result",
            "native": "/mnt/hdd8tb/mdb/wsp-case-study/" + native_job + "/wsp-result"}
        config["device"] = "cpu"
    if vram_mb is not None:
        if cpu_only or not measurement_job or vram_mb < measurement_peak_mb:
            raise ValueError("VRAM exception requires a preceding measured native loading job")
        config["reservation_exception"] = {"ram_mb": 5517, "vram_mb": vram_mb,
            "measurement_job": measurement_job, "observed_peak_vram_mb": measurement_peak_mb,
            "scope": "prior observed peak plus declared headroom for changed continuation geometry; update from completed peak"}
    save(run / "plan.json", config)
    payload = run / "payload"
    payload.mkdir()
    files = {"worker.py": HERE / "worker.py", "retrieve_weights.py": HERE / "retrieve_weights.py",
             "flux_family_custody.py": HERE.parent / "flux_family_custody.py",
             **{path.name: path for path in (base / "sources").iterdir()}}
    if phase == "saved-audit":
        files["audit_saved.py"] = HERE / "audit_saved.py"
    if phase == "recover":
        files["recover_saved.py"] = HERE / "recover_saved.py"
    save(payload / "preflight.v1.json", {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": [config["model_path"], config["output_scratch"],
                                 *config.get("saved_audit_roots", {}).values(),
                                 *([config["recovery_root"]] if phase == "recover" else [])],
        "hashed_roots": list(files) + ["preflight.v1.json"]})
    files["preflight.v1.json"] = payload / "preflight.v1.json"
    wheel_map = {p.name: p for p in wheels()}
    requirements = ["torch==2.13.0", "numpy==2.5.4", "safetensors==0.8.0", "pyyaml==6.0.3", "psutil==7.2.2"]
    if phase in {"smoke", "discovery"}:
        requirements += ["transformers==5.14.1", "accelerate==1.14.0"]
    environment = {"python": "3.14.6", "uv_link_mode": "symlink", "uv_cache_dir": "/mnt/hdd8tb/mdb/uv-cache",
        "torch_home": "/mnt/hdd8tb/mdb/torch", "extra_model_roots": "/mnt/big/llm-models", "requirements": requirements}
    request = {"schema": "mdb-wsp-case-study-request-v1", "config": config, "execution_environment": environment,
        "source_hashes": {name: sha(p) for name, p in files.items()}, "runtime_wheels": {name: sha(p) for name, p in wheel_map.items()},
        "model": "qwen3.5-0.8b", "backend": "hf", "device": config["device"], "dtype": "bfloat16",
        "needs": {"cpu": True} if cpu_only else {"cuda": True}, "batch_size": 1, "seq_len": config["max_prompt_tokens"] + config["max_new_tokens"],
        "backend_options": {"physical_batch_size": 1, "full_prefill_logits": False},
        "task_family": "analysis" if model_free else "recorder", "launcher_sha256": sha(__file__),
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0 if model_free else 1,
            model_free=model_free, logical_branches=1 if cpu_only else 4,
            logical_operations=1 if cpu_only else 12, logical_checkpoint_replays=0, max_local_work_items=256,
            extra={**({"task_family": "analysis"} if model_free else {}),
                "physical_forward_census": {"source_prefills": 0 if cpu_only else 3,
                "maximum_native_root_calls": 0 if cpu_only else 3 + 12 * config["max_new_tokens"]}})}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save(run / "request.json", request)
    shutil.copyfile(__file__, run / "frozen-launcher.py")
    for name, path in {**files, **wheel_map}.items():
        if path.resolve() != (payload / name).resolve():
            (payload / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, payload / name)
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    (payload / "saturn_run_config.json").write_bytes(raw)
    save(payload / "saturn_debug_request.json", request)
    packages = {}
    for name, path in wheel_map.items():
        package = "mdb" if name.startswith("mdb_") else "saturn_pub" if name.startswith("saturn_") else "mrun"
        with zipfile.ZipFile(path) as archive:
            packages[package] = {member[len(package) + 1:]: hashlib.sha256(archive.read(member)).hexdigest()
                for member in archive.namelist() if member.startswith(package + "/") and not member.endswith("/")}
    save(payload / "payload-seal.json", {"packages": packages, "wheels": request["runtime_wheels"], "files": request["source_hashes"],
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "worker": "worker.py", "worker_sha256": request["source_hashes"]["worker.py"]})
    save(run / "staged-command.json", {"command": command(request)})
    return {"run": str(run), "request_sha256": request["request_sha256"], "phase": phase}


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    request = json.loads((run / "request.json").read_text())
    if (run / "submission.json").exists():
        raise ValueError("finite request already submitted")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("staged payload differs")
    phase = request["config"]["phase"]
    model_free = phase in {"saved-audit", "recover"}
    reservation = request["config"].get("reservation_exception", {})
    os.environ.setdefault("MRUN_URL", "http://192.168.68.68:9025")
    job = guarded_mrun_launch(command(request), request=request, experiment="mdb-wsp-case-study-" + phase,
        task="analysis" if model_free else "recorder", model=None if model_free else request["model"], payload=run / "payload", prefer_host="beast",
        pin="beast" if model_free else None,
        ram_mb=reservation.get("ram_mb"), vram_mb=reservation.get("vram_mb"),
        gpu=phase != "audit" and not model_free, needs=request["needs"], timeout_s=840, retry_on_kill=False, preflight=True,
        note="WSP report recovery: zero model loads/forwards, retain failed report and original sealed native arrays, verify copied bytes and forecast/outcome manifest; no scientific reexecution. One CPU lease." if phase == "recover" else
             "WSP saved-only independent reader: zero model loads/forwards, pinned static and selected native evidence; recompute support, output directions, first-logit/token equivalences and retained local errors. One CPU lease." if phase == "saved-audit" else
             "WSP exact checkpoint audit: zero native forwards; two sparse BF16 edits, file/header/tensor hashes and actual support/SVD diagnostics. CPU only, one finite lease, no retries." if phase == "audit" else
             "WSP native panel: one Qwen3.5 0.8B load, 3 prompts x 4 arms x <=" + str(request["config"]["max_new_tokens"]) + " own-token steps, B1 BF16 CUDA; fresh hybrid caches per arm, sealed local forecasts; finite exploratory lease, no benchmark claim.")
    value = {"job_id": str(job), "request_sha256": request["request_sha256"]}
    save(run / "submission.json", value)
    return value


def collect(run):
    import runpy
    job = json.loads((run / "submission.json").read_text())["job_id"]
    result = runpy.run_path(str(HERE.parent / "flux_family_custody.py"))["collect"](run, job,
        limits={"max_archive_bytes": 64 << 20, "max_expanded_bytes": 128 << 20, "max_member_bytes": 32 << 20})
    if result.get("collection_status") == "not-terminal":
        return {key: result[key] for key in ("collection_status", "job_id", "state")}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "stage", "submit", "collect"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--name", default="audit-01")
    parser.add_argument("--phase", choices=("audit", "smoke", "discovery", "saved-audit", "recover"), default="audit")
    parser.add_argument("--vram-mb", type=float)
    parser.add_argument("--measurement-job")
    parser.add_argument("--measurement-peak-mb", type=float, default=2014)
    parser.add_argument("--sampling", choices=("greedy", "think35"), default="greedy")
    parser.add_argument("--native-run", default="smoke-03", help="source run for a saved-only audit or failed-report recovery")
    args = parser.parse_args()
    base = args.base.resolve()
    result = fetch(base) if args.action == "fetch" else stage(base, args.name, args.phase, args.vram_mb, args.measurement_job, args.measurement_peak_mb, args.sampling, args.native_run) if args.action == "stage" else submit(base / args.name) if args.action == "submit" else collect(base / args.name)
    print(json.dumps(result, sort_keys=True))
