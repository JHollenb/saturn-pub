"""One admitted resident worker for the CLI's open/resume session."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import torch

from mdb.bench_loading import load_bench_model
from mdb.bench import BenchSession
from mdb.bench_cli import run_commands
from mdb.bench_debugger import attach
from mdb.debugger_protocol import DebugRequest


def main():
    raw = Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if sha256(raw).hexdigest() != request["config_bytes_sha256"]:
        raise RuntimeError("sealed bench configuration changed")
    config = json.loads(raw)
    if "phase_vram_budget_mb" in config:
        os.environ["VRAM_LIMIT_MB"] = str(config["phase_vram_budget_mb"] * 1e6 / (1 << 20))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    spec, model, tokenizer = load_bench_model(config["model"], device="cuda")
    root = Path(config["root"])
    kwargs = {"resident_state_bytes": config.get("resident_state_bytes", 512 << 20)} if spec.family != "qwen2" else {}
    bench = (BenchSession.resume(model, tokenizer, root=root) if config["action"] == "resume" else
        BenchSession.from_resident(model, tokenizer, root=root, model_id=spec.model_id,
            revision=spec.revision, session_id=f"bench-{os.environ['MRUN_JOB_ID']}", lane=config["lane"],
            continuation_capacity=config["capacity"], resident_kv_bytes=config["resident_kv_bytes"], **kwargs))
    try:
        if config["action"] == "open":
            if spec.family in {"qwen2", "mamba"}: bench.prefill(config["prompt"])
            else: bench.prepare(config["prompt"], **{key: config[key] for key in
                ("seed", "height", "width", "num_inference_steps", "guidance_scale")})
        if config["commands"] is not None:
            results = run_commands(bench, config["commands"])
        else:
            job = os.environ["MRUN_JOB_ID"]
            credential = os.environ.get("MRUN_DEBUG_CREDENTIAL")
            controller, poller = attach(bench, job_id=job,
                endpoint=f"{os.environ['MRUN_SERVER_URL'].rstrip('/')}/api/jobs/{job}/debug",
                token=None if credential else os.environ.get("MRUN_SERVER_TOKEN"),
                debug_credential=credential, debug_credential_id=os.environ.get("MRUN_DEBUG_CREDENTIAL_ID"))
            controller.handle(DebugRequest(operation="pause"))
            cursor = bench.cursor()
            controller.safe_point("token" if spec.family in {"qwen2", "mamba"} else "denoise", cursor.get("token", cursor.get("step", 0)))
            print("SATURN_BENCH_READY="+json.dumps(bench.inspect(), sort_keys=True), flush=True)
            deadline = time.monotonic()+config["hold_seconds"]
            while time.monotonic() < deadline:
                poller.serve_once(controller)
                if controller.status()["state"] == "running":
                    cursor = bench.cursor()
                    boundary = None
                    if spec.family == "qwen2":
                        if cursor["token"] < config["max_new_tokens"]:
                            kind = "token" if cursor["next_layer"] == bench.vm.runtime.codec.layers else "layer"
                            bench.step(kind)
                            boundary = (kind, bench.cursor()["token"] if kind == "token" else bench.cursor()["next_layer"])
                    elif spec.family == "mamba":
                        if cursor["token"] < config["max_new_tokens"]:
                            kind = "token" if cursor["next_layer"] == len(bench.adapter.layers) else "layer"
                            bench.step(kind)
                            boundary = (kind, bench.cursor()["token"] if kind == "token" else bench.cursor()["next_layer"])
                    elif cursor["phase"] != "complete":
                        kind = "block" if cursor["phase"] in {"conditioner", "denoise", "blocks"} else "scheduler" if cursor["phase"] == "scheduler" else "vae" if cursor["phase"] == "vae" else "commit"
                        bench.step(kind)
                        now = bench.cursor()
                        boundary = ("denoise", now["step"]) if now["step"] != cursor["step"] else (kind, now["block"] if kind == "block" else now["step"])
                    if boundary: controller.safe_point(*boundary)
                time.sleep(0.05)
            results = [{"op": "live-debugger", "result": bench.inspect()}]
        print("SATURN_BENCH_REPORT="+json.dumps({"session_id": bench.session_id, "root": str(root),
            "results": results, "memory": bench.inspect("memory"), "manifest": bench._publish()}, sort_keys=True), flush=True)
    finally:
        bench.close()


if __name__ == "__main__":
    main()
