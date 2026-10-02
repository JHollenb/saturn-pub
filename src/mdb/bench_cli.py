"""CLI launcher and script dispatcher sharing BenchSession's implementation."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile


def run_commands(bench, commands):
    results = []
    for command in commands:
        arguments = dict(command)
        op = arguments.pop("op")
        if op not in bench.capabilities()["operations"] or op in {"close", "prefill"}:
            raise ValueError(f"unsupported bench script operation: {op}")
        results.append({"op": op, "result": getattr(bench, op)(**arguments),
                        "session_id": bench.session_id, "cut_id": bench.branch.cut})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mdb bench")
    subs = parser.add_subparsers(dest="action", required=True)
    for action in ("open", "resume"):
        p = subs.add_parser(action)
        p.add_argument("--model", default="qwen2.5-0.5b-instruct")
        p.add_argument("--root", required=True, help="durable absolute directory on the selected worker host")
        if action == "open":
            p.add_argument("--prompt", required=True)
        else:
            p.add_argument("--legacy-numerical-environment", type=Path,
                           help="explicit qualified environment JSON for older Mamba identities")
        p.add_argument("--lane", choices=("virtual", "native"), default="virtual")
        p.add_argument("--capacity", type=int, default=128)
        p.add_argument("--kv-budget-bytes", type=int, default=64 << 20)
        p.add_argument("--state-budget-bytes", type=int, default=512 << 20)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--height", type=int, default=512)
        p.add_argument("--width", type=int, default=512)
        p.add_argument("--steps", type=int, default=4)
        p.add_argument("--guidance-scale", type=float, default=1.0)
        p.add_argument("--commands", type=Path, help="JSON array of finite Python-equivalent bench calls")
        p.add_argument("--hold-seconds", type=int, default=300, help="bounded live attach window when no script is supplied")
        p.add_argument("--max-new-tokens", type=int, default=32)
        p.add_argument("--dry-run", action="store_true")
    inspect = subs.add_parser("inspect")
    inspect.add_argument("--root", required=True)
    verify_parser = subs.add_parser("verify")
    verify_parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    from mdb.circuit_runtime import DirectoryArtifactStore
    if args.action == "verify":
        from mdb.bench_verify import verify
        print(json.dumps(verify(args.root), sort_keys=True))
        return 0
    if args.action == "inspect":
        root = Path(args.root)
        ref = json.loads((root/"selected.json").read_text())["manifest"]
        print(json.dumps({"manifest_ref": ref, **DirectoryArtifactStore(root/"artifacts").get(ref)}, sort_keys=True))
        return 0
    if not Path(args.root).is_absolute() or args.capacity <= 0 or args.kv_budget_bytes <= 0 or args.state_budget_bytes <= 0 or args.max_new_tokens < 0 or not 1 <= args.hold_seconds <= 600:
        parser.error("root must be absolute; capacity/budget positive; attach window 1..600 seconds")
    if min(args.height, args.width, args.steps) <= 0 or not 0 <= args.seed < 2**63 or not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        parser.error("positive diffusion geometry, valid seed and finite nonnegative guidance required")
    from mdb.bench_loading import resolve_bench_model
    from mdb.tools import scheduler_api
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract, source_hash_manifest, stage_worker, worker_command
    spec = resolve_bench_model(args.model)
    profile = None
    resource_basis = None
    if spec.family == "flux":
        from mdb.bench_resources import diffusion_resources
        profile, resource_basis = diffusion_resources(spec.registry_name, height=args.height,
            width=args.width, steps=args.steps, state_budget_bytes=args.state_budget_bytes)
    config = {"action": args.action, "model": spec.registry_name, "root": args.root,
        "prompt": getattr(args, "prompt", None), "lane": args.lane, "capacity": args.capacity,
        "resident_kv_bytes": args.kv_budget_bytes, "hold_seconds": args.hold_seconds,
        "max_new_tokens": args.max_new_tokens,
        "resident_state_bytes": args.state_budget_bytes, "seed": args.seed, "height": args.height,
        "width": args.width, "num_inference_steps": args.steps, "guidance_scale": args.guidance_scale,
        "commands": json.loads(args.commands.read_text()) if args.commands else None}
    if getattr(args, "legacy_numerical_environment", None) is not None:
        config["legacy_numerical_environment"] = json.loads(args.legacy_numerical_environment.read_text())
    if resource_basis and resource_basis.get("phase_vram_budget_decimal_mb") is not None:
        config["phase_vram_budget_mb"] = resource_basis["phase_vram_budget_decimal_mb"]
    raw = (json.dumps(config, sort_keys=True)+"\n").encode()
    worker = Path(__file__).parent/"workers/run_saturn_bench.py"
    from .payload import package_files
    sources = [("worker.py", worker), *((f"mdb/{name}", path) for name, path in package_files("mdb").items())]
    needs = {"cuda": True, **({"saturn_debug_credential_v1": True} if args.commands is None else {})}
    request = {"schema": "saturn-bench-launch-v1", "model": spec.to_dict(),
        "task_family": "forward",
        "resource_profile": profile.as_dict() if profile else None,
        "resource_basis": resource_basis,
        "needs": needs,
        "scheduler_endpoint": scheduler_api()._resolve(),
        "config_bytes_sha256": sha256(raw).hexdigest(), "source_hashes": source_hash_manifest(sources),
        "scheduling_contract": one_lease_scheduling_contract(logical_operations=1,
            logical_checkpoint_replays=1, checkpoint_policy="one resident worker; all branches in-process")}
    from .execution_environment import intent
    request["execution_environment"] = intent()
    request["request_sha256"] = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps({"config": config, "request": request}, sort_keys=True))
        return 0
    with tempfile.TemporaryDirectory(prefix="mdb-bench-") as directory:
        stage = Path(directory)
        stage_worker(stage, worker=worker, config=config, request=request, config_bytes=raw)
        command = worker_command(worker.name, include_observability_extras=False)
        command = [command[0], f"MRUN_SERVER_URL={request['scheduler_endpoint']}", "CUBLAS_WORKSPACE_CONFIG=:4096:8", *command[1:]]
        job = guarded_mrun_launch(command,
            request=request, experiment="mdb-open-bench", task=f"bench-{args.action}",
            mrun_config={**request, "model": spec.registry_name, "model_identity": spec.to_dict()},
            model=spec.registry_name, payload=stage, prefer_host="beast", gpu=True,
            needs=needs,
            ram_mb=profile.ram_mb if profile else None, vram_mb=profile.vram_mb if profile else None,
            disk_gb=profile.disk_gb if profile else None,
            cpu_threads=profile.cpu_threads if profile else None,
            detach=True, timeout_s=840, retry_on_kill=False, preflight=True,
            note=f"Open Bench {spec.registry_name}: one resident model, serialized native stages; bounded live attach or finite script; no nested lease or retries.")
    print(json.dumps({"job_id": job, "root": args.root,
                     "attach": f"mdb attach {job}" if args.commands is None else None,
                     "request_sha256": request["request_sha256"]}, sort_keys=True))
    return 0
