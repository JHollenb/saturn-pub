"""Refine audited early asymmetric transfer at sequential joint0 writes."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def stage(run, predecessor):
    index = 0
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command
    data = predecessor / "extracted/flux-anime-origin-result"
    old, audit, frontiers, manifest = (read(p) for p in (data / "plan.json", predecessor / "analysis-v1.json", data / "frontiers.json", data / "files.json"))
    if audit["mechanics_status"] != "valid":
        raise ValueError("audited scout required")
    inputs = {role: frontiers[role + "/joint0"] for role in ("base", "source")}
    relatives = set()
    for digest in inputs.values():
        rel = "typed-store/cuts/" + digest + ".json"
        cut = read(data / rel)
        relatives.add(rel)
        relatives.update("typed-store/pages/" + h + ".json" for h in cut["slots"].values())
        relatives.update("typed-store/blobs/" + h + ".safetensors" for h in cut["physical_blobs"])
    external = {}
    for rel in sorted(relatives):
        p = data / rel
        if sha(p) != manifest[rel]:
            raise ValueError("scout CAS differs")
        external[rel] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    config = {k: old[k] for k in ("model", "model_path", "expected_checkpoint_sha256", "cpu_threads", "part_rows")}
    config.update(schema="mdb-anime-origin-joint-plan-v1", joint_index=0, max_adapter_steps=600, max_wall_seconds=720,
        input_cuts=inputs, external_inputs=external, output_scratch="/home/beast/.cache/mdb-research", external_input_root=read(data / "summary.json")["output_root"] + "/flux-anime-origin-result",
        queries=[201, 233, 297, 425, 489, 681, 405, 745], output_dims=list(range(0, 3072, 96)),
        historical_pngs={"native.base": read(data / "rows/native.base.json")["image"]["sha256"], "sham": read(data / "rows/native.base.json")["image"]["sha256"],
            **{"native." + support: read(data / "rows" / ("base." + support + ".joint0.json"))["image"]["sha256"] for support in ("upper", "lower")}},
        scope="Exploratory sequential joint0 refinement after asymmetric upper-head transfer in a reciprocal depth scout. Two native histories and matched upper/lower source futures known. All mixed R/A/M futures new. Source A/M writes observed before sampled sequential-BF16 forecast seal; not held-out semantic rule confirmation or raw-prompt RGB prediction.",
        hypotheses={
            "H-mechanics": "Native base,upper,lower and sham match scout PNGs. Source direct block/native parity and complete RAMT reconstruction exact; preserve sequential native BF16 edge order.",
            "H-R-versus-writes": "Tentative: attention and downstream MLP response carry more of the companion effect here than the incoming local residual alone. Preserve residual-driven cases and incomplete/countervailing effects.",
            "H-locality": "Attention changes outside the edited band even with exact outside incoming I. Unlike parallel single18, sequential MLP differences can also be outside the band because attention changed their receiving state. MLP remains pointwise; no contradiction or semantic label follows.",
            "H-numeric": "Freeze complete1024x3072 sequential BF16 image-update predictions from observed R,A,M,ga,gm before mixed candidate calls. Store lossless BF16 bits and8x32 samples; preserve all errors. No independent attention/MLP or RGB prediction.",
            "H-scope": "One selected earlier joint block, one scene and supplied geometry. Appearance origin/minimality and semantic identity are not established by sufficient source transplants."})
    run.mkdir(parents=True, exist_ok=False)
    save(run / "plan.json", config)
    here = Path(__file__).resolve().parent
    worker = here / "flux_anime_origin_joint.py"
    extras = [(here / "flux_scene_inventory_campaign.py", "flux_scene_inventory_campaign.py"), (here / "flux_joint_writes.py", "flux_joint_writes.py"), (here / "discovery_upgrades.py", "campaign_support.py"), (run / "plan.json", "plan.json")]
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, *(n for _, n in extras), preflight.name], "path_token_allowlist": ["flux-anime-origin-joint-result", "input-store", config["model_path"], config["external_input_root"], config["output_scratch"]]})
    extras.append((preflight, preflight.name))
    env = intent()
    if env["uv_cache_dir"].startswith("/Users/"):
        raise ValueError("Linux cache profile required")
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-anime-origin-joint-request-v1", "config": config, "model": config["model"], "task_family": "forward", "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "needs": {"cuda": True, "host": "beast"}, "execution_environment": env, "runtime_environment": {"PYTHONDONTWRITEBYTECODE": "1"}, "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "source_hashes": {n: sha(p) for p, n in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=20, logical_branches=20, logical_checkpoint_replays=20, max_local_work_items=20, checkpoint_policy="Audited joint0 scout frontiers.3sources+14RAM+2coherent+sham20RGB,20candidatecalls21Acts540nativeTransitions.14full1024x3072 forecasts plus8x32 samples. Max600nativeSteps. B1,1load0encode,no retries."),
        "resource_exception": {"ram_mb": 10000, "vram_mb": 13000, "calibration_job": "job-8505274b3899", "measured_peak_rss_mb": 6587.8, "measured_peak_vram_mb": 9602, "reason": "Same bounded20-case R/A/M native sequential-port geometry; source point earlier, longer suffix but no larger physical batch or port arrays."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    command = worker_command(worker.name)
    command.insert(1, "PYTHONDONTWRITEBYTECODE=1")
    job = guarded_mrun_launch(command, request=request, model=config["model"], task="forward", experiment="mdb-flux-anime-origin-joint", payload=run / "payload", prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=10000, vram_mb=13000, cpu_threads=4, est_wall_s=360, timeout_s=720, retry_on_kill=False, preflight=True,
        note="Klein bridge joint0 R/A/M factorial.20RGB20candidateblocks21Acts540nativeSteps,CUDA BF16 physicalB1,1load0encodes. Known source PNG controls, mixed futures new.14 complete1024x3072 source-assisted forecasts plus8x32 samples. Stop600nativeSteps/720s,no retries.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--predecessor", type=Path)
    p.add_argument("--collect")
    args = p.parse_args()
    if args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](args.run_dir, args.collect, log_storage="gzip", max_compressed_bytes=512 << 20, max_expanded_bytes=1 << 30)
    else:
        result = stage(args.run_dir.resolve(), args.predecessor.resolve())
    print(json.dumps(result, sort_keys=True))
