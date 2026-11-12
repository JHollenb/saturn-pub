"""One measured CUDA lease for actual anime ports and eight local restorations."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def stage(run_dir, predecessor, *, submit=True):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-anime-result"
    audit = json.loads((predecessor / "analysis-v1.json").read_text())
    if audit["mechanics_status"] != "valid":
        raise ValueError("audited predecessor required")
    old = json.loads((data / "plan.json").read_text())
    manifest = json.loads((data / "files.json").read_text())
    cuts, relatives, historical = {}, set(), {}
    for scene in ("festival", "bridge"):
        for variant in ("left", "both"):
            key = scene + "." + variant
            row = json.loads((data / "rows" / (key + ".json")).read_text())
            cuts[key], historical[key] = row["edited_cut"], row["image"]["sha256"]
            if sha(data / row["image"]["path"]) != historical[key]:
                raise ValueError("historical image differs")
            relative = "typed-store/cuts/" + cuts[key] + ".json"
            record = json.loads((data / relative).read_text())
            relatives.add(relative)
            relatives.update("typed-store/pages/" + h + ".json" for h in record["slots"].values())
            relatives.update("typed-store/blobs/" + h + ".safetensors" for h in record["physical_blobs"])
        historical[scene + ".rescue.sham"] = historical[scene + ".both"]
    external = {}
    for relative in sorted(relatives):
        path = data / relative
        if sha(path) != manifest[relative]:
            raise ValueError("historical CAS differs")
        external[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
    config = {"schema": "mdb-flux-anime-writers-plan-v1", "model": old["model"], "model_path": old["model_path"],
        "expected_checkpoint_sha256": old["expected_checkpoint_sha256"], "cpu_threads": 4,
        "input_cuts": cuts, "external_inputs": external,
        "external_input_root": json.loads((predecessor / "job.json").read_text())["result"]["diagnostics"]["cwd"] + "/flux-anime-result",
        "historical_pngs": historical, "left_rows": {s: old["supports"][s]["masks"]["left"] for s in ("festival", "bridge")},
        "max_adapter_steps": 750, "max_wall_seconds": 600,
        "scope": "Known anime left/both final images and exact step2 project cuts. No joint0 ports or rescue RGB yet. Same checkpoint, clock, base C. Manual geometric left support. Local image-only port restores leave native text and companion image updates intact; later feedback may repair them.",
        "hypotheses": {
            "H-first-writer": "Exact left latent and text produce exact left input projection and joint0 normalized attention input. First selected-side cross-region difference appears in joint0 attention return, then MLP input/return and post-block image.",
            "H-equation": "Observed actual native A/M returns close the sequential BF16 residual equation exactly. Restoring both fixed-left A/M returns reproduces single-branch fixed-left post-joint0 image exactly, retaining native-both text and other rows.",
            "H-replay": "Four native traces and two sham rescues reproduce six historical PNGs exactly.",
            "H-consumer": "Tentative: joint0 fixed-left A restoration shifts left hair toward the single-edit cue (black festival, violet bridge). M-only may alter appearance; AM may have larger effect. No effect is retained as persistence/repair evidence, not proof that attention is irrelevant.",
            "H-limits": "No automatic semantic decoder, independent RGB forecast or complete scene account. Only first joint0 block of the remaining step2 suffix, not the earliest step of generation."}}
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    worker = here / "flux_anime_writers.py"
    extras = [(here / n, n) for n in ("flux_scene_inventory_campaign.py", "flux_joint_writes.py")]
    extras += [(here / "discovery_upgrades.py", "campaign_support.py"), (run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, *(n for _, n in extras), preflight.name],
        "path_token_allowlist": ["flux-anime-writers-result", "input-store", config["model_path"], config["external_input_root"]]})
    extras.append((preflight, preflight.name))
    environment = intent()
    if submit and environment["uv_cache_dir"].startswith("/Users/"):
        raise ValueError("Linux worker profile required")
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-anime-writers-request-v1", "config": config, "model": config["model"], "task_family": "forward",
        "needs": {"cuda": True, "host": "beast"}, "backend": "diffusers", "device": "cuda", "dtype": "bfloat16",
        "batch_size": 1, "height": 512, "width": 512, "execution_environment": environment,
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "source_hashes": {n: sha(p) for p, n in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=12, logical_branches=12,
            logical_checkpoint_replays=12, max_local_work_items=12,
            checkpoint_policy="4known native traces +8local joint0 port restorations, 6exactPNGcontrols.1load0encode,B1sequential,no retries."),
        "resource_exception": {"ram_mb": 10000, "vram_mb": 13000, "calibration_job": "job-42ccc017f516",
            "measured_peak_rss_mb": 4944, "measured_peak_vram_mb": 9260,
            "reason": "Same installed-wheel CUDA BF16 resident512px4step lane;0encodes.12futures vs16calibration. Bounded additional BF16 joint0 ports and typed cuts."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    if not submit:
        return {"submitted": False, "run_dir": str(run_dir)}
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-anime-writers", payload=run_dir / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=10000, vram_mb=13000, cpu_threads=4, est_wall_s=300, timeout_s=600,
        retry_on_kill=False, preflight=True,
        note="Klein4B anime first-writer CUDA BF16 B1.4known traces+8local joint0 A/M rescues,6exactRGBcontrols.1load0encode.2scenes.8explicit candidate-block calls;native receipts separate. Stop750transitions/12images/600s;no retries.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--predecessor", type=Path)
    p.add_argument("--stage-only", action="store_true")
    p.add_argument("--status")
    p.add_argument("--collect")
    args = p.parse_args()
    if args.status:
        from mdb.job_support import terminal_job
        _, job, logs = terminal_job(args.status)
        phases = [line for line in logs.splitlines() if line.startswith("MDB_FLUX_ANIME_WRITERS_PHASE ")]
        result = {"job_id": args.status, "state": job["state"], "last_phases": phases[-2:],
            "sealed_phase": next((line for line in phases if '"kind": "predictions_sealed"' in line), None),
            "errors": logs[-3000:] if job["state"] == "failed" else None}
    elif args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir, args.collect, log_storage="gzip", max_compressed_bytes=512 << 20, max_expanded_bytes=1 << 30)
    else:
        result = stage(args.run_dir.resolve(), args.predecessor.resolve(), submit=not args.stage_only)
    print(json.dumps(result, sort_keys=True))
