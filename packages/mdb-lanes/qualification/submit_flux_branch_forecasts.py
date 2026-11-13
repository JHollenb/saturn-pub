"""One admitted resident run: base plus three sealed, edited image forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def stage(run_dir, predecessor):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    run_dir.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-concept-result"
    manifest, source, old, report = (json.loads(path.read_text()) for path in
        (data / "files.json", data / "source.json", data / "plan.json", predecessor / "report.json"))
    if report["job_id"] != "job-c2a467736b50" or report["archive_sha256"] != "24454082c80c1c646b8a7e4948b948fa4765f49ff5ca4b1f4edff3e964cc409d":
        raise ValueError("unexpected observed carrier source")
    if sha(data / "source.json") != manifest["source.json"]:
        raise ValueError("source differs")
    names = ["base", "relation", "color", "shape"]
    cuts = {name: source["cuts"][name]["0"]["projected"] for name in names}
    relatives = set()
    for cut_id in [cuts["base"]]:
        relative = f"typed-store/cuts/{cut_id}.json"
        cut = json.loads((data / relative).read_text())
        relatives.add(relative)
        relatives.update(f"typed-store/pages/{value}.json" for value in cut["slots"].values())
        relatives.update(f"typed-store/blobs/{value}.safetensors" for value in cut["physical_blobs"])
    extras = []
    for relative in sorted(relatives):
        path = data / relative
        if sha(path) != manifest[relative]:
            raise ValueError("observed carrier CAS differs")
        extras.append((path, relative.replace("typed-store/", "input-store/", 1)))
    for name in names[1:]:
        cut_path = data / "typed-store/cuts" / (cuts[name] + ".json")
        cut = json.loads(cut_path.read_text())
        page_relative = "typed-store/pages/" + cut["slots"]["text"] + ".json"
        page = json.loads((data / page_relative).read_text())
        blob_relative = "typed-store/blobs/" + page["tensor_blob"] + ".safetensors"
        for relative in (cut_path.relative_to(data).as_posix(), page_relative, blob_relative):
            if sha(data / relative) != manifest[relative]:
                raise ValueError("source tensor port provenance differs")
        record = run_dir / (name + ".carrier.json")
        save(record, {"cut": cut, "page": page, "source_cut_file_sha256": sha(cut_path),
            "scope": "authenticated text tensor extraction; not a complete replayable StateCut"})
        extras += [(record, "carriers/" + name + ".json"), (data / blob_relative, "carriers/" + name + ".safetensors")]
    config = {key: old[key] for key in ("model", "model_path", "expected_checkpoint_sha256", "height", "width", "steps", "guidance_scale", "cpu_threads")}
    config.update(schema="mdb-flux-branch-forecast-plan-v1", seed=47, input_cuts=cuts, branch_ids=names, prompts=old["prompts"],
        predecessor={"job_id": report["job_id"], "archive_sha256": report["archive_sha256"], "source_sha256": sha(data / "source.json")},
        max_transitions=111, max_total_transitions=444, max_images=8, witness_values=4096, max_cache_bytes=10 << 30,
        max_wall_seconds=840, max_archive_bytes=256 << 20,
        observed="stored native prompt carriers from the verified seed31 panel; a fresh seed47 latent and one native base projection; three declared text-only Acts before joint0",
        scope="four own-state numerical forecasts from common-parent initial states: unchanged base and relation/color/shape working-text replacements; all RGB sealed before any native suffix; fixed base conditioning reread later",
        arithmetic="separately wired functional equations sharing low-level Torch CUDA BF16 kernels with native reference",
        limits="observed encoder carriers; full supplied weights; no speed/compression claim; full intermediate RAM-only, offline witnesses and full endpoints; qualitative semantic hypotheses distinct from numerical forecasts",
        hypotheses={"relation": "early changed working text reverses object ordering despite subsequent base conditioning rereads",
            "color": "early changed working text swaps object color assignment despite subsequent base conditioning rereads",
            "shape": "early shape-assignment text changes shapes, potentially mixed with later base conditioning; no clean transfer assumed"},
        terminal_status="unassessed")
    save(run_dir / "plan.json", config)
    worker = here / "flux_branch_forecasts.py"
    extras += [(here / name, name) for name in ("flux_trajectory_forecast_campaign.py", "flux_functional_denoiser.py", "flux_functional_vae.py")]
    extras += [(here / "discovery_upgrades.py", "campaign_support.py"), (run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, "input-store", "carriers", "plan.json",
        "flux_trajectory_forecast_campaign.py", "flux_functional_denoiser.py", "flux_functional_vae.py", "campaign_support.py", preflight.name],
        "path_token_allowlist": ["flux-branch-forecast-result", "input-store", "carriers", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-branch-forecast-request-v1", "config": config, "task_family": "forward", "model": config["model"],
        "needs": {"cuda": True}, "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=8, logical_branches=8,
            logical_checkpoint_replays=4, max_local_work_items=16, checkpoint_policy="one fresh projected parent; three text Acts; four functional forecasts globally sealed then four native suffixes; no encoder, no retries"),
        "resource_exception": {"ram_mb": 24000, "vram_mb": 13000, "cpu_threads": 4,
            "calibration_job": "job-c2a467736b50", "measured_peak_vram_mb": 9470., "measured_peak_rss_mb": 10697.6,
            "ram_calibration_job": "job-898a27a308ae", "ram_calibration_peak_rss_mb": 12567.6,
            "ram_calibration_cache_bytes": 1952514048, "new_cache_cap_bytes": config["max_cache_bytes"],
            "reason": "same batch-one resident512px4step checkpoint and VAE; no encoder. Prior peak minus old1.953GB cache plus new capped10GiB cache and512MiB transaction budget is below24GB; reserve24GB rather than queued32GB. Prior automatic VRAM ceiling exceeded host."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    command = worker_command(worker.name)
    job = guarded_mrun_launch(command, request=request, model=config["model"], task="forward", experiment="mdb-flux-branch-forecasts",
        payload=run_dir / "payload", prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=24000, vram_mb=13000,
        cpu_threads=4, est_wall_s=600, timeout_s=840, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 512px4steps freshseed47. Reuse observed conditioning/text, newlatent+oneprojection; base+relation/color/shape textActs. Four functional444transition+VAE forecasts ALL sealed before four native444transition+VAE comparisons. One model load, zero encodes, CPUcache<=10GiB, 8PNGs, <=840s; no retries; retain all mismatches.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir, args.collect, max_compressed_bytes=256 << 20, max_expanded_bytes=1 << 30)
    else:
        if args.predecessor is None:
            parser.error("--predecessor required")
        result = stage(args.run_dir.resolve(), args.predecessor.resolve())
    print(json.dumps(result))
