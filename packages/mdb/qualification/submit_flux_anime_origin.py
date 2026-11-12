"""Launch reciprocal late-step appearance transport curves through public MDB."""
import argparse
import hashlib
import json
import runpy
import subprocess
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def stage(run, predecessor):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-anime-depth-result"
    audit, old, points, manifest = (read(p) for p in (predecessor / "analysis-v1.json", data / "plan.json", data / "frontiers.json", data / "files.json"))
    if audit["mechanics_status"] != "valid":
        raise ValueError("audited depth source required")
    inputs = {role: points["bridge." + variant + "/s3.project.None"] for role, variant in (("base", "both"), ("source", "left"))}
    relatives = set()
    for digest in inputs.values():
        rel = "typed-store/cuts/" + digest + ".json"
        cut = read(data / rel)
        relatives.add(rel)
        relatives.update("typed-store/pages/" + h + ".json" for h in cut["slots"].values())
        relatives.update("typed-store/blobs/" + h + ".safetensors" for h in cut["physical_blobs"])
    run.mkdir(exist_ok=False, parents=True)
    filelist = run / "input-files.txt"
    filelist.write_text("\n".join(sorted(relatives)) + "\n")
    # Restore just two source closures, preserving every existing local file.
    cold = "/mnt/hdd8tb/cold-storage/mbp1/domains/2026-11-04-artifact-offload/artifacts/mdb-flux-anime-20261103/depth-run-01/extracted/flux-anime-depth-result"
    subprocess.run(["rtk", "proxy", "rsync", "-ar", "--ignore-existing", "--no-owner", "--no-group", "--protect-args", "--files-from=" + str(filelist), "beast:" + cold + "/", str(data) + "/"], check=True)
    external = {}
    for rel in sorted(relatives):
        p = data / rel
        if sha(p) != manifest[rel]:
            raise ValueError("restored source CAS differs")
        external[rel] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    part = read(predecessor.parent / "parts-run-01/plan.json")
    historical = {role: read(data / "rows" / ("bridge." + variant + ".json"))["image"]["sha256"] for role, variant in (("base", "both"), ("source", "left"))}
    config = {k: old[k] for k in ("model", "model_path", "expected_checkpoint_sha256", "cpu_threads")}
    config.update(schema="mdb-anime-origin-plan-v1", input_cuts=inputs, external_inputs=external, external_input_root=cold,
        part_rows=part["part_rows"]["bridge"], max_wall_seconds=720, output_scratch="/home/beast/.cache/mdb-research",
        historical_pngs={prefix + role: h for role, h in historical.items() for prefix in ("native.", "sham.")},
        points=[{"name": phase + str(i), "phase": phase, "index": i} for phase, i in (("joint", 0), ("joint", 4), ("single", 0), ("single", 6), ("single", 10), ("single", 14), ("single", 18))],
        scope="Known bridge late block18 R/A/M outcomes; two own-state step3 projection frontiers. Corrected SSD-output/NVMe-package replication after cancelled HDD journal-stalled job-29b0209a2212 and bootstrap-only job-5c4862203de2. Preliminary native/joint0 PNGs from first incomplete attempt inspected and execution-null; original qualitative hypotheses retained. NVMe Torch2.13.0/Diffusers0.39.0 import mechanics13.05s qualified without model execution. Test reciprocal upper/lower image-only transfer along seven native boundaries. Maps conditional transferability and repair, not concept birth; later single-boundary curve and refined-site results remain unseen.",
        hypotheses={
            "H-mechanics": "Two natural continuations and two shams match historical PNGs exactly. Band Acts preserve nonimage state and all outside image rows. 32 RGB, zero candidate calls, one load, no encodes.",
            "H-late": "Tentative: late upper transfer changes head preferentially and lower changes armor/costume preferentially in both directions. Signed pixels and native visual observations may disagree and remain retained.",
            "H-depth": "Tentative: early band transfers are more susceptible to downstream repair or broader collateral than late ones; no monotonicity is assumed. Recipient/source direction can differ.",
            "H-formation": "A difference in transfer response across adjacent sampled sites identifies an interval for the next writer/repair cut. It does not prove concept birth or minimal necessity.",
            "H-scope": "One bridge seed, supplied geometry, no calibrated iris/identity variables or independent prompt-to-image forecasting. Original two histories differ outside the band; each intervention preserves its own receiving closure."})
    save(run / "plan.json", config)
    worker = here / "flux_anime_origin.py"
    extras = [(here / "flux_scene_inventory_campaign.py", "flux_scene_inventory_campaign.py"), (here / "discovery_upgrades.py", "campaign_support.py"), (run / "plan.json", "plan.json")]
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, *(n for _, n in extras), preflight.name], "path_token_allowlist": ["flux-anime-origin-result", "input-store", config["model_path"], cold, config["output_scratch"]]})
    extras.append((preflight, preflight.name))
    env = intent()
    if env["uv_cache_dir"].startswith("/Users/"):
        raise ValueError("Linux worker cache profile required")
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-anime-origin-request-v1", "config": config, "model": config["model"], "task_family": "forward", "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "needs": {"cuda": True, "host": "beast"}, "execution_environment": env, "runtime_environment": {"PYTHONDONTWRITEBYTECODE": "1"}, "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "source_hashes": {name: sha(p) for p, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=32, logical_branches=32, logical_checkpoint_replays=32, max_local_work_items=32, checkpoint_policy="2native traces shared7frontiers;28reciprocal band edits+2shams.32RGB528transitions30Acts.1load0encode0candidate,B1sequential,no retries."),
        "resource_exception": {"ram_mb": 10000, "vram_mb": 13000, "calibration_job": "job-72cf2fcc451d", "measured_peak_rss_mb": 7448, "measured_peak_vram_mb": 10472, "reason": "Same resident CUDA BF16 B1 no-encode native replay lane;14 retained frontiers, no candidate concat ports. Bounded envelope from audited22-future depth job."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    command = worker_command(worker.name)
    command.insert(1, "PYTHONDONTWRITEBYTECODE=1")
    job = guarded_mrun_launch(command, request=request, model=config["model"], task="forward", experiment="mdb-flux-anime-origin", payload=run / "payload", prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=10000, vram_mb=13000,
        cpu_threads=4, est_wall_s=360, timeout_s=720, retry_on_kill=False, preflight=True,
        note="Klein bridge backward origin scout:7depth points x2bands x2directions,2native+2shams=32RGB. CUDA BF16 physicalB1,1load0encode0candidate. Expected528nativeSteps30Acts;stop600steps34Acts720s,no retries. Map transfer/repair without monotonic semantic gates.")
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
