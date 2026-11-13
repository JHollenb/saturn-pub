"""Submit a bounded22-future time/depth and carried-state native panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as stream:
        json.dump(v, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def stage(run, predecessor, *, submit=True):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-anime-result"
    audit, old = (json.loads(p.read_text()) for p in (predecessor / "analysis-v1.json", data / "plan.json"))
    if audit["mechanics_status"] != "valid":
        raise ValueError("audited native predecessor required")
    cuts, historical, relatives = {}, {}, set()
    manifest = json.loads((data / "files.json").read_text())
    for scene in ("festival", "bridge"):
        for variant in ("left", "both"):
            key = scene + "." + variant
            row = json.loads((data / "rows" / (key + ".json")).read_text())
            cuts[key], historical[key] = row["edited_cut"], row["image"]["sha256"]
            if sha(data / row["image"]["path"]) != historical[key]:
                raise ValueError("native reference differs")
            relative = "typed-store/cuts/" + cuts[key] + ".json"
            cut = json.loads((data / relative).read_text())
            relatives.add(relative)
            relatives.update("typed-store/pages/" + h + ".json" for h in cut["slots"].values())
            relatives.update("typed-store/blobs/" + h + ".safetensors" for h in cut["physical_blobs"])
        historical[scene + ".L3.full"] = historical[scene + ".left"]
        historical[scene + ".sham.I3.19"] = historical[scene + ".both"]
    external = {}
    for relative in sorted(relatives):
        p = data / relative
        if sha(p) != manifest[relative]:
            raise ValueError("historical CAS differs")
        external[relative] = {"sha256": sha(p), "bytes": p.stat().st_size}
    branches = [{"id": f"I{s}.{i}", "clock": f"s{s}.single.{i}", "slots": ["image"], "support": "left"}
                for s in (2, 3) for i in (10, 19)]
    branches += [{"id": f"IT{s}.19", "clock": f"s{s}.single.19", "slots": ["image", "text"], "support": "full"} for s in (2, 3)]
    branches += [{"id": "L3." + support, "clock": "s3.project.None", "slots": ["latent"], "support": support} for support in ("left", "full")]
    branches += [{"id": "sham.I3.19", "clock": "s3.single.19", "slots": ["image"], "support": "left", "sham": True}]
    config = {"schema": "mdb-flux-anime-depth-plan-v1", "model": old["model"], "model_path": old["model_path"],
        "expected_checkpoint_sha256": old["expected_checkpoint_sha256"], "cpu_threads": 4, "max_adapter_steps": 850,
        "max_wall_seconds": 720, "input_cuts": cuts, "external_inputs": external,
        "external_input_root": json.loads((predecessor / "job.json").read_text())["result"]["diagnostics"]["cwd"] + "/flux-anime-result",
        "historical_pngs": historical, "left_rows": {s: old["supports"][s]["masks"]["left"] for s in ("festival", "bridge")}, "branches": branches,
        "scope": "Known step2 project anime left/both cuts, prior final RGB and joint0 A/M restoration outcomes. Observe own natural suffixes before transplants at declared clocks. No new depth/transplant RGB yet. Native operation traces are diagnostics; complete typed states retained only at specified frontiers. Candidate shape/identity labels remain informal.",
        "hypotheses": {
            "H-replay": "Four natural PNGs, two full step3 latent restores and two late image shams reproduce historical outputs exactly: eight mechanics controls.",
            "H-state-formation": "The equal selected-left step2 latent becomes different after native attention/MLP/readout/scheduler updates; measure the entire time/depth trajectory without assuming monotonicity.",
            "H-step2-IT": "Restoring complete source image+text before the last step2 single block makes native velocity exactly the observed source velocity. The next selected-left latent equals the source because its current left latent was exact; companion latent can still differ. Subsequent feedback may preserve or undo source appearance.",
            "H-Euler": "For each complete I/T restoration, freeze the next-latent tensor before running that branch using recipient FP32 carried latent plus native dt*observed source BF16 velocity, then BF16 cast. Exact same-program forecast, not RGB-from-text or donor-free prediction.",
            "H-appearance": "Tentative: late image-only writes change finer rendering more readily than coarse hair/cast binding; restoring carried left latent at step3 has a larger hair/outfit effect than a single late working-image restore. Preserve misses, facial/prop/relational changes and collateral without forcing these labels or monotonic scores.",
            "H-scope": "Two original anime scenes/seeds and one checkpoint. No calibrated face/eye identity instrument, automatic semantic segmentation or necessary/minimal object program claim."}}
    run.mkdir(parents=True, exist_ok=False)
    save(run / "plan.json", config)
    worker = here / "flux_anime_depth.py"
    extras = [(here / "flux_scene_inventory_campaign.py", "flux_scene_inventory_campaign.py"),
              (here / "discovery_upgrades.py", "campaign_support.py"), (run / "plan.json", "plan.json")]
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, *(n for _, n in extras), preflight.name],
        "path_token_allowlist": ["flux-anime-depth-result", "input-store", config["model_path"], config["external_input_root"]]})
    extras.append((preflight, preflight.name))
    env = intent()
    if submit and env["uv_cache_dir"].startswith("/Users/"):
        raise ValueError("qualified Linux cache profile required")
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-anime-depth-request-v1", "config": config, "task_family": "forward", "model": config["model"],
        "needs": {"cuda": True, "host": "beast"}, "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "execution_environment": env, "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {n: sha(p) for p, n in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=22, logical_branches=22,
            logical_checkpoint_replays=22, max_local_work_items=22, checkpoint_policy="4native suffix traces+18typed later transplants;shared frontiers save prefix work.8knownPNGcontrols.1load0encode,B1sequential,no retries."),
        "resource_exception": {"ram_mb": 10000, "vram_mb": 13000, "calibration_job": "job-cbcb59a0fe22", "measured_peak_rss_mb": 6485,
            "measured_peak_vram_mb": 9300, "reason": "Same qualified resident CUDA BF16 B1.0encodes0candidateblocks, shared prefixes. Bounded extra native frontier cuts and temporary CPU local feature snapshots within existing10GB/13GB envelope."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    if not submit:
        return {"submitted": False}
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-anime-depth", payload=run / "payload", prefer_host="beast", gpu=True, needs=request["needs"],
        ram_mb=10000, vram_mb=13000, cpu_threads=4, est_wall_s=360, timeout_s=720, retry_on_kill=False, preflight=True,
        note="Klein anime time/depth CUDA BF16 B1:4native traces+18late transplants=22RGB,8knowncontrols.1load0encode0candidateblocks. Shared typed frontiers; expected582transitions/22Acts.4Euler numeric forecasts. Stop850transitions/720s;no retries.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run)}


def stage_parts(run, predecessor):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command
    from saturn_pub.store import LocalStore

    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-anime-depth-result"
    old, audit, points = (json.loads(p.read_text()) for p in (data / "plan.json", predecessor / "analysis-v1.json", data / "frontiers.json"))
    if audit["mechanics_status"] != "valid":
        raise ValueError("audited depth panel required")
    manifest = json.loads((data / "files.json").read_text())
    cuts = {s + "." + v: points[s + "." + v + "/s3.single.19"] for s in ("festival", "bridge") for v in ("left", "both")}
    relatives, historical, part_rows = set(), {}, {}
    store = LocalStore(data / "typed-store")
    for scene in ("festival", "bridge"):
        ids = store.load(cuts[scene + ".both"], device="cpu").payload["img_ids"][0]
        upper = [i for i in old["left_rows"][scene] if float(ids[i, 1]) < 11]
        lower = sorted(set(old["left_rows"][scene]) - set(upper))
        if not upper or not lower or sorted(upper + lower) != old["left_rows"][scene]:
            raise ValueError("geometric partition differs")
        part_rows[scene] = {"upper": upper, "lower": lower}
        both = json.loads((data / "rows" / (scene + ".both.json")).read_text())["image"]["sha256"]
        local = json.loads((data / "rows" / (scene + ".I3.19.json")).read_text())["image"]["sha256"]
        historical.update({scene + ".native": both, scene + ".sham": both, scene + ".whole-left": local})
    for digest in cuts.values():
        relative = "typed-store/cuts/" + digest + ".json"
        c = json.loads((data / relative).read_text())
        relatives.add(relative)
        relatives.update("typed-store/pages/" + h + ".json" for h in c["slots"].values())
        relatives.update("typed-store/blobs/" + h + ".safetensors" for h in c["physical_blobs"])
    external = {}
    for relative in sorted(relatives):
        p = data / relative
        if sha(p) != manifest[relative]:
            raise ValueError("source frontier CAS differs")
        external[relative] = {"sha256": sha(p), "bytes": p.stat().st_size}
    config = {k: old[k] for k in ("model", "model_path", "expected_checkpoint_sha256", "cpu_threads", "left_rows")}
    branches = [{"id": "native", "clock": "s3.single.19", "slots": [], "support": "full"},
        {"id": "sham", "clock": "s3.single.19", "slots": ["image"], "support": "left", "sham": True}]
    branches += [{"id": name, "clock": "s3.single.19", "slots": ["image"], "support": support}
                 for name, support in (("whole-left", "left"), ("upper", "upper"), ("lower", "lower"))]
    config.update(schema="mdb-flux-anime-parts-plan-v1", stage="parts", max_adapter_steps=60, max_wall_seconds=480, max_images=10,
        start_point={"logical_step": 3, "phase": "single", "index": 19}, input_cuts=cuts, external_inputs=external,
        external_input_root=json.loads((predecessor / "job.json").read_text())["result"]["diagnostics"]["cwd"] + "/flux-anime-depth-result",
        historical_pngs=historical, part_rows=part_rows, branches=branches,
        scope="Observed depth panel and anonymous detail review; import exact step3 before-last-single-block source/recipient states. Partition the already tested selected-left working-image rows into native H<11 upper band and complementary lower band. Geometric upper band includes head, shoulder/hood and environment; no pure face/eye semantic mask. No new upper/lower RGB yet.",
        hypotheses={"H-replay": "Six exact controls: two untouched native, two shams and two whole-left repeats of I3.19.",
            "H-partition": "Upper/lower are disjoint and their union exactly covers348 selected-left native rows. All nonimage state and unselected image rows stay exact.",
            "H-upper": "Tentative: upper-only late working-image transplant changes the observed left head/hair toward the black-bob festival or violet-haired bridge source while preserving more recipient lower costume/prop state than whole-left. Observe face/eye contours without assigning calibrated identity.",
            "H-lower": "Tentative: lower-only changes clothing/armor/props more than left head; head transfer by lower-only would expose distributed/nonlocal dependence.",
            "H-scope": "Two contexts, geometric bands, no independent RGB prediction or automatic semantic factorization. Preserve crossed and off-target outcomes."})
    run.mkdir(parents=True, exist_ok=False)
    save(run / "plan.json", config)
    worker = here / "flux_anime_depth.py"
    extras = [(here / "flux_scene_inventory_campaign.py", "flux_scene_inventory_campaign.py"),
              (here / "discovery_upgrades.py", "campaign_support.py"), (run / "plan.json", "plan.json")]
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, *(n for _, n in extras), preflight.name],
        "path_token_allowlist": ["flux-anime-depth-result", "input-store", config["model_path"], config["external_input_root"]]})
    extras.append((preflight, preflight.name))
    env = intent()
    if env["uv_cache_dir"].startswith("/Users/"):
        raise ValueError("Linux cache profile required")
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-anime-parts-request-v1", "config": config, "model": config["model"], "task_family": "forward",
        "needs": {"cuda": True, "host": "beast"}, "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "execution_environment": env, "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {n: sha(p) for p, n in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=10, logical_branches=10,
            logical_checkpoint_replays=10, max_local_work_items=10, checkpoint_policy="Imported native late frontiers,10sequential RGB;6knowncontrols4new band edits.1load0encode30transitions8Acts;no retries."),
        "resource_exception": {"ram_mb": 10000, "vram_mb": 13000, "calibration_job": "job-72cf2fcc451d", "measured_peak_rss_mb": 7448,
            "measured_peak_vram_mb": 10472, "reason": "Same measured installed-wheel lane. Only4imported late cuts,30transitions vs582,10decodes vs22; no trace cache."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward", experiment="mdb-flux-anime-parts",
        payload=run / "payload", prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=10000, vram_mb=13000, cpu_threads=4,
        est_wall_s=150, timeout_s=480, retry_on_kill=False, preflight=True,
        note="Klein anime geometric upper/lower late-image partition. CUDA BF16 B1.10RGB6controls4new.1load0encode,30native transitions8Acts from imported step3single19.Upper96rows/lower252 per scene;not semantic masks. Stop60transitions/480s;no retries.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--predecessor", type=Path)
    p.add_argument("--stage-only", action="store_true")
    p.add_argument("--parts", action="store_true")
    p.add_argument("--status")
    p.add_argument("--collect")
    args = p.parse_args()
    if args.status:
        from mdb.job_support import terminal_job
        _, job, logs = terminal_job(args.status)
        phases = [v for v in logs.splitlines() if v.startswith("MDB_FLUX_ANIME_DEPTH_PHASE ")]
        result = {"job_id": args.status, "state": job["state"], "last_phases": phases[-2:],
            "sealed_phase": next((v for v in phases if '"kind": "predictions_sealed"' in v), None),
            "errors": logs[-4000:] if job["state"] == "failed" else None}
    elif args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir, args.collect, log_storage="gzip", max_compressed_bytes=512 << 20, max_expanded_bytes=1 << 30)
    elif args.parts:
        result = stage_parts(args.run_dir.resolve(), args.predecessor.resolve())
    else:
        result = stage(args.run_dir.resolve(), args.predecessor.resolve(), submit=not args.stage_only)
    print(json.dumps(result, sort_keys=True))
