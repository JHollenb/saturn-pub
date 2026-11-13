"""Launch the finite bridge residual/attention/MLP block18 factorial."""
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


def stage(run, predecessor):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-anime-depth-result"
    audit, old, frontiers = (json.loads(p.read_text()) for p in (predecessor / "analysis-v1.json", data / "plan.json", data / "frontiers.json"))
    if audit["mechanics_status"] != "valid":
        raise ValueError("audited depth sources required")
    inputs = {role: frontiers["bridge." + source + "/s3.single.10"] for role, source in (("base", "both"), ("source", "left"))}
    relatives, external = set(), {}
    manifest = json.loads((data / "files.json").read_text())
    for value in inputs.values():
        rel = "typed-store/cuts/" + value + ".json"
        cut = json.loads((data / rel).read_text())
        relatives.add(rel)
        relatives.update("typed-store/pages/" + h + ".json" for h in cut["slots"].values())
        relatives.update("typed-store/blobs/" + h + ".safetensors" for h in cut["physical_blobs"])
    for rel in sorted(relatives):
        p = data / rel
        if sha(p) != manifest[rel]:
            raise ValueError("retained CAS differs")
        external[rel] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    part_plan = json.loads((predecessor.parent / "parts-run-01/plan.json").read_text())
    config = {k: old[k] for k in ("model", "model_path", "expected_checkpoint_sha256", "cpu_threads")}
    config.update(schema="mdb-anime-factorial-plan-v1", input_cuts=inputs, external_inputs=external,
        external_input_root=json.loads((predecessor / "job.json").read_text())["result"]["diagnostics"]["cwd"] + "/flux-anime-depth-result",
        part_rows=part_plan["part_rows"]["bridge"], queries=[201, 233, 297, 425, 489, 681, 405, 745], output_dims=list(range(0, 3072, 96)),
        expected_base_png=json.loads((data / "rows/bridge.both.json").read_text())["image"]["sha256"], max_wall_seconds=600,
        scope="Known bridge late-band step19 outcomes. Replay imported own-state single10 sources to single18; prepare upper/lower I-only sources with matched incoming T/L/context. Three source block calls/continuations observed before source-assisted sampled FP64/BF16 projection forecasts. No mixed factorial images or block18 band results observed yet. Native and candidate calls counted separately.",
        hypotheses={
            "H-mechanics": "Three direct source block evaluations exactly reproduce native single18 I/T. Native base/sham exact historical PNG; full R/A/M/T reconstructs each own band-source image and native future. No Navigator hooks modified.",
            "H-locality": "At the fused pre-output concat, MLP-feature differences are confined to the edited upper/lower image positions, while attention can change image/text positions outside that band. Incoming T and time/gate are exact.",
            "H-R": "Tentative: the incoming residual is already an important appearance carrier; R-only more strongly shifts head/costume toward band-source than A-only or M-only at this late block. Keep signed magnitudes and misses.",
            "H-A-M": "Tentative: M edits concentrate more on the edited image band, A has broader collateral; nonlinear readout and following block can mix their effects. Do not infer semantic role from internal energy alone.",
            "H-numeric": "After observing source operands, freeze8query x32output FP64 fused projections and BF16-edge residual updates before edited candidates. Compare signed errors without numerical error gates; this is not an exact CUDA GEMM emulator.",
            "H-scope": "One bridge seed83, block18. Upper/lower geometry includes mixtures. No independent raw-prompt RGB or abstract face/armor operation claim. Last block19 exit remains read-only; this assay uses existing typed writable exit19."})
    run.mkdir(parents=True, exist_ok=False)
    save(run / "plan.json", config)
    worker = here / "flux_anime_factorial.py"
    extras = [(here / "flux_scene_inventory_campaign.py", "flux_scene_inventory_campaign.py"),
        (here / "flux_single_contraction.py", "flux_single_contraction.py"), (here / "discovery_upgrades.py", "campaign_support.py"),
        (run / "plan.json", "plan.json")]
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, *(n for _, n in extras), preflight.name],
        "path_token_allowlist": ["flux-anime-factorial-result", "input-store", config["model_path"], config["external_input_root"]]})
    extras.append((preflight, preflight.name))
    env = intent()
    if env["uv_cache_dir"].startswith("/Users/"):
        raise ValueError("Linux cache profile required")
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-anime-factorial-request-v1", "config": config, "model": config["model"], "task_family": "forward",
        "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "needs": {"cuda": True, "host": "beast"}, "execution_environment": env, "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {n: sha(p) for p, n in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=20, logical_branches=20,
            logical_checkpoint_replays=20, max_local_work_items=20, checkpoint_policy="Bridge2sourceprefixes16steps;3native sources+14RAMfactorial+2coherent+sham20RGB.1load0encodes20candidateblocks96nativeSteps21Acts. Source-assisted numerical forecasts sealed before mixed calls. No retries."),
        "resource_exception": {"ram_mb": 10000, "vram_mb": 13000, "calibration_job": "job-72cf2fcc451d", "measured_peak_rss_mb": 7448,
            "measured_peak_vram_mb": 10472, "reason": "Qualified CUDA BF16 B1 noencode lane. One scene, bounded three source concat ports and17 edited candidate projections;10GBRAM/13GBVRAM measured envelope, no larger physical batch."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward", experiment="mdb-flux-anime-factorial",
        payload=run / "payload", prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=10000, vram_mb=13000,
        cpu_threads=4, est_wall_s=300, timeout_s=600, retry_on_kill=False, preflight=True,
        note="Klein bridge block18 R/A/M factorial upper/lower. CUDA BF16 B1,1load0encode.20RGB20candidateblocks96nativeSteps21Acts. Native fused concat GEMM preserved; sample forecasts8x32 before mixedcalls. Finite110steps25Acts600s;no retries.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--predecessor", type=Path)
    p.add_argument("--status")
    p.add_argument("--collect")
    args = p.parse_args()
    if args.status:
        from mdb.job_support import terminal_job
        _, job, logs = terminal_job(args.status)
        phases = [v for v in logs.splitlines() if v.startswith("MDB_FLUX_ANIME_FACTORIAL_PHASE ")]
        result = {"job_id": args.status, "state": job["state"], "last_phases": phases[-2:],
            "seal_phase": next((v for v in phases if '"kind": "predictions_sealed"' in v), None),
            "numeric_phase": next((v for v in phases if '"kind": "numeric_predictions_sealed"' in v), None),
            "errors": logs[-4000:] if job["state"] == "failed" else None}
    elif args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir, args.collect, log_storage="gzip", max_compressed_bytes=512 << 20, max_expanded_bytes=1 << 30)
    else:
        result = stage(args.run_dir.resolve(), args.predecessor.resolve())
    print(json.dumps(result, sort_keys=True))
