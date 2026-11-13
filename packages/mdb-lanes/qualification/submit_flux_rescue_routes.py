"""Submit one bounded paired-rescue handoff and early text-KV assay."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def stage(run_dir, predecessor, *, submit=True):
    from mrun.protocol import RAM_KILL_FACTOR, kill_ceiling_mb

    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    h = runpy.run_path(str(here / "summarize_context_window.py"))
    prior = h["source_run"](predecessor, "flux-carriers-result", max_archive_bytes=512 << 20, max_member_bytes=128 << 20)
    data, old = prior["data"], prior["plan"]
    audit = json.loads((predecessor / "analysis-v1.json").read_text())
    if (prior["source"]["job_id"] != "job-a1afc382b4fe"
            or prior["source"]["archive_sha256"] != "dc4eac744092bd10845b2e9242b9b939241eafe82a2f7b6e6d4619715d27922b"
            or audit["source"] != prior["source"] or audit["mechanics_status"] != "valid"
            or audit["instrument_status"] != "native-replay-qualified"):
        raise ValueError("source is not the audited same-latent conditioning rescue")
    rows = {name: json.loads((data / "rows" / (row + ".json")).read_text())
            for name, row in (("zero", "L1.C0"), ("own", "L1.C0.rescue3"))}
    cuts = {name: next(item["post_cut"] for item in row["project_reads"] if item["logical_step"] == 3)
            for name, row in rows.items()}
    inventory = json.loads((data / "files.json").read_text())
    relatives = set()
    for cut_id in cuts.values():
        relative = f"typed-store/cuts/{cut_id}.json"
        cut = json.loads((data / relative).read_text())
        point = cut["execution_point"]
        if (cut["fingerprint"] != cut_id or point["phase"] != "joint" or point["logical_step"] != 3
                or point["index"] != 0 or point["local_clock"] != 0):
            raise ValueError("source cut is not the paired final-step joint0 entry")
        relatives.add(relative)
        relatives.update("typed-store/pages/" + page + ".json" for page in cut["slots"].values())
        relatives.update("typed-store/blobs/" + blob + ".safetensors" for blob in cut["physical_blobs"])
    extras, input_hashes = [], {}
    for relative in sorted(relatives):
        path = data / relative
        if sha(path) != inventory[relative]:
            raise ValueError("source CAS differs from original inventory")
        target = relative.replace("typed-store/", "input-store/", 1)
        extras.append((path, target))
        input_hashes[target] = sha(path)
    campaign = runpy.run_path(str(here / "flux_rescue_routes_campaign.py"))
    config = {key: old[key] for key in ("model", "model_path", "expected_checkpoint_sha256", "prompt", "seed", "height", "width", "steps", "cpu_threads")}
    config.update(schema="mdb-flux-rescue-routes-plan-v1", input_cuts=cuts, input_store="input-store",
        source_selection="source-selection.json", selector={"logical_step": 3, "phase": "joint", "index": 0},
        branches=campaign["BRANCHES"], max_adapter_steps=260, max_acts=11, max_candidate_calls=3,
        max_images=9, max_vae_calls=9, max_wall_seconds=600, max_archive_bytes=512 << 20, archive_compression_level=1,
        expected_pngs={name: row["image"]["sha256"] for name, row in rows.items()},
        hypotheses={
            "H-entry": "Paired native joint0 inputs have identical latent, projected image, complete time/modulation/gates/RoPE context, schedule and IDs; only working T and stored C differ. Image Q/K/V are expected bit-exact. Text Q from the zero-C source is not assumed numerically zero.",
            "H-local-KV": "Freeze implementation prediction: replacing only positioned text K/V in the zero-source joint0 call with own-source text K/V reproduces own-source image attention return, receiving MLP return and post-block image bytes. Effective Q (including text Q) and image K/V remain the zero-source tensors. Text output need not reproduce own-source T. Retain finite-precision misses and qualify the local instrument separately.",
            "H-controls": "Zero and own reference Z4 latent/PNG reproduce prior L1.C0 and L1.C0.rescue3. Copy sham matches zero. Late coherent I/T and early coherent I/T match own final latent/PNG despite retaining stored C0: no further conditioning projection or denoiser numerical use of stored C occurs in the remaining final step. Schema/custody checks may still inspect C. Early KV image-only and both-return arms install identical image bytes; their selected T differs.",
            "H-late-image": "Tentative componentwise visual prediction: the own-source whole-joint-stack image carrier with zero-source text at single0 carries some own-rescue improvement in red appearance and blue appearance/surface clarity relative to zero. Preserve separate color, shape, protrusion and clutter observations; no composite pass. It contains cumulative attention and MLP effects, not an isolated attention operand.",
            "H-exploratory": "Text-only and combination ranks, clutter/geometry repair, early-KV final effects and all numerical distance ranks are exploratory. Weak final early-KV effect may reflect compensation in four remaining joint blocks. Early-KV and late-image intervene at different frontiers and are not predicted to have identical futures.",
            "H-scope": "One observed native seed47 history and actual altered L3; source reference is its own paired C read, not a foreign donor. Whole banks and whole image/text returns, no head/token/object address, universal concept dictionary or independent RGB forecast. C-zero remains an artificial condition; every future, countertrend and miss is retained."
        },
        scope="One authenticated altered Z3 latent, two saved postprojection joint0 inputs. Paired five-joint prefixes shared once; six late single0 futures plus three early joint0-exit futures.220 adapter transitions,3 direct selected-block calculations,11 typed Acts,9 final PNGs/VAE calls; no encoder/noise draw. Main image/text handoff factorial and narrower early text-KV source diagnostic. No independent RGB prediction or transferable semantic law.",
        terminal_status="unassessed")
    campaign["validate_config"](config)
    run_dir.mkdir(parents=True, exist_ok=False)
    selection = {"schema": "mdb-flux-rescue-routes-selection-v1", "prior_job_id": prior["source"]["job_id"],
        "archive_sha256": prior["source"]["archive_sha256"], "request_sha256": prior["request"]["request_sha256"],
        "source_record": {"path": "prior-source.json", "sha256": sha(data / "source.json")},
        "audit_sha256": sha(predecessor / "analysis-v1.json"), "input_cuts": cuts, "files": input_hashes,
        "expected_pngs": config["expected_pngs"], "prior_rows": {name: {"path": f"prior-{name}-row.json", "sha256": sha(data / "rows" / (row["id"] + ".json"))} for name, row in rows.items()},
        "scope": "Known source-reference images are replay controls. New factorial/KV intervention images are unseen at hypothesis sealing."}
    save(run_dir / "source-selection.json", selection)
    save(run_dir / "plan.json", config)
    worker = here / "flux_rescue_routes_campaign.py"
    extras += [(here / name, name) for name in ("flux_rescue_routes.py", "flux_joint_self.py")]
    extras += [(here / "discovery_upgrades.py", "campaign_support.py"), (data / "source.json", "prior-source.json"),
        (predecessor / "analysis-v1.json", "prior-audit.json"), (run_dir / "plan.json", "plan.json"),
        (run_dir / "source-selection.json", "source-selection.json")]
    extras += [(data / "rows" / (row["id"] + ".json"), f"prior-{name}-row.json") for name, row in rows.items()]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, "flux_rescue_routes.py", "flux_joint_self.py", "campaign_support.py", "prior-source.json",
            "prior-zero-row.json", "prior-own-row.json", "prior-audit.json", "plan.json", "source-selection.json", "input-store", preflight.name],
        "path_token_allowlist": ["flux-rescue-routes-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    ram_mb, vram_mb = 16000, 13000
    request = {"schema": "mdb-flux-rescue-routes-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "backend": "diffusers", "device": "cuda",
        "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512, "needs": {"cuda": True},
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=9, logical_branches=9,
            logical_checkpoint_replays=9, max_local_work_items=12, checkpoint_policy="Two audited Z3joint0 entries; share paired5joint prefixes.6late22-step futures+3early26-step futures=220adapter steps;3direct joint0calls,11Acts,9finalPNG. One model/noencoder/noise/retries."),
        "resource_exception": {"ram_mb": ram_mb, "vram_mb": vram_mb, "ram_kill_factor": RAM_KILL_FACTOR,
            "ram_kill_ceiling_mb": kill_ceiling_mb(ram_mb), "calibration_job": "job-a1afc382b4fe",
            "measured_peak_rss_mb": 2670.8, "measured_peak_vram_mb": 9342,
            "failed_execution": {"job_id": "job-cfe22f928044", "state": "killed_ram", "phase": "loading before seal",
                "declared_ram_mb": 6144, "kill_ceiling_mb": 6758, "observed_kill_tree_rss_mb": 7024,
                "sampled_meta_peak_rss_mb": 1129.3, "scientific_outcomes": 0, "archive_completed": False},
            "reason": "Restore the previously qualified16000MB RAM envelope/17600MB kill ceiling for the same CUDA BF16 B1 model/VAE. The2670.8MB predecessor sample missed startup transients: attempt01 exceeded6758MB before sealing during loading/residency setup. This is a resource-only manual resubmission of the same specimen/panel/worker/source cuts, not an automatic retry or new scientific trial. Keep13GB VRAM, no encoder or repeated source projection; retain failed custody and actual completion peaks. Reservation is headroom, not a speedup claim."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    if not submit:
        return {"run_dir": str(run_dir), "submitted": False, "request_sha256": request["request_sha256"]}
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-rescue-routes", payload=run_dir / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4, est_wall_s=180,
        timeout_s=600, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 512px finalZ3 own-C partialrescue. Sharedpaired5joint prefixes;6late22step+3early26step=9futures/220transitions/3directjoint0calls/11Acts/9VAE. Image-text handofffactorial+earlytextKV diagnostic. Noenc/newnoise/foreign donor/retries. <=260steps/600s,gzip1; allmissesretained.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir, args.collect, log_storage="gzip", max_compressed_bytes=512 << 20, max_expanded_bytes=1 << 30)
    else:
        if args.predecessor is None:
            parser.error("--predecessor is required for a new submission")
        result = stage(args.run_dir.resolve(), args.predecessor.resolve(), submit=not args.stage_only)
    print(json.dumps(result))
