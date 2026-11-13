"""Submit thirteen native joint4 source/mediation futures from audited old cuts."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def stage(run_dir, predecessor):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-joint-writes-result"
    audit, recovery, old = (json.loads((predecessor / n).read_text()) for n in ("analysis-v1.json", "recovery.json", "plan.json"))
    if (sha(predecessor / "analysis-v1.json") != "158a1a5e52540f928326c16570f94dfbd86f101637380a26b47c72e139212e7e"
            or audit["mechanics_status"] != "valid" or recovery["job_id"] != "job-3319beda6ea6"
            or sha(data / "files.json") != recovery["remote_manifest_sha256"]
            or recovery["remote_manifest_sha256"] != "e68bbdd6ffa98b99a4e4a309429280d8e9156f44139459eb6b41f3c429caa5f8"):
        raise ValueError("unexpected or unaudited recovered predecessor")
    prefixes = json.loads((data / "observed-prefixes.json").read_text())
    cuts = {name: prefixes["prefixes"][name]["4"]["parent"] for name in ("base", "changed")}
    inventory = json.loads((data / "files.json").read_text())
    relatives = set()
    for cut in cuts.values():
        relative = f"typed-store/cuts/{cut}.json"
        record = json.loads((data / relative).read_text())
        if any(record["execution_point"].get(k) != v for k, v in {"logical_step": 1, "phase": "joint", "index": 4}.items()):
            raise ValueError("incorrect selected clock")
        relatives.add(relative)
        relatives.update("typed-store/pages/" + h + ".json" for h in record["slots"].values())
        relatives.update("typed-store/blobs/" + h + ".safetensors" for h in record["physical_blobs"])
    extras = []
    for relative in sorted(relatives):
        if sha(data / relative) != inventory[relative]:
            raise ValueError("source CAS differs")
        extras.append((data / relative, relative.replace("typed-store/", "input-store/", 1)))
    run_dir.mkdir(parents=True, exist_ok=False)
    selection = run_dir / "source-selection.json"
    save(selection, {"job_id": recovery["job_id"], "raw_manifest_sha256": recovery["remote_manifest_sha256"],
        "audit_sha256": sha(predecessor / "analysis-v1.json"), "cuts": cuts,
        "predecessor_status": "failed final archive publication; complete raw files recovered without inference and independently audited"})
    branches = [{"id": "A.base.recompute", "kind": "output"},
                {"id": "A.base.M.changed", "kind": "output", "lock_changed_mlp": True}]
    branches += [{"id": "qkv." + "".join(bits), "kind": "qkv", "mask": "".join(bits)} for bits in itertools.product("01", repeat=3)]
    branches += [{"id": "textQ.base", "kind": "qkv", "text_query": True, "mask": "000"}]
    config = {k: old[k] for k in ("model", "model_path", "expected_checkpoint_sha256", "seed", "height", "width", "steps", "cpu_threads", "prompts")}
    config.update(schema="mdb-flux-joint-sources-plan-v1", input_cuts=cuts, branches=branches,
        max_images=13, max_adapter_steps=1100, max_candidate_calls=16, max_wall_seconds=840,
        historical_pngs={name: json.loads((data / "rows" / (original + ".json")).read_text())["image"]["sha256"]
            for name, original in (("base", "base"), ("changed", "changed"), ("A.base.recompute", "j4.A"))},
        hypotheses={
            "H-replay": "Native base/changed and recomputed-MLP attention replacement reproduce their three historical PNGs exactly.",
            "H-source-corners": "qkv000 exactly reproduces changed; qkv111 (base imageQ/textKV/imageKV, recipient textQ) exactly reproduces A.base.recompute, with base image-attention return exactly. Compare every retained scheduler array and PNG.",
            "H-textQ": "Base text queries alone leave current image and final common-text-clamped future exactly changed; their altered text return is disposed of at joint4 exit.",
            "H-direct-shape": "Tentative new prediction: base image attention with original changed MLP return held fixed still renders a cube-like red object. A mixed or spherical outcome is a counterexample, retained separately.",
            "H-source-rank": "Tentative: p(qkv010)<p(qkv001): text K/V restoration is more baseward than image K/V restoration. p is latent geometry, not a shape score.",
            "H-source-interaction": "Tentative: p(qkv110)<min(p(qkv100),p(qkv010)): image-query plus text-KV restoration is stronger baseward than either alone. Other source factorial outcomes exploratory.",
            "H-collateral": "Red-left/blue-right and blue cube expected to persist; retain mixed shapes, other-object edits, scale, lighting and pose separately."},
        scope="Seed47 step1 joint4 source/mediation panel. Observed matched entry cuts; common base text after joint4 and base conditioning beforestep2. Native positioned Q/K/V source factorial and fixed-vs-recomputed image MLP. Thirteen dependent native futures; no independent numerical RGB forecast.",
        terminal_status="unassessed")
    save(run_dir / "plan.json", config)
    worker = here / "flux_joint_sources_campaign.py"
    extras += [(here / n, n) for n in ("flux_joint_sources.py", "flux_joint_writes.py", "flux_handoff_campaign.py", "flux_trajectory_forecast_campaign.py")]
    extras += [(here / "discovery_upgrades.py", "campaign_support.py"), (selection, selection.name), (run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name,
        "flux_joint_sources.py", "flux_joint_writes.py", "flux_handoff_campaign.py", "flux_trajectory_forecast_campaign.py",
        "campaign_support.py", "input-store", "plan.json", selection.name, preflight.name],
        "path_token_allowlist": ["flux-joint-sources-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-joint-sources-request-v1", "config": config, "task_family": "forward", "model": config["model"],
        "needs": {"cuda": True}, "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {n: sha(p) for p, n in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=13, logical_branches=13,
            logical_checkpoint_replays=13, max_local_work_items=13,
            checkpoint_policy="2 observedjoint4 parents;2sourceport/native paritycalls;13 inprocess RGBfutures,<=16candidateblocks. Common text/C reset. Noencoder/retries; compact retainedports without duplicate candidate images."),
        "resource_exception": {"ram_mb": 16000, "vram_mb": 13000, "calibration_job": "job-3319beda6ea6",
            "measured_peak_rss_mb": 3208.7, "measured_peak_vram_mb": 9480,
            "reason": "Same CUDA BF16 B1 512px4step resident model/VAE; extra two positioned QKV ports remain bounded, no encoder/forecast cache. Keep calibrated16GB RAM/13GB VRAM. Prior job completed science but failed inline archive publication."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-joint-sources", payload=run_dir / "payload", prefer_host="beast", gpu=True, needs=request["needs"],
        ram_mb=16000, vram_mb=13000, cpu_threads=4, est_wall_s=420, timeout_s=840, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 seed47 joint4.2refs+recomputeA+lockedM+8Q/textKV/imageKV masks+textQnull=13RGB. <=1100transitions/16candidateblocks/840s. Common Tbase afterj4/Cbase beforestep2.1load/0encodes/no retries. Preserve misses; no duplicate candidate image arrays in archive.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--predecessor", type=Path)
    p.add_argument("--collect")
    a = p.parse_args()
    if a.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            a.run_dir, a.collect, log_storage="gzip", max_compressed_bytes=512 << 20, max_expanded_bytes=1 << 30)
    else:
        if a.predecessor is None:
            p.error("--predecessor required")
        result = stage(a.run_dir.resolve(), a.predecessor.resolve())
    print(json.dumps(result))
