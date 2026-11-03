"""Factor a step-one shape response into native residual/attention/MLP ports."""
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


def stage(run_dir, predecessor, *, recipient_controls=False):
    from mrun.protocol import RAM_KILL_FACTOR, kill_ceiling_mb

    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    helper = runpy.run_path(str(here / "summarize_context_window.py"))
    prior = helper["source_run"](predecessor, "flux-handoff-result", max_archive_bytes=256 << 20, max_member_bytes=128 << 20)
    data, old, report = prior["data"], prior["plan"], json.loads((predecessor / "report.json").read_text())
    audit = json.loads((predecessor / "analysis-v1.json").read_text())
    if (report["job_id"] != "job-6a6fcb685a20" or report["archive_sha256"] != "9ff1c29917cf6339d1d80bc70a8b429a36bda7f3d301297da97f2c6aa9de4f7d"
            or audit["source"]["job_id"] != report["job_id"] or audit["source"]["archive_sha256"] != report["archive_sha256"]
            or audit["mechanics_status"] != "valid" or old.get("panel") != "shape-step1"):
        raise ValueError("unexpected or unaudited predecessor")
    frontiers = json.loads((data / "frontiers.json").read_text())
    if frontiers != audit["frontiers"]:
        raise ValueError("frontiers differ from audited histories")
    cuts = {name: frontiers[name]["after_joint4"] for name in ("base", "changed")}
    inventory = json.loads((data / "files.json").read_text())
    relatives = set()
    for cut_id in cuts.values():
        relative = f"typed-store/cuts/{cut_id}.json"
        cut = json.loads((data / relative).read_text())
        if any(cut["execution_point"].get(key) != value for key, value in {"logical_step": 1, "phase": "single", "index": 0}.items()):
            raise ValueError("wrong source occurrence")
        relatives.add(relative)
        relatives.update("typed-store/pages/" + value + ".json" for value in cut["slots"].values())
        relatives.update("typed-store/blobs/" + value + ".safetensors" for value in cut["physical_blobs"])
    extras = []
    for relative in sorted(relatives):
        if sha(data / relative) != inventory[relative]:
            raise ValueError("source CAS differs")
        extras.append((data / relative, relative.replace("typed-store/", "input-store/", 1)))
    run_dir.mkdir(parents=True, exist_ok=False)
    selection = run_dir / "source-selection.json"
    save(selection, {"predecessor_job": report["job_id"], "archive_sha256": report["archive_sha256"],
        "audit_sha256": sha(predecessor / "analysis-v1.json"), "cuts": cuts,
        "scope": "Observed matched step1 after-joint4 frontiers; known coarse shape outcomes inform selection, not fresh suffix predictions."})
    config = {key: old[key] for key in ("model", "model_path", "expected_checkpoint_sha256", "seed", "height", "width", "steps", "cpu_threads", "prompts")}
    config.update(schema="mdb-flux-shape-writer-plan-v1", panel="shape-step1-writer", block=4, start_step=1,
        prefix_selector={"logical_step": 1, "phase": "single", "index": 0},
        selector={"logical_step": 1, "phase": "single", "index": 4}, reset_step=2,
        input_cuts=cuts, predecessor={"job_id": report["job_id"], "archive_sha256": report["archive_sha256"], "selection_sha256": sha(selection)},
        historical_pngs={name: json.loads((data / "rows" / (original + ".json")).read_text())["image"]["sha256"]
            for name, original in (("base", "base"), ("changed", "changed"), ("merge.111", "after_single4.restore_image"))},
        max_images=11, max_adapter_steps=1100, max_wall_seconds=840, max_archive_bytes=512 << 20,
        hypotheses={
            "H-controls": "copy.sham and merge.000 exactly reproduce changed; merge.111 reconstructs base image at exit while retaining changed text, so it reproduces the coarse after_single4.restore_image, not native base.",
            "H-replay": "Both references exactly reproduce coarse reference PNGs under common base conditioning from step2.",
            "H-RA-rank": "Tentative extrapolation from earlier relation work: final latent p(merge.110) < min(p(merge.100), p(merge.010)), with base=0 and changed=1. All misses remain observations.",
            "H-A-vs-M": "Tentative extrapolation: attention-only replacement moves farther baseward than MLP-only: p010 < p001. This is not a semantic law or gate.",
            "H-shape": "MLP-only is predicted to retain the red sphere; RA is predicted to produce a cube-like or mixed red object. Record uncertainty, ordering, blue shape, scale and palette separately.",
            "H-limits": "This tests conditional operand effects at one overlap boundary; it does not identify the first or minimal shape writer. Fixed recipient MLP or attention can still be essential to later computation."},
        scope="seed47 step1 single4 native R/A/M image-row factorial; observed matched prefixes and two port captures; recipient post-block text retained; all branches reset C tobase before step2; empirical hypotheses, no independent RGB forecast",
        terminal_status="unassessed")
    if recipient_controls:
        config.update(schema="mdb-flux-recipient-controls-plan-v1", panel="shape-step1-recipient-controls", block=5,
            selector={"logical_step": 1, "phase": "single", "index": 5},
            historical_pngs={key: value for key, value in config["historical_pngs"].items() if key in {"base", "changed"}},
            max_images=14, max_adapter_steps=1300, max_wall_seconds=900,
            hypotheses={
                "H-controls": "common-base-C copy.sham and merge.000 exactly reproduce changed; own-C self.11 exactly reproduces self.native in all scheduler arrays and PNG. Native base/changed reproduce historical PNGs.",
                "H-adjacent-A-vs-M": "Tentative adjacent-boundary extrapolation: common-base-C final p010 < p001. No magnitude forecast; all misses retained.",
                "H-adjacent-AM": "Tentative composition prediction: common-base-C final p011 < min(p010,p001). Nonmonotonic outcomes remain evidence.",
                "H-self-shape": "Tentative recipient-only visual prediction: setting its image-row M operand to zero at single5 while preserving A and own future C retains two separate objects and a sphere-like red object. Keep mixed/uncertain shape and collateral observations.",
                "H-self-drift": "Tentative recipient-only prediction: zero A causes larger final latent L2 drift from own-C self.native than zero M. Full/half/zero responses need not be monotonic.",
                "H-scope": "Observed step1 prefixes and native ports are allowed before seal. Donor rows and self rows have different future-C policies. No exact image forecast, universal semantic operation, raw-prompt prediction, or minimal-writer claim."},
            scope="Adjacent step1 single5 R0 A/M donor factorial under common base C from2, plus recipient-only native-operand gains0/.5/1 with ownC throughout and no donor tensor in the recipe;14 dependent native futures, observed prefixes and ports, no RGB numerical forecasts",
            recipient_gain_rows={"self.11": [1., 1.], "self.10": [1., 0.], "self.01": [0., 1.],
                "self.00": [0., 0.], "self.1h": [1., .5], "self.h1": [.5, 1.]})
    save(run_dir / "plan.json", config)
    worker = here / "flux_merge_campaign.py"
    extras += [(here / name, name) for name in ("flux_parallel_merge.py", "flux_trajectory_forecast_campaign.py")]
    extras += [(here / "discovery_upgrades.py", "campaign_support.py"), (run_dir / "plan.json", "plan.json"), (selection, selection.name)]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name,
        "flux_parallel_merge.py", "flux_trajectory_forecast_campaign.py", "campaign_support.py", "plan.json", selection.name, "input-store", preflight.name],
        "path_token_allowlist": ["flux-merge-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    ram_mb, vram_mb = 16000, 13000
    if kill_ceiling_mb(ram_mb) >= 66512 * .95:
        raise ValueError("RAM kill envelope cannot fit reference host")
    request = {"schema": "mdb-flux-recipient-controls-request-v1" if recipient_controls else "mdb-flux-shape-writer-request-v1", "config": config, "task_family": "forward", "model": config["model"],
        "needs": {"cuda": True}, "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=config["max_images"], logical_branches=config["max_images"],
            logical_checkpoint_replays=config["max_images"], max_local_work_items=16,
            checkpoint_policy=("reuse two observed frontiers, advance five blocks each, capture ports,14 in-process futures; donor panel common future C reset; recipient-only panel retains own C; no encoder/retries"
                if recipient_controls else "reuse two observed frontiers, advance four blocks each, observe selected ports, eleven in-process futures, common future C reset, no encoder/retries")),
        "resource_exception": {"ram_mb": ram_mb, "vram_mb": vram_mb, "ram_kill_factor": RAM_KILL_FACTOR,
            "ram_kill_ceiling_mb": kill_ceiling_mb(ram_mb), "reference_host_mb": 66512,
            "calibration_job": report["job_id"], "measured_peak_rss_mb": 2651.2, "measured_peak_vram_mb": 9284,
            "native_calibration_job": "job-c2a467736b50", "native_measured_peak_rss_mb": 10697.6,
            "reason": "Same resident CUDA BF16 B1 512px4step model and VAE; no encoder/forecast cache. Retain historical16GB native margin for bounded retained ports/archive;13GB VRAM validated."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    note = ("Klein4B CUDA BF16 B1 512px seed47. Adjacent step1single5: R0 A/M factorial(common baseC from2) plus recipient-only A/M gains0/.5/1(ownC/no donor).14RGB,10prefixsteps+2portcaptures,10projections,one load/noencoder. <=1300steps/900s,no retries. Preserve all misses/collateral."
        if recipient_controls else "Klein4B CUDA BF16 B1 512px seed47. Step1 single4 R/A/M factorial from2 observedjoint4 frontiers;8 prefixsteps+2portcaptures,11RGB=2refs+copy+8masks. Retainrecipienttext; resetCbase beforestep2 everybranch. <=1100steps/840s,no retries,one load/noencoder. Preserve misses and mixtures.")
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-recipient-controls" if recipient_controls else "mdb-flux-shape-writer", payload=run_dir / "payload", prefer_host="beast", gpu=True, needs=request["needs"],
        ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4, est_wall_s=480 if recipient_controls else 420,
        timeout_s=config["max_wall_seconds"], retry_on_kill=False, preflight=True, note=note)
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
            args.run_dir, args.collect, log_storage="gzip", max_compressed_bytes=512 << 20, max_expanded_bytes=1 << 30)
    else:
        if args.predecessor is None:
            parser.error("--predecessor required")
        result = stage(args.run_dir.resolve(), args.predecessor.resolve())
    print(json.dumps(result))
