"""Submit a finite carried-latent versus future-conditioning replay panel."""
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

    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    h = runpy.run_path(str(here / "summarize_context_window.py"))
    prior = h["source_run"](predecessor, "flux-joint-self-result", max_archive_bytes=512 << 20, max_member_bytes=128 << 20)
    data, old = prior["data"], prior["plan"]
    audit = json.loads((predecessor / "analysis-v1.json").read_text())
    source = json.loads((data / "source.json").read_text())
    native = json.loads((data / "rows/native.json").read_text())
    if (prior["source"]["job_id"] != "job-1f1587e6983b"
            or prior["source"]["archive_sha256"] != "db2c1af6a818bcb481eb5abb7b02126af9e0c6f2615061e77267f0c09800637e"
            or audit["source"] != prior["source"] or audit["mechanics_status"] != "valid"):
        raise ValueError("source is not the audited native-history campaign")
    cuts = {"Z0": source["initial_cut"], "Z2": next(item["cut"] for item in native["schedulers"] if item["step"] == 2)}
    inventory = json.loads((data / "files.json").read_text())
    relatives = set()
    for name, cut_id in cuts.items():
        relative = f"typed-store/cuts/{cut_id}.json"
        body = json.loads((data / relative).read_text())
        if (body["fingerprint"] != cut_id or body["execution_point"]["phase"] != "project"
                or body["execution_point"]["logical_step"] != int(name[1:]) or body["execution_point"]["local_clock"] != 0):
            raise ValueError("source cut clock differs")
        relatives.add(relative)
        relatives.update("typed-store/pages/" + page + ".json" for page in body["slots"].values())
        relatives.update("typed-store/blobs/" + blob + ".safetensors" for blob in body["physical_blobs"])
    extras, input_hashes = [], {}
    for relative in sorted(relatives):
        path = data / relative
        if sha(path) != inventory[relative]:
            raise ValueError("source CAS bytes differ from audited inventory")
        target = relative.replace("typed-store/", "input-store/", 1)
        extras.append((path, target))
        input_hashes[target] = sha(path)
    campaign = runpy.run_path(str(here / "flux_carriers_campaign.py"))
    config = {key: old[key] for key in ("model", "model_path", "expected_checkpoint_sha256", "prompt", "seed", "height", "width", "steps", "cpu_threads")}
    config.update(schema="mdb-flux-context-carriers-plan-v1", input_cuts=cuts,
        input_store="input-store", source_selection="source-selection.json", selector={"logical_step": 2, "phase": "project", "index": None},
        branches=campaign["BRANCHES"], preview_rows=["native", "L1.C0", "L0.C1", "L0.C0"], preview_steps=[3],
        max_adapter_steps=650, max_final_images=11, max_preview_images=4, max_images=15,
        max_vae_calls=16, max_wall_seconds=600, max_archive_bytes=512 << 20, archive_compression_level=1,
        expected_native_png_sha256=native["image"]["sha256"],
        hypotheses={
            "H-carried-state": "Tentative componentwise visual forecast: with C=0 at both future projections, exact native L2 retains clearer evidence than the derived noise anchor of (a) two separate objects, (b) red-left/blue-right assignment, (c) red sphere-like form, and (d) blue cube-like form. Preserve four separate directions, ties, mixed or unknown outcomes; no composite pass.",
            "H-future-source": "Tentative componentwise visual forecast: from the derived noise anchor, own C yields clearer (a) red coloration, (b) blue coloration, (c) red sphere-like form, and (d) blue cube-like form than C=0. Spatial ordering is exploratory. No requirement that all attributes agree.",
            "H-late-rescue": "Tentative componentwise visual forecast: restoring original own C before Z3 projection improves prompt-associated color assignment and object shape identity relative to continuing the exact same altered Z3 with C=0. Assess the native-L2 and anchor histories separately; complete native recovery and ordering are exploratory.",
            "H-doses": "Half latent-residual and half C responses are exploratory. Half embedding amplitude is not half source influence. Retain actual installed arrays, projected text/image, signed downstream changes and nonmonotonic responses.",
            "H-controls": "Untouched native continuation reproduces its prior native PNG. Copy sham reproduces current native scheduler arrays and PNG exactly. The direct preview helper reproduces the final public adapter decode exactly. Rescue branches start from their own actual edited zero-C Z3 cut, retaining its latent and metadata.",
            "H-scope": "Observed native Z0 and Z2 from one fresh literal prompt/seed are allowed inputs. N2=sigma2*Z0 is a derived zero-endpoint anchor, not observed noise inside L2; R2=L2-N2 is an anchor-relative residual, not proved scene memory. C-zero is an artificial conditioner input; model priors, biases, positions and time remain. No foreign donor, new encoder, independent RGB forecast or universal semantic operation. No extra numerical rank is predicted."
        },
        scope="One native seed47 prompt history replayed from observed Z2. Eight latent/C dose cells, copy sham, two own-C rescues from actual edited Z3. Native L2 versus FP32-derived sigma2*Z0 anchor with half residual dose; own C gains1/.5/0. Nine complete two-step futures plus two shared-prefix one-step suffixes:11 logical futures,560 physical adapter transitions,18 Acts,20 actual projection reads,11 final PNGs+4 noisy previews+one VAE parity call. Conditional dependence/recoverability, no semantic memory-erasure claim.",
        terminal_status="unassessed")
    campaign["validate_config"](config)
    run_dir.mkdir(parents=True, exist_ok=False)
    selection = {"schema": "mdb-flux-native-carrier-selection-v1", "prior_job_id": prior["source"]["job_id"],
        "archive_sha256": prior["source"]["archive_sha256"], "request_sha256": prior["request"]["request_sha256"],
        "source_record": {"path": "prior-source.json", "sha256": sha(data / "source.json")},
        "audit_sha256": sha(predecessor / "analysis-v1.json"), "prior_native_row_sha256": sha(data / "rows/native.json"),
        "expected_native_png_sha256": native["image"]["sha256"], "input_cuts": cuts, "files": input_hashes,
        "scope": "Two observed cuts from one authenticated native prompt history; prior native image is a known replay control. New intervention and rescue images are unseen when hypotheses are sealed."}
    save(run_dir / "source-selection.json", selection)
    save(run_dir / "plan.json", config)
    worker = here / "flux_carriers_campaign.py"
    extras += [(here / name, name) for name in ("flux_carriers.py", "flux_joint_self.py")]
    extras += [(here / "discovery_upgrades.py", "campaign_support.py"),
        (data / "source.json", "prior-source.json"), (data / "rows/native.json", "prior-native-row.json"),
        (predecessor / "analysis-v1.json", "prior-audit.json"),
        (run_dir / "plan.json", "plan.json"), (run_dir / "source-selection.json", "source-selection.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, "flux_carriers.py", "flux_joint_self.py", "campaign_support.py", "prior-source.json",
            "prior-native-row.json", "prior-audit.json", "plan.json", "source-selection.json", "input-store", preflight.name],
        "path_token_allowlist": ["flux-carriers-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    ram_mb, vram_mb = 16000, 13000
    if kill_ceiling_mb(ram_mb) >= 66512 * .95:
        raise ValueError("declared RAM kill ceiling cannot fit measured reference host")
    request = {"schema": "mdb-flux-context-carriers-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "backend": "diffusers", "device": "cuda",
        "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512, "needs": {"cuda": True},
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=11,
            logical_branches=11, logical_checkpoint_replays=11, max_local_work_items=12,
            checkpoint_policy="Hydrate observed native Z0/Z2; no encoder. Nine two-step futures; rescue two actual altered Z3 cuts for one step each.560 physical transitions/20 projection reads/18 Acts/15 PNGs/16 VAE calls. One load, no external replay or retries."),
        "resource_exception": {"ram_mb": ram_mb, "vram_mb": vram_mb, "ram_kill_factor": RAM_KILL_FACTOR,
            "ram_kill_ceiling_mb": kill_ceiling_mb(ram_mb), "reference_host_mb": 66512,
            "calibration_job": "job-1f1587e6983b", "measured_peak_rss_mb": 16059, "measured_peak_vram_mb": 9282,
            "no_encoder_calibration_job": "job-947ba45b47f9", "no_encoder_peak_rss_mb": 3210,
            "reason": "Same qualified resident CUDA BF16 B1 512px4step model/VAE. Reuse8.50MB observed CAS, skip encoder, retain bounded project ports and fewer futures. Preserve measured16GB reservation/17.6GB kill ceiling and13GB VRAM envelope; level1 export qualified on complete prior evidence."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    if not submit:
        return {"run_dir": str(run_dir), "submitted": False, "request_sha256": request["request_sha256"]}
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-context-carriers", payload=run_dir / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4, est_wall_s=240,
        timeout_s=config["max_wall_seconds"], retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 512px seed47 nativeZ2 latent/C carriers.8dosecells+sham+2own-C Z3rescues=11futures. Shared edited prefixes:560steps/18Acts/20projectreads/15PNG/16VAE; noencoder/donor/retries,one load. sigma-noise anchor/C0 are artificial controls; retain all outcomes. <=650steps/600s,gzip1.")
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
