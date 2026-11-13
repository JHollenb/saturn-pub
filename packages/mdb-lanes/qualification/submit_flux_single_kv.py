"""Submit a finite first-single-block text-key/value route assay."""
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
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    here = Path(__file__).resolve().parent
    h = runpy.run_path(str(here / "summarize_context_window.py"))
    prior = h["source_run"](predecessor, "flux-rescue-routes-result", max_archive_bytes=512 << 20,
                              max_member_bytes=128 << 20)
    data, old = prior["data"], prior["plan"]
    audit = json.loads((predecessor / "analysis-v1.json").read_text())
    if (prior["source"]["job_id"] != "job-b0da807f5d60"
            or prior["source"]["archive_sha256"] != "fc47bc6de4709de88812d71e301209f5ddaf20ac7af1a174019af7385a265208"
            or audit["source"] != prior["source"] or audit["mechanics_status"] != "valid"
            or audit["instrument_status"] != "native-replay-qualified"):
        raise ValueError("source is not the audited image/text handoff occurrence")
    rows = {name: json.loads((data / "rows" / (row + ".json")).read_text())
            for name, row in (("zero", "zero"), ("text", "late.text"))}
    cuts = {"zero": rows["zero"]["parent"], "text": rows["text"]["acts"][-1]["post_act"]}
    inventory = json.loads((data / "files.json").read_text())
    relatives, fields = set(), {}
    for name, cut_id in cuts.items():
        relative = f"typed-store/cuts/{cut_id}.json"
        cut = json.loads((data / relative).read_text())
        point = cut["execution_point"]
        if (cut["fingerprint"] != cut_id or point["phase"] != "single" or point["logical_step"] != 3
                or point["index"] != 0 or point["local_clock"] != 0):
            raise ValueError("source cut is not final-step single0 entry")
        fields[name] = cut["slots"]
        relatives.add(relative)
        relatives.update("typed-store/pages/" + page + ".json" for page in cut["slots"].values())
        relatives.update("typed-store/blobs/" + blob + ".safetensors" for blob in cut["physical_blobs"])
    if (set(fields["zero"]) != set(fields["text"]) or fields["zero"]["text"] == fields["text"]["text"]
            or any(fields["zero"][key] != fields["text"][key] for key in fields["zero"] if key != "text")):
        raise ValueError("single0 pair must differ only in working text")
    extras, input_hashes = [], {}
    for relative in sorted(relatives):
        path = data / relative
        if sha(path) != inventory[relative]:
            raise ValueError("source CAS differs from immutable inventory")
        target = relative.replace("typed-store/", "input-store/", 1)
        extras.append((path, target))
        input_hashes[target] = sha(path)
    campaign = runpy.run_path(str(here / "flux_single_kv_campaign.py"))
    config = {key: old[key] for key in ("model", "model_path", "expected_checkpoint_sha256", "prompt", "seed", "height", "width", "steps", "cpu_threads")}
    config.update(schema="mdb-flux-single-kv-plan-v1", input_cuts=cuts, input_store="input-store",
        source_selection="source-selection.json", selector={"logical_step": 3, "phase": "single", "index": 0},
        branches=campaign["BRANCHES"], max_adapter_steps=191, max_acts=10, max_candidate_calls=5,
        max_images=9, max_vae_calls=9, max_wall_seconds=600, max_archive_bytes=512 << 20,
        archive_compression_level=1, expected_pngs={name: row["image"]["sha256"] for name, row in rows.items()},
        hypotheses={
            "H-entry": "Imported zero/text single0 entries differ only in working T, with identical image, latent, stored C0, complete context/modulation/gates/RoPE, positions, clock and schedule. Image Q/K/V and image MLP preactivation/activation match exactly between native source calls.",
            "H-selection": "K-only substitutes only positioned text K; V-only only text V; KV both. Every edited call keeps all recipient Q and all image K/V exact. V-only preserves baseline image-query attention weights mathematically; K-only and KV share image-query weights. K edits change normalization over both image and text banks. No semantic effect ordering is predicted.",
            "H-local-KV": "Freeze structural equality prediction: KV reproduces text-source image attention-dispatch output, native fused attention-plus-parallel-MLP image projection return, and postblock image bytes. Image MLP features are unchanged for K/V/KV edits. Candidate text need not match the text-source native return. Preserve native fused concat/projection and BF16 rounding. Full fused projection/MLP/image arrays are structural equality predictions; sampled attention has a separate numerical forecast.",
            "H-numeric-attention": "After two allowed native zero/text source calls reveal QKV operands, independently calculate FP64 softmax attention for fixed image queries [0,31,32,255,511,512,767,1023], all heads, under K/V/KV selection. Seal actual numeric predicted attention arrays and text-bank masses in a second public transaction BEFORE any edited K/V/KV call or adapter prefix. Predictor reads only source QKV, never edited outputs. Compare full saved samples with native BF16 dispatch; preserve RMS/max/signed errors without a numerical gate. No independent MLP/fused projection/RGB or raw-prompt forecast is claimed.",
            "H-controls": "Native direct zero/text candidates reproduce actual single0 adapter image/text exits. Zero/text references reproduce prior zero/late.text final latent and PNG; text reference is not prior full own-C recovery. Copy sham equals zero. Coherent native text-source postblock I/T equals text reference. KV.image and KV.both install identical image bytes with separately recorded selected T.",
            "H-text-policy": "K.image/V.image/KV.image retain native zero postblock T. KV.both installs KV candidate postblock T. text.text installs native text-source postblock T only. coherent installs native text-source postblock I/T. All hybrids start at native zero post-single0 with stored C0 unchanged, no further conditioning projection.",
            "H-exploratory": "K/V effect directions, ranks, signed interaction, downstream repair or amplification, relative text-carrier influence, and all visible appearance comparisons are exploratory. Weak final effects can reflect later changes; no isolated route is certified absent. Local attention routing/content comparisons do not exhaust text-query, text-residual, text-MLP or later-update routes.",
            "H-scope": "One observed seed47 altered latent, one final-step clock, paired T histories from its own context; no foreign prompt donor, new encoding or new noise. Whole text banks and complete selected returns; no object/head/token address, universal symbol law or independent prompt-to-image prediction. Nine dependent futures; all outcomes retained."
        },
        scope="One final-step single0 I0/T0 and I0/T1 pair from audited prior handoff. Share two native single0 prefixes; nine 21-transition suffixes =191 adapter transitions,5 direct block calculations,10 Acts,9 final PNG/VAE. Separate immediate text-key vs text-value image effects and selected working-text continuation. All rankings/semantics exploratory.",
        terminal_status="unassessed")
    campaign["validate_config"](config)
    run_dir.mkdir(parents=True, exist_ok=False)
    selection = {"schema": "mdb-flux-single-kv-selection-v1", "prior_job_id": prior["source"]["job_id"],
        "archive_sha256": prior["source"]["archive_sha256"], "request_sha256": prior["request"]["request_sha256"],
        "source_record": {"path": "prior-source.json", "sha256": sha(data / "source.json")},
        "prefix_record": {"path": "prior-prefixes.json", "sha256": sha(data / "prefixes.json")},
        "audit_sha256": sha(predecessor / "analysis-v1.json"), "input_cuts": cuts, "files": input_hashes,
        "expected_pngs": config["expected_pngs"], "prior_rows": {name: {"path": f"prior-{name}-row.json", "sha256": sha(data / "rows" / (row["id"] + ".json"))} for name, row in rows.items()},
        "scope": "Known zero and late.text finals are replay references. New K/V/split-return intervention outcomes unseen when hypotheses are sealed."}
    save(run_dir / "source-selection.json", selection)
    save(run_dir / "plan.json", config)
    worker = here / "flux_single_kv_campaign.py"
    extras += [(here / "flux_single_kv.py", "flux_single_kv.py"),
        (here / "discovery_upgrades.py", "campaign_support.py"), (data / "source.json", "prior-source.json"),
        (data / "prefixes.json", "prior-prefixes.json"), (predecessor / "analysis-v1.json", "prior-audit.json"),
        (run_dir / "plan.json", "plan.json"), (run_dir / "source-selection.json", "source-selection.json")]
    extras += [(data / "rows" / (row["id"] + ".json"), f"prior-{name}-row.json") for name, row in rows.items()]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, "flux_single_kv.py", "campaign_support.py", "prior-source.json",
            "prior-prefixes.json", "prior-zero-row.json", "prior-text-row.json", "prior-audit.json", "plan.json",
            "source-selection.json", "input-store", preflight.name],
        "path_token_allowlist": ["flux-single-kv-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    ram_mb, vram_mb = 16000, 13000
    request = {"schema": "mdb-flux-single-kv-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "backend": "diffusers", "device": "cuda",
        "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512, "needs": {"cuda": True},
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=9, logical_branches=9,
            logical_checkpoint_replays=9, max_local_work_items=14,
            checkpoint_policy="Two audited single0 inputs, two native prefixes shared; nine21-step suffixes=191transitions plus5direct single0 calculations,10Acts,9finalPNG. One model/noencoder/noise/retries."),
        "resource_exception": {"ram_mb": ram_mb, "vram_mb": vram_mb, "ram_kill_factor": RAM_KILL_FACTOR,
            "ram_kill_ceiling_mb": kill_ceiling_mb(ram_mb), "calibration_job": "job-b0da807f5d60",
            "measured_peak_rss_mb": 3194.3, "measured_peak_vram_mb": 9698,
            "failed_startup_lower_bound": {"job_id": "job-cfe22f928044", "observed_tree_rss_mb": 7023.8,
                "kill_ceiling_mb": 6758.4, "phase": "loading before seal; no scientific outcome"},
            "reason": "Reuse qualified16GBRAM/13GBVRAM CUDA BF16 B1 Klein/VAE envelope. Prior low sampled resident peaks miss startup transient measured above7GB; automatic4608MB sizing is not sufficient calibration. No encoder/new projection/noise. New five direct single0 calls retain selected image MLP ports and deduplicate arrays; finite9futures in one600s lease. Reservation is headroom, not a speedup claim."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    if not submit:
        return {"run_dir": str(run_dir), "submitted": False, "request_sha256": request["request_sha256"]}
    return submit_staged(run_dir)


def submit_staged(run_dir):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command

    if (run_dir / "submission.json").exists():
        raise ValueError("staged assay already has a scheduler admission")
    request = json.loads((run_dir / "request.json").read_text())
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("staged request digest differs")
    if any(sha(run_dir / "payload" / name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("staged payload bytes differ")
    config = json.loads((run_dir / "plan.json").read_text())
    if config != request["config"]:
        raise ValueError("staged plan differs from request")
    resource = request["resource_exception"]
    job = guarded_mrun_launch(worker_command("flux_single_kv_campaign.py"), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-single-kv", payload=run_dir / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"], cpu_threads=4, est_wall_s=180,
        timeout_s=600, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 512px final single0, sameI/C/time and differentT. K-only/V-only/KV native attention+parallelMLP;9futures,191adaptertransitions+5directblocks/10Acts/9VAE. Numeric8query predictions sealed after2sourcecalls beforeeditedcalls. Native textretention explicit. Noencoder/noise/foreignprompt/retries;600s,gzip1,allmissesretained.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--submit-staged", action="store_true")
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir, args.collect, log_storage="gzip", max_compressed_bytes=512 << 20,
            max_expanded_bytes=1 << 30)
    elif args.submit_staged:
        result = submit_staged(args.run_dir.resolve())
    else:
        if args.predecessor is None:
            parser.error("--predecessor is required for a new submission")
        result = stage(args.run_dir.resolve(), args.predecessor.resolve(), submit=not args.stage_only)
    print(json.dumps(result))
