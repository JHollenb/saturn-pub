"""Resume an authenticated equal-image/different-text pair at single block1."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

PRIOR_JOB = "job-b87315cd3534"
PRIOR_ARCHIVE = "e8e565211c979debd20b91d2fa79293880c9fd0d2147be407b1141a82eb32de3"
PRIOR_AUDIT = "7d2466af001e68dee2cf89fe3976221b168080fbaa041f99df785249a99c8d99"
CONTEXT = "p1-s53"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def stage(run_dir, predecessor, *, submit=True):
    from mrun.protocol import RAM_KILL_FACTOR, kill_ceiling_mb

    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    here = Path(__file__).resolve().parent
    custody = runpy.run_path(str(here / "flux_family_custody.py"))["verify"](predecessor)
    audit_path = predecessor / "analysis-v1.json"
    audit, transport, prior_request = (read(path) for path in (
        audit_path, predecessor / "report.json", predecessor / "request.json"))
    if (custody["job_id"] != PRIOR_JOB or custody["archive_sha256"] != PRIOR_ARCHIVE
            or sha(audit_path) != PRIOR_AUDIT or audit["mechanics_status"] != "valid"
            or audit["contexts"][CONTEXT]["instrument_status"] != "native-replay-qualified"
            or audit["source"]["archive_sha256"] != PRIOR_ARCHIVE
            or audit["source"]["request_sha256"] != custody["request_sha256"]):
        raise ValueError("successor source is not the audited equal-image occurrence")
    data = predecessor / "extracted" / transport["archive_root"]
    case = data / "contexts" / CONTEXT
    source = read(case / "source.json")
    rows = {role: read(case / "rows" / (row_id + ".json"))
        for role, row_id in (("zero", "KV.image"), ("text", "coherent"))}
    cuts = {role: row["acts"][-1]["post_act"] for role, row in rows.items()}
    manifest = read(data / "files.json")
    if sha(data / "files.json") != transport["files_manifest_sha256"]:
        raise ValueError("original producer manifest differs from transport")
    relatives, slots = set(), {}
    for role, fingerprint in cuts.items():
        relative = f"contexts/{CONTEXT}/typed-store/cuts/{fingerprint}.json"
        cut = read(data / relative)
        point = cut["execution_point"]
        if (cut["fingerprint"] != fingerprint or point["logical_step"] != 3 or point["phase"] != "single"
                or point["index"] != 1 or point["local_clock"] != 1):
            raise ValueError("successor source is not pre-single1")
        slots[role] = cut["slots"]
        relatives.add(relative)
        relatives.update(f"contexts/{CONTEXT}/typed-store/pages/{page}.json" for page in cut["slots"].values())
        relatives.update(f"contexts/{CONTEXT}/typed-store/blobs/{blob}.safetensors" for blob in cut["physical_blobs"])
    if (set(slots["zero"]) != set(slots["text"]) or slots["zero"]["text"] == slots["text"]["text"]
            or any(slots["zero"][key] != slots["text"][key] for key in slots["zero"] if key != "text")):
        raise ValueError("successor pair must differ only in working T")
    extras, input_hashes, source_files = [], {}, {}
    def original(path, target):
        relative = path.relative_to(data).as_posix()
        digest = sha(path)
        if manifest.get(relative) != digest:
            raise ValueError("selected source differs from original producer manifest")
        extras.append((path, target))
        source_files[target] = {"prior_path": relative, "sha256": digest, "bytes": path.stat().st_size}
        return {"path": target, "sha256": digest}
    for relative in sorted(relatives):
        target = relative.replace(f"contexts/{CONTEXT}/typed-store/", "input-store/", 1)
        input_hashes[target] = original(data / relative, target)["sha256"]
    selection = {"schema": "mdb-flux-single-successor-selection-v1", "prior_job_id": PRIOR_JOB,
        "archive_sha256": PRIOR_ARCHIVE, "request_sha256": custody["request_sha256"], "context_id": CONTEXT,
        "input_cuts": cuts, "files": input_hashes,
        "source_record": original(case / "source.json", "prior-context-source.json"),
        "prefix_record": original(case / "prefixes.json", "prior-prefixes.json"),
        "prior_rows": {role: original(case / "rows" / (row["id"] + ".json"), f"prior-{role}-row.json") for role, row in rows.items()},
        "expected_pngs": {role: row["image"]["sha256"] for role, row in rows.items()},
        "expected_final_latents": {role: row["final_latent"]["descriptor"] for role, row in rows.items()},
        "source_files": source_files,
        "scope": "PriorKV.image and coherent actual post-Act entries have identical I and all other state exceptT. Prior endpoints are known replay controls; new single1 ports/edits/intermediate trajectories have not been observed by this campaign."}
    for field, path, target in (
        ("audit_record", audit_path, "prior-audit.json"),
        ("prior_request", predecessor / "request.json", "prior-request.json"),
        ("prior_report", predecessor / "report.json", "prior-transport-report.json"),
        ("prior_manifest", data / "files.json", "prior-producer-manifest.json")):
        selection[field] = {"path": target, "sha256": sha(path)}
        extras.append((path, target))
    engine = runpy.run_path(str(here / "flux_single_contexts.py"))
    config = {key: prior_request["config"][key] for key in (
        "model", "model_path", "expected_checkpoint_sha256", "height", "width", "steps", "cpu_threads")}
    config.update(schema="mdb-flux-single-successor-plan-v1", input_cuts=cuts,
        input_store="input-store", source_selection="source-selection.json", single_index=1,
        context={"id": CONTEXT, "prompt_id": "p1", "prompt": source["context"]["prompt"], "seed": 53},
        context_id=CONTEXT, selector={"logical_step": 3, "phase": "single", "index": 1},
        branches=engine["BRANCHES"], capture_reference_trajectory=True,
        attention_query_indices=[0, 31, 32, 255, 511, 512, 767, 1023],
        max_adapter_steps=182, max_acts=10, max_candidate_calls=5, max_images=9, max_vae_calls=9,
        max_wall_seconds=540, archive_compression_level=1,
        expected_pngs=selection["expected_pngs"], expected_final_latents=selection["expected_final_latents"],
        hypotheses={
            "H-first-reader": "Prospective site prediction: image states first differ after native single1. At its input onlyT differs; imageQKV and parallel imageMLP preactivation/activation remain exact. If rounding/cancellation gives equalI, retain the miss and fixed full reference trajectories; report first actual later divergence or none. Never select/reseal a later intervention after seeing this outcome.",
            "H-local-KV": "Selecting native textK (after normalization/RoPE) and textV from high into low reproduces high native image attention dispatch, fused parallel attention+MLP projection and post-single1 image exactly. K/V edits retain recipientQ and imageK/V; imageMLP remains exact. Selected candidateT need not match high nativeT.",
            "H-numeric": "After exactly two allowed native low/high source captures, predict FP64 softmax(QK/sqrtD)V at the same fixed8imagequeries/all24heads128dims underK,V,KV. Seal numericalarrays before editedcalls or adapterprefixes. K/V are novel conditionaltargets;KV is representedhigh consistencycontrol. No fittedparameters, no independentMLP/rawprompt/RGBprediction; QKV-onlytrusteddataflow, nothermeticisolation.",
            "H-controls": "Bothdirectnative candidates equal actualadapterexits. Low/high finalPNG and native latent identities reproduce priorKV.image/coherent. Sham equals low; coherentpost-single1I/T replacement equals high. SourcepaironlyTdiffers at complete typed frontier.",
            "H-text-direction": "Tentative new-consumer prediction: post-single1 text.text ends closer to high/text reference than eachK.image,V.image,KV.image, checked separately in latentL2 andRGBMAE. Retain all signedmargins/misses/disagreement. Onecontext and dependentfutures, no universalmediationfraction.",
            "H-trajectory": "Observe both native referenceI/T trajectories at everyremaining live-slotboundary and retainarrays for independentdistancecurves. Firstdivergence, amplification, attenuation and exactreconvergence remain observations; no wholetrajectorynumericforecast. Selected completeStateCuts preserve replayfrontiers; receipts alone are not independentkernel-read traces.",
            "H-latent-carrier": "Structural prediction: the sameZ3 latent remains byte-exact in both references until schedulercommit, whileI/T evolve. Readout produces native noise_prediction; commit updates latent and clears I/T/context/noise. Retain actual post-readoutnoise via complete public StateCuts. Any Euler residual comparison is an offline diagnostic, not a presealed numerical Z4 forecast.",
            "H-scope": "One authenticated p1seed53 altered history at the nextconsumer single1, no freshpromptreplication, encoder, noise or foreigncontext. Paired source states and priorendpointcontrols are observed; newfirstwrite/local structuralpredictions sealedbeforecalls and numericconditionalpredictions sealedaftertwo allowedsourcecalls. All9dependent nativefutures retained; noobjectsemanticdictionary orautomaticfutureenumeration."},
        scope="Resume actual equalI/differentT at logical3single1 from audited priorcontextp1-s53. One load,0encodes/noise,182adaptersteps+5directblocks/10Acts/192receipts/9VAE. Two presealed stages; native low/high per-boundaryI/T arrays+selected fullcuts; all9futures and countertrends retained.",
        terminal_status="unassessed")
    runpy.run_path(str(here / "flux_single_successor_campaign.py"))["validate_config"](config)
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "source-verification.json", custody)
    save(run_dir / "source-selection.json", selection)
    save(run_dir / "plan.json", config)
    worker = here / "flux_single_successor_campaign.py"
    extras += [(here / name, name) for name in ("flux_single_contexts.py", "flux_single_kv.py", "flux_family_custody.py")]
    extras += [(run_dir / name, name) for name in ("plan.json", "source-selection.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, "flux_single_contexts.py", "flux_single_kv.py", "flux_family_custody.py", "plan.json",
            "source-selection.json", "prior-context-source.json", "prior-prefixes.json", "prior-zero-row.json", "prior-text-row.json",
            "prior-audit.json", "prior-request.json", "prior-transport-report.json", "prior-producer-manifest.json", "input-store", preflight.name],
        "path_token_allowlist": ["flux-single-successor-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-single-successor-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "backend": "diffusers", "device": "cuda",
        "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512, "needs": {"cuda": True},
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=9,
            logical_branches=9, logical_checkpoint_replays=9, max_local_work_items=16,
            checkpoint_policy="Authenticated22.5MB pairedsingle1 CAS; noencoding/noise. Two nativeprefixes then9x20transition suffixes. FullI/Treferencearrays/selectedcuts, two publicseals beforetargetcalls, boundedmultipart transport. One load/no retries."),
        "resource_exception": {"ram_mb": 16000, "vram_mb": 13000, "disk_gb": 8,
            "ram_kill_factor": RAM_KILL_FACTOR, "ram_kill_ceiling_mb": kill_ceiling_mb(16000),
            "calibration_job": PRIOR_JOB, "measured_peak_rss_mb": 8830.2, "measured_peak_vram_mb": 10036,
            "reason": "Use qualified16GBRAM/13GBVRAM noencoder CUDA BF16B1 lane; predecessorfullencoder campaign peaked8830.2/10036MB. Retain per-referenceI/Tboundaryarrays under bounded3GiBexpanded/1.5GiBcompressed transport,8GB transientdisk. Reservation is headroom, not a speedupclaim."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    return submit_staged(run_dir) if submit else {"run_dir": str(run_dir), "submitted": False, "request_sha256": request["request_sha256"]}


def submit_staged(run_dir):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command

    if (run_dir / "submission.json").exists():
        raise ValueError("successor assay already has scheduler admission")
    request = read(run_dir / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("staged request self seal differs")
    if any(sha(run_dir / "payload" / name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("staged payload differs")
    config, resource = read(run_dir / "plan.json"), request["resource_exception"]
    if config != request["config"]:
        raise ValueError("staged plan differs")
    job = guarded_mrun_launch(worker_command("flux_single_successor_campaign.py"), request=request,
        model=config["model"], task="forward", experiment="mdb-flux-single-successor", payload=run_dir / "payload",
        prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"],
        disk_gb=resource["disk_gb"], cpu_threads=4, est_wall_s=360, timeout_s=600, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16B1 512px: actualp1seed53 equalI/differentT pre-single1 fromjob-b87315cd3534. Presealfirstimage-divergence+KVlaw;2sourcecaptures thennumericK/Vsealbeforeeditedcalls.9futures/182adaptersteps/5direct/10Acts/9VAE; full2referenceI/Ttrajectories+selectedcuts. One residentload,0encoder/noise/retries,600s; preserve misses/repair/rankreversals.")
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
        result = runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](args.run_dir.resolve(), args.collect)
    elif args.submit_staged:
        result = submit_staged(args.run_dir.resolve())
    else:
        if args.predecessor is None:
            parser.error("--predecessor required for new stage")
        result = stage(args.run_dir.resolve(), args.predecessor.resolve(), submit=not args.stage_only)
    print(json.dumps(result))
