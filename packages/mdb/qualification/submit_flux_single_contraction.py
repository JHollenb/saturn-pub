"""Replay authenticated single2 exits and dissect the single6 contraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

PRIOR_JOB = "job-b8e38f8a326f"
PRIOR_ARCHIVE = "7ba8da5c2b0093310697760d397868932cad3b8b08fc3fd7cba01eac33c469d0"
PRIOR_AUDIT = "b28a9e223544c4e7646b9cd03ba2c500181848c28d9a71353fda90ef070da7c6"


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


def selected_exit(row, index):
    entries = [entry for entry in row["reference_trajectory"] if entry["point"]["phase"] == "single"
               and entry["point"]["index"] == index and entry["point"]["logical_step"] == 3]
    if len(entries) != 1:
        raise ValueError("source trajectory frontier missing or ambiguous")
    entry = entries[0]
    return {key: entry[key] for key in ("cut", "point", "receipt_id")} | {
        "arrays": {slot: value["descriptor"] for slot, value in entry["arrays"].items()}}


def stage(run_dir, predecessor, *, submit=True):
    from mrun.protocol import RAM_KILL_FACTOR, kill_ceiling_mb

    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    here = Path(__file__).resolve().parent
    custody = runpy.run_path(str(here / "flux_family_custody.py"))["verify"](predecessor)
    audit_path = predecessor / "analysis-v1.json"
    audit, transport, prior_request = [read(path) for path in (
        audit_path, predecessor / "report.json", predecessor / "request.json")]
    if (custody["job_id"] != PRIOR_JOB or custody["archive_sha256"] != PRIOR_ARCHIVE
            or sha(audit_path) != PRIOR_AUDIT or audit["mechanics_status"] != "valid"
            or audit["instrument_status"] != "native-replay-qualified"
            or audit["source"]["archive_sha256"] != PRIOR_ARCHIVE
            or audit["source"]["request_sha256"] != custody["request_sha256"]):
        raise ValueError("contraction source is not the audited successor history")
    data = predecessor / "extracted" / transport["archive_root"]
    manifest = read(data / "files.json")
    if sha(data / "files.json") != transport["files_manifest_sha256"]:
        raise ValueError("original producer manifest differs")
    rows = {role: read(data / "rows" / (role + ".json")) for role in ("zero", "text")}
    inputs = {role: selected_exit(row, 3) for role, row in rows.items()}
    relatives, slots = set(), {}
    for role, frontier in inputs.items():
        cut_path = "typed-store/cuts/" + frontier["cut"] + ".json"
        cut = read(data / cut_path)
        if (cut["fingerprint"] != frontier["cut"] or cut["execution_point"] != frontier["point"]
                or cut["execution_point"]["local_clock"] != 3):
            raise ValueError("source complete cut differs from recorded after-single2 exit")
        slots[role] = cut["slots"]
        relatives.add(cut_path)
        relatives.update("typed-store/pages/" + page + ".json" for page in cut["slots"].values())
        relatives.update("typed-store/blobs/" + blob + ".safetensors" for blob in cut["physical_blobs"])
    if (set(slots["zero"]) != set(slots["text"])
            or any(slots["zero"][key] != slots["text"][key] for key in slots["zero"] if key not in {"image", "text"})
            or any(slots["zero"][key] == slots["text"][key] for key in ("image", "text"))):
        raise ValueError("after-single2 sources must differ in I/T only")
    extras, imported, source_files = [], {}, {}

    def original(path, target):
        relative = path.relative_to(data).as_posix()
        digest = sha(path)
        if manifest.get(relative) != digest:
            raise ValueError("selected original differs from authenticated producer manifest")
        extras.append((path, target))
        source_files[target] = {"prior_path": relative, "sha256": digest, "bytes": path.stat().st_size}
        return {"path": target, "sha256": digest}

    for relative in sorted(relatives):
        target = relative.replace("typed-store/", "input-store/", 1)
        imported[target] = original(data / relative, target)["sha256"]
    selection = {"schema": "mdb-flux-single-contraction-selection-v1", "prior_job_id": PRIOR_JOB,
        "archive_sha256": PRIOR_ARCHIVE, "request_sha256": custody["request_sha256"], "context_id": "p1-s53",
        "input_cuts": {role: value["cut"] for role, value in inputs.items()}, "files": imported,
        "source_record": original(data / "source.json", "prior-source.json"),
        "prior_rows": {role: original(data / "rows" / (role + ".json"), "prior-" + role + "-row.json") for role in rows},
        "expected_frontiers": {role: selected_exit(row, 6) for role, row in rows.items()},
        "expected_outputs": {role: selected_exit(row, 7) for role, row in rows.items()},
        "expected_pngs": {role: row["image"]["sha256"] for role, row in rows.items()},
        "expected_final_latents": {role: row["final_latent"]["descriptor"] for role, row in rows.items()},
        "source_files": source_files,
        "scope": "Two actual retained after-single2 cuts, full-history I/T contrast. Replay after5 and after6 expected descriptors/cut identities and endpoint PNGs are already observed controls. New mixed feature calculations and forecasts are not reference first executions."}
    for field, path, target in (
        ("audit_record", audit_path, "prior-audit.json"),
        ("prior_request", predecessor / "request.json", "prior-request.json"),
        ("prior_report", predecessor / "report.json", "prior-transport-report.json"),
        ("prior_manifest", data / "files.json", "prior-producer-manifest.json")):
        selection[field] = {"path": target, "sha256": sha(path)}
        extras.append((path, target))
    worker_module = runpy.run_path(str(here / "flux_single_contraction_campaign.py"))
    helper = runpy.run_path(str(here / "flux_single_contraction.py"))
    config = {key: prior_request["config"][key] for key in (
        "model", "model_path", "expected_checkpoint_sha256", "height", "width", "steps", "cpu_threads")}
    config.update(schema="mdb-flux-single-contraction-plan-v1", input_cuts=selection["input_cuts"],
        input_store="input-store", source_selection="source-selection.json", single_index=6,
        selector={"logical_step": 3, "phase": "single", "index": 6},
        context_id="p1-s53", branches=helper["BRANCHES"], query_indices=[0, 31, 32, 255, 511, 512, 767, 1023],
        output_dims=list(range(0, 3072, 96)), max_adapter_steps=143, max_acts=10,
        max_candidate_calls=5, max_images=9, max_vae_calls=9, max_wall_seconds=540,
        archive_compression_level=1, expected_pngs=selection["expected_pngs"],
        expected_final_latents=selection["expected_final_latents"], hypotheses={
            "H-replay": "Known controls: replay after-single2 through3/4/5 exactly matches authenticated priorafter5 completecutidentity/liveI/T/Z3 descriptors. Native6prefixes andboth endpoints reproduce previous observed controls. Nofreshcontextclaim.",
            "H-partitions": "Only selected imageA/MLP slices entering nativefusedprojectionchange. EverymixedcandidateT exactlyequals low nativepost6T; textfeatures andimage residual/gates unchanged. AMimageprojection equalshigh nativeimageprojection, but AMpostI neednot equalhigh becauseentryresidualdiffers.",
            "H-numeric": "After exactly two allowed native source captures, predict the FP64 whole fused linear projection and gated residual update at eight fixed image queries and 32 fixed output dimensions. Use only source A/M, selected weight rows/bias and recipient R/gate. Seal before edited A/M/AM calls or native single6 prefixes. A/M projections and all low-residual mixed updates are new conditional targets; AM projection is represented by the observed high projection and is a consistency control. Retain FP64 and explicit BF16 variant errors without fitted coefficients or error gates. No independent MLP-feature or RGB forecast; trusted dataflow, not hermetic isolation.",
            "H-M-over-A": "Tentative unsupportedworkinghypothesis: M.image iscloser than A.image tohigh localpost6I inL2. Retainmissandall magnitudes; finalorderingexploratory, notpredicted.",
            "H-controls": "Sham equalslow andcoherentpost6I/T equals high inlatentidentity/PNG. AM.text andcoherent sharehighpostT/highimagefeaturewrite butdifferineffectiveimage residualwithrounding; noadditivedistanceassumption.",
            "H-contraction": "Known nativeI-distance contraction isreplayed. OfflineFP64effectivewrite Ueff=postI-preI yields exactvectoridentity dI'=dR+dUeff; report2dot+norm² andnativegate/projection/residual-addroundingrho separately. DonotidentifyroundedA/Msliceswithanexactadditiveprogram.",
            "H-scope": "Oneobservedp1seed53history,single6; bothI/Tvary. AttentionversusMLPfeaturewritesencodewholehighhistory, nottext-onlycausalpaths. Nine dependentnativefutures; nosemanticrepair/objectdictionary/rawpromptforecast/automaticfamilyinduction."},
        scope="Single6 parallelfeaturefactorial+residualcontrast fromauthenticated replay;1load0encode/noise,143adaptersteps5directblocks10Acts153receipts9VAE. Newconditionalprojectionsamples sealedaftertwoobservedsourcecalls. NativeRGBcontinuations andcounterexamplesretained.",
        terminal_status="unassessed")
    worker_module["validate_config"](config)
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "source-verification.json", custody)
    save(run_dir / "source-selection.json", selection)
    save(run_dir / "plan.json", config)
    worker = here / "flux_single_contraction_campaign.py"
    extras += [(here / name, name) for name in ("flux_single_contraction.py", "flux_family_custody.py")]
    extras += [(run_dir / name, name) for name in ("plan.json", "source-selection.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, "flux_single_contraction.py", "flux_family_custody.py", "plan.json", "source-selection.json",
            "prior-source.json", "prior-zero-row.json", "prior-text-row.json", "prior-audit.json", "prior-request.json",
            "prior-transport-report.json", "prior-producer-manifest.json", "input-store", preflight.name],
        "path_token_allowlist": ["flux-single-contraction-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-single-contraction-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "backend": "diffusers", "device": "cuda",
        "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512, "needs": {"cuda": True},
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=9, logical_branches=9,
            logical_checkpoint_replays=9, max_local_work_items=16,
            checkpoint_policy="Two imported after2cuts,6source replaytransitions,2native6prefixes,9x15suffixes,5directblocks. Featurefactorial/nativeprojection ports, twoindependentseals, boundedmultipart; one load/no retries."),
        "resource_exception": {"ram_mb": 16000, "vram_mb": 13000, "disk_gb": 8,
            "ram_kill_factor": RAM_KILL_FACTOR, "ram_kill_ceiling_mb": kill_ceiling_mb(16000),
            "calibration_job": PRIOR_JOB, "measured_peak_rss_mb": 6937.6, "measured_peak_vram_mb": 9992,
            "reason": "Samequalified4B CUDA BF16B1 noencoder lane; priorresidentrun measured6937.6MBRSS9992MBVRAM. Fullconcatports retainedunderboundedtransport; nofullweighttensor retained.16GB/13GB headroomnotaspeedupclaim."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    return submit_staged(run_dir) if submit else {"run_dir": str(run_dir), "submitted": False, "request_sha256": request["request_sha256"]}


def submit_staged(run_dir):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command

    if (run_dir / "submission.json").exists():
        raise ValueError("contraction assay already admitted")
    request = read(run_dir / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("request self seal differs")
    if any(sha(run_dir / "payload" / name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("staged payload changed")
    config, resource = read(run_dir / "plan.json"), request["resource_exception"]
    if config != request["config"]:
        raise ValueError("staged plan changed")
    job = guarded_mrun_launch(worker_command("flux_single_contraction_campaign.py"), request=request,
        model=config["model"], task="forward", experiment="mdb-flux-single-contraction", payload=run_dir / "payload",
        prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"],
        disk_gb=resource["disk_gb"], cpu_threads=4, est_wall_s=300, timeout_s=600, retry_on_kill=False, preflight=True,
        note="Klein 4B CUDA BF16, batch 1, 512px: single6 contraction. Replay two authenticated after-single2 cuts for six total transitions and verify prior after-single5 state. Capture two sources, seal 8x32 sampled fused projections, then calculate three A/M/AM edits. Nine native futures: 143 transitions, ten Acts, nine VAE decodes. One resident load, no encoding/noise/retries, 600s stop. Full-history feature writes; retain rounding, known reference controls and prediction misses.")
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
            parser.error("--predecessor is required")
        result = stage(args.run_dir.resolve(), args.predecessor.resolve(), submit=not args.stage_only)
    print(json.dumps(result))
