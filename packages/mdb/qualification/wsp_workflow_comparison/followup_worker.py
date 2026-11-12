"""Installed worker: storage contrast, task calibration, then native search."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import time
from pathlib import Path

from protocol import digest, fixture, plain, save


def filesystem(path):
    path = str(Path(path).resolve())
    rows = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    candidates = [r for r in rows if path == r[1] or path.startswith(r[1].rstrip("/")+"/")]
    row = max(candidates, key=lambda r: len(r[1]))
    return {"mount": row[1], "device": row[0], "filesystem": row[2], "options": row[3]}


def storage(config, root):
    import numpy as np
    import torch
    from worker_common import fingerprint, score

    from mdb.wsp import WeightStudy, verify_weight_study
    from mdb.wsp_weights import ParameterPatch

    model, splits, _ = fixture(config["seed"], "amplification", include_evaluation=False)
    identity = fingerprint(model)
    site = next(n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear))
    original = dict(model.named_parameters())[site+".weight"].detach().clone()
    doses = [.5, 1., 1.5]
    scratch = Path(config["volatile_scratch"]) / os.environ["MRUN_JOB_ID"]
    scratch.mkdir(parents=True, exist_ok=False)
    if filesystem(scratch)["filesystem"] != "tmpfs":
        raise ValueError("volatile comparison requires actual tmpfs")
    profiles = [("hdd", "per_probe"), ("tmpfs", "per_probe"), ("hdd", "shared_study"), ("tmpfs", "shared_study")]
    rows = []
    reference = []
    for gain in doses:
        started = time.perf_counter()
        try:
            with torch.no_grad():
                dict(model.named_parameters())[site+".weight"].copy_(original*gain)
            result = score(model, splits)
        finally:
            with torch.no_grad():
                dict(model.named_parameters())[site+".weight"].copy_(original)
        reference.append({"gain": gain, "result": result, "seconds": time.perf_counter()-started})
    for repetition in range(5):
        for offset in range(4):
            medium, mode = profiles[(offset+repetition) % 4]
            name = "r"+str(repetition)+"-"+medium+"-"+mode
            directory = (root if medium == "hdd" else scratch) / name
            directory.mkdir()
            sync = {"calls": 0, "seconds": 0., "file_calls": 0, "directory_calls": 0}
            actual_fsync = os.fsync
            def fsync(fd):
                start = time.perf_counter()
                try:
                    return actual_fsync(fd)
                finally:
                    sync["calls"] += 1
                    sync["seconds"] += time.perf_counter()-start
                    sync["directory_calls" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_calls"] += 1
            timings, outcomes = {}, []
            def timed(label, callback):
                start = time.perf_counter()
                try:
                    return callback()
                finally:
                    timings[label] = timings.get(label, 0.)+time.perf_counter()-start
            def new_study(path):
                return WeightStudy(path, model=model,
                    binding={"model": identity, "checkpoint": identity, "execution": "CPU FP64",
                        "consumer": "same discovery and preservation MSE", "horizon": "five residual writes"},
                    input_identity={"splits": digest({k: plain(v[0]) for k, v in splits.items()})},
                    source_identity={"seed": config["seed"], "doses": doses})
            def add(study, index, gain):
                patch = ParameterPatch(model, {site+".weight": original*gain}, model_identity=identity)
                study.add_arm("gain"+str(index), role="edited", patch=patch, state_policy="restart",
                    predictor=lambda context: {"predictions": {"gain": gain, "consumer": "unknown until native execution"}})
            started = time.perf_counter()
            os.fsync = fsync
            try:
                studies = []
                if mode == "shared_study":
                    study = timed("construct", lambda: new_study(directory / "study"))
                    studies.append(study)
                    for index, gain in enumerate(doses):
                        timed("add_arm", lambda index=index, gain=gain: add(study, index, gain))
                    timed("seal", study.seal)
                for index, gain in enumerate(doses):
                    if mode == "per_probe":
                        study = timed("construct", lambda index=index: new_study(directory / ("study"+str(index))))
                        studies.append(study)
                        timed("add_arm", lambda: add(study, index, gain))
                        timed("seal", study.seal)
                    def execute(context):
                        return {"observations": score(model, splits), "work": {"physical_forwards": 1, "batch_size": 16},
                            "state": {"policy": "restart", "recurrent_state": False}}
                    timed("execute_and_record", lambda: study.run("gain"+str(index), execute))
                    receipt = study.tx.observation("gain"+str(index))
                    outcomes.append(plain(receipt["result"]["observations"]))
                    if mode == "per_probe":
                        timed("close", study.close)
                        timed("verify", lambda: verify_weight_study(study.tx.root))
                if mode == "shared_study":
                    timed("close", study.close)
                    timed("verify", lambda: verify_weight_study(study.tx.root))
            finally:
                os.fsync = actual_fsync
                for study in locals().get("studies", []):
                    if study.tx.phase != "closed":
                        study.close()
            elapsed = time.perf_counter()-started
            if fingerprint(model) != identity or outcomes != [r["result"] for r in reference]:
                raise ValueError("storage execution/restoration differs from common reference")
            native_files = sorted(p for p in directory.rglob("*") if p.is_file())
            row = {"repetition": repetition, "warmup": repetition == 0, "medium": medium, "mode": mode,
                "seconds": elapsed, "stages": timings, "fsync": sync, "native_forwards": 3,
                "parameter_restoration": True, "reference_outputs_equal": True,
                "files": len(native_files), "bytes": sum(p.stat().st_size for p in native_files)}
            start = time.perf_counter()
            if medium == "tmpfs":
                shutil.copytree(directory, root / name)
                for p in native_files:
                    copied = root / name / p.relative_to(directory)
                    if hashlib.sha256(p.read_bytes()).digest() != hashlib.sha256(copied.read_bytes()).digest():
                        raise ValueError("tmpfs custody copy differs")
            row["post_measurement_custody_copy_seconds"] = time.perf_counter()-start
            rows.append(row)
            save(root / (name+".json"), row)
            print("FOLLOWUP_PROGRESS storage "+name+" seconds="+str(round(elapsed, 4)), flush=True)
    summary = {}
    for medium, mode in profiles:
        selected = [r for r in rows if not r["warmup"] and (r["medium"], r["mode"]) == (medium, mode)]
        summary[medium+"-"+mode] = {"median_seconds": float(np.median([r["seconds"] for r in selected])),
            "median_fsync_seconds": float(np.median([r["fsync"]["seconds"] for r in selected])),
            "fsync_calls": [r["fsync"]["calls"] for r in selected]}
    result = {"rows": rows, "summary": summary, "plain_reference": reference,
        "filesystems": {"hdd": filesystem(root), "tmpfs": filesystem(scratch)},
        "scope": "same public API/fsync calls and native outputs; tmpfs is volatile, copied after timing; shared study predeclares candidates and changes transaction grouping",
        "work": {"native_forwards": 63, "oracle_generation_forwards": 2}, "terminal_status": "not-evaluated"}
    save(root / "storage.json", result)
    return {"profiles": summary, "native_forwards": 63}


def calibrate(config, root):
    import numpy as np
    from native_workflow import arithmetic_cases
    from qwen_profile import Consumer, load

    net, tokenizer = load(config, root)
    rows = []
    for style in ("last_digit", "compact_modulo", "small_sum"):
        examples = arithmetic_cases(config["seed"], style)
        model = Consumer(net, tokenizer, config["seed"], examples=examples)
        logp = model.anchor_logp.cpu().numpy()
        targets = model.target_ids[:8].cpu().numpy()
        digits = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
        predicted = logp[:8].argmax(-1)
        restricted = np.asarray(digits)[logp[:8, digits].argmax(-1)]
        np.savez_compressed(root / (style+".npz"), logp=logp, target_ids=model.target_ids.cpu().numpy(),
            input_ids=model.input_ids.cpu().numpy(), attention_mask=model.attention_mask.cpu().numpy())
        rows.append({"style": style, "examples": examples, "correct": int(np.count_nonzero(predicted == targets)),
            "digit_restricted_correct": int(np.count_nonzero(restricted == targets)),
            "predictions": [tokenizer.decode([int(t)]) for t in predicted],
            "target_nll": (-logp[np.arange(8), targets]).tolist(), "input_identity": model.input_identity,
            "baseline_task_nll": float(-logp[np.arange(8), targets].mean())})
        print("FOLLOWUP_PROGRESS calibrate "+style+" correct="+str(rows[-1]["correct"]), flush=True)
    save(root / "calibration.json", rows)
    return {"profiles": rows, "physical_network_forwards": 3,
        "scope": "task-instrument discovery; same seed, no edit selection or held-out release"}


def run():
    import torch
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    root = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "followup-result"
    root.mkdir(parents=True, exist_ok=False)
    save(root / "request.json", request)
    save(root / "runtime.json", {"python": platform.python_version(),
        "packages": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "transformers", "mdb-runtime", "saturn-pub")},
        "device": config["device"], "dtype": config["dtype"], "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "matmul_precision": torch.get_float32_matmul_precision()})
    started = time.perf_counter()
    if config["phase"] == "recover":
        source = Path(config["recovery_root"])
        original_request = json.loads((source / "request.json").read_text())
        if original_request["request_sha256"] != config["recovery_request_sha256"]:
            raise ValueError("recovery source request differs")
        inventory = json.loads((source / "files.json").read_text())
        for n, h in inventory.items():
            path = (source / n).resolve()
            if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != h:
                raise ValueError("recovery source file differs")
        for path in source.rglob("*"):
            if path.is_file():
                relative = path.relative_to(source)
                destination = root / (Path("recovery") / relative if str(relative) in {"files.json", "summary.json", "request.json"} else relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        result = {"scope": "saved-only recovery after packaging failure; original failed job remains failed",
            "source_job": config["recovery_job"], "source_request_sha256": config["recovery_request_sha256"],
            "source_inventory_sha256": hashlib.sha256((source / "files.json").read_bytes()).hexdigest(),
            "verified_source_files": len(inventory), "target_model_loads": 0, "target_model_forwards": 0}
    elif config["phase"] == "audit":
        from native_audit import analyze
        result = analyze(config, root)
    elif config["phase"] in {"storage", "calibrate"}:
        result = {"storage": storage, "calibrate": calibrate}[config["phase"]](config, root)
    else:
        from native_workflow import run_phase
        result = run_phase(config, root)
    summary = {"schema": "mdb-workflow-followup-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "phase": config["phase"], "result": result,
        "elapsed_seconds": time.perf_counter()-started, "mechanics_status": "valid",
        "instrument_status": "observed", "trend_status": "uninterpreted", "terminal_status": "not-evaluated"}
    save(root / "summary.json", summary)
    save(root / "files.json", {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()})
    from flux_family_custody import emit
    emit(root, summary, limits=config.get("custody_limits", config.get("limits")), compression_level=1)


if __name__ == "__main__":
    run()
