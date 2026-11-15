"""Finite mrun wrapper for the synthetic, model-free E18 custody benchmark."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

BENCHMARK_SHA256 = "3e8b81894cde296cf5ff15509ba68c6c35ef60020b640b1a733d201d1005fd66"
FINALIZER_SHA256 = "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config_path):
    started = time.monotonic()
    faulthandler.dump_traceback_later(90, repeat=True)
    config = json.loads(Path(config_path).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if digest != request["request_sha256"] or request["config"] != config:
        raise ValueError("Exact source-bound calibration request/config required")
    for name, expected in request["source_hashes"].items():
        if sha(name) != expected:
            raise ValueError("Executed calibration source differs: " + name)
    base = Path(__file__).resolve().parent
    benchmark, finalizer = base / "benchmark_flux_e18_custody.py", base / "flux_image_tail_microscope.py"
    if sha(benchmark) != BENCHMARK_SHA256 or sha(finalizer) != FINALIZER_SHA256:
        raise ValueError("Qualified benchmark/finalizer changed")
    job_id = os.environ["MRUN_JOB_ID"]
    output = Path(config["output_scratch"]) / job_id / config["output_name"]
    print("MDB_E18_CUSTODY_BOOTSTRAP " + json.dumps({"job_id": job_id, "synthetic": True}), flush=True)
    report = runpy.run_path(str(benchmark))["run"]({key: config[key] for key in ("device", "profile", "repeats", "cpu_threads")} | {"output": str(output)})
    summary = {"schema": "mdb-e18-custody-calibration-summary-v1", "job_id": job_id,
               "request_sha256": request["request_sha256"], "benchmark_source_sha256": BENCHMARK_SHA256,
               "report_sha256": sha(output / "report.json"), "synthetic": True, "device": report["device"],
               "complete_cuts": report["complete_cuts"], "timed_steps_per_mode": report["timed_steps_per_mode"],
               "exact_metadata_and_store_equivalence": report["exact_metadata_and_store_equivalence"],
               "build_seconds": report["build_seconds"],
               "timings": {name: {key: row[key] for key in ("metadata_seconds", "store_seconds", "store_calls", "source_rule_payload_reads", "bytes")}
                           for name, row in report["observations"].items()},
               "neural_forwards": 0, "Session_receipts": 0, "worker_seconds": time.monotonic() - started,
               "limits": report["limits"]}
    (output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n")
    ready = runpy.run_path(str(finalizer))["finalize"](output, job_id=job_id,
        request_sha256=request["request_sha256"], **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    run(parser.parse_args().config)
