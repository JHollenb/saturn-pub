"""One finite installed-package ordinary-execution analysis worker."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def run(config):
    import torch

    import mdb.analysis_metrics
    import mdb.attention_ranking
    import mdb.native_discovery_learning
    from mdb.native_discovery import file_hash, save_json, verify_run
    from mdb.native_discovery_run import run_pretrained

    # Which code actually runs: sha256 of the imported modules (compare with the source tree).
    imported = {m.__name__: {"path": m.__file__, "sha256": file_hash(m.__file__)} for m in (
        mdb.analysis_metrics, mdb.attention_ranking, mdb.native_discovery_learning)}
    print("MDB_IMPORTED_SOURCE=" + json.dumps(imported, sort_keys=True), flush=True)

    if not os.environ.get("MRUN_JOB_ID"):
        raise RuntimeError("an admitted mrun lease is required")
    started = time.monotonic()
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(config["output_scratch"]) / ("native-discovery-" + os.environ["MRUN_JOB_ID"])
    result = run_pretrained(corpus=Path("corpus.json"), model=config["model_path"], out=root,
        device=config["device"], max_tokens=config["max_tokens"], stages=config["stages"],
        features=config["features"], steps=config["steps"], interval_steps=config["interval_steps"],
        seed=config["seed"], max_wall_seconds=config["max_wall_seconds"],
        **{k: config[k] for k in ("arms", "sealed_position_split", "contrast_nominator") if k in config})
    retrieval = subprocess.run([sys.executable, str(root / "retrieve_weights.py"), config["model_path"]],
                               check=True, capture_output=True, text=True)
    audit = verify_run(root)
    summary = {**result, "job_id": os.environ["MRUN_JOB_ID"], "verification": audit,
        "retrieval_verification": json.loads(retrieval.stdout), "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if config["device"] == "cuda" else 0,
        "config": config, "imported_source": imported,
        "report": json.loads((root / "report.json").read_text())}
    save_json(root.parent / (root.name + "-summary.json"), summary)
    archive = root.parent / (root.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(root, arcname="evidence")
    print("MDB_NATIVE_DISCOVERY_RESULT=" + json.dumps({"summary": summary,
        "artifact": {"path": str(archive), "sha256": file_hash(archive), "bytes": archive.stat().st_size}},
        sort_keys=True), flush=True)


if __name__ == "__main__":
    run(json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text()))
