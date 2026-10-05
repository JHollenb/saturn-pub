"""Continue the retained, completed memory panel through source-deleted recovery."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

from family_helpers import load, signature
from mdb.assay_custody import remove_source_store
from mdb.bench import BenchSession
from mdb.bench_verify import verify


def main():
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    cache = request["cache_materialization"]
    if sha256(Path(cache["path"]).read_bytes()).hexdigest() != cache["sha256"]:
        raise RuntimeError("cache inventory changed")
    parent = Path("/mnt/hdd8tb/mdb/qualification") / config["parent_job"]
    panel_bytes = (parent / "panel-before-export.json").read_bytes()
    if sha256(panel_bytes).hexdigest() != config["panel_sha256"]:
        raise RuntimeError("retained memory panel changed")
    panel = json.loads(panel_bytes)
    if panel["identity"] != config["expected_identity"]:
        raise RuntimeError("retained memory specimen changed")
    import re
    if not re.fullmatch(r"job-[0-9a-f]{12}", config["parent_job"]):
        raise RuntimeError("invalid parent job")
    job = os.environ["MRUN_JOB_ID"]
    if not re.fullmatch(r"job-[0-9a-f]{12}", job):
        raise RuntimeError("invalid job")
    root = parent.parent / job
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    before = verify(parent / "export")
    shutil.copytree(parent / "export", root / "export")
    if verify(root / "export") != before:
        raise RuntimeError("copied export differs")
    # This store belongs solely to the failed, terminal parent assay. The export
    # was sealed and independently checked before removing its original source.
    removal = remove_source_store(parent / "session", parent)
    spec, model, tokenizer, weights, files = load(config)
    with BenchSession.resume(model, tokenizer, root=root / "export", model_sha256=weights) as b:
        if b.identity != panel["identity"]:
            raise RuntimeError("resident identity changed")
        exact = signature(b.generate(len(config["expected"]))) == [tuple(r) for r in config["expected"]]
    closed = verify(root / "export")
    mechanics = {"native": panel["native_exact"], "sharing": panel["shared"],
        "parent_unchanged": panel["parent_unchanged"],
        "private_replay": all(r["exact"] for r in panel["private_replays"].values()),
        "eviction": panel["panel_evictions"] > 0, "hydration": panel["panel_hydrations"] > 0,
        "fresh_after_source_removal": exact and removal["removed"]}
    report = {"schema": "mdb-mamba-memory-completion-v1", "parent_job": config["parent_job"],
        "parent_panel_sha256": config["panel_sha256"], "parent_proof": config["parent_proof"],
        "identity": panel["identity"], "weight_files": files, "source_removal": removal,
        "pressure": panel["pressure"], "panel_evictions": panel["panel_evictions"],
        "panel_hydrations": panel["panel_hydrations"], "private_replays": panel["private_replays"],
        "fresh_exact": exact, "fresh_pid": os.getpid(), "closed_verification": closed,
        "mechanics": mechanics, "mechanics_status": "valid" if all(mechanics.values()) else "invalid",
        "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"],
        "elapsed_seconds": time.monotonic()-started,
        "boundary": "completion of retained memory panel; prior failed jobs remain immutable"}
    (root / "report.json").write_text(json.dumps(report, indent=2))
    (root / "parent-panel.json").write_bytes(panel_bytes)
    with tarfile.open(root / "bench-artifacts.tar.gz", "w:gz") as archive:
        for path in root.iterdir():
            if path.name != "bench-artifacts.tar.gz": archive.add(path, arcname=path.name)
    print("MDB_MEMORY_REPORT="+json.dumps(report), flush=True)
    if report["mechanics_status"] != "valid": raise RuntimeError("incomplete memory mechanics")


if __name__ == "__main__": main()
