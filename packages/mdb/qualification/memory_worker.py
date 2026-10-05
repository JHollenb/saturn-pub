"""Finish the Mamba memory panel without repeating the completed parent suite."""
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import torch
from family_helpers import load, signature
from mdb.bench import BenchSession, tensor_digest
from mdb.bench_pages import allocations
from mdb.bench_verify import verify
from mdb.assay_custody import remove_source_store


def produce(config, root):
    spec, model, tokenizer, weights, weight_files = load(config)
    expected = [tuple(row) for row in config["expected"]]
    ids = tokenizer(config["prompt"], return_tensors="pt").input_ids.to("cuda")
    # Use the qualified native slow path, including the recurrent cache flags.
    b = BenchSession.from_resident(model, tokenizer, root=root/"session",
        model_id=spec.model_id, revision=spec.revision, model_sha256=weights,
        session_id=os.environ["MRUN_JOB_ID"], resident_state_bytes=config["state_budget_bytes"])
    if b.identity != config["expected_identity"]:
        raise RuntimeError("memory specimen differs from the qualified parent lane")
    native = b.adapter._slow(None, ids)
    native_rows = []
    for _ in expected:
        token = int(native.logits[:, -1].argmax(-1))
        native = b.adapter._slow(native.cache_params, torch.tensor([[token]], device="cuda"))
        native_rows.append((token, tensor_digest(native.logits[:, -1])))
    if native_rows != expected:
        raise RuntimeError("current native consumer differs from retained parent reference")
    del native
    parent = b.prefill(config["prompt"])
    parent_digest = sha256(json.dumps(b.artifacts.get(parent), sort_keys=True).encode()).hexdigest()
    b.fork("memory-b", cut=parent); b.fork("memory-c", cut=parent)
    shared = any(len(row["owners"]) == 3 for row in b.inspect("memory")["unique_allocations"])
    event_start = len(b.pages.events)
    private, pressure = {}, {}
    for branch in ("main", "memory-b", "memory-c"):
        b.pages.budget = config["state_budget_bytes"]
        b.checkout(branch)
        first = b.step("token")
        if signature([first]) != expected[:1]:
            raise RuntimeError("private first token differs")
        private[branch] = b.checkpoint(branch+"-private")
        active = allocations(b.state)
        union = allocations(list(b.pages.pool.values())); union.update(active)
        budget = sum(active.values())
        if sum(union.values()) <= budget:
            raise RuntimeError("memory instrument has no inactive resident pressure")
        b.pages.budget = budget
        b.pages.admit(b.state)
        inspection = b.inspect("memory")
        if inspection["resident_bytes"] > budget:
            raise RuntimeError("eviction did not honor the measured budget")
        pressure[branch] = {"active_bytes": budget, "union_before_bytes": sum(union.values()),
                           "resident_after_bytes": inspection["resident_bytes"]}
    replays = {}
    for branch, cut in private.items():
        b.pages.budget = config["state_budget_bytes"]
        b.checkout(branch); b.restore(cut)
        replays[branch] = {"cut": cut, "exact": signature(b.generate(len(expected)-1)) == expected[1:]}
    b.pages.budget = config["state_budget_bytes"]
    b.checkout("main"); b.restore(parent)
    parent_unchanged = parent_digest == sha256(json.dumps(b.artifacts.get(parent), sort_keys=True).encode()).hexdigest()
    panel_events = b.pages.events[event_start:]
    memory = b.inspect("memory")
    report = {"identity": b.identity, "weight_files": weight_files, "native_exact": True,
        "shared": shared, "parent_unchanged": parent_unchanged, "private_replays": replays,
        "pressure": pressure, "panel_evictions": sum(e["kind"] == "evict" for e in panel_events),
        "panel_hydrations": sum(e["kind"] == "hydrate" for e in panel_events), "memory": memory}
    (root/"panel-before-export.json").write_text(json.dumps(report, indent=2))
    b.export(root/"export"); b.close()
    report["source_removal"] = remove_source_store(root/"session", root)
    return report


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--resume":
        config = json.loads(Path(sys.argv[2]).read_text())
        closure = verify(config["export"])
        spec, model, tokenizer, weights, _ = load(config)
        with BenchSession.resume(model, tokenizer, root=config["export"], model_sha256=weights) as b:
            exact = signature(b.generate(len(config["expected"]))) == [tuple(r) for r in config["expected"]]
            identity = b.identity
        print("MDB_MEMORY_FRESH="+json.dumps({"exact": exact, "identity": identity,
            "verification": closure, "closed_verification": verify(config["export"]), "pid": os.getpid()}), flush=True)
        return
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    cache = request["cache_materialization"]
    cache_bytes = Path(cache["path"]).read_bytes()
    if sha256(cache_bytes).hexdigest() != cache["sha256"] or json.loads(cache_bytes)["missing"]:
        raise RuntimeError("runtime cache materialization differs from the admitted request")
    import re
    if not re.fullmatch(r"job-[0-9a-f]{12}", os.environ["MRUN_JOB_ID"]):
        raise RuntimeError("invalid job identity")
    root = Path("/mnt/hdd8tb/mdb/qualification")/os.environ["MRUN_JOB_ID"]
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = produce(config, root)
    gc.collect(); torch.cuda.empty_cache()
    resume = root/"resume.json"
    resume.write_text(json.dumps({**config, "export": str(root/"export")}))
    child = subprocess.run([sys.executable, __file__, "--resume", str(resume)],
        capture_output=True, text=True, timeout=600)
    (root/"fresh.stdout").write_text(child.stdout); (root/"fresh.stderr").write_text(child.stderr)
    if child.returncode: raise RuntimeError(child.stderr[-3000:])
    report["fresh"] = next(json.loads(l.split("=", 1)[1]) for l in child.stdout.splitlines() if l.startswith("MDB_MEMORY_FRESH="))
    mechanics = {"native": report["native_exact"], "sharing": report["shared"],
        "parent_unchanged": report["parent_unchanged"],
        "private_replay": all(r["exact"] for r in report["private_replays"].values()),
        "eviction": report["panel_evictions"] > 0, "hydration": report["panel_hydrations"] > 0,
        "fresh_after_source_removal": report["fresh"]["exact"] and not (root/"session").exists()}
    report.update(schema="mdb-mamba-memory-proof-v1", parent_proof=config["parent_proof"],
        mechanics=mechanics, mechanics_status="valid" if all(mechanics.values()) else "invalid",
        elapsed_seconds=time.monotonic()-start, request_sha256=request["request_sha256"],
        source_hashes=request["source_hashes"], boundary="focused memory completion; prior three-case execution proof remains independently sealed")
    (root/"report.json").write_text(json.dumps(report, indent=2))
    with tarfile.open(root/"bench-artifacts.tar.gz", "w:gz") as archive:
        for path in root.iterdir():
            if path.name != "bench-artifacts.tar.gz": archive.add(path, arcname=path.name)
    print("MDB_MEMORY_REPORT="+json.dumps(report), flush=True)
    if report["mechanics_status"] != "valid": raise RuntimeError("memory mechanics incomplete")


if __name__ == "__main__": main()
