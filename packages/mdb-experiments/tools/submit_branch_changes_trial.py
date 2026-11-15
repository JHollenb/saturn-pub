"""Build public wheels and submit the bounded trial through mrun.

Invoke through common exec so the scheduler credentials and client are configured.
No source-path injection or remote checkout mutation is used.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from mrun.client.api import Api
from mrun.client.submit import submit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    payload = Path(tempfile.mkdtemp(prefix="mdb-branch-changes-"))
    wheels = payload / "wheels"
    for source in (root, package):
        subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheels), str(source)], check=True)
    for name in ("branch_changes_trial.py", "retrieve_weights.py"):
        shutil.copy2(package / "tools" / name, payload / name)
    sources = {str(p.relative_to(payload)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in payload.rglob("*") if p.is_file()}
    (payload / "sources.json").write_text(json.dumps(sources, indent=2, sort_keys=True))
    wheel_args = [part for p in sorted(wheels.glob("*.whl"))
                  for part in ("--with", str(p.relative_to(payload)))]
    cmd = ["env", "UV_CACHE_DIR=.uv-cache", "uv", "run", "--project", "{env}", "--no-sync", *wheel_args,
           "python", "branch_changes_trial.py", "--model", "/mnt/big/llm-models/Qwen2.5-1.5B"]
    api = Api()
    hosts = api.json("GET", "/api/hosts")
    print("active jobs", [(h.get("name", h.get("host")), h.get("active_jobs")) for h in hosts])
    job = submit(experiment="mdb-branch-changes", cmd=cmd, payload=payload,
        model="qwen2.5-1.5b", backend="hf", dtype="bf16", device="cuda",
        host="beast", needs={"cuda": True}, timeout_s=600, retry_on_kill=False,
        config={"retry_on_kill": False, "max_seq_len": 22,
                "backend_options": {"physical_batch_size": 1, "full_prefill_logits": True}},
        preflight=True, note="One resident BF16/eager load; four same-parent K/V arms, two explained blocks per arm, one selected neuron restore and one fresh-order conditional forecast. Scalar replay preserves port contract. Stop after finite panel; 600 s timeout; no retries.")
    record = {"job_id": job, "payload": str(payload), "sources": sources, "command": cmd}
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
