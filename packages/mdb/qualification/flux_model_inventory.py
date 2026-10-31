"""Resolve and hash the fleet's Diffusers checkpoint without loading a model."""
from __future__ import annotations

import hashlib
import json
import os
import runpy
from pathlib import Path


def main():
    from mdb.bench_loading import local_checkpoint_root

    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    if request["config"] != config:
        raise ValueError("inventory configuration differs from request")
    output = Path("flux-inventory-result")
    output.mkdir(exist_ok=False)
    root = local_checkpoint_root(config["model"])
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}:
            continue
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                checksum.update(block)
        files[path.relative_to(root).as_posix()] = {"sha256": checksum.hexdigest(), "bytes": path.stat().st_size}
    file_hashes = {name: value["sha256"] for name, value in files.items()}
    digest = hashlib.sha256(json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    configs = {name: json.loads((root / name).read_text()) for name in
               ["model_index.json", "transformer/config.json", "text_encoder/config.json", "scheduler/scheduler_config.json"]
               if (root / name).is_file()}
    result = {"schema": "mdb-flux-model-inventory-v1", "model": config["model"], "model_path": str(root),
              "checkpoint_sha256": digest, "files": files, "configs": configs, "model_loads": 0,
              "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"]}
    (output / "inventory.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    helper = runpy.run_path("campaign_support.py")
    helper["emit"](output, result)


if __name__ == "__main__":
    main()
