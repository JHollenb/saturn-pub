"""Execute only the hash-bound public packages shipped with this experiment."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"source custody mismatch: {name}")
    request = json.loads((root / "request.json").read_text())
    if request["device"] == "cpu":
        # Public global-RNG snapshots can initialize every visible CUDA device.
        # CPU leases must not create an unadmitted GPU context.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(root / "vendor"))
    os.environ["SPF_LM_MANIFEST_SHA256"] = hashlib.sha256(
        (root / "source-manifest.json").read_bytes()
    ).hexdigest()
    entry = {"algorithmic": "algorithmic.py", "memory_futures": "memory_futures.py"}.get(
        request["stage"], "run.py"
    )
    sys.argv[0] = str(root / entry)
    runpy.run_path(str(root / entry), run_name="__main__")


if __name__ == "__main__":
    main()
