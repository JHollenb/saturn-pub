#!/usr/bin/env python3
"""Generic model-virtual-memory page-op worker for `saturn-debug vm ...`.

Reads the VM op request from ``SATURN_RUN_CONFIG`` (keys: ``op``, ``model``,
``object``, ``dtype``, and op params), builds the Mamba state VM, runs the one
page operation through the Model OS kernel, and emits one gzip+b64 marker with
the evidence receipt.  All model execution is on the mrun lease; the VM mechanics
are in :mod:`saturn.mamba_state_vm`.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from mdb.mvm_worker_support import run_mamba_op

MARKER = "SATURN_MVM_OP_GZIP_B64="


def main() -> int:
    started = time.time()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    report: dict[str, Any] = {"schema": "saturn-mvm-op-worker-v1", "status": "ok", "errors": []}
    try:
        with torch.inference_mode():
            report.update(run_mamba_op(config))
        print(json.dumps({"op": report.get("op"), "ok": True}), flush=True)
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["errors"].append(str(exc))
        report["traceback"] = traceback.format_exc()
        print(json.dumps({"op": config.get("op"), "error": str(exc)}), flush=True)
    report["wall_s_total"] = round(time.time() - started, 2)
    try:
        if torch.cuda.is_available():
            report["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except Exception:
        pass
    print(MARKER + base64.b64encode(gzip.compress(json.dumps(report).encode())).decode(), flush=True)
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
