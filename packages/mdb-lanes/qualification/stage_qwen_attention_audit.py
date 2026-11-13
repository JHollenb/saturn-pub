"""Reproduce the frozen saved-only attention audit or signed-vector supplement."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROLLER = "stage_saved_flux_mask_pair_audit.py"
CONTROLLER_SHA256 = "e15e228136093a1af4c5cfa0f7e262b1a031bd607a035731a94e8678fa0ca122"
READERS = {
    False: {
        "worker": "audit_qwen_block0_attention.py",
        "sha256": "5ab0ffc52537384341158f41edaa44d546232b7d4030d1c35a8949db81d11f8b",
        "helpers": [], "operations": 5, "experiment": "mdb-qwen-block0-saved-audit",
        "scope": "Seven-call block0 source and fixed-bank triangle full arrays,actual operands/profiler/tails/historical controls;zero new recurrence.",
    },
    True: {
        "worker": "audit_qwen_attention_bridge.py",
        "sha256": "bfd6349675a8056cbb8554c699094b67b1d51a6698515a7c58380a8c76965021",
        "helpers": ["audit_qwen_block0_attention.py"], "operations": 6,
        "experiment": "mdb-qwen-attention-signed-bridge",
        "scope": "Exact shared-prefix A/short,B/long,C/long and full signedtriangle-to-historical bridges;selected sixports only,zero model/attention/recipe recurrence.",
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(run, *, study=None, bridge=False, submit=False):
    controller = HERE / CONTROLLER
    reader = READERS[bridge]
    worker = HERE / reader["worker"]
    if sha(controller) != CONTROLLER_SHA256 or sha(worker) != reader["sha256"]:
        raise ValueError("Qualified controller or saved reader changed")
    for name in reader["helpers"]:
        if sha(HERE / name) != READERS[False]["sha256"]:
            raise ValueError("Frozen saved-array helper changed")
    utilities = runpy.run_path(str(controller))
    if submit:
        request = json.loads((Path(run) / "request.json").read_text())
        if request["source_hashes"].get(worker.name) != reader["sha256"]:
            raise ValueError("Staged reader selection differs")
        return utilities["submit"](run, worker=worker)
    if study is None:
        raise ValueError("Original completed study is required to stage")
    return utilities["stage"](run, study, worker=worker, helpers=reader["helpers"],
                              operations=reader["operations"], experiment=reader["experiment"], scope=reader["scope"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--bridge", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and args.study is None:
        parser.error("--study is required to stage")
    print(json.dumps(execute(args.run, study=args.study, bridge=args.bridge, submit=args.submit), sort_keys=True))
