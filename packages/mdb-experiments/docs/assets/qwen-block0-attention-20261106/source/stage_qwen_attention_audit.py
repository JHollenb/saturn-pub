"""Reproduce the frozen saved-only attention audit or signed-vector supplement."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROLLER = "stage_saved_flux_mask_pair_audit.py"
CONTROLLER_SHA256 = "afbb71dfe56b2ed1749c77469dbde8a6c85f554b38831d52f54dd30a647f2ea0"
READERS = {
    False: {
        "worker": "audit_qwen_block0_attention.py",
        "sha256": "00de6d42e167351b95667b07723e04320f798265b0155e101008990e114abc20",
        "helpers": [], "operations": 5, "experiment": "mdb-qwen-block0-saved-audit",
        "scope": "Seven-call block0 source and fixed-bank triangle full arrays,actual operands/profiler/tails/historical controls;zero new recurrence.",
    },
    True: {
        "worker": "audit_qwen_attention_bridge.py",
        "sha256": "e4b7b24d54c2ffcd574cb1fc41e39e4dfa1d9ff8db73005567ce15a3fc4dd3dc",
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
