"""Stage a source-pinned, saved-only H18 history CPU audit near raw evidence.

The frozen controller keeps CPU2/RAM3GiB,240s estimate and600s outer timeout.
The E18 predecessor measured245.61s: the historical estimate is slightly low,
not a runtime guarantee. No new neural/tokenizer/parameter recurrence is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROLLER = "stage_saved_flux_mask_pair_audit.py"
CONTROLLER_SHA256 = "e15e228136093a1af4c5cfa0f7e262b1a031bd607a035731a94e8678fa0ca122"
WORKER = "audit_flux_h18_state_history.py"
WORKER_SHA256 = "f76d0e89d1e355ad6b840e623d82480f8b6bc5c78cb6dccd71bfa7bc4ea34437"
SCOPE = ("OwnH18history/direct-vs-coherent E,4genuine FLUX Acts/224advances/8actualcontext operands; "
         "fullsavedstate,signedresponses,sixoriginal decoder images;zero neural/tokenizer recurrence.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(run, *, study=None, submit=False):
    controller, worker = HERE / CONTROLLER, HERE / WORKER
    if sha(controller) != CONTROLLER_SHA256 or sha(worker) != WORKER_SHA256:
        raise ValueError("Qualified saved controller/reader changed")
    account = runpy.run_path(str(worker))  # Definitions only; no producer import.
    if len(account["HELPERS"]) != 7 or set(account["HELPERS"]) != set(account["PINS"]):
        raise ValueError("Saved-only helper closure differs")
    for name, digest in account["PINS"].items():
        if sha(HERE / name) != digest:
            raise ValueError("Qualified saved account helper changed: " + name)
    utilities = runpy.run_path(str(controller))
    if submit:
        request = json.loads((Path(run) / "request.json").read_text())
        expected = {WORKER: WORKER_SHA256, **account["PINS"], Path(__file__).name: sha(__file__)}
        if any(request["source_hashes"].get(name) != digest for name, digest in expected.items()):
            raise ValueError("Staged saved reader/frontend/helper differs")
        resource = request["resource_exception"]
        if any(resource.get(key) != value for key, value in dict(cpu_threads=2, ram_mb=3072, vram_mb=0, est_wall_s=240).items()):
            raise ValueError("Frozen CPU resource estimate differs")
        return utilities["submit"](run, worker=worker)
    if study is None:
        raise ValueError("Completed original H18 study required")
    # Include this thin frontend as reproducible source, in addition to exactly
    # seven saved-only numerical/receipt helpers and the generic controller.
    return utilities["stage"](run, study, worker=worker,
        helpers=[*account["HELPERS"], Path(__file__).name], experiment="mdb-flux-h18-history-saved-audit",
        operations=6, scope=SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and args.study is None:
        parser.error("--study is required to stage")
    print(json.dumps(execute(args.run, study=args.study, submit=args.submit), sort_keys=True))


if __name__ == "__main__":
    main()
