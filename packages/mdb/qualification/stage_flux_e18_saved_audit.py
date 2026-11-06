"""Stage one full saved-only E18 Act/continuation audit near its raw data."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROLLER = "stage_saved_flux_mask_pair_audit.py"
CONTROLLER_SHA256 = "afbb71dfe56b2ed1749c77469dbde8a6c85f554b38831d52f54dd30a647f2ea0"
WORKER = "audit_flux_e18_conditioning_act.py"
WORKER_SHA256 = "b9690429f05257bac7f6a9bd276e37e411d4fd7960cd33caa893faf5d87ec70c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(run, *, study=None, submit=False):
    controller, worker = HERE / CONTROLLER, HERE / WORKER
    if sha(controller) != CONTROLLER_SHA256 or sha(worker) != WORKER_SHA256:
        raise ValueError("Qualified saved controller/reader changed")
    account = runpy.run_path(str(worker))
    for name, digest in account["PINS"].items():
        if sha(HERE / name) != digest:
            raise ValueError("Qualified saved account helper changed: " + name)
    utilities = runpy.run_path(str(controller))
    if submit:
        request = json.loads((Path(run) / "request.json").read_text())
        if request["source_hashes"].get(WORKER) != WORKER_SHA256:
            raise ValueError("Staged saved reader differs")
        return utilities["submit"](run, worker=worker)
    if study is None:
        raise ValueError("Completed original E18 study required")
    return utilities["stage"](run, study, worker=worker, helpers=list(account["HELPERS"]),
        experiment="mdb-flux-e18-saved-audit", operations=6,
        scope="OwnQwen,E18patchCAS+4Acts/224advances/8operands,complete branch states,signed responses and six actual decoder images;zero neural/tokenizer calls.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and args.study is None:
        parser.error("--study is required to stage")
    print(json.dumps(execute(args.run, study=args.study, submit=args.submit), sort_keys=True))
