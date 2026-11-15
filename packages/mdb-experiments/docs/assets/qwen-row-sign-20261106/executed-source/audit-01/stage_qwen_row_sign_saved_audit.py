"""Source-pinned saved-only row-sign CPU audit frontend.

The unchanged generic controller uses CPU2/RAM3GiB, 240s estimated and600s
outer timeout, one lease and no retries. No model or learned operation runs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROLLER = "stage_saved_flux_mask_pair_audit.py"
CONTROLLER_SHA256 = "afbb71dfe56b2ed1749c77469dbde8a6c85f554b38831d52f54dd30a647f2ea0"
WORKER = "audit_qwen_row_sign.py"
WORKER_SHA256 = "218bf64a05739a4af547b8e26c3ccf2f5d4f60ecbd4d2e73bc6ad691af8a5667"
HELPER_PINS = {
    "audit_qwen_sign_shortcut.py": "d33613bbb1d71ee3b2007fb007e5266acaf141994091264bed50d27342237e2c",
    "audit_flux_raw_to_image.py": "e8075dfe29401b39678b479b7f13681fd5f2ce56a44ff9e5761a2f74b4431835",
    "audit_flux_functional_qwen_prefix.py": "5004786be76e1f576d2df65e6d1225f8fdd41c4a7e09e5ed8a6c23057db78a97",
    "audit_flux_functional_pass.py": "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80",
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
SCOPE = ("Inherited-functional-state conditional row-sign frontier; two edited forecasts and three native component "
         "observations, full fields/windows/signed responses and custody; zero neural/tokenizer/learned recurrence.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(run, *, study=None, submit=False):
    controller, worker = HERE / CONTROLLER, HERE / WORKER
    if sha(controller) != CONTROLLER_SHA256 or sha(worker) != WORKER_SHA256:
        raise ValueError("Qualified saved controller/reader changed")
    account = runpy.run_path(str(worker))
    if account["PINS"] != HELPER_PINS or tuple(account["HELPERS"]) != tuple(HELPER_PINS):
        raise ValueError("Saved-only helper closure differs")
    for name, digest in HELPER_PINS.items():
        if sha(HERE / name) != digest:
            raise ValueError("Qualified saved account helper changed: " + name)
    utilities = runpy.run_path(str(controller))
    if submit:
        request = json.loads((Path(run) / "request.json").read_text())
        expected = {WORKER: WORKER_SHA256, **HELPER_PINS, Path(__file__).name: sha(__file__)}
        if any(request["source_hashes"].get(name) != digest for name, digest in expected.items()):
            raise ValueError("Staged saved reader/frontend/helper differs")
        resource = dict(cpu_threads=2, ram_mb=3072, vram_mb=0, disk_gb=1, est_wall_s=240)
        if any(request["resource_exception"].get(key) != value for key, value in resource.items()):
            raise ValueError("Frozen CPU resource estimate differs")
        return utilities["submit"](run, worker=worker)
    if study is None:
        raise ValueError("Completed original row-sign study required")
    return utilities["stage"](run, study, worker=worker,
        helpers=[*account["HELPERS"], Path(__file__).name], experiment="mdb-qwen-row-sign-saved-audit",
        operations=3, scope=SCOPE)


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
