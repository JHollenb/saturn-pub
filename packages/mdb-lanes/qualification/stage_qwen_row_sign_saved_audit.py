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
CONTROLLER_SHA256 = "e15e228136093a1af4c5cfa0f7e262b1a031bd607a035731a94e8678fa0ca122"
WORKER = "audit_qwen_row_sign.py"
WORKER_SHA256 = "33932f9f182f9d28588f3ce6f486241af7edf135b37be6439c0b6954080ce187"
HELPER_PINS = {
    "audit_qwen_sign_shortcut.py": "f1e3d115e76215eebc920f0f62251bc5a9d7f9f0e66ad7fa3403090206f37682",
    "audit_flux_raw_to_image.py": "f8c1a39f9b86b9002e079db084887aece105304f58e28e1ae5f4fa146a0ae4bf",
    "audit_flux_functional_qwen_prefix.py": "2331126c9b3d918b1ac5376a6a13250c14806bdbcf151277f59320440e191c7a",
    "audit_flux_functional_pass.py": "43e163b7543ec225da007b6ecc87418c7c0654f464c36c8e7f1e5234642aeaa8",
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
