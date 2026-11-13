"""Source-pinned near-data saved-only block19 CPU audit frontend.

The frozen controller uses CPU2/RAM3GiB,240s estimate/600s outer,one lease and
no retries. Only compact metadata/retrievers leave the saved array roots.
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
WORKER = "audit_qwen_error_consumer.py"
WORKER_SHA256 = "0c9df3b4330cb225b7da7d2d72d7fb820daa4367df5c2e18f4de24daa326e0bb"
HELPER_PINS = {
    "audit_qwen_cached_qkv.py": "e04990fbe73a1dab3c3a4f9a480ee70bcac5e955cf946ed95a8f958021a281b9",
    "audit_qwen_row_sign.py": "33932f9f182f9d28588f3ce6f486241af7edf135b37be6439c0b6954080ce187",
    "audit_qwen_sign_shortcut.py": "f1e3d115e76215eebc920f0f62251bc5a9d7f9f0e66ad7fa3403090206f37682",
    "audit_flux_raw_to_image.py": "f8c1a39f9b86b9002e079db084887aece105304f58e28e1ae5f4fa146a0ae4bf",
    "audit_flux_functional_qwen_prefix.py": "2331126c9b3d918b1ac5376a6a13250c14806bdbcf151277f59320440e191c7a",
    "audit_flux_functional_pass.py": "43e163b7543ec225da007b6ecc87418c7c0654f464c36c8e7f1e5234642aeaa8",
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
RESOURCE = dict(cpu_threads=2, ram_mb=3072, vram_mb=0, disk_gb=1, est_wall_s=240)
SCOPE = ("Conditional block19 explicit saved input worlds;two full functional forecasts/two native observations,complete "
    "head/projection/R/M/H20 endpoints and signed responses,prefix/finite-addition accounts;zero model/tokenizer/learned recurrence.")


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
        if any(request["resource_exception"].get(key) != value for key, value in RESOURCE.items()):
            raise ValueError("Frozen CPU resource estimate differs")
        return utilities["submit"](run, worker=worker)
    if study is None:
        raise ValueError("Completed original block19 study required")
    return utilities["stage"](run, study, worker=worker, helpers=[*account["HELPERS"], Path(__file__).name],
        experiment="mdb-qwen-error-consumer-saved-audit", operations=2, scope=SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path); parser.add_argument("--study", type=Path); parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and args.study is None:
        parser.error("--study is required to stage")
    print(json.dumps(execute(args.run, study=args.study, submit=args.submit), sort_keys=True))


if __name__ == "__main__":
    main()
