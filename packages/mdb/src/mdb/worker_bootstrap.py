"""Verify payload/config/package identity before importing a model worker."""
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

from .payload import seal_package


def main():
    seal = json.loads(Path("payload-seal.json").read_text())
    for name, expected in seal["packages"].items():
        if seal_package(name) != expected:
            raise RuntimeError("installed worker package differs from sealed payload: " + name)
    for name, expected in seal["wheels"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("payload wheel bytes changed: " + name)
    for name, expected in seal.get("files", {}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("payload file bytes changed: " + name)
    worker = Path(sys.argv[1])
    if worker.name != seal["worker"] or hashlib.sha256(worker.read_bytes()).hexdigest() != seal["worker_sha256"]:
        raise RuntimeError("worker identity changed")
    raw = Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != seal["config_bytes_sha256"]:
        raise RuntimeError("worker configuration identity changed")
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    environment = request.get("execution_environment")
    if environment:
        import platform
        for key, field in (("UV_CACHE_DIR", "uv_cache_dir"), ("UV_LINK_MODE", "uv_link_mode"), ("TORCH_HOME", "torch_home"),
                           ("LLM_MODELS_EXTRA_ROOT", "extra_model_roots")):
            if os.environ.get(key) != environment[field]:
                raise RuntimeError("worker environment intent changed: " + key)
        if platform.python_version() != environment["python"]:
            raise RuntimeError("worker Python identity changed")
    from .payload import package_files
    installed = {package + "/" + relative: path for package in seal["packages"]
                 for relative, path in package_files(package).items()}
    for name, expected in request.get("source_hashes", {}).items():
        source = installed.get(name, Path(name))
        if name == "worker.py":
            source = worker
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError("declared source identity changed: " + name)
    if request.get("config_bytes_sha256") and hashlib.sha256(raw).hexdigest() != request["config_bytes_sha256"]:
        raise RuntimeError("worker configuration identity changed")
    if request.get("request_sha256"):
        body = {key: value for key, value in request.items() if key != "request_sha256"}
        # The launch-v1 embedded digest uses the original JSON encoding.
        # mrun's outer request marker separately uses canonical compact JSON.
        encoded = json.dumps(body, sort_keys=True).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != request["request_sha256"]:
            raise RuntimeError("worker request identity changed")
    if environment:
        from .execution_environment import receipt
        observed = receipt()
        Path("execution-environment.json").write_text(json.dumps(observed, sort_keys=True) + "\n")
        print("MDB_EXECUTION_ENVIRONMENT=" + json.dumps(observed, sort_keys=True), flush=True)
    # run_path replaces argv[0] only. Remove the bootstrap's worker argument
    # so an argparse-based worker sees exactly its own command-line options.
    bootstrap_argv = sys.argv
    sys.argv = [str(worker), *bootstrap_argv[2:]]
    try:
        runpy.run_path(str(worker), run_name="__main__")
    finally:
        sys.argv = bootstrap_argv


if __name__ == "__main__":
    main()
