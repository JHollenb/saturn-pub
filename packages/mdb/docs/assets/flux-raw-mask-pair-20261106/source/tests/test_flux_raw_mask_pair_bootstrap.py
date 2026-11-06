"""Exercise the paired worker bootstrap/finalizer without loading a model."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parents[1]
QUALIFICATION = HERE / "qualification"
WORKER = QUALIFICATION / "flux_raw_mask_pair.py"
FINALIZER = QUALIFICATION / "flux_image_tail_microscope.py"

HARNESS = r'''
import json
import os
import runpy
import sys
from pathlib import Path

root = Path.cwd()
namespace = runpy.run_path(str(root / "flux_raw_mask_pair.py"), run_name="paired_bootstrap_target")


def model_free_run(config, *, progress):
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    assert config["_request_record"] == request
    assert config["request_sha256"] == request["request_sha256"]
    output = Path(config["output"])
    assert str(output) == os.environ["BOOTSTRAP_EXPECTED_OUTPUT"]
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "model-free-paired-bootstrap-fixture",
        "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": config["request_sha256"],
        "neural_calls": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")
    progress({"kind": "model-free-paired-fixture-reached"})
    return summary


namespace["worker"].__globals__["run"] = model_free_run
namespace["main"].__globals__["run"] = model_free_run
if os.environ.get("BOOTSTRAP_MODE") == "env-main":
    namespace["main"]()
else:
    namespace["worker"](os.environ["SATURN_RUN_CONFIG"])
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paired_worker_bootstrap_identity_output_and_finalizer(tmp_path):
    for mode in ("direct", "env-main"):
        directory = tmp_path / mode
        directory.mkdir()
        shutil.copyfile(WORKER, directory / WORKER.name)
        shutil.copyfile(FINALIZER, directory / FINALIZER.name)
        (directory / "bootstrap_fixture.py").write_text(HARNESS)
        config = {
            "output_scratch": str(directory / "outputs"),
            "output_name": "paired-bootstrap-result",
            "transport_limits": {"max_expanded_bytes": 1 << 20, "max_member_bytes": 1 << 18},
        }
        request = {
            "schema": "synthetic-paired-bootstrap-request-v1",
            "config": config,
            "source_hashes": {name: sha(directory / name) for name in (WORKER.name, FINALIZER.name)},
        }
        request["request_sha256"] = hashlib.sha256(
            json.dumps(request, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        request_path = directory / "request.json"
        config_path = directory / "config.json"
        request_path.write_text(json.dumps(request, sort_keys=True) + "\n")
        config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
        job_id = "job-synthetic-paired-bootstrap-" + mode
        expected_output = Path(config["output_scratch"]) / job_id / config["output_name"]
        env = {
            **os.environ,
            "SATURN_DEBUG_REQUEST": str(request_path),
            "SATURN_RUN_CONFIG": str(config_path),
            "MRUN_JOB_ID": job_id,
            "BOOTSTRAP_MODE": mode,
            "BOOTSTRAP_EXPECTED_OUTPUT": str(expected_output),
            "CUDA_VISIBLE_DEVICES": "",
        }
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(directory / "bootstrap_fixture.py")],
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert "MDB_RAW_MASK_PAIR_BOOTSTRAP" in result.stdout
        assert "model-free-paired-fixture-reached" in result.stdout
        markers = [
            json.loads(line.split(" ", 1)[1])
            for line in result.stdout.splitlines()
            if line.startswith("MDB_DIRECTORY_READY ")
        ]
        assert len(markers) == 1
        marker = markers[0]
        assert marker["job_id"] == job_id
        assert marker["request_sha256"] == request["request_sha256"]
        assert marker["output_root"] == str(expected_output)
        assert marker["files_manifest_sha256"] == sha(expected_output / "files.json")
        assert marker["summary_sha256"] == sha(expected_output / "summary.json")
        inventory = json.loads((expected_output / "files.json").read_text())
        assert set(inventory["files"]) == {"summary.json"}
