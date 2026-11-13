"""Actual raw-worker bootstrap/finalizer with model-free run substitution only."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parents[1] / "qualification"

HARNESS = r'''import json,os,runpy,sys
from pathlib import Path
root=Path.cwd()
namespace=runpy.run_path(str(root/'flux_raw_to_image.py'),run_name='controlled_bootstrap_target')
def model_free_run(config,*,progress):
 assert 'torch' not in sys.modules and 'transformers' not in sys.modules
 request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text())
 assert config['_request_record']==request
 assert config['request_sha256']==request['request_sha256']
 output=Path(config['output'])
 assert str(output)==os.environ['BOOTSTRAP_EXPECTED_OUTPUT']
 ledger=root/'fake-science.jsonl'
 with ledger.open('a') as stream:stream.write(json.dumps(dict(output=str(output),job_id=os.environ['MRUN_JOB_ID'],request_sha256=config['request_sha256']))+'\n')
 output.mkdir(parents=True,exist_ok=False)
 summary=dict(schema='model-free-bootstrap-fixture',job_id=os.environ['MRUN_JOB_ID'],request_sha256=config['request_sha256'],calls=1,neural_calls=0)
 (output/'summary.json').write_text(json.dumps(summary,sort_keys=True)+'\n')
 (output/'received-config.json').write_text(json.dumps(config,sort_keys=True)+'\n')
 progress(dict(kind='model-free-fixture-reached',calls=1))
 return summary
namespace['worker'].__globals__['run']=model_free_run
if os.environ['BOOTSTRAP_MODE']=='env-main':
 namespace['main']()
else:
 namespace['worker'](os.environ['SATURN_RUN_CONFIG'])
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raw_worker_default_request_identity_custody_finalizer_and_env_main(tmp_path):
    for case, mode, fault in (("direct", "direct", None), ("env-main", "env-main", None),
        ("source-changed", "direct", "source"), ("config-changed", "direct", "config"),
        ("compact-identity", "direct", "identity")):
        directory = tmp_path / case
        directory.mkdir()
        for name in ("flux_raw_to_image.py", "flux_image_tail_microscope.py"):
            shutil.copyfile(HERE / name, directory / name)
        (directory / "bound-source.txt").write_text("synthetic immutable source binding\n")
        harness = directory / "bootstrap_fixture.py"
        harness.write_text(HARNESS)
        config = dict(output_scratch=str(directory / "outputs"), output_name="model-free-result",
            nested_binding=dict(z=[1, 2], a="default JSON separators are part of the MDB request ABI"),
            transport_limits=dict(max_expanded_bytes=1 << 20, max_member_bytes=1 << 18))
        request = dict(schema="synthetic-bootstrap-request-v1", config=config,
            source_hashes={name: sha(directory / name) for name in ("flux_raw_to_image.py", "flux_image_tail_microscope.py", "bound-source.txt")})
        # Match the stager/MDB protocol, deliberately independent of the
        # worker's compact canonical() helper for other evidence identities.
        request_identity = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
        compact_identity = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        assert request_identity != compact_identity
        request["request_sha256"] = compact_identity if fault == "identity" else request_identity
        request_path, config_path = directory / "request.json", directory / "config.json"
        request_path.write_text(json.dumps(request, sort_keys=True) + "\n")
        config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
        if fault == "source":
            (directory / "bound-source.txt").write_text("changed after source seal\n")
        elif fault == "config":
            config_path.write_text(json.dumps({**config, "output_name": "unbound-output"}))
        job_id = "job-synthetic-raw-bootstrap-" + case
        expected_output = Path(config["output_scratch"]) / job_id / config["output_name"]
        env = {**os.environ, "SATURN_DEBUG_REQUEST": str(request_path), "SATURN_RUN_CONFIG": str(config_path), "MRUN_JOB_ID": job_id,
            "BOOTSTRAP_MODE": mode, "BOOTSTRAP_EXPECTED_OUTPUT": str(expected_output), "CUDA_VISIBLE_DEVICES": ""}
        result = subprocess.run([sys.executable, "-I", "-B", str(harness)], cwd=directory, env=env, capture_output=True, text=True, timeout=20)
        if fault:
            assert result.returncode != 0
            message = {"source": "staged source differs", "config": "request/config differs", "identity": "request identity differs"}[fault]
            assert message in result.stderr
            assert not (directory / "fake-science.jsonl").exists()
            assert not expected_output.exists() and "MDB_DIRECTORY_READY" not in result.stdout
            continue
        assert result.returncode == 0, result.stderr
        calls = (directory / "fake-science.jsonl").read_text().splitlines()
        assert len(calls) == 1 and json.loads(calls[0])["output"] == str(expected_output)
        phase = [line for line in result.stdout.splitlines() if line.startswith("MDB_RAW_TO_IMAGE_PHASE ")]
        assert len(phase) == 1 and json.loads(phase[0].split(" ", 1)[1])["kind"] == "model-free-fixture-reached"
        markers = [json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")]
        assert len(markers) == 1
        marker = markers[0]
        assert marker["job_id"] == job_id and marker["request_sha256"] == request_identity
        assert marker["output_root"] == str(expected_output)
        assert marker["files_manifest_sha256"] == sha(expected_output / "files.json")
        assert marker["summary_sha256"] == sha(expected_output / "summary.json")
        inventory = json.loads((expected_output / "files.json").read_text())
        assert inventory["job_id"] == job_id and inventory["request_sha256"] == request_identity
        assert set(inventory["files"]) == {"summary.json", "received-config.json"}
        assert marker["file_count"] == len(inventory["files"]) == 2
        assert marker["expanded_bytes"] == inventory["expanded_bytes"] == sum(row["bytes"] for row in inventory["files"].values())
        for name, row in inventory["files"].items():
            assert sha(expected_output / name) == row["sha256"] and (expected_output / name).stat().st_size == row["bytes"]
        observed = json.loads((expected_output / "received-config.json").read_text())
        assert observed == {**config, "request_sha256": request_identity, "_request_record": request, "output": str(expected_output)}
