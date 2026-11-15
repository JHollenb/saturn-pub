"""Authenticate actual source/request/layout metadata only; never stage."""
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parents[1]
source = ROOT / "attempt-01/cohort/qualification/stage_qwen_cached_qkv.py"
expected = "bf5ccf76dda3059d3b222981efec483c8fe5d57bb93beb182ca6592e0edaa460"
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
s = runpy.run_path(str(source)); h = s["utilities"]()
parent = ARTIFACTS / "mdb-qwen-sign-shortcut-20261106"
row = ARTIFACTS / "mdb-qwen-row-sign-20261106"
p, protocol, paths = s["metadata"](parent / "run-02", parent / "compact-collection-01/extracted/binding-minimal-result",
    ROOT.parent / "selection-v1", parent / "frontier-metadata-v1", row / "selection-v1/qwen-block18-parameter-artifacts.json")
measurement = s["measured_component"](row / "run-02/job.json", p)
worker = s["producer"]()
config = worker["make_config"]({**{key: str(path) for key,path in paths.items()},
    **{key + "_sha256": h["sha"](path) for key,path in paths.items()},
    "parameter_root": p["config"]["parameter_root"], "input_root": protocol["parent_raw_root"]})
counts = worker["validate_config"](config)
assert worker["PHYSICAL_PLAN"] == protocol["physical_plan"] and worker["SOURCE_KEYS"] == set(protocol["source_arrays"])
receipt = dict(schema="mdb-cached-qkv-actual-metadata-proof-v1", frontend_sha256=expected,
    metadata_sha256={key: h["sha"](path) for key,path in paths.items()}, source_arrays=len(protocol["source_arrays"]),
    original_qkv_strides={key: protocol["source_arrays"][key]["stride"] for key in ("q", "k", "v")},
    original_qkv_shapes={key: protocol["source_arrays"][key]["shape"] for key in ("q", "k", "v")},
    measurement=measurement, count_contract=counts, original_request_sha256=p["request_sha256"],
    actual_source_arrays_or_weights_read=0, models=0, CUDA=0, stage_executed=False, jobs=0)
out = ROOT / "actual-metadata-proof.json"
with out.open("x") as f:
    json.dump(receipt, f, sort_keys=True, indent=2); f.write("\n")
print(json.dumps(dict(path=str(out), sha256=h["sha"](out)), sort_keys=True))
