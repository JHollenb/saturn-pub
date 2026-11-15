"""Read metadata only using the frozen controller snapshot; never stage."""
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parents[1]
SOURCE = ROOT / "attempt-06/cohort/qualification/stage_qwen_row_sign.py"
EXPECTED = "bfe6fbf84faaaf3eabec8930a9c868dc72c9a55a0568d01da1fe3f89b4c50cce"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
controller = runpy.run_path(str(SOURCE))
h = controller["utilities"]()
parent = ARTIFACTS / "mdb-qwen-sign-shortcut-20261106"
request, protocol, paths = controller["metadata"](parent / "run-02",
    parent / "compact-collection-01/extracted/binding-minimal-result",
    ROOT.parent / "selection-v1", parent / "frontier-metadata-v1")
receipt = dict(schema="mdb-row-sign-parent-metadata-verification-v1",
    source_sha256=EXPECTED, verifier_sha256=h["sha"](__file__),
    parent_request_sha256=request["request_sha256"], source_arrays=13,
    original_recipe_count=398, selected_recipe_count=11,
    source_metadata_sha256={key: h["sha"](path) for key, path in paths.items()},
    activation_bytes_read=0, checkpoint_bytes_read=0, model_calls=0,
    stage_executed=False, jobs_submitted=0,
    scope="Original metadata/source-wrapper/global-seal/recipe-entry equality only; does not read activation arrays or checkpoint payloads, and makes no numerical or runtime claim.")
output = ROOT / "parent-metadata-verification.json"
with output.open("x") as stream:
    json.dump(receipt, stream, sort_keys=True, indent=2)
    stream.write("\n")
print(json.dumps(dict(path=str(output), sha256=h["sha"](output)), sort_keys=True))
