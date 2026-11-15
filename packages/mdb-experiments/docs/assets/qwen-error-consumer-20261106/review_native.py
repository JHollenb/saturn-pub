"""Review the exact finite two-world request; no scientific accuracy gate."""
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
QUAL = BASE / "saturn-pub/packages/mdb/qualification"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frontend = QUAL / "stage_qwen_error_consumer.py"
    if sha(frontend) != "c7574da612f8548e0574e627496e273c6ac8f81e8b04f5984fb368759d21cf28":
        raise ValueError("Qualified native frontend changed")
    module = runpy.run_path(str(frontend))
    request = module["inspect_run"](ROOT / "run-01")
    protocol = json.loads((ROOT / "selection-v1/protocol.json").read_text())
    qualification = {
        "producer-qualification-v1/qualification.json": "3eae562539b741592a86efe40843da3b3b3b808e6e89b2a5f5d33cfe818ec5ef",
        "audit-qualification-v1/qualification.json": "f54a5f6327cc8377462b3235231561d78528dc6a5ec5971564420782c38fee21",
        "frontend-qualification-v1/attempt-02/qualification.json": "9a01d920bc3f0e891a2c79d264fd3ff9eb7acf70d65c6f4e90a30467bfe75aab",
    }
    for name, expected in qualification.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Qualification changed: " + name)
    if request["request_sha256"] != "d0bb13015d8bcb735f0de5cc2a4587d8aa96eb13b7af51d6c6af391015cdf827":
        raise ValueError("Exact staged request differs")
    if sha(ROOT / "selection-v1/protocol.json") != "4e62bc3fa8adcd059ac12f775effa0010752e3c5e0cca2a2ebc4702ecef7f847":
        raise ValueError("Frozen science changed")
    if request["workload_geometry"] != protocol["physical_plan"] or protocol["scientific_gates"]:
        raise ValueError("Physical/scientific plan differs")
    if request["sizing"] != dict(mode="intent-first", reservation=None, timeout_s=300) or request["needs"] != dict(cuda=True, host="beast"):
        raise ValueError("Declared sizing/device differs")
    command = module["command"](request)
    if command[-2:] != ["--config", "saturn_run_config.json"]:
        raise ValueError("Actual worker entry differs")
    report = dict(schema="mdb-error-consumer-finite-source-review-v1", request_sha256=request["request_sha256"],
        review_status="ready-for-finite-run", blocking_issues=[], reviewer="root exact source/ABI/resource review",
        qualification_receipts=qualification, source_and_wheel_files_authenticated=len(request["source_hashes"])+len(request["runtime_wheels"]),
        command=command, selection_sha256=sha(ROOT / "selection-v1/protocol.json"),
        source_scope=dict(worlds=2, context_arrays=5, archive_bytes=5256576, original_parameter_tensors=11,
            original_parameter_payload_bytes=201861632, source_lineage="Two turn14 presealed forecasts; five turn13 original context arrays. No H20/native target is a forecast input."),
        source_review=[
            "Read shared explicit-world evaluator, adapter, saved reader and both frontend contracts. No fictitious row-sign intervention or implicit baseline at block19.",
            "Both complete own-input forecasts and all five complete FP64 response hashes precede global seal and exactly two native targets. Functional modules denied before release.",
            "Native O projection is an actual passive output hook, R actual RMS input, M actual down output, H20 actual return; actual SDPA QKV/head receipts retained.",
            "Two source roots, exact array bytes/layout/context and original parameter payloads are authenticated. Complete original checkpoint recipe streaming remains near-data execution, not proved by synthetic fixture.",
            "Four SDPA/28linear/30nativeentries/one hydration; source/parameter immutability before/after. Shared native primitives, not independent CUDA arithmetic.",
            "Saved numeric/rawbit/zero-sign differences, input-prefix premise and all finite-addition/energy/direction accounts remain observations. Descriptive cross terms do not establish causal mediation.",
            "Aggregate SDPA receipts are written after targets and joined to presealed forecast operands; no pretarget sidecar claim."],
        sizing=request["sizing"], scheduling=request["scheduling_contract"], expected_wall_seconds=60,
        expected_over_15_minutes=False, scientific_accuracy_gates=0,
        stopping_policy="Model/input/source/schema/shape/layout/device/finite/custody/resource mechanics only; preserve all numerical counterexamples.",
        narrative_review=dict(latest="holistic-review-11.json", next="holistic review12 running against completed15/16 and prospective17; not an admission gate"),
        inherited_style_diagnostic="Unchanged frozen turn14 source retains its known I001 import-spacing diagnostic; no numerical/source repair claimed.")
    with (ROOT / "run-01/finite-run-review-v1.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(request_sha256=request["request_sha256"], source_and_wheel_files=report["source_and_wheel_files_authenticated"],
        review_sha256=sha(ROOT / "run-01/finite-run-review-v1.json"), sizing=request["sizing"]["mode"], scientific_accuracy_gates=0), sort_keys=True))


if __name__ == "__main__":
    main()
