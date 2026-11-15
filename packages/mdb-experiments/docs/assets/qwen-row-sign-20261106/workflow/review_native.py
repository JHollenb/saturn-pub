"""Root admission review of the fixed conditional row-sign component assay."""
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUAL = ROOT.parents[1] / "saturn-pub/packages/mdb/qualification"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = ROOT / "run-02"
    request = read(run / "request.json")
    assert request["request_sha256"] == "fb202763231fad27712f2b525aa1ea0d041496c13079d6b4a4a7425d36e18a77"
    stage_path = QUAL / "stage_qwen_row_sign.py"
    assert sha(stage_path) == request["launcher_sha256"] == "bfe6fbf84faaaf3eabec8930a9c868dc72c9a55a0568d01da1fe3f89b4c50cce"
    namespace = runpy.run_path(str(stage_path))
    inspected = namespace["inspect_run"](run)
    assert inspected["request_sha256"] == request["request_sha256"]
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        assert sha(run / "payload" / name) == expected, name
    assert request["source_hashes"]["qwen_row_sign_campaign.py"] == "fbd8eceaf1a9ef8917ecdc39a98ff960940aab9da8ecb4697c25c1a33d545f47"
    proofs = {
        "producer-qualification-v3/qualification.json": "9adccec7c8f310c32f27b040af0fbe3a85530eca02521d15905166ba5609c716",
        "controller-qualification-v1/native-qualification-v2.json": "bf8cd8d35a597d6960aaef525f03e51d22bb1a6f9cd8a59cd69c30e5bddaa3cd",
        "helper-qualification-v1/qualification.json": "e7a28111dee9c2b0012b78acfb2a1929fb62cb02d75dc65e66ad43e135fb1c1e",
    }
    for name, expected in proofs.items():
        assert sha(ROOT / name) == expected, name
    qualification = read(ROOT / "producer-qualification-v3/qualification.json")
    assert qualification["producer_sha256"] == request["source_hashes"]["qwen_row_sign_campaign.py"]
    assert qualification["test_sha256"] == sha(QUAL.parent / "tests/test_qwen_row_sign_campaign.py")
    command = read(run / "staged-command.json")["command"]
    assert command[-3:] == ["qwen_row_sign_campaign.py", "--config", "saturn_run_config.json"]
    assert request["resource_exception"] | dict(calibration="") == dict(ram_mb=4096, vram_mb=2048, cpu_threads=4, disk_gb=3, est_wall_s=60, timeout_s=300, calibration="")
    config = request["config"]
    assert config["max_wall_seconds"] == 240 and request["needs"] == dict(cuda=True, host="beast")
    protocol = read(run / "payload/protocol.json")
    assert len(protocol["source_arrays"]) == 13 and protocol["fixed_split"] == 256
    assert request["source_hashes"]["derived-predictions.json"] == "34d59492473e3496681da2cedb1d2a5a8c42d51ebb96900f3937782f81db3f4d"
    assert request["workload_geometry"]["native_block_calls"] == 3 and request["workload_geometry"]["functional_full_negative_blocks"] == 2
    report = dict(schema="mdb-qwen-row-sign-root-finite-review-v1", reviewer="root", request_sha256=request["request_sha256"],
        review_status="ready-for-finite-run", blocking_issues=[], qualifications=proofs,
        source_and_wheel_files_rehashed=len(request["source_hashes"]) + len(request["runtime_wheels"]),
        frontier=dict(kind="architectural", origin="inherited-functional-state", raw_input_prediction=False, genuine_record_prefix_capture=False),
        inspected_source="Full loader, source whitelist/global seal, BF16 original11recipe restoration, native-source/SDPA pins, positive-only shortcut before functional reference, global target guard, actual native context/ports, finite counts and publication entrypoint.",
        independent_instrument_review="Reader author confirmed final tiny saved fixture accounts/Tx/native input/context/transpose/census; original numerical outcomes remain unobserved.",
        resources=request["resource_exception"], timing=dict(estimated_s=60, assay_timer_s=240, whole_worker_s=300),
        memory_reason="201.9MB original component;13inherited arrays44.6MB; B1/512 sequential two F forecasts/two shortcuts/three native targets. 4GiB host and2GiB CUDA declarations include bounded arrays, transient payloads and runtime workspace; successor peaks remain unmeasured.",
        integrity_only_hard_guards=True, accuracy_gates=[],
        scientific_scope="Support-closed versus mixed rows and three endpoint/own-positive response errors. Full and window accounts overlap. No semantic, universal, QwenAct or independent-kernel claim.",
        previous_staged_draft="run-01 preserved unsubmitted; source.json metadata scope was added before this final request.")
    with (run / "finite-run-review-v1.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"request_sha256": request["request_sha256"], "review_status": report["review_status"], "review_sha256": sha(run / "finite-run-review-v1.json")}))


if __name__ == "__main__":
    main()
