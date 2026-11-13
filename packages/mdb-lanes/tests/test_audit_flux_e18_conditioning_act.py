"""Read the producer's persisted genuine-Act/Session tiny fixture, never models."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parents[1] / "qualification"
A = runpy.run_path(str(HERE / "audit_flux_e18_conditioning_act.py"))


@pytest.fixture(scope="module")
def saved():
    value = os.environ.get("MDB_E18_SAVED_FIXTURE")
    if not value:
        pytest.skip("Pass the producer-generated saved tiny fixture; this reader never creates a model.")
    root = Path(value)
    pair, old, q, streamer, remaining = A["helpers"]()
    spec = A["read"](root / "fixture.json")
    assert spec["profile"] == "tiny-cpu-public-Act-Session"
    acts = A["read"](root / "act-records.json")
    own = {lane: torch.from_numpy(old["load_array"](root, record).copy()).to(torch.bfloat16) for lane, record in spec["own_E"].items()}
    report, cuts = A["act_accounts"](root, acts, own, columns=spec["conditioning_columns"])
    rows = {case: {lane: A["read"](root / "cases" / case / (lane + "-state-rows.json")) for lane in ("forecast", "native")} for case in A["CASES"]}
    return dict(root=root, spec=spec, acts=acts, own=own, report=report, cuts=cuts, rows=rows, old=old, streamer=streamer)


def test_actual_four_patch_receipts_mask_positive_zero_and_own_lanes(saved):
    from saturn_pub.values import describe

    assert sum(len(value) for value in saved["report"].values()) == 4
    for lane in ("forecast", "native"):
        assert saved["report"][lane]["sham"]["changed_values"] == 0
        assert saved["report"][lane]["e18-zero"]["changed_values"] > 0
        assert bool((saved["cuts"][lane]["e18-zero"].payload["conditioning"][..., 2:4].view(torch.int16) == 0).all())
    # Actual fixture deliberately differs in E9 across independent lanes.
    assert describe(saved["own"]["forecast"]) != describe(saved["own"]["native"])
    assert saved["own"]["forecast"][0, 0, 2].view(torch.int16).item() == -32768
    assert saved["cuts"]["forecast"]["e18-zero"].payload["conditioning"][0, 0, 2].view(torch.int16).item() == 0


def test_actual_native_receipts_and_eight_context_operand_joins(saved):
    report = A["receipt_accounts"](saved["root"], saved["acts"], saved["cuts"], saved["rows"], transitions=28)
    assert report["actual_acts"] == 4 and report["actual_native_advances"] == 56 and report["total_receipt_occurrences"] == 60
    assert len(report["advances"]) == 56
    consumer = A["context_accounts"](A["read"](saved["root"] / "context-consumers.json"), saved["cuts"], steps=4)
    assert consumer["actual_occurrences"] == 8


def test_actual_full_own_state_trajectories_keep_native_and_functional_provenance(saved):
    for case in A["CASES"]:
        load = A["state_loader"](saved["root"] / "cases" / case, saved["old"], saved["streamer"], dict(profile="tiny-cpu"), identity_cut=saved["cuts"]["forecast"][case])
        rows = saved["rows"][case]
        result = saved["old"]["trajectory_accounts"](rows["forecast"][1:], rows["native"][1:], {lane: value[0] for lane, value in rows.items()},
            load, joints=2, singles=2, steps=4)
        assert result["complete_transitions"] == 28 and result["independent_neural_calls"] == 0
        assert rows["native"][0]["cut"] == saved["cuts"]["native"][case].fingerprint
        assert rows["forecast"][0]["parent"] == saved["cuts"]["forecast"][case].fingerprint
        assert result["first_bit_discrepancy"] is not None  # Own-lane synthetic E intentionally differs.


def test_actual_prospective_recursive_response_hashes_and_serializer_variants(saved):
    from saturn_pub.store import LocalStore

    responses = A["read"](saved["root"] / "forecast-responses.json")
    assert len(responses["denoiser"]) == 29
    for index, row in enumerate(responses["denoiser"]):
        states = {lane: {case: dict(A["tensor_leaves"](LocalStore(saved["root"] / "cases" / case / (lane + "-state-store")).load(saved["rows"][case][lane][index]["cut"]).payload))
            for case in A["CASES"]} for lane in ("forecast", "native")}
        assert all(set(states[lane][case]) == set(row["fields"]) for lane in states for case in A["CASES"])
        for name in row["fields"]:
            values = {lane: {case: saved["old"]["field_to_numpy"](states[lane][case][name]) for case in A["CASES"]} for lane in states}
            report = A["response_account"](values, row["fields"][name], saved["old"])
            assert report["prospective_full_hash_verified"]
    decoder = A["read"](saved["root"] / "cases/sham/decoder-forecast.json")
    assert "content_sha256" in decoder["decoded"] and "content_sha256" not in decoder["pixels"]
    assert saved["old"]["load_array"](saved["root"] / "cases/sham", decoder["pixels"]).dtype == np.uint8


@pytest.mark.parametrize("mutation", ["crosslane", "columns", "shape", "receipt", "zero-mode"])
def test_hostile_actual_patch_records_refuse(saved, mutation):
    acts = copy.deepcopy(saved["acts"])
    own = dict(saved["own"])
    if mutation == "crosslane":
        own["native"] = own["forecast"]
    elif mutation == "columns":
        acts["native"]["e18-zero"]["columns"] = [0, 2]
    elif mutation == "shape":
        acts["forecast"]["sham"]["exact_shape_mask"] = [1]
    elif mutation == "zero-mode":
        acts["forecast"]["e18-zero"]["receipt"]["act"]["parameters"] = acts["forecast"]["sham"]["receipt"]["act"]["parameters"]
    else:
        acts["native"]["e18-zero"]["receipt"]["fingerprint"] = "0" * 64
    with pytest.raises(ValueError):
        A["act_accounts"](saved["root"], acts, own, columns=saved["spec"]["conditioning_columns"])


def test_hostile_actual_consumer_and_advance_receipt_refuse(saved):
    rows = copy.deepcopy(saved["rows"])
    rows["sham"]["native"][1]["receipt"] = rows["e18-zero"]["native"][1]["receipt"]
    with pytest.raises(ValueError, match="receipt sidecar"):
        A["receipt_accounts"](saved["root"], saved["acts"], saved["cuts"], rows, transitions=28)
    consumers = A["read"](saved["root"] / "context-consumers.json")
    consumers[0]["source_act_cut"] = saved["cuts"]["native"]["e18-zero"].fingerprint
    with pytest.raises(ValueError, match="own acted conditioning"):
        A["context_accounts"](consumers, saved["cuts"], steps=4)


def test_actual_forecast_act_CAS_frontier_excludes_native_only_pages(saved):
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction

    root = saved["root"]
    tx = root / "forecast-transaction"
    verify_transaction(tx, expected_seal_sha256=saved["spec"]["forecast_seal"]["seal_sha256"])
    prediction = saved["old"]["plain"](load_forecast_tree(tx, A["read"](tx / "forecasts/both-complete-forecasts.json")["predictions"]))
    expected = A["act_store_closure"](root, saved["acts"]["forecast"])
    assert {name for name in prediction["external_files"] if name.startswith("acts-store/")} == expected
    assert expected != A["act_store_closure"](root, saved["acts"]["native"])


def test_actual_serializers_and_public_transaction_cover_native_decoder_frontier(saved, tmp_path):
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    root = tmp_path / "envelope"
    shutil.copytree(saved["root"], root)
    shutil.rmtree(root / "forecast-transaction")
    (root / "fixture.json").unlink()
    shutil.rmtree(root / "fixture")
    old = runpy.run_path(str(HERE / "flux_raw_to_image.py"))
    for case in A["CASES"]:
        for lane in ("native-own", "native-reference"):
            # Actual frozen serializer and actual native decoder path spelling, without a VAE call.
            record = old["_save_tensor"](root / "cases" / case, "decoder/" + lane + "/decoder-input.npy", torch.zeros(1, dtype=torch.bfloat16))
            A["save"](root / "cases" / case / ("decoder-" + lane + ".json"), dict(decoder_input=record))
            assert A["native_member"]("cases/" + case + "/" + record["path"])
    config = dict(forecast_transaction_bytes=16 << 20, intervention=dict(address="conditioning", mode="replace", columns=[2, 4]))
    inventory = lambda: {p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=A["sha"](p)) for p in root.rglob("*") if p.is_file()}
    initial = inventory()
    expected = A["pretarget_files"](root, initial, saved["acts"]["forecast"])
    external = {name: initial[name] for name in expected}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    tx = ForecastTransaction(root / "forecast-transaction", model_identity="flux2-klein-4b-original",
        input_identity=hashlib.sha256(canonical(dict(prompt=A["PROMPT"], seed=277, intervention=config["intervention"]))).hexdigest(),
        source_identity=hashlib.sha256(canonical({})).hexdigest(),
        frontier=dict(kind="raw-input", declared_intervention="conditioning-Act", prompt=A["PROMPT"], seed=277, intervention=config["intervention"], source_inventory=external),
        budgets=dict(max_predictions=1, max_executions=1, max_native_calls=1, max_bytes=16 << 20, max_files=10000, max_wall_ms=120000))
    tx.predict("e18-complete-composition", lambda _ctx: {"predictions": dict(external_files=external, acts=saved["acts"]["forecast"], native_targets_at_seal=0,
        responses=dict(path="forecast-responses.json", sha256=A["sha"](root / "forecast-responses.json"), orientation="e18-zero-minus-sham"))})
    seal = tx.seal()
    # Tiny source has56 real advances and zero VAE calls; no production counts invented.
    tx.execute("native-own-E-branches", lambda _ctx: dict(native_transitions=56, native_vae_calls=0, native_act_receipts=2, context_operand_occurrences=8))
    tx.close()
    summary = dict(forecast_seal=seal, transaction=verify_transaction(root / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"]),
        counts=dict(native_transitions=56, native_vae_calls=0), context_operand_occurrences=8)
    result, prediction = A["transaction"](root, config, dict(source_files={}), summary, dict(files=inventory()), saved["old"])
    assert result["phase"] == "closed" and set(prediction["external_files"]) == expected
    assert "frontend/native.json" not in expected
    assert all(not A["native_member"](name) for name in expected)
    # A new forecast-like file after seal must make complete frontier equality refuse.
    (root / "unsealed-forecast.json").write_text("{}")
    with pytest.raises(ValueError, match="complete pretarget"):
        A["transaction"](root, config, dict(source_files={}), summary, dict(files=inventory()), saved["old"])


def test_response_misses_are_observations_and_false_hash_refuses():
    old = A["helpers"]()[1]
    values = dict(forecast=dict(sham=np.array([1., -0.], np.float32), **{"e18-zero": np.array([0., 0.], np.float32)}),
        native=dict(sham=np.array([1., 0.], np.float32), **{"e18-zero": np.array([.5, 0.], np.float32)}))
    raw = np.array([-1., 0.], dtype="<f8")
    seal = dict(shape=[2], elements=2, dtype="float64", orientation="e18-zero-minus-sham", sha256=hashlib.sha256(raw.tobytes()).hexdigest())
    report = A["response_account"](values, seal, old)
    assert report["signed_response_error"]["changed_values"] == 1
    with pytest.raises(ValueError, match="response hash"):
        A["response_account"](values, {**seal, "sha256": "0" * 64}, old)


def test_genuine_receipt_fingerprint_seal_and_duplicate_occurrences_allowed(saved):
    body = saved["acts"]["forecast"]["sham"]["receipt"]
    assert A["sealed_receipt"](body) == A["sealed_receipt"](copy.deepcopy(body))
    damaged = copy.deepcopy(body)
    damaged["operation"] = "continue"
    with pytest.raises(ValueError, match="content seal"):
        A["sealed_receipt"](damaged)
    assert json.loads(json.dumps(body))["fingerprint"] == body["fingerprint"]


def test_actual_saved_worker_compact_packaging_and_marker(saved, tmp_path, monkeypatch, capfd):
    from PIL import Image

    root = tmp_path / "source"
    shutil.copytree(saved["root"], root)
    names = ("plan.json", "request.json", "source.json", "summary.json", "native-contract.json", "files.json", "runtime-contract.json", "input.json",
        "forecast-qwen.json", "native-qwen.json", "act-records.json", "forecast-acts.json", "native-acts.json", "context-consumers.json", "receipt-emissions.jsonl", "timeline.jsonl", "forecast-responses.json")
    # Controlled transport fixture only: source account is mocked, genuine Act/store fixtures are unchanged.
    for name in names:
        if not (root / name).exists():
            A["save"](root / name, dict(transport_fixture=True))
    for case in A["CASES"]:
        base = root / "cases" / case
        pixels = saved["old"]["load_array"](base, A["read"](base / "decoder-forecast.json")["pixels"])
        for lane in ("forecast", "native-own", "native-reference"):
            path = base / "transport-images" / (lane + ".png")
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(pixels).save(path)
            record = dict(png=dict(path=path.relative_to(base).as_posix(), bytes=path.stat().st_size, sha256=A["sha"](path)))
            (base / ("decoder-" + lane + ".json")).write_text(json.dumps(record))
    config = dict(native_root=str(root), native_manifest_sha256="transport-fixture", native_request_sha256="transport-fixture", output_scratch=str(tmp_path / "out"))
    envelope = dict(config=config, source_hashes={str(HERE / "audit_flux_e18_conditioning_act.py"): A["sha"](HERE / "audit_flux_e18_conditioning_act.py")})
    envelope["request_sha256"] = A["identity"](envelope)
    A["save"](tmp_path / "config.json", config)
    A["save"](tmp_path / "request.json", envelope)
    monkeypatch.setenv("SATURN_DEBUG_REQUEST", str(tmp_path / "request.json"))
    monkeypatch.setenv("SATURN_RUN_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MRUN_JOB_ID", "job-reader-transport")

    def account(actual_root, output, **_kwargs):
        assert actual_root == root
        output.mkdir(parents=True)
        result = dict(job_id="job-source-transport", source=dict(source_files={}))
        A["save"](output / "audit.json", result)
        return result, dict(files={})

    monkeypatch.setitem(A["worker"].__globals__, "audit", account)
    A["worker"]()
    marker = json.loads(next(line.removeprefix("MDB_DIRECTORY_READY ") for line in capfd.readouterr().out.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(marker["output_root"])
    assert marker["job_id"] == "job-reader-transport" and marker["request_sha256"] == envelope["request_sha256"]
    assert marker["files_manifest_sha256"] == A["sha"](output / "files.json")
    assert len(list(output.glob("*.png"))) == 6 and not list(output.rglob("*.npy"))
    assert A["read"](output / "summary.json")["model_calls"] == 0
