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
A = runpy.run_path(str(HERE / "audit_flux_h18_state_history.py"))


@pytest.fixture(scope="module")
def saved():
    value = os.environ.get("MDB_H18_SAVED_FIXTURE")
    if not value:
        pytest.skip("Pass the producer-generated saved tiny fixture; this reader never creates a model.")
    root = Path(value)
    pair, old, q, streamer, remaining = A["helpers"]()
    spec = A["read"](root / "fixture.json")
    fq, nq = (A["read"](root / (lane + "-qwen-history.json")) for lane in ("forecast", "native"))
    cfg = dict(history=spec["geometry"], max_length=fq["input_ids"]["shape"][1],
        attention_dispatch=fq["profile"]["attention_dispatch"], rope_buffer_policy=fq["profile"]["rope_buffer_policy"])
    qcfg = dict(num_hidden_layers=nq["counts"]["baseline_blocks"], hidden_size=fq["common"]["0"]["shape"][-1],
        head_dim=nq["sdpa_calls"][0]["QKV"][0]["shape"][-1], num_attention_heads=nq["sdpa_calls"][0]["QKV"][0]["shape"][1],
        num_key_value_heads=fq["first_suffix_ports"]["plus"]["k"]["shape"][1], vocab_size=nq["baseline_logits_shape"][-1])
    qr, own_np, mask = A["qwen_accounts"](root, fq, nq, cfg, old, q, qcfg)
    own = {lane: {name: torch.from_numpy(value.copy()).to(torch.bfloat16) for name, value in records.items()} for lane, records in own_np.items()}
    columns = [qcfg["hidden_size"], 3 * qcfg["hidden_size"]]
    acts = A["read"](root / "act-records.json")
    report, cuts = A["act_accounts"](root, acts, own, columns=columns)
    rows = {case: {lane: A["read"](root / "cases" / case / (lane + "-state-rows.json")) for lane in ("forecast", "native")} for case in A["CASES"]}
    return dict(root=root, spec=spec, acts=acts, own=own, report=report, cuts=cuts, rows=rows,
        old=old, streamer=streamer, qr=qr, fq=fq, nq=nq, config=cfg, qconfig=qcfg, q=q, columns=columns)


def test_actual_qwen_histories_firstsuffix_ports_and_value_vs_bits(saved):
    assert saved["qr"]["native_counts"]["SDPA"] == 7
    assert saved["qr"]["functional_counts"]["SDPA"] == 6
    assert saved["qr"]["native_context"]["consumed_occurrences"] == 2
    assert saved["qr"]["genuine_qwen_act"] is False
    assert saved["qr"]["oddness"]["forecast"]["extra_learned_projections"] == 0
    assert saved["qr"]["oddness"]["native"]["oddness"]["h"]["numerical_changed"] == 0
    zeros = np.array([0., -0.], np.float32)
    diagnostic = A["oddness"](zeros, np.array([0., 0.], np.float32))
    assert diagnostic["numerical_changed"] == 0 and diagnostic["zero_signbit_mismatches"] == 1


def test_actual_four_patch_receipts_hold_negative_H18_and_only_contrast_H27(saved):
    from saturn_pub.values import describe

    assert sum(len(value) for value in saved["report"].values()) == 4
    width = saved["columns"][0]
    for lane in ("forecast", "native"):
        direct, coherent = (saved["cuts"][lane][case].payload["conditioning"] for case in A["CASES"])
        assert describe(direct[..., :2 * width]) == describe(coherent[..., :2 * width])
        assert torch.equal(direct[..., width:2 * width].view(torch.int16), (-saved["own"][lane]["base_conditioning"][..., width:2 * width]).view(torch.int16))
        assert saved["report"][lane]["direct"]["selected_values"] == saved["report"][lane]["coherent"]["selected_values"]


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
        assert "first_bit_discrepancy" in result  # Every numerical result is retained, without a quality gate.


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
    decoder = A["read"](saved["root"] / "cases/direct/decoder-forecast.json")
    assert "content_sha256" in decoder["decoded"] and "content_sha256" not in decoder["pixels"]
    assert saved["old"]["load_array"](saved["root"] / "cases/direct", decoder["pixels"]).dtype == np.uint8


@pytest.mark.parametrize("mutation", ["crosslane", "columns", "shape", "receipt", "zero-mode"])
def test_hostile_actual_patch_records_refuse(saved, mutation):
    acts = copy.deepcopy(saved["acts"])
    own = copy.deepcopy(saved["own"])
    if mutation == "crosslane":
        own["native"]["direct"].zero_()
    elif mutation == "columns":
        acts["native"]["coherent"]["columns"] = [0, 2]
    elif mutation == "shape":
        acts["forecast"]["direct"]["exact_shape_mask"] = [1]
    elif mutation == "zero-mode":
        acts["forecast"]["coherent"]["receipt"]["act"]["parameters"] = acts["forecast"]["direct"]["receipt"]["act"]["parameters"]
    else:
        acts["native"]["coherent"]["receipt"]["fingerprint"] = "0" * 64
    with pytest.raises(ValueError):
        A["act_accounts"](saved["root"], acts, own, columns=saved["columns"])


def test_hostile_actual_consumer_and_advance_receipt_refuse(saved):
    rows = copy.deepcopy(saved["rows"])
    rows["direct"]["native"][1]["receipt"] = rows["coherent"]["native"][1]["receipt"]
    with pytest.raises(ValueError, match="receipt sidecar"):
        A["receipt_accounts"](saved["root"], saved["acts"], saved["cuts"], rows, transitions=28)
    consumers = A["read"](saved["root"] / "context-consumers.json")
    consumers[0]["source_act_cut"] = saved["cuts"]["native"]["coherent"].fingerprint
    with pytest.raises(ValueError, match="own acted conditioning"):
        A["context_accounts"](consumers, saved["cuts"], steps=4)


def test_actual_forecast_act_CAS_frontier_excludes_native_only_pages(saved):
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    root = saved["root"]
    tx = root / "forecast-transaction"
    verify_transaction(tx, expected_seal_sha256=saved["spec"]["forecast_seal"]["seal_sha256"])
    prediction = saved["old"]["plain"](load_forecast_tree(tx, A["read"](tx / "forecasts/two-complete-futures.json")["predictions"]))
    expected = A["act_store_closure"](root, saved["acts"]["forecast"])
    assert {name for name in prediction["external_files"] if name.startswith("acts-store/")} == expected
    # Value-identical native and forecast cuts may share CAS members. Chronology
    # and actual receipt occurrences remain separate even when fingerprints agree.
    assert A["act_store_closure"](root, saved["acts"]["native"]) <= {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_actual_serializers_and_public_transaction_cover_native_decoder_frontier(saved, tmp_path):
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    root = tmp_path / "envelope"
    shutil.copytree(saved["root"], root)
    shutil.rmtree(root / "forecast-transaction")
    (root / "fixture.json").unlink()
    old = runpy.run_path(str(HERE / "flux_raw_to_image.py"))
    for case in A["CASES"]:
        for lane in ("native-own", "native-reference"):
            # Actual frozen serializer and actual native decoder path spelling, without a VAE call.
            record = old["_save_tensor"](root / "cases" / case, "decoder/" + lane + "/decoder-input.npy", torch.zeros(1, dtype=torch.bfloat16))
            A["save"](root / "cases" / case / ("decoder-" + lane + ".json"), dict(decoder_input=record))
            assert A["native_member"]("cases/" + case + "/" + record["path"])
    config = dict(history=saved["config"]["history"], forecast_transaction_bytes=16 << 20, intervention=dict(address="conditioning", mode="replace", columns=[2, 6]))
    inventory = lambda: {p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=A["sha"](p)) for p in root.rglob("*") if p.is_file()}
    initial = inventory()
    expected = A["pretarget_files"](root, initial, saved["acts"]["forecast"])
    external = {name: initial[name] for name in expected}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    tx = ForecastTransaction(root / "forecast-transaction", model_identity="flux2-klein-4b-original",
        input_identity=hashlib.sha256(canonical(dict(prompt=A["PROMPT"], seed=277, intervention=config["intervention"]))).hexdigest(),
        source_identity=hashlib.sha256(canonical({})).hexdigest(),
        frontier=dict(kind="raw-input", prompt=A["PROMPT"], seed=277, history=config["history"], intervention=config["intervention"], source_inventory=external),
        budgets=dict(max_predictions=1, max_executions=1, max_native_calls=1, max_bytes=16 << 20, max_files=10000, max_wall_ms=120000))
    tx.predict("h18-complete-composition", lambda _ctx: {"predictions": dict(external_files=external, acts=saved["acts"]["forecast"], native_targets_at_seal=0,
        responses=dict(path="forecast-responses.json", sha256=A["sha"](root / "forecast-responses.json"), orientation="coherent-minus-direct"))})
    seal = tx.seal()
    # Tiny source has56 real advances and zero VAE calls; no production counts invented.
    tx.execute("native-own-history-branches", lambda _ctx: dict(native_transitions=56, native_vae_calls=0, qwen_native_blocks=7, native_act_receipts=2, context_operand_occurrences=8))
    tx.close()
    summary = dict(forecast_seal=seal, transaction=verify_transaction(root / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"]),
        counts=dict(native_transitions=56, native_vae_calls=0, qwen_native_blocks=7), context_operand_occurrences=8)
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
    values = dict(forecast=dict(direct=np.array([1., -0.], np.float32), coherent=np.array([0., 0.], np.float32)),
        native=dict(direct=np.array([1., 0.], np.float32), coherent=np.array([.5, 0.], np.float32)))
    raw = np.array([-1., 0.], dtype="<f8")
    seal = dict(shape=[2], elements=2, dtype="float64", orientation="coherent-minus-direct", sha256=hashlib.sha256(raw.tobytes()).hexdigest())
    report = A["response_account"](values, seal, old)
    assert report["signed_response_error"]["changed_values"] == 1
    with pytest.raises(ValueError, match="response hash"):
        A["response_account"](values, {**seal, "sha256": "0" * 64}, old)


def test_genuine_receipt_fingerprint_seal_and_duplicate_occurrences_allowed(saved):
    body = saved["acts"]["forecast"]["direct"]["receipt"]
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
    names = ("plan.json", "request.json", "source.json", "summary.json", "native-contract.json", "files.json", "runtime-contract.json", "input.json", "count-contract.json",
        "forecast-qwen-history.json", "native-qwen-history.json", "act-records.json", "forecast-acts.json", "native-acts.json", "context-consumers.json", "receipt-emissions.jsonl", "timeline.jsonl", "forecast-responses.json")
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
    envelope = dict(config=config, source_hashes={str(HERE / "audit_flux_h18_state_history.py"): A["sha"](HERE / "audit_flux_h18_state_history.py")})
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


@pytest.mark.parametrize("kind", ["negation", "context", "index", "response", "module"])
def test_hostile_actual_history_recipes_contexts_and_native_census_refuse(saved, kind):
    fq, nq = copy.deepcopy(saved["fq"]), copy.deepcopy(saved["nq"])
    edit = str(saved["spec"]["geometry"]["edit_index"])
    if kind == "negation":
        nq["minus"][edit] = nq["common"][edit]
    elif kind == "context":
        nq["consumed"][0]["context"]["position_ids"] = {"tensor": nq["input_ids"]}
    elif kind == "index":
        nq["consumed"][0]["index"] += 1
    elif kind == "response":
        fq["conditioning_response"]["sha256"] = "0" * 64
    else:
        nq["module_entries"] = [row for row in nq["module_entries"] if row["name"] != "lm_head"]
    with pytest.raises(ValueError):
        A["qwen_accounts"](saved["root"], fq, nq, saved["config"], saved["old"], saved["q"], saved["qconfig"])


def test_nonzero_port_miss_is_retained_with_later_exit_not_numerical_gate(saved):
    nq = copy.deepcopy(saved["nq"])
    nq["first_suffix_ports"]["minus"]["gate"] = nq["first_suffix_ports"]["minus"]["up"]
    result, _, _ = A["qwen_accounts"](saved["root"], saved["fq"], nq, saved["config"], saved["old"], saved["q"], saved["qconfig"])
    assert result["first_suffix"]["minus"]["gate"]["changed_values"] > 0
    assert "Hnext" in result["first_suffix"]["minus"]
    assert result["first_suffix_first_bit_discrepancy"]["minus"] is not None


def test_actual_metadata_authentication_standard_request_and_changed_identity(tmp_path):
    root = tmp_path / "envelope"
    root.mkdir()
    counts = dict(component_hydrations=3, cuda_placements=6, cpu_offloads=6, noise_draws=2, functional_transitions=224,
        native_transitions=224, functional_decode_calls=2, native_vae_calls=4, native_qwen_calls=1, act_receipts=4,
        receipt_occurrences=228, guard_refusals=0, qwen_functional_blocks=36, qwen_native_blocks=45,
        qwen_native_authorizations=10, qwen_functional_SDPA=36, qwen_native_SDPA=45)
    source_hashes = {"flux_h18_state_history.py": A["WORKER_SHA256"], "qwen_h18_state_history.py": A["HISTORY_SHA256"]}
    config = dict(schema="mdb-flux-h18-state-history-plan-v1", prompt=A["PROMPT"], seed=277, branches=list(A["CASES"]),
        conditioning_columns=[2560, 7680], history=dict(edit_index=18, horizon=27, selected=[9, 18, 27], genuine_qwen_act=False),
        helper_source_sha256={"qwen_h18_state_history.py": A["HISTORY_SHA256"]})
    request = dict(config=config, source_hashes=source_hashes, envelope="synthetic authentication mechanics only; no production outcome")
    request["request_sha256"] = A["identity"](request)
    A["save"](root / "request.json", request)
    A["save"](root / "plan.json", {**config, "request_sha256": request["request_sha256"]})
    A["save"](root / "runtime-contract.json", {})
    seal = dict(seal_sha256="synthetic")
    A["save"](root / "source.json", dict(job_id="synthetic", request_sha256=request["request_sha256"], worker_sha256=A["WORKER_SHA256"],
        helper_source_sha256=config["helper_source_sha256"], forecast_seal=seal, runtime_contract_sha256=A["sha"](root / "runtime-contract.json")))
    A["save"](root / "summary.json", dict(job_id="synthetic", request_sha256=request["request_sha256"], counts=counts, contract=counts, forecast_seal=seal, context_operand_occurrences=8))
    A["save"](root / "native-contract.json", dict(counts=counts, forecast_module_entries=0, context_operand_occurrences=8, history_helper_sha256=A["HISTORY_SHA256"], genuine_qwen_act=False))
    A["save"](root / "count-contract.json", counts)
    def inventory():
        files = {p.name: dict(bytes=p.stat().st_size, sha256=A["sha"](p)) for p in root.iterdir() if p.name != "files.json"}
        return dict(schema="mdb-image-tail-files-v1", request_sha256=request["request_sha256"], job_id="synthetic", files=files, expanded_bytes=sum(row["bytes"] for row in files.values()))
    A["save"](root / "files.json", inventory())
    A["authenticate"](root, A["sha"](root / "files.json"), request["request_sha256"], A["helpers"]()[1])
    request["config"]["seed"] = 278
    (root / "request.json").write_text(json.dumps(request))
    (root / "files.json").write_text(json.dumps(inventory()))
    with pytest.raises(ValueError, match="request/effective"):
        A["authenticate"](root, A["sha"](root / "files.json"), request["request_sha256"], A["helpers"]()[1])


def test_primary_regions_ports_and_diagnostics_are_separate_coverage(saved):
    case_reports = {}
    for case in A["CASES"]:
        load = A["state_loader"](saved["root"] / "cases" / case, saved["old"], saved["streamer"], dict(profile="tiny-cpu"), identity_cut=saved["cuts"]["forecast"][case])
        rows = saved["rows"][case]
        trajectory = saved["old"]["trajectory_accounts"](rows["forecast"][1:], rows["native"][1:], {lane: value[0] for lane, value in rows.items()}, load, joints=2, singles=2, steps=4)
        case_reports[case] = dict(denoiser=trajectory)
    coverage = A["coverage_account"](saved["qr"], case_reports, dict(denoiser=[]))
    assert coverage["primary_qwen_whole_fields"]["fields"] == 11
    assert coverage["qwen_query_region_diagnostics"]["fields"] == 22
    assert coverage["aligned_first_suffix_port_comparisons"]["fields"] == 30
    assert coverage["primary_denoiser_whole_fields"]["fields"] > 56
    assert coverage["primary_qwen_whole_fields"]["positions"] == sum(row["full"]["sample_count"] for family in saved["qr"]["fields"].values() for row in family.values())
