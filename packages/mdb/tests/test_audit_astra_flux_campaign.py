"""Model-free accounts using the frozen producer's actual serializer and public CAS."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import runpy
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from saturn_pub.core import Adapter, Session
from saturn_pub.store import LocalStore

from mdb.forecast_transaction import ForecastTransaction, verify_transaction

HERE = Path(__file__).parents[1] / "qualification"
A = runpy.run_path(str(HERE / "audit_astra_flux_campaign.py"))
P = runpy.run_path(str(HERE / "flux_gauge_spectral_campaign.py"))
OLD = P["helpers"]()


def test_actual_rich_minimal_serializers_bits_and_corruption(tmp_path):
    value = torch.tensor([-.0, .5, -2.], dtype=torch.bfloat16)
    record = OLD["_save_tensor"](tmp_path, "value.npy", value)
    got = A["array"](tmp_path, record)
    assert np.signbit(got[0]) and A["metric"](got, value.float().numpy())["bit_exact"]
    minimal = OLD["_save_bytes"](tmp_path, "bytes.npy", np.arange(4, dtype=np.uint8))
    assert A["array"](tmp_path, minimal).tolist() == [0, 1, 2, 3]
    (tmp_path / "value.npy").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="file differs"):
        A["array"](tmp_path, record)


def test_actual_reduced_law_serialization_preserves_native_miss(tmp_path):
    save = lambda path, value: P["save"](tmp_path / path, value)
    rec = lambda path, value: OLD["_save_tensor"](tmp_path, path + ".npy", value)
    x = torch.tensor([[[1., 2.], [-1., .5]]], dtype=torch.bfloat16)
    coordinate = torch.tensor([1., -1.], dtype=torch.float64)
    directions = {"top": torch.tensor([1., 0.], dtype=torch.float64), "bottom": torch.tensor([0., 1.], dtype=torch.float64), "random": torch.tensor([1., 1.], dtype=torch.float64) / 2 ** .5}
    spectral = dict(coordinate=rec("coordinate", coordinate), vectors={k: rec(k, v) for k, v in directions.items()}, directions={k: dict(eigenvalue=float(i), rayleigh=float(i)) for i, k in enumerate(directions)})
    forecast, native, laws = {}, {}, {}
    for i, (name, v) in enumerate(directions.items()):
        plus, minus = P["directional_inputs"](x, v)
        # Saved synthetic writes; no learned or module operations are executed.
        yp, yn = plus * .25, minus * .125
        even = P["even_write"](yp, yn)
        records = {sign: dict(input=rec(name + sign + "input", operand), output=rec(name + sign + "output", out)) for sign, operand, out in (("plus", plus, yp), ("minus", minus, yn))}
        records["even"] = rec(name + "even", even); forecast[name] = records
        nplus, nminus = yp.clone(), yn.clone(); nplus[0, 0, 0] += .5
        native[name] = {sign: dict(source_input=records[sign]["input"], actual_input=OLD["_tensor_signature"](operand), output=rec(name + sign + "native", out)) for sign, operand, out in (("plus", plus, nplus), ("minus", minus, nminus))}
        native[name]["even"] = rec(name + "nativeeven", P["even_write"](nplus, nminus))
        laws[name] = P["reduced_law_prediction"](x, v, float(i), even, coordinate, plus, minus)
    save("ff-forecasts.json", dict(tiny=forecast)); save("ff-native.json", dict(tiny=native))
    save("reduced-law-predictions.json", dict(roots=dict(tiny=dict(directions=laws)), native_targets_at_formation=0))
    save("spectral-source.json", spectral); save("ff-applicability.json", dict(linear_in_bias_none=True, linear_out_bias_none=True))
    save("writer/tiny/account.json", dict(original=dict(x=rec("x", x))))
    report = A["ff_accounts"](tmp_path)
    assert report["roots"]["tiny"]["top"]["forecast_minus_native"]["changed"] == 1
    assert len(report["roots"]["tiny"]["top"]["native"]["values"]) == 2
    # Simulate a legitimate one-ULP alternative FP64 reduction using the actual
    # producer scalar serializer. The complete row bytes remain authenticated;
    # independent recomputation must report the miss without discarding it.
    rounded = P["read"](tmp_path / "reduced-law-predictions.json")
    row = rounded["roots"]["tiny"]["directions"]["top"]["source_input_norm_squared"]
    values = torch.tensor(row["values"], dtype=torch.float64).reshape(row["shape"])
    values = torch.nextafter(values, torch.full_like(values, float("inf")))
    rounded["roots"]["tiny"]["directions"]["top"]["source_input_norm_squared"] = P["scalar_rows"](values)
    (tmp_path / "reduced-law-predictions.json").write_text(json.dumps(rounded))
    retained = A["ff_accounts"](tmp_path)
    assert retained["roots"]["tiny"]["top"]["independent_scalar_reduction_errors"]["source_input_norm_squared"]["changed"] == 2
    bad = P["read"](tmp_path / "reduced-law-predictions.json")
    bad["roots"]["tiny"]["directions"]["top"]["ideal_unrounded"]["values"][0] += 1
    (tmp_path / "reduced-law-predictions.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="presealed scalar row bytes"):
        A["ff_accounts"](tmp_path)


def test_complete_signed_response_hash_strict_reduction_rounding_diagnostic():
    generator = torch.Generator().manual_seed(84)
    left, right = (torch.randn(1, 128, 256, generator=generator).to(torch.bfloat16) for _ in range(2))
    sealed = P["signed_fields"](left, right)["/"]
    actual = A["signed"](left.float().numpy(), right.float().numpy())
    report = A["verify_signed_identity"](actual, sealed)
    assert report["sealed_reduction"]["rms"] == sealed["rms"]
    assert report["independently_reduced"]["rms"] == actual["rms"]
    alternate_reduction = dict(sealed, rms=float(np.nextafter(sealed["rms"], np.inf)))
    assert A["verify_signed_identity"](actual, alternate_reduction)["sealed_reduction"]["rms"] != sealed["rms"]
    changed = left.clone(); changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match="complete signed-response bytes"):
        A["verify_signed_identity"](A["signed"](changed.float().numpy(), right.float().numpy()), sealed)


class ArithmeticAdapter(Adapter):
    model_identity = "saved-audit-mechanics"
    execution = dict(family="arithmetic", version=1)

    def addresses(self, state):
        return ("image", "text")

    def validate(self, state):
        assert set(state) == {"image", "text", "clock"}

    def boundary(self, state):
        return str(state["clock"])

    def advance(self, state):
        return dict(state, clock=state["clock"] + 1)


def test_real_public_act_and_retained_boundary_hash_join(tmp_path):
    session = Session(ArithmeticAdapter(), dict(image=torch.zeros(1, 2, 4, dtype=torch.bfloat16), text=torch.ones(1, 1, 4, dtype=torch.bfloat16), clock=0))
    store = LocalStore(tmp_path / "state-store"); branch = tmp_path / "cases/tiny/gauge"; branch.mkdir(parents=True)
    records = []
    for slot in ("image", "text"):
        before, after, receipt = P["apply_slot_edit"](session, slot, P["gauge_replacement"](session.read(slot), "gauge"))
        store.save(before); store.save(after)
        records.append(dict(slot=slot, before=P["cut_record"](before), after=P["cut_record"](after), receipt=receipt.to_dict(), all_other_slots_unchanged=True))
    P["save"](branch / "forecast-acts.json", records)
    report = A["act_accounts"](tmp_path, store, branch, "gauge", "forecast")
    assert len(report) == 2 and report[0]["centered_leakage_rms"] == 0
    tree = P["tensor_tree"](after.payload, OLD)
    assert len(A["verify_boundary"](after.payload, tree)) == 2
    tree["image"]["tensor"]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="boundary hash"):
        A["verify_boundary"](after.payload, tree)


def test_same_parent_real_acts_have_equal_receipts_but_two_occurrences():
    session = Session(ArithmeticAdapter(), dict(image=torch.zeros(1, 2, 4, dtype=torch.bfloat16), text=torch.ones(1, 1, 4, dtype=torch.bfloat16), clock=0))
    parent = session.capture(retain=False); emitted = []
    for lane in ("forecast", "native"):
        branch = session.fork(parent)
        _, _, receipt = P["apply_slot_edit"](branch, "image", P["gauge_replacement"](branch.read("image"), "gauge"))
        A["receipt_valid"](receipt.to_dict())
        emitted.append(dict(lane=lane, receipt=receipt.to_dict()))
    assert emitted[0]["receipt"]["fingerprint"] == emitted[1]["receipt"]["fingerprint"]
    report = A["receipt_statistics"](emitted)
    assert report["physical_occurrences"] == 2 and report["distinct_fingerprints"] == 1
    assert report["repeated_fingerprint_occurrences"] == 1


def test_actual_transaction_global_twenty_future_seal_and_hostile_member(tmp_path):
    P["save"](tmp_path / "ff-forecasts.json", {})
    for name in ("reduced-law-predictions.json", "forecast-responses.json", "spectral-source.json", "ff-applicability.json"):
        P["save"](tmp_path / name, {})
    session = Session(ArithmeticAdapter(), dict(image=torch.zeros(1, 2, 4, dtype=torch.bfloat16), text=torch.ones(1, 1, 4, dtype=torch.bfloat16), clock=0))
    state = session.capture(retain=False); LocalStore(tmp_path / "state-store").save(state)
    P["save"](tmp_path / "source-frontiers.json", dict(roots={}))
    P["save"](tmp_path / "case/forecast-rows.json", [dict(retained=P["cut_record"](state))])
    P["save"](tmp_path / "case/forecast-acts.json", [])
    inventory = {str(p.relative_to(tmp_path)): dict(sha256=A["sha"](p), bytes=p.stat().st_size) for p in tmp_path.rglob("*") if p.is_file()}
    tx = ForecastTransaction(tmp_path / "forecast-transaction", model_identity="synthetic", input_identity="a" * 64, source_identity="b" * 64,
        frontier=dict(kind="architectural"), array_storage="files-v1", budgets=dict(max_predictions=1, max_executions=1, max_bytes=1 << 20))
    tx.frontier("native-prefixes", {}); tx.predict("all-futures", lambda _ctx: dict(predictions=dict(targets_at_seal=0, paths={str(i): dict(root="case") for i in range(20)}, external_files=inventory)))
    seal = tx.seal(); tx.execute("original-native-targets", lambda _ctx: dict(calls=20)); tx.close()
    timeline = [dict(number=i, kind="forecast_suffix_completed", seconds=float(i)) for i in range(20)]
    timeline += [dict(number=20, kind="all_forecasts_globally_sealed", seconds=20., seal_sha256=seal["seal_sha256"]), dict(number=21, kind="native_targets_released", seconds=21., seal_sha256=seal["seal_sha256"])]
    timeline += [dict(number=22+i, kind="native_suffix_completed", seconds=22.+i) for i in range(20)]
    (tmp_path / "timeline.jsonl").write_text("\n".join(json.dumps(r) for r in timeline))
    summary = dict(forecast_seal=seal["seal_sha256"], transaction=verify_transaction(tmp_path / "forecast-transaction"))
    assert A["transaction"](tmp_path, summary, inventory)["sealed_files"] == len(inventory)
    broken = dict(inventory); broken["ff-forecasts.json"] = dict(sha256="c"*64, bytes=3)
    with pytest.raises(ValueError, match="sealed publication"):
        A["transaction"](tmp_path, summary, broken)


def test_only_two_bn_payloads_header_binding_and_normalization(tmp_path):
    arrays = {"bn.running_mean": torch.zeros(128, dtype=torch.bfloat16), "bn.running_var": torch.ones(128, dtype=torch.bfloat16)}
    payloads = {k: v.view(torch.uint16).numpy().tobytes() for k, v in arrays.items()}
    header = {name: dict(dtype="BF16", shape=[128], data_offsets=[i*256, (i+1)*256]) for i, name in enumerate(arrays)}
    raw_header = json.dumps(header).encode(); archive = tmp_path / "checkpoint"; archive.mkdir()
    checkpoint = archive / "tiny.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"".join(payloads.values()) + b"unread unrelated archive payload")
    rows = []
    for name, value in arrays.items():
        buffer = io.BytesIO(); np.save(buffer, value.view(torch.uint16).numpy(), allow_pickle=False)
        npy = buffer.getvalue(); prefix = npy[:-256]
        rows.append(dict(path=name + ".npy", sha256=hashlib.sha256(npy).hexdigest(), npy_prefix_b64=base64.b64encode(prefix).decode(),
            source=dict(header[name], tensor=name, file=checkpoint.name, payload_sha256=hashlib.sha256(payloads[name]).hexdigest())))
    manifest = dict(artifacts=rows, provenance=dict(header_bytes=len(raw_header), header_sha256=hashlib.sha256(raw_header).hexdigest(), checkpoint_sha256="declared-not-reread"))
    P["save"](tmp_path / "manifest.json", manifest)
    source = dict(sources=dict(vae_manifest=dict(path="manifest.json"), vae_retrieval_script=dict(path="retrieve.py", sha256="a"*64)))
    bn, account = A["bn_buffers"](tmp_path, dict(vae_parameter_root=str(archive)), source)
    assert account["decoded_payload_bytes"] == 512 and not account["whole_checkpoint_hash_recomputed"] and account["persistent_copies"] == 0
    payload = dict(latent=torch.arange(512).reshape(1, 4, 128).to(torch.bfloat16), img_ids=torch.tensor([[[0,0,0],[0,0,1],[0,1,0],[0,1,1]]]))
    result = A["decoder_argument"](payload, bn, 1e-5, 4, 4, 1)
    assert result.shape == (1, 32, 4, 4) and np.sort(result.ravel()).tolist() == np.sort(payload["latent"].float().numpy().ravel()).tolist()
    altered = checkpoint.read_bytes(); checkpoint.write_bytes(altered[:9] + b"x" + altered[10:])
    with pytest.raises(ValueError, match="header differs"):
        A["bn_buffers"](tmp_path, dict(vae_parameter_root=str(archive)), source)


def test_original_pixel_consumer_and_nonzero_response_no_gate(tmp_path):
    decoded = torch.tensor([[[[-1., 0.], [.5, 1.]], [[0., 0.], [0., 0.]], [[1., 0.], [-.5, -1.]]]], dtype=torch.bfloat16)
    pixels = np.round(np.clip(A["rne"](A["rne"](decoded.float().numpy()*.5)+.5), 0, 1).transpose(0,2,3,1)[0]*255).astype(np.uint8)
    Image.fromarray(pixels).save(tmp_path / "image.png")
    record = dict(decoded=OLD["_save_tensor"](tmp_path, "decoded.npy", decoded), pixels=OLD["_save_bytes"](tmp_path, "pixels.npy", pixels),
        png=dict(path="image.png", sha256=A["sha"](tmp_path / "image.png"), bytes=(tmp_path / "image.png").stat().st_size))
    assert np.array_equal(A["image_record"](tmp_path, record)[1], pixels)
    changed = pixels.copy(); changed[0,0,0] += 1
    assert A["metric"](changed, pixels)["changed"] == 1
    assert A["signed"](changed, pixels)["signed_mean"] > 0
    assert A["metric"](np.array([-.0], np.float32), np.array([0.], np.float32))["signed_zero_differences"] == 1


def test_cli_config_contract_and_no_generative_imports():
    config = A["make_config"](dict(native_root="saved", native_manifest_sha256="a"*64, native_request_sha256="b"*64, output_scratch="out"))
    assert A["validate_config"](config)["output_name"] == "astra-flux-audit-result"
    source = (HERE / "audit_astra_flux_campaign.py").read_text()
    assert "import torch" not in source and "runpy" not in source and "transformers" not in source and "diffusers" not in source


def test_actual_published_plan_request_schema_without_worker_key():
    """Identity-only subset of hash-authenticated actual JSON; no arrays read."""
    packet = A["read"](Path(__file__).parents[1] / "fixtures/astra-original-identity.json")
    plan, request = packet["plan"], packet["request"]
    assert "worker_sha256" not in plan and "worker_sha256" not in request["config"]
    assert set(plan) - set(request["config"]) == {"job_id", "request_sha256", "output"}
    args = (plan, packet["source"], packet["summary"], request, packet["wrapper"], request["request_sha256"])
    assert A["original_identity"](*args) == request["request_sha256"]
    changed = copy.deepcopy(plan); changed["dose"] = .5
    with pytest.raises(ValueError, match="executed config/source"):
        A["original_identity"](changed, *args[1:])
    wrong_source = copy.deepcopy(request)
    wrong_source["source_hashes"]["flux_gauge_spectral_campaign.py"] = "c" * 64
    wrong_source["request_sha256"] = hashlib.sha256(json.dumps({k:v for k,v in wrong_source.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    # Even a coherent synthetic request identity cannot substitute another worker.
    source, summary, wrapper, plan = (copy.deepcopy(packet[k]) for k in ("source", "summary", "wrapper", "plan"))
    for document in (source, summary, wrapper, plan):
        document["request_sha256"] = wrong_source["request_sha256"]
    with pytest.raises(ValueError, match="executed config/source"):
        A["original_identity"](plan, source, summary, wrong_source, wrapper, wrong_source["request_sha256"])


def test_actual_four_public_fork_splices_require_entire_typed_closure():
    """Actual hash-authenticated native cut JSON, no tensor payload retrieval."""
    from saturn_pub.values import digest

    packet = A["read"](Path(__file__).parents[1] / "fixtures/astra-fork-splices.json")
    assert len(packet) == 4
    for case in packet:
        common, consumed = (case["cuts"][key]["value"] for key in ("common", "consumed"))
        proof = A["fork_clone_proof"](common, consumed)
        assert proof["common_cut"] != proof["consumed_fork_cut"] and proof["fork_parent"] == proof["common_cut"]
        assert proof["full_manifest_except_parent_exact"]
        changed = copy.deepcopy(consumed)
        changed["manifest"]["parent"] = "c" * 64; changed["logical_parent"] = "c" * 64
        changed["fingerprint"] = digest(changed["manifest"])
        with pytest.raises(ValueError, match="parent is not"):
            A["fork_clone_proof"](common, changed)
        changed = copy.deepcopy(consumed)
        changed["manifest"]["execution"]["guidance"] = 2.
        changed["fingerprint"] = digest(changed["manifest"])
        with pytest.raises(ValueError, match="typed state/execution closure"):
            A["fork_clone_proof"](common, changed)


def test_actual_worker_request_identity_and_normal_compact_finalizer(tmp_path, monkeypatch, capsys):
    worker = A["worker"]; globals_ = worker.__globals__
    source = tmp_path / "audit_astra_flux_campaign.py"
    source.write_bytes((HERE / source.name).read_bytes())
    config = A["make_config"](dict(native_root="no-read-original", native_manifest_sha256="a"*64, native_request_sha256="b"*64, output_scratch=str(tmp_path / "outputs")))
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    request = dict(config=config, source_hashes={source.name: A["sha"](source)})
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    request_path = tmp_path / "request.json"; request_path.write_text(json.dumps(request))
    monkeypatch.setenv("MRUN_JOB_ID", "job-tiny-saved-audit"); monkeypatch.setenv("SATURN_DEBUG_REQUEST", str(request_path))
    calls = []
    def fake_audit(root, output, **kwargs):
        calls.append((root, kwargs)); output.mkdir(parents=True)
        report = dict(native_job_id="job-tiny-original", mechanics_status="fixture-only", trend_status="miss retained")
        A["save"](output / "audit.json", report)
        return report
    monkeypatch.setitem(globals_, "audit", fake_audit)
    ready = worker(config_path)
    assert len(calls) == 1 and ready["request_sha256"] == request["request_sha256"]
    assert "MDB_DIRECTORY_READY " in capsys.readouterr().out and ready["file_count"] == 2
    assert A["read"](Path(ready["output_root"]) / "files.json")["job_id"] == "job-tiny-saved-audit"
    source.write_text("changed")
    with pytest.raises(ValueError, match="executed source"):
        worker(config_path)
    assert len(calls) == 1
