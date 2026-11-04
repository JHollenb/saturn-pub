"""Model-free saved-array regressions for the parallel single-block assay."""
import base64
import hashlib
import io
import json
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1] / "qualification"
AUDIT = runpy.run_path(str(HERE / "analyze_flux_single_kv.py"))


@pytest.mark.parametrize("select_k,select_v", [(False, False), (True, False), (False, True), (True, True)])
def test_key_and_value_selectors_preserve_all_queries_and_image_bank(select_k, select_v):
    observed = {key: np.arange(24, dtype=np.float32).reshape(1, 6, 2, 2) + offset
                for key, offset in (("q", 0), ("k", 30), ("v", 70))}
    donor = {key: value + 200 for key, value in observed.items()}
    used = {key: value.copy() for key, value in observed.items()}
    for key, selected in (("k", select_k), ("v", select_v)):
        if selected:
            used[key][:, :2] = donor[key][:, :2]
    assert all(AUDIT["operand_selection"](observed, used, donor, text_rows=2,
        select_k=select_k, select_v=select_v).values())
    used["v"][:, 3] += 1
    with pytest.raises(ValueError, match="image bank changed"):
        AUDIT["operand_selection"](observed, used, donor, text_rows=2, select_k=select_k, select_v=select_v)


def test_key_only_must_not_secretly_replace_values_or_text_queries():
    observed = {key: np.zeros((1, 4, 1, 2), dtype=np.float32) for key in ("q", "k", "v")}
    donor = {key: np.ones_like(value) for key, value in observed.items()}
    used = {key: value.copy() for key, value in observed.items()}
    used["k"][:, :2] = 1
    used["v"][:, :2] = 1
    with pytest.raises(ValueError, match="wrong text K/V"):
        AUDIT["operand_selection"](observed, used, donor, text_rows=2, select_k=True, select_v=False)
    used["v"][:] = 0
    used["q"][:, :2] = 1
    with pytest.raises(ValueError, match="Q or image bank"):
        AUDIT["operand_selection"](observed, used, donor, text_rows=2, select_k=True, select_v=False)


def test_parallel_whole_projection_residual_rounding():
    q = runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py"))["quantize_bf16"]
    residual = np.array([[[32., .125], [.5, -.25]]], dtype=np.float32)
    projection = np.array([[[.3, .25], [.5, -.125]]], dtype=np.float32)
    gate = np.array([[[.5, -.25]]], dtype=np.float32)
    output = q(residual + q(gate * projection))
    assert AUDIT["parallel_residual"](residual, gate, projection, output, q)["rounded_whole_projection_residual_exact"]
    with pytest.raises(ValueError, match="parallel-block residual"):
        AUDIT["parallel_residual"](residual, gate, projection, output + 1, q)


def test_KV_interaction_is_signed_full_vector_not_distance_arithmetic():
    cells = {"zero": np.array([10., 10.]), "K.image": np.array([12., 10.]),
             "V.image": np.array([10., 7.]), "KV.image": np.array([13., 5.])}
    result = AUDIT["kv_interaction"](cells)
    assert result["l2"] == pytest.approx(5 ** .5)
    assert result["signed_mean"] == -.5 and result["max_abs"] == 2
    assert "KV.image" in result["formula"]


def test_numeric_attention_counterexample_is_retained_without_error_gate():
    q = runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py"))["quantize_bf16"]
    actual = np.zeros((1, 1025, 1, 2), dtype=np.float32)
    prediction = {"instrument_status": "applicable", "query_indices": [0, 31, 32, 255, 511, 512, 767, 1023], "cases": {}}
    records = {}
    for name in ("K", "V", "KV"):
        fp64 = np.full((1, 8, 1, 2), 100., dtype=np.float64)
        prediction["cases"][name] = {"attention": fp64, "attention_bf16": q(fp64.astype(np.float32)),
            "text_mass": np.full((1, 1, 8), .5, dtype=np.float64),
            "role": "native-text represented consistency control" if name == "KV" else "novel conditional forecast"}
        records[name] = {"text_rows": 1, "attention_dispatch_output": {"id": name}}
    result = AUDIT["inspect_attention_samples"](prediction, records, lambda _: actual, q)
    assert result["instrument_status"] == "applicable" and result["cases"]["K"]["fp64_error"]["rms"] == 100
    assert result["cases"]["K"]["predicted_cast_bit_exact"] is False
    assert "unmeasured" in result["cases"]["V"]["text_mass"]["authority"]
    prediction["cases"]["K"]["attention_bf16"][0, 0, 0, 0] = 99
    with pytest.raises(ValueError, match="forecast/cast/mass ABI"):
        AUDIT["inspect_attention_samples"](prediction, records, lambda _: actual, q)


def test_edited_target_before_numeric_seal_is_unresolved_not_main_row_filter():
    timeline = [{"kind": "candidate_start", "candidate": "zero", "sequence": 0},
                {"kind": "candidate_complete", "candidate": "zero", "sequence": 1},
                {"kind": "candidate_start", "candidate": "text", "sequence": 2},
                {"kind": "candidate_complete", "candidate": "text", "sequence": 3},
                {"kind": "attention_forecast_sealed", "sequence": 4, "counts": {
                    "candidate_calls": 2, "adapter_steps": 0, "acts": 0, "images": 0, "vae_calls": 0}}]
    for index, name in enumerate(("K", "V", "KV")):
        timeline.extend([{"kind": "candidate_start", "candidate": name, "sequence": 5 + index * 2},
                         {"kind": "candidate_complete", "candidate": name, "sequence": 6 + index * 2}])
    assert AUDIT["numeric_timing"](timeline)["qualified"]
    timeline[5]["sequence"] = 3
    result = AUDIT["numeric_timing"](timeline)
    assert not result["qualified"] and result["status"] == "prospective-timing-unresolved"


def test_public_secondary_seal_binds_only_sources_and_saved_target_samples(tmp_path):
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    h = runpy.run_path(str(HERE / "summarize_context_window.py"))
    q = runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py"))["quantize_bf16"]
    queries = [0, 31, 32, 255, 511, 512, 767, 1023]
    profile = {"extra_positional_arguments": 0, "attn_mask_present": False, "dropout_p": 0., "is_causal": False,
               "scale": None, "enable_gqa": False, "attention_kwargs_present": False, "backend": None,
               "parallel_config_present": False}
    arrays, candidates = {}, {}
    for name in ("zero", "text", "K", "V", "KV"):
        path = name + ".npy"
        arrays[path] = np.zeros((1, 1025, 1, 2), dtype=np.float32)
        candidates[name] = {"text_rows": 1, "dispatch_profile": profile,
            "attention_dispatch_output": {"path": path, "sha256": "saved-" + name, "descriptor": {"dtype": "torch.bfloat16"}},
            "observed_qkv": {key: {"path": path, "sha256": "saved-" + name,
                                   "descriptor": {"dtype": "torch.bfloat16"}} for key in ("q", "k", "v")}}
    counts = {"candidate_calls": 2, "adapter_steps": 0, "acts": 0, "images": 0, "vae_calls": 0}
    binding = {"native_candidates": {name: {"input_parent": name, "qkv": candidates[name]["observed_qkv"],
                                           "dispatch_profile": profile} for name in ("zero", "text")},
               "query_indices": queries, "text_rows": 1, "main_seal_sha256": "0" * 64, "preseal_counts": counts}
    prediction = {"instrument_status": "applicable", "query_indices": queries, "cases": {}}
    samples = {}
    for name in ("K", "V", "KV"):
        value = np.ones((1, 8, 1, 2), dtype=np.float64)
        prediction["cases"][name] = {"attention": value, "attention_bf16": q(value.astype(np.float32)),
            "text_mass": np.full((1, 1, 8), .2, dtype=np.float64),
            "role": "native-text represented consistency control" if name == "KV" else "novel conditional forecast"}
        actual = np.zeros((1, 8, 1, 2), dtype=np.float32)
        errors = AUDIT["sample_error"](value, actual)
        samples[name] = {"attention": actual, "actual_native_dtype": "torch.bfloat16",
            "source_dispatch_array": candidates[name]["attention_dispatch_output"], "comparisons": {
                "fp64": errors, "cast_bf16": errors}, "actual_rms": 0., "text_mass_measured": False}
    tx = ForecastTransaction(tmp_path / "attention-transaction", model_identity="tiny-observed-operand-fixture",
        input_identity="tiny-sources", source_identity="tiny-forecaster", frontier={"kind": "architectural"})
    with tx:
        tx.frontier("source_binding", binding)
        tx.predict("sampled_attention", lambda _: {"predictions": prediction})
        seal = tx.seal()
        tx.execute("edited_candidates", lambda _: {"instrument_status": "applicable", "samples": samples,
            "counts": {**counts, "candidate_calls": 5}})
    record = {"seal": seal, "verification": verify_transaction(tmp_path / "attention-transaction", expected_seal_sha256=seal["seal_sha256"]),
              "source_binding": binding, "instrument_status": "applicable", "text_mass_measured": False,
              "comparisons": {name: {"comparisons": sample["comparisons"], "actual_rms": 0.} for name, sample in samples.items()}}
    (tmp_path / "attention-forecast.json").write_text(json.dumps(record))
    timeline = [{"kind": kind, "candidate": name, "sequence": index} for index, (kind, name) in enumerate([
        ("candidate_start", "zero"), ("candidate_complete", "zero"), ("candidate_start", "text"), ("candidate_complete", "text")])]
    timeline.append({"kind": "attention_forecast_sealed", "sequence": 4, "counts": counts,
        "seal": seal, "source_binding_sha256": h["sha"](h["canonical"](binding))})
    for index, name in enumerate(("K", "V", "KV")):
        timeline.extend([{"kind": "candidate_start", "candidate": name, "sequence": 5 + index * 2},
                         {"kind": "candidate_complete", "candidate": name, "sequence": 6 + index * 2}])
    e = SimpleNamespace(data=tmp_path, read=h["read"], h=h, array=lambda rec: arrays[rec["path"]],
        typed={"quantize_bf16": q}, run={"source": {"transaction": {"seal_sha256": "0" * 64}}})
    source = {"input_cuts": {"zero": "zero", "text": "text"}, "attention_forecast_plan": {"query_indices": queries}}
    result = AUDIT["audit_attention_transaction"](e, source, candidates, timeline)
    assert result["prospective_claim"] == "qualified"
    assert result["cases"]["K"]["fp64_error"]["signed_mean"] == -1
    assert result["cases"]["K"]["predicted_cast_bit_exact"] is False
    candidates["zero"]["observed_qkv"]["q"] = candidates["KV"]["attention_dispatch_output"]
    with pytest.raises(ValueError, match="non-source operands"):
        AUDIT["audit_attention_transaction"](e, source, candidates, timeline)


def test_auditor_and_publisher_import_without_model_frameworks():
    script = '''
import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','diffusers','transformers','saturn','common'}:raise RuntimeError(fullname)
sys.meta_path.insert(0,Block())
for path in sys.argv[1:]:runpy.run_path(path)
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(HERE / "analyze_flux_single_kv.py"),
                             str(HERE / "publish_flux_single_kv.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_failed_publication_role_never_relabels_external_job_success(tmp_path):
    publisher = runpy.run_path(str(HERE / "publish_flux_single_kv.py"))
    notice = publisher["execution_notice"]({"job_state": "failed"})
    assert "failed during REPORT serialization" in notice and "unchanged completed" in notice
    assert "audited separately" in notice and "not a rewritten producer-job success" in notice
    counts = {"adapter_steps": 191, "candidate_calls": 5, "acts": 10, "images": 9, "vae_calls": 9,
              "receipt_emissions": 201, "prompt_encodings": 0, "noise_draws": 0, "model_loads": 1, "physical_batch": 1}
    data = tmp_path / "extracted/flux-single-kv-result"
    data.mkdir(parents=True)
    failure = {"schema": "mdb-flux-single-kv-failure-v1", "job_id": "same-job", "request_sha256": "request",
               "mechanics_status": "errored", "error": "TypeError: report serialization"}
    (tmp_path / "job.json").write_text(json.dumps({"state": "failed", "job_id": "same-job"}))
    (tmp_path / "report.json").write_text(json.dumps({**failure, "archive_sha256": "archive"}))
    (data / "failure.json").write_text(json.dumps(failure))
    (data / "summary.json").write_text(json.dumps({"job_id": "same-job", "request_sha256": "request", "counts": counts,
        "seal": {"seal_sha256": "main"}}))
    (data / "attention-forecast.json").write_text(json.dumps({"seal": {"seal_sha256": "attention"}}))
    for name in ("transaction", "attention-transaction"):
        (data / name / "events").mkdir(parents=True)
        (data / name / "events/0.json").write_text(json.dumps({"kind": "observation_persisted"}))
    (data / "transaction/observations").mkdir()
    (data / "transaction/observations/native.json").write_text(json.dumps({"value": {
        "counts": counts, "rows": [{}] * 9, "candidates": {name: {} for name in ("zero", "text", "K", "V", "KV")}}}))
    (data / "timeline.jsonl").write_text("")
    helper = {"source_run": lambda *args, **kwargs: {"source": {}}, "read": lambda path: json.loads(path.read_text()),
        "verify_transaction": lambda *args, **kwargs: {"phase": "closed"},
        "NumericalTree": lambda *args: SimpleNamespace(decode=lambda value: value),
        "sha": lambda raw: hashlib.sha256(raw).hexdigest()}
    report = AUDIT["source_run"](helper, tmp_path, failed=True)
    assert report["execution_status"]["job_state"] == "failed"
    assert report["execution_status"]["failed_publication_role"] is True
    with pytest.raises(ValueError, match="external job state"):
        AUDIT["source_run"](helper, tmp_path, failed=False)
    (tmp_path / "job.json").write_text(json.dumps({"state": "succeeded", "job_id": "other-job"}))
    with pytest.raises(ValueError, match="external job state"):
        AUDIT["source_run"](helper, tmp_path, failed=True)


def failed_archive_fixture(tmp_path):
    from mdb.forecast_transaction import ForecastTransaction

    helper = runpy.run_path(str(HERE / "summarize_context_window.py"))
    data = tmp_path / "extracted/fixture-result"
    data.mkdir(parents=True)
    tx = ForecastTransaction(data / "transaction", model_identity="tiny", input_identity="tiny-input",
        source_identity="tiny-source", frontier={"kind": "architectural"})
    with tx:
        tx.predict("panel", lambda _: {"predictions": {"declared": True}})
        tx.seal()
        tx.execute("native", lambda _: {"saved": True})
    (data / "transaction-report.json").write_text(json.dumps(tx.report()))
    (data / "plan.json").write_text("{}")
    manifest = {str(path.relative_to(data)): helper["sha"](path.read_bytes()) for path in data.rglob("*") if path.is_file()}
    (data / "files.json").write_text(json.dumps(manifest))
    request = {"config": {}, "source_hashes": {}}
    request["request_sha256"] = helper["sha"](json.dumps(request, sort_keys=True, allow_nan=False).encode())
    failure = {"schema": "mdb-fixture-failure-v1", "job_id": "failed-job", "request_sha256": request["request_sha256"],
               "mechanics_status": "errored", "error": "TypeError: post-science transport report"}
    (data / "failure.json").write_text(json.dumps(failure))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(data.rglob("*")):
            if path.is_file():
                archive.add(path, arcname="fixture-result/" + str(path.relative_to(data)), recursive=False)
    raw = buffer.getvalue()
    (tmp_path / "worker.log").write_text("MDB_DISCOVERY_UPGRADES_ARCHIVE_V1 " + base64.b64encode(raw).decode() + "\n")
    (tmp_path / "report.json").write_text(json.dumps({**failure, "archive_sha256": helper["sha"](raw)}))
    (tmp_path / "request.json").write_text(json.dumps(request))
    (tmp_path / "job.json").write_text(json.dumps({"state": "failed", "job_id": "failed-job"}))
    return helper, data, manifest, {"failure.json": helper["sha"]((data / "failure.json").read_bytes())}


def test_original_failed_archive_and_manifest_remain_unchanged(tmp_path):
    helper, data, manifest, extras = failed_archive_fixture(tmp_path)
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = helper["source_run"](tmp_path, "fixture-result", failed=True, failed_manifest_extras=extras)
    assert result["manifest"] == manifest and result["source"]["retained_failed_manifest_extras"] == extras
    assert result["source"]["verified_artifact_files"] == len(manifest)
    assert result["source"]["metadata_recovery"]["failed_manifest_extras"]["failure.json"]["sha256"] == extras["failure.json"]
    assert json.loads((tmp_path / "job.json").read_text())["state"] == "failed"
    assert before == {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="inventory differs"):
        helper["source_run"](tmp_path, "fixture-result", failed=True)


def test_failed_manifest_extra_rejects_success_overlap_bad_hash_and_payload(tmp_path):
    helper, data, manifest, extras = failed_archive_fixture(tmp_path)
    with pytest.raises(ValueError, match="explicit failed"):
        helper["source_run"](tmp_path, "fixture-result", failed=False, failed_manifest_extras=extras)
    with pytest.raises(ValueError, match="only bounded failure"):
        helper["source_run"](tmp_path, "fixture-result", failed=True, failed_manifest_extras={"array.npy": "0" * 64})
    with pytest.raises(ValueError, match="size/hash"):
        helper["source_run"](tmp_path, "fixture-result", failed=True, failed_manifest_extras={"failure.json": "0" * 64})
    report = helper["read"](tmp_path / "report.json")
    with pytest.raises(ValueError, match="disjoint"):
        helper["failed_extra_metadata"](data, extras, {**manifest, **extras}, report, published_manifest=True)
    value = helper["read"](data / "failure.json")
    value["payload"] = [1, 2, 3]
    (data / "failure.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exact outer failure"):
        helper["failed_extra_metadata"](data, {"failure.json": helper["sha"]((data / "failure.json").read_bytes())},
            manifest, report, published_manifest=True)


def test_publisher_keeps_nine_rows_under_unresolved_controls(tmp_path, monkeypatch):
    module = runpy.run_path(str(HERE / "publish_flux_single_kv.py"))
    real_run_path = runpy.run_path
    common = real_run_path(str(HERE / "publish_flux_carriers.py"))
    root, output = tmp_path / "run", tmp_path / "published"
    data, tx = root / "extracted/flux-single-kv-result", root / "tx"
    data.mkdir(parents=True)
    tx.mkdir()
    (data / "rows").mkdir()
    (data / "typed-store/cuts").mkdir(parents=True)
    write = lambda path, value: path.write_text(json.dumps(value))
    for name in ("source.json", "prefixes.json", "candidates.json"):
        write(data / name, {})
    write(data / "attention-forecast.json", {})
    for name in ("seal.json", "frontiers/source_binding.json", "forecasts/sampled_attention.json", "observations/edited_candidates.json"):
        path = data / "attention-transaction" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, {})
    write(tx / "seal.json", {})
    write(tx / "observations.json", {})
    (tx / "observations").mkdir()
    write(tx / "observations/native.json", {"value": {}})
    names = ["zero", "text", "sham", "K.image", "V.image", "KV.image", "KV.both", "text.text", "coherent"]
    rows = []
    for i, name in enumerate(names):
        image = data / (name + ".png")
        Image.new("RGB", (512, 512), (i * 20, 17, 9)).save(image)
        latent = data / (name + ".npy")
        np.save(latent, np.array([i], dtype=np.float32), allow_pickle=False)
        for cut in ("parent", "final"):
            write(data / "typed-store/cuts" / (cut + ".json"), {"fingerprint": cut})
        row = {"id": name, "parent": "parent", "final_cut": "final", "acts": [],
               "image": {"path": image.name, "sha256": common["sha"](image)},
               "final_latent": {"path": latent.name, "sha256": common["sha"](latent)}}
        write(data / "rows" / (name + ".json"), row)
        rows.append(row)
    observed = {"rows": rows}
    run = {"data": data, "plan": {"branches": [{"id": n} for n in names]}, "transaction": tx,
           "source": {"archive_sha256": "pinned"}, "tree": type("Tree", (), {"decode": lambda self, _: observed})()}
    helper = HERE / "summarize_context_window.py"
    metric = {"base_l2": 2.5, "exact_base": False}
    audit = {"mechanics_status": "valid", "source": run["source"], "auditor_source_sha256": common["sha"](HERE / "analyze_flux_single_kv.py"),
             "audit_helpers": {helper.name: common["sha"](helper)}, "instrument_status": "replay-unresolved",
             "counts": {"images": 9, "vae_calls": 9}, "comparisons": {n: {b: {"latent": metric, "rgb": metric} for b in ("zero", "text")} for n in names},
             "native_paths": {n: {} for n in names}, "replay_qualification": {"controls": {"sham": False}},
             "control_observations": {"sham": False}, "candidate_checks": {}, "paired_source_checks": {}, "sealed_hypotheses": {}, "limits": []}
    audit["text_propagation_differentials"] = {}
    audit["numerical_attention_forecast"] = {"prospective_claim": "not established"}
    audit["image_KV_interactions"] = {key: {"l2": 1.} for key in ("Z4_latent", "final_RGB")}
    audit_path = root / "audit.json"
    write(root / "job.json", {"state": "succeeded"})
    audit["execution_status"] = {"job_state": "succeeded", "failed_publication_role": False,
        "scope": "Scientific mechanics evaluated separately from external worker/transport status; no status rewrite."}
    write(audit_path, audit)
    monkeypatch.setattr(module["runpy"], "run_path", lambda path: {
        "source_run": lambda *a, **k: run, "read": lambda path: json.loads(path.read_text())
    } if Path(path).name == helper.name else real_run_path(path))
    result = module["publish"](root, audit_path, output)
    assert len(result["rows"]) == 9 and result["instrument_status"] == "replay-unresolved"
    assert result["control_observations"]["sham"] is False
    with Image.open(output / "single-kv-finals.png") as sheet:
        for row in rows:
            with Image.open(data / row["image"]["path"]) as original:
                assert sheet.crop(result["figures"]["finals"]["tiles"][row["id"]]["image_box"]).tobytes() == original.tobytes()
    with pytest.raises(ValueError, match="already exists"):
        module["publish"](root, audit_path, output)
