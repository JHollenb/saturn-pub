"""Model-free successor source, site, trajectory and counterexample checks."""
from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "qualification"
A = runpy.run_path(str(HERE / "analyze_flux_single_successor.py"))
H = runpy.run_path(str(HERE / "summarize_context_window.py"))


def test_shared_site_defaults_and_explicit_next_consumer_census():
    site = A["PANEL"]["audit_site"]()
    assert site == {"single_index": 0, "prefix_exit_key": "post_single0", "suffix_native_steps": 21,
                    "adapter_steps": 191, "receipt_emissions": 201}
    site = A["PANEL"]["audit_site"](A["SuccessorSource"]({}))
    assert site == {"single_index": 1, "prefix_exit_key": "post_single1", "suffix_native_steps": 20,
                    "adapter_steps": 182, "receipt_emissions": 192}
    with pytest.raises(ValueError, match="site"):
        A["PANEL"]["audit_site"](SimpleNamespace(single_index=True))
    with pytest.raises(ValueError, match="contract"):
        A["PANEL"]["audit_site"](SimpleNamespace(single_index=1, suffix_native_steps=21))


def source_inventory_fixture(tmp_path):
    names = {"input-store/cuts/cut.json": "typed-store/cuts/cut.json",
             "prior-context-source.json": "source.json", "prior-prefixes.json": "prefixes.json",
             "prior-zero-row.json": "rows/KV.image.json", "prior-text-row.json": "rows/coherent.json"}
    source_files, manifest = {}, {}
    for target, original in names.items():
        path = tmp_path / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(target.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        prior_path = "contexts/p1-s53/" + original
        manifest[prior_path] = digest
        source_files[target] = {"prior_path": prior_path, "sha256": digest, "bytes": path.stat().st_size}
    selection = {"files": {"input-store/cuts/cut.json": source_files["input-store/cuts/cut.json"]["sha256"]},
        "source_record": {"path": "prior-context-source.json"}, "prefix_record": {"path": "prior-prefixes.json"},
        "prior_rows": {"zero": {"path": "prior-zero-row.json"}, "text": {"path": "prior-text-row.json"}},
        "source_files": source_files}
    return selection, manifest


def test_source_bytes_are_joined_to_original_manifest_not_selection_self_hash(tmp_path):
    selection, manifest = source_inventory_fixture(tmp_path)
    result = A["verify_source_inventory"](tmp_path, selection, manifest, H)
    assert result["files"] == 1 and result["original_manifest_joined_files"] == 5
    path = tmp_path / "prior-zero-row.json"
    path.write_bytes(b"changed but selection self-consistent")
    selection["source_files"][path.name].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    with pytest.raises(ValueError, match="original producer manifest"):
        A["verify_source_inventory"](tmp_path, selection, manifest, H)


def test_source_inventory_rejects_extra_cas_and_foreign_original_path(tmp_path):
    selection, manifest = source_inventory_fixture(tmp_path)
    (tmp_path / "input-store/extra").write_bytes(b"extra")
    with pytest.raises(ValueError, match="extra files"):
        A["verify_source_inventory"](tmp_path, selection, manifest, H)
    (tmp_path / "input-store/extra").unlink()
    selection["source_files"]["prior-zero-row.json"]["prior_path"] = "../other/row.json"
    with pytest.raises(ValueError, match="unsafe or foreign"):
        A["verify_source_inventory"](tmp_path, selection, manifest, H)


def test_native_implementation_closure_allows_encoder_only_extra_and_refuses_drift():
    prior = {"/old/pipeline.py": "p", "/old/transformer.py": "t", "/old/adapter.py": "a",
             "/old/modeling_qwen3.py": "q"}
    current = {"/new/pipeline.py": "p", "/new/transformer.py": "t", "/new/adapter.py": "a"}
    assert A["native_implementation_closure"](current, prior) == {"pipeline.py": "p", "transformer.py": "t", "adapter.py": "a"}
    current["/new/adapter.py"] = "changed"
    with pytest.raises(ValueError, match="byte closure"):
        A["native_implementation_closure"](current, prior)
    del current["/new/adapter.py"]
    with pytest.raises(ValueError, match="omitted"):
        A["native_implementation_closure"](current, prior)
    with pytest.raises(ValueError, match="ambiguous"):
        A["basename_hashes"]({"/one/a.py": "1", "/two/a.py": "2"})


def reference_fixture(phase="single", *, step=3, index=2):
    point = {"logical_step": step, "phase": phase, "index": index}
    slots = ["latent"] if step == 4 else ["image", "text", "latent"]
    if phase == "scheduler":
        slots.append("noise_prediction")
    arrays, values = {}, {}
    for i, slot in enumerate(slots):
        value = np.full((1, 2, 2), i + 1, dtype=np.float32)
        desc = {"shape": [1, 2, 2], "dtype": "torch.float32", "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        arrays[slot] = {"descriptor": desc, "full_array": {"descriptor": desc, "path": slot},
                        "rms": float(i + 1), "l2": float(2 * (i + 1))}
        values[slot] = value
    selected = phase in {"readout", "scheduler"} or step == 4 or index in {2, 3, 10}
    entry = {"kind": "prefix", "receipt_id": "receipt", "cut": "cut", "point": point,
             "arrays": arrays, "retained_full_cut": selected}
    receipt = {"fingerprint": "receipt", "result": "cut"}
    e = SimpleNamespace(point=lambda _: point, array=lambda record: values[record["path"]],
        slot=lambda _, name: values[name], store={"cuts": {"cut": {}}})
    return e, entry, receipt, values


def test_every_retained_reference_array_is_checked_against_full_cut():
    e, entry, receipt, values = reference_fixture()
    entry["arrays"]["image"]["rms"] += 1e-12
    stats, arrays = A["verify_reference_entry"](e, entry, receipt, first=True)
    assert stats["image"]["reported_rms_minus_offline"] > 0  # retain float diagnostic, no quality gate
    assert set(arrays) == {"image", "text", "latent"}
    e.slot = lambda _, name: values[name] + 1
    with pytest.raises(ValueError, match="full StateCut"):
        A["verify_reference_entry"](e, entry, receipt, first=True)


def test_final_absence_and_real_post_readout_noise_are_explicit():
    e, entry, receipt, _ = reference_fixture("project", step=4, index=None)
    stats, _ = A["verify_reference_entry"](e, entry, receipt, first=True)
    assert set(stats) == {"latent"}
    entry["arrays"]["image"] = entry["arrays"]["latent"]
    with pytest.raises(ValueError, match="live-slot inventory"):
        A["verify_reference_entry"](e, entry, receipt, first=True)
    e, entry, receipt, _ = reference_fixture("scheduler", index=None)
    assert "noise_prediction" in A["verify_reference_entry"](e, entry, receipt, first=True)[0]
    del entry["arrays"]["noise_prediction"]
    with pytest.raises(ValueError, match="live-slot inventory"):
        A["verify_reference_entry"](e, entry, receipt, first=True)


@pytest.mark.parametrize("edit", ["receipt", "cut", "point", "retention"])
def test_reference_receipt_and_selected_frontier_custody(edit):
    e, entry, receipt, _ = reference_fixture()
    if edit == "receipt":
        entry["receipt_id"] = "other"
    elif edit == "cut":
        entry["cut"] = "other"
    elif edit == "point":
        entry["point"] = {"phase": "wrong"}
    else:
        entry["retained_full_cut"] = False
    with pytest.raises(ValueError):
        A["verify_reference_entry"](e, entry, receipt, first=True)


def curves(values):
    return [{"index": i, "point": {"phase": "single", "index": i + 2},
             "distances": {"image": {"exact_base": value == 0, "base_l2": value}}}
            for i, value in enumerate(values)]


def test_first_write_miss_later_divergence_and_partial_repair_are_not_aborts():
    initial = {"image": {"exact_base": True, "base_l2": 0}}
    result = A["trajectory_trends"](curves([0, 4, 2, 0, 3]), initial)
    assert result["H_first_reader"]["observed"] == "miss"
    assert result["H_first_reader"]["first_image_divergence"]["curves_index"] == 1
    assert result["image_l2_changes"][2]["signed_l2_change_from_previous"] == -2
    assert result["exact_image_reconvergence_points"] == [{"phase": "single", "index": 5}]
    assert "no adaptive" in result["H_first_reader"]["action_on_miss"]
    result = A["trajectory_trends"](curves([0, 0]), initial)
    assert result["H_first_reader"]["first_image_divergence"] is None
    assert result["H_first_reader"]["observed"] == "miss"
    assert A["trajectory_trends"](curves([1, 2]), initial)["H_first_reader"]["observed"] == "matched"


def test_known_endpoint_reference_mismatch_is_a_retained_control():
    rows = {name: {"image": {"sha256": name}, "final_latent": {"descriptor": {"value": name}}}
            for name in ("zero", "text")}
    selection = {"expected_pngs": {"zero": "other", "text": "text"},
                 "expected_final_latents": {name: {"value": name} for name in rows}}
    e = SimpleNamespace(array=lambda _: np.zeros((1,)), png=lambda _: np.zeros((1,)))
    result = A["prior_endpoint_controls"](e, selection, rows)
    assert result["zero"] == {"png_equals_prior": False, "final_latent_equals_prior": True}
    assert result["text"]["png_equals_prior"]


def test_current_parent_verifies_once_and_retains_actual_outer_vs_panel_load(tmp_path):
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    data = tmp_path / "extracted/flux-single-successor-result"
    data.mkdir(parents=True)
    tx = ForecastTransaction(data / "transaction", model_identity="fixture", input_identity="fixture-input",
                             source_identity="fixture-source", frontier={"kind": "architectural"})
    with tx:
        tx.predict("single_kv_hypotheses", lambda _: {"predictions": {"fixture": True}})
        seal = tx.seal()
        tx.execute("native", lambda _: {"fixture": True})
    plan = {"fixture": True}
    request = {"config": plan, "request_sha256": "current-request", "source_hashes": {}}
    source = {"job_id": "current-job", "request_sha256": "current-request", "source_hashes": {}, "model_identity": "fixture"}
    summary = {"job_id": "current-job", "request_sha256": "current-request", "model_identity": "fixture",
        "counts": dict(A["PANEL_COUNTS"]), "seal": seal,
        "transaction_verification": verify_transaction(data / "transaction", expected_seal_sha256=seal["seal_sha256"])}
    write(data / "plan.json", plan)
    write(data / "summary.json", summary)
    outer = {**summary, "counts": {**summary["counts"], "model_loads": 1},
             "panel_summary_sha256": A["sha_file"](data / "summary.json")}
    write(data / "outer-summary.json", outer)
    write(data / "source.json", source)
    write(data / "plan-frozen.json", {"job_id": "current-job", "request_sha256": "current-request",
        "plan_sha256": A["sha_file"](data / "plan.json"), "model_loads": 0, "adapter_steps": 0, "candidate_calls": 0})
    write(tmp_path / "request.json", request)
    write(tmp_path / "job.json", {"job_id": "current-job", "state": "succeeded"})
    write(tmp_path / "report.json", {"job_id": "current-job", "request_sha256": "current-request",
        "archive_root": "flux-single-successor-result", "report": outer})
    calls = []
    def verify(root, *, limits):
        calls.append(root)
        return {"mechanics_status": "verified", "job_id": "current-job", "request_sha256": "current-request"}
    result = A["current_parent"](tmp_path, custody_verify=verify)
    assert calls == [tmp_path.resolve()]
    assert result["summary"]["counts"]["model_loads"] == 0
    assert result["outer"]["counts"]["model_loads"] == 1


def test_framework_free_auditor_import():
    script = '''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','diffusers','transformers','saturn','common'}:raise RuntimeError(fullname)
sys.meta_path.insert(0,Block())
runpy.run_path(sys.argv[1])
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(HERE / "analyze_flux_single_successor.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
