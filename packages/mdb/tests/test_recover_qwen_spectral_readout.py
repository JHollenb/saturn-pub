"""Saved tiny native evidence only; never import a model or load a checkpoint."""
from __future__ import annotations

import copy
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import pytest

R = runpy.run_path(str(Path(__file__).parents[1] / "qualification/recover_qwen_spectral_readout.py"))


@pytest.fixture
def saved(tmp_path):
    value = os.environ.get("MDB_QWEN_SPECTRAL_TINY_FIXTURE")
    if not value:
        pytest.skip("Explicit retained random-native fixture path required; no model executes here")
    source = Path(value); assert source.is_dir()
    target = tmp_path / "saved"; shutil.copytree(source, target)
    return target


def test_config_is_structural_and_defaults_to_actual_failed_run03():
    config = R["make_config"]({key: "1"*64 for key in R["PIN_KEYS"]})
    assert config["native_job_id"] == "job-956049ba0eeb"
    assert config["native_request_sha256"].startswith("4fd54f32")
    assert R["validate_config"](config)["model_calls"] == 0 and len(R["HELPERS"]) == 2
    assert config["source_inventory"] == "source-inventory.json" and config["source_job"] == "source-job.json"
    changed = dict(config, native_request_sha256="b"*64)
    with pytest.raises(ValueError, match="fixed recovery contract"):
        R["validate_config"](changed)


def test_exact_source_inventory_refuses_extra_future_and_changed_file(tmp_path):
    root = tmp_path / "source"; root.mkdir(); (root / "summary.json").write_text("{}\n")
    inventory = R["capture_inventory"](root, job_id="job-fixture", request_sha256="1"*64)
    R["authenticate_inventory"](root, inventory, job_id="job-fixture", request_sha256="1"*64)
    (root / "extra-future.npy").write_bytes(b"not a numerical payload; hostile inventory member")
    with pytest.raises(ValueError, match="inventory/content differs"):
        R["authenticate_inventory"](root, inventory, job_id="job-fixture", request_sha256="1"*64)
    (root / "extra-future.npy").unlink(); (root / "summary.json").write_text("changed")
    with pytest.raises(ValueError, match="inventory/content differs"):
        R["authenticate_inventory"](root, inventory, job_id="job-fixture", request_sha256="1"*64)


def test_actual_saved_tiny_publication_strips_version_subclass_and_copies_no_arrays(saved, tmp_path):
    assert not any(name in sys.modules for name in R["BLOCKED"])
    before = R["capture_inventory"](saved, job_id="job-fixture", request_sha256="1"*64)

    class Version(str):
        pass

    guard = R["NoModelImports"](); sys.meta_path.insert(0, guard)
    try:
        records, verified = R["publish_saved"](saved, tmp_path / "recovery", execution=dict(backend="cpu", dtype="float32", torch=Version("tiny-version")),
            source_identity=dict(fixture="1"*64), run_id="job-recovery-fixture", source_config=dict(seed=37, prompts=[dict(id="tiny")]))
    finally:
        sys.meta_path.remove(guard)
    assert len(records["rows"]) == len(records["atlas_ids"]) == 4 and verified["verified"]["verified"]
    assert verified["complete_scientific_counts"]["physical_entries"] == 25
    assert verified["random_control_seed_accounts"]["hypotheses"]["source_config_callback"]["bit_changed_total"] == 0
    account = tmp_path / "recovery/original-account"
    assert (account / "summary.json").read_bytes() == (saved / "summary.json").read_bytes()
    assert (account / "transaction/seal.json").read_bytes() == (saved / "transaction/seal.json").read_bytes()
    assert not list((tmp_path / "recovery").rglob("*.npy*")) and not any(name in sys.modules for name in R["BLOCKED"])
    assert R["capture_inventory"](saved, job_id="job-fixture", request_sha256="1"*64) == before
    catalog = list((account / "catalog").glob("*.json"))
    assert type(json.loads(catalog[0].read_text())["binding"]["execution"]["torch"]) is str


def test_public_callback_insertion_order_differs_from_saved_canonical_reload(tmp_path):
    from mdb.forecast_transaction import ForecastTransaction, load_forecast_tree

    txroot = tmp_path / "metadata-transaction"; observed = {}
    with ForecastTransaction(txroot, model_identity="metadata-only-order-fixture", input_identity="no-model",
            source_identity="no-model", frontier=dict(kind="architectural")) as tx:
        tx.frontier("prefixes", {"zeta": dict(value=1), "alpha": dict(value=2)})

        def predict(context):
            observed["callback"] = list(context.frontiers["prefixes"])
            return dict(predictions=dict(order=observed["callback"]))

        tx.predict("cohort", predict); tx.seal()
        tx.execute("metadata", lambda context: dict(order=list(context.frontiers["prefixes"])))
    value = R["read"](txroot / "frontiers/prefixes.json")["value"]
    assert observed["callback"] == ["zeta", "alpha"]
    assert list(load_forecast_tree(txroot, value)) == ["alpha", "zeta"]


def test_multiprompt_receipts_join_unique_ids_without_imposing_map_order():
    prefixes = dict(alpha=None, zeta=None)
    prediction = dict(prompts={name: dict(selection=dict(rows=rows)) for name, rows in
        (("zeta", ["0,0", "1,0"]), ("alpha", ["0,0", "0,1"]))})
    pre = dict(prefixes=2, physical_entries=2, final_MLP_entries=0, denied_preseal_entries=0)
    counts = dict(denied_preseal_entries=0, physical_entries=50)
    journal = [dict(number=index, kind="genuine_prefix_complete", prompt=name, final_suffix_entries=0) for index, name in enumerate(("zeta", "alpha"))]
    journal.append(dict(number=2, kind="global_forecasts_sealed", seal_sha256="1"*64, pretarget_counts=pre))
    journal.extend(dict(number=index+3, kind="prompt_targets_complete", prompt=name,
        selected_future_rows=prediction["prompts"][name]["selection"]["rows"], counts=counts) for index, name in enumerate(("zeta", "alpha")))

    def check(value):
        return R["check_guard_journal"](value, prefixes, prediction, dict(pretarget_counts=pre), dict(counts=counts), "1"*64)

    observed = check(journal)
    assert observed["guard_capture_order"] == observed["target_completion_order"] == ["zeta", "alpha"]
    assert list(prefixes) == ["alpha", "zeta"]
    for mutation in ("duplicate", "selection", "preseal-target", "preseal-entry"):
        changed = copy.deepcopy(journal)
        if mutation == "duplicate":
            changed[4]["prompt"] = "zeta"
        elif mutation == "selection":
            changed[3]["selected_future_rows"] = ["0,0", "1,1"]
        elif mutation == "preseal-target":
            changed[1]["kind"] = "prompt_targets_complete"
        else:
            changed[0]["final_suffix_entries"] = 1
        with pytest.raises(ValueError):
            check(changed)


def test_complete_random_inputs_resolve_order_and_keep_mismatch_as_diagnostic():
    import numpy as np

    actual_order = ["zeta", "alpha"]; prompts = {}
    for index, name in enumerate(actual_order):
        direction = np.random.default_rng(38+index).standard_normal(8); direction /= np.linalg.norm(direction)
        prompts[name] = dict(parity=dict(random=dict(requested_raw_norm=2.3, positive_input=(2.3*direction).astype(np.float32))))
    config = dict(seed=37, prompts=[dict(id=name) for name in actual_order])
    verification = dict(callback_decode_order=actual_order, serialized_frontier_order=["alpha", "zeta"])
    account = R["random_seed_accounts"](dict(prompts=prompts), config, verification)
    assert account["hypotheses"]["source_config_callback"]["bit_changed_total"] == 0
    assert account["hypotheses"]["serialized_metadata_reload"]["bit_changed_total"] > 0
    prompts["alpha"]["parity"]["random"]["positive_input"].view(np.uint32)[0] ^= 1
    changed = R["random_seed_accounts"](dict(prompts=prompts), config, verification)
    assert changed["hypotheses"]["source_config_callback"]["bit_changed_total"] == 1
    assert "mismatch observed" in changed["instrument_status"] and changed["numerical_gate"] is False


@pytest.mark.parametrize("where", ["preseal", "posttarget", "journal"])
def test_corrupted_retained_science_is_refused_without_any_model(saved, where):
    summary = R["read"](saved / "summary.json"); tx = saved / "transaction"
    if where == "journal":
        rows = [json.loads(line) for line in (saved / "timeline.jsonl").read_text().splitlines()]
        rows[1]["pretarget_counts"]["final_MLP_entries"] = 1
        (saved / "timeline.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    elif where == "preseal":
        path = tx / "forecasts/cohort.json"; path.write_bytes(path.read_bytes()+b" ")
    else:
        pre = R["read"](tx / "seal.json")["files"]
        path = next(path for path in (tx / "arrays").glob("*.npyb") if path.relative_to(tx).as_posix() not in pre)
        value = bytearray(path.read_bytes()); value[-1] ^= 1; path.write_bytes(value)
    with pytest.raises((ValueError, RuntimeError)):
        R["transaction_accounts"](saved, summary)
    assert not any(name in sys.modules for name in R["BLOCKED"])


def test_cli_request_uses_default_identity_and_relative_metadata_paths(tmp_path, monkeypatch, capsys):
    config = R["make_config"]({key: "1"*64 for key in R["PIN_KEYS"]} | dict(output_scratch=str(tmp_path / "scratch")))
    source = Path(__file__).parents[1] / "qualification"
    names = ["recover_qwen_spectral_readout.py", *R["HELPERS"]]
    for name in names:
        shutil.copyfile(source / name, tmp_path / name)
    request = dict(config=config, source_hashes={name: R["sha"](tmp_path / name) for name in names})
    request["request_sha256"] = R["request_identity"](request)
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    request_path = tmp_path / "request.json"; request_path.write_text(json.dumps(request))
    monkeypatch.setenv("SATURN_DEBUG_REQUEST", str(request_path)); monkeypatch.setenv("MRUN_JOB_ID", "job-recovery-fixture")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    calls = []

    def saved_only(config, output, **context):
        calls.append(dict(config=config, output=str(output), **context))
        return dict(output_root=str(output), job_id=context["job_id"], request_sha256=context["request_sha256"], model_calls=0)

    monkeypatch.setitem(R["worker"].__globals__, "recover", saved_only)
    R["worker"](config_path)
    assert len(calls) == 1 and Path(calls[0]["config"]["source_inventory"]) == tmp_path / "source-inventory.json"
    assert calls[0]["own_request"] == request and "MDB_DIRECTORY_READY " in capsys.readouterr().out
    changed = copy.deepcopy(request); changed["config"]["model_calls"] = 1; request_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="CPU request/config identity differs"):
        R["worker"](config_path)
