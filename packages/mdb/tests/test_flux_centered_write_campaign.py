"""Bounded model-free payload, public Act, source and bootstrap mechanics."""
from __future__ import annotations

import ast
import hashlib
import json
import runpy
from pathlib import Path

import pytest
import torch
from saturn_pub.core import Adapter, Session
from saturn_pub.store import LocalStore
from saturn_pub.values import describe
from torch.nn import functional as F

from mdb.forecast_transaction import (
    ForecastTransaction,
    ForecastTransactionError,
    verify_transaction,
)

HERE = Path(__file__).parents[1]
FILE = HERE / "qualification/flux_centered_write_campaign.py"
M = runpy.run_path(str(FILE))


def source_config(tmp_path):
    fixture = HERE / "fixtures/centered-source"
    parent = M["read"](fixture / "parent-config.json")
    config = M["make_config"](parent)
    selection = tmp_path / "selection.json"
    M["save"](selection, dict(contexts=list(M["CONTEXTS"])))
    for key in M["path_keys"]():
        path = selection if key == "selection_protocol" else fixture / Path(config[key]).name
        config[key] = str(path); config[key + "_sha256"] = M["sha"](path)
    return config


def tensor_fixture():
    gate = torch.tensor([[[.75, -.5, 1., .25]]], dtype=torch.bfloat16)
    top = torch.tensor([[[1., -2., .5, 3.], [3., 1., -.5, 2.], [2., 4., 1., -.5], [4., -1., 3., .5]]], dtype=torch.bfloat16)
    return gate, dict(top=top, bottom=top.roll(1, -1), random=top.flip(-1))


def test_common_uniform_centered_norms_and_signed_output():
    gate, evens = tensor_fixture(); payloads, report = M["centered_payloads"](gate, evens)
    assert set(payloads) == {"top-positive", "top-negative", "bottom-positive", "random-positive"}
    assert torch.equal(payloads["top-negative"].view(torch.int16), (-payloads["top-positive"]).view(torch.int16))
    targets = [row["intended_uniform_row_norm"]["values"] for row in report["arms"].values()]
    assert all(value == targets[0] for value in targets) and len(set(targets[0])) == 1
    assert len(set(report["source_norms"]["top"]["values"])) > 1
    assert all(row["zero_source_rows"] == 0 for row in report["arms"].values())
    assert all(row["ideal_to_cast_rms"] > 0 for row in report["arms"].values())


def test_zero_centered_source_rows_are_retained_without_repair():
    gate, evens = tensor_fixture(); evens["bottom"] = torch.zeros_like(evens["bottom"])
    payloads, report = M["centered_payloads"](gate, evens)
    assert torch.count_nonzero(payloads["bottom-positive"]) == 0
    assert report["arms"]["bottom-positive"]["zero_source_rows"] == 4
    assert report["target_norm"] > 0


class ArithmeticAdapter(Adapter):
    model_identity = "centered-additive-fixture"
    execution = dict(family="arithmetic", version=1)

    def addresses(self, state):
        return ("image", "text")

    def validate(self, state):
        assert set(state) == {"image", "text", "clock"}

    def boundary(self, state):
        return str(state["clock"])

    def advance(self, state):
        return dict(state, clock=state["clock"] + 1)


def test_additive_acts_consume_own_lane_state_and_realized_rounding(tmp_path):
    gate, evens = tensor_fixture(); payloads, _ = M["centered_payloads"](gate, evens)
    store = LocalStore(tmp_path / "store"); results = []
    for offset in (0., 16.):
        session = Session(ArithmeticAdapter(), dict(image=torch.full((1,4,4), offset, dtype=torch.bfloat16), text=torch.ones(1,2,4,dtype=torch.bfloat16), clock=0))
        common = session.capture(retain=False); store.save(common)
        branch = session.fork(common); actual_start = branch.capture(retain=False); store.save(actual_start)
        assert actual_start.parent == common.fingerprint and actual_start.fingerprint != common.fingerprint
        before, after, receipt = M["apply_slot_edit"](branch, "image", payloads["top-positive"], mode="add")
        store.save(before); store.save(after)
        assert receipt.to_dict()["parent"] == actual_start.fingerprint
        assert torch.equal(after.payload["image"], before.payload["image"] + payloads["top-positive"])
        assert describe(after.payload["text"]) == describe(before.payload["text"])
        account = M["displacement_account"](before.payload["image"], after.payload["image"])
        assert len(account["centered_row_norm"]["values"]) == 4
        step = branch.continue_(); assert step.to_dict()["parent"] == after.fingerprint
        results.append(after.payload["image"])
    assert not torch.equal(results[0], results[1])


def test_original_equation_functions_preserved_from_frozen_reference():
    old = ast.parse((HERE / "qualification/flux_gauge_spectral_campaign.py").read_text())
    new = ast.parse(FILE.read_text())
    functions = lambda tree: {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("step_equations", "selected_joint0", "directional_inputs", "even_write", "frontend", "native_joint0", "native_repeat_decode"):
        assert functions(old)[name] == functions(new)[name]


def test_actual_absolute_source_validation_and_inherited_vector_bits(tmp_path):
    config = source_config(tmp_path)
    assert M["validate_config"](config)["image_futures"] == 12
    target = tmp_path / "inherited"; target.mkdir()
    values, report = M["inherited_directions"](config, target, M["helpers"](), device="cpu")
    assert set(values) == {"top", "bottom", "random"}
    assert report["work"]["operator_calls"] == 0 and report["inherited_source"]["metadata_sha256"] == M["DIRECTION_METADATA_SHA256"]
    original = M["read"](config["direction_source_metadata"])
    assert all(report["vectors"][name]["content_sha256"] == original["vectors"][name]["content_sha256"] for name in values)
    config["dose"] = .5
    with pytest.raises(ValueError, match="frozen campaign"):
        M["validate_config"](config)


def test_real_global_twelve_future_transaction_before_targets(tmp_path):
    calls = []
    tx = ForecastTransaction(tmp_path / "tx", model_identity="tiny", input_identity="a"*64, source_identity="b"*64,
        frontier=dict(kind="architectural"), array_storage="files-v1", budgets=dict(max_predictions=1,max_executions=1,max_bytes=1<<20))
    tx.frontier("native-prefixes", dict(roots=2))
    tx.predict("all-futures", lambda _ctx: dict(predictions=dict(targets_at_seal=0, paths={str(i):{} for i in range(12)})))
    tx.seal(); tx.execute("original-native-targets", lambda _ctx: calls.append("after-seal") or dict(targets=12)); tx.close()
    assert calls == ["after-seal"] and verify_transaction(tmp_path / "tx")["phase"] == "closed"
    hostile = ForecastTransaction(tmp_path / "premature", model_identity="tiny", input_identity="a"*64, source_identity="b"*64,
        frontier=dict(kind="architectural"), array_storage="files-v1")
    with pytest.raises(ForecastTransactionError):
        hostile.execute("too-soon", lambda _ctx: calls.append("invalid"))
    assert calls == ["after-seal"]


def test_module_guard_refuses_entry_and_releases_global_hooks():
    identity = torch.nn.Identity(); phase = dict(kind="forecast-denoiser"); released = dict(native=False); observed = {}
    linear, sdpa = F.linear, F.scaled_dot_product_attention
    with pytest.raises(ValueError, match="module forward forbidden"):
        with M["instrument"]([identity], phase, released, observed):
            identity(torch.ones(1))
    assert F.linear is linear and F.scaled_dot_product_attention is sdpa and not identity._forward_pre_hooks


def test_request_bound_actual_worker_finalizer_model_free_stub(tmp_path, monkeypatch, capfd):
    config = source_config(tmp_path); config["output_scratch"] = str(tmp_path / "outputs")
    path = tmp_path / "config.json"; path.write_text(json.dumps(config))
    source = tmp_path / FILE.name; source.write_bytes(FILE.read_bytes())
    request = dict(config=config, source_hashes={FILE.name:M["sha"](source)})
    request["request_sha256"] = hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    req = tmp_path / "request.json"; req.write_text(json.dumps(request))
    monkeypatch.setenv("SATURN_DEBUG_REQUEST",str(req)); monkeypatch.setenv("MRUN_JOB_ID","job-centered-mechanics")
    calls = []
    def fake_run(derived, progress=None):
        calls.append(derived); output = Path(derived["output"]); output.mkdir(parents=True)
        summary = dict(job_id=derived["job_id"],request_sha256=derived["request_sha256"],scope="model-free bootstrap only")
        M["save"](output / "summary.json",summary); return summary
    monkeypatch.setitem(M["worker"].__globals__,"run",fake_run)
    ready = M["worker"](path)
    assert len(calls) == 1 and ready["request_sha256"] == request["request_sha256"] and "MDB_DIRECTORY_READY " in capfd.readouterr().out
    source.write_text("changed")
    with pytest.raises(ValueError,match="executed source changed"):
        M["worker"](path)
    assert len(calls) == 1
