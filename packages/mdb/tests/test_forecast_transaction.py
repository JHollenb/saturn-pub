from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

import mdb.forecast_transaction as forecast_module
from mdb.forecast_transaction import (
    ForecastPayload,
    ForecastTransaction,
    ForecastTransactionError,
    main,
    verify_transaction,
)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = torch.nn.Linear(2, 2)
        self.suffix = torch.nn.Linear(2, 1)
        self.suffix_calls = 0
        with torch.no_grad():
            self.prefix.weight.copy_(torch.eye(2))
            self.prefix.bias.fill_(1)
            self.suffix.weight.fill_(2)
            self.suffix.bias.fill_(3)

    def forward(self, value):
        return self.suffix(self.prefix(value))


def transaction(tmp_path, *, model=None, prefix=False, **kwargs):
    return ForecastTransaction(tmp_path / "tx", model=model,
        model_identity={"checkpoint": "tiny-fixed"}, input_identity={"case": "raw"},
        source_identity={"file_sha256": "declared-source"},
        allowed_prefix={"stop_before": "suffix", "observe": ["prefix"]} if prefix else None,
        denied_modules=("suffix",) if model is not None else (),
        frontier=None if prefix else {"kind": "raw-input"}, **kwargs)


def simple_forecast(context):
    return {"predictions": {"value": np.array([1.25], dtype=np.float64)},
            "payloads": {"U": ForecastPayload(np.array([1.123456789], dtype=np.float64),
                dtype="float32", provenance={"source": "analytical prediction"})}}


def test_bounded_script_captures_real_prefix_globally_seals_then_executes(tmp_path):
    model = Tiny().eval()
    entered = []
    hook = model.suffix.register_forward_pre_hook(lambda module, args: entered.append(True))
    with transaction(tmp_path, model=model, prefix=True) as tx:
        def predict(context):
            assert entered == []
            value = context.frontiers["a"]["boundary"]["args"][0]
            return {"predictions": {"steps": {"0": value @ context.weights["suffix.weight"].T
                                                           + context.weights["suffix.bias"]}},
                    "payloads": {"U": ForecastPayload(value)}}

        result = tx.run_script([
            {"op": "capture", "name": "a", "args": (torch.tensor([[2., 4.]]),)},
            {"op": "capture", "name": "b", "args": (torch.tensor([[8., 9.]]),)},
            {"op": "predict", "name": "trajectory", "callback": predict},
            {"op": "predict", "name": "other", "callback": simple_forecast},
            {"op": "seal"},
            {"op": "execute", "name": "actual", "callback": lambda context: context.model(torch.tensor([[2., 4.]]))},
        ])
        assert entered == [True]
        assert np.array_equal(tx.prediction("trajectory", path=("steps", "0")), tx.observation("actual"))
        assert result[-2]["forecasts"] == ["trajectory", "other"]
        events = tx.report()["events"]
        names = [event["kind"] for event in events]
        assert names.index("globally_sealed") < names.index("native_execution_released") < names.index("observation_persisted")
        assert tx.report()["work"]["captures"] == 2
        assert tx.report()["work"]["native_calls"] == 3
        assert verify_transaction(tx.root, expected_seal_sha256=result[-2]["seal_sha256"])["verified"]
    hook.remove()
    assert not model._forward_pre_hooks and not model.prefix._forward_pre_hooks


def test_raw_architectural_frontier_supports_global_trajectory_without_capture(tmp_path):
    tx = transaction(tmp_path, weights={"W": np.eye(2, dtype=np.float32)})
    tx.frontier("inputs", {"registers": np.array([1, 2], dtype=np.uint64),
                            "weights": np.eye(2, dtype=np.float32), "steps": 3})
    tx.predict("trajectory", lambda context: {"predictions": {"steps": {str(i):
        context.frontiers["inputs"]["registers"] + i for i in range(3)}}})
    tx.seal()
    assert tx.report()["work"]["native_calls"] == 0
    np.testing.assert_array_equal(tx.prediction("trajectory", path=("steps", "2")), [3, 4])
    tx.execute("conditional", lambda context: {"supplied": context.frontiers["inputs"]["steps"],
                                                "forecast": context.predictions["trajectory"]})
    assert tx.observation("conditional")["supplied"] == 3
    tx.close()


def test_callback_views_are_immutable_and_weights_are_fresh_owned_copies(tmp_path):
    model = Tiny().eval()
    tx = transaction(tmp_path, model=model)
    tx.frontier("inputs", np.ones(2, dtype=np.float32))
    seen = []

    def predict(context):
        seen.append(context.weights["prefix.weight"])
        for value in (context.weights["prefix.weight"], context.frontiers["inputs"]):
            with pytest.raises(ValueError):
                value.setflags(write=True)
            with pytest.raises(ValueError):
                value.flat[0] = 99
        return {"predictions": context.weights["prefix.weight"]}

    tx.predict("first", predict)
    tx.predict("second", predict)
    assert seen[0] is not seen[1] and not np.shares_memory(seen[0], seen[1])
    np.testing.assert_array_equal(tx.prediction("second"), np.eye(2))
    tx.close()


def test_payload_cast_clones_bytes_and_returns_owned_tensor_without_aliasing(tmp_path):
    tx = transaction(tmp_path)
    returned = np.array([1.123456789], dtype=np.float64)
    tx.predict("p", lambda context: {"predictions": returned, "payloads": {
        "U": ForecastPayload(returned, provenance={"coordinate": "mixer-output"})}})
    returned[:] = 77
    record = json.loads((tx.root / "forecasts/p.json").read_text())
    assert record["payloads"]["U"]["cast"] == {"from": "float64", "to": "float32", "policy": "explicit-astype"}
    assert record["payloads"]["U"]["provenance"]["coordinate"] == "mixer-output"
    assert tx.payload("p", "U").dtype == np.float32
    assert tx.prediction("p")[0] != 77
    tensor = tx.payload("p", "U", as_tensor=True)
    tensor.zero_()
    assert tx.payload("p", "U")[0] != 0
    tx.close()


@pytest.mark.parametrize("direct_submodule", [False, True])
def test_native_attempt_is_latched_even_when_prediction_callback_catches_it(tmp_path, direct_submodule):
    model = Tiny().eval()
    tx = transaction(tmp_path, model=model)

    def predict(context):
        try:
            (model.prefix if direct_submodule else model)(torch.ones(1, 2))
        except ForecastTransactionError:
            pass
        return {"predictions": np.zeros(1)}

    with pytest.raises(ForecastTransactionError, match="latched"):
        tx.predict("bad", predict)
    assert tx.phase == "failed" and tx.report()["work"]["denied_attempts"] == 1
    assert not (tx.root / "seal.json").exists()
    assert any(event["kind"] == "violation" for event in tx.report()["events"])
    assert not model._forward_pre_hooks and not model.prefix._forward_pre_hooks


def test_background_native_attempt_is_rejected_and_join_does_not_deadlock(tmp_path):
    model = Tiny().eval()
    tx = transaction(tmp_path, model=model)
    errors = []

    def predict(context):
        def background():
            try:
                model.suffix(torch.ones(1, 2))
            except ForecastTransactionError as error:
                errors.append(str(error))
        thread = threading.Thread(target=background)
        thread.start()
        thread.join(2)
        assert not thread.is_alive()
        return {"predictions": np.zeros(1)}

    with pytest.raises(ForecastTransactionError, match="latched"):
        tx.predict("bad", predict)
    assert errors and tx.phase == "failed"


@pytest.mark.parametrize("operation", ["nested", "background"])
def test_reentrant_or_background_transaction_actions_are_latched_without_deadlock(tmp_path, operation):
    tx = transaction(tmp_path)

    def predict(context):
        def call():
            try:
                tx.predict("inner", simple_forecast)
            except ForecastTransactionError:
                pass
        if operation == "nested":
            call()
        else:
            thread = threading.Thread(target=call)
            thread.start()
            thread.join(2)
            assert not thread.is_alive()
        return {"predictions": np.zeros(1)}

    with pytest.raises(ForecastTransactionError, match="latched"):
        tx.predict("bad", predict)


def test_release_verifies_all_payload_bytes_and_cleans_hooks_on_tamper(tmp_path):
    model = Tiny().eval()
    tx = transaction(tmp_path, model=model)
    tx.predict("p", simple_forecast)
    tx.seal()
    payload = tx._predictions["p"]["payloads"]["U"]["value"]["path"]
    (tx.root / payload).write_bytes(b"tampered")
    executed = []
    with pytest.raises(ForecastTransactionError, match="sealed artifact changed"):
        tx.execute("bad", lambda context: executed.append(True))
    assert not executed and tx.phase == "failed" and not model._forward_pre_hooks


def test_seal_tamper_refused_and_no_overwrite_of_completed_forecast(tmp_path):
    tx = transaction(tmp_path)
    tx.predict("p", simple_forecast)
    original = (tx.root / "forecasts/p.json").read_bytes()
    with pytest.raises(ForecastTransactionError, match="new name"):
        tx.predict("p", simple_forecast)
    assert (tx.root / "forecasts/p.json").read_bytes() == original
    other = ForecastTransaction(tmp_path / "second", model_identity="m", input_identity="i",
        source_identity="s", frontier={"kind": "empty"})
    other.predict("p", simple_forecast)
    other.seal()
    (other.root / "seal.json").write_text("{}")
    with pytest.raises(ForecastTransactionError, match="seal changed"):
        other.release()


def test_observation_is_durable_before_optional_analysis_failure(tmp_path):
    tx = transaction(tmp_path)
    tx.predict("p", simple_forecast)
    tx.seal()

    def analysis(value):
        assert (tx.root / "observations/a.json").exists()
        assert tx.report()["events"][-1]["kind"] == "observation_persisted"
        np.testing.assert_array_equal(value["actual"], [4, 5])
        raise ValueError("analysis instrument failed")

    with pytest.raises(ValueError, match="instrument"):
        tx.execute("a", lambda context: {"actual": np.array([4, 5])}, analysis=analysis)
    assert tx.phase == "failed" and tx.report()["observations"] == ["a"]
    np.testing.assert_array_equal(tx.observation("a")["actual"], [4, 5])


def test_finite_prediction_and_native_call_budgets_preserve_work(tmp_path):
    tx = transaction(tmp_path, budgets={"max_predictions": 1})
    tx.predict("one", simple_forecast)
    with pytest.raises(ForecastTransactionError, match="prediction budget"):
        tx.predict("two", simple_forecast)
    assert tx.report()["work"]["predictions"] == 1 and (tx.root / "forecasts/one.json").exists()
    model = Tiny().eval()
    other = ForecastTransaction(tmp_path / "native", model=model, model_identity="m", input_identity="i",
        source_identity="s", frontier={"kind": "empty"}, budgets={"max_native_calls": 1})
    other.predict("p", simple_forecast)
    other.seal()
    other.execute("first", lambda context: model(torch.ones(1, 2)))
    with pytest.raises(ForecastTransactionError, match="native-call budget"):
        other.execute("second", lambda context: model(torch.ones(1, 2)))
    assert other.report()["work"]["native_calls"] == 1


def test_script_is_prevalidated_and_does_not_import_callback_strings(tmp_path):
    tx = transaction(tmp_path)
    with pytest.raises(ForecastTransactionError, match="callables"):
        tx.run_script([{"op": "frontier", "name": "unused", "value": 1},
                       {"op": "predict", "name": "bad", "callback": "package:callable"}])
    assert tx.report()["frontiers"] == []
    with pytest.raises(ForecastTransactionError, match="arguments"):
        tx.run_script([{"op": "seal", "unexpected": True}])
    tx.close()


def test_existing_root_cannot_be_overwritten_and_optional_capture_is_explicit(tmp_path):
    tx = transaction(tmp_path)
    with pytest.raises(FileExistsError):
        transaction(tmp_path)
    tx.close()
    with pytest.raises(ForecastTransactionError, match="explicit frontier"):
        ForecastTransaction(tmp_path / "unbound", model_identity="m", input_identity="i", source_identity="s")


def test_offline_verify_chain_cli_and_no_framework_import(tmp_path, capsys):
    tx = transaction(tmp_path)
    tx.predict("p", simple_forecast)
    sealed = tx.seal()
    tx.execute("actual", lambda context: np.array([3]))
    tx.close()
    result = verify_transaction(tx.root, expected_seal_sha256=sealed["seal_sha256"],
                                 expected_event_head=tx.report()["event_head"])
    assert result["verified"] and result["phase"] == "closed"
    assert main(["verify", str(tx.root), "--expected-seal", sealed["seal_sha256"]]) == 0
    assert json.loads(capsys.readouterr().out)["verified"]
    code = "import importlib.util,sys; p=sys.argv[1]; s=importlib.util.spec_from_file_location('forecast',p); m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m); assert 'torch' not in sys.modules and 'numpy' not in sys.modules;m.verify_transaction(sys.argv[2]);assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    source = Path(forecast_module.__file__).resolve()
    completed = subprocess.run([sys.executable, "-I", "-c", code, str(source), str(tx.root)],
                               capture_output=True, text=True, timeout=20, env=dict(os.environ))
    assert completed.returncode == 0, completed.stderr
    events = sorted((tx.root / "events").glob("*.json"))
    altered = json.loads(events[-1].read_text())
    altered["kind"] = "forged"
    events[-1].write_text(json.dumps(altered))
    with pytest.raises(ForecastTransactionError, match="receipt binding"):
        verify_transaction(tx.root)
