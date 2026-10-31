"""Composition mechanics for default native screening and durable publications."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mdb.discovery_batch import BatchProfile, BatchRow, FrozenNativeBatch
from mdb.discovery_runtime import ReplayForward
from mdb.discovery_storage import load_frontier, save_frontier
from mdb.forecast_transaction import ForecastTransaction, verify_transaction
from mdb.forward_recording import record_forward


def test_default_batch_cut_and_forecast_bundles_compose_without_changing_outputs(tmp_path):
    class Native(torch.nn.Module):
        def forward(self, value):
            return {"out": (value + 2) * 3}

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = Native().eval().requires_grad_(False)
        profile = BatchProfile("tiny-stack", "add-mul", ((1, 2),), {"out": (1, 2)})
        batch = FrozenNativeBatch(model, profile)
        rows = [BatchRow(f"lane_{i}", "input_00", "main", 0,
                        (torch.tensor([[i, i + 1]], dtype=torch.float32),)) for i in range(4)]
        screened = batch.screen(rows)
        assert not screened.errors and len(screened.attempts) == 1
        assert screened.attempts[0]["physical_batch_size"] == 4

        # Selected component work retains its B1 geometry and uses scalar replay.
        selected = rows[2]
        capture = record_forward(model, selected.roots, model_identity="tiny-stack",
                                 execution={"backend": "cpu-fp32"}, input_identity=selected.binding())
        runtime = ReplayForward(capture)
        while runtime.trace["events"][runtime.pc]["operator"] != "aten.mul.Tensor":
            runtime.advance_one()
        incoming = runtime.read(runtime.trace["events"][runtime.pc]["inputs"][0]).numpy()
        saved = save_frontier(runtime, tmp_path / "store")
        assert saved["storage_layout"] == "bundle-v1"
        restored = load_frontier(tmp_path / "store", saved["cut_id"])
        assert restored.pc == runtime.pc and not restored.durable_origin["prefix_replayed"]

        def suffix(context):
            while not restored.complete:
                restored.advance_one()
            return restored.outputs()

        with ForecastTransaction(tmp_path / "forecast", model_identity="tiny-stack",
                input_identity=selected.binding(), source_identity={"trace": capture.trace["fingerprint"]},
                frontier={"kind": "architectural"}) as tx:
            tx.frontier("selected", incoming)
            record = tx.predict("suffix", lambda context: {
                "predictions": {"out": context.frontiers["selected"] * 3}})
            assert record["predictions"]["items"]["out"]["storage"] == "bundle-v1"
            assert restored.pc == runtime.pc  # Predicting did not run the suffix.
            tx.seal()
            tx.execute("suffix", suffix)
            np.testing.assert_array_equal(tx.prediction("suffix")["out"], tx.observation("suffix")["out"])
            np.testing.assert_array_equal(tx.observation("suffix")["out"], screened.rows[2].outputs["out"].numpy())
        assert verify_transaction(tx.root, expected_seal_sha256=tx.report()["seal_sha256"],
                                  expected_event_head=tx.report()["event_head"])["verified"]
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("storage", ["files-v1", "bundle-v1"])
@pytest.mark.parametrize("reader", ["context_window_campaign", "summarize_raw_isa_sequence"])
def test_campaign_readers_decode_both_layouts_without_changing_row_values(tmp_path, storage, reader):
    source = Path(__file__).resolve().parents[1] / "qualification" / f"{reader}.py"
    decoder = runpy.run_path(str(source))["decode_tree"]
    values = {"first": np.arange(4, dtype=np.float32), "second": np.arange(5, dtype=np.float64),
              "before": {"flags": [0, 1], "offset": 7}}
    with ForecastTransaction(tmp_path / "tx", model_identity="tiny", input_identity="input",
            source_identity="reader-mechanics", frontier={"kind": "raw-input"}, array_storage=storage) as tx:
        record = tx.predict("row", lambda context: {"predictions": values})
        node = record["predictions"]
        decoded = decoder(node, tx.root) if reader == "context_window_campaign" else decoder(tx.root, node)
        np.testing.assert_array_equal(decoded["first"], values["first"])
        np.testing.assert_array_equal(decoded["second"], values["second"])
        assert decoded["before"] == values["before"]
        decoded.update(evidence_role="prior")
        tx.seal()


@pytest.mark.parametrize("campaign", ["context_window_campaign", "raw_isa_sequence_campaign"])
@pytest.mark.parametrize("requested_storage", [None, "files-v1"])
def test_campaign_cli_selects_bundles_by_default_and_preserves_legacy_override(
        tmp_path, monkeypatch, capsys, campaign, requested_storage):
    source = Path(__file__).resolve().parents[1] / "qualification" / f"{campaign}.py"
    main = runpy.run_path(str(source))["main"]
    staged = []

    def stage(*args):
        staged.append(args)
        return {"staged": True}

    monkeypatch.setitem(main.__globals__, "stage", stage)
    arguments = [campaign, "--run-dir", str(tmp_path / "run"), "--isa-root", str(tmp_path / "isa"),
                 "--plan", str(tmp_path / "plan.json")]
    if campaign == "context_window_campaign":
        arguments += ["--prior-root", str(tmp_path / "prior")]
    if requested_storage is not None:
        arguments += ["--array-storage", requested_storage]
    monkeypatch.setattr(sys, "argv", arguments)
    main()
    assert len(staged) == 1
    index = 5 if campaign == "context_window_campaign" else 3
    assert staged[0][index] == (requested_storage or "bundle-v1")
    assert '"staged": true' in capsys.readouterr().out
