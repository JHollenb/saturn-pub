"""Installed-package CPU qualification of sealed parallel-writer interventions."""
from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers.models.transformers import transformer_flux2 as native
from mdb_lanes.forecast_transaction import ForecastTransaction, ForecastTransactionError, load_forecast_tree, verify_transaction
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/flux_single_writer_square.py"
W = runpy.run_path(str(PATH))
T = runpy.run_path(str(HERE / "tests/test_flux_single_stack_trace.py"))
M = T["M"]


def fixture(root):
    config, _, _ = T["fixture"](root)
    config["block_count"] = 2
    T["C"]["run"](config)
    source = Path(config["output"])
    inputs = {}
    for role in W["ROLES"]:
        metadata = json.loads((source / f"boundaries/{role}/00.json").read_text())
        inputs[role] = dict(store="typed-store", cut=metadata["cut"], entry_metadata=f"boundaries/{role}/00.json",
                            exit_metadata=f"boundaries/{role}/01.json", final_metadata=f"boundaries/{role}/02.json")
    result = dict(config, selected_block=0, suffix_block=1, inputs=inputs, input_root=str(source), output=str(root / "square"),
                  helper_source_sha256={name: M["sha"](HERE / "qualification" / name) for name in W["HELPERS"]})
    result["input_files"] = {file.relative_to(source).as_posix(): dict(sha256=M["sha"](file), bytes=file.stat().st_size)
                             for file in source.rglob("*") if file.is_file()}
    return result


def decode_operand(output, record):
    assert record["encoding"] == "bf16-little-endian-uint16-npy-v1"
    path = output / record["path"]
    assert M["sha"](path) == record["sha256"]
    bits = np.load(path, allow_pickle=False)
    return torch.from_numpy(bits.astype(np.uint16)).view(torch.bfloat16)


def test_eight_sealed_native_corners_complete_state_and_actual_operands(tmp_path):
    config = fixture(tmp_path)
    original_dispatch = native.dispatch_attention_fn
    summary = W["run"](config)
    assert native.dispatch_attention_fn is original_dispatch
    assert summary["counts"] == W["contract"]()["counts"]
    assert summary["complete_typed_cuts"] == 19
    assert summary["intervention_typed_cuts"] == 16 and summary["source_typed_cuts"] == 3
    assert summary["act_receipts"] == summary["session_continue_receipts"] == 0
    assert summary["vae_calls"] == summary["encodes"] == summary["noise_draws"] == 0
    assert all(all(value.values()) for value in summary["prediction_exact"].values())
    assert all(row[stream]["exact"] for row in summary["source_replay"].values() for stream in ("image", "text"))
    output = Path(config["output"])
    verification = verify_transaction(output / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    assert verification["verified"]
    seal = json.loads((output / "transaction/seal.json").read_text())
    assert len(seal["forecasts"]) == 8 and seal["pre_release_work"]["executions"] == 0
    timeline = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    sealed = next(index for index, row in enumerate(timeline) if row["kind"] == "globally_sealed")
    assert timeline[sealed]["target_entries_at_seal"] == 0
    assert [row["stage"] for row in timeline[:sealed] if row["kind"] == "native_entry"] == ["source"] * 3
    assert [row["stage"] for row in timeline[sealed:] if row["kind"] == "native_entry"] == ["target"] * 8 + ["suffix"] * 8
    frontier = json.loads((output / "transaction/frontiers/observed_sources.json").read_text())
    sources = load_forecast_tree(output / "transaction", frontier["value"])
    source_calls = json.loads((output / "source-calls.json").read_text())
    store = LocalStore(output / "typed-store")
    for role in W["ROLES"]:
        observed = decode_operand(output, source_calls[role]["observed_concat"])
        assert torch.equal(observed, torch.cat([torch.from_numpy(sources[role][key].copy()).to(torch.bfloat16) for key in ("H", "M")], dim=-1))
        assert source_calls[role]["hook_return"] == "None"
        source_cut = store.load(source_calls[role]["output"]["cut"])
        source_cut.verify()
        traced_row = json.loads((Path(config["input_root"]) / config["inputs"][role]["exit_metadata"]).read_text())
        traced_cut = LocalStore(Path(config["input_root"]) / config["inputs"][role]["store"]).load(traced_row["cut"])
        assert all(torch.equal(source_cut.payload[stream], traced_cut.payload[stream]) for stream in ("image", "text"))
    for case in W["cases"]():
        name, role = case["id"], case["recipient"]
        row = json.loads((output / "cases" / (name + ".local.json")).read_text())
        final = json.loads((output / "cases" / (name + ".final.json")).read_text())
        parent = LocalStore(Path(config["input_root"]) / "typed-store").load(config["inputs"][role]["cut"])
        local, continued = store.load(row["local"]["cut"]), store.load(final["final"]["cut"])
        local.verify(); continued.verify()
        assert local.parent == parent.fingerprint and continued.parent == local.fingerprint
        assert local.execution["component_trace"]["id"] == W["EXECUTION_ID"]
        assert local.payload["cursor"] == 1 and continued.payload["cursor"] == 2 and continued.payload["phase"] == "readout"
        for cut in (local, continued):
            for key in parent.payload:
                if key not in ("image", "text", "phase", "cursor"):
                    assert describe(cut.payload[key]) == describe(parent.payload[key])
        receiving, consumed = (decode_operand(output, row["operands"][key]) for key in ("receiving", "consumed"))
        own = torch.cat([torch.from_numpy(sources[role][key].copy()).to(torch.bfloat16) for key in ("H", "M")], dim=-1)
        expected = torch.cat([torch.from_numpy(sources[case[key]][key].copy()).to(torch.bfloat16) for key in ("H", "M")], dim=-1)
        assert torch.equal(receiving, own) and torch.equal(consumed, expected)
        if name.endswith("11"):
            assert all(row["own_replay"][stream]["exact"] and final["own_final_replay"][stream]["exact"] for stream in ("image", "text"))
    assert not (output / "arrays").exists()
    assert all(not (output / row["path"]).exists() for row in json.loads((output / "parameter-artifacts.json").read_text())["artifacts"])
    assert len(list((output / "operands").glob("*.npy"))) <= 7
    accounts = json.loads((output / "accounts.json").read_text())
    for stage in ("local", "final"):
        for stream in ("image", "text"):
            assert accounts["signed_joint_restoration"][stage][stream] == accounts["growth"][stage + ".00"][stream]["excess_growth"] - accounts["growth"][stage + ".11"][stream]["excess_growth"]


def test_wrong_named_bank_refused_and_global_native_state_restored(tmp_path):
    config = fixture(tmp_path)
    binding = json.loads(Path(config["binding"]).read_text())
    binding["final_blocks"]["0"]["parameters"]["qkv_mlp_weight"]["parameter_tensor"] = "wrong.weight"
    Path(config["binding"]).write_text(json.dumps(binding))
    config["binding_sha256"] = M["sha"](config["binding"])
    original = native.dispatch_attention_fn
    debug, threads = torch.get_deterministic_debug_mode(), torch.get_num_threads()
    with pytest.raises(ValueError, match="parameter identity"):
        W["run"](config)
    assert native.dispatch_attention_fn is original
    assert torch.get_deterministic_debug_mode() == debug and torch.get_num_threads() == threads
    assert not (Path(config["output"]) / "transaction").exists()


def test_trace_replay_miss_is_observation_and_source_clock_is_integrity(tmp_path):
    config = fixture(tmp_path)
    path = Path(config["input_root"]) / config["inputs"]["HH"]["exit_metadata"]
    record = json.loads(path.read_text())
    record["fields"]["image"]["descriptor"]["bf16_le_sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    config["input_files"][config["inputs"]["HH"]["exit_metadata"]] = dict(sha256=M["sha"](path), bytes=path.stat().st_size)
    result = W["run"](config)
    assert not result["source_replay"]["HH"]["image"]["exact"]
    assert result["mechanics_status"] == "complete" and result["counts"]["native_blocks"] == 19
    wrong = dict(config, suffix_block=2)
    with pytest.raises(ValueError, match="only final native single"):
        W["validate_config"](wrong)


@pytest.mark.parametrize("failure", ["preseal-entry", "corrupted-seal", "wrong-receiving"])
def test_target_guard_failure_modes_and_cleanup(tmp_path, monkeypatch, failure):
    config = fixture(tmp_path)
    original_execute = ForecastTransaction.execute
    original_dispatch = native.dispatch_attention_fn
    debug, threads = torch.get_deterministic_debug_mode(), torch.get_num_threads()
    def execute(self, name, callback, **kwargs):
        if failure == "preseal-entry":
            return callback(None)
        def guarded(context):
            if failure == "corrupted-seal":
                with (self.root / "seal.json").open("ab") as stream:
                    stream.write(b"\n")
                return callback(context)
            source = context.frontiers["observed_sources"]["HL"]
            coherent = torch.cat([torch.from_numpy(source[key].copy()).to(torch.bfloat16) for key in ("H", "M")], dim=-1)
            expected = context.frontiers["observed_sources"]["HH"]
            assert not torch.equal(coherent, torch.cat([torch.from_numpy(expected[key].copy()).to(torch.bfloat16) for key in ("H", "M")], dim=-1)), "wrong-source instrument must differ at first HH recipient"
            def corrupt(module, args):
                if isinstance(module, torch.nn.Linear) and module.in_features == coherent.shape[-1] and module.out_features == config["module_config"]["dim"]:
                    return (coherent,)
            handle = torch.nn.modules.module.register_module_forward_pre_hook(corrupt)
            try:
                return callback(context)
            finally:
                handle.remove()
        return original_execute(self, name, guarded, **kwargs)
    monkeypatch.setattr(ForecastTransaction, "execute", execute)
    error, message = ((ForecastTransactionError, "global forecast seal changed") if failure == "corrupted-seal" else
                      (ValueError, "target entered before sealed execute" if failure == "preseal-entry" else "recipient operand fidelity differs"))
    with pytest.raises(error, match=message):
        W["run"](config)
    assert native.dispatch_attention_fn is original_dispatch
    assert torch.get_deterministic_debug_mode() == debug and torch.get_num_threads() == threads
    output = Path(config["output"])
    rows = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    targets = [row for row in rows if row["kind"] == "native_entry" and row["stage"] == "target"]
    assert len(targets) == (1 if failure == "wrong-receiving" else 0)
    assert not (output / "cases/HH.00.local.json").exists()


def test_actual_worker_cli_finalizes_and_joins_source_sidecar(tmp_path):
    config = fixture(tmp_path)
    config["request_sha256"] = "1" * 64
    path = tmp_path / "writer-config.json"
    path.write_text(json.dumps(config))
    process = subprocess.run([sys.executable, "-I", "-B", str(PATH), "--config", str(path)], cwd=tmp_path,
        env={**os.environ, "MRUN_JOB_ID": "tiny-writer-cli"}, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    ready = json.loads(next(line.split(" ", 1)[1] for line in process.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(config["output"])
    assert ready["files_manifest_sha256"] == M["sha"](output / "files.json") and ready["expanded_bytes"] < 4 << 20
    source = json.loads((output / "source.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert source["job_id"] == summary["job_id"] == "tiny-writer-cli"
    assert source["request_sha256"] == summary["request_sha256"] == "1" * 64
    contract = json.loads((output / "native-contract.json").read_text())
    assert contract["inference_mode"] is True and contract["autocast"] is False
    verification = verify_transaction(output / "transaction", expected_event_head=summary["transaction"]["event_head"])
    assert verification["phase"] == "closed"
