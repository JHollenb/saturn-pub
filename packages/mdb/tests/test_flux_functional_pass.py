"""Installed-package mechanics for the complete four-case own-state pass."""
from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
import os
import runpy
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from diffusers.models.transformers import transformer_flux2 as native
from mdb.forecast_transaction import ForecastTransaction, verify_transaction
from safetensors.torch import save_file
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.core import StateCut
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/flux_functional_pass.py"
W = runpy.run_path(str(PATH))
M = runpy.run_path(str(HERE / "qualification/flux_image_tail_microscope.py"))


def fixture(root):
    import mdb.parameter_artifacts as parameter_api

    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = dict(in_channels=4, num_layers=2, num_single_layers=2, attention_head_dim=4,
                num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
                axes_dims_rope=[2, 2], guidance_embeds=False, mlp_ratio=2.0)
    with torch.random.fork_rng():
        torch.manual_seed(71)
        model = Flux2Transformer2DModel(**spec).eval().to(torch.bfloat16).requires_grad_(False)
        latent = torch.randn(1, 4, 4, dtype=torch.bfloat16).transpose(1, 2)
        state = dict(latent=latent, conditioning=torch.randn(1, 3, 6, dtype=torch.bfloat16),
                     img_ids=torch.arange(16, dtype=torch.int64).reshape(1, 4, 4),
                     txt_ids=torch.arange(12, dtype=torch.int64).reshape(1, 3, 4),
                     timesteps=torch.tensor([1000., 670., 330.]), sigmas=torch.tensor([1., .67, .33, 0.]),
                     step=1, phase="project", cursor=0, image=None, text=None, context=None, noise_prediction=None)
    scheduler = FlowMatchEulerDiscreteScheduler()
    adapter = Flux2KleinAdapter(model, scheduler, granularity="operation", device="cpu")
    input_root, historical_root = root / "inputs", root / "verification-only"
    sources, historical = {}, {}
    for role, change in (("sham", 0.), ("HH", .125), ("HL", -.0625)):
        z = torch.empty_strided(latent.shape, latent.stride(), dtype=latent.dtype)
        z.copy_(latent + change * torch.arange(16).reshape(1, 4, 4).to(torch.bfloat16))
        payload = {**state, "latent": z}
        cut = StateCut("tiny-original-closure", {"backend": "cpu", "dtype": "torch.bfloat16"}, adapter.boundary(payload), None, payload, adapter.point(payload).to_dict())
        LocalStore(input_root / "typed-store").save(cut)
        sources[role] = dict(store="typed-store", cut=cut.fingerprint)
        current = payload
        with torch.inference_mode():
            for _ in range(7):
                current = adapter.advance(current)
        final = StateCut(cut.model_identity, cut.execution, adapter.boundary(current), cut.fingerprint, current, adapter.point(current).to_dict())
        LocalStore(historical_root / "typed-store").save(final)
        historical[role] = dict(store="typed-store", cut=final.fingerprint)
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    path = checkpoint / "weights.safetensors"
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, path)
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
    artifacts = []
    for name, value in model.state_dict().items():
        bits = value.contiguous().view(torch.uint16).numpy().astype("<u2")
        raw = bits.tobytes()
        buffer = io.BytesIO()
        np.save(buffer, bits, allow_pickle=False)
        encoded = buffer.getvalue()
        artifacts.append(dict(path="parameters/" + hashlib.sha256(name.encode() + encoded).hexdigest() + ".npy", bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest(),
            npy_prefix_b64=base64.b64encode(encoded[:-len(raw)]).decode(), source=dict(file=path.name, file_sha256=M["sha"](path), tensor=name, dtype="BF16", shape=list(bits.shape), data_offsets=header[name]["data_offsets"], payload_sha256=hashlib.sha256(raw).hexdigest())))
    files = {"transformer_config": root / "transformer-config.json", "scheduler_config": root / "scheduler-config.json",
             "parameter_manifest": root / "parameter-artifacts.json", "binding": root / "binding.json", "retrieval_script": root / "retrieve_weights.py"}
    M["save"](files["transformer_config"], spec)
    scheduler.to_json_file(files["scheduler_config"])
    M["save"](files["parameter_manifest"], dict(schema="mdb-parameter-artifacts-v1", artifacts=artifacts))
    M["save"](files["binding"], dict(native_names=sorted(model.state_dict())))
    shutil.copyfile(parameter_api.__file__, files["retrieval_script"])
    config = dict(profile="tiny-cpu", device="cpu", cpu_threads=1, max_wall_seconds=120, witness_values=4,
                  input_root=str(input_root), inputs=sources, historical_root=str(historical_root), historical=historical,
                  expected_source_model_identity="tiny-original-closure", source_reduction_policy=True,
                  parameter_root=str(checkpoint), output=str(root / "result"), native_source_sha256=M["sha"](inspect.getfile(native)),
                  helper_source_sha256={name: M["sha"](HERE / "qualification" / name) for name in W["HELPERS"]})
    for key, file in files.items():
        config[key], config[key + "_sha256"] = str(file), M["sha"](file)
    for root_key, files_key in (("input_root", "input_files"), ("historical_root", "historical_files")):
        base = Path(config[root_key])
        config[files_key] = {p.relative_to(base).as_posix(): dict(sha256=M["sha"](p), bytes=p.stat().st_size) for p in base.rglob("*") if p.is_file()}
    return config


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    config = fixture(tmp_path_factory.mktemp("functional-pass"))
    return config, W["run"](config)


def test_complete_sealed_four_case_pass_and_own_state_consumers(completed):
    config, summary = completed
    output = Path(config["output"])
    declared = W["contract"](config)
    assert summary["totals"]["model_loads"] == 1 and summary["totals"]["historical_cut_reads"] == 3
    assert summary["complete_forecast_cuts"] == summary["complete_native_cuts"] == 28
    assert summary["complete_consumer_cuts"] == 8
    for lane in ("forecast", "native", "consumers"):
        assert all(summary["counts"][lane][key] == value for key, value in declared[lane].items())
    assert summary["counts"]["forecast"]["module_entries"] == 0
    assert summary["act_receipts"] == summary["session_continue_receipts"] == 0
    for name in ("input", "forecast"):
        verify_transaction(output / (name + "-transaction"), expected_seal_sha256=summary[name + "_seal"]["seal_sha256"])
    events = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    input_seal = next(i for i, row in enumerate(events) if row["kind"] == "inputs_globally_sealed")
    cohort_seal = next(i for i, row in enumerate(events) if row["kind"] == "cohort_globally_sealed")
    assert input_seal < cohort_seal
    assert all(row["lane"] == "forecast" for row in events[:cohort_seal] if row["kind"] == "boundary_completed")
    assert all(i > cohort_seal for i, row in enumerate(events) if row["kind"] in ("native_entry", "native_scheduler_entry", "historical_cut_read"))
    assert all(row["native_transitions_completed"] == 28 for row in events if row["kind"] == "historical_cut_read")
    comparisons = json.loads((output / "comparisons.json").read_text())
    for case in W["CASES"]:
        assert all(v["exact"] for row in comparisons["boundaries"][case] for v in row["fields"].values())
        assert all(v["exact"] for row in comparisons["boundaries"][case] for v in row["primitive_ports"].values())
        predicted_rows = json.loads((output / "forecast-rows" / (case + ".json")).read_text())
        native_rows = json.loads((output / "native-rows" / (case + ".json")).read_text())
        for lane, rows in (("prediction", predicted_rows), ("native", native_rows)):
            previous = json.loads((output / "source.json").read_text())["input_records"][case]["cut"]
            for row in rows:
                cut = LocalStore(output / (lane + "-store")).load(row["state"]["cut"])
                cut.verify()
                assert cut.parent == previous
                assert cut.execution["component_pass"]["lane"] == ("forecast" if lane == "prediction" else lane)
                previous = cut.fingerprint
            assert (cut.payload["step"], cut.payload["phase"], cut.payload["cursor"]) == (2, "project", 0)
            assert all(cut.payload[key] is None for key in ("image", "text", "context", "noise_prediction"))
        consumers = comparisons["consumers"][case]
        for origin in ("predicted", "native_reference"):
            assert consumers[origin]["predicted"] is False
            parent = LocalStore(output / ("prediction-store" if origin == "predicted" else "native-store")).load(consumers[origin]["input_cut"])
            assert describe(parent.payload) == consumers[origin]["consumed_state"]
        assert all(value["exact"] for value in consumers["comparison"].values())
    assert all(value["latent"]["exact"] for value in comparisons["historical"].values())
    accounts = json.loads((output / "response-accounts.json").read_text())
    assert accounts["ordering_prediction"] is None
    assert all(value["exact"] for row in accounts["boundaries"] for fields in row["responses"].values() for value in fields.values())
    assert not (output / "parameters").exists()


def test_fixed_input_derivation_is_sealed_with_native_layout(completed):
    config, _ = completed
    output = Path(config["output"])
    source = json.loads((output / "source.json").read_text())
    store = LocalStore(output / "input-store")
    sham, hh, new = (store.load(source["input_records"][case]["cut"]) for case in ("sham", "HH", "new-3over8"))
    z, h, actual = sham.payload["latent"], hh.payload["latent"], new.payload["latent"]
    assert torch.equal(actual, (z.float() + (3 / 8) * (h.float() - z.float())).to(torch.bfloat16))
    assert actual.stride() == z.stride() and actual.dtype == torch.bfloat16
    assert new.parent == sham.fingerprint and new.execution["input_derivation"]["coefficient"] == [3, 8]
    assert new.execution["input_derivation"]["act_receipt"] is None
    geometry = json.loads((output / "input-geometry.json").read_text())
    assert geometry["coefficient"] == [3, 8] and geometry["changed_from_sham"] > 0


def test_guard_forbids_real_module_forward_in_functional_forecast(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    original_helpers = W["run"].__globals__["helpers"]
    def helpers(config):
        graph, m, recipes = original_helpers(config)
        def forbidden(model, latent, *args):
            return model.x_embedder(latent)
        graph["functional_project"] = forbidden
        # Preserve the globals discovery contract while replacing only the
        # public prediction entry with an actual forbidden module invocation.
        forbidden.__wrapped__ = original_helpers(config)[0]["functional_project"].__wrapped__
        return graph, m, recipes
    monkeypatch.setitem(W["run"].__globals__, "helpers", helpers)
    original = (native.dispatch_attention_fn, native.apply_rotary_emb, FlowMatchEulerDiscreteScheduler.step)
    with pytest.raises(ValueError, match="original native module.forward"):
        W["run"](config)
    assert (native.dispatch_attention_fn, native.apply_rotary_emb, FlowMatchEulerDiscreteScheduler.step) == original
    assert not (Path(config["output"]) / "forecast-transaction").exists()


@pytest.mark.parametrize("fault", ["outside-window", "tampered-prediction-file"])
def test_target_seal_and_external_file_guards(tmp_path, monkeypatch, fault):
    config = fixture(tmp_path)
    original_execute = ForecastTransaction.execute
    def execute(self, name, callback, **kwargs):
        if fault == "outside-window":
            return callback(None)
        def tamper(context):
            path = next((Path(config["output"]) / "prediction-store/blobs").glob("*.safetensors"))
            with path.open("ab") as stream:
                stream.write(b"x")
            return callback(context)
        return original_execute(self, name, tamper, **kwargs)
    monkeypatch.setattr(ForecastTransaction, "execute", execute)
    original = (native.dispatch_attention_fn, native.apply_rotary_emb, FlowMatchEulerDiscreteScheduler.step)
    with pytest.raises(ValueError, match="outside globally sealed execution" if fault == "outside-window" else "sealed external evidence signature changed"):
        W["run"](config)
    assert (native.dispatch_attention_fn, native.apply_rotary_emb, FlowMatchEulerDiscreteScheduler.step) == original
    events = [json.loads(line) for line in (Path(config["output"]) / "timeline.jsonl").read_text().splitlines()]
    assert not any(row["kind"] == "native_entry" for row in events)


def test_source_whitelist_excludes_future_cut_and_historical_misses_are_diagnostic(tmp_path):
    config = fixture(tmp_path)
    wrong = {**config, "input_files": {**config["input_files"], "native/future.json": dict(sha256="0" * 64, bytes=0)}}
    with pytest.raises(ValueError, match="exactly complete declared cuts"):
        W["validate_config"](wrong)
    old_store = LocalStore(Path(config["historical_root"]) / "typed-store")
    old = old_store.load(config["historical"]["HH"]["cut"])
    state = old.payload
    state["latent"] = state["latent"] + torch.tensor(.125, dtype=torch.bfloat16)
    changed = StateCut(old.model_identity, old.execution, old.boundary, old.parent, state, old.execution_point)
    replacement = tmp_path / "changed-history"
    for role in W["ROLES"]:
        cut = changed if role == "HH" else old_store.load(config["historical"][role]["cut"])
        LocalStore(replacement / "typed-store").save(cut)
        config["historical"][role]["cut"] = cut.fingerprint
    config["historical_root"] = str(replacement)
    config["historical_files"] = {p.relative_to(replacement).as_posix(): dict(sha256=M["sha"](p), bytes=p.stat().st_size) for p in replacement.rglob("*") if p.is_file()}
    summary = W["run"](config)
    assert summary["mechanics_status"] == "complete"
    historical = json.loads((Path(config["output"]) / "comparisons.json").read_text())["historical"]
    assert historical["HH"]["latent"]["exact"] is False


def test_actual_cli_finalizer(tmp_path):
    config = fixture(tmp_path)
    config["request_sha256"] = "1" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH), "--config", str(path)], cwd=tmp_path,
        env={**os.environ, "MRUN_JOB_ID": "tiny-functional-pass"}, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    ready = json.loads(next(line.split(" ", 1)[1] for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(config["output"])
    assert ready["files_manifest_sha256"] == M["sha"](output / "files.json")
    source, summary = (json.loads((output / name).read_text()) for name in ("source.json", "summary.json"))
    assert source["job_id"] == summary["job_id"] == "tiny-functional-pass"
    assert source["request_sha256"] == summary["request_sha256"] == "1" * 64
    verify_transaction(output / "forecast-transaction", expected_event_head=summary["transaction"]["event_head"])
