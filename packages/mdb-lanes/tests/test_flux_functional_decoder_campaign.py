"""Installed-package decoder replication mechanics; synthetic weights only."""
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
from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from safetensors.torch import save_file
from saturn_pub.core import StateCut
from saturn_pub.store import LocalStore

from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/flux_functional_decoder_campaign.py"
W = runpy.run_path(str(PATH))


def recipes(root, model):
    root.mkdir(parents=True)
    path = root / "vae.safetensors"
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, path)
    with path.open("rb") as stream:
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
    rows = []
    for name, value in model.state_dict().items():
        if not (name.startswith(("decoder.", "post_quant_conv.")) or name in ("bn.running_mean", "bn.running_var")):
            continue
        bits = value.contiguous().view(torch.uint16).numpy().astype("<u2")
        raw = bits.tobytes()
        stream = io.BytesIO()
        np.save(stream, bits, allow_pickle=False)
        encoded = stream.getvalue()
        rows.append(dict(path="parameters/" + hashlib.sha256(name.encode() + encoded).hexdigest() + ".npy", bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest(),
            npy_prefix_b64=base64.b64encode(encoded[:-len(raw)]).decode(), source=dict(file=path.name, file_sha256=W["sha"](path), tensor=name, dtype="BF16", shape=list(bits.shape), data_offsets=header[name]["data_offsets"], payload_sha256=hashlib.sha256(raw).hexdigest())))
    return dict(schema="mdb-parameter-artifacts-v1", artifacts=rows)


def fixture(root):
    import mdb_lanes.parameter_artifacts as api

    root.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng():
        torch.manual_seed(1041)
        vae = AutoencoderKLFlux2(block_out_channels=(4, 4), decoder_block_out_channels=(4, 4),
            down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
            layers_per_block=1, latent_channels=1, norm_num_groups=1, sample_size=8,
            use_post_quant_conv=True, mid_block_add_attention=True).eval().to(torch.bfloat16).requires_grad_(False)
        vae.bn.running_mean.copy_(torch.tensor([-.25, .125, .375, -.125], dtype=torch.bfloat16))
        vae.bn.running_var.copy_(torch.tensor([.5, 1.5, .75, 2.], dtype=torch.bfloat16))
        latent = torch.randn(1, 4, 4, dtype=torch.bfloat16).transpose(1, 2)
        state = dict(latent=latent, conditioning=torch.randn(1, 3, 6, dtype=torch.bfloat16),
            img_ids=torch.tensor([[[0, y, x, 0] for y in range(2) for x in range(2)]], dtype=torch.int64),
            txt_ids=torch.tensor([[[0, 0, 0, x] for x in range(3)]], dtype=torch.int64),
            timesteps=torch.tensor([1000., 750., 500., 250.]), sigmas=torch.tensor([1., .75, .5, .25, 0.]),
            step=4, phase="project", cursor=0, image=None, text=None, context=None, noise_prediction=None)
    inputs = {}
    for case, dose in zip(W["CASES"], (0., .125, -.0625, .046875), strict=True):
        z = torch.empty_strided(latent.shape, latent.stride(), dtype=latent.dtype)
        z.copy_((latent.float() + dose * torch.arange(16).reshape(1, 4, 4)).to(torch.bfloat16))
        execution = dict(backend="cpu", dtype="torch.bfloat16", component_pass=dict(id="mdb-flux-functional-remaining-v1", lane="forecast", case=case, synthetic_fixture=True))
        cut = StateCut("tiny-original-closure", execution, "synthetic-final-S4", None, {**state, "latent": z},
            dict(family="flux2-klein", logical_step=4, phase="project", operator="project", index=0, local_clock=0, edge="before"))
        LocalStore(root / "inputs/typed-store").save(cut)
        inputs[case] = dict(store="typed-store", cut=cut.fingerprint)
    paths = dict(vae_config=root / "vae-config.json", parameter_manifest=root / "parameter-artifacts.json", retrieval_script=root / "retrieve_weights.py")
    vae.to_json_file(paths["vae_config"])
    paths["parameter_manifest"].write_text(json.dumps(recipes(root / "checkpoint", vae)))
    shutil.copyfile(api.__file__, paths["retrieval_script"])
    config = dict(profile="tiny-cpu", device="cpu", cpu_threads=1, max_wall_seconds=120, chunk_bytes=256, witness_values=4, height=8, width=8,
        input_root=str(root / "inputs"), inputs=inputs, expected_source_model_identity="tiny-original-closure", source_reduction_policy=True,
        parameter_root=str(root / "checkpoint"), output=str(root / "result"), source_provenance=dict(scope="synthetic fixture only"),
        helper_source_sha256={name: W["sha"](HERE / "qualification" / name) for name in W["HELPERS"]},
        upstream_source_sha256={"pipeline_flux2_klein.py": W["sha"](inspect.getfile(Flux2KleinPipeline)), "flux2_image_processor.py": W["sha"](inspect.getfile(Flux2ImageProcessor))})
    for key, path in paths.items():
        config[key], config[key + "_sha256"] = str(path), W["sha"](path)
    config["input_files"] = {p.relative_to(Path(config["input_root"])).as_posix(): dict(sha256=W["sha"](p), bytes=p.stat().st_size) for p in Path(config["input_root"]).rglob("*") if p.is_file()}
    return config


def arrays(output, lane, row):
    cut = LocalStore(output / (lane + "-store")).load(row["cut"])
    cut.verify()
    assert cut.payload["chunk_keys"] == row["chunk_keys"]
    return torch.cat([cut.payload[key] for key in row["chunk_keys"]]).reshape(row["descriptor"]["shape"])


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    config = fixture(tmp_path_factory.mktemp("decoder"))
    return config, W["run"](config)


def test_complete_macro_chunks_primitive_hashes_and_same_latent_consumers(completed):
    config, summary = completed
    output = Path(config["output"])
    assert summary["totals"]["functional_decode_calls"] == summary["totals"]["native_decode_calls"] == 4
    assert summary["totals"]["forecast_module_entries"] == summary["totals"]["diagnostic_module_entries"] == 0
    assert summary["totals"]["diagnostic_prefix_calls"] == 0
    assert summary["first_micro_difference"] == summary["first_macro_difference"] == dict.fromkeys(W["CASES"])
    assert summary["counts"]["forecast"] == summary["counts"]["native"]
    verify_transaction(output / "forecast-transaction", expected_seal_sha256=summary["forecast_seal"]["seal_sha256"])
    events = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    seal = next(i for i, row in enumerate(events) if row["kind"] == "cohort_globally_sealed")
    assert events[seal]["native_decode_calls"] == 0
    assert sum(row["kind"] == "forecast_case_completed" for row in events[:seal]) == 4
    assert all(i > seal for i, row in enumerate(events) if row["kind"].startswith("native_"))
    forecasts = json.loads((output / "forecast-macros.json").read_text())
    natives = json.loads((output / "native-macros.json").read_text())
    for case in W["CASES"]:
        assert len(forecasts[case]) == 10
        for name, row in forecasts[case].items():
            assert torch.equal(arrays(output, "forecast", row), arrays(output, "native", natives[case][name]))
            assert all(p.stat().st_size <= config["chunk_bytes"] + 256 for p in (output / "forecast-store/blobs").glob("*"))
        assert forecasts[case]["latent_unpatched"]["descriptor"]["sha256"] != forecasts[case]["decoder_input"]["descriptor"]["sha256"]
        assert (output / "forecast-images" / (case + ".png")).read_bytes() == (output / "native-images" / (case + ".png")).read_bytes()
    assert len(json.loads((output / "signed-responses.json").read_text())) == 3
    for key in ("vae_config", "parameter_manifest", "retrieval_script"):
        assert (output / Path(config[key]).name).read_bytes() == Path(config[key]).read_bytes()
    assert not (output / "parameters").exists()


def test_module_forecast_guard_and_cleanup(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    original = W["run"].__globals__["helpers"]
    hooks, sdpa = dict(torch.nn.modules.module._global_forward_pre_hooks), torch.nn.functional.scaled_dot_product_attention
    def changed(config):
        graph, *others = original(config)
        graph["functional_decode"] = lambda model, argument, **kwargs: model.post_quant_conv(argument)
        return [graph, *others]
    monkeypatch.setitem(W["run"].__globals__, "helpers", changed)
    with pytest.raises(ValueError, match="functional decoder entered native"):
        W["run"](config)
    assert dict(torch.nn.modules.module._global_forward_pre_hooks) == hooks
    assert torch.nn.functional.scaled_dot_product_attention is sdpa


@pytest.mark.parametrize("fault", ["outside-window", "tampered-prediction"])
def test_global_seal_prevents_target_entry(tmp_path, monkeypatch, fault):
    config = fixture(tmp_path)
    original = ForecastTransaction.execute
    hooks = dict(torch.nn.modules.module._global_forward_pre_hooks)
    def execute(self, name, callback, **kwargs):
        if fault == "outside-window":
            return callback(None)
        def tamper(context):
            with (Path(config["output"]) / "forecast-images/sham.png").open("ab") as stream:
                stream.write(b"x")
            return callback(context)
        return original(self, name, tamper, **kwargs)
    monkeypatch.setattr(ForecastTransaction, "execute", execute)
    with pytest.raises(ValueError, match="outside globally sealed execution" if fault == "outside-window" else "signature changed"):
        W["run"](config)
    assert dict(torch.nn.modules.module._global_forward_pre_hooks) == hooks
    events = [json.loads(line) for line in (Path(config["output"]) / "timeline.jsonl").read_text().splitlines()]
    assert not any(row["kind"].startswith("native_") for row in events)


@pytest.mark.parametrize("replay_matches", [True, False])
def test_first_micro_miss_retains_native_and_bounded_prefix(tmp_path, monkeypatch, replay_matches):
    config = fixture(tmp_path)
    original = torch.nn.Conv2d.forward
    change = dict(armed=False, diagnostic=False)
    def progress(row):
        change["armed"] = row["kind"] == "native_decode_entry" and row["case"] == "HH"
        if row["kind"] == "diagnostic_prefix_started":
            change["diagnostic"] = True
    def altered(self, value):
        result = original(self, value)
        if change["armed"]:
            change["armed"] = False
            result = result + .125
        return result
    monkeypatch.setattr(torch.nn.Conv2d, "forward", altered)
    original_helpers = W["run"].__globals__["helpers"]
    def changed_helpers(config):
        graph, *others = original_helpers(config)
        functional = graph["functional_decode"]
        def changed(model, argument, *, observe, **kwargs):
            def observe_changed(name, kind, inputs, value):
                return observe(name, kind, inputs, value + .25 if change["diagnostic"] and not replay_matches else value)
            return functional(model, argument, observe=observe_changed, **kwargs)
        graph["functional_decode"] = changed
        return [graph, *others]
    monkeypatch.setitem(W["run"].__globals__, "helpers", changed_helpers)
    summary = W["run"](config, progress=progress)
    assert summary["totals"]["native_decode_calls"] == 4
    assert summary["totals"]["diagnostic_prefix_calls"] == 1
    diagnostic = summary["diagnostics"]["HH"]
    assert diagnostic["status"] == ("hash-matched-reconstruction" if replay_matches else "replay-differs-from-original-sealed-hash")
    assert ("forecast_full" in diagnostic) == replay_matches
    assert diagnostic["native_full"] == summary["first_micro_difference"]["HH"]["native_full"]
    if replay_matches:
        assert diagnostic["full_pair_metric"]["rms"] > 0
    assert summary["first_micro_difference"]["HH"]["id"] == "post_quant_conv.0"
    events = [json.loads(line) for line in (Path(config["output"]) / "timeline.jsonl").read_text().splitlines()]
    assert max(i for i, row in enumerate(events) if row["kind"] == "native_case_completed") < next(i for i, row in enumerate(events) if row["kind"] == "diagnostic_prefix_started")


def test_equal_values_wrong_source_lineage_rejected(tmp_path):
    config = fixture(tmp_path)
    store = LocalStore(Path(config["input_root"]) / "typed-store")
    old = store.load(config["inputs"]["sham"]["cut"])
    execution = {**old.execution, "component_pass": {**old.execution["component_pass"], "lane": "native"}}
    wrong = StateCut(old.model_identity, execution, old.boundary, old.parent, old.payload, old.execution_point)
    store.save(wrong)
    config["inputs"]["sham"]["cut"] = wrong.fingerprint
    config["input_files"] = {name: dict(sha256=W["sha"](Path(config["input_root"]) / name), bytes=(Path(config["input_root"]) / name).stat().st_size) for name in W["source_closure"](Path(config["input_root"]), config["inputs"])}
    with pytest.raises(ValueError, match="predicted source lineage"):
        W["run"](config)


def test_worker_cli_exports_closed_raw_directory(tmp_path):
    config = fixture(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, "-I", str(PATH), "--config", str(path)], cwd=tmp_path, env={**os.environ, "MRUN_JOB_ID": "tiny-decoder-cli"}, text=True, capture_output=True, check=True)
    assert "MDB_DIRECTORY_READY" in result.stdout
    summary = json.loads((Path(config["output"]) / "summary.json").read_text())
    assert summary["job_id"] == "tiny-decoder-cli"
    assert (Path(config["output"]) / "files.json").exists()


def test_optional_catalog_failure_preserves_core_exports_and_final_checks(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    def broken(*args):
        output = Path(config["output"])
        assert all((output / name).is_file() for name in ("source.json", "native-contract.json", "summary.json"))
        assert json.loads((output / "summary.json").read_text())["final_resident_parameter_check"] == "verified-before-optional-catalog"
        raise ValueError("intentional optional reporter failure")
    monkeypatch.setitem(W["run"].__globals__, "endpoint_comparison", broken)
    summary = W["run"](config)
    assert summary["mechanics_status"] == "complete" and summary["optional_catalog"]["status"] == "reporting-error"
    assert summary["optional_catalog"]["primary_evidence_preserved"] is True


def test_endpoint_catalog_declares_more_than_default_million_elements():
    from mdb_lanes.equation_comparison import compare_equations

    arrays = dict(decoded=np.zeros((1, 3, 512, 512), np.float32), pixels=np.zeros((1, 512, 512, 3), np.uint8))
    with pytest.raises(ValueError, match="element budget"):
        compare_equations(arrays, arrays)
    report = W["endpoint_comparison"](arrays, arrays)
    assert sum(v["residual"]["elements"] for v in report["fields"].values()) == 1_572_864
