"""Synthetic installed-package mechanics for remaining passes and real decoders."""
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
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from diffusers.models.transformers import transformer_flux2 as native
from safetensors.torch import save_file
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.core import StateCut
from saturn_pub.store import LocalStore

from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/flux_functional_remaining.py"
W = runpy.run_path(str(PATH))
M = runpy.run_path(str(HERE / "qualification/flux_image_tail_microscope.py"))


def recipes(root, model, filename, selected):
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, path)
    with path.open("rb") as stream:
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
    rows = []
    for name, value in model.state_dict().items():
        if not selected(name):
            continue
        bits = value.contiguous().view(torch.uint16).numpy().astype("<u2")
        raw = bits.tobytes()
        stream = io.BytesIO()
        np.save(stream, bits, allow_pickle=False)
        encoded = stream.getvalue()
        rows.append(dict(path="parameters/" + hashlib.sha256(name.encode() + encoded).hexdigest() + ".npy", bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest(),
            npy_prefix_b64=base64.b64encode(encoded[:-len(raw)]).decode(), source=dict(file=path.name, file_sha256=M["sha"](path), tensor=name, dtype="BF16", shape=list(bits.shape), data_offsets=header[name]["data_offsets"], payload_sha256=hashlib.sha256(raw).hexdigest())))
    return dict(schema="mdb-parameter-artifacts-v1", artifacts=rows)


def decoder_sources(vae):
    import diffusers.image_processor as image_processor
    import diffusers.models.attention_processor as attention
    import diffusers.models.resnet as resnet
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    return {"autoencoder_kl_flux2.py": M["sha"](inspect.getfile(AutoencoderKLFlux2)), "vae.py": M["sha"](inspect.getfile(type(vae.decoder))),
        "resnet.py": M["sha"](inspect.getfile(resnet)), "attention_processor.py": M["sha"](inspect.getfile(attention)),
        "image_processor.py": M["sha"](inspect.getfile(image_processor)), "flux2_image_processor.py": M["sha"](inspect.getfile(Flux2ImageProcessor)), "pipeline_flux2_klein.py": M["sha"](inspect.getfile(Flux2KleinPipeline))}


def fixture(root):
    import mdb_lanes.parameter_artifacts as api

    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = dict(in_channels=4, num_layers=2, num_single_layers=2, attention_head_dim=4, num_attention_heads=2,
                joint_attention_dim=6, timestep_guidance_channels=8, axes_dims_rope=[2, 2], guidance_embeds=False, mlp_ratio=2.)
    with torch.random.fork_rng():
        torch.manual_seed(137)
        model = Flux2Transformer2DModel(**spec).eval().to(torch.bfloat16).requires_grad_(False)
        vae = AutoencoderKLFlux2(block_out_channels=(4, 4), decoder_block_out_channels=(4, 4),
            down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
            layers_per_block=1, latent_channels=1, norm_num_groups=1, sample_size=8,
            use_post_quant_conv=True, mid_block_add_attention=True).eval().to(torch.bfloat16).requires_grad_(False)
        latent = torch.randn(1, 4, 4, dtype=torch.bfloat16).transpose(1, 2)
        state = dict(latent=latent, conditioning=torch.randn(1, 3, 6, dtype=torch.bfloat16),
            img_ids=torch.tensor([[[0, y, x, 0] for y in range(2) for x in range(2)]], dtype=torch.int64),
            txt_ids=torch.tensor([[[0, 0, 0, x] for x in range(3)]], dtype=torch.int64),
            timesteps=torch.tensor([1000., 750., 500., 250.]), sigmas=torch.tensor([1., .75, .5, .25, 0.]),
            step=2, phase="project", cursor=0, image=None, text=None, context=None, noise_prediction=None)
    scheduler = FlowMatchEulerDiscreteScheduler()
    adapter = Flux2KleinAdapter(model, scheduler, granularity="operation", device="cpu")
    inputs, references = {}, {}
    for case, dose in (("sham", 0.), ("HH", .125), ("HL", -.0625), ("new-3over8", .046875)):
        z = torch.empty_strided(latent.shape, latent.stride(), dtype=latent.dtype)
        z.copy_((latent.float() + dose * torch.arange(16).reshape(1, 4, 4)).to(torch.bfloat16))
        for lane, directory, records in (("forecast", "inputs", inputs), ("native", "references", references)):
            payload = {**state, "latent": z}
            execution = dict(backend="cpu", dtype="torch.bfloat16", component_pass=dict(id="mdb-flux-functional-pass-v1", lane=lane, case=case, synthetic_fixture=True))
            cut = StateCut("tiny-original-closure", execution, adapter.boundary(payload), None, payload, adapter.point(payload).to_dict())
            LocalStore(root / directory / "typed-store").save(cut)
            records[case] = dict(store="typed-store", cut=cut.fingerprint)
    checkpoint = root / "checkpoint"
    files = {key: root / (key + ".json") for key in ("transformer_config", "scheduler_config", "parameter_manifest", "binding", "vae_config", "vae_parameter_manifest")}
    M["save"](files["parameter_manifest"], recipes(checkpoint, model, "weights.safetensors", lambda name: True))
    M["save"](files["vae_parameter_manifest"], recipes(root / "vae-checkpoint", vae, "vae.safetensors", lambda name: name.startswith(("decoder.", "post_quant_conv.")) or name in ("bn.running_mean", "bn.running_var")))
    M["save"](files["transformer_config"], spec)
    scheduler.to_json_file(files["scheduler_config"])
    vae.to_json_file(files["vae_config"])
    M["save"](files["binding"], dict(native_names=sorted(model.state_dict())))
    files["retrieval_script"] = root / "retrieve_weights.py"
    shutil.copyfile(api.__file__, files["retrieval_script"])
    config = dict(profile="tiny-cpu", device="cpu", cpu_threads=1, max_wall_seconds=120, witness_values=4, height=8, width=8,
        input_root=str(root / "inputs"), inputs=inputs, reference_root=str(root / "references"), references=references,
        expected_source_model_identity="tiny-original-closure", source_reduction_policy=True, parameter_root=str(checkpoint),
        vae_parameter_root=str(root / "vae-checkpoint"), decoder_source_sha256=decoder_sources(vae),
        output=str(root / "result"), native_source_sha256=M["sha"](inspect.getfile(native)),
        source_provenance=dict(scope="synthetic fixture only; no archived scientific producer"),
        helper_source_sha256={name: M["sha"](HERE / "qualification" / name) for name in W["HELPERS"]})
    for key, path in files.items():
        config[key], config[key + "_sha256"] = str(path), M["sha"](path)
    for root_key, files_key in (("input_root", "input_files"), ("reference_root", "reference_files")):
        base = Path(config[root_key])
        config[files_key] = {p.relative_to(base).as_posix(): dict(sha256=M["sha"](p), bytes=p.stat().st_size) for p in base.rglob("*") if p.is_file()}
    return config


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    config = fixture(tmp_path_factory.mktemp("remaining"))
    return config, W["run"](config)


def test_two_remaining_passes_and_real_decoder_seals(completed):
    config, summary = completed
    output = Path(config["output"])
    assert summary["complete_forecast_cuts"] == summary["complete_native_cuts"] == 56
    assert summary["real_decoder_consumers"] == 8
    assert summary["counts"] == {lane: W["contract"](config)[lane] for lane in ("forecast", "native")}
    assert summary["totals"]["vae_calls"] == 8 and summary["totals"]["reference_cut_reads"] == 4
    assert summary["decoder_counts"]["predicted"] == summary["decoder_counts"]["native_reference"]
    assert summary["decoder_counts"]["predicted"]["sdpa"] == 4
    assert summary["counts"]["forecast"]["module_entries"] == 0
    for name, seal in (("input", "input_seal"), ("latent-forecast", "latent_seal"), ("forecast", "forecast_seal")):
        verify_transaction(output / (name + "-transaction"), expected_seal_sha256=summary[seal]["seal_sha256"])
    events = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    latent_seal = next(i for i, row in enumerate(events) if row["kind"] == "latents_globally_sealed")
    cohort_seal = next(i for i, row in enumerate(events) if row["kind"] == "cohort_globally_sealed")
    assert sum(row["kind"] == "real_decoder_completed" and row["origin"] == "predicted" for row in events[latent_seal:cohort_seal]) == 4
    assert all(i > cohort_seal for i, row in enumerate(events) if row["kind"] in ("native_entry", "native_scheduler_entry", "reference_cut_read"))
    assert all(row["native_future_transitions"] == 56 for row in events if row["kind"] == "real_decoder_completed" and row["origin"] == "native_reference")
    comparisons = json.loads((output / "comparisons.json").read_text())
    assert comparisons["first_covered_mismatch"] == dict.fromkeys(W["CASES"])
    for case in W["CASES"]:
        for lane, directory, source_key in (("forecast", "prediction-store", "inputs"), ("native", "native-store", "references")):
            rows = json.loads((output / (lane + "-rows") / (case + ".json")).read_text())
            parent = config[source_key][case]["cut"]
            for number, row in enumerate(rows):
                cut = LocalStore(output / directory).load(row["state"]["cut"])
                cut.verify()
                assert cut.parent == parent
                assert cut.execution["component_pass"]["lane"] == lane
                assert row["before"]["step"] == 2 + number // 7
                parent = cut.fingerprint
            assert cut.payload["step"] == 4 and all(cut.payload[key] is None for key in ("image", "text", "context", "noise_prediction"))
        assert comparisons["consumers"][case]["comparison"]["exact"]
        assert not comparisons["consumers"][case]["predicted"]["independent_decoder_prediction"]
        assert comparisons["consumers"][case]["actual_decoder_input_comparison"]["exact"]
        assert comparisons["consumers"][case]["predicted"]["input_cut"] != comparisons["consumers"][case]["native_reference"]["input_cut"]
    source = json.loads((output / "source.json").read_text())
    for key in ("vae_config", "vae_parameter_manifest"):
        copied = output / Path(config[key]).name
        assert copied.read_bytes() == Path(config[key]).read_bytes()
        assert M["sha"](copied) == config[key + "_sha256"]
    assert {case: row["cut"] for case, row in source["input_records"].items()} == {case: row["cut"] for case, row in config["inputs"].items()}
    assert not (output / "input-geometry.json").exists() and not (output / "parameters").exists()
    assert all(not field.endswith("encoder.weight") for field in source["original_decoder_tensor_sha256"])


def test_forecast_forbids_module_entry_and_restores_global_hooks(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    original = W["run"].__globals__["helpers"]
    before = dict(torch.nn.modules.module._global_forward_pre_hooks)
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    def helpers(config):
        graph, m, recipes = original(config)
        def forbidden(model, latent, *args):
            return model.x_embedder(latent)
        forbidden.__wrapped__ = graph["functional_project"].__wrapped__
        graph["functional_project"] = forbidden
        return graph, m, recipes
    monkeypatch.setitem(W["run"].__globals__, "helpers", helpers)
    with pytest.raises(ValueError, match="original native module.forward"):
        W["run"](config)
    assert dict(torch.nn.modules.module._global_forward_pre_hooks) == before
    assert torch.nn.functional.scaled_dot_product_attention is original_sdpa


@pytest.mark.parametrize("fault", ["outside-window", "tampered-prediction"])
def test_final_seal_blocks_reference_reads_and_native_entry(tmp_path, monkeypatch, fault):
    config = fixture(tmp_path)
    original = ForecastTransaction.execute
    hooks = dict(torch.nn.modules.module._global_forward_pre_hooks)
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    def execute(self, name, callback, **kwargs):
        if name != "native_targets_and_reference_decoders":
            return original(self, name, callback, **kwargs)
        if fault == "outside-window":
            return callback(None)
        def tamper(context):
            path = next((Path(config["output"]) / "predicted-decoder").rglob("image.png"))
            with path.open("ab") as stream:
                stream.write(b"x")
            return callback(context)
        return original(self, name, tamper, **kwargs)
    monkeypatch.setattr(ForecastTransaction, "execute", execute)
    with pytest.raises(ValueError, match="outside globally sealed execution" if fault == "outside-window" else "sealed external evidence signature changed"):
        W["run"](config)
    assert dict(torch.nn.modules.module._global_forward_pre_hooks) == hooks
    assert torch.nn.functional.scaled_dot_product_attention is original_sdpa
    events = [json.loads(line) for line in (Path(config["output"]) / "timeline.jsonl").read_text().splitlines()]
    assert not any(row["kind"] in ("reference_cut_read", "native_entry") for row in events)


def test_equal_valued_native_input_is_rejected_as_prediction(tmp_path):
    config = fixture(tmp_path)
    config["input_root"], config["inputs"], config["input_files"] = config["reference_root"], config["references"], config["reference_files"]
    with pytest.raises(ValueError, match="archived prior forecast lineage"):
        W["run"](config)


def test_numerical_reference_miss_is_retained_without_stopping_collection(tmp_path):
    config = fixture(tmp_path)
    destination = tmp_path / "changed-reference"
    for case, source in config["references"].items():
        old = LocalStore(Path(config["reference_root"]) / source["store"]).load(source["cut"])
        payload = dict(old.payload)
        if case == "HH":
            value = payload["latent"]
            changed = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype)
            changed.copy_(value + .125)
            payload["latent"] = changed
        cut = StateCut(old.model_identity, old.execution, old.boundary, old.parent, payload, old.execution_point)
        LocalStore(destination / "typed-store").save(cut)
        config["references"][case] = dict(store="typed-store", cut=cut.fingerprint)
    config["reference_root"] = str(destination)
    config["reference_files"] = {p.relative_to(destination).as_posix(): dict(sha256=M["sha"](p), bytes=p.stat().st_size) for p in destination.rglob("*") if p.is_file()}
    summary = W["run"](config)
    assert summary["mechanics_status"] == "complete" and summary["totals"]["vae_calls"] == 8
    assert summary["first_covered_mismatch"]["HH"]["number"] == 0
    assert summary["complete_native_cuts"] == 56


def test_actual_cli_and_directory_finalizer(tmp_path):
    config = fixture(tmp_path)
    config["request_sha256"] = "a" * 64
    config["transport_limits"] = dict(max_expanded_bytes=64 << 20, max_member_bytes=16 << 20)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    env = {**os.environ, "MRUN_JOB_ID": "job-tiny-remaining"}
    result = subprocess.run([sys.executable, "-I", str(PATH), "--config", str(path)], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MDB_DIRECTORY_READY" in result.stdout
    summary = json.loads((Path(config["output"]) / "summary.json").read_text())
    assert summary["job_id"] == "job-tiny-remaining" and summary["totals"]["vae_calls"] == 8
