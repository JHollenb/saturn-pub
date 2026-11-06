"""Installed public-wheel tiny CPU qualification; no real model or mrun job."""
from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
import runpy
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers.models.transformers import transformer_flux2 as native
from safetensors.torch import save_file
from saturn_pub.core import StateCut
from saturn_pub.store import LocalStore

HERE = Path(__file__).parents[1]
CORE_PATH = HERE / "qualification/flux_image_tail_microscope.py"
C = runpy.run_path(str(CORE_PATH))


def fixture(root):
    import mdb.parameter_artifacts as parameter_api
    from mdb.forward_recording import tensor_descriptor

    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(23)
        module_config = dict(dim=8, num_attention_heads=2, attention_head_dim=4, mlp_ratio=2.0, eps=1e-6, bias=False)
        block = native.Flux2TransformerBlock(**module_config).eval().to(torch.bfloat16).requires_grad_(False)
        # Deliberately strided entry. The public cut store must preserve layout.
        image = torch.randn(1, 4, 16, dtype=torch.bfloat16)[..., ::2]
        text = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        mod = torch.randn(1, 48, dtype=torch.bfloat16) / 4
        context = dict(img_mod=mod, txt_mod=mod.flip(-1), rope=(torch.ones(7, 4), torch.zeros(7, 4)))
        parents = {}
        for role, incoming in (("sham", image), ("edit", image + torch.tensor(.03125, dtype=torch.bfloat16))):
            payload = dict(image=incoming, text=text, context=context, latent=torch.zeros(1, 4, 4, dtype=torch.bfloat16),
                           step=1, phase="joint", cursor=0)
            parents[role] = StateCut("tiny-joint-source", {"backend": "cpu", "dtype": "bfloat16"}, "joint.0", None, payload)
    input_root = root / "inputs"
    weights = {key: value.detach().float().numpy().copy() for key, value in block.state_dict().items()}
    inputs = {}
    with torch.inference_mode():
        for role, cut in parents.items():
            LocalStore(input_root / "typed-store").save(cut)
            p = cut.payload
            text_out, image_out = block(p["image"], p["text"], p["context"]["img_mod"], p["context"]["txt_mod"], p["context"]["rope"])
            inputs[role] = dict(store="typed-store", cut=cut.fingerprint,
                               historical_image_descriptor=C["historical_descriptor"](image_out),
                               historical_text_descriptor=C["historical_descriptor"](text_out))
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    file = checkpoint / "weights.safetensors"
    named = {"transformer_blocks.0." + key: value.detach().contiguous() for key, value in block.state_dict().items()}
    save_file(named, file)
    with file.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
    artifacts, parameters = [], {}
    for key, value in block.state_dict().items():
        tensor_name = "transformer_blocks.0." + key
        bits = value.contiguous().view(torch.uint16).numpy().astype("<u2")
        raw = bits.tobytes()
        stream = io.BytesIO()
        np.save(stream, bits, allow_pickle=False)
        encoded = stream.getvalue()
        record = dict(path="arrays/" + hashlib.sha256(key.encode() + encoded).hexdigest() + ".npy", bytes=len(encoded),
                      sha256=hashlib.sha256(encoded).hexdigest(), npy_prefix_b64=base64.b64encode(encoded[:-len(raw)]).decode(),
                      source=dict(file=file.name, file_sha256=C["sha"](file), tensor=tensor_name, dtype="BF16",
                                  shape=list(bits.shape), data_offsets=header[tensor_name]["data_offsets"], payload_sha256=hashlib.sha256(raw).hexdigest()))
        artifacts.append(record)
        parameters[key] = dict(path=record["path"], sha256=record["sha256"], descriptor=tensor_descriptor(value),
                               parameter_tensor=tensor_name, stored_shape=list(bits.shape))
    manifest = root / "parameter-artifacts.json"
    C["save"](manifest, dict(schema="mdb-parameter-artifacts-v1", artifacts=artifacts))
    script = root / "retrieve_weights.py"
    shutil.copyfile(parameter_api.__file__, script)
    source_binding = root / "source-binding.json"
    C["save"](source_binding, {"fixture": True})
    config = dict(profile="tiny-cpu", device="cpu", cpu_threads=1, source_reduction_policy=True, reference_head=True,
                  max_wall_seconds=60, query_chunk=2, module_config=module_config, inputs=inputs, input_root=str(input_root),
                  expected_model_identity="tiny-joint-source", native_source_sha256=C["sha"](inspect.getfile(native)),
                  helper_source_sha256={name: C["sha"](HERE / "qualification" / name) for name in C["HELPERS"]},
                  parameter_root=str(checkpoint), parameter_manifest=str(manifest), retrieval_script=str(script),
                  source_binding=str(source_binding), joint_parameters=parameters, output=str(root / "result"))
    return config, parents, weights


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("image-tail")
    config, parents, weights = fixture(root)
    before = torch.get_rng_state().clone()
    policy = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    summary = C["run"](config)
    assert torch.equal(torch.get_rng_state(), before)  # Meta construction never randomizes actual parameters.
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == policy
    ready = C["finalize"](config["output"], job_id="tiny", request_sha256="fixture")
    return config, parents, weights, summary, ready


def test_two_unchanged_calls_and_all_lanes_have_exact_census(completed):
    config, _, _, summary, _ = completed
    assert summary["counts"] == dict(native_blocks=2, native_attention=2, linear=48, layernorm=14, rmsnorm=8, swiglu=10,
                                     full_model_loads=0, vae_calls=0, encodes=0, noise_draws=0, reference_heads=2, mathematical_affines=22)
    assert summary["native_replay"] == {role: dict(image=True, text=True) for role in ("sham", "edit")}
    cases = json.loads((Path(config["output"]) / "cases.json").read_text())
    assert set(cases) == {"sham", "edit"}
    for row in cases.values():
        assert set(row["actual"]) == set(C["PORTS"])
        assert set(row["native"]) == {"actual-head", "opposite-reduction", "local-opposite", "reference-head"}
        assert set(row["independent"]) == {"actual-head", "actual-after", "local", "reference-head"}
        assert set(row["native"]["actual-head"]["fields"]) == set(C["PORTS"]) - {"q", "k", "v", "text"}
        assert all(x["changed_values"] == 0 for x in row["native"]["actual-head"]["comparisons"].values())
        # Original local residual/modulation arithmetic closes on actual operands.
        assert all(row["independent"]["local"]["comparisons"][key]["changed_values"] == 0
                   for key in ("after_attention", "mlp_input", "image"))


def test_exact_saved_operands_geometry_and_recipe_only_weights(completed):
    config, parents, _, _, ready = completed
    data = Path(config["output"])
    core = C["dependencies"](config)
    cases = json.loads((data / "cases.json").read_text())
    loaded = LocalStore(Path(config["input_root"]) / "typed-store").load(parents["sham"].fingerprint)
    assert loaded.payload["image"].stride() == parents["sham"].payload["image"].stride()
    for role, row in cases.items():
        q, k, v = [core["H"]["load_array"](data, row["actual"][name]) for name in ("q", "k", "v")]
        assert q.shape == k.shape == v.shape == (1, 7, 2, 4)
        assert core["H"]["load_array"](data, row["actual"]["head"]).shape == (1, 4, 8)
        assert row["source_cut"] == parents[role].fingerprint
    manifest = json.loads((data / "files.json").read_text())
    assert ready["files_manifest_sha256"] == C["sha"](data / "files.json")
    assert all(C["sha"](data / path) == record["sha256"] for path, record in manifest["files"].items())
    recipes = json.loads((data / "parameter-artifacts.json").read_text())
    assert len(recipes["artifacts"]) == 16 and all(not (data / row["path"]).exists() for row in recipes["artifacts"])


def test_historical_miss_is_retained_and_complete(tmp_path):
    config, parents, weights = fixture(tmp_path)
    config["reference_head"] = False
    config["inputs"]["edit"]["historical_image_descriptor"]["bf16_le_sha256"] = "0" * 64
    result = C["run"](config, parents=parents, weights=weights)
    assert result["mechanics_status"] == "complete"
    assert not result["native_replay"]["edit"]["image"]
    assert result["native_replay"]["sham"]["image"]
    assert result["counts"]["native_blocks"] == 2 and result["counts"]["mathematical_affines"] == 16
    assert (Path(config["output"]) / "cases/edit.json").is_file()


def test_capture_exception_restores_dispatch_and_unrelated_hooks(tmp_path):
    config, parents, _ = fixture(tmp_path)
    block = native.Flux2TransformerBlock(**config["module_config"]).eval().to(torch.bfloat16)
    original = native.dispatch_attention_fn
    calls = []
    def unrelated(module, args):
        calls.append(True)
        raise RuntimeError("fixture failure")
    hook = block.norm2.register_forward_pre_hook(unrelated)
    counts = dict.fromkeys(("native_blocks", "native_attention", "linear", "layernorm", "rmsnorm", "swiglu"), 0)
    before = {name: (len(module._forward_hooks), len(module._forward_pre_hooks)) for name, module in block.named_modules()}
    with pytest.raises(RuntimeError, match="fixture failure"), torch.inference_mode():
        C["capture_joint"](block, parents["sham"].payload, native, counts)
    assert calls and native.dispatch_attention_fn is original
    assert before == {name: (len(module._forward_hooks), len(module._forward_pre_hooks)) for name, module in block.named_modules()}
    hook.remove()


@pytest.mark.parametrize("change", ("model", "clock", "source", "input-file", "parameter-name"))
def test_refuses_wrong_source_before_native_calls(tmp_path, change):
    config, parents, weights = fixture(tmp_path)
    if change == "model":
        config["expected_model_identity"] = "wrong"
    elif change == "source":
        config["native_source_sha256"] = "0" * 64
    elif change == "input-file":
        config["input_files"] = {"missing": {"sha256": "0" * 64, "bytes": 1}}
    elif change == "parameter-name":
        config["joint_parameters"]["attn.to_q.weight"]["parameter_tensor"] = "transformer_blocks.0.attn.to_k.weight"
        weights = None
    else:
        p = parents["edit"].payload
        p["cursor"] = 1
        cut = StateCut("tiny-joint-source", {}, "wrong", None, p)
        parents["edit"] = cut
        config["inputs"]["edit"]["cut"] = cut.fingerprint
    with pytest.raises(ValueError):
        C["run"](config, parents=parents, weights=weights)
    assert not (Path(config["output"]) / "captures").exists()


def test_comparator_orientation_and_zero_denominator():
    a, n = np.array([1., -1.], np.float32), np.zeros(2, np.float32)
    value = C["measure"](a, n)
    assert value["signed_mean"] == 0 and value["rms"] == 1 and value["native_relative_rms"] is None
    assert C["measure"](a + 1, a)["signed_mean"] == 1


def test_finalization_refuses_overwrite_and_bounds(completed, tmp_path):
    config, _, _, _, _ = completed
    with pytest.raises(FileExistsError):
        C["finalize"](config["output"])
    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="member bound"):
        C["finalize"](tmp_path, max_member_bytes=1)


def test_worker_cli_runs_through_actual_finalization_and_directory_marker(tmp_path):
    config, _, _ = fixture(tmp_path)
    path = tmp_path / "config.json"
    C["save"](path, config)
    execution = subprocess.run([sys.executable, "-I", "-B", str(CORE_PATH), "--config", str(path)],
                               cwd=tmp_path, capture_output=True, text=True, check=False)
    assert execution.returncode == 0, execution.stdout + execution.stderr
    lines = execution.stdout.splitlines()
    assert lines[0] == "MDB_IMAGE_TAIL_SETUP bootstrap"
    marker = json.loads(next(line.split(" ", 1)[1] for line in lines if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(config["output"])
    assert marker["files_manifest_sha256"] == C["sha"](output / "files.json")
    assert marker["summary_sha256"] == C["sha"](output / "summary.json")
    assert marker["worker_seconds_including_bootstrap"] > 0
    assert json.loads((output / "summary.json").read_text())["counts"]["native_blocks"] == 2
