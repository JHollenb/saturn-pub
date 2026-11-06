"""Tiny installed-wheel component trace, source recipes and honest lineage."""
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
from mdb.forward_recording import tensor_descriptor
from safetensors.torch import save_file

from saturn_pub.core import StateCut
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/flux_single_stack_trace.py"
C = runpy.run_path(str(PATH))
M = runpy.run_path(str(HERE / "qualification/flux_image_tail_microscope.py"))


def fixture(root):
    import mdb.parameter_artifacts as parameter_api

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = dict(dim=8, num_attention_heads=2, attention_head_dim=4, mlp_ratio=2.0, eps=1e-6, bias=False)
    with torch.random.fork_rng():
        torch.manual_seed(43)
        blocks = [native.Flux2SingleTransformerBlock(**spec).eval().to(torch.bfloat16).requires_grad_(False) for _ in range(3)]
        image = torch.randn(1, 8, 4, dtype=torch.bfloat16).transpose(1, 2)
        text = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        context = dict(single_mod=torch.randn(1, 24, dtype=torch.bfloat16) / 4,
                       rope=(torch.ones(7, 4), torch.zeros(7, 4)), temb=torch.ones(1, 8, dtype=torch.bfloat16))
        parents = {}
        for role, delta in (("sham", 0), ("HH", .0625), ("HL", -.03125)):
            incoming = torch.empty_strided(image.shape, image.stride(), dtype=image.dtype)
            incoming.copy_(image + delta)
            payload = dict(image=incoming, text=text + delta / 2, context=context,
                           latent=torch.zeros(1, 4, 2, dtype=torch.bfloat16), conditioning=torch.ones(1, 3, 8, dtype=torch.bfloat16),
                           img_ids=torch.arange(4, dtype=torch.int64), txt_ids=torch.arange(3, dtype=torch.int64),
                           timesteps=torch.arange(4, dtype=torch.float32), sigmas=torch.arange(5, dtype=torch.float32),
                           noise_prediction=None, step=1, phase="single", cursor=0)
            parents[role] = StateCut("tiny-single-source", {"backend": "cpu", "dtype": "bfloat16"}, "single.0", None, payload,
                                    dict(logical_step=1, phase="single", index=0, local_clock=0, next_operation="single_block.0"))
    inputs_root = root / "inputs"
    inputs, expected = {}, {}
    with torch.inference_mode():
        for role, cut in parents.items():
            LocalStore(inputs_root / "typed-store").save(cut)
            p = cut.payload
            combined = torch.cat([p["text"], p["image"]], dim=1)
            for block in blocks:
                combined = block(combined, None, p["context"]["single_mod"], p["context"]["rope"])
            expected[role] = {"image": combined[:, 3:].clone(), "text": combined[:, :3].clone()}
            rows = [{} for _ in range(26)]
            rows[5] = dict(logical_step=1, local_index=5, capture=dict(cut=cut.fingerprint, retained_full_cut=True),
                           outputs={name: {"descriptor": M["historical_descriptor"](p[name])} for name in ("image", "text")})
            rows[25] = dict(logical_step=1, local_index=25,
                            outputs={name: {"descriptor": M["historical_descriptor"](value)} for name, value in expected[role].items()})
            ledger = "native/" + C["BRANCHES"][role] + ".jsonl"
            (inputs_root / ledger).parent.mkdir(parents=True, exist_ok=True)
            (inputs_root / ledger).write_text("".join(json.dumps(row) + "\n" for row in rows))
            inputs[role] = dict(branch=C["BRANCHES"][role], store="typed-store", cut=cut.fingerprint, ledger=ledger, entry_row=5, exit_row=25,
                                historical_image_descriptor=rows[25]["outputs"]["image"]["descriptor"],
                                historical_text_descriptor=rows[25]["outputs"]["text"]["descriptor"])
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    path = checkpoint / "weights.safetensors"
    tensors = {f"single_transformer_blocks.{index}." + key: value.detach().contiguous()
               for index, block in enumerate(blocks) for key, value in block.state_dict().items()}
    save_file(tensors, path)
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
    artifacts, banks = [], {}
    for index, block in enumerate(blocks):
        parameters = {}
        for alias, name in C["PARAMETERS"].items():
            if name not in block.state_dict():
                parameters[alias] = None
                continue
            value = block.state_dict()[name]
            tensor = f"single_transformer_blocks.{index}." + name
            bits = value.contiguous().view(torch.uint16).numpy().astype("<u2")
            raw = bits.tobytes()
            buffer = io.BytesIO()
            np.save(buffer, bits, allow_pickle=False)
            encoded = buffer.getvalue()
            record = dict(path="parameters/" + hashlib.sha256(tensor.encode() + encoded).hexdigest() + ".npy", bytes=len(encoded),
                          sha256=hashlib.sha256(encoded).hexdigest(), npy_prefix_b64=base64.b64encode(encoded[:-len(raw)]).decode(),
                          source=dict(file=path.name, file_sha256=M["sha"](path), tensor=tensor, dtype="BF16", shape=list(bits.shape),
                                      data_offsets=header[tensor]["data_offsets"], payload_sha256=hashlib.sha256(raw).hexdigest()))
            artifacts.append(record)
            parameters[alias] = dict(path=record["path"], sha256=record["sha256"], descriptor=tensor_descriptor(value),
                                     parameter_tensor=tensor, stored_shape=list(bits.shape), storage="checkpoint-recipe-v1")
        banks[str(index)] = dict(block_index=index, native_parameter_prefix=f"single_transformer_blocks.{index}.",
            heads=2, head_dim=4, layernorm_eps=1e-6, qk_eps=1e-6, parameters=parameters,
            native_parameters={name: tensor_descriptor(value) for name, value in block.state_dict().items()})
    manifest, binding, source_binding, script = [root / name for name in ("parameter-artifacts.json", "binding.json", "source-binding.json", "retrieve_weights.py")]
    M["save"](manifest, dict(schema="mdb-parameter-artifacts-v1", artifacts=artifacts))
    M["save"](binding, {"final_blocks": banks})
    M["save"](source_binding, {"fixture": True, "scheduler_status": "failed", "manifest_origin": "collector-derived-source-manifest"})
    shutil.copyfile(parameter_api.__file__, script)
    config = dict(profile="tiny-cpu", device="cpu", cpu_threads=1, block_count=3, max_wall_seconds=60, module_config=spec,
                  source_reduction_policy=True, input_root=str(inputs_root), inputs=inputs, expected_model_identity="tiny-single-source",
                  native_source_sha256=M["sha"](inspect.getfile(native)), parameter_root=str(checkpoint), output=str(root / "result"),
                  helper_source_sha256={name: M["sha"](HERE / "qualification" / name) for name in C["HELPERS"]},
                  source_provenance={"scheduler_status": "failed", "collection_mode": "authenticated-ssh-manifest-recovery"})
    for key, file in (("parameter_manifest", manifest), ("binding", binding), ("source_binding", source_binding), ("retrieval_script", script)):
        config[key], config[key + "_sha256"] = str(file), M["sha"](file)
    config["input_files"] = {file.relative_to(inputs_root).as_posix(): dict(sha256=M["sha"](file), bytes=file.stat().st_size)
                             for file in inputs_root.rglob("*") if file.is_file()}
    return config, parents, expected


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    config, parents, expected = fixture(tmp_path_factory.mktemp("single-stack"))
    return config, parents, expected, C["run"](config)


def test_full_native_loop_census_lineage_and_historical_exits(completed):
    config, parents, expected, summary = completed
    assert summary["counts"] == C["contract"](config)["counts"]
    assert summary["complete_typed_cuts"] == 12
    assert summary["act_receipts"] == summary["session_continue_receipts"] == 0
    assert summary["vae_calls"] == summary["encodes"] == summary["noise_draws"] == 0
    assert all(row[stream]["exact"] for row in summary["native_replay"].values() for stream in ("image", "text"))
    store = LocalStore(Path(config["output"]) / "typed-store")
    for role in C["ROLES"]:
        previous = parents[role].fingerprint
        for index in range(4):
            row = json.loads((Path(config["output"]) / f"boundaries/{role}/{index:02d}.json").read_text())
            cut = store.load(row["cut"])
            cut.verify()
            if index:
                assert cut.parent == previous
                assert cut.execution["component_trace"]["id"] == C["EXECUTION_ID"]
                assert cut.payload["cursor"] == index
                assert cut.payload["phase"] == ("readout" if index == 3 else "single")
            previous = cut.fingerprint
        for stream in ("image", "text"):
            assert torch.equal(cut.payload[stream], expected[role][stream])


def test_own_carried_context_and_noncontiguous_entry_preserved(completed):
    config, parents, _, _ = completed
    store = LocalStore(Path(config["output"]) / "typed-store")
    for role, parent in parents.items():
        p = parent.payload
        assert p["image"].stride()[-1] == 4
        for index in range(4):
            row = json.loads((Path(config["output"]) / f"boundaries/{role}/{index:02d}.json").read_text())
            payload = store.load(row["cut"]).payload
            for key in p:
                if key not in ("image", "text", "phase", "cursor"):
                    assert describe(payload[key]) == describe(p[key])
            assert row["fields"]["image"]["storage"]["kind"] == "typed-cut-slot"
    assert not (Path(config["output"]) / "arrays").exists()
    manifest = json.loads((Path(config["output"]) / "parameter-artifacts.json").read_text())
    assert all(not (Path(config["output"]) / row["path"]).exists() for row in manifest["artifacts"])


def test_signed_accounts_and_negative_increment_undefined_ratio():
    fields = {role: {stream: np.array([0., 0.]) for stream in ("image", "text")} for role in C["ROLES"]}
    fields["HH"]["image"] = np.array([3., -3.])
    first = C["boundary_accounts"](fields)
    assert first["HH"]["image"]["relative_to_sham_rms"] is None
    assert first["HH"]["image"]["HH_over_HL_response_rms"] is None
    fields["HH"]["image"] = np.array([1., -1.])
    second = C["boundary_accounts"](fields, first)
    assert second["HH"]["image"]["rms_absolute_increment"] == -2
    assert second["HH"]["image"]["signed_mean"] == 0


def test_replay_miss_is_retained_without_discarding_native_panel(tmp_path):
    config, _, _ = fixture(tmp_path)
    path = Path(config["input_root"]) / config["inputs"]["HH"]["ledger"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[25]["outputs"]["image"]["descriptor"]["bf16_le_sha256"] = "0" * 64
    config["inputs"]["HH"]["historical_image_descriptor"] = rows[25]["outputs"]["image"]["descriptor"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config["input_files"][config["inputs"]["HH"]["ledger"]] = dict(sha256=M["sha"](path), bytes=path.stat().st_size)
    report = C["run"](config)
    assert not report["native_replay"]["HH"]["image"]["exact"]
    assert report["counts"]["native_blocks"] == 9 and report["mechanics_status"] == "complete"


def test_wrong_named_recipe_is_refused_and_dispatch_restored(tmp_path):
    config, _, _ = fixture(tmp_path)
    binding = json.loads(Path(config["binding"]).read_text())
    binding["final_blocks"]["0"]["parameters"]["qkv_mlp_weight"]["parameter_tensor"] = "wrong.weight"
    Path(config["binding"]).write_text(json.dumps(binding))
    config["binding_sha256"] = M["sha"](config["binding"])
    original = native.dispatch_attention_fn
    with pytest.raises(ValueError, match="parameter identity"):
        C["run"](config)
    assert native.dispatch_attention_fn is original
    assert not (Path(config["output"]) / "calls/00.json").exists()


def test_source_digest_and_clock_refused_before_native(tmp_path):
    config, _, _ = fixture(tmp_path)
    config["binding_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source metadata"):
        C["run"](config)
    assert not Path(config["output"]).exists()
    config["inputs"]["HH"]["entry_row"] = 6
    with pytest.raises(ValueError, match="historical branch/cursor"):
        C["validate_config"](config)


def test_actual_worker_tail_finalizes_recipe_only_directory(tmp_path):
    config, _, _ = fixture(tmp_path)
    config["request_sha256"] = "1" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    process = subprocess.run([sys.executable, "-I", "-B", str(PATH), "--config", str(path)], cwd=tmp_path,
                             capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    ready = json.loads(next(line.split(" ", 1)[1] for line in process.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    assert ready["files_manifest_sha256"] == M["sha"](Path(config["output"]) / "files.json")
    assert ready["expanded_bytes"] < 4 << 20
