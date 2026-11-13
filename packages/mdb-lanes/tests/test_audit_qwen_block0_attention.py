"""Actual producer serializers, synthetic saved tensors, and hostile custody."""
from __future__ import annotations

import base64
import hashlib
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
from safetensors.torch import save_file

from mdb_lanes import parameter_artifacts

HERE = Path(__file__).parents[1] / "qualification"
A = runpy.run_path(str(HERE / "audit_qwen_block0_attention.py"))
W = runpy.run_path(str(HERE / "qwen_block0_attention_microscope.py"))
assert W["sha"](HERE / "flux_raw_to_image.py") == "b04db3835522a007a15293ef177e4463cf05e9d01122d47d42531dc72bd9700a"
OLD = runpy.run_path(str(HERE / "flux_raw_to_image.py"))


def rewrite(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def finalize(root):
    path = root / "files.json"
    path.unlink(missing_ok=True)
    finalizer = runpy.run_path(str(HERE / "flux_image_tail_microscope.py"))
    return finalizer["finalize"](root, job_id="job-synthetic-block0", request_sha256=A["read"](root / "request.json")["request_sha256"],
        max_expanded_bytes=8 << 20, max_member_bytes=2 << 20)["files_manifest_sha256"]


def fixture(tmp_path):
    """No neural model or attention call: use the actual record writer throughout."""
    torch.set_num_threads(1)
    torch.manual_seed(127)
    root = tmp_path / "raw"
    root.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    values = {name: torch.arange(4, dtype=torch.bfloat16).reshape(2, 2) + index for index, name in enumerate(sorted(A["PARAMETERS"]))}
    archive = checkpoint / "tiny.safetensors"
    save_file(values, str(archive))
    data = archive.read_bytes()
    header = json.loads(data[8:8 + struct.unpack("<Q", data[:8])[0]])
    recipes, before = [], {}
    for name, value in values.items():
        bits = value.view(torch.uint16).numpy()
        buffer = io.BytesIO()
        np.save(buffer, bits, allow_pickle=False)
        encoded, raw = buffer.getvalue(), bits.tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        recipes.append(dict(path="parameters/" + name + ".npy", bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest(),
            npy_prefix_b64=base64.b64encode(encoded[:-len(raw)]).decode(), source=dict(file=archive.name, file_sha256=W["sha"](archive),
                tensor=name, dtype="BF16", shape=list(value.shape), data_offsets=header[name]["data_offsets"], payload_sha256=digest)))
        before[name] = dict(shape=list(value.shape), dtype="torch.bfloat16", sha256=digest)
    W["save"](root / "qwen-config.json", dict(head_dim=2, num_attention_heads=4, num_key_value_heads=2))
    W["save"](root / "qwen-parameter-artifacts.json", dict(schema="mdb-parameter-artifacts-v1", artifacts=recipes))
    shutil.copyfile(parameter_artifacts.__file__, root / "retrieve_weights.py")
    original = root / "original-inputs"
    original.mkdir()
    cases, live, specs = {}, {}, {}
    record = W["tensor_record"]
    for index, case in enumerate(A["CASES"]):
        # All ports are arbitrary saved evidence; zero neural calls are needed by a reader fixture.
        tail = {name: torch.randn(1, 5, 6, dtype=torch.bfloat16) for name in ("H0", "normalized", "q_raw", "k_raw", "v_raw", "projection", "after_attention", "mlp_input", "gate", "up", "activation", "mlp", "H1")}
        q = torch.randn(1, 5, 4, 2, dtype=torch.bfloat16).transpose(1, 2)
        kv = torch.randn(1, 5, 2, 2, dtype=torch.bfloat16).transpose(1, 2)
        tail.update(q=q, k=kv, v=kv.clone(), q_norm=q.clone(), k_norm=kv.clone(), head=q.clone(),
            cos=torch.ones(1, 5, 2, dtype=torch.bfloat16), sin=torch.zeros(1, 5, 2, dtype=torch.bfloat16))
        native = {name: value.clone() for name, value in tail.items()}
        # Preserve a deliberate numerical miss and a layout-only difference as observations.
        native["H1"][0, 0, 0] += 1
        native["q"] = tail["q"].contiguous()
        live[case] = dict(functional=tail, native=native)
        mask = torch.ones((1, 1, 5, 5), dtype=torch.bool).tril()
        if case == "short":
            mask[..., -1] = False
        cases[case] = dict(functional={name: record(root, "functional/" + case + "/" + name, value) for name, value in tail.items()},
            native={name: record(root, "native/" + case + "/" + name, value) for name, value in native.items()},
            dispatch=dict(is_causal=case == "long"), original_mask=None if case == "long" else record(root, "functional/short/mask", mask),
            attended_rows=4 if case == "short" else 5, retained_rows=5,
            source_sidecars={}, matched_functional_vs_native={}, historical_replay={}, module_entries=["", "self_attn"])
        base = original / "cases" / case
        ids = torch.tensor([[3, 7, 9, 11 + index, 13]])
        attended = torch.tensor([[1, 1, 1, 1, int(case == "long")]])
        specs[case] = {}
        for lane in ("forecast", "native"):
            sidecar = dict(input_ids=OLD["_save_tensor"](base, lane + "/ids.npy", ids), attention_mask=OLD["_save_tensor"](base, lane + "/mask.npy", attended),
                hidden_states=[OLD["_save_tensor"](base, lane + "/H0.npy", tail["H0"]), OLD["_save_tensor"](base, lane + "/H1.npy", native["H1"])])
            name = "cases/" + case + "/" + lane + "-qwen.json"
            W["save"](original / name, sidecar)
            specs[case][lane + "_qwen"] = name
        cases[case]["source_sidecars"] = specs[case]
    input_files = {path.relative_to(original).as_posix(): dict(bytes=path.stat().st_size, sha256=W["sha"](path)) for path in original.rglob("*") if path.is_file()}
    W["save"](original / "files.json", dict(schema="mdb-image-tail-files-v1", job_id="job-original-pair", request_sha256="a" * 64,
        files=input_files, expanded_bytes=sum(row["bytes"] for row in input_files.values())))
    provenance = dict(job_id="job-original-pair", request_sha256="a" * 64, manifest_sha256=W["sha"](original / "files.json"))
    arms, arm_live = {}, {}
    for index, arm in enumerate(A["ARMS"]):
        ports = {name: live["long"]["functional"][name].clone() for name in ("head", "projection", "after_attention", "mlp_input", "gate", "up", "activation", "mlp", "H1")}
        for name in ports:
            ports[name] += index * .25
        arm_live[arm] = ports
        arms[arm] = {name: record(root, "triangle/" + arm + "/" + name, value) for name, value in ports.items()}
    repeat = lambda value: value[:, :, None].expand(1, 2, 2, 5, 2).reshape(1, 4, 5, 2)
    inputs = dict(explicit_causal_mask=record(root, "triangle-inputs/explicit-mask", torch.ones(1, 1, 5, 5, dtype=torch.bool).tril()),
        repeated_k=record(root, "triangle-inputs/repeated-k", repeat(live["long"]["functional"]["k"])),
        repeated_v=record(root, "triangle-inputs/repeated-v", repeat(live["long"]["functional"]["v"])))
    rows = []
    for index, label in enumerate(["functional.short", "functional.long", *A["ARMS"], "native.short", "native.long"]):
        if index in (0, 1, 5, 6):
            case = A["CASES"][index] if index < 2 else A["CASES"][index - 5]
            lane = "functional" if index < 2 else "native"
            ports = live[case][lane]
            q, k, v = (ports[name] for name in ("q", "k", "v"))
            mask = None if case == "long" else torch.from_numpy(A["array"](root, cases[case]["original_mask"]))
            causal, gqa = case == "long", case == "long"
            if not gqa:
                k, v = repeat(k), repeat(v)
            out = ports["head"]
        else:
            ports = live["long"]["functional"]
            q, k, v = (ports[name] for name in ("q", "k", "v"))
            if index != 4:
                k, v = repeat(k), repeat(v)
            mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril() if index == 2 else None
            causal, gqa = index != 2, index == 4
            out = arm_live[A["ARMS"][index - 2]]["head"]
        rows.append(dict(label=label, ordinal=index, is_causal=causal, enable_gqa=gqa, scale=2 ** -.5, dropout_p=0.,
            **{name: record(root, "sdpa/" + str(index) + "/" + name, value) for name, value in (("query", q), ("key", k), ("value", v), ("output", out))},
            mask=None if mask is None else record(root, "sdpa/" + str(index) + "/mask", mask),
            profiler=dict(status="not-requested", selected_kernel="unprofiled")))
    plan = dict(schema="mdb-qwen-block0-attention-plan-v1", profile="tiny-cpu", device="cpu", source_reduction_policy=True,
        fixed_bank_case="long", cases=specs, shared_prefix_rows=3, input_files=input_files, source_provenance=provenance,
        input_root="/unused/original-source", parameter_root=str(checkpoint), native_sources={"qwen3": "b" * 64}, output=str(root))
    for key, name in (("qwen_config", "qwen-config.json"), ("qwen_manifest", "qwen-parameter-artifacts.json"), ("retrieval_script", "retrieve_weights.py")):
        plan[key], plan[key + "_sha256"] = "/staged/" + name, W["sha"](root / name)
    worker_sha = W["sha"](HERE / "qwen_block0_attention_microscope.py")
    assert A["WORKER_SHA256"] == worker_sha
    request = dict(config=plan, source_hashes={"qwen_block0_attention_microscope.py": worker_sha, **A["HELPER_PINS"]})
    request["request_sha256"] = A["identity"](request)
    W["save"](root / "request.json", request)
    W["save"](root / "plan.json", {**plan, "request_sha256": request["request_sha256"]})
    W["save"](root / "cases.json", cases)
    triangle = dict(arms=arms, inputs=inputs, accounts={}, fixed_bank_case="long", full_causal_equivalence_rows=5, historical_bridge_valid_prefix_rows=3)
    W["save"](root / "triangle.json", triangle)
    seal_cases = {case: {key: value for key, value in cases[case].items() if key in ("functional", "dispatch", "source_sidecars", "original_mask", "attended_rows", "retained_rows")} for case in A["CASES"]}
    rotary = record(root, "rotary-source/inv_freq", torch.tensor([1.], dtype=torch.float32))
    W["save"](root / "runtime-contract.json", dict(device="cpu", versions=dict(torch=torch.__version__), flags=dict(
        deterministic_algorithms=True, float32_matmul_precision="highest", cuda_tf32=False, cuda_bf16_reduced_precision_reduction=True,
        cudnn_tf32=False, cudnn_deterministic=True, cudnn_benchmark=False, cpu_threads=1)))
    external_files = {path.relative_to(root).as_posix(): dict(bytes=path.stat().st_size, sha256=W["sha"](path)) for path in root.rglob("*") if path.is_file()
        and (path.relative_to(root).as_posix() in ("plan.json", "request.json", "runtime-contract.json", "qwen-config.json", "qwen-parameter-artifacts.json", "retrieve_weights.py")
             or path.relative_to(root).as_posix().startswith(("original-inputs/", "rotary-source/", "functional/", "triangle/", "triangle-inputs/"))
             or any(path.relative_to(root).as_posix().startswith("sdpa/" + str(index) + "/") for index in range(5)))}
    W["save"](root / "functional-seal.json", dict(schema="mdb-qwen-block0-functional-fields-v1", cases=seal_cases, triangle=arms, grid_inputs=inputs,
        sdpa_receipts=rows[:5], external_files=external_files, parameter_identity=before, native_block_calls=0))
    seal_digest = W["sha"](root / "functional-seal.json")
    W["save"](root / "sdpa.json", rows)
    W["save"](root / "pair-comparisons.json", {})
    W["save"](root / "per-token-head.json", {})
    timeline = [dict(kind="functional_sources_and_triangle_complete", seconds=1., sdpa_calls=5, native_block_calls=0),
        dict(kind="functional_fields_sealed", seconds=1.1, sdpa_calls=5, native_block_calls=0, seal_sha256=seal_digest),
        dict(kind="native_block_entry", seconds=1.2, case="short", functional_seal_sha256=seal_digest),
        dict(kind="native_block_completed", seconds=2., case="short", sdpa_calls=6),
        dict(kind="native_block_entry", seconds=2.2, case="long", functional_seal_sha256=seal_digest),
        dict(kind="native_block_completed", seconds=3., case="long", sdpa_calls=7)]
    W["save"](root / "timeline.json", timeline)
    (root / "timeline.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in timeline))
    payload_bytes = sum(row["source"]["data_offsets"][1] - row["source"]["data_offsets"][0] for row in recipes)
    W["save"](root / "source.json", dict(schema="mdb-qwen-block0-attention-source-v1", job_id="job-synthetic-block0", request_sha256=request["request_sha256"],
        worker_sha256=worker_sha, native_sources={"qwen3": dict(sha256="b" * 64, path="/installed/qwen3.py")}, helper_sources=A["HELPER_PINS"],
        input_files=input_files, source_provenance=provenance, parameter_identity_before=before, parameter_identity_after=before,
        functional_seal_sha256=seal_digest, rotary_buffers_before=dict(inv_freq=rotary), rotary_buffers_after=dict(inv_freq={key: rotary[key] for key in ("shape", "dtype", "raw_sha256")}),
        selected_recipe_names=sorted(values), selected_payload_bytes=payload_bytes, recipe_reads=12, recipe_payload_reads=12, source_payload_inventory_bytes=payload_bytes))
    W["save"](root / "summary.json", dict(schema="mdb-qwen-block0-attention-summary-v1", job_id="job-synthetic-block0", request_sha256=request["request_sha256"],
        contract=W["contract"](), actual=dict(native_block_calls=2, native_rotary_calls=2, sdpa_calls=7, source_parameter_reads=12, selected_payload_bytes=payload_bytes), shared_prefix_rows=3))
    return root, finalize(root), request["request_sha256"]


def test_saved_only_full_reader_retains_misses_actual_serializers_and_catalog(tmp_path):
    root, manifest, request = fixture(tmp_path)
    result, _ = A["audit"](root, tmp_path / "compact", expected_manifest=manifest, expected_request=request)
    assert result["matched_functional_vs_native"]["short"]["H1"]["changed_values"] == 1
    assert not result["capture_layout"]["short"]["q"]["capture_stride_exact"]
    assert result["capture_layout"]["short"]["q"]["functional_stride"] != result["capture_layout"]["short"]["q"]["native_stride"]
    assert result["triangle_accounts"]["H1"]["prefix_telescope"]["first_energy"] > 0
    assert len(result["sdpa"]) == 7 and result["new_model_calls"] == result["new_tokenizer_calls"] == result["new_attention_calls"] == 0
    assert A["read"](tmp_path / "compact/catalog/provenance.json")["records"] == 5
    assert not list((tmp_path / "compact").rglob("*.npy"))


def test_actual_writer_bf16_signed_zero_and_capture_stride(tmp_path):
    value = torch.tensor([[0., -0., 1.], [2., 3., 4.]], dtype=torch.bfloat16).t()
    record = W["tensor_record"](tmp_path, "noncontiguous", value)
    restored = A["array"](tmp_path, record)
    assert record["stride"] == list(value.stride())
    assert restored.tobytes() == value.contiguous().float().numpy().tobytes()
    wrong = {**record, "raw_sha256": "0" * 64}
    with pytest.raises(ValueError, match="raw identity"):
        A["array"](tmp_path, wrong)
    difference = A["metric"](np.array([0.], np.float32), np.array([-0.], np.float32))
    assert difference["changed_values"] == 0 and not difference["bitwise_exact"]
    legacy = OLD["_save_tensor"](tmp_path, "legacy-bf16.npy", value)
    assert A["array"](tmp_path, legacy).tobytes() == restored.tobytes()
    byte = OLD["_save_bytes"](tmp_path, "legacy-byte.npy", torch.tensor([0, 255, 3], dtype=torch.uint8))
    assert A["array"](tmp_path, byte).tolist() == [0, 255, 3]


@pytest.mark.parametrize("mutation", ["bank", "original", "seal", "seventh", "member", "future-frontier"])
def test_hostile_custody_and_actual_consumed_operand_refuse(tmp_path, mutation):
    root, manifest, request = fixture(tmp_path)
    if mutation == "bank":
        rows = A["read"](root / "sdpa.json")
        rows[2]["query"] = rows[0]["query"]
        rewrite(root / "sdpa.json", rows)
    elif mutation == "original":
        source = A["read"](root / "plan.json")
        source["source_provenance"]["job_id"] = "wrong"
        rewrite(root / "plan.json", source)
    elif mutation == "seal":
        seal = A["read"](root / "functional-seal.json")
        seal["triangle"][A["ARMS"][0]]["H1"] = seal["triangle"][A["ARMS"][1]]["H1"]
        rewrite(root / "functional-seal.json", seal)
    elif mutation == "seventh":
        rows = A["read"](root / "sdpa.json")
        rows.pop()
        rewrite(root / "sdpa.json", rows)
    elif mutation == "future-frontier":
        seal = A["read"](root / "functional-seal.json")
        name = "native/short/H1.npy"
        seal["external_files"][name] = dict(bytes=(root / name).stat().st_size, sha256=A["sha"](root / name))
        rewrite(root / "functional-seal.json", seal)
    else:
        (root / "unlisted.txt").write_text("hostile")
    if mutation != "member":
        manifest = finalize(root)
    with pytest.raises(ValueError):
        A["audit"](root, tmp_path / "refused", expected_manifest=manifest, expected_request=request)


def test_coherent_receipt_replacement_refuses_fixed_bank(tmp_path):
    root, _, _ = fixture(tmp_path)
    cases, triangle, rows = (A["read"](root / name) for name in ("cases.json", "triangle.json", "sdpa.json"))
    rows[2]["query"] = rows[0]["query"]
    with pytest.raises(ValueError, match="triangle changed fixed Q"):
        A["receipts"](root, cases, triangle, rows, A["read"](root / "qwen-config.json"))


def test_cli_saved_only_without_framework_imports(tmp_path):
    root, manifest, request = fixture(tmp_path)
    script = """import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','tokenizers'}:
        raise AssertionError('saved reader imported a model framework: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
path=sys.argv.pop(1)
sys.argv[0]=path
runpy.run_path(path,run_name='__main__')
"""
    command = [sys.executable, "-I", "-B", "-c", script, str(HERE / "audit_qwen_block0_attention.py"), "--root", str(root), "--output", str(tmp_path / "cli"),
        "--expected-manifest", manifest, "--expected-request", request]
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert A["read"](tmp_path / "cli/audit.json")["new_model_calls"] == 0


def test_actual_saved_worker_compact_marker_and_request(tmp_path):
    root, manifest, request = fixture(tmp_path)
    config = dict(native_root=str(root), native_manifest_sha256=manifest, native_request_sha256=request, output_scratch=str(tmp_path / "out"))
    envelope = dict(config=config, source_hashes={str(HERE / "audit_qwen_block0_attention.py"): A["sha"](HERE / "audit_qwen_block0_attention.py")})
    envelope["request_sha256"] = A["identity"](envelope)
    W["save"](tmp_path / "config.json", config)
    W["save"](tmp_path / "request.json", envelope)
    command = [sys.executable, "-I", "-B", str(HERE / "audit_qwen_block0_attention.py")]
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
        "SATURN_RESIDENT_WORKER": "1", "SATURN_RUN_CONFIG": str(tmp_path / "config.json"), "SATURN_DEBUG_REQUEST": str(tmp_path / "request.json"),
        "MRUN_JOB_ID": "job-synthetic-saved-reader"})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    ready = json.loads(next(line.removeprefix("MDB_DIRECTORY_READY ") for line in completed.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(ready["output_root"])
    assert ready["job_id"] == "job-synthetic-saved-reader" and ready["request_sha256"] == envelope["request_sha256"]
    assert ready["files_manifest_sha256"] == A["sha"](output / "files.json")
    assert A["read"](output / "source-files.json")["job_id"] == "job-synthetic-block0"
    assert not list(output.rglob("*.npy"))
