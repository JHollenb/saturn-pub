"""Custody, identity, and persistence tests for standalone Jacobian I/O."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mdb.jacobian_io import (
    FrozenJacobianLens,
    JacobianReader,
    JacobianWriter,
    PreparedJWrite,
)
from mdb.jacobian_workspace import (
    JacobianWorkspaceError,
    metadata_sha256,
    tensor_payload_identity,
)
from mdb.tensor_hash import prefixed_tensor_sha256, tensor_bytes_sha256


def _sha(character: str) -> str:
    return character * 64


def _header_sha(value: torch.Tensor) -> str:
    return prefixed_tensor_sha256(f"{value.dtype}|{tuple(value.shape)}|".encode(), value)


def _manifest(transport: torch.Tensor, decoder: torch.Tensor) -> dict:
    body = {
        "schema": "saturn-jacobian-lens-v1",
        "model_identity": {
            "model_id": "fixture/model",
            "weight_sha256": _sha("1"),
            "config_sha256": _sha("2"),
            "tokenizer_sha256": _sha("3"),
            "model_abi_sha256": _sha("a"),
        },
        "execution_compatibility": {
            "fitting_lane": "cpu-fp32-eager",
            "permitted_application_lanes": ["cpu-fp32-eager"],
        },
        "coordinate_map": {
            "residual_width": 3,
            "source_layer": 4,
            "completed_layer_boundary": 5,
            "hook_convention": "decoder-block-output",
            "source_position_semantics": "zero-based supplied token position",
        },
        "transport": {
            "frozen": True,
            "shape": [3, 3],
            "dtype": str(transport.dtype),
            "sha256": tensor_bytes_sha256(transport),
            "payload": "transport.safetensors#transport",
        },
        "decoder": {
            "residual_width": 3,
            "vocab_size": 5,
            "final_norm": "declared-rmsnorm",
            "output_head": "declared-decoder-weight",
            "unembedding_weight_sha256": _header_sha(decoder),
        },
        "estimator": {"name": "target-sum/source-mean"},
        "dictionary": {
            "atom_derivation": "decoder-weight-times-transport",
            "norm_scale_convention": "unit-l2-after-transport",
            "sparse_interpretation": "unperformed",
        },
    }
    return {**body, "fingerprint": metadata_sha256(body)}


def _fixture(dtype: torch.dtype = torch.float32):
    transport = torch.tensor([[1.0, 0.2, -0.1], [0.0, 0.5, 0.3], [-0.4, 0.1, 1.2]], dtype=dtype)
    decoder = torch.tensor(
        [
            [0.2, 0.5, -0.3],
            [-0.7, 0.1, 0.4],
            [0.9, -0.2, 0.6],
            [0.1, 0.8, -0.5],
            [-0.4, -0.2, 0.3],
        ],
        dtype=dtype,
    )
    rms_weight = torch.ones(3, dtype=dtype)
    manifest = _manifest(transport, decoder)
    manifest["decoder"]["rms_weight_sha256"] = _header_sha(rms_weight)
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    manifest["fingerprint"] = metadata_sha256(body)
    lens = FrozenJacobianLens(manifest, transport)
    reader = JacobianReader(
        lens,
        decoder,
        model_identity=manifest["model_identity"],
        application_lane="cpu-fp32-eager",
        rms_weight=rms_weight,
    )
    return transport, decoder, manifest, lens, reader


def _source(residual: torch.Tensor, **updates) -> dict:
    source = {
        "model_abi_sha256": _sha("a"),
        "cut_ref": "cut://fixture/one",
        "branch": "native",
        "token_position": 7,
        "completed_layer_boundary": 5,
        "input_tokens_sha256": _sha("b"),
        "intervention_lineage": [],
        "frame_binding": {"frame_id": "frame-1"},
        "numerical_contract": "fixture-contract",
        "activation": tensor_payload_identity(residual),
    }
    source.update(updates)
    return source


def _consumer() -> dict:
    return {"name": "native-test-suffix", "horizon": 2, "extra": {"mode": "fixture"}}


def _bound_reader(*, session_id: str | None = None) -> JacobianReader:
    _, decoder, manifest, lens, _ = _fixture()
    binding = {
        "model_abi_sha256": _sha("a"),
        "numerical_contract": "fixture-contract",
    }
    if session_id is not None:
        binding["session_id"] = session_id
    return JacobianReader(
        lens,
        decoder,
        model_identity=manifest["model_identity"],
        application_lane="cpu-fp32-eager",
        rms_weight=torch.ones(3),
        execution_binding=binding,
    )


def test_lens_snapshots_inputs_returns_copies_and_rejects_corruption(tmp_path: Path):
    transport, _, manifest, lens, _ = _fixture()
    expected = transport.clone()
    transport.zero_()
    manifest["coordinate_map"]["source_layer"] = 99
    assert torch.equal(lens.transport_for("cpu"), expected)
    returned = lens.transport_for("cpu")
    returned.zero_()
    assert torch.equal(lens.transport_for("cpu"), expected)
    assert lens.manifest["coordinate_map"]["source_layer"] == 4
    with pytest.raises(JacobianWorkspaceError, match="requested dtype"):
        lens.transport_for("cpu", torch.float64)

    lens.save(tmp_path)
    reopened = FrozenJacobianLens.open(tmp_path)
    assert torch.equal(reopened.transport_for("cpu"), expected)
    artifact = tmp_path / "tensors.safetensors"
    damaged = bytearray(artifact.read_bytes())
    damaged[-1] ^= 1
    artifact.write_bytes(damaged)
    with pytest.raises(JacobianWorkspaceError, match="file hash"):
        FrozenJacobianLens.open(tmp_path)


def test_lens_normalizes_a_missing_fingerprint():
    transport, decoder, manifest, _, _ = _fixture()
    manifest.pop("fingerprint")
    lens = FrozenJacobianLens(manifest, transport)
    assert lens.manifest["fingerprint"] == metadata_sha256(manifest)

    malformed = dict(manifest, fingerprint="not-a-digest")
    with pytest.raises(JacobianWorkspaceError, match="fingerprint"):
        FrozenJacobianLens(malformed, transport)


def test_lens_rejects_transport_hash_and_nonfinite():
    transport, decoder, manifest, _, _ = _fixture()
    bad = transport.clone()
    bad[0, 0] += 1
    with pytest.raises(JacobianWorkspaceError, match="payload SHA-256"):
        FrozenJacobianLens(manifest, bad)
    nonfinite = transport.clone()
    nonfinite[0, 0] = float("nan")
    corrupt_manifest = _manifest(nonfinite, decoder)
    with pytest.raises(JacobianWorkspaceError, match="non-finite"):
        FrozenJacobianLens(corrupt_manifest, nonfinite)


def test_reader_rejects_model_lane_dtype_and_header_decoder_digest():
    transport, decoder, manifest, lens, _ = _fixture()
    bad_identity = dict(manifest["model_identity"], model_id="other")
    with pytest.raises(JacobianWorkspaceError, match="model identity mismatch"):
        JacobianReader(
            lens,
            decoder,
            model_identity=bad_identity,
            application_lane="cpu-fp32-eager",
            rms_weight=torch.ones(3),
        )
    with pytest.raises(JacobianWorkspaceError, match="not permitted"):
        JacobianReader(
            lens,
            decoder,
            model_identity=manifest["model_identity"],
            application_lane="cuda-bf16",
            rms_weight=torch.ones(3),
        )
    with pytest.raises(JacobianWorkspaceError, match="dtype"):
        JacobianReader(
            lens,
            decoder.double(),
            model_identity=manifest["model_identity"],
            application_lane="cpu-fp32-eager",
            rms_weight=torch.ones(3, dtype=torch.float64),
        )

    raw_only_manifest = _manifest(transport, decoder)
    raw_only_manifest["decoder"]["unembedding_weight_sha256"] = tensor_bytes_sha256(decoder)
    body = {key: value for key, value in raw_only_manifest.items() if key != "fingerprint"}
    raw_only_manifest["fingerprint"] = metadata_sha256(body)
    raw_only_lens = FrozenJacobianLens(raw_only_manifest, transport)
    with pytest.raises(JacobianWorkspaceError, match="decoder weight"):
        JacobianReader(
            raw_only_lens,
            decoder,
            model_identity=raw_only_manifest["model_identity"],
            application_lane="cpu-fp32-eager",
            rms_weight=torch.ones(3),
        )


def test_reader_binds_source_and_wraps_original_observation_metadata():
    _, _, _, _, reader = _fixture()
    residual = torch.tensor([1.0, -2.0, 0.5])
    observation = reader.read(
        residual,
        source=_source(residual),
        token_subset=[4, 2, 0],
        top_k=2,
        logical_address={"layer": 4, "position": 7},
    )
    assert observation.metadata["source"]["activation"] == tensor_payload_identity(residual)
    assert observation.metadata["lens"]["fingerprint"]
    assert observation.metadata["address"] == {"layer": 4, "position": 7}
    assert observation.metadata["transport_policy"].startswith("left-apply-J")
    assert observation.metadata["readout_policy"]["token_subset"] == [4, 2, 0]
    json.dumps(observation.metadata, allow_nan=False)

    with pytest.raises(JacobianWorkspaceError, match="activation identity"):
        reader.read(
            residual, source=_source(residual, activation=tensor_payload_identity(residual + 1))
        )
    with pytest.raises(JacobianWorkspaceError, match="model ABI"):
        reader.read(residual, source=_source(residual, model_abi_sha256=_sha("c")))
    with pytest.raises(JacobianWorkspaceError, match="layer boundary"):
        reader.read(residual, source=_source(residual, completed_layer_boundary=4))


def test_reader_detects_caller_mutation_of_hash_bound_decoder():
    _, decoder, _, _, reader = _fixture()
    residual = torch.tensor([1.0, -2.0, 0.5])
    decoder[0, 0] += 1
    with pytest.raises(JacobianWorkspaceError, match="changed after"):
        reader.read(residual, source=_source(residual))


def test_optional_execution_binding_rejects_wrong_or_missing_source_fields():
    reader = _bound_reader()
    residual = torch.tensor([1.0, -2.0, 0.5])
    observation = reader.read(residual, source=_source(residual))
    assert observation.metadata["execution_binding"] == {
        "model_abi_sha256": _sha("a"),
        "numerical_contract": "fixture-contract",
    }
    with pytest.raises(JacobianWorkspaceError, match="model ABI mismatch"):
        reader.read(residual, source=_source(residual, model_abi_sha256=_sha("c")))
    with pytest.raises(JacobianWorkspaceError, match="numerical contract mismatch"):
        reader.read(
            residual,
            source=_source(residual, numerical_contract="different-contract"),
        )
    missing = _source(residual)
    missing.pop("numerical_contract")
    with pytest.raises(JacobianWorkspaceError, match="source.numerical_contract"):
        reader.read(residual, source=missing)


def test_execution_binding_is_checked_for_prepare_and_validate():
    reader = _bound_reader(session_id="session-1")
    writer = JacobianWriter(reader)
    residual = torch.tensor([1.0, -2.0, 0.5])
    source = _source(residual, session_id="session-1")
    prepared = writer.prepare_steer(
        residual,
        source=source,
        token_id=2,
        dose=0.5,
        consumer=_consumer(),
    )
    assert prepared.metadata["execution_binding"] == reader.execution_binding
    assert torch.equal(
        writer.validate(prepared, residual=residual, source=source), prepared.candidate
    )

    with pytest.raises(JacobianWorkspaceError, match="numerical contract mismatch"):
        writer.prepare_steer(
            residual,
            source=_source(
                residual,
                session_id="session-1",
                numerical_contract="wrong-contract",
            ),
            token_id=2,
            dose=0.5,
            consumer=_consumer(),
        )
    with pytest.raises(JacobianWorkspaceError, match="session ID mismatch"):
        writer.validate(
            prepared,
            residual=residual,
            source=_source(residual, session_id="session-2"),
        )
    missing_session = _source(residual)
    with pytest.raises(JacobianWorkspaceError, match="source.session_id"):
        writer.validate(prepared, residual=residual, source=missing_session)


def test_execution_binding_requires_abi_and_numerical_contract():
    _, decoder, manifest, lens, _ = _fixture()
    common = {
        "lens": lens,
        "decoder_weight": decoder,
        "model_identity": manifest["model_identity"],
        "application_lane": "cpu-fp32-eager",
        "rms_weight": torch.ones(3),
    }
    with pytest.raises(JacobianWorkspaceError, match="model_abi_sha256"):
        JacobianReader(**common, execution_binding={"numerical_contract": "fixture"})
    with pytest.raises(JacobianWorkspaceError, match="numerical_contract"):
        JacobianReader(**common, execution_binding={"model_abi_sha256": _sha("a")})


class _NativeNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        variance = value.float().pow(2).mean(-1, keepdim=True)
        return (value.float() * torch.rsqrt(variance + 1e-6)).to(value.dtype) * self.weight


def test_native_norm_and_output_head_weights_are_bound_to_declared_tensors():
    transport, decoder, manifest, lens, _ = _fixture()
    norm_weight = torch.ones(3)
    norm = _NativeNorm(norm_weight)
    head = torch.nn.Linear(3, 5, bias=False)
    with torch.no_grad():
        head.weight.copy_(decoder)
    reader = JacobianReader(
        lens,
        decoder,
        model_identity=manifest["model_identity"],
        application_lane="cpu-fp32-eager",
        rms_weight=norm_weight,
        final_norm=norm,
        output_head=head,
    )
    residual = torch.tensor([1.0, -2.0, 0.5])
    reader.read(residual, source=_source(residual))
    with torch.no_grad():
        head.weight[0, 0] += 1
    with pytest.raises(JacobianWorkspaceError, match="output head weight changed"):
        reader.read(residual, source=_source(residual))

    wrong_head = torch.nn.Linear(3, 5, bias=False)
    with pytest.raises(JacobianWorkspaceError, match="does not match decoder"):
        JacobianReader(
            lens,
            decoder,
            model_identity=manifest["model_identity"],
            application_lane="cpu-fp32-eager",
            final_norm=norm,
            output_head=wrong_head,
        )

    with torch.no_grad():
        head.weight.copy_(decoder)
        norm.weight[0] += 1
    with pytest.raises(JacobianWorkspaceError, match="RMS weight SHA-256 mismatch"):
        JacobianReader(
            lens,
            decoder,
            model_identity=manifest["model_identity"],
            application_lane="cpu-fp32-eager",
            final_norm=norm,
            output_head=head,
        )


def test_native_module_parameter_rebinding_fails_closed_even_with_identical_bytes():
    _, decoder, manifest, lens, _ = _fixture()
    residual = torch.tensor([1.0, -2.0, 0.5])

    def make_reader():
        norm = _NativeNorm(torch.ones(3))
        head = torch.nn.Linear(3, 5, bias=False)
        with torch.no_grad():
            head.weight.copy_(decoder)
        reader = JacobianReader(
            lens,
            decoder,
            model_identity=manifest["model_identity"],
            application_lane="cpu-fp32-eager",
            rms_weight=norm.weight,
            final_norm=norm,
            output_head=head,
        )
        return reader, norm, head

    reader, _, head = make_reader()
    head.weight = torch.nn.Parameter(head.weight.detach().clone(), requires_grad=False)
    with pytest.raises(JacobianWorkspaceError, match="output head weight was rebound"):
        reader.read(residual, source=_source(residual))

    reader, norm, _ = make_reader()
    norm.weight = torch.nn.Parameter(norm.weight.detach().clone(), requires_grad=False)
    with pytest.raises(JacobianWorkspaceError, match="final norm weight was rebound"):
        reader.read(residual, source=_source(residual))


def test_writer_exact_noop_recipe_and_mutation_staleness():
    _, _, _, _, reader = _fixture()
    writer = JacobianWriter(reader)
    residual = torch.tensor([[1.0, -2.0, 0.5]])
    prepared = writer.prepare_span(
        residual,
        source=_source(residual),
        token_ids=[1, 3],
        target_coordinates=[0.25, -0.5],
        dose=0,
        consumer=_consumer(),
    )
    assert torch.equal(prepared.candidate, residual)
    assert prepared.candidate.data_ptr() != residual.data_ptr()
    assert prepared.metadata["recipe"]["operation"] == "signed-span-coordinate-edit"
    assert prepared.metadata["recipe"]["dose"] == 0.0
    assert prepared.metadata["recipe"]["noop_exact"] is True
    assert (
        writer.validate(prepared, residual=residual, source=_source(residual)).data_ptr()
        != prepared.candidate.data_ptr()
    )

    residual[0, 0] += 1
    with pytest.raises(JacobianWorkspaceError, match="activation identity|stale"):
        writer.validate(prepared, residual=residual, source=_source(residual))

    residual = torch.tensor([[1.0, -2.0, 0.5]])
    prepared = writer.prepare_steer(
        residual,
        source=_source(residual),
        token_id=2,
        dose=-1.5,
        consumer=_consumer(),
    )
    prepared.candidate[0, 0] += 1
    with pytest.raises(JacobianWorkspaceError, match="candidate identity"):
        writer.validate(prepared, residual=residual, source=_source(residual))


def test_writer_bfloat16_upcast_receipt_and_single_token_rule():
    _, _, _, _, reader = _fixture(torch.bfloat16)
    writer = JacobianWriter(reader)
    residual = torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16)
    prepared = writer.prepare_steer(
        residual,
        source=_source(residual),
        token_id=2,
        dose=0.75,
        consumer=_consumer(),
    )
    assert prepared.candidate.dtype == torch.bfloat16
    assert prepared.metadata["recipe"]["work_dtype"] == "torch.float32"
    assert prepared.metadata["recipe"]["native_cast_applied"] is True
    with pytest.raises(JacobianWorkspaceError, match="exactly one token"):
        writer.prepare_steer(
            residual.repeat(2, 1),
            source=_source(residual.repeat(2, 1)),
            token_id=2,
            dose=1,
            consumer=_consumer(),
        )


def test_prepared_write_persists_and_reopens_in_upstream_blocked_subprocess(tmp_path: Path):
    _, _, _, _, reader = _fixture()
    residual = torch.tensor([1.0, -2.0, 0.5])
    prepared = JacobianWriter(reader).prepare_steer(
        residual,
        source=_source(residual),
        token_id=2,
        dose=0.5,
        consumer=_consumer(),
    )
    prepared.save(tmp_path)
    reopened = PreparedJWrite.open(tmp_path)
    assert torch.equal(reopened.candidate, prepared.candidate)
    assert reopened.metadata == prepared.metadata

    package_src = Path(__file__).resolve().parents[1] / "src"
    script = (
        "import builtins,sys; original=builtins.__import__; "
        "builtins.__import__=lambda name,g=None,l=None,f=(),level=0: "
        "(_ for _ in ()).throw(AssertionError(name)) if level==0 and name.split('.')[0] in "
        "{'saturn','common','mrun'} else original(name,g,l,f,level); "
        "from mdb.jacobian_io import PreparedJWrite; "
        "p=PreparedJWrite.open(sys.argv[1]); "
        "print(p.metadata['schema'], p.candidate.shape)"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_src)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    assert "saturn-j-prepared-write-v1" in completed.stdout

    manifest_path = tmp_path / "manifest.json"
    envelope = json.loads(manifest_path.read_text())
    envelope["metadata"]["source"]["branch"] = "corrupt"
    manifest_path.write_text(json.dumps(envelope))
    with pytest.raises(JacobianWorkspaceError, match="fingerprint"):
        PreparedJWrite.open(tmp_path)


def test_consumer_and_source_are_strict_json():
    _, _, _, _, reader = _fixture()
    residual = torch.tensor([1.0, -2.0, 0.5])
    writer = JacobianWriter(reader)
    with pytest.raises(JacobianWorkspaceError, match="consumer.name"):
        writer.prepare_steer(
            residual,
            source=_source(residual),
            token_id=1,
            dose=1,
            consumer={"horizon": 1},
        )
    with pytest.raises(JacobianWorkspaceError, match="strict JSON"):
        writer.prepare_steer(
            residual,
            source=_source(residual),
            token_id=1,
            dose=1,
            consumer={"name": "x", "horizon": 1, "extra": torch.ones(1)},
        )
