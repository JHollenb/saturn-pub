"""Dependency-light algebra tests for the standalone J-space observer."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from mdb.jacobian_workspace import (
    JacobianWorkspaceError,
    derive_dictionary_rows,
    metadata_sha256,
    normalized_atom_steering,
    observe_j,
    span_coordinate_edit,
    tensor_payload_identity,
    validate_lens_metadata,
)


def _sha(character: str) -> str:
    return character * 64


def _manifest() -> dict:
    body = {
        "schema": "saturn-jacobian-lens-v1",
        "model_identity": {
            "model_id": "fixture/model",
            "weight_sha256": _sha("1"),
            "config_sha256": _sha("2"),
            "tokenizer_sha256": _sha("3"),
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
            "dtype": "torch.float32",
            "sha256": _sha("4"),
            "payload": "transport.pt#J",
        },
        "decoder": {
            "residual_width": 3,
            "vocab_size": 5,
            "final_norm": "native-rmsnorm",
            "output_head": "native-lm-head",
        },
        "estimator": {"name": "target-sum/source-mean"},
        "dictionary": {
            "atom_derivation": "decoder-weight-times-transport",
            "norm_scale_convention": "unit-l2-after-transport",
            "sparse_interpretation": "unperformed",
        },
    }
    return {**body, "fingerprint": metadata_sha256(body)}


def test_lens_metadata_binds_payload_model_lane_and_coordinates():
    manifest = _manifest()
    validated = validate_lens_metadata(
        manifest,
        matrix_shape=(3, 3),
        matrix_dtype=torch.float32,
        application_lane="cpu-fp32-eager",
        expected_model_identity={"weight_sha256": _sha("1")},
        expected_coordinate={"source_layer": 4, "completed_layer_boundary": 5},
    )
    assert validated == manifest
    assert json.loads(json.dumps(validated, allow_nan=False)) == manifest

    bad_lane = copy.deepcopy(manifest)
    bad_lane["execution_compatibility"]["permitted_application_lanes"] = ["cuda-bf16"]
    body = {key: value for key, value in bad_lane.items() if key != "fingerprint"}
    bad_lane["fingerprint"] = metadata_sha256(body)
    with pytest.raises(JacobianWorkspaceError, match="not permitted"):
        validate_lens_metadata(
            bad_lane,
            matrix_shape=(3, 3),
            matrix_dtype=torch.float32,
            application_lane="cpu-fp32-eager",
        )

    with pytest.raises(JacobianWorkspaceError, match="shape mismatch"):
        validate_lens_metadata(
            manifest, matrix_shape=(4, 4), matrix_dtype=torch.float32
        )
    with pytest.raises(JacobianWorkspaceError, match="model identity mismatch"):
        validate_lens_metadata(
            manifest,
            matrix_shape=(3, 3),
            matrix_dtype=torch.float32,
            expected_model_identity={"tokenizer_sha256": _sha("9")},
        )


def test_lens_metadata_rejects_tensor_metadata_and_sparse_claims():
    manifest = _manifest()
    manifest["runtime_tensor"] = torch.ones(1)
    with pytest.raises(JacobianWorkspaceError, match="strict JSON"):
        validate_lens_metadata(
            manifest, matrix_shape=(3, 3), matrix_dtype=torch.float32
        )

    manifest = _manifest()
    manifest["dictionary"]["sparse_interpretation"] = "learned-code"
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    manifest["fingerprint"] = metadata_sha256(body)
    with pytest.raises(JacobianWorkspaceError, match="cannot claim"):
        validate_lens_metadata(
            manifest, matrix_shape=(3, 3), matrix_dtype=torch.float32
        )


class _NativeRMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        variance = value.float().pow(2).mean(-1, keepdim=True)
        return (value.float() * torch.rsqrt(variance + self.eps)).to(value.dtype) * self.weight


def test_observer_matches_native_decoder_without_changing_module_flags():
    residual = torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -0.75]])
    transport = torch.tensor(
        [[1.0, 0.2, -0.1], [0.0, 0.5, 0.3], [-0.4, 0.1, 1.2]]
    )
    decoder = torch.tensor(
        [
            [0.2, 0.5, -0.3],
            [-0.7, 0.1, 0.4],
            [0.9, -0.2, 0.6],
            [0.1, 0.8, -0.5],
            [-0.4, -0.2, 0.3],
        ]
    )
    rms_weight = torch.tensor([1.2, 0.7, 1.1])
    eps = 1e-5
    norm = _NativeRMSNorm(rms_weight, eps).train()
    head = torch.nn.Linear(3, 5, bias=False)
    with torch.no_grad():
        head.weight.copy_(decoder)
    head.train()

    native = observe_j(
        residual,
        transport,
        decoder,
        final_norm=norm,
        output_head=head,
        token_subset=[4, 2, 0],
        top_k=2,
    )
    algebraic = observe_j(
        residual,
        transport,
        decoder,
        rms_weight,
        eps=eps,
        token_subset=[4, 2, 0],
        top_k=2,
    )
    expected_transport = residual @ transport.T
    assert torch.equal(native.transported, expected_transport)
    assert torch.equal(native.raw_scores, expected_transport @ decoder.T)
    assert torch.equal(native.normalized_scores, head(norm(expected_transport)))
    assert torch.equal(native.normalized_scores, algebraic.normalized_scores)
    assert torch.equal(native.selected_token_ids, algebraic.selected_token_ids)
    assert native.metadata["readout_coverage"] == "declared-subset"
    assert native.metadata["sparse_interpretation"] == "unperformed"
    assert norm.training and head.training
    assert not native.transported.requires_grad
    json.dumps(native.metadata, allow_nan=False)


def test_dictionary_rows_are_selected_wu_j_with_declared_scaling():
    transport = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
    decoder = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, -2.0]])
    selected = derive_dictionary_rows(transport, decoder, [2, 0], normalize=False)
    expected = decoder[[2, 0]] @ transport
    assert torch.equal(selected.rows, expected)
    assert selected.metadata["norm_scale_convention"] == "none"
    assert selected.metadata["atom_derivation"] == "decoder-weight-times-transport"

    normalized = derive_dictionary_rows(transport, decoder, [2, 0])
    assert torch.allclose(
        normalized.rows,
        expected / torch.linalg.vector_norm(expected, dim=-1, keepdim=True),
    )
    assert normalized.metadata["norm_scale_convention"] == "unit-l2-after-transport"
    assert normalized.metadata["pre_normalization_l2"] == pytest.approx(
        torch.linalg.vector_norm(expected, dim=-1).tolist()
    )


def test_normalized_steering_and_span_coordinates_have_exact_noops():
    residual = torch.tensor([3.0, 2.0])
    atom = torch.tensor([3.0, 4.0])
    steered = normalized_atom_steering(residual, atom, dose=2.5)
    assert torch.allclose(steered.candidate, residual + torch.tensor([1.5, 2.0]))
    assert steered.metadata["actual_delta_norm"] == pytest.approx(2.5)

    no_steer = normalized_atom_steering(residual, atom, dose=0.0)
    assert torch.equal(no_steer.candidate, residual)
    assert no_steer.candidate.data_ptr() != residual.data_ptr()
    assert torch.equal(no_steer.delta, torch.zeros_like(residual))
    assert no_steer.metadata["noop_exact"] is True

    # Columns of atoms.T are [1, 0] and [1, 1], so residual coordinates are [1, 2].
    atoms = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    target = torch.tensor([4.0, -1.0])
    edited = span_coordinate_edit(residual, atoms, target)
    expected = residual + atoms.T @ (target - torch.tensor([1.0, 2.0]))
    assert torch.allclose(edited.candidate, expected)
    assert edited.metadata["diagnostics"]["source_coordinates"] == pytest.approx([1.0, 2.0])
    assert edited.metadata["diagnostics"]["target_coordinates"] == [4.0, -1.0]
    assert edited.metadata["diagnostics"]["rank"] == 2
    assert "not sparse codes" in edited.metadata["diagnostics"]["coordinate_semantics"]

    no_edit = span_coordinate_edit(residual, atoms, target, dose=0.0)
    assert torch.equal(no_edit.candidate, residual)
    assert no_edit.candidate.data_ptr() != residual.data_ptr()
    assert no_edit.metadata["noop_exact"] is True
    json.dumps(no_edit.metadata, allow_nan=False)


def test_shape_finite_and_payload_identity_contracts():
    residual = torch.ones(3)
    transport = torch.eye(3)
    decoder = torch.ones(4, 3)
    rms = torch.ones(3)
    with pytest.raises(JacobianWorkspaceError, match="non-finite"):
        observe_j(
            residual,
            transport.clone().index_put_((torch.tensor([0]), torch.tensor([0])), torch.tensor([float("nan")])),
            decoder,
            rms,
        )
    with pytest.raises(JacobianWorkspaceError, match="residual_width"):
        observe_j(residual, torch.eye(2), decoder, rms)
    with pytest.raises(JacobianWorkspaceError, match="torch.Tensor"):
        span_coordinate_edit(residual, [[1.0, 0.0, 0.0]], torch.ones(1))

    identity = tensor_payload_identity(transport)
    assert identity == tensor_payload_identity(transport.clone())
    reshaped = tensor_payload_identity(transport.reshape(1, 9))
    assert identity["sha256"] == reshaped["sha256"]
    assert identity["identity_sha256"] != reshaped["identity_sha256"]
    json.dumps(identity, allow_nan=False)
