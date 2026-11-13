"""Saved bytes and image diagnostics, without trained models or equations."""
from __future__ import annotations

import hashlib
import json
import runpy
import struct
from pathlib import Path

import numpy as np
import pytest

AUDIT = runpy.run_path(str(Path(__file__).parents[1] / "qualification/analyze_flux_mlp_forecast.py"))


def test_bf16_halfway_values_use_ties_to_even_for_both_signs():
    # Exact halfway FP32 bit patterns: retained low bit is alternately even/odd.
    bits = np.array([0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000], dtype=np.uint32)
    expected = np.array([0x3f800000, 0x3f820000, 0xbf800000, 0xbf820000], dtype=np.uint32)
    actual = AUDIT["quantize_bf16"](bits.view(np.float32))
    np.testing.assert_array_equal(actual.view(np.uint32), expected)
    np.testing.assert_array_equal(bits, [0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000])


def test_native_descriptor_uses_bf16_bytes_and_refuses_unrounded_witness():
    value = np.array([1., -2.], dtype=np.float32)
    result = AUDIT["descriptor"](value, bf16=True)
    assert result["sha256"] == hashlib.sha256(bytes.fromhex("803f00c0")).hexdigest()
    assert result["dtype"] == "torch.bfloat16"
    with pytest.raises(ValueError, match="exactly widened"):
        AUDIT["descriptor"](np.array([1.001], dtype=np.float32), bf16=True)
    with pytest.raises(ValueError, match="nonfinite"):
        AUDIT["quantize_bf16"]([np.inf])


def scene(*, reverse=False, alpha=False):
    pixels = np.full((12, 24, 4 if alpha else 3), 255, dtype=np.uint8)
    pixels[3:9, 2:8, :3] = [255, 0, 0] if not reverse else [0, 0, 255]
    pixels[3:9, 16:22, :3] = [0, 0, 255] if not reverse else [255, 0, 0]
    return pixels


def test_color_order_is_signed_and_not_a_shape_or_semantic_gate():
    left, right = (AUDIT["image_diagnostics"](scene(reverse=flag)) for flag in (False, True))
    assert left["red_minus_blue_normalized_x"] < 0 < right["red_minus_blue_normalized_x"]
    assert left["red"]["largest"]["pixels"] == 36
    assert left["red"]["largest"]["centroid_xy"] == [4.5, 5.5]
    assert left["semantic_gate"] is False
    assert "unassessed" in left["shape_identity"]


def test_missing_or_transparent_color_remains_unknown_without_failure():
    pixels = scene(alpha=True)
    pixels[3:9, 2:8, 3] = 0
    result = AUDIT["image_diagnostics"](pixels)
    assert result["red"]["largest"] is None
    assert result["red_minus_blue_normalized_x"] is None
    assert result["blue"]["largest"]["pixels"] == 36
    assert result["semantic_gate"] is False


def test_pixel_collateral_retains_both_halves_and_sign():
    baseline = np.zeros((4, 8, 3), dtype=np.uint8)
    changed = baseline.copy()
    changed[:, :4] = 10
    result = AUDIT["pixel_difference"](changed, baseline)
    assert result["changed_pixel_fraction"] == .5
    assert result["left_half_MAE"] == 10 and result["right_half_MAE"] == 0
    assert result["signed_mean"] == 5


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def publish(root, directory, suffix, raw):
    identifier = hashlib.sha256(raw).hexdigest()
    path = root / directory / (identifier + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return identifier, path


@pytest.fixture
def typed_store(tmp_path):
    raw = bytes.fromhex("803f00c0")
    header = canonical({"tensor_0": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}})
    blob, blob_path = publish(tmp_path, "blobs", ".safetensors", struct.pack("<Q", len(header)) + header + raw)
    described = {"kind": "tensor", "sha256": hashlib.sha256(raw).hexdigest(), "tensor": hashlib.sha256(raw).hexdigest(),
                 "shape": [2], "dtype": "torch.bfloat16", "stride": [1], "layout": "strided-dense"}
    page = {"schema": "saturn-pub-slot-page-v1", "encoded": {"kind": "tensor", "name": "tensor_0", "shape": [2], "stride": [1]},
            "value": described, "tensor_blob": blob}
    page_id, _ = publish(tmp_path, "pages", ".json", canonical(page))
    null = {"kind": "scalar", "type": "none", "value": None}
    manifest = {"schema": "saturn-pub-statecut-v2", "model_identity": "fixture", "execution": {},
                "boundary": "saved", "parent": None, "execution_point": null, "surface": null,
                "payload": {"kind": "mapping", "items": [{"key": {"kind": "scalar", "type": "str", "value": "latent"}, "value": described}]}}
    identifier = hashlib.sha256(canonical(manifest)).hexdigest()
    metadata = {"schema": "saturn-pub-local-cut-v2", "manifest": manifest, "fingerprint": identifier,
                "logical_parent": None, "execution_point": None, "surface": None,
                "slots": {"latent": page_id}, "physical_blobs": [blob]}
    path = tmp_path / "cuts" / (identifier + ".json")
    path.parent.mkdir()
    path.write_bytes(canonical(metadata))
    return tmp_path, blob_path, path, metadata


def test_typed_custody_reads_raw_bf16_without_native_hydration(typed_store):
    root, _, _, _ = typed_store
    assert AUDIT["audit_store"](root)["counts"] == {"cuts": 1, "pages": 1, "tensor_blobs": 1}


def test_typed_blob_corruption_is_a_mechanics_error(typed_store):
    root, path, _, _ = typed_store
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="blob CAS differs"):
        AUDIT["audit_store"](root)


def test_changed_logical_parent_cannot_hide_behind_same_manifest(typed_store):
    root, _, path, metadata = typed_store
    metadata["logical_parent"] = "a" * 64
    path.write_bytes(canonical(metadata))
    with pytest.raises(ValueError, match="seal/metadata"):
        AUDIT["audit_store"](root)
