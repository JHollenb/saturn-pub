"""Generic source-layout mechanics, including payload bits and all source dtypes."""
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

C = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_component_experiment.py"))
P = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_cached_qkv.py"))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64, torch.bool])
def test_strided_restore_preserves_logical_bits_and_dtype(tmp_path, dtype):
    value = torch.arange(24).reshape(1, 3, 2, 4).transpose(1, 2).to(dtype)
    if dtype.is_floating_point:
        value[0, 0, 0, 0] = -0.
    record = C["tensor_record"](tmp_path, "source.npy", value)
    restored, receipt = C["restore_array"](tmp_path / "source.npy", record, torch.device("cpu"))
    assert restored.stride() == value.stride() and restored.dtype == dtype
    assert C["live_descriptor"](restored)["content_sha256"] == record["content_sha256"]
    assert receipt["value_bit_exact"]


def test_overlapping_or_gapped_layouts_are_rejected():
    for stride in ([0, 1], [1, 1], [4, 1]):
        with pytest.raises(ValueError, match="nonoverlapping and dense"):
            C["dense_layout"]([2, 3], stride)


def test_signed_dense_preserves_layout_and_signed_zeros():
    source = torch.arange(24, dtype=torch.bfloat16).reshape(1, 3, 2, 4).transpose(1, 2)
    source[:, :, -1] = 0; source[0, 0, -1, 0] = -0.
    signs = torch.tensor([1, 1, -1], dtype=torch.bfloat16)
    output = P["signed_dense"](source, signs)
    expected = source.float().numpy() * signs.float().numpy().reshape(1, 1, 3, 1)
    assert output.stride() == source.stride()
    assert np.array_equal(output.float().numpy(), expected)
    assert np.array_equal(np.signbit(output.float().numpy()), np.signbit(expected))
