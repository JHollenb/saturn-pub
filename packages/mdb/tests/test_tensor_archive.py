"""Lossless native BF16 archives, hostile metadata and framework-free decoding."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from mdb.forward_recording import tensor_descriptor
from mdb.tensor_archive import TensorArchiveError, load_tensor_archive, write_tensor_archive


def test_all_bf16_patterns_roundtrip_including_subnormals_signed_zero_and_nan_payloads(tmp_path):
    bits = np.arange(65536, dtype=np.uint16)
    native = torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(256, 256).T
    record = write_tensor_archive(tmp_path / "bits.npy", native)
    assert record["stored_dtype"] == "uint16" and record["encoding"] == "bf16-bits-v1"
    assert record["descriptor"] == tensor_descriptor(native) and record["native_stride"] == list(native.stride())
    decoded = load_tensor_archive(tmp_path, record)
    expected = (bits.reshape(256, 256).T.astype(np.uint32) << 16)
    np.testing.assert_array_equal(decoded.view(np.uint32), expected)
    assert decoded.dtype == np.float32 and not decoded.flags.writeable
    with pytest.raises(ValueError):
        decoded.setflags(write=True)
    other = load_tensor_archive(tmp_path, record)
    assert not np.shares_memory(decoded, other)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_floats_preserve_bits_dtype_and_noncontiguous_values(tmp_path, dtype):
    value = torch.tensor([[0., -0., float("inf")], [float("-inf"), float("nan"), .125]], dtype=dtype).T
    record = write_tensor_archive(tmp_path / "value.npy", value)
    result = load_tensor_archive(tmp_path, record)
    assert result.dtype == (np.float32 if dtype == torch.float32 else np.float64)
    assert result.tobytes() == value.contiguous().numpy().tobytes()
    assert record["descriptor"] == tensor_descriptor(value)


def test_legacy_widened_bf16_and_fp32_descriptors_still_load(tmp_path):
    for name, value in (("bf16", torch.tensor([0., -0., .125, -3.25], dtype=torch.bfloat16)),
            ("fp32", torch.tensor([1., 2., 3.], dtype=torch.float32))):
        path = tmp_path / (name + ".npy")
        np.save(path, value.float().numpy(), allow_pickle=False)
        record = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "descriptor": tensor_descriptor(value),
            "stored_dtype": "float32", "stored_shape": list(value.shape), "native_stride": list(value.stride())}
        np.testing.assert_array_equal(load_tensor_archive(tmp_path, record), value.float().numpy())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_legacy_fortran_order_records_preserve_logical_values_and_native_hash(tmp_path, dtype):
    value = torch.arange(12, dtype=dtype).reshape(3, 4).T
    native = value.float().numpy()
    assert native.flags.f_contiguous and not native.flags.c_contiguous
    path = tmp_path / "legacy-fortran.npy"
    np.save(path, native, allow_pickle=False)
    record = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "descriptor": tensor_descriptor(value),
        "stored_dtype": "float32", "stored_shape": list(value.shape), "native_stride": list(value.stride())}
    result = load_tensor_archive(tmp_path, record)
    np.testing.assert_array_equal(result, native)
    assert result.tobytes() == value.float().contiguous().numpy().tobytes() and not result.flags.writeable
    bad = copy.deepcopy(record)
    bad["descriptor"]["sha256"] = "0" * 64
    with pytest.raises(TensorArchiveError, match="native"):
        load_tensor_archive(tmp_path, bad)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_new_encoded_archives_still_reject_fortran_order(tmp_path, dtype):
    value = torch.arange(12, dtype=dtype).reshape(3, 4).T
    path = tmp_path / "encoded.npy"
    record = write_tensor_archive(path, value)
    stored = np.asfortranarray(np.load(path, allow_pickle=False))
    np.save(path, stored, allow_pickle=False)
    changed = {**record, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(TensorArchiveError, match="C-order"):
        load_tensor_archive(tmp_path, changed)


def test_native_int64_position_ids_and_large_signed_values_preserve_exact_bits(tmp_path):
    # Native pipeline txt_ids are int64, rather than floating model activations.
    ids = torch.zeros((1, 7, 4), dtype=torch.int64)
    ids[0, :, 3] = torch.arange(7, dtype=torch.int64)
    ids[0, :4, 0] = torch.tensor([-(2 ** 63), 2 ** 63 - 1, 2 ** 53 + 1, -(2 ** 53 + 1)], dtype=torch.int64)
    value = ids.transpose(1, 2)
    record = write_tensor_archive(tmp_path / "txt-ids.npy", value)
    result = load_tensor_archive(tmp_path, record)
    assert record["stored_dtype"] == "int64" and record["encoding"] == "native-npy-v1"
    assert record["descriptor"] == tensor_descriptor(value) and record["native_stride"] == list(value.stride())
    assert result.dtype == np.int64 and result.tobytes() == value.contiguous().numpy().tobytes()
    assert not result.flags.writeable


def test_integer_descriptor_confusion_unsupported_dtype_and_trailing_bytes_refused(tmp_path):
    path = tmp_path / "ids.npy"
    record = write_tensor_archive(path, torch.arange(8, dtype=torch.int64).reshape(1, 2, 4))
    original = path.read_bytes()
    for dtype in (np.uint64, np.int32, np.float64):
        np.save(path, np.arange(8, dtype=dtype).reshape(1, 2, 4), allow_pickle=False)
        changed = {**record, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "stored_dtype": str(np.dtype(dtype))}
        with pytest.raises(TensorArchiveError):
            load_tensor_archive(tmp_path, changed)
    path.write_bytes(original + b"trailing")
    with pytest.raises(TensorArchiveError, match="framing"):
        load_tensor_archive(tmp_path, {**record, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with pytest.raises(TensorArchiveError, match="supports"):
        write_tensor_archive(tmp_path / "unsupported.npy", torch.arange(3, dtype=torch.int32))


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_scalar_and_empty_shapes(tmp_path, shape):
    value = torch.zeros(shape, dtype=torch.bfloat16)
    record = write_tensor_archive(tmp_path / "value.npy", value)
    assert load_tensor_archive(tmp_path, record).shape == shape


def test_changed_file_native_hash_geometry_encoding_and_budget_rejected(tmp_path):
    record = write_tensor_archive(tmp_path / "value.npy", torch.tensor([1., 2.], dtype=torch.bfloat16))
    for key, value in (("path", "../value.npy"), ("stored_shape", [1, 2]), ("stored_dtype", "float32"),
            ("encoding", "numeric-uint16"), ("native_byteorder", "unknown"), ("native_stride", [-1])):
        with pytest.raises(TensorArchiveError):
            load_tensor_archive(tmp_path, {**record, key: value})
    bad = copy.deepcopy(record)
    bad["descriptor"]["sha256"] = "0" * 64
    with pytest.raises(TensorArchiveError, match="native"):
        load_tensor_archive(tmp_path, bad)
    for bound in (1, True, 0):
        with pytest.raises(TensorArchiveError):
            load_tensor_archive(tmp_path, record, max_bytes=bound)
    path = tmp_path / "value.npy"
    original = path.read_bytes()
    for changed in (original[:-1], original + b"trailing"):
        path.write_bytes(changed)
        with pytest.raises(TensorArchiveError):
            load_tensor_archive(tmp_path, {**record, "sha256": hashlib.sha256(changed).hexdigest()})


def test_symlink_ancestor_and_overwriting_refused(tmp_path):
    record = write_tensor_archive(tmp_path / "value.npy", torch.ones(2))
    with pytest.raises(FileExistsError):
        write_tensor_archive(tmp_path / "value.npy", torch.zeros(2))
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(TensorArchiveError, match="symlink"):
        load_tensor_archive(tmp_path, {**record, "path": "alias/value.npy"})


def test_reader_in_fresh_process_blocks_torch_and_model_packages(tmp_path):
    record = write_tensor_archive(tmp_path / "value.npy", torch.tensor([1., -0., .125], dtype=torch.bfloat16))
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    code = """import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','diffusers','transformers','saturn','saturn_pub'}: raise ImportError('framework blocked: '+fullname)
sys.meta_path.insert(0,Block())
from mdb.tensor_archive import load_tensor_archive
value=load_tensor_archive(sys.argv[1],json.load(open(sys.argv[2])))
assert value.tolist()==[1.,-0.,.125] and not value.flags.writeable
assert 'torch' not in sys.modules
print('framework-free')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path), str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "framework-free"
