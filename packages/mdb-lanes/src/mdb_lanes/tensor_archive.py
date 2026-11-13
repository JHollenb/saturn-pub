"""Lossless, non-pickle numerical tensor archives with a Torch-free reader.

BF16 is stored as little-endian uint16 bit patterns, never integer magnitudes.
The reader widens those bits to FP32 exactly. Native FP32/FP64/int64 remain unchanged.
The native descriptor always describes the original tensor, including its SHA.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path


class TensorArchiveError(ValueError):
    """Malformed, changed or unsupported lossless tensor evidence."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise TensorArchiveError("archive path must be explicit relative POSIX text")
    parts = relative.split("/")
    if Path(relative).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise TensorArchiveError("unsafe tensor archive path")
    path = Path(root)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise TensorArchiveError("tensor archive root contains a symlink")
    root = path.resolve()
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise TensorArchiveError("tensor archive path contains a symlink")
    if not path.resolve().is_relative_to(root):
        raise TensorArchiveError("tensor archive escapes root")
    return path


def write_tensor_archive(path, value, *, root=None):
    """Write once; return JSON metadata. Imports Torch only for this operation.

    ``path`` is an NPY destination; ``root`` makes the returned path root-relative.
    Numerical evidence is serialized in C order; original view strides are metadata.
    No arbitrary Python objects, parameter names or model code are serialized.
    """
    import numpy as np
    import torch

    from mdb_lanes.forward_recording import tensor_descriptor

    path = Path(path)
    root = path.parent if root is None else Path(root)
    try:
        relative = path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError as error:
        raise TensorArchiveError("tensor archive destination escapes root") from error
    path = _path(root, relative)
    if not isinstance(value, torch.Tensor) or value.dtype not in {torch.bfloat16, torch.float32, torch.float64, torch.int64}:
        raise TensorArchiveError("tensor archive supports native BF16, FP32, FP64 and int64")
    descriptor = tensor_descriptor(value)
    data = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        array = data.reshape(-1).view(torch.uint16).numpy().reshape(data.shape).astype("<u2", copy=False)
        encoding = "bf16-bits-v1"
    else:
        storage_dtype = {torch.float32: "<f4", torch.float64: "<f8", torch.int64: "<i8"}[value.dtype]
        array = data.numpy().astype(storage_dtype, copy=False)
        encoding = "native-npy-v1"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    return {"path": relative, "sha256": _sha(path.read_bytes()), "descriptor": descriptor,
        "stored_dtype": str(array.dtype), "stored_shape": list(array.shape), "native_stride": list(value.stride()),
        "encoding": encoding, "native_byteorder": sys.byteorder}


def load_tensor_archive(root, record, *, max_bytes=256 << 20):
    """Verify file/native hashes and return fresh immutable NumPy values.

    BF16 values widen exactly to FP32 by ``uint32(bits) << 16`` and a bit view.
    Legacy widened-FP32 records are also verified against their native descriptor.
    Legacy records without an encoding field may use Fortran-order NPY storage;
    their native hashes still describe logical C-order tensor bytes.
    Legacy native-byte hashes are interpreted as little endian. Nonfinite bit
    patterns, signed zero, empty shapes and BF16 subnormals remain lossless.
    """
    import numpy as np

    if type(max_bytes) is not int or max_bytes < 1 or not isinstance(record, Mapping):
        raise TensorArchiveError("invalid tensor archive metadata or byte budget")
    try:
        path = _path(root, record["path"])
        if not path.is_file() or path.stat().st_size > max_bytes:
            raise TensorArchiveError("tensor archive missing or exceeds byte budget")
        raw = path.read_bytes()
        if len(raw) > max_bytes or not re.fullmatch("[0-9a-f]{64}", record["sha256"]) or _sha(raw) != record["sha256"]:
            raise TensorArchiveError("tensor archive file digest/size differs")
        stream = io.BytesIO(raw)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise TensorArchiveError("unsupported tensor archive NPY version")
        if ((fortran and "encoding" in record) or dtype not in (np.dtype("<u2"), np.dtype("<f4"), np.dtype("<f8"), np.dtype("<i8"))
                or stream.tell() + math.prod(shape) * dtype.itemsize != len(raw)):
            raise TensorArchiveError("tensor archive framing/dtype/C-order differs")
        descriptor = record["descriptor"]
        if (list(shape) != record["stored_shape"] or list(shape) != descriptor["shape"]
                or str(dtype) != record["stored_dtype"] or len(record["native_stride"]) != len(shape)
                or any(type(stride) is not int or stride < 0 for stride in record["native_stride"])):
            raise TensorArchiveError("tensor archive shape/dtype/stride metadata differs")
        data = np.load(io.BytesIO(raw), allow_pickle=False)
        order = record.get("native_byteorder", "little")
        if order not in {"little", "big"}:
            raise TensorArchiveError("unsupported native byte order")
        endian = "<" if order == "little" else ">"
        if descriptor["dtype"] == "torch.bfloat16" and dtype == np.dtype("<u2"):
            if record.get("encoding") != "bf16-bits-v1":
                raise TensorArchiveError("uint16 tensor archive requires BF16 bit encoding")
            native = data.astype(endian + "u2", copy=False).tobytes()
            decoded = (data.astype(np.uint32) << 16).view(np.float32)
        elif descriptor["dtype"] == "torch.bfloat16" and dtype == np.dtype("<f4"):
            bits = data.astype(np.float32, copy=False).view(np.uint32)
            if np.any(bits & np.uint32(0xffff)):
                raise TensorArchiveError("legacy BF16 archive contains non-BF16 FP32 values")
            native = (bits >> 16).astype(endian + "u2").tobytes()
            decoded = data
        elif (descriptor["dtype"], str(dtype)) in {("torch.float32", "float32"), ("torch.float64", "float64")}:
            native = data.astype(endian + ("f4" if dtype.itemsize == 4 else "f8"), copy=False).tobytes()
            decoded = data
        elif descriptor["dtype"] == "torch.int64" and dtype == np.dtype("<i8"):
            native = data.astype(endian + "i8", copy=False).tobytes()
            decoded = data
        else:
            raise TensorArchiveError("stored dtype/native tensor descriptor differ")
        if _sha(native) != descriptor["sha256"]:
            raise TensorArchiveError("native tensor digest differs")
        # A byte-backed result cannot have writeability re-enabled by a caller.
        return np.frombuffer(decoded.tobytes(), dtype=decoded.dtype).reshape(shape)
    except TensorArchiveError:
        raise
    except (KeyError, TypeError, ValueError, OSError, EOFError, OverflowError) as error:
        raise TensorArchiveError("invalid tensor archive: " + str(error)) from error


__all__ = ["TensorArchiveError", "write_tensor_archive", "load_tensor_archive"]
