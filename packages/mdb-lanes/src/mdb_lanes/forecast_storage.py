"""Immutable numerical-tree readers for legacy files and closed array bundles.

Imports remain framework-free until a caller asks to decode a numerical value.
Bundle offsets address complete, unchanged NPY files, never raw tensor storage.
"""
from __future__ import annotations

import ast
import hashlib
import io
import math
import re
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

BUNDLE_MAGIC = b"MDBNPYBUNDLE1\n"


class ForecastStorageError(RuntimeError):
    """Malformed or changed immutable numerical evidence."""


def _npy(raw, start=0):
    """Return a numeric NPY boundary/descriptor without importing NumPy."""
    if raw[start:start + 6] != b"\x93NUMPY" or start + 10 > len(raw):
        raise ForecastStorageError("invalid NPY leaf framing")
    version = raw[start + 6:start + 8]
    if version == b"\x01\x00":
        width, code = 2, "<H"
    elif version in (b"\x02\x00", b"\x03\x00"):
        width, code = 4, "<I"
    else:
        raise ForecastStorageError("unsupported NPY leaf version")
    if start + 8 + width > len(raw):
        raise ForecastStorageError("truncated NPY header")
    length = struct.unpack(code, raw[start + 8:start + 8 + width])[0]
    data = start + 8 + width + length
    if length > 10000 or data > len(raw):
        raise ForecastStorageError("truncated or oversized NPY header")
    try:
        header = ast.literal_eval(raw[start + 8 + width:data].decode("utf8" if version == b"\x03\x00" else "latin1"))
    except (ValueError, SyntaxError, UnicodeError) as error:
        raise ForecastStorageError("invalid NPY header") from error
    if not isinstance(header, dict) or set(header) != {"descr", "shape", "fortran_order"}:
        raise ForecastStorageError("invalid NPY header fields")
    shape, dtype = header["shape"], header["descr"]
    if (not isinstance(shape, tuple) or len(shape) > 64 or any(type(x) is not int or x < 0 for x in shape)
            or type(header["fortran_order"]) is not bool or header["fortran_order"]):
        raise ForecastStorageError("NPY leaves require canonical C-order numeric arrays")
    match = re.fullmatch(r"([<>|=])([biuf])([1-9][0-9]*)", dtype) if isinstance(dtype, str) else None
    if match is None:
        raise ForecastStorageError("NPY leaf dtype must be real numeric")
    order, kind, itemsize = match[1], match[2], int(match[3])
    if itemsize not in {"b": (1,), "i": (1, 2, 4, 8), "u": (1, 2, 4, 8), "f": (2, 4, 8, 16)}[kind]:
        raise ForecastStorageError("unsupported numeric NPY item size")
    if order not in (("|",) if itemsize == 1 else ("<", ">")):
        raise ForecastStorageError("NPY byte order is not canonical")
    end = data + math.prod(shape) * itemsize
    if end > len(raw):
        raise ForecastStorageError("truncated NPY array data")
    native = "<" if sys.byteorder == "little" else ">"
    if order in ("|", "=", native) or itemsize == 1:
        name = {"b": "bool", "i": "int", "u": "uint", "f": "float"}[kind]
        dtype_name = name if kind == "b" else name + str(itemsize * 8)
    else:
        dtype_name = dtype
    return end, {"shape": list(shape), "dtype": dtype_name}, data


def _path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ForecastStorageError("array path must be explicit text")
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts) or Path(relative).is_absolute():
        raise ForecastStorageError("unsafe array path")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ForecastStorageError("array path contains a symbolic link")
    if not path.resolve().is_relative_to(root):
        raise ForecastStorageError("array path escapes transaction directory")
    return path


class ArrayReader:
    """One decode call, with verified bundle bytes shared only within that call."""

    def __init__(self, root):
        root = Path(root)
        if root.is_symlink():
            raise ForecastStorageError("transaction directory is a symbolic link")
        self.root, self.bundles = root.resolve(), {}

    @staticmethod
    def digest(raw, expected, label):
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ForecastStorageError("invalid " + label + " identity")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ForecastStorageError(label + " seal differs")

    def raw(self, descriptor):
        shape = descriptor.get("shape")
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ForecastStorageError("invalid stored array shape")
        path = _path(self.root, descriptor["path"])
        storage = descriptor.get("storage", "files-v1")
        if storage == "files-v1":
            raw = path.read_bytes()
        elif storage == "bundle-v1":
            identifier = descriptor["bundle_sha256"]
            key = (str(path), identifier)
            if key not in self.bundles:
                bundle = path.read_bytes()
                self.digest(bundle, identifier, "array bundle")
                if not bundle.startswith(BUNDLE_MAGIC):
                    raise ForecastStorageError("invalid array bundle framing")
                boundaries, position = {}, len(BUNDLE_MAGIC)
                while position < len(bundle):
                    end, metadata, data = _npy(bundle, position)
                    boundaries[position] = (end - position, metadata, data - position)
                    position = end
                self.bundles[key] = bundle, boundaries
            bundle, boundaries = self.bundles[key]
            offset, length = descriptor["offset"], descriptor["nbytes"]
            if (type(offset) is not int or type(length) is not int
                    or offset < len(BUNDLE_MAGIC) or length < 1
                    or offset > len(bundle) or length > len(bundle) - offset):
                raise ForecastStorageError("array bundle range is invalid")
            if offset not in boundaries or boundaries[offset][0] != length:
                raise ForecastStorageError("array bundle range is not a complete framed leaf")
            raw = bundle[offset:offset + length]
        else:
            raise ForecastStorageError("unsupported array storage: " + str(storage))
        self.digest(raw, descriptor["sha256"], "stored array")
        end, metadata, data = _npy(raw)
        if end != len(raw) or any(descriptor.get(key) != value for key, value in metadata.items()):
            raise ForecastStorageError("stored array descriptor differs from NPY header")
        self.digest(raw[data:], descriptor["content_sha256"], "stored array content")
        return raw

    def array(self, descriptor):
        import numpy as np

        if not isinstance(descriptor, Mapping) or descriptor.get("kind") != "array":
            raise ForecastStorageError("expected an array descriptor")
        raw = self.raw(descriptor)
        stream = io.BytesIO(raw)
        value = np.load(stream, allow_pickle=False)
        if not isinstance(value, np.ndarray) or stream.tell() != len(raw):
            raise ForecastStorageError("array leaf must contain exactly one NPY array")
        if (value.dtype.kind not in "biuf" or not np.isfinite(value).all()
                or list(value.shape) != descriptor["shape"] or str(value.dtype) != descriptor["dtype"]):
            raise ForecastStorageError("stored array descriptor differs")
        content = value.tobytes(order="C")
        self.digest(content, descriptor["content_sha256"], "stored array content")
        # bytes is the ultimate backing owner: setflags(write=True) cannot
        # mutate another callback's view or the retained evidence.
        return np.frombuffer(content, dtype=value.dtype).reshape(value.shape)

    def tree(self, descriptor):
        kind = descriptor["kind"]
        if kind == "array":
            return self.array(descriptor)
        if kind == "mapping":
            return MappingProxyType({key: self.tree(value) for key, value in descriptor["items"].items()})
        if kind in ("list", "tuple"):
            return tuple(self.tree(value) for value in descriptor["items"])
        if kind == "scalar":
            return descriptor["value"]
        raise ForecastStorageError("unknown stored descriptor")
