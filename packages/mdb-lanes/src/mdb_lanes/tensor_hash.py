"""Memory-bounded raw-byte hashing for tensor evidence.

The digest conventions in Saturn are deliberately byte exact.  They do not,
however, require a payload-sized CPU tensor followed by a second payload-sized
Python ``bytes`` object.  For contiguous model payloads, these helpers stream
the same logical byte sequence through an existing hash object in bounded host
chunks.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

DEFAULT_TENSOR_HASH_CHUNK_BYTES = 64 * 1024 * 1024


def update_hash_with_tensor_bytes(
    hasher: Any,
    value: torch.Tensor,
    *,
    chunk_bytes: int = DEFAULT_TENSOR_HASH_CHUNK_BYTES,
) -> int:
    """Append tensor bytes in logical contiguous order and return their count.

    Host transfer/serialization staging for a contiguous source is bounded by
    ``chunk_bytes``.  A non-contiguous source may first require a full contiguous
    materialization on its current device.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a torch.Tensor")
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
        raise ValueError("chunk_bytes must be a positive integer")
    source = value.detach().contiguous()
    if source.ndim == 0:
        source = source.reshape(1)
    byte_view = source.view(torch.uint8).reshape(-1)
    total = int(byte_view.numel())
    for offset in range(0, total, chunk_bytes):
        chunk = byte_view[offset : offset + chunk_bytes]
        if chunk.device.type != "cpu":
            chunk = chunk.to(device="cpu")
        host = chunk.contiguous().numpy()
        hasher.update(memoryview(host))
    return total


def tensor_bytes_sha256(
    value: torch.Tensor,
    *,
    chunk_bytes: int = DEFAULT_TENSOR_HASH_CHUNK_BYTES,
) -> str:
    """Hash only the raw contiguous bytes of ``value``."""

    hasher = hashlib.sha256()
    update_hash_with_tensor_bytes(hasher, value, chunk_bytes=chunk_bytes)
    return hasher.hexdigest()


def prefixed_tensor_sha256(
    prefix: bytes,
    value: torch.Tensor,
    *,
    chunk_bytes: int = DEFAULT_TENSOR_HASH_CHUNK_BYTES,
) -> str:
    """Hash ``prefix + contiguous_tensor_bytes`` without materializing either sum."""

    hasher = hashlib.sha256(prefix)
    update_hash_with_tensor_bytes(hasher, value, chunk_bytes=chunk_bytes)
    return hasher.hexdigest()


__all__ = [
    "DEFAULT_TENSOR_HASH_CHUNK_BYTES",
    "prefixed_tensor_sha256",
    "tensor_bytes_sha256",
    "update_hash_with_tensor_bytes",
]
