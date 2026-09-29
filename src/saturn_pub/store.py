"""Verified local custody: content-addressed JSON and safetensors, never pickle."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core import Receipt, StateCut
from .values import canonical, tensor


def _safe_id(value: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected a lowercase SHA-256 identifier")
    return value


def _publish(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("existing artifact does not match its content address")
    finally:
        temporary.unlink()


def _encode(value: Any, tensors: dict[str, Any]) -> Any:
    if tensor(value):
        name = f"tensor_{len(tensors)}"
        tensors[name] = value.detach().cpu().contiguous().clone()
        return {"kind": "tensor", "name": name}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("durable state mappings require string keys")
        return {
            "kind": "mapping",
            "items": {key: _encode(item, tensors) for key, item in value.items()},
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode(item, tensors) for item in value],
        }
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"unsupported durable value: {type(value).__name__}")
    canonical(value)
    return {"kind": "scalar", "value": value}


def _decode(value: Any, tensors: dict[str, Any]) -> Any:
    kind = value["kind"]
    if kind == "tensor":
        return tensors[value["name"]]
    if kind == "mapping":
        return {key: _decode(item, tensors) for key, item in value["items"].items()}
    if kind in ("list", "tuple"):
        items = [_decode(item, tensors) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "scalar":
        return value["value"]
    raise ValueError("unknown stored value kind")


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(self, cut: StateCut) -> str:
        cut.verify()
        tensors: dict[str, Any] = {}
        encoded = _encode(cut.payload, tensors)
        blob_id = None
        if tensors:
            from safetensors.torch import save

            blob = save(tensors)
            blob_id = hashlib.sha256(blob).hexdigest()
            _publish(self.root / "blobs" / f"{blob_id}.safetensors", blob)
        metadata = {
            "manifest": cut.manifest(),
            "fingerprint": cut.fingerprint,
            "encoded": encoded,
            "blob": blob_id,
        }
        _publish(self.root / "cuts" / f"{cut.fingerprint}.json", canonical(metadata))
        return cut.fingerprint

    def load(self, identifier: str, *, device: str = "cpu") -> StateCut:
        data = json.loads((self.root / "cuts" / f"{_safe_id(identifier)}.json").read_bytes())
        tensors = {}
        if data["blob"] is not None:
            from safetensors.torch import load

            blob_id = _safe_id(data["blob"])
            blob = (self.root / "blobs" / f"{blob_id}.safetensors").read_bytes()
            if hashlib.sha256(blob).hexdigest() != blob_id:
                raise ValueError("tensor blob digest mismatch")
            tensors = {key: item.to(device) for key, item in load(blob).items()}
        manifest = data["manifest"]
        cut = StateCut(
            manifest["model_identity"],
            manifest["execution"],
            manifest["boundary"],
            manifest["parent"],
            _decode(data["encoded"], tensors),
        )
        if cut.fingerprint != identifier or data["fingerprint"] != identifier:
            raise ValueError("StateCut manifest digest mismatch")
        if cut.manifest() != manifest:
            raise ValueError("stored descriptor does not match payload")
        return cut

    def receipt(self, receipt: Receipt) -> str:
        _publish(
            self.root / "receipts" / f"{receipt.fingerprint}.json", canonical(receipt.to_dict())
        )
        return receipt.fingerprint

    def inspect(self, identifier: str) -> dict[str, Any]:
        """Read and verify the metadata descriptor without importing torch."""
        data = json.loads((self.root / "cuts" / f"{_safe_id(identifier)}.json").read_bytes())
        if hashlib.sha256(canonical(data["manifest"])).hexdigest() != identifier:
            raise ValueError("manifest digest mismatch")
        return data["manifest"]
