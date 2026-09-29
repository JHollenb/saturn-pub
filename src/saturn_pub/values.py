"""Small value utilities shared by the runtime and durable store."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def tensor(value: Any) -> bool:
    # Avoid importing torch when inspecting only metadata.
    return type(value).__module__.startswith("torch") and hasattr(value, "detach")


def clone(value: Any) -> Any:
    if tensor(value):
        # Snapshot ABI: preserve safe dense tensor layout because native kernels
        # can be bitwise layout-sensitive. Durable storage seals and restores it.
        return value.detach().clone(memory_format=__import__("torch").preserve_format)
    if isinstance(value, Mapping):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    if isinstance(value, list):
        return [clone(item) for item in value]
    return copy.deepcopy(value)


def _scalar(value: Any) -> dict[str, Any]:
    if value is None:
        return {"kind": "scalar", "type": "none", "value": None}
    if type(value) is bool:
        return {"kind": "scalar", "type": "bool", "value": value}
    if type(value) is int:
        return {"kind": "scalar", "type": "int", "value": value}
    if type(value) is float:
        canonical(value)
        return {"kind": "scalar", "type": "float", "value": value}
    if type(value) is str:
        return {"kind": "scalar", "type": "str", "value": value}
    raise ValueError(f"unsupported sealed value: {type(value).__name__}")


def describe(value: Any) -> Any:
    """Return a typed, canonical content descriptor.

    Tags are intentional: JSON alone cannot distinguish tuples from lists and
    Python considers values such as ``True`` and ``1`` equal.  State and
    operation seals must preserve those distinctions.
    """
    if tensor(value):
        raw = value.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8)
        content_digest = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
        return {
            "kind": "tensor",
            "sha256": content_digest,
            # Compatibility alias for callers that consumed the v1 summary.
            "tensor": content_digest,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "layout": "strided-dense",
            "stride": list(value.stride()),
        }
    if isinstance(value, Mapping):
        items = [{"key": describe(key), "value": describe(item)} for key, item in value.items()]
        items.sort(key=lambda item: canonical(item["key"]))
        return {"kind": "mapping", "items": items}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [describe(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [describe(item) for item in value]}
    return _scalar(value)


def legacy_describe_v1(value: Any) -> Any:
    """Reproduce the v1 descriptor solely to verify historical artifacts.

    This format is not authoritative for value typing because it collapsed
    lists/tuples and did not tag scalars.  New seals must use :func:`describe`.
    """
    if tensor(value):
        raw = value.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8)
        return {
            "tensor": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {key: legacy_describe_v1(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [legacy_describe_v1(item) for item in value]
    canonical(value)
    return value


def identity(model: Any, configuration: Any) -> str:
    """Bind actual parameter/buffer content, rather than trusting a model name."""
    return digest({"config": configuration, "state": describe(model.state_dict())})
