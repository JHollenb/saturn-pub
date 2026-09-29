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
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    if isinstance(value, list):
        return [clone(item) for item in value]
    return copy.deepcopy(value)


def describe(value: Any) -> Any:
    if tensor(value):
        raw = value.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8)
        return {
            "tensor": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {key: describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    canonical(value)
    return value


def identity(model: Any, configuration: Any) -> str:
    """Bind actual parameter/buffer content, rather than trusting a model name."""
    return digest({"config": configuration, "state": describe(model.state_dict())})
