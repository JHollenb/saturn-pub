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


class TensorFacts:
    """Facts about a tensor's content (its ``describe`` descriptor, that it is all finite),
    remembered while the tensor object lives and its version counter, data pointer, shape,
    stride, dtype and device are unchanged.

    The key is the same guard the ``shares_state`` contract already relies on (``core``: a
    sealed tensor is immutable, every in-place write bumps its version counter, which views
    share). A tensor without a version counter (an inference tensor) is never remembered. A
    write that bypasses the counter (through ``.data``, ``numpy()`` or a raw storage) is not
    seen here, as it is not seen by the in-place guard; ``describe`` and ``StateCut.verify``
    never use this memo and always re-hash.

    Writes that bypass the version counter (``.data``, ``numpy()``, dlpack, raw kernels) are
    not detected, the same as in main's in-place guard. For the native K/V cache this means
    the decoder holds the state's tensors by reference only while no foreign forward hook is
    registered; foreign hooks force the copying path.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[Any, tuple[Any, ...], dict[str, Any]]] = {}

    @staticmethod
    def _key(value: Any) -> tuple[Any, ...] | None:
        if value.is_inference():
            return None
        return (
            value._version,
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
        )

    def get(self, value: Any, fact: str) -> Any:
        entry = self._entries.get(id(value))
        if entry is None or entry[0]() is not value:
            return None
        key = self._key(value)
        if key is None or key != entry[1]:
            return None
        return entry[2].get(fact)

    def put(self, value: Any, fact: str, result: Any) -> None:
        import weakref

        key = self._key(value)
        if key is None:
            return
        ident = id(value)
        entry = self._entries.get(ident)
        if entry is not None and entry[0]() is value and entry[1] == key:
            entry[2][fact] = result
            return

        def forget(ref: Any, ident: int = ident) -> None:
            current = self._entries.get(ident)
            if current is not None and current[0] is ref:
                self._entries.pop(ident, None)

        self._entries[ident] = (weakref.ref(value, forget), key, {fact: result})

    def __len__(self) -> int:
        return len(self._entries)


TENSOR_FACTS = TensorFacts()


def _describe_tensor(value: Any) -> dict[str, Any]:
    raw = value.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8)
    content_digest = hashlib.sha256(raw.numpy()).hexdigest()  # the buffer, not a bytes copy
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


def _copy_descriptor(described: dict[str, Any]) -> dict[str, Any]:
    return {**described, "shape": list(described["shape"]), "stride": list(described["stride"])}


def describe(value: Any) -> Any:
    """Return a typed, canonical content descriptor.

    Tags are intentional: JSON alone cannot distinguish tuples from lists and
    Python considers values such as ``True`` and ``1`` equal.  State and
    operation seals must preserve those distinctions.
    """
    return _describe(value, False)


def describe_remembered(value: Any) -> Any:
    """``describe(value)``, hashing a tensor only when ``TENSOR_FACTS`` holds no descriptor
    for it (same object, same version counter, pointer and layout): for sealing a state whose
    tensors are shared with earlier seals (a layer transition replaces one K/V layer of L).
    The descriptor is the very one ``describe`` gives; verification paths use ``describe``."""
    return _describe(value, True)


def _describe(value: Any, remember: bool) -> Any:
    if tensor(value):
        if not remember:
            return _describe_tensor(value)
        known = TENSOR_FACTS.get(value, "describe")
        if known is None:
            known = _describe_tensor(value)
            TENSOR_FACTS.put(value, "describe", known)
        return _copy_descriptor(known)
    if isinstance(value, Mapping):
        return describe_mapping({key: _describe(item, remember) for key, item in value.items()})
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_describe(item, remember) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_describe(item, remember) for item in value]}
    return _scalar(value)


def describe_mapping(described: Mapping[Any, Any]) -> dict[str, Any]:
    """``describe`` of a mapping whose items are already described (``{key: describe(item)}``).

    Lets a caller that holds the descriptor of every item seal the mapping without hashing the
    items again; ``describe`` of a mapping is defined through this function, so both agree.
    """
    items = [{"key": describe(key), "value": item} for key, item in described.items()]
    if all(type(key) is str and _PLAIN_KEY.fullmatch(key) for key in described):
        # canonical() of a plain string key is '{"kind":"scalar","type":"str","value":"KEY"}':
        # with no character that JSON escapes and none below '"', the byte order of those
        # encodings is the code-point order of the keys, so sort without encoding them
        items.sort(key=lambda item: item["key"]["value"])
    else:
        items.sort(key=lambda item: canonical(item["key"]))
    return {"kind": "mapping", "items": items}


# printable ASCII above '"' except '\\': canonical JSON leaves them as they are, and each sorts
# after the closing quote, so a key that is a prefix of another sorts first in both orders
_PLAIN_KEY = __import__("re").compile(r"[#-\[\]-~]*")


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
