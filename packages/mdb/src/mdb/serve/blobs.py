"""Read recorded tensors with numpy only (no torch), verifying every digest on the way.

Two sources in a project store (saturn_pub ``LocalStore`` layout):

* ``tensors/<ref>.safetensors``: single tensors by content address (``put_tensor``). Parsed
  here with the same checks as ``LocalStore._read_tensor`` (file sha256 = ref, payload sha256 =
  header metadata) because ``get_tensor`` needs torch for bfloat16; bf16 is widened exactly to
  float32 (it is the top half of a float32).
* cut slots: ``cuts/<cut>.json`` -> slot page ``pages/<page>.json`` -> ``blobs/<blob>.safetensors``.
  ``LocalStore.load`` hydrates these through ``safetensors.torch``; this module parses the same
  bytes with numpy and checks the tensor's sha256 against the page's sealed value descriptor.

Arrays above ``MAX_ELEMENTS`` are refused, never partially read: the safetensors header is read
first (a few KiB) and the element count is checked against the limit BEFORE the payload is read
or hashed, so a huge blob costs one small read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MAX_ELEMENTS = 1 << 26
MAX_HEADER = 16 << 20  # a safetensors header above this is malformed, not worth reading
MAX_FILE = 1 << 30  # largest blob file this server will read and hash
MAX_JSON = 16 << 20  # largest cut manifest / slot page this server will parse

_CODES = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
          "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8",
          "U16": "uint16", "U32": "uint32", "U64": "uint64", "BOOL": "bool"}


class BlobError(LookupError):
    """A tensor cannot be produced. ``refused``: found but failing verification; ``limit``:
    over the size limit (refused before any payload was read)."""

    def __init__(self, reason, *, refused=False, limit=False):
        super().__init__(reason)
        self.reason, self.refused, self.limit = reason, refused or limit, limit


def _hex(value, label):
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise BlobError(f"{label} is not a sha256 hex id")
    return value


def _store_file(path: Path) -> bool:
    """True when ``path`` is a regular file whose real path stays under the store root (the
    grandparent of every ``<root>/<dir>/<name>`` file). A symlink pointing outside the store is
    treated exactly like a missing file: same answer whether or not its target exists."""
    try:
        real = path.resolve(strict=True)
        return real.is_file() and real.is_relative_to(path.parent.parent.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def _read_verified(path: Path, identifier: str, label: str, max_bytes: int = MAX_FILE) -> bytes:
    try:
        size = path.stat().st_size if _store_file(path) else None
    except OSError:
        size = None
    if size is None:
        raise BlobError(f"{label} {identifier[:12]} is not in the store")
    if size > max_bytes:
        raise BlobError(f"{label} {identifier[:12]} is {size} bytes (> {max_bytes}); refused "
                        "before reading", limit=True)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise BlobError(f"{label} {identifier[:12]} cannot be read ({error.strerror})") from error
    if hashlib.sha256(raw).hexdigest() != identifier:
        raise BlobError(f"{label} {identifier[:12]} digest mismatch", refused=True)
    return raw


def _json_file(path: Path, label: str, *, verify: str | None = None):
    """The parsed JSON of a small store file; every failure is a ``BlobError``."""
    if verify is not None:
        raw = _read_verified(path, verify, label, MAX_JSON)
    else:
        try:
            if not _store_file(path):
                raise BlobError(f"{label} {path.stem[:12]} is not in the store")
            if path.stat().st_size > MAX_JSON:
                raise BlobError(f"{label} {path.stem[:12]} is over {MAX_JSON} bytes", limit=True)
            raw = path.read_bytes()
        except OSError as error:
            raise BlobError(f"{label} {path.stem[:12]} cannot be read "
                            f"({error.strerror})") from error
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError) as error:
        raise BlobError(f"{label} {path.stem[:12]} is not JSON", refused=True) from error
    if not isinstance(value, dict):
        raise BlobError(f"{label} {path.stem[:12]} is not a JSON object", refused=True)
    return value


def _count(shape, name):
    if not isinstance(shape, list) or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape):
        raise BlobError(f"tensor {name!r} has a malformed shape", refused=True)
    count = 1
    for n in shape:
        count *= n
    return count


def _precheck(path: Path, name: str, label: str):
    """Read only the safetensors header of ``path`` and refuse an over-limit tensor before the
    payload is read or hashed. A missing file is left to ``_read_verified``."""
    if not _store_file(path):
        return
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
            if len(head) < 8:
                return
            size = int.from_bytes(head, "little")
            if size < 2 or size > MAX_HEADER:
                return  # the verified read reports it as malformed
            header = json.loads(handle.read(size))
    except (OSError, ValueError, RecursionError):
        return
    entry = header.get(name) if isinstance(header, dict) else None
    if not isinstance(entry, dict):
        return
    count = _count(entry.get("shape"), name)
    if count > MAX_ELEMENTS:
        raise BlobError(f"{label} tensor has {count} elements (> {MAX_ELEMENTS}); refused "
                        "before reading its payload", limit=True)


def _from_bytes(payload: bytes, dtype: str, shape):
    import numpy as np

    count = 1
    for n in shape:
        count *= n
    if count > MAX_ELEMENTS:
        raise BlobError(f"tensor has {count} elements (> {MAX_ELEMENTS}); refused", limit=True)
    if dtype == "bfloat16":
        raw = np.frombuffer(payload, dtype="<u2").astype(np.uint32) << 16
        return raw.view(np.float32).reshape(shape)
    little = np.dtype(dtype).newbyteorder("<")
    return np.frombuffer(payload, dtype=little).reshape(shape).astype(np.dtype(dtype))


def _safetensors(raw: bytes):
    if len(raw) < 8:
        raise BlobError("safetensors file is truncated", refused=True)
    size = int.from_bytes(raw[:8], "little")
    if size < 2 or 8 + size > len(raw):
        raise BlobError("safetensors header is malformed", refused=True)
    try:
        header = json.loads(raw[8:8 + size])
    except ValueError as error:
        raise BlobError("safetensors header is not JSON", refused=True) from error
    return header, raw[8 + size:]


def _entry(header, data, name):
    entry = header.get(name)
    if not isinstance(entry, dict) or entry.get("dtype") not in _CODES:
        raise BlobError(f"tensor {name!r} is absent from its blob", refused=True)
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) for v in offsets):
        raise BlobError(f"tensor {name!r} has malformed offsets", refused=True)
    begin, end = offsets
    _count(entry.get("shape"), name)
    if not 0 <= begin <= end <= len(data):
        raise BlobError("tensor offsets fall outside the blob", refused=True)
    return _CODES[entry["dtype"]], list(entry["shape"]), data[begin:end]


def tensor_blob(store_root: Path, ref: str):
    """``(array, info)`` for a ``put_tensor`` blob."""
    ref = _hex(ref, "tensor ref")
    path = Path(store_root) / "tensors" / f"{ref}.safetensors"
    _precheck(path, "tensor", "tensor")
    raw = _read_verified(path, ref, "tensor")
    header, data = _safetensors(raw)
    meta = header.get("__metadata__") or {}
    dtype, shape, payload = _entry(header, data, "tensor")
    if hashlib.sha256(payload).hexdigest() != meta.get("sha256"):
        raise BlobError("tensor payload digest mismatch", refused=True)
    array = _from_bytes(payload, dtype, shape)
    return array, {"shape": shape, "dtype": dtype, "sha256": meta["sha256"],
                   "via": [f"tensors/{ref[:12]}", "payload sha256 verified"]}


def _navigate(encoded, described, item, slot):
    if encoded.get("kind") == "tensor":
        if item is not None:
            raise BlobError(f"slot {slot!r} holds one tensor; no item index")
        return encoded["name"], described
    if encoded.get("kind") in ("list", "tuple"):
        items, descs = encoded.get("items") or [], (described or {}).get("items") or []
        if item is None:
            raise BlobError(f"slot {slot!r} is a sequence of {len(items)}; give an item index")
        if not 0 <= item < len(items) or items[item].get("kind") != "tensor":
            raise BlobError(f"slot {slot!r} has no tensor at item {item}")
        return items[item]["name"], descs[item] if item < len(descs) else None
    if encoded.get("kind") == "scalar":
        raise BlobError(f"slot {slot!r} holds no tensor at this cut (scalar "
                        f"{encoded.get('value')!r})")
    raise BlobError(f"slot {slot!r} has unsupported kind {encoded.get('kind')!r}")


def _slot_page(root: Path, cut: str, slot: str):
    """``(page_id, page)`` of a retained slot after checking the cut manifest and page digests."""
    from saturn_pub.values import digest

    cut = _hex(cut, "cut id")
    metadata = _json_file(root / "cuts" / f"{cut}.json", "cut")
    try:
        sealed = metadata.get("fingerprint") == cut and digest(metadata.get("manifest")) == cut
    except (TypeError, ValueError) as error:
        raise BlobError("cut manifest is malformed", refused=True) from error
    if not sealed:
        raise BlobError("cut manifest digest mismatch", refused=True)
    slots = metadata.get("slots")
    page_id = slots.get(slot) if isinstance(slots, dict) else None
    if page_id is None:
        raise BlobError(f"cut {cut[:12]} retains no slot {slot!r}")
    page = _json_file(root / "pages" / f"{_hex(page_id, 'page')}.json", "slot page",
                      verify=page_id)
    return page_id, page


def _navigated(page, item, slot):
    try:
        return _navigate(page.get("encoded") or {}, page.get("value"), item, slot)
    except (KeyError, TypeError, AttributeError) as error:
        raise BlobError(f"slot {slot!r} has a malformed page", refused=True) from error


def cut_slot(store_root: Path, cut: str, slot: str, item: int | None = None):
    """``(array, info)`` for one tensor of a retained cut slot."""
    root = Path(store_root)
    page_id, page = _slot_page(root, cut, slot)
    name, described = _navigated(page, item, slot)
    blob = page.get("tensor_blob")
    if blob is None:
        raise BlobError(f"slot {slot!r} page has no tensor blob", refused=True)
    path = root / "blobs" / f"{_hex(blob, 'blob')}.safetensors"
    _precheck(path, name, "slot")
    header, data = _safetensors(_read_verified(path, blob, "slot blob"))
    dtype, shape, payload = _entry(header, data, name)
    described = described if isinstance(described, dict) else {}
    expected = described.get("sha256")
    if expected is None or hashlib.sha256(payload).hexdigest() != expected:
        raise BlobError("slot tensor bytes differ from the sealed descriptor", refused=True)
    if list(described.get("shape") or shape) != shape:
        raise BlobError("slot tensor shape differs from the sealed descriptor", refused=True)
    array = _from_bytes(payload, dtype, shape)
    return array, {"shape": shape, "dtype": dtype, "sha256": expected,
                   "via": [f"cut {cut[:12]}", f"slot {slot}" + ("" if item is None else
                                                                f"[{item}]"),
                           f"page {page_id[:12]}", f"blob {blob[:12]}",
                           "tensor sha256 verified against the sealed descriptor"]}


def slot_info(store_root: Path, cut: str, slot: str, item: int | None = None):
    """Shape/dtype of a cut slot tensor from its sealed page descriptor (no tensor bytes read).

    Raises ``BlobError`` (never another exception) when the cut, page or tensor blob is missing:
    the caller reports that as unresolved."""
    root = Path(store_root)
    _, page = _slot_page(root, cut, slot)
    _, described = _navigated(page, item, slot)
    blob = page.get("tensor_blob")
    if blob is None or not _store_file(root / "blobs" / f"{_hex(blob, 'blob')}.safetensors"):
        raise BlobError(f"slot {slot!r} tensor blob is not in the store")
    described = described if isinstance(described, dict) else {}
    return {"shape": described.get("shape"), "dtype": str(described.get("dtype", "")).replace(
        "torch.", "")}
