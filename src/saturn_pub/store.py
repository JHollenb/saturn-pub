"""Verified local custody with per-slot content-addressed pages, never pickle."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core import Receipt, StateCut
from .values import canonical, clone, describe, digest, legacy_describe_v1, tensor

_LEGACY_CAVEAT = (
    "saturn-pub-statecut-v1 typing is non-authoritative historical evidence: "
    "lists/tuples and JSON scalar types were not independently sealed"
)


def _safe_id(value: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected a lowercase SHA-256 identifier")
    return value


def _publish(path: Path, data: bytes) -> bool:
    if path.suffix == ".safetensors":
        from .shared_store import publish_configured

        shared = publish_configured(path, data)
        if shared is not None:
            return shared
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        pass
    else:
        if existing != data:
            raise ValueError("existing artifact does not match its content address")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    admitted = False
    try:
        try:
            os.link(temporary, path)
            admitted = True
            try:
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError:
                # Some filesystems do not expose directory fsync; the file itself
                # was already flushed before the immutable link was published.
                pass
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("existing artifact does not match its content address")
    finally:
        temporary.unlink()
    return admitted


def _scalar_type(value: Any) -> str:
    if value is None:
        return "none"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        canonical(value)
        return "float"
    if type(value) is str:
        return "str"
    raise ValueError(f"unsupported durable value: {type(value).__name__}")


def _encode(value: Any, tensors: dict[str, Any]) -> Any:
    if tensor(value):
        name = f"tensor_{len(tensors)}"
        tensors[name] = value.detach().cpu().contiguous().clone()
        return {
            "kind": "tensor",
            "name": name,
            "shape": list(value.shape),
            "stride": list(value.stride()),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("durable state mappings require string keys")
        return {
            "kind": "mapping",
            "items": {key: _encode(item, tensors) for key, item in sorted(value.items())},
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode(item, tensors) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode(item, tensors) for item in value]}
    return {"kind": "scalar", "type": _scalar_type(value), "value": value}


def _decode(value: Any, tensors: dict[str, Any], used: set[str] | None = None) -> Any:
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise ValueError("invalid stored value descriptor")
    used = set() if used is None else used
    kind = value["kind"]
    if kind == "tensor":
        name = value.get("name")
        if not isinstance(name, str) or name not in tensors or name in used:
            raise ValueError("invalid or repeated tensor page reference")
        used.add(name)
        stored = tensors[name]
        shape = value.get("shape")
        stride = value.get("stride")
        if (
            not isinstance(shape, list)
            or not isinstance(stride, list)
            or len(shape) != len(stride)
            or any(type(size) is not int or size < 0 for size in shape)
            or any(type(step) is not int or step < 0 for step in stride)
            or tuple(shape) != tuple(stored.shape)
        ):
            raise ValueError("invalid tensor layout descriptor")
        # Accept permutations of a compact dense layout. This bounds physical
        # storage to numel and excludes overlaps, broadcast views, and sparse
        # as_strided layouts whose hydration could amplify resources.
        expected = 1
        for step, size in sorted(
            ((step, size) for size, step in zip(shape, stride) if size > 1),
            key=lambda item: item[0],
        ):
            if step != expected:
                raise ValueError("tensor layout is not non-overlapping dense")
            expected *= size
        if stored.numel() and expected != stored.numel():
            raise ValueError("tensor layout resource contract mismatch")
        if tuple(stride) == tuple(stored.stride()):
            return stored
        restored = __import__("torch").empty_strided(
            tuple(shape), tuple(stride), dtype=stored.dtype, device=stored.device
        )
        restored.copy_(stored)
        return restored
    if kind == "mapping":
        items = value.get("items")
        if not isinstance(items, dict) or any(not isinstance(key, str) for key in items):
            raise ValueError("invalid stored mapping")
        return {key: _decode(item, tensors, used) for key, item in items.items()}
    if kind in ("list", "tuple"):
        encoded_items = value.get("items")
        if not isinstance(encoded_items, list):
            raise ValueError("invalid stored sequence")
        items = [_decode(item, tensors, used) for item in encoded_items]
        return tuple(items) if kind == "tuple" else items
    if kind == "scalar":
        scalar = value.get("value")
        if value.get("type") != _scalar_type(scalar):
            raise ValueError("stored scalar type tag mismatch")
        return scalar
    raise ValueError("unknown stored value kind")


def _decode_v1(value: Any, tensors: dict[str, Any]) -> Any:
    kind = value["kind"]
    if kind == "tensor":
        return tensors[value["name"]]
    if kind == "mapping":
        return {key: _decode_v1(item, tensors) for key, item in value["items"].items()}
    if kind in ("list", "tuple"):
        items = [_decode_v1(item, tensors) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "scalar":
        return value["value"]
    raise ValueError("unknown legacy stored value kind")


def _move(value: Any, device: str) -> Any:
    if tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return clone(value)


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._last_admission: dict[str, Any] | None = None

    @property
    def last_admission(self) -> dict[str, Any] | None:
        return None if self._last_admission is None else clone(self._last_admission)

    def _save_slot(self, value: Any) -> tuple[str, set[str], int, int]:
        tensors: dict[str, Any] = {}
        encoded = _encode(value, tensors)
        blob_id = None
        blob_bytes = b""
        admitted = 0
        if tensors:
            from safetensors.torch import save

            blob_bytes = save(tensors)
            blob_id = hashlib.sha256(blob_bytes).hexdigest()
            if _publish(self.root / "blobs" / f"{blob_id}.safetensors", blob_bytes):
                admitted += len(blob_bytes)
        page = canonical(
            {
                "schema": "saturn-pub-slot-page-v1",
                "encoded": encoded,
                "tensor_blob": blob_id,
                "value": describe(value),
            }
        )
        page_id = hashlib.sha256(page).hexdigest()
        if _publish(self.root / "pages" / f"{page_id}.json", page):
            admitted += len(page)
        return (
            page_id,
            (set() if blob_id is None else {blob_id}),
            len(page) + len(blob_bytes),
            admitted,
        )

    def save(self, cut: StateCut) -> str:
        cut.verify()
        slots: dict[str, str] = {}
        blobs: set[str] = set()
        logical_bytes = admitted_bytes = 0
        for name, value in sorted(cut.payload.items()):
            if not isinstance(name, str):
                raise ValueError("durable state slots require string names")
            page_id, physical, logical, admitted = self._save_slot(value)
            slots[name] = page_id
            blobs.update(physical)
            logical_bytes += logical
            admitted_bytes += admitted
        metadata = {
            "schema": "saturn-pub-local-cut-v2",
            "manifest": cut.manifest(),
            "fingerprint": cut.fingerprint,
            "logical_parent": cut.parent,
            "slots": slots,
            "physical_blobs": sorted(blobs),
            "execution_point": None if cut.execution_point is None else dict(cut.execution_point),
            "surface": None if cut.surface is None else dict(cut.surface),
        }
        body = canonical(metadata)
        logical_bytes += len(body)
        if _publish(self.root / "cuts" / f"{cut.fingerprint}.json", body):
            admitted_bytes += len(body)
        self._last_admission = {
            "cut": cut.fingerprint,
            "logical_bytes": logical_bytes,
            "admitted_bytes": admitted_bytes,
            "deduplicated_bytes": logical_bytes - admitted_bytes,
            "slot_pages": len(slots),
            "physical_blobs": len(blobs),
        }
        return cut.fingerprint

    def save_report(self, cut: StateCut) -> dict[str, Any]:
        self.save(cut)
        assert self._last_admission is not None
        return clone(self._last_admission)

    def _read_page(self, page_id: str) -> tuple[Any, dict[str, Any]]:
        identifier = _safe_id(page_id)
        raw = (self.root / "pages" / f"{identifier}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != identifier:
            raise ValueError("slot page digest mismatch")
        page = json.loads(raw)
        if page.get("schema") != "saturn-pub-slot-page-v1":
            raise ValueError("unknown slot page schema")
        tensors: dict[str, Any] = {}
        blob_id = page.get("tensor_blob")
        if blob_id is not None:
            from safetensors.torch import load

            blob_id = _safe_id(blob_id)
            blob = (self.root / "blobs" / f"{blob_id}.safetensors").read_bytes()
            if hashlib.sha256(blob).hexdigest() != blob_id:
                raise ValueError("tensor blob digest mismatch")
            tensors = load(blob)
        used: set[str] = set()
        value = _decode(page.get("encoded"), tensors, used)
        if used != set(tensors):
            raise ValueError("slot page contains unreferenced tensor data")
        if describe(value) != page.get("value"):
            raise ValueError("slot page descriptor does not match hydrated value")
        return value, page

    def _load_v2(self, identifier: str, data: dict[str, Any], *, device: str) -> StateCut:
        manifest = data.get("manifest")
        if not isinstance(manifest, dict) or digest(manifest) != identifier:
            raise ValueError("StateCut manifest digest mismatch")
        if data.get("fingerprint") != identifier:
            raise ValueError("stored StateCut fingerprint mismatch")
        if data.get("logical_parent") != manifest.get("parent"):
            raise ValueError("logical ancestry does not match the StateCut manifest")
        slots = data.get("slots")
        if not isinstance(slots, dict) or any(
            not isinstance(name, str) or not isinstance(page, str) for name, page in slots.items()
        ):
            raise ValueError("invalid StateCut slot inventory")
        payload: dict[str, Any] = {}
        observed_blobs: set[str] = set()
        for name, page_id in slots.items():
            value, page = self._read_page(page_id)
            payload[name] = value
            if page.get("tensor_blob") is not None:
                observed_blobs.add(page["tensor_blob"])
        if sorted(observed_blobs) != data.get("physical_blobs"):
            raise ValueError("physical blob inventory mismatch")
        cut = StateCut(
            manifest["model_identity"],
            manifest["execution"],
            manifest["boundary"],
            manifest["parent"],
            payload,
            data.get("execution_point"),
            data.get("surface"),
        )
        if cut.fingerprint != identifier or cut.manifest() != manifest:
            raise ValueError("stored descriptor does not match hydrated payload")
        cut.verify()
        if device != "cpu":
            cut = StateCut(
                cut.model_identity,
                cut.execution,
                cut.boundary,
                cut.parent,
                _move(cut.payload, device),
                cut.execution_point,
                cut.surface,
            )
            if cut.fingerprint != identifier:
                raise ValueError("device hydration changed StateCut identity")
        return cut

    def _load_v1(self, identifier: str, data: dict[str, Any], *, device: str) -> StateCut:
        warnings.warn(_LEGACY_CAVEAT, RuntimeWarning, stacklevel=3)
        manifest = data.get("manifest")
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != "saturn-pub-statecut-v1"
            or digest(manifest) != identifier
            or data.get("fingerprint") != identifier
        ):
            raise ValueError("legacy StateCut manifest digest mismatch")
        tensors: dict[str, Any] = {}
        if data.get("blob") is not None:
            from safetensors.torch import load

            blob_id = _safe_id(data["blob"])
            blob = (self.root / "blobs" / f"{blob_id}.safetensors").read_bytes()
            if hashlib.sha256(blob).hexdigest() != blob_id:
                raise ValueError("legacy tensor blob digest mismatch")
            tensors = load(blob)
        payload = _decode_v1(data.get("encoded"), tensors)
        expected = {
            "schema": "saturn-pub-statecut-v1",
            "model_identity": manifest["model_identity"],
            "execution": manifest["execution"],
            "boundary": manifest["boundary"],
            "parent": manifest["parent"],
            "payload": legacy_describe_v1(payload),
        }
        if expected != manifest:
            raise ValueError("legacy descriptor does not match hydrated payload")
        return StateCut(
            manifest["model_identity"],
            manifest["execution"],
            manifest["boundary"],
            manifest["parent"],
            _move(payload, device),
            {
                "authority": "non-authoritative-historical-v1",
                "source_fingerprint": identifier,
                "caveat": _LEGACY_CAVEAT,
            },
            None,
        )

    def load(self, identifier: str, *, device: str = "cpu") -> StateCut:
        identifier = _safe_id(identifier)
        data = json.loads((self.root / "cuts" / f"{identifier}.json").read_bytes())
        if data.get("schema") == "saturn-pub-local-cut-v2":
            return self._load_v2(identifier, data, device=device)
        return self._load_v1(identifier, data, device=device)

    def receipt(self, receipt: Receipt) -> str:
        _publish(
            self.root / "receipts" / f"{receipt.fingerprint}.json", canonical(receipt.to_dict())
        )
        return receipt.fingerprint

    def inspect(self, identifier: str) -> dict[str, Any]:
        identifier = _safe_id(identifier)
        data = json.loads((self.root / "cuts" / f"{identifier}.json").read_bytes())
        manifest = data.get("manifest")
        if not isinstance(manifest, dict) or digest(manifest) != identifier:
            raise ValueError("manifest digest mismatch")
        if data.get("fingerprint") != identifier:
            raise ValueError("stored StateCut fingerprint mismatch")
        if manifest.get("schema") == "saturn-pub-statecut-v1":
            warnings.warn(_LEGACY_CAVEAT, RuntimeWarning, stacklevel=2)
            return {**manifest, "historical_typing_caveat": _LEGACY_CAVEAT}
        if data.get("schema") != "saturn-pub-local-cut-v2":
            raise ValueError("unknown local cut schema")
        return manifest

    def export_graph(
        self, heads: Mapping[str, str], *, include_data: bool = False
    ) -> dict[str, Any]:
        if not heads or any(not isinstance(name, str) or not name for name in heads):
            raise ValueError("export heads require non-empty string names")
        pending = list(heads.values())
        cuts: dict[str, Any] = {}
        page_ids: set[str] = set()
        blob_ids: set[str] = set()
        while pending:
            identifier = _safe_id(pending.pop())
            if identifier in cuts:
                continue
            path = self.root / "cuts" / f"{identifier}.json"
            if not path.exists():
                raise ValueError(f"local ancestor is unavailable: {identifier}")
            data = json.loads(path.read_bytes())
            if (
                data.get("schema") != "saturn-pub-local-cut-v2"
                or digest(data["manifest"]) != identifier
            ):
                raise ValueError("ReplayBundle exports require authoritative verified v2 cuts")
            cuts[identifier] = data
            page_ids.update(data["slots"].values())
            blob_ids.update(data["physical_blobs"])
            parent = data.get("logical_parent")
            if parent is not None and (self.root / "cuts" / f"{parent}.json").exists():
                pending.append(parent)
        objects: dict[str, Any] = {"pages": {}, "blobs": {}}
        if include_data:
            for identifier in sorted(page_ids):
                raw = (self.root / "pages" / f"{_safe_id(identifier)}.json").read_bytes()
                if hashlib.sha256(raw).hexdigest() != identifier:
                    raise ValueError("cannot export a corrupt slot page")
                objects["pages"][identifier] = base64.b64encode(raw).decode("ascii")
            for identifier in sorted(blob_ids):
                raw = (self.root / "blobs" / f"{_safe_id(identifier)}.safetensors").read_bytes()
                if hashlib.sha256(raw).hexdigest() != identifier:
                    raise ValueError("cannot export a corrupt tensor blob")
                objects["blobs"][identifier] = base64.b64encode(raw).decode("ascii")
        return {
            "schema": "saturn-pub-local-graph-v1",
            "heads": dict(heads),
            "cuts": cuts,
            "objects": objects,
            "data_included": include_data,
        }

    def import_graph(self, graph: Mapping[str, Any], *, allow_data: bool = False) -> None:
        if graph.get("schema") != "saturn-pub-local-graph-v1":
            raise ValueError("unknown local graph schema")
        if not graph.get("data_included") or not allow_data:
            raise ValueError("cut data import requires data_included and allow_data=True")
        cuts = dict(graph.get("cuts", {}))
        objects = dict(graph.get("objects", {}))
        pages = dict(objects.get("pages", {}))
        blobs = dict(objects.get("blobs", {}))
        decoded_pages: dict[str, bytes] = {}
        decoded_blobs: dict[str, bytes] = {}
        heads = dict(graph.get("heads", {}))
        if not heads or any(identifier not in cuts for identifier in heads.values()):
            raise ValueError("imported graph head is absent from its cut inventory")
        for identifier, encoded in pages.items():
            raw = base64.b64decode(encoded, validate=True)
            if hashlib.sha256(raw).hexdigest() != _safe_id(identifier):
                raise ValueError("imported slot page digest mismatch")
            decoded_pages[identifier] = raw
        for identifier, encoded in blobs.items():
            raw = base64.b64decode(encoded, validate=True)
            if hashlib.sha256(raw).hexdigest() != _safe_id(identifier):
                raise ValueError("imported tensor blob digest mismatch")
            decoded_blobs[identifier] = raw
        for identifier, data in cuts.items():
            identifier = _safe_id(identifier)
            if data.get("schema") != "saturn-pub-local-cut-v2":
                raise ValueError("imported graph contains a non-v2 cut")
            if data.get("fingerprint") != identifier or digest(data.get("manifest")) != identifier:
                raise ValueError("imported cut manifest digest mismatch")
            if not set(data.get("slots", {}).values()).issubset(decoded_pages):
                raise ValueError("imported graph is missing a slot page")
            if not set(data.get("physical_blobs", ())).issubset(decoded_blobs):
                raise ValueError("imported graph is missing a tensor blob")
        hydrated_pages: dict[str, Any] = {}
        for identifier, raw in decoded_pages.items():
            page = json.loads(raw)
            if page.get("schema") != "saturn-pub-slot-page-v1":
                raise ValueError("imported graph contains an unknown slot page")
            tensors: dict[str, Any] = {}
            blob_id = page.get("tensor_blob")
            if blob_id is not None:
                if blob_id not in decoded_blobs:
                    raise ValueError("imported slot page references a missing tensor blob")
                from safetensors.torch import load

                tensors = load(decoded_blobs[blob_id])
            used: set[str] = set()
            value = _decode(page.get("encoded"), tensors, used)
            if used != set(tensors) or describe(value) != page.get("value"):
                raise ValueError("imported slot page does not hydrate to its descriptor")
            hydrated_pages[identifier] = value
        for identifier, data in cuts.items():
            manifest = data["manifest"]
            payload = {name: hydrated_pages[page] for name, page in data["slots"].items()}
            observed_blobs = sorted(
                {
                    json.loads(decoded_pages[page]).get("tensor_blob")
                    for page in data["slots"].values()
                    if json.loads(decoded_pages[page]).get("tensor_blob") is not None
                }
            )
            if observed_blobs != data["physical_blobs"]:
                raise ValueError("imported cut physical blob inventory mismatch")
            cut = StateCut(
                manifest["model_identity"],
                manifest["execution"],
                manifest["boundary"],
                manifest["parent"],
                payload,
                data.get("execution_point"),
                data.get("surface"),
            )
            if cut.fingerprint != identifier or cut.manifest() != manifest:
                raise ValueError("imported cut does not hydrate to its identity")
        for identifier, raw in decoded_blobs.items():
            _publish(self.root / "blobs" / f"{identifier}.safetensors", raw)
        for identifier, raw in decoded_pages.items():
            _publish(self.root / "pages" / f"{identifier}.json", raw)
        for identifier, data in cuts.items():
            _publish(self.root / "cuts" / f"{identifier}.json", canonical(data))

    # Single tensors by content address (occurrence values). Each is one safetensors file,
    # tensors/<sha256 of the file>.safetensors, holding one tensor named "tensor" with
    # little-endian payload bytes; __metadata__ records the payload sha256. The file is written
    # here byte for byte (canonical header, space-padded to 8 bytes), so numpy arrays need no
    # torch and bfloat16 round-trips through torch only.
    #
    # Temporary files (``tensors/tmp*``, unique per write: tempfile picks a fresh random name)
    # that a killed process left behind are removed by the next successful put once they are
    # older than ``_STALE_TMP_SECONDS``; a file a live process is writing is younger than that.
    # A blob hard-linked into a shared pool (``shared_store``) is never repaired from here: if
    # it is corrupt the pooled object is, and the pool is repaired there.

    def put_tensor(self, value: Any) -> dict[str, Any]:
        dtype, shape, payload = _tensor_payload(value)
        payload_sha = hashlib.sha256(payload).hexdigest()
        header = {
            "__metadata__": {"schema": _TENSOR_SCHEMA, "sha256": payload_sha},
            "tensor": {
                "dtype": _SAFETENSORS_DTYPES[dtype][0],
                "shape": shape,
                "data_offsets": [0, len(payload)],
            },
        }
        raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        raw_header += b" " * (-len(raw_header) % 8)
        data = len(raw_header).to_bytes(8, "little") + raw_header + payload
        ref = hashlib.sha256(data).hexdigest()
        path = self.root / "tensors" / f"{ref}.safetensors"
        self._repair_tensor(path, data)
        _publish(path, data)
        self._remove_stale_temporaries(path.parent)
        return {
            "ref": ref,
            "shape": shape,
            "dtype": dtype,
            "sha256": payload_sha,
            "bytes": len(payload),
        }

    @staticmethod
    def _remove_stale_temporaries(directory: Path) -> None:
        """Delete ``tmp*`` files older than ``_STALE_TMP_SECONDS`` (best effort, never raises)."""
        import time

        cutoff = time.time() - _STALE_TMP_SECONDS
        try:
            entries = list(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.name.startswith("tmp"):
                continue
            try:
                if entry.is_file() and not entry.is_symlink() and entry.stat().st_mtime < cutoff:
                    entry.unlink(missing_ok=True)
            except OSError:
                continue

    @staticmethod
    def _repair_tensor(path: Path, data: bytes) -> None:
        """Replace an on-disk blob whose bytes differ from its content address, atomically.

        The correct bytes are written to a temporary file in the same directory, fsynced and
        renamed over the damaged file, so a reader sees the old or the new file, never a mix.
        A blob that is a directory is a ``ValueError``; a damaged blob hard-linked into a shared
        pool is a ``ValueError`` naming the pooled object (repair it in the pool).
        """
        if path.is_dir():
            raise ValueError(f"tensor blob path {path} is a directory, not a file")
        try:
            existing = path.read_bytes()
        except FileNotFoundError:
            return
        if existing == data:
            return
        LocalStore._refuse_pooled_blob(path, data)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @staticmethod
    def _refuse_pooled_blob(path: Path, data: bytes) -> None:
        """A corrupt blob that is the same inode as its shared-pool object is not repaired here."""
        try:
            if path.stat().st_nlink < 2:
                return
            from .shared_store import configured_store

            pool = configured_store(path)
            if pool is None:
                return
            pooled = pool.object_path(hashlib.sha256(data).hexdigest())
            if pooled.exists() and os.path.samefile(path, pooled):
                raise ValueError(
                    f"the pooled object {pooled} is corrupt (it is hard-linked at {path}); "
                    "repair it in the shared pool: remove the object there and publish the "
                    "tensor again; this store does not repair a pooled blob"
                )
        except ValueError:
            raise
        except OSError:
            return

    @staticmethod
    def _tensor_ref(ref: Any) -> str:
        if not isinstance(ref, str):
            raise ValueError(
                f"expected a lowercase SHA-256 identifier (text), not {type(ref).__name__}"
            )
        return _safe_id(ref)

    def has_tensor(self, ref: str, *, verify: bool = False) -> bool:
        """Whether the blob exists; with ``verify`` also that its size, header and hash are sound."""
        path = self.root / "tensors" / f"{self._tensor_ref(ref)}.safetensors"
        if not path.is_file():
            return False
        if not verify:
            return True
        try:
            self._read_tensor(ref)
        except (ValueError, OSError):
            return False
        return True

    def tensor_info(self, ref: str) -> dict[str, Any]:
        """Verified descriptor of a stored tensor, without materialising it."""
        dtype, shape, payload_sha, payload = self._read_tensor(ref)
        return {
            "ref": ref,
            "shape": shape,
            "dtype": dtype,
            "sha256": payload_sha,
            "bytes": len(payload),
        }

    def get_tensor(
        self, ref: str, *, device: str | None = None, framework: str | None = None
    ) -> Any:
        """Rebuild a stored tensor after verifying both digests.

        Returns a numpy array unless ``device`` is given, ``framework="torch"``, or the dtype
        has no numpy form (bfloat16): then a torch tensor (torch is imported only then). The
        numpy array is a writable copy (never a view of the verified bytes); so is the torch one.
        """
        if framework not in (None, "numpy", "torch"):
            raise ValueError("framework is 'numpy' or 'torch'")
        dtype, shape, _, payload = self._read_tensor(ref)
        if framework == "numpy" and (device is not None or dtype == "bfloat16"):
            raise ValueError("a bfloat16 tensor or a device placement needs framework='torch'")
        if framework == "torch" or device is not None or dtype == "bfloat16":
            import torch

            torch_dtype = getattr(torch, dtype)
            if not payload:
                restored = torch.empty(shape, dtype=torch_dtype)
            else:
                restored = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                restored = restored.view(torch_dtype).reshape(shape)
            return restored if device is None else restored.to(device)
        import numpy as np

        little = np.dtype(dtype).newbyteorder("<")
        return np.frombuffer(payload, dtype=little).reshape(shape).astype(np.dtype(dtype))

    def _read_tensor(self, ref: str) -> tuple[str, list[int], str, bytes]:
        identifier = self._tensor_ref(ref)
        blob = self.root / "tensors" / f"{identifier}.safetensors"
        if blob.is_dir():
            raise ValueError(f"tensor blob path {blob} is a directory, not a file")
        data = blob.read_bytes()
        if hashlib.sha256(data).hexdigest() != identifier:
            raise ValueError("tensor blob digest mismatch")
        size = int.from_bytes(data[:8], "little") if len(data) >= 8 else -1
        if size < 2 or 8 + size > len(data):
            raise ValueError("tensor blob header is malformed")
        try:
            header = json.loads(data[8 : 8 + size])
            entry = header["tensor"]
            metadata = header["__metadata__"]
            codes = {code: name for name, (code, _) in _SAFETENSORS_DTYPES.items()}
            if (
                not isinstance(header, dict)
                or not isinstance(entry, dict)
                or not isinstance(metadata, dict)
                or not isinstance(entry["dtype"], str)
            ):
                raise ValueError("header sections are mappings and the dtype is text")
            dtype = codes[entry["dtype"]]
            shape = entry["shape"]
            begin, end = entry["data_offsets"]
        except (ValueError, KeyError, TypeError, RecursionError) as error:
            raise ValueError("tensor blob header is malformed") from error
        if (
            set(header) != {"__metadata__", "tensor"}
            or type(begin) is not int
            or type(end) is not int
            or metadata.get("schema") != _TENSOR_SCHEMA
            or not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
        ):
            raise ValueError("tensor blob header is malformed")
        payload = data[8 + size :]
        count = 1
        for n in shape:
            count *= n
        expected = count * _SAFETENSORS_DTYPES[dtype][1]
        if (begin, end) != (0, len(payload)) or len(payload) != expected:
            raise ValueError("tensor blob payload does not match its header")
        payload_sha = hashlib.sha256(payload).hexdigest()
        if metadata.get("sha256") != payload_sha:
            raise ValueError("tensor payload digest mismatch")
        return dtype, shape, payload_sha, payload


_TENSOR_SCHEMA = "saturn-pub-tensor-v1"
_STALE_TMP_SECONDS = 3600
# dtype name -> (safetensors code, bytes per element)
_SAFETENSORS_DTYPES = {
    "bool": ("BOOL", 1),
    "uint8": ("U8", 1),
    "int8": ("I8", 1),
    "int16": ("I16", 2),
    "uint16": ("U16", 2),
    "int32": ("I32", 4),
    "uint32": ("U32", 4),
    "int64": ("I64", 8),
    "uint64": ("U64", 8),
    "float16": ("F16", 2),
    "bfloat16": ("BF16", 2),
    "float32": ("F32", 4),
    "float64": ("F64", 8),
}


def _tensor_payload(value: Any) -> tuple[str, list[int], bytes]:
    """(dtype name, shape, little-endian bytes) of a torch tensor or numpy array."""
    if tensor(value):
        import sys

        import torch

        if sys.byteorder != "little":
            raise ValueError("torch tensor blobs require a little-endian host")
        # Lazy conjugate/negation views carry a flag the raw storage does not: materialise them.
        value = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
        dtype = str(value.dtype).removeprefix("torch.")
        if dtype not in _SAFETENSORS_DTYPES:
            raise ValueError(f"unsupported tensor dtype: {dtype}")
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        return dtype, list(value.shape), payload
    if type(value).__module__ == "numpy" and hasattr(value, "dtype"):
        import numpy as np

        array = np.asarray(value)  # ascontiguousarray would turn a 0-d array into 1-d
        if not array.flags.c_contiguous:
            array = array.copy(order="C")
        dtype = array.dtype.name
        if dtype not in _SAFETENSORS_DTYPES or dtype == "bfloat16":
            raise ValueError(f"unsupported array dtype: {array.dtype}")
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        return dtype, list(array.shape), array.tobytes()
    raise ValueError("put_tensor expects a torch tensor or a numpy array")
