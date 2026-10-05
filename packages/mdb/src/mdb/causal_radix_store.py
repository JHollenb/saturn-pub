"""Durable physical segments for a verified causal radix DAG.

The reader returns ordered immutable segments. It never reads a source page
blob or silently concatenates a logical page. Consumers may decode each
segment in order under their own declared numerical ABI.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mdb.causal_page_dag import CausalPageManifest, CausalPageRow
from mdb.causal_page_store import manifest_from_row_payloads
from mdb.causal_radix_dag import compile_causal_radix_dag
from mdb.vm_payload_cache import PayloadRef, PayloadStore

SCHEMA = "saturn-causal-radix-physical-v1"


class RadixStoreError(ValueError):
    """The physical row map or its declared ancestry is invalid."""


def _canon(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SegmentedRead:
    page_id: str
    model_abi_sha256: str
    node_sha256s: tuple[str, ...]
    segments: tuple[bytes, ...]
    source_payload_sha256: str

    @property
    def row_bytes(self) -> int:
        return sum(map(len, self.segments))

    def materialize_for_reference(self) -> bytes:
        """Explicit serial reference only; native segmented users avoid this."""
        return b"".join(self.segments)


class CausalRadixStore:
    def __init__(self, root: str | Path, *, read_only: bool = False) -> None:
        self.root = Path(root).expanduser().resolve()
        self.payloads = PayloadStore(self.root / "cas", read_only=read_only)
        if read_only:
            if not (self.root / "mounts").is_dir(): raise RadixStoreError("read-only mounts absent")
        else:
            (self.root / "mounts").mkdir(parents=True, exist_ok=True)

    def export_mount(self, mount_sha256: str, destination: CausalRadixStore) -> str:
        """Copy a verified immutable mount and its complete byte closure.

        The destination owns its blobs; it does not retain scratch-path or
        source-store pointers. Publish the mount last, after verifying every
        source path and hydrating every physical segment.
        """
        mount = self._mount(mount_sha256)
        abi = mount["plan"]["model_abi_sha256"]
        for source in mount["sources"]:
            self.read(mount_sha256, source["page_id"], model_abi_sha256=abi)
        for item in mount["physical"]:
            ref = PayloadRef.from_dict(item["payload_ref"])
            raw = self.payloads.hydrate(ref, native_consumer="radix-mount-export").payload
            copied = destination.payloads.put_bytes(raw, role="causal-radix-segment")
            if copied.to_dict() != ref.to_dict():
                raise RadixStoreError("export changed physical segment identity")
        raw_mount = _canon(mount) + b"\n"
        target = destination.root / "mounts" / f"{mount_sha256}.json"
        descriptor, temporary = tempfile.mkstemp(prefix=".radix-export.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw_mount)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                if target.read_bytes() != raw_mount:
                    raise RadixStoreError("immutable exported mount collision") from error
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(temporary)
        for source in mount["sources"]:
            destination.read(mount_sha256, source["page_id"], model_abi_sha256=abi)
        return mount_sha256

    def publish(
        self,
        *,
        model_abi_sha256: str,
        pages: Mapping[str, Sequence[bytes]],
    ) -> str:
        if not pages:
            raise RadixStoreError("at least one page is required")
        owned = {name: tuple(bytes(row) for row in rows) for name, rows in pages.items()}
        sources = tuple(
            manifest_from_row_payloads(page_id=name, model_abi_sha256=model_abi_sha256, rows=rows)
            for name, rows in sorted(owned.items())
        )
        plan = compile_causal_radix_dag(sources)
        payload_by_identity: dict[tuple[str, int], bytes] = {}
        for rows in owned.values():
            for row in rows:
                payload_by_identity[(_sha(row), len(row))] = row
        physical = []
        for segment in plan.segments:
            raw = b"".join(payload_by_identity[identity] for identity in segment.payload_identity)
            ref = self.payloads.put_bytes(raw, role="causal-radix-segment")
            physical.append({"node_sha256": segment.node_sha256, "payload_ref": ref.to_dict()})
        body = {
            "schema": SCHEMA,
            "plan": plan.to_dict(),
            "sources": [source.to_dict() for source in sources],
            "physical": physical,
        }
        mount_sha = _sha(_canon(body))
        raw_mount = _canon({**body, "mount_sha256": mount_sha}) + b"\n"
        target = self.root / "mounts" / f"{mount_sha}.json"
        from .bench_publication import stage_write
        if stage_write(target, raw_mount): return mount_sha
        descriptor, temporary = tempfile.mkstemp(prefix=".radix.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw_mount)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                if target.read_bytes() != raw_mount:
                    raise RadixStoreError("immutable mount collision") from error
        finally:
            os.unlink(temporary)
        return mount_sha

    def _mount(self, mount_sha256: str) -> dict:
        if len(mount_sha256) != 64 or any(c not in "0123456789abcdef" for c in mount_sha256):
            raise RadixStoreError("invalid mount digest")
        raw = (self.root / "mounts" / f"{mount_sha256}.json").read_bytes()
        mount = json.loads(raw)
        body = {key: value for key, value in mount.items() if key != "mount_sha256"}
        if mount.get("mount_sha256") != mount_sha256 or _sha(_canon(body)) != mount_sha256:
            raise RadixStoreError("mount digest changed")
        if mount.get("schema") != SCHEMA:
            raise RadixStoreError("mount schema changed")
        sources = tuple(
            CausalPageManifest(
                page_id=item["page_id"],
                model_abi_sha256=item["model_abi_sha256"],
                payload_sha256=item["payload_sha256"],
                rows=tuple(CausalPageRow(**row) for row in item["rows"]),
            )
            for item in mount["sources"]
        )
        expected = compile_causal_radix_dag(sources)
        if expected.to_dict() != mount["plan"]:
            raise RadixStoreError("radix plan or ancestry changed")
        physical = {item["node_sha256"]: item["payload_ref"] for item in mount["physical"]}
        if len(physical) != len(expected.segments) or set(physical) != {
            s.node_sha256 for s in expected.segments
        }:
            raise RadixStoreError("physical node map changed")
        return mount

    def read(self, mount_sha256: str, page_id: str, *, model_abi_sha256: str,
             audit: bool = False) -> SegmentedRead:
        mount = self._mount(mount_sha256)
        plan = mount["plan"]
        if model_abi_sha256 != plan["model_abi_sha256"]:
            raise RadixStoreError("model ABI changed")
        paths = {path["page_id"]: path for path in plan["paths"]}
        if page_id not in paths:
            raise RadixStoreError("page is absent")
        path = paths[page_id]
        nodes = {node["node_sha256"]: node for node in plan["segments"]}
        refs = {item["node_sha256"]: item["payload_ref"] for item in mount["physical"]}
        parent = plan["root_sha256"]
        segments = []
        for node_sha in path["node_sha256s"]:
            node = nodes[node_sha]
            if node["required_parent_sha256"] != parent:
                raise RadixStoreError("wrong parent")
            ref = PayloadRef.from_dict(refs[node_sha])
            raw = (self.payloads.audit(ref) if audit else
                   self.payloads.hydrate(ref, native_consumer="radix-segment-reader")).payload
            if len(raw) != node["payload_bytes"]:
                raise RadixStoreError("segment byte count changed")
            offset = 0
            for row in node["rows"]:
                stop = offset + row["payload_bytes"]
                if _sha(raw[offset:stop]) != row["payload_sha256"]:
                    raise RadixStoreError("segment row digest changed")
                offset = stop
            segments.append(raw)
            parent = node_sha
        digest = hashlib.sha256()
        for raw in segments:
            digest.update(raw)
        if digest.hexdigest() != path["source_payload_sha256"]:
            raise RadixStoreError("logical page digest changed")
        return SegmentedRead(
            page_id,
            model_abi_sha256,
            tuple(path["node_sha256s"]),
            tuple(segments),
            digest.hexdigest(),
        )

    def append(
        self,
        mount_sha256: str,
        *,
        parent_page_id: str,
        new_page_id: str,
        rows: Sequence[bytes],
        model_abi_sha256: str,
    ) -> str:
        """Publish a new immutable family; the old mount remains rollback state."""
        mount = self._mount(mount_sha256)
        if model_abi_sha256 != mount["plan"]["model_abi_sha256"]:
            raise RadixStoreError("model ABI changed")
        pages: dict[str, tuple[bytes, ...]] = {}
        for source in mount["sources"]:
            page_id = source["page_id"]
            read = self.read(mount_sha256, page_id, model_abi_sha256=model_abi_sha256)
            source_rows = []
            for segment in read.segments:
                # Source boundaries are reconstructed from its verified row manifest.
                source_rows.append(segment)
            flat = b"".join(source_rows)
            offset = 0
            decoded = []
            for row in source["rows"]:
                stop = offset + row["payload_bytes"]
                decoded.append(flat[offset:stop])
                offset = stop
            pages[page_id] = tuple(decoded)
        if parent_page_id not in pages or new_page_id in pages:
            raise RadixStoreError("invalid append identity")
        pages[new_page_id] = pages[parent_page_id] + tuple(bytes(row) for row in rows)
        return self.publish(model_abi_sha256=model_abi_sha256, pages=pages)
