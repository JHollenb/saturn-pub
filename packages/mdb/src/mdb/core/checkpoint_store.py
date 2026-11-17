"""Where checkpoint bytes live: a local project, a directory or an S3-compatible bucket (#517).

A checkpoint (``mdb.engine.checkpoint``) is a sealed ``checkpoint-state`` record plus blobs: its
document and every tensor part, each a saturn_pub tensor blob addressed by the sha256 of its
bytes. A store holds both under one layout::

    <root>/tensors/<sha256>.safetensors   the blob bytes, exactly as a project store holds them
    <root>/records/<record id>.json       the sealed record

Backends: ``ProjectStore`` (a project's own store and record index), ``DirectoryStore`` (a
plain directory, ``file:///path``) and ``S3Store`` (``s3://bucket/prefix``; boto3, the
``mdb-runtime[s3]`` extra; MinIO through ``endpoint``). Credentials come from the environment
only (boto3's chain: ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``); a store's ``uri`` names
no credential and is what a record says.

``copy`` moves a checkpoint between two stores: blobs the target already has are not sent
again (dedup across jobs), and every blob read is checked against its sha256 before it is
written anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_HEX = set("0123456789abcdef")


def _sha(text: str) -> str:
    if len(text) != 64 or not set(text) <= _HEX:
        raise ValueError(f"{text!r} is not a sha256")
    return text


def _blob_key(sha: str) -> str:
    return f"tensors/{_sha(sha)}.safetensors"


def _record_key(record_id: str) -> str:
    return f"records/{record_id}.json"


class MissingBlob(ValueError):
    """A blob the store does not hold (``sha256`` names it)."""

    def __init__(self, sha256: str, uri: str):
        self.sha256 = sha256
        super().__init__(f"checkpoint blob {sha256} is missing from {uri}")


class Store:
    """Blobs and records by key. Backends implement ``_get`` (None when absent), ``_put``,
    ``_has`` and ``_keys``; the sha256 checks are here, once."""

    uri = ""

    def _get(self, key: str) -> bytes | None:
        raise NotImplementedError

    def _put(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def _has(self, key: str) -> bool:
        return self._get(key) is not None

    def _keys(self, prefix: str) -> list[str]:
        raise NotImplementedError

    # blobs --------------------------------------------------------------------------------

    def has(self, sha: str) -> bool:
        return self._has(_blob_key(sha))

    def get(self, sha: str) -> bytes:
        """The blob ``sha``, checked: ``MissingBlob`` when absent, ``ValueError`` when its
        bytes differ from their sha256."""
        data = self._get(_blob_key(sha))
        if data is None:
            raise MissingBlob(sha, self.uri)
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"checkpoint blob {sha} in {self.uri} differs from its sha256")
        return data

    def put(self, sha: str, data: bytes) -> None:
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"blob bytes differ from their sha256 {sha}")
        self._put(_blob_key(sha), data)

    # records ------------------------------------------------------------------------------

    def put_record(self, record: dict[str, Any]) -> None:
        self._put(_record_key(record["id"]), json.dumps(record, sort_keys=True).encode())

    def find_record(self, ref: str) -> dict[str, Any] | None:
        """The verified record whose id starts with ``ref`` (None when there is none)."""
        from mdb.core.kinds import CHECKPOINT_STATE  # noqa: F401 (declares the kind)
        from mdb.core.records import verify_record

        ids = sorted({k.rsplit("/", 1)[-1].removesuffix(".json")
                      for k in self._keys(f"records/{ref}")})
        if not ids:
            return None
        if len(ids) > 1:
            raise ValueError(f"checkpoint record prefix {ref} is ambiguous in {self.uri}; give "
                             "more digits")
        data = self._get(_record_key(ids[0]))
        if data is None:
            return None
        record = verify_record(json.loads(data))
        if record["id"] != ids[0]:
            raise ValueError(f"record {ids[0]} in {self.uri} holds record {record['id']}")
        return record

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.uri!r})"


class DirectoryStore(Store):
    """A plain directory (local disk, a mounted share)."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().absolute()
        self.uri = self.root.as_uri()

    def _path(self, key: str) -> Path:
        return self.root / key

    def _get(self, key: str) -> bytes | None:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            return None

    def _has(self, key: str) -> bool:
        return self._path(key).is_file()

    def _put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)

    def _keys(self, prefix: str) -> list[str]:
        folder, _, stem = prefix.rpartition("/")
        directory = self.root / folder
        if not directory.is_dir():
            return []
        return [f"{folder}/{p.name}" for p in directory.iterdir() if p.name.startswith(stem)]


class S3Store(Store):
    """An S3-compatible bucket (MinIO: give ``endpoint``, else ``AWS_ENDPOINT_URL``). Needs
    boto3 (``pip install 'mdb-runtime[s3]'``)."""

    def __init__(self, uri: str, *, endpoint: str | None = None, client: Any = None):
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"{uri!r} is not an s3://bucket/prefix URI")
        if parsed.username or parsed.password:
            raise ValueError("an s3 URI names no credentials: they come from the environment")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.uri = f"s3://{self.bucket}/{self.prefix}".rstrip("/")
        self.endpoint = endpoint or os.environ.get("AWS_ENDPOINT_URL") or None
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as error:
                raise ImportError("an s3:// checkpoint store needs boto3: pip install "
                                  "'mdb-runtime[s3]'") from error
            self._client = boto3.client(
                "s3", endpoint_url=self.endpoint,
                region_name=os.environ.get("AWS_DEFAULT_REGION") or "us-east-1",
                config=Config(s3={"addressing_style": "path"}))
        return self._client

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    @staticmethod
    def _absent(error: Exception) -> bool:
        code = str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))
        return code in ("404", "NoSuchKey", "NotFound")

    def _get(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except Exception as error:  # botocore.exceptions.ClientError (boto3 is optional)
            if self._absent(error):
                return None
            raise

    def _has(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception as error:
            if self._absent(error):
                return False
            raise

    def _put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def _keys(self, prefix: str) -> list[str]:
        out, cut = [], len(self._key(""))
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=self._key(prefix))
        for page in pages:
            out += [item["Key"][cut:] for item in page.get("Contents", [])]
        return out


class ProjectStore(Store):
    """A project's own store (``store/tensors``) and record index: the local end of ``copy``."""

    def __init__(self, project: Any):
        self.project = project
        self.uri = f"project:{project.root}"

    def _path(self, sha: str) -> Path:
        return self.project.store.root / _blob_key(sha)

    def has(self, sha: str) -> bool:
        return self._path(sha).is_file()

    def get(self, sha: str) -> bytes:
        try:
            data = self._path(sha).read_bytes()
        except FileNotFoundError:
            raise MissingBlob(sha, self.uri) from None
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"checkpoint blob {sha} in {self.uri} differs from its sha256")
        return data

    def put(self, sha: str, data: bytes) -> None:
        from saturn_pub.store import _publish

        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"blob bytes differ from their sha256 {sha}")
        _publish(self._path(sha), data)

    def put_record(self, record: dict[str, Any]) -> None:
        from mdb.core import index

        self.project.open_session(record["session_id"], model_identity=record["model_identity"],
                                  execution=record["execution"])
        index.put_record(self.project, record)

    def find_record(self, ref: str) -> dict[str, Any] | None:
        from mdb.engine.checkpoint import find_record

        try:
            return find_record(self.project, ref)
        except ValueError as error:
            if "no checkpoint record" in str(error):
                return None
            raise


def open_store(uri: Any, *, endpoint: str | None = None) -> Store:
    """A store from ``uri``: a ``Store``, a ``Project``, ``s3://bucket/prefix``, ``file:///dir``
    or a directory path."""
    if isinstance(uri, Store):
        return uri
    if hasattr(uri, "store") and hasattr(uri, "sessions"):
        return ProjectStore(uri)
    text = str(uri)
    if text.startswith("s3://"):
        from mdb.core import settings

        return S3Store(text, endpoint=endpoint or settings.value("storage.endpoint"))
    if text.startswith("file://"):
        return DirectoryStore(urlparse(text).path)
    if "://" in text:
        raise ValueError(f"unknown checkpoint store {text!r}: s3://bucket/prefix, file:///dir "
                         "or a directory")
    return DirectoryStore(text)


def is_store_uri(value: Any) -> bool:
    return isinstance(value, Store) or (isinstance(value, str)
                                        and value.startswith(("s3://", "file://")))


def configured(given: Any = None) -> Store | None:
    """The checkpoint store ``given`` names, else ``[storage] checkpoint_store``, else None."""
    from mdb.core import settings

    uri = settings.value("storage.checkpoint_store", given)
    return None if uri is None else open_store(uri)


def cache_project() -> Any:
    """The local project that holds checkpoints read from (or written to) a store when no
    project is attached: ``[storage] checkpoint_cache``, default ``$XDG_CACHE_HOME/mdb/
    checkpoints``."""
    from mdb.core import settings
    from mdb.core.project import Project

    root = settings.value("storage.checkpoint_cache")
    if root is None:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        root = Path(base) / "mdb" / "checkpoints"
    return Project(root)


def blobs_of(record: dict[str, Any], doc: dict[str, Any]) -> list[str]:
    """The blobs a checkpoint names: its document, then every part (sha256 of the blob)."""
    refs = [record["body"]["document"]["ref"]]
    refs += [p["ref"] for p in doc["parts"]]
    return list(dict.fromkeys(refs))


def copy(record: dict[str, Any], doc: dict[str, Any] | None, source: Store, target: Store
         ) -> dict[str, Any]:
    """Copy checkpoint ``record`` (document ``doc``; None: read from ``source``) and its blobs
    from ``source`` to ``target``, blobs first and the record last (a record never names
    missing bytes). A blob ``target`` has is not sent again. Returns counts."""
    sent = kept = size = 0
    read: dict[str, bytes] = {}
    if doc is None:
        ref = record["body"]["document"]["ref"]
        read[ref] = source.get(ref)
        doc = _document(read[ref])
    for sha in blobs_of(record, doc):
        if target.has(sha):
            kept += 1
            continue
        data = read.pop(sha, None) or source.get(sha)
        target.put(sha, data)
        sent += 1
        size += len(data)
    target.put_record(record)
    return {"store": target.uri, "sent": sent, "present": kept, "bytes": size}


def _document(raw: bytes) -> dict[str, Any]:
    """The JSON document inside a uint8 tensor blob (``bounded.put_json``); the reader checks
    it against the record's sha256 (``checkpoint.load_record``)."""
    size = int.from_bytes(raw[:8], "little")
    return json.loads(raw[8 + size:])


__all__ = ["DirectoryStore", "MissingBlob", "ProjectStore", "S3Store", "Store", "blobs_of",
           "cache_project", "configured", "copy", "is_store_uri", "open_store"]
