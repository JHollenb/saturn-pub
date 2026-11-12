"""Immutable whole-file sharing without changing StateCut or array identities.

Local paths remain regular files (hardlinks), so legacy readers and manifests
need no changes. The pool is local and on the same filesystem as its clients.
This is byte sharing, not semantic equivalence or BF16/FP32 conversion.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import uuid
from pathlib import Path

CONFIG_NAME = ".mdb-storage.json"
SCHEMA = "saturn-pub-shared-files-v1"
TENSOR_SUFFIXES = frozenset({".safetensors", ".npy", ".npyb"})


def safe_path(path):
    path = Path(os.path.abspath(path))
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise ValueError("shared storage refuses symbolic links: " + str(part))
    return path


def _identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def file_identity(path):
    """Hash a stable regular file; fail if its inode or contents change."""
    path = safe_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("shared storage requires a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise ValueError("source changed before hashing")
        while chunk := stream.read(4 << 20):
            digest.update(chunk)
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise ValueError("source changed while hashing")
    if _identity(path.stat()) != _identity(before):
        raise ValueError("source path changed while hashing")
    return digest.hexdigest(), before


def _fsync_dir(path):
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass  # Some supported filesystems do not expose directory fsync.


class SharedStore:
    """No automatic GC; all managed payloads are read-only and hash verified."""

    def __init__(self, root):
        self.root = safe_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / "sha256"
        self.objects.mkdir(exist_ok=True)

    def object_path(self, digest):
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("expected a lowercase SHA-256")
        return safe_path(self.objects / digest[:2] / digest)

    def _check(self, path, digest, size):
        actual, st = file_identity(path)
        if actual != digest or st.st_size != size:
            raise ValueError("shared object content-address mismatch: " + str(path))
        if st.st_mode & 0o222:
            raise ValueError("shared object must be read-only: " + str(path))
        return st

    def intern(self, path, *, expected_sha256=None):
        """Atomically replace one sealed file by a byte-identical shared link.

        Source stays available throughout, including on EXDEV or interruption.
        Inputs must be closed evidence, never active mutable model checkpoints.
        """
        path = safe_path(path)
        digest, before = file_identity(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("source differs from frozen storage plan: " + str(path))
        target = self.object_path(digest)
        target.parent.mkdir(exist_ok=True)
        if before.st_dev != self.root.stat().st_dev:
            raise ValueError("shared storage requires the same filesystem")
        # Seal before publication. Byte identity does not include permission bits.
        os.chmod(path, stat.S_IMODE(before.st_mode) & ~0o222)
        try:
            os.link(path, target)
            _fsync_dir(target.parent)
        except FileExistsError:
            pass
        stored = self._check(target, digest, before.st_size)
        current = path.stat()
        if _identity(current) != _identity(before):
            raise ValueError("source changed before shared-link replacement")
        changed = (current.st_dev, current.st_ino) != (stored.st_dev, stored.st_ino)
        released = current.st_size if changed and current.st_nlink == 1 else 0
        blocks = current.st_blocks * 512 if changed and current.st_nlink == 1 else 0
        if changed:
            temporary = path.parent / (".mdb-share-" + uuid.uuid4().hex)
            try:
                os.link(target, temporary)
                if _identity(path.stat()) != _identity(before):
                    raise ValueError("source changed during shared-link replacement")
                os.replace(temporary, path)
                _fsync_dir(path.parent)
            finally:
                temporary.unlink(missing_ok=True)
        if not os.path.samefile(path, target):
            raise ValueError("shared-link verification failed")
        return {"path": str(path), "sha256": digest, "bytes": before.st_size,
                "object": str(target), "relinked": changed,
                "released_inode_logical_bytes": released,
                "released_inode_stat_allocated_bytes": blocks}

    def publish(self, path, raw):
        """Publish immutable bytes via the pool; return local admission status."""
        path = safe_path(path)
        digest = hashlib.sha256(raw).hexdigest()
        target = self.object_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(exist_ok=True)
        if path.parent.stat().st_dev != self.root.stat().st_dev:
            raise ValueError("shared storage requires the same filesystem")
        if not target.exists():
            fd, name = tempfile.mkstemp(dir=target.parent, prefix=".publish-")
            temporary = Path(name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.fchmod(stream.fileno(), 0o444)
                try:
                    os.link(temporary, target)
                    _fsync_dir(target.parent)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
        self._check(target, digest, len(raw))
        try:
            os.link(target, path)
            _fsync_dir(path.parent)
            return True
        except FileExistsError:
            actual, st = file_identity(path)
            if actual != digest or st.st_size != len(raw):
                raise ValueError("existing artifact differs from immutable shared bytes")
            self.intern(path, expected_sha256=digest)
            return False


def configure(scope, root):
    scope, root = safe_path(scope), safe_path(root)
    scope.mkdir(parents=True, exist_ok=True)
    pool = SharedStore(root)
    if scope.stat().st_dev != pool.root.stat().st_dev:
        raise ValueError("configuration requires the same filesystem")
    config = scope / CONFIG_NAME
    body = (json.dumps({"schema": SCHEMA, "root": str(root)}, sort_keys=True, indent=2) + "\n").encode()
    if config.exists():
        if config.read_bytes() != body:
            raise ValueError("storage scope is already configured differently")
    else:
        # Exclusive, small configuration file; existing configuration is never replaced.
        with config.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_dir(scope)
    return config


def configured_store(path):
    override = os.environ.get("MDB_SHARED_TENSOR_STORE")
    if override:
        return SharedStore(override)
    path = safe_path(path)
    for parent in path.parents:
        config = parent / CONFIG_NAME
        if config.exists():
            config = safe_path(config)
            value = json.loads(config.read_text())
            if value.get("schema") != SCHEMA or not Path(value["root"]).is_absolute():
                raise ValueError("invalid shared storage configuration")
            return SharedStore(value["root"])
    return None


def publish_configured(path, raw):
    """None means unconfigured, preserving the caller's legacy publication."""
    pool = configured_store(path)
    return None if pool is None else pool.publish(path, raw)


def share_configured_file(path):
    """Admit a closed NPY/safetensors export from an external scientific writer."""
    pool = configured_store(path)
    return None if pool is None else pool.intern(path)
