"""Root-scoped durable publication batches for the single-writer bench.

Existing stores remain independently durable outside this context. Inside it,
their files are materialized for immediate verification, while a sealed recovery
pack owns the durable bytes. The selected pointer is published only after that
pack is fsynced. Recovery verifies the complete chain before changing any files.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path, PurePosixPath
import base64
import fcntl
import gzip
import json
import os
import tempfile
import time
import zlib
from uuid import uuid4
from weakref import WeakValueDictionary

_ACTIVE = ContextVar("saturn_bench_publication", default=None)
SCHEMA = "saturn-bench-publication-v1"
OWNERS = WeakValueDictionary()
_RECOVERED = {}


def _checked_path(root, path):
    if not path.is_relative_to(root): raise ValueError("publication path outside root")
    for part in (path, *path.parents):
        if part == root: break
        if part.is_symlink(): raise ValueError("publication path traverses a symlink")
    return path


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _replace(path, raw, *, durable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".bench-write-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            if durable: os.fsync(stream.fileno())
        os.replace(temporary, path)
        if durable:
            directory = os.open(path.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def stage_write(path, raw, *, immutable=True):
    """Return False outside the owning bench; never intercept another store."""
    active = _ACTIVE.get()
    path = Path(path).absolute()
    if active is None or not path.is_relative_to(active.root): return False
    if path.resolve() != path: raise ValueError("noncanonical publication path")
    active.stage(path, bytes(raw), immutable=immutable)
    return True


class BenchPublication:
    def __init__(self, root, *, mode="journal"):
        if mode not in {"journal", "legacy"}: raise ValueError("invalid bench publication mode")
        self.root = Path(root).resolve()
        OWNERS[str(self.root)] = self
        self.owner_lock = None
        self.mode = mode
        self._namespace_durable = False
        self.nonce = None
        self.records, self.originals = {}, {}
        self._atomic_selection = False
        self._deferred_manifest = None
        recovered = _RECOVERED.get(str(self.root))
        self.durable_paths = set((recovered or {}).get("committed", ()))
        candidates = (recovered or {}).get("candidates", {})
        self.stats = {"commits": 0, "staged_writes": 0, "staged_bytes": 0,
            "pack_bytes": 0, "pack_bytes_written": 0, "pack_bytes_flushed": 0,
            "file_fsyncs": 0, "directory_fsyncs": 0, "namespace_fsyncs": 0, "identity_fsyncs": 0,
            "publication_seconds": 0.0, "recovered_files": 0}
        self.stats["recovery_issues"] = (recovered or {}).get("issues", [])
        self.stats["quarantined_files"] = (recovered or {}).get("quarantined_files", 0)
        # Adopt old durable stores and candidate files left by an interrupted
        # writer. Store cache-hit paths can bypass stage_write, so adoption must
        # happen before another cut is allowed to refer to them.
        if mode == "journal" and self.root.exists():
            if recovered is None and (self.root/"journal").exists():
                self.recover(self.root)
                recovered = _RECOVERED[str(self.root)]
                candidates = recovered["candidates"]
                self.stats["recovery_issues"] = recovered["issues"]
                self.stats["quarantined_files"] = recovered.get("quarantined_files", 0)
            if (recovered or {}).get("quarantine_incomplete"):
                raise ValueError("candidate quarantine incomplete; retry recovery before publication")
            if (recovered or {}).get("legacy_unsafe"):
                raise ValueError("unqualified legacy migration evidence; retain custody before journal publication")
            identity = _checked_path(self.root, self.root/"journal/identity.json")
            if identity.exists(): self.nonce = self._read_identity(identity)
            for path in self.root.rglob("*"):
                if not path.is_file(): continue
                _checked_path(self.root, path)
                name = path.relative_to(self.root).as_posix()
                if (name in self.durable_paths or name == "selected.json"
                    or path.name == ".writer.lock" or name.startswith("journal/")): continue
                if path.is_symlink(): raise ValueError("publication path traverses a symlink")
                raw = path.read_bytes()
                if name in candidates and sha256(raw).hexdigest() != candidates[name]:
                    raise ValueError("recovered candidate changed before adoption")
                self.records[name] = {"path": name, "immutable": name in candidates,
                    "sha256": sha256(raw).hexdigest(), "bytes": base64.b64encode(raw).decode()}
        _RECOVERED.pop(str(self.root), None)

    @contextmanager
    def initializing(self):
        """Include new store manifests in the first cut's durable closure."""
        if self.mode == "legacy":
            yield
            return
        token = _ACTIVE.set(self)
        try: yield
        finally: _ACTIVE.reset(token)

    @contextmanager
    def transaction(self):
        if self.mode == "legacy":
            yield
            return
        current = _ACTIVE.get()
        if current is self:
            yield
            return
        if current is not None: raise RuntimeError("nested bench publication roots")
        token = _ACTIVE.set(self)
        try:
            yield
            if self.records:
                selected = self.root/"selected.json"
                if selected.exists():
                    self.select(json.loads(selected.read_bytes())["manifest"])
                else:
                    # Failed/unfinished initialization has no selectable session.
                    self._retain_candidate(self._seal(None))
                    self.records.clear(); self.originals.clear()
        except BaseException:
            # Retain immutable candidate bytes, but never publish candidate cursors.
            try:
                if self.records: self._retain_candidate(self._seal(None))
            except (OSError, ValueError):
                pass
            for name, raw in self.originals.items():
                path = self.root/name
                if raw is None: path.unlink(missing_ok=True)
                else: _replace(path, raw)
            # Cache hits in a retry may bypass the store's publish helper. Keep
            # retained immutable bytes in the next selectable pack as well.
            self.records = {name: row for name, row in self.records.items() if row["immutable"]}
            self.originals.clear()
            raise
        finally:
            _ACTIVE.reset(token)

    def stage(self, path, raw, *, immutable):
        path = Path(path).absolute()
        _checked_path(self.root, path)
        if path.resolve() != path: raise ValueError("noncanonical publication path")
        name = path.relative_to(self.root).as_posix()
        if name == "selected.json" or path.name == ".writer.lock" or name.startswith("journal/"):
            raise ValueError("reserved publication path")
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != self.root):
            raise ValueError("publication path traverses a symlink")
        exists = path.exists()
        if immutable and exists:
            if path.read_bytes() != raw: raise ValueError("immutable publication collision")
            if name in self.records or name in self.durable_paths: return
        if not immutable and name not in self.originals:
            self.originals[name] = path.read_bytes() if exists else None
        self.records[name] = {"path": name, "immutable": immutable,
            "sha256": sha256(raw).hexdigest(), "bytes": base64.b64encode(raw).decode()}
        _replace(path, raw)
        self.stats["staged_writes"] += 1
        self.stats["staged_bytes"] += len(raw)

    def _seal(self, manifest):
        selected = _checked_path(self.root, self.root/"selected.json")
        pointer = json.loads(selected.read_bytes()) if selected.exists() else None
        parent = pointer.get("journal") if pointer else None
        if self.nonce is None:
            self.nonce = uuid4().hex
            _replace(_checked_path(self.root, self.root/"journal/identity.json"),
                _json({"schema": SCHEMA, "nonce": self.nonce}), durable=True)
            self.stats["identity_fsyncs"] += 2
        body = {"schema": SCHEMA, "parent": parent, "manifest": manifest,
                "nonce": self.nonce, "legacy_parent_manifest": pointer["manifest"] if pointer and "journal" not in pointer else None,
                "writes": [self.records[k] for k in sorted(self.records)]}
        raw = gzip.compress(_json(body), compresslevel=1, mtime=0)
        digest = sha256(raw).hexdigest()
        path = _checked_path(self.root, self.root/"journal"/(digest+".pack"))
        if path.exists():
            if path.read_bytes() != raw: raise ValueError("publication pack collision")
            # A prior attempt may have renamed this pack and then failed its
            # directory flush. Existence alone does not establish durability.
            descriptor = os.open(path, os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
            directory = os.open(path.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        else:
            _replace(path, raw, durable=True)
            self.stats["pack_bytes_written"] += len(raw)
        self.stats["file_fsyncs"] += 1; self.stats["directory_fsyncs"] += 1
        if not self._namespace_durable:
            # Persist journal and session directory entries even when the first
            # operation fails before any selected pointer can be acknowledged.
            for parent in (self.root, self.root.parent):
                directory = os.open(parent, os.O_RDONLY)
                try: os.fsync(directory)
                finally: os.close(directory)
                self.stats["namespace_fsyncs"] += 1
            self._namespace_durable = True
        self.stats["pack_bytes"] += len(raw)
        self.stats["pack_bytes_flushed"] += len(raw)
        return digest

    def _retain_candidate(self, digest):
        path = _checked_path(self.root, self.root/"journal/candidates.json")
        index = json.loads(path.read_bytes()) if path.exists() else {"schema": SCHEMA, "nonce": self.nonce, "packs": []}
        if not isinstance(index, dict) or index.get("schema") != SCHEMA or not isinstance(index.get("packs"), list):
            raise ValueError("invalid candidate index")
        if digest not in index["packs"]:
            _replace(path, _json({"schema": SCHEMA, "nonce": self.nonce, "packs": [*index["packs"], digest]}), durable=True)
            self.stats["file_fsyncs"] += 1; self.stats["directory_fsyncs"] += 1

    @contextmanager
    def atomic_selection(self):
        """Publish a multi-cut operation once, retaining failed candidate bytes.

        The owning transaction stages immutable payloads and captures mutable
        originals. Intermediate cuts are readable by the serialized owner but
        cannot advance the durable selected pointer. Its normal exception path
        retains a candidate pack and restores mutable store cursors.
        """
        if self.mode != "journal":
            raise ValueError("atomic selection requires journal publication mode")
        if self._atomic_selection:
            yield
            return
        self._atomic_selection = True
        self._deferred_manifest = None
        try:
            yield
            manifest = self._deferred_manifest
            self._atomic_selection = False
            if manifest is not None:
                self.select(manifest)
        finally:
            self._atomic_selection = False
            self._deferred_manifest = None

    def select(self, manifest):
        if self._atomic_selection:
            self._deferred_manifest = manifest
            return
        retry = _RECOVERED.get(str(self.root), {})
        if retry.get("quarantine_incomplete") or (self.root/"journal/quarantine.pending.json").exists():
            raise ValueError("candidate quarantine incomplete; retry recovery before publication")
        selected = _checked_path(self.root, self.root/"selected.json")
        if self.mode == "legacy":
            raw = _json({"manifest": manifest})
            if not stage_write(selected, raw, immutable=False):
                _replace(selected, raw, durable=True)
            return
        started = time.perf_counter()
        selected = _checked_path(self.root, self.root/"selected.json")
        previous = selected.read_bytes() if selected.exists() else None
        digest = self._seal(manifest)
        try:
            _replace(selected, _json({"manifest": manifest, "journal": digest,
                "publication_schema": SCHEMA, "nonce": self.nonce}), durable=True)
        except BaseException:
            # Rename may have succeeded before the directory flush failed.
            if previous is None:
                selected.unlink(missing_ok=True)
                directory = os.open(self.root, os.O_RDONLY)
                try: os.fsync(directory)
                finally: os.close(directory)
            else: _replace(selected, previous, durable=True)
            raise
        self.stats["commits"] += 1
        self.stats["file_fsyncs"] += 1; self.stats["directory_fsyncs"] += 1
        self.stats["publication_seconds"] += time.perf_counter()-started
        self.durable_paths.update(self.records)
        self.records.clear(); self.originals.clear()

    @classmethod
    def _read_identity(cls, path):
        identity = json.loads(path.read_bytes())
        nonce = identity.get("nonce") if isinstance(identity, dict) else None
        if (not isinstance(identity, dict) or identity.get("schema") != SCHEMA or not isinstance(nonce, str)
            or len(nonce) != 32 or any(c not in "0123456789abcdef" for c in nonce)):
            raise ValueError("invalid publication identity")
        return nonce

    @classmethod
    def recover(cls, root):
        root = Path(root).resolve()
        selected_path = _checked_path(root, root/"selected.json")
        selected = json.loads(selected_path.read_bytes()) if selected_path.exists() else None
        legacy = selected is not None and "journal" not in selected
        if selected is not None and not legacy and selected.get("publication_schema") != SCHEMA:
            raise ValueError("selected publication schema changed")
        identity = _checked_path(root, root/"journal/identity.json")
        nonce = cls._read_identity(identity) if identity.exists() else None
        if selected is not None and not legacy and selected.get("nonce") is not None and selected["nonce"] != nonce:
            raise ValueError("selected publication identity changed")
        def read_pack(digest):
            if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("invalid publication chain")
            raw = _checked_path(root, root/"journal"/(digest+".pack")).read_bytes()
            if sha256(raw).hexdigest() != digest: raise ValueError("publication pack checksum changed")
            try: body = json.loads(gzip.decompress(raw))
            except (OSError, EOFError, ValueError, zlib.error) as exc:
                raise ValueError("invalid publication pack encoding") from exc
            if not isinstance(body, dict) or body.get("schema") != SCHEMA: raise ValueError("publication schema changed")
            names = set()
            for row in body["writes"]:
                name = PurePosixPath(row["path"])
                if (name.is_absolute() or ".." in name.parts or not name.parts
                    or str(name) != row["path"] or row["path"] in names
                    or row["path"] == "selected.json" or name.name == ".writer.lock" or name.parts[0] == "journal"
                    or type(row.get("immutable")) is not bool):
                    raise ValueError("invalid publication member")
                names.add(row["path"])
                data = base64.b64decode(row["bytes"], validate=True)
                if sha256(data).hexdigest() != row["sha256"]: raise ValueError("publication member checksum changed")
                row["data"] = data
            return body
        chain, seen = [], set()
        digest = selected["journal"] if selected is not None and not legacy else None
        while digest is not None:
            if digest in seen: raise ValueError("invalid publication chain")
            seen.add(digest)
            body = read_pack(digest)
            if body.get("nonce") not in {None, nonce}:
                raise ValueError("selected pack publication identity changed")
            chain.append(body); digest = body["parent"]
        if selected is not None and not legacy and (not chain or chain[0]["manifest"] != selected["manifest"]):
            raise ValueError("selected manifest differs from publication pack")
        if selected is not None and not legacy and selected.get("nonce") is not None and chain[0].get("nonce") != nonce:
            raise ValueError("selected pack publication identity changed")
        # Only the selected chain has cursor authority. Verified immutable rows
        # in unselected packs retain rejected candidates after cache deletion.
        files = {}
        for body in reversed(chain):
            for row in body["writes"]:
                old = files.get(row["path"])
                if row["immutable"] and old is not None and old != row["data"]:
                    raise ValueError("immutable publication history changed")
                files[row["path"]] = row["data"]
        candidates, issues = {}, []
        index_path = root/"journal/candidates.json"
        indexed = []
        try:
            _checked_path(root, index_path)
            if index_path.exists():
                index = json.loads(index_path.read_bytes())
                if not isinstance(index, dict) or index.get("schema") != SCHEMA or not isinstance(index.get("packs"), list):
                    raise ValueError("invalid candidate index")
                if index.get("nonce") is not None and index["nonce"] != nonce:
                    raise ValueError("candidate index publication identity changed")
                indexed = index["packs"]
        except (OSError, ValueError) as exc:
            issues.append({"kind": "candidate-index-invalid", "error": str(exc)})
        pending_retry = _checked_path(root, root/"journal/quarantine.pending.json")
        if pending_retry.exists(): issues.append({"kind": "quarantine-retry", "error": "previous cleanup incomplete"})
        journal = _checked_path(root, root/"journal")
        # A crash after the pack flush but before selection/index publication
        # leaves an unindexed pack. Its selected-parent digest is the lineage
        # authority; corrupt/foreign orphans stay diagnostics, never cursors.
        pending = list(indexed)
        for path in sorted(journal.glob("*.pack")):
            if path.stem not in pending and path.stem not in seen: pending.append(path.stem)
        for digest in pending:
            if isinstance(digest, str) and digest in seen: continue
            try:
                body = read_pack(digest)
                if nonce is None or body.get("nonce") != nonce:
                    raise ValueError("candidate publication identity differs from session")
                rows = {row["path"]: row["data"] for row in body["writes"] if row["immutable"]}
                if (selected is not None and body["parent"] is None
                    and all(files.get(name) == data for name, data in rows.items())):
                    continue  # bootstrap evidence already present in the selected closure
                if not (body["parent"] in seen or (selected is None and body["parent"] is None)
                    or (legacy and body["parent"] is None and body.get("legacy_parent_manifest") == selected["manifest"])):
                    raise ValueError("candidate parent outside selected lineage")
                for name, data in rows.items():
                    previous = files.get(name, candidates.get(name))
                    if previous is not None and previous != data:
                        raise ValueError("immutable candidate history changed")
                    _checked_path(root, root/name)
                candidates.update({name: data for name, data in rows.items() if name not in files})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append({"kind": "candidate-pack-invalid", "pack": digest, "error": str(exc)})
        recovered = {**files, **candidates}
        if legacy and issues:
            _RECOVERED[str(root)] = {"committed": set(), "candidates": {}, "issues": issues, "legacy_unsafe": True}
            return {"recovered_files": 0, "packs": 0, "candidate_issues": issues}
        # Validate every destination before materializing the first member.
        for name in recovered: _checked_path(root, root/name)
        restored = 0
        for name, data in recovered.items():
            path = root/name
            if not path.exists() or path.read_bytes() != data:
                _replace(path, data); restored += 1
        quarantined = []
        if issues:
            # A failed forensic write/cleanup must leave a fail-closed retry state.
            _RECOVERED[str(root)] = {"committed": set(files), "candidates": {}, "issues": issues,
                "quarantine_incomplete": True}
            pending_path = _checked_path(root, journal/"quarantine.pending.json")
            _replace(pending_path, _json({"schema": SCHEMA, "nonce": nonce, "issues": issues}), durable=True)
            # Do not let a bad pack's materialized CAS cache hit bypass future
            # publication. Preserve unqualified bytes in a forensic pack, then
            # remove their cache paths so only revalidated writes can replace them.
            for path in sorted(root.rglob("*")):
                if not path.is_file() and not path.is_symlink(): continue
                name = path.relative_to(root).as_posix()
                if name in recovered or name == "selected.json" or path.name == ".writer.lock" or name.startswith("journal/"): continue
                raw = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
                quarantined.append({"path": name, "kind": "symlink" if path.is_symlink() else "file",
                    "sha256": sha256(raw).hexdigest(), "bytes": base64.b64encode(raw).decode()})
            if quarantined:
                raw = gzip.compress(_json({"schema": SCHEMA, "kind": "quarantined-candidate-cache",
                    "issues": issues, "writes": quarantined}), compresslevel=1, mtime=0)
                path = _checked_path(root, journal/"quarantine"/(sha256(raw).hexdigest()+".pack"))
                _replace(path, raw, durable=True)
                directory = os.open(journal, os.O_RDONLY)
                try: os.fsync(directory)
                finally: os.close(directory)
                for row in quarantined: (root/row["path"]).unlink()
                for parent in {str((root/row["path"]).parent) for row in quarantined}:
                    directory = os.open(parent, os.O_RDONLY)
                    try: os.fsync(directory)
                    finally: os.close(directory)
            pending_path.unlink(missing_ok=True)
            directory = os.open(journal, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        _RECOVERED[str(root)] = {"committed": set(files),
            "candidates": {name: sha256(data).hexdigest() for name, data in candidates.items()}, "issues": issues,
            "quarantined_files": len(quarantined)}
        return {"recovered_files": restored, "packs": len(chain), **({"candidate_issues": issues} if issues else {})}


def writer_path(root):
    root = Path(root).resolve()
    return _checked_path(root, root/".writer.lock")


def recover_locked(root):
    """Acquire the session writer before recovery; hand it to the resident owner."""
    root = Path(root).resolve()
    writer = writer_path(root).open("a+")
    try:
        try: fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            from .errors import BenchError
            raise BenchError("another writer owns this session") from None
        BenchPublication.recover(root)
        if not (root/"selected.json").exists():
            from .errors import BenchError
            raise BenchError("no acknowledged session cursor; from_resident can retain initialization evidence")
        selected = json.loads((root/"selected.json").read_bytes())
        return writer, selected
    except BaseException:
        writer.close(); raise
