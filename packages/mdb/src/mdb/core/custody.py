"""Record custody and record sinks (gap audit 2026-11-14, P8).

The showcase audit could prove sealing order only from git commit times, flagged that a prereg's
"written" time is unreliable and that ``REVISION`` pinned the library but not the driver; three
drivers kept only record ids, so the sealed bodies were never stored. Two debugger switches fix
that for every sealed record (forecast, trace, watch-hits, circuit-map, supplied-forecast,
branch, pool, ...), because they act inside ``records.seal_record``:

* ``declare(script=, prereg=, kernel_path=)``: custody for this process. While declared, every
  record sealed gets ``body.custody`` = ``{schema, script, script_sha256, prereg,
  prereg_sha256, libraries, kernel_path, declared_at}`` (files hashed when declared: the
  bytes that ran, not the bytes at audit time; ``libraries`` = versions of mdb, saturn_pub,
  torch, transformers and every interop library already imported). A record that already carries
  ``custody`` (``Branches``, ``Pool``: they always do) keeps its own. Undeclared, records are
  sealed exactly as before (the REPL goldens and every stored id are unchanged).
* ``records_to(path)``: a sink. Every record sealed after it (in any module) is appended to
  ``path`` as one canonical JSON line (``records.stored_form``, the digest's form), so a record
  persists with no project attached; ``Sink.records()`` reads them back verified. Several sinks
  may be open; ``Sink.close()`` stops one.

The environment can declare custody for a job payload with no code change: ``MDB_SCRIPT`` and
``MDB_PREREG`` (paths) and ``MDB_KERNEL_PATH`` (any one of them declares) are read on first use
(an unreadable file raises on every seal, never once);  ``MDB_RECORDS`` opens a
sink.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = "mdb-custody-v1"
CORE_LIBRARIES = ("mdb", "saturn_pub", "torch", "transformers")
INTEROP_LIBRARIES = ("nnsight", "sae_lens", "transformer_lens", "pyvene", "circuit_tracer",
                     "safetensors", "numpy")
_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"declared": None, "env_read": False, "env_error": None}
_SINKS: list[Sink] = []


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _version(name: str) -> str | None:
    module = sys.modules.get(name)
    if module is None:
        return None
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as dist_version

            version = dist_version(name.replace("_", "-"))
        except Exception:
            version = "unknown"
    return str(version)


def libraries() -> dict[str, str]:
    """Versions of the core and interop libraries this process has imported (nothing is
    imported here: a record about a run was made with what is loaded)."""
    out = {}
    for name in (*CORE_LIBRARIES, *INTEROP_LIBRARIES):
        found = _version(name)
        if found is not None:
            out[name] = found
    return out


def _main_script() -> str | None:
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    return path if path and Path(path).is_file() else None


def declare(*, script: str | Path | None = None, prereg: str | Path | None = None,
            kernel_path: str | None = None, note: str | None = None) -> dict[str, Any]:
    """Declare custody for every record sealed from now on (returns the custody block).
    ``script`` defaults to the ``__main__`` file when there is one."""
    script = script if script is not None else _main_script()
    block: dict[str, Any] = {"schema": SCHEMA, "script": None, "script_sha256": None,
                             "prereg": None, "prereg_sha256": None,
                             "kernel_path": kernel_path or "undeclared",
                             "declared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if script is not None:
        block["script"], block["script_sha256"] = str(script), file_sha256(script)
    if prereg is not None:
        block["prereg"], block["prereg_sha256"] = str(prereg), file_sha256(prereg)
    if note:
        block["note"] = str(note)
    with _LOCK:
        _STATE["declared"] = block
        _STATE["env_read"] = True
        _STATE["env_error"] = None
    return block_with_libraries(block)


def clear() -> None:
    """Forget declared custody (records seal as before)."""
    with _LOCK:
        _STATE["declared"] = None
        _STATE["env_read"] = True
        _STATE["env_error"] = None


def _from_env() -> None:
    """Declare custody from the environment once. A failure (a missing ``MDB_SCRIPT`` file) is
    kept and raised on EVERY later call: custody never drops silently after the first error."""
    if _STATE["env_read"]:
        if _STATE["env_error"] is not None:
            raise _STATE["env_error"]
        return
    try:
        script, prereg = os.environ.get("MDB_SCRIPT"), os.environ.get("MDB_PREREG")
        kernel = os.environ.get("MDB_KERNEL_PATH")
        if script or prereg or kernel:
            declare(script=script, prereg=prereg, kernel_path=kernel)
        sink = os.environ.get("MDB_RECORDS")
        if sink and not any(str(s.path) == sink for s in _SINKS):
            records_to(sink)
    except BaseException as error:
        _STATE["env_read"], _STATE["env_error"] = True, error
        raise
    _STATE["env_read"] = True


def block_with_libraries(block: dict[str, Any]) -> dict[str, Any]:
    return {**block, "libraries": libraries()}


def current() -> dict[str, Any] | None:
    """The declared custody block with library versions now, or None when undeclared."""
    _from_env()
    block = _STATE["declared"]
    return None if block is None else block_with_libraries(block)


def custody(kernel_path: str | None = None) -> dict[str, Any]:
    """A custody block for a record that always carries one (branches, pools): the declared
    custody if any, else the ``__main__`` script (when a file) and the library versions;
    ``kernel_path`` overrides the declared one."""
    block = current()
    if block is None:
        script = _main_script()
        block = block_with_libraries({
            "schema": SCHEMA, "script": script,
            "script_sha256": file_sha256(script) if script else None,
            "prereg": None, "prereg_sha256": None, "kernel_path": "undeclared",
            "declared_at": None})
    if kernel_path is not None:
        block["kernel_path"] = kernel_path
    return block


class Sink:
    """An append-only JSONL file of sealed records (one canonical record per line)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        from .records import stored_form

        line = stored_form(record)
        with _LOCK, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self.count += 1

    def records(self, *, verify: bool = True) -> list[dict[str, Any]]:
        """The records written to this file (by any process), verified unless ``verify=False``."""
        from .records import verify_record

        out = []
        if not self.path.exists():
            return out
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                out.append(verify_record(record) if verify else record)
        return out

    def close(self) -> None:
        with _LOCK:
            if self in _SINKS:
                _SINKS.remove(self)

    def __enter__(self) -> Sink:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def records_to(path: str | Path) -> Sink:
    """Open a sink: every record sealed from now on is appended to ``path``."""
    sink = Sink(path)
    with _LOCK:
        _SINKS.append(sink)
    return sink


def sinks() -> list[Sink]:
    return list(_SINKS)


def attach_custody(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    """``body`` with the declared custody added (unchanged when undeclared or already there)."""
    if "custody" in body:
        return body
    block = current()
    return body if block is None else {**body, "custody": block}


def emit(record: dict[str, Any]) -> None:
    """Hand a sealed record to every open sink (``records.seal_record`` calls this)."""
    _from_env()
    for sink in list(_SINKS):
        sink.write(record)


__all__ = ["CORE_LIBRARIES", "INTEROP_LIBRARIES", "SCHEMA", "Sink", "attach_custody", "clear",
           "current", "custody", "declare", "emit", "file_sha256", "libraries", "records_to",
           "sinks"]
