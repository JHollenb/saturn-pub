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
  may be open; ``Sink.close()`` stops one. A v3 record's stored form keeps its execution profile
  by sha256, so before the first record of each profile the sink writes the profile once as an
  ``{"mdb_execution": "mdb-execution-v1", "sha256", "execution"}`` line; ``Sink.records()``
  reads v3 and v2 lines (one file may hold both).

  Sink names (pre-release custody fix, 2026-11-14): shard jobs that all opened
  ``records-<tag>.jsonl`` were collected into one directory and only the last shard's records
  survived. So ``path`` may be omitted (``records-<run>.jsonl`` in the working directory), a
  directory (``<dir>/records-<run>.jsonl``) or contain ``{run}``; ``<run>`` is ``run_tag()``
  (``MRUN_JOB_ID`` when set, a timestamp and the pid: unique per process and job). An explicit
  file that already holds data and was not opened by this process is refused
  (``FileExistsError``) unless ``append=True``.

  Payloads: a record too large for one line keeps a sha256 pointer to the full payload
  (``bounded.put_json``; ``state: "absent"`` with no project store, e.g. a compact
  ``Branches.seal``'s ``arms_payload``). While a sink is open the payload bytes are also written
  to it as one ``{"mdb_payload": ...}`` line, so the full payload (per-arm probe values) is sealed
  in the file; ``Sink.load_json(pointer)`` reads it back sha256-checked. Records are unchanged:
  a sink never alters a record's id.

The environment can declare custody for a job payload with no code change: ``MDB_SCRIPT`` and
``MDB_PREREG`` (paths) and the kernel path (``custody.kernel_path`` of ``mdb.core.settings``:
``MDB_CUSTODY_KERNEL_PATH``, the older ``MDB_KERNEL_PATH``, or a config file; any one of them
declares) are read on first use (an unreadable file raises on every seal, never once). The
record sink is ``records.sink`` (``MDB_RECORDS_SINK``, the older ``MDB_RECORDS``, or a config
file; same naming rules) with ``records.append`` and ``records.payloads``. The sink is opened on
first use even when custody was declared in code first.

Config files: when any was loaded (``settings.current().provenance()``), every record carries
custody, and the block has ``config`` = ``{schema, sources: [{path, sha256, layer}], resolved,
origins}``: the files' bytes and the resolved values of the settings that reach a record. With
no config file, nothing changes (record ids are as before).
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
# env_read: custody from the environment/settings done (also set by declare/clear);
# sinks_read: the configured sink opened (independent: declaring custody in code first must not
# skip the sink).
_STATE: dict[str, Any] = {"declared": None, "env_read": False, "sinks_read": False,
                          "env_error": None}
_SINKS: list[Sink] = []
_OWNED: set[str] = set()          # sink files this process opened (it may reopen and append)
PAYLOAD_SCHEMA = "mdb-payload-v1"
_RUN_TAG: list[str] = []


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


def _settings() -> Any:
    from . import settings

    return settings


def _from_env() -> None:
    """Declare custody and open the sink from the environment and settings once. A failure (a
    missing ``MDB_SCRIPT`` file, a bad config file) is kept and raised on EVERY later call:
    custody never drops silently after the first error. Setting ``_STATE["env_read"] = False``
    re-reads both (the environment changed)."""
    if _STATE["env_read"] and _STATE["sinks_read"]:
        if _STATE["env_error"] is not None:
            raise _STATE["env_error"]
        return
    again = not _STATE["env_read"]
    try:
        settings = _settings()
        resolved = settings.refresh() if again else settings.current()
        if again:
            script, prereg = os.environ.get("MDB_SCRIPT"), os.environ.get("MDB_PREREG")
            kernel = resolved.get("custody.kernel_path")
            if script or prereg or kernel:
                declare(script=script, prereg=prereg, kernel_path=kernel)
        if again or not _STATE["sinks_read"]:
            sink = resolved.get("records.sink")
            if sink:
                target = sink_path(sink)
                if not any(s.path == target.absolute() or s.path == target for s in _SINKS):
                    records_to(target, append=resolved.get("records.append"),
                               payloads=resolved.get("records.payloads"))
    except BaseException as error:
        _STATE["env_read"], _STATE["sinks_read"], _STATE["env_error"] = True, True, error
        raise
    _STATE["env_read"], _STATE["sinks_read"], _STATE["env_error"] = True, True, None


def config_block() -> dict[str, Any] | None:
    """The loaded config files and resolved recorded settings, or None with no config file."""
    _from_env()
    return _settings().current().provenance()


def block_with_libraries(block: dict[str, Any]) -> dict[str, Any]:
    return {**block, "libraries": libraries()}


def _undeclared() -> dict[str, Any]:
    script = _main_script()
    return {"schema": SCHEMA, "script": script,
            "script_sha256": file_sha256(script) if script else None,
            "prereg": None, "prereg_sha256": None, "kernel_path": "undeclared",
            "declared_at": None}


def current() -> dict[str, Any] | None:
    """The declared custody block with library versions now, or None when undeclared and no
    config file was loaded. With a config file, a block is always returned (the ``__main__``
    script when undeclared) and carries ``config``."""
    _from_env()
    block = _STATE["declared"]
    config = config_block()
    if config is not None:
        block = {**(block if block is not None else _undeclared()), "config": config}
    return None if block is None else block_with_libraries(block)


def custody(kernel_path: str | None = None) -> dict[str, Any]:
    """A custody block for a record that always carries one (branches, pools): the declared
    custody if any, else the ``__main__`` script (when a file) and the library versions;
    ``kernel_path`` overrides the declared one."""
    block = current()
    if block is None:
        block = block_with_libraries(_undeclared())
    if kernel_path is not None:
        block["kernel_path"] = kernel_path
    return block


class Sink:
    """An append-only JSONL file of sealed records (one canonical record per line)."""

    def __init__(self, path: str | Path, *, payloads: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.payload_count = 0
        self.payloads_on = bool(payloads)
        self._executions: set[str] = set()   # execution profiles this sink has written

    def _append(self, line: str) -> None:
        with _LOCK, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write(self, record: dict[str, Any]) -> None:
        """Append one record's stored form. A v3 record keeps its execution profile by sha256,
        so the profile is written first, once per sink, as an ``mdb_execution`` line."""
        from .records import EXECUTION_SCHEMA, SCHEMA, stored_record
        from .records import _canonical as canonical

        stored = stored_record(record)
        if record.get("schema") == SCHEMA and "execution" in record:
            digest = stored["execution_sha256"]
            if digest not in self._executions:
                self._append(canonical({"mdb_execution": EXECUTION_SCHEMA, "sha256": digest,
                                        "execution": record["execution"]}))
                self._executions.add(digest)
        self._append(canonical(stored))
        self.count += 1

    def write_payload(self, digest: str, raw: bytes) -> None:
        """One payload line: the exact JSON bytes a record's pointer names by ``sha256``."""
        if not self.payloads_on:
            return
        line = json.dumps({"mdb_payload": PAYLOAD_SCHEMA, "sha256": digest, "bytes": len(raw),
                           "raw": raw.decode("utf-8")}, sort_keys=True, separators=(",", ":"))
        self._append(line)
        self.payload_count += 1

    def _lines(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        from .pointers import RecordError
        from .records import parse_line

        out = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            # a payload line may exceed the record bound (it carries a bounded object's bytes)
            big = line.startswith('{"bytes":') and '"mdb_payload"' in line[:200]
            value = parse_line(line, f"{self.path}:{number}", limit=None if big else _LIMIT())
            if not isinstance(value, dict):
                raise RecordError(f"{self.path}:{number}: not a record, payload or execution line")
            out.append(value)
        return out

    def records(self, *, verify: bool = True) -> list[dict[str, Any]]:
        """The records written to this file (by any process), as logical records (v3 and v2),
        verified unless ``verify=False``. Payload and execution lines are not records and are
        skipped (``payloads()``, ``executions()``)."""
        from .records import load_record

        lines = self._lines()
        executions = _executions(lines)
        return [load_record(r, executions=executions, verify=verify) for r in lines
                if "mdb_payload" not in r and "mdb_execution" not in r]

    def executions(self) -> dict[str, Any]:
        """sha256 -> execution profile for every ``mdb_execution`` line, each checked."""
        return _executions(self._lines())

    def payloads(self) -> dict[str, bytes]:
        """sha256 -> payload bytes for every payload line, each checked against its sha256."""
        out = {}
        for line in self._lines():
            if "mdb_payload" not in line:
                continue
            raw = line["raw"].encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != line["sha256"]:
                raise ValueError(f"payload {line['sha256'][:12]} differs from its sha256")
            out[line["sha256"]] = raw
        return out

    def load_json(self, pointer: dict[str, Any]) -> Any:
        """The full object a ``bounded.put_json`` pointer names, from this file's payloads."""
        raw = self.payloads().get(pointer["sha256"])
        if raw is None:
            raise ValueError(f"no payload {pointer['sha256'][:12]} in {self.path}")
        return json.loads(raw)

    def close(self) -> None:
        with _LOCK:
            if self in _SINKS:
                _SINKS.remove(self)

    def __enter__(self) -> Sink:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _LIMIT() -> int:
    from .records import MAX_LINE_BYTES

    return MAX_LINE_BYTES


def _executions(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """sha256 -> profile of every ``mdb_execution`` line; a malformed or edited one is a
    ``RecordError``."""
    from .pointers import RecordError
    from .records import seal_digest

    out = {}
    for line in lines:
        if "mdb_execution" not in line:
            continue
        try:
            sha, profile = line["sha256"], line["execution"]
            same = seal_digest(profile) == sha
        except (KeyError, TypeError, ValueError) as error:
            raise RecordError(f"execution line is malformed: {type(error).__name__}: "
                              f"{error}") from error
        if not same:
            raise RecordError(f"execution profile {sha[:12]} differs from its sha256")
        out[sha] = profile
    return out


def run_tag() -> str:
    """This process's sink tag: ``<MRUN_JOB_ID>-`` (when set) + start time + ``pid<pid>``."""
    if not _RUN_TAG:
        job = os.environ.get("MRUN_JOB_ID", "").strip()
        parts = [job] if job else []
        parts += [time.strftime("%Y%m%dT%H%M%S"), f"pid{os.getpid()}"]
        _RUN_TAG.append("".join(c if c.isalnum() or c in "-_." else "_"
                                for c in "-".join(parts)))
    return _RUN_TAG[0]


def sink_path(path: str | Path | None = None) -> Path:
    """Where a sink named ``path`` writes: ``None`` -> ``./records-<run>.jsonl``; a directory
    (existing, ending in a separator, or a name with no suffix that does not exist yet, so
    ``Path("out/sinks")`` is a directory and ``Path("out/records.jsonl")`` a file) ->
    ``<dir>/records-<run>.jsonl``; ``{run}`` in the name
    -> replaced by ``run_tag()``; any other path as given."""
    name = f"records-{run_tag()}.jsonl"
    if path is None:
        return Path.cwd() / name
    text = str(path)
    if "{run}" in text:
        return Path(text.replace("{run}", run_tag()))
    as_path = Path(text)
    if (text.endswith(("/", os.sep)) or as_path.is_dir()
            or (not as_path.suffix and not as_path.exists())):
        return as_path / name
    return as_path


def records_to(path: str | Path | None = None, *, append: bool = False,
               payloads: bool = True) -> Sink:
    """Open a sink: every record sealed from now on is appended to ``sink_path(path)``.

    A file that already holds data and that this process did not open is refused unless
    ``append=True`` (two shards writing one name lose records when collected); use a directory,
    ``{run}`` or no path for a unique per-process name. ``payloads=False`` keeps the file to
    records only (no ``mdb_payload`` lines). Calling it again for a path that already has an open
    sink returns that sink (so re-running a notebook cell does not write every record twice)."""
    target = sink_path(path).absolute()
    key = str(target)
    if (not append and key not in _OWNED and target.is_file()
            and target.stat().st_size > 0):
        raise FileExistsError(
            f"record sink {target} already holds records this process did not write; pass "
            "append=True to add to it, or give a directory / a name with {run} / no path for a "
            f"unique per-run file (records-{run_tag()}.jsonl)")
    with _LOCK:
        for open_sink in _SINKS:
            if str(open_sink.path.absolute()) == key:
                open_sink.payloads_on = bool(payloads)
                return open_sink
    sink = Sink(target, payloads=payloads)
    with _LOCK:
        _OWNED.add(key)
        _SINKS.append(sink)
    return sink


def emit_payload(digest: str, raw: bytes) -> None:
    """Hand a payload's bytes to every open sink (``bounded.put_json`` calls this when it has no
    store, so a pointer with ``state: "absent"`` is still sealed in the sink file)."""
    _from_env()
    for sink in list(_SINKS):
        sink.write_payload(digest, raw)


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


__all__ = ["CORE_LIBRARIES", "INTEROP_LIBRARIES", "PAYLOAD_SCHEMA", "SCHEMA", "Sink",
           "attach_custody", "clear", "config_block", "current", "custody", "declare", "emit",
           "emit_payload", "file_sha256", "libraries", "records_to", "run_tag", "sink_path", "sinks"]
