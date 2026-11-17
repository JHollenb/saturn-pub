"""Checkpoints: a run's state at a cut as content-addressed pointers to its tensors (#517).

One checkpoint object serves ``Run.checkpoint(durable=...)``, ``Run.dump`` / ``Debugger.run_from``
(the core files of mdb 0.2.2) and ``Debugger.checkpoint`` / ``restart`` (a project):

* **What is stored.** The native ``StateCut`` of every trail mark kept (``history``), slot by
  slot: per layer the key/value cache, the hidden carrier of a row in flight (a cut inside a row
  is a layer boundary ``t<r>.L<i>.h_out``), logits, the sampled token, the stepped-prompt slot,
  plus the cut's boundary, execution point, surface and **parent fingerprint**. Each tensor is
  stored as parts (the block form the engine already holds: the prefix cache once, then one
  part per appended row), each part addressed by the sha256 of its bytes; identical parts are
  stored once. The cursor (a port inside the pending layer and the edits staged there), the
  committed edits of every transition kept, the edit and mask tensors (including those of the
  path before the first kept mark), the active holds (described, not re-armed), the run's
  recipe (``Run.checkpoint()``: provenance and the input of ``verify``) and optional captured
  values complete it.
* **Where.** A self-contained file (``torch.save`` of plain data, so ``weights_only`` loading
  works) or a project (``mdb.core.project.Project``): every part a tensor blob of its store, the
  document a JSON blob, the record indexed under the session. A checkpoint is addressed by its
  file path or by its record id (a unique prefix of at least 8 hex digits).
* **Record.** A sealed ``checkpoint-state`` record (schema ``mdb-record-v3``) holding the
  document's sha256, every part's sha256, the cut fingerprints, the prompt digest, the
  procedure, the lineage (``root``: ``prompt`` when the cut's fingerprints chain back to a run
  of the prompt; ``checkpoint`` when they chain back to a foreign root, a replica or a legacy
  core file) and custody.
* **Rehydration** (``Checkpoint.run``) rebuilds every kept cut from its parts, recomputes its
  fingerprint from the content and refuses one that differs from the record; the run's trail is
  the kept marks with their committed edits and the cursor is put back (port, staged edits).
  Nothing before the cursor is recomputed; moving to a port inside the pending layer is one
  probe of that layer, exactly as on the source run. The fingerprints are the source run's,
  so checkpoints transfer between the source and the reloaded run in both directions.
* **Honest identity.** A cut's fingerprint is the sha256 of the model identity, the execution
  contract (dtype, device, numeric scope, library versions), the state content, the execution
  point and the parent fingerprint. Rehydration recomputes it from the stored content and the
  recorded parent, so a changed tensor, another model or another execution never verifies. The
  parent is recorded, not re-derived: ``verify`` (``run(..., verify=True)``) replays the recipe
  from the lineage root (the prompt) and checks it reaches the same fingerprint (measured).
* **Replicas.** Another dtype, device or weights of the same shape is refused unless
  ``replica=True``: then the cursor's state (cast once to the target dtype) roots a new lineage,
  labelled a replica, and ``run.core_check`` compares the captured values (``null_check``).

Structural refusals: another family, layer count, KV heads, head_dim or vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .ports import tensor_digest

SCHEMA = "mdb-checkpoint-v2"            # the state document
FILE_SCHEMA = "mdb-checkpoint-file-v2"  # a self-contained checkpoint file
KIND = "checkpoint-state"               # the record kind (declared in mdb.core.kinds)
HISTORIES = ("cursor", "row", "root")
_FILE_KEYS = {"schema", "record", "document", "tensors"}
_SHAS_PER_TEXT = 60                     # packed sha256 hex per record string (<= 4 KiB)
_MAX_INLINE_SHAS = 60 * _SHAS_PER_TEXT  # beyond: the record holds their digest only (the
                                        # sealed document lists them)


def shape_of(adapter: Any) -> dict[str, Any]:
    return {"family": adapter.spec.model_type, "layers": int(adapter.layers),
            "kv_heads": int(adapter.num_kv_heads), "head_dim": int(adapter.head_dim),
            "vocab": int(adapter.vocab_size)}


def check_shape(want: Mapping[str, Any], adapter: Any, what: str = "checkpoint") -> None:
    have = shape_of(adapter)
    differ = [k for k in want if want[k] != have.get(k)]
    if differ:
        raise ValueError(f"{what} of another model shape: " + ", ".join(
            f"{k} {want[k]} vs {have.get(k)}" for k in differ))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


def _plain(value: Any, what: str) -> Any:
    """``value`` as plain JSON data (a JSON round trip), refusing anything that is not."""
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{what} is not plain data: {error}") from error


def aligned_blocks(length: int) -> list[tuple[int, int]]:
    """``[start, end)`` row blocks of ``0..length``: power-of-two sizes, largest first, each
    aligned to its size (``length`` 100 is 64 + 32 + 4): at most ``log2(length) + 1`` blocks,
    and a longer run of the same rows shares every block but the last few."""
    out, start = [], 0
    size = 1 << max(length.bit_length() - 1, 0)
    while start < length:
        while start + size > length:
            size >>= 1
        out.append((start, start + size))
        start += size
    return out


class _Writer:
    """Collects the parts (content-addressed) and the tensor descriptors of one checkpoint."""

    def __init__(self, store: Any = None):
        self.store = store
        self.parts: list[dict[str, Any]] = []
        self.blobs: dict[str, torch.Tensor] = {}
        self._part_index: dict[str, int] = {}
        self._part_seen: dict[int, tuple[Any, int]] = {}
        self.tensors: list[dict[str, Any]] = []
        self._seen: dict[Any, tuple[Any, int]] = {}
        self._keep: list[Any] = []         # chunks named by part memo keys stay alive meanwhile
        self.bytes = 0

    def part(self, t: torch.Tensor, memo: Any = None) -> int:
        key0 = memo if memo is not None else id(t)
        seen = self._part_seen.get(key0)
        if seen is not None and (memo is not None or seen[0] is t):
            return seen[1]
        flat = t.detach().to("cpu").contiguous()
        sha = tensor_digest(flat)
        key = part_key(sha, _dtype_name(flat), list(flat.shape))
        index = self._part_index.get(key)
        if index is None:
            pointer: dict[str, Any] = {"sha256": sha, "shape": list(flat.shape),
                                       "dtype": _dtype_name(flat)}
            if self.store is not None:
                pointer["ref"] = self.store.put_tensor(flat)["ref"]
            else:
                # a compact copy: torch.save of a view would write its whole storage
                self.blobs[key] = flat.clone(memory_format=torch.contiguous_format)
            self.bytes += flat.numel() * flat.element_size()
            self.parts.append(pointer)
            index = self._part_index[key] = len(self.parts) - 1
        self._part_seen[key0] = (t, index)
        return index

    def tensor(self, value: Any) -> int:
        """The descriptor index of a tensor or ``saturn_pub.core.Blocks`` (one per held
        tensor: the same chunks give the same index)."""
        from saturn_pub.core import Blocks

        if not isinstance(value, Blocks):
            value = Blocks((value,), None, tuple(value.shape), tuple(value.stride()),
                           value.dtype)
        key = (value.dim, tuple(id(c) for c in value.chunks))
        seen = self._seen.get(key)
        if seen is not None:
            return seen[1]
        chunks, axis = value.chunks, value.dim
        if axis is None:
            parts = [self.part(chunks[0])]
        else:
            parts = self._aligned(chunks, axis, int(value.shape[axis]))
        descriptor = {"dtype": str(value.dtype).removeprefix("torch."),
                      "shape": list(value.shape), "stride": list(value.stride), "axis": axis,
                      "parts": parts}
        self.tensors.append(descriptor)
        index = len(self.tensors) - 1
        self._seen[key] = (value, index)       # keeps the chunks alive while ids are keys
        return index

    def _aligned(self, chunks: tuple[torch.Tensor, ...], axis: int, length: int) -> list[int]:
        """Parts of the rows ``0..length`` held as ``chunks`` along ``axis``: the first chunk
        whole (the prefix cache every cut of the run shares; for a rehydrated run, the part it
        was rehydrated from), then the rows after it in ``aligned_blocks`` blocks counted from
        its end, so a cut one row later shares every part but the last few. A block inside one
        chunk is hashed once per chunk."""
        offsets, at = [], 0
        for chunk in chunks:
            offsets.append(at)
            at += int(chunk.shape[axis])
        base = int(chunks[0].shape[axis])
        blocks = ([(0, base)] if base else []) + [
            (base + a, base + b) for a, b in aligned_blocks(length - base)]
        out = []
        for start, end in blocks:
            inside = [(c, o) for c, o in zip(chunks, offsets)
                      if o <= start and end <= o + int(c.shape[axis])]
            if inside:
                chunk, offset = inside[0]
                piece = chunk.narrow(axis, start - offset, end - start)
                out.append(self.part(piece, memo=(id(chunk), start - offset, end - start)))
                self._keep.append(chunk)
                continue
            pieces = [c.narrow(axis, max(start, o) - o, min(end, o + int(c.shape[axis]))
                               - max(start, o))
                      for c, o in zip(chunks, offsets)
                      if o < end and start < o + int(c.shape[axis])]
            out.append(self.part(torch.cat(pieces, dim=axis)))
        return out

    def encode(self, value: Any) -> Any:
        from saturn_pub.core import Blocks

        if isinstance(value, (torch.Tensor, Blocks)):
            return {"kind": "tensor", "t": self.tensor(value)}
        if isinstance(value, Mapping):
            if any(not isinstance(k, str) for k in value):
                raise ValueError("a checkpoint state mapping needs string keys")
            return {"kind": "mapping", "items": {k: self.encode(v) for k, v in value.items()}}
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [self.encode(v) for v in value]}
        if isinstance(value, list):
            return {"kind": "list", "items": [self.encode(v) for v in value]}
        if value is None or type(value) in (bool, int, float, str):
            return {"kind": "scalar", "value": value}
        raise ValueError(f"unsupported checkpoint state value: {type(value).__name__}")


def part_key(sha: str, dtype: str, shape: list[int]) -> str:
    return f"{sha}.{dtype}.{'x'.join(str(int(s)) for s in shape) or 'scalar'}"


def _packed(shas: list[str]) -> list[str]:
    return ["".join(shas[i:i + _SHAS_PER_TEXT]) for i in range(0, len(shas), _SHAS_PER_TEXT)]


def _unpacked(texts: list[str]) -> list[str]:
    return [t[i:i + 64] for t in texts for i in range(0, len(t), 64)]


# --------------------------------------------------------------------------- write

def _mark_address(run: Any, index: int) -> str:
    return run.address(run._mark_key(run._trail[index]))


def _recipe_at(run: Any, index: int) -> dict[str, Any]:
    """``Run.checkpoint()`` of trail mark ``index`` (the cursor's own when it is the last)."""
    from saturn_pub.values import digest

    from .run import CHECKPOINT_SCHEMA

    if index == len(run._trail) - 1:
        return run.checkpoint(pin=False)
    mark = run._trail[index]
    body = {"schema": CHECKPOINT_SCHEMA, "name": None,
            "model_identity": run.adapter.model_identity, "prompt_sha256": digest(run.tokens),
            "prefill": run.prefill, "where": _mark_address(run, index), "cut": mark.fingerprint,
            "transitions": mark.transitions, "port": None, "lookahead": run.lookahead,
            "numeric_scope": run._numeric_scope(), "edits": [],
            "path_edits": run._path_manifest(index)}
    supplied = [[m.transitions, m.supplied] for m in [*run._ancestry, *run._trail[:index]]
                if m.supplied is not None]
    if supplied:
        body["supplied"] = supplied
    return body


def _mark_entry(mark: Any, committed: bool) -> dict[str, Any]:
    """A trail mark as plain data: its fingerprint and place and, when the transition after it
    was committed (``committed``), that transition's edits, cells, records and fed token."""
    entry: dict[str, Any] = {
        "fingerprint": mark.fingerprint, "transitions": mark.transitions, "row": mark.row,
        "label": mark.label, "phase": mark.phase, "layer": mark.layer,
        "supplied": mark.supplied if committed else None, "cut": None,
        "edits": [], "cells": [], "fired": [], "cell_fired": []}
    if committed:
        entry["edits"] = [e.manifest() for e in mark.edits]
        entry["cells"] = [c.manifest() for c in mark.cells]
        entry["fired"] = _plain(list(mark.fired), "a fired edit record")
        entry["cell_fired"] = _plain(list(mark.cell_fired), "a cell realisation record")
    return entry


def _target_index(run: Any, at: str | None, row: int | None) -> int:
    if at is not None and row is not None:
        raise ValueError("give at= or row=, not both")
    if row is not None:
        at = f"t{int(row)}.start"
    if at is None:
        return len(run._trail) - 1
    key = run.parse(at)
    found = [i for i, m in enumerate(run._trail) if run._mark_key(m) == key]
    if not found:
        if key == run._key():
            return len(run._trail) - 1
        where = run.address(key)
        if key < run._mark_key(run._trail[0]):
            raise ValueError(f"the run holds no {where} boundary: it is before the run root "
                             f"{run.root}; save at a stop the run has reached")
        raise ValueError(f"the run holds no {where} boundary (first stepped row "
                         f"t{run.first_row}, cursor {run.where}); save at the cursor or at a "
                         "boundary the run has passed (a row start, a layer boundary)")
    return found[-1]


def _later_edits(run: Any, index: int) -> list[str]:
    """What the run changed after trail mark ``index`` that a checkpoint there would not hold."""
    out = [f"{run.address(run._mark_key(m))} ({len(m.edits) + len(m.cells)} committed)"
           for m in run._trail[index:-1] if m.edits or m.cells]
    if index != len(run._trail) - 1 and (run._edits or run._cells):
        out.append(f"{run.where} ({len(run._edits) + len(run._cells)} staged)")
    return out


def _first_kept(run: Any, index: int, history: str) -> int:
    if history == "cursor":
        return index
    if history == "root":
        return 0
    row = run._trail[index].row
    return min(i for i in range(index + 1) if run._trail[i].row == row)


def _holds(run: Any, writer: Any, from_row: int, holds: str) -> list[dict[str, Any]]:
    """The active holds, each with the source values it reads from ``from_row`` on (tensor
    indices), its source's cursor and cut, so a reloaded run re-arms it exactly; ``holds``
    ``"drop"``: described and marked dropped (the reloaded run continues free)."""
    from .holds import source_addresses
    from .refs import run_label

    out = []
    for hold in run.hold_objects():
        entry = {**hold.describe(), "cells": [dict(c) for c in hold.cells],
                 "dropped": holds == "drop"}
        if holds == "rearm":
            values = {}
            for address in source_addresses(run, hold, from_row):
                try:
                    value = hold.source.read(address)
                except ValueError:
                    continue            # not computed by the source (yet): refused on reach
                values[address] = writer.tensor(value.detach().contiguous())
            entry["snapshot"] = {"source": run_label(hold.source), "where": hold.source.where,
                                 "cut": hold.source.checkpoint(pin=False)["cut"],
                                 "tokens": list(hold.source.tokens), "values": values}
        out.append(entry)
    return out


def build(run: Any, *, store: Any = None, name: str | None = None, at: str | None = None,
          row: int | None = None, history: str = "row",
          capture: tuple[str, ...] | None = None, holds: str = "rearm"
          ) -> tuple[dict[str, Any], _Writer, dict]:
    """The state document of ``run`` at ``at`` (default: the cursor) and its writer (parts put
    in ``store`` when given, else held for a file), plus the record body facts."""
    if history not in HISTORIES:
        raise ValueError(f"history is one of {', '.join(HISTORIES)}")
    index = _target_index(run, at, row)
    where = run.where if index == len(run._trail) - 1 else _mark_address(run, index)
    later = _later_edits(run, index)
    cursor_is_cut = index == len(run._trail) - 1
    last = len(run.tokens) - 1
    if capture is None:
        computed = run.parse(f"t{last}.logits") <= run._key()
        capture = (f"t{last}.logits",) if computed and not later else ()
    elif later and capture:
        raise ValueError(
            f"the run changed its state after {where}: {', '.join(later)}; a captured value "
            "would include those edits but the checkpoint holds the state at "
            f"{where}; save at the cursor (at=None) or with capture=()")
    if later and not cursor_is_cut:
        raise ValueError(
            f"the run edited after {where}: {', '.join(later)}; a checkpoint at {where} would "
            "not hold those edits; save at the cursor (at=None, the default: a cut inside a "
            "row is kept exactly) or before editing")
    first = _first_kept(run, index, history)
    stored = {first, index}
    stored.update(i for i in range(first, index + 1)
                  if run._trail[i].label == "start" or run._trail[i].row == run._trail[index].row)
    from saturn_pub.core import cut_blocks

    writer = _Writer(store)
    ancestry = [_mark_entry(m, True) for m in [*run._ancestry, *run._trail[:first]]
                if m.edits or m.cells or m.supplied is not None]
    marks = []
    for i in range(first, index + 1):
        mark = run._trail[i]
        entry = _mark_entry(mark, i < index)
        if i in stored:
            cut = run._cut_of(i)
            if cut.fingerprint != mark.fingerprint:
                raise RuntimeError(f"the cut of {_mark_address(run, i)} differs from its mark")
            entry["cut"] = {
                "boundary": cut.boundary, "parent": cut.parent,
                "execution_point": _plain(dict(cut.execution_point or {}), "an execution point")
                if cut.execution_point is not None else None,
                "surface": _plain(dict(cut.surface), "a state surface")
                if cut.surface is not None else None,
                "payload": writer.encode(cut_blocks(cut))}
        marks.append(entry)
    tensors = run.edit_tensors()
    edit_index = {sha: writer.tensor(t) for sha, t in sorted(tensors.items())}
    captured = {}
    for address in capture:
        value = run.read(address).detach().to("cpu").contiguous().clone()
        captured[address] = writer.tensor(value)
    if holds not in ("rearm", "drop"):
        raise ValueError("holds is 'rearm' (default) or 'drop'")
    if run.hold_objects() and not cursor_is_cut and holds == "rearm":
        raise ValueError(f"the run holds ports ({', '.join(h.name for h in run.hold_objects())})"
                         f" now, not necessarily at {where}: save at the cursor to re-arm them, "
                         "or pass holds='drop' (recorded)")
    held = _holds(run, writer, run._trail[index].row, holds)
    staged_edits = [e.manifest() for e in run._edits] if cursor_is_cut else []
    staged_cells = [c.manifest() for c in run._cells] if cursor_is_cut else []
    adapter = run.adapter
    doc = {
        "schema": SCHEMA, "name": name,
        "model_identity": _plain(adapter.model_identity, "the model identity"),
        "execution": _plain(dict(adapter.execution), "the execution contract"),
        "shape": shape_of(adapter),
        "prompt": list(run.tokens), "prefill": int(run.prefill), "lookahead": run.lookahead,
        "history": history, "where": where,
        "cursor": {"port": run._port if cursor_is_cut else None,
                   "edits": staged_edits, "cells": staged_cells},
        "ancestry": ancestry,
        "marks": marks,
        "edit_tensors": edit_index,
        "captured": captured,
        "holds": held,
        "recipe": _recipe_at(run, index),
        "lineage": dict(run.lineage),
        "parts": writer.parts, "tensors": writer.tensors,
    }
    doc = _plain(doc, "the checkpoint document")
    facts = {"index": index, "first": first, "where": where, "stored": len(stored)}
    return doc, writer, facts


def _body(run: Any, doc: dict[str, Any], writer: _Writer, facts: dict[str, Any],
          document: dict[str, Any]) -> dict[str, Any]:
    from saturn_pub.values import digest

    from mdb.core.custody import custody
    from mdb.core.kinds import procedure
    from mdb.core.records import num

    cut_mark = doc["marks"][-1]
    root_mark = doc["marks"][0]
    shas = sorted({p["sha256"] for p in doc["parts"]})
    recipe = doc["recipe"]
    adapter = run.adapter
    body = {
        "schema": SCHEMA, "name": doc["name"], "where": doc["where"],
        "port": doc["cursor"]["port"],
        "cut": {"fingerprint": cut_mark["fingerprint"], "transitions": cut_mark["transitions"],
                "row": cut_mark["row"], "label": cut_mark["label"]},
        "root": {"fingerprint": root_mark["fingerprint"], "transitions": root_mark["transitions"],
                 "row": root_mark["row"], "label": root_mark["label"]},
        "history": doc["history"], "marks": len(doc["marks"]), "stored_cuts": facts["stored"],
        "prompt": {"tokens": list(run.tokens)[:64], "length": len(run.tokens),
                   "sha256": digest(list(run.tokens))},
        "procedure": procedure(run),
        "shape": doc["shape"],
        "source": {"dtype": _dtype_name_of(adapter), "device": str(adapter.device)},
        "lineage": doc["lineage"],
        "recipe": {"sha256": _sha_text(_canonical(recipe)),
                   "path_edits": len(recipe["path_edits"]), "staged": len(recipe["edits"])},
        "edited": bool(recipe["path_edits"] or recipe["edits"]),
        "holds": [{"name": h["name"], "address": h["address"], "dropped": h["dropped"]}
                  for h in doc["holds"]][:64],
        "captured": sorted(doc["captured"])[:64],
        "document": document,
        "tensors": {"count": len(shas),
                    "sha256": _packed(shas) if len(shas) <= _MAX_INLINE_SHAS else None,
                    "parts_sha256": _sha_text(_canonical(shas))},
        "bytes": num(float(writer.bytes), "measured"),
        "custody": custody("mdb-stepped"),
    }
    return body


def _dtype_name_of(adapter: Any) -> str:
    return str(adapter.dtype).removeprefix("torch.")


def _session_id(run: Any) -> str:
    from saturn_pub.values import digest

    from mdb.core.pointers import seal_digest

    return "mdb-run:" + seal_digest({"model": run.adapter.model_identity,
                                     "prompt": digest(run.tokens)})[:32]


def _seal(run: Any, body: dict[str, Any], session_id: str, execution: Any = None
          ) -> dict[str, Any]:
    from mdb.core.kinds import CHECKPOINT_STATE  # noqa: F401 (declares the kind)
    from mdb.core.records import seal_record

    return seal_record(session_id=session_id, kind=KIND,
                       model_identity=run.adapter.model_identity,
                       execution=dict(run.adapter.execution if execution is None else execution),
                       authority="measured", body=body, run=run)


def save(run: Any, target: Any = None, *, project: Any = None, session_id: str | None = None,
         name: str | None = None, at: str | None = None, row: int | None = None,
         history: str = "row", capture: tuple[str, ...] | None = None,
         holds: str = "rearm") -> dict[str, Any]:
    """Write a checkpoint of ``run`` at ``at`` (default: the cursor; ``row=K`` is
    ``at="t<K>.start"``) and return its sealed record. ``target`` a path: a self-contained file;
    ``project``: the project's store and index (``session_id``: the record's session, default
    the project's session bound to this model when there is exactly one). ``history``:
    ``cursor`` (only the cut), ``row`` (default: every boundary of the cut's row before it, so
    the reloaded run rewinds inside that row) or ``root`` (back to the run's root: row starts
    as cuts, the boundaries between them replayed on demand). ``capture``: addresses whose
    values are stored for a replica's ``null_check`` (default: the last prompt row's logits when
    the run computed them). ``holds``: ``rearm`` (default) keeps each active hold with the
    source values it reads from the checkpoint's row on, so the reloaded run continues held,
    bit for bit; ``drop`` records the holds as dropped and the reloaded run continues free."""
    if (target is None) == (project is None):
        raise ValueError("save to a path or to a project, one of them")
    from mdb.core.bounded import put_json

    started = time.perf_counter()
    store = project.store if project is not None else None
    doc, writer, facts = build(run, store=store, name=name, at=at, row=row, history=history,
                               capture=capture, holds=holds)
    text = _canonical(doc)
    if project is not None:
        session = session_id or _project_session(project, run)
        document = put_json(doc, store=store)
        document = {k: document[k] for k in ("state", "ref", "bytes", "sha256", "tensor")}
        if document["sha256"] != _sha_text(text):
            raise RuntimeError("the stored document differs from its canonical text")
        row_ = project.session(session)
        record = _seal(run, _body(run, doc, writer, facts, document), session,
                       execution=json.loads(row_["execution"]))
        from mdb.core import index

        index.put_record(project, record)
    else:
        document = {"state": "file", "bytes": len(text.encode()), "sha256": _sha_text(text)}
        record = _seal(run, _body(run, doc, writer, facts, document),
                       session_id or _session_id(run))
        torch.save({"schema": FILE_SCHEMA, "record": json.dumps(record), "document": text,
                    "tensors": writer.blobs}, str(target))
    record_seconds = time.perf_counter() - started
    _LAST_WRITE.update(seconds=record_seconds, bytes=writer.bytes, parts=len(writer.parts))
    return record


_LAST_WRITE: dict[str, Any] = {}


def _project_session(project: Any, run: Any) -> str:
    sessions = [s for s in project.sessions()
                if s["model_identity"] == _plain(run.adapter.model_identity, "identity")
                and _canonical(s["execution"]) == _canonical(_plain(dict(run.adapter.execution),
                                                                     "execution"))]
    if len(sessions) != 1:
        raise ValueError(f"the project has {len(sessions)} sessions bound to this model and "
                         "execution; name one (session_id=)")
    return sessions[0]["session_id"]


# --------------------------------------------------------------------------- read

class Checkpoint:
    """A checkpoint read back (``load``): its record, document and parts, checked; tensors are
    read and verified when first needed (``run``), and kept, so many runs share them."""

    def __init__(self, record: dict[str, Any], doc: dict[str, Any], *, path: str | None = None,
                 blobs: dict[str, torch.Tensor] | None = None, store: Any = None):
        self.record, self.doc, self.path = record, doc, path
        self._blobs, self._store = blobs, store
        self._parts: dict[int, torch.Tensor] = {}
        self._blocks: dict[tuple[int, str], Any] = {}
        self._cuts: dict[tuple[str, int], Any] = {}
        self.prompt = [int(t) for t in doc["prompt"]]
        self.split = int(doc["marks"][-1]["row"])
        self.timings: dict[str, float] = {}

    # identity ----------------------------------------------------------------------------

    @property
    def id(self) -> str:
        return self.record["id"]

    @property
    def where(self) -> str:
        return self.doc["where"]

    @property
    def fingerprint(self) -> str:
        return self.doc["marks"][-1]["fingerprint"]

    @property
    def recipe(self) -> dict[str, Any]:
        return dict(self.doc["recipe"])

    @property
    def lineage(self) -> dict[str, Any]:
        return dict(self.doc["lineage"])

    def describe(self) -> dict[str, Any]:
        body = self.record["body"]
        return {"record": self.id, "path": self.path, "where": self.where,
                "cut": self.fingerprint, "history": self.doc["history"],
                "marks": len(self.doc["marks"]), "parts": len(self.doc["parts"]),
                "bytes": body["bytes"]["value"], "lineage": self.lineage}

    # tensors -----------------------------------------------------------------------------

    def _part(self, index: int) -> torch.Tensor:
        found = self._parts.get(index)
        if found is not None:
            return found
        pointer = self.doc["parts"][index]
        if self._store is not None:
            from mdb.core.pointers import Unresolved, resolve_tensor, tensor_pointer

            try:
                value = resolve_tensor(tensor_pointer(pointer["ref"], sha256=pointer["sha256"],
                                                      shape=pointer["shape"],
                                                      dtype=pointer["dtype"]),
                                       store=self._store, framework="torch")
            except Unresolved as error:
                raise ValueError(f"checkpoint part {pointer['sha256'][:12]}: {error}") from error
        else:
            value = self._blobs[part_key(pointer["sha256"], pointer["dtype"], pointer["shape"])]
            if (tensor_digest(value) != pointer["sha256"] or list(value.shape) != pointer["shape"]
                    or _dtype_name(value) != pointer["dtype"]):
                raise ValueError(f"checkpoint tensor {pointer['sha256'][:12]} differs from its "
                                 "sha256")
        self._parts[index] = value
        return value

    def blocks(self, index: int, device: Any = "cpu") -> Any:
        """Tensor ``index`` of the document as ``saturn_pub.core.Blocks`` of its parts (on
        ``device``; cached, so every cut that holds the tensor shares one object)."""
        from saturn_pub.core import Blocks

        key = (index, str(device))
        found = self._blocks.get(key)
        if found is not None:
            return found
        desc = self.doc["tensors"][index]
        parts = [self._part(i) for i in desc["parts"]]
        if str(device) != "cpu":
            parts = [p.to(device) for p in parts]
        dtype = getattr(torch, desc["dtype"])
        if any(p.dtype != dtype for p in parts):
            raise ValueError("checkpoint tensor parts differ from its dtype")
        if desc["axis"] is None and len(parts) != 1:
            raise ValueError("a checkpoint tensor without an axis has one part")
        found = self._blocks[key] = Blocks(tuple(parts), desc["axis"], tuple(desc["shape"]),
                                           tuple(desc["stride"]), dtype)
        return found

    def tensor(self, index: int) -> torch.Tensor:
        """Tensor ``index`` of the document, rebuilt from its parts (CPU)."""
        try:
            return self.blocks(index).tensor()
        except ValueError as error:
            raise ValueError(f"checkpoint tensor {index}: {error}") from error

    def _decode(self, value: Any, device: Any, *, blocks: bool = False) -> Any:
        kind = value["kind"]
        if kind == "tensor":
            if blocks:
                return self.blocks(value["t"], device)
            t = self.tensor(value["t"])
            return t if str(t.device) == str(device) else t.to(device)
        if kind == "mapping":
            return {k: self._decode(v, device, blocks=blocks) for k, v in value["items"].items()}
        if kind == "list":
            return [self._decode(v, device, blocks=blocks) for v in value["items"]]
        if kind == "tuple":
            return tuple(self._decode(v, device, blocks=blocks) for v in value["items"])
        if kind == "scalar":
            return value["value"]
        raise ValueError(f"unknown checkpoint value kind {kind!r}")

    # binding -----------------------------------------------------------------------------

    def binding(self, adapter: Any) -> dict[str, Any]:
        """How ``adapter`` relates to the source: same identity, same execution (exact), dtype."""
        check_shape(self.doc["shape"], adapter)
        same_identity = _plain(adapter.model_identity, "identity") == self.doc["model_identity"]
        same_execution = _canonical(_plain(dict(adapter.execution), "execution")) == \
            _canonical(self.doc["execution"])
        return {"same_identity": same_identity, "same_execution": same_execution,
                "exact": same_identity and same_execution,
                "source_dtype": self.doc["execution"].get("dtype", "").removeprefix("torch."),
                "dtype": _dtype_name_of(adapter)}

    def _refuse(self, adapter: Any, replica: bool) -> dict[str, Any]:
        bound = self.binding(adapter)
        if bound["exact"] or replica:
            return bound
        if not bound["same_identity"]:
            raise ValueError("checkpoint of another model (model identity differs); load it as "
                             "a labelled replica with replica=True")
        mine = _flatten(_plain(dict(adapter.execution), "execution"))
        theirs = _flatten(self.doc["execution"])
        differ = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        raise ValueError("checkpoint of another execution (" + "; ".join(
            f"{k}: checkpoint {theirs.get(k, 'missing')!r}, here {mine.get(k, 'missing')!r}"
            for k in differ[:6]) + "); load it as a labelled replica with replica=True")

    def cut(self, adapter: Any, index: int) -> Any:
        """The ``StateCut`` of kept mark ``index`` (exact binding), fingerprint verified."""
        from saturn_pub.core import StateCut

        key = (str(adapter.device), index)
        found = self._cuts.get(key)
        if found is not None:
            return found
        mark = self.doc["marks"][index]
        stored = mark["cut"]
        if stored is None:
            raise ValueError(f"mark {index} of the checkpoint keeps no cut")
        payload = self._decode(stored["payload"], adapter.device, blocks=True)
        try:
            cut = StateCut.from_blocks(adapter.model_identity, adapter.execution,
                                       stored["boundary"], stored["parent"], payload,
                                       execution_point=stored["execution_point"],
                                       surface=stored["surface"])
        except ValueError as error:
            raise ValueError(f"checkpoint state at mark {index}: {error}") from error
        if cut.fingerprint != mark["fingerprint"]:
            raise ValueError(f"checkpoint state at mark {index} differs from its fingerprint "
                             f"{mark['fingerprint'][:12]} (changed tensors or metadata)")
        self._cuts[key] = cut
        return cut

    def _rearm(self, run: Any) -> None:
        """Re-arm the holds the source had active (``save(holds="rearm")``), each reading the
        source values the checkpoint kept (``holds.Snapshot``)."""
        from .holds import Hold, Snapshot

        for h in self.doc["holds"]:
            if h["dropped"]:
                continue
            snap = h["snapshot"]
            values = {a: self.tensor(i) for a, i in snap["values"].items()}
            source = Snapshot(snap["source"], snap["where"], snap["cut"], values, snap["tokens"])
            run._holds.append(Hold(h["name"], h["port"], tuple(h["layers"]),
                                   None if h["rows"] is None else tuple(h["rows"]), source,
                                   h["address"], cells=[dict(c) for c in h["cells"]]))
        run._hold_key = None

    def _held(self) -> dict[str, torch.Tensor]:
        out = {}
        for sha, index in self.doc["edit_tensors"].items():
            value = self.tensor(index)
            if tensor_digest(value) != sha:
                raise ValueError(f"checkpoint edit tensor {sha[:12]} differs from its sha256")
            out[sha] = value
        return out

    # runs --------------------------------------------------------------------------------

    def run(self, adapter: Any, *, lookahead: str | None = None, name: str | None = None,
            replica: bool = False, check: bool = True, verify: bool | str = False,
            root: Any = None) -> Any:
        """A ``Run`` of ``adapter`` at the checkpoint's cursor (module docstring). ``replica``:
        allow another dtype, device or weights (a new lineage at the cursor). ``check``: the
        replica null check (``run.core_check``; values were captured), else the result says
        there was no null check. The run is labelled ``from checkpoint <id>, path unverified``
        (``run.from_checkpoint``) until its path is verified. ``verify=True``: replay the
        recipe from the lineage root (``root`` for a foreign one) first and raise
        ``CheckpointUnverified`` unless it reaches the checkpoint's fingerprint (the run is then
        ``path verified``); ``verify="report"``: record the result as ``run.core["verified"]``
        and never raise."""
        if verify not in (False, True, "report"):
            raise ValueError(f"verify is False, True or 'report', not {verify!r}")
        if verify is True:
            result = self.verify(adapter, root=root)
        started = time.perf_counter()
        bound = self._refuse(adapter, replica)
        lookahead = self.doc["lookahead"] if lookahead is None else lookahead
        if bound["exact"]:
            run = self._exact(adapter, lookahead)
        else:
            run = self._replica(adapter, lookahead)
        seconds = time.perf_counter() - started
        run.name = name or f"core@{self.id[:12]}"
        run.core = {"record": self.id, "path": self.path, "split": self.split,
                    "where": self.where, "cut": self.fingerprint,
                    "history": self.doc["history"], "exact": bound["exact"],
                    "replica": not bound["exact"], "lineage": dict(run.lineage),
                    "source_dtype": bound["source_dtype"], "dtype": bound["dtype"],
                    "same_identity": bound["same_identity"],
                    "holds": [h["name"] for h in self.doc["holds"] if not h["dropped"]],
                    "holds_dropped": [h["name"] for h in self.doc["holds"] if h["dropped"]],
                    "load_seconds": seconds}
        run.from_checkpoint = {"checkpoint": self.id, "path": "unverified"}
        if not bound["exact"]:
            run.from_checkpoint["replica"] = True
        run.core_check = None
        if check and self.doc["captured"]:
            run.core_check = self.null_check(adapter, lookahead=lookahead, replica=replica)
        elif check and not bound["exact"]:
            run.core_check = {"state": "unresolved", "ok": None,
                              "reason": "no null check: the checkpoint captured no value to "
                                        "compare the replica with (save it with capture=)"}
        if not bound["exact"]:
            ok = (run.core_check or {}).get("ok")
            run.from_checkpoint["null_check"] = None if ok is None else bool(ok)
        if verify is True:
            run.core["verified"] = result
        elif verify == "report":
            run.core["verified"] = self.verify(adapter, root=root, report=True)
        if run.core.get("verified", {}).get("ok"):
            run.from_checkpoint["path"] = "verified"
        return run

    def _exact(self, adapter: Any, lookahead: str) -> Any:
        from saturn_pub.core import Session

        from .run import Run

        marks = self.doc["marks"]
        for i in range(1, len(marks)):
            # the kept marks are consecutive trail marks: each cut chains to the one before
            stored = marks[i]["cut"]
            if stored is not None and stored["parent"] != marks[i - 1]["fingerprint"]:
                raise ValueError(f"checkpoint mark {i} ({marks[i]['label']}) has parent "
                                 f"{str(stored['parent'])[:12]}, not the previous mark "
                                 f"{marks[i - 1]['fingerprint'][:12]} (a broken chain)")
        cuts = {i: self.cut(adapter, i) for i, m in enumerate(marks) if m["cut"] is not None}
        # the cuts hold the parts (block form) or the tensors themselves: the concatenated
        # copies are not needed any more (rebuilt from the parts if ever asked for again)
        self._blocks.clear()
        run = Run(adapter, self.prompt, root=cuts[0], root_transitions=marks[0]["transitions"],
                  prefill=self.doc["prefill"], lookahead=lookahead)
        if run._trail[0].fingerprint != marks[0]["fingerprint"]:
            raise ValueError("the rehydrated root differs from the checkpoint's fingerprint")
        run.hold(self._held())
        self._rearm(run)
        run._ancestry = [self._mark(run, m, None, ancestor=True) for m in self.doc["ancestry"]]
        run.lineage = self._lineage()
        run._origin = self
        trail = [run._trail[0]]
        trail += [self._mark(run, m, cuts.get(i)) for i, m in enumerate(marks[1:], 1)]
        self._fill(run, trail[0], marks[0])
        run._trail = trail
        if len(marks) > 1:
            run.session = Session.from_cut(adapter, cuts[len(marks) - 1])
        self._cursor(run)
        return run

    @staticmethod
    def _fill(run: Any, mark: Any, m: dict[str, Any]) -> Any:
        mark.edits = tuple(run._edit_from(e) for e in m["edits"])
        mark.cells = tuple(run._edit_from(c) for c in m["cells"])
        mark.fired = tuple(dict(x) for x in m["fired"])
        mark.cell_fired = tuple(dict(x) for x in m["cell_fired"])
        mark.supplied = m["supplied"]
        return mark

    def _mark(self, run: Any, m: dict[str, Any], cut: Any, *, ancestor: bool = False) -> Any:
        from .run import _Mark

        mark = _Mark(m["fingerprint"], cut, m["row"], m["label"], m["phase"], m["layer"],
                     m["transitions"], ancestor=ancestor)
        return self._fill(run, mark, m)

    def _cursor(self, run: Any) -> None:
        from .edits import CellEdit
        from .ports import PortEdit

        cursor = self.doc["cursor"]
        staged = [run._edit_from(m) for m in [*cursor["edits"], *cursor["cells"]]]
        run._edits = [e for e in staged if isinstance(e, PortEdit)]
        run._cells = [e for e in staged if isinstance(e, CellEdit)]
        run._cache, run._port = None, None
        if cursor["port"] is not None:
            port = run.program.spec(cursor["port"]).name
            run._ensure_cache(port)
            run._port = port
        if run.where != self.where:
            raise RuntimeError(f"the rehydrated cursor is at {run.where}, not {self.where}")

    def _lineage(self) -> dict[str, Any]:
        lineage = dict(self.doc["lineage"])
        lineage.setdefault("checkpoints", [])
        lineage["checkpoints"] = [*lineage["checkpoints"], self.id][-8:]
        return lineage

    def _replica(self, adapter: Any, lookahead: str) -> Any:
        from saturn_pub.core import Session

        from .run import Run

        marks = self.doc["marks"]
        stored = marks[-1]["cut"]
        state = self._decode(stored["payload"], "cpu")
        state.pop("__saturn_external__", None)
        device, dtype = adapter.device, adapter.dtype

        def move(t: Any) -> Any:
            if isinstance(t, Mapping):
                return {k: move(v) for k, v in t.items()}
            if isinstance(t, (list, tuple)):
                return type(t)(move(x) for x in t)
            if not isinstance(t, torch.Tensor):
                return t
            if t.is_floating_point():
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)

        state = move(state)
        root = Session(adapter, state).capture(retain=False)
        run = Run(adapter, self.prompt, root=root, root_transitions=marks[-1]["transitions"],
                  prefill=self.doc["prefill"], lookahead=lookahead)
        run.hold(self._held())
        self._rearm(run)
        run._ancestry = [self._mark(run, m, None, ancestor=True)
                         for m in [*self.doc["ancestry"], *marks[:-1]]
                         if m["edits"] or m["cells"] or m["supplied"] is not None]
        run.lineage = {"root": "checkpoint", "checkpoint": self.id, "replica": True,
                       "cut": root.fingerprint, "source_cut": self.fingerprint,
                       "checkpoints": [self.id]}
        run._origin = self
        self._cursor(run)
        return run

    def null_check(self, adapter: Any, *, lookahead: str = "none",
                   replica: bool = True) -> dict[str, Any]:
        """Run a fresh run of this checkpoint to every captured address and compare (measured):
        per address ``max_abs_diff`` (float64) and, for logits, ``argmax_equal``; ``ok`` when
        every argmax agrees (and every difference is 0 on the same dtype and identity)."""
        if not self.doc["captured"]:
            return {"state": "unresolved", "reason": "the checkpoint captured no value"}
        bound = self._refuse(adapter, replica)
        twin = self._exact(adapter, lookahead) if bound["exact"] else self._replica(adapter,
                                                                                  lookahead)
        out: dict[str, Any] = {"checks": {}, "authority": "measured"}
        same = bound["exact"]
        ok = True
        for address in sorted(self.doc["captured"], key=twin.parse):
            if twin.parse(address) > twin._key():
                twin.until(address)
            got = twin.read(address).detach().to("cpu")
            want = self.tensor(self.doc["captured"][address])
            diff = float((got.to(torch.float64) - want.to(torch.float64)).abs().max())
            entry: dict[str, Any] = {"max_abs_diff": diff}
            if address.endswith("logits"):
                entry["argmax_equal"] = int(got.reshape(-1).argmax()) == int(
                    want.reshape(-1).argmax())
                entry["argmax"] = [int(want.reshape(-1).argmax()), int(got.reshape(-1).argmax())]
                ok = ok and entry["argmax_equal"]
            if same:
                ok = ok and diff == 0.0
            out["checks"][address] = entry
        out.update(ok=ok, same_dtype_and_identity=same,
                   rule="argmax equal at every captured logits" + (
                       " and bit-identical values (same dtype and identity)" if same else ""))
        return out

    def verify(self, adapter: Any, *, root: Any = None, report: bool = False) -> dict[str, Any]:
        """Verify by recompute (measured): replay the recipe from the lineage root (a run of the
        prompt; ``root``, a ``Run`` or ``Checkpoint``, for a foreign lineage) with the stored
        edit tensors and compare the cut it reaches with the checkpoint's fingerprint. Raises
        ``CheckpointUnverified`` unless it does (also when it cannot be checked: another model
        or execution, or a foreign root and no ``root=``); ``report``: return the result
        (``ok``, ``state``) instead of raising."""
        result = self._verify(adapter, root)
        if not result["ok"] and not report:
            raise CheckpointUnverified(result)
        return result

    def _verify(self, adapter: Any, root: Any) -> dict[str, Any]:
        from .run import ReplayMismatch, Run

        started = time.perf_counter()
        lineage = self.doc["lineage"]
        if self.binding(adapter)["exact"] is False:
            return {"ok": False, "state": "unresolved", "checkpoint": self.id,
                    "reason": "verify needs the source's model and execution"}
        if lineage.get("root") != "prompt" and root is None:
            return {"ok": False, "state": "unresolved", "checkpoint": self.id,
                    "reason": "the lineage root is a foreign checkpoint "
                              f"({lineage.get('checkpoint')}); pass root= to verify from it"}
        recipe = self.recipe
        if root is None:
            replay = Run(adapter, self.prompt, prefill=recipe["prefill"],
                         lookahead=recipe["lookahead"])
        elif isinstance(root, Checkpoint):
            replay = root.run(adapter, check=False)
        elif isinstance(root, (str, Path)):
            replay = load(root).run(adapter, check=False)
        else:
            replay = root.fresh()
        replay.hold(self._held())
        try:
            replay.restore(recipe)
            reached = replay._trail[-1].fingerprint
        except ReplayMismatch as error:
            reached = error.reached
        ok = reached == self.fingerprint
        return {"ok": ok, "state": "resolved", "checkpoint": self.id, "expected": self.fingerprint, "reached": reached,
                "authority": "measured", "seconds": time.perf_counter() - started,
                "native_calls": dict(replay.calls),
                "rule": "the recipe replayed from the lineage root reaches the checkpoint's cut "
                        "fingerprint (state content, execution contract and ancestry)"}


class CheckpointUnverified(ValueError):
    """A checkpoint whose path did not verify by recompute (``result`` says why)."""

    def __init__(self, result: dict[str, Any]):
        self.result = result
        if result.get("state") == "resolved":
            text = (f"checkpoint {str(result.get('checkpoint'))[:16]} did not verify: the recipe "
                    f"replayed from its lineage root reaches {str(result['reached'])[:12]}, not "
                    f"{str(result['expected'])[:12]}")
        else:
            text = (f"checkpoint {str(result.get('checkpoint'))[:16]} cannot be verified: "
                    f"{result['reason']}")
        super().__init__(text + " (verify='report' returns the result instead)")


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): value}


def _verify_document(record: dict[str, Any], text: str) -> dict[str, Any]:
    body = record["body"]
    if body.get("schema") != SCHEMA:
        raise ValueError(f"not an {SCHEMA} checkpoint record")
    if _sha_text(text) != body["document"]["sha256"]:
        raise ValueError("checkpoint document differs from its sealed sha256")
    doc = json.loads(text)
    if doc.get("schema") != SCHEMA:
        raise ValueError("checkpoint document has another schema")
    shas = sorted({p["sha256"] for p in doc["parts"]})
    listed = body["tensors"]["sha256"]
    if (listed is not None and _unpacked(listed) != shas) or body["tensors"]["count"] != len(
            shas) or body["tensors"]["parts_sha256"] != _sha_text(_canonical(shas)):
        raise ValueError("checkpoint parts differ from the sha256s in its record")
    from saturn_pub.values import digest

    if digest(doc["prompt"]) != body["prompt"]["sha256"] or len(doc["prompt"]) != \
            body["prompt"]["length"]:
        raise ValueError("checkpoint prompt differs from its sealed prompt sha256")
    if doc["marks"][-1]["fingerprint"] != body["cut"]["fingerprint"] or \
            doc["marks"][0]["fingerprint"] != body["root"]["fingerprint"]:
        raise ValueError("checkpoint marks differ from the fingerprints in its record")
    if _sha_text(_canonical(doc["recipe"])) != body["recipe"]["sha256"]:
        raise ValueError("checkpoint recipe differs from its sealed sha256")
    if _plain(record["model_identity"], "identity") != doc["model_identity"]:
        raise ValueError("checkpoint document belongs to another model than its record")
    return doc


def load_file(path: str | Path) -> Any:
    """A checkpoint file (this schema, or a legacy ``mdb-core-file-v1`` core file)."""
    from mdb.core.kinds import CHECKPOINT_STATE  # noqa: F401 (declares the kind)
    from mdb.core.records import verify_record

    data = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not an mdb checkpoint")
    from . import core_file

    if data.get("schema") == core_file.SCHEMA:
        return core_file.load_v1(data, path)
    if data.get("schema") != FILE_SCHEMA:
        raise ValueError(f"{path} is not an mdb checkpoint")
    if set(data) != _FILE_KEYS:
        raise ValueError("checkpoint file holds other entries than its schema")
    record = verify_record(json.loads(data["record"]))
    doc = _verify_document(record, data["document"])
    keys = {part_key(p["sha256"], p["dtype"], p["shape"]) for p in doc["parts"]}
    if set(data["tensors"]) != keys:
        raise ValueError("checkpoint file tensors differ from its document")
    checkpoint = Checkpoint(record, doc, path=str(path), blobs=data["tensors"])
    for index in range(len(doc["parts"])):
        checkpoint._part(index)            # every part's sha256, before anything is used
    return checkpoint


def find_record(project: Any, ref: str) -> dict[str, Any]:
    """The ``checkpoint-state`` record ``ref`` (an id or a unique prefix of >= 8 hex digits) of
    any session of ``project``."""
    from mdb.core import index
    from mdb.core.kinds import CHECKPOINT_STATE  # noqa: F401 (declares the kind)

    ref = str(ref).strip().lower()
    if len(ref) < 8 or any(c not in "0123456789abcdef" for c in ref):
        raise ValueError(f"{ref!r} is not a record id (8 or more hex digits)")
    found = []
    for session in project.sessions():
        for record in index.list_records(project, session["session_id"], kind=KIND):
            if record.get("verified") is False:
                continue
            if record["id"].startswith(ref):
                found.append(record)
    if not found:
        raise ValueError(f"no checkpoint record {ref} in project {project.root}")
    if len({r["id"] for r in found}) > 1:
        raise ValueError(f"checkpoint record prefix {ref} is ambiguous; give more digits")
    return found[0]


def load_record(project: Any, record: dict[str, Any]) -> Checkpoint:
    from mdb.core.bounded import load_json

    pointer = record["body"]["document"]
    doc = load_json(pointer, store=project.store)
    text = _canonical(doc)
    doc = _verify_document(record, text)
    return Checkpoint(record, doc, store=project.store)


def find_named(project: Any, name: str | None, *, session_id: str | None = None,
               cut: str | None = None) -> dict[str, Any] | None:
    """The newest ``checkpoint-state`` record named ``name`` (None: any name; optionally at cut
    ``cut``), in ``session_id`` first, else in any session."""
    from mdb.core import index
    from mdb.core.kinds import CHECKPOINT_STATE  # noqa: F401 (declares the kind)

    sessions = [s["session_id"] for s in project.sessions()]
    if session_id in sessions:
        sessions = [session_id] + [s for s in sessions if s != session_id]
    for sid in sessions:
        named = [r for r in index.list_records(project, sid, kind=KIND)
                 if r.get("verified") is not False
                 and (name is None or r["body"].get("name") == name)
                 and (cut is None or r["body"]["cut"]["fingerprint"] == cut)]
        if named:
            return named[-1]
    return None


def load(ref: Any, *, project: Any = None) -> Any:
    """A checkpoint from a path, a record id (with ``project``) or a loaded object."""
    from . import core_file

    if isinstance(ref, (Checkpoint, core_file.CoreFile)):
        return ref
    if isinstance(ref, Mapping) and "id" in ref and ref.get("kind") == KIND:
        if project is None:
            raise ValueError("a checkpoint record needs the project that holds it")
        return load_record(project, find_record(project, ref["id"]))
    if isinstance(ref, Mapping) and isinstance(ref.get("durable"), Mapping):
        durable = ref["durable"]
        ref = durable.get("path") or durable["record"]
    is_file = isinstance(ref, Path) or (isinstance(ref, str) and Path(ref).expanduser().is_file())
    record = None
    if isinstance(ref, str) and project is not None:
        try:
            record = find_record(project, ref)
        except ValueError:
            if not is_file:
                raise
    if is_file and record is not None:
        raise ValueError(f"{ref!r} names both a file and checkpoint record {record['id'][:16]} "
                         "of the project: pass Path(...) for the file, or the record itself (the dict "
                         "states.find_record returns) for the record")
    if is_file:
        return load_file(Path(ref).expanduser())
    if record is not None:
        return load_record(project, record)
    raise ValueError(f"no checkpoint at {ref!r}: give a file path, or a record id with a "
                     "project attached")


__all__ = ["Checkpoint", "CheckpointUnverified", "FILE_SCHEMA", "HISTORIES", "KIND", "SCHEMA", "build", "find_named",
           "find_record", "load", "load_file", "load_record", "save", "shape_of"]
