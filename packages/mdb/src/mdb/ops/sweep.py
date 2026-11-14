"""``sweep``: the position x layer activation-patching sweep, measured natively (M35).

``sweep PORT [rows=R0:R1] [layers=L0:L1] = ORIGIN --consumer C`` patches ``t<r>.L<l>.PORT``
with the value of the run ``ORIGIN`` at the same address (``set t<r>.L<l>.PORT =
ORIGIN:t<r>.L<l>.PORT``, the cross-run port edit of ``set``), runs natively to the consumer,
reads it, and goes on to the next cell; the session is restored once at the end and checked
unchanged. Every number is ``measured``: the base consumer value, each patched value, and the
delta ``patched - base`` (arithmetic on two measured values). No prediction is made.

Rows must be stepped rows of both runs (``start ... prefill=K``: a prefilled row has no ports).
The cells are visited from the latest layer entry to the earliest: each one rewinds to its
entry ``t<r>.L<l>.h`` with one restore of a trail mark the earlier cells never touched (the
trail below an edit is that of the unedited run), so no cell replays the prompt from row 0;
the final restore replays from the earliest entry only (``Run._replay_base``). Cost model
(asserted, the counts in the result are measured): a cell ``(r, l)`` with the consumer in row
``c`` costs ``L - l`` native layer calls in row ``r`` plus ``L`` per row ``r+1..c`` (plus the
post-layer transitions of each row), one probe at the entry, and at most one full-layer probe
of the origin run to read its value (cached per layer); the final restore costs one replay from
the earliest entry to the cursor.

The result is one bounded JSON record: the rows x layers table of deltas inline (``delta`` a
``{value: [[...]], authority: measured}`` claim, indexed [layer][row]; ``tokens`` per row),
or, when it exceeds ``INLINE_BYTES`` or the record's item bound (``MAX_INLINE_ITEMS`` items
across nested lists, so about 50 cells), a sha256-checked JSON blob
(``mdb.core.bounded.put_json``: in the project store, else held in memory on the run) with the
top ``top`` cells by ``|delta|`` inline.
"""

from __future__ import annotations

import re
import time
from typing import Any

from mdb.core.bounded import put_json
from mdb.core.pointers import RecordError, check_bounded

from .builtin import procedure_json, project_of
from .errors import BadArgs, Refused
from .registry import Arg, Op, register

INLINE_BYTES = 32 * 1024     # a table larger than this goes to a blob (the record bound is 64 KiB)
MAX_CELLS = 4096
_SPAN = re.compile(r"(?P<a>\d+)(?::(?P<b>\d+))?")
_PORT = re.compile(r"(?P<port>[A-Za-z_]\w*)(?P<selector>\[[^\]]*\])?")
USAGE = ("sweep PORT [rows=R0:R1] [layers=L0:L1] = ORIGIN --consumer C [top=K]  "
         "(PORT: h, m, A[h2], ...; ranges half-open; ORIGIN a run name)")


def _span(text: str | None, default: tuple[int, int], what: str, limit: int) -> list[int]:
    """``R`` or ``A:B`` (half-open) -> the indices, inside ``0..limit-1``."""
    if text is None:
        a, b = default
    else:
        match = _SPAN.fullmatch(str(text).strip())
        if match is None:
            raise BadArgs(f"{what} is N or A:B (half-open), not {text!r}")
        a = int(match["a"])
        b = int(match["b"]) if match["b"] is not None else a + 1
    if not 0 <= a < b <= limit:
        raise Refused(f"{what} {a}:{b} is outside 0:{limit} (or empty)")
    return list(range(a, b))


def _origin_name(text: str) -> str:
    """``NAME`` or ``NAME:ADDR`` (as ``@pair:ADDR`` binds in a prompt set) -> ``NAME``."""
    name = text.strip().partition(":")[0].strip()
    if not name or name.startswith("@"):
        raise BadArgs(f"the origin is a run name (info runs), not {text!r}")
    return name


def _refuse_edits(run: Any, first: tuple[int, int]) -> None:
    if run.edits or run.cells:
        raise Refused("edits are staged at the cursor: a sweep steps back to each cell's layer, "
                      "which drops them; commit or abort them first")
    later = [e for e in run.committed_edits() if (int(e["row"]), int(e["layer"])) >= first]
    later += [e for e in run.committed_cells() if (int(e["row"]), int(e["layer"])) >= first]
    if later:
        found = later[0]
        raise Refused(f"the committed edit at t{found['row']}.L{found['layer']} "
                      f"({found['port']}) comes at or after the first swept cell: stepping back "
                      "would drop it from the measured runs")


def _token_texts(ctx: Any, rows: list[int]) -> list[dict[str, Any]]:
    run, tokenizer = ctx.run, ctx.tokenizer
    out = []
    held = [int(t) for t in run.session.read("tokens")[0].tolist()]
    tokens = held if len(held) > len(run.tokens) else list(run.tokens)   # prompt, then sampled
    for row in rows:
        item: dict[str, Any] = {"row": row}
        if row < len(tokens):        # a generated row's own token is known once it is sampled
            item["token"] = tokens[row]
            if tokenizer is not None:
                item["text"] = tokenizer.decode([tokens[row]])
        out.append(item)
    return out


def _calls(*runs: Any) -> dict[str, int]:
    total = {"layer": 0, "transition": 0, "probe": 0}
    for run in runs:
        for key in total:
            total[key] += int(getattr(run, "calls", {}).get(key, 0))
    return total


def sweep(ctx: Any, port: str, rows: str | None, layers: str | None, origin: str,
          consumer: str, top: int) -> dict[str, Any]:
    from mdb.analysis.compare import _guard, consumer_stop, consumer_value, restore_exactly
    from mdb.analysis.forecast import Logit, Margin, Refusal

    from .predict import build_edit, parse_consumer, session_digest

    run = ctx.run
    found = _PORT.fullmatch(port.strip())
    if found is None or found["port"] not in run.program.names:
        raise BadArgs(f"PORT is a port name with an optional selector ({', '.join(run.program.names)}"
                      f"; e.g. h, m, A[h2]), not {port!r}")
    spec = run.program.spec(found["port"])
    if not spec.editable:
        raise Refused(f"port {spec.name} is not editable: {spec.edit}")
    selector = found["selector"] or ""
    branch = _origin_name(origin)
    debugger = ctx.debugger
    donor = getattr(debugger, "runs", {}).get(branch)
    if donor is run:
        raise Refused(f"the origin {branch} is the swept run itself; start the clean run under "
                      "another name (start name=clean ...)")
    try:
        chosen = parse_consumer(ctx, consumer)
        stop = consumer_stop(run, chosen)
    except Refusal as error:
        raise Refused(str(error)) from None
    if not isinstance(chosen, (Margin, Logit)):
        raise Refused(f"a sweep consumer is one number per cell: a margin M[a,b] or a logit "
                      f"logits[i], not the whole stop {stop}")
    stop_key = run.parse(stop)
    last = stop_key[0]
    row_list = _span(rows, (run.first_row, last + 1), "rows", last + 1)
    layer_list = _span(layers, (0, run.adapter.layers), "layers", run.adapter.layers)
    if len(row_list) * len(layer_list) > MAX_CELLS:
        raise Refused(f"{len(row_list)} x {len(layer_list)} cells exceed {MAX_CELLS}; sweep "
                      "narrower rows= or layers=")
    if row_list[0] < run.first_row:
        raise Refused(f"rows {row_list[0]}..: {run.prefilled_reason(row_list[0])}")
    if donor is not None and row_list[0] < donor.first_row:
        raise Refused(f"the origin {branch} prefilled row {row_list[0]} (prefill="
                      f"{donor.first_row}): start it with `start name={branch} prefill=K ...` "
                      f"for a K <= {row_list[0]}")
    cells = [(r, l) for r in row_list for l in layer_list]
    for r, l in cells:
        if run.parse(f"t{r}.L{l}.{spec.name}") >= stop_key:
            raise Refused(f"the consumer {stop} is not downstream of t{r}.L{l}.{spec.name}; "
                          f"sweep rows that end at or before t{last}")
    _refuse_edits(run, (row_list[0], layer_list[0]))

    before = session_digest(ctx)
    checkpoint = run.checkpoint(pin=False)   # its edits are all before the first cell: kept
    try:
        _guard(run, [checkpoint])
    except Refusal as error:
        raise Refused(str(error)) from None
    calls0 = _calls(run, donor) if donor is not None else _calls(run)
    started = time.perf_counter()
    order = sorted(cells, key=lambda cell: run.parse(f"t{cell[0]}.L{cell[1]}.h"), reverse=True)
    deltas: dict[tuple[int, int], float] = {}
    base_value: float | None = None
    dtype = None
    try:
        if stop_key > run.parse(run.where):
            run.until(stop)
        native = run.read(stop)
        dtype = str(native.dtype).removeprefix("torch.")
        base_value = float(consumer_value(chosen, native))
        for r, l in order:
            entry = f"t{r}.L{l}.h"
            if run.parse(run.where) > run.parse(entry):
                run.rewind(entry)
            elif run.where != entry:
                run.until(entry)
            address = f"t{r}.L{l}.{spec.name}{selector}"
            edit = build_edit(ctx, address, f"{branch}:{address}", "replace")
            run.set(edit.address, edit.value, "replace", mask=edit.mask,
                    selection=edit.selection, origin=edit.origin)
            run.until(stop)
            deltas[r, l] = float(consumer_value(chosen, run.read(stop))) - base_value
    finally:
        restore_exactly(run, checkpoint)
    seconds = time.perf_counter() - started
    after = session_digest(ctx)
    if after != before:
        raise RuntimeError("sweep changed the debugger session")
    calls1 = _calls(run, donor) if donor is not None else _calls(run)
    calls = {key: calls1[key] - calls0[key] for key in calls1}

    grid = [[deltas[r, l] for r in row_list] for l in layer_list]   # rows of the table: layers
    table = {"layers": layer_list, "rows": row_list,
             "delta": {"value": grid, "authority": "measured"},
             "tokens": _token_texts(ctx, row_list),
             "note": "delta[i][j] = consumer patched at t<rows[j]>.L<layers[i]> minus base"}
    ranked = sorted(cells, key=lambda cell: (-abs(deltas[cell]), cell))[:top]
    out: dict[str, Any] = {
        "port": f"{spec.name}{selector}", "origin": branch, "consumer": consumer, "stop": stop,
        "cells": {"count": len(cells)},
        "base": {"value": base_value, "authority": "measured", "dtype": dtype},
        "top": [{"row": r, "layer": l, "delta": {"value": deltas[r, l], "authority": "measured"}}
                for r, l in ranked],
        "prefill": procedure_json(run),
        "cost": {
            "native": {**{key: {"value": n, "authority": "measured"} for key, n in calls.items()},
                       "note": "native calls made by this run and the origin run during the "
                               "sweep (layer: decoder-layer transitions; probe: layer calls up "
                               "to a stop or a cached full-layer read)"},
            "seconds": {"value": seconds, "authority": "measured"},
            "model": "per cell (r, l), consumer in row c: (L - l) + L x (c - r) layer calls, "
                     "plus the post-layer transitions of rows r..c, one entry probe and at most "
                     "one origin probe; once: a replay from the earliest entry (restore)",
        },
        "session": {"unchanged": True, "debugger_before": before, "debugger_after": after},
        "where": run.where,
    }
    if donor is not None:
        out["origin_prefill"] = procedure_json(donor)
    try:   # inline when it fits a record part: bytes, and items across nested lists
        out["table"] = check_bounded(table, "sweep table", max_bytes=INLINE_BYTES)
    except RecordError:
        project = project_of(ctx)
        store = project.store if project is not None and not project.read_only else None
        pointer = put_json(table, store=store, run=run)
        pointer.pop("tensor", None)
        out["table_pointer"] = pointer
    return out


register(Op("sweep", "Patch PORT from another run at every (row, layer) of a range, measure the "
            "consumer natively for each, restore: one table of deltas (activation patching)",
            (Arg("port", "str", required=True, help="the port, with an optional selector (h, m, "
                 "A[h2])"),
             Arg("origin", "str", required=True, positional=False,
                 help="the run the value comes from (start name=NAME ...)"),
             Arg("consumer", "str", required=True, positional=False,
                 help="M[a,b], logits[i] or a stop"),
             Arg("rows", "str", positional=False, help="R or R0:R1 (half-open; default the "
                 "run's stepped rows up to the consumer)"),
             Arg("layers", "str", positional=False, help="L or L0:L1 (default all layers)"),
             Arg("top", "int", default=8, minimum=1, maximum=64, positional=False,
                 help="cells listed by |delta|")),
            "navigate", True, lambda ctx, port, origin, consumer, rows, layers, top:
            sweep(ctx, port, rows, layers, origin, consumer, top), usage=USAGE))


def parse_line(text: str) -> dict[str, Any]:
    """``PORT [rows=..] [layers=..] [top=K] = ORIGIN --consumer C`` -> op args."""
    from .lines import _consumer_option, split

    consumer, rest = _consumer_option(text)
    words = [word for word, _, _ in split(rest)]
    args: dict[str, Any] = {}
    if consumer is not None:
        args["consumer"] = consumer
    i = 0
    while i < len(words):
        word = words[i]
        key, equals, value = word.partition("=")
        if word == "=" and i + 1 < len(words) and "origin" not in args:
            args["origin"], i = words[i + 1], i + 2
            continue
        if word.startswith("=") and len(word) > 1 and "origin" not in args:   # `=clean`
            args["origin"], i = word[1:], i + 1
            continue
        if equals and key in ("rows", "layers", "top", "origin", "consumer") and value:
            args[key] = int(value) if key == "top" and value.isdigit() else value
        elif "port" not in args and not equals and "origin" not in args:
            args["port"] = word
        else:
            raise BadArgs(USAGE)
        i += 1
    if "port" not in args or "origin" not in args or "consumer" not in args:
        raise BadArgs(USAGE)
    return args


def _register_parser() -> None:
    from . import lines

    lines.SPECIAL.setdefault("sweep", parse_line)


_register_parser()
