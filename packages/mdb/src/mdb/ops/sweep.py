"""``sweep``: the position x layer activation-patching sweep, measured natively (M35).

``sweep PORT [rows=R0:R1] [layers=L0:L1] = ORIGIN --consumer C`` patches ``t<r>.L<l>.PORT``
with the value of the run ``ORIGIN`` at the same address (``set t<r>.L<l>.PORT =
ORIGIN:t<r>.L<l>.PORT``, the cross-run port edit of ``set``), runs natively to the consumer,
reads it, and goes on to the next cell; the session is restored once at the end and checked
unchanged. Every number is ``measured``: the base consumer value, each patched value, and the
delta ``patched - base`` (arithmetic on two measured values). No prediction is made.

Rows must be stepped rows of both runs (``start ... prefill=K``: a prefilled row has no ports).
The cells are visited from the latest layer boundary to the earliest: each one rewinds to the
boundary before its layer (``t<r>.emb`` or ``t<r>.L<l-1>.h_out``) with one restore of a trail
mark the earlier cells never touched (the trail below an edit is that of the unedited run), so
no cell replays the prompt; the edit is staged there with no probe (M54). At the end the run
comes back to the trail it held before the sweep (``Run.excursion``/``come_back``: a restore of
the held cut, no replay). Cost model (derived before the sweep runs; the counts in the result
are measured): a cell ``(r, l)`` with the consumer in row ``c`` costs ``L - l`` native layer
calls in row ``r`` plus ``L`` per row ``r+1..c`` (plus the post-layer transitions of each row)
and at most one full-layer probe of the origin run to read its value (cached per layer); the
base reading costs the forward from the cursor to the consumer. The rows between a cell and
the consumer are the cost that remains: each is stepped at batch 1 under the run's split, and
batching them would change the procedure (M26). The plan is printed before the sweep runs
(``Context.notice``) and a plan over ``max_calls`` (default ``MAX_LAYER_CALLS``) is refused.

The result is bounded JSON: the rows x layers table of deltas inline (``delta`` a
``{value: [[...]], authority: measured}`` claim, indexed [layer][row]; ``tokens`` per row),
or, when it exceeds ``INLINE_BYTES`` or the record's item bound (``MAX_INLINE_ITEMS`` items
across nested lists, so about 50 cells), a sha256-checked JSON blob
(``mdb.core.bounded.put_json``: in the project store, else held in memory on the run) with the
top ``top`` cells by ``|delta|`` and the heat text (``heat``) inline. It is also sealed as a
``sweep`` record (``core.kinds.SWEEP``: the table always by blob pointer, the procedure of both
runs, no wall clock), stored when a writable project is attached (``stored``).
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
SCHEMA = "mdb-sweep-v1"
HEAT_BYTES = 4000            # the heat text of a blob table (a record text holds 4 KiB)
# The default cap on planned decoder-layer calls (``max_calls=`` overrides it). About 5 minutes
# at the 5.2 ms per layer commit measured on Qwen2.5-1.5B bf16 (RTX 4080, README); the full
# 22 rows x 28 layers of that model's 22-token prompt plan about 190k.
MAX_LAYER_CALLS = 50_000
_SPAN = re.compile(r"(?P<a>\d+)(?::(?P<b>\d+))?")
_PORT = re.compile(r"(?P<port>[A-Za-z_]\w*)(?P<selector>\[[^\]]*\])?")
USAGE = ("sweep PORT [rows=R0:R1] [layers=L0:L1] = ORIGIN --consumer C [top=K] [max_calls=N]  "
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


def _layer_calls(run: Any, after: tuple[int, int], upto: tuple[int, int]) -> int:
    """Decoder-layer transitions committed moving forward from stop key ``after`` to ``upto``:
    the ``L<l>.h_out`` stops in ``(after, upto]`` (one native layer call each)."""
    stops = run.stops()
    outs = [i for i, label in enumerate(stops) if label.endswith(".h_out")]
    return sum(1 for row in range(after[0], upto[0] + 1) for i in outs
               if after < (row, i) <= upto)


def _boundary(run: Any, row: int, layer: int) -> str:
    """The boundary stop before layer ``layer`` of row ``row`` (``emb`` or ``L<l-1>.h_out``):
    a trail mark whose pending layer is ``layer``, reached by a restore with no probe."""
    return f"t{row}.emb" if layer == 0 else f"t{row}.L{layer - 1}.h_out"


def _heat(grid: list[list[float]], rows: list[int], layers: list[int],
          tokens: list[dict[str, Any]]) -> dict[str, Any]:
    """The heat table as text, for a table kept as a blob: one line per layer, each cell the
    signed tenths of the largest |delta| (``+9``..``-9``, ``.`` below a twentieth, ``0`` exact);
    the REPL draws it as it draws an inline table."""
    scale = max((abs(v) for line in grid for v in line), default=0.0)

    def code(value: float) -> str:
        if value == 0:
            return "0"
        if scale == 0 or abs(value) / scale < 0.05:
            return "."
        return ("+" if value > 0 else "-") + str(min(9, max(1, round(abs(value) / scale * 10))))

    out: dict[str, Any] = {
        "rows": f"{rows[0]}:{rows[-1] + 1}", "layers": f"{layers[0]}:{layers[-1] + 1}",
        "scale": {"value": scale, "authority": "measured"},
        "cells": "\n".join(" ".join(code(v) for v in line) for line in grid),
        "meaning": "one line per layer, one cell per row: the signed tenths of the largest "
                   "|delta| (9 caps), '.' below a twentieth, '0' exact"}
    if len(out["cells"].encode()) > HEAT_BYTES:
        out.pop("cells")
        out["state"] = "absent"
        out["reason"] = f"the heat text exceeds {HEAT_BYTES} bytes: read the table blob"
    texts = [item.get("text") for item in tokens]
    if len(texts) <= 64 and all(isinstance(t, str) for t in texts):
        out["tokens"] = texts
    return out


def sweep(ctx: Any, port: str, rows: str | None, layers: str | None, origin: str,
          consumer: str, top: int, max_calls: int = MAX_LAYER_CALLS) -> dict[str, Any]:
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

    # the plan (cost model, derived before any native call): the forward to the consumer for
    # the base reading, then per cell the layers from its boundary to the consumer
    here = run.parse(run.where)
    plan_base = _layer_calls(run, here, stop_key) if stop_key > here else 0
    per_cell = {(r, l): _layer_calls(run, run.parse(_boundary(run, r, l)), stop_key)
                for r, l in cells}
    planned = plan_base + sum(per_cell.values())
    plan_text = (f"sweep plan: {len(cells)} cells ({len(row_list)} rows x {len(layer_list)} "
                 f"layers), {planned} layer calls planned (base {plan_base}, cells "
                 f"{planned - plan_base}; cap {max_calls})")
    if planned > max_calls:
        raise Refused(f"{plan_text}: over the cap. Each cell runs natively from its layer to the "
                      f"consumer, so rows far before the consumer cost {run.adapter.layers} layer "
                      "calls per row in between; narrow rows= or layers=, or pass "
                      f"max_calls={planned} to run it anyway")
    notice = getattr(ctx, "notice", None)
    if notice is not None:
        notice(plan_text)

    before = session_digest(ctx)
    checkpoint = run.checkpoint(pin=False)   # its edits are all before the first cell: kept
    try:
        _guard(run, [checkpoint])
    except Refusal as error:
        raise Refused(str(error)) from None
    calls0 = _calls(run, donor) if donor is not None else _calls(run)
    started = time.perf_counter()
    # latest boundary first: each cell rewinds to a trail mark the earlier cells never touched
    order = sorted(cells, key=lambda cell: run.parse(_boundary(run, *cell)), reverse=True)
    deltas: dict[tuple[int, int], float] = {}
    base_value: float | None = None
    dtype = None
    held = run.excursion()      # the unedited trail, to come back to without a replay
    try:
        if stop_key > run.parse(run.where):
            run.until(stop)
        native = run.read(stop)
        dtype = str(native.dtype).removeprefix("torch.")
        base_value = float(consumer_value(chosen, native))
        for r, l in order:
            boundary = _boundary(run, r, l)
            if run.parse(run.where) > run.parse(boundary):
                run.rewind(boundary)          # a restore of a held mark: no native call
            elif run.where != boundary:
                run.until(boundary)
            address = f"t{r}.L{l}.{spec.name}{selector}"
            edit = build_edit(ctx, address, f"{branch}:{address}", "replace")
            run.set(edit.address, edit.value, "replace", mask=edit.mask,
                    selection=edit.selection, origin=edit.origin)
            run.until(stop)
            deltas[r, l] = float(consumer_value(chosen, run.read(stop))) - base_value
    finally:
        try:
            run.come_back(held)
        except BaseException:
            restore_exactly(run, checkpoint)
            raise
        del held
    seconds = time.perf_counter() - started
    after = session_digest(ctx)
    if after != before:
        raise RuntimeError("sweep changed the debugger session")
    calls1 = _calls(run, donor) if donor is not None else _calls(run)
    calls = {key: calls1[key] - calls0[key] for key in calls1}

    grid = [[deltas[r, l] for r in row_list] for l in layer_list]   # rows of the table: layers
    tokens = _token_texts(ctx, row_list)
    table = {"layers": layer_list, "rows": row_list,
             "delta": {"value": grid, "authority": "measured"},
             "tokens": tokens,
             "note": "delta[i][j] = consumer patched at t<rows[j]>.L<layers[i]> minus base"}
    ranked = sorted(cells, key=lambda cell: (-abs(deltas[cell]), cell))[:top]
    model = ("per cell (r, l), consumer in row c: (L - l) + L x (c - r) layer calls from the "
             "boundary before layer l (a restore of a held trail mark), plus the post-layer "
             "transitions of rows r..c and at most one origin probe; once: the forward to the "
             "consumer for the base; the return is a restore of the held trail (no call)")
    out: dict[str, Any] = {
        "port": f"{spec.name}{selector}", "origin": branch, "consumer": consumer, "stop": stop,
        "cells": {"count": len(cells)},
        "base": {"value": base_value, "authority": "measured", "dtype": dtype},
        "top": [{"row": r, "layer": l, "delta": {"value": deltas[r, l], "authority": "measured"}}
                for r, l in ranked],
        "prefill": procedure_json(run),
        "cost": {
            "planned": {"layer": {"value": planned, "authority": "derived-exact"},
                        "cap": {"value": max_calls, "authority": "supplied"},
                        "note": "decoder-layer transitions the cost model gives before the sweep "
                                "runs (the cap refuses a larger plan; max_calls= overrides)"},
            "native": {**{key: {"value": n, "authority": "measured"} for key, n in calls.items()},
                       "note": "native calls made by this run and the origin run during the "
                               "sweep (layer: decoder-layer transitions; probe: layer calls up "
                               "to a stop or a cached full-layer read)"},
            "seconds": {"value": seconds, "authority": "measured"},
            "model": model,
        },
        "session": {"unchanged": True, "debugger_before": before, "debugger_after": after},
        "where": run.where,
    }
    if donor is not None:
        out["origin_prefill"] = procedure_json(donor)
    project = project_of(ctx)
    store = project.store if project is not None and not project.read_only else None
    pointer = put_json(table, store=store, run=run)
    try:   # inline when it fits a record part: bytes, and items across nested lists
        out["table"] = check_bounded(table, "sweep table", max_bytes=INLINE_BYTES)
    except RecordError:
        shown = dict(pointer)
        shown.pop("tensor", None)
        out["table_pointer"] = shown
        out["heat"] = _heat(grid, row_list, layer_list, tokens)
    record = _record(ctx, run, donor, out, pointer, row_list, layer_list, model)
    out["record"] = record["id"]
    if store is not None:
        from mdb.core import index

        out["stored"] = index.put_record(project, record)
    return out


def _record(ctx: Any, run: Any, donor: Any, out: dict[str, Any], pointer: dict[str, Any],
            rows: list[int], layers: list[int], model: str) -> dict[str, Any]:
    """The sweep as one ``sweep`` record: no array inline (the table is the blob ``pointer``;
    rows and layers as half-open spans), the procedure of both runs, no wall clock (a record id
    is the same for the same sweep, REPL or server)."""
    from mdb.analysis.forecast import run_session_id
    from mdb.core.kinds import procedure
    from mdb.core.records import num, seal_record

    table: dict[str, Any] = {"state": pointer["state"], "sha256": pointer["sha256"]}
    pointers = []
    if pointer["state"] == "stored":
        table.update(ref=pointer["ref"], tensor_sha256=pointer["tensor"]["sha256"],
                     bytes=num(pointer["bytes"], "derived-exact"))
        pointers.append(pointer["tensor"])
    else:
        table["reason"] = pointer["reason"]
    body: dict[str, Any] = {
        "schema": SCHEMA, "port": out["port"], "origin": out["origin"],
        "consumer": out["consumer"], "stop": out["stop"],
        "rows": f"{rows[0]}:{rows[-1] + 1}", "layers": f"{layers[0]}:{layers[-1] + 1}",
        "cells": out["cells"], "top": out["top"], "table": table,
        "base": {k: out["base"][k] for k in ("value", "authority")},
        "dtype": out["base"]["dtype"],
        "cost": {"planned": out["cost"]["planned"], "native": out["cost"]["native"],
                 "model": model},
        "procedure": procedure(run),
        "session": {"before": out["session"]["debugger_before"],
                    "after": out["session"]["debugger_after"], "unchanged": True},
    }
    if "heat" in out:
        body["heat"] = out["heat"]
    if donor is not None:
        body["origin_procedure"] = procedure(donor)
    project = project_of(ctx)
    session_id = (ctx.debugger.session_id if project is not None and not project.read_only
                  else run_session_id(run))
    try:
        return seal_record(session_id=session_id, kind="sweep",
                           model_identity=run.adapter.model_identity,
                           execution=dict(run.adapter.execution), authority="measured",
                           body=check_bounded(body, "sweep record"), pointers=pointers)
    except RecordError as error:
        raise Refused(f"the sweep result does not fit one record ({error})") from None


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
                 help="cells listed by |delta|"),
             Arg("max_calls", "int", default=MAX_LAYER_CALLS, minimum=1, positional=False,
                 help=f"refuse a sweep whose planned decoder-layer calls exceed this (default "
                      f"{MAX_LAYER_CALLS}; pass the planned count to run a larger one)")),
            "navigate", True, lambda ctx, port, origin, consumer, rows, layers, top, max_calls:
            sweep(ctx, port, rows, layers, origin, consumer, top, max_calls), usage=USAGE))


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
        if equals and key in ("rows", "layers", "top", "max_calls", "origin", "consumer") \
                and value:
            args[key] = (int(value) if key in ("top", "max_calls") and value.isdigit()
                         else value)
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
