"""The built-in ops over ``mdb.engine.Run``. Import is torch-free; each op imports the engine
when it runs.

Addresses accepted by ops (``ADDR``): a canonical occurrence ``t3.L0.p[h0,src=2]``, the same
without the row (the cursor's row), a bare port of the focus/cursor layer (``p[h0]``), a
global (``emb fn logits sample start``), or a cache cell ``cache.L0.v[kv0,row=2]`` read at the
current clock. Origins (``boiling:...``, ``@cut+edit:...``) and ``~program`` are refused until
branches exist.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from mdb.core import address as addresses
from mdb.core.records import num

from . import graph
from .errors import BadArgs, Refused
from .registry import Arg, op, slot

GLOBALS = (*addresses.DEFAULT_GLOBALS, "start")
_ROW = re.compile(r"t(?P<row>\d+)\.(?P<rest>.+)")
_LAYER_PORT = re.compile(r"L(?P<layer>\d+)\.(?P<port>\w+)")
STATS = ("norm", "min", "max", "mean", "abs_max")
MAX_DISPLAYS = 8
MAX_CHECKPOINTS = 4096
STEP_CAP = 4096          # most units one step/next/reverse-step may take
PAGE = 32                # default page size of every list; 64 is the most one result may hold
GLOBAL_OPERATORS = {
    "start": "tokens of the context so far (before the pending row is embedded)",
    "emb": "emb = embed(token)",
    "fn": "fn = final_norm(h_out of the last layer)",
    "logits": "logits = lm_head(fn)",
    "sample": "sample = argmax(logits) (greedy)",
}


def _bp():
    from mdb.engine import breakpoints

    return breakpoints


def _num(value: Any, authority: str = "measured") -> dict[str, Any]:
    """A labelled number; a NaN/Inf has no honest numeric claim, so it is an unresolved field
    that says which non-finite value it was."""
    if isinstance(value, float) and not math.isfinite(value):
        return _unresolved(f"non-finite value ({'nan' if value != value else str(value)})")
    return num(value, authority)


# Run.read_info says "derived" for a replica recomputed from a supplied value; the record
# vocabulary (mdb.core.pointers.AUTHORITIES) has no bare "derived".
_AUTHORITY = {"measured": "measured", "supplied": "supplied", "derived": "derived-exact",
              "derived-exact": "derived-exact", "derived-local": "derived-local",
              "bound": "bound"}


def _unresolved(reason: str) -> dict[str, str]:
    """A field that was requested but could not be computed (never silently dropped)."""
    return {"state": "unresolved", "reason": reason}


def _page(items: list, offset: int, limit: int) -> tuple[list, dict[str, int]]:
    """One page of a list: the slice and ``total``/``offset``/``limit`` (never a refusal for
    length; an ``offset`` past the end is an empty page)."""
    return items[offset:offset + limit], {"total": len(items), "offset": offset, "limit": limit}


_OFFSET = Arg("offset", "int", default=0, minimum=0, positional=False, help="first item of the page")
_LIMIT = Arg("limit", "int", default=PAGE, minimum=1, maximum=64, positional=False,
             help="items per page (default 32, at most 64)")


def _labelled(key: str, value: Any) -> Any:
    """Plain JSON with every number labelled (measured) unless it is a structural integer."""
    from .registry import INTEGER_KEYS

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and key in INTEGER_KEYS:
        return value
    if isinstance(value, (int, float)):
        return _num(value)
    if isinstance(value, dict):
        return {k: _labelled(k, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_labelled(key, v) for v in value]
    return str(value)


# --------------------------------------------------------------------------- addresses


@dataclass
class Target:
    text: str
    run_target: str | None
    row: int | None
    label: str | None
    selector: addresses.Selector
    cell: addresses.StateCell | None = None

    @property
    def layer_port(self) -> tuple[int, str] | None:
        if self.label is None:
            return None
        match = _LAYER_PORT.fullmatch(self.label)
        return None if match is None else (int(match["layer"]), match["port"])


def _program(ctx):
    if ctx.run is not None:
        return ctx.run.program
    if ctx.debugger is None:
        raise Refused("no model attached")
    return ctx.debugger.program


def _split(where: str) -> tuple[int, str]:
    match = _ROW.fullmatch(where)
    return int(match["row"]), match["rest"]


def _current_layer(ctx) -> int | None:
    for text in (ctx.focus, ctx.run.where):
        if text:
            match = _LAYER_PORT.search(text)
            if match is not None:
                return int(match["layer"])
    return ctx.run.pending_layer


def _target(ctx, text: str | None) -> Target:
    run = ctx.run
    program = run.program
    text = (text or "").strip() or ctx.focus or run.where
    if "~" in text:
        raise Refused("~program addresses are not built yet: A6 (compare)")
    base, bracket, tail = text.partition("[")
    if ":" in base or base.startswith("@"):
        raise Refused("origins (branches, @cut+edit) are not built yet: A6 (compare)")
    if base.startswith("cache."):
        try:
            cell = addresses.parse(text, ports=program.names, global_ports=GLOBALS)
        except addresses.AddressError as error:
            raise BadArgs(f"bad address {text!r}: {error}") from None
        if cell.layer >= run.adapter.layers:
            raise BadArgs(f"no layer {cell.layer} in a {run.adapter.layers}-layer model")
        return Target(addresses.format_address(cell), None, None, None, cell.selector, cell)
    match = _ROW.fullmatch(base)
    row = int(match["row"]) if match else _split(run.where)[0]
    rest = match["rest"] if match else base
    if rest in program.names:
        layer = _current_layer(ctx)
        if layer is None:
            raise Refused(f"bare port {rest!r} needs a layer here; write L<i>.{rest}")
        label = f"L{layer}.{rest}"
    elif rest in GLOBALS or _LAYER_PORT.fullmatch(rest):
        label = rest
    else:
        raise BadArgs(
            f"bad address {text!r}: expected t<row>.L<i>.<port>, L<i>.<port>, a port "
            f"({', '.join(program.names)}), a global ({', '.join(GLOBALS)}) or cache.L<i>.k|v"
        )
    try:
        parsed = addresses.parse(
            f"t{row}.{label}{bracket}{tail}", ports=program.names, global_ports=GLOBALS
        )
    except addresses.AddressError as error:
        raise BadArgs(f"bad address {text!r}: {error}") from None
    if parsed.port == "M":
        raise Refused("the declared margin M is not built yet: A5 (explain)")
    layer_port = _LAYER_PORT.fullmatch(label)
    if layer_port is not None and int(layer_port["layer"]) >= run.adapter.layers:
        raise BadArgs(f"no layer {layer_port['layer']} in a {run.adapter.layers}-layer model")
    return Target(
        addresses.format_address(parsed), f"t{row}.{label}", row, label, parsed.selector
    )


def _prov_json(raw: dict[str, Any]) -> dict[str, Any]:
    """``breakpoints.provenance`` as JSON: the label of the shown value and why."""
    authority = raw.get("authority", "measured")
    state = raw.get("state", "resolved")
    if authority is None and state != "resolved":
        # the engine declined to label the value (a replica whose check failed or never ran):
        # no authority is claimed, so no number about it is either
        out: dict[str, Any] = {"state": state, "reason": str(raw.get("reason") or state),
                               "edited": bool(raw.get("edited"))}
    elif authority not in _AUTHORITY:
        raise Refused(f"engine reported an unknown authority {authority!r}")
    else:
        out = {"authority": _AUTHORITY[authority], "edited": bool(raw.get("edited"))}
    if raw.get("by"):
        by = raw["by"]
        out["by"] = by if isinstance(by, str) else ",".join(map(str, by))
    kind = raw.get("kind")
    if kind and kind != "native":
        out["edit_kind"] = str(kind)
    if raw.get("under_edits_total"):
        out["under_edits"] = [dict(item) for item in raw["under_edits"]]
        out["under_edits_total"] = int(raw["under_edits_total"])
    if raw.get("mixed"):
        out["mixed"] = {k: int(v) for k, v in raw["mixed"].items()}
    if raw.get("same_port_edit"):
        out["same_port_edit"] = dict(raw["same_port_edit"])
    for key in ("realisation_exact", "realisation_max_abs_error"):
        if raw.get(key) is not None:
            out[key] = raw[key] if key == "realisation_exact" else _num(raw[key], "measured")
    if raw.get("replica_equal") is None:
        # None is "not applicable" or "not reached". A replica derived from captured operands
        # is labelled derived-exact either way; when its comparison with native has not run
        # the output says so.
        if raw.get("replica_checked") is False:
            out["replica_checked"] = False
            if raw.get("port") in ("s", "p"):
                out["replica_check_at"] = "p"
        elif raw.get("port") in ("s", "A") and kind == "replica":
            reason = "replica check not reached"
            out["replica_equal"] = _unresolved(reason)
            out["replica_max_abs_diff"] = _unresolved(reason)
    else:
        out["replica_equal"] = bool(raw["replica_equal"])
        out["replica_checked"] = True
        if raw.get("max_abs_diff") is not None:
            out["replica_max_abs_diff"] = _num(raw["max_abs_diff"])
    return out


def _read_info(ctx, target: Target, selection=None) -> dict[str, Any]:
    """Provenance of a value, from ``Run.read_info`` / ``Run.edit_mask`` (never a default)."""
    bp = _bp()
    if target.cell is not None:
        return _prov_json(bp.cell_provenance(ctx.run, target.cell.layer, selection,
                                             target.cell.tensor))
    raw = bp.provenance(ctx.run, target.run_target, selection)
    raw["port"] = target.layer_port[1] if target.layer_port else None
    return _prov_json(raw)


def _read(ctx, target: Target):
    run = ctx.run
    if target.cell is not None:
        # The committed boundary state (the cursor's cut): rows written so far, not the row
        # the pending layer is about to write.
        return run.read_cache(target.cell.layer, target.cell.tensor)
    return run.read(target.run_target)


def _axes(ctx, target: Target) -> tuple[str, ...]:
    bp = _bp()
    if target.cell is not None:
        return bp.CELL_AXES
    return bp.stop_axes(ctx.run.program, target.label)


def _select(ctx, target: Target, value):
    bp = _bp()
    axes = _axes(ctx, target)
    row = target.row if target.label not in (None, "start") and "row" in axes else None
    try:
        return bp.select(value, axes, target.selector, head_dim=ctx.run.adapter.head_dim,
                         row=row, tokenizer=ctx.tokenizer)
    except bp.SelectionRange as error:
        raise Refused(f"{target.text}: {error}") from None
    except bp.SelectionError as error:
        raise BadArgs(f"{target.text}: {error}") from None


def _check_selector(ctx, target: Target) -> None:
    """Bound-check a target's selector against the axis sizes that are knowable without a
    value (a refusal naming the axis and its size, before anything is stored)."""
    if not target.selector.keys and not target.selector.tokens:
        return
    bp = _bp()
    axes = _axes(ctx, target)
    try:
        bp.plan(axes, bp.axis_sizes(ctx.run.adapter, axes), target.selector,
                head_dim=ctx.run.adapter.head_dim, row=None, tokenizer=_NoTokenizer())
    except bp.SelectionRange as error:
        raise Refused(f"{target.text}: {error}") from None
    except bp.SelectionError as error:
        raise BadArgs(f"{target.text}: {error}") from None


class _NoTokenizer:
    """Token-text selectors are resolved when the value is read, not when it is stored."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0]


def _text(ctx, token_id: int) -> str | None:
    tokenizer = ctx.tokenizer
    if tokenizer is None:
        return None
    return tokenizer.decode([int(token_id)])


def _token(ctx, token_id: int) -> dict[str, Any]:
    out: dict[str, Any] = {"id": int(token_id)}
    text = _text(ctx, token_id)
    if text is not None:
        out["text"] = text
    return out


def _tokens(ctx, ids: list[int], keep: int = 16) -> dict[str, Any]:
    return {"tokens": [_token(ctx, i) for i in ids[-keep:]], "tokens_total": len(ids)}


def _edited(ctx) -> bool:
    return bool(ctx.run.edits) or bool(ctx.run.checkpoint()["path_edits"])


def _pointer(ctx, target: Target, value) -> tuple[dict | None, str | None]:
    from mdb.core.pointers import RecordError, occurrence_pointer
    from mdb.engine.ports import tensor_digest

    if target.cell is not None:
        return None, "cache cells are read at the current clock; no occurrence pointer"
    if target.label == "start":
        return None, "start is the token list, not a port occurrence"
    if _edited(ctx):
        return None, "edited run: occurrence origins (@cut+edit) are not wired yet"
    run = ctx.run
    cut = None
    layer_port = target.layer_port
    if layer_port is not None and run.pending_layer == layer_port[0] and (
        _split(run.where)[0] == target.row
    ):
        cut = run.checkpoint()["cut"]
    address = addresses.parse(target.run_target, ports=run.program.names,
                              global_ports=GLOBALS)
    authority, reason = _claim(_read_info(ctx, target))
    if authority is None:
        return None, f"no occurrence pointer: the value's label was refused ({reason})"
    try:
        pointer = occurrence_pointer(
            address,
            program="stepped",
            shape=list(value.shape),
            dtype=str(value.dtype).removeprefix("torch."),
            sha256=tensor_digest(value.contiguous()),
            cut=cut,
            authority=authority,
            ports=run.program.names,
            global_ports=addresses.DEFAULT_GLOBALS,
        )
    except RecordError as error:
        return None, f"no occurrence pointer: {error}"
    return pointer, None


def _claim(provenance: dict[str, Any]) -> tuple[str | None, str | None]:
    """``(authority, reason)`` of a value's provenance; no authority means its numbers are
    withheld (``{state: "unresolved", reason}``) with the reason the engine gave."""
    return provenance.get("authority"), provenance.get("reason")


def _stats(raw: dict[str, Any], authority: str | None = "measured",
           reason: str | None = None) -> dict[str, Any]:
    """Labelled statistics. With NaN/Inf elements they run over the finite ones
    (``over: "finite"``) beside integer ``nonfinite`` counts; with no finite element every
    statistic is ``{state: "unresolved", reason}``; so is every statistic of a value whose label
    the engine refused."""
    out: dict[str, Any] = {}
    if authority is None:
        for name in (*STATS, "argmax_index"):
            out[name] = _unresolved(reason or "no authority label")
        out["argmax_index"] = list(raw["argmax_index"]) if "argmax_index" in raw else []
    elif "unresolved" in raw:
        for name in (*STATS, "argmax_index"):
            out[name] = _unresolved(raw["unresolved"])
    else:
        out = {name: _num(raw[name], authority) for name in STATS}
        out["argmax_index"] = list(raw["argmax_index"])
    out["numel"] = int(raw["numel"])
    if "nonfinite" in raw:
        out["over"] = "finite"
        out["nonfinite"] = {k: int(v) for k, v in raw["nonfinite"].items()}
    return out


def _json_values(part) -> tuple[list, dict[str, int] | None]:
    """A flat tensor slice as JSON numbers; NaN/Inf become the strings ``"nan"``, ``"inf"``,
    ``"-inf"`` (JSON has none), with their counts."""
    import torch

    if not part.is_floating_point():
        return [int(v) for v in part.tolist()], None
    values = part.to(torch.float64).tolist()
    counts = _bp().nonfinite(part)
    if not any(counts.values()):
        return values, None
    named = {float("inf"): "inf", float("-inf"): "-inf"}
    return [("nan" if v != v else named[v]) if (v != v or v in named) else v
            for v in values], counts


def _reduction(ctx, target: Target, *, k: int = 5, slice_count: int = 0,
               pointer: bool = True) -> dict[str, Any]:
    bp = _bp()
    value = _read(ctx, target)
    selection = _select(ctx, target, value)
    provenance = _read_info(ctx, target, selection)
    authority, reason = _claim(provenance)
    out: dict[str, Any] = {
        "address": target.text,
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "axes": list(selection.axes),
        "provenance": provenance,
    }
    if list(selection.value.shape) != list(value.shape):
        out["selected_shape"] = list(selection.value.shape)
    if not value.is_floating_point():
        ids = [int(i) for i in selection.value.reshape(-1).tolist()]
        out.update(_tokens(ctx, ids, keep=32))
    else:
        if selection.value.numel() == 0:
            raise BadArgs(f"{target.text}: the selection is empty")
        out["stats"] = _stats(bp.reduce(selection), authority, reason)
        vocab = "vocab" in selection.axes
        entries = []
        for index, item in bp.top(selection, k):
            entry: dict[str, Any] = {
                "index": index,
                "value": _num(item, authority) if authority else _unresolved(reason or "refused"),
            }
            if vocab and ctx.tokenizer is not None:
                entry["text"] = _text(ctx, index[-1])
            entries.append(entry)
        out["k"] = k
        out["top"] = entries
        if slice_count:
            flat, counts = _json_values(selection.value.reshape(-1)[:slice_count])
            out["slice"] = ({"offset": 0, "values": flat, "authority": authority} if authority
                            else {"offset": 0, "values": _unresolved(reason or "refused")})
            if counts:
                out["slice"]["nonfinite"] = counts
    if pointer:
        found, reason = _pointer(ctx, target, value)
        if found is not None:
            out["pointer"] = found
        else:
            out["pointer_omitted"] = reason
    return out


# --------------------------------------------------------------------------- stops


def _stop(ctx, where: str | None = None) -> dict[str, Any]:
    run = ctx.run
    row, label = _split(where or run.where)
    info: dict[str, Any] = {"address": f"t{row}.{label}", "row": row}
    match = _LAYER_PORT.fullmatch(label)
    if match is not None:
        spec = run.program.spec(match["port"])
        info.update(
            layer=int(match["layer"]),
            port=spec.name,
            sublayer=spec.sublayer,
            operator=spec.operator,
            proofs=list(spec.proofs),
        )
    else:
        info.update(port=label, operator=GLOBAL_OPERATORS.get(label, label), proofs=[])
    return info


def _displays(ctx) -> list[dict[str, Any]]:
    from .errors import OpError

    out = []
    for number, text in sorted(ctx.displays.items())[:MAX_DISPLAYS]:
        try:
            item = _reduction(ctx, _target(ctx, text), k=3, pointer=False)
            item.pop("axes", None)
            out.append({"id": number, **item})
        except (OpError, ValueError) as error:
            out.append({"id": number, "address": text, "unavailable": str(error)})
    return out


def _moved(ctx, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    ctx.focus, ctx.focus_stack = None, []
    out: dict[str, Any] = {"where": ctx.run.where, "stop": _stop(ctx)}
    if extra:
        out.update(extra)
    if ctx.displays:
        out["displays"] = _displays(ctx)
    return out


def _hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for hit in hits:
        item = {key: hit[key] for key in ("id", "kind", "pattern", "mode") if key in hit}
        for key in ("old", "new"):
            if key in hit:
                prov = hit.get(f"{key}_provenance") or {"authority": "measured"}
                item[f"{key}_provenance"] = _prov_json(prov)
                item[key] = _stats(hit[key], *_claim(item[f"{key}_provenance"]))
        out.append(item)
    return out


# --------------------------------------------------------------------------- run / start


def _start(ctx, prompt) -> None:
    if ctx.debugger is None:
        raise Refused("no model attached; use mdb.attach(model) or --tiny")
    if prompt is None:
        if ctx.run is None:
            raise BadArgs("start needs a prompt (text or token ids)")
        prompt = list(ctx.run.tokens)
    run = ctx.debugger.run(prompt)  # the one implementation of start: cursor at t<last>.L0.h
    ctx.set_run(run)


_PROMPT = Arg("prompt", "prompt", help="prompt text (needs a tokenizer) or token ids; "
              "omitted = the current run's prompt")


@op("start", "Prefill a prompt and stop at the pending token's first port (L0.h)", _PROMPT,
    effect="native", needs_live=False, usage="start [TEXT | ID ID ...]")
def _op_start(ctx, prompt):
    _start(ctx, prompt)
    return _moved(ctx, {"prompt_tokens": len(ctx.run.tokens)})


@op("run", "Like start, then continue to the first armed breakpoint or watch", _PROMPT,
    Arg("budget", "int", default=8, minimum=0, maximum=STEP_CAP, positional=False,
        help="token rows continue may enter (minimum 1, at most 4096)"),
    effect="native", needs_live=False, usage="run [TEXT | ID ID ...] [budget=N]")
def _op_run(ctx, prompt, budget):
    _start(ctx, prompt)
    if not ctx.table.armed():
        return _moved(ctx, {"prompt_tokens": len(ctx.run.tokens), "reason": "no-breakpoints"})
    return _continue(ctx, budget=budget, extra=None,
                     more={"prompt_tokens": len(ctx.run.tokens)})


# --------------------------------------------------------------------------- cursor


@op("where", "Show the cursor, its operator and the context tokens", effect="read")
def _op_where(ctx):
    run = ctx.run
    out = {"where": run.where, "stop": _stop(ctx), "staged": len(run.edits),
           "lookahead": getattr(run, "lookahead", None)}
    if ctx.focus:
        out["focus"] = ctx.focus
    past = run.status().get("computed_past_cursor") or []
    if past:  # computed by the native call but not held: say so rather than hide it
        out["computed_past_cursor"] = list(past)
    out.update(_tokens(ctx, [int(t) for t in run.session.read("tokens")[0].tolist()]))
    return out


@op("frame", "Show or set the focus address (print/list/backtrace default to it)",
    Arg("address", "address", help="an address, or 'cursor' to follow the cursor again"),
    effect="navigate", usage="frame [ADDR | cursor]")
def _op_frame(ctx, address):
    if address == "cursor":
        ctx.focus, ctx.focus_stack = None, []
    elif address is not None:
        target = _target(ctx, address)
        ctx.focus, ctx.focus_stack = target.text, []
    focus = ctx.focus or ctx.run.where
    out: dict[str, Any] = {"focus": focus, "cursor": ctx.run.where}
    target = _target(ctx, focus)
    if target.label is not None:
        out["frame"] = _stop(ctx, target.run_target)
    return out


_N = Arg("n", "int", default=1, minimum=1, maximum=STEP_CAP, help="repeat count (at most 4096)")
_BUDGET = Arg("budget", "int", default=8, minimum=0, maximum=STEP_CAP, positional=False,
              help="token rows that may be entered before stopping (minimum 1, at most 4096)")
_ENGINE_UNITS = {
    "port": lambda label: True,
    "sublayer": lambda label: label in addresses.DEFAULT_GLOBALS or label == "start"
    or label.endswith((".r", ".h_out")),
    "layer": lambda label: label in addresses.DEFAULT_GLOBALS or label == "start"
    or label.endswith(".h_out"),
    "token": lambda label: label == "start",
}


def _walk(run, unit: str, n: int, *, reverse: bool) -> tuple[int, int]:
    """Where ``n`` unit steps end, computed from the stop order alone (nothing moves)."""
    accept = _ENGINE_UNITS[unit]
    stops = run.stops()
    size = len(stops)
    key = run.parse(run.where)
    root = (len(run.tokens) - 1, 0)
    for _ in range(n):
        while True:
            row, index = key
            if reverse:
                key = (row, index - 1) if index else (row - 1, size - 1)
            else:
                key = (row, index + 1) if index + 1 < size else (row + 1, 0)
            if reverse and key < root:
                raise Refused(f"cannot reverse before the run root t{root[0]}.{stops[root[1]]}; "
                              "nothing moved")
            if accept(stops[key[1]]):
                break
    return key


def _capped(ctx, unit: str, n: int, budget: int, *, reverse: bool) -> None:
    """Refuse, before anything moves, a unit step that would enter more token rows than
    ``budget`` (layer and token units; port and sublayer are bounded by ``n`` <= 4096)."""
    run = ctx.run
    target = _walk(run, unit, n, reverse=reverse)
    here = run.parse(run.where)
    entered = abs(target[0] - here[0])
    limit = max(1, budget)
    if unit in ("layer", "token") and entered > limit:
        raise Refused(
            f"{'reverse-' if reverse else ''}step {n} {unit} would enter {entered} token rows "
            f"(budget {limit}); pass budget=N (at most {STEP_CAP}). Nothing moved.")


def _step_forward(ctx, n: int, unit: str, budget: int) -> dict[str, Any]:
    _capped(ctx, unit, n, budget, reverse=False)
    for _ in range(n):
        try:
            ctx.run.step(unit, 1)
        except ValueError as error:
            if "capacity" not in str(error):
                raise
            return _moved(ctx, {"reason": "end-of-context"})
    return _moved(ctx)


def _step_back(ctx, n: int, unit: str, budget: int) -> dict[str, Any]:
    _capped(ctx, unit, n, budget, reverse=True)
    ctx.run.reverse_step(unit, n)
    return _moved(ctx)


@op("step", "Step one port (or N units; layer/token units enter at most `budget` rows)", _N,
    Arg("unit", "choice", default="port", choices=("port", "sublayer", "layer", "token")),
    _BUDGET, effect="navigate", motion=True,
    usage="step [N] [port|sublayer|layer|token] [budget=ROWS]")
def _op_step(ctx, n, unit, budget):
    return _step_forward(ctx, n, unit, budget)


@op("next", "Step to the last port of the sublayer (r, then h_out)", _N,
    effect="navigate", motion=True, usage="next [N]")
def _op_next(ctx, n):
    return _step_forward(ctx, n, "sublayer", 1)


@op("finish", "From a port inside a layer: run to that layer's h_out. From h_out or a "
    "token-level stop: run to this row's sample", effect="navigate", motion=True)
def _op_finish(ctx):
    run = ctx.run
    row, label = _split(run.where)
    found = _LAYER_PORT.fullmatch(label)
    if found is not None and found["port"] != "h_out":
        run.until(f"t{row}.L{found['layer']}.h_out")
    elif label == "sample":
        raise Refused(f"already at t{row}.sample, the end of the row; use `step token`")
    else:
        run.until(f"t{row}.sample")
    return _moved(ctx)


def _run_to(ctx, target: str, budget: int, *, name: str) -> dict[str, Any]:
    """Run to an address, stopping earlier at an armed breakpoint or watch."""
    bp = _bp()
    run = ctx.run
    found = _target(ctx, target)
    if found.cell is not None:
        raise BadArgs(f"{name} takes a port occurrence, not a cache cell")
    if name == "advance" and found.selector:
        raise BadArgs("advance takes a port occurrence without selectors")
    key, here = run.parse(found.run_target), run.parse(run.where)
    if key == here and name == "until":
        return _moved(ctx)
    if key <= here:
        raise Refused(f"{found.run_target} is "
                      + ("behind the cursor " if name == "until" else "not ahead of the cursor ")
                      + f"{run.where}; use reverse-step or restart")
    if key[0] > run.adapter.max_positions - 1:
        raise Refused(f"{found.run_target} is past the model's context limit "
                      f"({run.adapter.max_positions} positions)")
    pattern = bp.parse_pattern(found.run_target, run.program)
    item = bp.Breakpoint(0, found.run_target, pattern, kind="advance")
    budget = max(budget, key[0] - here[0] + 1)
    out = _continue(ctx, budget=budget, extra=[item])
    if name == "until" and out["reason"] == "advance":
        out.update(reason="reached", hits=[], hits_total=0)
        out.pop("omitted", None)
    return out


@op("until", "Run to an address, stopping earlier at an armed breakpoint or watch",
    Arg("target", "address", required=True), _BUDGET, effect="navigate", motion=True,
    usage="until ADDR [budget=ROWS]")
def _op_until(ctx, target, budget):
    return _run_to(ctx, target, budget, name="until")


@op("advance", "Run to an address, stopping earlier at an armed breakpoint or watch",
    Arg("target", "address", required=True), _BUDGET, effect="navigate", motion=True,
    usage="advance ADDR [budget=ROWS]")
def _op_advance(ctx, target, budget):
    return _run_to(ctx, target, budget, name="advance")


def _selection_of(hits: list[dict[str, Any]]) -> str | None:
    """The selector of the first hit's pattern (``[h0]``), which ``print`` then shows."""
    for hit in hits:
        pattern = hit.get("pattern", "")
        if "[" in pattern:
            return pattern[pattern.index("["):]
    return None


def _continue(ctx, *, budget, extra, more=None) -> dict[str, Any]:
    bp = _bp()
    run = ctx.run
    result = bp.run_forward(run, ctx.table, budget=budget, extra=extra,
                            tokenizer=ctx.tokenizer, max_positions=run.adapter.max_positions)
    return _stopped(ctx, result, {"budget": max(1, budget), **(more or {})})


def _stopped(ctx, result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """The result of a forward/reverse run-to-hit: the first 32 hits (``hits_total`` and
    ``omitted`` say how many there were), an ``error`` when a condition failed to evaluate,
    and the cursor, always."""
    hits = result["hits"]
    payload.update(reason=result["reason"], hits=_hits(hits[:PAGE]), hits_total=len(hits))
    if len(hits) > PAGE:
        payload["omitted"] = len(hits) - PAGE
    if "error" in result:
        payload["error"] = dict(result["error"])
    out = _moved(ctx, payload)
    selection = _selection_of(hits)
    if selection is not None:
        ctx.focus = f"{ctx.run.where}{selection}"
        out["focus"] = ctx.focus
    return out


@op("continue", "Run to the next armed breakpoint or watch hit (N: ignore the breakpoint "
    "stopped at N-1 more times, as gdb)", Arg("n", "int", default=1, minimum=1),
    _BUDGET, effect="navigate", motion=True, usage="continue [N] [budget=ROWS]")
def _op_continue(ctx, n, budget):
    more = {}
    if n > 1:
        stop = ctx.table.last_stop if ctx._table is not None else None
        ids = [i for i in (stop["ids"] if stop else []) if i in ctx.table.items]
        if stop is not None and stop["where"] == ctx.run.where and ids:
            for number in ids:
                ctx.table.items[number].ignore = n - 1
            more["ignored"] = {"ids": ids, "count": n - 1}
        else:
            more["note"] = "not stopped at a breakpoint; the count is ignored"
    return _continue(ctx, budget=budget, extra=None, more=more)


# --------------------------------------------------------------------------- breakpoints


_COND = Arg("condition", "condition", positional=False,
            help="e.g. norm > 3 and argmax == 2; reductions: norm min max mean abs_max argmax "
                 "numel; v[...] indexes")


def _adapter(ctx):
    return ctx.run.adapter if ctx.run is not None else getattr(ctx.debugger, "adapter", None)


def _table_error(error: Exception):
    """Table/selector failures as typed errors: out-of-range is a refusal naming the axis and
    its size, anything malformed is bad arguments."""
    bp = _bp()
    if isinstance(error, bp.SelectionRange):
        return Refused(str(error))
    return BadArgs(str(error))


def _add_break(ctx, pattern, condition, temporary):
    bp = _bp()
    try:
        item = ctx.table.add_break(pattern, _program(ctx), condition=condition,
                                   temporary=temporary, adapter=_adapter(ctx))
    except (addresses.AddressError, bp.ConditionError, bp.SelectionError) as error:
        raise _table_error(error) from None
    return {"breakpoint": item.describe()}


@op("break", "Set a breakpoint on an address pattern (L*.p, t5.L1.o, logits, cache.L0.v[row=2]); cache cells are evaluated at boundary stops only",
    Arg("pattern", "pattern", required=True), _COND, effect="mutate", needs_live=False,
    usage="break PATTERN [if CONDITION]")
def _op_break(ctx, pattern, condition):
    return _add_break(ctx, pattern, condition, False)


@op("tbreak", "Set a breakpoint that deletes itself after its first hit",
    Arg("pattern", "pattern", required=True), _COND, effect="mutate", needs_live=False,
    usage="tbreak PATTERN [if CONDITION]")
def _op_tbreak(ctx, pattern, condition):
    return _add_break(ctx, pattern, condition, True)


def _ids(found: list[int]) -> dict[str, Any]:
    """Ids affected by an op, the first 32 of them (``total`` and ``omitted`` say the rest)."""
    out: dict[str, Any] = {"ids": found[:PAGE], "total": len(found)}
    if len(found) > PAGE:
        out["omitted"] = len(found) - PAGE
    return out


@op("delete", "Delete breakpoints and watches (all when no ids are given)",
    Arg("ids", "ints"), effect="mutate", needs_live=False, usage="delete [ID ...]")
def _op_delete(ctx, ids):
    gone = ctx.table.delete(ids)
    out: dict[str, Any] = {"deleted": gone[:PAGE], "deleted_total": len(gone)}
    if len(gone) > PAGE:
        out["omitted"] = len(gone) - PAGE
    return out


@op("enable", "Enable breakpoints and watches (all when no ids are given)",
    Arg("ids", "ints"), effect="mutate", needs_live=False, usage="enable [ID ...]")
def _op_enable(ctx, ids):
    return {"enabled": True, **_ids(ctx.table.set_enabled(ids, True))}


@op("disable", "Disable breakpoints and watches without deleting them (all when no ids)",
    Arg("ids", "ints"), effect="mutate", needs_live=False, usage="disable [ID ...]")
def _op_disable(ctx, ids):
    return {"enabled": False, **_ids(ctx.table.set_enabled(ids, False))}


@op("condition", "Set the condition of breakpoint N (a watch's predicate); no expression "
    "removes it", Arg("id", "int", required=True, minimum=1),
    Arg("expr", "condition", positional=False), effect="mutate", needs_live=False,
    usage="condition N [EXPR]")
def _op_condition(ctx, id, expr):  # noqa: A002 - the arg name is the wire name
    bp = _bp()
    try:
        item = ctx.table.set_condition(id, expr, _program(ctx), _adapter(ctx))
    except (addresses.AddressError, bp.ConditionError, bp.SelectionError) as error:
        raise _table_error(error) from None
    return {"watch" if item.kind == "watch" else "breakpoint": item.describe(),
            "removed": expr is None}


@op("ignore", "Ignore the next COUNT hits of breakpoint N (they are counted, not reported)",
    Arg("id", "int", required=True, minimum=1), Arg("count", "int", required=True, minimum=0),
    effect="mutate", needs_live=False, usage="ignore N COUNT")
def _op_ignore(ctx, id, count):  # noqa: A002
    item = ctx.table.set_ignore(id, count)
    return {"watch" if item.kind == "watch" else "breakpoint": item.describe(), "ignore": count}


@op("watch", "Watch an address: stop when it changes (or a predicate holds: level/edge); cache cells (cache.L<i>.k|v) are evaluated at boundary stops only",
    Arg("address", "pattern", required=True),
    Arg("mode", "choice", default="change", choices=("change", "level", "edge")),
    Arg("predicate", "condition", positional=False), effect="mutate", needs_live=False,
    usage="watch ADDR [change|level|edge] [if PREDICATE]")
def _op_watch(ctx, address, mode, predicate):
    bp = _bp()
    try:
        item = ctx.table.add_watch(address, _program(ctx), predicate=predicate, mode=mode,
                                   adapter=_adapter(ctx))
    except ValueError as error:  # Address/Condition/SelectionError, or a mode without predicate
        raise _table_error(error) from None
    out: dict[str, Any] = {"watch": item.describe()}
    if ctx.run is not None:
        try:
            primed = bp.prime(ctx.run, item, tokenizer=ctx.tokenizer)
        except Exception as error:  # noqa: BLE001 - nothing stays armed that cannot be read
            ctx.table.delete([item.id])
            if isinstance(error, ValueError):
                raise _table_error(error) from None
            raise
        out["baseline"] = ctx.run.where if primed else None
    return out


@op("display", "Show an address after every stop (no address: list displays)",
    Arg("address", "address"), effect="mutate", usage="display [ADDR]")
def _op_display(ctx, address):
    if address is not None:
        if len(ctx.displays) >= MAX_DISPLAYS:
            raise Refused(f"at most {MAX_DISPLAYS} displays")
        target = _target(ctx, address)
        _check_selector(ctx, target)
        if target.cell is None and target.label is not None:
            text = address.strip()
        else:
            text = target.text
        ctx.displays[ctx.next_display_id()] = text
    return {"displays": _displays(ctx)}


@op("undisplay", "Remove displays (all when no ids are given)", Arg("ids", "ints"),
    effect="mutate", needs_live=False, usage="undisplay [ID ...]")
def _op_undisplay(ctx, ids):
    if ids is None:
        gone = sorted(ctx.displays)
        ctx.displays.clear()
    else:
        missing = [i for i in ids if i not in ctx.displays]
        if missing:
            raise Refused(f"no display {missing[0]}")
        gone = sorted(ids)
        for i in ids:
            del ctx.displays[i]
    return {"deleted": gone}


# --------------------------------------------------------------------------- values


@op("print", "Reduce a value: shape, dtype, stats, top-k by |value|, pointer",
    Arg("address", "address"),
    Arg("top", "int", default=5, minimum=1, maximum=16, positional=False),
    Arg("slice", "int", default=0, minimum=0, maximum=32, positional=False,
        help="also show the first N selected values"),
    effect="read", usage="print [ADDR] [top=K] [slice=N]")
def _op_print(ctx, address, top, slice):  # noqa: A002 - the arg name is the wire name
    return _reduction(ctx, _target(ctx, address), k=top, slice_count=slice)


@op("x", "Examine raw values: a bounded slice of the selected, flattened value",
    Arg("address", "address"), Arg("count", "int", default=16, minimum=1, maximum=64),
    Arg("offset", "int", default=0, minimum=0), effect="read",
    usage="x [ADDR] [COUNT] [OFFSET]")
def _op_x(ctx, address, count, offset):
    target = _target(ctx, address)
    value = _read(ctx, target)
    selection = _select(ctx, target, value)
    provenance = _read_info(ctx, target, selection)
    flat = selection.value.reshape(-1)
    if offset >= flat.numel():
        raise BadArgs(f"offset {offset} is past the {flat.numel()} selected values")
    part = flat[offset:offset + count]
    values, counts = _json_values(part)
    out = {
        "address": target.text,
        "shape": list(value.shape),
        "selected_shape": list(selection.value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "offset": offset,
        "count": len(values),
        "numel": int(flat.numel()),
    }
    if part.is_floating_point():
        authority, reason = _claim(provenance)
        out["values"] = ({"values": values, "authority": authority} if authority
                         else _unresolved(reason or "refused"))
        if counts:
            out["nonfinite"] = counts
        out["provenance"] = provenance
    else:
        out.update(_tokens(ctx, values, keep=64))
    return out


def _edit_json(record: dict[str, Any]) -> dict[str, Any]:
    """A staged or committed edit (``Run.staged_info`` / ``Run.committed_edits`` record): the
    requested value ``supplied``, the selection and mask fields, and every realisation field
    (cast and realised errors, ``changed_fraction``, replica fields, ``p`` row sums) kept,
    numbers labelled ``measured`` except the row-sum tolerance, which is ``supplied``."""
    value = record["value"]
    out: dict[str, Any] = {"port": record["port"], "op": record["op"], "kind": value["kind"]}
    if value["kind"] == "scalar":
        out["value"] = _num(value["value"], "supplied")
    else:
        out.update(sha256=value["sha256"], shape=value["shape"],
                   dtype=value["dtype"].removeprefix("torch."))
    for key, item in record.items():
        if key in ("port", "op", "value"):
            continue
        if key == "row_sum_tolerance":
            out[key] = _num(item, "supplied")
        else:
            out[key] = _labelled(key, item)
    return out


def _value_source(ctx, value):
    """A scalar, or the tensor at another computed address of the same run."""
    if not isinstance(value, str):
        return value, None
    try:
        return float(value), None
    except ValueError:
        pass
    source = _target(ctx, value)
    tensor = _read(ctx, source)
    return _select(ctx, source, tensor).value, source.text


def _finite_limit(run, value) -> None:
    import torch

    dtype = run.adapter.dtype
    if isinstance(value, (int, float)) and not isinstance(value, bool) and dtype.is_floating_point:
        limit = float(torch.finfo(dtype).max)
        if abs(float(value)) > limit:
            raise Refused(
                f"{value!r} is outside the finite range of {str(dtype).removeprefix('torch.')} "
                f"(|x| <= {limit:.6g}); nothing was staged")


@op("set", "Stage an edit at a port of the pending layer: set ADDR = VALUE [op]",
    Arg("target", "address", required=True), Arg("value", "value", required=True),
    Arg("op", "choice", default="replace", choices=("replace", "add", "scale")),
    effect="mutate", usage="set ADDR = VALUE|ADDR [replace|add|scale]")
def _op_set(ctx, target, value, op):
    import torch

    run = ctx.run
    found = _target(ctx, target)
    if found.layer_port is None:
        raise Refused(f"edits apply to ports of the pending layer, not {found.text}")
    value, source = _value_source(ctx, value)
    _finite_limit(run, value)
    kwargs: dict[str, Any] = {}
    if found.selector:
        # ONE edit: the selector becomes a mask over the port (shape known without running
        # anything), so replace/add/scale apply to the selected elements only and the rest
        # stay bit-identical. Same behaviour ahead of the cursor as at it.
        bp = _bp()
        shape = tuple(run.port_shape(found.layer_port[1]))
        axes = _axes(ctx, found)
        row = found.row if "row" in axes else None
        try:
            picks = bp.plan(axes, list(shape), found.selector, head_dim=run.adapter.head_dim,
                            row=row, tokenizer=ctx.tokenizer)
        except bp.SelectionRange as error:
            raise Refused(f"{found.text}: {error}") from None
        except bp.SelectionError as error:
            raise BadArgs(f"{found.text}: {error}") from None
        index = tuple(
            torch.tensor(list(range(shape[axis]) if pick is None else pick), dtype=torch.long).view(
                [-1 if a == axis else 1 for a in range(len(shape))])
            for axis, pick in enumerate(picks)
        )
        mask = torch.zeros(shape, dtype=torch.bool)
        mask[index] = True
        chosen = tuple(shape[a] if pick is None else len(pick) for a, pick in enumerate(picks))
        if isinstance(value, torch.Tensor):
            if torch.broadcast_shapes(value.shape, chosen) != chosen:
                raise BadArgs(f"{source} does not broadcast to the selection {list(chosen)}")
            full = torch.zeros(shape, dtype=run.adapter.dtype)
            full[index] = value.to(run.adapter.dtype).expand(chosen)
            value = full
        kwargs = {"mask": mask, "selection": found.text.partition("[")[2].rstrip("]")}
    manifest = run.set(found.run_target, value, op, **kwargs)  # atomic: failure stages nothing
    staged = run.staged_info()
    out = {"target": found.text, "edits": [_edit_json(staged[-1] if staged else manifest)],
           "staged": len(run.edits), "where": run.where}
    if source is not None:
        out["source"] = source
    if found.selector:
        out["realised_as"] = f"one masked {op} on the selected elements"
    return out


@op("commit", "Commit the pending layer with its staged edits (cursor to its h_out)",
    effect="mutate", motion=True)
def _op_commit(ctx):
    committed = len(ctx.run.edits)
    try:
        ctx.run.commit()
    except ValueError as error:
        if committed:
            raise Refused(
                f"cannot commit: the engine refused the layer state ({error}). Staged edits "
                "that make a value non-finite (inf/nan) cannot be committed: inspect the "
                "edited ports with `print`, then run `abort` to drop the staged edits."
            ) from error
        raise
    return _moved(ctx, {"committed": committed})


@op("abort", "Drop the staged edits of the pending layer", effect="mutate")
def _op_abort(ctx):
    dropped = len(ctx.run.edits)
    ctx.run.abort()
    return _moved(ctx, {"count": dropped})


# --------------------------------------------------------------------------- program


def _checks(ctx) -> dict[str, Any]:
    out = {}
    for name, check in ctx.run.checks.items():
        item = {key: value for key, value in check.items() if isinstance(value, bool)}
        if "max_abs_diff" in check:
            item["max_abs_diff"] = _num(check["max_abs_diff"])
        out[name] = item
    return out


@op("list", "The layer's port program: operator text, proof IDs, cursor marked",
    Arg("layer", "int", minimum=0), effect="read", usage="list [LAYER]")
def _op_list(ctx, layer):
    run = ctx.run
    row, label = _split(run.where)
    focus_port = None
    if layer is None:
        layer = _current_layer(ctx)
        if layer is None:
            layer = 0 if label in ("start", "emb") else run.adapter.layers - 1
    if layer >= run.adapter.layers:
        raise BadArgs(f"no layer {layer} in a {run.adapter.layers}-layer model")
    match = _LAYER_PORT.fullmatch(label)
    at = match["port"] if match is not None and int(match["layer"]) == layer else None
    if ctx.focus:
        found = _LAYER_PORT.search(ctx.focus)
        if found is not None and int(found["layer"]) == layer:
            focus_port = found["port"].split("[")[0]
    here = run.parse(run.where)
    rows = []
    for spec in run.program.ports:
        key = run.parse(f"t{row}.L{layer}.{spec.name}")
        rows.append({
            "port": spec.name,
            "sublayer": spec.sublayer,
            "operator": spec.operator,
            "proofs": list(spec.proofs),
            "editable": spec.editable,
            "cursor": spec.name == at,
            "focus": spec.name == focus_port,
            "computed": key <= here,
        })
    out = {"layer": layer, "row": row, "where": run.where, "ports": rows}
    if run.pending_layer == layer and ctx.run.checks:
        out["checks"] = _checks(ctx)
    return out


def _focus_parts(ctx, address: str | None) -> tuple[int, str]:
    target = _target(ctx, address)
    if target.label is None:
        raise BadArgs("backtrace and up/down follow port occurrences, not cache cells")
    return target.row, target.label


def _producers(program, layers: int, label: str) -> list[str]:
    """Every producer frame of a stop, nearest-first (the first is what ``up`` follows)."""
    match = _LAYER_PORT.fullmatch(label)
    if match is None:
        found = {"emb": "start", "fn": f"L{layers - 1}.h_out", "logits": "fn",
                 "sample": "logits"}.get(label)
        return [] if found is None else [found]
    layer, port = int(match["layer"]), match["port"]
    inputs = graph.producers(program)[port]
    if inputs:
        return [f"L{layer}.{item}" for item in inputs]
    return ["emb" if layer == 0 else f"L{layer - 1}.h_out"]


def _producer(program, layers: int, label: str) -> str | None:
    found = _producers(program, layers, label)
    return found[0] if found else None


def _consumer(program, layers: int, label: str) -> str | None:
    match = _LAYER_PORT.fullmatch(label)
    if match is None:
        return {"start": "emb", "emb": "L0.h", "fn": "logits", "logits": "sample"}.get(label)
    layer, port = int(match["layer"]), match["port"]
    outputs = graph.consumers(program)[port]
    if outputs:
        return f"L{layer}.{outputs[0]}"
    return f"L{layer + 1}.h" if layer + 1 < layers else "fn"


@op("backtrace", "Producer chain from the focus port back to the layer input",
    Arg("address", "address"), effect="read", usage="backtrace [ADDR]")
def _op_backtrace(ctx, address):
    run = ctx.run
    row, label = _focus_parts(ctx, address)
    match = _LAYER_PORT.fullmatch(label)
    frames = []
    if match is None:
        levels, item = [], label
        while item is not None and _LAYER_PORT.fullmatch(item) is None:
            levels.append([item])
            item = _producer(run.program, run.adapter.layers, item)
        if item is not None:
            levels.append([item])
        for level, names in enumerate(levels):
            for name in names:
                frames.append({"level": level, **_stop(ctx, f"t{row}.{name}")})
        chain = graph.chain_text(levels)
    else:
        layer = int(match["layer"])
        levels = graph.layering(run.program, match["port"])
        edges = graph.producers(run.program)
        extra = graph.pseudo(run.program)
        for level, names in enumerate(levels):
            for name in names:
                frame = _stop(ctx, f"t{row}.L{layer}.{name}")
                frame["level"] = level
                frame["inputs"] = [p for p in edges[name] if any(p in lv for lv in levels)]
                if extra[name]:
                    frame["reads"] = list(extra[name])  # pseudo-nodes: cache, prev_layer
                frames.append(frame)
        chain = graph.chain_text(levels)
    return {"focus": f"t{row}.{label}", "chain": chain, "frames": frames}


@op("up", "Move the focus to a producer (toward the layer input), N times; `up PORT` picks "
    "the producer by name, the result lists the alternatives", _N,
    Arg("port", "str", help="the producer to follow first, e.g. o from r"),
    effect="navigate", usage="up [N] [PORT]")
def _op_up(ctx, n, port):
    run = ctx.run
    layers = run.adapter.layers
    row, label = _focus_parts(ctx, None)
    stack = list(ctx.focus_stack)
    here = ctx.focus or run.where
    options: list[str] = []
    for step in range(n):  # computed on copies: a refusal leaves the focus where it was
        options = _producers(run.program, layers, label)
        if not options:
            raise Refused(f"no frame beyond t{row}.{label}")
        chosen = options[0]
        if port is not None and step == 0:
            named = [o for o in options if o.split(".")[-1] == port or o == port]
            if not named:
                raise Refused(f"t{row}.{label} has no producer {port!r}; producers: "
                              + ", ".join(options))
            chosen = named[0]
        stack.append(here)  # one stack entry per frame, so `down 1` walks back one frame
        label = chosen
        here = f"t{row}.{label}"
    ctx.focus, ctx.focus_stack = here, stack
    out: dict[str, Any] = {"focus": ctx.focus, "frame": _stop(ctx, ctx.focus),
                           "producers": options, "alternatives": [o for o in options
                                                                    if o != label]}
    if run.parse(ctx.focus) > run.parse(run.where):
        out["computed"] = False
    return out


@op("down", "Move the focus back toward its consumer, N times", _N, effect="navigate",
    usage="down [N]")
def _op_down(ctx, n):
    run = ctx.run
    focus, stack = ctx.focus, list(ctx.focus_stack)
    for _ in range(n):
        if stack:
            focus = stack.pop()
            continue
        row, label = _focus_parts(ctx, focus)
        nxt = _consumer(run.program, run.adapter.layers, label)
        if nxt is None:
            raise Refused(f"no consumer frame after t{row}.{label}")
        focus = f"t{row}.{nxt}"
    ctx.focus, ctx.focus_stack = focus, stack
    if ctx.focus == run.where:
        ctx.focus = None
    focus = ctx.focus or run.where
    return {"focus": focus, "frame": _stop(ctx, _target(ctx, focus).run_target)}


# --------------------------------------------------------------------------- info


def _program_info(ctx) -> dict[str, Any]:
    program = _program(ctx)
    adapter = ctx.run.adapter if ctx.run is not None else ctx.debugger.adapter
    return {
        "family": program.family,
        "modeling": program.modeling,
        "layers": adapter.layers,
        "hidden_size": adapter.hidden_size,
        "heads": adapter.num_heads,
        "kv_heads": adapter.num_kv_heads,
        "head_dim": adapter.head_dim,
        "vocab_size": adapter.vocab_size,
        "max_positions": adapter.max_positions,
        "dtype": str(adapter.dtype).removeprefix("torch."),
        "device": str(adapter.device),
        "attention": adapter.execution.get("attention"),
        "lookahead": getattr(ctx.run, "lookahead", None),
        "port_names": list(program.names),
        "globals": list(GLOBALS),
    }


@op("info", "Show debugger tables or the program (lists are paged: offset=, limit=)",
    Arg("subject", "choice", required=True,
        choices=("ports", "breakpoints", "watch", "display", "checkpoints", "edits",
                 "program")),
    _OFFSET, _LIMIT,
    effect="read", needs_live=False, usage="info ports|breakpoints|watch|display|checkpoints|"
    "edits|program [offset=N] [limit=N]")
def _op_info(ctx, subject, offset, limit):
    if subject == "program":
        return {"subject": subject, "program": _program_info(ctx)}
    if subject == "ports":
        program = _program(ctx)
        return {"subject": subject, "family": program.family,
                "ports": [spec.to_dict() for spec in program.ports]}
    if subject in ("breakpoints", "watch"):
        items = [item.describe() for item in ctx.table.listed()] if ctx._table else []
        if subject == "watch":
            items = [item for item in items if item["kind"] == "watch"]
        page, meta = _page(items, offset, limit)
        return {"subject": subject, "items": page, **meta}
    if subject == "display":
        shown, meta = _page([{"id": number, "address": text}
                             for number, text in sorted(ctx.displays.items())], offset, limit)
        return {"subject": subject, "displays": shown, **meta}
    if subject == "checkpoints":
        shown, meta = _page(_checkpoint_list(ctx), offset, limit)
        return {"subject": subject, "checkpoints": shown, **meta}
    if ctx.run is None:
        raise Refused("info edits needs a live run")
    staged = ctx.run.staged_info()
    groups: dict[int, dict[str, Any]] = {}
    for record in ctx.run.committed_edits():
        group = groups.setdefault(record["transition"], {
            "transition": record["transition"], "layer": record["layer"], "row": record["row"],
            "edits": []})
        group["edits"].append(record)
    entries = list(groups.values())
    # One window over the staged edits followed by the committed layers.
    total = len(staged) + len(entries)
    out_staged = [_edit_json(m) for m in staged[offset:offset + limit]]
    lo = max(0, offset - len(staged))
    hi = max(0, offset + limit - len(staged))
    out_committed = []
    for entry in entries[lo:hi]:
        edits = entry["edits"]
        item = {"transition": entry["transition"], "layer": entry["layer"], "row": entry["row"],
                "edits": [_edit_json(m) for m in edits[:limit]],
                "edits_total": len(edits)}
        if len(edits) > limit:
            item["omitted"] = len(edits) - limit
        out_committed.append(item)
    return {"subject": subject, "staged": out_staged, "committed": out_committed,
            "where": ctx.run.where, "staged_total": len(staged),
            "committed_total": len(entries), "total": total, "offset": offset, "limit": limit}


# --------------------------------------------------------------------------- checkpoints


def _checkpoint_list(ctx) -> list[dict[str, Any]]:
    return [
        {"name": name, "where": cp["where"], "staged": len(cp["edits"]),
         "committed": len(cp["path_edits"])}
        for name, cp in ctx.checkpoints.items()
    ]


@op("checkpoint", "Name the cursor position (about 1 KB of plain data)",
    Arg("name", "str"), effect="mutate", usage="checkpoint [NAME]")
def _op_checkpoint(ctx, name):
    if len(ctx.checkpoints) >= MAX_CHECKPOINTS and name not in ctx.checkpoints:
        raise Refused(f"at most {MAX_CHECKPOINTS} checkpoints; reuse a name or start again")
    name = name or ctx.next_checkpoint_name()
    checkpoint = ctx.run.checkpoint(name)
    ctx.checkpoints[name] = checkpoint
    result = {"name": name, "where": checkpoint["where"],
              "bytes": len(json.dumps(checkpoint, sort_keys=True))}
    if _project(ctx) is not None:  # durable: recipe, edit blobs and cursor go to the project
        stored = ctx.debugger.checkpoint(ctx.run, name)
        result["durable"] = {"recipe": stored["recipe"], "blobs": stored["blobs"]}
    return result


def _project(ctx):
    return getattr(ctx.debugger, "project", None)


@op("checkpoints", "List named checkpoints (paged: offset=, limit=)", _OFFSET, _LIMIT,
    effect="read", needs_live=False, usage="checkpoints [offset=N] [limit=N]")
def _op_checkpoints(ctx, offset, limit):
    shown, meta = _page(_checkpoint_list(ctx), offset, limit)
    return {"checkpoints": shown, **meta}


@op("restart", "Return to a named checkpoint (rewinds or replays)",
    Arg("name", "str", required=True), effect="navigate", motion=True, needs_live=False,
    usage="restart NAME")
def _op_restart(ctx, name):
    if ctx.run is not None and name in ctx.checkpoints:
        ctx.run.restore(ctx.checkpoints[name])
    elif _project(ctx) is not None:  # a checkpoint an earlier process stored in the project
        ctx.set_run(ctx.debugger.restart(name))
    elif ctx.run is None:
        raise Refused("no run to rewind; start one with run, or attach with project= to "
                      "restart a stored checkpoint")
    else:
        known = ", ".join(ctx.checkpoints) or "none"
        raise Refused(f"no checkpoint {name!r} (have: {known})")
    ctx.table.reset_watches()
    return _moved(ctx, {"name": name})


# --------------------------------------------------------------------------- reverse


@op("reverse-step", "Step back one port (or N units; layer/token units enter at most `budget` "
    "rows)", _N,
    Arg("unit", "choice", default="port", choices=("port", "sublayer", "layer", "token")),
    _BUDGET, effect="navigate", motion=True,
    usage="reverse-step [N] [unit] [budget=ROWS]")
def _op_reverse_step(ctx, n, unit, budget):
    return _step_back(ctx, n, unit, budget)


@op("reverse-next", "Step back to the previous sublayer end", _N, effect="navigate",
    motion=True, usage="reverse-next [N]")
def _op_reverse_next(ctx, n):
    return _step_back(ctx, n, "sublayer", 1)


@op("reverse-finish", "Step back to the boundary before the pending layer (its input)",
    effect="navigate", motion=True)
def _op_reverse_finish(ctx):
    return _step_back(ctx, 1, "layer", 1)


@op("reverse-continue", "Run backwards to the previous breakpoint or watch hit",
    effect="navigate", motion=True)
def _op_reverse_continue(ctx):
    result = _bp().run_reverse(ctx.run, ctx.table, tokenizer=ctx.tokenizer)
    return _stopped(ctx, result, {})


# --------------------------------------------------------------------------- slots

for _name, _summary, _milestone in (
    ("forecast", "Seal a forecast at the cursor before the native step", "A5"),
    ("verify", "Check a sealed forecast against the native receipt", "A5"),
    ("whatif", "Predicted versus measured effect of an edit on a consumer", "A5"),
    ("explain", "Per-layer readout of a consumer with proof IDs and remainder", "A5"),
    ("reads", "Who reads a value: attention reads with mass and causal verdict", "A6"),
    ("symbols", "Symbol bindings at the cursor with readers and lifetimes", "A6"),
    ("circuit", "Circuit records: nodes, edges, consumer, unlisted remainder", "A6"),
    ("compare", "Compare branches (sham, A, B, A+B, predicted-only)", "A6"),
    ("seek", "Seek to an occurrence through recorded projects", "A6"),
    ("export", "Export a walkthrough bundle of records and pointers", "B1"),
):
    slot(_name, _summary, _milestone)
