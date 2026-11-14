"""The prediction verbs: ``explain``, ``forecast``, ``verify`` and ``whatif``.

They replace the A5 slots of ``builtin`` (``register(..., replace=True)``). Import is
torch-free; each op imports the analysis layer when it runs.

- ``explain ADDR``: the exact accounting of a margin ``M[a,b]`` (``analysis.explain.
  explain_margin``) or of one port (``explain_port``) of the row the cursor is at the
  ``logits``/``sample`` of.
- ``forecast [ADDR|M[a,b]]``: seal a prediction of the next stop (or of a later stop of the
  cursor's row, or a margin) from values at the cursor (``analysis.forecast.seal``).
- ``verify [IDS]``: compare the open forecasts whose target the cursor has passed with the
  native values (``analysis.forecast.verify``).
- ``whatif ADDR=EXPR [replace|add|scale|mix DOSE] consumer=C``: predicted, then natively
  measured change of a consumer under one edit, the session left exactly where it was
  (``analysis.propagate.whatif``). ADDR may be a state cell ``cache.L<i>.k|v[kv,row,d]``; EXPR
  may be ``ORIGIN:ADDR`` (a value of another run) for a port and a cell alike.

Assignments (``parse_assignment``, no ``eval``): ``ADDR = EXPR``, ``ADDR += X`` (add X),
``ADDR -= X`` (add -X), ``ADDR *= X`` (scale X), ``ADDR /= X`` (scale by the float64 1/X, which
the edit casts to the model dtype: not a rounded division); ``-=`` and ``/=`` take a number.

Margins are ``M[a,b]`` / ``t<row>.M[a,b]`` with token ids or quoted token texts (one token each);
a single logit is ``logits[i]`` / ``t<row>.logits[i=7]``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mdb.core.pointers import seal_digest

from .builtin import cell_request, port_request, resolve_target
from .errors import BadArgs, Refused
from .registry import Arg, Op, register

_MARGIN = re.compile(r"(?:t(?P<row>\d+)\.)?M\[(?P<a>[^,\]]+),(?P<b>[^\]]+)\]")
_LOGIT = re.compile(r"(?:t(?P<row>\d+)\.)?logits\[(?:i=)?(?P<i>\d+)\]")
_ROW = re.compile(r"t(?P<row>\d+)\.")


def _row(ctx, match) -> int:
    return int(match["row"]) if match["row"] else int(_ROW.match(ctx.run.where)["row"])


def _in_vocab(ctx, token: int) -> int:
    """``token`` when it indexes the logits; ``Refused`` with the valid range otherwise."""
    vocab = int(ctx.run.adapter.model.lm_head.weight.shape[0])
    if not 0 <= token < vocab:
        raise Refused(f"token id {token} is outside the vocabulary (valid ids 0..{vocab - 1})")
    return token


def _token_id(ctx, text: str) -> int:
    """A token id (inside the vocabulary), or the single id of a quoted token text."""
    text = text.strip()
    if re.fullmatch(r"\d+", text):
        return _in_vocab(ctx, int(text))
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        word = json.loads(text) if text[0] == '"' else text[1:-1]
        if ctx.tokenizer is None:
            raise BadArgs(f"token text {text} needs a tokenizer; this model has none, give ids")
        ids = ctx.tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise Refused(f"token text {text} is {len(ids)} tokens, not one")
        return int(ids[0])
    raise BadArgs(f"a margin side is a token id or a quoted token text, got {text!r}")


def _margin(ctx, text: str):
    from mdb.analysis.forecast import Margin

    match = _MARGIN.fullmatch(text.strip())
    if match is None:
        return None
    return Margin(_row(ctx, match), _token_id(ctx, match["a"]), _token_id(ctx, match["b"]))


def record_view(record: dict[str, Any]) -> dict[str, Any]:
    """A sealed record for an op result: every field except ``execution`` (the adapter's
    execution settings, plain configuration numbers), which is replaced by its digest; the
    record's ``id`` still seals the whole envelope."""
    out = {key: value for key, value in record.items() if key != "execution"}
    out["execution_sha256"] = seal_digest(record["execution"])
    return out


def _whole_stop(ctx, text: str, verb: str) -> str:
    """The canonical ``t<row>.<stop>`` of an address naming one whole stop."""
    found = resolve_target(ctx, text)
    if found.cell is not None:
        raise Refused(f"{verb} takes a stop of the token rows, not the cache cell {found.text}")
    if found.selector:
        raise Refused(f"{verb} works on a whole stop; drop the selector from {found.text}")
    return found.run_target


# --------------------------------------------------------------------------- explain


def _explain(ctx, address):
    from mdb.analysis import explain as explain_mod

    run = ctx.run
    row = int(_ROW.match(run.where)["row"])
    margin = _margin(ctx, address)
    if margin is None:
        found = resolve_target(ctx, address)
        if found.layer_port is None:
            raise Refused(f"explain takes a margin M[a,b] or a layer port, not {found.text}")
        target_row = found.row
    else:
        target_row = margin.row
    stop = run.where.split(".", 1)[1]
    if target_row != row or stop not in ("logits", "sample"):
        raise Refused(f"explain reads the whole captured row t{target_row}: run `until "
                      f"t{target_row}.logits` first (cursor {run.where})")
    if margin is not None:
        result = explain_mod.explain_margin(run, margin.a, margin.b)
        project = getattr(ctx.debugger, "project", None)
        explain_mod.bound_terms(
            result, run=run,
            store=project.store if project is not None and not project.read_only else None)
        result["tokens"] = {side: {"token": token} for side, token in result["tokens"].items()}
        result["edits"] = [{"layer": int(layer), "edits": records}
                           for layer, records in sorted(result["edits"].items())]
        return {"target": margin.text, "kind": "margin", "explain": result}
    layer, port = found.layer_port
    result = explain_mod.explain_port(run, layer, port)
    identity = result.pop("result", None)
    for key in ("reconstruction", "native"):
        result.pop(key, None)
    if identity is not None:
        result["identity"] = identity.summary()
    return {"target": found.text.partition("[")[0], "kind": "port", "explain": result}


# --------------------------------------------------------------------------- forecast / verify


def _forecast(ctx, target):
    from mdb.analysis import forecast

    if target is None:
        chosen = None
    else:
        chosen = _margin(ctx, target) or _whole_stop(ctx, target, "forecast")
    try:
        project = getattr(ctx.debugger, "project", None)
        store = project.store if project is not None and not project.read_only else None
        view = forecast.seal(ctx.run, chosen, store=store)
    except forecast.Refusal as error:
        raise Refused(str(error)) from None
    entry = forecast.ledger(ctx.run)[view["id"]]
    return {"forecast": view, "record": record_view(entry.record)}


def _verify(ctx, ids):
    from mdb.analysis import forecast

    entries = forecast.ledger(ctx.run)
    unknown = [i for i in ids or () if i not in entries]
    if unknown:
        raise Refused(f"no forecast {unknown[0]} (have: {', '.join(map(str, entries)) or 'none'})")
    results = forecast.verify(ctx.run, ids)
    return {"forecasts": results, "count": len(results), "where": ctx.run.where}


# --------------------------------------------------------------------------- whatif


def session_digest(ctx) -> str:
    """What ``whatif`` must leave unchanged at the debugger level: the run's state digest,
    breakpoints and watches, displays, focus and named checkpoints."""
    from mdb.analysis.propagate import state_digest

    return seal_digest({
        "run": state_digest(ctx.run),
        "table": [item.describe() for item in ctx.table.listed()],
        "displays": sorted([number, text] for number, text in ctx.displays.items()),
        "focus": [ctx.focus, list(ctx.focus_stack)],
        "checkpoints": sorted(ctx.checkpoints),
    })


def build_edit(ctx, target: str, value: Any, op: str, dose: float | None = None):
    """The requested edit as ``propagate.Edit``. A port: ``builtin.port_request`` (a selector
    becomes a mask over it; a value is a scalar, the value at another computed address of this
    run, or ``ORIGIN:ADDR``, a value of another run). A state cell (``cache.L<i>.k|v[...]``):
    ``builtin.cell_request`` (ops ``replace|add|scale|mix``, values also ``ORIGIN:ADDR``)."""
    from mdb.analysis.propagate import Edit

    found = resolve_target(ctx, target)
    if found.cell is not None:
        request = cell_request(ctx, found, value, op, dose)
        cell = {key: request[key] for key in ("layer", "tensor", "rows", "kv", "d")}
        return Edit(found.text, request["value"], op, None, request["selection"],
                    request["source"], cell=cell, dose=dose, origin=request["origin"])
    if found.layer_port is None:
        raise Refused(f"edits apply to layer ports or state cells, not {found.text}")
    if op == "mix" or dose is not None:
        raise Refused("mix DOSE applies to state cells (cache.L<i>.k|v)")
    request = port_request(ctx, found, value)
    return Edit(found.run_target, request["value"], op, request["mask"], request["selection"],
                request["source"], origin=request["origin"])


def parse_consumer(ctx, text: str):
    """A whatif/compare consumer: a margin ``M[a,b]``, a logit ``logits[i]`` or a whole stop."""
    from mdb.analysis.forecast import Logit

    margin = _margin(ctx, text)
    if margin is not None:
        return margin
    match = _LOGIT.fullmatch(text.strip())
    if match is not None:
        return Logit(_row(ctx, match), _in_vocab(ctx, int(match["i"])))
    return _whole_stop(ctx, text, "a whatif consumer")


_SUGAR = {"+": "add", "-": "add", "*": "scale", "/": "scale"}


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_assignment(text: str, *, usage: str = "ADDR=EXPR [replace|add|scale|mix DOSE]"
                     ) -> tuple[str, Any, str | None, float | None]:
    """``ADDR = EXPR [OP [DOSE]]`` or ``ADDR OP= X`` (``+= -= *= /=``) -> ``(address, value, op,
    dose)``; ``op`` None when none was written. Parsed, never evaluated: EXPR is a number or an
    address (``ORIGIN:ADDR`` included); ``-=`` and ``/=`` take a number (``-= X`` is ``add -X``,
    ``/= X`` is ``scale`` by the float64 ``1/X``)."""
    depth, at = 0, -1
    for index, char in enumerate(text):
        depth += char == "["
        depth -= char == "]"
        if char == "=" and depth == 0:
            at = index
            break
    if at <= 0:
        raise BadArgs(f"write {usage}")
    left, right = text[:at].strip(), text[at + 1:].split()
    sugar = left[-1] if left and left[-1] in _SUGAR else None
    if sugar is not None:
        left = left[:-1].strip()
    if not left or not right:
        raise BadArgs(f"write {usage}")
    expr, rest = right[0], right[1:]
    number = _number(expr)
    value: Any = expr if number is None else number
    op: str | None = None
    dose: float | None = None
    if sugar is not None:
        if rest:
            raise BadArgs(f"{sugar}= takes one value and no op word")
        op = _SUGAR[sugar]
        if sugar in "-/":
            if number is None:
                raise BadArgs(f"{sugar}= takes a number, not {expr!r}")
            if sugar == "/" and number == 0:
                raise BadArgs("/= 0 divides by zero")
            value = -number if sugar == "-" else 1.0 / number
        return left, value, op, dose
    if rest:
        op = rest[0]
        if op == "mix":
            if len(rest) != 2 or _number(rest[1]) is None:
                raise BadArgs("mix takes a dose: ADDR = VALUE mix DOSE")
            dose = _number(rest[1])
        elif len(rest) != 1:
            raise BadArgs(f"write {usage}")
    return left, value, op, dose


def split_assignment(target: str, value: Any) -> tuple[str, Any]:
    """``ADDR=EXPR`` in one word (``=`` outside brackets) -> ``(ADDR, EXPR)``."""
    depth = 0
    for index, char in enumerate(target):
        depth += char == "["
        depth -= char == "]"
        if char == "=" and depth == 0:
            left, right = target[:index].strip(), target[index + 1:].strip()
            if value is not None or not left or not right:
                raise BadArgs("write whatif ADDR=EXPR [replace|add|scale] consumer=C")
            try:
                return left, float(right)
            except ValueError:
                return left, right
    if value is None:
        raise BadArgs("whatif needs a value: whatif ADDR=EXPR [replace|add|scale] consumer=C")
    return target, value


def _whatif(ctx, target, value, op, consumer, dose=None):
    from mdb.analysis import forecast, propagate

    address, value = split_assignment(target, value)
    edit = build_edit(ctx, address, value, op, dose)
    chosen = parse_consumer(ctx, consumer)
    before = session_digest(ctx)
    project = getattr(ctx.debugger, "project", None)
    store = project.store if project is not None and not project.read_only else None
    try:
        result = propagate.whatif(ctx.run, edit, chosen, store=store)
    except forecast.Refusal as error:
        raise Refused(str(error)) from None
    after = session_digest(ctx)
    if after != before:
        raise RuntimeError("whatif changed the debugger session")
    report = dict(result.report)
    report["session"] = {**report["session"], "debugger_before": before,
                         "debugger_after": after}
    return {**report, "where": ctx.run.where, "sealed_record": record_view(result.record)}


# --------------------------------------------------------------------------- registration

register(Op("explain", "Exact accounting of a margin M[a,b] or of one port, with proof IDs "
            "and the measured remainder (cursor at the row's logits/sample)",
            (Arg("address", "address", required=True),), "read", True, _explain,
            usage="explain M[a,b]|ADDR"), replace=True)
register(Op("forecast", "Seal a prediction of the next stop (or a later stop of this row, or a "
            "margin M[a,b]) from values at the cursor, before native code runs",
            (Arg("target", "address"),), "mutate", True, _forecast,
            usage="forecast [ADDR|M[a,b]]"), replace=True)
register(Op("verify", "Compare open forecasts whose target the cursor has passed with the "
            "native values (confirmed or not by the measured residual, shown in dtype spacings, "
            "against its floor 3 x sqrt(rounded ops) x spacing; order)",
            (Arg("ids", "ints"),), "mutate", True, _verify, usage="verify [ID ...]"),
         replace=True)
register(Op("whatif", "Predicted, then natively measured change of a consumer under one edit; "
            "the session is left exactly where it was",
            (Arg("target", "str", required=True), Arg("value", "value"),
             Arg("op", "choice", default="replace", choices=("replace", "add", "scale", "mix")),
             Arg("consumer", "str", required=True, positional=False),
             Arg("dose", "number", positional=False, help="mix only (state cells)")),
            "navigate", True, _whatif,
            usage="whatif ADDR=EXPR [replace|add|scale|mix DOSE] "
                  "consumer=M[a,b]|logits[i]|ADDR"),
         replace=True)

# old private names, kept for one commit
_record_view, _session_digest, _edit = record_view, session_digest, build_edit
_consumer, _split_assignment = parse_consumer, split_assignment
