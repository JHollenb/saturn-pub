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
- ``whatif ADDR=EXPR [replace|add|scale] consumer=C``: predicted, then natively measured change
  of a consumer under one edit, the session left exactly where it was
  (``analysis.propagate.whatif``).

Margins are ``M[a,b]`` / ``t<row>.M[a,b]`` with token ids or quoted token texts (one token each);
a single logit is ``logits[i]`` / ``t<row>.logits[i=7]``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mdb.core.pointers import seal_digest

from .builtin import _bp, _finite_limit, _target, _value_source
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


def _record_view(record: dict[str, Any]) -> dict[str, Any]:
    """A sealed record for an op result: every field except ``execution`` (the adapter's
    execution settings, plain configuration numbers), which is replaced by its digest; the
    record's ``id`` still seals the whole envelope."""
    out = {key: value for key, value in record.items() if key != "execution"}
    out["execution_sha256"] = seal_digest(record["execution"])
    return out


def _whole_stop(ctx, text: str, verb: str) -> str:
    """The canonical ``t<row>.<stop>`` of an address naming one whole stop."""
    found = _target(ctx, text)
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
        found = _target(ctx, address)
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
        view = forecast.seal(ctx.run, chosen)
    except forecast.Refusal as error:
        raise Refused(str(error)) from None
    entry = forecast.ledger(ctx.run)[view["id"]]
    return {"forecast": view, "record": _record_view(entry.record)}


def _verify(ctx, ids):
    from mdb.analysis import forecast

    entries = forecast.ledger(ctx.run)
    unknown = [i for i in ids or () if i not in entries]
    if unknown:
        raise Refused(f"no forecast {unknown[0]} (have: {', '.join(map(str, entries)) or 'none'})")
    results = forecast.verify(ctx.run, ids)
    return {"forecasts": results, "count": len(results), "where": ctx.run.where}


# --------------------------------------------------------------------------- whatif


def _session_digest(ctx) -> str:
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


def _edit(ctx, target: str, value: Any, op: str):
    """The requested edit as ``propagate.Edit``: a selector becomes a mask over the port, a
    value is a scalar or the value at another computed address of this run."""
    import torch

    from mdb.analysis.propagate import Edit

    run = ctx.run
    found = _target(ctx, target)
    if found.layer_port is None:
        raise Refused(f"edits apply to layer ports, not {found.text}")
    if isinstance(value, str) and ":" in value.partition("[")[0]:
        raise Refused(f"{value}: a value from another origin needs the origin resolver, which "
                      "is not built yet; use a scalar or an address of this run")
    value, source = _value_source(ctx, value)
    _finite_limit(run, value)
    mask = selection = None
    if found.selector:
        bp = _bp()
        shape = tuple(run.port_shape(found.layer_port[1]))
        axes = bp.stop_axes(run.program, found.label)
        try:
            picks = bp.plan(axes, list(shape), found.selector, head_dim=run.adapter.head_dim,
                            row=found.row if "row" in axes else None, tokenizer=ctx.tokenizer)
        except bp.SelectionRange as error:
            raise Refused(f"{found.text}: {error}") from None
        except bp.SelectionError as error:
            raise BadArgs(f"{found.text}: {error}") from None
        index = tuple(
            torch.tensor(list(range(shape[a]) if pick is None else pick), dtype=torch.long).view(
                [-1 if b == a else 1 for b in range(len(shape))])
            for a, pick in enumerate(picks))
        mask = torch.zeros(shape, dtype=torch.bool)
        mask[index] = True
        chosen = tuple(shape[a] if pick is None else len(pick) for a, pick in enumerate(picks))
        if isinstance(value, torch.Tensor):
            if torch.broadcast_shapes(value.shape, chosen) != chosen:
                raise BadArgs(f"{source} does not broadcast to the selection {list(chosen)}")
            full = torch.zeros(shape, dtype=run.adapter.dtype)
            full[index] = value.to(run.adapter.dtype).expand(chosen)
            value = full
        selection = found.text.partition("[")[2].rstrip("]")
    return Edit(found.run_target, value, op, mask, selection, source)


def _consumer(ctx, text: str):
    from mdb.analysis.forecast import Logit

    margin = _margin(ctx, text)
    if margin is not None:
        return margin
    match = _LOGIT.fullmatch(text.strip())
    if match is not None:
        return Logit(_row(ctx, match), _in_vocab(ctx, int(match["i"])))
    return _whole_stop(ctx, text, "a whatif consumer")


def _split_assignment(target: str, value: Any) -> tuple[str, Any]:
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


def _whatif(ctx, target, value, op, consumer):
    from mdb.analysis import forecast, propagate

    address, value = _split_assignment(target, value)
    edit = _edit(ctx, address, value, op)
    chosen = _consumer(ctx, consumer)
    before = _session_digest(ctx)
    try:
        result = propagate.whatif(ctx.run, edit, chosen)
    except forecast.Refusal as error:
        raise Refused(str(error)) from None
    after = _session_digest(ctx)
    if after != before:
        raise RuntimeError("whatif changed the debugger session")
    report = dict(result.report)
    report["session"] = {**report["session"], "debugger_before": before,
                         "debugger_after": after}
    return {**report, "where": ctx.run.where, "sealed_record": _record_view(result.record)}


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
            "native values (residual, bound, order)",
            (Arg("ids", "ints"),), "mutate", True, _verify, usage="verify [ID ...]"),
         replace=True)
register(Op("whatif", "Predicted, then natively measured change of a consumer under one edit; "
            "the session is left exactly where it was",
            (Arg("target", "str", required=True), Arg("value", "value"),
             Arg("op", "choice", default="replace", choices=("replace", "add", "scale")),
             Arg("consumer", "str", required=True, positional=False)),
            "navigate", True, _whatif,
            usage="whatif ADDR=EXPR [replace|add|scale] consumer=M[a,b]|logits[i]|ADDR"),
         replace=True)
