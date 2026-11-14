"""Named symbols declared from a transplant result (issue M23).

A transplant is a ``whatif`` or ``compare`` edit that writes a value taken from another run (or
another address of the same run) at one source occurrence, ``t18.L20.v[kv0] = clean:t18.L20.
v[kv0]``, and measures a consumer natively. ``symbol NAME = whatif`` (the last ``whatif`` of
this session), ``symbol NAME = compare.a|compare.b`` (a side of the last ``compare``) or
``symbol NAME = EACH:ITEM`` (one item's whatif of an ``each`` result, in memory or stored)
turns it into a reusable name. The ``symbol`` record (``core.kinds.SYMBOL``) binds:

* ``role``: the writer, row-free (``L20.v[kv0]``: port, layer and selector, or a cache cell
  with ``row=@``), and its readers as ``reads`` lists them at the consumer's row (mass
  ``measured``, effect ``derived-exact P19``, causal verdict ``measured``), or why not;
* ``row``: the source row, role-relative (``t@word``) when the run's prompt is an item of a
  loaded prompt set with a role at that row (or ``role=NAME`` names it, ``supplied``),
  otherwise absolute, with the prompt's and the prefix's sha256;
* ``consumer`` and ``evidence``: the measured effect that justifies the name (base, edited,
  delta ``measured``; the verdict). A result whose verdict is not ``effect`` is refused: no
  record is made (a name stands for a measured effect, never for its absence);
* ``origin``: the donor value (run, address, sha256, cut, recipe), the run, its prefill
  procedure and its edits, where the result came from.

Use: the name stands for its address where an address is expected (``print NAME``, ``whatif
NAME = clean:NAME --consumer C``, ``break NAME``, ``reads NAME``...); ``registry.dispatch``
calls ``expand`` before the op, and the result lists each use under ``symbol_uses``. The row
binds on the run the address is read in (``ORIGIN:NAME``: on run ORIGIN): its declaring prompt;
a prompt-set item with the same tokens and the role; the same tokens before the row. Inside
``each``, ``NAME`` binds through the item's role (``t@word``; ``@pair:NAME`` through the pair's)
and an absolute symbol through the item's prefix. A use that does not bind is refused with the
reason. Records are pointers and labelled numbers only (64 KiB).
"""

from __future__ import annotations

import re
import weakref
from typing import Any

from mdb.core.pointers import RecordError, seal_digest

from .context import Context
from .errors import BadArgs, OpError, Refused
from .registry import hook

SCHEMA = "mdb-symbol-v1"
MAX_READERS = 8
MAX_EDITS = 16
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ADDRESS_LIKE = re.compile(r"(?:t|L|h|kv)\d+")
RESERVED = frozenset({"Unbound", "last", "pair", "cache", "whatif", "compare", "show", "delete",
                      "cursor", "M"})
_REFERENCE = re.compile(r"(?P<kind>whatif|compare\.(?P<side>[ab]))|"
                        r"(?P<each>each\d+|[0-9a-f]{12,64}):(?P<item>[A-Za-z_][A-Za-z0-9_]*)")
_OCCURRENCE = re.compile(r"t(?P<row>\d+)\.(?P<writer>L(?P<layer>\d+)\.(?P<port>\w+)"
                         r"(?:\[(?P<sel>[^\]]*)\])?)")
_CELL = re.compile(r"cache\.L(?P<layer>\d+)\.(?P<port>k|v)\[(?P<sel>[^\]]*)\]")
# the verbs whose arguments are addresses (or ADDR = VALUE sides): a name there is expanded
ADDRESS_ARGS = {
    "print": ("address",), "x": ("address",), "display": ("address",), "frame": ("address",),
    "backtrace": ("address",), "explain": ("address",), "forecast": ("target",),
    "until": ("target",), "advance": ("target",), "seek": ("target",), "break": ("pattern",),
    "tbreak": ("pattern",), "watch": ("address",), "set": ("target", "value"),
    "whatif": ("target", "value"), "compare": ("a", "b"), "reads": ("source",),
}

_STORE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()   # debugger -> {name: record}
_LAST: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()    # ctx -> {op: (result, run)}


def store(debugger: Any) -> dict[str, dict[str, Any]]:
    """The named symbols of ``debugger`` (name -> sealed ``symbol`` record)."""
    if debugger not in _STORE:
        _STORE[debugger] = {}
    return _STORE[debugger]


def is_reference(text: str) -> bool:
    """Whether the right side of ``symbol NAME = ...`` names a result, not addresses."""
    return _REFERENCE.fullmatch(text.strip()) is not None


# --------------------------------------------------------------------------- binding


def _sha(tokens: list[int]) -> str:
    return seal_digest([int(t) for t in tokens])


def _tokens(run: Any) -> list[int]:
    return [int(t) for t in run.tokens]


def address_at(body: dict[str, Any], row: int | str) -> str:
    """The symbol's address at ``row`` (an int, or ``@ROLE`` for a template)."""
    writer = body["role"]["writer"]
    if writer["form"] == "cell":
        return writer["address"].replace("row=@", f"row={row}")
    return f"t{row}.{writer['address']}"


def _set_items(debugger: Any, tokens: list[int], role: str) -> list[tuple[str, int]]:
    """``(set/item, row)`` of every loaded prompt-set item with these tokens and this role."""
    from . import sets

    found = []
    for name, found_set in sorted(sets.sets_of(debugger).items()):
        for item in found_set.items:
            if role not in item.roles:
                continue
            try:
                same = debugger.tokens(item.prompt) == tokens
            except Refused:
                continue
            if same:
                found.append((f"{name}/{item.name}", int(item.roles[role])))
    return found


def bind(debugger: Any, record: dict[str, Any], tokens: list[int]) -> tuple[int, str]:
    """``(row, how)`` of a symbol on a prompt, or ``Refused`` with the reason."""
    body = record["body"]
    row = body["row"]
    name = body["name"]
    if _sha(tokens) == row["prompt_sha256"]:
        return row["row"], "its declaring prompt"
    if row["rule"] == "role":
        role = row["role"]
        if role == "last":
            return len(tokens) - 1, "role @last"
        found = _set_items(debugger, tokens, role)
        rows = sorted({r for _, r in found})
        if len(rows) == 1:
            return rows[0], f"role @{role} of {found[0][0]}"
        if len(rows) > 1:
            raise Refused(f"symbol {name}: role @{role} names rows {rows} in the loaded set items "
                          f"with this prompt ({', '.join(f for f, _ in found)})")
    at = row["row"]
    if at < len(tokens) and _sha(tokens[:at]) == row["prefix_sha256"]:
        return at, f"same tokens as its declaring prompt before row {at}"
    what = (f"role @{row['role']} (row {at} of its declaring prompt)" if row["rule"] == "role"
            else f"absolute row t{at} (no role named that row when it was declared)")
    tried = ("no loaded prompt-set item with role "
             f"@{row['role']} has this prompt, and " if row["rule"] == "role" else "")
    raise Refused(f"symbol {name} binds {what}; on this prompt ({len(tokens)} tokens) {tried}"
                  f"the tokens before row {at} differ from its declaring prompt's, so the row "
                  "is not known: use it inside `each SET ...` where the item's role binds it, or "
                  "declare it from a run of this prompt")


def _bind_run(ctx: Any, record: dict[str, Any], origin: str | None) -> tuple[int, str, Any]:
    debugger = ctx.debugger
    if origin is None:
        run = ctx.run
        if run is None:
            raise Refused(f"symbol {record['body']['name']}: no live run to bind it on")
    else:
        run = (getattr(debugger, "runs", None) or {}).get(origin)
        if run is None:
            known = ", ".join(getattr(debugger, "runs", None) or ()) or "none"
            raise Refused(f"symbol {record['body']['name']} in {origin}:: {origin} is not a live "
                          f"run of this debugger (runs: {known}); a symbol binds its row on a "
                          "live run's prompt")
    row, how = bind(debugger, record, _tokens(run))
    return row, how, run


def _pattern(names: list[str]) -> re.Pattern:
    joined = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(r"(?<![\w.@$:\[\"'\-])(?:(?P<origin>[A-Za-z_][\w\-]*):)?"
                      rf"(?P<name>{joined})(?![\w.\[:\"'\-])")


def _assigns(text: str) -> bool:
    depth = 0
    for char in text:
        depth += char == "["
        depth -= char == "]"
        if char == "=" and depth == 0:
            return True
    return False


def expand(ctx: Any, text: str, *, notes: list[dict[str, Any]] | None = None) -> str:
    """``text`` with every symbol name (and ``ORIGIN:NAME``) replaced by its address on the
    run it is read in; ``Refused`` when one does not bind."""
    debugger = getattr(ctx, "debugger", None)
    named = _STORE.get(debugger) if debugger is not None else None
    if not named:
        return text

    def one(match: re.Match) -> str:
        record = named[match["name"]]
        origin = match["origin"]
        row, how, run = _bind_run(ctx, record, origin)
        address = address_at(record["body"], row)
        note: dict[str, Any] = {"symbol": match["name"],
                                "address": f"{origin}:{address}" if origin else address,
                                "row": row, "how": how}
        tokens = _tokens(run)
        if row < len(tokens):
            note["token"] = tokens[row]
            if ctx.tokenizer is not None:
                note["text"] = ctx.tokenizer.decode([tokens[row]])
        if notes is not None:
            notes.append(note)
        return f"{origin}:{address}" if origin else address

    return _pattern(list(named)).sub(one, text)


def expand_for_item(debugger: Any, text: str, tokens: list[int]) -> str:
    """``text`` for one prompt-set item (``each``): a role symbol becomes its role template
    (``t@word.L20.v[kv0]``, bound by the item's roles next), an absolute one its row when the
    item's tokens before it are the declaring prompt's; ``ORIGIN:NAME`` binds on run ORIGIN.
    ``Refused`` with the reason otherwise."""
    named = _STORE.get(debugger)
    if not named:
        return text

    def one(match: re.Match) -> str:
        record = named[match["name"]]
        body = record["body"]
        if match["origin"]:
            row, _, _ = _bind_run(Context(debugger), record, match["origin"])
            return f"{match['origin']}:{address_at(body, row)}"
        if body["row"]["rule"] == "role":
            return address_at(body, f"@{body['row']['role']}")
        row, _ = bind(debugger, record, tokens)
        return address_at(body, row)

    return _pattern(list(named)).sub(one, text)


def _before(ctx: Any, name: str, values: dict[str, Any]) -> tuple[dict[str, Any], list]:
    fields = ADDRESS_ARGS.get(name)
    debugger = getattr(ctx, "debugger", None)
    if not fields or debugger is None or not _STORE.get(debugger):
        return values, []
    notes: list[dict[str, Any]] = []
    out = dict(values)
    for key in fields:
        value = out.get(key)
        if not isinstance(value, str) or (name == "compare" and not _assigns(value)):
            continue      # a compare side without `=` is a checkpoint name, not an address
        out[key] = expand(ctx, value, notes=notes)
    return out, notes[:16]


def _after(ctx: Any, name: str, result: dict[str, Any]) -> None:
    run = getattr(ctx, "run", None)
    if name in ("whatif", "compare") and run is not None:
        _LAST.setdefault(ctx, {})[name] = (result, weakref.ref(run))


hook(before=_before, after=_after)


# --------------------------------------------------------------------------- declaring


def check_name(ctx: Any, name: str) -> None:
    program = ctx.run.program if ctx.run is not None else ctx.debugger.program
    from .builtin import GLOBALS

    if not _NAME.fullmatch(name) or name in RESERVED or _ADDRESS_LIKE.fullmatch(name) \
            or name in program.names or name in GLOBALS:
        raise Refused(f"a symbol name is a word ([A-Za-z_][A-Za-z0-9_]*) that is not an address, "
                      f"a port, a stop or reserved ({', '.join(sorted(RESERVED))}), not {name!r}")
    runs = getattr(ctx.debugger, "runs", None) or {}
    if name in runs:
        raise Refused(f"{name} is a run of this debugger; a symbol needs another name")


def _source(text: str) -> dict[str, Any]:
    """The writer of a source address: row, row-free address, form, reads source."""
    text = text.strip()
    match = _OCCURRENCE.fullmatch(text)
    if match is not None:
        row, layer, port, sel = (int(match["row"]), int(match["layer"]), match["port"],
                                 match["sel"])
        reads = why = None
        if port in ("k", "v"):
            if sel is None:
                reads = f"t{row}.L{layer}.{port}"
            elif re.fullmatch(r"kv\d+", sel):
                reads = f"cache.L{layer}.{port}[{sel},row={row}]"
            else:
                why = (f"reads follows whole KV heads; the source selects [{sel}]")
        else:
            why = (f"reads follows cache cells (k, v); a {port} value is read by the later stops "
                   "of its own row, not by attention")
        return {"row": row, "layer": layer, "port": port, "form": "occurrence",
                "writer": match["writer"], "reads": reads, "why": why, "text": text}
    match = _CELL.fullmatch(text)
    if match is None:
        raise Refused(f"a symbol's source is a port occurrence t<r>.L<l>.<port>[sel] or a cache "
                      f"cell cache.L<l>.k|v[...,row=<r>], not {text!r}")
    parts = [p.strip() for p in match["sel"].split(",") if p.strip()]
    rows = [p for p in parts if p.startswith("row=")]
    if len(rows) != 1 or not rows[0][4:].isdigit():
        raise Refused(f"a symbol binds one source row; {text} selects "
                      f"{rows[0] if rows else 'no row'}")
    row = int(rows[0][4:])
    rest = [p for p in parts if not p.startswith("row=")]
    writer = f"cache.L{match['layer']}.{match['port']}[" + ",".join(
        "row=@" if p.startswith("row=") else p for p in parts) + "]"
    whole = all(re.fullmatch(r"kv\d+", p) for p in rest)
    return {"row": row, "layer": int(match["layer"]), "port": match["port"], "form": "cell",
            "writer": writer, "text": text,
            "reads": (f"cache.L{match['layer']}.{match['port']}[{','.join(rest + [f'row={row}'])}]"
                      if whole else None),
            "why": None if whole else f"reads follows whole KV heads; the source selects {text}"}


def _num(item: Any) -> Any:
    """A labelled number as stored: ``{value, authority[, proof]}`` or a state with a reason."""
    if isinstance(item, dict) and "value" in item and "authority" in item:
        return {k: item[k] for k in ("value", "authority", "proof") if k in item}
    if isinstance(item, dict) and item.get("state"):
        return {"state": item["state"], "reason": str(item.get("reason", item["state"]))}
    return {"state": "unresolved", "reason": "not in the result"}


def _consumer(text: str) -> dict[str, Any]:
    match = re.fullmatch(r"t(?P<row>\d+)\.(?P<rest>.+)", text)
    out: dict[str, Any] = {"address": text}
    if match is not None:
        out["row"] = int(match["row"])
    return out


def _from_whatif(result: dict[str, Any], label: str) -> dict[str, Any]:
    edit = result["edit"]
    address = edit["address"]
    if "[" not in address and edit.get("selection"):
        address = f"{address}[{edit['selection']}]"
    origin = edit.get("origin")
    donor = edit.get("source")
    if edit.get("op") != "replace" or not (isinstance(origin, dict) or isinstance(donor, str)):
        raise Refused(f"{label} is not a transplant: it {edit.get('op')}s "
                      f"{'a scalar' if not isinstance(donor, str) else donor} at {address}; a "
                      "symbol is declared from a replace whose value is another run's (ORIGIN:"
                      "ADDR) or another address's")
    effect = result.get("effect") or {}
    consumer = dict(result["consumer"])
    measured = result["measured"]
    evidence = {"op": "whatif", "base": _num(measured["base"]),
                "edited": _num(measured["edited"]), "delta": _num(measured["delta"]),
                "exact": _num(effect.get("exact")), "emulated": _num(effect.get("emulated")),
                "floor": _num(effect.get("floor")), "verdict": effect.get("verdict"),
                "basis": effect.get("basis", ""), "record": result.get("record", "")}
    procedure = ((result.get("sealed_record") or {}).get("body") or {}).get("procedure")
    return {"source": address, "donor": donor, "origin": origin, "evidence": evidence,
            "consumer": {k: consumer[k] for k in ("address", "row", "kind", "indices")
                         if k in consumer},
            "where": result.get("where"), "procedure": procedure,
            "note": effect.get("note") or effect.get("reason")}


def _from_compare(ctx: Any, run: Any, result: dict[str, Any], side: str) -> dict[str, Any]:
    from .builtin import resolve_target
    from .predict import parse_assignment

    text = result["sides"][side]
    if not _assigns(text):
        raise Refused(f"compare side {side} is the checkpoint {text!r}, not a transplant")
    address, value, op, _ = parse_assignment(text)
    if (op or "replace") != "replace" or not isinstance(value, str):
        raise Refused(f"compare side {side} ({text}) is not a transplant: a symbol is declared "
                      "from a replace whose value is another run's (ORIGIN:ADDR) or address's")
    local = ctx if ctx.run is run else Context(ctx.debugger, run)
    found = resolve_target(local, address)
    verdict = result["verdicts"][side]
    evidence = {"op": "compare", "base": _num(result["measured"]["base"]["value"]),
                "edited": _num(result["measured"][side]["value"]),
                "delta": _num(result["effects"][side]), "floor": _num(verdict.get("floor")),
                "verdict": verdict.get("verdict"), "basis": verdict.get("basis", ""),
                "record": ""}
    from mdb.core.kinds import procedure

    return {"source": found.text, "donor": value, "origin": None, "evidence": evidence,
            "consumer": _consumer(result["consumer"]), "where": result.get("where"),
            "procedure": procedure(run), "note": None}


def _reference(ctx: Any, text: str) -> tuple[dict[str, Any], Any, Any, str]:
    """``(transplant, run, item, from)`` for a result reference."""
    from . import sets
    from .registry import dispatch

    match = _REFERENCE.fullmatch(text.strip())
    if match["each"]:
        try:
            found = dispatch(ctx, "promptset", {"action": "result", "name": match["each"],
                                                "arg": match["item"]})
            summary = dispatch(ctx, "promptset", {"action": "result",
                                                  "name": match["each"]})["summary"]
        except OpError as error:
            raise Refused(f"{text}: {error}") from None
        result = found["result"]
        if result.get("op") != "whatif":
            raise Refused(f"{text} is a {result.get('op')} result; a symbol is declared from an "
                          "item's whatif")
        known = sets.sets_of(ctx.debugger).get(summary["set"])
        if known is None or match["item"] not in {i.name for i in known.items}:
            raise Refused(f"{text}: set {summary['set']} is not loaded, so the item's roles are "
                          "unknown; load it again (set load)")
        bound = sets.Bound(ctx.debugger, known.item(match["item"]))
        run = (getattr(ctx.debugger, "runs", None) or {}).get(f"{summary['set']}-{match['item']}")
        if run is not None and _tokens(run) != bound.tokens:
            run = None
        return _from_whatif(result, text), run, bound, f"{text} (set {summary['set']})"
    entry = _LAST.get(ctx, {}).get("whatif" if match["kind"] == "whatif" else "compare")
    if entry is None:
        raise Refused(f"no {match['kind'].split('.')[0]} result in this session yet: run one "
                      "(whatif ADDR = ORIGIN:ADDR --consumer C), then symbol NAME = "
                      f"{match['kind']}")
    result, ref = entry
    run = ref()
    if match["kind"] == "whatif":
        return _from_whatif(result, "the last whatif"), run, None, "the last whatif"
    if run is None:
        raise Refused("the run of the last compare is gone; run compare again")
    side = match["side"].upper()
    return _from_compare(ctx, run, result, side), run, None, f"side {side} of the last compare"


def _row_rule(ctx: Any, tokens: list[int], row: int, role: str | None, item: Any
              ) -> dict[str, Any]:
    if item is not None:
        names = sorted(r for r, v in item.roles.items() if v == row and r != "last")
        where = {name: f"set item {item.item.name}" for name in names}
    else:
        where = _roles_at(ctx.debugger, tokens, row)
        names = sorted(where)
    if not names and row == len(tokens) - 1:
        names, where = ["last"], {"last": "the prompt's last row"}
    if role is not None:
        if names and role not in names:
            raise Refused(f"role={role}: row {row} is role {', '.join('@' + n for n in names)} "
                          f"of {where[names[0]]}, not @{role}")
        source = where.get(role, "supplied (role= on the declaration)")
        return {"rule": "role", "role": role, "role_source": source}
    if len(names) > 1:
        raise Refused(f"row {row} is roles {', '.join('@' + n for n in names)}: name one with "
                      "role=NAME")
    if names:
        return {"rule": "role", "role": names[0], "role_source": where[names[0]]}
    return {"rule": "absolute"}


def _roles_at(debugger: Any, tokens: list[int], row: int) -> dict[str, str]:
    """Role name -> ``set/item`` of the loaded set items with these tokens and a role at row."""
    from . import sets

    out: dict[str, str] = {}
    for name, found_set in sorted(sets.sets_of(debugger).items()):
        for item in found_set.items:
            roles = [r for r, v in item.roles.items() if v == row]
            if not roles:
                continue
            try:
                same = debugger.tokens(item.prompt) == tokens
            except Refused:
                continue
            if same:
                for role in roles:
                    out.setdefault(role, f"set item {name}/{item.name}")
    return out


def _readers(ctx: Any, run: Any, source: dict[str, Any], consumer: dict[str, Any], top: int,
             causal: bool) -> dict[str, Any]:
    from .registry import dispatch

    if source["reads"] is None:
        return {"state": "unresolved", "reason": source["why"]}
    if run is None:
        return {"state": "unresolved",
                "reason": "the run the result was measured on is gone (released or replaced); "
                          "keep it (each ... keep, or declare before moving on) to list readers"}
    here = int(run.where.split(".")[0][1:])
    row = consumer.get("row")
    if row is None or row > here:
        return {"state": "unresolved",
                "reason": f"the cursor {run.where} is before the consumer's row t{row}: reads "
                          f"lists the reads made there; run until t{row}.logits and declare "
                          "again"}
    local = ctx if ctx.run is run else Context(ctx.debugger, run)
    try:
        found = dispatch(local, "reads", {"source": source["reads"], "rows": str(row),
                                          "consumer": consumer["address"], "top": top,
                                          "causal": causal})
    except OpError as error:
        return {"state": "refused", "reason": f"reads {source['reads']}: {error}"}
    listed = []
    for read in found["reads"][:top]:
        entry: dict[str, Any] = {"name": read["name"], "row": read["row"],
                                 "layer": read["layer"], "mass": _num(read["mass"]),
                                 "effect": _num(read.get("to_consumer"))}
        check = read.get("causal") if causal else None
        if isinstance(check, dict) and "delta" in check:
            entry["causal"] = {"verdict": str(check.get("verdict", "unresolved")),
                               "delta": _num(check["delta"])}
        elif isinstance(check, dict):
            entry["causal"] = {"verdict": str(check.get("verdict") or check.get("state")
                                              or "unresolved"),
                               "reason": str(check.get("reason", ""))}
        listed.append(entry)
    omitted = found.get("omitted") or {}
    out: dict[str, Any] = {
        "state": "resolved", "source": source["reads"], "rows": [row],
        "consumer": consumer["address"], "at": run.where,
        "count": len(found["reads"]) + int(omitted.get("count", 0) or 0), "listed": listed,
        "authority": "as reads reports them: mass measured, effect derived-exact P19 (the read's "
                     "exact share of the consumer), causal verdict measured (the read removed "
                     "natively)"}
    if found.get("not_computed"):
        out["not_computed"] = list(found["not_computed"])[:8]
    return out


def declare(ctx: Any, name: str, reference: str, *, role: str | None = None, readers: int = 6,
            causal: bool = True) -> dict[str, Any]:
    """``symbol NAME = REFERENCE``: seal a ``symbol`` record from a transplant result."""
    from mdb.core.records import seal_record

    from .inspect import _edit_list, _symbol_store

    debugger = ctx.debugger
    if debugger is None:
        raise Refused("symbols need an attached model")
    check_name(ctx, name)
    if name in _symbol_store(ctx):
        raise Refused(f"{name} is a declared cell symbol (symbol {name} = ADDR...); delete it or "
                      "choose another name")
    transplant, run, item, came_from = _reference(ctx, reference)
    evidence = transplant["evidence"]
    if evidence["verdict"] != "effect":
        note = f" ({transplant['note']})" if transplant.get("note") else ""
        raise Refused(f"not declared: {came_from} has verdict {evidence['verdict']!r}{note}; a "
                      "symbol stands for a measured effect (verdict effect), so no record is made")
    source = _source(transplant["source"])
    tokens = item.tokens if item is not None else _tokens(run) if run is not None else None
    if tokens is None:
        raise Refused(f"the run of {came_from} is gone; its prompt is unknown")
    row = source["row"]
    rule = _row_rule(ctx, tokens, row, role, item)
    row_view: dict[str, Any] = {"row": row, **rule, "prompt_rows": len(tokens),
                                "prompt_sha256": _sha(tokens), "prefix_sha256": _sha(tokens[:row])}
    if row < len(tokens):
        row_view["token"] = tokens[row]
        if ctx.tokenizer is not None:
            row_view["text"] = ctx.tokenizer.decode([tokens[row]])
    writer = {"layer": source["layer"], "port": source["port"], "form": source["form"],
              "address": source["writer"]}
    body: dict[str, Any] = {"schema": SCHEMA, "name": name}
    body["row"] = row_view
    template = (f"@{rule['role']}" if rule["rule"] == "role" else row)
    row_view["address"] = address_at({"role": {"writer": writer}}, template)
    body["role"] = {
        "writer": writer,
        "meaning": "the writer (port and selector at its layer, or the cache cell) without its "
                   "row; the readers are the reads of the source at the consumer's row",
        "readers": _readers(ctx, run, source, transplant["consumer"], readers, causal),
    }
    body["consumer"] = transplant["consumer"]
    body["evidence"] = {**evidence, "edit": f"{source['text']} = {transplant['donor']}",
                        "authority": "base, edited and delta measured natively; the verdict is "
                                     "the result's own (P20)"}
    origin: dict[str, Any] = {"from": came_from, "where": transplant.get("where") or ""}
    if isinstance(transplant["origin"], dict):
        donor = transplant["origin"]
        origin["donor"] = {k: donor[k] for k in ("origin", "via", "address", "sha256", "cut",
                                                 "recipe", "where") if donor.get(k) is not None}
    else:
        origin["donor"] = {"address": str(transplant["donor"])}
    if run is not None:
        origin["run"] = getattr(run, "name", None) or ""
        edits = _edit_list(run)
        origin["edits"] = {k: v[:MAX_EDITS] for k, v in edits.items()}
    if transplant.get("procedure"):
        body["procedure"] = transplant["procedure"]
    body["origin"] = origin
    adapter = run.adapter if run is not None else debugger.adapter
    session = getattr(debugger, "session_id", None)
    project = getattr(debugger, "project", None)
    writable = project is not None and not project.read_only and session is not None
    try:
        record = seal_record(session_id=session if writable else f"mdb-symbol:{_sha(tokens)[:32]}",
                             kind="symbol", model_identity=adapter.model_identity,
                             execution=dict(adapter.execution), authority="measured", body=body)
    except RecordError as error:
        raise Refused(f"symbol {name}: the record exceeds the record bound: {error}") from None
    store(debugger)[name] = record
    out: dict[str, Any] = {"action": "declare", "symbol": name, "record": record["id"],
                           "body": body}
    if writable:
        from mdb.core import index

        out["stored"] = [index.put_record(project, record)]
    return out


def show(ctx: Any, name: str) -> dict[str, Any]:
    from .predict import record_view

    debugger = ctx.debugger
    named = store(debugger) if debugger is not None else {}
    if name not in named:
        raise Refused(f"no named symbol {name!r} (have: {', '.join(sorted(named)) or 'none'})")
    record = named[name]
    return {"action": "show", "symbol": name, "record": record_view(record),
            "here": here(ctx, record)}


def here(ctx: Any, record: dict[str, Any]) -> dict[str, Any]:
    """The symbol's binding on the current run: its address and how, or the refusal."""
    if getattr(ctx, "run", None) is None:
        return {"state": "refused", "reason": "no live run"}
    try:
        row, how, _ = _bind_run(ctx, record, None)
    except Refused as error:
        return {"state": "refused", "reason": str(error)}
    return {"address": address_at(record["body"], row), "row": row, "how": how}


def delete(ctx: Any, name: str) -> dict[str, Any]:
    named = store(ctx.debugger) if ctx.debugger is not None else {}
    if name not in named:
        raise Refused(f"no named symbol {name!r} (have: {', '.join(sorted(named)) or 'none'})")
    record = named.pop(name)
    return {"action": "delete", "symbol": name, "record": record["id"],
            "note": "removed from this session; a record stored in a project stays there"}


def listing(ctx: Any) -> list[dict[str, Any]]:
    """One line per named symbol for ``symbols``: source, evidence and its binding here."""
    debugger = getattr(ctx, "debugger", None)
    out = []
    for name, record in sorted(store(debugger).items() if debugger is not None else ()):
        body = record["body"]
        out.append({"symbol": name, "source": body["row"]["address"],
                    "consumer": body["consumer"]["address"],
                    "delta": body["evidence"]["delta"], "verdict": body["evidence"]["verdict"],
                    "record": record["id"], "here": here(ctx, record)})
    return out[:16]


def parse_args(rest: str) -> dict[str, Any]:
    """The words after ``symbol NAME = REFERENCE``: ``role=R readers=K causal=false``."""
    out: dict[str, Any] = {}
    for word in rest.split():
        key, equals, value = word.partition("=")
        if not equals or key not in ("role", "readers", "causal"):
            raise BadArgs("write symbol NAME = whatif|compare.a|compare.b|EACH:ITEM [role=R] "
                          "[readers=K] [causal=false]")
        if key == "readers":
            if not value.isdigit():
                raise BadArgs("readers must be an integer")
            out[key] = int(value)
        elif key == "causal":
            if value.lower() not in ("true", "false", "on", "off", "1", "0"):
                raise BadArgs("causal must be true or false")
            out[key] = value.lower() in ("true", "on", "1")
        else:
            out[key] = value
    return out


__all__ = ["ADDRESS_ARGS", "MAX_READERS", "SCHEMA", "address_at", "bind", "declare", "delete",
           "expand", "expand_for_item", "is_reference", "listing", "show", "store"]
