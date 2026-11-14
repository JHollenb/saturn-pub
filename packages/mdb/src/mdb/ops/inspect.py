"""The inspection verbs: ``reads``, ``symbols``, ``symbol``, ``circuit``, ``compare`` and
``seek``.

They replace the A6 slots of ``builtin`` (``register(..., replace=True)``). Import is
torch-free; each op imports the analysis layer when it runs.

- ``reads SOURCE``: the attention reads of a source row (``t2``, ``t2.L0.v``,
  ``cache.L1.v[row=2]``) at the cursor's row (``rows=`` for later rows), with mass, exact
  contribution and a native causal verdict (``analysis.reads``).
- ``symbols``: the symbol table at the cursor (``analysis.symbols``); ``symbol NAME =
  ADDR[,ADDR...]`` declares a user symbol over cache rows; ``symbol NAME = whatif`` (or
  ``compare.a|b``, ``EACH:ITEM``) a named symbol from a transplant result (``ops.named``),
  listed under ``named`` by ``symbols``, shown by ``symbol show NAME``.
- ``circuit CONSUMER [top=K] [depth=D]`` builds a circuit record; ``circuit test ID ref=NAME``
  patches it natively from a reference run (``ref=set:NAME``: a prompt set's per-component
  mean) and seals a ``circuit-test`` record; ``circuit show ID`` prints it as a walk
  (``analysis.circuits``, ``analysis.circuit_test``).
- ``compare A B [consumer=C]``: base, A, B and A+B measured natively, effects and the
  interaction (``analysis.compare``). A side is an edit ``ADDR=VALUE [op]`` or a checkpoint
  name (in memory, else durable).
- ``seek ADDR``: move the session to an occurrence, forward (``until``), backward
  (``reverse-step``) or, with a project, by replaying a stored recipe (``ORIGIN:ADDR`` with a
  checkpoint name or session id as origin, or an occurrence pointer as JSON).

With a project attached, ``symbols`` and ``circuit`` index their sealed records in it.
Declared symbols belong to the debugger: they are bound again on the run the session moves to
(``start``, a recipe ``seek``) once their rows are computed, and listed ``unresolved`` until
then; the circuit ledger moves with a recipe ``seek``.
"""

from __future__ import annotations

import json
import re
import weakref
from collections import Counter
from typing import Any

from . import named
from .builtin import GLOBALS, moved, project_of
from .errors import BadArgs, Refused
from .predict import build_edit, parse_assignment, parse_consumer, record_view, session_digest
from .registry import Arg, Op, register

_ROW = re.compile(r"t(?P<row>\d+)\.")
_OPS = ("replace", "add", "scale", "mix")
_SYMBOLS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _refusal():
    from mdb.analysis.forecast import Refusal

    return Refusal


def _declared_consumer(ctx, text: str | None):
    return None if text is None else parse_consumer(ctx, text)


def _session_id(ctx) -> str | None:
    project = project_of(ctx)
    return None if project is None or project.read_only else ctx.debugger.session_id


def _store(ctx, records: list[dict[str, Any]]) -> list[str]:
    """Index sealed records in the attached project; the ids stored."""
    if _session_id(ctx) is None:
        return []
    from mdb.core import index

    return [index.put_record(ctx.debugger.project, record) for record in records]


def _unchanged(ctx, before: str, verb: str) -> None:
    if session_digest(ctx) != before:
        raise RuntimeError(f"{verb} changed the debugger session")


# --------------------------------------------------------------------------- reads


def _reads(ctx, source, rows, consumer, top, causal, min_effect):
    from mdb.analysis import reads as reads_mod

    before = session_digest(ctx)
    try:
        result = reads_mod.reads(ctx.run, source, rows=rows,
                                 consumer=_declared_consumer(ctx, consumer), top=top,
                                 causal=causal, min_effect=float(min_effect))
    except _refusal() as error:
        raise Refused(str(error)) from None
    _unchanged(ctx, before, "reads")
    report = result.report
    tokenizer = ctx.tokenizer
    if tokenizer is not None and "token" in report["source"]:
        report["source"]["text"] = tokenizer.decode([report["source"]["token"]])
    return {**report, "where": ctx.run.where}


# --------------------------------------------------------------------------- symbols


def _symbol_store(ctx) -> dict[str, list[str]]:
    """The symbols declared through this debugger (name -> addresses), whatever its run."""
    if ctx.debugger not in _SYMBOLS:
        _SYMBOLS[ctx.debugger] = {}
    return _SYMBOLS[ctx.debugger]


def _bind_declared(ctx) -> dict[str, dict[str, Any]]:
    """Declare on the current run every symbol of the debugger it does not hold yet; those that
    cannot be bound (rows not computed) are returned with the reason."""
    from mdb.analysis import symbols

    held = symbols.declared(ctx.run)
    pending = {}
    for name, addresses in _symbol_store(ctx).items():
        if name in held and held[name].addresses == addresses:
            continue
        try:
            symbols.declare(ctx.run, name, addresses)
        except _refusal() as error:
            pending[name] = {"addresses": addresses, "reason": str(error)}
    return pending


def _symbols(ctx, consumer, rows, min_mass, readers, offset, limit):
    from mdb.analysis import symbols

    pending = _bind_declared(ctx)
    try:
        found = symbols.table(ctx.run, consumer=_declared_consumer(ctx, consumer), rows=rows,
                              min_mass=float(min_mass), readers=readers, offset=offset,
                              limit=limit, session_id=_session_id(ctx),
                              tokenizer=ctx.tokenizer, pending=pending)
    except _refusal() as error:
        raise Refused(str(error)) from None
    stored = _store(ctx, found.records)
    report = dict(found.report)
    if stored:
        report["stored"] = stored
    listed = named.listing(ctx)
    if listed:
        report["named"] = listed     # transplant symbols (M23); not part of the table record
    return report


def _symbol(ctx, name, addresses=None, result=None, action="declare", role=None, readers=6,
            causal=True):
    from mdb.analysis import symbols

    if action == "show":
        return named.show(ctx, name)
    if action == "delete":
        return named.delete(ctx, name)
    if (addresses is None) == (result is None):
        raise BadArgs("write symbol NAME = ADDR[,ADDR...] (cache cells) or symbol NAME = "
                      "whatif|compare.a|compare.b|EACH:ITEM (a transplant result)")
    if ctx.run is None and result is None:
        raise Refused("symbol needs a live run; start one with `start`")
    if result is not None:
        return named.declare(ctx, name, result, role=role, readers=readers, causal=causal)
    if name in named.store(ctx.debugger):
        raise Refused(f"{name} is a named symbol (symbol show {name}); delete it or choose "
                      "another name")
    parts = [part.strip() for part in addresses.split(",") if part.strip()]
    _bind_declared(ctx)
    try:
        binding = symbols.declare(ctx.run, name, parts)
    except _refusal() as error:
        raise Refused(str(error)) from None
    _symbol_store(ctx)[name] = binding.addresses
    return {"symbol": binding.name, "definitions": binding.addresses,
            "rows": sorted({cell.row for cell in binding.cells}),
            "declared": sorted(_symbol_store(ctx))}


# --------------------------------------------------------------------------- circuit


CIRCUIT_TEST_NEEDS_REF = (
    "circuit test needs a reference: circuit test ID ref=NAME, where NAME is another run of "
    "this debugger (e.g. the opposite-answer prompt: `run name=NAME \"...\"` then `until "
    "logits`), or ref=set:NAME for the per-component mean over a prompt set. Zero-ablation is "
    "not offered: zeroing components is off-distribution, and on Qwen2.5-1.5B zeroing the "
    "complement moved the margin by 27.75 against a circuit effect of 1.81, so it said nothing "
    "(issue M4)")


def _page_store(ctx, write: bool = True):
    """The project store a circuit's child pages go to (``write``) or are read from, or None
    (a read-only project is still read)."""
    project = getattr(ctx.debugger, "project", None)
    if project is None or (write and project.read_only):
        return None
    return project.store


def _reference(ctx, ref: str):
    """``(run, name, prompts)`` for ``ref=``: a named run of the debugger, or ``set:NAME`` (the
    set's items as ``(item, token ids)``)."""
    debugger = ctx.debugger
    if debugger is None:
        raise Refused("circuit test ref= names a run of the debugger; there is none here")
    if ref.startswith("set:"):
        from . import sets

        name = ref[4:]
        found = sets.sets_of(debugger).get(name)
        if found is None:
            raise Refused(f"no prompt set {name!r} (have: "
                          f"{', '.join(sets.sets_of(debugger)) or 'none'}; promptset load NAME "
                          "FILE)")
        prompts = []
        for item in found.items:
            try:
                prompts.append((item.name, debugger.tokens(item.prompt)))
            except Refused as error:
                raise Refused(f"set {name} item {item.name}: {error}") from None
        return None, name, prompts
    runs = getattr(debugger, "runs", None) or {}
    if ref not in runs:
        raise Refused(f"no run {ref!r} to take reference values from (runs: "
                      f"{', '.join(runs) or 'none'}; or ref=set:NAME)")
    if runs[ref] is ctx.run:
        raise Refused(f"ref={ref} is the current run: name another run (info runs)")
    return runs[ref], ref, None


def _circuit(ctx, target, id, top, depth, ref=None, node=None):  # noqa: A002 - the wire name
    from mdb.analysis import circuits

    run = ctx.run
    ledger = circuits.ledger(run)
    if target in ("test", "show"):
        if id is None:
            raise BadArgs(f"write circuit {target} ID")
        if id not in ledger:
            raise Refused(f"no circuit {id} (have: {', '.join(map(str, ledger)) or 'none'})")
        circuit = ledger[id]
        if target == "show" and node is not None:
            try:
                found = circuits.page(circuit, node, store=_page_store(ctx, False))
            except _refusal() as error:
                raise Refused(str(error)) from None
            out = {"action": "show", "id": id, "node": node, "page": found,
                   "record": {"id": circuit.record["id"]}}
            if getattr(ctx.debugger, "carried_circuits", {}).get(id) is circuit:
                out["from_previous_run"] = True
            else:
                texts = _source_texts(ctx, found["node"])
                if texts:
                    out["token_text"] = texts
            return out
        if node is not None:
            raise BadArgs("node= belongs to circuit show ID")
        if ref is not None and target == "show":
            raise BadArgs("ref= belongs to circuit test ID")
        if target == "show":
            out = {"action": "show", "id": id, "record": record_view(circuit.record),
                   "tests": circuit.tests[-1:]}
            if circuit.tests:
                nodes = circuit.tests[-1].get("nodes", {})
                # patched from the reference: the circuit's nodes in the knock-out, the
                # `+N more` groups in the keep-only
                out["ablated"] = {"retained": nodes.get("circuit", {}).get("refs", []),
                                  "complement": nodes.get("group", {}).get("refs", [])}
            if getattr(ctx.debugger, "carried_circuits", {}).get(id) is circuit:
                out["from_previous_run"] = True   # built by an earlier run: its tokens are not
                return out                         # this run's, so no token text is attached
            texts = _source_texts(ctx, circuit.record["body"]["root"])
            if texts:
                out["token_text"] = texts
            return out
        if ref is None:
            raise Refused(CIRCUIT_TEST_NEEDS_REF)
        if getattr(ctx.debugger, "carried_circuits", {}).get(id) is circuit:
            raise Refused(f"circuit {id} was built by an earlier run: its nodes are that run's "
                          "rows; build it again on this run to test it")
        other, name, prompts = _reference(ctx, ref)
        before = session_digest(ctx)
        try:
            found = circuits.test(run, circuit, other, name=name, prompts=prompts,
                                  session_id=_session_id(ctx), store=_page_store(ctx))
        except _refusal() as error:
            raise Refused(str(error)) from None
        _unchanged(ctx, before, "circuit test")
        out = {"action": "test", "id": id, "test": found["result"], "where": run.where}
        stored = _store(ctx, [found["record"]])
        if stored:
            out["stored"] = stored
        return out
    if ref is not None:
        raise BadArgs("ref= belongs to circuit test ID")
    if id is not None:
        raise BadArgs("circuit CONSUMER takes top= and depth=, not an id")
    try:
        circuit = circuits.build(run, parse_consumer(ctx, target), top=top, depth=depth,
                                 session_id=_session_id(ctx), store=_page_store(ctx))
    except _refusal() as error:
        raise Refused(str(error)) from None
    out = {"action": "build", "id": circuit.id, "record": record_view(circuit.record)}
    stored = _store(ctx, [circuit.record])
    if stored:
        out["stored"] = stored
    return out


def _source_texts(ctx, root: dict[str, Any]) -> dict[str, str]:
    """Token text of each source-row leaf (node id -> text), when a tokenizer is attached."""
    tokenizer, tokens = ctx.tokenizer, ctx.run.tokens
    if tokenizer is None:
        return {}
    out = {}
    stack = [root]
    while stack and len(out) < 64:
        node = stack.pop()
        stack += node.get("children", [])
        if node["kind"] == "source":
            row = int(node["name"][1:])
            if row < len(tokens):
                out[node["id"]] = tokenizer.decode([int(tokens[row])])
    return out


# --------------------------------------------------------------------------- compare


def _side(ctx, text: str):
    """An edit ``ADDR=VALUE [op]`` (``predict.parse_assignment``: also ``ADDR += X`` and the
    other sugar, ``mix DOSE``, a state cell target and an ``ORIGIN:ADDR`` value) or a checkpoint
    name (in memory first, then durable)."""
    from mdb.analysis.compare import Side

    words = text.split()
    if _assigns(text):
        usage = "ADDR=VALUE [replace|add|scale|mix DOSE] or ADDR += X"
        address, value, op, dose = parse_assignment(text, usage=usage)
        if (op or "replace") not in _OPS:
            raise BadArgs(f"an edit side is {usage}, not {text!r}")
        return Side(text, edits=[build_edit(ctx, address, value, op or "replace", dose)])
    if len(words) != 1:
        raise BadArgs(f"a side is an edit ADDR=VALUE [op] or a checkpoint name, not {text!r}")
    name = words[0]
    if name in ctx.checkpoints:
        return Side(name, checkpoint=ctx.checkpoints[name])
    project = project_of(ctx)
    if project is not None:
        from mdb.core import recipe as recipes

        found = recipes.find_checkpoint(project, name, session_id=ctx.debugger.session_id)
        if found is not None:
            try:
                other = ctx.debugger.replay(recipes.load(project, found["body"]["recipe"]))
            except (recipes.RecipeError, ValueError) as error:
                raise Refused(f"checkpoint {name!r} cannot be replayed here: {error}") from None
            return Side(f"{name} (durable)", run=other)
    known = ", ".join(ctx.checkpoints) or "none"
    raise Refused(f"no checkpoint {name!r} (have: {known}) and it is not an edit ADDR=VALUE")


def _assigns(text: str) -> bool:
    """Whether a compare side is an assignment: an ``=`` outside brackets."""
    depth = 0
    for char in text:
        depth += char == "["
        depth -= char == "]"
        if char == "=" and depth == 0:
            return True
    return False


def _compare(ctx, a, b, consumer):
    from mdb.analysis import compare

    run = ctx.run
    if consumer is None:
        consumer = f"t{_ROW.match(run.where)['row']}.logits"
    chosen = parse_consumer(ctx, consumer)
    before = session_digest(ctx)
    try:
        result = compare.compare(run, _side(ctx, a), _side(ctx, b), chosen)
    except _refusal() as error:
        raise Refused(str(error)) from None
    after = session_digest(ctx)
    if after != before:
        raise RuntimeError("compare changed the debugger session")
    report = result["report"]
    report["session"] = {**report["session"], "debugger_before": before,
                         "debugger_after": after}
    return {**report, "consumer": getattr(chosen, "text", chosen), "where": run.where}


# --------------------------------------------------------------------------- seek


def _seek_within(run, address: str) -> list[str]:
    """Move ``run`` to ``address``: forward with ``until``, backward by layer boundaries. All
    or nothing: a target before the run's first stepped row is refused before any move, and a
    failure part way restores the starting cursor."""
    moves = []
    key = run.parse(address)
    root = run.parse(run.root)
    if key < root:
        if key[0] < run.prefill:
            raise ValueError(run.prefilled_reason(key[0]))
        raise ValueError(f"{address} is before this run's root {run.root}")
    start = run.checkpoint()
    try:
        if key < run.parse(run.where):
            while run.parse(run.where) > key:
                run.reverse_step("layer")
            moves.append(f"reverse-step layer to {run.where}")
        if key > run.parse(run.where):
            run.until(address)
            moves.append(f"until {address}")
    except BaseException:
        from mdb.analysis.compare import restore_exactly

        restore_exactly(run, start)
        raise
    return moves


def _edit_list(run) -> dict[str, list[str]]:
    """The run's committed and staged edits as text (``t<r>.L<l>.<port> <op> <value>``)."""
    def text(row, entry):
        value = entry["value"]
        shown = value.get("value") if value.get("kind") == "scalar" else value.get("sha256",
                                                                                   "")[:12]
        sel = f"[{entry['selection']}]" if entry.get("selection") else ""
        return f"t{row}.L{entry['layer']}.{entry['port']}{sel} {entry['op']} {shown}"

    row = int(_ROW.match(run.where)["row"])
    return {"committed": [text(e["row"], e) for e in run.committed_edits()],
            "staged": [text(row, e) for e in run.staged_info()]}


def _dropped(before: dict[str, list[str]], after: dict[str, list[str]]) -> dict[str, Any]:
    """The edits a move dropped and the committed edits it turned back into staged ones."""
    now = Counter(after["committed"]) + Counter(after["staged"])
    dropped = list((Counter(before["committed"]) + Counter(before["staged"]) - now).elements())
    restaged = list((Counter(before["committed"]) & Counter(after["staged"])).elements())
    return {"dropped_edits": dropped[:16], "restaged_edits": restaged[:16]}


def _replayed(ctx, recipe_id: str):
    from mdb.core import recipe as recipes

    try:
        return ctx.debugger.replay(recipes.load(ctx.debugger.project, recipe_id))
    except (recipes.RecipeError, ValueError) as error:
        raise Refused(f"the stored recipe {recipe_id[:12]} cannot be replayed here: "
                      f"{error}") from None


def _origin_recipe(ctx, branch: str, address: str) -> str:
    """The recipe standing for origin ``branch``: a durable checkpoint of that name, or the
    recipe the occurrence index (or the session cursor) of session ``branch`` holds."""
    from mdb.core import index
    from mdb.core import recipe as recipes

    project = ctx.debugger.project
    named = recipes.find_checkpoint(project, branch, session_id=ctx.debugger.session_id)
    if named is not None:
        return named["body"]["recipe"]
    if branch in {r["session_id"] for r in project.sessions()}:
        # exported occurrences are indexed under their origin-qualified address
        row = index.find_occurrence(project, branch, f"{branch}:{address}",
                                    layers=ctx.run.adapter.layers,
                                    ports=ctx.run.program.names, global_ports=GLOBALS)
        if row is not None and row["recipe_id"]:
            return row["recipe_id"]
        cursor = index.get_cursor(project, branch)
        if cursor is not None:
            return index.get_record(project, branch, cursor)["body"]["recipe"]
    known = ", ".join(getattr(ctx.debugger, "runs", None) or ()) or "none"
    raise Refused(f"origin {branch}: names no run (have: {known}), no durable checkpoint and no "
                  "session with a stored recipe in this project")


def _seek_named(ctx, name: str, plain: str):
    """``seek NAME:ADDR``: make the run named ``NAME`` current (as ``frame NAME:``), then move it
    to ``plain``. All or nothing: a failed move puts the previous run back as the current one."""
    from .builtin import _switch

    previous, target = ctx.run, ctx.debugger.runs[name]
    if target is not previous:
        _switch(ctx, target)
    before = _edit_list(target)
    try:
        moves = _seek_within(target, plain)
    except ValueError as error:
        if target is not previous:
            _switch(ctx, previous)
        raise Refused(f"cannot reach {name}:{plain}: {error}") from None
    return moved(ctx, {"target": plain, "moves": moves, "via": "run", "run": name,
                       **_dropped(before, _edit_list(target))})


def _seek(ctx, target):
    from mdb.core import address as addresses
    from mdb.core.pointers import RecordError, validate_occurrence_pointer

    run = ctx.run
    text = target.strip()
    moves: list[str] = []
    origin = None
    if text.startswith("{"):
        try:
            pointer = validate_occurrence_pointer(json.loads(text), ports=run.program.names,
                                                  global_ports=GLOBALS)
        except (ValueError, RecordError) as error:
            raise BadArgs(f"not an occurrence pointer: {error}") from None
        parsed = addresses.from_json(pointer["address"], ports=run.program.names,
                                         global_ports=GLOBALS)
        recipe_id = pointer["recipe"]
        origin = "pointer"
    else:
        head = text.partition("[")[0]
        relative = (":" not in head and not head.startswith(("@", "cache."))
                    and not _ROW.match(head))
        # `L20.p` and `logits` mean the cursor's row, as they do for `until` and `print`
        full = f"t{_ROW.match(run.where)['row']}.{text}" if relative else text
        try:
            parsed = addresses.parse(full, ports=run.program.names, global_ports=GLOBALS)
        except addresses.AddressError as error:
            raise BadArgs(f"bad address {text!r}: {error}") from None
        recipe_id = None
        if parsed.origin is not None:
            origin = parsed.origin.branch or "@cut"
    if not isinstance(parsed, addresses.Occurrence) or parsed.selector != addresses.Selector():
        raise Refused("seek moves to an occurrence t<row>.<stop> (no selector, no cache cell)")
    plain = addresses.format_address(addresses.Occurrence(parsed.token, parsed.port,
                                                          parsed.layer))
    runs = getattr(ctx.debugger, "runs", None) or {}
    named = origin if origin in runs else None  # a run started with name=NAME: switch to it
    if named is not None:
        return _seek_named(ctx, named, plain)
    own = origin is None or origin == getattr(ctx.debugger, "session_id", None)
    before = _edit_list(run)
    if own and recipe_id is None:
        try:
            moves = _seek_within(run, plain)
        except ValueError as error:
            if project_of(ctx) is None:
                raise Refused(f"cannot reach {plain}: {error}; an occurrence outside this run "
                              "is reachable only through a stored recipe (attach with project= "
                              "and checkpoint)") from None
            recipe_id = _origin_recipe(ctx, ctx.debugger.session_id, plain)
        else:
            return moved(ctx, {"target": plain, "moves": moves, "via": "run",
                                **_dropped(before, _edit_list(run))})
    if project_of(ctx) is None:
        raise Refused(f"{text}: no run named {origin!r} (have: {', '.join(runs) or 'none'}); "
                      "reaching another origin needs a stored recipe; attach with project=")
    if origin == "@cut":
        raise Refused("an @cut origin names no recipe; use a checkpoint name or a pointer")
    if recipe_id is None:
        recipe_id = _origin_recipe(ctx, origin, plain)
    replayed = _replayed(ctx, recipe_id)
    moves.append(f"replay recipe {recipe_id[:12]} to {replayed.where}")
    try:
        moves += _seek_within(replayed, plain)
    except ValueError as error:
        raise Refused(f"the stored recipe reaches {replayed.where}, but not {plain}: "
                      f"{error}") from None
    from mdb.analysis import circuits

    circuits.ledger(replayed).update(circuits.ledger(run))
    ctx.set_run(replayed)
    ctx.debugger.live = replayed
    _bind_declared(ctx)
    return moved(ctx, {"target": plain, "moves": moves, "via": "recipe",
                        "recipe": recipe_id, **_dropped(before, _edit_list(replayed))})


# --------------------------------------------------------------------------- registration

_CONSUMER = Arg("consumer", "str", positional=False,
                help="M[a,b], logits[i] or a residual port / o address")
register(Op("reads", "Who reads a value: attention reads of a source row with mass, exact "
            "contribution and a native causal verdict",
            (Arg("source", "str", required=True,
                 help="t<r>, t<r>.L<l>.k|v or cache.L<l>.k|v[row=<r>]"),
             Arg("rows", "str", positional=False, help="reader rows: R, A:B or all"),
             _CONSUMER,
             Arg("top", "int", default=8, minimum=1, maximum=32, positional=False,
                 help="reads listed with a verdict; the rest are summed"),
             Arg("causal", "bool", default=True, positional=False,
                 help="measure each listed read's verdict natively"),
             Arg("min_effect", "number", default=0.0, positional=False,
                 help="a measured change above this is causal")),
            "navigate", True, _reads,
            usage="reads SOURCE [rows=R|A:B|all] [consumer=C] [top=K] [causal=false]"),
         replace=True)
register(Op("symbols", "Symbol table at the cursor: bindings with definition sites, lifetimes, "
            "readers, status and an Unbound row (stored as records with a project)",
            (_CONSUMER, Arg("rows", "str", positional=False, help="reader rows: R, A:B or all"),
             Arg("min_mass", "number", default=0.1, positional=False,
                 help="a read above this mass marks the lifetime"),
             Arg("readers", "int", default=3, minimum=1, maximum=8, positional=False),
             Arg("offset", "int", default=0, minimum=0, positional=False),
             Arg("limit", "int", default=16, minimum=1, maximum=16, positional=False)),
            "mutate", True, _symbols,
            usage="symbols [consumer=C] [rows=R|A:B|all] [min_mass=X] [offset=N] [limit=N]"),
         replace=True)
register(Op("symbol", "Declare a user symbol over cache rows (symbol NAME = ADDR[,ADDR...]), or a "
            "named symbol from a transplant result (symbol NAME = whatif|compare.a|compare.b|"
            "EACH:ITEM): its role, source row, readers and measured effect, usable as an "
            "address; symbol show|delete NAME",
            (Arg("name", "str", required=True),
             Arg("addresses", "str", positional=False, help="cache cells: ADDR[,ADDR...]"),
             Arg("result", "str", positional=False,
                 help="whatif, compare.a, compare.b (the last of this session) or EACH:ITEM "
                      "(an each result's item)"),
             Arg("action", "choice", default="declare", choices=("declare", "show", "delete"),
                 positional=False),
             Arg("role", "str", positional=False,
                 help="the source row's role when several (or none) name it"),
             Arg("readers", "int", default=6, minimum=1, maximum=named.MAX_READERS,
                 positional=False, help="readers listed from reads (top K)"),
             Arg("causal", "bool", default=True, positional=False,
                 help="measure each listed reader's causal verdict natively")),
            "mutate", False, _symbol,
            usage="symbol NAME = ADDR[,ADDR...] | symbol NAME = whatif|compare.a|compare.b|"
                  "EACH:ITEM [role=R] [readers=K] [causal=false] | symbol show|delete NAME"),
         replace=True)
register(Op("circuit", "Circuit records: build from the exact decomposition (nodes, edges, "
            "remainder), test by native patching from a reference run, show as a walk",
            (Arg("target", "str", required=True, help="a consumer, or test / show"),
             Arg("id", "int", minimum=1),
             Arg("top", "int", default=4, minimum=1, maximum=8, positional=False),
             Arg("depth", "int", default=2, minimum=1, maximum=3, positional=False),
             Arg("node", "str", positional=False,
                 help="circuit show ID: the node whose page to read (n12), for a group "
                      "the record stores as a child page"),
             Arg("ref", "str", positional=False,
                 help="circuit test ID: the reference run whose values are patched in (a run "
                      "name, or set:NAME for a prompt set's per-component mean)")),
            "mutate", True, _circuit,
            usage="circuit CONSUMER [top=K] [depth=D] | circuit test ID ref=NAME|set:NAME | "
                  "circuit show ID [node=NODEID]"),
         replace=True)
register(Op("compare", "Compare branches: base, A, B and A+B measured natively, effects and "
            "the interaction",
            (Arg("a", "str", required=True, help="ADDR=VALUE [op] or a checkpoint name"),
             Arg("b", "str", required=True, help="ADDR=VALUE [op] or a checkpoint name"),
             _CONSUMER),
            "navigate", True, _compare,
            usage="compare A B [consumer=C]  (A, B: ADDR=VALUE [op] or a checkpoint name)"),
         replace=True)
register(Op("seek", "Move the session to an occurrence: forward, backward, or by replaying a "
            "stored recipe (ORIGIN:ADDR or an occurrence pointer)",
            (Arg("target", "str", required=True),), "navigate", True, _seek,
            usage="seek ADDR | seek ORIGIN:ADDR | seek POINTER-JSON"), replace=True)
