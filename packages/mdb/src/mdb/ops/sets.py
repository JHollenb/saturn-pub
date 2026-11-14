"""Prompt sets and ``each``: one op line run on every prompt of a set, plus a summary.

The engine stays batch 1: ``each`` makes one run per prompt (``start`` then ``until AT``
through the registry, exactly what a user would type), runs the op line on it through the
same ``dispatch`` a single-prompt call uses, and releases the run (``keep`` holds them as
named runs ``SET-ITEM``). Runs are made one after the other, not held together.

A prompt set is a list of items, from a JSONL file (``promptset load NAME FILE``, the REPL's
``set load NAME FILE``) or given inline (``define``):

    {"name": "paris", "tokens": [1, 5, 9, 3], "roles": {"subject": 1},
     "vars": {"answer": 4, "foil": 9}, "pair": "rome"}

``prompt`` (text, needs a tokenizer) or ``tokens``; ``roles`` name rows (``last`` is always the
last row); ``vars`` name tokens (an id, or a text that is one token); ``pair`` names another
item of the set. An op line addresses them as a template, bound per item before dispatch, so
every per-item result holds ordinary canonical addresses:

* ``t@ROLE`` -> ``t<row>`` (``t@subject.L1.A[h1]``, ``t@last.logits``); ``=@ROLE`` -> ``=<row>``
  (``[row=@subject]``, ``rows=@subject``); an argument that is exactly ``@ROLE`` -> ``<row>``;
* ``$NAME`` -> the token id (``M[$answer,$foil]``);
* ``@pair:ADDR`` -> ``pair-SET-PAIR:ADDR``: the value at ADDR (roles bound to the paired item)
  of the paired item's run, a clean run of that prompt at the same stop (``AT``), started once
  per ``each`` and released with the item runs. It is the cross-run value of ``set``/``whatif``
  /``compare`` (``ORIGIN:ADDR``), so it reaches what those reach: a port of the run's last row
  or a cache cell. An address with no row (``L20.A[h5]``) means the paired item's last row,
  ``pair-SET-PAIR:t<its last>.L20.A[h5]``, whatever length the two prompts are; the item row's
  ``donor`` lists the donor addresses used.

``prefill=K`` or ``prefill=@ROLE`` (M54) starts each item's run, and its pair's clean run,
under that split, bound per item (``@ROLE``: that item's own role row), so the rows from it on
are stepped with ports (``sweep``, cross-row ``set``/``whatif`` at earlier rows); the item's row
in the summary holds the bound value (``prefill``).

An item whose template does not bind (an undefined role or variable, a role past its prompt,
no pair, a split past its prompt) or whose op is refused is a per-item refusal with its
reason; the other items run.

Output: the ``each`` op returns ONE summary record (bounded: per-item rows hold key numbers, the
item result's digest and a pointer to it, never the result itself); the per-item results are the
ops' ordinary results, kept in memory under the summary id (``promptset result ID ITEM``). With
a writable project attached each result is also a sha256-checked JSON blob in the project store
(``core.bounded.put_json``; the row's ``blob`` is its pointer) and the summary is stored as a
``prompt-set`` record, so ``promptset result RECORD ITEM`` (RECORD = the summary's ``stored``
id) reads the result back in a new process, verifying the blob; without a project the row says
the result is held in memory only.
Statistics: ``mdb.analysis.summary``.

State: the caller's context (run, cursor, focus, tables, checkpoints) and the debugger's live
run, run table, circuit carry and declared symbols are what they were before; each item runs
in a fresh context (no breakpoints, displays, checkpoints or focus of the caller's).
"""

from __future__ import annotations

import json
import re
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mdb.core.pointers import RecordError, check_bounded, seal_digest

from . import lines
from .context import Context
from .errors import BadArgs, OpError, Refused
from .registry import Arg, Op, dispatch, register, registry, validate

MAX_ITEMS = 64          # one summary row per item: a record list holds at most 64 items
MAX_NUMERIC = 40        # statistics listed in the summary record (the rest are counted)
MAX_VERDICTS = 16
MAX_RESULTS = 16        # summaries (with their item results) kept in memory per debugger
DEFAULT_AT = "t@last.logits"
_FORBIDDEN = ("each", "promptset")
_ROLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ITEM_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SET_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_PAIR = re.compile(r"@pair:(?P<addr>\S+)")
_T_ROLE = re.compile(r"(?<![A-Za-z0-9_])t@(?P<role>[A-Za-z_]\w*)")
_EQ_ROLE = re.compile(r"=@(?P<role>[A-Za-z_]\w*)")
_BARE_ROLE = re.compile(r"@(?P<role>[A-Za-z_]\w*)")
_VAR = re.compile(r"\$(?P<var>[A-Za-z_]\w*)")

_SETS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()      # debugger -> {name: set}
_RESULTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()   # debugger -> {id: entry}
_OWNED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()     # debugger -> run names


# --------------------------------------------------------------------------- sets


@dataclass(frozen=True)
class Item:
    name: str
    prompt: Any                      # text or a list of token ids
    roles: dict[str, int] = field(default_factory=dict)
    vars: dict[str, Any] = field(default_factory=dict)
    pair: str | None = None


@dataclass(frozen=True)
class PromptSet:
    name: str
    items: tuple[Item, ...]
    source: str = "inline"

    def item(self, name: str) -> Item:
        return next(item for item in self.items if item.name == name)


def _item(raw: Any, where: str) -> Item:
    if not isinstance(raw, dict):
        raise BadArgs(f"{where}: an item is a JSON object")
    known = {"name", "prompt", "tokens", "roles", "vars", "pair"}
    extra = sorted(set(raw) - known)
    if extra:
        raise BadArgs(f"{where}: unknown key {extra[0]!r} (keys: {', '.join(sorted(known))})")
    name = raw.get("name")
    if not isinstance(name, str) or not _ITEM_NAME.fullmatch(name):
        raise BadArgs(f"{where}: name must be a word ([A-Za-z_][A-Za-z0-9_]*)")
    where = f"{where} ({name})"
    if ("prompt" in raw) == ("tokens" in raw):
        raise BadArgs(f"{where}: give prompt (text) or tokens (ids), one of them")
    if "tokens" in raw:
        tokens = raw["tokens"]
        if (not isinstance(tokens, list) or not tokens
                or any(type(t) is not int or t < 0 for t in tokens)):
            raise BadArgs(f"{where}: tokens is a non-empty list of token ids")
        prompt: Any = list(tokens)
    else:
        prompt = raw["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise BadArgs(f"{where}: prompt is non-empty text")
    roles = raw.get("roles") or {}
    if not isinstance(roles, dict):
        raise BadArgs(f"{where}: roles is an object ROLE: row")
    for role, row in roles.items():
        if not _ROLE_NAME.fullmatch(role) or role in ("last", "pair"):
            raise BadArgs(f"{where}: role {role!r} is not a word, or is reserved (last, pair)")
        if type(row) is not int or row < 0:
            raise BadArgs(f"{where}: role {role} must be a row number >= 0")
    variables = raw.get("vars") or {}
    if not isinstance(variables, dict):
        raise BadArgs(f"{where}: vars is an object NAME: token id or token text")
    for key, value in variables.items():
        if not _ROLE_NAME.fullmatch(key):
            raise BadArgs(f"{where}: variable {key!r} is not a word")
        if not ((type(value) is int and value >= 0) or (isinstance(value, str) and value)):
            raise BadArgs(f"{where}: variable {key} is a token id or a token text")
    pair = raw.get("pair")
    if pair is not None and (not isinstance(pair, str) or pair == name):
        raise BadArgs(f"{where}: pair names another item of the set")
    return Item(name, prompt, dict(roles), dict(variables), pair)


def make_set(name: str, rows: list[Any], source: str = "inline") -> PromptSet:
    """A checked prompt set (``BadArgs`` names the item and the key that is wrong)."""
    if not isinstance(name, str) or not _SET_NAME.fullmatch(name):
        raise BadArgs(f"set name {name!r} must be a word ([A-Za-z_][A-Za-z0-9_]*)")
    if not isinstance(rows, list) or not rows:
        raise BadArgs(f"set {name}: no items")
    if len(rows) > MAX_ITEMS:
        raise BadArgs(f"set {name}: {len(rows)} items (at most {MAX_ITEMS}; split the set)")
    items = tuple(_item(raw, f"item {i + 1}") for i, raw in enumerate(rows))
    names = [item.name for item in items]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise BadArgs(f"set {name}: item name {repeated[0]!r} repeats")
    for item in items:
        if item.pair is not None and item.pair not in names:
            raise BadArgs(f"set {name}: item {item.name} pairs with {item.pair!r}, "
                          "which is not in the set")
    return PromptSet(name, items, source)


def read_jsonl(path: str) -> list[Any]:
    """The JSON objects of a JSONL file (blank lines and ``#`` lines skipped). Errors name
    the line, never its content."""
    try:
        text = Path(path).expanduser().read_text()
    except OSError as error:
        raise BadArgs(f"cannot read {path}: {error.strerror or type(error).__name__}") from None
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise BadArgs(f"{path} line {number}: not JSON ({error.msg})") from None
    return rows


def _inline_rows(text: str) -> list[Any]:
    """Items given inline: a JSON list of item objects, or JSONL (one object per line)."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise BadArgs(f"inline items line {number}: not JSON ({error.msg})") from None
        return rows
    return value if isinstance(value, list) else [value]


def sets_of(debugger: Any) -> dict[str, PromptSet]:
    if debugger not in _SETS:
        _SETS[debugger] = {}
    return _SETS[debugger]


def define(debugger: Any, name: str, rows: list[Any], source: str = "inline") -> PromptSet:
    """Keep a prompt set under ``name`` on this debugger (replacing one of that name)."""
    found = make_set(name, rows, source)
    sets_of(debugger)[name] = found
    return found


def load(debugger: Any, name: str, path: str) -> PromptSet:
    return define(debugger, name, read_jsonl(path), source=str(path))


# --------------------------------------------------------------------------- binding


class Unbound(Exception):
    """A template that does not bind for one item: the item is refused, the set goes on."""


class Bound:
    """An item with its tokens: rows of its roles (``last`` included) and lazily its vars."""

    def __init__(self, debugger: Any, item: Item):
        self.item = item
        self.debugger = debugger
        try:
            self.tokens = debugger.tokens(item.prompt)
        except Refused as error:
            raise Unbound(f"prompt: {error}") from None
        self.roles = {**item.roles, "last": len(self.tokens) - 1}

    def row(self, role: str) -> int:
        if role not in self.roles:
            raise Unbound(f"no role {role!r} (roles: {', '.join(sorted(self.roles))})")
        row = self.roles[role]
        if row >= len(self.tokens):
            raise Unbound(f"role {role} = row {row} is past the prompt ({len(self.tokens)} "
                          "tokens)")
        return row

    def var(self, name: str) -> int:
        if name not in self.item.vars:
            raise Unbound(f"no variable {name!r} (vars: {', '.join(sorted(self.item.vars))})")
        value = self.item.vars[name]
        if isinstance(value, int):
            return value
        tokenizer = self.debugger.tokenizer
        if tokenizer is None:
            raise Unbound(f"variable {name} is text ({value!r}); this model has no tokenizer, "
                          "give a token id")
        ids = tokenizer.encode(value, add_special_tokens=False)
        if len(ids) != 1:
            raise Unbound(f"variable {name} = {value!r} is {len(ids)} tokens, not one")
        return int(ids[0])


def _bind_plain(text: str, item: Bound) -> str:
    whole = _BARE_ROLE.fullmatch(text)
    if whole is not None:
        return str(item.row(whole["role"]))
    text = _T_ROLE.sub(lambda m: f"t{item.row(m['role'])}", text)
    text = _EQ_ROLE.sub(lambda m: f"={item.row(m['role'])}", text)
    return _VAR.sub(lambda m: str(item.var(m["var"])), text)


_ROWED = re.compile(r"(?:t-?\d+(?![\w]))|cache\.")


def _pair_address(addr: str, pair: Bound) -> str:
    """``addr`` bound for the paired item. A port address with no row (``L20.A[h5]``) names
    the paired item's own last row (``t<last>.L20.A[h5]``), not whatever row the asking item's
    cursor is at; ``t@ROLE`` and a cache cell (it carries ``row=``) already name theirs."""
    bound = _bind_plain(addr, pair)
    if _ROWED.match(bound) or bound.isdigit():
        return bound
    return f"t{pair.row('last')}.{bound}"


def bind_text(text: str, item: Bound, pair: Bound | None = None,
              pair_run: str | None = None) -> str:
    """The template ``text`` with ``@pair:ADDR``, ``t@ROLE``, ``=@ROLE``, ``@ROLE`` and
    ``$NAME`` replaced for ``item`` (``Unbound`` when one does not resolve)."""
    out, last = [], 0
    for match in _PAIR.finditer(text):
        out.append(_bind_plain(text[last:match.start()], item))
        if pair is None or pair_run is None:
            raise Unbound(f"@pair: item {item.item.name} has no pair")
        out.append(f"{pair_run}:{_pair_address(match['addr'], pair)}")
        last = match.end()
    out.append(_bind_plain(text[last:], item))
    return "".join(out)


def bind_prefill(text: str | None, item: Bound) -> int | None:
    """``each ... prefill=K|@ROLE`` for one item: ``K``, or the row of the item's role ``ROLE``
    (``@last`` steps only the last row, as the default does); ``Unbound`` when the role is
    missing or the split is past the item's prompt. None: the default split."""
    if text is None:
        return None
    text = str(text).strip()
    if re.fullmatch(r"\d+", text):
        prefill = int(text)
    elif _BARE_ROLE.fullmatch(text):
        prefill = item.row(text[1:])
    else:
        raise BadArgs(f"prefill is a row count K or @ROLE (a role of each item), not {text!r}")
    if prefill > len(item.tokens) - 1:
        raise Unbound(f"prefill={prefill} is past the prompt ({len(item.tokens)} tokens: "
                      f"prefill is 0..{len(item.tokens) - 1})")
    return prefill


def _bind_args(name: str, args: dict[str, Any], item: Bound, pair: Bound | None,
               pair_run: str | None) -> dict[str, Any]:
    kinds = {arg.name: arg.type for arg in registry()[name].args}
    out = {}
    for key, value in args.items():
        if isinstance(value, str):
            bound = bind_text(value, item, pair, pair_run)
            if bound != value and kinds.get(key) == "value":
                try:
                    bound = float(bound)
                except ValueError:
                    pass
            value = bound
        out[key] = value
    return out


@dataclass(frozen=True)
class Step:
    name: str
    args: dict[str, Any]
    text: str


def template(line: str) -> list[Step]:
    """The op line(s) of ``each`` (``;`` separates lines), parsed and checked once with the
    placeholders as text; ``BadArgs`` for the whole set when a line does not parse."""
    steps = []
    for part in (p.strip() for p in line.split(";")):
        if not part:
            continue
        verb, _, rest = part.partition(" ")
        name, rest = lines.route(lines.resolve_verb(verb), rest)
        found = registry()[name]
        if name in _FORBIDDEN or found.effect == "native":
            raise BadArgs(f"each runs an op on each item's own run; {name} cannot be one of "
                          "its lines")
        args = lines.parse(name, rest)
        validate(found, args)
        steps.append(Step(name, args, part))
    if not steps:
        raise BadArgs("each needs an op line: each SET: OP LINE")
    return steps


# --------------------------------------------------------------------------- running


@dataclass
class EachResult:
    """``summary``: the bounded summary record (what the ``each`` op returns); ``results``: item
    name -> the op's ordinary result (refused items absent); ``statistics``: the full
    ``analysis.summary.summarize`` output (not capped)."""

    summary: dict[str, Any]
    results: dict[str, dict[str, Any]]
    statistics: dict[str, Any]


def _run_name(set_name: str, item: str) -> str:
    return f"{set_name}-{item}"


def _pair_name(set_name: str, item: str) -> str:
    return f"pair-{set_name}-{item}"


def _reason(error: BaseException) -> str:
    return getattr(error, "message", None) or str(error) or type(error).__name__


def _goto(ctx: Context, at: str) -> None:
    run = ctx.run
    try:
        here = run.parse(run.where) == run.parse(at)
    except ValueError:
        here = False
    if not here:
        dispatch(ctx, "until", {"target": at})


def _snapshot(ctx: Any, debugger: Any) -> dict[str, Any]:
    from .inspect import _symbol_store
    from .predict import session_digest

    return {
        "live": debugger.live, "runs": dict(debugger.runs),
        "run_checkpoints": dict(getattr(debugger, "run_checkpoints", {}) or {}),
        "carried": dict(getattr(debugger, "carried_circuits", {}) or {}),
        "symbols": dict(_symbol_store(Context(debugger))),
        "digest": session_digest(ctx) if getattr(ctx, "run", None) is not None else None,
        "run": getattr(ctx, "run", None),
    }


def _restore(debugger: Any, saved: dict[str, Any], kept: dict[str, Any]) -> None:
    from .inspect import _symbol_store

    debugger.runs.clear()
    debugger.runs.update(saved["runs"])
    debugger.runs.update(kept)
    debugger.live = saved["live"]
    if hasattr(debugger, "run_checkpoints"):
        debugger.run_checkpoints.clear()
        debugger.run_checkpoints.update(saved["run_checkpoints"])
    if hasattr(debugger, "carried_circuits"):
        debugger.carried_circuits.clear()
        debugger.carried_circuits.update(saved["carried"])
    store = _symbol_store(Context(debugger))
    store.clear()
    store.update(saved["symbols"])


def _check_names(debugger: Any, ctx: Any, names: list[str]) -> None:
    owned = _OWNED.setdefault(debugger, set())
    for name in names:
        held = debugger.runs.get(name)
        if held is None:
            continue
        if held is getattr(ctx, "run", None):
            raise Refused(f"each would replace run {name}, the current run; frame another "
                          "run first")
        if name not in owned:
            raise Refused(f"each would replace run {name}, which it did not make; rename the "
                          "set or that run")


def run_each(ctx: Any, set_name: str, line: str, *, at: str | None = None,
             keep: bool = False, prefill: str | None = None) -> EachResult:
    """Run ``line`` on every item of the set ``set_name`` (see the module docstring).
    ``prefill`` (``K`` or ``@ROLE``): each item's run, and its pair's clean run, is started
    with that split, bound per item (``bind_prefill``); None is the default split."""
    from mdb.analysis import summary as stats

    debugger = getattr(ctx, "debugger", None)
    if debugger is None:
        raise Refused("each needs an attached model")
    found = sets_of(debugger).get(set_name)
    if found is None:
        known = ", ".join(sorted(sets_of(debugger))) or "none; set load NAME FILE"
        raise Refused(f"no prompt set {set_name!r} (sets: {known})")
    if prefill is not None:
        prefill = str(prefill).strip()
        if not (re.fullmatch(r"\d+", prefill) or _BARE_ROLE.fullmatch(prefill)):
            raise BadArgs(f"prefill is a row count K or @ROLE (a role of each item), not "
                          f"{prefill!r}")
    at = (at or DEFAULT_AT).strip()
    steps = template(line)
    uses_pair = "@pair:" in line
    names = [_run_name(found.name, item.name) for item in found.items]
    donors_needed = sorted({item.pair for item in found.items if item.pair} if uses_pair
                           else set())
    _check_names(debugger, ctx, names + [_pair_name(found.name, p) for p in donors_needed])
    saved = _snapshot(ctx, debugger)
    made: dict[str, Any] = {}
    donors: dict[str, tuple[Any, str | None]] = {}
    rows, results = [], {}
    try:
        for item in found.items:
            row, result = _one(debugger, saved, found, item, steps, at, uses_pair, made,
                               donors, prefill)
            if keep and _run_name(found.name, item.name) in made:
                row["run"] = _run_name(found.name, item.name)
            rows.append(row)
            if result is not None:
                results[item.name] = result
            if not keep:      # released now: one item run is held at a time (donors aside)
                released = made.pop(_run_name(found.name, item.name), None)
                if released is not None and debugger.runs.get(released.name) is released:
                    debugger.runs.pop(released.name)
                debugger.live = saved["live"]
    finally:
        kept = made if keep else {}
        _OWNED.setdefault(debugger, set()).update(kept)
        _restore(debugger, saved, kept)
    if saved["run"] is not None:
        from .predict import session_digest

        if getattr(ctx, "run", None) is not saved["run"] or session_digest(ctx) != saved["digest"]:
            raise RuntimeError("each changed the caller's session")
    _put_blobs(debugger, rows, results)
    full = stats.summarize([results[name] for name in results])
    summary = _summary(found, line, steps, at, keep, rows, full, saved["digest"])
    if prefill is not None:
        summary["prefill"] = prefill    # the template; each item's row holds its bound split
    entry_id = _remember(debugger, summary, results)
    summary["id"] = entry_id
    _store(debugger, summary)
    return EachResult(summary, results, full)


def _one(debugger: Any, saved: dict[str, Any], found: PromptSet, item: Item,
         steps: list[Step], at: str, uses_pair: bool, made: dict[str, Any],
         donors: dict[str, tuple[Any, str | None]],
         prefill: str | None = None) -> tuple[dict[str, Any], Any]:
    """One item: bind, make its run, run the steps. Returns its summary row and its result."""
    from mdb.analysis import summary as stats

    row: dict[str, Any] = {"name": item.name}
    try:
        bound = Bound(debugger, item)
        split = bind_prefill(prefill, bound)
        pair = pair_run = None
        if uses_pair and item.pair is not None:
            pair = Bound(debugger, found.item(item.pair))
            pair_run = _pair_name(found.name, item.pair)
        calls = [(step, _bind_args(step.name, step.args, bound, pair, pair_run))
                 for step in steps]
        target = bind_text(at, bound)
    except Unbound as error:
        return {**row, "state": "refused", "reason": str(error)}, None
    if split is not None:
        row["prefill"] = split
    if pair is not None:
        reason = _donor(debugger, saved, found, pair, pair_run, at, made, donors, prefill)
        if reason is not None:
            return {**row, "state": "refused", "reason": f"pair {item.pair}: {reason}"}, None
    name = _run_name(found.name, item.name)
    debugger.live = saved["live"]          # each item run carries what a fresh start would
    local = Context(debugger)
    try:
        dispatch(local, "start", {"prompt": bound.tokens, "name": name,
                                  **({} if split is None else {"prefill": split})})
        made[name] = local.run
        _goto(local, target)
        result = None
        for number, (step, args) in enumerate(calls, 1):
            try:
                result = dispatch(local, step.name, args)
            except OpError as error:
                where = f"line {number} ({step.name})" if len(calls) > 1 else step.name
                return {**row, "state": "refused", "reason": f"{where}: {_reason(error)}"}, None
    except OpError as error:
        return {**row, "state": "refused", "reason": f"at {target}: {_reason(error)}"}, None
    row["state"] = result.get("state", "resolved")
    row["result"] = seal_digest(result)
    if isinstance(result.get("record"), str):
        row["record"] = result["record"]
    row["key"] = stats.key_columns(steps[-1].name, result)
    row["bytes"] = int(local.run.held_bytes()["total"])
    bound_args = {k: str(v) for k, v in calls[-1][1].items() if v != steps[-1].args.get(k)}
    if bound_args:
        row["bound"] = bound_args
    if pair is not None:
        found_at = sorted({m[0] for _, args in calls for v in args.values()
                           if isinstance(v, str)
                           for m in re.finditer(re.escape(pair_run) + r":[^\s,;=]+(?:\[[^\]]*\])?",
                                                v)})
        if found_at:
            row["donor"] = found_at
    return row, result


def _donor(debugger: Any, saved: dict[str, Any], found: PromptSet, pair: Bound,
           run_name: str, at: str, made: dict[str, Any],
           donors: dict[str, tuple[Any, str | None]], prefill: str | None = None) -> str | None:
    """Make (once) the clean run of the paired item at its ``at`` stop, under ``prefill``
    bound for the paired item (its own role row); the refusal reason."""
    if pair.item.name in donors:
        return donors[pair.item.name][1]
    reason = None
    debugger.live = saved["live"]
    local = Context(debugger)
    try:
        target = bind_text(at, pair)
        split = bind_prefill(prefill, pair)
        dispatch(local, "start", {"prompt": pair.tokens, "name": run_name,
                                  **({} if split is None else {"prefill": split})})
        made[run_name] = local.run
        _goto(local, target)
    except Unbound as error:
        reason = str(error)
    except OpError as error:
        reason = _reason(error)
    donors[pair.item.name] = (local.run, reason)
    return reason


# --------------------------------------------------------------------------- the record


def _summary(found: PromptSet, line: str, steps: list[Step], at: str, keep: bool,
             rows: list[dict[str, Any]], full: dict[str, Any], digest: str | None
             ) -> dict[str, Any]:
    refused = [row for row in rows if row["state"] == "refused"]
    ran = len(rows) - len(refused)
    if refused:
        names = ", ".join(row["name"] for row in refused[:8])
        more = f" +{len(refused) - 8} more" if len(refused) > 8 else ""
        exclusion = (f"{len(refused)} of {len(rows)} items refused ({names}{more}) and "
                     f"excluded; statistics over the {ran} others")
    else:
        exclusion = f"none refused; statistics over all {ran} items"
    key_paths = [path for row in rows if "key" in row for path in row["key"]["paths"]]
    numeric = sorted(full["numeric"],
                     key=lambda s: key_paths.index(s["path"]) if s["path"] in key_paths
                     else len(key_paths))
    for row in rows:
        if "key" in row:
            row["key"].pop("paths", None)
    out: dict[str, Any] = {
        "set": found.name, "line": line, "verb": steps[-1].name, "at": at,
        "total": len(rows), "items": rows,
        "statistics": {
            "n": full["n"], "exclusion": exclusion,
            "numeric": numeric[:MAX_NUMERIC],
            "verdicts": full["verdicts"][:MAX_VERDICTS],
            "omitted": {"count": max(0, len(numeric) - MAX_NUMERIC)
                        + max(0, len(full["verdicts"]) - MAX_VERDICTS)},
            "partial": {"count": full["partial"],
                        "meaning": "paths present in some items only: not summarised"},
        },
        "runs": "kept as named runs SET-ITEM (frame NAME:)" if keep else "released",
    }
    if digest is not None:
        out["session"] = {"unchanged": True, "debugger": digest}
    return out


def _remember(debugger: Any, summary: dict[str, Any], results: dict[str, Any]) -> str:
    ledger = _RESULTS.setdefault(debugger, {})
    number = 1 + max((int(k[4:]) for k in ledger), default=0)
    key = f"each{number}"
    ledger[key] = {"summary": summary, "results": results}
    while len(ledger) > MAX_RESULTS:
        ledger.pop(next(iter(ledger)))
    return key


def _writable(debugger: Any) -> Any:
    project = getattr(debugger, "project", None)
    return None if project is None or project.read_only else project


def _put_blobs(debugger: Any, rows: list[dict[str, Any]], results: dict[str, Any]) -> None:
    """Each item's result as a sha256-checked JSON blob (an op result is not a valid record
    body, so it cannot be a record); the row's ``blob`` is the pointer. No writable project:
    the row says the result is held in memory only."""
    from mdb.core.bounded import put_json

    project = _writable(debugger)
    for row in rows:
        if row["name"] not in results:
            continue
        if project is None:
            row["blob"] = {"state": "absent",
                           "reason": "held in memory only (no writable project)"}
        else:
            row["blob"] = put_json(results[row["name"]], store=project.store)


def _stored_entry(debugger: Any, record_id: str) -> dict[str, Any] | None:
    """The stored ``prompt-set`` summary named ``record_id`` (a summary's ``stored``), or None."""
    project = getattr(debugger, "project", None)
    if project is None:
        return None
    from mdb.core import index
    from mdb.core.pointers import RecordError

    try:
        record = index.get_record(project, debugger.session_id, record_id)
    except (RecordError, LookupError, KeyError, OSError):
        return None
    return record["body"] if record.get("kind") == "prompt-set" else None


def load_item_result(debugger: Any, summary: dict[str, Any], item: str) -> dict[str, Any]:
    """The item's result from its blob (checked against the blob's sha256 and the row's result
    digest); ``Refused`` when the row has no stored blob or the blob fails a check."""
    from mdb.core.bounded import load_json
    from mdb.core.pointers import Unresolved

    row = next((r for r in summary["items"] if r["name"] == item), None)
    blob = None if row is None else row.get("blob")
    if blob is None or blob.get("state") != "stored":
        raise Refused(f"{item!r} has no stored result: "
                      + ("not in the set or refused" if row is None or "result" not in row
                         else (blob or {}).get("reason", "held in memory only")))
    try:
        result = load_json(blob, store=debugger.project.store)
    except (ValueError, Unresolved) as error:
        raise Refused(f"stored result of {item!r} refused: "
                      f"{getattr(error, 'reason', None) or error}") from None
    if seal_digest(result) != row["result"]:
        raise Refused(f"stored result of {item!r} refused: it differs from the row's digest")
    return result


def _store(debugger: Any, summary: dict[str, Any]) -> None:
    """With a writable project: the summary as one ``prompt-set`` record. Item results are
    referenced by digest and by blob pointer (and by the op's own record id where the op seals
    one, as ``whatif`` does); an op result is not a record body (its pointers and bounds are not
    ``num`` claims), so the results themselves are the blobs ``_put_blobs`` stored."""
    project = _writable(debugger)
    if project is None:
        return
    from mdb.core import index
    from mdb.core.records import declare_kind, seal_record

    declare_kind("prompt-set", required=("set", "line", "items", "statistics"),
                 plain_numeric=("items.*.bytes", "items.*.prefill", "items.*.blob.bytes", "items.*.blob.tensor.shape.*", "total", "statistics.n",
                                "statistics.numeric.*.n", "statistics.numeric.*.agree.count",
                                "statistics.numeric.*.by_authority.*.n",
                                "statistics.numeric.*.by_authority.*.agree.count",
                                "statistics.verdicts.*.counts.*.count",
                                "statistics.omitted.count", "statistics.partial.count"))
    try:
        record = seal_record(session_id=debugger.session_id, kind="prompt-set",
                             model_identity=debugger.adapter.model_identity,
                             execution=dict(debugger.adapter.execution),
                             authority="derived-exact",
                             body={k: v for k, v in summary.items() if k != "session"})
        summary["stored"] = index.put_record(project, record)
    except RecordError as error:
        summary["stored"] = {"state": "unresolved", "reason": f"not stored: {error}"}


# --------------------------------------------------------------------------- ops


def _op_each(ctx, set, line, at, keep, prefill=None):  # noqa: A002 - the wire name
    return run_each(ctx, set, line, at=at, keep=keep, prefill=prefill).summary


def _item_view(item: Item) -> dict[str, Any]:
    out: dict[str, Any] = {"name": item.name}
    if isinstance(item.prompt, str):
        out["prompt"] = item.prompt if len(item.prompt) <= 200 else item.prompt[:200] + "..."
    else:
        out["prompt_tokens"] = len(item.prompt)
        out["ids"] = list(item.prompt[:16])
    out["roles"] = [{"name": k, "row": v} for k, v in item.roles.items()]
    out["vars"] = [{"name": k, "token": v} if isinstance(v, int) else {"name": k, "text": v}
                   for k, v in item.vars.items()]
    if item.pair is not None:
        out["pair"] = item.pair
    return out


def _op_promptset(ctx, action, name, arg):
    debugger = getattr(ctx, "debugger", None)
    if debugger is None:
        raise Refused("prompt sets need an attached model")
    known = sets_of(debugger)
    if action == "list":
        return {"action": "list", "sets": [{"set": n, "total": len(s.items), "source": s.source}
                                           for n, s in sorted(known.items())][:64]}
    if not name:
        raise BadArgs(f"promptset {action} needs a name")
    if action == "load":
        if not arg:
            raise BadArgs("write set load NAME FILE")
        found = load(debugger, name, arg)
    elif action == "define":
        if not arg:
            raise BadArgs("promptset define NAME ITEMS: ITEMS is a JSON list of items (or "
                          "JSONL, one item per line)")
        found = define(debugger, name, _inline_rows(arg))
    elif action == "show":
        if name not in known:
            raise Refused(f"no prompt set {name!r} (sets: {', '.join(sorted(known)) or 'none'})")
        found = known[name]
    else:  # result
        entry = _RESULTS.get(debugger, {}).get(name)
        if entry is None:
            stored = _stored_entry(debugger, name)
            if stored is None:
                held = ", ".join(_RESULTS.get(debugger, {})) or "none"
                raise Refused(f"no each result {name!r} in memory (held: {held}) and no stored "
                              "summary of that record id in the project")
            if not arg:
                return {"action": "result", "id": name, "summary": stored}
            return {"action": "result", "id": name, "item": arg,
                    "result": load_item_result(debugger, stored, arg), "from": "stored blob"}
        if not arg:
            return {"action": "result", "id": name, "summary": entry["summary"]}
        if arg not in entry["results"]:
            raise Refused(f"{name} holds no result for item {arg!r} (refused, or not in the "
                          "set)")
        return {"action": "result", "id": name, "item": arg, "result": entry["results"][arg]}
    return {"action": action, "set": found.name, "source": found.source,
            "total": len(found.items), "items": [_item_view(item) for item in found.items]}


register(Op("each", "Run one op line on every prompt of a set (one run per prompt): one result "
            "per prompt and a summary (n, mean, min, max, sign agreement, verdict counts)",
            (Arg("set", "str", required=True, help="a prompt set (set load NAME FILE)"),
             Arg("line", "str", required=True, positional=False,
                 help="the op line(s); t@ROLE, =@ROLE, $VAR and @pair:ADDR bind per item"),
             Arg("at", "str", default=DEFAULT_AT, positional=False,
                 help="where each item's run stops before the line runs"),
             Arg("keep", "bool", default=False, positional=False,
                 help="keep the item runs as named runs SET-ITEM"),
             Arg("prefill", "str", positional=False,
                 help="each item's prefill split: K, or @ROLE (that role's row of each item; "
                      "the pair's clean run takes its own role row); default all but the last")),
            "native", False, _op_each,
            usage="each SET [at=ADDR] [prefill=K|@ROLE] [keep]: OP LINE [; OP LINE ...]"))
register(Op("promptset", "Prompt sets: load a JSONL file, show or list sets, read a stored each "
            "result (the REPL's set load / set show)",
            (Arg("action", "choice", required=True, choices=("load", "define", "show", "list", "result")),
             Arg("name", "str", help="the set (load, show) or the each result id (result)"),
             Arg("arg", "str", help="load: the JSONL file (not served: send the set inline); "
                                    "define: the items as JSON; result: an item name")),
            "mutate", False, _op_promptset,
            usage="set load NAME FILE | set define NAME ITEMS | set show NAME | "
                  "promptset list | "
                  "promptset result ID [ITEM]"))


def check_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """The summary record under the pointer rule (64 KiB, no arrays); raises ``RecordError``."""
    return check_bounded(summary, "each summary")
