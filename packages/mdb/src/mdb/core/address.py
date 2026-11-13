"""One occurrence address: the CLI argument, the record key and the UI URL fragment.

Grammar (no whitespace outside quoted strings; every character is consumed or rejected)::

    address  := [origin ":"] target [selectors] ["~" program]
    pattern  := [origin ":"] ptarget [selectors] ["~" program]       # breakpoints, matches()
    origin   := branch | "@" cutref {"+" edit}
    target   := clock "." port            # port occurrence inside layer L (clock has .L)
              | clock "." global          # emb fn logits M sample (clock has no .L)
              | "cache.L" INT "." ("k"|"v")   # a state cell, read at the current clock
    clock    := "t" INT ["." "s" INT] ["." "L" INT]   # token row; diffusion step; layer
    selectors:= "[" sel {"," sel} "]"
    sel      := key ["="] (INT | INT ":" INT | INT "|" INT {"|" INT})   # keys h kv i src row d
              | QUOTED ["=" INT]          # token text (JSON string), optionally its resolved id
    ptarget  := pclock ["." (port | global | "*")] | "cache.L" (INT|"*") ["." ("k"|"v"|"*")]
    pclock   := one or more of "t" (INT|"*"), "s" (INT|"*"), "L" (INT|"*") in that order,
                joined by "."

INT has no sign and no leading zero. ``a:b`` is half-open, ``a|b`` a set. Canonical form
(``format``) orders keyed selectors h, kv, i, src, row, d, writes ``h5``/``kv0`` bare and every
other key with ``=``, sorts sets, and keeps token selectors in their given order after the
keyed ones (``M[" hot"," cold"]`` is an ordered contrast). Extensions over the plan grammar:
``~program`` (which program produced the value: a stepped occurrence and a prefill occurrence
are different values) and ``"text"=ID`` (a token text with its resolved id).

An ``Occurrence`` (``t18.L20.v``: the value a port wrote at that clock) and a ``StateCell``
(``cache.L20.v[row=18]``: the cache cell as read at the current clock) are different types and
never compare equal, even where their values coincide at write time.

Matching and formatting defaults (intended; ``matches`` is subset semantics, see its docstring):

==========================================  ==================================================
pattern or address written                  meaning
==========================================  ==================================================
pattern with no ``tN``/``sN``/``LN``        that component is unconstrained (any value or none)
pattern with no ``origin:``                 matches every origin (with one: must be equal)
pattern with no ``~program``                matches every program (with one: must be equal)
pattern ``t*.L20.o``                        ``t*`` is "no constraint": it prints as ``L20.o``
pattern ``L*`` / ``s*``                     the address must HAVE a layer / step
pattern selector key (``h5``)               address must select a subset of it on that key
no pattern selector                         any selector matches
``h=5:6``                                   one element: normalised to ``h5`` (same for every key)
==========================================  ==================================================

``matches`` answers "is this address inside the pattern"; ``matches_stop`` answers "does a
debugger stop here": it ignores selectors on both sides (a breakpoint ``L20.p[h5]`` fires at the
stop ``t21.L20.p``). Global ports accept only their own selectors (``GLOBAL_SELECTORS``).

State-slot mapping (``to_state_address``) for the saturn_pub decoder adapter
(``adapters/decoder.py``; slots tokens, keys, values, hidden, logits, sampled_token). ``L`` is the
model's layer count (``layers=``, required for ``h``/``h_out``); layers are ``0..L-1``:

======================  =============================  ==========================================
address                 slot, selector                 clock where the slot holds it
======================  =============================  ==========================================
cache.LK.k[kv,row,d]    keys, (K, 0, kv, row, d)       any clock after row was written
cache.LK.v[kv,row,d]    values, (K, 0, kv, row, d)     same (selector keys must be a prefix)
tN.emb[d]               hidden, (0, 0, d)              logical_step N, phase layer, index 0
tN.LK.h[d]  K<L         hidden, (0, 0, d)              logical_step N, phase layer, index K
tN.LK.h_out[d]  K<L-1   hidden, (0, 0, d)              logical_step N, phase layer, index K+1
tN.L{L-1}.h_out[d]      hidden, (0, 0, d)              layer: N, phase layer, index L;
                                                       operation: N, phase normalization
tN.fn[d]                hidden, (0, 0, d)              operation granularity: N, phase readout
tN.logits[i | "tok"=id] logits, (0, id)                layer: N+1, phase embed; operation: N,
                                                       phase sample
tN.sample               tokens, (0, N+1)  (layer)      layer: N+1, phase embed
                        sampled_token, (0, 0) (op)     operation: N, phase commit
======================  =============================  ==========================================

``from_state_address(to_state_address(a)) == state_canonical(a)``: the one spelling of a hidden
value is ``value_key`` (``tN.emb`` and ``tN.L0.h`` are one state, ``tN.LK.h_out`` and
``tN.L{K+1}.h`` are one state) and a logits token selector with its id reads back as ``i``.

Every other port (a q k v qr kr s p A o r n g u sg f m) and the derived margin ``M`` is a port
occurrence with no state slot: it exists only inside a layer call (or as a derived value) and
is reached by the engine's port program, never by ``Session.read_port``. Origins and programs
other than ``stepped`` are not part of a ``StateAddress``: the caller picks the session.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

DEFAULT_PORTS = ("h", "a", "q", "k", "v", "qr", "kr", "s", "p", "A", "o", "r", "n", "g", "u",
                 "sg", "f", "m", "h_out")
DEFAULT_GLOBALS = ("emb", "fn", "logits", "M", "sample")
CACHE_TENSORS = ("k", "v")
SELECTOR_KEYS = ("h", "kv", "i", "src", "row", "d")
BARE_KEYS = ("h", "kv")
CELL_KEYS = ("kv", "row", "d")
_KEY_ORDER = {key: n for n, key in enumerate(SELECTOR_KEYS)}
_DIGITS = "0123456789"
_NAME_START = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
_NAME_CHARS = _NAME_START + _DIGITS
_REF_CHARS = _NAME_CHARS + "-"
_SLOT_CLOCK = {"layer": ("embed", "layer"), "operation": ("sample", "commit")}
MAX_INT = 10 ** 15 - 1   # every integer is exact in JSON/JavaScript

# Which selectors a global port accepts. ``keys``: allowed selector keys; ``tokens``: at most this
# many token texts; ``either``: keys and tokens are alternatives, not both; ``exactly``: a
# concrete address names exactly this many items, as token texts or as a set of that many ``i``.
# Patterns need only be subsets. A caller may add or change entries (``global_ports`` that are
# not listed here accept any selector).
GLOBAL_SELECTORS = {
    "emb": {"keys": ("d",), "tokens": 0},
    "fn": {"keys": ("d",), "tokens": 0},
    "logits": {"keys": ("i",), "tokens": 64, "either": True},
    "M": {"keys": ("i",), "tokens": 2, "either": True, "exactly": 2},
    "sample": {"keys": (), "tokens": 0},
}


class AddressError(ValueError):
    """A malformed address; ``column`` (1-based) points at the first bad character."""

    def __init__(self, reason, text=None, column=None):
        self.reason, self.text, self.column = reason, text, column
        message = reason if column is None else f"column {column}: {reason}"
        if text is not None and column is not None:
            message += f"\n  {text}\n  {' ' * (column - 1)}^"
        super().__init__(message)


class NoStateSlot(ValueError):
    """The address names a port occurrence (or form) with no saturn_pub state slot."""


# --------------------------------------------------------------------------- value types

@dataclass(frozen=True)
class Span:
    """Half-open integer range ``start:stop``."""
    start: int
    stop: int

    def __post_init__(self):
        _nonneg(self.start, "span start")
        _nonneg(self.stop, "span stop")
        if self.stop <= self.start:
            raise AddressError("a range needs stop > start")

    def __contains__(self, value):
        return self.start <= value < self.stop


@dataclass(frozen=True)
class AnyOf:
    """A set of at least two integers, kept sorted."""
    values: tuple

    def __post_init__(self):
        values = tuple(self.values)
        for value in values:
            _nonneg(value, "set item")
        if len(values) < 2 or len(set(values)) != len(values):
            raise AddressError("a set needs at least two distinct items")
        object.__setattr__(self, "values", tuple(sorted(values)))


@dataclass(frozen=True)
class Token:
    """A token selector: its text, and once resolved by the caller's tokenizer, its id."""
    text: str
    id: int | None = None

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text:
            raise AddressError("a token selector needs non-empty text")
        _encodable(self.text, "token text")
        if self.id is not None:
            _nonneg(self.id, "token id")


@dataclass(frozen=True)
class Selector:
    keys: tuple = ()      # ((key, int | Span | AnyOf), ...) in canonical key order
    tokens: tuple = ()    # (Token, ...) in given order

    def __post_init__(self):
        keys = tuple((k, v) for k, v in self.keys)
        names = [k for k, _ in keys]
        if len(set(names)) != len(names):
            raise AddressError("duplicate selector key")
        for key, value in keys:
            if key not in _KEY_ORDER:
                raise AddressError(f"unknown selector key {key!r}")
            if isinstance(value, bool) or not isinstance(value, (int, Span, AnyOf)):
                raise AddressError(f"selector {key} needs an integer, range or set")
            if isinstance(value, int):
                _nonneg(value, f"selector {key}")
        # A one-element range is the integer: h=5:6 is h5.
        keys = tuple((k, v.start if isinstance(v, Span) and v.stop == v.start + 1 else v)
                     for k, v in keys)
        object.__setattr__(self, "keys", tuple(sorted(keys, key=lambda kv: _KEY_ORDER[kv[0]])))
        tokens = tuple(self.tokens)
        if any(not isinstance(t, Token) for t in tokens):
            raise AddressError("token selectors must be Token values")
        object.__setattr__(self, "tokens", tokens)

    @classmethod
    def of(cls, *tokens, **keys):
        return cls(tuple(keys.items()), tuple(t if isinstance(t, Token) else Token(t)
                                              for t in tokens))

    def get(self, key, default=None):
        return dict(self.keys).get(key, default)

    def __bool__(self):
        return bool(self.keys or self.tokens)

    def to_json(self):
        out = {}
        for key, value in self.keys:
            if isinstance(value, Span):
                out[key] = {"start": value.start, "stop": value.stop}
            elif isinstance(value, AnyOf):
                out[key] = list(value.values)
            else:
                out[key] = value
        if self.tokens:
            out["tokens"] = [{"text": t.text} if t.id is None else {"text": t.text, "id": t.id}
                             for t in self.tokens]
        return out

    @classmethod
    def from_json(cls, value):
        if not isinstance(value, dict):
            raise AddressError("selector JSON must be an object")
        keys, tokens = [], ()
        for key, item in value.items():
            if key == "tokens":
                if not isinstance(item, list) or not item:
                    raise AddressError("selector tokens must be a non-empty list")
                tokens = tuple(_token_json(t) for t in item)
            elif isinstance(item, dict):
                if set(item) != {"start", "stop"}:
                    raise AddressError(f"selector {key} range needs start and stop")
                keys.append((key, Span(item["start"], item["stop"])))
            elif isinstance(item, list):
                try:
                    ordered = sorted(item) == item
                except TypeError:
                    raise AddressError(f"selector {key} set holds non-integers") from None
                if not ordered:
                    raise AddressError(f"selector {key} set is not in canonical order")
                keys.append((key, AnyOf(tuple(item))))
            else:
                keys.append((key, item))
        return cls(tuple(keys), tokens)


def _token_json(value):
    if not isinstance(value, dict) or not {"text"} <= set(value) <= {"text", "id"}:
        raise AddressError("a token selector is {text, id?}")
    return Token(value["text"], value.get("id"))


@dataclass(frozen=True)
class Origin:
    """Where the value lives: a named branch, or a cut plus ordered edit ids."""
    branch: str | None = None
    cut: str | None = None
    edits: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "edits", tuple(self.edits))
        if (self.branch is None) == (self.cut is None):
            raise AddressError("an origin is a branch or a cut, not both or neither")
        if self.branch is not None:
            if self.edits:
                raise AddressError("ordered edits follow a cut ('@cut+edit'), not a branch")
            if not _is_name(self.branch, _REF_CHARS):
                raise AddressError(f"invalid branch name {self.branch!r}")
        else:
            for ref in (self.cut, *self.edits):
                if not isinstance(ref, str) or not ref or any(c not in _REF_CHARS for c in ref):
                    raise AddressError(f"invalid cut or edit reference {ref!r}")

    def to_json(self):
        if self.branch is not None:
            return {"branch": self.branch}
        return {"cut": self.cut, "edits": list(self.edits)}

    @classmethod
    def from_json(cls, value):
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) not in ({"branch"}, {"cut", "edits"}):
            raise AddressError("origin JSON is {branch} or {cut, edits}")
        if "branch" in value:
            return cls(branch=value["branch"])
        if not isinstance(value["edits"], list):
            raise AddressError("origin edits must be a list")
        return cls(cut=value["cut"], edits=tuple(value["edits"]))


@dataclass(frozen=True)
class Occurrence:
    """The value a port (or global) produced at a clock: ``t21.L20.p[h5,src=18]``."""
    token: int
    port: str
    layer: int | None = None
    step: int | None = None
    selector: Selector = Selector()
    origin: Origin | None = None
    program: str | None = None

    kind = "occurrence"

    def __post_init__(self):
        _nonneg(self.token, "token row")
        for name in ("layer", "step"):
            if getattr(self, name) is not None:
                _nonneg(getattr(self, name), name)
        _check_port_name(self.port)
        _check_parts(self.selector, self.origin, self.program)
        if _is_global(self.port):
            if self.layer is not None:
                raise AddressError(f"{self.port!r} is a global and takes no layer")
            _check_global_selector(self.port, self.selector, partial=False)
        elif self.port in DEFAULT_PORTS and self.layer is None:
            raise AddressError(f"port {self.port!r} needs a layer")

    def to_json(self):
        clock = {"token": self.token}
        if self.step is not None:
            clock["step"] = self.step
        if self.layer is not None:
            clock["layer"] = self.layer
        origin = None if self.origin is None else self.origin.to_json()
        out = {"kind": "occurrence", "origin": origin, "clock": clock, "port": self.port,
               "selector": self.selector.to_json()}
        if self.program is not None:
            out["program"] = self.program
        return out

    def __str__(self):
        return format_address(self)


@dataclass(frozen=True)
class StateCell:
    """A key/value cache cell read at the current clock: ``cache.L20.v[kv0,row=18]``."""
    layer: int
    tensor: str
    selector: Selector = Selector()
    origin: Origin | None = None

    kind = "cell"

    def __post_init__(self):
        _nonneg(self.layer, "layer")
        if self.tensor not in CACHE_TENSORS:
            raise AddressError("a cache cell is k or v")
        _check_parts(self.selector, self.origin, None)
        bad = [k for k, _ in self.selector.keys if k not in CELL_KEYS]
        if bad or self.selector.tokens:
            raise AddressError("a cache cell selects only kv, row and d")

    def to_json(self):
        return {"kind": "cell", "origin": None if self.origin is None else self.origin.to_json(),
                "cell": {"layer": self.layer, "tensor": self.tensor},
                "selector": self.selector.to_json()}

    def __str__(self):
        return format_address(self)


Address = Union[Occurrence, StateCell]

ANY = "*"   # in a pattern: the component is present with any value


@dataclass(frozen=True)
class Pattern:
    """A wildcard address for breakpoints. ``None`` = unconstrained, ``ANY`` = present."""
    kind: str                      # "occurrence" | "cell"
    token: int | None = None
    step: int | str | None = None
    layer: int | str | None = None
    port: str | None = None        # occurrence port/global, or cell tensor
    selector: Selector = Selector()
    origin: Origin | None = None
    program: str | None = None

    def __post_init__(self):
        if self.kind not in ("occurrence", "cell"):
            raise AddressError("a pattern is an 'occurrence' or a 'cell' pattern")
        if self.token == ANY:
            object.__setattr__(self, "token", None)   # t* is "no constraint"
        if self.token is not None:
            _nonneg(self.token, "token row")
        for name in ("step", "layer"):
            value = getattr(self, name)
            if value is not None and value != ANY:
                _nonneg(value, name)
        _check_parts(self.selector, self.origin, self.program)
        if self.kind == "cell":
            if self.token is not None or self.step is not None or self.program is not None:
                raise AddressError("a cache-cell pattern has only a layer and a tensor")
            if self.layer is None:
                raise AddressError("a cache-cell pattern needs a layer (or '*')")
            if self.port is not None and self.port not in CACHE_TENSORS:
                raise AddressError("a cache cell is k or v")
            if [k for k, _ in self.selector.keys if k not in CELL_KEYS] or self.selector.tokens:
                raise AddressError("a cache cell selects only kv, row and d")
            return
        if self.port is not None:
            _check_port_name(self.port)
            if _is_global(self.port):
                if self.layer is not None:
                    raise AddressError(f"{self.port!r} is a global and takes no layer")
                _check_global_selector(self.port, self.selector, partial=True)

    def __str__(self):
        return format_pattern(self)


def _nonneg(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AddressError(f"{label} must be a non-negative integer")
    if value > MAX_INT:
        raise AddressError(f"{label} has more than 15 digits")


def _encodable(text, label):
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise AddressError(f"{label} cannot be encoded as UTF-8 (lone surrogate)") from None


def _is_name(value, chars):
    return (isinstance(value, str) and bool(value) and value[0] in _NAME_START
            and all(c in chars for c in value))


def _check_port_name(port):
    if not _is_name(port, _NAME_CHARS):
        raise AddressError(f"invalid port name {port!r}")
    if port[0] in "tsL" and port[1:2].isdigit():   # t5, s1x, L2norm, t5a: all read as a clock
        raise AddressError(f"port name {port!r} would read as a clock component")


def _is_global(port):
    return port in GLOBAL_SELECTORS or port in DEFAULT_GLOBALS


def _check_global_selector(port, selector, *, partial):
    """Refuse selectors a global port does not take (a pattern may name a subset)."""
    spec = GLOBAL_SELECTORS.get(port)
    if spec is None:
        return
    bad = [k for k, _ in selector.keys if k not in spec["keys"]]
    if bad:
        raise AddressError(f"{port!r} does not select by {bad[0]!r}"
                           + (f" (it takes {', '.join(spec['keys'])})" if spec["keys"] else
                              " (it takes no selector)"))
    if len(selector.tokens) > spec["tokens"]:
        raise AddressError(f"{port!r} takes " + ("no token texts" if not spec["tokens"] else
                                                  f"at most {spec['tokens']} token texts"))
    if spec.get("either") and selector.keys and selector.tokens:
        raise AddressError(f"{port!r} selects by keys or by token texts, not both")
    exactly = spec.get("exactly")
    if exactly and not partial:
        value = selector.get("i")
        count = len(selector.tokens) if selector.tokens else (
            len(value.values) if isinstance(value, AnyOf) else 0)
        if count != exactly:
            raise AddressError(f"{port!r} names exactly {exactly} token texts (or {exactly} "
                               "values of i)")


def _check_parts(selector, origin, program):
    if not isinstance(selector, Selector):
        raise AddressError("selector must be a Selector")
    if origin is not None and not isinstance(origin, Origin):
        raise AddressError("origin must be an Origin")
    if program is not None and not _is_name(program, _REF_CHARS):
        raise AddressError(f"invalid program name {program!r}")


# --------------------------------------------------------------------------- parser

class _Scan:
    def __init__(self, text):
        if not isinstance(text, str):
            raise AddressError("an address is text")
        self.text, self.pos = text, 0
        for column, char in enumerate(text):
            if "\ud800" <= char <= "\udfff":
                raise AddressError("an address is UTF-8 text; found a lone surrogate", text,
                                   column + 1)

    def fail(self, reason, pos=None):
        raise AddressError(reason, self.text, (self.pos if pos is None else pos) + 1)

    def peek(self, n=1):
        return self.text[self.pos:self.pos + n]

    def done(self):
        return self.pos >= len(self.text)

    def eat(self, literal):
        if self.text.startswith(literal, self.pos):
            self.pos += len(literal)
            return True
        return False

    def expect(self, literal, what):
        if not self.eat(literal):
            self.fail(f"expected {what}, found {self._found()}")

    def _found(self):
        if self.done():
            return "end of address"
        char = self.text[self.pos]
        return "whitespace" if char.isspace() else repr(char)

    def integer(self, what, *, wildcard=False):
        if wildcard and self.eat("*"):
            return ANY
        start = self.pos
        while not self.done() and self.text[self.pos] in _DIGITS:
            self.pos += 1
        digits = self.text[start:self.pos]
        if not digits:
            self.fail(f"expected {what}, found {self._found()}")
        if len(digits) > 1 and digits[0] == "0":
            self.fail(f"{what} has a leading zero", start)
        if len(digits) > 15:
            self.fail(f"{what} has more than 15 digits (every integer must be exact in JSON)",
                      start)
        return int(digits)

    def run(self, chars):
        start = self.pos
        while not self.done() and self.text[self.pos] in chars:
            self.pos += 1
        return self.text[start:self.pos]

    def name(self, what):
        start = self.pos
        if self.done() or self.text[self.pos] not in _NAME_START:
            self.fail(f"expected {what}, found {self._found()}")
        return self.run(_NAME_CHARS), start


def _names(ports, global_ports):
    ports = DEFAULT_PORTS if ports is None else tuple(ports)
    global_ports = DEFAULT_GLOBALS if global_ports is None else tuple(global_ports)
    for name in (*ports, *global_ports):
        if not _is_name(name, _NAME_CHARS):
            raise AddressError(f"invalid port name {name!r} in the port list")
        if name[0] in "tsL" and name[1:2].isdigit():
            raise AddressError(f"port name {name!r} would read as a clock component")
    overlap = set(ports) & set(global_ports)
    if overlap:
        raise AddressError(f"names are both layer ports and globals: {sorted(overlap)}")
    return frozenset(ports), frozenset(global_ports)


def _origin(scan):
    text = scan.text
    if scan.eat("@"):
        start = scan.pos
        cut = scan.run(_REF_CHARS)
        if not cut:
            scan.fail("expected a cut reference after '@'", start)
        edits = []
        while scan.eat("+"):
            start = scan.pos
            edit = scan.run(_REF_CHARS)
            if not edit:
                scan.fail("expected an edit id after '+'", start)
            edits.append(edit)
        scan.expect(":", "':' after the origin")
        return Origin(cut=cut, edits=tuple(edits))
    end = scan.pos
    while end < len(text) and text[end] in _REF_CHARS:
        end += 1
    if end < len(text) and text[end] == ":" and end > scan.pos:
        if text[scan.pos] not in _NAME_START:
            scan.fail("a branch name starts with a letter or '_'")
        branch = text[scan.pos:end]
        scan.pos = end + 1
        return Origin(branch=branch)
    return None


def _selectors(scan, kind):
    keys, tokens, seen = [], [], set()
    if not scan.eat("["):
        return Selector()
    while True:
        start = scan.pos
        if scan.peek() == '"' and kind == "cell":
            scan.fail("a cache cell selects only kv, row and d")
        if scan.peek() == '"':
            try:
                text, end = json.JSONDecoder().raw_decode(scan.text, scan.pos)
            except json.JSONDecodeError:
                scan.fail("unterminated or invalid quoted token text")
            if not isinstance(text, str) or not text:
                scan.fail("a token selector needs non-empty text")
            if any("\ud800" <= c <= "\udfff" for c in text):
                scan.fail("token text cannot be encoded as UTF-8 (lone surrogate escape)", start)
            scan.pos = end
            token_id = scan.integer("a token id") if scan.eat("=") else None
            tokens.append(Token(text, token_id))
        else:
            key = scan.run("abcdefghijklmnopqrstuvwxyz")
            if not key:
                scan.fail(f"expected a selector key or quoted token, found {scan._found()}")
            if key not in _KEY_ORDER:
                scan.fail(f"unknown selector key {key!r} (keys: {', '.join(SELECTOR_KEYS)})",
                          start)
            if kind == "cell" and key not in CELL_KEYS:
                scan.fail("a cache cell selects only kv, row and d", start)
            if key in seen:
                scan.fail(f"duplicate selector key {key!r}", start)
            seen.add(key)
            scan.eat("=")
            first = scan.integer(f"a value for {key}")
            if scan.eat(":"):
                at = scan.pos
                stop = scan.integer("the range end")
                if stop <= first:
                    scan.fail("a range needs end > start", at)
                value = Span(first, stop)
            elif scan.peek() == "|":
                items = [first]
                while scan.eat("|"):
                    at = scan.pos
                    item = scan.integer("a set item")
                    if item in items:
                        scan.fail(f"duplicate set item {item}", at)
                    items.append(item)
                value = AnyOf(tuple(items))
            else:
                value = first
            keys.append((key, value))
        if scan.eat("]"):
            break
        scan.expect(",", "',' or ']'")
    return Selector(tuple(keys), tuple(tokens))


def _program(scan):
    if not scan.eat("~"):
        return None
    start = scan.pos
    program = scan.run(_REF_CHARS)
    if not program or program[0] not in _NAME_START:
        scan.fail("expected a program name after '~'", start)
    return program


def _finish(scan):
    if not scan.done():
        scan.fail(f"unexpected {scan._found()} after the address")


def _clock(scan, *, wildcard):
    """t INT [.s INT] [.L INT]; a pattern takes any of t/s/L in that order, '*' for INT."""
    if not wildcard:
        if not scan.eat("t"):
            scan.fail(f"expected 't<row>' or 'cache.L<layer>', found {scan._found()}")
        values = {"t": scan.integer("a token row")}
    else:
        values = {}
    starts = _DIGITS + ("*" if wildcard else "")
    for letter, what in (("t", "a token row"), ("s", "a diffusion step"), ("L", "a layer")):
        if letter in values:
            continue
        prefix = ("." if values else "") + letter
        after = scan.text[scan.pos + len(prefix):scan.pos + len(prefix) + 1]
        if scan.text.startswith(prefix, scan.pos) and after and after in starts:
            scan.pos += len(prefix)
            values[letter] = scan.integer(what, wildcard=wildcard)
    if not values:
        scan.fail(f"expected a clock (t<row>, s<step> or L<layer>), found {scan._found()}")
    token = values.get("t")
    return (None if token == ANY else token), values.get("s"), values.get("L")


def _target(scan, ports, global_ports, *, wildcard, check_ports=True):
    if scan.eat("cache."):
        if not scan.eat("L"):
            scan.fail("expected 'L<layer>' after 'cache.'")
        layer = scan.integer("a layer", wildcard=wildcard)
        tensor = None
        if scan.eat("."):
            start = scan.pos
            if wildcard and scan.eat("*"):
                tensor = None
            else:
                tensor = scan.run(_NAME_CHARS)
                if tensor not in CACHE_TENSORS:
                    scan.fail("a cache cell is k or v", start)
        elif not wildcard:
            scan.fail("expected '.k' or '.v' after the cache layer")
        return "cell", None, None, layer, tensor
    token, step, layer = _clock(scan, wildcard=wildcard)
    port = None
    if scan.eat("."):
        if wildcard and scan.eat("*"):
            port = None
        else:
            nxt = scan.text[scan.pos + 1:scan.pos + 2]
            if scan.peek() in ("t", "s", "L") and nxt and nxt in _DIGITS + "*":
                scan.fail("clock components come once each, in the order t, s, L")
            port, start = scan.name("a port name")
            if not check_ports:
                # Any well-formed name; the structural rule is the constructor's.
                if port[0] in "tsL" and port[1:].isdigit():
                    scan.fail(f"port name {port!r} would read as a clock component", start)
            elif port in global_ports:
                if layer is not None:
                    scan.fail(f"{port!r} is a global and takes no layer", start)
            elif port in ports:
                if layer is None and not wildcard:
                    scan.fail(f"port {port!r} needs a layer (t<row>.L<layer>.{port})", start)
            else:
                scan.fail(f"unknown port {port!r}", start)
    elif not wildcard:
        scan.fail(f"expected '.<port>', found {scan._found()}")
    return "occurrence", token, step, layer, port


def parse(text, *, ports=None, global_ports=None, check_ports=True) -> Address:
    """Parse one concrete address; ``ports``/``global_ports`` default to the Qwen2 port list.

    ``check_ports=False`` accepts any well-formed port name (the structural rules of the
    constructors still apply: a default layer port needs a layer, a global takes none), so
    ``parse(format(a), check_ports=False) == a`` for every constructible ``Occurrence``.
    """
    ports, global_ports = _names(ports, global_ports) if check_ports else (None, None)
    scan = _Scan(text)
    origin = _origin(scan)
    kind, token, step, layer, port = _target(scan, ports, global_ports, wildcard=False,
                                             check_ports=check_ports)
    selector = _selectors(scan, kind)
    at = scan.pos
    program = _program(scan)
    _finish(scan)
    try:
        if kind == "cell":
            if program is not None:
                scan.fail("a state cell has no program (it is read at the current clock)", at)
            return StateCell(layer, port, selector, origin)
        return Occurrence(token, port, layer, step, selector, origin, program)
    except AddressError as error:
        if error.column is not None:
            raise
        raise AddressError(error.reason, text, at + 1) from None


def parse_pattern(text, *, ports=None, global_ports=None, check_ports=True) -> Pattern:
    """Parse a breakpoint pattern: ``L*.p``, ``t21.L20``, ``t*.L20.o``, ``cache.L*.v``."""
    ports, global_ports = _names(ports, global_ports) if check_ports else (None, None)
    scan = _Scan(text)
    origin = _origin(scan)
    kind, token, step, layer, port = _target(scan, ports, global_ports, wildcard=True,
                                             check_ports=check_ports)
    selector = _selectors(scan, kind)
    at = scan.pos
    program = _program(scan)
    _finish(scan)
    if kind == "cell" and program is not None:
        scan.fail("a state cell has no program", at)
    try:
        return Pattern(kind, token, step, layer, port, selector, origin, program)
    except AddressError as error:
        if error.column is not None:
            raise
        raise AddressError(error.reason, text, at + 1) from None


# --------------------------------------------------------------------------- format / JSON

def _format_origin(origin):
    if origin is None:
        return ""
    if origin.branch is not None:
        return origin.branch + ":"
    return "@" + origin.cut + "".join("+" + e for e in origin.edits) + ":"


def _format_selector(selector):
    parts = []
    for key, value in selector.keys:
        if isinstance(value, Span):
            parts.append(f"{key}={value.start}:{value.stop}")
        elif isinstance(value, AnyOf):
            parts.append(f"{key}=" + "|".join(str(v) for v in value.values))
        elif key in BARE_KEYS:
            parts.append(f"{key}{value}")
        else:
            parts.append(f"{key}={value}")
    for token in selector.tokens:
        text = json.dumps(token.text, ensure_ascii=False)
        parts.append(text if token.id is None else f"{text}={token.id}")
    return "[" + ",".join(parts) + "]" if parts else ""


def format_address(address) -> str:
    """Canonical text of an address (stable: equal addresses give equal text)."""
    if isinstance(address, StateCell):
        return (_format_origin(address.origin) + f"cache.L{address.layer}.{address.tensor}"
                + _format_selector(address.selector))
    if not isinstance(address, Occurrence):
        raise TypeError("format expects an Occurrence or StateCell (use format_pattern)")
    clock = f"t{address.token}"
    if address.step is not None:
        clock += f".s{address.step}"
    if address.layer is not None:
        clock += f".L{address.layer}"
    return (_format_origin(address.origin) + clock + "." + address.port
            + _format_selector(address.selector)
            + ("" if address.program is None else "~" + address.program))


format = format_address  # the public name in the plan; shadows the builtin inside this module


def format_pattern(pattern) -> str:
    def part(letter, value):
        return None if value is None else f"{letter}{value}"

    if pattern.kind == "cell":
        text = f"cache.L{pattern.layer}" + ("" if pattern.port is None else "." + pattern.port)
    else:
        clock = [p for p in (part("t", pattern.token), part("s", pattern.step),
                             part("L", pattern.layer)) if p]
        text = ".".join(clock or ["t*"])
        if pattern.port is not None:
            text += "." + pattern.port
    return (_format_origin(pattern.origin) + text + _format_selector(pattern.selector)
            + ("" if pattern.program is None else "~" + pattern.program))


def to_json(address):
    return address.to_json()


def from_json(value, *, ports=None, global_ports=None, check_ports=True) -> Address:
    """Strict inverse of ``to_json``; the port name is checked against the port lists.

    ``check_ports=False`` checks only that the port is a well-formed name (stored pointers and
    records are validated without the engine's port program).
    """
    if check_ports:
        ports, global_ports = _names(ports, global_ports)
    if not isinstance(value, dict):
        raise AddressError("address JSON must be an object")
    kind = value.get("kind")
    if kind == "cell":
        if set(value) != {"kind", "origin", "cell", "selector"}:
            raise AddressError("cell JSON is {kind, origin, cell, selector}")
        cell = value["cell"]
        if not isinstance(cell, dict) or set(cell) != {"layer", "tensor"}:
            raise AddressError("cell JSON needs {layer, tensor}")
        return StateCell(cell["layer"], cell["tensor"], Selector.from_json(value["selector"]),
                         Origin.from_json(value["origin"]))
    if kind != "occurrence":
        raise AddressError("address JSON kind is 'occurrence' or 'cell'")
    if not {"kind", "origin", "clock", "port", "selector"} <= set(value) <= {
            "kind", "origin", "clock", "port", "selector", "program"}:
        raise AddressError("occurrence JSON is {kind, origin, clock, port, selector, program?}")
    clock = value["clock"]
    if not isinstance(clock, dict) or not {"token"} <= set(clock) <= {"token", "step", "layer"}:
        raise AddressError("clock JSON is {token, step?, layer?}")
    if "program" in value and value["program"] is None:
        raise AddressError("omit program rather than writing null")
    address = Occurrence(clock["token"], value["port"], clock.get("layer"), clock.get("step"),
                         Selector.from_json(value["selector"]), Origin.from_json(value["origin"]),
                         value.get("program"))
    if check_ports:
        _check_port(address, ports, global_ports)
    return address


def _check_port(address, ports, global_ports):
    if address.port in global_ports:
        if address.layer is not None:
            raise AddressError(f"{address.port!r} is a global and takes no layer")
    elif address.port in ports:
        if address.layer is None:
            raise AddressError(f"port {address.port!r} needs a layer")
    else:
        raise AddressError(f"unknown port {address.port!r}")


def coerce(value, *, ports=None, global_ports=None) -> Address:
    """Address from an Address, its text, or its JSON."""
    if isinstance(value, (Occurrence, StateCell)):
        return value
    if isinstance(value, str):
        return parse(value, ports=ports, global_ports=global_ports)
    return from_json(value, ports=ports, global_ports=global_ports)


# --------------------------------------------------------------------------- matching

def _value_within(inner, outer):
    """True when every index selected by ``inner`` is selected by ``outer``."""
    def members(v):
        if isinstance(v, int):
            return None, {v}
        if isinstance(v, AnyOf):
            return None, set(v.values)
        return v, None

    inner_span, inner_set = members(inner)
    outer_span, outer_set = members(outer)
    if outer_span is not None:
        if inner_span is not None:
            return outer_span.start <= inner_span.start and inner_span.stop <= outer_span.stop
        return all(i in outer_span for i in inner_set)
    if inner_span is not None:
        return set(range(inner_span.start, inner_span.stop)) <= outer_set
    return inner_set <= outer_set


def _clock_ok(want, have):
    if want is None:
        return True
    if want == ANY:
        return have is not None
    return want == have


def _token_equal(want, have):
    """Resolved ids decide when both sides carry one; text only when an id is missing."""
    if want.id is not None and have.id is not None:
        return want.id == have.id
    return want.text == have.text


def _match(pattern, address, *, selectors, ports, global_ports):
    if isinstance(pattern, str):
        pattern = parse_pattern(pattern, ports=ports, global_ports=global_ports)
    if isinstance(address, str):
        address = parse(address, ports=ports, global_ports=global_ports)
    if pattern.kind != address.kind:
        return False
    if pattern.origin is not None and pattern.origin != address.origin:
        return False
    if address.kind == "cell":
        if not _clock_ok(pattern.layer, address.layer):
            return False
        if pattern.port is not None and pattern.port != address.tensor:
            return False
    else:
        if pattern.token is not None and pattern.token != address.token:
            return False
        if not (_clock_ok(pattern.step, address.step) and _clock_ok(pattern.layer, address.layer)):
            return False
        if pattern.port is not None and pattern.port != address.port:
            return False
        if pattern.program is not None and pattern.program != address.program:
            return False
    if not selectors:
        return True
    have = dict(address.selector.keys)
    for key, want in pattern.selector.keys:
        if key not in have or not _value_within(have[key], want):
            return False
    if pattern.selector.tokens:
        want, got = pattern.selector.tokens, address.selector.tokens
        return len(want) == len(got) and all(_token_equal(w, g) for w, g in zip(want, got))
    return True


def matches(pattern, address, *, ports=None, global_ports=None) -> bool:
    """Whether ``address`` lies inside ``pattern`` (subset semantics; use for filtering and
    "is this value one the pattern names").

    Omitted pattern parts are unconstrained; ``L*``/``s*`` require the address to have a
    layer/step; ``t*`` equals an omitted token. A pattern selector key requires the address to
    select a subset of it on that key; pattern tokens must equal the address tokens in order
    (by resolved id when both carry one, else by text). A pattern without an origin or program
    matches every origin or program; with one it must be equal. For a debugger breakpoint use
    ``matches_stop``, which ignores selectors.
    """
    return _match(pattern, address, selectors=True, ports=ports, global_ports=global_ports)


def matches_stop(pattern, address, *, ports=None, global_ports=None) -> bool:
    """Whether a debugger stopped at ``address`` fires ``pattern`` (use for breakpoints).

    Origin, clock, port and program are compared exactly as ``matches`` does; selectors are
    IGNORED on both sides, because a debugger stops at whole ports: the breakpoint ``L20.p[h5]``
    fires at the stop ``t21.L20.p``. Use ``matches`` to ask whether a value is inside the
    pattern's selection.
    """
    return _match(pattern, address, selectors=False, ports=ports, global_ports=global_ports)


# --------------------------------------------------------------------------- one key per value

def _check_layer(address, layers):
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("layers must be the model's decoder layer count (a positive integer)")
    if address.layer is None or address.layer >= layers:
        raise AddressError(f"layer {address.layer} does not exist (the model has {layers})")


def _canonical_value(value):
    """A set of consecutive integers is the range (``h=1|2`` is ``h=1:3``)."""
    if isinstance(value, AnyOf) and value.values[-1] - value.values[0] + 1 == len(value.values):
        return Span(value.values[0], value.values[-1] + 1)
    return value


def _canonical_selector(port, selector):
    """One spelling per selected set: ranges for consecutive sets, and a logits token with a
    resolved id as ``i`` (``logits["x"=5]``, ``logits["y"=5]`` and ``logits[i=5]`` are one)."""
    if (port == "logits" and len(selector.tokens) == 1 and selector.tokens[0].id is not None
            and not selector.keys):
        return Selector((("i", selector.tokens[0].id),))
    if GLOBAL_SELECTORS.get(port, {}).get("exactly"):
        # a port that names an exact count of values (``M[i=14|15]``) keeps its set: a range
        # would no longer count as that many, and the canonical form must be constructible
        return selector
    return Selector(tuple((k, _canonical_value(v)) for k, v in selector.keys), selector.tokens)


def value_key(address, *, layers) -> Address:
    """The one canonical spelling of an address's value (the record and index key).

    ``tN.emb`` is ``tN.L0.h``; ``tN.Lk.h_out`` is ``tN.L{k+1}.h`` for ``k < layers-1`` (the last
    layer's ``h_out`` stays: there is no layer ``L`` to enter). A selector is canonical too: a
    set of consecutive integers is the range (``h=1|2`` -> ``h=1:3``) and a logits token with a
    resolved id is ``i`` (``logits["x"=5]`` -> ``logits[i=5]``); any other token selector,
    origin and program are kept. Every other address is its own key. A layer that does not
    exist is an ``AddressError``.
    """
    from dataclasses import replace

    address = coerce(address)
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("layers must be the model's decoder layer count (a positive integer)")
    if isinstance(address, StateCell):
        return replace(address, selector=_canonical_selector("cache", address.selector))
    address = replace(address, selector=_canonical_selector(address.port, address.selector))
    if address.port == "emb":
        return replace(address, port="h", layer=0)
    if address.port == "h":
        _check_layer(address, layers)
    elif address.port == "h_out":
        _check_layer(address, layers)
        if address.layer < layers - 1:
            return replace(address, port="h", layer=address.layer + 1)
    return address


def aliases(address, *, layers) -> tuple:
    """Every spelling of the address's value, ``value_key`` first (the rest in text order).

    The spellings are the port forms (``h``, ``h_out``, ``emb``) with the canonical selector and
    with the selector as written (``h=1|2`` beside the canonical ``h=1:3``). A selector is
    carried to an alias that can hold it (``emb`` takes only ``d``); when it has no meaning on
    the alias target only the other forms are returned. Never raises on a valid address.
    """
    from dataclasses import replace

    written = coerce(address)
    key = value_key(address, layers=layers)
    found = {format_address(key): key}
    selectors = [key.selector] + ([written.selector] if written.selector != key.selector else [])
    for selector in selectors:
        spelled = replace(key, selector=selector)
        found.setdefault(format_address(spelled), spelled)
        if isinstance(key, Occurrence) and key.port == "h":
            try:
                alias = (replace(spelled, port="h_out", layer=key.layer - 1) if key.layer > 0
                         else replace(spelled, port="emb", layer=None))
            except AddressError:
                continue
            found.setdefault(format_address(alias), alias)
    rest = sorted(text for text in found if found[text] != key)
    return (key, *(found[text] for text in rest))


def state_canonical(address, *, layers) -> Address:
    """What ``from_state_address(to_state_address(address))`` returns for a mappable address:
    ``value_key`` (which spells a resolved logits token as ``i``) and no origin or program."""
    from dataclasses import replace

    key = value_key(address, layers=layers)
    if isinstance(key, StateCell):
        return replace(key, origin=None)
    return replace(key, origin=None, program=None)


# --------------------------------------------------------------------------- saturn_pub mapping

def _need_layers(layers, what):
    if layers is None:
        raise ValueError(f"{what} needs layers= (the model's decoder layer count)")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("layers must be a positive integer")


def state_clock(address, *, granularity="layer", layers=None):
    """The decoder clock at which the address's state slot holds its value.

    Returns ``{logical_step, phase, index}`` (``index`` is None outside phase ``layer``), or
    None for a cache cell (readable at any later clock). Raises ``NoStateSlot`` otherwise.
    ``h`` and ``h_out`` need ``layers=`` (ValueError without it); a layer the model does not
    have is ``NoStateSlot``.
    """
    address = coerce(address)
    if granularity not in _SLOT_CLOCK:
        raise ValueError("granularity is 'layer' or 'operation'")
    if isinstance(address, StateCell):
        if layers is not None and address.layer >= layers:
            raise NoStateSlot(f"layer {address.layer} does not exist (the model has {layers})")
        return None
    if address.step is not None:
        raise NoStateSlot("the decoder adapter has no diffusion-step clock")
    if address.program not in (None, "stepped"):
        raise NoStateSlot(f"program {address.program!r} is not the stepped session's state")
    t, port = address.token, address.port
    if port == "emb":
        return {"logical_step": t, "phase": "layer", "index": 0}
    if port in ("h", "h_out"):
        _need_layers(layers, f"the {port} port")
        k = address.layer
        if k >= layers:
            raise NoStateSlot(f"layer {k} does not exist (the model has {layers})")
        if port == "h":
            return {"logical_step": t, "phase": "layer", "index": k}
        if k + 1 < layers:
            return {"logical_step": t, "phase": "layer", "index": k + 1}
        if granularity == "layer":
            return {"logical_step": t, "phase": "layer", "index": layers}
        return {"logical_step": t, "phase": "normalization", "index": None}
    if port == "fn":
        if granularity != "operation":
            raise NoStateSlot("the final-norm output is a state only at operation granularity")
        return {"logical_step": t, "phase": "readout", "index": None}
    if port in ("logits", "sample"):
        if granularity == "layer":
            return {"logical_step": t + 1, "phase": "embed", "index": None}
        return {"logical_step": t, "phase": "sample" if port == "logits" else "commit",
                "index": None}
    raise NoStateSlot(f"{format_address(address)} is a port occurrence with no state slot")


def _hidden_selector(selector):
    extra = [k for k, _ in selector.keys if k != "d"]
    d = selector.get("d")
    if extra or selector.tokens or (d is not None and not isinstance(d, int)):
        raise NoStateSlot("the hidden state selects only one integer d")
    return () if d is None else (0, 0, d)


_LIMIT_KEYS = ("heads", "kv_groups", "vocab", "layers", "first_step")


def _first_step(limits):
    """The session's first stepped row (``limits['first_step']``), or None when not supplied."""
    return limits.get("first_step")


def _refuse_unstepped(what, row, limits):
    """``sample`` and ``logits`` of a row are the model's own output only when the session
    stepped that row (the prompt rows before the first step were prefilled or teacher-forced:
    the token a layer-granularity ``tN.sample`` maps to is then a prompt token, not a sample).
    Without the first stepped row nothing can be said, so the conversion refuses."""
    first = _first_step(limits)
    if first is None:
        raise NoStateSlot(f"{what} needs limits['first_step'] (the session's first stepped "
                          "row) to say the session produced it")
    if row < first:
        raise NoStateSlot(f"{what} of row {row} was never stepped (the session starts at "
                          f"row {first})")


def _limits(limits, layers):
    """``limits`` (heads, kv_groups, vocab, layers: positive ints) and the layer count to use."""
    if limits is None:
        return {}, layers
    if not isinstance(limits, Mapping) or set(limits) - set(_LIMIT_KEYS):
        raise ValueError("limits is a mapping with keys " + ", ".join(_LIMIT_KEYS))
    for key, value in limits.items():
        floor = 0 if key == "first_step" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < floor:
            raise ValueError(f"limits[{key!r}] must be "
                             + ("a non-negative" if floor == 0 else "a positive") + " integer")
    return dict(limits), (limits.get("layers") if layers is None else layers)


def to_state_address(address, *, point=None, state_schema=None, granularity="layer", layers=None,
                     limits=None):
    """The saturn_pub ``StateAddress`` holding this address's value (table in the docstring).

    ``point`` (an ``ExecutionPoint``) is required for every port occurrence: an address whose
    clock was not checked and bound could read a different element (``t3.L0.h`` is not
    ``t3.L1.h``), so without it ``ValueError`` is raised; ``state_clock(address, layers=)``
    computes the clock without binding. A cache cell has no clock and takes ``point`` only to
    bind it. With ``point`` the clock is checked, its fingerprint bound, and a cache cell whose
    ``row`` is past the point's logical step is ``NoStateSlot``. ``layers`` (the model's layer
    count) is required for ``h``/``h_out``. ``limits`` (optional ``heads``, ``kv_groups``,
    ``vocab``, ``layers``): with it a cache cell with ``kv`` >= ``kv_groups``, a logits ``i``
    (or resolved token id) >= ``vocab`` and a layer >= ``layers`` are ``NoStateSlot``; ``heads``
    is accepted for completeness (no state-slot address selects a head). ``first_step`` (a
    non-negative int: the first row the session stepped, i.e. its starting logical step) is
    REQUIRED for ``logits`` and ``sample``: a row below it was prefilled or teacher-forced,
    never sampled, so those two ports of such a row are ``NoStateSlot``, and without
    ``first_step`` they are refused rather than guessed. Otherwise nothing is range-checked
    without ``limits``.
    """
    from saturn_pub.contracts import StateAddress

    address = coerce(address)
    limits, layers = _limits(limits, layers)
    clock = state_clock(address, granularity=granularity, layers=layers)   # NoStateSlot first
    if point is None and not isinstance(address, StateCell):
        raise ValueError("point= is required to bind an occurrence address")
    if point is not None and clock is not None:
        have = {"logical_step": point.logical_step, "phase": point.phase,
                "index": point.index if point.phase == "layer" else None}
        if have != clock:
            raise NoStateSlot(f"{format_address(address)} is not in state at {have}")
    bound = None if point is None else point.fingerprint
    if isinstance(address, StateCell):
        sel = address.selector
        values = [sel.get(k) for k in CELL_KEYS]
        prefix = []
        for value in values:
            if value is None:
                break
            prefix.append(value)
        if any(v is not None for v in values[len(prefix):]) or any(
                not isinstance(v, int) for v in prefix):
            raise NoStateSlot("cache selectors must be integers forming a prefix of kv, row, d")
        if point is not None and sel.get("row") is not None and sel.get("row") > point.logical_step:
            raise NoStateSlot(f"cache row {sel.get('row')} is past logical step "
                              f"{point.logical_step}")
        if sel.get("kv") is not None and "kv_groups" in limits and sel.get("kv") >= limits["kv_groups"]:
            raise NoStateSlot(f"kv group {sel.get('kv')} does not exist "
                              f"(the model has {limits['kv_groups']})")
        slot = "keys" if address.tensor == "k" else "values"
        return StateAddress(slot, (address.layer, 0, *prefix), "cache", bound, state_schema)
    port = address.port
    if port in ("emb", "h", "h_out", "fn"):
        return StateAddress("hidden", _hidden_selector(address.selector), "carrier", bound,
                            state_schema)
    if port in ("logits", "sample"):
        _refuse_unstepped(f"{format_address(address)}", address.token, limits)
    if port == "logits":
        sel = address.selector
        ids = [t.id for t in sel.tokens]
        if (sel.keys and sel.tokens) or [k for k, _ in sel.keys] not in ([], ["i"]):
            raise NoStateSlot("logits select one vocabulary entry by i or by a resolved token")
        if len(ids) > 1 or None in ids:
            raise NoStateSlot("logits need exactly one token selector resolved to an id")
        index = sel.get("i") if sel.keys else (ids[0] if ids else None)
        if index is not None and not isinstance(index, int):
            raise NoStateSlot("logits select one integer vocabulary index")
        if index is not None and "vocab" in limits and index >= limits["vocab"]:
            raise NoStateSlot(f"vocabulary index {index} does not exist "
                              f"(the vocabulary has {limits['vocab']})")
        return StateAddress("logits", () if index is None else (0, index), "observation",
                            bound, state_schema)
    if port != "sample":
        raise NoStateSlot(f"{format_address(address)} is a port occurrence with no state slot")
    if address.selector:
        raise NoStateSlot("the sampled token takes no selector")
    if granularity == "operation":
        return StateAddress("sampled_token", (0, 0), "decision", bound, state_schema)
    return StateAddress("tokens", (0, address.token + 1), "cursor", bound, state_schema)


_SLOT_ROLE = {"keys": "cache", "values": "cache", "hidden": "carrier", "logits": "observation",
              "sampled_token": "decision", "tokens": "cursor"}


def from_state_address(state, point, *, layers=None, granularity="layer", origin=None,
                       limits=None):
    """Inverse of ``to_state_address`` at ``point``: the canonical address of a state value.

    ``point`` is the ``ExecutionPoint`` the value was read at (and must match
    ``state.execution_point`` when that is bound). It refuses exactly the (state, point) pairs
    ``to_state_address`` cannot produce, and for every producible pair returns
    ``value_key``'s spelling (``tN.L0.h`` rather than ``tN.emb``, ``tN.L{k+1}.h`` rather than
    ``tN.Lk.h_out``, the last layer's ``h_out``, ``i`` for a logits token). ``layers`` is
    required for the hidden state. ``limits['first_step']`` is required for the ``logits``,
    ``sampled_token`` and ``tokens`` slots (same rule as ``to_state_address``: a row below the
    session's first stepped row is refused, and so is any of them without it).
    """
    limits, layers = _limits(limits, layers)
    if state.execution_point is not None and state.execution_point != point.fingerprint:
        raise NoStateSlot("state address is bound to another execution point")
    if granularity not in _SLOT_CLOCK:
        raise ValueError("granularity is 'layer' or 'operation'")
    if state.role and state.role != _SLOT_ROLE.get(state.slot):
        raise NoStateSlot(f"role {state.role!r} does not belong to slot {state.slot!r}")
    sel = state.selector
    if any(isinstance(v, bool) or v < 0 for v in sel):
        raise NoStateSlot("state selectors are non-negative integers")
    if state.slot in ("keys", "values"):
        if layers is not None and sel and sel[0] >= layers:
            raise NoStateSlot(f"layer {sel[0]} does not exist (the model has {layers})")
        if not 2 <= len(sel) <= 5 or sel[1] != 0:
            raise NoStateSlot("cache selector is (layer, 0, kv?, row?, d?)")
        if len(sel) >= 4 and sel[3] > point.logical_step:
            raise NoStateSlot(f"cache row {sel[3]} is past logical step {point.logical_step}")
        keys = tuple(zip(CELL_KEYS, sel[2:]))
        return StateCell(sel[0], "k" if state.slot == "keys" else "v", Selector(keys), origin)
    t = point.logical_step
    if state.slot == "hidden":
        _need_layers(layers, "the hidden state")
        if sel and (len(sel) != 3 or sel[:2] != (0, 0)):
            raise NoStateSlot("hidden selector is () or (0, 0, d)")
        selector = Selector((("d", sel[2]),)) if sel else Selector()
        if point.phase == "readout" and granularity == "operation":
            return Occurrence(t, "fn", None, None, selector, origin)
        if point.phase == "normalization" and granularity == "operation":
            return Occurrence(t, "h_out", layers - 1, None, selector, origin)
        if point.phase != "layer" or point.index is None or point.index > layers:
            raise NoStateSlot(f"no hidden-state address at phase {point.phase!r}")
        if point.index == layers:
            if granularity != "layer":
                raise NoStateSlot("operation granularity has no layer phase after the last layer")
            return Occurrence(t, "h_out", layers - 1, None, selector, origin)
        return Occurrence(t, "h", point.index, None, selector, origin)
    if state.slot == "logits":
        if sel and (len(sel) != 2 or sel[0] != 0):
            raise NoStateSlot("logits selector is () or (0, id)")
        want = "embed" if granularity == "layer" else "sample"
        if point.phase != want or (granularity == "layer" and t < 1):
            raise NoStateSlot(f"logits are addressed at phase {want!r}, not {point.phase!r}")
        selector = Selector((("i", sel[1]),)) if sel else Selector()
        token = t - 1 if granularity == "layer" else t
        _refuse_unstepped(f"logits at step {t}", token, limits)
        return Occurrence(token, "logits", None, None, selector, origin)
    if state.slot == "sampled_token" and granularity == "operation":
        if sel != (0, 0) or point.phase != "commit":
            raise NoStateSlot("the sampled token is (0, 0) at phase 'commit'")
        _refuse_unstepped(f"the sampled token at step {t}", t, limits)
        return Occurrence(t, "sample", None, None, Selector(), origin)
    if (state.slot == "tokens" and granularity == "layer" and len(sel) == 2 and sel[1] == t
            and sel[0] == 0 and t >= 1 and point.phase == "embed"):
        _refuse_unstepped(f"the sampled token at step {t}", t - 1, limits)
        return Occurrence(t - 1, "sample", None, None, Selector(), origin)
    raise NoStateSlot(f"slot {state.slot!r} with selector {sel} has no address at this point")
