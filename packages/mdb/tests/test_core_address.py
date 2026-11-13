import json
import random

import pytest

from mdb.core import address as A
from mdb.core.address import (
    AddressError,
    AnyOf,
    NoStateSlot,
    Occurrence,
    Origin,
    Selector,
    Span,
    StateCell,
    Token,
)

EXAMPLES = [
    't21.L20.p[h5,src=18]',
    't21.L20.f[i=4031]',
    't21.logits[" hot"]',
    'boiling:t21.logits[" hot"]',
    '@pre+e2:t21.M[" hot"," cold"]',
    'cache.L20.v[kv0,row=18]',
    't3.s12.L4.p[h2]',
]
PATTERNS = ['L*.p', 't21.L20', 't*.L20.o']


@pytest.mark.parametrize("text", EXAMPLES)
def test_examples_round_trip(text):
    address = A.parse(text)
    assert A.format(address) == text
    assert A.parse(A.format(address)) == address
    encoded = json.dumps(address.to_json(), sort_keys=True)
    assert A.from_json(json.loads(encoded)) == address
    assert A.coerce(address.to_json()) == address == A.coerce(text)


def test_example_json_shapes():
    assert A.parse('@pre+e2:t21.M[" hot"," cold"]').to_json() == {
        "kind": "occurrence", "origin": {"cut": "pre", "edits": ["e2"]},
        "clock": {"token": 21}, "port": "M",
        "selector": {"tokens": [{"text": " hot"}, {"text": " cold"}]}}
    assert A.parse("t3.s12.L4.p[h2]~prefill").to_json() == {
        "kind": "occurrence", "origin": None, "clock": {"token": 3, "step": 12, "layer": 4},
        "port": "p", "selector": {"h": 2}, "program": "prefill"}
    assert A.parse("boiling:cache.L20.v[kv0,row=18]").to_json() == {
        "kind": "cell", "origin": {"branch": "boiling"}, "cell": {"layer": 20, "tensor": "v"},
        "selector": {"kv": 0, "row": 18}}


@pytest.mark.parametrize("text, canonical", [
    ("t1.L0.p[src=18,h=5]", "t1.L0.p[h5,src=18]"),
    ("t1.L0.p[row18,kv=0,d4,i=3]", "t1.L0.p[kv0,i=3,row=18,d=4]"),
    ("t1.L0.p[h=3|1|2]", "t1.L0.p[h=1|2|3]"),
    ("t1.L0.p[src0:4]", "t1.L0.p[src=0:4]"),
    ('t1.logits[" hot"=2560]', 't1.logits[" hot"=2560]'),
    ('t1.L2.p[" x",h1]', 't1.L2.p[h1," x"]'),
    ('t1.logits["caf\\u00e9"]', 't1.logits["café"]'),
    ('t1.logits["a\\"b"]', 't1.logits["a\\"b"]'),
])
def test_canonical_ordering(text, canonical):
    address = A.parse(text)
    assert A.format(address) == canonical
    assert A.parse(canonical) == address


def test_token_order_is_kept_and_matters():
    assert A.parse('t1.M[" hot"," cold"]') != A.parse('t1.M[" cold"," hot"]')
    assert A.format(A.parse('t1.M[" cold"," hot"]')) == 't1.M[" cold"," hot"]'


def test_program_and_resolved_id_are_part_of_identity():
    assert A.parse("t1.L0.v~prefill") != A.parse("t1.L0.v~stepped") != A.parse("t1.L0.v")
    assert A.parse('t1.logits[" hot"=5]') != A.parse('t1.logits[" hot"]')


def test_occurrence_and_state_cell_never_equal():
    occurrence, cell = A.parse("t18.L20.v"), A.parse("cache.L20.v[row=18]")
    assert isinstance(occurrence, Occurrence) and isinstance(cell, StateCell)
    assert occurrence != cell and cell != occurrence
    assert occurrence.kind == "occurrence" and cell.kind == "cell"
    assert len({occurrence, cell, A.parse("t18.L20.v")}) == 2
    assert StateCell(20, "v") != Occurrence(0, "v", 20)
    assert not A.matches("cache.L20.v", occurrence) and not A.matches("t18.L20.v", cell)
    with pytest.raises(AddressError, match="no program"):
        A.parse("cache.L20.v~stepped")


@pytest.mark.parametrize("text", PATTERNS + ["cache.L*.v", "cache.L20", "t21", "t21.*",
                                             "s3.L*.q", "@pre:t*.L1.p[h5]", "t1.L1.p~prefill"])
def test_patterns_round_trip(text):
    pattern = A.parse_pattern(text)
    assert A.parse_pattern(A.format_pattern(pattern)) == pattern


WILDCARD_TABLE = [
    ("L*.p", "t21.L20.p[h5,src=18]", True),
    ("L*.p", "t21.L20.o", False),
    ("L*.p", "t3.s12.L4.p[h2]", True),
    ("L*.logits", None, "error"),
    ("t21.L20", "t21.L20.p[h5,src=18]", True),
    ("t21.L20", "t21.L20.f[i=4031]", True),
    ("t21.L20", "t21.L19.p", False),
    ("t21.L20", "t21.logits", False),
    ("t21.L20", "cache.L20.v[row=21]", False),
    ("t21", "t21.logits", True),
    ("t*.L20.o", "t7.L20.o", True),
    ("t*.L20.o", "t7.L20.o[h1]", True),
    ("t*.L20.o", "t7.L21.o", False),
    ("t*.L20.o", "boiling:t7.L20.o", True),
    ("@pre:t*.L20.o", "t7.L20.o", False),
    ("@pre:t*.L20.o", "@pre:t7.L20.o", True),
    ("@pre:t*.L20.o", "@pre+e1:t7.L20.o", False),
    ("s*.L1.p", "t1.L1.p", False),
    ("s*.L1.p", "t1.s2.L1.p", True),
    ("t1.L1.p[h5]", "t1.L1.p[h5,src=2]", True),
    ("t1.L1.p[h5]", "t1.L1.p", False),
    ("t1.L1.p[h5]", "t1.L1.p[h6]", False),
    ("t1.L1.p[h=0:8]", "t1.L1.p[h=2|7]", True),
    ("t1.L1.p[h=0:8]", "t1.L1.p[h=4:9]", False),
    ("t1.L1.p[h=1|2|3]", "t1.L1.p[h=1:3]", True),
    ("t1.L1.p[h=1|3]", "t1.L1.p[h=1:3]", False),
    ("cache.L*.v", "cache.L20.v[kv0,row=18]", True),
    ("cache.L*.v", "cache.L20.k[kv0,row=18]", False),
    ("cache.L20", "cache.L20.k", True),
    ("cache.L20.v[row=18]", "cache.L20.v[kv0,row=18]", True),
    ("cache.L20.v[row=0:18]", "cache.L20.v[kv0,row=18]", False),
    ('t21.M[" hot"]', 't21.M[" hot"," cold"]', False),
    ('t21.M[" hot"," cold"]', 't21.M[" hot"," cold"]', True),
    ("t1.L1.v~prefill", "t1.L1.v", False),
    ("t1.L1.v", "t1.L1.v~prefill", True),
]


@pytest.mark.parametrize("pattern, address, expected", WILDCARD_TABLE)
def test_wildcard_truth_table(pattern, address, expected):
    if expected == "error":
        with pytest.raises(AddressError):
            A.parse_pattern(pattern)
        return
    assert A.matches(pattern, address) is expected
    assert A.matches(A.parse_pattern(pattern), A.parse(address)) is expected


def test_patterns_are_not_addresses():
    for text in PATTERNS:
        with pytest.raises(AddressError):
            A.parse(text)


MALFORMED = [
    ("", 1, "expected 't<row>'"),
    ("t21.L20.p[h5,src=18]x", 21, "unexpected 'x'"),
    ("t21.L20.p]", 10, "unexpected ']'"),
    ("x21.L20.p", 1, "expected 't<row>'"),
    ("t.L20.p", 2, "expected a token row"),
    ("t021.L1.p", 2, "leading zero"),
    ("t21.L020.p", 6, "leading zero"),
    ("t21.L20", 8, "expected '.<port>'"),
    ("t21.L20.", 9, "expected a port name"),
    ("t21.L20.zz", 9, "unknown port 'zz'"),
    ("t21.p", 5, "needs a layer"),
    ("t21.L2.logits", 8, "global and takes no layer"),
    ("t21.L20.s3.p", 9, "order t, s, L"),
    ("t21.L20.p[]", 11, "expected a selector key"),
    ("t21.L20.p[h5", 13, "expected ',' or ']'"),
    ("t21.L20.p[h5,]", 14, "expected a selector key"),
    ("t21.L20.p[q5]", 11, "unknown selector key 'q'"),
    ("t21.L20.p[h5,h6]", 14, "duplicate selector key 'h'"),
    ("t21.L20.p[h]", 12, "expected a value for h"),
    ("t21.L20.p[h=4:4]", 15, "end > start"),
    ("t21.L20.p[h=1|1]", 15, "duplicate set item 1"),
    ("t21.L20.p[h=-1]", 13, "expected a value for h"),
    ('t21.logits[" hot]', 12, "unterminated"),
    ('t21.logits[""]', 12, "non-empty text"),
    ("t21.L20.p[h5, src=18]", 14, "found whitespace"),
    (" t21.L20.p", 1, "found whitespace"),
    ("@:t21.logits", 2, "cut reference"),
    ("@pre+:t21.logits", 6, "edit id"),
    ("@pre t21.logits", 5, "':' after the origin"),
    ("9branch:t21.logits", 1, "starts with a letter"),
    ("cache.L20.q", 11, "k or v"),
    ("cache.L20", 10, "'.k' or '.v'"),
    ("cache.20.v", 7, "'L<layer>'"),
    ('cache.L20.v[h5]', 13, "only kv, row and d"),
    ('cache.L20.v[" a"]', 13, "only kv, row and d"),
    ("t21.L20.p~", 11, "program name"),
    ("t21.L20.p~stepped~x", 18, "unexpected '~'"),
    ("t1.L1.p[h5]x[h6]", 12, "unexpected 'x'"),
]


@pytest.mark.parametrize("text, column, reason", MALFORMED)
def test_malformed_rejected_with_column(text, column, reason):
    with pytest.raises(AddressError) as caught:
        A.parse(text)
    assert caught.value.column == column, str(caught.value)
    assert reason in caught.value.reason
    assert str(caught.value).startswith(f"column {column}: ")
    caret = str(caught.value).splitlines()[-1]
    assert caret.index("^") == column + 1  # two-space indent


def test_port_lists_are_supplied_by_the_caller():
    assert A.parse("t1.L0.zz", ports=("zz",)).port == "zz"
    with pytest.raises(AddressError, match="unknown port 'p'"):
        A.parse("t1.L0.p", ports=("zz",))
    assert A.parse("t1.out", global_ports=("out",)).port == "out"
    with pytest.raises(AddressError, match="read as a clock"):
        A.parse("t1.L0.p", ports=("s1",))
    with pytest.raises(AddressError, match="both layer ports and globals"):
        A.parse("t1.L0.p", ports=("p",), global_ports=("p",))
    with pytest.raises(AddressError, match="unknown port"):
        A.from_json({"kind": "occurrence", "origin": None, "clock": {"token": 1, "layer": 0},
                     "port": "zz", "selector": {}})


def test_from_json_is_strict():
    good = A.parse("t1.L0.p[h=0:2]").to_json()
    for broken in (
        {**good, "extra": 1},
        {**good, "program": None},
        {**good, "kind": "cellar"},
        {**good, "clock": {"token": 1, "layer": 0, "row": 3}},
        {**good, "selector": {"h": [2, 1]}},
        {**good, "selector": {"h": {"start": 2}}},
        {**good, "selector": {"h": True}},
        {**good, "origin": {"branch": "b", "edits": []}},
        {**good, "clock": {"token": -1, "layer": 0}},
    ):
        with pytest.raises(AddressError):
            A.from_json(broken)


def _random_address(rng):
    def ref():
        return rng.choice(["pre", "post", "b1", "x-2", "a_b", "0f" * 32])

    origin = rng.choice([None, Origin(branch=rng.choice(["boiling", "freeze", "b_2"])),
                         Origin(cut=ref(), edits=tuple(ref() for _ in range(rng.randint(0, 3))))])
    keys = []
    pool = list(A.CELL_KEYS) if rng.random() < 0.4 else list(A.SELECTOR_KEYS)
    for key in rng.sample(pool, rng.randint(0, len(pool))):
        roll = rng.random()
        if roll < 0.6:
            value = rng.randint(0, 5000)
        elif roll < 0.8:
            start = rng.randint(0, 100)
            value = Span(start, start + rng.randint(1, 50))
        else:
            value = AnyOf(tuple(rng.sample(range(64), rng.randint(2, 5))))
        keys.append((key, value))
    if pool == list(A.CELL_KEYS) and rng.random() < 0.5:
        return StateCell(rng.randint(0, 80), rng.choice("kv"), Selector(tuple(keys)), origin)
    texts = [" hot", " cold", "é", 'q"uote', "back\\slash", "\n", "∑"]
    tokens = tuple(Token(rng.choice(texts), rng.choice([None, rng.randint(0, 150000)]))
                   for _ in range(rng.randint(0, 2)))
    is_global = rng.random() < 0.3
    port = rng.choice(A.DEFAULT_GLOBALS if is_global else A.DEFAULT_PORTS)
    selector = Selector(tuple(keys), tokens)
    if is_global:   # a global takes only its own selectors (A.GLOBAL_SELECTORS)
        selector = {
            "emb": Selector((("d", rng.randint(0, 4000)),)), "fn": Selector(),
            "sample": Selector(), "logits": rng.choice([
                Selector((("i", rng.randint(0, 150000)),)), Selector(tuple(), tokens[:1])]),
            "M": rng.choice([
                Selector((), (Token(rng.choice(texts[:2])), Token(rng.choice(texts[2:])))),
                # two values of i, often consecutive (i=14|15 must have a constructible key)
                Selector((("i", AnyOf(rng.choice([(14, 15), (7, 8), (0, 1), (3, 9), (20, 21)]))),))]),
        }[port]
    return Occurrence(
        rng.randint(0, 4096), port, None if is_global else rng.randint(0, 80),
        rng.choice([None, rng.randint(0, 50)]), selector, origin,
        rng.choice([None, "stepped", "prefill", "cache-only"]))


def test_generated_addresses_round_trip():
    rng = random.Random(20261112)
    seen = set()
    for _ in range(600):
        address = _random_address(rng)
        text = A.format(address)
        assert A.parse(text) == address, text
        assert A.format(A.parse(text)) == text
        assert A.from_json(json.loads(json.dumps(address.to_json()))) == address
        assert A.matches(A.parse_pattern(text), address), text
        seen.add(text)
    assert len(seen) > 500


# --------------------------------------------------------------------------- parser limits (F6)

@pytest.mark.parametrize("text, column", [
    ("t1000000000000000.L0.p", 2),                 # 16 digits
    ("t1.L1000000000000000.p", 5),
    ("t1.L0.p[h1000000000000000]", 10),
    ("t1.L0.p[h=0:1000000000000000]", 13),
    ("t1.L0.p[h=1|1000000000000000]", 13),
    ('t1.logits[" a"=1000000000000000]', 16),
    ("cache.L1.v[row=1000000000000000]", 16),
])
def test_integers_are_limited_to_15_digits(text, column):
    with pytest.raises(AddressError, match="more than 15 digits") as caught:
        A.parse(text)
    assert caught.value.column == column, str(caught.value)


def test_fifteen_digits_are_accepted_and_exact():
    text = "t999999999999999.L0.p[h999999999999999]"
    address = A.parse(text)
    assert address.token == 10 ** 15 - 1 and A.format(address) == text
    assert json.loads(json.dumps(address.to_json())) == address.to_json()
    with pytest.raises(AddressError, match="more than 15 digits"):
        Occurrence(10 ** 15, "p", 0)
    with pytest.raises(AddressError, match="more than 15 digits"):
        Selector((("h", 10 ** 15),))
    with pytest.raises(AddressError, match="more than 15 digits"):
        Span(0, 10 ** 15)


@pytest.mark.parametrize("text", [
    "t1.L0.p\ud800", "t1.L0.p[h5]\udfff", 't1.logits["\ud800"]', "\ud800t1.L0.p",
    't1.logits["\\ud800"]', 't1.logits["\\udc00x"]',
])
def test_lone_surrogates_are_rejected(text):
    with pytest.raises(AddressError, match="surrogate|UTF-8"):
        A.parse(text)
    with pytest.raises(AddressError):
        A.parse_pattern(text)


def test_unencodable_token_text_is_not_constructible():
    for text in ("\ud800", "a\udfffb"):
        with pytest.raises(AddressError, match="UTF-8"):
            Token(text)
    assert A.parse('t1.logits["\\ud83d\\ude00"]').selector.tokens[0].text == "\U0001F600"


# --------------------------------------------------------------------------- structure (F7)

def _build_attempts(rng):
    """Random (possibly invalid) objects: anything constructible must round trip."""
    def maybe(*options):
        return rng.choice(options)

    names = ["p", "q", "emb", "logits", "M", "h", "h_out", "zz", "q_norm", "t5", "s2", "L9",
             "x_1", "A", "sg", "fn", "sample", "_u", "9x", "", "a-b", "a b"]
    for _ in range(3000):
        kind = rng.choice(["occ", "occ", "cell", "pat", "patcell"])
        try:
            sel_keys = []
            for key in rng.sample(A.SELECTOR_KEYS, rng.randint(0, 4)):
                sel_keys.append((key, maybe(rng.randint(0, 99), Span(3, 3 + rng.randint(0, 3)),
                                            Span(7, 7 + rng.randint(1, 3)))))
            tokens = tuple(Token(rng.choice(["a", " hot", "é"]), maybe(None, 7))
                           for _ in range(rng.randint(0, 3)))
            selector = Selector(tuple(sel_keys), tokens)
            origin = maybe(None, Origin(branch="b"), Origin(cut="c1", edits=("e",)))
            program = maybe(None, "stepped", "x-y")
            if kind == "occ":
                yield Occurrence(rng.randint(0, 9), rng.choice(names),
                                 maybe(None, rng.randint(0, 9)), maybe(None, 3), selector, origin,
                                 program)
            elif kind == "cell":
                cell_keys = tuple((k, v) for k, v in sel_keys if k in A.CELL_KEYS)
                yield StateCell(rng.randint(0, 9), maybe("k", "v", "q"),
                                maybe(selector, Selector(cell_keys)), origin)
            elif kind == "pat":
                yield A.Pattern("occurrence", maybe(None, 3, "*"), maybe(None, 2, "*"),
                                maybe(None, 4, "*"), maybe(None, rng.choice(names)), selector,
                                origin, program)
            else:
                yield A.Pattern("cell", maybe(None, 3), None, maybe(None, 4, "*"),
                                maybe(None, "k", "v", "q"), selector, origin, program)
        except AddressError:
            continue


def test_everything_constructible_round_trips_without_port_lists():
    rng = random.Random(7)
    built = {"occ": 0, "cell": 0, "pat": 0}
    unknown = 0
    for obj in _build_attempts(rng):
        if isinstance(obj, A.Pattern):
            text = A.format_pattern(obj)
            assert A.parse_pattern(text, check_ports=False) == obj, text
            built["pat"] += 1
        else:
            text = A.format(obj)
            assert A.parse(text, check_ports=False) == obj, text
            assert A.from_json(json.loads(json.dumps(obj.to_json())), check_ports=False) == obj
            built["occ" if isinstance(obj, Occurrence) else "cell"] += 1
            unknown += isinstance(obj, Occurrence) and obj.port in ("zz", "q_norm", "x_1", "_u")
    assert min(built.values()) > 100 and unknown > 20, (built, unknown)


@pytest.mark.parametrize("build, message", [
    (lambda: Occurrence(1, "p"), "needs a layer"),
    (lambda: Occurrence(1, "h_out"), "needs a layer"),
    (lambda: Occurrence(1, "logits", 3), "global and takes no layer"),
    (lambda: Occurrence(1, "emb", 0), "global and takes no layer"),
    (lambda: Occurrence(1, "t5", 0), "clock component"),
    (lambda: Occurrence(1, "L9", 0), "clock component"),
    (lambda: Occurrence(1, "9x", 0), "invalid port name"),
    (lambda: Occurrence(1, "a-b", 0), "invalid port name"),
    (lambda: Occurrence(1, "emb", None, None, Selector.of(h=1)), "does not select by 'h'"),
    (lambda: Occurrence(1, "sample", None, None, Selector.of(i=1)), "takes no selector"),
    (lambda: Occurrence(1, "fn", None, None, Selector.of("x")), "no token texts"),
    (lambda: Occurrence(1, "logits", None, None, Selector.of("a", i=1)), "not both"),
    (lambda: Occurrence(1, "M", None, None, Selector.of("a")), "exactly 2"),
    (lambda: Occurrence(1, "M", None, None, Selector.of("a", "b", "c")), "at most 2"),
    (lambda: Occurrence(1, "M", None, None, Selector.of(d=1)), "does not select by 'd'"),
    (lambda: A.Pattern("occurrence", port="emb", layer=3), "takes no layer"),
    (lambda: A.Pattern("occurrence", port="emb", layer="*"), "takes no layer"),
    (lambda: A.Pattern("cell", layer=None), "needs a layer"),
    (lambda: A.Pattern("cell", token=3, layer=1), "only a layer and a tensor"),
    (lambda: A.Pattern("cell", layer=1, port="h"), "k or v"),
    (lambda: A.Pattern("cell", layer=1, selector=Selector.of(h=1)), "only kv, row and d"),
    (lambda: A.Pattern("cell", layer=1, program="x"), "only a layer and a tensor"),
    (lambda: A.Pattern("occurrence", port="t5"), "clock component"),
    (lambda: A.Pattern("thing"), "pattern is an"),
    (lambda: A.Pattern("occurrence", token=-1), "non-negative"),
    (lambda: A.Pattern("occurrence", port="sample", selector=Selector.of(d=1)), "does not select"),
])
def test_structural_rules_hold_for_direct_construction(build, message):
    with pytest.raises(AddressError, match=message):
        build()


def test_parse_applies_the_same_rules_with_and_without_port_lists():
    for text in ("t1.p", "t1.L1.logits", "t1.emb[h5]", "t1.sample[d=1]", 't1.M[" a"]'):
        with pytest.raises(AddressError):
            A.parse(text)
        with pytest.raises(AddressError):
            A.parse(text, check_ports=False)
    assert A.parse("t1.L1.zz[h1]", check_ports=False).port == "zz"
    assert A.parse("t1.zz[h1]", check_ports=False).layer is None   # unknown name: either shape
    with pytest.raises(AddressError, match="unknown port"):
        A.parse("t1.L1.zz")
    with pytest.raises(AddressError, match="clock component"):
        A.parse("t1.L1.s2", check_ports=False)


def test_global_selector_table_is_data_a_caller_can_extend():
    assert A.parse('t1.logits[" a"," b"]').selector.tokens[1].text == " b"
    assert A.parse("t1.logits[i=3]").selector.get("i") == 3
    assert A.parse("t1.M[i=3|9]").selector.get("i") == A.AnyOf((3, 9))
    assert A.parse('t1.emb[d=3]') and A.parse("t1.fn[d=0]") and A.parse("t1.sample")
    assert "kl" not in A.GLOBAL_SELECTORS
    try:
        A.GLOBAL_SELECTORS["kl"] = {"keys": ("i",), "tokens": 0}
        assert A.parse("t2.kl[i=4]", global_ports=("kl",)).selector.get("i") == 4
        with pytest.raises(AddressError, match="does not select by 'd'"):
            A.parse("t2.kl[d=4]", global_ports=("kl",))
        with pytest.raises(AddressError, match="takes no layer"):
            A.Occurrence(1, "kl", 3)
    finally:
        A.GLOBAL_SELECTORS.pop("kl", None)
    assert A.parse("t2.kl[d=4]", global_ports=("kl",)).port == "kl"   # unlisted: any selector


# --------------------------------------------------------------------------- matching oddities

@pytest.mark.parametrize("text, canonical", [
    ("t1.L0.p[h=5:6]", "t1.L0.p[h5]"),
    ("t1.L0.p[src=0:1]", "t1.L0.p[src=0]"),
    ("t1.L0.p[kv=7:8,row=2:3,d=9:10]", "t1.L0.p[kv7,row=2,d=9]"),
    ("cache.L1.v[row=4:5]", "cache.L1.v[row=4]"),
    ("t1.L0.p[h=5:7]", "t1.L0.p[h=5:7]"),
])
def test_a_one_element_range_is_the_integer(text, canonical):
    address = A.parse(text)
    assert A.format(address) == canonical and address == A.parse(canonical)
    assert A.matches(canonical.split("[")[0], address)


@pytest.mark.parametrize("text", ["t1.L0.p[h=4:4]", "t1.L0.p[h=5:4]", "t1.L0.p[h=0:0]"])
def test_empty_and_reversed_ranges_are_rejected(text):
    with pytest.raises(AddressError, match="end > start"):
        A.parse(text)
    with pytest.raises(AddressError):
        Span(4, 4)
    with pytest.raises(AddressError):
        Span(5, 4)


def test_single_element_range_in_a_pattern_equals_the_integer():
    assert A.parse_pattern("L*.p[h=5:6]") == A.parse_pattern("L*.p[h5]")
    assert A.matches("t1.L1.p[h=5:7]", "t1.L1.p[h5]") and A.matches("L*.p[h5]", "t1.L1.p[h=5:6]")


def test_token_selectors_compare_ids_when_both_sides_carry_one():
    assert A.matches('t1.logits[" hot"=5]', 't1.logits[" hot"=5]')
    assert not A.matches('t1.logits[" hot"=5]', 't1.logits[" hot"=6]')     # same text, other id
    assert A.matches('t1.logits["hot"=5]', 't1.logits[" hot"=5]')           # same id, other text
    assert A.matches('t1.logits[" hot"]', 't1.logits[" hot"=6]')            # id missing on one side
    assert A.matches('t1.logits[" hot"=5]', 't1.logits[" hot"]')
    assert not A.matches('t1.logits[" hot"]', 't1.logits[" cold"=6]')       # text compared
    assert not A.matches('t1.logits[" hot"]', 't1.logits[" hot"," cold"]')  # count differs
    assert A.matches('t1.M[" a"=1," b"]', 't1.M[" x"=1," b"=9]')   # 1 == 1; " b" == " b" (id missing)
    assert not A.matches('t1.M[" a"=1," b"]', 't1.M[" a"=2," b"]')
    assert not A.matches('t1.M[" a"=1," b"]', 't1.M[" x"=1," c"=9]')


def test_matches_stop_ignores_selectors_and_nothing_else():
    assert A.matches_stop("L20.p[h5]", "t21.L20.p")
    assert not A.matches("L20.p[h5]", "t21.L20.p")
    assert A.matches_stop("L20.p[h5]", "t21.L20.p[h9,src=1]")
    assert A.matches_stop("L20.p", "t21.L20.p[h9]")
    assert A.matches_stop('t21.M[" hot"]', 't21.M[" a"," b"]')
    assert A.matches_stop("cache.L20.v[row=18]", "cache.L20.v[kv0]")
    for pattern, address in [("L20.p[h5]", "t21.L20.o"), ("L21.p[h5]", "t21.L20.p"),
                             ("t22.L20.p[h5]", "t21.L20.p"), ("@pre:L20.p", "t21.L20.p"),
                             ("L20.p~prefill", "t21.L20.p"), ("L*.p[h5]", "t21.logits"),
                             ("cache.L20.v", "t21.L20.p"), ("s*.L1.p", "t1.L1.p[h1]")]:
        assert not A.matches_stop(pattern, address), (pattern, address)
    assert A.matches_stop(A.parse_pattern("L20.p[h5]"), A.parse("t21.L20.p"))
    assert "matches_stop" in A.matches.__doc__ and "matches" in A.matches_stop.__doc__


# --------------------------------------------------------------------------- one key per value

def test_value_key_and_aliases():
    L = 4
    key = lambda text: A.format(A.value_key(text, layers=L))          # noqa: E731
    assert key("t7.emb") == "t7.L0.h"
    assert key("t7.emb[d=3]") == "t7.L0.h[d=3]"
    assert key("t7.L0.h_out") == "t7.L1.h"
    assert key("t7.L2.h_out[d=3]") == "t7.L3.h[d=3]"
    assert key("t7.L3.h_out") == "t7.L3.h_out"                       # the last layer stays
    assert key("t7.L3.h") == "t7.L3.h" and key("t7.L1.h[d=2]") == "t7.L1.h[d=2]"
    assert key("boiling:t7.L0.h_out~prefill") == "boiling:t7.L1.h~prefill"
    assert key("t7.L1.p") == "t7.L1.p" and key('t7.logits[" a"]') == 't7.logits[" a"]'
    assert key("cache.L1.k[row=2]") == "cache.L1.k[row=2]"
    for bad in ("t7.L4.h", "t7.L4.h_out", "t7.L9.h"):
        with pytest.raises(AddressError, match="does not exist"):
            A.value_key(bad, layers=L)
    with pytest.raises(ValueError):
        A.value_key("t7.emb", layers=0)
    for text in ("t7.emb", "t7.L0.h", "t7.L0.h_out", "t7.L2.h_out[d=3]", "t7.L3.h_out",
                 "t7.L2.p"):
        spelled = [A.format(a) for a in A.aliases(text, layers=L)]
        assert spelled[0] == key(text)
        assert key(text) == key(spelled[-1]) and all(key(x) == spelled[0] for x in spelled)
    assert [A.format(a) for a in A.aliases("t7.L2.h", layers=L)] == ["t7.L2.h", "t7.L1.h_out"]
    assert [A.format(a) for a in A.aliases("t7.emb", layers=L)] == ["t7.L0.h", "t7.emb"]
    assert [A.format(a) for a in A.aliases("t7.L3.h_out", layers=L)] == ["t7.L3.h_out"]
    assert [A.format(a) for a in A.aliases("t7.L2.p", layers=L)] == ["t7.L2.p"]


# --------------------------------------------------------------------------- saturn_pub mapping

def _point(step, phase, index=None):
    from saturn_pub.contracts import ExecutionPoint

    return ExecutionPoint("qwen2", step, phase, phase, index, "before", "next",
                          0 if index is None else index)


FIRST = {"first_step": 0}      # the session's first stepped row (sample/logits need it)


def _point_for(text, granularity="layer"):
    clock = A.state_clock(text, layers=32, granularity=granularity)
    return _point(clock["logical_step"], clock["phase"], clock["index"])


def test_state_address_table():
    from saturn_pub.contracts import StateAddress

    cases = [
        ("cache.L20.v[kv0,row=18]", StateAddress("values", (20, 0, 0, 18), "cache")),
        ("cache.L3.k", StateAddress("keys", (3, 0), "cache")),
        ("cache.L3.k[kv1,row=2,d=5]", StateAddress("keys", (3, 0, 1, 2, 5), "cache")),
        ("t4.emb", StateAddress("hidden", (), "carrier")),
        ("t4.L2.h[d=7]", StateAddress("hidden", (0, 0, 7), "carrier")),
        ("t4.L2.h_out", StateAddress("hidden", (), "carrier")),
        ("t4.L5.h_out", StateAddress("hidden", (), "carrier")),
        ("t4.logits[i=11]", StateAddress("logits", (0, 11), "observation")),
        ('t4.logits[" hot"=11]', StateAddress("logits", (0, 11), "observation")),
        ("t4.sample", StateAddress("tokens", (0, 5), "cursor")),
    ]
    for text, expected in cases:
        point = None if text.startswith("cache") else _point_for(text)
        got = A.to_state_address(text, layers=32, point=point, limits=FIRST)
        assert (got.slot, got.selector, got.role) == (expected.slot, expected.selector,
                                                      expected.role), text
        assert got.execution_point == (None if point is None else point.fingerprint)
    clock = lambda text, **kw: A.state_clock(text, layers=6, **kw)    # noqa: E731
    assert clock("t4.L2.h") == {"logical_step": 4, "phase": "layer", "index": 2}
    assert clock("t4.L5.h") == {"logical_step": 4, "phase": "layer", "index": 5}
    assert clock("t4.L2.h_out") == {"logical_step": 4, "phase": "layer", "index": 3}
    assert clock("t4.L5.h_out") == {"logical_step": 4, "phase": "layer", "index": 6}
    assert clock("t4.L5.h_out", granularity="operation") == {
        "logical_step": 4, "phase": "normalization", "index": None}
    assert clock("t4.logits") == {"logical_step": 5, "phase": "embed", "index": None}
    assert clock("t4.logits", granularity="operation") == {
        "logical_step": 4, "phase": "sample", "index": None}
    assert clock("cache.L1.v") is None
    assert A.state_clock("t4.emb") == {"logical_step": 4, "phase": "layer", "index": 0}
    assert A.to_state_address("t4.sample", granularity="operation", limits=FIRST,
                              point=_point(4, "commit")).slot == "sampled_token"
    assert A.to_state_address("t4.fn", granularity="operation",
                              point=_point(4, "readout")).slot == "hidden"


def test_hidden_conversions_need_the_layer_count_and_refuse_missing_layers():
    for call in (lambda: A.to_state_address("t4.L2.h", point=_point(4, "layer", 2)),
                 lambda: A.state_clock("t4.L2.h_out"),
                 lambda: A.from_state_address(_state("hidden", ()), _point(4, "layer", 2))):
        with pytest.raises(ValueError, match="needs layers="):
            call()
    for bad_layers in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            A.to_state_address("t4.L0.h", layers=bad_layers, point=_point(4, "layer", 0))
    for text in ("t4.L6.h", "t4.L6.h_out", "t4.L99.h"):
        with pytest.raises(NoStateSlot, match="does not exist"):
            A.to_state_address(text, layers=6, point=_point(4, "layer", 0))
        with pytest.raises(NoStateSlot, match="does not exist"):
            A.to_state_address(text, layers=6, granularity="operation",
                               point=_point(4, "layer", 0))
    with pytest.raises(NoStateSlot, match="does not exist"):
        A.to_state_address("cache.L6.k", layers=6)
    assert A.to_state_address("cache.L5.k", layers=6).selector == (5, 0)


def _state(slot, selector, **kw):
    from saturn_pub.contracts import StateAddress

    return StateAddress(slot, selector, **kw)


@pytest.mark.parametrize("text", [
    "t21.L20.p[h5,src=18]", "t21.L20.v", "t21.L20.q", "t21.L20.f[i=4031]", "t21.L20.o",
    "t21.M[\" hot\",\" cold\"]", "t3.s12.L4.h", "t4.L2.h~prefill", "t4.L2.h[h1]",
    "t4.logits[\" hot\"]", "t4.logits[i=0:5]", "cache.L1.v[row=3]", "cache.L1.v[kv=0:2]",
    "t4.fn",
])
def test_addresses_without_a_state_slot(text):
    with pytest.raises(NoStateSlot):
        A.to_state_address(text, layers=6, point=_point(4, "layer", 2))


def test_state_address_binds_and_checks_the_clock():
    point = _point(4, "layer", 2)
    bound = A.to_state_address("t4.L2.h[d=1]", point=point, state_schema="s", layers=6)
    assert bound.execution_point == point.fingerprint and bound.state_schema == "s"
    with pytest.raises(NoStateSlot, match="not in state"):
        A.to_state_address("t4.L3.h", point=point, layers=6)
    assert A.to_state_address("cache.L0.k[kv0]", point=point).execution_point == point.fingerprint
    # h_out of the last layer: layer granularity holds it at index L, operation at normalization
    assert A.to_state_address("t4.L5.h_out", point=_point(4, "layer", 6), layers=6)
    assert A.to_state_address("t4.L5.h_out", point=_point(4, "normalization"), layers=6,
                              granularity="operation")
    with pytest.raises(NoStateSlot, match="not in state"):
        A.to_state_address("t4.L5.h_out", point=_point(4, "layer", 5), layers=6)


def test_from_state_address_inverts_the_table():
    layer = _point(4, "layer", 2)
    expect = {
        ("hidden", (0, 0, 7), layer): "t4.L2.h[d=7]",
        ("hidden", (), _point(4, "layer", 0)): "t4.L0.h",
        ("hidden", (), _point(4, "layer", 5)): "t4.L5.h",
        ("hidden", (), _point(4, "layer", 6)): "t4.L5.h_out",
        ("hidden", (0, 0, 1), _point(4, "layer", 6)): "t4.L5.h_out[d=1]",
        ("values", (20, 0, 0, 18), _point(18, "layer", 2)): "cache.L20.v[kv0,row=18]",
        ("logits", (0, 11), _point(5, "embed")): "t4.logits[i=11]",
        ("tokens", (0, 5), _point(5, "embed")): "t4.sample",
    }
    for (slot, selector, point), text in expect.items():
        assert A.from_state_address(_state(slot, selector), point, limits=FIRST,
                                    layers=32 if slot == "values" else 6) == A.parse(text)
    assert A.from_state_address(_state("hidden", ()), _point(4, "normalization"), layers=6,
                                granularity="operation") == A.parse("t4.L5.h_out")
    assert A.from_state_address(_state("hidden", (0, 0, 3)), _point(4, "readout"), layers=6,
                                granularity="operation") == A.parse("t4.fn[d=3]")
    assert A.from_state_address(_state("logits", (0, 3)), _point(4, "sample"), layers=6,
                                granularity="operation", limits=FIRST) == A.parse("t4.logits[i=3]")
    assert A.from_state_address(_state("sampled_token", (0, 0)), _point(4, "commit"), layers=6,
                                granularity="operation", limits=FIRST) == A.parse("t4.sample")
    for text, at in (("t4.L2.h[d=7]", layer), ("cache.L20.v[kv0,row=18]", _point(18, "layer", 2))):
        address = A.parse(text)
        assert A.from_state_address(A.to_state_address(address, layers=32, point=at), at,
                                    layers=32) == address
    with pytest.raises(NoStateSlot, match="another execution point"):
        A.from_state_address(_state("hidden", (), execution_point="0" * 64), layer, layers=6)
    with pytest.raises(NoStateSlot):
        A.from_state_address(_state("layer", ()), layer, layers=6)
    # refusals: wrong phase, wrong selector shape, a layer that does not exist, a wrong role
    for slot, selector, point, kw in [
            ("hidden", (), _point(4, "embed"), {}),
            ("hidden", (), _point(4, "layer", 7), {}),
            ("hidden", (), _point(4, "readout"), {}),                       # layer granularity
            ("hidden", (), _point(4, "normalization"), {}),                 # layer granularity
            ("hidden", (0,), layer, {}), ("hidden", (0, 0), layer, {}), ("hidden", (1, 0, 2), layer, {}),
            ("logits", (0, 3), layer, {}), ("logits", (1, 3), _point(5, "embed"), {}),
            ("logits", (0, 3), _point(0, "embed"), {}),                     # no token before row 0
            ("tokens", (0, 5), layer, {}), ("tokens", (0, 4), _point(5, "embed"), {}),
            ("tokens", (0, 5), _point(5, "embed"), {"granularity": "operation"}),
            ("sampled_token", (0, 0), _point(4, "commit"), {}),             # layer granularity
            ("sampled_token", (), _point(4, "commit"), {"granularity": "operation"}),
            ("sampled_token", (0, 0), _point(4, "sample"), {"granularity": "operation"}),
            ("keys", (3,), layer, {}), ("keys", (3, 1), layer, {}), ("keys", (6, 0), layer, {}),
            ("values", (3, 0, 1, 2, 3, 4), layer, {}), ("values", (3, 0, -1), layer, {})]:
        with pytest.raises(NoStateSlot):
            A.from_state_address(_state(slot, selector), point, layers=6, **kw)
    with pytest.raises(NoStateSlot, match="role"):
        A.from_state_address(_state("hidden", (), role="cache"), layer, layers=6)


# ----------------------------------------------- every mappable address against a real Session

def _walk_session(family, granularity, prompt=(3, 5, 7, 9), generated=3):
    """Walk a tiny decoder Session through ``generated`` tokens; at every clock convert every
    address of the table and compare ``Session.read_port`` with direct indexing of the state.

    Returns the per-category counts and the values read (for the independent forward)."""
    import torch

    from saturn_pub.adapters.decoder import DecoderAdapter

    adapter = DecoderAdapter.tiny(family, granularity=granularity)
    L, P, G = adapter.layers, len(prompt), generated
    T = P + G
    session = adapter.session(list(prompt))
    rows = range(P - 1, T - 1)
    addrs = []
    for n in rows:
        addrs += [f"t{n}.emb[d=2]", f"t{n}.emb", f"t{n}.fn[d=2]", f"t{n}.fn",
                  f"t{n}.logits[i=17]", f"t{n}.logits[i=3]", f't{n}.logits[" x"=17]',
                  f"t{n}.logits", f"t{n}.sample"]
        for k in range(L):
            addrs += [f"t{n}.L{k}.h[d=2]", f"t{n}.L{k}.h", f"t{n}.L{k}.h_out[d=2]",
                      f"t{n}.L{k}.h_out"]
        addrs += [f"t{n}.L{L}.h", f"t{n}.L{L}.h_out[d=2]", f"t{n}.L{L + 3}.h"]   # no such layer
    cells = []
    for k in range(L):
        for tensor, kv in (("k", 1), ("v", 1)):
            cells += [f"cache.L{k}.{tensor}", f"cache.L{k}.{tensor}[kv{kv}]",
                      f"cache.L{k}.{tensor}[kv{kv},row=2]",
                      f"cache.L{k}.{tensor}[kv{kv},row=2,d=4]"]
            cells += [f"cache.L{k}.{tensor}[kv{kv},row={r}]" for r in range(T + 1)]
            cells += [f"cache.L{k}.{tensor}[kv{kv},row={r},d=4]" for r in range(T + 1)]
    count = {"converted": {}, "read": {}, "refused": 0, "cache_converted": 0, "cache_read": 0,
             "cache_refused_by_state": 0, "cache_refused_by_row": 0}
    seen = []
    clocks = []
    point = None
    while True:
        state = session._state
        point = adapter.point(state)
        tokens = state["tokens"].shape[1]
        clocks.append((point, tokens, state["hidden"] is not None, state["layer"]))
        for text in addrs:
            address = A.parse(text)
            try:
                sa = A.to_state_address(address, point=point, granularity=granularity, layers=L,
                                         limits={"first_step": P - 1})
            except NoStateSlot:
                count["refused"] += 1
                continue
            category = address.port if address.port not in ("h_out",) else "h_out"
            count["converted"][category] = count["converted"].get(category, 0) + 1
            expected = _direct(address, state, granularity)
            if expected is None:
                with pytest.raises(ValueError):
                    session.read_port(sa)
                continue
            got = session.read_port(sa)
            assert torch.equal(got.reshape(-1), expected.reshape(-1)), (text, point)
            count["read"][category] = count["read"].get(category, 0) + 1
            seen.append((address, got, point))
            # the inverse reads the same value back at the same point
            back = A.from_state_address(sa, point, layers=L, granularity=granularity,
                                         limits={"first_step": P - 1})
            assert back == A.state_canonical(address, layers=L), (text, back)
        for text in cells:
            address = A.parse(text)
            try:
                sa = A.to_state_address(address, point=point, granularity=granularity, layers=L)
            except NoStateSlot:
                assert address.selector.get("row") > point.logical_step   # C10: row beyond the step
                count["cache_refused_by_row"] += 1
                continue
            count["cache_converted"] += 1
            expected = _direct(address, state, granularity)
            if expected is None:
                with pytest.raises(ValueError):
                    session.read_port(sa)
                count["cache_refused_by_state"] += 1
                continue
            assert torch.equal(session.read_port(sa).reshape(-1), expected.reshape(-1)), (text, point)
            assert A.from_state_address(sa, point, layers=L) == address
            count["cache_read"] += 1
        if tokens >= T and (granularity == "layer" and state["hidden"] is None
                            or granularity == "operation" and state["phase"] == "embed"):
            break
        session.continue_(1)
    return adapter, session, count, seen, clocks, (L, P, G, T)


def _direct(address, state, granularity):
    """The element the address names, by plain indexing of the decoder state (None: absent)."""
    if isinstance(address, StateCell):
        value = (state["keys"] if address.tensor == "k" else state["values"])[address.layer][0]
        try:
            for key in ("kv", "row", "d"):
                index = address.selector.get(key)
                if index is None:
                    break
                value = value[index]
        except IndexError:
            return None
        return value
    port, d = address.port, address.selector.get("d")
    if port in ("emb", "h", "h_out", "fn"):
        value = state["hidden"][0, 0]
        return value if d is None else value[d]
    if port == "logits":
        if state["logits"] is None:
            return None
        index = address.selector.get("i")
        if index is None and address.selector.tokens:
            index = address.selector.tokens[0].id
        return state["logits"][0] if index is None else state["logits"][0, index]
    if granularity == "operation":
        return None if state["sampled_token"] is None else state["sampled_token"][0, 0]
    return state["tokens"][0, address.token + 1]


def _expected_counts(granularity, L, G, clocks, T):
    op = granularity == "operation"
    per_row = {"emb": 2, "h": 2 * L, "h_out": 2 * L, "fn": 2 if op else 0, "logits": 4, "sample": 1}
    converted = {k: v * G for k, v in per_row.items() if v}
    cache_rows = 0
    cache_converted = 0
    refused_by_row = 0
    for point, tokens, has_hidden, layer in clocks:
        refused_by_row += L * 2 * 2 * max(0, T - point.logical_step)
        for k in range(L):
            written = tokens - 1 + (1 if has_hidden and k < layer else 0)
            for _tensor in range(2):
                cache_converted += 1 + 1 + 1 + (T + 1) * 2 + 1 + 0
                cache_rows += 1 + 1 + (1 if written > 2 else 0) * 2 + 2 * min(T + 1, written)
    return converted, cache_converted - refused_by_row, cache_rows, refused_by_row


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
@pytest.mark.parametrize("granularity", ["layer", "operation"])
def test_every_mappable_address_reads_exactly_the_named_element(family, granularity):
    torch = pytest.importorskip("torch")
    adapter, session, count, seen, clocks, (L, P, G, T) = _walk_session(family, granularity)
    converted, cache_converted, cache_read, by_row = _expected_counts(granularity, L, G, clocks, T)
    # the count the table predicts, so an empty loop cannot pass
    assert count["converted"] == converted, count
    assert count["read"] == converted, count
    assert count["cache_converted"] == cache_converted > 0
    assert count["cache_refused_by_row"] == by_row > 0
    assert count["cache_read"] == cache_read > 0
    assert count["cache_refused_by_state"] == cache_converted - cache_read
    # every address either converts at a clock or is refused there (a layer that does not exist,
    # fn at layer granularity and every other clock refuse): nothing is silently skipped
    per_row = 2 + 4 * L + 2 + 4 + 1 + 3
    assert count["refused"] == len(clocks) * G * per_row - sum(converted.values())
    # independent truth: a full forward over the final tokens with hooks on the same model
    tokens = session._state["tokens"]
    layer_in, layer_out, norm_out, hooks = {}, {}, {}, []
    for i, module in enumerate(adapter._layer_modules):
        hooks.append(module.register_forward_pre_hook(
            lambda m, args, i=i: layer_in.__setitem__(i, args[0].detach().clone())))
        hooks.append(module.register_forward_hook(
            lambda m, args, out, i=i: layer_out.__setitem__(
                i, (out[0] if isinstance(out, tuple) else out).detach().clone())))
    hooks.append(getattr(adapter.backbone, adapter.spec.final_norm).register_forward_hook(
        lambda m, a, o: norm_out.__setitem__(0, o.detach().clone())))
    with torch.no_grad():
        out = adapter.model(tokens, use_cache=True)
    for hook in hooks:
        hook.remove()
    checked = 0
    for address, got, point in seen:
        n, port, d = address.token, address.port, address.selector.get("d")
        if port == "emb":
            truth = layer_in[0][0, n]
        elif port == "h":
            truth = layer_in[address.layer][0, n]
        elif port == "h_out":
            truth = layer_out[address.layer][0, n]
        elif port == "fn":
            truth = norm_out[0][0, n]
        elif port == "logits":
            index = address.selector.get("i")
            if index is None and address.selector.tokens:
                index = address.selector.tokens[0].id
            truth = out.logits[0, n] if index is None else out.logits[0, n, index]
            d = None
        else:
            truth = tokens[0, n + 1]
            d = None
        if d is not None:
            truth = truth[d]
        assert got.numel() == truth.numel(), (address, got.shape, truth.shape)
        assert torch.allclose(got.float().reshape(-1), truth.float().reshape(-1), atol=1e-4,
                              rtol=1e-4), str(address)
        checked += 1
    assert checked == sum(converted.values())
    # the cache at the final clock equals the independent forward's cache (every layer and row)
    pkv = out.past_key_values
    final = adapter.point(session._state)
    for k in range(L):
        for tensor, source in (("k", pkv.layers[k].keys), ("v", pkv.layers[k].values)):
            sa = A.to_state_address(f"cache.L{k}.{tensor}[kv1,row={T - 2},d=4]", point=final,
                                    granularity=granularity, layers=L)
            assert torch.allclose(session.read_port(sa).float(), source[0, 1, T - 2, 4].float(),
                                  atol=1e-4)
    # inverse exactness: accepted exactly when some address of the table produces it
    L_, produced = L, {}
    forward_list = []
    for n in range(P - 2, T - 1):
        forward_list += [f"t{n}.emb[d=2]", f"t{n}.emb", f"t{n}.fn[d=2]", f"t{n}.fn",
                         f"t{n}.logits[i=17]", f"t{n}.logits[i=3]", f't{n}.logits[" x"=17]',
                         f"t{n}.logits", f"t{n}.sample"]
        for k in range(L):
            forward_list += [f"t{n}.L{k}.h[d=2]", f"t{n}.L{k}.h", f"t{n}.L{k}.h_out[d=2]",
                             f"t{n}.L{k}.h_out"]
    for k in range(L):
        for tensor in "kv":
            forward_list += [f"cache.L{k}.{tensor}", f"cache.L{k}.{tensor}[kv1]",
                             f"cache.L{k}.{tensor}[kv1,row=2]", f"cache.L{k}.{tensor}[kv1,row=2,d=4]"]
    for point, *_ in clocks:
        produced[point] = set()
        for text in forward_list:
            try:
                sa = A.to_state_address(text, point=point, granularity=granularity, layers=L,
                                        limits={"first_step": P - 1})
            except NoStateSlot:
                continue
            produced[point].add((sa.slot, sa.selector))
    universe = [("hidden", s) for s in [(), (0, 0, 2), (0,), (0, 0), (1, 0, 2), (0, 0, 2, 3)]]
    universe += [("logits", s) for s in [(), (0, 17), (0, 3), (1, 17), (17,), (0, 17, 1)]]
    universe += [("tokens", (0, n + 1)) for n in range(P - 2, T - 1)] + [
        ("tokens", s) for s in [(0, T + 5), (1, 2), (0,), (0, 1, 2)]]
    universe += [("sampled_token", s) for s in [(0, 0), (), (0, 1)]]
    universe += [(slot, s) for slot in ("keys", "values") for k in range(L_)
                 for s in [(k, 0), (k, 0, 1), (k, 0, 1, 2), (k, 0, 1, 2, 4), (k,), (k, 1),
                           (k, 0, 1, 2, 4, 5), (L_, 0), (k, 0, -1)]]
    sampled_checked = 0
    for point, *_ in clocks:
        for slot, selector in universe:
            try:
                A.from_state_address(_state(slot, selector), point, layers=L,
                                     granularity=granularity, limits={"first_step": P - 1})
                accepted = True
            except NoStateSlot:
                accepted = False
            producible = (slot, selector) in produced[point]
            assert accepted == producible, (slot, selector, point)
            sampled_checked += 1
    assert sampled_checked == len(clocks) * len(universe) > 0


# --------------------------------------------------------------------------- round 3 (C4, C5, C8, C10)

def test_c4_an_unbound_occurrence_conversion_is_impossible():
    """The auditor's case: ``t3.L0.h`` converted without ``point=`` read ``t3.L1.h`` later."""
    for text in ("t3.L0.h", "t3.emb", "t3.logits", "t3.sample", "t3.L1.h_out[d=2]"):
        with pytest.raises(ValueError, match="point= is required to bind an occurrence address"):
            A.to_state_address(text, layers=4)
    with pytest.raises(ValueError, match="point= is required"):
        A.to_state_address(A.parse("t3.L0.h"), layers=4, granularity="operation")
    # the unbound clock is still computed without binding
    assert A.state_clock("t3.L0.h", layers=4) == {"logical_step": 3, "phase": "layer", "index": 0}
    # a bound conversion at the right clock reads the named element; at another clock it refuses
    point = _point(3, "layer", 0)
    assert A.to_state_address("t3.L0.h", layers=4, point=point).execution_point == point.fingerprint
    with pytest.raises(NoStateSlot, match="not in state"):
        A.to_state_address("t3.L0.h", layers=4, point=_point(3, "layer", 1))
    # unmappable addresses stay NoStateSlot (not ValueError) with or without a point
    with pytest.raises(NoStateSlot):
        A.to_state_address("t3.L0.q", layers=4)
    # a cache cell has no clock: it converts without a point
    assert A.to_state_address("cache.L1.k[kv0]").execution_point is None


def _clocklike(name):
    return name[0] in "tsL" and name[1:2].isdigit()


def test_c5_clock_like_port_names_are_refused_everywhere():
    for name in ("s1x", "L2norm", "t5a", "s0_", "t5", "L9", "s12", "t0z9"):
        with pytest.raises(AddressError, match="clock component"):
            Occurrence(0, name, 1)
        with pytest.raises(AddressError, match="clock component"):
            A.parse("t0.L1.p", ports=(name,), global_ports=())
        with pytest.raises(AddressError, match="clock component"):
            A.parse("t0.L1.p", ports=("p",), global_ports=(name,))
        with pytest.raises(AddressError, match="clock component"):
            A.Pattern("occurrence", port=name)
    for name in ("Lx", "s", "t", "L", "sg", "t_5", "s_1", "Lx2"):   # names, not clock components
        address = Occurrence(0, name, 1)
        assert A.parse(A.format(address), check_ports=False) == address


def test_c5_everything_constructible_round_trips_over_20k_cases():
    rng = random.Random(20261113)
    alphabet = "tsL0123456789xa_ -"
    names = ["p", "h", "q_norm", "s1x", "L2norm", "t5a", "s0_", "t5", "Lx", "s", "emb", "logits"]
    constructed = refused_clocklike = 0
    for _ in range(24000):
        if rng.random() < 0.5:
            name = rng.choice(names)
        else:
            name = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6)))
        layer = rng.choice([None, 0, 3])
        try:
            address = Occurrence(rng.randint(0, 99), name, layer)
        except AddressError:
            refused_clocklike += _clocklike(name)
            continue
        assert not _clocklike(name), name
        constructed += 1
        text = A.format(address)
        assert A.parse(text, check_ports=False) == address, text
        assert A.format(A.parse(text, check_ports=False)) == text
        assert A.from_json(json.loads(json.dumps(address.to_json())), check_ports=False) == address
    assert constructed > 2000 and refused_clocklike > 2000, (constructed, refused_clocklike)


def test_c8_one_value_one_key():
    L = 4
    key = lambda text: A.format(A.value_key(text, layers=L))          # noqa: E731
    one = {key(t) for t in ('t0.logits["x"=5]', 't0.logits["y"=5]', "t0.logits[i=5]",
                            "t0.logits[i5]")}
    assert one == {"t0.logits[i=5]"}
    assert key("t0.L1.q[h=1:3]") == key("t0.L1.q[h=1|2]") == "t0.L1.q[h=1:3]"
    assert key("t0.L1.q[h=1|2|3]") == "t0.L1.q[h=1:4]"
    assert key("t0.L1.q[h=1|3]") == "t0.L1.q[h=1|3]"                 # not consecutive: a set
    assert key("cache.L1.k[kv=0|1]") == "cache.L1.k[kv=0:2]"
    assert key('t0.logits[" a"]') == 't0.logits[" a"]'               # unresolved token: kept
    assert key('t0.M[" a"=1," b"=2]') == 't0.M[" a"=1," b"=2]'       # an ordered contrast: kept
    for text in ("t0.logits[i=5]", "t0.L1.q[h=1:3]", "t0.L1.h[h5]", "cache.L1.k[kv0]"):
        once = A.value_key(text, layers=L)
        assert A.value_key(once, layers=L) == once
    # state_canonical still equals what the state round trip returns
    assert A.state_canonical('t4.logits["a"=5]', layers=L) == A.parse("t4.logits[i=5]")


def test_c8_aliases_do_not_raise_on_valid_addresses_with_selectors():
    L = 4
    texts = ["t12.L0.h[i=28069000079599,src=24:29]", "t12.L0.h[h5]", 't12.L0.h["a"]',
             "t12.L0.h[d=3]", "t12.L0.h[h=1|2,d=3]", "t7.emb[d=3]", "t7.emb[d=1:3]",
             "t7.L2.h[h5]", "t7.L2.h[i=4]", "t7.L2.h_out[h5]", 't7.L3.h_out["a"]',
             "s2:t5.L0.h[src=3]~prefill"]
    for text in texts:
        als = A.aliases(text, layers=L)
        key = A.value_key(text, layers=L)
        assert als[0] == key
        assert all(A.value_key(a, layers=L) == key for a in als)
    assert [A.format(a) for a in A.aliases("t7.emb[d=1|2]", layers=L)] == [
        "t7.L0.h[d=1:3]", "t7.L0.h[d=1|2]", "t7.emb[d=1:3]", "t7.emb[d=1|2]"]
    # carried where valid, canonical only where the selector means nothing on the alias target
    assert [A.format(a) for a in A.aliases("t12.L0.h[d=3]", layers=L)] == [
        "t12.L0.h[d=3]", "t12.emb[d=3]"]
    assert [A.format(a) for a in A.aliases("t12.L0.h[h5]", layers=L)] == ["t12.L0.h[h5]"]
    assert [A.format(a) for a in A.aliases("t7.L2.h[h5]", layers=L)] == [
        "t7.L2.h[h5]", "t7.L1.h_out[h5]"]


def test_consecutive_m_set_has_a_constructible_canonical_form():
    """Audit round 3: ``t3.M[i=14|15]`` is valid but value_key, aliases and state_canonical
    raised (the range spelling no longer counted as exactly 2)."""
    a = A.parse("t3.M[i=14|15]")
    for layers in (1, 2, 24):
        assert A.format_address(A.value_key(a, layers=layers)) == "t3.M[i=14|15]"
        assert A.aliases(a, layers=layers)[0] == A.value_key(a, layers=layers)
        assert A.state_canonical(a, layers=layers) == a
    assert A.format_address(A.value_key("t3.M[i=14|16]", layers=2)) == "t3.M[i=14|16]"


def test_sample_and_logits_of_a_row_the_session_never_stepped_are_refused():
    """Audit round 3: at layer granularity ``t{P-2}.sample`` mapped to tokens[0, P-1], the
    teacher-forced last prompt token, and read as if sampled. Both directions refuse a row
    below the session's first step, and refuse sample/logits when it is not supplied."""
    first = {"first_step": 3}                      # prompt of 4: rows 0..2 prefilled
    for gran, sample_pt, logits_pt in (
            ("layer", lambda row: _point(row + 1, "embed"), lambda row: _point(row + 1, "embed")),
            ("operation", lambda row: _point(row, "commit"), lambda row: _point(row, "sample"))):
        for text in ("t{}.sample", "t{}.logits", "t{}.logits[i=2]"):
            port = "sample" if "sample" in text else "logits"
            at = (sample_pt if port == "sample" else logits_pt)
            ok, bad = at(3), at(2)
            assert A.to_state_address(text.format(3), point=ok, granularity=gran, layers=2,
                                      limits=first)
            with pytest.raises(NoStateSlot, match="never stepped"):
                A.to_state_address(text.format(2), point=bad, granularity=gran, layers=2,
                                   limits=first)
            with pytest.raises(NoStateSlot, match="first_step"):          # not supplied: refuse
                A.to_state_address(text.format(3), point=ok, granularity=gran, layers=2)
            with pytest.raises(NoStateSlot, match="first_step"):
                A.to_state_address(text.format(3), point=ok, granularity=gran, layers=2,
                                   limits={"vocab": 10})
    # the inverse: the same pairs
    inv = [("layer", "tokens", (0, 4), _point(4, "embed")), ("layer", "logits", (0, 1), _point(4, "embed")),
           ("operation", "sampled_token", (0, 0), _point(3, "commit")),
           ("operation", "logits", (0, 1), _point(3, "sample"))]
    for gran, slot, selector, point in inv:
        assert A.from_state_address(_state(slot, selector), point, layers=2, granularity=gran,
                                    limits=first).token == 3
        with pytest.raises(NoStateSlot, match="first_step"):
            A.from_state_address(_state(slot, selector), point, layers=2, granularity=gran)
        with pytest.raises(NoStateSlot, match="never stepped"):
            A.from_state_address(_state(slot, selector), point, layers=2, granularity=gran,
                                 limits={"first_step": 4})
    # hidden states and cache cells do not need it
    assert A.from_state_address(_state("hidden", ()), _point(3, "layer", 0), layers=2).token == 3
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError):
            A.to_state_address("t3.sample", point=_point(4, "embed"), layers=2,
                               limits={"first_step": bad})


def test_c8_aliases_fuzz_has_no_failures():
    rng = random.Random(8)
    checked = 0
    for _ in range(6000):
        try:
            address = _random_address(rng)
        except AddressError:
            continue
        for layers in (1, 4, 24):
            try:
                key = A.value_key(address, layers=layers)
            except AddressError:
                continue                              # a layer the model does not have
            als = A.aliases(address, layers=layers)   # must not raise
            assert als[0] == key and all(A.value_key(a, layers=layers) == key for a in als)
            assert address in als                      # the spelling as written is one of them
            assert A.value_key(key, layers=layers) == key
            checked += 1
    assert checked > 5000


def test_c10_out_of_range_conversions_are_refused():
    # a row past the point's logical step
    point = _point(4, "layer", 2)
    assert A.to_state_address("cache.L1.k[kv0,row=4]", point=point).selector == (1, 0, 0, 4)
    for text in ("cache.L1.k[kv0,row=5]", "cache.L1.v[kv0,row=99,d=1]"):
        with pytest.raises(NoStateSlot, match="past logical step 4"):
            A.to_state_address(text, point=point)
    assert A.to_state_address("cache.L1.k[kv0,row=99]").selector == (1, 0, 0, 99)   # unbound: as before
    state = _state("keys", (1, 0, 0, 5))
    with pytest.raises(NoStateSlot, match="past logical step"):
        A.from_state_address(state, point, layers=6)
    # limits
    limits = {"heads": 8, "kv_groups": 2, "vocab": 100, "layers": 6, "first_step": 0}
    assert A.to_state_address("cache.L5.k[kv1]", limits=limits).selector == (5, 0, 1)
    for text in ("cache.L5.k[kv2]", "cache.L0.v[kv9,row=1]"):
        with pytest.raises(NoStateSlot, match="kv group"):
            A.to_state_address(text, limits=limits)
        assert A.to_state_address(text)                               # no limits: unchanged
    with pytest.raises(NoStateSlot, match="does not exist"):
        A.to_state_address("cache.L6.k", limits=limits)
    logits_point = _point(5, "embed")
    assert A.to_state_address("t4.logits[i=99]", point=logits_point, limits=limits).selector == (0, 99)
    for text in ("t4.logits[i=100]", 't4.logits["a"=100]', "t4.logits[i=99999]"):
        with pytest.raises(NoStateSlot, match="vocabulary"):
            A.to_state_address(text, point=logits_point, limits=limits)
        assert A.to_state_address(text, point=logits_point, limits=FIRST)   # no vocab limit: unchanged
    # layers from limits when layers= is not given; layers= wins when both are given
    hidden = _point(4, "layer", 5)
    assert A.to_state_address("t4.L5.h", point=hidden, limits=limits)
    with pytest.raises(NoStateSlot, match="does not exist"):
        A.to_state_address("t4.L6.h", point=_point(4, "layer", 6), limits=limits)
    for bad in ({"vocab": 0}, {"nope": 1}, {"heads": True}, [1]):
        with pytest.raises(ValueError):
            A.to_state_address("cache.L1.k", limits=bad)
