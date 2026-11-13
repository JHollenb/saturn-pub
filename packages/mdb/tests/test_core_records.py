import json

import pytest

from mdb.core import records as R
from mdb.core.pointers import MAX_RECORD_BYTES, RecordError, _canonical, occurrence_pointer

R.declare_kind("test-margin", numeric=("margin", "effects.*", "layers.*.value"),
               required=("margin", "consumer"), plain_numeric=("layers.*.index",))
R.declare_kind("test-free", plain_numeric=("x", "n", "ids"))


def _body():
    return {"margin": R.num(-4.75, "measured", unit="logit"), "consumer": 't21.M[" hot"]',
            "effects": {"a": R.num(3.875, "measured"),
                        "b": R.num(0.875, "derived-exact", proof="P20")},
            "layers": [{"index": 3, "value": R.num(0.5, "bound")}]}


def _seal(body=None, *, kind="test-margin", authority="measured", pointers=()):
    return R.seal_record(session_id="s1", kind=kind, model_identity="model-x",
                         execution={"dtype": "bfloat16"}, authority=authority,
                         body=_body() if body is None else body, pointers=pointers)


def _hand_seal(kind, body, pointers=()):
    """A record sealed without ``seal_record`` (what a hostile or old writer could store)."""
    from mdb.core.pointers import seal_digest

    envelope = {"schema": R.SCHEMA, "session_id": "s1", "kind": kind, "model_identity": "model-x",
                "execution": {"dtype": "bfloat16"}, "authority": "measured",
                "pointers": list(pointers),
                "body": body}
    digest = seal_digest(envelope)
    return {**envelope, "id": digest, "digest": digest}


def test_seal_and_verify():
    record = _seal()
    assert set(record) == set(R.FIELDS)
    assert record["schema"] == "mdb-record-v2" and record["id"] == record["digest"]
    assert R.verify_record(json.loads(json.dumps(record))) == record
    assert _seal() == record  # deterministic
    envelope = {k: v for k, v in record.items() if k not in ("id", "digest")}
    from mdb.core.pointers import seal_digest
    assert record["digest"] == seal_digest(envelope)


@pytest.mark.parametrize("path, value", [
    (("body", "margin", "value"), -4.5),
    (("body", "consumer"), "t22.M"),
    (("authority",), "derived-local"),
    (("session_id",), "s2"),
    (("execution", "dtype"), "float32"),
    (("model_identity",), "model-y"),
    (("id",), "0" * 64),
    (("digest",), "0" * 64),
    (("pointers",), []),
])
def test_tamper_is_detected(path, value):
    record = json.loads(json.dumps(_seal(pointers=[_pointer()])))
    target = record
    for key in path[:-1]:
        target = target[key]
    assert target[path[-1]] != value
    target[path[-1]] = value
    with pytest.raises(RecordError, match="seal differs"):
        R.verify_record(record)


def test_envelope_fields_are_exact():
    record = _seal()
    with pytest.raises(RecordError, match="record fields"):
        R.verify_record({**record, "extra": 1})
    with pytest.raises(RecordError, match="record fields"):
        R.verify_record({k: v for k, v in record.items() if k != "authority"})
    with pytest.raises(RecordError, match="schema"):
        R.verify_record({**record, "schema": "mdb-record-v1"})


def test_authority_is_required():
    with pytest.raises(RecordError, match="authority must be one of"):
        _seal(authority=None)
    with pytest.raises(RecordError, match="authority must be one of"):
        _seal(authority="observed")
    with pytest.raises(RecordError, match="authority must be one of"):
        R.num(1.0, "estimated")
    assert R.num(2, "bound", proof="P31") == {"value": 2, "authority": "bound", "proof": "P31"}


def _with(**changes):
    return {**_body(), **changes}


@pytest.mark.parametrize("body, message", [
    (_with(margin=-4.75), "body.margin must be a num"),
    (_with(effects={"a": 0.5}), "body.effects.a must be a num"),
    (_with(layers=[{"index": 1, "value": 0.5}]), "body.layers\\[0\\].value must be a num"),
    (_with(margin={"value": -4.75}), "has no authority"),
    (_with(margin={"value": -4.75, "authority": "guess"}), "authority"),
    (_with(margin={"value": True, "authority": "measured"}), "finite number"),
    ({k: v for k, v in _body().items() if k != "consumer"}, "requires body.consumer"),
    ({k: v for k, v in _body().items() if k != "margin"}, "requires body.margin"),
    (_with(deep={"v": {"value": 0.25}}), "body.deep.v has no authority"),
    (_with(effects={"a": R.num(1, "measured")}, layers=[{"index": 1}]),
     "body.layers\\[0\\].value is missing"),
    (_with(layers=7), "body.layers must be a mapping or list"),
    ({k: v for k, v in _body().items() if k != "effects"}, "body.effects is missing"),
])
def test_bare_numbers_and_claims_without_authority_are_refused(body, message):
    with pytest.raises(RecordError, match=message):
        _seal(body)


def test_num_refuses_non_finite_and_bool():
    for bad in (float("nan"), float("inf"), True, "1.0", None):
        with pytest.raises(RecordError, match="finite number"):
            R.num(bad, "measured")
    assert R.num(-0.0, "measured")["value"] == 0.0


def test_undeclared_kind_is_refused_when_sealing_and_when_verifying():
    with pytest.raises(RecordError, match="not declared"):
        _seal({"x": 1}, kind="never-declared")
    hand = _hand_seal("never-declared", {"x": "text"})
    with pytest.raises(RecordError, match="not declared"):
        R.verify_record(hand)
    with pytest.raises(RecordError, match="not declared"):
        R.verify_record(hand, require_kind=False)   # the argument no longer relaxes it
    record = _seal({"x": 1, "claim": R.num(1.5, "supplied")}, kind="test-free")
    stored = json.loads(json.dumps(record))
    R._KINDS.pop("test-free")
    try:
        with pytest.raises(RecordError, match="not declared"):
            R.verify_record(stored)
    finally:
        R.declare_kind("test-free", plain_numeric=("x", "n", "ids"))
    assert R.verify_record(stored) == record


def test_redeclaring_a_kind_differently_fails():
    assert R.declare_kind("test-free", plain_numeric=("x", "n", "ids")) == R.kind_spec("test-free")
    with pytest.raises(RecordError, match="already declared differently"):
        R.declare_kind("test-free", numeric=("x",))
    with pytest.raises(RecordError, match="already declared differently"):
        R.declare_kind("test-free", plain_numeric=("x", "n", "ids", "more"))


def _sized(n):
    """A test-free record whose filler text totals ``n`` ASCII characters."""
    filler = {f"f{i:03d}": "x" * min(4000, n - 4000 * i) for i in range(-(-n // 4000))}
    return _seal({"filler": filler}, kind="test-free")


def test_record_bound_at_exactly_the_limit():
    # With 17 chunks fixed, the size grows by exactly one byte per filler character.
    start = 16 * 4000 + 1
    size = len(_canonical(_sized(start)).encode())
    n = start + MAX_RECORD_BYTES - size
    assert size < MAX_RECORD_BYTES and n + 1 <= 17 * 4000
    record = _sized(n)
    assert len(_canonical(record).encode()) == MAX_RECORD_BYTES
    assert R.verify_record(record) == record
    with pytest.raises(RecordError, match=f"{MAX_RECORD_BYTES + 1} bytes"):
        _sized(n + 1)


def _multibyte(extra):
    """A test-free record of 16 full 4000-byte chunks (text of 3-byte characters) and a last
    chunk sized so the stored form is exactly ``MAX_RECORD_BYTES + extra`` bytes."""
    def text(nbytes):
        return "\u2211" * (nbytes // 3) + "x" * (nbytes % 3)

    def build(last):
        filler = {f"f{i:03d}": text(4000) for i in range(16)}
        filler["f016"] = text(last)
        return {"filler": filler}

    def size(last):
        return len(R.stored_bytes(_hand_seal("test-free", build(last))))

    last = 1 + MAX_RECORD_BYTES + extra - size(1)
    assert 1 <= last <= 4000
    assert size(last) == MAX_RECORD_BYTES + extra
    return build(last)


def test_size_bound_is_measured_on_the_stored_non_ascii_form():
    body = _multibyte(0)
    record = _seal(body, kind="test-free")
    assert len(R.stored_bytes(record)) == MAX_RECORD_BYTES
    assert len(R.stored_form(record)) < MAX_RECORD_BYTES        # characters, not bytes
    assert len(R.stored_form(record).encode("ascii", "backslashreplace")) > MAX_RECORD_BYTES
    assert R.verify_record(json.loads(json.dumps(record, ensure_ascii=False))) == record
    with pytest.raises(RecordError, match=f"{MAX_RECORD_BYTES + 1} bytes"):
        _seal(_multibyte(1), kind="test-free")
    with pytest.raises(RecordError, match="bytes"):
        R.verify_record(_hand_seal("test-free", _multibyte(1)))


def test_lone_surrogates_are_a_record_error():
    with pytest.raises(RecordError, match="UTF-8"):
        _seal({"x": "\ud800"}, kind="test-free")


def test_records_hold_no_arrays():
    import numpy as np

    with pytest.raises(RecordError, match="array or tensor"):
        _seal({"x": np.zeros(3)}, kind="test-free")
    with pytest.raises(RecordError, match="store it by pointer"):
        _seal({"x": list(range(65))}, kind="test-free")
    assert _seal({"x": np.int32(2), "n": R.num(np.int64(3), "measured")},
                 kind="test-free")["body"] == {"x": 2, "n": {"value": 3, "authority": "measured"}}


def _pointer():
    return occurrence_pointer("t18.L20.v[kv0]", program="stepped", shape=[1, 128],
                              dtype="bfloat16", recipe="r" * 8, event=2, cut="cd" * 32,
                              authority="measured")


def test_occurrence_kind():
    body = {"history": {"token_ids": [1, 2, 3]},
            "transform": {"window": "t18", "convention": "P07"},
            "retained": [{"term": "L20.h5", "value": R.num(3.1, "derived-local", proof="P14")}],
            "unlisted": R.num(0.0, "derived-exact", proof="P20"), "consumer": 't21.M[" hot"]',
            "evidence": [{"kind": "run", "run_id": "job-1"}]}
    record = _seal(body, kind="occurrence", pointers=[_pointer()])
    assert R.verify_record(record)["pointers"][0]["address"]["port"] == "v"
    with pytest.raises(RecordError, match="pointers of kind \\['occurrence'\\]"):
        _seal(body, kind="occurrence")
    with pytest.raises(RecordError, match="must be a num"):
        _seal({**body, "unlisted": 0.0}, kind="occurrence", pointers=[_pointer()])
    with pytest.raises(RecordError, match="requires body.transform"):
        _seal({k: v for k, v in body.items() if k != "transform"}, kind="occurrence",
              pointers=[_pointer()])
    with pytest.raises(RecordError, match="state cell"):
        _seal(body, kind="occurrence", pointers=[{**_pointer(), "address": {
            "kind": "cell", "origin": None, "cell": {"layer": 1, "tensor": "v"},
            "selector": {}}}])


def test_occurrence_pointer_integers_obey_the_15_digit_cap():
    """Audit round 3: shape entries and event were unbounded (10**20 sealed; every other
    integer in a record is capped at 15 digits)."""
    for kwargs in ({"shape": [10**15]}, {"shape": [10**18]}, {"shape": [1, 10**5000]},
                   {"shape": [1], "event": 10**15}, {"shape": [1], "event": 10**20}):
        with pytest.raises(RecordError):
            occurrence_pointer("t1.L0.v", program="stepped", dtype="float32",
                               **{"shape": [1], **kwargs})
    top = 10**15 - 1
    good = occurrence_pointer("t1.L0.v", program="stepped", dtype="float32", shape=[top],
                              event=top)
    assert R.verify_record(_seal(_occ_body(), kind="occurrence", pointers=[good]))["pointers"]
    # a hand-built pointer (not made by occurrence_pointer) is refused at sealing too
    for change in ({"event": 10**20}, {"event": 10**15},
                   {"value": {**good["value"], "shape": [10**20]}}):
        with pytest.raises(RecordError):
            _seal(_occ_body(), kind="occurrence", pointers=[{**good, **change}])
        with pytest.raises(RecordError):
            R.num(1.0, "measured", pointer={**good, **change})
    with pytest.raises(RecordError):
        R._typed_occurrence({**good, "event": 10**20})


def _occ_body():
    return {"history": {"token_ids": [1, 2, 3]},
            "transform": {"window": "t18", "convention": "P07"},
            "retained": [{"term": "L20.h5", "value": R.num(3.1, "derived-local", proof="P14")}],
            "unlisted": R.num(0.0, "derived-exact", proof="P20"), "consumer": 't21.M[" hot"]',
            "evidence": [{"kind": "run", "run_id": "job-1"}]}


def test_num_pointer_is_validated():
    claim = R.num(1.0, "measured", pointer=_pointer())
    assert R.check_num(claim) == claim
    with pytest.raises(RecordError):
        R.num(1.0, "measured", pointer={"kind": "occurrence"})


def test_legacy_circuit_record_is_checked_by_saturn_digest():
    from saturn_pub.values import digest

    body = {"schema": "mdb-circuit-investigation-v1", "session_id": "s1", "kind": "focus",
            "model_identity": "model-x", "execution": {}, "node_id": "n1"}
    assert R.verify_legacy(body, digest(body)) == {"record_id": digest(body), **body}
    with pytest.raises(RecordError, match="seal differs"):
        R.verify_legacy({**body, "node_id": "n2"}, digest(body))
    with pytest.raises(RecordError, match="legacy"):
        R.verify_legacy({**body, "schema": "other"}, digest(body))


# --------------------------------------------------------------------------- authority rule

R.declare_kind("test-shapes", numeric=("claim",), required=("claim",),
               plain_numeric=("shape", "ids", "meta.count", "rows.*.index", "sizes.**"))
_CLAIM = R.num(1.5, "measured")


def _shape_body(**extra):
    return {"claim": _CLAIM, **extra}


# (name, body, accepted)
SHAPES = [
    ("only the claim", _shape_body(), True),
    ("text values", _shape_body(note="a", tag=["x", "y"]), True),
    ("null and bool", _shape_body(flag=True, none=None, flags=[False, None]), True),
    ("structural shape list", _shape_body(shape=[1, 128, 4096]), True),
    ("structural ids", _shape_body(ids=[3, 5, 7]), True),
    ("structural nested", _shape_body(meta={"count": 3}), True),
    ("structural under list of mappings", _shape_body(rows=[{"index": 0}, {"index": 4}]), True),
    ("structural sizes mapping", _shape_body(sizes={"a": 1, "b": {"c": 2}}), True),
    ("labelled number elsewhere", _shape_body(extra=R.num(2, "bound")), True),
    ("labelled numbers in a list", _shape_body(extra=[R.num(2, "bound"), R.num(.5, "derived-local")]),
     True),
    ("labelled number at depth 4", _shape_body(a={"b": [{"c": R.num(1, "supplied")}]}), True),
    ("empty containers", _shape_body(a=[], b={}), True),
    ("bare int", _shape_body(n=1), False),
    ("bare float", _shape_body(x=0.25), False),
    ("bare number in a list", _shape_body(x=[1, 2]), False),
    ("bare number in a nested list", _shape_body(x=[[1.5]]), False),
    ("bare number at depth 2", _shape_body(a={"b": 3}), False),
    ("bare number at depth 3", _shape_body(a={"b": {"c": 3}}), False),
    ("bare number at depth 4 in a list of mappings", _shape_body(a=[{"b": [{"c": 1.5}]}]), False),
    ("bare number at depth 4 in mappings", _shape_body(a={"b": {"c": {"d": 7}}}), False),
    ("bare number at depth 5", _shape_body(a=[{"b": [{"c": [2]}]}]), False),
    ("bare number beside structural key", _shape_body(meta={"count": 3, "kl": 0.01}), False),
    ("bare number under a sibling of a structural key", _shape_body(rows=[{"index": 0, "kl": 1}]),
     False),
    ("structural key name at the wrong depth", _shape_body(a={"shape": [1]}), False),
    ("bare number under an unrelated mapping key", _shape_body(evidence=[{"kl": 0.0076}]), False),
    ("tuple of bare numbers", _shape_body(x=(1, 2)), False),
    ("claim mapping without authority", _shape_body(x={"value": 0.5}), False),
    ("claim mapping with numeric string value", _shape_body(x={"value": "0.5"}), False),
    ("claim mapping with extra key", _shape_body(x={"value": 0.5, "authority": "measured",
                                                    "note": "n"}), False),
    ("claim mapping with unknown authority", _shape_body(x={"value": 0.5, "authority": "guess"}),
     False),
    ("claim mapping with bool value", _shape_body(x={"value": True, "authority": "measured"}),
     False),
    ("claim mapping with string value and authority",
     _shape_body(x={"value": "0.5", "authority": "measured"}), False),
    ("claim mapping with a mapping value", _shape_body(x={"value": {"a": 1}, "authority":
                                                             "measured"}), False),
    ("declared claim bare", {"claim": 1.5}, False),
    ("declared claim is a list of claims", {"claim": [_CLAIM]}, False),
    ("declared claim is a numeric string", {"claim": "1.5"}, False),
    ("declared claim is a string with authority label", {"claim": {"value": "1.5",
                                                                     "authority": "measured"}},
     False),
    ("declared claim missing", {"other": _CLAIM}, False),
    ("declared claim null", {"claim": None}, False),
    ("declared claim bool", {"claim": True}, False),
    ("declared claim with extra key", {"claim": {**_CLAIM, "extra": 1}}, False),
    ("structural NaN", _shape_body(shape=[float("nan")]), False),
    ("structural Inf", _shape_body(ids=[float("inf")]), False),
    ("claim NaN", {"claim": {"value": float("nan"), "authority": "measured"}}, False),
    ("labelled number with a bare number in its pointer", _shape_body(
        x={**R.num(1, "measured"), "pointer": {"kind": "run", "run_id": "r", "kl": 0.1}}), False),
]


def test_the_table_has_enough_shapes():
    assert len(SHAPES) >= 30


@pytest.mark.parametrize("name, body, accepted", SHAPES, ids=[n for n, _, _ in SHAPES])
def test_authority_rule_table(name, body, accepted):
    try:
        hand = _hand_seal("test-shapes", json.loads(json.dumps(body, allow_nan=True)))
    except ValueError:   # a NaN has no digest: nothing to hand-seal
        hand = None
    if accepted:
        record = _seal(body, kind="test-shapes")
        assert R.verify_record(record) == record
        assert R.verify_record(hand) == hand
    else:
        with pytest.raises(RecordError):
            _seal(body, kind="test-shapes")
        # verify_record applies the same rule to a record some other writer sealed (a NaN does
        # not survive the digest: the seal itself is refused)
        if hand is not None:
            with pytest.raises(RecordError):
                R.verify_record(hand)


_OCC_PTR = occurrence_pointer("t18.L20.v[kv0]", program="stepped", shape=[1, 128],
                              dtype="bfloat16")
_OCC_BASE = {"history": {"token_ids": [1, 2, 3]},
             "transform": {"window": "t18", "convention": "P07"},
             "retained": [{"term": "L20.h5", "value": R.num(3.1, "derived-local", proof="P14")}],
             "unlisted": R.num(0.0, "derived-exact", proof="P20"), "consumer": 't21.M[" hot"]',
             "evidence": [{"kind": "run", "run_id": "job-1"}]}


def _occ(body, pointers=None):
    return R.seal_record(session_id="s1", kind="occurrence", model_identity="m",
                         execution={"dtype": "bf16"}, authority="measured", body=body,
                         pointers=[_OCC_PTR] if pointers is None else pointers)


@pytest.mark.parametrize("name, body", [
    ("retained = list of bare floats", {**_OCC_BASE, "retained": [3.1, -2.0]}),
    ("retained item without 'value' key", {**_OCC_BASE, "retained": [{"term": "x", "weight": 3.1}]}),
    ("retained = bare number", {**_OCC_BASE, "retained": 3.1}),
    ("retained item value bare", {**_OCC_BASE, "retained": [{"term": "x", "value": 3.1}]}),
    ("num-like with extra key", {**_OCC_BASE, "evidence": [{"value": 0.5, "note": "measured-ish"}]}),
    ("value as string", {**_OCC_BASE, "evidence": [{"value": "0.5"}]}),
    ("bare number under undeclared key", {**_OCC_BASE, "evidence": [{"kind": "run", "kl": 0.0076}]}),
    ("unlisted as list of num", {**_OCC_BASE, "unlisted": [R.num(0.0, "measured")]}),
    ("unlisted bool", {**_OCC_BASE, "unlisted": True}),
    ("num value bool", {**_OCC_BASE, "unlisted": {"value": True, "authority": "measured"}}),
    ("num value as string w/ authority",
     {**_OCC_BASE, "unlisted": {"value": "0.0", "authority": "measured"}}),
    ("num authority wrong case", {**_OCC_BASE, "unlisted": {"value": 0.0, "authority": "Measured"}}),
    ("unlisted missing", {k: v for k, v in _OCC_BASE.items() if k != "unlisted"}),
    ("bare number in transform key outside the declared structure",
     {**_OCC_BASE, "consumer": {"margin": 1.5}}),
])
def test_every_listed_bypass_is_refused(name, body):
    with pytest.raises(RecordError):
        _occ(body)
    with pytest.raises(RecordError):
        R.verify_record(_hand_seal("occurrence", body, [_OCC_PTR]))


def test_occurrence_baseline_and_structural_keys_are_accepted():
    record = _occ(_OCC_BASE)
    assert R.verify_record(record) == record
    structural = {**_OCC_BASE, "history": {"token_ids": [1, 2, 3], "parent_row": 7},
                  "transform": {"window": 4, "head": 0, "convention": "P07"},
                  "retained": [{"src": 3, "row": 1, "value": R.num(1, "measured")}],
                  "evidence": [{"kind": "file", "size": 1024}]}
    assert R.verify_record(_occ(structural)) == _occ(structural)
    assert _occ({**_OCC_BASE, "retained": {}})["body"]["retained"] == {}   # explicit: no terms


def test_pointers_of_unknown_kind_are_refused():
    for pointer in ({"kind": "blob", "kl": 0.0076}, {"kind": "url"}, {"kind": "file2"}):
        with pytest.raises(RecordError, match="pointer kind"):
            _occ(_OCC_BASE, pointers=[_OCC_PTR, pointer])
        with pytest.raises(RecordError, match="pointer kind"):
            R.verify_record(_hand_seal("occurrence", _OCC_BASE, [_OCC_PTR, pointer]))
        with pytest.raises(RecordError, match="pointer kind"):
            R.num(1.0, "measured", pointer=pointer)
    smuggled = {"kind": "run", "run_id": "r", "kl": 0.5}
    with pytest.raises(RecordError, match="unknown fields"):
        _occ(_OCC_BASE, pointers=[_OCC_PTR, smuggled])
    assert _occ(_OCC_BASE, pointers=[_OCC_PTR, {"kind": "run", "run_id": "r"}])


def test_nan_and_inf_are_refused_everywhere():
    for bad in (float("nan"), float("inf"), float("-inf")):
        for body in ({**_OCC_BASE, "history": {"token_ids": [bad]}},
                     {**_OCC_BASE, "transform": {"window": bad}},
                     {**_OCC_BASE, "unlisted": {"value": bad, "authority": "measured"}}):
            with pytest.raises(RecordError):
                _occ(body)
    stored = {**_hand_seal("occurrence", _OCC_BASE, [_OCC_PTR])}
    stored["body"] = {**_OCC_BASE, "history": {"token_ids": [float("nan")]}}
    with pytest.raises(RecordError, match="not finite"):
        R.verify_record(stored)


def test_legacy_circuit_records_are_exempt_and_unchanged():
    from saturn_pub.values import digest

    body = {"schema": "mdb-circuit-investigation-v1", "session_id": "s1", "kind": "inspection",
            "model_identity": "m", "execution": {}, "score": 0.5, "rows": [1, 2, {"kl": 3.5}],
            "note": "bare numbers everywhere"}
    legacy = R.verify_legacy(body, digest(body))
    assert legacy == {"record_id": digest(body), **body}
    with pytest.raises(RecordError, match="not declared|bare number|schema"):
        R.verify_record({**_hand_seal("inspection", {"score": 0.5}), "schema": body["schema"]})


# --------------------------------------------------------------------------- round 3: C2 / C7

_M = R.num(1.0, "measured")
_FILE = {"kind": "file", "locator": {"root": "r", "path": "a.npy"},
         "digest": {"sha256": "a" * 64, "size": 12}}
_PARAM = {"kind": "parameter", "parameter": "w",
          "model": {"name": "m", "identity_status": "unresolved", "resolve_by": "x"}}


def _both_refuse(kind, body, pointers=(), **kw):
    """Refused by ``seal_record`` and, hand-sealed (a hostile writer), by ``verify_record``."""
    from mdb.core.pointers import seal_digest

    with pytest.raises(RecordError):
        R.seal_record(session_id="s1", kind=kind, model_identity=kw.get("mi", "m"),
                      execution=kw.get("ex", {"dtype": "bf16"}), authority="measured", body=body,
                      pointers=list(pointers))
    envelope = {"schema": R.SCHEMA, "session_id": "s1", "kind": kind,
                "model_identity": kw.get("mi", "m"), "execution": kw.get("ex", {"dtype": "bf16"}),
                "authority": "measured", "pointers": list(pointers), "body": body}
    try:
        digest = seal_digest(envelope)
    except ValueError:            # a huge int cannot even be digested here: verify must still refuse
        digest = "0" * 64
    with pytest.raises(RecordError):
        R.verify_record({**envelope, "id": digest, "digest": digest})


@pytest.mark.parametrize("name, body", [
    ("transform holds a measurement", {**_OCC_BASE, "transform": {"window": 4, "logit_diff": 4.2,
                                                                  "accuracy": 0.97}}),
    ("transform window is a float", {**_OCC_BASE, "transform": {"window": 4.5}}),
    ("transform undeclared int", {**_OCC_BASE, "transform": {"window": 4, "matches": 38}}),
    ("history holds a measurement", {**_OCC_BASE, "history": {"token_ids": [1], "loss": 2.31}}),
    ("history token id float", {**_OCC_BASE, "history": {"token_ids": [1.5]}}),
    ("history token id negative", {**_OCC_BASE, "history": {"token_ids": [-1]}}),
    ("retained src is a float", {**_OCC_BASE, "retained": [{"value": _M, "src": 0.731}]}),
    ("retained src is a dict", {**_OCC_BASE, "retained": [{"value": _M, "src": {"acc": 0.9}}]}),
    ("retained src is a list of floats", {**_OCC_BASE, "retained": [{"value": _M,
                                                                     "src": [0.1, 0.2]}]}),
    ("evidence size is a dict of measurements",
     {**_OCC_BASE, "evidence": [{"size": {"acc": 0.97, "kl": 0.0076}}]}),
    ("evidence size is a dict of ints", {**_OCC_BASE, "evidence": [{"size": {"a": 1}}]}),
    ("evidence size negative", {**_OCC_BASE, "evidence": [{"size": -3}]}),
    ("evidence index float", {**_OCC_BASE, "evidence": [{"index": 38.5}]}),
    ("structural int past 15 digits", {**_OCC_BASE, "evidence": [{"size": 10 ** 15}]}),
    ("structural int 10**5000", {**_OCC_BASE, "history": {"token_ids": [10 ** 5000]}}),
])
def test_c2_structural_holes_are_closed(name, body):
    _both_refuse("occurrence", body, [_OCC_PTR])


def test_c2_structural_leaves_still_take_non_negative_ints():
    body = {**_OCC_BASE, "history": {"token_ids": [0, 1, 10 ** 15 - 1], "parent_row": 7},
            "transform": {"window": 4, "head": 0},
            "retained": [{"src": 3, "row": 1, "token": 5, "layer": 2, "head": 1,
                          "value": R.num(1, "measured")}],
            "evidence": [{"size": 1024, "event": 2, "index": 0}]}
    record = _occ(body)
    assert R.verify_record(record) == record


def test_c2_a_subtree_is_structural_only_with_an_explicit_double_star():
    R.declare_kind("test-subtree", plain_numeric=("whole.**", "leaf"))
    good = {"whole": {"a": 1, "b": {"c": [2, 3]}}, "leaf": 4}
    assert _seal(good, kind="test-subtree")
    for bad in ({"whole": {"a": 0.5}}, {"whole": {"b": {"c": -1}}}, {"leaf": 0.5},
                {"leaf": {"x": 1}}, {"whole": {"a": 1.5}}, {"leaf": 10 ** 15}):
        _both_refuse("test-subtree", bad)
    # without the marker a path names only its own leaf (a nested mapping is refused)
    R.declare_kind("test-leaf-only", plain_numeric=("sizes",))
    assert _seal({"sizes": [1, 2]}, kind="test-leaf-only")
    _both_refuse("test-leaf-only", {"sizes": {"a": 1}})
    with pytest.raises(RecordError, match="'\\*\\*' only as the last"):
        R.declare_kind("test-bad-marker", plain_numeric=("a.**.b",))


def test_c2_pointer_free_fields_carry_no_numbers():
    ok = {**_FILE, "array": {"shape": [1, 32], "dtype": "float32"}, "selector": [None, 1, [2, 3]],
          "json_pointer": "/a/b", "role": "plan"}
    R.declare_kind("test-plain", numeric=("margin",), required=("margin",))
    assert _seal({"margin": _M}, kind="test-plain", pointers=[ok])
    assert _seal({"margin": _M}, kind="test-plain", pointers=[{**_FILE, "array": "tok_w"}])
    for field, bad in [("array", {"mean": 0.731}), ("array", {"shape": [1.5], "dtype": "f"}),
                       ("array", {"shape": [1], "dtype": "f", "kl": 0.1}), ("array", 0.5),
                       ("array", {"shape": [-1], "dtype": "f"}),
                       ("selector", [0.97]), ("selector", [-1]), ("selector", [True]),
                       ("selector", [[0.5]]), ("selector", 3), ("selector", {"a": 1}),
                       ("role", 0.5), ("role", {"a": 1}), ("json_pointer", 7)]:
        _both_refuse("test-plain", {"margin": _M}, [{**_FILE, field: bad}])
        with pytest.raises(RecordError):
            R.num(1.0, "measured", pointer={**_FILE, field: bad})
    for digest in ({"sha256": "a" * 64, "size": 0.5}, {"sha256": "a" * 64, "size": -1},
                   {"sha256": "a" * 64, "size": True}):
        _both_refuse("test-plain", {"margin": _M}, [{**_FILE, "digest": digest}])
    _both_refuse("test-plain", {"margin": _M}, [{**_FILE, "locator": {"root": "r", "path": 5}}])
    assert _seal({"margin": _M}, kind="test-plain", pointers=[{**_PARAM, "selector": [0, 3]}])
    for bad in ([0.731], [-1], ["a"], 5):
        _both_refuse("test-plain", {"margin": _M}, [{**_PARAM, "selector": bad}])
    _both_refuse("test-plain", {"margin": _M}, [{**_PARAM, "model": {
        **_PARAM["model"], "resolve_by": 0.731}}])
    # run pointer: plan must be a file pointer carrying no numbers; plan_sha256 is a digest
    assert _seal({"margin": _M}, kind="test-plain", pointers=[
        {"kind": "run", "run_id": "r1", "plan_sha256": "a" * 64, "plan": _FILE}])
    for plan in ({**_FILE, "array": {"mean": 0.731}}, {"acc": 0.97}, 0.5, {"kind": "run",
                                                                            "run_id": "r"}):
        _both_refuse("test-plain", {"margin": _M}, [{"kind": "run", "run_id": "r1", "plan": plan}])
    _both_refuse("test-plain", {"margin": _M}, [{"kind": "run", "run_id": "r1",
                                                 "plan_sha256": 0.731}])


def test_c2_execution_and_identity_carry_no_measurements():
    R.declare_kind("test-plain", numeric=("margin",), required=("margin",))
    for execution in ({"dtype": "bf16", "acc": 0.97}, {"nested": {"kl": 0.0076}},
                      {"counts": [0.5]}, {"n": -3}, {"n": 10 ** 15}, {"d": {"value": 0.5}}):
        _both_refuse("test-plain", {"margin": _M}, ex=execution)
    _both_refuse("test-plain", {"margin": _M}, mi={"name": "m", "score": 0.5})
    # what the adapters really store: counts, device index, thread counts, configuration floats
    real = {"family": "x", "device": {"type": "cpu", "index": None}, "batch": {"size": 1,
            "guidance": 1.0}, "numeric_scope": {"torch_num_threads": 1, "autocast": False},
            "eta": 0.0, "logit_softcap": {"final": 30.0, "attention": 50.0}, "sliding_window": 4096,
            "version": 1, "graph_nodes": 12}
    assert R.seal_record(session_id="s", kind="test-plain", model_identity={"name": "m", "n": 3},
                         execution=real, authority="measured", body={"margin": _M})


def test_c2_strings_that_look_like_numbers_and_booleans_are_text_by_design():
    """Documented out of scope: a string is text and a boolean is not a numeric claim."""
    assert "string that looks like a number is text" in R.__doc__
    R.declare_kind("test-plain", numeric=("margin",), required=("margin",))
    assert _seal({"margin": _M, "accuracy": "0.97", "significant": True}, kind="test-plain")


def test_c7_integers_are_capped_at_15_digits_and_huge_ones_are_record_errors():
    assert R.num(10 ** 15 - 1, "measured")["value"] == 10 ** 15 - 1
    assert R.num(-(10 ** 15 - 1), "measured")["value"] == -(10 ** 15 - 1)
    for big in (10 ** 15, -(10 ** 15), 10 ** 30, 10 ** 5000, -(10 ** 5000)):
        with pytest.raises(RecordError, match="15 decimal digits"):
            R.num(big, "measured")
        with pytest.raises(RecordError):
            R.check_num({"value": big, "authority": "measured"})
    R.declare_kind("test-plain", numeric=("margin",), required=("margin",))
    for body in ({"margin": _M, "n": 10 ** 5000}, {"margin": _M, "n": [10 ** 5000]},
                 {"margin": {"value": 10 ** 5000, "authority": "measured"}}):
        _both_refuse("test-plain", body)
    _both_refuse("test-plain", {"margin": _M}, ex={"n": 10 ** 5000})
    _both_refuse("test-plain", {"margin": _M}, [{**_FILE, "digest": {"sha256": "a" * 64,
                                                                    "size": 10 ** 5000}}])
    _both_refuse("occurrence", {**_OCC_BASE, "history": {"token_ids": [10 ** 5000]}}, [_OCC_PTR])
    stored = _occ(_OCC_BASE)
    stored["body"] = {**_OCC_BASE, "history": {"token_ids": [10 ** 5000]}}
    with pytest.raises(RecordError):
        R.verify_record(stored)
    assert issubclass(RecordError, ValueError)
