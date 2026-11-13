"""The session substrate copies, hashes and validates a state once per call, and still refuses
what it refused before: a mutated cut, a changed model parameter, a non-finite state.

Counts are taken on a tiny native decoder with the copy, hash and check functions wrapped."""

import collections

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub import core, values  # noqa: E402
from saturn_pub.adapters import decoder  # noqa: E402
from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from saturn_pub.core import Session, StateCut  # noqa: E402
from saturn_pub.values import describe, digest  # noqa: E402


@pytest.fixture
def adapter():
    return DecoderAdapter.tiny("qwen2", granularity="operation")


@pytest.fixture
def counts(monkeypatch):
    seen: collections.Counter = collections.Counter()
    clone, describe_, guards = values.clone, values.describe, decoder._tensor_guards
    check, boolean = DecoderAdapter.validate_state, torch.Tensor.__bool__

    def counted_clone(value):
        if isinstance(value, dict) and "keys" in value and "values" in value:
            seen["state copies"] += 1
        return clone(value)

    def counted_describe(value):
        if values.tensor(value):
            seen["tensor hashes"] += 1
        return describe_(value)

    def counted_guards(*args, **kwargs):
        seen["frozen-model walks"] += 1
        return guards(*args, **kwargs)

    def counted_check(self, state):
        seen["state validations"] += 1
        return check(self, state)

    def counted_bool(self):
        seen["host syncs"] += 1
        return boolean(self)

    monkeypatch.setattr(core, "clone", counted_clone)
    monkeypatch.setattr(values, "describe", counted_describe)
    monkeypatch.setattr(core, "describe", counted_describe)
    monkeypatch.setattr(decoder, "_tensor_guards", counted_guards)
    monkeypatch.setattr(DecoderAdapter, "validate_state", counted_check)
    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)
    return seen


def _tensors(state) -> int:
    return sum(len(v) if isinstance(v, list) else values.tensor(v) for v in state.values())


def _at_layer(adapter, layer=1):
    session = adapter.session([3, 1, 4, 1])
    while not (session._state["phase"] == "layer" and session._state["layer"] == layer):
        session.continue_()
    return session


def test_from_cut_hashes_copies_and_validates_once(adapter, counts):
    session = _at_layer(adapter)
    cut = session.capture()
    counts.clear()
    hydrated = Session.from_cut(adapter, cut)
    assert counts["tensor hashes"] == _tensors(cut._payload)  # verify, once
    assert counts["state copies"] == 1
    assert counts["state validations"] == 1 and counts["frozen-model walks"] == 1
    assert counts["host syncs"] <= 2  # tokens in range, all finite (one fused check)
    [receipt] = [r.to_dict() for r in hydrated.receipts]
    assert receipt["operation"] == "restore"
    assert receipt["parent"] == receipt["result"] == cut.fingerprint
    assert hydrated.capture().fingerprint == cut.fingerprint


def test_restore_hashes_copies_and_validates_once(adapter, counts):
    session = _at_layer(adapter)
    cut = session.capture()
    session.continue_()
    counts.clear()
    receipt = session.restore(cut).to_dict()
    assert counts["tensor hashes"] == _tensors(cut._payload)
    assert counts["state copies"] == 1
    assert counts["state validations"] == 1 and counts["frozen-model walks"] == 1
    assert receipt["result"] == cut.fingerprint == session.capture().fingerprint


def test_a_transition_hashes_each_slot_once_and_capture_after_it_hashes_nothing(adapter, counts):
    session = _at_layer(adapter)
    counts.clear()
    receipt = session.continue_().to_dict()
    after = _tensors(session._state)
    assert counts["tensor hashes"] == after  # the new state's slots, each once
    assert counts["state copies"] == 1  # the candidate the native transition runs on
    assert counts["state validations"] == 1
    assert counts["frozen-model walks"] == 2  # before (as capture did) and after the native call
    counts.clear()
    cut = session.capture()
    assert counts["tensor hashes"] == 0 and counts["state validations"] == 0
    assert counts["state copies"] == 1  # the cut's own payload
    assert receipt["result"] == cut.fingerprint
    # the fingerprint sealed from known descriptors is the one a full re-hash gives
    assert digest(cut.manifest()) == cut.fingerprint
    cut.verify()


def test_known_descriptors_seal_the_same_fingerprints_as_hashing(adapter):
    session = adapter.session([2, 7, 1, 8])
    fingerprints = []
    for _ in range(12):  # embed, layers, norm, readout, sample, commit and the next row
        session.continue_()
        cut = session.capture()
        rebuilt = StateCut(cut.model_identity, cut.execution, cut.boundary, cut.parent,
                           cut.payload, cut.execution_point, cut.surface)  # hashes everything
        assert rebuilt.fingerprint == cut.fingerprint
        fingerprints.append(cut.fingerprint)
    assert [r.to_dict()["result"] for r in session.receipts] == fingerprints


def test_a_mutated_cut_is_refused(adapter):
    session = _at_layer(adapter)
    cut = session.capture()
    cut._payload["hidden"].add_(1)
    with pytest.raises(ValueError, match="StateCut payload or metadata changed"):
        Session.from_cut(adapter, cut)
    with pytest.raises(ValueError, match="StateCut payload or metadata changed"):
        session.restore(cut)
    with pytest.raises(ValueError, match="StateCut payload or metadata changed"):
        session.fork(cut)


@pytest.mark.parametrize("call", ["from_cut", "restore", "continue_", "capture", "fork"])
def test_a_changed_model_parameter_is_refused(adapter, call):
    session = _at_layer(adapter)
    cut = session.capture()
    with torch.no_grad():
        next(adapter.model.parameters()).add_(1)
    with pytest.raises(ValueError, match="rewrap"):
        {
            "from_cut": lambda: Session.from_cut(adapter, cut),
            "restore": lambda: session.restore(cut),
            "continue_": lambda: session.continue_(),
            "capture": lambda: session.capture(),
            "fork": lambda: session.fork(),
        }[call]()


@pytest.mark.parametrize(
    ("slot", "message"),
    [("keys", "non-finite K/V"), ("hidden", "invalid hidden carrier")],
)
def test_a_non_finite_state_is_refused(adapter, slot, message):
    session = _at_layer(adapter, adapter.layers - 1)
    cut = session.capture()
    payload = cut.payload
    if slot == "keys":
        payload["keys"][1][..., 0, 0] = float("nan")
    else:
        payload["hidden"][..., 0] = float("inf")
    forged = StateCut(cut.model_identity, cut.execution, cut.boundary, cut.parent, payload,
                      cut.execution_point, cut.surface)
    with pytest.raises(ValueError, match=message):
        Session.from_cut(adapter, forged)
    with pytest.raises(ValueError, match=message):
        session.restore(forged)
    with pytest.raises(ValueError, match=message):
        Session(adapter, payload)


def test_one_layer_of_a_cache_slot_and_one_slot_of_a_cut_copy_only_that(adapter, monkeypatch):
    session = _at_layer(adapter, adapter.layers - 1)
    cut = session.capture()
    copied = []
    clone = torch.Tensor.clone

    def counted(self, *args, **kwargs):
        copied.append(self.numel())
        return clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counted)
    layer = session.read("keys", 1)
    assert copied == [layer.numel()]
    copied.clear()
    hidden = cut.slot("hidden")
    assert copied == [hidden.numel()]
    monkeypatch.undo()
    assert torch.equal(layer, session.read("keys")[1])
    assert describe(hidden) == describe(cut.payload["hidden"])
    hidden.add_(1)  # a private copy: the cut is unchanged
    cut.verify()
