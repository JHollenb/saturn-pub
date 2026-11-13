from contextlib import contextmanager

import pytest

from saturn_pub import Adapter, PortStop, Session

HISTORICAL_TRANSITION_KEYS = {
    "before",
    "after",
    "native_operation",
    "structural_operation",
    "occurrence",
}


class Counter(Adapter):
    model_identity = "counter-v1"
    execution = {"family": "counter"}

    def boundary(self, state):
        return "step"

    def validate(self, state):
        if type(state["value"]) is not int or state["value"] < 0:
            raise ValueError("nonnegative integer required")

    def addresses(self, state):
        return ("value",)

    def advance(self, state):
        return {"value": state["value"] + 1}


class PortedCounter(Counter):
    def ports(self, state):
        return ("increment",)

    def advance(self, state, *, ports=None):
        delta = 1 if ports is None else ports.increment(state["value"])
        return {"value": state["value"] + delta}


class Controller:
    def __init__(self, *, stop=False, delta=1, fail=False):
        self.stop, self.delta, self.fail, self.seen = stop, delta, fail, []

    def increment(self, value):
        self.seen.append(value)
        if self.fail:
            raise RuntimeError("controller failure")
        if self.stop:
            raise PortStop("increment")
        return self.delta

    def manifest(self):
        return [{"port": "increment", "delta": self.delta}] if self.delta != 1 else None


def _counts(session):
    return len(session.receipts), len(session.attempts), len(session.history)


def test_adapter_without_ports_keeps_working_and_refuses_controllers():
    session = Session(Counter(), {"value": 0})
    assert session.adapter.ports(session.capture(retain=False).payload) == ()
    receipt = session.continue_(2)
    assert session.read("value") == 2
    entry = receipt.to_dict()["transitions"][0]
    assert set(entry) == HISTORICAL_TRANSITION_KEYS
    before = session.capture(retain=False).fingerprint
    with pytest.raises(ValueError, match="does not accept a port controller"):
        session.continue_(ports=Controller())
    with pytest.raises(ValueError, match="no ports"):
        session.probe(Controller())
    assert session.capture(retain=False).fingerprint == before
    assert _counts(session) == (1, 0, 0)


def test_probe_is_read_only_and_portstop_is_normal_completion():
    session = Session(PortedCounter(), {"value": 3})
    before = session.capture(retain=False)
    controller = Controller(stop=True)
    assert session.probe(controller) == {"boundary": "step", "stopped": True}
    completed = Controller()
    assert session.probe(completed) == {"boundary": "step", "stopped": False}
    assert controller.seen == completed.seen == [3]
    with pytest.raises(RuntimeError, match="controller failure"):
        session.probe(Controller(fail=True))
    assert session.capture(retain=False).fingerprint == before.fingerprint
    assert session.read("value") == 3
    assert _counts(session) == (0, 0, 0)


def test_continue_records_port_edits_only_when_declared():
    plain = Session(PortedCounter(), {"value": 0})
    inert = Session(PortedCounter(), {"value": 0})
    edited = Session(PortedCounter(), {"value": 0})
    first = plain.continue_()
    second = inert.continue_(ports=Controller())
    assert first._body == second._body
    assert plain.capture(retain=False).fingerprint == inert.capture(retain=False).fingerprint
    receipt = edited.continue_(ports=Controller(delta=5))
    entry = receipt.to_dict()["transitions"][0]
    assert entry["port_edits"] == [{"port": "increment", "delta": 5}]
    assert set(entry) - {"port_edits"} == HISTORICAL_TRANSITION_KEYS
    assert edited.read("value") == 5


def test_portstop_during_continue_is_a_refused_attempt():
    session = Session(PortedCounter(), {"value": 0})
    with pytest.raises(PortStop):
        session.continue_(ports=Controller(stop=True))
    assert _counts(session) == (0, 1, 0)
    assert session.read("value") == 0


class LayerProbe:
    """A generic decoder controller: capture the MLP output of one layer, optionally stop."""

    def __init__(self, layer, *, stop):
        self.layer, self.stop, self.value, self.installs = layer, stop, None, 0

    @contextmanager
    def installed(self, adapter, layer_index, layer_module):
        if layer_index != self.layer:
            yield
            return
        self.installs += 1

        def hook(module, args, output):
            self.value = output.detach().clone()
            if self.stop:
                raise PortStop("mlp")

        handle = layer_module.mlp.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


def test_decoder_probe_leaves_state_and_cache_untouched():
    torch = pytest.importorskip("torch")
    from saturn_pub.adapters.decoder import DecoderAdapter

    adapter = DecoderAdapter.tiny("qwen2", granularity="operation")
    session = adapter.session([1, 2, 3, 4])
    with pytest.raises(ValueError, match="no ports"):
        session.probe(LayerProbe(0, stop=True))
    session.continue_()
    state = session.capture(retain=False)
    ports = adapter.ports(state.payload)
    assert ports[0] == "model.layers.0" and "model.layers.0.mlp.down_proj" in ports
    keys = [k.clone() for k in state.payload["keys"]]
    stopped, full = LayerProbe(0, stop=True), LayerProbe(0, stop=False)
    assert session.probe(stopped)["stopped"] is True
    assert session.probe(full)["stopped"] is False
    assert torch.equal(stopped.value, full.value)
    after = session.capture(retain=False)
    assert after.fingerprint == state.fingerprint
    assert all(torch.equal(a, b) for a, b in zip(keys, after.payload["keys"]))
    assert _counts(session) == (1, 0, 0)
    assert len(adapter.model.model.layers[0].mlp._forward_hooks) == 0


def test_decoder_receipts_byte_identical_with_inert_controller():
    from saturn_pub.adapters.decoder import DecoderAdapter

    adapter = DecoderAdapter.tiny("qwen3", granularity="operation")
    plain, observed = adapter.session([5, 6]), adapter.session([5, 6])
    controller = LayerProbe(1, stop=False)
    for _ in range(adapter.layers + 5):
        plain.continue_()
        observed.continue_(ports=controller)
    assert controller.installs == 1
    assert [r._body for r in plain.receipts] == [r._body for r in observed.receipts]
    assert plain.capture(retain=False).fingerprint == observed.capture(retain=False).fingerprint
    for receipt in plain.receipts:
        assert set(receipt.to_dict()["transitions"][0]) == HISTORICAL_TRANSITION_KEYS


class Lifecycle(Controller):
    """Applies a delta only at value 0 and records the lifecycle calls the Session makes."""

    def __init__(self, *, refuse=False):
        super().__init__()
        self.events, self.applied, self.refuse = [], [], refuse

    def begin(self):
        self.events.append("begin")
        self.applied = []

    def increment(self, value):
        self.events.append("advance")
        if value == 0:
            self.applied.append({"port": "increment", "value": value})
            return 5
        return 1

    def manifest(self):
        self.events.append("manifest")
        if self.refuse:
            raise ValueError("staged edit did not fire")
        applied, self.applied = self.applied, []
        return applied or None


def test_lifecycle_begin_precedes_advance_and_manifest_follows_it():
    session = Session(PortedCounter(), {"value": 0})
    controller = Lifecycle()
    session.probe(controller)
    assert controller.events == ["begin", "advance"]
    controller.events.clear()
    session.continue_(2, ports=controller)
    assert controller.events == ["begin", "advance", "manifest"] * 2


def test_probe_never_influences_a_later_receipt():
    session = Session(PortedCounter(), {"value": 0})
    controller = Lifecycle()
    session.probe(controller)  # applies at value 0 and leaves a stale record behind
    assert controller.applied
    receipt = session.continue_(1, ports=Lifecycle())  # value 0 -> 5 with another controller
    session.probe(controller)  # value is 5 now: the probe applies nothing
    entry = session.continue_(1, ports=controller).to_dict()["transitions"][0]
    assert "port_edits" not in entry
    assert "port_edits" in receipt.to_dict()["transitions"][0]


def test_refusing_manifest_fails_the_transition_without_a_receipt():
    session = Session(PortedCounter(), {"value": 0})
    before = session.capture(retain=False).fingerprint
    with pytest.raises(ValueError, match="did not fire"):
        session.continue_(ports=Lifecycle(refuse=True))
    assert session.capture(retain=False).fingerprint == before
    assert _counts(session) == (0, 1, 0)


def test_decoder_probe_of_an_unaddressed_layer_calls_probe_check():
    from saturn_pub.adapters.decoder import DecoderAdapter

    class Strict(LayerProbe):
        def probe_check(self):
            if not self.installs:
                raise ValueError("controller never met its layer")

    adapter = DecoderAdapter.tiny("qwen2", granularity="operation")
    session = adapter.session([1, 2, 3, 4])
    session.continue_()
    with pytest.raises(ValueError, match="never met"):
        session.probe(Strict(1, stop=False))
    assert session.probe(Strict(0, stop=False))["stopped"] is False
    assert _counts(session) == (1, 0, 0)


class ExternalCounter(PortedCounter):
    """An adapter whose native call mutates state outside the session state (a registered
    side-effect closure): ``external`` is captured and restored by the Session."""

    def __init__(self):
        self.external = 100

    def capture_external(self):
        return self.external

    def restore_external(self, snapshot):
        self.external = snapshot

    def advance(self, state, *, ports=None):
        self.external += 1
        return super().advance(state, ports=ports)


@pytest.mark.parametrize("how", ["normal", "stopping", "raising"])
def test_probe_restores_adapter_external_state(how):
    adapter = ExternalCounter()
    session = Session(adapter, {"value": 3})
    controller = Controller(stop=how == "stopping", fail=how == "raising")
    if how == "raising":
        with pytest.raises(RuntimeError, match="controller failure"):
            session.probe(controller)
    else:
        assert session.probe(controller)["stopped"] is (how == "stopping")
    assert controller.seen == [3]  # the native call did run (and mutated the external state)
    assert adapter.external == 100
    assert session.read("value") == 3 and _counts(session) == (0, 0, 0)


class Reach(Controller):
    """A controller that addresses one site and refuses a call that cannot reach it."""

    def __init__(self, site, **kwargs):
        super().__init__(**kwargs)
        self.site, self.calls = site, []

    def check_ahead(self, ahead):
        self.calls.append(tuple(ahead))
        if self.site not in ahead:
            raise ValueError(f"site {self.site} is not reached by {tuple(ahead)}")


class Ahead(PortedCounter):
    def ports_ahead(self, state, steps):
        return tuple(range(state["value"], state["value"] + steps))


def test_continue_asks_the_controller_before_running_anything():
    session = Session(Ahead(), {"value": 0})
    cut = session.capture(retain=False).fingerprint
    far = Reach(5)
    with pytest.raises(ValueError, match="site 5 is not reached"):
        session.continue_(2, ports=far)
    assert far.seen == [] and far.calls == [(0, 1)]  # nothing ran
    assert session.capture(retain=False).fingerprint == cut and _counts(session) == (0, 0, 0)
    ok = Reach(1)
    session.continue_(2, ports=ok)
    assert session.read("value") == 2
    # an adapter that cannot say (no ports_ahead) is not asked
    plain = Session(PortedCounter(), {"value": 0})
    plain.continue_(1, ports=Reach(99))


def test_end_call_runs_after_the_last_transition_and_can_refuse():
    class Tracked(Controller):
        def __init__(self, refuse=False):
            super().__init__()
            self.events, self.refuse = [], refuse

        def begin_call(self):
            self.events.append("begin_call")

        def end_call(self):
            self.events.append("end_call")
            if self.refuse:
                raise ValueError("never installed")

    session = Session(PortedCounter(), {"value": 0})
    tracked = Tracked()
    session.continue_(2, ports=tracked)
    assert tracked.events == ["begin_call", "end_call"]
    before = session.capture(retain=False).fingerprint
    with pytest.raises(ValueError, match="never installed"):
        session.continue_(1, ports=Tracked(refuse=True))
    assert session.capture(retain=False).fingerprint == before


def test_decoder_ports_ahead_names_the_layer_of_each_coming_transition():
    from saturn_pub.adapters.decoder import DecoderAdapter

    adapter = DecoderAdapter.tiny("qwen2", granularity="operation")
    layers = adapter.layers
    session = adapter.session([1, 2, 3])
    state = session.capture(retain=False).payload
    assert adapter.ports_ahead(state, 2) == (None, 0)  # embed, then layer 0
    ahead = adapter.ports_ahead(state, layers + 9)
    assert ahead[:1] == (None,) and ahead[1 : 1 + layers] == tuple(range(layers))
    assert ahead[1 + layers : 6 + layers] == (None,) * 5  # norm, readout, sample, commit, embed
    assert ahead[6 + layers] == 0  # the next row's layer 0
    # the prediction equals what a real session does, for both granularities
    for granularity in ("operation", "layer"):
        real = DecoderAdapter.tiny("qwen2", granularity=granularity)
        walk = real.session([1, 2, 3])
        predicted = real.ports_ahead(walk.capture(retain=False).payload, 3 * layers + 12)
        seen = []
        for _ in predicted:
            state = walk.capture(retain=False).payload
            ran_layer = state["hidden"] is not None and state["layer"] < real.layers
            seen.append(state["layer"] if ran_layer else None)
            walk.continue_()
        assert tuple(seen) == predicted, granularity
    for _ in range(layers + 2):
        session.continue_()  # now at readout
    state = session.capture(retain=False).payload
    assert adapter.ports_ahead(state, 3) == (None, None, None)  # readout, sample, commit
    assert adapter.ports_ahead(state, 5) == (None, None, None, None, 0)
