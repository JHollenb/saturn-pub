"""Mechanics-only two-bank clock organism for symbol lifecycle joins.

The adapter is an explicit public ``saturn_pub`` organism.  Its transition
ticks bank A and a global cursor while preserving bank B.  The phase and age
fields are supplied fixture metadata; no Mamba, VM, or learned clock claim is
made here.
"""

from __future__ import annotations

import time

import pytest
from saturn_pub import Session, StateAddress
from saturn_pub.contracts import ExecutionPoint, SlotSpec, SurfaceManifest, TransitionSpec
from saturn_pub.core import Adapter
from saturn_pub.symbols import SymbolBinding, SymbolTable

from mdb import BenchSession, Project, SaturnPubDebuggerAdapter
from mdb.symbol_lifecycle import ClockWitness, SymbolLifecycle, SymbolOccurrence

PHASE_MODULUS = 2
SESSION_ID = "experiment"
EXPIRY_RULE = {
    "kind": "local-age",
    "operator": ">=",
    "threshold": 2,
    "source": "supplied-fixture-rule",
}


class TwoBankClockAdapter(Adapter):
    """Pure public state machine whose only native transition ticks bank A."""

    model_identity = "public-two-bank-clock-v1"
    execution = {"family": "clock-organism", "adapter": "two-bank-clock-v1"}

    def boundary(self, state):
        return f"tick-a:{state['a_clock']}/global:{state['global_clock']}"

    def point(self, state):
        return ExecutionPoint(
            "clock-organism",
            state["global_clock"],
            f"phase-{state['a_clock'] % PHASE_MODULUS}",
            operator="tick-bank-a",
            index=state["global_clock"],
            next_operation="tick-bank-a",
            local_clock=state["a_clock"],
        )

    def surface(self, state):
        return SurfaceManifest(
            (
                SlotSpec("a_value", role="carrier", writable=True, consumer="two-bank-native"),
                SlotSpec("b_value", role="carrier", writable=True, consumer="two-bank-native"),
                SlotSpec("a_clock", role="runtime", consumer="two-bank-native"),
                SlotSpec("b_clock", role="runtime", consumer="two-bank-native"),
                SlotSpec("global_clock", role="runtime", consumer="two-bank-native"),
            ),
            "two-bank-clock-state-v1",
            "two-bank-native",
        )

    def addresses(self, state):
        return tuple(state)

    def validate(self, state):
        expected = {"a_value", "b_value", "a_clock", "b_clock", "global_clock"}
        if set(state) != expected or any(
            type(value) is not int or value < 0 for value in state.values()
        ):
            raise ValueError("two-bank organism requires nonnegative integer state")

    def transition(self, state):
        return TransitionSpec(
            "tick-bank-a",
            reads=("a_value", "a_clock", "global_clock"),
            writes=("a_value", "a_clock", "global_clock"),
            consumer="two-bank-native",
            footprint="declared",
        )

    def advance(self, state):
        return {
            **state,
            "a_value": state["a_value"] + 1,
            "a_clock": state["a_clock"] + 1,
            "global_clock": state["global_clock"] + 1,
        }


def _native_session():
    return Session(
        TwoBankClockAdapter(),
        {"a_value": 0, "b_value": 0, "a_clock": 0, "b_clock": 0, "global_clock": 0},
    )


def _binding(adapter, bank):
    slot = f"{bank}_value"
    clock = f"{bank}_clock"
    return SymbolBinding.bind(
        adapter.session,
        f"organism.{bank}",
        (StateAddress(slot),),
        context=f"bank-{bank}",
        context_slots=(slot, clock, "global_clock"),
        evidence=("fixture://two-bank-clock-organism",),
        claim=(
            f"Supplied {bank} bank carrier at a native cut; the local-age expiry "
            "rule remains descriptive evidence."
        ),
    )


def _clock(binding, cut, *, local_age, global_age):
    domain = binding.context
    phase = local_age % PHASE_MODULUS
    return ClockWitness(
        f"{domain}-phase-clock",
        "phase",
        domain,
        phase,
        provenance=(f"cas://sha256/{cut}",),
        status="observed",
        metadata={
            "source_cut": cut,
            "model_execution": binding.execution,
            "consumer": "two-bank-native",
            "reset": "native-zero-cut",
            "age": local_age,
            "global_age": global_age,
            "clock_domain": domain,
            "phase_modulus": PHASE_MODULUS,
            "expiry_rule": EXPIRY_RULE,
        },
    )


def _lifecycle(binding, cut, *, local_age, global_age, occurrence_id):
    clock = _clock(binding, cut, local_age=local_age, global_age=global_age)
    occurrence = SymbolOccurrence.for_binding(
        binding,
        occurrence_id=occurrence_id,
        session_id=SESSION_ID,
        source_cut=cut,
        locations=binding.locations,
        branch="main",
        execution=binding.execution,
        clock=clock,
        evidence_refs=(f"cas://sha256/{cut}",),
        status="observed",
        metadata={
            "clock_domain": binding.context,
            "expiry_rule": EXPIRY_RULE,
            "interpretation": "supplied-age-observation-only",
        },
    )
    return SymbolLifecycle.define(binding, definitions=(occurrence,))


def _open_two_bank(tmp_path):
    project = Project(tmp_path / "project")
    adapter = SaturnPubDebuggerAdapter(_native_session(), project=project, session_id=SESSION_ID)
    owner = BenchSession.from_backend(adapter, identity=adapter.identity)
    debug = owner.debugger()

    a0, b0 = _binding(adapter, "a"), _binding(adapter, "b")
    adapter.symbols = SymbolTable((a0, b0))
    adapter.symbol_context = "bank-a"
    before = adapter.checkpoint("before-a-ticks")["cut_id"]

    debug.step(2)
    assert adapter.session.read("a_clock") == 2
    assert adapter.session.read("b_clock") == 0
    assert adapter.session.read("global_clock") == 2

    a2, b2 = _binding(adapter, "a"), _binding(adapter, "b")
    adapter.symbols = SymbolTable((a2, b2))
    adapter.symbol_context = "bank-a"
    after = adapter.checkpoint("after-a-ticks")["cut_id"]

    records = {
        "a0": _lifecycle(a0, before, local_age=0, global_age=0, occurrence_id="a-before"),
        "b0": _lifecycle(b0, before, local_age=0, global_age=0, occurrence_id="b-before"),
        "a2": _lifecycle(a2, after, local_age=2, global_age=2, occurrence_id="a-after"),
        "b2": _lifecycle(b2, after, local_age=0, global_age=2, occurrence_id="b-after"),
    }
    for record in records.values():
        project.save_symbol_lifecycle(SESSION_ID, record)
    return project, adapter, owner, debug, records, before, after


def _match(project, port, cut):
    result = project.symbols_at(SESSION_ID, port, cut_id=cut, selector=())
    assert len(result["matches"]) == 1
    return result["matches"][0]


def _local_age_for_domain(clock, domain):
    """The fixture's explicit association check; Project does no policy inference."""
    if clock["domain"] != domain or clock["metadata"].get("clock_domain") != domain:
        raise ValueError("clock domain mismatch")
    return clock["metadata"]["age"]


def test_project_query_keeps_local_and_global_age_at_native_cuts(tmp_path):
    project, adapter, owner, _, _, before, after = _open_two_bank(tmp_path)
    try:
        a_before = _match(project, "a_value", before)
        b_before = _match(project, "b_value", before)
        a_after = _match(project, "a_value", after)
        b_after = _match(project, "b_value", after)

        assert a_before["clock"]["metadata"]["age"] == 0
        assert b_before["clock"]["metadata"]["age"] == 0
        assert a_after["clock"]["metadata"]["age"] == 2
        assert b_after["clock"]["metadata"]["age"] == 0
        assert a_after["clock"]["metadata"]["global_age"] == 2
        assert b_after["clock"]["metadata"]["global_age"] == 2
        assert a_after["clock"]["domain"] == "bank-a"
        assert b_after["clock"]["domain"] == "bank-b"
        assert adapter.session.read("b_clock") == 0
        assert b_after["validity_status"] == "unknown"
        assert b_after["clock"]["metadata"]["expiry_rule"] == EXPIRY_RULE
    finally:
        owner.close()


def test_restore_and_replay_reproduce_local_age_without_host_wallclock(tmp_path, monkeypatch):
    project, adapter, owner, _, _, before, after = _open_two_bank(tmp_path)
    try:
        before_cut = project.load_cut(SESSION_ID, before)
        after_cut = project.load_cut(SESSION_ID, after)

        replayed = adapter.session.replay(before_cut, steps=2)
        assert replayed.read("a_clock") == 2
        assert replayed.read("b_clock") == 0
        assert replayed.read("global_clock") == 2

        adapter.session.restore(before_cut)
        assert adapter.session.read("a_clock") == 0
        assert adapter.session.read("b_clock") == 0
        adapter.session.restore(after_cut)
        assert adapter.session.read("a_clock") == 2
        assert adapter.session.read("b_clock") == 0

        saved = project.symbols_at(SESSION_ID, "b_value", cut_id=after, selector=())
        monkeypatch.setattr(time, "time", lambda: 10**15)
        assert project.symbols_at(SESSION_ID, "b_value", cut_id=after, selector=()) == saved
    finally:
        owner.close()


def test_phase_wrap_and_wrong_bank_clock_remain_explicit_witnesses(tmp_path):
    project, _, owner, _, records, before, after = _open_two_bank(tmp_path)
    try:
        a_before = _match(project, "a_value", before)
        a_after = _match(project, "a_value", after)
        assert a_before["clock"]["reading"] == a_after["clock"]["reading"] == 0
        assert a_before["clock"]["metadata"]["age"] == 0
        assert a_after["clock"]["metadata"]["age"] == 2
        # The wrapped phase is not enough to recover elapsed age.
        assert a_after["clock"]["metadata"]["phase_modulus"] == PHASE_MODULUS

        b_after = records["b2"]
        wrong_clock = _clock(
            records["a2"].binding,
            after,
            local_age=2,
            global_age=2,
        )
        wrong = b_after.invalidate(
            scope="occurrence:b-after",
            reason="retain a-bank witness separately; it is not b-bank local time",
            evidence_refs=("policy://clock-domain-association",),
            clock=wrong_clock,
            metadata={"clock_domain": "bank-a", "interpretation": "separate-witness"},
        )
        saved = project.save_symbol_lifecycle(SESSION_ID, wrong)
        assert saved["lifecycle"]["invalidations"][0]["clock"]["domain"] == "bank-a"

        queried = project.symbols_at(SESSION_ID, "b_value", cut_id=after, selector=())
        # The original sealed lifecycle and the later invalidation are both
        # retained; neither row is rewritten into a new local-age policy.
        assert len(queried["matches"]) == 2
        primary = next(row for row in queried["matches"] if not row["invalidations"])
        with_invalidations = next(row for row in queried["matches"] if row["invalidations"])
        assert _local_age_for_domain(primary["clock"], "bank-b") == 0
        assert with_invalidations["clock"]["domain"] == "bank-b"
        assert with_invalidations["invalidations"][0]["clock"]["domain"] == "bank-a"
        with pytest.raises(ValueError, match="clock domain"):
            _local_age_for_domain(with_invalidations["invalidations"][0]["clock"], "bank-b")
    finally:
        owner.close()
