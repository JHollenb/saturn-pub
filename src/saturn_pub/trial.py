"""Arbitrate a standard-instrument reading against the native consumer.

This module ports the owner's *Instrument Trial* into the public toolkit. It has two
public jobs, neither of which imports a model framework at module load:

* :func:`verify_bundle` re-derives the shipped Instrument Trial verdicts from a
  hash-pinned receipt bundle using frozen mechanical decision rules. Stdlib only, no
  torch; an outside researcher can reproduce the six case verdicts offline.

* :func:`arbitrate` takes a verdict a standard instrument already declared (activation
  patching, a linear/dictionary probe, a cosine readout, an SAE ablation, or any
  external reader) and lets the *unchanged native consumer* decide the verdict-grade
  question, using saturn-pub's own Session / fork / compare / Receipt. It returns a
  :class:`TrialRow` whose verdict is ``agree``, ``invert`` or ``inconclusive`` and which
  records the frozen :class:`DecisionRule` that produced it.

The arbiter in every case is the rest of the model run to completion, not an internal
metric. A reading is a candidate; only the consumer decides. The :class:`Reading` carries
a ``source`` seam so a reading produced by an external tool (a future circuit-tracer
``native_edge_test``) can be arbitrated the same way as one declared by hand.

See :mod:`saturn_pub` docs ``docs/trial.md`` and the seven measurement traps in
:data:`TRAPS`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .core import Act, Adapter, Receipt, Session, StateCut
from .values import clone, describe, digest

# --------------------------------------------------------------------------------------
# Offline bundle verifier: frozen mechanical decision rules, re-derived from receipts.
# --------------------------------------------------------------------------------------

NECESSITY_THRESHOLD = 0.15
COSINE_RECOVERED_THRESHOLD = 0.99

_TRIAL_DATA = Path(__file__).resolve().parent / "trial_data"
_PACKAGE_BUNDLE = _TRIAL_DATA / "bundle"
_PACKAGE_EXPECTED = _TRIAL_DATA / "expected.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _derive_case_1(bundle: Path) -> dict[str, str]:
    sweep = _load(bundle / "case-1" / "native-object-mediation.json")
    necessity = [b for b in sweep["panel"]["branches"] if b["kind"] == "necessity"]
    passing = [b for b in necessity if b["donor_progress"] <= NECESSITY_THRESHOLD]
    community = "necessary component exists" if passing else "no necessary component"
    ledger = _load(bundle / "case-1" / "circuit-panel-ledger.json")
    row = next(r for r in ledger["rows"] if r["axis_id"] == "identity_cat_fox")
    saturn = (
        "the 4-site route is jointly necessary as a time-accumulated object"
        if (
            row["level"] == "certified"
            and row["aggregate"]["max_route_ablation_progress"] <= NECESSITY_THRESHOLD
        )
        else "route not certified"
    )
    return {"community": community, "saturn": saturn}


def _derive_case_2_probe(bundle: Path) -> dict[str, str]:
    rep = _load(bundle / "case-2" / "probe-job-f58bcd26f295-report.json")
    gates = rep["gates"]
    if not (gates["parity_ok"] and gates["g1_ok"]):
        return {"community_strengthened": "arm broken (gates failed)"}
    primary = str(rep["model"]["mapper_seeds"][0])
    return {"community_strengthened": rep["foreign_scores"][primary]["verdict"]}


def _derive_case_2(bundle: Path) -> dict[str, str]:
    rep = _load(bundle / "case-2" / "xcond-job-e1ba0cbed889-report.json")
    resid = rep["readout"]["carrier_resid"]
    floor = rep["readout"]["resid_floor_deer_vs_tiger"]
    above = {a: c for a, c in resid.items() if c > floor}
    community = "no decodable content" if not above else "decodable: " + max(above, key=above.get)
    e3 = [b for b in rep["branches"] if b["label"] == "E3:transplant_foreign_subject_into_native"]
    sham = [b for b in rep["branches"] if b["label"] == "E2a:sham_noise_subject_rows"]
    moved = all(b["roi"]["fox"]["progress"] > 0.3 for b in e3)
    sham_flat = all(abs(b["roi"]["fox"]["progress"]) < 0.1 for b in sham)
    gates_exact = all(v == 0.0 for v in rep["gates"].values())
    saturn = (
        "register carries content (transplant moves the subject region "
        "on both seeds, sham flat, exact gates)"
        if moved and sham_flat and gates_exact
        else "not established"
    )
    return {"community": community, "saturn": saturn}


def _derive_case_3(bundle: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"community": {}, "saturn": {}}
    for path in sorted((bundle / "case-3").glob("whitening-job-*-report.json")):
        rep = _load(path)
        for mr in rep["model_reports"]:
            model = mr["model"]
            fractions = mr["transform_negative_fraction_overall"]
            raw = fractions["raw"]
            if not mr["parity_within_tolerance"]:
                out["community"][model] = "arm broken (parity gate failed)"
                continue
            out["community"][model] = (
                "representations differ across families"
                if raw > 0.30
                else "no representational difference detected"
            )
            transformed = min(fractions["centered"], fractions["zca"])
            if raw > 0.30 and transformed <= 0.5 * raw:
                out["saturn"][model] = (
                    "the split is an artifact of fitted-direction transfer "
                    "(collapses under centering/whitening); native geometry "
                    "not divergent"
                )
            elif raw <= 0.30:
                out["saturn"][model] = "no representational difference (consistent with raw)"
            else:
                out["saturn"][model] = (
                    "split persists under whitening; representational difference not ruled out"
                )
    return out


def _derive_case_4(bundle: Path) -> dict[str, str]:
    rep = _load(bundle / "case-4" / "tecm-v2-job-3a259f6d1b9e-report.json")
    smol = rep["comparisons"]["flux2-smol"]
    tecm_cos = [smol[p]["tecm_carrier_cosine"] for p in smol]
    community = (
        "state recovered"
        if all(c >= COSINE_RECOVERED_THRESHOLD for c in tecm_cos)
        else "state not recovered"
    )
    duplicate_mad = rep["checkpoint_control"]["same_process_native_duplicate"]["rgb_mad"]
    render_mad = [smol[p]["tecm_vs_native"]["rgb_mad"] for p in smol]
    template_cos = [smol[p]["template_carrier_cosine"] for p in smol]
    behaviorally_wrong = (
        min(render_mad) > 50.0
        and duplicate_mad == 0.0
        and all(c >= COSINE_RECOVERED_THRESHOLD for c in template_cos)
    )
    saturn = (
        "state not recovered (consumer contradicts the cosine; content-free template matches it)"
        if behaviorally_wrong
        else "not established"
    )
    return {"community": community, "saturn": saturn}


def _derive_case_5(bundle: Path) -> dict[str, str]:
    cells = []
    for path in sorted((bundle / "case-5").glob("mediation-job-*-report.json")):
        rep = _load(path)
        for _axis, entry in rep["axes"].items():
            for gate in entry["certificate"]["gates"]:
                if gate["id"] == "mediation":
                    cells.append(bool(gate["passed"]))
    n_pass = sum(cells)
    saturn = (
        f"not a step artifact: {n_pass}/{len(cells)} axis x step cells hold; failure localized"
        if n_pass >= len(cells) - 1 and len(cells) >= 12
        else "step-generality not established"
    )
    return {
        "community": "mediation holds at the measured step; no claim about other steps",
        "saturn": saturn,
    }


def verify_bundle(
    bundle: str | Path | None = None, expected: str | Path | None = None
) -> dict[str, Any]:
    """Re-derive and check every Instrument Trial case verdict from a receipt bundle.

    With no arguments this checks the bundle shipped inside the package. It verifies
    every receipt against its SHA-256 in ``manifest.json``, re-derives the six case
    verdicts from the raw receipt scalars with the frozen decision rules above, and
    compares them to ``expected.json``. The return value reports ``ok`` plus per-file
    hash status and per-arm derived/expected verdicts. Stdlib only; no torch.
    """
    bundle_path = Path(bundle) if bundle is not None else _PACKAGE_BUNDLE
    expected_path = Path(expected) if expected is not None else _PACKAGE_EXPECTED
    manifest = _load(bundle_path / "manifest.json")
    hashes: list[dict[str, Any]] = []
    hash_ok = True
    for rel, want in sorted(manifest.items()):
        path = bundle_path / rel
        if not path.is_file():
            raise ValueError(f"structural error: missing bundled file {rel}")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = got == want
        hash_ok = hash_ok and ok
        hashes.append({"file": rel, "ok": ok, "sha256": got})
    expected_verdicts = _load(expected_path)
    derived = {
        "case-1": _derive_case_1(bundle_path),
        "case-2": _derive_case_2(bundle_path),
        "case-2-probe": _derive_case_2_probe(bundle_path),
        "case-3": _derive_case_3(bundle_path),
        "case-4": _derive_case_4(bundle_path),
        "case-5": _derive_case_5(bundle_path),
    }
    verdicts: dict[str, Any] = {}
    verdict_ok = True
    for case, arms in sorted(derived.items()):
        for arm, value in sorted(arms.items()):
            want = expected_verdicts.get(case, {}).get(arm)
            ok = value == want
            verdict_ok = verdict_ok and ok
            verdicts[f"{case}/{arm}"] = {"derived": value, "expected": want, "ok": ok}
    return {
        "ok": bool(hash_ok and verdict_ok),
        "hash_ok": bool(hash_ok),
        "verdict_ok": bool(verdict_ok),
        "bundle": str(bundle_path),
        "hashes": hashes,
        "verdicts": verdicts,
        "derived": derived,
    }


# --------------------------------------------------------------------------------------
# Consumer-gated arbitration of a standard-instrument reading.
# --------------------------------------------------------------------------------------

_VERDICTS = frozenset({"agree", "invert", "inconclusive"})
Driver = Callable[[Session], None]


def grade(reading: "Reading", rule: "DecisionRule", effect: float) -> tuple[str, str, str]:
    """Grade a reading against a consumer effect under a frozen rule, controls aside.

    Returns ``(verdict, classification, reason)``. This is exactly the decision the arbiter
    applies once its controls pass, factored out so an external operator -- for example the
    circuit-tracer native edge test, which measures the consumer effect with a forward hook
    rather than a forked :class:`Session` -- can reach the same verdict under the identical
    frozen rule. ``classification`` is the rule's ``present``/``absent``/``ambiguous`` class.
    """
    classification = rule.classify(effect)
    if reading.declined:
        return "inconclusive", classification, "the instrument declined the verdict-grade question"
    if classification == "ambiguous":
        return (
            "inconclusive",
            classification,
            "the consumer effect fell in the rule's ambiguity band",
        )
    present = classification == "present"
    if present == reading.asserts_effect:
        return "agree", classification, "the native consumer agrees with the instrument reading"
    reason = (
        "the native consumer shows the carrier is load-bearing where the "
        "instrument read no effect"
        if present
        else "the native consumer shows no effect where the instrument read one"
    )
    return "invert", classification, reason


@dataclass(frozen=True)
class Reading:
    """A verdict-grade reading a standard instrument declared before arbitration.

    ``asserts_effect`` records the *direction* of the claim as a boolean so the native
    consumer can contradict it: ``True`` when the instrument says the carrier is
    load-bearing / the content is present / the state was recovered, ``False`` when it
    says the opposite. ``declined`` marks a reading that explicitly refused the
    verdict-grade (or generality) question; such a reading is never graded a misreport.

    ``source`` is the seam for an external reader. A reading declared by hand keeps the
    default ``"declared"``; :meth:`from_external` tags one produced by another tool (for
    example a future circuit-tracer ``native_edge_test``) so provenance stays explicit.
    """

    instrument: str
    claim: str
    asserts_effect: bool
    declined: bool = False
    detail: Mapping[str, Any] = field(default_factory=dict)
    source: str = "declared"
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, str) or not self.instrument:
            raise ValueError("reading requires a non-empty instrument label")
        if not isinstance(self.claim, str) or not self.claim:
            raise ValueError("reading requires a non-empty claim")
        if type(self.asserts_effect) is not bool or type(self.declined) is not bool:
            raise ValueError("asserts_effect and declined must be booleans")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("reading source must be a non-empty string")
        object.__setattr__(self, "detail", MappingProxyType(clone(dict(self.detail))))
        object.__setattr__(self, "fingerprint", digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "saturn-pub-trial-reading-v1",
            "instrument": self.instrument,
            "claim": self.claim,
            "asserts_effect": self.asserts_effect,
            "declined": self.declined,
            "detail": describe(dict(self.detail)),
            "source": self.source,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_external(
        cls,
        tool: str,
        claim: str,
        asserts_effect: bool,
        *,
        detail: Mapping[str, Any] | None = None,
        declined: bool = False,
    ) -> Reading:
        """Wrap a reading produced by an external tool, keeping its provenance."""
        if not isinstance(tool, str) or not tool:
            raise ValueError("external reading requires the producing tool name")
        return cls(
            instrument=f"external:{tool}",
            claim=claim,
            asserts_effect=asserts_effect,
            declined=declined,
            detail={} if detail is None else detail,
            source=tool,
        )


@dataclass(frozen=True)
class DecisionRule:
    """A frozen, content-addressed rule that turns a consumer measurement into a class.

    The consumer-closure ``effect`` evaluator supplied to :func:`arbitrate` returns a
    scalar under ``metric`` for which higher means "the native consumer confirms the
    thing the reading was about". This rule thresholds it: ``>= present_threshold`` is
    ``present``, ``<= absent_threshold`` is ``absent``, anything between is ``ambiguous``
    (and arbitrates to ``inconclusive``). The rule's :attr:`fingerprint` is recorded on
    every row so the decision rule is auditable after the fact.
    """

    name: str
    version: str
    metric: str
    present_threshold: float
    absent_threshold: float
    require_exact_gate: bool = True
    require_sham_flat: bool = False
    sham_metric: str | None = None
    sham_tolerance: float = 0.1
    description: str = ""
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.metric:
            raise ValueError("decision rule requires name, version, and a consumer metric key")
        for value in (self.present_threshold, self.absent_threshold, self.sham_tolerance):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("decision-rule thresholds must be numeric")
        if float(self.present_threshold) < float(self.absent_threshold):
            raise ValueError("present_threshold must be >= absent_threshold (higher = present)")
        if type(self.require_exact_gate) is not bool or type(self.require_sham_flat) is not bool:
            raise ValueError("gate requirements must be booleans")
        if self.require_sham_flat and not self.sham_metric:
            raise ValueError("require_sham_flat needs a sham_metric key")
        if self.sham_metric is not None and not isinstance(self.sham_metric, str):
            raise ValueError("sham_metric must be a string key or None")
        object.__setattr__(self, "present_threshold", float(self.present_threshold))
        object.__setattr__(self, "absent_threshold", float(self.absent_threshold))
        object.__setattr__(self, "sham_tolerance", float(self.sham_tolerance))
        object.__setattr__(self, "fingerprint", digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "saturn-pub-trial-decision-rule-v1",
            "name": self.name,
            "version": self.version,
            "metric": self.metric,
            "present_threshold": self.present_threshold,
            "absent_threshold": self.absent_threshold,
            "require_exact_gate": self.require_exact_gate,
            "require_sham_flat": self.require_sham_flat,
            "sham_metric": self.sham_metric,
            "sham_tolerance": self.sham_tolerance,
            "description": self.description,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    def classify(self, effect: float) -> str:
        """Return ``present`` / ``absent`` / ``ambiguous`` for a consumer effect scalar."""
        value = float(effect)
        if value >= self.present_threshold:
            return "present"
        if value <= self.absent_threshold:
            return "absent"
        return "ambiguous"


@dataclass(frozen=True)
class TrialRow:
    """One arbitrated verdict: the reading, the frozen rule, and what the consumer did."""

    reading: Reading
    rule: DecisionRule
    verdict: str
    classification: str
    consumer_effect: float | None
    reason: str
    controls: Mapping[str, Any]
    measurements: Mapping[str, Any]
    parent: str
    receipt: str
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.verdict not in _VERDICTS:
            raise ValueError(f"unknown trial verdict: {self.verdict}")
        object.__setattr__(self, "controls", MappingProxyType(clone(dict(self.controls))))
        object.__setattr__(self, "measurements", MappingProxyType(clone(dict(self.measurements))))
        object.__setattr__(self, "fingerprint", digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "saturn-pub-trial-row-v1",
            "reading": self.reading.fingerprint,
            "decision_rule": self.rule.fingerprint,
            "verdict": self.verdict,
            "classification": self.classification,
            "consumer_effect": self.consumer_effect,
            "reason": self.reason,
            "controls": describe(dict(self.controls)),
            "measurements": describe(dict(self.measurements)),
            "parent": self.parent,
            "receipt": self.receipt,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "saturn-pub-trial-row-v1",
            "fingerprint": self.fingerprint,
            "reading": self.reading.to_dict(),
            "decision_rule": self.rule.to_dict(),
            "verdict": self.verdict,
            "classification": self.classification,
            "consumer_effect": self.consumer_effect,
            "reason": self.reason,
            "controls": dict(self.controls),
            "measurements": dict(self.measurements),
            "parent": self.parent,
            "receipt": self.receipt,
            "arbiter": "unchanged native consumer; a reading is a candidate, not a verdict",
        }


def _as_driver(spec: Act | Sequence[Act] | Driver, steps: int) -> Driver:
    if isinstance(spec, Act):
        acts: tuple[Act, ...] = (spec,)
    elif (
        isinstance(spec, (tuple, list)) and spec and all(isinstance(item, Act) for item in spec)
    ):
        acts = tuple(spec)
    elif callable(spec):
        return spec
    else:
        raise ValueError("intervention must be an Act, a sequence of Acts, or a driver callable")

    def drive(branch: Session) -> None:
        for act in acts:
            branch.apply(act)
        if steps:
            branch.continue_(steps)

    return drive


def _native_driver(spec: Act | Sequence[Act] | Driver | None, steps: int) -> Driver:
    if spec is None:
        return lambda branch: branch.continue_(steps) if steps else None
    return _as_driver(spec, steps)


def arbitrate(
    session_or_adapter: Session | Adapter,
    reading: Reading,
    rule: DecisionRule,
    *,
    effect: Callable[[Mapping[str, Session]], Mapping[str, Any]],
    candidate: Act | Sequence[Act] | Driver,
    steps: int = 1,
    native: Act | Sequence[Act] | Driver | None = None,
    sham: Act | Sequence[Act] | Driver | None = None,
    parent: StateCut | None = None,
) -> TrialRow:
    """Let the native consumer arbitrate a standard-instrument ``reading``.

    The ``candidate`` is the intervention the instrument's claim is about (an
    :class:`~saturn_pub.Act`, a sequence of Acts applied then run ``steps``, or a driver
    callable that drives a freshly forked branch to the consumer-observable end). A
    matching unmodified ``native`` arm and an optional ``sham`` control are run from the
    same parent. ``effect`` is a consumer-closure evaluator; it receives the finished
    branches (``{"native", "candidate"[, "sham"]}``) and must return the rule's ``metric``
    (and ``sham_metric`` when the rule requires a flat sham). The frozen ``rule`` classes
    the effect; the verdict compares that class to what the reading asserted:

    * agree -- the consumer confirms what the instrument claimed,
    * invert -- the consumer contradicts it,
    * inconclusive -- the reading declined, the effect was ambiguous, or a control failed.

    torch is never imported here; whatever adapter the caller built owns the tensors.
    """
    if isinstance(session_or_adapter, Session):
        session = session_or_adapter
        parent_cut = parent or session.capture()
    elif isinstance(session_or_adapter, Adapter):
        if parent is None:
            raise ValueError("arbitrating from an adapter requires a parent StateCut")
        session = Session.from_cut(session_or_adapter, parent)
        parent_cut = parent
    else:
        raise ValueError("arbitrate needs a Session, or an Adapter plus a parent StateCut")
    if not isinstance(reading, Reading) or not isinstance(rule, DecisionRule):
        raise ValueError("arbitrate needs a Reading and a DecisionRule")
    if not callable(effect):
        raise ValueError("effect must be a consumer-closure evaluator callable")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    candidate_driver = _as_driver(candidate, steps)
    native_driver = _native_driver(native, steps)

    native_branch = session.fork(parent_cut)
    native_driver(native_branch)
    candidate_branch = session.fork(parent_cut)
    candidate_driver(candidate_branch)

    branches: dict[str, Session] = {"native": native_branch, "candidate": candidate_branch}
    if sham is not None:
        sham_branch = session.fork(parent_cut)
        _as_driver(sham, steps)(sham_branch)
        branches["sham"] = sham_branch

    controls: dict[str, Any] = {}
    controls_ok = True
    reason = ""
    if rule.require_exact_gate:
        native_check = session.fork(parent_cut)
        native_driver(native_check)
        exact_gate = bool(native_branch.compare(native_check)["equal_payload"])
        controls["exact_gate"] = exact_gate
        if not exact_gate:
            controls_ok = False
            reason = "exact-replay gate failed: the unmodified path did not reproduce"

    metrics = dict(effect(branches))
    if rule.metric not in metrics:
        raise ValueError(f"effect evaluator did not return the rule metric: {rule.metric}")
    consumer_effect = float(metrics[rule.metric])
    classification = rule.classify(consumer_effect)

    if rule.require_sham_flat:
        if rule.sham_metric not in metrics:
            raise ValueError(f"effect evaluator did not return the sham metric: {rule.sham_metric}")
        sham_value = float(metrics[rule.sham_metric])
        sham_flat = abs(sham_value) <= rule.sham_tolerance
        controls["sham_flat"] = sham_flat
        controls["sham_value"] = sham_value
        if controls_ok and not sham_flat:
            controls_ok = False
            reason = "sham control was not flat; the effect is not specific to the intervention"

    if not controls_ok:
        verdict = "inconclusive"
        classification = "controls-failed"
    else:
        verdict, classification, reason = grade(reading, rule, consumer_effect)

    receipt = Receipt.make(
        operation="trial.arbitrate",
        parent=parent_cut.fingerprint,
        model_identity=session.adapter.model_identity,
        execution=dict(session.adapter.execution),
        reading=reading.to_dict(),
        decision_rule=rule.to_dict(),
        verdict=verdict,
        classification=classification,
        consumer_effect=consumer_effect,
        controls=controls,
        measurements=describe(metrics),
        native=native_branch.capture(retain=False).fingerprint,
        candidate=candidate_branch.capture(retain=False).fingerprint,
    )
    session.receipts.append(receipt)
    return TrialRow(
        reading=reading,
        rule=rule,
        verdict=verdict,
        classification=classification,
        consumer_effect=consumer_effect,
        reason=reason,
        controls=controls,
        measurements=metrics,
        parent=parent_cut.fingerprint,
        receipt=receipt.fingerprint,
    )


# --------------------------------------------------------------------------------------
# TRAPS: seven ways to be confidently wrong, as a documented, programmatic checklist.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Trap:
    """One measurement failure mode and the gate that kills it."""

    number: int
    name: str
    failure: str
    gate: str

    def to_dict(self) -> dict[str, Any]:
        return {"number": self.number, "name": self.name, "failure": self.failure, "gate": self.gate}


TRAPS: tuple[Trap, ...] = (
    Trap(
        1,
        "The latent-only false positive",
        "Internal positions of a late merged block showed high alignment with intervention "
        "outcomes while the served output never changed, because the architecture discards "
        "those positions. Activation-magnitude rankings reliably surface these disposal sites.",
        "Confirm every internal claim at an observable the model actually serves "
        "(the two-consumer rule).",
    ),
    Trap(
        2,
        "Contrast collapse inflates alignment",
        "When an intervention collapses the two contrast conditions toward each other, every "
        "later patch scores a higher alignment cosine while absolute effects are negligible.",
        "Verify baseline contrast viability; report baseline-normalized progress, never raw "
        "alignment.",
    ),
    Trap(
        3,
        "The scaffold dominates similarity",
        "Shared structure (positional scaffolds, templates, formatting) dominates global "
        "similarity: a 0.996-cosine reconstruction is compatible with zero semantic content "
        "(bundle case 4 -- a content-free template scores the same).",
        "Test with semantic contrasts, or not at all.",
    ),
    Trap(
        4,
        "A clean output is not a correct output",
        "Strong generative priors render plausible outputs from degenerate internal states.",
        "Output quality checks cannot substitute for output content checks.",
    ),
    Trap(
        5,
        "Terminal readout masquerades as origin",
        "Writing a complete target state at the last pre-output boundary reproduces the target "
        "perfectly and identifies nothing but the readout.",
        "Cross-check late-site interventions with delta-based screens before any 'the concept "
        "lives late' conclusion.",
    ),
    Trap(
        6,
        "Relaxation artifacts read as structure",
        "Graph/flow analyses composed over intervention evidence inherit their relaxations' "
        "artifacts: an empty minimum cut was a property of the relaxation, not evidence the "
        "route has no bottleneck.",
        "Derived graph objects are search guides, never certificates.",
    ),
    Trap(
        7,
        "A live site is not a circuit",
        "An early result named one block-call 'the counting circuit' from a behavioral "
        "association; under strict replication it survived 5 of 16 paired tests and was demoted.",
        "Behavioral liveness is a candidate -- promotion needs the full battery, and every "
        "positive result owes its survival to a control designed to kill it.",
    ),
)


def traps() -> tuple[Trap, ...]:
    """Return the seven measurement traps as a programmatic checklist."""
    return TRAPS


# --------------------------------------------------------------------------------------
# Offline CLI entry point for the bundle verifier.
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the shipped (or a supplied) bundle; exit 0 = reproduced, 3 = mismatch."""
    parser = argparse.ArgumentParser(
        prog="saturn-pub-trial-verify",
        description="Re-derive the Instrument Trial verdicts from a hash-pinned bundle.",
    )
    parser.add_argument("--bundle", type=Path, default=None, help="bundle directory")
    parser.add_argument("--expected", type=Path, default=None, help="expected verdicts JSON")
    args = parser.parse_args(argv)
    try:
        result = verify_bundle(args.bundle, args.expected)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, StopIteration) as err:
        print(f"structural error: {err}")
        return 2
    for row in result["hashes"]:
        print(f"  {'ok' if row['ok'] else 'HASH MISMATCH':14s} {row['file']}")
    for key, row in sorted(result["verdicts"].items()):
        label = "ok" if row["ok"] else "VERDICT MISMATCH"
        print(f"  {label:18s} {key}: {row['derived']}")
    if result["ok"]:
        print("RESULT: all hashes and all mechanical verdicts verified")
        return 0
    print("RESULT: FAILED verification")
    return 3


__all__ = [
    "Reading",
    "DecisionRule",
    "TrialRow",
    "Trap",
    "arbitrate",
    "grade",
    "verify_bundle",
    "traps",
    "TRAPS",
    "NECESSITY_THRESHOLD",
    "COSINE_RECOVERED_THRESHOLD",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
