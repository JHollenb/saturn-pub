"""First-divergence (and friends) bisect over an ordered cut lattice.

Debugging a branched model run keeps reducing to one question asked many times:
*at which cut does a property first hold?* This module owns only the search
mechanics and the receipt ledger. Model execution stays behind caller-supplied
callbacks, so the same core serves token cuts, layer cuts, denoising steps, and
accepted training intervals without importing any runtime (or ``torch``).

Three verbs share one engine:

* :func:`first_true` -- binary search for the boundary of a monotone predicate;
* :func:`first_divergence` -- binary search for the first cut where two
  exact-replay arms stop agreeing, compared through content digests of typed
  boundary state;
* :func:`splice_bisect` -- binary search for the smallest cut whose behavioral
  transfer score crosses a threshold.

Every probe is recorded, in probe order, in a hash-chained ledger: each record
carries ``prev_sha256`` and its own ``sha256`` over canonical JSON of the record
minus the ``sha256`` field, so a saved receipt can be re-verified without
re-running the callbacks. Non-monotone lattices are a *finding*, not an error:
the result carries ``monotonicity_violation`` instead of raising.

Integration with ``saturn_pub``: the callbacks are the only seam. To bisect two
retained ``StateCut`` arms (or two ``ReplayBundle`` branches), the caller builds
a :class:`CutHandle` lattice over the step index and supplies a probe that reads
each arm's digest at a cut -- see :func:`first_divergence_pairs` and
``examples/evidence_bisect.py``. Nothing here imports the rest of the package.

Claim boundary: this is mechanics-only infrastructure. A bisect receipt
localizes *where* a caller-supplied signal changes, not *why* it changes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ._canonical import GENESIS_SHA256, content_fingerprint

BISECT_SCHEMA = "saturn-pub-divergence-bisect-v1"


@dataclass
class CutHandle:
    """One addressable cut in an ordered lattice.

    ``index`` is the caller's coordinate for the cut (token position, layer,
    accepted training step). The lattice order is the list order, and indices
    must increase strictly along it.
    """

    cut_id: str
    index: int
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ArmDigests(Protocol):
    """Caller callback: both arms' digests at one cut, computed once per cut.

    Implementations typically replay two retained ``saturn_pub`` arms up to
    ``cut.index`` and return each arm's ``StateCut.fingerprint`` (or a selected
    port digest). The engine probes each lattice position at most once, so an
    expensive native replay runs a logarithmic number of times, not linearly.
    """

    def __call__(self, cut: CutHandle) -> tuple[str, str]: ...


@dataclass
class ProbeRecord:
    """One executed probe, sealed into the hash chain.

    Predicate and splice probes fill ``verdict``; divergence probes fill
    ``digest_a``/``digest_b`` and leave ``verdict`` as ``None``. ``sha256``
    covers the canonical JSON of every other field, and ``prev_sha256`` links to
    the previous probe (``GENESIS_SHA256`` for the first).
    """

    index: int
    cut_id: str
    verdict: bool | None
    digest_a: str | None
    digest_b: str | None
    evidence_sha256: str
    prev_sha256: str
    sha256: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def observed_true(self) -> bool:
        """The probe's boolean reading, derived for divergence probes."""

        if self.verdict is not None:
            return self.verdict
        return self.digest_a != self.digest_b


@dataclass
class BisectResult:
    """Receipt for one bisect run: the located boundary plus its probe ledger.

    ``first_true_index``/``first_true_cut_id`` are ``None`` when the property
    never fires on the lattice (or the lattice is empty). For
    :func:`first_divergence` the "true" property is "the two digests differ".
    ``linear_probe_count`` is the number of reads a naive scan would have made
    (the lattice size); comparing it with ``probe_count`` is the measured
    economics of the search.
    """

    verb: str
    first_true_index: int | None
    first_true_cut_id: str | None
    probes: list[ProbeRecord]
    probe_count: int
    lattice_size: int
    monotonicity_violation: bool
    ledger_sha256: str | None

    @property
    def linear_probe_count(self) -> int:
        """Reads a linear scan of the whole lattice would have taken."""

        return self.lattice_size

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": BISECT_SCHEMA,
            "verb": self.verb,
            "first_true_index": self.first_true_index,
            "first_true_cut_id": self.first_true_cut_id,
            "probe_count": self.probe_count,
            "linear_probe_count": self.linear_probe_count,
            "lattice_size": self.lattice_size,
            "monotonicity_violation": self.monotonicity_violation,
            "ledger_sha256": self.ledger_sha256,
            "probes": [probe.to_json_dict() for probe in self.probes],
        }


class _Ledger:
    """Appends probe records and threads the hash chain through them."""

    def __init__(self) -> None:
        self.probes: list[ProbeRecord] = []
        self._prev = GENESIS_SHA256

    def record(
        self,
        *,
        index: int,
        cut_id: str,
        evidence: Any,
        verdict: bool | None = None,
        digest_a: str | None = None,
        digest_b: str | None = None,
    ) -> ProbeRecord:
        body = {
            "index": index,
            "cut_id": cut_id,
            "verdict": verdict,
            "digest_a": digest_a,
            "digest_b": digest_b,
            "evidence_sha256": content_fingerprint(evidence),
            "prev_sha256": self._prev,
        }
        record = ProbeRecord(**body, sha256=content_fingerprint(body))
        self.probes.append(record)
        self._prev = record.sha256
        return record


def verify_ledger(probes: Sequence[ProbeRecord | Mapping[str, Any]]) -> bool:
    """Re-hash a probe chain; True iff every record and every link agrees.

    Any entry that is not a :class:`ProbeRecord` or a mapping makes the chain
    unverifiable and returns ``False`` (never raises).
    """

    prev = GENESIS_SHA256
    for probe in probes:
        if isinstance(probe, ProbeRecord):
            record = probe.to_json_dict()
        elif isinstance(probe, Mapping):
            record = dict(probe)
        else:
            return False
        claimed = record.get("sha256")
        body = {key: value for key, value in record.items() if key != "sha256"}
        if record.get("prev_sha256") != prev or content_fingerprint(body) != claimed:
            return False
        prev = str(claimed)
    return True


def _validate_cuts(cuts: Sequence[CutHandle]) -> None:
    for earlier, later in zip(cuts, cuts[1:], strict=False):
        if later.index <= earlier.index:
            raise ValueError(
                "cut handles must be ordered by strictly increasing index; "
                f"got {earlier.index} then {later.index}"
            )


def _observed_violation(probes: Sequence[ProbeRecord]) -> bool:
    """True when the recorded probes contradict the monotone contract."""

    seen_true = False
    for probe in sorted(probes, key=lambda record: record.index):
        if probe.observed_true:
            seen_true = True
        elif seen_true:
            return True
    return False


def _bisect(
    verb: str,
    cuts: Sequence[CutHandle],
    probe_cut: Callable[[CutHandle], tuple[bool, dict[str, Any]]],
    *,
    verify_endpoints: bool,
) -> BisectResult:
    """Shared engine: cached binary search with a hash-chained probe ledger.

    ``probe_cut`` returns ``(observed_true, record_kwargs)`` where the kwargs
    feed :meth:`_Ledger.record`. Each lattice position is probed at most once.
    """

    cuts = list(cuts)
    _validate_cuts(cuts)
    ledger = _Ledger()
    cache: dict[int, bool] = {}

    def probe(position: int) -> bool:
        if position not in cache:
            cut = cuts[position]
            observed, kwargs = probe_cut(cut)
            ledger.record(index=cut.index, cut_id=cut.cut_id, **kwargs)
            cache[position] = observed
        return cache[position]

    size = len(cuts)
    found: int | None = None
    if size == 0:
        found = None
    elif verify_endpoints:
        first = probe(0)
        last = probe(size - 1) if size > 1 else first
        if not last:
            found = None
        elif first:
            found = 0
        else:
            low, high = 1, size - 1
            while low < high:
                mid = (low + high) // 2
                if probe(mid):
                    high = mid
                else:
                    low = mid + 1
            found = low
    else:
        low, high = 0, size - 1
        while low < high:
            mid = (low + high) // 2
            if probe(mid):
                high = mid
            else:
                low = mid + 1
        found = low if probe(low) else None

    return BisectResult(
        verb=verb,
        first_true_index=cuts[found].index if found is not None else None,
        first_true_cut_id=cuts[found].cut_id if found is not None else None,
        probes=ledger.probes,
        probe_count=len(ledger.probes),
        lattice_size=size,
        monotonicity_violation=_observed_violation(ledger.probes),
        ledger_sha256=ledger.probes[-1].sha256 if ledger.probes else None,
    )


def first_true(
    cuts: Sequence[CutHandle],
    predicate: Callable[[CutHandle], tuple[bool, dict[str, Any]]],
    *,
    verify_endpoints: bool = True,
) -> BisectResult:
    """Locate the first cut where a monotone predicate fires.

    Contract: the predicate reads ``False ... False True ... True`` over the
    lattice. With ``verify_endpoints`` the first and last cuts are probed up
    front: a False last cut short-circuits to a no-fire result, a True first cut
    short-circuits to the first index. Observed contradictions of the monotone
    contract are recorded on the result, never raised.
    """

    def probe_cut(cut: CutHandle) -> tuple[bool, dict[str, Any]]:
        verdict, evidence = predicate(cut)
        verdict = bool(verdict)
        return verdict, {"verdict": verdict, "evidence": dict(evidence)}

    return _bisect("first_true", cuts, probe_cut, verify_endpoints=verify_endpoints)


def first_divergence(
    cuts: Sequence[CutHandle],
    digest_a: Callable[[CutHandle], str],
    digest_b: Callable[[CutHandle], str],
) -> BisectResult:
    """Locate the first cut where two exact-replay arms stop agreeing.

    Contract: the arms share a prefix, then diverge permanently, as read through
    content digests of the typed boundary state at each cut. Equal digests at
    the last cut yield a no-divergence result; a divergence that "heals" by the
    last cut is recorded as a monotonicity violation.

    ``digest_a`` and ``digest_b`` are each called once per probed cut. When a
    single replay yields both arms' digests together, prefer
    :func:`first_divergence_pairs` so the expensive work runs once per cut.
    """

    def probe_cut(cut: CutHandle) -> tuple[bool, dict[str, Any]]:
        left = str(digest_a(cut))
        right = str(digest_b(cut))
        return _divergence_record(left, right)

    return _bisect("first_divergence", cuts, probe_cut, verify_endpoints=True)


def first_divergence_pairs(
    cuts: Sequence[CutHandle],
    digests: ArmDigests,
) -> BisectResult:
    """First divergence where one callback returns both arms' digests per cut.

    This is the ``saturn_pub`` integration form: ``digests(cut)`` replays the
    native and candidate arms (retained ``StateCut`` branches or
    ``ReplayBundle`` heads) to ``cut.index`` and returns
    ``(native_digest, candidate_digest)``. The pair callback is invoked at most
    once per lattice position, so a native model replay runs only on the
    logarithmic set of probed cuts.
    """

    def probe_cut(cut: CutHandle) -> tuple[bool, dict[str, Any]]:
        left, right = digests(cut)
        return _divergence_record(str(left), str(right))

    return _bisect("first_divergence", cuts, probe_cut, verify_endpoints=True)


def _divergence_record(left: str, right: str) -> tuple[bool, dict[str, Any]]:
    diverged = left != right
    return diverged, {
        "digest_a": left,
        "digest_b": right,
        "evidence": {"digest_a": left, "digest_b": right, "diverged": diverged},
    }


def splice_bisect(
    cuts: Sequence[CutHandle],
    splice_score: Callable[[CutHandle], tuple[float, dict[str, Any]]],
    threshold: float,
    *,
    larger_transfers: bool = True,
) -> BisectResult:
    """Locate the smallest cut whose splice transfer score crosses a threshold.

    ``splice_score`` reports the behavioral transfer when arm-A state is spliced
    into the arm-B suffix at a cut. Crossing is inclusive: with
    ``larger_transfers`` a score of exactly ``threshold`` counts as crossed, and
    with ``larger_transfers=False`` the comparison flips to ``<=``. The monotone
    contract and violation recording match :func:`first_true`.
    """

    threshold = float(threshold)

    def probe_cut(cut: CutHandle) -> tuple[bool, dict[str, Any]]:
        score, evidence = splice_score(cut)
        score = float(score)
        crossed = score >= threshold if larger_transfers else score <= threshold
        return crossed, {
            "verdict": crossed,
            "evidence": {
                "score": score,
                "threshold": threshold,
                "larger_transfers": larger_transfers,
                "evidence": dict(evidence),
            },
        }

    return _bisect("splice_bisect", cuts, probe_cut, verify_endpoints=True)


def make_cuts(count: int, *, prefix: str = "cut") -> list[CutHandle]:
    """Build a contiguous lattice of cut handles indexed ``0 .. count - 1``."""

    return [CutHandle(cut_id=f"{prefix}/{position}", index=position) for position in range(count)]


def probe_economics(result: BisectResult) -> dict[str, Any]:
    """Probes actually spent versus a linear scan of the same lattice."""

    linear = result.linear_probe_count
    probed = result.probe_count
    return {
        "probe_count": probed,
        "linear_probe_count": linear,
        "saved_probes": max(linear - probed, 0),
        "speedup": (linear / probed) if probed else None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bisect a synthetic boolean lattice and verify bisect receipts"
    )
    commands = parser.add_subparsers(dest="bisect_command", required=True)
    demo = commands.add_parser(
        "demo", help='run first_true over a JSON file: {"lattice": [false, false, true, true]}'
    )
    demo.add_argument("lattice", type=Path, help="JSON file with a boolean 'lattice' array")
    demo.add_argument("--no-verify-endpoints", action="store_true")
    demo.add_argument("--out", type=Path, help="also write the receipt JSON to this path")
    verify = commands.add_parser("verify", help="re-check a saved receipt ledger")
    verify.add_argument("receipt", type=Path, help="receipt JSON (BisectResult or bare probe list)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bisect_command == "demo":
        payload = json.loads(args.lattice.read_text(encoding="utf-8"))
        lattice = payload.get("lattice") if isinstance(payload, dict) else None
        if not isinstance(lattice, list) or not all(isinstance(item, bool) for item in lattice):
            raise SystemExit('lattice file must be {"lattice": [false, false, true, ...]}')

        def predicate(cut: CutHandle) -> tuple[bool, dict[str, Any]]:
            return lattice[cut.index], {"lattice_value": lattice[cut.index]}

        result = first_true(
            make_cuts(len(lattice)),
            predicate,
            verify_endpoints=not args.no_verify_endpoints,
        )
        text = json.dumps(result.to_json_dict(), indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    payload = json.loads(args.receipt.read_text(encoding="utf-8"))
    probes = payload.get("probes", []) if isinstance(payload, dict) else payload
    if not isinstance(probes, list):
        raise SystemExit("receipt must be a BisectResult object or a bare probe list")
    chain_valid = verify_ledger(probes)
    head_matches = True
    if isinstance(payload, dict) and payload.get("ledger_sha256") is not None:
        head_matches = (
            bool(probes)
            and isinstance(probes[-1], Mapping)
            and probes[-1].get("sha256") == payload["ledger_sha256"]
        )
    verified = chain_valid and head_matches
    print(
        json.dumps(
            {
                "receipt": str(args.receipt),
                "probe_count": len(probes),
                "chain_valid": chain_valid,
                "head_matches": head_matches,
                "verified": verified,
            },
            sort_keys=True,
        )
    )
    return 0 if verified else 1


__all__ = [
    "BISECT_SCHEMA",
    "GENESIS_SHA256",
    "ArmDigests",
    "BisectResult",
    "CutHandle",
    "ProbeRecord",
    "first_divergence",
    "first_divergence_pairs",
    "first_true",
    "make_cuts",
    "probe_economics",
    "splice_bisect",
    "verify_ledger",
]


if __name__ == "__main__":
    raise SystemExit(main())
