"""One panel generator for attention experiments: typed arms, mandatory controls, budgets, seals, verdicts.

Framework-free: plain dataclasses and JSON. Nothing here imports torch; numpy is imported lazily
(random controls only), and :mod:`mdb.attention_port` (itself torch-free at import) only when an
arm is mapped to port cuts.

Two arm families share one record type, :class:`Arm`:

* read cuts (``family="read"``): ``cut`` (cut the unit set), ``keep`` (cut its complement), in
  both cut types (``cut_type="zero"`` = ``renormalize=False``, ``"renorm"`` = ``renormalize=True``),
  with their floor arms, matched random controls, other-item controls and references;
* value replacement (``family="value"``, the Codex family): ``singleton``, ``complement``,
  ``coalition``, ``dose``, ``wrong_target``, ``wrong_payload``, ``reciprocal``, ``sham``,
  ``replay``, ``fixed_distance`` and the references ``native`` / ``donor_native``.

Mappings: :func:`port_cuts` (arm -> ``attention_port`` ``Cut`` specs), :func:`port_kwargs` /
:func:`check_port` (the ``AttentionPort`` settings the plan was hashed with), :func:`value_spec`
(arm -> side, payload side, target row, payload row, cells, dose) and ONE dose blend,
:func:`blend`. :func:`to_legacy` writes the hand-written ``{arm, kind, cells, dose}`` schema.

Analysis: :func:`attach_results` refuses results whose plan hash differs or whose seal does not
verify; :func:`arm_effects` gives each test arm's effect beside its floor; :func:`verdicts`
returns one record per unit set with status ``supported`` (full scope) / ``set_only`` (the weaker,
named scope) / ``not_supported`` / ``unresolved`` / ``control_missing``, the alternative
explanations ruled out and remaining, and a slot for a picture judgement.

See ``docs/ATTENTION-PANELS.md`` ("What a verdict does and does not say").
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass, field, replace

SCHEMA = "mdb-attention-panel-v1"
RESULTS_SCHEMA = "mdb-attention-panel-results-v1"
SEAL_SCHEMA = "mdb-attention-panel-seal-v1"
CUT_TYPES = ("zero", "renorm")
PREFIX = {"zero": "z", "renorm": "m"}
ROLES = ("test", "control", "floor", "reference")
READ_KINDS = ("cut", "keep", "floor_sham", "floor_sham_recompute", "floor_cut_recompute",
              "condition", "native", "all_cut", "deletion", "sham_all")
VALUE_KINDS = ("singleton", "complement", "coalition", "dose", "wrong_target", "wrong_payload",
               "reciprocal", "sham", "replay", "fixed_distance", "native", "donor_native")
# ``set_only``: the claim holds for this unit set against the alternatives that were ruled out,
# but at least one alternative (``remaining_alternatives``) is open; it is NOT ``supported``.
STATUSES = ("supported", "set_only", "not_supported", "unresolved", "control_missing")
WAIVABLE = ("floors", "both_cut_types", "random_cut", "random_keep", "other_item", "partial_universe")
VALUE_WAIVABLE = ("sham", "native_reference")
VALUE_CONTROLS = ("sham", "wrong_payload", "wrong_target", "reciprocal")
VALUE_REFERENCES = ("native", "replay", "donor_native")
# alternative explanations a verdict must rule out, per claim
ALTERNATIVES = {"necessary": ("generic_damage", "mass", "set_size"), "sufficient": ("mass", "set_size")}
# what a beaten random control rules out, by its mass relation to the target set
RULES_OUT_BY_MASS = {"unknown": ("set_size",), "less": ("set_size",), "matched_by_reuse": ("set_size",),
                     "matched": ("mass", "set_size"), "more": ("mass", "set_size")}
DEFAULT_RECOVERY = 0.7
FLUX_LANE_CAP = 256
# Arm fields added after the first release: left out of the JSON when at their default
_OMIT_IF_DEFAULT = {"rules_out": (), "mass_relation": None}


class PanelError(ValueError):
    """A refused panel, budget, seal or result attachment."""


# ----------------------------------------------------------------------------------- utilities

def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _plain(value):
    """numpy scalars/arrays -> Python numbers/lists (recursively), so plans hash as JSON."""
    if type(value).__module__ == "numpy":
        return value.tolist() if hasattr(value, "ndim") and value.ndim else value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(v) for v in value)
    return value


def _number(value, name, *, integer=False, allow_none=False):
    value = _plain(value)
    if value is None and allow_none:
        return None
    kind = numbers.Integral if integer else numbers.Real
    if isinstance(value, bool) or not isinstance(value, kind) or not math.isfinite(float(value)):
        raise PanelError(f"{name} must be a finite {'integer' if integer else 'number'}")
    return int(value) if integer else float(value)


def _unit(u):
    u = _plain(u)
    if isinstance(u, numbers.Integral) and not isinstance(u, bool):
        return (int(u),)
    out = tuple(int(x) for x in u)
    if not out:
        raise PanelError("a unit needs at least one coordinate")
    return out


def units_of(values):
    """Sorted, de-duplicated tuple of integer unit tuples."""
    return tuple(sorted({_unit(u) for u in values}))


def _strict_units(values, what):
    """Like :func:`units_of` but refuses an empty list and duplicate units."""
    raw = [_unit(u) for u in values]
    if not raw:
        raise PanelError(f"{what} is empty")
    if len(set(raw)) != len(raw):
        dup = sorted({u for u in raw if raw.count(u) > 1})
        raise PanelError(f"{what} lists units more than once: {dup[:3]}")
    return tuple(sorted(raw))


def _rows(spec):
    if spec is None:
        return ()
    if isinstance(spec, str):
        return (spec,)
    return tuple(spec)


def _fill(template, item):
    return tuple(t.format(item=item) for t in template)


def _row_table_digest(rows):
    """sha256 of a row table as the port reads it (per name: sorted distinct integer rows)."""
    if not isinstance(rows, dict):
        raise PanelError("a row table is a {name: rows} mapping")
    return sha256_json({str(k): sorted({int(r) for r in v}) for k, v in rows.items()})


def _waiver_text(waiver, name):
    if waiver is None:
        return None
    if not isinstance(waiver, str) or not waiver.strip():
        raise PanelError(f"{name} needs a written reason (a nonempty string)")
    return waiver


# ---------------------------------------------------------------------------------- the record

@dataclass(frozen=True)
class Arm:
    """One branch of a panel (JSON-serialisable; no tensors, no callables).

    ``units``: the (block, head) pairs (read) or (layer, kv group) cells (value) the arm acts on;
    ``None`` = every unit (an all-cut). ``unit_set`` names the set (``"keep-own60"`` names the
    complement of ``own60``). ``item``: the item whose result this arm serves (word, source);
    ``rows_item``: the item whose rows are cut (differs only for an other-item control).
    ``receivers``/``sources``: row names for a read cut (resolved by the port's row table).
    ``condition``: input variant the forward runs under (``"iso:red"``, ``"delete:red"``, ``None``
    = the unmodified input). ``reference``: arm id the outcome is measured against.
    ``control_for``: test arm ids this arm is a control for. ``floor_of``: the cut whose
    arithmetic floor this arm measures. ``twin``: same arm in the other cut type.
    ``separates_identity``: for a control, whether it holds mass fixed while varying the units
    (``None`` = unknown). ``claim``: ``"necessary"`` (cut) or ``"sufficient"`` (keep).
    ``rules_out``: the alternative explanations this control rules out when the test beats it
    beyond the floor (``set_size``, ``mass``, ``generic_damage``). ``mass_relation``: a random
    control's mass against the target's (``matched``, ``matched_by_reuse``, ``less``, ``more``,
    ``unknown``).
    """
    arm: str
    family: str
    kind: str
    role: str
    item: str | None = None
    units: tuple | None = None
    unit_set: str | None = None
    rows_item: str | None = None
    receivers: tuple = ()
    sources: tuple = ()
    cut_type: str | None = None
    zero_method: str | None = None
    condition: str | None = None
    reference: str | None = None
    control_for: tuple = ()
    floor_of: str | None = None
    twin: str | None = None
    claim: str | None = None
    control: str | None = None
    separates_identity: bool | None = None
    bundle: str | None = None
    # value family
    dose: float | None = None
    target_row: str | None = None
    payload_row: str | None = None
    notes: tuple = ()
    # controls: what a beaten control rules out, and a random control's mass relation
    rules_out: tuple = ()
    mass_relation: str | None = None

    def __post_init__(self):
        if self.family not in ("read", "value"):
            raise PanelError("family must be 'read' or 'value'")
        kinds = READ_KINDS if self.family == "read" else VALUE_KINDS
        if self.kind not in kinds:
            raise PanelError(f"unknown {self.family} kind: {self.kind}")
        if self.role not in ROLES:
            raise PanelError(f"unknown role: {self.role}")
        if self.cut_type is not None and self.cut_type not in CUT_TYPES:
            raise PanelError("cut_type must be 'zero' or 'renorm'")
        if self.zero_method not in (None, "delta", "recompute"):
            raise PanelError("zero_method must be None, 'delta' or 'recompute'")
        if self.dose is not None and (isinstance(self.dose, bool) or not math.isfinite(float(self.dose))):
            raise PanelError("dose must be a finite number")
        if any(r not in ALTERNATIVES["necessary"] for r in self.rules_out):
            raise PanelError(f"rules_out entries must be in {ALTERNATIVES['necessary']}")
        if self.mass_relation not in (None, *RULES_OUT_BY_MASS):
            raise PanelError(f"mass_relation must be None or one of {tuple(RULES_OUT_BY_MASS)}")

    @property
    def renormalize(self):
        return None if self.cut_type is None else self.cut_type == "renorm"

    def to_dict(self):
        out = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name in _OMIT_IF_DEFAULT and value == _OMIT_IF_DEFAULT[name]:
                continue          # keeps hashes of plans written before these fields existed
            if name == "units" and value is not None:
                value = [list(u) for u in value]
            elif isinstance(value, tuple):
                value = list(value)
            out[name] = value
        return out

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if d.get("units") is not None:
            d["units"] = tuple(tuple(u) for u in d["units"])
        for name in ("receivers", "sources", "control_for", "notes", "rules_out"):
            d[name] = tuple(d.get(name) or ())
        return cls(**d)


# ----------------------------------------------------------------------------------- the panel

@dataclass
class Panel:
    """A finite, seal-able list of arms run in every context (one (prompt, seed) per context).

    ``execution`` (hashed with the plan): the settings the arms must be executed under:
    port precision, zero method per cut type, empty-row policy, row-table digest, universe
    geometry, forwards per image. See :func:`port_kwargs`, :func:`check_port`, :meth:`check_universe`.
    """
    family: str
    arms: list
    contexts: list
    sets: dict = field(default_factory=dict)
    controls: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)
    waivers: dict = field(default_factory=dict)
    port: dict = field(default_factory=dict)
    aliases: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    execution: dict = field(default_factory=dict)

    def __post_init__(self):
        ids = [a.arm for a in self.arms]
        if len(ids) != len(set(ids)):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise PanelError(f"duplicate arm ids: {dup[:5]}")
        if not self.contexts or len(set(map(str, self.contexts))) != len(self.contexts):
            raise PanelError("a panel needs distinct contexts")
        self.contexts = [str(c) for c in self.contexts]

    # identity ------------------------------------------------------------------------------
    def by_id(self):
        return {a.arm: a for a in self.arms}

    def to_json(self):
        out = {"schema": SCHEMA, "family": self.family, "contexts": list(self.contexts),
               "arms": [a.to_dict() for a in self.arms], "sets": self.sets, "controls": self.controls,
               "identity": self.identity, "waivers": self.waivers, "port": self.port,
               "aliases": self.aliases, "notes": list(self.notes), "branches": self.branch_count}
        if self.execution:          # absent from plans written before the execution block existed
            out["execution"] = self.execution
        return out

    @classmethod
    def from_json(cls, d):
        if d.get("schema") != SCHEMA:
            raise PanelError("not an attention panel plan")
        return cls(family=d["family"], arms=[Arm.from_dict(a) for a in d["arms"]], contexts=d["contexts"],
                   sets=d.get("sets", {}), controls=d.get("controls", {}), identity=d.get("identity", {}),
                   waivers=d.get("waivers", {}), port=d.get("port", {}), aliases=d.get("aliases", {}),
                   notes=d.get("notes", []), execution=d.get("execution", {}))

    def canonical(self):
        return canonical_json(self.to_json())

    def sha256(self):
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def seal(self, predictions=None):
        """Seal the plan (and optional predictions keyed by arm id or unit set) before execution."""
        plan_sha = self.sha256()
        record = {"schema": SEAL_SCHEMA, "plan_sha256": plan_sha}
        if predictions is not None:
            predictions = json.loads(canonical_json(_plain(predictions)))  # finite JSON only, a copy
            record["predictions"] = predictions
            record["predictions_sha256"] = sha256_json(predictions)
        record["seal_sha256"] = sha256_json(record)
        return record

    def verify_seal(self, seal):
        """Recompute the plan, predictions and seal hashes; raise on any mismatch. See :func:`verify_seal`."""
        return verify_seal(self, seal)

    def check_universe(self, observed_blocks, observed_heads):
        """Worker check against the live model; MUST be called before running keep arms.

        A keep arm cuts the complement of its set over the plan's universe, so the universe
        must be every (block, head) of the model. Raises if the live geometry differs from the
        geometry recorded in the plan, or if the plan's universe is not the full grid (unless
        the plan was built with ``waive['partial_universe']``).
        """
        geo = self.execution.get("universe")
        if not geo:
            raise PanelError("the plan records no universe geometry (not a read panel built by read_panel)")
        b = _number(observed_blocks, "observed_blocks", integer=True)
        h = _number(observed_heads, "observed_heads", integer=True)
        if (b, h) != (geo["blocks"], geo["heads"]):
            raise PanelError(f"live model has {b} blocks x {h} heads; the plan was built for "
                             f"{geo['blocks']} x {geo['heads']}: keep arms would cut the wrong complement")
        if not geo["full_grid"] and "partial_universe" not in self.waivers:
            raise PanelError(f"the plan's universe ({geo['n_units']} units) is not every (block, head) of a "
                             f"{b} x {h} model; keep complements would leave units uncut")
        return True

    # budget --------------------------------------------------------------------------------
    @property
    def branch_count(self):
        return len(self.arms) * len(self.contexts)

    def check_budget(self, cap):
        """Refuse a panel whose total branch count (arms x contexts) exceeds ``cap``."""
        if isinstance(cap, bool) or not isinstance(cap, numbers.Integral) or cap < 1:
            raise PanelError("cap must be a positive integer")
        if self.branch_count > cap:
            raise PanelError(f"panel has {self.branch_count} branches ({len(self.arms)} arms x "
                             f"{len(self.contexts)} contexts), over the cap of {cap}; split it with leases()")
        return self.branch_count

    def leases(self, cap=FLUX_LANE_CAP, *, max_total=None, seal=None):
        """Deterministic split into leases of at most ``cap`` branches.

        Atoms are (context, bundle): a test set's arms, its floors, twins and controls stay in one
        lease for one context. Atoms are packed in plan order (context-major); an atom larger
        than ``cap`` is refused. ``max_total`` refuses the whole panel first. Each lease carries
        the plan hash, the seal hash (``None`` when ``seal`` is not given; a given seal is
        verified first) and ``depends_on``: (context, arm) references run in another lease, so
        outcomes are scored only after those leases have run.
        """
        if max_total is not None:
            self.check_budget(max_total)
        if isinstance(cap, bool) or not isinstance(cap, numbers.Integral) or cap < 1:
            raise PanelError("cap must be a positive integer")
        seal_sha = None
        if seal is not None:
            verify_seal(self, seal)
            seal_sha = seal["seal_sha256"]
        bundles = []
        for a in self.arms:
            key = a.bundle or a.arm
            if not bundles or bundles[-1][0] != key:
                if any(k == key for k, _ in bundles):
                    raise PanelError(f"bundle {key!r} is not contiguous in the plan")
                bundles.append((key, []))
            bundles[-1][1].append(a.arm)
        plan_sha = self.sha256()
        leases, current = [], []
        for context in self.contexts:
            for key, arms in bundles:
                if len(arms) > cap:
                    raise PanelError(f"bundle {key!r} has {len(arms)} arms, over the lease cap {cap}")
                if len(current) + len(arms) > cap:
                    leases.append(current)
                    current = []
                current += [[context, arm] for arm in arms]
        if current:
            leases.append(current)
        ids = self.by_id()
        out = []
        for i, b in enumerate(leases):
            inside = {(c, a) for c, a in b}
            deps = sorted({(c, ids[a].reference) for c, a in b
                           if ids[a].reference in ids and (c, ids[a].reference) not in inside})
            body = {"plan_sha256": plan_sha, "lease": i, "branches": b}
            if seal_sha is not None:
                body["seal_sha256"] = seal_sha
            out.append({"plan_sha256": plan_sha, "seal_sha256": seal_sha, "lease": i, "of": len(leases),
                        "branches": b, "depends_on": [list(d) for d in deps], "lease_sha256": sha256_json(body)})
        return out


def verify_seal(panel, seal):
    """Recompute the plan hash, the predictions hash and the seal hash of ``seal``; raise on mismatch.

    A seal from :meth:`Panel.seal` is fully recomputed. A seal from
    :func:`seal_with_forecast_transaction` (it has ``path`` and no ``schema``) is checked for the
    plan hash only; its predictions are verified by ``ForecastTransaction.verify()``.
    """
    if not isinstance(seal, dict):
        raise PanelError("a seal is a dict")
    if seal.get("plan_sha256") != panel.sha256():
        raise PanelError("the seal was made for a different plan (plan sha256 recomputed and compared)")
    if seal.get("schema") is None and "path" in seal:
        return seal
    if seal.get("schema") != SEAL_SCHEMA:
        raise PanelError("not an attention panel seal")
    if ("predictions" in seal) != ("predictions_sha256" in seal):
        raise PanelError("seal predictions and their hash must come together")
    if "predictions" in seal and sha256_json(seal["predictions"]) != seal["predictions_sha256"]:
        raise PanelError("sealed predictions were altered after sealing (predictions sha256 recomputed)")
    body = {k: v for k, v in seal.items() if k != "seal_sha256"}
    if sha256_json(body) != seal.get("seal_sha256"):
        raise PanelError("seal record was altered after sealing (seal sha256 recomputed)")
    return seal


# --------------------------------------------------------------------------- read panel generator

@dataclass(frozen=True)
class RandomControls:
    """Matched random unit sets via :func:`mdb.attention_ranking.matched_control_sets`.

    ``strata``: one label per universe unit (e.g. joint / single). ``match_blocks``: match the
    number of distinct blocks (block = unit[0]). ``mass``: per-unit mass (e.g. the item's
    read-mass share), required to say anything about identity vs mass; ``mass_tolerance``
    requests mass matching. ``disjoint``: draw outside the target. ``overlap_sd``: a
    mass-matched, non-disjoint draw whose overlap with the target exceeds chance mean +
    ``overlap_sd`` sd is recorded as ``matched_by_reuse`` (it does not rule out mass).
    ``n`` must be >= 1 (no random control = ``random=None`` plus a written waiver).
    numpy scalars/arrays are converted to Python numbers at construction.
    """
    n: int = 1
    strata: tuple | None = None
    match_blocks: bool = True
    mass: tuple | None = None
    mass_tolerance: float | None = None
    disjoint: bool = False
    seed: int = 0
    overlap_sd: float = 3.0

    def __post_init__(self):
        n = _number(self.n, "RandomControls.n", integer=True)
        if n < 1:
            raise PanelError("RandomControls(n=0) emits no random control; pass random=None with "
                             "waive['random_cut'] / waive['random_keep'] = written reason instead")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "seed", _number(self.seed, "RandomControls.seed", integer=True))
        tol = _number(self.mass_tolerance, "RandomControls.mass_tolerance", allow_none=True)
        if tol is not None and tol < 0:
            raise PanelError("RandomControls.mass_tolerance must be >= 0")
        object.__setattr__(self, "mass_tolerance", tol)
        if self.mass is not None:
            object.__setattr__(self, "mass", tuple(_number(m, "RandomControls.mass") for m in _plain(self.mass)))
        if self.strata is not None:
            object.__setattr__(self, "strata", tuple(_plain(self.strata)))
        sd = _number(self.overlap_sd, "RandomControls.overlap_sd")
        if sd <= 0:
            raise PanelError("RandomControls.overlap_sd must be > 0")
        object.__setattr__(self, "overlap_sd", sd)
        object.__setattr__(self, "match_blocks", bool(self.match_blocks))
        object.__setattr__(self, "disjoint", bool(self.disjoint))


def _flat(universe):
    index = {u: i for i, u in enumerate(universe)}
    if len(index) != len(universe):
        raise PanelError("universe units must be distinct")
    return index


def _mass_bound(universe, target_idx, strata, mass, disjoint):
    """Upper bound on the mass a stratum-count-matched set can reach (ignores the block constraint)."""
    import numpy as np
    strata = np.zeros(len(universe), np.int64) if strata is None else np.asarray(strata)
    mass = np.asarray(mass, np.float64)
    allowed = np.ones(len(universe), bool)
    if disjoint:
        allowed[target_idx] = False
    total = 0.0
    for s in np.unique(strata[target_idx]):
        need = int((strata[target_idx] == s).sum())
        pool = np.sort(mass[allowed & (strata == s)])[::-1]
        total += float(pool[:need].sum())
    return total


def _chance_overlap(target_idx, strata, n_units):
    """Mean and sd of the overlap of a uniform stratum-count-matched draw with the target (hypergeometric)."""
    labels = [0] * n_units if strata is None else list(strata)
    mean = var = 0.0
    for s in set(labels[i] for i in target_idx):
        c = sum(1 for i in target_idx if labels[i] == s)
        n = sum(1 for x in labels if x == s)
        mean += c * c / n
        if n > 1:
            var += c * (c / n) * ((n - c) / n) * ((n - c) / (n - 1))
    return mean, math.sqrt(var)


def _mass_relation(report, reuse_above):
    m = report.get("mass", {})
    if "target" not in m:
        return "unknown"
    if m.get("matched"):
        if reuse_above is not None and report["overlap_with_target"] > reuse_above:
            return "matched_by_reuse"
        return "matched"
    return "less" if m["control"] <= m["target"] else "more"


def _draw_random(name, units, universe, spec, index, item):
    """(entry, [(set, mass_relation), ...] or None)."""
    from .attention_ranking import matched_control_sets
    target = [index[u] for u in units]
    blocks = [u[0] for u in universe] if spec.match_blocks else None
    entry = {"target": name, "item": item, "n": spec.n, "seed": spec.seed, "disjoint": spec.disjoint,
             "mass_tolerance": spec.mass_tolerance, "match_blocks": spec.match_blocks,
             "overlap_sd": spec.overlap_sd}
    try:
        drawn = matched_control_sets(target, len(universe), strata=spec.strata, blocks=blocks, mass=spec.mass,
                                     mass_tolerance=spec.mass_tolerance, disjoint=spec.disjoint,
                                     seed=spec.seed, n_sets=spec.n)
    except ValueError as error:
        entry.update(status="control_missing", reason=str(error), sets=[])
        return _plain(entry), None
    sets, reports = [], []
    for d in drawn:
        sets.append(tuple(universe[int(i)] for i in d["units"]))
        reports.append(_plain(d["report"]))
    entry["reports"] = reports
    entry["sets"] = [[list(u) for u in s] for s in sets]
    reuse_above = None
    if not spec.disjoint:
        mean, sd = _chance_overlap(target, spec.strata, len(universe))
        reuse_above = mean + spec.overlap_sd * sd
    relations = [_mass_relation(r, reuse_above) for r in reports]
    entry["mass_relations"] = relations
    if spec.mass is None:
        entry["identity"] = {"separable": None, "why": "no per-unit mass supplied: a random set that is not "
                             "mass-matched tests mass and identity together"}
    elif spec.mass_tolerance is None:
        entry["identity"] = {"separable": False, "why": "mass supplied but not matched (no tolerance): the "
                             "control tests mass as well as identity",
                             "target_mass": reports[0]["mass"]["target"],
                             "control_mass": [r["mass"]["control"] for r in reports]}
    else:
        ok = all(r["mass"]["matched"] for r in reports)
        info = {"separable": bool(ok), "target_mass": reports[0]["mass"]["target"],
                "control_mass": [r["mass"]["control"] for r in reports], "tolerance": spec.mass_tolerance}
        if ok and not spec.disjoint:
            mean, sd = _chance_overlap(target, spec.strata, len(universe))
            overlaps = [r["overlap_with_target"] for r in reports]
            info.update(overlap_with_target=overlaps, chance_overlap_mean=mean, chance_overlap_sd=sd,
                        overlap_sd_threshold=spec.overlap_sd)
            if max(overlaps) > mean + spec.overlap_sd * sd:
                info["separable"] = False
                info["why"] = (f"mass was reached by re-using target units (overlap {max(overlaps)} of "
                               f"{len(target)}; chance {mean:.1f} +/- {sd:.1f}, threshold mean + "
                               f"{spec.overlap_sd:g} sd): the control shares the identity it should vary; "
                               "use disjoint=True or the other-item control")
        if not ok:
            bound = _mass_bound(universe, target, spec.strata, spec.mass, spec.disjoint)
            reach = info["target_mass"] - spec.mass_tolerance
            info["max_reachable_mass_bound"] = bound
            info["why"] = ("random sets could not be mass-matched: " +
                           (f"no {'disjoint ' if spec.disjoint else ''}stratum-matched set can reach the "
                            f"target mass (bound {bound:.4g} < {reach:.4g})" if bound < reach else
                            "swap search ended outside tolerance") +
                           "; identity cannot be separated from mass by random draws, use the other-item control")
        entry["identity"] = info
    entry["status"] = "emitted"
    return _plain(entry), list(zip(sets, relations))


@dataclass(frozen=True)
class ReadPanelSpec:
    """Everything the read-panel generator needs; see :func:`read_panel`.

    ``sets[item][name]`` = units. ``cut_sets``/``keep_sets``: names that get test arms.
    ``cut_types``: default types for test arms; ``cut_type_overrides`` / ``keep_type_overrides``
    per set name. ``external_controls``: ``{control_set_name: target_set_name}`` for hand-drawn
    control sets (they inherit the target's cut types). ``nested``: set names in increasing
    size; each larger set gets ``<big>-minus-<small>`` added-units-alone test arms.
    ``other_items``: ``{item: (other, ...)}``; default = the next item (cyclic).

    Execution (hashed into the plan's ``execution`` block): ``precision`` (``"native"`` /
    ``"legacy"``; ``None`` = ``port["precision"]`` if that is a valid value, else ``"native"``),
    ``zero_method`` (the per-arm zero method; ``None`` = the port's own, recorded as
    ``port_zero_method``, default ``"delta"``), ``empty`` (the port's empty-row policy),
    ``row_table`` (bound by digest; the worker must pass the same table), ``forwards_per_image``.
    """
    items: tuple
    universe: tuple
    sets: dict
    contexts: tuple = ("default",)
    cut_sets: tuple = ()
    keep_sets: tuple = ()
    cut_types: tuple = CUT_TYPES
    cut_type_overrides: dict = field(default_factory=dict)
    keep_type_overrides: dict = field(default_factory=dict)
    external_controls: dict = field(default_factory=dict)
    random: RandomControls | None = RandomControls()
    other_items: dict | None = None
    nested: tuple = ()
    nested_pairs: str = "smallest"
    receivers: tuple = ("image", "others:{item}")
    sources: tuple = ("own:{item}",)
    condition: str | None = "iso:{item}"
    references: tuple = ("condition", "all_cut", "deletion")
    reference_types: tuple | None = None
    floors: bool = True
    zero_method: str | None = None
    waive: dict = field(default_factory=dict)
    port: dict = field(default_factory=dict)
    precision: str | None = None
    port_zero_method: str = "delta"
    empty: str = "raise"
    row_table: dict | None = None
    forwards_per_image: int | None = None


def _types(t):
    t = tuple(t)
    if not t or any(x not in CUT_TYPES for x in t) or len(set(t)) != len(t):
        raise PanelError("cut types must be a nonempty subset of ('zero', 'renorm')")
    return tuple(x for x in CUT_TYPES if x in t)


def _geometry(universe, strata):
    two = all(len(u) == 2 for u in universe)
    blocks = 1 + max(u[0] for u in universe) if two else None
    heads = 1 + max(u[1] for u in universe) if two else None
    full = bool(two and len(universe) == blocks * heads and min(min(u) for u in universe) >= 0)
    geo = {"n_units": len(universe), "blocks": blocks, "heads": heads, "full_grid": full,
           "units_sha256": sha256_json([list(u) for u in universe])}
    if strata is not None:
        counts = {}
        for s in strata:
            counts[str(s)] = counts.get(str(s), 0) + 1
        geo["strata_counts"] = counts
        geo["strata_sha256"] = sha256_json([str(s) for s in strata])
    return geo


def _execution(spec, universe):
    port_precision = spec.port.get("precision") if isinstance(spec.port, dict) else None
    if port_precision not in ("native", "legacy"):
        port_precision = None
    precision = spec.precision or port_precision or "native"
    if precision not in ("native", "legacy"):
        raise PanelError("precision must be 'native' or 'legacy'")
    if spec.precision is not None and port_precision is not None and spec.precision != port_precision:
        raise PanelError(f"precision {spec.precision!r} contradicts port['precision'] = {port_precision!r}")
    if spec.port_zero_method not in ("delta", "recompute"):
        raise PanelError("port_zero_method must be 'delta' or 'recompute'")
    if spec.empty not in ("raise", "zero"):
        raise PanelError("empty must be 'raise' or 'zero'")
    forwards = None if spec.forwards_per_image is None else _number(spec.forwards_per_image, "forwards_per_image",
                                                                      integer=True)
    if forwards is not None and forwards < 1:
        raise PanelError("forwards_per_image must be >= 1")
    rows = None
    if spec.row_table is not None:
        rows = {"sha256": _row_table_digest(spec.row_table),
                "names": sorted(str(k) for k in spec.row_table),
                "sizes": {str(k): len({int(r) for r in v}) for k, v in spec.row_table.items()}}
    strata = spec.random.strata if spec.random is not None else None
    if strata is not None and len(strata) != len(universe):
        raise PanelError("RandomControls.strata needs one label per universe unit")
    unbound = [k for k, v in (("row_table", rows), ("forwards_per_image", forwards)) if v is None]
    return {"port": "mdb.attention_port.AttentionPort", "precision": precision,
            "zero_method": {"zero": spec.zero_method or spec.port_zero_method, "renorm": None,
                            "port_default": spec.port_zero_method, "per_arm": spec.zero_method,
                            "floors": "recompute (attention_port.floor_arms)"},
            "empty": spec.empty, "row_table": rows, "forwards_per_image": forwards,
            "universe": _geometry(universe, strata), "unbound": unbound}


def read_panel(spec):
    """Emit the full read-cut panel for ``spec`` (see the module and doc for the controls)."""
    waive = dict(spec.waive)
    for key, why in waive.items():
        if key not in WAIVABLE or not isinstance(why, str) or not why.strip():
            raise PanelError(f"waiver {key!r} needs a known key {WAIVABLE} and a written reason")
    universe = tuple(_unit(u) for u in spec.universe)
    if not universe:
        raise PanelError("the universe is empty")
    index = _flat(universe)
    items = tuple(spec.items)
    if not items:
        raise PanelError("at least one item is required")
    if len(set(items)) != len(items):
        raise PanelError("items must be distinct")
    if not spec.floors and "floors" not in waive:
        raise PanelError("floors=False needs waive['floors'] = reason (every no-renormalisation cut needs "
                         "its recompute sham and recompute twin, every renormalising cut its sham)")
    references = tuple(spec.references)
    unknown_refs = [r for r in references if r not in ("condition", "native", "deletion", "sham_all", "all_cut")]
    if unknown_refs:
        raise PanelError(f"unknown references {unknown_refs}")
    if spec.condition is not None and "condition" not in references:
        raise PanelError("cut arms run under the condition (e.g. iso:<item>) but references lack 'condition': "
                         "their outcome would include the conditioning change; add 'condition' to references "
                         "or pass condition=None")
    if spec.keep_sets and "all_cut" not in references:
        raise PanelError("keep sets need the all-cut reference (the effect a keep arm recovers); add 'all_cut' "
                         "to references")
    execution = _execution(spec, universe)
    if spec.keep_sets and not execution["universe"]["full_grid"] and "partial_universe" not in waive:
        raise PanelError("keep arms cut the complement over the universe, but the universe is not a full "
                         "(block, head) grid; pass every unit or waive['partial_universe'] = reason")
    other = spec.other_items
    if other is None:
        other = {it: ((items[(i + 1) % len(items)],) if len(items) > 1 else ()) for i, it in enumerate(items)}
    else:
        other = {k: _rows(v) for k, v in other.items()}
        for it, others in other.items():
            if it not in items:
                raise PanelError(f"other_items names {it!r}, which is not an item")
            for o in others:
                if o == it:
                    raise PanelError(f"other_items maps {it!r} to itself: an other-item control needs another item")
                if o not in items:
                    raise PanelError(f"other_items maps {it!r} to {o!r}, which is not an item (no rows, no sets)")
    used = set((*spec.cut_sets, *spec.keep_sets, *spec.external_controls, *spec.external_controls.values(),
                *spec.nested))
    sets = {}
    for item in items:
        if item not in spec.sets:
            raise PanelError(f"no sets given for item {item!r}")
        named = {}
        for k, v in spec.sets[item].items():
            named[k] = _strict_units(v, f"set {k} of {item}") if k in used else units_of(v)
        for name, units in named.items():
            missing = [u for u in units if u not in index]
            if missing:
                raise PanelError(f"set {name} of {item} has units outside the universe: {missing[:3]}")
            if name in used and len(units) == len(universe):
                raise PanelError(f"set {name} of {item} is the whole universe: its cut is the all-cut reference and "
                                 "its keep arm cuts nothing")
        sets[item] = named
    for name in (*spec.cut_sets, *spec.keep_sets, *spec.external_controls, *spec.external_controls.values(),
                 *spec.nested):
        for item in items:
            if name not in sets[item]:
                raise PanelError(f"set {name!r} is not defined for item {item!r}")
    # nested-size series: added units alone
    nested_sets = []
    if spec.nested:
        if spec.nested_pairs not in ("smallest", "all", "consecutive"):
            raise PanelError("nested_pairs must be 'smallest', 'all' or 'consecutive'")
        names = list(spec.nested)
        pairs = ([(names[0], b) for b in names[1:]] if spec.nested_pairs == "smallest" else
                 [(names[i], names[i + 1]) for i in range(len(names) - 1)] if spec.nested_pairs == "consecutive"
                 else [(a, b) for i, a in enumerate(names) for b in names[i + 1:]])
        for small, big in pairs:
            name = f"{big}-minus-{small}"
            for item in items:
                if not set(sets[item][small]) < set(sets[item][big]):
                    raise PanelError(f"nested series broken for {item}: {small} is not a proper subset of {big}")
                sets[item][name] = units_of(set(sets[item][big]) - set(sets[item][small]))
            nested_sets.append(name)
    cut_sets = tuple(dict.fromkeys((*spec.cut_sets, *nested_sets)))
    default = _types(spec.cut_types)
    cut_types = {s: _types(spec.cut_type_overrides.get(s, default)) for s in cut_sets}
    keep_types = {s: _types(spec.keep_type_overrides.get(s, default)) for s in spec.keep_sets}
    if any(len(t) < 2 for t in (*cut_types.values(), *keep_types.values())) and "both_cut_types" not in waive:
        raise PanelError("every test set needs both cut types (a result counts only if both agree); pass "
                         "waive['both_cut_types'] = reason to emit fewer")
    external_for = {}
    for c, target in spec.external_controls.items():
        external_for.setdefault(target, []).append(c)
    if spec.random is None:
        lacking = [s for s in cut_sets if s not in external_for]
        if lacking and "random_cut" not in waive:
            raise PanelError(f"cut sets {lacking} have no random control; give RandomControls or "
                             "waive['random_cut'] = reason")
        if spec.keep_sets and "random_keep" not in waive:
            raise PanelError("keep sets need a random keep control (complement of a matched random set); "
                             "give RandomControls or waive['random_keep'] = reason")
    if any(not other.get(it) for it in items) and "other_item" not in waive:
        raise PanelError("every item needs an other-item control (same units, another item's rows); pass "
                         "other_items or waive['other_item'] = reason")
    ref_types = default if spec.reference_types is None else _types(spec.reference_types)

    arms, controls, identity, notes = [], {}, {}, []

    def name_of(cut_type, core, item):
        return f"{PREFIX[cut_type]}:{core}:{item}"

    def ref_id(item):
        if spec.condition is not None:          # 'condition' is in references (checked above)
            return spec.condition.format(item=item)
        return f"native:{item}" if "native" in references else "native"   # "native" = unmodified input

    def cut_arm(item, core, kind, role, units, unit_set, cut_type, bundle, rows_item=None, **extra):
        rows_item = rows_item or item
        return Arm(arm=name_of(cut_type, core, item), family="read", kind=kind, role=role, item=item,
                   units=units, unit_set=unit_set, rows_item=rows_item,
                   receivers=_fill(spec.receivers, rows_item), sources=_fill(spec.sources, rows_item),
                   cut_type=cut_type, zero_method=spec.zero_method,
                   condition=None if spec.condition is None else spec.condition.format(item=item),
                   reference=ref_id(item), bundle=bundle, **extra)

    def with_floors(arm):
        out = [arm]
        if not spec.floors:
            return out
        clear = dict(control_for=(), twin=None, control=None, separates_identity=None, claim=None,
                     rules_out=(), mass_relation=None)
        if arm.cut_type == "renorm":
            out.append(replace(arm, arm=arm.arm + "/sham", kind="floor_sham", role="floor", floor_of=arm.arm, **clear))
        else:
            out.append(replace(arm, arm=arm.arm + "/sham_recompute", kind="floor_sham_recompute", role="floor",
                               floor_of=arm.arm, **clear))
            out.append(replace(arm, arm=arm.arm + "/cut_recompute", kind="floor_cut_recompute", role="floor",
                               floor_of=arm.arm, **clear))
        return out

    def complement(units):
        inside = set(units)
        return tuple(u for u in universe if u not in inside)

    def twin_ids(item, core, types):
        return {t: name_of(t, core, item) for t in types}

    def family_arms(item, core, kind, role, units, unit_set, types, bundle, **extra):
        ids = twin_ids(item, core, types)
        out = []
        for t in types:
            twin = ids["renorm" if t == "zero" else "zero"] if len(types) == 2 else None
            ext = dict(extra)
            if "control_for" in ext:
                ext["control_for"] = tuple(x for x in ext["control_for"] if x.startswith(PREFIX[t] + ":"))
            out += with_floors(cut_arm(item, core, kind, role, units, unit_set, t, bundle, twin=twin, **ext))
        return out

    for item in items:
        named = sets[item]
        cond = None if spec.condition is None else spec.condition.format(item=item)
        refs = []
        bundle = f"ref:{item}"
        if "condition" in references and cond is not None:
            refs.append(Arm(arm=cond, family="read", kind="condition", role="reference", item=item,
                            condition=cond, reference="native", bundle=bundle))
        if "native" in references:
            refs.append(Arm(arm=f"native:{item}", family="read", kind="native", role="reference", item=item,
                            bundle=bundle))
        if "deletion" in references:
            refs.append(Arm(arm=f"del:{item}", family="read", kind="deletion", role="reference", item=item,
                            condition=f"delete:{item}", reference="native", bundle=bundle))
        if "sham_all" in references:
            refs.append(Arm(arm=f"z:sham-all:{item}", family="read", kind="sham_all", role="floor", item=item,
                            cut_type="zero", zero_method="recompute", condition=cond, reference=ref_id(item),
                            bundle=bundle, notes=("every head, block and row through the FP32 recompute path; "
                                                  "nothing removed",)))
        if "all_cut" in references:
            for t in ref_types:
                refs += with_floors(cut_arm(item, "none", "all_cut", "reference", None, "all", t, bundle,
                                            twin=(name_of("renorm" if t == "zero" else "zero", "none", item)
                                                  if len(ref_types) == 2 else None)))
        arms += refs
        others = tuple(other.get(item, ()))
        for claim, names, types_of in (("necessary", cut_sets, cut_types), ("sufficient", spec.keep_sets, keep_types)):
            for s in names:
                types = types_of[s]
                units = named[s]
                keep = claim == "sufficient"
                core = f"keep-{s}" if keep else s
                act_units = complement(units) if keep else units
                kind = "keep" if keep else "cut"
                bundle = f"{core}:{item}"
                tests = family_arms(item, core, kind, "test", act_units, core, types, bundle, claim=claim)
                arms += tests
                test_ids = tuple(a.arm for a in tests if a.role == "test")
                # external (hand-drawn) controls for cut sets: mass and structure unknown, so they
                # can refute a claim but rule nothing out
                if not keep:
                    for c in external_for.get(s, ()):
                        arms += family_arms(item, c, "cut", "control", named[c], c, types, bundle,
                                            control_for=test_ids, control="external", claim=claim,
                                            separates_identity=None, mass_relation="unknown")
                # matched random controls (cut and keep)
                if spec.random is not None:
                    key = f"{item}:{core}"
                    entry, drawn = _draw_random(s, units, universe, spec.random, index, item)
                    controls[key] = entry
                    if drawn is None:
                        identity.setdefault(f"{item}:{s}", {})["random_" + kind] = {
                            "status": "control_missing", "why": entry["reason"]}
                    else:
                        identity.setdefault(f"{item}:{s}", {})["random_" + kind] = entry["identity"]
                        for k, (rset, relation) in enumerate(drawn):
                            rcore = f"keep-rand{k}-{s}" if keep else f"rand{k}-{s}"
                            arms += family_arms(item, rcore, kind, "control",
                                                complement(rset) if keep else units_of(rset), rcore, types, bundle,
                                                control_for=test_ids, control="random", claim=claim,
                                                separates_identity=relation == "matched",
                                                mass_relation=relation, rules_out=RULES_OUT_BY_MASS[relation])
                # other item, same units: rules out "cutting these units damages any word"
                # (cut claims); it does not hold mass fixed. For keep claims it is a reported
                # reference only (how much the other word matters), never support.
                for o in others:
                    ocore = f"{core}@rows-{o}"
                    arms += family_arms(item, ocore, kind, "control", act_units, core, types, bundle,
                                        rows_item=o, control_for=test_ids, control="other_item", claim=claim,
                                        separates_identity=False,
                                        rules_out=() if keep else ("generic_damage",))
                    identity.setdefault(f"{item}:{s}", {})[f"other_item_{kind}"] = {
                        "separable": False, "other": o,
                        "rules_out": [] if keep else ["generic_damage"],
                        "why": ("reported reference only: how much the other item's rows matter when only these "
                                "units are kept; never counted as support for sufficiency" if keep else
                                "same units on another item's rows: rules out 'cutting these units damages any "
                                "word'; it does not hold mass fixed (the other item's mass through these units "
                                "differs), so it does not separate identity from mass")}
    if any(a.reference == "native" for a in arms):
        arms.insert(0, Arm(arm="native", family="read", kind="native", role="reference", bundle="native",
                           notes=("the unmodified input: the reference iso/del (and condition-free) arms are "
                                  "scored against",)))
    if spec.random is None:
        notes.append("random controls: none generated (" + "; ".join(
            f"{k}: {waive[k]}" for k in ("random_cut", "random_keep") if k in waive) + ")")
    port = {"zero_method": spec.zero_method or "port default", **spec.port}
    return Panel(family="read", arms=arms, contexts=list(spec.contexts),
                 sets={item: {k: [list(u) for u in v] for k, v in sets[item].items()} for item in items},
                 controls=controls, identity=identity, waivers=waive, port=port, notes=notes,
                 execution=execution)


def sets_from_ranking(ranking, sizes, *, prefix="own"):
    """Top-``n`` unit sets from an :func:`mdb.attention_ranking.contrast_ranking` result."""
    import numpy as np
    shape = tuple(ranking["unit_shape"])
    order = list(ranking["order"])
    out = {}
    for n in sizes:
        out[f"{prefix}{n}"] = units_of(tuple(int(x) for x in np.unravel_index(int(i), shape)) for i in order[:n])
    return out


def item_rows(word_rows, text_rows, total_rows):
    """Row table for the default read templates: ``image``, ``own:<item>``, ``others:<item>``."""
    table = {"image": list(range(int(text_rows), int(total_rows)))}
    for item, rows in word_rows.items():
        own = sorted({int(r) for r in rows})
        table[f"own:{item}"] = own
        table[f"others:{item}"] = [j for j in range(int(text_rows)) if j not in set(own)]
    return table


# --------------------------------------------------------------------------- arm -> port calls

def port_kwargs(panel, rows=None):
    """The ``AttentionPort(backend, **kwargs)`` arguments the plan was hashed with.

    ``rows``: the worker's row table; required (and checked against the plan's digest) when the
    plan bound one. Returns ``{"rows", "zero_method", "precision", "empty"}``.
    """
    ex = panel.execution
    if not ex or "precision" not in ex:
        raise PanelError("the plan has no execution block (not built by read_panel)")
    bound = ex.get("row_table")
    if bound is not None:
        if rows is None:
            raise PanelError("the plan binds a row table (by digest); pass the same table as rows=")
        if _row_table_digest(rows) != bound["sha256"]:
            raise PanelError("row table differs from the one the plan was hashed with")
    return {"rows": None if rows is None else dict(rows), "zero_method": ex["zero_method"]["port_default"],
            "precision": ex["precision"], "empty": ex["empty"]}


def check_port(panel, port, *, forwards=None):
    """Raise unless a live ``AttentionPort`` runs with the plan's precision, zero method, empty
    policy and (when bound) row table and forwards per image."""
    ex = panel.execution
    if not ex or "precision" not in ex:
        raise PanelError("the plan has no execution block (not built by read_panel)")
    live = {"precision": getattr(port, "precision", None), "zero_method": getattr(port, "zero_method", None),
            "empty": getattr(port, "empty", None)}
    want = {"precision": ex["precision"], "zero_method": ex["zero_method"]["port_default"], "empty": ex["empty"]}
    bad = {k: (live[k], want[k]) for k in want if live[k] != want[k]}
    if bad:
        raise PanelError("port settings differ from the plan: " +
                         ", ".join(f"{k} {v[0]!r} (plan {v[1]!r})" for k, v in bad.items()))
    if ex.get("row_table") is not None and _row_table_digest(getattr(port, "rows", {}) or {}) != ex["row_table"]["sha256"]:
        raise PanelError("the port's row table differs from the one the plan was hashed with")
    if ex.get("forwards_per_image") is not None and forwards is not None and int(forwards) != ex["forwards_per_image"]:
        raise PanelError(f"forwards per image {forwards} differs from the plan's {ex['forwards_per_image']}")
    return True


def port_cuts(arm, *, zero_method=None, panel=None, port=None):
    """Map a read arm to ``attention_port`` ``Cut`` specs (``()`` for an arm that cuts nothing).

    Floor arms come from :func:`mdb.attention_port.floor_arms` of the base cut, so the
    panel and the port cannot disagree about what a floor is. ``arm.condition`` (the input
    variant) is the caller's responsibility; the port does not see it. With ``panel``, the arm
    must be that panel's arm and ``zero_method`` may not contradict the plan; with ``panel`` and
    ``port``, :func:`check_port` runs first, so a plan cannot execute under other settings.
    """
    from . import attention_port as ap
    if arm.family != "read":
        raise PanelError("port_cuts maps read arms; use value_spec for value arms")
    if panel is not None:
        if panel.by_id().get(arm.arm) != arm:
            raise PanelError(f"arm {arm.arm!r} is not this panel's arm")
        ex = panel.execution.get("zero_method", {})
        if zero_method is not None and arm.zero_method is None and zero_method != ex.get("port_default"):
            raise PanelError(f"zero_method {zero_method!r} contradicts the plan's {ex.get('port_default')!r}")
        if port is not None:
            check_port(panel, port)
    elif port is not None:
        raise PanelError("port= needs panel= to check against")
    if arm.kind in ("condition", "native", "deletion"):
        return ()
    if arm.kind == "sham_all":
        return (ap.sham(renormalize=False, zero_method="recompute"),)
    heads = None if arm.units is None else [tuple(u) for u in arm.units]
    if heads is not None and any(len(u) != 2 for u in heads):
        raise PanelError("read units must be (block, head) pairs")
    zm = arm.zero_method if arm.zero_method is not None else zero_method
    base = ap.cut(list(arm.receivers), list(arm.sources), heads=heads, renormalize=arm.cut_type == "renorm",
                  zero_method=None if arm.cut_type == "renorm" else zm)
    if arm.kind in ("cut", "keep", "all_cut"):
        return (base,)
    floors = ap.floor_arms(base)
    name = {"floor_sham": "sham", "floor_sham_recompute": "sham_recompute",
            "floor_cut_recompute": "cut_recompute"}[arm.kind]
    return (floors[name],)


# --------------------------------------------------------------------------- value panel generator

LEGACY_VALUE_KIND = {"singleton": "transfer", "complement": "transfer", "coalition": "transfer",
                     "dose": "transfer", "wrong_target": "wrong-target", "wrong_payload": "wrong-payload",
                     "reciprocal": "reciprocal", "sham": "sham", "replay": "native", "native": "native",
                     "donor_native": "donor-native", "fixed_distance": "fixed-distance"}


def _cell_label(cell):
    return "L" + "-KV".join(str(c) for c in cell) if len(cell) == 2 else "-".join(map(str, cell))


def value_panel(cells, *, label="all", singletons=True, complements=True, groups=None, doses=(0.5,),
                controls=VALUE_CONTROLS, references=("native", "replay", "donor_native"), comparators=None,
                fixed_distance=False, reference_dose=0, contexts=("default",), item=None, cell_label=_cell_label,
                waive=None, allow_extrapolation=False):
    """Value-replacement (Codex) panel over ``cells`` ((layer, kv group) tuples).

    Arms whose executable content (legacy kind, cell set, dose) coincides are emitted once; the
    dropped names are recorded in ``panel.aliases`` (with two cells, every complement equals a
    singleton; :func:`resolve_aliases` maps results named by an alias). ``replay`` is exempt: it
    repeats ``native`` on purpose (determinism check). ``reference_dose`` is the inert dose
    written on native arms (workers skip replacement for them); the 2026-11-12 plans used 0
    (value_coalition) and 1 (frozen pair, resolver).

    Mandatory unless ``waive={"sham" | "native_reference": "written reason"}``: the ``sham``
    control and the ``native`` reference. Doses must lie in [0, 1] unless
    ``allow_extrapolation=True``. Bundles: the references form one lease atom, and the coalition
    test, its doses and every control of it (sham, wrong payload/target, reciprocal, fixed
    distance) form another, so a sham never lands in a different lease from its coalition.
    """
    waive = dict(waive or {})
    for key, why in waive.items():
        if key not in VALUE_WAIVABLE or not isinstance(why, str) or not why.strip():
            raise PanelError(f"value waiver {key!r} needs a known key {VALUE_WAIVABLE} and a written reason")
    controls, references = tuple(controls), tuple(references)
    if any(c not in VALUE_CONTROLS for c in controls):
        raise PanelError(f"unknown value controls {[c for c in controls if c not in VALUE_CONTROLS]}")
    if any(r not in VALUE_REFERENCES for r in references):
        raise PanelError(f"unknown value references {[r for r in references if r not in VALUE_REFERENCES]}")
    if "sham" not in controls and "sham" not in waive:
        raise PanelError("a value panel needs the sham control (payload from the recipient itself); pass "
                         "waive['sham'] = reason to omit it")
    if "native" not in references and "native_reference" not in waive:
        raise PanelError("a value panel needs the native reference (every test is scored against it); pass "
                         "waive['native_reference'] = reason to omit it")

    def check_dose(d, what):
        d = _number(d, what)
        if not allow_extrapolation and not 0 <= d <= 1:
            raise PanelError(f"{what} {d:g} is outside [0, 1]; pass allow_extrapolation=True to extrapolate")
        return d

    doses = tuple(check_dose(d, "dose") for d in doses)
    reference_dose = check_dose(reference_dose, "reference_dose")
    cells = units_of(cells)
    if not cells:
        raise PanelError("a value panel needs cells")
    full = cells
    rows = []
    ref_bundle, test_bundle = f"ref:{label}", f"coalition:{label}"

    def add(kind, role, name, units, dose, **extra):
        rows.append(Arm(arm=name, family="value", kind=kind, role=role, item=item, units=units,
                        unit_set=extra.pop("unit_set", None), dose=float(dose), **extra))

    ref_kinds = [k for k in ("native", "replay", "donor_native") if k in references]
    for k in ref_kinds:
        add(k, "reference", {"native": "native-recipient", "replay": "replay-recipient",
                             "donor_native": "native-donor"}[k], full, reference_dose, unit_set=label,
            bundle=ref_bundle)
    test_all = f"coalition-{label}"
    if "sham" in controls:
        add("sham", "control", f"sham-{label}", full, 1, unit_set=label, control_for=(test_all,),
            reference="native-recipient", bundle=test_bundle)
    add("coalition", "test", test_all, full, 1, unit_set=label, reference="native-recipient", bundle=test_bundle)
    for d in doses:
        add("dose", "test", f"dose{d:g}-{label}", full, d, unit_set=label, reference="native-recipient",
            bundle=test_bundle)
    for k in ("wrong_payload", "wrong_target"):
        if k in controls:
            add(k, "control", f"{k.replace('_', '-')}-{label}", full, 1, unit_set=label,
                control_for=(test_all,), reference="native-recipient", bundle=test_bundle)
    if "reciprocal" in controls:
        add("reciprocal", "control", f"reciprocal-{label}", full, 1, unit_set=label, control_for=(test_all,),
            reference="native-donor", bundle=test_bundle)
    if fixed_distance:
        add("fixed_distance", "control", f"fixed-distance-{label}", full, 1, unit_set=label,
            control_for=(test_all,), reference="native-recipient", bundle=test_bundle)
    for gname, gcells in (groups or {}).items():
        g = units_of(gcells)
        if not set(g) <= set(full):
            raise PanelError(f"group {gname} has cells outside the panel")
        add("coalition", "test", f"coalition-{gname}", g, 1, unit_set=gname, reference="native-recipient")
    for c in full if singletons else ():
        add("singleton", "test", f"only-{cell_label(c)}", (c,), 1, unit_set=cell_label(c),
            reference="native-recipient")
    for c in full if complements else ():
        add("complement", "test", f"without-{cell_label(c)}", tuple(x for x in full if x != c), 1,
            unit_set=f"without-{cell_label(c)}", reference="native-recipient")
    for cname, ccells in (comparators or {}).items():
        add("coalition", "reference", f"comparator-{cname}", units_of(ccells), 1, unit_set=cname,
            reference="native-recipient")
    seen, arms, aliases = {}, [], {}
    for a in rows:
        key = (LEGACY_VALUE_KIND[a.kind], a.units, a.dose if a.kind not in ("native", "donor_native") else None)
        if a.kind != "replay" and key in seen:
            aliases[a.arm] = seen[key]
            continue
        if a.kind != "replay":
            seen[key] = a.arm
        arms.append(a)
    execution = {"family": "value", "reference_dose": reference_dose, "allow_extrapolation": bool(allow_extrapolation),
                 "blend": "mdb.attention_panels.blend (dose 1: exact payload copy; other doses: FP32 lerp "
                          "rounded once to the recipient dtype)"}
    return Panel(family="value", arms=arms, contexts=list(contexts), aliases=aliases,
                 sets={"cells": [list(c) for c in full]}, waivers=waive, execution=execution)


def value_spec(arm):
    """Arm -> replacement spec. ``None`` for arms that replace nothing (native, replay, donor native).

    Rows are symbolic: ``"source"`` (the declared/nominated source row), ``"wrong"`` (source - 1
    in the 2026-11-12 workers) and ``"fixed-distance"``; the caller resolves them. This is the
    single copy of the mapping hand-written in the value-coalition, frozen-value-pair and
    resolver workers. Details those workers hide (read from their source, 2026-11-12):

    * the resolver worker (``resolver_execution_worker.py`` 79-89) resolves ``target_row`` from
      the nomination on the RUN side (``run_side(arm)``) and ``payload_row`` from the nomination
      on the PAYLOAD side (``payload_side``); the two can differ for ``sham`` and ``reciprocal``;
    * ``"fixed-distance"`` is ``len(tokens) - 4`` of the respective side's prompt (target: run
      side, payload: payload side), not a fixed offset from the source;
    * the frozen-pair and resolver workers call ``value_coalition.replacement`` without a dose,
      so they run every arm at FULL dose (1) whatever ``dose`` says; only the value-coalition
      worker honours ``dose`` (and its ``replacement`` accepts only 0.5 and 1);
    * :func:`blend` returns ``payload.clone()`` at dose 1 (the payload's dtype), ``before.clone()``
      at dose 0, and otherwise an FP32 lerp cast to ``before``'s dtype.
    """
    if arm.family != "value":
        raise PanelError("value_spec maps value arms")
    side = "donor" if arm.kind in ("donor_native", "reciprocal") else "recipient"
    if arm.kind in ("native", "replay", "donor_native"):
        return None
    payload_side = "recipient" if arm.kind in ("sham", "reciprocal") else "donor"
    target = "wrong" if arm.kind == "wrong_target" else "fixed-distance" if arm.kind == "fixed_distance" else "source"
    payload = ("wrong" if arm.kind == "wrong_payload" else "fixed-distance" if arm.kind == "fixed_distance"
               else "source")
    return {"side": side, "payload_side": payload_side, "target_row": target, "payload_row": payload,
            "cells": [list(c) for c in arm.units], "dose": arm.dose}


def run_side(arm):
    """Which prompt the arm's forward runs on (also defined for arms that replace nothing)."""
    return "donor" if arm.kind in ("donor_native", "reciprocal") else "recipient"


def blend(before, payload, dose):
    """THE dose blend (duck-typed tensors; no framework import).

    dose 1: an exact copy of ``payload`` (no arithmetic; the result has ``payload``'s dtype).
    dose 0: a copy of ``before``. Other doses: FP32 linear interpolation
    ``before + dose * (payload - before)`` rounded once to ``before``'s dtype. So if the two
    dtypes differ, dose 1 and dose 0.999 return different dtypes. Bit-identical to
    ``value_coalition.replacement`` (tested), where both sides share a dtype.
    ``forward_interventions._blend`` differs for non-FP32 parents: it interpolates in the
    parent's own dtype; it is left unchanged (hash-bound callers).
    """
    if isinstance(dose, bool) or not isinstance(dose, numbers.Real) or not math.isfinite(float(dose)):
        raise PanelError("dose must be a finite number")
    dose = float(dose)
    if dose == 1:
        return payload.clone()
    if dose == 0:
        return before.clone()
    b = before.float()
    return (b + dose * (payload.float() - b)).to(before.dtype)


def to_legacy(panel):
    """Hand-written ``{arm, kind, cells, dose}`` rows (value panels)."""
    if panel.family != "value":
        raise PanelError("to_legacy writes value panels")
    return [{"arm": a.arm, "kind": LEGACY_VALUE_KIND[a.kind], "cells": [tuple(c) for c in a.units],
             "dose": a.dose} for a in panel.arms]


# --------------------------------------------------------------------------- sealing adapters

def seal_with_forecast_transaction(panel, root, predictions, *, model_identity, input_identity,
                                   execution_scope="panel executed elsewhere (mrun lane); outcomes attached "
                                                   "by plan sha256"):
    """Seal plan + predictions through :class:`mdb.forecast_transaction.ForecastTransaction`.

    Uses the capture-optional declaration (``frontier={'kind': 'empty'}``, no bound model, no
    weights): the plan's canonical JSON is published as a frontier and the predictions as one
    forecast, then globally sealed. The transaction's model guards (CPU FP32 bound model) do not
    apply to a FLUX panel executed on a remote lane, so execution is not routed through
    ``execute``; outcomes are attached with :func:`attach_results`, which checks the plan hash.
    """
    from .forecast_transaction import ForecastTransaction
    plan_sha = panel.sha256()
    ids = set(panel.by_id())
    unknown = [k for k in predictions if k not in ids and not any(k == f"{a.item}:{a.unit_set}" for a in panel.arms)]
    if unknown:
        raise PanelError(f"predictions name unknown arms/sets: {unknown[:5]}")
    tx = ForecastTransaction(root, model_identity=model_identity, input_identity=input_identity,
                             source_identity={"attention_panel_sha256": plan_sha}, frontier={"kind": "empty"},
                             execution_scope=execution_scope)
    tx.frontier("panel_plan", {"sha256": plan_sha, "canonical_json": panel.canonical()})
    tx.predict("panel_predictions", lambda context: {"predictions": dict(predictions)})
    seal = tx.seal()
    return tx, {"plan_sha256": plan_sha, "seal_sha256": seal["seal_sha256"], "path": seal["path"]}


# --------------------------------------------------------------------------- results + analysis

def results_record(panel, outcomes, *, pair_distances=None, seal=None, metric=None):
    """Build a results record bound to the plan hash.

    ``outcomes[context][arm_id]`` = scalar distance (>= 0) of that arm's output from its
    ``reference`` arm (e.g. pixel MAD vs ``iso:<item>``). ``pair_distances[context]["a|b"]`` =
    direct distance between two arms; every ``<cut>|<cut>/cut_recompute`` pair listed by
    :func:`twin_pairs` is needed for a complete floor.
    """
    rec = {"schema": RESULTS_SCHEMA, "plan_sha256": panel.sha256(), "metric": metric,
           "outcomes": outcomes, "pair_distances": pair_distances or {}}
    if seal is not None:
        rec["seal_sha256"] = seal["seal_sha256"]
    return rec


def twin_pairs(panel):
    """The ``"<cut>|<cut>/cut_recompute"`` pair-distance keys a worker must measure in every context
    (the delta-vs-recompute twin gap of every no-renormalisation cut)."""
    return [f"{a.floor_of}|{a.arm}" for a in panel.arms if a.kind == "floor_cut_recompute"]


def resolve_aliases(panel, outcomes):
    """``outcomes[context][name]`` with alias names (``panel.aliases``) mapped to the emitted arm.

    Raises if an alias and its target both carry a value and the values differ.
    """
    out = {}
    for context, rows in outcomes.items():
        row = {}
        for name, value in rows.items():
            target = panel.aliases.get(name, name)
            if target in row and row[target] != value:
                raise PanelError(f"{name!r} is an alias of {target!r} but carries a different value in {context!r}")
            row[target] = value
        out[context] = row
    return out


def _distance(value, what):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise PanelError(f"{what} must be a finite number")
    if value < 0:
        raise PanelError(f"{what} is negative ({value}); outcomes are distances (>= 0)")


def attach_results(panel, results, *, seal=None):
    """Validate ``results`` against ``panel``; refuse a plan-hash mismatch, a seal that does not
    verify or that the results do not cite, unknown arms or contexts, and non-finite or negative
    outcomes and pair distances."""
    if not isinstance(results, dict) or results.get("schema") != RESULTS_SCHEMA:
        raise PanelError("not an attention panel results record")
    if results.get("plan_sha256") != panel.sha256():
        raise PanelError("results were produced for a different plan (sha256 mismatch); refusing to attach")
    if seal is not None:
        verify_seal(panel, seal)
        if results.get("seal_sha256") != seal.get("seal_sha256"):
            raise PanelError("results do not cite this panel's seal")
    ids = panel.by_id()
    for context, rows in results["outcomes"].items():
        if context not in panel.contexts:
            raise PanelError(f"unknown context {context!r}")
        for arm, value in rows.items():
            if arm not in ids:
                if arm in panel.aliases:
                    raise PanelError(f"{arm!r} is an alias of {panel.aliases[arm]!r}; map results with "
                                     "resolve_aliases(panel, outcomes)")
                raise PanelError(f"unknown arm {arm!r}")
            _distance(value, f"outcome of {arm!r} in {context!r}")
    for context, rows in (results.get("pair_distances") or {}).items():
        if context not in panel.contexts:
            raise PanelError(f"pair distances for unknown context {context!r}")
        for key, value in rows.items():
            a, sep, b = key.partition("|")
            if not sep or a not in ids or b not in ids:
                raise PanelError(f"bad pair distance key {key!r} (need 'arm|arm' with known arms)")
            _distance(value, f"pair distance {key!r} in {context!r}")
    return results


def _pair(results, context, a, b):
    rows = (results.get("pair_distances") or {}).get(context, {})
    for key in (f"{a}|{b}", f"{b}|{a}"):
        if key in rows:
            return float(rows[key]), "pair distance"
    return None, None


def _floor(panel, results, arm_id, context):
    """(floor, parts) for one cut arm in one context; floor None if a floor arm is missing.

    ``parts["floor_incomplete"]`` is True when the delta-vs-recompute pair distance was not
    supplied and the scalar difference (a lower bound, can be 0) stood in for it.
    """
    out = results["outcomes"].get(context, {})
    floors = [a for a in panel.arms if a.floor_of == arm_id]
    a = panel.by_id()[arm_id]
    parts = {}
    if a.cut_type == "renorm":
        f = [x for x in floors if x.kind == "floor_sham"]
        if not f or f[0].arm not in out:
            return None, {"missing": "sham"}
        parts["sham"] = out[f[0].arm]
        parts["floor_incomplete"] = False
        return parts["sham"], parts
    sham = [x for x in floors if x.kind == "floor_sham_recompute"]
    twin = [x for x in floors if x.kind == "floor_cut_recompute"]
    if not sham or not twin or sham[0].arm not in out or twin[0].arm not in out or arm_id not in out:
        return None, {"missing": "sham_recompute and/or cut_recompute"}
    parts["sham_recompute"] = out[sham[0].arm]
    gap, source = _pair(results, context, arm_id, twin[0].arm)
    parts["floor_incomplete"] = gap is None
    if gap is None:
        gap, source = abs(out[arm_id] - out[twin[0].arm]), "|scalar difference| (lower bound on the pair distance)"
    parts["twin_gap"], parts["twin_gap_source"] = gap, source
    return max(parts["sham_recompute"], gap), parts


def _floor_factor(floor_factor, waiver):
    ff = _number(floor_factor, "floor_factor")
    waiver = _waiver_text(waiver, "floor_factor_waiver")
    if ff <= 0 and waiver is None:
        raise PanelError("floor_factor <= 0 makes every effect clear its floor; pass floor_factor_waiver = reason")
    return ff, waiver


def arm_effects(panel, results, *, floor_factor=1.0, floor_factor_waiver=None):
    """Effect of every test/control/reference cut arm per context, beside its floor."""
    attach_results(panel, results)
    ff, _ = _floor_factor(floor_factor, floor_factor_waiver)
    rows = []
    for a in panel.arms:
        if a.family != "read" or a.kind not in ("cut", "keep", "all_cut"):
            continue
        for context in panel.contexts:
            out = results["outcomes"].get(context, {})
            if a.arm not in out:
                continue
            f, parts = _floor(panel, results, a.arm, context)
            rows.append({"arm": a.arm, "context": context, "role": a.role, "cut_type": a.cut_type,
                         "effect": out[a.arm], "floor": f, "floor_parts": parts, "floor_factor": ff,
                         "floor_incomplete": parts.get("floor_incomplete", f is None),
                         "inside_floor": None if f is None else bool(out[a.arm] <= ff * f)})
    return rows


_PASS = "pass"


def _rank(statuses):
    for s in ("control_missing", "not_supported", "unresolved", "set_only", "supported", _PASS):
        if s in statuses:
            return s
    return "control_missing"


def _compare(panel, results, context, c, out, ff, f_test, margin_of):
    """One control against the test in one context (shared by both claims)."""
    if c.arm not in out:
        return {"control": c.arm, "kind": c.control, "outcome": "missing"}, False
    fc, cparts = _floor(panel, results, c.arm, context)
    margin = margin_of(out[c.arm])
    bound = ff * (f_test if fc is None else max(f_test, fc))
    beyond = bool(margin > bound)
    can_refute = c.mass_relation != "more"
    # a refutation with the smaller (test-only) bound also holds with the larger one
    refutes = bool(can_refute and not beyond)
    decided = fc is not None or refutes
    row = {"control": c.arm, "kind": c.control, "mass_relation": c.mass_relation,
           # a random/external set that did not hold mass fixed: beating it does not rule out mass
           "confounded_with_mass": c.control in ("random", "external") and c.mass_relation in (
               "unknown", "less", "matched_by_reuse"),
           "control_effect": out[c.arm], "margin": margin, "floor": bound, "control_floor_missing": fc is None,
           "beyond_floor": beyond if decided else None, "refutes": refutes,
           "rules_out": sorted(c.rules_out) if (beyond and fc is not None) else []}
    if not can_refute and not beyond:
        row["note"] = "heavier than the target: an equal effect is explained by its extra mass (neither refutes nor supports)"
    return row, bool(fc is not None and cparts.get("floor_incomplete"))


def _profiled(panel, c):
    """A control arm with ``mass_relation``/``rules_out`` filled in for plans written before
    those fields existed (other-item arms then carried ``separates_identity=True``; random arms
    are profiled from their stored draw report)."""
    if c.control == "other_item" and c.separates_identity is True and not c.rules_out:
        return replace(c, rules_out=("generic_damage",) if c.claim == "necessary" else ())
    if c.control == "random" and c.mass_relation is None:
        core = c.unit_set or ""
        keep = core.startswith("keep-")
        body = core.removeprefix("keep-")
        k, _, target = body.removeprefix("rand").partition("-")
        entry = panel.controls.get(f"{c.item}:{'keep-' if keep else ''}{target}", {})
        reports = entry.get("reports") or []
        if not k.isdigit() or int(k) >= len(reports):
            return replace(c, mass_relation="unknown", rules_out=RULES_OUT_BY_MASS["unknown"])
        relation = _mass_relation(reports[int(k)], None)
        info = entry.get("identity") or {}
        if relation == "matched" and info.get("separable") is False:
            relation = "matched_by_reuse"
        return replace(c, mass_relation=relation, rules_out=RULES_OUT_BY_MASS[relation])
    if c.control == "external" and c.mass_relation is None:
        return replace(c, mass_relation="unknown")
    return c


def _necessity_row(panel, results, t, controls, context, ff):
    out = results["outcomes"].get(context, {})
    row = {"context": context}
    if t.arm not in out:
        return {**row, "status": "control_missing", "why": "test outcome missing"}
    e = out[t.arm]
    f, parts = _floor(panel, results, t.arm, context)
    row.update(effect=e, floor=f, floor_parts=parts)
    if f is None:
        return {**row, "status": "control_missing", "why": "floor arm missing"}
    incomplete = bool(parts.get("floor_incomplete"))
    comps = []
    for c in controls:
        comp, inc = _compare(panel, results, context, c, out, ff, f, lambda ce: e - ce)
        comps.append(comp)
        incomplete |= inc
    row["controls"] = comps
    row["floor_incomplete"] = incomplete
    row["ruled_out"] = sorted({r for x in comps for r in x.get("rules_out", ())})
    if e <= ff * f:
        return {**row, "status": "unresolved", "why": "test effect inside its floor"}
    refuters = [x for x in comps if x.get("refutes")]
    if refuters:
        return {**row, "status": "not_supported", "refuted_by": [x["control"] for x in refuters],
                "why": "a control with no more mass than the target did as much damage (" +
                       ", ".join(f"{x['control']} {x['control_effect']:.4g} vs test {e:.4g}" for x in refuters) + ")"}
    if incomplete:
        return {**row, "status": "unresolved", "reason": "twin_gap_not_measured",
                "why": "twin gap not measured (no delta-vs-recompute pair distance): the floor is a lower bound"}
    if not set(row["ruled_out"]) & {"set_size", "generic_damage"}:
        return {**row, "status": "control_missing",
                "why": "no control rules out a generic-damage explanation (no random or other-item control beaten "
                       "beyond the floor with outcome and floor present)"}
    return {**row, "status": _PASS}


def _sufficiency_row(panel, results, t, controls, allcut, context, ff, threshold):
    out = results["outcomes"].get(context, {})
    row = {"context": context}
    if t.arm not in out:
        return {**row, "status": "control_missing", "why": "test outcome missing"}
    k = out[t.arm]
    f, parts = _floor(panel, results, t.arm, context)
    row.update(effect=k, keep_effect=k, floor=f, floor_parts=parts)
    if f is None:
        return {**row, "status": "control_missing", "why": "floor arm missing"}
    if allcut is None or allcut.arm not in out:
        return {**row, "status": "control_missing", "why": "all-cut reference missing for a keep claim"}
    a = out[allcut.arm]
    fa, aparts = _floor(panel, results, allcut.arm, context)
    row.update(all_cut_effect=a, all_cut_floor=fa)
    if fa is None:
        return {**row, "status": "control_missing", "why": "all-cut floor missing"}
    incomplete = bool(parts.get("floor_incomplete") or aparts.get("floor_incomplete"))
    if a <= ff * fa:
        return {**row, "floor_incomplete": incomplete, "status": "unresolved",
                "why": "cutting every unit is inside its floor: nothing to recover"}
    recovery = 1 - k / a
    row["recovery"] = recovery
    comps, refs = [], []
    for c in controls:
        if c.control != "random":
            if c.arm in out:
                refs.append({"control": c.arm, "kind": c.control, "control_effect": out[c.arm],
                             "recovery": 1 - out[c.arm] / a,
                             "note": "reported reference only; never counted for sufficiency"})
            continue
        comp, inc = _compare(panel, results, context, c, out, ff, f, lambda ce: ce - k)
        if "control_effect" in comp:
            comp["recovery"] = 1 - comp["control_effect"] / a
        comps.append(comp)
        incomplete |= inc
    row.update(controls=comps, references=refs, floor_incomplete=incomplete,
               ruled_out=sorted({r for x in comps for r in x.get("rules_out", ())}))
    if recovery < threshold:
        return {**row, "status": "not_supported", "finding": "low_recovery",
                "why": f"keeping the set recovers {recovery:.3g} of the all-cut effect, below {threshold:g}"}
    usable = [x for x in comps if "control_effect" in x and x["beyond_floor"] is not None]
    if not usable:
        return {**row, "status": "control_missing",
                "why": "no random keep control with outcome and floor (other-item keep does not count)"}
    refuters = [x for x in usable if x["refutes"]]
    if refuters:
        return {**row, "status": "not_supported", "refuted_by": [x["control"] for x in refuters],
                "finding": "any_same_size_set_suffices",
                "why": "a same-size random keep recovers as much (" +
                       ", ".join(f"{x['control']} recovery {x['recovery']:.3g}" for x in refuters) +
                       f" vs {recovery:.3g}): any same-size set suffices, so this set is not specific"}
    if incomplete:
        return {**row, "status": "unresolved", "reason": "twin_gap_not_measured",
                "why": "twin gap not measured (no delta-vs-recompute pair distance): the floor is a lower bound"}
    if "set_size" not in row["ruled_out"]:
        return {**row, "status": "control_missing", "why": "no random keep control beaten beyond the floor"}
    return {**row, "status": _PASS}


def _scope(claim, ruled):
    remaining = sorted(set(ALTERNATIVES[claim]) - set(ruled))
    if not remaining:
        return "supported", "identity", remaining
    return "set_only", "needed_set" if claim == "necessary" else "sufficient_set", remaining


def _statement(claim, item, unit_set, scope, remaining):
    what = (f"cutting {unit_set} damages {item} more than" if claim == "necessary"
            else f"keeping only {unit_set} recovers {item} better than")
    if scope == "identity":
        return (f"{what} a same-size set of matched mass and more than cutting these units on another item's rows: "
                "the units' identity matters for this item" if claim == "necessary" else
                f"{what} a same-size random set of matched mass: these units carry the item")
    return (f"WEAKER CLAIM ({scope}): {what} the controls that were beaten; NOT shown: "
            + ", ".join(remaining) + " (these alternatives remain open)")


def verdicts(panel, results, *, judgements=None, floor_factor=1.0, recovery_threshold=DEFAULT_RECOVERY,
             twin_gap_waiver=None, floor_factor_waiver=None):
    """One verdict record per (item, unit set, claim). Rules: docs/ATTENTION-PANELS.md,
    "What a verdict does and does not say".

    Necessity (cut, "these units are needed for this item"), per cut type and context:
    ``unresolved`` if the effect is inside its floor; ``not_supported`` if any control whose mass
    is not known to exceed the target's (random, other-item, external) has an effect not smaller
    than the test's by more than ``floor_factor`` x the larger floor (``refuted_by``);
    ``unresolved`` (``twin_gap_not_measured``) if a delta-vs-recompute pair distance is missing
    and ``twin_gap_waiver`` is not given; ``control_missing`` if no beaten control rules out
    ``set_size`` or ``generic_damage``. Otherwise the ruled-out alternatives decide the status:
    ``supported`` (scope ``identity``) only if ``set_size``, ``mass`` and ``generic_damage`` are
    all ruled out, else ``set_only`` (scope ``needed_set``) with ``remaining_alternatives``.

    Sufficiency (keep, "these units alone carry the item"): the all-cut effect must clear its
    floor; recovery = 1 - keep_effect / all_cut_effect must reach ``recovery_threshold``;
    a matched random KEEP control must recover clearly less (``not_supported`` with finding
    ``any_same_size_set_suffices`` otherwise); no random keep -> ``control_missing``. The
    other-item keep control is a reported reference only. Scope ``identity`` needs ``mass``
    ruled out too, else ``set_only`` (scope ``sufficient_set``).

    Both cut types must be present and pass; alternatives count as ruled out only if ruled out
    in every context and both cut types. Across contexts the worst status wins.

    ``judgements[(item, unit_set, claim)] = {"status", "by", "note"}``: a picture judgement on
    whether the effect is real. It may move the status between supported / set_only /
    not_supported / unresolved, but cannot repair a missing control, a missing cut type or an
    incomplete floor (``picture_blocked_by``), and an upgrade is capped by the scope the
    controls allow. Unknown keys raise.
    """
    attach_results(panel, results)
    ff, ff_waiver = _floor_factor(floor_factor, floor_factor_waiver)
    threshold = _number(recovery_threshold, "recovery_threshold")
    if not 0 < threshold <= 1:
        raise PanelError("recovery_threshold must be in (0, 1]")
    twin_waiver = _waiver_text(twin_gap_waiver, "twin_gap_waiver")
    judgements = judgements or {}
    arms = panel.by_id()
    groups = {}
    for a in panel.arms:
        if a.family == "read" and a.role == "test":
            groups.setdefault((a.item, a.unit_set, a.claim), []).append(a)
    unknown = [k for k in judgements if k not in groups]
    if unknown:
        raise PanelError(f"judgements name unknown (item, unit_set, claim) keys: {unknown[:3]}; known: "
                         f"{sorted(groups)[:6]}")
    records = []
    for (item, unit_set, claim), tests in groups.items():
        per_type, all_rows, any_incomplete, refuted_by, findings = {}, [], False, [], set()
        for t in tests:
            controls = [_profiled(panel, c) for c in panel.arms if t.arm in c.control_for]
            allcut = arms.get(f"{PREFIX[t.cut_type]}:none:{item}")
            ctx_rows = []
            for context in panel.contexts:
                if claim == "sufficient":
                    row = _sufficiency_row(panel, results, t, controls, allcut, context, ff, threshold)
                else:
                    row = _necessity_row(panel, results, t, controls, context, ff)
                if row.get("reason") == "twin_gap_not_measured" and twin_waiver is not None:
                    row = {**row, "status": _PASS, "twin_gap_waived": twin_waiver}
                    if claim == "necessary" and not set(row["ruled_out"]) & {"set_size", "generic_damage"}:
                        row["status"] = "control_missing"
                    if claim == "sufficient" and "set_size" not in row["ruled_out"]:
                        row["status"] = "control_missing"
                if row["status"] == _PASS:
                    row["status"] = _scope(claim, row["ruled_out"])[0]
                any_incomplete |= bool(row.get("floor_incomplete"))
                refuted_by += [{"cut_type": t.cut_type, "context": context, "control": c}
                               for c in row.get("refuted_by", ())]
                if row.get("finding"):
                    findings.add(row["finding"])
                ctx_rows.append(row)
                all_rows.append(row)
            status = _rank({r["status"] for r in ctx_rows})
            ruled = set.intersection(*(set(r.get("ruled_out", ())) for r in ctx_rows))
            if status in ("supported", "set_only"):
                status = _scope(claim, ruled)[0]
            per_type[t.cut_type] = {
                "arm": t.arm, "status": status, "contexts": ctx_rows, "ruled_out": sorted(ruled),
                "mass_confounded_controls": [c.arm for c in controls
                                             if c.control == "random" and c.mass_relation != "matched"],
                "separating_controls": [c.arm for c in controls if c.mass_relation == "matched"],
                "refuting_controls": sorted({x["control"] for x in refuted_by if x["cut_type"] == t.cut_type})}
        present = set(per_type)
        statuses = {per_type[t]["status"] for t in present}
        missing = [t for t in CUT_TYPES if t not in present]
        passing = {"supported", "set_only"}
        agree = len(present) == 2 and (len(statuses) == 1 or statuses <= passing)
        # an alternative is ruled out only if ruled out in every context of every cut type
        ruled = set.intersection(*(set(r.get("ruled_out", ())) for r in all_rows)) if all_rows else set()
        if missing:
            scalar = "control_missing"
            why = f"cut type(s) {missing} not run: a result counts only if both cut types agree"
        elif statuses <= passing:
            scalar, _, _ = _scope(claim, ruled)
            why = "both cut types agree"
        elif agree:
            scalar, why = statuses.pop(), "both cut types agree"
        elif "control_missing" in statuses:
            scalar, why = "control_missing", "a cut type lacks a control"
        elif statuses & passing and "not_supported" in statuses:
            scalar, why = "not_supported", "cut types disagree; the result does not count"
        else:
            scalar, why = _rank(statuses), "cut types disagree"
        _, scope, remaining = _scope(claim, ruled)
        if scalar not in passing:
            scope = None
        core = unit_set.removeprefix("keep-")
        entry = panel.controls.get(f"{item}:{unit_set}", {})
        thresholds = {"floor_factor": ff, "floor_factor_waiver": ff_waiver,
                      "refute_rule": "control effect >= test effect - floor_factor * max(test floor, control floor)",
                      "overlap_sd": entry.get("overlap_sd"), "mass_tolerance": entry.get("mass_tolerance")}
        if claim == "sufficient":
            thresholds["recovery_threshold"] = threshold
        rec = {"item": item, "unit_set": unit_set, "claim": claim, "scalar_status": scalar, "why": why,
               "claim_scope": scope,
               "claim_statement": None if scope is None else _statement(claim, item, unit_set, scope, remaining),
               "ruled_out": sorted(ruled), "remaining_alternatives": remaining,
               "refuted_by": refuted_by, "findings": sorted(findings),
               "floor_incomplete": any_incomplete, "twin_gap_waiver": twin_waiver, "thresholds": thresholds,
               "cut_types": per_type, "cut_types_agree": agree, "missing_cut_types": missing,
               "waivers": dict(panel.waivers), "identity": panel.identity.get(f"{item}:{core}"),
               "picture": None, "overridden_by_picture": False, "picture_blocked_by": [], "status": scalar}
        j = judgements.get((item, unit_set, claim))
        if j is not None:
            if j.get("status") not in STATUSES or not j.get("by"):
                raise PanelError("a picture judgement needs a status in STATUSES and who made it ('by')")
            rec["picture"] = dict(j)
            blocked = []
            if j["status"] in passing:
                if missing:
                    blocked.append("missing_cut_type")
                if scalar == "control_missing":
                    blocked.append("control_missing")
                if any_incomplete and twin_waiver is None:
                    blocked.append("floor_incomplete")
                minimal = {"set_size", "generic_damage"} if claim == "necessary" else {"set_size"}
                if not ruled & minimal and "control_missing" not in blocked:
                    blocked.append("no_control_rules_out_generic_damage")
            if blocked:
                rec["picture_blocked_by"] = blocked
            else:
                status = j["status"]
                if status in passing:
                    status, scope, remaining = _scope(claim, ruled)
                    rec.update(claim_scope=scope, remaining_alternatives=remaining,
                               claim_statement=_statement(claim, item, unit_set, scope, remaining))
                    if status != j["status"]:
                        rec["picture_capped"] = f"picture said {j['status']}; the controls allow at most {status}"
                rec["status"] = status
            rec["overridden_by_picture"] = rec["status"] != scalar
        if rec["status"] not in passing:
            rec["claim_scope"], rec["claim_statement"] = None, None
        records.append(rec)
    return records
