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

Mappings: :func:`port_cuts` (arm -> ``attention_port`` ``Cut`` specs), :func:`value_spec`
(arm -> side, payload side, target row, payload row, cells, dose) and ONE dose blend,
:func:`blend`. :func:`to_legacy` writes the hand-written ``{arm, kind, cells, dose}`` schema.

Analysis: :func:`attach_results` refuses results whose plan hash differs; :func:`arm_effects`
gives each test arm's effect beside its floor; :func:`verdicts` returns one record per unit set
with status ``supported`` / ``not_supported`` / ``unresolved`` (inside floor) /
``control_missing`` and a slot for a picture judgement that overrides the scalar.

See ``docs/ATTENTION-PANELS.md``.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass, field, replace

SCHEMA = "mdb-attention-panel-v1"
RESULTS_SCHEMA = "mdb-attention-panel-results-v1"
CUT_TYPES = ("zero", "renorm")
PREFIX = {"zero": "z", "renorm": "m"}
ROLES = ("test", "control", "floor", "reference")
READ_KINDS = ("cut", "keep", "floor_sham", "floor_sham_recompute", "floor_cut_recompute",
              "condition", "native", "all_cut", "deletion", "sham_all")
VALUE_KINDS = ("singleton", "complement", "coalition", "dose", "wrong_target", "wrong_payload",
               "reciprocal", "sham", "replay", "fixed_distance", "native", "donor_native")
STATUSES = ("supported", "not_supported", "unresolved", "control_missing")
WAIVABLE = ("floors", "both_cut_types", "random_cut", "random_keep", "other_item")
FLUX_LANE_CAP = 256


class PanelError(ValueError):
    """A refused panel, budget, seal or result attachment."""


# ----------------------------------------------------------------------------------- utilities

def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _unit(u):
    if isinstance(u, numbers.Integral) and not isinstance(u, bool):
        return (int(u),)
    out = tuple(int(x) for x in u)
    if not out:
        raise PanelError("a unit needs at least one coordinate")
    return out


def units_of(values):
    """Sorted, de-duplicated tuple of integer unit tuples."""
    return tuple(sorted({_unit(u) for u in values}))


def _rows(spec):
    if spec is None:
        return ()
    if isinstance(spec, str):
        return (spec,)
    return tuple(spec)


def _fill(template, item):
    return tuple(t.format(item=item) for t in template)


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
    ``separates_identity``: for a control, whether it can separate unit identity from mass
    (``None`` = unknown). ``claim``: ``"necessary"`` (cut) or ``"sufficient"`` (keep).
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

    @property
    def renormalize(self):
        return None if self.cut_type is None else self.cut_type == "renorm"

    def to_dict(self):
        out = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
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
        for name in ("receivers", "sources", "control_for", "notes"):
            d[name] = tuple(d.get(name) or ())
        return cls(**d)


# ----------------------------------------------------------------------------------- the panel

@dataclass
class Panel:
    """A finite, sealed-able list of arms run in every context (one (prompt, seed) per context)."""
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
        return {"schema": SCHEMA, "family": self.family, "contexts": list(self.contexts),
                "arms": [a.to_dict() for a in self.arms], "sets": self.sets, "controls": self.controls,
                "identity": self.identity, "waivers": self.waivers, "port": self.port,
                "aliases": self.aliases, "notes": list(self.notes),
                "branches": self.branch_count}

    @classmethod
    def from_json(cls, d):
        if d.get("schema") != SCHEMA:
            raise PanelError("not an attention panel plan")
        return cls(family=d["family"], arms=[Arm.from_dict(a) for a in d["arms"]], contexts=d["contexts"],
                   sets=d.get("sets", {}), controls=d.get("controls", {}), identity=d.get("identity", {}),
                   waivers=d.get("waivers", {}), port=d.get("port", {}), aliases=d.get("aliases", {}),
                   notes=d.get("notes", []))

    def canonical(self):
        return canonical_json(self.to_json())

    def sha256(self):
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def seal(self, predictions=None):
        """Seal the plan (and optional predictions keyed by arm id or unit set) before execution."""
        plan_sha = self.sha256()
        record = {"schema": "mdb-attention-panel-seal-v1", "plan_sha256": plan_sha}
        if predictions is not None:
            json.loads(canonical_json(predictions))  # finite JSON only
            record["predictions"] = predictions
            record["predictions_sha256"] = sha256_json(predictions)
        record["seal_sha256"] = sha256_json(record)
        return record

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

    def leases(self, cap=FLUX_LANE_CAP, *, max_total=None):
        """Deterministic split into leases of at most ``cap`` branches.

        Atoms are (context, bundle): a test set's arms, its floors, twins and controls stay in one
        lease for one context. Atoms are packed in plan order (context-major); an atom larger
        than ``cap`` is refused. ``max_total`` refuses the whole panel first.
        """
        if max_total is not None:
            self.check_budget(max_total)
        if isinstance(cap, bool) or not isinstance(cap, numbers.Integral) or cap < 1:
            raise PanelError("cap must be a positive integer")
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
        return [{"plan_sha256": plan_sha, "lease": i, "of": len(leases), "branches": b,
                 "lease_sha256": sha256_json({"plan_sha256": plan_sha, "lease": i, "branches": b})}
                for i, b in enumerate(leases)]


# --------------------------------------------------------------------------- read panel generator

@dataclass(frozen=True)
class RandomControls:
    """Matched random unit sets via :func:`mdb.attention_ranking.matched_control_sets`.

    ``strata``: one label per universe unit (e.g. joint / single). ``match_blocks``: match the
    number of distinct blocks (block = unit[0]). ``mass``: per-unit mass (e.g. the item's
    read-mass share), required to say anything about identity vs mass; ``mass_tolerance``
    requests mass matching. ``disjoint``: draw outside the target.
    """
    n: int = 1
    strata: tuple | None = None
    match_blocks: bool = True
    mass: tuple | None = None
    mass_tolerance: float | None = None
    disjoint: bool = False
    seed: int = 0


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


def _draw_random(name, units, universe, spec, index, item):
    from .attention_ranking import matched_control_sets
    target = [index[u] for u in units]
    blocks = [u[0] for u in universe] if spec.match_blocks else None
    entry = {"target": name, "item": item, "n": spec.n, "seed": spec.seed, "disjoint": spec.disjoint,
             "mass_tolerance": spec.mass_tolerance, "match_blocks": spec.match_blocks}
    try:
        drawn = matched_control_sets(target, len(universe), strata=spec.strata, blocks=blocks, mass=spec.mass,
                                     mass_tolerance=spec.mass_tolerance, disjoint=spec.disjoint,
                                     seed=spec.seed, n_sets=spec.n)
    except ValueError as error:
        entry.update(status="control_missing", reason=str(error), sets=[])
        return entry, None
    sets, reports = [], []
    for d in drawn:
        sets.append(tuple(universe[int(i)] for i in d["units"]))
        reports.append(d["report"])
    entry["reports"] = reports
    entry["sets"] = [[list(u) for u in s] for s in sets]
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
            info.update(overlap_with_target=overlaps, chance_overlap_mean=mean, chance_overlap_sd=sd)
            if max(overlaps) > mean + 3 * sd:
                info["separable"] = False
                info["why"] = (f"mass was reached by re-using target units (overlap {max(overlaps)} of "
                               f"{len(target)}; chance {mean:.1f} +/- {sd:.1f}): the control shares the "
                               "identity it should vary; use disjoint=True or the other-item control")
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
    return entry, sets


@dataclass(frozen=True)
class ReadPanelSpec:
    """Everything the read-panel generator needs; see :func:`read_panel`.

    ``sets[item][name]`` = units. ``cut_sets``/``keep_sets``: names that get test arms.
    ``cut_types``: default types for test arms; ``cut_type_overrides`` / ``keep_type_overrides``
    per set name. ``external_controls``: ``{control_set_name: target_set_name}`` for hand-drawn
    control sets (they inherit the target's cut types). ``nested``: set names in increasing
    size; each larger set gets ``<big>-minus-<small>`` added-units-alone test arms.
    ``other_items``: ``{item: (other, ...)}``; default = the next item (cyclic).
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


def _types(t):
    t = tuple(t)
    if not t or any(x not in CUT_TYPES for x in t) or len(set(t)) != len(t):
        raise PanelError("cut types must be a nonempty subset of ('zero', 'renorm')")
    return tuple(x for x in CUT_TYPES if x in t)


def read_panel(spec):
    """Emit the full read-cut panel for ``spec`` (see the module and doc for the controls)."""
    waive = dict(spec.waive)
    for key, why in waive.items():
        if key not in WAIVABLE or not isinstance(why, str) or not why.strip():
            raise PanelError(f"waiver {key!r} needs a known key {WAIVABLE} and a written reason")
    universe = tuple(_unit(u) for u in spec.universe)
    index = _flat(universe)
    items = tuple(spec.items)
    if not items:
        raise PanelError("at least one item is required")
    if not spec.floors and "floors" not in waive:
        raise PanelError("floors=False needs waive['floors'] = reason (every no-renormalisation cut needs "
                         "its recompute sham and recompute twin, every renormalising cut its sham)")
    other = spec.other_items
    if other is None:
        other = {it: ((items[(i + 1) % len(items)],) if len(items) > 1 else ()) for i, it in enumerate(items)}
    sets = {}
    for item in items:
        named = {k: units_of(v) for k, v in spec.sets[item].items()}
        for name, units in named.items():
            missing = [u for u in units if u not in index]
            if missing:
                raise PanelError(f"set {name} of {item} has units outside the universe: {missing[:3]}")
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

    def cut_arm(item, core, kind, role, units, unit_set, cut_type, bundle, rows_item=None, **extra):
        rows_item = rows_item or item
        return Arm(arm=name_of(cut_type, core, item), family="read", kind=kind, role=role, item=item,
                   units=units, unit_set=unit_set, rows_item=rows_item,
                   receivers=_fill(spec.receivers, rows_item), sources=_fill(spec.sources, rows_item),
                   cut_type=cut_type, zero_method=spec.zero_method,
                   condition=None if spec.condition is None else spec.condition.format(item=item),
                   reference=ref_id(item), bundle=bundle, **extra)

    def ref_id(item):
        if "condition" in spec.references and spec.condition is not None:
            return spec.condition.format(item=item)
        return f"native:{item}" if "native" in spec.references else "native"   # "native" = unmodified input

    def with_floors(arm):
        out = [arm]
        if not spec.floors:
            return out
        if arm.cut_type == "renorm":
            out.append(replace(arm, arm=arm.arm + "/sham", kind="floor_sham", role="floor", floor_of=arm.arm,
                               control_for=(), twin=None, control=None, separates_identity=None, claim=None))
        else:
            out.append(replace(arm, arm=arm.arm + "/sham_recompute", kind="floor_sham_recompute", role="floor",
                               floor_of=arm.arm, control_for=(), twin=None, control=None,
                               separates_identity=None, claim=None))
            out.append(replace(arm, arm=arm.arm + "/cut_recompute", kind="floor_cut_recompute", role="floor",
                               floor_of=arm.arm, control_for=(), twin=None, control=None,
                               separates_identity=None, claim=None))
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
        if "condition" in spec.references and cond is not None:
            refs.append(Arm(arm=cond, family="read", kind="condition", role="reference", item=item,
                            condition=cond, reference="native", bundle=bundle))
        if "native" in spec.references:
            refs.append(Arm(arm=f"native:{item}", family="read", kind="native", role="reference", item=item,
                            bundle=bundle))
        if "deletion" in spec.references:
            refs.append(Arm(arm=f"del:{item}", family="read", kind="deletion", role="reference", item=item,
                            condition=f"delete:{item}", reference="native", bundle=bundle))
        if "sham_all" in spec.references:
            refs.append(Arm(arm=f"z:sham-all:{item}", family="read", kind="sham_all", role="floor", item=item,
                            cut_type="zero", zero_method="recompute", condition=cond, reference=ref_id(item),
                            bundle=bundle, notes=("every head, block and row through the FP32 recompute path; "
                                                  "nothing removed",)))
        if "all_cut" in spec.references:
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
                # external (hand-drawn) controls for cut sets
                if not keep:
                    for c in external_for.get(s, ()):
                        arms += family_arms(item, c, "cut", "control", named[c], c, types, bundle,
                                            control_for=test_ids, control="external", claim=claim,
                                            separates_identity=None)
                # matched random controls (cut and keep)
                if spec.random is not None:
                    key = f"{item}:{core}"
                    entry, drawn = _draw_random(s, units, universe, spec.random, index, item)
                    controls[key] = entry
                    sep = entry.get("identity", {}).get("separable")
                    if drawn is None:
                        identity.setdefault(f"{item}:{s}", {})["random_" + kind] = {
                            "status": "control_missing", "why": entry["reason"]}
                    else:
                        identity.setdefault(f"{item}:{s}", {})["random_" + kind] = entry["identity"]
                        for k, rset in enumerate(drawn):
                            rcore = f"keep-rand{k}-{s}" if keep else f"rand{k}-{s}"
                            arms += family_arms(item, rcore, kind, "control",
                                                complement(rset) if keep else units_of(rset), rcore, types, bundle,
                                                control_for=test_ids, control="random", claim=claim,
                                                separates_identity=sep)
                # other item, same units
                for o in others:
                    ocore = f"{core}@rows-{o}"
                    arms += family_arms(item, ocore, kind, "control", act_units, core, types, bundle,
                                        rows_item=o, control_for=test_ids, control="other_item", claim=claim,
                                        separates_identity=True)
                    identity.setdefault(f"{item}:{s}", {})[f"other_item_{kind}"] = {
                        "separable": True, "other": o,
                        "why": "same units on another item's rows: the mass the units carry for this item "
                               "is not what is cut, so an equal effect means the units, not the item, matter"}
    if spec.random is None:
        notes.append("random controls: none generated (" + "; ".join(
            f"{k}: {waive[k]}" for k in ("random_cut", "random_keep") if k in waive) + ")")
    port = {"zero_method": spec.zero_method or "port default", **spec.port}
    return Panel(family="read", arms=arms, contexts=list(spec.contexts),
                 sets={item: {k: [list(u) for u in v] for k, v in sets[item].items()} for item in items},
                 controls=controls, identity=identity, waivers=waive, port=port, notes=notes)


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

def port_cuts(arm, *, zero_method=None):
    """Map a read arm to ``attention_port`` ``Cut`` specs (``()`` for an arm that cuts nothing).

    Floor arms come from :func:`mdb.attention_port.floor_arms` of the base cut, so the
    panel and the port cannot disagree about what a floor is. ``arm.condition`` (the input
    variant) is the caller's responsibility; the port does not see it.
    """
    from . import attention_port as ap
    if arm.family != "read":
        raise PanelError("port_cuts maps read arms; use value_spec for value arms")
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
                controls=("sham", "wrong_payload", "wrong_target", "reciprocal"),
                references=("native", "replay", "donor_native"), comparators=None, fixed_distance=False,
                reference_dose=0, contexts=("default",), item=None, cell_label=_cell_label):
    """Value-replacement (Codex) panel over ``cells`` ((layer, kv group) tuples).

    Arms whose executable content (legacy kind, cell set, dose) coincides are emitted once; the
    dropped names are recorded in ``panel.aliases`` (with two cells, every complement equals a
    singleton). ``replay`` is exempt: it repeats ``native`` on purpose (determinism check).
    ``reference_dose`` is the inert dose written on native arms (workers skip replacement for
    them); the 2026-11-12 plans used 0 (value_coalition) and 1 (frozen pair, resolver).
    """
    cells = units_of(cells)
    if not cells:
        raise PanelError("a value panel needs cells")
    full = cells
    rows = []

    def add(kind, role, name, units, dose, **extra):
        rows.append(Arm(arm=name, family="value", kind=kind, role=role, item=item, units=units,
                        unit_set=extra.pop("unit_set", None), dose=float(dose), **extra))

    ref_kinds = [k for k in ("native", "replay", "donor_native") if k in references]
    for k in ref_kinds:
        add(k, "reference", {"native": "native-recipient", "replay": "replay-recipient",
                             "donor_native": "native-donor"}[k], full, reference_dose, unit_set=label)
    test_all = f"coalition-{label}"
    if "sham" in controls:
        add("sham", "control", f"sham-{label}", full, 1, unit_set=label, control_for=(test_all,),
            reference="native-recipient")
    add("coalition", "test", test_all, full, 1, unit_set=label, reference="native-recipient")
    for d in doses:
        add("dose", "test", f"dose{d:g}-{label}", full, d, unit_set=label, reference="native-recipient")
    for k in ("wrong_payload", "wrong_target"):
        if k in controls:
            add(k, "control", f"{k.replace('_', '-')}-{label}", full, 1, unit_set=label,
                control_for=(test_all,), reference="native-recipient")
    if "reciprocal" in controls:
        add("reciprocal", "control", f"reciprocal-{label}", full, 1, unit_set=label, control_for=(test_all,),
            reference="native-donor")
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
    if fixed_distance:
        add("fixed_distance", "control", f"fixed-distance-{label}", full, 1, unit_set=label,
            control_for=(test_all,), reference="native-recipient")
    seen, arms, aliases = {}, [], {}
    for a in rows:
        key = (LEGACY_VALUE_KIND[a.kind], a.units, a.dose if a.kind not in ("native", "donor_native") else None)
        if a.kind != "replay" and key in seen:
            aliases[a.arm] = seen[key]
            continue
        if a.kind != "replay":
            seen[key] = a.arm
        arms.append(a)
    return Panel(family="value", arms=arms, contexts=list(contexts), aliases=aliases,
                 sets={"cells": [list(c) for c in full]})


def value_spec(arm):
    """Arm -> replacement spec. ``None`` for arms that replace nothing (native, replay, donor native).

    Rows are symbolic: ``"source"`` (the declared/nominated source row), ``"wrong"`` (source - 1
    in the 2026-11-12 workers) and ``"fixed-distance"`` (a fixed offset from the end); the
    caller resolves them. This is the single copy of the mapping hand-written in the
    value-coalition, frozen-value-pair and resolver workers.
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

    dose 1: an exact copy of ``payload`` (no arithmetic). dose 0: a copy of ``before``. Other
    doses: FP32 linear interpolation ``before + dose * (payload - before)`` rounded once to
    ``before``'s dtype. Bit-identical to ``value_coalition.replacement`` (tested).
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

    ``outcomes[context][arm_id]`` = scalar distance of that arm's output from its
    ``reference`` arm (e.g. pixel MAD vs ``iso:<item>``). ``pair_distances[context]["a|b"]`` =
    direct distance between two arms (used for the delta/recompute twin gap when present).
    """
    rec = {"schema": RESULTS_SCHEMA, "plan_sha256": panel.sha256(), "metric": metric,
           "outcomes": outcomes, "pair_distances": pair_distances or {}}
    if seal is not None:
        rec["seal_sha256"] = seal["seal_sha256"]
    return rec


def attach_results(panel, results, *, seal=None):
    """Validate ``results`` against ``panel``; refuse a plan-hash mismatch or unknown arms."""
    if not isinstance(results, dict) or results.get("schema") != RESULTS_SCHEMA:
        raise PanelError("not an attention panel results record")
    if results.get("plan_sha256") != panel.sha256():
        raise PanelError("results were produced for a different plan (sha256 mismatch); refusing to attach")
    if seal is not None:
        if seal.get("plan_sha256") != panel.sha256() or results.get("seal_sha256") != seal.get("seal_sha256"):
            raise PanelError("results do not cite this panel's seal")
    ids = panel.by_id()
    for context, rows in results["outcomes"].items():
        if context not in panel.contexts:
            raise PanelError(f"unknown context {context!r}")
        for arm, value in rows.items():
            if arm not in ids:
                raise PanelError(f"unknown arm {arm!r}")
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise PanelError(f"outcome of {arm!r} in {context!r} must be a finite number")
    for context, rows in results.get("pair_distances", {}).items():
        for key, value in rows.items():
            a, _, b = key.partition("|")
            if a not in ids or b not in ids or not math.isfinite(value):
                raise PanelError(f"bad pair distance {key!r}")
    return results


def _pair(results, context, a, b):
    rows = results.get("pair_distances", {}).get(context, {})
    for key in (f"{a}|{b}", f"{b}|{a}"):
        if key in rows:
            return float(rows[key]), "pair distance"
    return None, None


def _floor(panel, results, arm_id, context):
    """(floor, parts) for one cut arm in one context; floor None if a floor arm is missing."""
    arms = panel.by_id()
    out = results["outcomes"].get(context, {})
    floors = [a for a in panel.arms if a.floor_of == arm_id]
    a = arms[arm_id]
    parts = {}
    if a.cut_type == "renorm":
        f = [x for x in floors if x.kind == "floor_sham"]
        if not f or f[0].arm not in out:
            return None, {"missing": "sham"}
        parts["sham"] = out[f[0].arm]
        return parts["sham"], parts
    sham = [x for x in floors if x.kind == "floor_sham_recompute"]
    twin = [x for x in floors if x.kind == "floor_cut_recompute"]
    if not sham or not twin or sham[0].arm not in out or twin[0].arm not in out or arm_id not in out:
        return None, {"missing": "sham_recompute and/or cut_recompute"}
    parts["sham_recompute"] = out[sham[0].arm]
    gap, source = _pair(results, context, arm_id, twin[0].arm)
    if gap is None:
        gap, source = abs(out[arm_id] - out[twin[0].arm]), "|scalar difference| (lower bound on the pair distance)"
    parts["twin_gap"], parts["twin_gap_source"] = gap, source
    return max(parts["sham_recompute"], gap), parts


def arm_effects(panel, results, *, floor_factor=1.0):
    """Effect of every test/control/reference cut arm per context, beside its floor."""
    attach_results(panel, results)
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
                         "effect": out[a.arm], "floor": f, "floor_parts": parts,
                         "inside_floor": None if f is None else bool(out[a.arm] <= floor_factor * f)})
    return rows


def _rank(statuses):
    for s in ("control_missing", "not_supported", "unresolved", "supported"):
        if s in statuses:
            return s
    return "control_missing"


def verdicts(panel, results, *, judgements=None, floor_factor=1.0):
    """One verdict record per (item, unit set, claim).

    Per cut type and context: ``unresolved`` if the test effect (necessary) or the all-cut
    effect (sufficient) is inside its floor; ``control_missing`` if a floor arm or every
    identity-separating control (mass-matched random set or other-item control) is missing;
    ``supported`` if the effect beats every separating control by more than the larger floor
    (for a keep arm: is smaller than every control's effect by more than that); otherwise
    ``not_supported``. Across contexts the worst status wins (control_missing > not_supported >
    unresolved > supported). Across cut types the result counts only if both types are present
    and agree. Mass-confounded random controls are reported, never counted.

    ``judgements[(item, unit_set, claim)] = {"status": ..., "by": ..., "note": ...}`` records a
    human/auditor picture judgement; where it disagrees with the scalar, it wins and
    ``overridden_by_picture`` is True.
    """
    attach_results(panel, results)
    judgements = judgements or {}
    arms = panel.by_id()
    groups = {}
    for a in panel.arms:
        if a.family == "read" and a.role == "test":
            groups.setdefault((a.item, a.unit_set, a.claim), []).append(a)
    records = []
    for (item, unit_set, claim), tests in groups.items():
        per_type = {}
        for t in tests:
            ctx_rows = []
            controls = [c for c in panel.arms if t.arm in c.control_for]
            sep = [c for c in controls if c.separates_identity]
            confounded = [c for c in controls if not c.separates_identity]
            allcut = arms.get(f"{PREFIX[t.cut_type]}:none:{item}")
            for context in panel.contexts:
                out = results["outcomes"].get(context, {})
                row = {"context": context}
                if t.arm not in out:
                    row.update(status="control_missing", why="test outcome missing")
                    ctx_rows.append(row)
                    continue
                e = out[t.arm]
                f, parts = _floor(panel, results, t.arm, context)
                row.update(effect=e, floor=f, floor_parts=parts)
                if f is None:
                    row.update(status="control_missing", why="floor arm missing")
                    ctx_rows.append(row)
                    continue
                if claim == "sufficient":
                    if allcut is None or allcut.arm not in out:
                        row.update(status="control_missing", why="all-cut reference missing for a keep claim")
                        ctx_rows.append(row)
                        continue
                    fa, _ = _floor(panel, results, allcut.arm, context)
                    row["all_cut_effect"] = out[allcut.arm]
                    if fa is None:
                        row.update(status="control_missing", why="all-cut floor missing")
                        ctx_rows.append(row)
                        continue
                    if out[allcut.arm] <= floor_factor * max(fa, f):
                        row.update(status="unresolved", why="cutting every unit is inside the floor")
                        ctx_rows.append(row)
                        continue
                elif e <= floor_factor * f:
                    row.update(status="unresolved", why="test effect inside its floor")
                    ctx_rows.append(row)
                    continue
                comps = []
                for c in controls:
                    if c.arm not in out:
                        continue
                    fc, _ = _floor(panel, results, c.arm, context)
                    margin = (e - out[c.arm]) if claim == "necessary" else (out[c.arm] - e)
                    bound = None if fc is None else floor_factor * max(f, fc)
                    comps.append({"control": c.arm, "kind": c.control, "separates_identity": c.separates_identity,
                                  "control_effect": out[c.arm], "margin": margin, "floor": bound,
                                  "beyond_floor": None if bound is None else bool(margin > bound)})
                row["controls"] = comps
                counted = [x for x in comps if x["separates_identity"] and x["floor"] is not None]
                if not counted:
                    row.update(status="control_missing",
                               why="no identity-separating control with outcome and floor (random sets "
                                   "mass-confounded or missing; no other-item control)")
                elif all(x["beyond_floor"] for x in counted):
                    row["status"] = "supported"
                else:
                    row.update(status="not_supported", why="effect not beyond every separating control")
                ctx_rows.append(row)
            per_type[t.cut_type] = {"arm": t.arm, "status": _rank({r["status"] for r in ctx_rows}),
                                    "contexts": ctx_rows,
                                    "mass_confounded_controls": [c.arm for c in confounded],
                                    "separating_controls": [c.arm for c in sep]}
        present = set(per_type)
        statuses = {per_type[t]["status"] for t in present}
        missing = [t for t in CUT_TYPES if t not in present]
        agree = len(present) == 2 and len(statuses) == 1
        if missing:
            scalar = "control_missing"
            why = f"cut type(s) {missing} not run: a result counts only if both cut types agree"
        elif agree:
            scalar, why = statuses.pop(), "both cut types agree"
        elif "control_missing" in statuses:
            scalar, why = "control_missing", "a cut type lacks a control"
        elif statuses == {"supported", "not_supported"}:
            scalar, why = "not_supported", "cut types disagree; the result does not count"
        else:
            scalar, why = _rank(statuses), "cut types disagree"
        rec = {"item": item, "unit_set": unit_set, "claim": claim, "scalar_status": scalar, "why": why,
               "cut_types": per_type, "cut_types_agree": agree, "missing_cut_types": missing,
               "waivers": dict(panel.waivers), "identity": panel.identity.get(f"{item}:{unit_set.removeprefix('keep-')}"),
               "picture": None, "overridden_by_picture": False, "status": scalar}
        j = judgements.get((item, unit_set, claim))
        if j is not None:
            if j.get("status") not in STATUSES or not j.get("by"):
                raise PanelError("a picture judgement needs a status in STATUSES and who made it ('by')")
            rec["picture"] = dict(j)
            rec["status"] = j["status"]
            rec["overridden_by_picture"] = j["status"] != scalar
        records.append(rec)
    return records
