"""Conservative suffix-reuse plans from bound, declared transition footprints.

Planning never executes a model, grants a write or proves footprint completeness.
Actual replay and numerical laws remain native-consumer qualifications.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence


class DependencyReuseError(ValueError):
    """An identity, transition or sealed plan is malformed."""


def _digest(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError) as error:
        raise DependencyReuseError("plan must contain finite JSON") from error
    return hashlib.sha256(raw).hexdigest()


def _slots(value, label):
    if (isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
            or any(not isinstance(v, str) or not v for v in value)
            or len(set(value)) != len(value)):
        raise DependencyReuseError(label + " must contain distinct slot names")
    return tuple(value)


def plan_suffix_reuse(transitions, *, changed_slots, model_identity, execution_identity,
                      source_parent, target_parent, max_transitions=4096):
    """Track potentially changed state through an ordered, deterministic program.

    Unaffected declared reads permit a reuse candidate. An affected read marks
    every declared write as potentially changed. An identical-input overwrite
    clears that slot's prior difference. Unknown footprints taint the remaining
    plan. This planner does not reason from floating-point proximity or names.
    """
    changed = set(_slots(changed_slots, "changed_slots"))
    if (type(max_transitions) is not int or max_transitions < 1
            or isinstance(transitions, (str, bytes)) or not isinstance(transitions, Sequence)
            or not transitions or len(transitions) > max_transitions):
        raise DependencyReuseError("ordered transitions exceed their finite bound")
    for label, value in (("model_identity", model_identity), ("execution_identity", execution_identity),
                         ("source_parent", source_parent), ("target_parent", target_parent)):
        if not isinstance(value, str) or not value:
            raise DependencyReuseError(label + " is required")
    original = sorted(changed)
    rows, tainted = [], False
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Mapping):
            raise DependencyReuseError("each transition must be a mapping")
        spec = copy.deepcopy(dict(transition))
        if not isinstance(spec.get("operation"), str) or not spec["operation"]:
            raise DependencyReuseError("transition operation is required")
        reads = set(_slots(spec.get("reads", ()), "reads"))
        writes = set(_slots(spec.get("writes", ()), "writes"))
        invalidates = set(_slots(spec.get("invalidates", ()), "invalidates"))
        footprint = spec.get("footprint", "unknown")
        if footprint not in {"declared", "unknown"}:
            raise DependencyReuseError("unknown footprint status")
        affected = sorted(reads & changed)
        before = sorted(changed)
        tainted = tainted or footprint == "unknown"
        reusable = not tainted and not affected
        if reusable:
            changed.difference_update(writes)
            changed.update(invalidates)
        else:
            changed.update(writes | invalidates)
        rows.append({"index": index, "transition": spec,
                     "decision": "reuse-candidate" if reusable else "recompute-candidate",
                     "affected_reads": affected, "unknown_footprint_taint": tainted,
                     "changed_before": before, "changed_after": sorted(changed)})
    body = {"schema": "mdb-dependency-reuse-plan-v1", "model_identity": model_identity,
            "execution_identity": execution_identity, "source_parent": source_parent,
            "target_parent": target_parent, "initial_changed_slots": original, "rows": rows,
            "reuse_candidates": sum(r["decision"] == "reuse-candidate" for r in rows),
            "recompute_candidates": sum(r["decision"] == "recompute-candidate" for r in rows),
            "authority": "diagnostic-planning-only",
            "conditions": "Same deterministic model/execution program and complete declared footprints. Parent differences must include every changed state slot. Native replay and consumer checks qualify reuse; the planner does not prove these conditions."}
    return {**body, "fingerprint": _digest(body)}


def verify_reuse_plan(plan):
    if not isinstance(plan, Mapping) or plan.get("schema") != "mdb-dependency-reuse-plan-v1":
        raise DependencyReuseError("unsupported reuse plan")
    body = {k: v for k, v in plan.items() if k != "fingerprint"}
    if _digest(body) != plan.get("fingerprint"):
        raise DependencyReuseError("reuse plan fingerprint differs")
    replay = plan_suffix_reuse([r["transition"] for r in plan["rows"]],
        changed_slots=plan["initial_changed_slots"], model_identity=plan["model_identity"],
        execution_identity=plan["execution_identity"], source_parent=plan["source_parent"],
        target_parent=plan["target_parent"], max_transitions=max(4096, len(plan["rows"])))
    if replay != dict(plan):
        raise DependencyReuseError("reuse plan decisions differ from its footprints")
    return {"verified": True, "fingerprint": plan["fingerprint"],
            "reuse_candidates": plan["reuse_candidates"], "recompute_candidates": plan["recompute_candidates"]}


__all__ = ["DependencyReuseError", "plan_suffix_reuse", "verify_reuse_plan"]
