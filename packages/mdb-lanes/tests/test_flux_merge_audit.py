"""Saved-receipt conditioning reset regressions; no native model or equations."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "qualification"
AUDIT = runpy.run_path(str(HERE / "analyze_flux_merge.py"))
TYPED = runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py"))
PUBLIC = runpy.run_path(str(HERE / "summarize_context_window.py"))["public_receipt"]


def seal(value):
    value = {key: item for key, item in value.items() if key != "fingerprint"}
    value["fingerprint"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return value


def edge(parent, result, writes=("image", "text")):
    return {"parent": parent, "result": result, "operation": "continue", "steps": 1, "fingerprint": parent + ".receipt",
            "transitions": [{"native_operation": {"writes": list(writes)}}]}


def test_native_continuation_inserts_exactly_one_reset_between_own_states():
    edges = {"selected": edge("selected", "reset-parent"), "reset-result": edge("reset-result", "scheduler3"),
             "scheduler3": edge("scheduler3", "final")}
    reset = {"parent": "reset-parent", "result": "reset-result"}
    assert AUDIT["native_path"]("selected", "final", edges, reset=reset) == [
        "selected.receipt", "reset-result.receipt", "scheduler3.receipt"]
    with pytest.raises(ValueError, match="broken"):
        AUDIT["native_path"]("selected", "final", edges)
    with pytest.raises(ValueError, match="omitted"):
        AUDIT["native_path"]("scheduler3", "final", edges, reset=reset)


@pytest.mark.parametrize("damage", ["cycle", "conditioning-write", "truncation"])
def test_native_path_rejects_corrupt_lineage_not_scientific_effects(damage):
    edges = {"first": edge("first", "last"), "last": edge("last", "final")}
    if damage == "cycle":
        edges["last"]["result"] = "first"
    elif damage == "conditioning-write":
        edges["last"]["transitions"][0]["native_operation"]["writes"].append("conditioning")
    else:
        del edges["last"]
    with pytest.raises(ValueError):
        AUDIT["native_path"]("first", "final", edges)


def descriptor(name):
    return {"kind": "tensor", "shape": [1, 2], "dtype": "torch.bfloat16", "sha256": name, "stride": [2, 1]}


def encoded(values):
    return {"kind": "mapping", "items": [{"key": {"kind": "scalar", "value": key}, "value": value} for key, value in values.items()]}


def replacement():
    target = descriptor("base-conditioning")
    before = {"conditioning": descriptor("changed-conditioning"), "text": descriptor("native-text"), "latent": descriptor("own-latent")}
    after = {**before, "conditioning": target}
    act = {"operation": {"id": "saturn.builtin.replace"}, "writes": ["conditioning"], "parameters": encoded({"value": target})}
    receipt = seal({"schema": "saturn-pub-receipt-v1", "parent": "before", "result": "after", "operation": "apply",
                    "act": act, "writes": encoded({"conditioning": target})})
    action = {"parent": "before", "post_act": "after", "act": act, "receipt": receipt}
    return target, {"before": before, "after": after}, action


def test_conditioning_reset_preserves_own_latent_and_text_with_exact_payload_authority():
    target, cuts, action = replacement()
    check = lambda: AUDIT["replacement_closure"](action, cuts.__getitem__, slot="conditioning", expected=target,
        stored_receipt=action["receipt"], unpack=TYPED["unpack"], public_receipt=PUBLIC)
    assert check()["fingerprint"] == action["receipt"]["fingerprint"]
    cuts["after"]["latent"] = descriptor("donor-latent")
    with pytest.raises(ValueError, match="undeclared"):
        check()


def test_self_consistent_reset_receipt_cannot_substitute_unbound_conditioning():
    target, cuts, action = replacement()
    wrong = descriptor("different-donor")
    action = copy.deepcopy(action)
    action["act"]["parameters"] = encoded({"value": wrong})
    action["receipt"] = seal({**action["receipt"], "act": action["act"], "writes": encoded({"conditioning": wrong})})
    with pytest.raises(ValueError, match="payload"):
        AUDIT["replacement_closure"](action, cuts.__getitem__, slot="conditioning", expected=target,
            stored_receipt=action["receipt"], unpack=TYPED["unpack"], public_receipt=PUBLIC)
