"""Bounded native Qwen regeneration from sealed metadata, never activation roots.

The caller supplies the already-qualified native adapter and original model.
This module opens no artifact files, fits no predictor and exports no tensors.
Expected cuts are checked after execution and never install native state.
"""
from __future__ import annotations
import hashlib
import json

SCHEMA = "mdb-native-qwen-replay-recipe-v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seal(body):
    value = json.loads(json.dumps(body, allow_nan=False))
    if "fingerprint" in value:
        raise ValueError("recipe is already sealed")
    return {**value, "fingerprint": fingerprint(value)}


def validate(recipe):
    value = json.loads(json.dumps(recipe, allow_nan=False))
    expected = value.pop("fingerprint", None)
    if fingerprint(value) != expected:
        raise ValueError("recipe seal differs")
    if set(value) != {"schema", "model_identity", "execution", "roots", "events"} or value["schema"] != SCHEMA:
        raise ValueError("unsupported recipe fields or schema")
    if not isinstance(value["model_identity"], str) or not isinstance(value["execution"], dict):
        raise ValueError("recipe needs native model and execution identities")
    roots = value["roots"]
    if not isinstance(roots, list) or not 1 <= len(roots) <= 8:
        raise ValueError("root count outside bounded support")
    sessions = set()
    for root in roots:
        if set(root) != {"id", "token_ids"} or not isinstance(root["id"], str) or root["id"] in sessions:
            raise ValueError("roots require unique identities and input tokens only")
        tokens = root["token_ids"]
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 128 or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("invalid root token history")
        sessions.add(root["id"])
    events = value["events"]
    if not isinstance(events, list) or not 1 <= len(events) <= 256:
        raise ValueError("event count outside bounded support")
    known, steps = set(), 0
    for event in events:
        required = {"id", "session", "kind"}
        if not required <= set(event) or event["session"] not in sessions or event["id"] in known:
            raise ValueError("invalid or duplicate event identity")
        kind = event["kind"]
        extra = {"continue": {"steps"}, "prompt-token": {"token_id"}, "restore": {"source"},
                 "copy-values": {"source", "cells", "target_row", "source_row"}, "mark": set()}.get(kind)
        if extra is None or set(event) - required - extra - {"expected_cut"} or not extra <= set(event):
            raise ValueError("unsupported event fields or operation")
        if "expected_cut" in event and (not isinstance(event["expected_cut"], str) or len(event["expected_cut"]) != 64):
            raise ValueError("invalid comparison hash")
        if kind == "continue":
            if type(event["steps"]) is not int or not 1 <= event["steps"] <= 128:
                raise ValueError("invalid native step count")
            steps += event["steps"]
        if kind == "prompt-token" and (type(event["token_id"]) is not int or event["token_id"] < 0):
            raise ValueError("invalid prompt token")
        if kind in {"restore", "copy-values"} and event["source"] not in known:
            raise ValueError("reference must resolve to an earlier regenerated occurrence")
        if kind == "copy-values":
            cells = event["cells"]
            if (not isinstance(cells, list) or not 1 <= len(cells) <= 56
                    or any(not isinstance(c, list) or len(c) != 2 or any(type(x) is not int or x < 0 for x in c) for c in cells)
                    or len({tuple(c) for c in cells}) != len(cells)
                    or any(type(event[k]) is not int or event[k] < 0 for k in ("target_row", "source_row"))):
                raise ValueError("invalid native value selectors")
        known.add(event["id"])
    if steps > 2500:
        raise ValueError("native replay exceeds bounded step budget")
    return {**value, "fingerprint": expected}


class NativeReplay:
    """Materialize supported occurrences in memory through the native Session.

    `observe` is a caller-owned context factory for passive recapture. `inspect`
    receives the completed event and observation for comparison or inspection.
    Their results are never used as replay inputs. This is trusted Python, not
    an adversarial sandbox. References can be read through `materialize`.
    """
    def __init__(self, recipe, adapter):
        self.recipe = validate(recipe)
        if (adapter.model_identity != self.recipe["model_identity"]
                or dict(adapter.execution) != self.recipe["execution"]):
            raise ValueError("incompatible native model or numerical execution")
        self.adapter = adapter
        self.sessions, self.cuts, self.rows = {}, {}, []

    def run(self, *, observe=None, inspect=None):
        from contextlib import nullcontext
        from saturn_pub.core import Act
        if self.sessions:
            raise ValueError("use a fresh replay object for independent reconstruction")
        self.sessions = {root["id"]: self.adapter.session(root["token_ids"]) for root in self.recipe["roots"]}
        for event in self.recipe["events"]:
            session = self.sessions[event["session"]]
            kind, receipt = event["kind"], None
            with observe(event) if observe else nullcontext() as observation:
                if kind == "continue":
                    receipt = session.continue_(event["steps"])
                elif kind == "prompt-token":
                    receipt = self.adapter.stage_prompt_token(session, event["token_id"])
                elif kind == "restore":
                    receipt = session.restore(self.cuts[event["source"]])
                elif kind == "copy-values":
                    current = session.read("values")
                    donor = self.cuts[event["source"]].payload["values"]
                    target_row, source_row = event["target_row"], event["source_row"]
                    for layer, kv in event["cells"]:
                        a, b = current[layer], donor[layer]
                        if (a.shape[0] != b.shape[0] or a.shape[-1] != b.shape[-1]
                                or kv >= min(a.shape[1], b.shape[1])
                                or target_row >= a.shape[2] or source_row >= b.shape[2]):
                            raise ValueError("value-copy selector outside native cache")
                        a[:, kv, target_row] = b[:, kv, source_row]
                    receipt = session.apply(Act.replace("values", current))
            cut = session.capture(retain=False)
            if event.get("expected_cut", cut.fingerprint) != cut.fingerprint:
                raise ValueError("regenerated native closure differs at " + event["id"])
            self.cuts[event["id"]] = cut
            self.rows.append({"event": event["id"], "cut": cut.fingerprint,
                "receipt": receipt.fingerprint if receipt else None, "kind": kind,
                "execution_point": dict(cut.execution_point)})
            if inspect:
                inspect(event, session, observation)
        return self

    def materialize(self, occurrence, address):
        """Read a qualified public StateAddress at a regenerated occurrence."""
        from saturn_pub.core import Session
        if occurrence not in self.cuts:
            raise ValueError("occurrence has not been regenerated")
        return Session.from_cut(self.adapter, self.cuts[occurrence]).read_port(address)
