"""A pure-Python debugger demo with real, isolated in-memory state cuts."""

import json
from copy import deepcopy
from hashlib import sha256

from .debugger_adapters import AdapterCapabilities, CallbackDebuggerAdapter
from .debugger_protocol import DebuggerProtocolError


def _ref(body):
    return "counter://" + sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class CounterDebuggerAdapter(CallbackDebuggerAdapter):
    """Toy integer machine; no weights, model libraries or durable model ABI."""

    def __init__(self):
        self.active = "main"
        self.branches = {"main": {"value": 0, "cursor": 0, "cut": None}}
        self.cuts = {}
        self.pending = None
        caps = AdapterCapabilities(
            family="counter",
            boundaries=("step",),
            inspectable_slots=("value", "cursor", "memory"),
            writable_slots=("value",),
            mutable_closure=("value", "cursor"),
            exact_replay=True,
            state_schema_id="mdb-counter-memory-v1",
            backend="python",
            consumer="counter-plus-one",
            checkpoint_requires_pause=True,
        )
        super().__init__(
            caps,
            status=self.status,
            inspect=self.inspect,
            resolve_value=self.resolve_value,
            checkpoint=self.checkpoint,
            restore=self.restore,
            preview_act=self.preview_act,
            commit_act=self.commit_act,
            abort_act=self.abort_act,
            replay=self.replay,
            step=self._advance,
            continue_run=lambda: {"state": "running"},
            fork_branch=self.fork,
        )
        self.checkpoint("initial")

    @property
    def state(self):
        return self.branches[self.active]

    def _clear(self):
        if self.pending is not None:
            raise DebuggerProtocolError("commit or abort the pending Act first")

    def status(self):
        return {
            "branch": self.active,
            "cut_id": self.state["cut"],
            "cursor": {"phase": "counter", "step": self.state["cursor"]},
            "value": self.state["value"],
            "storage": "in-memory-demo",
        }

    def inspect(self, path=None):
        if path is None:
            return self.status()
        return {"path": path, "value": self.resolve_value(path)}

    def resolve_value(self, path):
        if path == "value":
            return self.state["value"]
        if path == "cursor":
            return self.status()["cursor"]
        if path == "memory":
            return {"branches": len(self.branches), "cuts": len(self.cuts), "storage": "in-memory"}
        raise DebuggerProtocolError(f"unknown Counter address: {path}")

    def checkpoint(self, name=None):
        self._clear()
        body = {
            "schema": "mdb-counter-cut-v1",
            "branch": self.active,
            "value": self.state["value"],
            "cursor": self.state["cursor"],
            "parent_cut": self.state["cut"],
            "name": name,
        }
        ref = _ref(body)
        self.cuts[ref] = deepcopy(body)
        self.state["cut"] = ref
        return {
            "cut_id": ref,
            "parent_cut": body["parent_cut"],
            "name": name,
            "branch": self.active,
            "cursor": self.status()["cursor"],
        }

    def read_cut(self, ref):
        if ref not in self.cuts or _ref(self.cuts[ref]) != ref:
            raise DebuggerProtocolError("unknown or changed Counter cut")
        return deepcopy(self.cuts[ref])

    def restore(self, ref):
        self._clear()
        body = self.read_cut(ref)
        self.state.update(value=body["value"], cursor=body["cursor"], cut=ref)
        return {**self.status(), "parent_cut": body["parent_cut"], "exact": True}

    def replay(self, ref):
        return {**self.restore(ref), "replay_scope": "exact-counter-state-restoration"}

    def _advance(self, count=1):
        self._clear()
        if type(count) is not int or count <= 0:
            raise DebuggerProtocolError("step count must be positive")
        self.state["value"] += count
        self.state["cursor"] += count
        return {**self.checkpoint("step"), "steps": count}

    def fork(self, name):
        self._clear()
        if not isinstance(name, str) or not name or name in self.branches:
            raise DebuggerProtocolError("fork requires a new branch name")
        self.branches[name] = deepcopy(self.state)
        self.active = name
        return self.checkpoint(name)

    def preview_act(self, act):
        self._clear()
        self.validate_write(act)
        if act.parent_cut != self.state["cut"] or act.path != "value":
            raise DebuggerProtocolError("Act parent or Counter address changed")
        if type(act.value) is not int:
            raise DebuggerProtocolError("Counter edits require an integer")
        if act.sign == "replace":
            value = act.value
        elif act.sign == "add":
            value = self.state["value"] + act.value
        elif act.sign == "scale":
            value = self.state["value"] * act.value
        else:
            raise DebuggerProtocolError("unsupported Counter edit")
        self.pending = (act.act_id, value, act.parent_cut)
        return {"act_id": act.act_id, "preview": True, "candidate_value": value}

    def commit_act(self, ref):
        if self.pending is None or self.pending[0] != ref:
            raise DebuggerProtocolError("pending Counter Act changed")
        _, value, _ = self.pending
        self.pending = None
        self.state["value"] = value
        return {**self.checkpoint("act-committed"), "act_id": ref, "committed": True}

    def abort_act(self, ref):
        if self.pending is None or self.pending[0] != ref:
            raise DebuggerProtocolError("pending Counter Act changed")
        _, _, parent = self.pending
        self.pending = None
        return {"act_id": ref, "aborted": True, "cut_id": parent}

    def optional_operations(self):
        return (*super().optional_operations(), "fork_branch")
