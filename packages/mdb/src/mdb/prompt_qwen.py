"""Opt-in Qwen prompt ingestion and KV ports over the public native adapter.

The inherited numerical program is unchanged. The extended write contract has
its own execution identity; legacy public adapter cuts remain incompatible.
"""
from dataclasses import replace

from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.core import Act
from saturn_pub.values import clone


class PromptQwenAdapter(QwenAdapter):
    def __init__(self, model):
        super().__init__(model, granularity="operation")
        self.execution = {**self.execution, "mdb_ports": "prompt-input-kv-v1"}

    def addresses(self, state):
        return (*super().addresses(state), "keys", "values")

    def surface(self, state):
        surface = super().surface(state)
        return replace(surface, state_schema="mdb-qwen-prompt-kv-state-v1",
            slots=tuple(replace(slot, writable=True) if slot.name in {"keys", "values"}
                or (slot.name == "tokens" and state["phase"] == "embed") else slot
                for slot in surface.slots))

    def write(self, state, writes, invalidates):
        if "tokens" not in writes:
            return super().write(state, writes, invalidates)
        if set(writes) != {"tokens"} or invalidates:
            raise ValueError("prompt ingestion writes only the pending token")
        old, new = state["tokens"], writes["tokens"]
        if (state["phase"] != "embed" or state["hidden"] is not None
                or old.shape != new.shape or old.dtype != new.dtype
                or not old[:, :-1].equal(new[:, :-1])):
            raise ValueError("prompt ingestion requires an unconsumed final token and fixed history")
        result = clone(state)
        result["tokens"] = clone(new)
        self.validate(result)
        return result

    def stage_prompt_token(self, session, token):
        tokens = session.read("tokens")
        tokens[:, -1] = token
        return session.apply(Act.replace("tokens", tokens))
