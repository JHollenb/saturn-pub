"""Read-only observation of native Qwen eager attention, including Bench calls.

The explicit context is process-local and single-threaded. It calls the same
native function once and returns the original objects unchanged. No tensor is
edited. Passive parity must be qualified in the actual execution lane.
"""
from __future__ import annotations


class AttentionObserver:
    def __init__(self, model):
        self.model = model
        self.rows = []
        self.handles = []
        self.active = False

    def __enter__(self):
        from transformers.models.qwen2 import modeling_qwen2
        if getattr(modeling_qwen2.eager_attention_forward, "_mdb_attention_observer", False):
            raise ValueError("nested eager-attention observers are unsupported")
        self.module = modeling_qwen2
        self.original = self.module.eager_attention_forward
        modules = {id(layer.self_attn): i for i, layer in enumerate(self.model.model.layers)}

        def observed(attention, query, key, value, *args, **kwargs):
            result = self.original(attention, query, key, value, *args, **kwargs)
            if id(attention) in modules:
                self.rows.append({"layer": modules[id(attention)],
                    "query": query.detach().clone(), "key": key.detach().clone(),
                    "value": value.detach().clone(), "attention": result[1].detach().clone(),
                    "head_return": result[0].detach().clone()})
            return result

        observed._mdb_attention_observer = True
        self.module.eager_attention_forward = observed
        self.active = True
        return self

    def __exit__(self, *args):
        if self.active:
            self.module.eager_attention_forward = self.original
            self.active = False
