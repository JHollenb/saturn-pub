"""Read-only observation of native Qwen eager attention, including Bench calls.

The explicit context is process-local and single-threaded. It calls the same
native function once and returns the original objects unchanged. No tensor is
edited. Passive parity must be qualified in the actual execution lane.

`mdb.attention_port` subclasses this observer for its Qwen backend through the
`_intercept` hook; the default hook keeps the original read-only behaviour.
"""
from __future__ import annotations


class AttentionObserver:
    def __init__(self, model, module=None):
        self.model = model
        self.rows = []
        self.handles = []
        self.active = False
        self._modeling_module = module

    def _intercept(self, layer, attention, query, key, value, args, kwargs):
        """Called once per native attention call of a decoder layer of `model`."""
        result = self.original(attention, query, key, value, *args, **kwargs)
        self.rows.append({"layer": layer,
            "query": query.detach().clone(), "key": key.detach().clone(),
            "value": value.detach().clone(), "attention": result[1].detach().clone(),
            "head_return": result[0].detach().clone()})
        return result

    def __enter__(self):
        if self._modeling_module is None:
            from transformers.models.qwen2 import modeling_qwen2
            self._modeling_module = modeling_qwen2
        if getattr(self._modeling_module.eager_attention_forward, "_mdb_attention_observer", False):
            raise ValueError("nested eager-attention observers are unsupported")
        self.module = self._modeling_module
        self.original = self.module.eager_attention_forward
        modules = {id(layer.self_attn): i for i, layer in enumerate(self.model.model.layers)}

        def observed(attention, query, key, value, *args, **kwargs):
            if id(attention) in modules:
                return self._intercept(modules[id(attention)], attention, query, key, value, args, kwargs)
            return self.original(attention, query, key, value, *args, **kwargs)

        observed._mdb_attention_observer = True
        self.module.eager_attention_forward = observed
        self.active = True
        return self

    def __exit__(self, *args):
        if self.active:
            self.module.eager_attention_forward = self.original
            self.active = False
