"""A module tree over the same tensors, for tools that rewrite the tree they are handed."""

from __future__ import annotations

from typing import Any


def shared_clone(model: Any) -> Any:
    """A new module tree over the SAME parameter and buffer tensors (no weight copy).

    Two tools rewrite the module tree they are given, measured on tiny Qwen2 (tests):
    ``TransformerBridge.boot_transformers(hf_model=...)`` replaces modules in place (33 -> 155
    modules; layers become ``BlockBridge``, attention a ``PositionEmbeddingsAttentionBridge``),
    and nnsight's ``.source`` replaces an attention module's ``forward`` with an instrumented
    copy that binds ``eager_attention_forward`` at rewrite time. After either, mdb's eager
    interceptor no longer sees that layer's attention. Handing the tool a shared clone leaves
    mdb's tree untouched while both read one copy of the weights.
    """
    import copy

    memo = {id(t): t for t in list(model.parameters()) + list(model.buffers())}
    return copy.deepcopy(model, memo)
