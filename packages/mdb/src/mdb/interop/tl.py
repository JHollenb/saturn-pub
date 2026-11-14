"""TransformerLens (v4 ``TransformerBridge``) <-> mdb values.

- ``bridge(model, tokenizer)``: a bridge that wraps the SAME Hugging Face module mdb attaches to
  (unprocessed: values are the native ones, bitwise; measured in ``tests/test_interop.py``).
- ``cache_values(cache, rows, ...)``: an ``ActivationCache`` (or any ``{hook: tensor}``) as mdb
  occurrence values ``t<r>.L<l>.<port>`` for ``Debugger.import_values``.
- ``import_cache(dbg, name, cache, tokens)``: the same, registered as the origin ``name``.
- ``trace_cache(trace)``: an ``mdb.trace`` capture as ``{TL hook name: tensor}`` (and as a TL
  ``ActivationCache`` when a bridge is given).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from . import names


def version() -> str:
    from importlib.metadata import version as _v

    return _v("transformer-lens")


def shared_clone(model: Any) -> Any:
    """A new module tree over the SAME parameter and buffer tensors (no weight copy).

    ``boot_transformers(hf_model=...)`` rewrites the module tree it is given in place
    (measured: 33 -> 155 modules, every decoder layer becomes a ``BlockBridge``, attention a
    ``PositionEmbeddingsAttentionBridge``; ``mdb.trace`` then refuses ``p`` because native eager
    attention no longer runs). Bridging a shared clone leaves mdb's module tree untouched while
    both read one copy of the weights."""
    import copy

    memo = {id(t): t for t in list(model.parameters()) + list(model.buffers())}
    return copy.deepcopy(model, memo)


def bridge(model: Any, tokenizer: Any, *, name: str | None = None,
           in_place: bool = False) -> Any:
    """``TransformerBridge.boot_transformers`` over ``model``'s weights, unprocessed.

    By default the bridge wraps ``shared_clone(model)``: same tensors, separate module tree, so
    mdb can stay attached to ``model``. ``in_place=True`` hands TL the module itself (TL then
    rewrites its tree). Never call ``enable_compatibility_mode()`` on a bridge that shares
    weights with a model in use: weight processing writes the shared parameters."""
    from transformer_lens.model_bridge import TransformerBridge

    param = next(model.parameters())
    target = model if in_place else shared_clone(model)
    return TransformerBridge.boot_transformers(
        name or getattr(model.config, "_name_or_path", None) or "hf-model", hf_model=target,
        tokenizer=tokenizer, device=param.device, dtype=param.dtype)


def processed(bridge_or_cache: Any) -> bool:
    """True when TransformerLens weight processing ran (fold_ln / centring / bias folding)."""
    model = getattr(bridge_or_cache, "model", bridge_or_cache)
    return bool(getattr(model, "_weights_processed", False))


def cache_values(cache: Any, rows: Iterable[int], *, layers: Iterable[int] | None = None,
                 ports: Sequence[str] | None = None, batch: int = 0) -> dict[str, Any]:
    """``{t<r>.L<l>.<port>: tensor}`` from a TL cache (bridge-native names or aliases), one
    batch element, mdb's single-row layout. Hooks with no mdb port are skipped."""
    out: dict[str, Any] = {}
    rows = list(rows)
    keys = list(cache.keys()) if hasattr(cache, "keys") else list(cache)
    for hook in keys:
        try:
            address = names.from_tl(hook)
        except names.NoName:
            continue
        if "." not in address:
            continue
        layer, port = address[1:].split(".", 1)
        layer = int(layer)
        if layers is not None and layer not in set(layers):
            continue
        if ports is not None and port not in ports:
            continue
        spec = names.BY_PORT[port]
        value = cache[hook][batch:batch + 1]
        for row in rows:
            key = f"t{row}.L{layer}.{port}"
            if key not in out:
                out[key] = names.to_mdb(spec.tl_layout, value, row).detach().clone()
    return out


def import_cache(dbg: Any, name: str, cache: Any, tokens: Sequence[int], *,
                 rows: Iterable[int] | None = None, layers: Iterable[int] | None = None,
                 ports: Sequence[str] | None = None, batch: int = 0,
                 bridge_model: Any = None) -> Any:
    """Register a TL cache as origin ``name`` of ``dbg`` (``Debugger.import_values``)."""
    rows = range(len(tokens)) if rows is None else rows
    values = cache_values(cache, rows, layers=layers, ports=ports, batch=batch)
    is_processed = processed(bridge_model) if bridge_model is not None else processed(cache)
    provenance = {"tool": "transformer_lens", "version": version(), "processed": is_processed,
                  "program": "TransformerBridge full forward", "batch_index": batch}
    return dbg.import_values(name, values, tokens=tokens, provenance=provenance)


def trace_cache(trace: Any, *, head_dim: int, call: int | None = None,
                bridge_model: Any = None) -> Any:
    """An ``mdb.trace`` capture as TL hook names (bridge-native), TL layouts. With
    ``bridge_model`` the result is a ``transformer_lens.ActivationCache``."""
    out: dict[str, Any] = {}
    for address in trace.ports:
        layer, port = names._split(address)
        if layer is None or port not in names.BY_PORT:
            continue
        spec = names.BY_PORT[port]
        if spec.tl is None:
            continue
        if trace.info(address, call)["state"] != "resolved":
            continue
        value = trace[address] if call is None else trace[address, call]
        out[spec.tl.format(l=layer)] = names.to_tool(spec.tl_layout, value, head_dim=head_dim)
    if bridge_model is None:
        return out
    from transformer_lens import ActivationCache

    return ActivationCache(out, bridge_model)
