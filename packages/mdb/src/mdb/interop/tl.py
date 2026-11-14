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
from .clone import shared_clone


def version() -> str:
    from importlib.metadata import version as _v

    return _v("transformer-lens")


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


def _bridge_of(bridge_or_cache: Any) -> Any:
    """The bridge behind a bridge or an ``ActivationCache`` (whose ``.model`` is the bridge).

    Never ``getattr(bridge, "model")``: ``TransformerBridge.__getattr__`` falls through to the
    wrapped Hugging Face module, so ``bridge.model`` is the HF ``Qwen3Model``, which has no
    ``_weights_processed`` (the old ``processed()`` read it there and always said False)."""
    if "_weights_processed" in getattr(bridge_or_cache, "__dict__", {}):
        return bridge_or_cache
    inner = getattr(bridge_or_cache, "__dict__", {}).get("model")
    if inner is not None and "_weights_processed" in getattr(inner, "__dict__", {}):
        return inner
    return None


def processed(bridge_or_cache: Any) -> bool:
    """True when TransformerLens weight processing ran on the bridge (``process_weights``, called
    by ``enable_compatibility_mode()`` unless ``no_processing=True``). TL sets
    ``self._weights_processed`` on the bridge itself (``bridge_core.py``, TL 4.4.0)."""
    found = _bridge_of(bridge_or_cache)
    return bool(found is not None and found.__dict__["_weights_processed"])


def processing(bridge_or_cache: Any) -> dict[str, Any]:
    """What TL weight processing did to this bridge, as far as TL 4.4.0's own gates decide it
    (``ProcessWeights.process_weights``, ``weight_processing.py``): ``fold_ln`` folds the norm
    gains into the next linear layers for LN and RMS; centring of the writing weights (and the
    centred-output value-bias fold) runs only for ``normalization_type`` LN/LNPre without
    ``final_rms``, so on RMSNorm models (Qwen2/Qwen3, Llama) the residual stream is NOT centred.
    Assumes the default flags of ``enable_compatibility_mode``; TL does not record the flags."""
    found = _bridge_of(bridge_or_cache)
    cfg = getattr(found, "cfg", None)
    norm = getattr(cfg, "normalization_type", None)
    final_rms = bool(getattr(cfg, "final_rms", False))
    done = processed(bridge_or_cache)
    return {"processed": done, "normalization_type": norm, "final_rms": final_rms,
            "residual_centred": bool(done and norm in ("LN", "LNPre") and not final_rms),
            "norm_gains_folded": bool(done and norm in ("LN", "LNPre", "RMS", "RMSPre"))}


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
    state = processing(bridge_model if bridge_model is not None else cache)
    if _bridge_of(bridge_model if bridge_model is not None else cache) is None:
        # a plain {hook: tensor} dict carries no bridge: say so instead of claiming "unprocessed"
        state["processed"] = None
    provenance = {"tool": "transformer_lens", "version": version(), "processed": state["processed"],
                  "processing": state,
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
