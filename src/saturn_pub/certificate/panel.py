"""Run the relational-induction causal battery on a native decoder adapter.

This produces the per-example forced-choice outcomes that
:func:`saturn_pub.certificate.gate.certify_induction_route` consumes. Every branch
terminates at the adapter's *own* native lexical consumer -- the resident model's final
normalization and output embedding (``model(...).logits``) -- so the certificate's
``native_consumer_continuation`` gate is about the real model, never a surrogate readout.

Five branches per panel, over the complete direct-source parent (every attention layer
and query head), mirroring the owner's private assay:

* ``clean`` -- the untouched forward.
* ``source_deletion`` -- delete the final query's attention edge to the earlier answer
  position at every head/layer (the necessity cut).
* ``match_deletion`` -- the same-sized deletion at the neighbor cue position (control).
* ``correct_repair`` -- re-run under the full source deletion, then restore every head's
  clean pre-output-projection activation at the query (sufficiency; with the full parent
  this restores the clean decision exactly).
* ``wrong_repair`` -- the same-shaped repair taken from a different-answer donor row
  (content specificity).

The attention-edge deletion and pre-``o_proj`` repair are finer than the decoder
adapter's declared Session state ports, so they are applied as explicit forward-hook
contexts on the adapter's resident native model -- the same frozen model the Session
executes. A Session/fork consumer cross-check confirms the stepped native consumer
reproduces the batched clean decision on a held example.

Supported attention families are the llama-style decoders whose per-head output
projection is a contiguous ``self_attn.o_proj`` slice: ``llama``, ``mistral``,
``mixtral``, ``gemma``, ``qwen2``, ``qwen3``. Other registered decoder families
(``gpt2``, ``gpt_neox``, ``phi``) use a different attention module layout and are
refused fail-closed rather than silently mismeasured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from .gate import (
    INDUCTION_PANEL_SCHEMA,
    NATIVE_CONSUMER,
    InductionCircuitPolicy,
    certify_induction_route,
)
from .workload import (
    InductionPanelSpec,
    build_relational_induction_workload,
    wrong_donor_indices,
)

_SUPPORTED_ATTENTION_FAMILIES = frozenset(
    {"llama", "mistral", "mixtral", "gemma", "qwen2", "qwen3"}
)


def _require_supported(adapter: Any) -> str:
    family = getattr(getattr(adapter, "spec", None), "model_type", None)
    if family not in _SUPPORTED_ATTENTION_FAMILIES:
        raise ValueError(
            f"induction panel supports llama-style attention families "
            f"{sorted(_SUPPORTED_ATTENTION_FAMILIES)}; got {family!r}. The owner's assay "
            f"deletes a final-query attention edge and repairs the pre-o_proj head output, "
            f"which requires a self_attn.o_proj with contiguous per-head slices."
        )
    return family


def _layer_modules(adapter: Any) -> Sequence[Any]:
    return getattr(adapter.backbone, adapter.spec.layers)


def _attention(adapter: Any, layer: int) -> Any:
    return _layer_modules(adapter)[layer].self_attn


def _out_proj(adapter: Any, layer: int) -> Any:
    return _layer_modules(adapter)[layer].self_attn.o_proj


def _head_slice(adapter: Any, head: int) -> slice:
    return slice(head * adapter.head_dim, (head + 1) * adapter.head_dim)


def _batch_tensors(
    items: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(item["tokens"]) for item in items)
    ids = torch.zeros(len(items), max_length, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, item in enumerate(items):
        tokens = torch.as_tensor(item["tokens"], dtype=torch.long, device=device)
        ids[row, : len(tokens)] = tokens
        mask[row, : len(tokens)] = 1
    return ids, mask, mask.sum(1) - 1


def _score(
    logits: torch.Tensor, items: Sequence[Mapping[str, Any]], last: torch.Tensor
) -> tuple[list[float], list[float]]:
    device = logits.device
    candidates = torch.as_tensor([item["cands"] for item in items], dtype=torch.long, device=device)
    labels = torch.as_tensor([item["label"] for item in items], dtype=torch.long, device=device)
    rows = torch.arange(len(items), device=device)
    scores = logits[rows, last].gather(1, candidates).float()
    selected = scores[rows, labels]
    masked = scores.clone()
    masked[rows, labels] = -torch.inf
    correct = (scores.argmax(1) == labels).float().cpu().tolist()
    margin = (selected - masked.max(1).values).cpu().tolist()
    return correct, margin


@contextmanager
def _source_deletion(
    adapter: Any,
    items: Sequence[Mapping[str, Any]],
    last: torch.Tensor,
    position_key: str,
    heads: Sequence[int],
) -> Any:
    """Zero the final query's attention to ``position_key`` at the declared heads."""

    handles = []
    head_list = list(heads)
    for layer in range(adapter.layers):

        def pre(
            module: Any, args: tuple, kwargs: dict, *, _items=items, _last=last, _key=position_key
        ):
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None:
                raise RuntimeError("eager attention did not expose attention_mask as a keyword")
            expanded = attention_mask.expand(
                len(_items), adapter.num_heads, attention_mask.shape[2], attention_mask.shape[3]
            ).clone()
            minimum = torch.finfo(expanded.dtype).min
            for row, item in enumerate(_items):
                expanded[row, head_list, int(_last[row]), int(item[_key])] = minimum
            kwargs["attention_mask"] = expanded
            return args, kwargs

        handles.append(_attention(adapter, layer).register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _capture_context(adapter: Any, store: dict[int, torch.Tensor]) -> Any:
    """Capture the clean pre-``o_proj`` activation at every layer (read-only)."""

    handles = []
    for layer in range(adapter.layers):

        def pre(module: Any, args: tuple, *, _layer=layer):
            store[_layer] = args[0].detach().clone()

        handles.append(_out_proj(adapter, layer).register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _repair_context(
    adapter: Any,
    context: Mapping[int, torch.Tensor],
    last: torch.Tensor,
    heads: Sequence[int],
    donor_rows: torch.Tensor | None,
) -> Any:
    """Restore the declared heads' clean (or donor) pre-``o_proj`` output at the query."""

    handles = []
    head_list = list(heads)
    rows = torch.arange(last.shape[0], device=last.device)
    source_rows = rows if donor_rows is None else donor_rows
    for layer in range(adapter.layers):

        def pre(module: Any, args: tuple, *, _layer=layer):
            changed = args[0].clone()
            clean = context[_layer]
            for head in head_list:
                segment = _head_slice(adapter, head)
                changed[rows, last, segment] = clean[source_rows, last[source_rows], segment].to(
                    changed.dtype
                )
            return (changed, *args[1:])

        handles.append(_out_proj(adapter, layer).register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _forward(adapter: Any, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return adapter.model(input_ids=ids, attention_mask=mask, use_cache=False).logits


def _full_family(adapter: Any) -> dict[int, tuple[int, ...]]:
    heads = tuple(range(adapter.num_heads))
    return {layer: heads for layer in range(adapter.layers)}


def _session_consumer_crosscheck(
    adapter: Any, item: Mapping[str, Any], clean_correct_first: float
) -> dict[str, Any]:
    """Drive the native consumer through a Session fork and confirm it agrees.

    Demonstrates that the stepped ``Session``/``fork`` native suffix (embed -> layers ->
    final norm -> lm head) reproduces the batched clean decision used for the panel.
    """

    tokens = [int(token) for token in item["tokens"]]
    probe = adapter.session(tokens)
    branch = probe.fork()
    branch.continue_(adapter.layers + 2)
    logits = branch.read("logits")[0]
    candidates = torch.as_tensor(item["cands"], dtype=torch.long, device=logits.device)
    scores = logits.index_select(0, candidates).float()
    predicted = int(scores.argmax().item())
    session_correct = float(predicted == int(item["label"]))
    return {
        "example_query_position": int(item["query"]),
        "session_candidate_argmax_label": predicted,
        "session_correct": session_correct,
        "batched_clean_correct": float(clean_correct_first),
        "agrees": session_correct == float(clean_correct_first),
    }


def measure_induction_panel(
    adapter: Any,
    spec: InductionPanelSpec,
    *,
    consumer: str = NATIVE_CONSUMER,
    batch_size: int = 32,
    family: Mapping[int, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Measure one sealed induction panel on a native decoder adapter.

    Returns a panel dict in the schema :data:`saturn_pub.certificate.gate.INDUCTION_PANEL_SCHEMA`
    ready for :func:`certify_induction_route`.
    """

    _require_supported(adapter)
    adapter.validate_execution()
    device = adapter.device
    universe = (
        {int(layer): tuple(sorted({int(h) for h in heads})) for layer, heads in family.items()}
        if family is not None
        else _full_family(adapter)
    )
    heads_by_layer = universe  # deletions/repairs use the same declared family per layer
    all_heads = tuple(range(adapter.num_heads))
    # The certified parent is every head; a per-layer head list is honored if supplied.

    items, classes = build_relational_induction_workload(spec)
    if max(max(item["cands"]) for item in items) >= int(adapter.vocab_size):
        raise ValueError("workload candidate token lies outside the model vocabulary")

    branches = {
        name: {"correct": [], "margin": []}
        for name in ("clean", "source_deletion", "match_deletion", "correct_repair", "wrong_repair")
    }
    first_clean_correct: float | None = None

    for start in range(0, len(items), batch_size):
        chosen = items[start : start + batch_size]
        ids, mask, last = _batch_tensors(chosen, device)
        heads = all_heads  # delete/repair across every head of the full parent

        # clean + capture the clean pre-o_proj context in one forward (hooks read-only)
        context: dict[int, torch.Tensor] = {}
        with _capture_context(adapter, context):
            clean_logits = _forward(adapter, ids, mask)
        clean_correct, clean_margin = _score(clean_logits, chosen, last)
        if first_clean_correct is None:
            first_clean_correct = clean_correct[0]

        with _source_deletion(adapter, chosen, last, "source", heads):
            source_logits = _forward(adapter, ids, mask)
        source_correct, source_margin = _score(source_logits, chosen, last)

        with _source_deletion(adapter, chosen, last, "match", heads):
            match_logits = _forward(adapter, ids, mask)
        match_correct, match_margin = _score(match_logits, chosen, last)

        donor = torch.as_tensor(
            wrong_donor_indices([item["label"] for item in chosen]), dtype=torch.long, device=device
        )
        with _source_deletion(adapter, chosen, last, "source", heads):
            with _repair_context(adapter, context, last, heads, None):
                repair_logits = _forward(adapter, ids, mask)
        repair_correct, repair_margin = _score(repair_logits, chosen, last)

        with _source_deletion(adapter, chosen, last, "source", heads):
            with _repair_context(adapter, context, last, heads, donor):
                wrong_logits = _forward(adapter, ids, mask)
        wrong_correct, wrong_margin = _score(wrong_logits, chosen, last)

        for name, (correct, margin) in (
            ("clean", (clean_correct, clean_margin)),
            ("source_deletion", (source_correct, source_margin)),
            ("match_deletion", (match_correct, match_margin)),
            ("correct_repair", (repair_correct, repair_margin)),
            ("wrong_repair", (wrong_correct, wrong_margin)),
        ):
            branches[name]["correct"].extend(correct)
            branches[name]["margin"].extend(margin)

    adapter.validate_execution()  # the frozen model must be intact after instrumentation
    crosscheck = _session_consumer_crosscheck(adapter, items[0], first_clean_correct or 0.0)

    execution = {
        "consumer": consumer,
        "backend": str(adapter.execution.get("adapter", "native-decoder-v1")),
        "device": device.type,
        "dtype": str(adapter.dtype),
        "attention": str(adapter.execution.get("attention", "eager")),
        "family": adapter.spec.model_type,
        "model_identity": adapter.model_identity,
        "session_consumer_crosscheck": crosscheck,
    }
    family_edges = {str(layer): list(heads) for layer, heads in sorted(heads_by_layer.items())}
    return {
        "schema": INDUCTION_PANEL_SCHEMA,
        "panel_id": spec.panel_id,
        "seed": spec.seed,
        "length": spec.length,
        "examples": len(items),
        "classes": classes,
        "route_family": family_edges,
        "route_edge_count": sum(len(heads) for heads in heads_by_layer.values()),
        "workload": spec.to_dict(),
        "branches": branches,
        "execution": execution,
    }


def certify_decoder_induction(
    adapter: Any,
    specs: Sequence[InductionPanelSpec],
    policy: InductionCircuitPolicy | None = None,
    *,
    batch_size: int = 32,
    family: Mapping[int, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Measure every declared panel on ``adapter`` and certify the fixed route family."""

    selected = policy or InductionCircuitPolicy()
    panels = [
        measure_induction_panel(
            adapter,
            spec,
            consumer=selected.consumer.consumer,
            batch_size=batch_size,
            family=family,
        )
        for spec in specs
    ]
    certificate = certify_induction_route(panels, selected)
    return {"certificate": certificate, "panels": panels}


__all__ = [
    "measure_induction_panel",
    "certify_decoder_induction",
]
