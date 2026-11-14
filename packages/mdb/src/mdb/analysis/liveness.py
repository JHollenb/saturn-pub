"""Causal liveness of symbol-table bindings (symbol-table additions review 2026-11-14, 2.11).

Mass says who is read, contrast says what the read carries, liveness says whether the consumer
needs it. Per binding, ``liveness`` measures the native consumer change when the consumer
row's reads of that binding are removed at every layer and head: ``p[h, j, s] = 0`` for every
value cell ``s`` the binding holds, no renormalisation (the ``rows.row_table`` arm dose). It is
realised by zeroing those value cells for row ``j``'s forward only: zeroing ``p[h, j, s]`` and
zeroing ``v_s`` give the same row-``j`` output when only row ``j`` reads the changed cells
(``rows.py``), so the ``mask`` and ``zero`` forms are the same number at the reader row; the
record says which was asked for.

Procedure (measured, batch 1, stepped): one native forward of rows ``0..j-1`` with a cache,
then per arm one forward of row ``j`` against that cache with the arm's value cells zeroed
(cells at row ``j`` itself are zeroed as they are appended). A null arm (no cell zeroed) runs
the same program; ``null_vs_stepped`` is its consumer difference from the run's own stepped
value, the noise ``floor`` of every verdict. Arms run one at a time (batch 1): issue T2
measured batched edited arms differing from stepped by up to 0.625 on Qwen3-1.7B, so no
batched arm decides a verdict here.

Verdicts (the ``reads.verdict`` vocabulary, ported as ``mdb_lanes.symbol_reads`` scopes): an
observed mass has causal status ``unknown``; a measured arm attaches a typed result with
``scope: set`` (all of the binding's reads at once), ``claim: necessary`` and
``verdict``: ``effect`` (``|delta| > max(floor, min_effect)``), ``null`` (exactly no change)
or ``below_floor``.

Refusals: a run carrying edits (the fresh forward would not hold them), a consumer that is not
a margin or a logit of row ``j``'s logits.
"""

from __future__ import annotations

from typing import Any

from .finite import MEASURED, SUPPLIED, num
from .forecast import Logit, Margin, Refusal

UNKNOWN = {"status": "unknown",
           "meaning": "observed mass and effect only; no intervention measured (mass is not "
                      "causation)"}
FORMS = ("mask", "zero")
DOSE = ("p[h, j, s] = 0 for every head and layer of the consumer row j and every value cell s "
        "of the binding (no renormalisation), realised by zeroing those value cells for row j's "
        "forward only")


def _torch():
    import torch

    return torch


def _token(run: Any, row: int) -> int:
    if row < len(run.tokens):
        return int(run.tokens[row])
    held = run.session.read("tokens")[0]
    return int(held[row])


def _value(target: Any, logits: Any) -> float:
    flat = logits.reshape(-1).double()
    if isinstance(target, Margin):
        return float(flat[target.a] - flat[target.b])
    return float(flat[target.i])


def _cells(binding: Any, j: int, layers: int, kv: int) -> dict[int, list[tuple[int, int]]]:
    """layer -> [(kv, row)] of the binding's value cells at rows <= j."""
    out: dict[int, list[tuple[int, int]]] = {}
    for cell in binding.cells:
        if cell.row > j or (cell.ports is not None and "v" not in cell.ports):
            continue
        for layer in (cell.layers if cell.layers is not None else range(layers)):
            for g in (cell.kv if cell.kv is not None else range(kv)):
                out.setdefault(layer, []).append((g, cell.row))
    return out


def liveness(run: Any, bindings: list[Any], reader: Any, *, form: str = "mask",
             min_effect: float = 0.0) -> dict[str, dict[str, Any]]:
    """Per binding name, the causal result (module docstring)."""
    torch = _torch()
    from .rows import decoder_parts

    if form not in FORMS:
        raise Refusal(f"causal form is one of {FORMS}, not {form!r}")
    if reader is None or not reader.scalar or not isinstance(reader.target, (Margin, Logit)):
        raise Refusal("causal liveness needs a margin or logit consumer (M[a,b] or logits[i])")
    j = int(reader.row)
    info = run.read_info(f"t{j}.logits")
    if info["edited"] or info["upstream_edits"]:
        raise Refusal("causal liveness runs a fresh native forward; this run carries edits "
                      "(remove them, or measure with whatif)")
    model = run.adapter.model
    base, _, _ = decoder_parts(run)
    device = next(model.parameters()).device
    layers, kv = run.adapter.layers, run.adapter.num_kv_heads
    tokens = [_token(run, r) for r in range(j + 1)]
    stepped = _value(reader.target, run.read(f"t{j}.logits"))
    with torch.no_grad():
        cache = None
        if j > 0:
            cache = base(input_ids=torch.tensor([tokens[:j]], device=device),
                         use_cache=True).past_key_values

        def arm(cells: dict[int, list[tuple[int, int]]]) -> float:
            saved, patched = [], []
            try:
                for layer, found in cells.items():
                    for g, row in found:
                        if row < j:
                            values = cache.layers[layer].values
                            saved.append((layer, g, row, values[0, g, row].clone()))
                            values[0, g, row] = 0
                own = {layer: sorted({g for g, row in found if row == j})
                       for layer, found in cells.items()}
                own = {layer: gs for layer, gs in own.items() if gs}
                if own:
                    patched = _zero_own(base, own)
                out = model(input_ids=torch.tensor([[tokens[j]]], device=device),
                            past_key_values=cache,
                            position_ids=torch.tensor([[j]], device=device), use_cache=True)
                value = _value(reader.target, out.logits[0, -1])
            finally:
                for restore in patched:
                    restore()
                if cache is not None:
                    cache.crop(j)
                    # the forward re-allocates the cache (concatenation): restore the cells in
                    # the tensors the cache holds now, not the ones zeroed before it ran
                    for layer, g, row, keep in saved:
                        cache.layers[layer].values[0, g, row] = keep
            return value

        null = arm({})
        floor = abs(null - stepped)
        out: dict[str, dict[str, Any]] = {}
        for binding in bindings:
            cells = _cells(binding, j, layers, kv)
            if not cells:
                out[binding.name] = {"status": "intervened", "form": form, "scope": "set",
                                     "claim": "necessary", "verdict": "null",
                                     "delta": num(0.0, MEASURED),
                                     "reason": "the binding holds no value cell read by row j"}
                continue
            value = arm(cells)
            delta = value - null
            if delta == 0.0:
                verdict = "null"
            elif abs(delta) > max(floor, float(min_effect)):
                verdict = "effect"
            else:
                verdict = "below_floor"
            out[binding.name] = {
                "status": "intervened", "form": form, "scope": "set", "claim": "necessary",
                "intervention": DOSE, "verdict": verdict,
                "delta": num(delta, MEASURED), "floor": num(floor, MEASURED),
                "min_effect": num(float(min_effect), SUPPLIED),
                "null_vs_stepped": num(null - stepped, MEASURED),
                "numerics": "batch 1, stepped from a fresh prefix cache; floor = |null arm - "
                            "the run's stepped consumer|"}
    return out


def _zero_own(base: Any, own: dict[int, list[int]]) -> list[Any]:
    """Zero the appended value rows of KV heads ``own[layer]`` (row j's own cells) as they are
    stored; returns the restore callables."""
    restores = []
    for layer, groups in own.items():
        attn = base.layers[layer].self_attn
        head_dim = _head_dim(base, attn)

        def hook(mod, inp, out, groups=groups, head_dim=head_dim):
            out = out.clone()
            for g in groups:
                out[..., -1:, g * head_dim:(g + 1) * head_dim] = 0
            return out
        handle = attn.v_proj.register_forward_hook(hook)
        restores.append(handle.remove)
    return restores


def _head_dim(base: Any, attn: Any) -> int:
    dim = getattr(attn, "head_dim", None)
    if dim:
        return int(dim)
    cfg = base.config
    return int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)


__all__ = ["DOSE", "FORMS", "UNKNOWN", "liveness"]
