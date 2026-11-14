"""Batched native sweeps of state-cell edits at a run's pending row.

``cell_sweep(run, arms)`` measures many alternative versions of the key/value cache in one
native forward of the pending row per chunk: arm ``i`` is a list of cell replacements
``Cell(layer, "k"|"v", row, kv, value)``; the cache of batch element ``i`` is the run's prefill
cache with those cells replaced (the value cast once to the cache dtype), and the pending token
runs through the model's own forward (``model(token, past_key_values=cache)``), batch ``B``. It
is the batched form of ``Run.set_cell(layer, t, value, rows=[row], kv=[kv])`` followed by
``until logits`` from the first port of the pending row: the edited cells are read only by the
pending row (the prefill rows already ran), exactly as a staged cell is realised there.

Numerics: a batch-1 forward is the Hugging Face incremental program the stepped run equals
(tested on the tiny models). A batch of ``B > 1`` may round differently (another matmul
shape). Each chunk therefore carries a null arm (no edit) at batch index 0, and the result
reports ``null_vs_stepped``: the largest ``|logit|`` difference between that null arm and the
run's stepped native logits, measured, so a caller can put it into its noise floor.

Labels: every returned logit is ``measured`` (native forward), under the edits of its arm.
Memory: one chunk's cache (``B`` copies of the prefill cache) and logits of the requested token
ids only; nothing else is kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from .forecast import Refusal

_ROW = re.compile(r"t(?P<row>\d+)\.L0\.")


@dataclass(frozen=True)
class Cell:
    """One cell replacement: ``cache.L<layer>.<tensor>[kv<kv>, row=<row>]`` = ``value``
    (a tensor of ``head_dim`` values)."""

    layer: int
    tensor: str
    row: int
    kv: int
    value: Any


def _check(run: Any) -> int:
    """The pending row, after refusing a cursor that is not at the pending row's first port or a
    run that carries edits (the sweep starts from the native prefill cache)."""
    first = run.program.names[0]
    match = _ROW.match(run.where)
    if match is None or not run.where.endswith(f".L0.{first}"):
        raise Refusal(f"cell_sweep needs the cursor at the pending row's first port "
                      f"(t<row>.L0.{first}, where start leaves it), not {run.where}")
    if run.edits or run.cells or run.committed_cells() or run.committed_edits():
        raise Refusal("cell_sweep starts from the native prefill cache: the run carries edits")
    row = int(match["row"])
    if row != len(run.tokens) - 1:
        raise Refusal("cell_sweep runs the last prompt row only")
    return row


def cell_sweep(run: Any, arms: Sequence[Sequence[Cell]], *, tokens: Sequence[int] | None = None,
               batch: int = 64, stepped_logits: Any = None) -> dict[str, Any]:
    """Logits of the pending row under each arm's cell replacements, measured natively in
    batches of ``batch`` (index 0 of every chunk is a null arm, not counted in ``batch``).

    ``tokens``: vocabulary ids to return (default: all). ``stepped_logits``: the run's own
    native logits at the pending row (to report ``null_vs_stepped``); when omitted the run is
    not moved and the check is not made.

    Returns ``{"logits": float32 tensor (len(arms), len(tokens)), "argmax": long (len(arms),),
    "null": float32 (chunks, len(tokens)), "null_vs_stepped": float or None, "batches": n}``.
    """
    import torch
    from transformers import DynamicCache

    row = _check(run)
    if batch < 1:
        raise Refusal("batch must be at least 1")
    adapter = run.adapter
    model = adapter.model
    layers = adapter.layers
    keys = [run.read_cache(layer, "k", staged=False) for layer in range(layers)]
    values = [run.read_cache(layer, "v", staged=False) for layer in range(layers)]
    length = int(keys[0].shape[2])
    if length != row:
        raise Refusal(f"the cache holds {length} rows, the pending row is t{row}")
    for arm in arms:
        for cell in arm:
            if cell.tensor not in ("k", "v"):
                raise Refusal("a cell is k or v")
            if not (0 <= cell.layer < layers and 0 <= cell.row < row
                    and 0 <= cell.kv < adapter.num_kv_heads):
                raise Refusal(f"no cell cache.L{cell.layer}.{cell.tensor}"
                              f"[kv{cell.kv},row={cell.row}] before t{row}")
            if int(cell.value.numel()) != adapter.head_dim:
                raise Refusal(f"a cell value has {adapter.head_dim} values, got "
                              f"{int(cell.value.numel())}")
    device = keys[0].device
    pick = None if tokens is None else torch.tensor(list(tokens), dtype=torch.long, device=device)
    token = int(run.tokens[row])
    out, argmax, nulls = [], [], []
    with torch.no_grad():
        for start in range(0, len(arms), batch):
            chunk = [[]] + [list(arm) for arm in arms[start:start + batch]]
            size = len(chunk)
            cache = DynamicCache(config=model.config)
            for layer in range(layers):
                k = keys[layer].expand(size, -1, -1, -1).clone()
                v = values[layer].expand(size, -1, -1, -1).clone()
                for index, arm in enumerate(chunk):
                    for cell in arm:
                        if cell.layer != layer:
                            continue
                        target = k if cell.tensor == "k" else v
                        target[index, cell.kv, cell.row] = cell.value.reshape(-1).to(
                            device=device, dtype=target.dtype)
                cache.update(k, v, layer)
            ids = torch.full((size, 1), token, dtype=torch.long, device=device)
            logits = model(ids, past_key_values=cache, use_cache=True).logits[:, -1]
            del cache
            chosen = logits if pick is None else logits.index_select(1, pick)
            argmax.append(logits.argmax(dim=-1)[1:].cpu())
            out.append(chosen[1:].float().cpu())
            nulls.append(chosen[0].float().cpu())
            del logits
    result: dict[str, Any] = {
        "logits": torch.cat(out) if out else torch.zeros(0),
        "argmax": torch.cat(argmax) if argmax else torch.zeros(0, dtype=torch.long),
        "null": torch.stack(nulls) if nulls else torch.zeros(0),
        "batches": len(out),
        "null_vs_stepped": None,
        "authority": "measured",
    }
    if stepped_logits is not None and nulls:
        native = stepped_logits.reshape(-1)
        native = native if pick is None else native.index_select(0, pick.to(native.device))
        native = native.float().cpu()
        result["null_vs_stepped"] = float(max((n - native).abs().max() for n in nulls))
    return result


__all__ = ["Cell", "cell_sweep"]
