"""The Qwen read-panel lane measured through the engine (``mdb.trace``, ``analysis.reads``,
``analysis.propagate.whatif``), with the lane's controls and verdicts kept on top.

What the engine does here, and what stays on the lane path (``attention_port``):

* attention weights at the receiver row: :func:`trace_attention` captures the native ``L<l>.p``
  of the caller's own forward with ``mdb.trace`` (passive: the forward's outputs are unchanged);
  :func:`run_attention` reads the same ports from a stepped engine ``Run``. Layout
  ``[layer][head][source]`` as the lane's ``receiver-attention.json``. The lane's own capture is
  an FP32 softmax recomputed from q and k (``attention_port.Read.probs``); the engine's is the
  native weight in the model dtype.
* per-source contribution: :func:`contributions` is ``analysis.reads.head_mix`` (``p[h,s]
  W_o,h v[kv(h),s]`` in float64, the zero-cut identity of ``predicted_write.zero_cut_delta``
  through ``o_proj``) read out to the margin by ``analysis.reads.readout`` (P19).
* measured cut: :func:`engine_cuts` is ``propagate.whatif_many`` of a zero cut ``p[h, s] = 0``
  at ``t<row>.L<l>.p`` (the engine's ``reads`` causal cut), one what-if per cut; several heads
  of one layer go in one masked edit. Its arithmetic is the lane's ``zero_method="recompute"``,
  not the lane's default ``delta`` (see :func:`expressible`).
* NOT through the engine (lane path kept): the renormalising cut (the engine has no op that
  renormalises the remaining weights), set cuts that span layers and every ``keep`` arm (a
  what-if takes one edit; ``keep`` cuts every other head of every layer), the lane's FP32
  arithmetic floors (``floor_*`` arms; the engine's floor is one dtype spacing and its emulated
  residual, reported per cut).

:func:`expressible` decides per arm; :func:`engine_rows` turns engine measurements into worker
rows (``{arm, margin, ...}``) that ``qwen_panels.read_results`` scores with the panel's controls
(random draws, mass matching, other item, verdicts) unchanged.

Torch and the engine are imported inside functions; importing this module loads neither.
"""
from __future__ import annotations

from . import attention_panels as P

ENGINE_PATH = ("mdb.attach -> Debugger.run(prefill=K) stepped Run; zero cut p[h,src]=0 at t<row>.L<l>.p "
               "through mdb.analysis.propagate.whatif_many; margin from the native edited logits")


class EngineLaneError(ValueError):
    """An arm or a request the engine path cannot take."""


# ------------------------------------------------------------------------------- arm mapping

def zero_method(arm, panel):
    """The zero method an arm's cut runs with: the arm's own, else the plan's port default
    (``attention_port`` default ``delta``); ``floor_cut_recompute`` arms always recompute."""
    if arm.kind == "floor_cut_recompute":
        return "recompute"
    if arm.zero_method is not None:
        return arm.zero_method
    return ((panel.execution or {}).get("zero_method") or {}).get("port_default") or "delta"


def expressible(arm, rows_table, panel=None, *, staged=False):
    """``(cuts, None)`` with ``cuts = [(layer, (heads...), receiver_row, source_row)]`` (one engine
    edit per layer) when the arm is a zero cut the engine makes with the same arithmetic in one
    what-if, else ``(None, reason)``. The native arm maps to ``([], None)`` (nothing cut).

    The engine's cut writes ``p[h, s] = 0`` and lets eager attention recompute the head mix, which
    is the lane's ``zero_method="recompute"`` (MEASURED bit for bit on tiny Qwen2 FP32, CPU,
    ``tests/test_qwen_engine.py``); the lane's default ``delta`` method subtracts ``P[cut] @ V[cut]``
    from the native output and rounds again, so a ``delta`` arm stays on the lane path. With
    ``panel`` the plan's default zero method is used; without it ``delta`` is assumed.

    ``staged``: also accept a set over several layers or source rows (measured natively by
    :func:`staged_cut_run`, one run per arm, without a what-if verdict)."""
    if arm.family != "read":
        return None, "not a read arm"
    if arm.kind == "native":
        return [], None
    if arm.kind not in ("cut", "floor_cut_recompute"):
        return None, (f"kind {arm.kind!r}: lane path (keep and all_cut cut every other head of every "
                      "layer; sham floors are the lane's own arithmetic)")
    if arm.cut_type != "zero":
        return None, "renormalising cut: the engine has no renormalising edit"
    method = zero_method(arm, panel) if panel is not None else (
        "recompute" if arm.kind == "floor_cut_recompute" else arm.zero_method or "delta")
    if method != "recompute":
        return None, (f"zero_method {method!r}: the engine's edit recomputes the head mix (the lane's "
                      "'recompute' arithmetic), not native_out - P[cut] @ V[cut]")
    if not arm.units:
        return None, "no units"
    layers = sorted({int(u[0]) for u in arm.units})
    if len(layers) != 1 and not staged:
        return None, f"units span layers {layers}: a what-if takes one edit"
    receivers = [r for name in arm.receivers for r in rows_table[name]]
    sources = [s for name in arm.sources for s in rows_table[name]]
    if len(receivers) != 1:
        return None, "the engine cut is made on one receiver row"
    return [(layer, tuple(sorted({int(u[1]) for u in arm.units if int(u[0]) == layer})), receivers[0], s)
            for layer in layers for s in sources], None


# ------------------------------------------------------------------------------- attention

def _p_rows(values, layers):
    """``[layer][head][source]`` float lists from per-layer ``p`` tensors ``[1, H, Q, S]`` (last query row)."""
    out = []
    for layer in range(layers):
        p = values[layer]
        out.append(p[0, :, -1, :].double().cpu().tolist())
    return out


def trace_attention(model, forward, *, layers, call=0):
    """Native attention weights ``[layer][head][source]`` of the last query row of traced call
    ``call`` while ``forward()`` runs (the caller's own forward, unchanged). Returns
    ``(attention, info)``; ``info`` holds the trace's labels of layer 0."""
    import mdb
    ports = [f"L{layer}.p" for layer in range(layers)]
    with mdb.trace(model, ports=ports) as t:
        result = forward()
    values = [t[f"L{layer}.p", call] for layer in range(layers)]
    info = {k: t.info("L0.p", call)[k] for k in ("state", "authority", "kind", "dtype", "axes")}
    info["calls"] = t.calls
    return _p_rows(values, layers), info, result


def attach_run(model, tokenizer, tokens, *, prefill, row=None):
    """An engine debugger and a stepped ``Run`` of ``tokens`` (rows ``0..prefill-1`` batched, the
    rest stepped with their tokens forced), stopped at ``t<row>.logits`` (default the last row)."""
    import mdb
    dbg = mdb.attach(model, tokenizer)
    run = dbg.run(list(tokens), prefill=prefill)
    row = len(tokens) - 1 if row is None else row
    run.until(f"t{row}.logits")
    return dbg, run


def run_attention(run, row, layers):
    """Native ``t<row>.L<l>.p`` read from a stepped run, ``[layer][head][source]``."""
    return _p_rows([run.read(f"t{row}.L{layer}.p") for layer in range(layers)], layers)


def run_logits(run, row):
    return run.read(f"t{row}.logits").reshape(-1)


# ------------------------------------------------------------------------------- contributions

def contributions(run, row, source, consumer):
    """Exact per-(layer, head) contribution of ``source``'s read at ``row`` to ``consumer`` (a
    ``Margin``): ``{(layer, head): {"mass", "to_consumer"}}`` (float64; ``analysis.reads``)."""
    from mdb.analysis import reads as R
    reader = R.readout(run, consumer)
    out = {}
    for layer in range(run.adapter.layers):
        mix = R.head_mix(run, row, layer, {})     # one float64 o_proj at a time (not all layers held)
        parts = mix.o_parts()
        for h in range(mix.p.shape[0]):
            value = reader.apply(parts[h, source]) if reader.reaches(layer, "attn") is None else None
            out[(layer, h)] = {"mass": float(mix.p[h, source]),
                               "to_consumer": None if value is None else float(value)}
    return out


# ------------------------------------------------------------------------------- measured cuts

def _mask(run, layer, heads, row, source):
    from mdb.analysis.reads import head_mask
    mask = None
    for h in heads:
        m = head_mask(run, row, layer, h, source)
        mask = m if mask is None else (mask | m)
    return mask


def engine_cuts(run, cuts, consumer):
    """Measure each zero cut ``(layer, heads, row, source)`` with ``whatif_many`` (one excursion,
    session restored). Returns one dict per cut: ``edited_logits`` (native, the consumer row),
    ``delta`` (native margin change), ``emulated`` and ``exact`` changes, the P20 ``verdict``,
    or ``refused`` with the reason."""
    from mdb.analysis.propagate import Edit, whatif_many
    from mdb.analysis.forecast import Refusal
    requests = []
    for layer, heads, row, source in cuts:
        selection = ",".join(f"h{h}" for h in heads) + f",src={source}"
        requests.append((Edit(f"t{row}.L{layer}.p", 0.0, "replace", _mask(run, layer, heads, row, source),
                              selection, "zero cut: the source removed from these heads"), consumer))
    out = []
    for cut, result in zip(cuts, whatif_many(run, requests, stages=False), strict=True):
        if isinstance(result, Refusal):
            out.append({"cut": cut, "refused": str(result)})
            continue
        report, effect = result.report, result.report["effect"]

        def value(node, *keys):
            for key in keys:
                node = node.get(key) if isinstance(node, dict) else None
            return node

        out.append({"cut": cut,
                    "edited_logits": result.tensors["native_edited"].reshape(-1),
                    "base_logits": result.tensors["native_base"].reshape(-1),
                    "delta": report["measured"]["delta"]["value"],
                    "emulated": value(effect, "emulated", "value"),
                    "exact": value(report, "exact", "written", "value"),
                    "residual_spacings": value(effect, "residual_spacings", "value"),
                    "verdict": effect["verdict"], "record": report["record"]})
    return out


def staged_cut_run(dbg, tokens, cuts, *, prefill, row, name="staged"):
    """A new engine run with every zero cut ``(layer, heads, row, source)`` of ``cuts`` staged at
    its layer (any number of layers and sources; one masked edit per layer), stopped at
    ``t<row>.logits``. A native measurement only: no sealed prediction and no P20 verdict (that
    needs ``whatif``, one edit). This is how a set cut over several layers goes through the
    engine; its arithmetic is the lane's ``recompute`` zero cut at every layer."""
    by_layer = {}
    for layer, heads, r, source in cuts:
        if r != row:
            raise EngineLaneError("staged cuts are made on one receiver row")
        by_layer.setdefault(int(layer), []).append((tuple(heads), int(source)))
    run = dbg.run(list(tokens), prefill=prefill, name=name)
    for layer in sorted(by_layer):
        run.until(f"t{row}.L{layer}.h")
        mask, picked = None, []
        for heads, source in by_layer[layer]:
            m = _mask(run, layer, heads, row, source)
            mask = m if mask is None else (mask | m)
            picked.append(",".join(f"h{h}" for h in heads) + f",src={source}")
        run.set(f"L{layer}.p", 0.0, "replace", mask=mask, selection=";".join(picked))
    run.until(f"t{row}.logits")
    return run


# ------------------------------------------------------------------------------- replay recipes

def v2_checkpoint(dbg, run, name):
    """Make ``run`` durable as a replay recipe v2 (``mdb.core.recipe``) in the debugger's project
    (``mdb.attach(..., project=DIR)``): edits, masks and cell doses as the engine records them.
    Returns ``(checkpoint summary, recipe)``; ``dbg.restart(name)`` replays it."""
    from mdb.core import recipe as R
    summary = dbg.checkpoint(run, name)
    return summary, R.load(dbg.project, summary["recipe"])


def read_recipe(recipe):
    """Verify a stored lane recipe of either schema and say which: ``("v1", recipe)`` for
    ``mdb-native-qwen-replay-recipe-v1`` (``native_replay_recipe.validate``), ``("v2", recipe)``
    for ``mdb-replay-recipe-v2`` (``mdb.core.recipe.verify``). Old v1 recipes stay readable."""
    schema = recipe.get("schema") if isinstance(recipe, dict) else None
    if schema == "mdb-native-qwen-replay-recipe-v1":
        from .native_replay_recipe import validate
        return "v1", validate(recipe)
    if schema == "mdb-replay-recipe-v2":
        from mdb.core import recipe as R
        return "v2", R.verify(recipe)
    raise EngineLaneError(f"unknown recipe schema {schema!r}")


# What replay recipe v2 does not express for the Qwen panel lane (ASSERTED from mdb.core.recipe and
# mdb.engine.edits; each is a statement about the schema, not a measurement):
V2_GAPS = {
    "renormalising read cut": "no op renormalises the remaining weights; only a supplied p tensor "
                              "(replace) could stand for it, which is not the lane's FP32 masked softmax",
    "delta zero cut": "native_out - P[cut] @ V[cut] rounded again is not an edit of a port; v2 holds the "
                      "recompute arithmetic (p[h,s] = 0 at the p port)",
    "FP32 blend dose": "cell mix DOSE interpolates in float64 and rounds once; attention_panels.blend "
                       "interpolates in FP32 (equal at dose 0 and 1; other doses not yet compared)",
    "FLUX": "no FLUX / diffusers port program in mdb.engine.ports",
    "lane batching": "the lane's prefill (rows 0..18 in one forward, then one row at a time) is v2's "
                     "prefill=19; other row groupings are not expressible",
}


def margin_of(logits, a, b):
    """``float(logits[a]) - float(logits[b])`` in Python float64 (the lane's margin arithmetic)."""
    return float(logits[a]) - float(logits[b])


def engine_rows(panel, run, rows_table, *, hot, cold, row, staged=None):
    """Worker rows for every engine-expressible arm of ``panel``, measured through the engine, and
    ``{arm: reason}`` for the arms left to the lane path. Rows carry ``arm``, ``margin`` (hot -
    cold of the native logits) and the engine's verdict, so ``qwen_panels.read_results`` scores
    them with the panel's own controls.

    ``staged``: ``{"dbg", "tokens", "prefill", "only"}`` measures the zero-recompute arms whose set
    spans layers or source rows by :func:`staged_cut_run` (``only``: a set of arm names to limit
    them to, or None for all); their rows carry ``engine.kind == "staged"`` and no verdict."""
    from mdb.analysis.forecast import Margin
    consumer = Margin(row, hot, cold)
    rows, lane_only, wanted = [], {}, []
    for arm in panel.arms:
        cuts, why = expressible(arm, rows_table, panel)
        if cuts is None and staged is not None and (staged.get("only") is None or arm.arm in staged["only"]):
            cuts, why = expressible(arm, rows_table, panel, staged=True)
            if cuts:
                run_s = staged_cut_run(staged["dbg"], staged["tokens"], cuts, prefill=staged["prefill"], row=row,
                                       name="staged")
                rows.append({"arm": arm.arm, "margin": margin_of(run_logits(run_s, row), hot, cold),
                             "path": "engine", "engine": {"kind": "staged", "cuts": [[c[0], list(c[1]), c[2], c[3]]
                                                                                   for c in cuts]}})
                continue
        if cuts is None:
            lane_only[arm.arm] = why
        elif not cuts:
            logits = run_logits(run, row)
            rows.append({"arm": arm.arm, "margin": margin_of(logits, hot, cold), "path": "engine",
                         "engine": {"kind": "native"}})
        else:
            if len(cuts) != 1:
                lane_only[arm.arm] = "several source rows: one edit per source row would be several what-ifs"
                continue
            wanted.append((arm, cuts[0]))
    for (arm, cut), measured in zip(wanted, engine_cuts(run, [c for _, c in wanted], consumer), strict=True):
        if "refused" in measured:
            lane_only[arm.arm] = "engine refused: " + measured["refused"]
            continue
        rows.append({"arm": arm.arm, "margin": margin_of(measured["edited_logits"], hot, cold), "path": "engine",
                     "engine": {k: measured[k] for k in ("delta", "emulated", "exact", "residual_spacings",
                                                         "verdict")}
                     | {"cut": [cut[0], list(cut[1]), cut[2], cut[3]]}})
    return rows, lane_only


def scored(panel, rows, results_fn):
    """``results_fn(panel, rows)`` (e.g. ``qwen_panels.read_results``) after checking every row
    names an arm of the panel; the controls and verdicts are the panel's, unchanged."""
    ids = panel.by_id()
    unknown = [r["arm"] for r in rows if r["arm"] not in ids]
    if unknown:
        raise EngineLaneError(f"rows name arms outside the panel: {unknown[:3]}")
    return results_fn(panel, rows)


__all__ = ["ENGINE_PATH", "EngineLaneError", "attach_run", "contributions", "engine_cuts", "engine_rows",
           "expressible", "margin_of", "run_attention", "run_logits", "scored", "trace_attention", "P"]
