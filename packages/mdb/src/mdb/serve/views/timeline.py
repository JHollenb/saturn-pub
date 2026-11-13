"""Timeline: one lens over the first two clock axes (token row x depth boundary).

Cells the recording never reached are not listed; ``fill`` states their evidence class
(``unresolved``) and why. A listed cell with ``v = 0`` means the recording passed through that
boundary and the lens counted nothing there, which is a recorded fact, not a missing value.
"""
from __future__ import annotations

from collections import Counter

from ..caps import take
from ..schema import TimelineBlock, TimelineCell, TokenCell, Window, unresolved
from .common import ViewError, cut_refs
from .sessions import LENSES, lens_infos
from .symbols import symbol_cells

MAX_SPAN_X, MAX_SPAN_Y, MAX_CELLS = 512, 256, 1500
NOT_REACHED = "not reached in the recorded history (no cut or native transition here)"


def _branch_scope(model, branch):
    if branch is None:
        return list(model.cut_ids), model.receipts
    cuts = model.branch_cuts(branch)
    if not cuts:
        raise ViewError("unknown_branch", f"no branch {branch!r} in session {model.sid}", 404)
    keep = set(cuts)
    return cuts, [r for r in model.receipts if r.get("result") in keep]


def _counts(model, lens, cuts, receipts):
    reached = set()
    for cut in cuts:
        if model.cell_of_cut.get(cut):
            reached.add(model.cell_of_cut[cut])
    transitions = Counter()
    for receipt in receipts:
        for step in receipt.get("transitions") or []:
            for key in ("before", "after"):
                cell = model.place(step.get(key))
                if cell:
                    reached.add(cell)
            cell = model.place(step.get("before"))
            if cell:
                transitions[cell] += 1
    cut_count = Counter(model.cell_of_cut[c] for c in cuts if model.cell_of_cut.get(c))
    record_count = Counter()
    for record in [*model.records(), *model.findings()]:
        for cut in cut_refs(record):
            if cut in cuts and model.cell_of_cut.get(cut):
                record_count[model.cell_of_cut[cut]] += 1
    symbols = {cell: len(ids) for cell, ids in symbol_cells(model).items()}
    if lens == "cuts":
        values = {cell: cut_count.get(cell, 0) for cell in reached}
    elif lens == "receipts":
        values = {cell: transitions.get(cell, 0) for cell in reached}
    elif lens == "records":
        values = {cell: record_count.get(cell, 0) for cell in reached}
    else:
        values = dict(symbols)
    return reached, values, cut_count, record_count, symbols


def _glyphs(model, cell, cuts, cut_count, record_count, symbols):
    out = []
    if cut_count.get(cell):
        out.append("cut")
    if model.selected in cuts and model.cell_of_cut.get(model.selected) == cell:
        out.append("selected")
    branches = {model.branch[c] for c in cuts if model.cell_of_cut.get(c) == cell}
    parents = {model.branch.get(model.cuts[c]["parent"]) for c in cuts
               if model.cell_of_cut.get(c) == cell and model.cuts[c]["parent"]}
    if len(branches) > 1 or (branches and parents and branches - parents - {None}):
        out.append("fork")
    if symbols.get(cell):
        out.append("symbol")
    if record_count.get(cell):
        out.append("record")
    return out


def timeline(model, *, lens="cuts", x0=None, x1=None, y0=None, y1=None, branch=None):
    if lens not in LENSES:
        raise ViewError("unknown_lens", f"lens is one of {', '.join(LENSES)}")
    axes = model.axes()
    nx, ny = len(axes[0].ticks), len(axes[1].ticks)
    x0 = 0 if x0 is None else x0
    y0 = 0 if y0 is None else y0
    x1 = min(nx, x0 + MAX_SPAN_X) if x1 is None else x1
    y1 = min(ny, y0 + MAX_SPAN_Y) if y1 is None else y1
    if not (0 <= x0 <= x1 and 0 <= y0 <= y1):
        raise ViewError("bad_window", "window needs 0 <= x0 <= x1 and 0 <= y0 <= y1")
    if x1 - x0 > MAX_SPAN_X or y1 - y0 > MAX_SPAN_Y:
        raise ViewError("bad_window", f"window spans at most {MAX_SPAN_X} x {MAX_SPAN_Y} cells")
    cuts, receipts = _branch_scope(model, branch)
    reached, values, cut_count, record_count, symbols = _counts(model, lens, cuts, receipts)
    unit = LENSES[lens][1]
    if lens == "symbols":
        fill = unresolved("no registered symbol location or read here; absence not established",
                          unit=unit)
    else:
        fill = unresolved(NOT_REACHED, unit=unit)
    cells, omitted = [], []
    for (x, y), v in sorted(values.items()):
        if x0 <= x < x1 and y0 <= y < y1:
            cells.append(TimelineCell(x=x, y=y, v=float(v), ev="measured",
                                      glyphs=_glyphs(model, (x, y), cuts, cut_count,
                                                     record_count, symbols)))
    cells = take(cells, MAX_CELLS, "cells", omitted,
                 reason="cells beyond the first (row-major); request a smaller window")
    tokens = []
    for x in range(x0, min(x1, nx)):
        if model.tokens is not None and x < len(model.tokens):
            tokens.append(TokenCell(x=x, id=model.tokens[x], ev="measured"))
        else:
            tokens.append(TokenCell(x=x, id=None, ev="unresolved",
                                    reason=model.tokens_reason or "row beyond the tokens slot"))
    block = TimelineBlock(lens=lens, lenses=lens_infos(model), unit=unit, axes=axes[:2],
                          win=Window(x0=x0, x1=x1, y0=y0, y1=y1), cells=cells, fill=fill,
                          tokens=tokens, branch=branch)
    return block, omitted
