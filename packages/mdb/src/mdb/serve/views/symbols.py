"""Symbol table view: bindings joined with lifecycles and the attention reads they record."""
from __future__ import annotations

from saturn_pub.values import digest

from .. import _legacy
from ..schema import (
    LifePoint,
    Lifetime,
    ReadEdge,
    ReaderView,
    ReadSourceView,
    SymbolDetail,
    SymbolPage,
    SymbolRow,
    UseView,
    num,
    unresolved,
)
from ..stored import as_dict, as_str, is_number
from .common import ViewError, address_cell, location_address, parse

PAGE = 25
LIST = 8
UNBOUND_REASON = ("not measured in this recording: the unbound share of a value needs the "
                  "engine's per-port captures at the cursor")


def _reads(model, lifecycle_record):
    """``SymbolRead`` dicts derived from an attention lifecycle (empty when it has none)."""
    return _legacy.lifecycle_reads(lifecycle_record["lifecycle"])


def _edge(model, read):
    body = read
    reader, source = body["reader"], body["source"]
    results = body["causal"]["results"]
    verdict = results[-1]["verdict"] if results else "untested"
    step = reader.get("receiver_row")
    addr = None
    if isinstance(step, int):
        src = source["rows"][0] if len(source["rows"]) == 1 else None
        addr = f"t{step}.L{reader['block']}.p[h{reader['head']}" + (
            f",src={src}]" if src is not None else "]")
    return ReadEdge(
        id=body["fingerprint"],
        source=ReadSourceView(name=source["name"], rows=source["rows"],
                              assignment=source["assignment"]),
        reader=ReaderView(block=reader["block"], head=reader["head"],
                          kvGroup=reader.get("kv_group"), step=reader.get("step"),
                          receivers=reader.get("receivers")),
        mass=num(body["mass"], "measured", unit="probability"),
        massKind=body["mass_kind"], verdict=verdict, addr=addr)


def _point(model, cut, label, point=None):
    cell = model.place(point) if point else model.cell_of_cut.get(cut)
    clock = {"token": cell[0], "layer": cell[1]} if cell else {"token": None, "layer": None}
    return LifePoint(cut=cut, clock=clock, label=label)


def _row(model, binding, by_binding) -> SymbolRow:
    fp = binding["fingerprint"]
    related = by_binding.get(fp, [])
    reads, uses, defs, reads_at = [], [], [], []
    for loc in binding["locations"]:
        for cut, desc in model.cuts.items():
            if loc.get("execution_point") == digest(desc["execution_point"]):
                defs.append(_point(model, cut, "bound at " + loc["slot"]))
                break
    for record in related:
        body = record["lifecycle"]
        for d in body["definitions"]:
            defs.append(_point(model, d["source_cut"], "definition " + d["occurrence_id"]))
        for u in body["uses"]:
            meta = u.get("metadata") or {}
            reads_at.append(_point(model, u.get("result_cut"), "use " + u["use_id"],
                                   meta.get("native_operation_point")))
        joins = {j["use"]: j for j in record["joins"]["uses"]}
        for u in body["uses"]:
            j = joins.get(u["use_id"], {})
            uses.append(UseView(id=u["use_id"], consumer=u["consumer"],
                                support=j.get("consumer_support", "unknown"),
                                closure=j.get("receipt_closure")))
        reads += _reads(model, record)
    edges = sorted((_edge(model, r) for r in reads), key=lambda e: -(e.mass.v or 0.0))
    locations = [location_address(model, loc) for loc in binding["locations"]]
    statuses = [r["lifecycle"]["statuses"] for r in related]
    confidence = binding.get("confidence")
    return SymbolRow(
        id=fp, symbol=binding["symbol"], context=binding["context"],
        status=binding["status"],
        assignment="supplied" if binding["status"] == "supplied" else (
            edges[0].source.assignment if edges else None),
        claim=binding["claim"],
        confidence=num(confidence, "supplied") if is_number(confidence) else None,
        abstainReason=binding.get("abstain_reason"),
        locations=locations[:LIST], locationsMore=max(0, len(locations) - LIST),
        readers=edges[:LIST], readersMore=max(0, len(edges) - LIST),
        uses=uses[:LIST], usesMore=max(0, len(uses) - LIST),
        verdict=(statuses[-1]["terminal"] if statuses else "no lifecycle recorded"),
        lifetime=Lifetime(defs=_dedupe(defs)[:LIST], reads=_dedupe(reads_at)[:LIST]),
        evidence=list(binding.get("evidence") or [])[:LIST],
        evidenceMore=max(0, len(binding.get("evidence") or []) - LIST),
        fingerprint=fp, lifecycles=[r["lifecycle_id"] for r in related][:LIST])


def build_rows(model):
    lifecycles = model.lifecycles()
    by_binding = {}
    for record in lifecycles:
        try:
            by_binding.setdefault(record["lifecycle"]["binding_fingerprint"], []).append(record)
        except (KeyError, TypeError) as error:
            model._problem("lifecycle", record.get("lifecycle_id"), error)
    rows = []
    for binding in model.symbols():
        try:
            rows.append(_row(model, binding, by_binding))
        except Exception as error:  # noqa: BLE001 - one malformed binding is one refused row
            model._problem("symbol", binding.get("fingerprint"), error)
    return rows, by_binding


def _dedupe(points):
    seen, out = set(), []
    for p in points:
        key = (p.cut, p.clock.get("token"), p.clock.get("layer"))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def symbol_cells(model):
    """``{(x, y): [binding fingerprints]}`` for the timeline symbol lens."""
    rows, _ = build_rows(model)
    cells = {}
    for row in rows:
        for point in [*row.lifetime.defs, *row.lifetime.reads]:
            x, y = point.clock.get("token"), point.clock.get("layer")
            if x is not None and y is not None:
                cells.setdefault((x, y), []).append(row.id)
    return cells


def symbol_page(model, *, at=None, q=None, page=0) -> SymbolPage:
    rows, _ = build_rows(model)
    if at:
        x, _ = address_cell(model, parse(at))
        if x is not None:
            rows = [r for r in rows if any(p.clock.get("token") is not None and
                                           p.clock["token"] <= x for p in r.lifetime.defs)]
    if q:
        needle = q.lower()
        rows = [r for r in rows if needle in r.symbol.lower() or needle in r.claim.lower()]
    total = len(rows)
    pages = max(1, -(-total // PAGE))
    if page < 0 or page >= pages:
        raise ViewError("bad_page", f"page {page} is outside 0..{pages - 1}")
    return SymbolPage(rows=rows[page * PAGE:(page + 1) * PAGE], page=page, pages=pages,
                      total=total, at=at, q=q, unbound=unresolved(UNBOUND_REASON),
                      coverage="registered bindings only; no match is not evidence of absence")


def symbol_detail(model, identifier) -> SymbolDetail:
    rows, by_binding = build_rows(model)
    row = next((r for r in rows if r.id == identifier), None)
    if row is None:
        raise ViewError("not_found", f"symbol {identifier} is not bound in {model.sid}", 404)
    binding = next(b for b in model.symbols() if b["fingerprint"] == identifier)
    related = by_binding.get(identifier, [])
    slices = []
    for record in related:
        slices += [d["slice_id"] for d in record["joins"]["definitions"] if d.get("slice_id")]
    for sliced in model.slices():
        if (sliced.get("symbol_binding") or {}).get("fingerprint") == identifier:
            slices.append(sliced["slice_id"])
    statuses = {k: v for k, v in as_dict(as_dict(related[-1]["lifecycle"]).get(
        "statuses")).items() if isinstance(v, str)} if related else {}
    compact = {k: v for k, v in binding.items() if k not in ("locations",)}
    return SymbolDetail(row=row, binding=compact, slices=list(dict.fromkeys(slices))[:LIST],
                        statuses=statuses,
                        claimBoundary=as_str(as_dict(related[-1]["lifecycle"]).get(
                            "claim_boundary")) or None if related else None)
