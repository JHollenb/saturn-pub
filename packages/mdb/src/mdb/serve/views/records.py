"""Records drawer: sealed records, findings, slices and symbol lifecycles of one session."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

from ...core.address import format_address, from_json
from ...core.circuit_pages import PAGE_SCHEMA, PageError, locate
from .. import blobs
from ..caps import NONFINITE, bound_json, canonical, nonfinite
from ..schema import (
    STRUCTURAL_KEYS,
    ArmRow,
    Link,
    NamedNum,
    Num,
    Omitted,
    PointerRef,
    RecordDetail,
    RecordPage,
    RecordSummary,
    StoredBody,
    num,
)
from ..schema import unresolved as unresolved_num
from ..stored import as_dict, as_int, as_list, as_str, is_number
from .common import (
    ViewError,
    blob_ref,
    cell_address,
    cut_refs,
    descriptor_ref,
    occurrence_row,
    record_id,
    short,
    spec_digest,
)

PAGE = 25
AUTH = {"measured", "derived-exact", "derived-local", "bound", "supplied"}
LINK_KEYS = ("focus_id", "plan_id", "slice_id", "investigation_id", "analysis_id")
ARM_LABEL = {0: "sham (no edit)", 1: "A", 2: "B", 3: "A+B"}


def _is_auth(value) -> bool:
    """A stored ``authority`` may be any JSON value (some records hold a dict there)."""
    return isinstance(value, str) and value in AUTH


def _v2_number(value):
    """The v2 form ``{value, authority[, unit, proof]}`` as a ``Num``, or None. This is the only
    form through which a stored body can carry a class."""
    if _is_auth(value.get("authority")) and is_number(value.get("value")):
        return num(value["value"], value["authority"], unit=as_str(value.get("unit")) or None,
                   proof=as_str(value.get("proof")) or None)
    return None


def _claims(value, path="", out=None, limit=4):
    """``num``-shaped claims in a v2 body, in a stable order."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(value, dict):
        claim = _v2_number(value)
        if claim is not None:
            out.append(NamedNum(name=path or "value", n=claim))
            return out
        for key in sorted(value):
            _claims(value[key], f"{path}.{key}" if path else key, out, limit)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _claims(item, f"{path}[{i}]", out, limit)
    return out


SUPPLIED_NOTE = "as recorded; this record carries no per-number authority"
CIRCUIT = "mdb-circuit-investigation-v1"
# (schema id, record kind, field path) -> (evidence class, where the field's meaning is defined).
# Only fields whose meaning the schema's writer defines are listed; "*" stands for one dynamic key
# (an arm mask). Every other bare float in a stored body is "supplied": nothing is guessed from a
# field's name. Writer: packages/mdb/src/mdb/circuit_investigation.py.
FIELD_CLASS = {
    # measure() output of one native arm run (circuit_investigation.py:639-652)
    (CIRCUIT, "comparison-arm", "measurement.value"): ("measured", "circuit_investigation.py:652"),
    # gradient . (actual - source) delta, local autograd (circuit_investigation.py:696)
    (CIRCUIT, "comparison-arm", "local_linear_prediction"): (
        "derived-local", "circuit_investigation.py:696"),
    # native effect minus the local prediction, a difference with a local term
    # (circuit_investigation.py:697-701)
    (CIRCUIT, "comparison-arm", "native_minus_local_prediction"): (
        "derived-local", "circuit_investigation.py:697"),
    # arm value minus sham value, both measured (circuit_investigation.py:739)
    (CIRCUIT, "comparison", "effects.*"): ("derived-exact", "circuit_investigation.py:739"),
    # measured[3] - measured[1] - measured[2] + measured[0] (circuit_investigation.py:728)
    (CIRCUIT, "comparison", "interaction"): ("derived-exact", "circuit_investigation.py:728"),
}


def _field_class(schema, kind, path):
    hit = FIELD_CLASS.get((schema, kind, path))
    if hit is None and "." in path:
        hit = FIELD_CLASS.get((schema, kind, path.rsplit(".", 1)[0] + ".*"))
    return hit[0] if hit else None


def _supplied(value, recorded, path, note=SUPPLIED_NOTE):
    return num(value, "supplied", proof=f"stored in the record body at {path}",
               note=note, recorded_authority=recorded or None).model_dump(mode="json")


LABEL_ITEMS = 64
NOT_A_NUMBER = "stored value is not a number"


def _forged(value, path):
    """A stored ``{v, ev}`` look-alike of the server's own wire shape: never passed through. Its
    ``v`` (when a finite number) is ``supplied`` and says which class the body claimed."""
    claimed = value.get("ev")
    text = claimed if isinstance(claimed, str) else str(claimed)
    return _supplied(value["v"], claimed[:200] if isinstance(claimed, str) else None, path,
                     note=f"stored body claims class {text[:40]}")


def label_floats(value, *, schema, kind, recorded, path="", key=None, omitted=None):
    """Every number of a stored body as a labelled number, never guessed:

    1. a v2 ``{value, authority}`` (authority one of the five classes, value a finite number)
       keeps its class; a stored ``{v, ev}`` look-alike does not (it is ``supplied`` and says
       what it claimed);
    2. a field in ``FIELD_CLASS`` gets the class the table gives it;
    3. any other number, int or float, is ``supplied`` with ``SUPPLIED_NOTE`` and the
       record-level authority string (``recorded_authority``), never measured, unless its key
       is in ``STRUCTURAL_KEYS`` and it is an int (an id, index or count);
    4. a NaN/inf is an unresolved number saying so; a bool or a string is not a number.
    A list longer than ``LABEL_ITEMS`` under a non-structural key is labelled up to that cap and
    the remainder is recorded in ``omitted`` (when given)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        if not is_number(value):
            return nonfinite() if isinstance(value, float) else unresolved_num(
                "stored number is out of range").model_dump(mode="json")
        if key in STRUCTURAL_KEYS and isinstance(value, int):
            return value
        cls = _field_class(schema, kind, path)
        if cls:
            return num(value, cls, proof=f"stored in the record body at {path}"
                       ).model_dump(mode="json")
        return _supplied(value, recorded, path or "value")
    if isinstance(value, dict):
        if "ev" in value and "v" in value and is_number(value["v"]):
            return _forged(value, path or "value")
        claim = _v2_number(value)
        if claim is not None:
            return claim.model_dump(mode="json")
        return {k: label_floats(v, schema=schema, kind=kind, recorded=recorded,
                                path=f"{path}.{k}" if path else str(k), key=k, omitted=omitted)
                for k, v in value.items()}
    if isinstance(value, list):
        items = value
        if key not in STRUCTURAL_KEYS and omitted is not None and len(items) > LABEL_ITEMS:
            omitted.append(Omitted(field=path or "body", count=len(items) - LABEL_ITEMS,
                                   reason="list items beyond the first; not labelled"))
            items = items[:LABEL_ITEMS]
        return [label_floats(v, schema=schema, kind=kind, recorded=recorded, path=path, key=key,
                             omitted=omitted) for v in items]
    return value


def _measured(arm):
    arm = as_dict(arm)
    m = as_dict(arm.get("measurement"))
    status = arm.get("status")
    if status != "measured":
        return unresolved_num(as_str(arm.get("reason")) or f"arm status {str(status)[:40]}")
    value = m.get("value")
    if isinstance(value, float) and not math.isfinite(value):
        return unresolved_num(NONFINITE)
    if not is_number(value):
        return unresolved_num(NOT_A_NUMBER)
    cls = _field_class(CIRCUIT, "comparison-arm", "measurement.value") or "measured"
    return num(value, cls, unit=as_str(m.get("units")) or None)


def _interaction(raw):
    """The comparison's ``interaction`` as a labelled number (derived-exact by FIELD_CLASS), or
    None when absent; unresolved when stored but not a finite number."""
    if "interaction" not in raw:
        return None
    value = raw["interaction"]
    if isinstance(value, float) and not math.isfinite(value):
        return unresolved_num(NONFINITE)
    if not is_number(value):
        return unresolved_num(NOT_A_NUMBER)
    return num(value, _field_class(CIRCUIT, "comparison", "interaction"), unit="A+B-A-B+sham")


def _placeholder(rid, kind, reason, order, state="refused") -> RecordSummary:
    reason = as_str(reason)[:300]
    marker = (unresolved_num(reason) if state != "refused" else
              Num(v=None, ev="refused", reason=reason))
    return RecordSummary(id=as_str(rid) or f"{kind}-{order}", schema="?", kind=as_str(kind) or "?",
                         title=reason[:96], status=state, authority="", order=order,
                         nums=[NamedNum(name="record", n=marker)])


def _ident(kind, raw):
    """The id a row answers to; never raises."""
    raw = as_dict(raw)
    key = {"record": None, "finding": "annotation_id", "slice": "slice_id",
           "lifecycle": "lifecycle_id"}[kind]
    return as_str(record_id(raw) if key is None else raw.get(key))


def _summary(entry, order) -> RecordSummary:
    """The row for one entry; a body that cannot be projected is a refused placeholder row."""
    try:
        return _project_summary(entry, order)
    except Exception as error:  # noqa: BLE001 - one malformed stored row must not fail the list
        kind, raw = entry
        return _placeholder(_ident(kind, raw), "record" if kind == "record" else kind,
                            f"refused: record body could not be read ({type(error).__name__})",
                            order)


def _project_summary(entry, order) -> RecordSummary:
    kind, raw = entry
    raw = as_dict(raw)
    if kind == "record" and raw.get("unreadable"):
        return _placeholder(record_id(raw), as_str(raw.get("kind")) or "?",
                            as_str(raw.get("unreadable")), order,
                            "refused" if raw.get("unreadable_state") == "refused"
                            else "unresolved").model_copy(update={
                                "schema_": as_str(raw.get("schema")) or "?"})
    if kind == "record":
        rid = record_id(raw)
        nums = []
        if raw.get("schema") == "mdb-record-v2":
            nums = _claims(raw.get("body"))
            title = as_str(raw.get("kind")) + " " + as_str(as_dict(raw.get("body")).get(
                "consumer"))
            status, authority = "sealed", as_str(raw.get("authority"))
        else:
            title = " ".join(x for x in (as_str(raw.get("kind")), as_str(raw.get("component"))
                                         or as_str(raw.get("view")) or as_str(raw.get("name")))
                             if x)
            status = as_str(raw.get("terminal_status")) or as_str(raw.get("status")) or "recorded"
            authority = as_str(raw.get("authority"))
            if raw.get("kind") == "comparison-arm":
                nums = [NamedNum(name="value", n=_measured(raw))]
            if raw.get("kind") == "comparison":
                interaction = _interaction(raw)
                if interaction is not None:
                    nums = [NamedNum(name="interaction", n=interaction)]
        refs = cut_refs(raw)
        return RecordSummary(id=rid, schema=as_str(raw.get("schema")) or "?",
                             kind=as_str(raw.get("kind")) or "?",
                             title=title.strip()[:96], status=status, authority=authority,
                             cut=refs[0] if refs else None,
                             focus=as_str(raw.get("focus_id")) or None, order=order, nums=nums)
    if kind == "finding":
        details = as_dict(raw.get("details"))
        return RecordSummary(id=_ident(kind, raw), schema=as_str(details.get("schema"))
                             or "finding", kind="finding",
                             title=as_str(raw.get("label"))[:96],
                             status=as_str(details.get("terminal_status")) or "recorded",
                             authority=as_str(as_dict(raw.get("provenance")).get("authority")),
                             cut=as_str(raw.get("cut_id")) or None,
                             focus=as_str(details.get("investigation_id")) or None, order=order)
    if kind == "slice":
        return RecordSummary(id=_ident(kind, raw), schema=as_str(raw.get("schema")) or "?",
                             kind="slice", title=as_str(raw.get("component")),
                             status="complete" if raw.get("complete") is True
                             else f"{len(as_list(raw.get('gaps')))} gaps",
                             authority=as_str(raw.get("authority")),
                             cut=as_str(raw.get("cut_id")) or None, order=order)
    body = as_dict(raw.get("lifecycle"))
    return RecordSummary(id=_ident(kind, raw), schema=as_str(body.get("schema")) or "?",
                         kind="symbol-lifecycle", title=as_str(body.get("symbol")),
                         status=as_str(as_dict(body.get("statuses")).get("terminal")) or "?",
                         authority=as_str(as_dict(raw.get("joins")).get("authority")),
                         order=order)


def _with_addr(model, summary):
    """Attach the address of the summary's cut cell, so a click can move the cursor there."""
    cell = model.cell_of_cut.get(summary.cut) if summary.cut else None
    if cell:
        summary.addr = cell_address(model, *cell)
    return summary


def _entries(model):
    out = [("record", r) for r in model.records()]
    out += [("finding", f) for f in model.findings()]
    out += [("slice", s) for s in model.slices()]
    out += [("lifecycle", c) for c in model.lifecycles()]
    out += [("record", r) for r in model.unreadable()]
    return out


def record_page(model, *, kind=None, page=0) -> RecordPage:
    rows = [_with_addr(model, _summary(e, i)) for i, e in enumerate(_entries(model))]
    kinds = Counter(r.kind for r in rows)
    if kind:
        rows = [r for r in rows if r.kind == kind]
    pages = max(1, -(-len(rows) // PAGE))
    if page < 0 or page >= pages:
        raise ViewError("bad_page", f"page {page} is outside 0..{pages - 1}")
    return RecordPage(rows=rows[page * PAGE:(page + 1) * PAGE], page=page, pages=pages,
                      total=len(rows), kinds=dict(sorted(kinds.items())))


def _mask(arm):
    mask = as_int(arm.get("mask"))
    return mask if mask is not None else -1


def _arms(model, comparison):
    by_id = {record_id(r): r for r in model.records() if isinstance(r, dict)}
    wanted = [a for a in as_list(comparison.get("arms")) if isinstance(a, str)]
    arms = sorted((by_id[a] for a in wanted if a in by_id), key=_mask)
    sham = next((a for a in arms if as_int(a.get("mask")) == 0), None)
    base = _measured(sham) if sham else None
    linear = as_dict(comparison.get("linearization")).get("status") == "native-primal-matched"
    rows = []
    for arm in arms:
        value = _measured(arm)
        if base is None or base.v is None or value.v is None:
            effect = unresolved_num("no measured sham arm" if base is None or base.v is None
                                    else "arm not measured")
        else:
            effect = num(value.v - base.v, "derived-exact", unit=value.unit,
                         proof="difference of measured arms")
        prediction = residual = None
        if linear and is_number(arm.get("local_linear_prediction")):
            prediction = num(arm["local_linear_prediction"], "derived-local", unit=value.unit,
                             proof="local-autograd")
            if is_number(arm.get("native_minus_local_prediction")):
                residual = num(arm["native_minus_local_prediction"], "derived-local",
                               unit=value.unit, proof="native - local")
        decision = as_dict(arm.get("measurement")).get("decision")
        mask = _mask(arm)
        rows.append(ArmRow(mask=mask, label=ARM_LABEL.get(mask, str(arm.get("mask"))[:24]),
                           status=as_str(arm.get("status")) or "?", value=value, effect=effect,
                           prediction=prediction, residual=residual,
                           decision=None if decision is None else str(decision)[:200],
                           record=as_str(record_id(arm)),
                           cut=as_str(arm.get("result_cut")) or None))
    return rows


def _bad_pointer(why):
    return PointerRef(digest=spec_digest({"kind": "unreadable-pointer", "why": why}),
                      status="frontier", reason=why)


def _pointers(model, raw):
    out = []
    for pointer in as_list(raw.get("pointers")):
        pointer = as_dict(pointer)
        if pointer.get("kind") != "occurrence":
            continue
        try:
            address = pointer["address"]
            row = occurrence_row(model, address)
            text = format_address(from_json(address, check_ports=False))
        except Exception:  # noqa: BLE001 - a stored pointer with no readable address
            out.append(_bad_pointer("stored occurrence pointer has no readable address"))
            continue
        if row and row.get("blob_id"):
            out.append(blob_ref(model, row["blob_id"], addr=text, via=["occurrence index"]))
        else:
            out.append(descriptor_ref(as_dict(pointer.get("value")), addr=text,
                                      why="occurrence not materialised"))
    found = []

    def scan(value, depth=0):
        if depth > 6 or len(found) >= 8:
            return
        if isinstance(value, dict):
            if value.get("kind") == "tensor" and isinstance(value.get("sha256"), str):
                found.append(value)
                return
            for v in value.values():
                scan(v, depth + 1)
        elif isinstance(value, list):
            for v in value[:16]:
                scan(v, depth + 1)

    scan({k: v for k, v in raw.items() if k not in ("pointers",)})
    out += [descriptor_ref(d, why="sealed descriptor") for d in found]
    return out[:12]


def _refused_detail(summary, reason):
    summary = summary.model_copy(update={"status": "refused", "nums": [NamedNum(
        name="record", n=Num(v=None, ev="refused", reason=reason))]})
    return RecordDetail(summary=summary, body={}, arms=None, pointers=[], links=[])


def _detail(model, entry, order, identifier):
    kind, raw = entry
    summary = _with_addr(model, _summary(entry, order))
    raw = as_dict(raw)
    if kind == "slice":
        graph = as_dict(raw.get("graph"))
        raw = {k: v for k, v in raw.items() if k != "graph"} | {
            "graph": {"snapshot_id": graph.get("snapshot_id"),
                      "nodes": len(as_list(graph.get("nodes"))),
                      "edges": len(as_list(graph.get("edges")))}}
    trimmed = {k: v for k, v in raw.items() if k not in ("model_identity", "execution")}
    links = [Link(rel=k, id=raw[k]) for k in LINK_KEYS if isinstance(raw.get(k), str)]
    links += [Link(rel="arm", id=a) for a in as_list(raw.get("arms")) if isinstance(a, str)]
    arms = (_arms(model, raw) if raw.get("kind") == "comparison" and not raw.get("unreadable")
            else None)
    pointers = _pointers(model, raw)
    for items in (16, 8, 4, 2):
        omitted = []
        body = bound_json(trimmed, field="body", omitted=omitted, max_items=items,
                          max_text=512 if items > 4 else 160)
        body = label_floats(body, schema=as_str(raw.get("schema")),
                            kind=as_str(raw.get("kind")), recorded=as_str(raw.get("authority"))
                            or None, omitted=omitted)
        detail = RecordDetail(summary=summary, body=body, arms=arms, pointers=pointers,
                              links=links[:32])
        if len(canonical(detail.model_dump(mode="json", by_alias=True))) < 56 * 1024:
            return detail, omitted
    return detail, omitted


DRAWN_KINDS = ("circuit", "symbol-table")
V2_KEYS = ("value", "authority", "proof", "unit", "note", "reason")
STORED_BYTES = 60 * 1024


def _lean(value, *, schema, kind, recorded, path="", key=None):
    """A stored body with every number labelled as ``label_floats`` labels it, but keeping a v2
    ``{value, authority, ...}`` in its own (op) form, so the body is the op's JSON the Live view
    draws. A v2 number's other keys (``bound``, ``within``) are kept and labelled too."""
    if isinstance(value, dict):
        if "ev" in value and "v" in value and is_number(value["v"]):
            return _forged(value, path or "value")
        if _v2_number(value) is not None:
            out = {k: value[k] for k in V2_KEYS if k in value and (
                k == "value" or isinstance(value[k], str))}
            for k, v in value.items():
                if k not in V2_KEYS:
                    out[k] = _lean(v, schema=schema, kind=kind, recorded=recorded,
                                   path=f"{path}.{k}" if path else str(k), key=k)
            return out
        return {k: _lean(v, schema=schema, kind=kind, recorded=recorded,
                         path=f"{path}.{k}" if path else str(k), key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [_lean(v, schema=schema, kind=kind, recorded=recorded, path=path, key=key)
                for v in value]
    return label_floats(value, schema=schema, kind=kind, recorded=recorded, path=path, key=key)


class _PageRefused(PageError):
    """A stored child page that is missing or fails its checks (not an unknown node id)."""


def _page_loader(model):
    """``circuit_pages.find``'s ``load`` for a recorded project: the page blob read from the
    project store with numpy only, its payload and JSON digests checked against the pointer."""
    def load(pointer):
        if pointer.get("state") != "stored":
            raise _PageRefused("this page was held in memory by a run without a project "
                               f"({as_str(pointer.get('reason')) or 'absent'}); the recorded "
                               "project does not hold it")
        try:
            array, info = blobs.tensor_blob(model.store_root, as_str(pointer.get("ref")))
            raw = array.tobytes()
            if (info["sha256"] != pointer.get("tensor_sha256") or info["dtype"] != "uint8"
                    or len(raw) != as_int(as_dict(pointer.get("bytes")).get("value"))):
                raise ValueError("the blob differs from its pointer")
            if hashlib.sha256(raw).hexdigest() != pointer.get("sha256"):
                raise ValueError("the page differs from the pointer's sha256")
            page = json.loads(raw)
            if not isinstance(page, dict) or page.get("schema") != PAGE_SCHEMA:
                raise ValueError("not a circuit page")
        except (blobs.BlobError, OSError, ValueError, AttributeError, TypeError) as error:
            raise _PageRefused(f"page {short(as_str(pointer.get('sha256')))} refused: "
                               f"{error}") from None
        return page
    return load


def stored_body(model, identifier, node=None) -> StoredBody:
    """The whole sealed body of one ``circuit`` or ``symbol-table`` record (``StoredBody``).
    Another kind, an unreadable row or a body over ``STORED_BYTES`` is refused, never cut: a
    drawn circuit or table with nodes silently missing would not close. A circuit record is
    paged (its root page is the body; a group that did not fit is a ``paged`` pointer to a child
    page): ``node`` (``n12``) returns that node's page instead, ``body`` being the page
    (``{schema, node, ...}``), read from the project store and checked against its sha256."""
    for raw in model.records():
        if not isinstance(raw, dict) or record_id(raw) != identifier:
            continue
        kind = as_str(raw.get("kind"))
        if raw.get("unreadable"):
            raise ViewError("unreadable", f"record {short(identifier)}: "
                            f"{as_str(raw.get('unreadable'))}", 409)
        if kind not in DRAWN_KINDS or raw.get("schema") != "mdb-record-v2":
            raise ViewError("not_drawn", f"record {short(identifier)} is a {kind or '?'} record; "
                            "only engine circuit and symbol-table records are drawn (the "
                            "Records drawer shows the rest)", 409)
        body = as_dict(raw.get("body"))
        shown, page_schema = body, as_str(body.get("schema"))
        if node is not None:
            if kind != "circuit":
                raise ViewError("not_drawn", f"record {short(identifier)} is a {kind} record; "
                                "only a circuit has node pages", 409)
            try:
                count = as_dict(body.get("count")).get("value", body.get("count"))
                shown = locate(as_dict(body.get("root")), as_int(count) or 0, node,
                               _page_loader(model))
            except _PageRefused as error:
                raise ViewError("page_refused", f"record {short(identifier)}: {error}",
                                409) from None
            except PageError as error:
                raise ViewError("no_node", f"record {short(identifier)}: {error}", 404) from None
            page_schema = PAGE_SCHEMA
        lean = _lean(shown, schema=as_str(raw.get("schema")), kind=kind,
                     recorded=as_str(raw.get("authority")) or None)
        size = len(canonical(lean))
        if size > STORED_BYTES:
            raise ViewError("over_cap", f"record {short(identifier)}: its body is {size} bytes, "
                            f"over the {STORED_BYTES} the page draws at once; the REPL prints "
                            "it whole", 409)
        ledger = body.get("ledger_id")
        return StoredBody(id=identifier, kind=kind, record_schema=page_schema,
                          authority=as_str(raw.get("authority")),
                          consumer=as_str(body.get("consumer")) or None,
                          where=as_str(body.get("where")) or None, body=lean,
                          ledger_id=(as_int(ledger.get("value") if isinstance(ledger, dict)
                                            else ledger) if ledger is not None else None),
                          node=node)
    raise ViewError("not_found", f"record {short(identifier)} is not in session {model.sid}", 404)


def record_detail(model, identifier) -> tuple[RecordDetail, list]:
    for order, entry in enumerate(_entries(model)):
        kind, raw = entry
        if _ident(kind, raw) != identifier:
            continue
        try:
            return _detail(model, entry, order, identifier)
        except Exception as error:  # noqa: BLE001 - refused detail, never a 500
            summary = _placeholder(identifier, "record" if kind == "record" else kind, "", order)
            return _refused_detail(summary, "refused: record body could not be read "
                                   f"({type(error).__name__})"), []
    raise ViewError("not_found", f"record {short(identifier)} is not in session {model.sid}", 404)
