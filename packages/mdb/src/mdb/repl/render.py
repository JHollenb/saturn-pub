"""Op result JSON -> compact text. Deterministic: fixed float format, no timings."""

from __future__ import annotations

import json
from typing import Any

STAT_ORDER = ("norm", "min", "max", "mean", "abs_max")


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.5g}"
        return "0" if text in ("-0", "0") else text
    return str(value)


def tag(authority: str, qualifier: str = "") -> str:
    return f"[{authority}, {qualifier}]" if qualifier else f"[{authority}]"


def qualifier(provenance: dict[str, Any] | None) -> str:
    """Why a value carries its label: ``under edit t3.L0.q``, ``replica of native``,
    ``realised through o``, ``mixed 4 written/12 untouched``, ``untouched, port edit: replace``."""
    if not provenance:
        return ""
    parts = []
    kind = provenance.get("edit_kind")
    if provenance.get("replica_checked") is False:
        at = provenance.get("replica_check_at")
        parts.append(f"replica unchecked until {at}" if at else "replica unchecked")
    elif kind == "replica":
        parts.append("replica of native")
    elif kind == "realisation":
        parts.append("realised through o")
    if provenance.get("mixed"):
        mixed = provenance["mixed"]
        parts.append(f"mixed {mixed['written']} written/{mixed['untouched']} untouched")
    if provenance.get("same_port_edit"):
        parts.append("untouched; port edited: " + ",".join(provenance["same_port_edit"]["ops"]))
    under = provenance.get("under_edits") or []
    if under:
        first = under[0]
        text = f"under edit {first['address']}"
        extra = provenance.get("under_edits_total", len(under)) - 1
        parts.append(text + (f" +{extra} more" if extra > 0 else ""))
    return ", ".join(parts)


def num(item: Any) -> str:
    if isinstance(item, dict) and item.get("state") == "unresolved":
        return f"unresolved ({item['reason']})"
    if isinstance(item, dict) and item.get("state") == "refused":
        return f"refused ({item['reason']})"
    if isinstance(item, dict) and "value" in item:
        return f"{fmt(item['value'])} {tag(item.get('authority', '?'))}"
    return fmt(item)


def _proofs(proofs: list[str]) -> str:
    return "{" + " ".join(proofs) + "}" if proofs else ""


def stop_line(stop: dict[str, Any]) -> str:
    parts = [stop["address"], stop.get("operator", "")]
    if stop.get("proofs"):
        parts.append(_proofs(stop["proofs"]))
    return "  ".join(part for part in parts if part)


def _nonfinite(counts: dict[str, Any]) -> str:
    parts = [f"{counts[k]} {label}" for k, label in (("nan", "nan"), ("posinf", "+inf"),
                                                      ("neginf", "-inf")) if counts.get(k)]
    return ", ".join(parts)


def stats_line(stats: dict[str, Any], why: str = "") -> str:
    present = [name for name in STAT_ORDER if name in stats]
    unresolved = [name for name in present if stats[name].get("state") == "unresolved"]
    tail = ""
    if stats.get("nonfinite"):
        tail = f"  (over finite elements; non-finite: {_nonfinite(stats['nonfinite'])})"
    if unresolved:
        return f"unresolved: {stats[unresolved[0]]['reason']}{tail}"
    authorities = {stats[name]["authority"] for name in present}
    body = " ".join(f"{name} {fmt(stats[name]['value'])}" for name in present)
    if len(authorities) == 1:
        return f"{body} {tag(authorities.pop(), why)}{tail}"
    return " ".join(f"{name} {num(stats[name])}" for name in present) + tail


def _token_text(tokens: list[dict[str, Any]]) -> str:
    out = []
    for token in tokens:
        text = token.get("text")
        out.append(str(token["id"]) if text is None else f"{token['id']}:{json.dumps(text)}")
    return " ".join(out)


def _index(index: list[int]) -> str:
    return "[" + ",".join(str(i) for i in index) + "]"


def reduction_lines(result: dict[str, Any], indent: str = "  ") -> list[str]:
    head = f"{result['address']}  {result['dtype']} {result['shape']}"
    if "selected_shape" in result:
        head += f" -> {result['selected_shape']}"
    if result.get("axes"):
        head += "  (" + ", ".join(result["axes"]) + ")"
    lines = [head]
    provenance = result.get("provenance") or {}
    why = qualifier(provenance)
    if provenance.get("state") == "refused":
        lines.append(f"{indent}label refused: {provenance.get('reason')}")
    if provenance.get("edited"):
        by = f" by {provenance['by']}" if provenance.get("by") else ""
        lines.append(f"{indent}edited{by}: {provenance.get('authority', 'refused')}")
    if provenance.get("realisation_exact") is False:
        lines.append(f"{indent}realisation inexact: max abs error "
                     f"{num(provenance.get('realisation_max_abs_error'))}")
    if "replica_equal" in provenance:
        found = provenance["replica_equal"]
        if isinstance(found, dict):
            lines.append(f"{indent}replica check: unresolved ({found['reason']})")
        else:
            text = "equal" if found else "differs"
            if "replica_max_abs_diff" in provenance:
                text += f" max_abs_diff {num(provenance['replica_max_abs_diff'])}"
            lines.append(f"{indent}replica check: {text}")
    if "margin" in result:
        margin = result["margin"]
        value = margin["value"]
        sides = f"logits[{_token_text([margin['a']])}] - logits[{_token_text([margin['b']])}]"
        lines.append(f"{indent}{sides} = " + (
            f"{fmt(value['value'])} {tag(value['authority'], why)}" if "authority" in value
            else num(value)))
        return lines
    if "tokens" in result:
        total = result.get("tokens_total", len(result["tokens"]))
        more = "" if total == len(result["tokens"]) else f" (last {len(result['tokens'])})"
        label = provenance.get("authority")
        qualified = f" {tag(label, why)}" if label else ""
        lines.append(f"{indent}tokens{more}: {_token_text(result['tokens'])}{qualified}")
    if "stats" in result:
        lines.append(indent + stats_line(result["stats"], why))
    nonfinite = (result.get("stats") or {}).get("nonfinite")
    if nonfinite and not result.get("top"):
        lines.append(f"{indent}top |v|: none (non-finite: {_nonfinite(nonfinite)})")
    if result.get("top") and any("authority" not in e["value"] for e in result["top"]):
        lines.append(f"{indent}top |v|: {num(result['top'][0]['value'])}")
    elif result.get("top"):
        authorities = {entry["value"]["authority"] for entry in result["top"]}
        items = []
        for entry in result["top"]:
            text = f"{_index(entry['index'])} {fmt(entry['value']['value'])}"
            if "text" in entry:
                text += f" {json.dumps(entry['text'])}"
            items.append(text)
        suffix = f" {tag(authorities.pop(), why)}" if len(authorities) == 1 else ""
        lines.append(f"{indent}top |v|: " + ", ".join(items) + suffix)
    if "slice" in result and not isinstance(result["slice"]["values"], list):
        lines.append(f"{indent}first values: {num(result['slice']['values'])}")
    elif "slice" in result:
        values = " ".join(fmt(v) for v in result["slice"]["values"])
        lines.append(f"{indent}first {len(result['slice']['values'])}: {values} "
                     f"{tag(result['slice']['authority'], why)}")
    if "pointer" in result:
        pointer = result["pointer"]
        sha = (pointer["value"].get("sha256") or "")[:16]
        cut = f" cut {pointer['cut'][:12]}" if pointer.get("cut") else ""
        lines.append(f"{indent}pointer {pointer['kind']} ~{pointer['program']} sha256 {sha}{cut}")
    elif "pointer_omitted" in result:
        lines.append(f"{indent}pointer omitted: {result['pointer_omitted']}")
    return lines


def _displays(result: dict[str, Any]) -> list[str]:
    lines = []
    for item in result.get("displays", ()):
        if "unavailable" in item:
            lines.append(f"{item['id']}: {item['address']}  <{item['unavailable']}>")
            continue
        body = reduction_lines(item, indent="     ")
        lines.append(f"{item['id']}: {body[0]}")
        lines.extend(body[1:])
    return lines


def _hit_lines(hits: list[dict[str, Any]]) -> list[str]:
    lines = []
    for hit in hits:
        kind = hit["kind"]
        if kind == "watch":
            line = f"Watch {hit['id']} ({hit['mode']}) {hit['pattern']}"
            if "old" in hit:
                line += f"  old {stats_line(hit['old'], qualifier(hit.get('old_provenance')))}"
            lines.append(line)
            lines.append(f"  new {stats_line(hit['new'], qualifier(hit.get('new_provenance')))}")
        elif kind == "advance":
            lines.append(f"Advanced to {hit['pattern']}")
        else:
            name = "Temporary breakpoint" if kind == "tbreak" else "Breakpoint"
            lines.append(f"{name} {hit['id']}, {hit['pattern']}")
    return lines


def _motion(result: dict[str, Any]) -> list[str]:
    lines = []
    reason = result.get("reason")
    if reason in ("budget", "end-of-context", "start-of-run", "no-breakpoints"):
        lines.append(f"stopped: {reason}")
    elif reason == "error":
        error = result.get("error") or {}
        lines.append(f"stopped: error in breakpoint {error.get('id')}: {error.get('message')}")
    lines.extend(_hit_lines(result.get("hits", ())))
    if result.get("omitted"):
        lines.append(f"... {result['omitted']} more of {result['hits_total']} hits not shown")
    if result.get("note"):
        lines.append(f"note: {result['note']}")
    if result.get("ignored"):
        lines.append(f"ignoring breakpoint {' '.join(map(str, result['ignored']['ids']))} "
                     f"{result['ignored']['count']} more time(s)")
    if "committed" in result:
        lines.append(f"committed {result['committed']} edit(s)"
                     + (f" and {result['committed_cells']} cell edit(s)"
                        if result.get("committed_cells") else ""))
    if result.get("op") == "abort":
        lines.append(f"dropped {result.get('count', 0)} staged edit(s)"
                     + (f" and {result['staged_cells']} cell edit(s)"
                        if result.get("staged_cells") else ""))
    if result.get("op") in ("start", "run") and result.get("run") not in (None, "main"):
        lines.append(f"run {result['run']}")
    if result.get("op") in ("start", "run"):
        lines.extend(_procedure_lines(result))
    lines.append(stop_line(result["stop"]))
    if result.get("focus"):
        lines.append(f"focus {result['focus']}")
    lines.extend(_displays(result))
    return lines


def _edit(edit: dict[str, Any]) -> str:
    op = edit["op"] + (f" {num(edit['dose'])}" if "dose" in edit else "")
    if edit["kind"] == "scalar":
        text = f"{edit['port']} {op} {num(edit['value'])}"
    else:
        text = (f"{edit['port']} {op} tensor {edit['shape']} {edit['dtype']} "
                f"sha256 {edit['sha256'][:12]}")
    if edit.get("origin"):
        origin = edit["origin"]
        text += f" from {origin['origin']}:{origin['address'].split(':', 1)[-1]} ({origin['via']})"
    if edit.get("cell"):
        text += f" [{edit['cell']['timing']} the read]"
    if "realised_exact" in edit:
        if edit["realised_exact"]:
            text += " (realised exactly"
        else:
            text += " (NOT realised exactly"
            if "realised_max_abs_error" in edit:
                text += f", max abs error {num(edit['realised_max_abs_error'])}"
        if edit.get("realised_dtype"):
            text += f" as {str(edit['realised_dtype']).removeprefix('torch.')}"
        if edit.get("realised_at"):
            text += f" via {edit['realised_at']}"
            if edit.get("realisation_exact") is False:  # the cost of going through o
                text += (f"; realisation inexact, max abs error "
                         f"{num(edit.get('realisation_max_abs_error'))}")
        text += ")"
    if edit.get("selection"):
        text += f" on [{edit['selection']}]"
        if "mask_count" in edit:
            text += f" ({edit['mask_count']} elements)"
    if edit.get("fired") is False:
        text += " (not fired yet)"
    if "changed_fraction" in edit:
        text += f" changed {num(edit['changed_fraction'])}"
    if "cast_max_abs_error" in edit and not edit.get("cast_exact", True):
        text += f" cast error {num(edit['cast_max_abs_error'])}"
    if "is_distribution" in edit:
        text += (f" row sums {num(edit.get('row_sum_min'))}..{num(edit.get('row_sum_max'))}"
                 f" tol {num(edit.get('row_sum_tolerance'))}"
                 f" {'a distribution' if edit['is_distribution'] else 'NOT a distribution'}")
    if edit.get("mix"):
        text += f" mix {edit['mix']}"
        if "replica_verified" in edit:
            text += " verified" if edit["replica_verified"] else " UNVERIFIED"
    return text


def _info(result: dict[str, Any]) -> list[str]:
    subject = result["subject"]
    if subject == "program":
        program = result["program"]
        lines = [f"{program['family']} ({program['modeling']}), {program['layers']} layers, "
                 f"d_model {program['hidden_size']}, heads {program['heads']}/"
                 f"{program['kv_heads']} x {program['head_dim']}, vocab {program['vocab_size']}, "
                 f"{program['dtype']} on {program['device']}, attention {program['attention']}"
                 + (f", lookahead {program['lookahead']}" if program.get("lookahead") else "")]
        lines.append("ports: " + " ".join(program["port_names"]))
        lines.append("globals: " + " ".join(program["globals"]))
        return lines
    if subject == "ports":
        return [f"{p['name']:<6}{p['sublayer']:<5}{p['operator']}  {_proofs(p['proofs'])}"
                + ("" if p["editable"] else "  (not editable)") for p in result["ports"]]
    if subject in ("breakpoints", "watch"):
        if not result["items"]:
            if result.get("total"):
                return [_page_note(result)]
            return ["no breakpoints or watches" if subject == "breakpoints" else "no watches"]
        lines = []
        for item in result["items"]:
            extra = ""
            if item["kind"] == "watch":
                extra = f" {item['mode']}"
                if item.get("predicate"):
                    extra += f" if {item['predicate']}"
            elif item.get("condition"):
                extra = f" if {item['condition']}"
            lines.append(f"{item['id']:<4}{item['kind']:<7}{item['pattern']}{extra}  "
                         f"hits {item['hits']}" + _flags(item))
        lines.extend(_more(result))
        return lines
    if subject == "display":
        lines = [f"{d['id']}: {d['address']}" for d in result["displays"]] or ["no displays"]
        return lines + _more(result)
    if subject == "checkpoints":
        return _checkpoints(result["checkpoints"]) + _more(result)
    if subject == "runs":
        return [f"{'*' if r['current'] else ' '} {r['name']:<10}{r['where']}  "
                f"{r['prompt_tokens']} tokens, staged {r['staged']}"
                + (f" + {r['staged_cells']} cell(s)" if r["staged_cells"] else "")
                + "".join(f", {line}" for line in _procedure_lines(r))
                for r in result["runs"]] + _more(result) or ["no runs"]
    lines = [f"staged at {result['where']}: " + (
        "; ".join(_edit(e) for e in result["staged"]) or "none")]
    for entry in result["committed"]:
        more = (f" (+{entry['omitted']} more of {entry['edits_total']})"
                if entry.get("omitted") else "")
        lines.append(f"committed L{entry['layer']} (transition {entry['transition']}): "
                     + "; ".join(_edit(e) for e in entry["edits"]) + more)
    cells = result.get("cells")
    if cells:
        omitted = cells.get("omitted") or {}
        for e in cells["staged"]:
            lines.append(f"staged cell: {_edit(e)}")
        if omitted.get("staged"):
            lines.append(f"staged cell: +{omitted['staged']} more of {cells['staged_total']} "
                         "(use limit= for the rest)")
        for e in cells["committed"]:
            lines.append(f"committed cell (t{e['row']} L{e['layer']} transition "
                         f"{e['transition']}): {_edit(e)}")
        if omitted.get("committed"):
            lines.append(f"committed cell: +{omitted['committed']} more of "
                         f"{cells['committed_total']} (use limit= for the rest)")
    return lines + _more(result)


def _flags(item: dict[str, Any]) -> str:
    flags = []
    if not item.get("enabled", True):
        flags.append("disabled")
    if item.get("ignore"):
        flags.append(f"ignore next {item['ignore']}")
    return f"  ({', '.join(flags)})" if flags else ""


def _page_note(result: dict[str, Any]) -> str:
    return f"(offset {result['offset']} is past the {result['total']} items)"


def _more(result: dict[str, Any]) -> list[str]:
    """A page footer when the list has more than was shown."""
    total, offset = result.get("total", 0), result.get("offset", 0)
    shown = offset + result.get("limit", 0)
    if total > shown or offset:
        return [f"(items {offset + 1}-{min(shown, total)} of {total}; "
                f"use offset= and limit= for the rest)"]
    return []


def _checkpoints(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["no checkpoints"]
    return [f"{c['name']:<10}{c['where']}  staged {c['staged']}, committed {c['committed']}"
            for c in items]


def _frame(stop: dict[str, Any]) -> str:
    return stop_line(stop)


def render(result: dict[str, Any]) -> str:
    """Compact text of one op result; a result whose state is not ``resolved`` says so."""
    text = _render(result)
    if result.get("state", "resolved") != "resolved":
        text += f"\nstate: {result['state']}"
    return text


def _render(result: dict[str, Any]) -> str:
    name = result.get("op")
    if name in ("start", "run", "step", "next", "finish", "until", "advance", "continue",
                "commit", "abort", "restart", "reverse-step", "reverse-next", "reverse-finish",
                "reverse-continue"):
        return "\n".join(_motion(result))
    if name == "where":
        lines = [stop_line(result["stop"])]
        if result.get("focus"):
            lines.append(f"focus {result['focus']}")
        total = result["tokens_total"]
        more = "" if total == len(result["tokens"]) else f" (last {len(result['tokens'])})"
        lines.append(f"tokens{more}: {_token_text(result['tokens'])}")
        lines.append(f"staged edits: {result['staged']}"
                     + (f" + {result['staged_cells']} cell edit(s)"
                        if result.get("staged_cells") else "")
                     + (f", lookahead {result['lookahead']}" if result.get("lookahead") else ""))
        for port in result.get("computed_past_cursor", ()):
            lines.append(f"{port} was computed by the native call but is not held")
        lines.extend(_procedure_lines(result))
        return "\n".join(lines)
    if name == "frame" and "run" in result:
        return f"run {result['run']}\n" + stop_line(result["stop"])
    if name == "frame":
        lines = [f"focus {result['focus']}  (cursor {result['cursor']})"]
        if "frame" in result:
            lines.append(_frame(result["frame"]))
        return "\n".join(lines)
    if name in ("up", "down"):
        line = _frame(result["frame"])
        if result.get("computed") is False:
            line += "  (not computed yet)"
        if result.get("alternatives"):
            line += "\n  also produced by: " + ", ".join(result["alternatives"])
        return line
    if name == "print":
        return "\n".join(reduction_lines(result))
    if name == "x":
        head = (f"{result['address']}  [{result['offset']}:{result['offset'] + result['count']}]"
                f" of {result['numel']}")
        if "values" in result:
            values = result["values"]
            if values.get("state") in ("unresolved", "refused"):
                return f"{head}: {num(values)}"
            text = f"{head}: " + " ".join(fmt(v) for v in values["values"]) + \
                f" {tag(values['authority'], qualifier(result.get('provenance')))}"
            if result.get("nonfinite"):
                text += f"  (non-finite: {_nonfinite(result['nonfinite'])})"
            return text
        return f"{head}: {_token_text(result['tokens'])}"
    if name == "list":
        lines = [f"L{result['layer']} at row {result['row']}  (cursor {result['where']})"]
        for row in result["ports"]:
            mark = "=>" if row["cursor"] else ("*" if row["focus"] else " ")
            state = "" if row["computed"] else "  ."
            lines.append(f"{mark:>2} {row['port']:<6}{row['operator']}  {_proofs(row['proofs'])}"
                         f"{state}")
        for check, value in result.get("checks", {}).items():
            text = "equal" if value.get("equal") else "differs"
            if "max_abs_diff" in value:
                text += f" max_abs_diff {num(value['max_abs_diff'])}"
            lines.append(f"   replica check {check}: {text}")
        return "\n".join(lines)
    if name == "backtrace":
        lines = [result["chain"]]
        for frame in result["frames"]:
            inputs = frame.get("inputs")
            reads = f"  <- {','.join(inputs)}" if inputs else ""
            if frame.get("reads"):
                reads += (" + " if inputs else "  <- ") + ",".join(frame["reads"])
            lines.append(f"#{frame['level']:<2} {_frame(frame)}{reads}")
        return "\n".join(lines)
    if name in ("break", "tbreak"):
        item = result["breakpoint"]
        cond = f" if {item['condition']}" if item["condition"] else ""
        label = "Temporary breakpoint" if item["kind"] == "tbreak" else "Breakpoint"
        return f"{label} {item['id']} at {item['pattern']}{cond}"
    if name == "watch":
        item = result["watch"]
        pred = f" if {item['predicate']}" if item["predicate"] else ""
        return f"Watch {item['id']} ({item['mode']}) on {item['pattern']}{pred}"
    if name in ("delete", "undisplay"):
        text = "deleted " + (" ".join(str(i) for i in result["deleted"]) or "nothing")
        if result.get("omitted"):
            text += f" ... ({result['deleted_total']} in all)"
        return text
    if name in ("enable", "disable"):
        text = f"{name}d " + (" ".join(str(i) for i in result["ids"]) or "nothing")
        if result.get("omitted"):
            text += f" ... ({result['total']} in all)"
        return text
    if name in ("condition", "ignore"):
        item = result.get("breakpoint") or result["watch"]
        if name == "ignore":
            return (f"Will ignore next {result['ignore']} crossings of breakpoint {item['id']}."
                    if result["ignore"] else f"Will stop next time breakpoint {item['id']} is "
                    "reached.")
        what = item.get("condition") if "condition" in item else item.get("predicate")
        return (f"Breakpoint {item['id']} now unconditional." if what is None
                else f"Breakpoint {item['id']}: if {what}")
    if name == "display":
        return "\n".join(_displays(result)) or "no displays"
    if name == "set" and "cell" in result:
        cell = result["cell"]
        lines = [f"staged cell {result['target']}: "
                 + "; ".join(_edit(e) for e in result["edits"])
                 + f"  ({result['staged_cells']} cell edit(s) staged)",
                 f"  realised {cell['timing']} {cell['read']} reads it (boundary "
                 f"{cell['boundary']}); reads already made are not recomputed"]
        if "source" in result:
            lines.append(f"  value from {result['source']}")
        return "\n".join(lines)
    if name == "set":
        lines = [f"staged at {result['target']}: " + "; ".join(_edit(e) for e in result["edits"])
                 + f"  ({result['staged']} staged)"]
        if "source" in result:
            lines.append(f"  value from {result['source']}")
        if "realised_as" in result:
            lines.append(f"  realised as {result['realised_as']}")
        return "\n".join(lines)
    if name == "checkpoint":
        return f"checkpoint {result['name']} at {result['where']} ({result['bytes']} bytes)"
    if name == "checkpoints":
        return "\n".join(_checkpoints(result["checkpoints"]) + _more(result))
    if name == "info":
        return "\n".join(_info(result))
    if name in _PREDICT:
        return "\n".join(_PREDICT[name](result))
    if name in _INSPECT:
        return "\n".join(_INSPECT[name](result))
    return json.dumps(result, sort_keys=True)


# --------------------------------------------------------------------------- explain/forecast/
# verify/whatif


def _claimed(item: Any) -> str:
    """A labelled number with its proof (``0.25 [derived-local P14]``), a labelled
    ``{max_abs, norm}`` pair, or the state and reason of a field that has no number."""
    if isinstance(item, dict) and item.get("state") in ("unresolved", "refused"):
        return num(item)
    if isinstance(item, dict) and "max_abs" in item and "value" not in item:
        return f"max_abs {_claimed(item['max_abs'])} norm {_claimed(item['norm'])}"
    proof = item.get("proof")
    return f"{fmt(item['value'])} [{item['authority']}{' ' + proof if proof else ''}]"


def _explain_lines(result: dict[str, Any]) -> list[str]:
    data = result["explain"]
    if data.get("state") == "refused":
        return [f"explain {result['target']}: refused ({data['reason']})"]
    if result["kind"] == "margin":
        lines = [f"explain {result['target']}  native {_claimed(data['native_margin'])}  "
                 f"float64 {_claimed(data['float64_margin'])}  eta {_claimed(data['eta'])}"]
        for term in data["terms"]:
            lines.append(f"  {term['name']:<19}{_claimed(term['value'])}")
        summary = data.get("terms_summary")
        if summary and summary["rest"]["count"]:
            rest = summary["rest"]
            lines.append(f"  +{rest['count']} more, sum = {_claimed(rest['sum'])}"
                         f"  (of {summary['count']} terms; full list "
                         f"{summary['pointer']['state']}, sha256 {summary['pointer']['sha256'][:16]})")
        for rem in data["remainders"]:
            line = f"  ~{rem['name']:<18}{_claimed(rem['value'])}"
            if "scale" in rem:
                line += (f"  scale {_claimed(rem['scale'])} k {rem['k']} "
                         f"{'within' if rem['within_rounding'] else 'OUTSIDE'}")
            elif rem.get("bound", {}).get("state") == "refused":
                line += f"  bound refused ({rem['bound']['reason']})"
            lines.append(line)
        lines.append(f"  listed_sum {_claimed(data['listed_sum'])}  closure "
                     f"{_claimed(data['closure'])}")
    else:
        lines = [f"explain {result['target']}  {data['function']} {data['proof']}: "
                 f"{data['operator']}  {_proofs(data['proofs_declared'])}"]
        if "eta_max_abs" in data:
            lines.append(f"  eta_max_abs {_claimed(data['eta_max_abs'])}  eta_rel "
                         f"{_claimed(data['eta_rel'])}  (against {data['against']})")
        if "edit_delta_max_abs" in data:
            lines.append(f"  written by an edit ({data['edit']['by']}): edit_delta_max_abs "
                         f"{_claimed(data['edit_delta_max_abs'])}")
        lines += [f"  note: {note}" for note in data.get("notes", ())]
    if data.get("state") not in (None, "resolved"):
        lines.append(f"  {data['state']}: {data.get('reason')}")
    return lines


def _predicted(body: dict[str, Any]) -> str:
    if "token" in body:
        return f"token {body['token']}"
    if "value" in body:
        return _claimed(body["value"])
    return f"{_claimed(body)}  sha256 {body['sha256'][:16]}"


def _forecast_lines(result: dict[str, Any]) -> list[str]:
    view = result["forecast"]
    summary = view["method_summary"]
    count, first, last = summary["count"], summary["first"][0], summary["last"][-1]
    method = (f"{last['identity']} {last['proof'] or ''}: {last['operator']}" if count == 1
              else f"{count} identities {first['stop']} .. {last['stop']} (last "
                   f"{last['identity']} {last['proof'] or ''})")
    pointer = summary["chain_pointer"]
    method += (f"; chain blob {pointer['ref'][:16]}" if pointer["state"] == "stored"
               else "; chain not stored (no project)")
    bound = view["bound"]
    vector = "sha256" in view["predicted"]
    spread = (f"bound {_claimed(bound['max'])}{' per element' if vector else ''}"
              if "max" in bound else f"bound {num(bound)}")
    return [f"forecast {view['id']}  {view['target']}  sealed at {view['sealed_at']}  "
            f"record {view['record'][:16]}",
            f"  method {method}",
            f"  predicted {_predicted(view['predicted'])}  {spread}"]


def _floor_text(item: dict[str, Any]) -> str:
    """``residual 0 = 0 spacings <= floor 0.0156 = 1 bfloat16 spacing (supplied)``: the residual
    native - emulated in dtype spacings next to the floor in spacings."""
    text = ""
    if "residual_spacings" in item:
        text += f" = {item['residual_spacings']['value']:.2g} spacings"
    text += f" {'<=' if item['verdict'] == 'confirmed' else '>'} floor {_claimed(item['floor'])}"
    if "floor_spacings" in item:
        count = item["floor_spacings"]["value"]
        text += (f" = {count:g} {item.get('dtype', '?')} spacing{'' if count == 1 else 's'} "
                 f"({item['floor_spacings']['authority']})")
    return text


def _verify_lines(result: dict[str, Any]) -> list[str]:
    if not result["forecasts"]:
        return [f"no open forecasts (cursor {result['where']})"]
    lines = []
    for item in result["forecasts"]:
        head = f"forecast {item['id']} {item['target']}: {item['verdict']}"
        if item["verdict"] == "open":
            lines.append(f"{head} ({item.get('reason', 'target not reached')})")
            continue
        if item["verdict"] == "edited":
            lines.append(f"{head} ({'; '.join(item['edited_after_seal'])}): not a forecast "
                         "failure")
        else:
            text = f"{head}  native {_claimed(item['native'])}" if "native" in item else head
            if "residual_max_abs" in item:
                text += f"  residual vs emulated {_claimed(item['residual_max_abs'])}"
            if "floor" in item:
                text += _floor_text(item) + f"  [P20, {item.get('state', 'resolved')}]"
            if "reason" in item:
                text += f"  ({item['reason']})"
            lines.append(text)
            effect = item.get("rounding_effect")
            if effect is not None:
                text = f"  rounding effect native - exact {_claimed(effect['max_abs'])}"
                if "spacings" in effect:
                    text += f" = {effect['spacings']['value']:.3g} spacings"
                if "bound_max" in item:
                    text += (f"; worst-case bound {_claimed(item['bound_max'])} "
                             f"({'holds' if item['within_bound'] else 'exceeded'}, information)")
                    if "worst_excess" in item:
                        text += f" (worst excess {_claimed(item['worst_excess'])})"
                lines.append(text)
            if "rounding_origin" in item:
                lines.append(f"  rounding origin (sealed): {item['rounding_origin']['text']}")
        order = item["order"]
        lineage = ("same cut (the target is a later port of the sealed layer)"
                   if order["same_cut"] else
                   f"cut {'descends' if order['descends'] else 'does not descend'} through "
                   f"{order['receipts_after_seal']['count']} receipts")
        lines.append(f"  order: sealed at {item['sealed_at']} (stop {order['sealed']['index']}) "
                     f"{'before' if order['sealed_before_target'] else 'NOT before'} the target "
                     f"(stop {order['target']['index']}); {lineage}")
    return lines


def _delta_name(consumer: dict[str, Any]) -> str:
    return {"margin": "\u0394M", "logit": "\u0394logit"}.get(consumer["kind"], "\u0394")


def _whatif_lines(result: dict[str, Any]) -> list[str]:
    edit, consumer, delta = result["edit"], result["consumer"], _delta_name(result["consumer"])
    value = (_claimed(edit["value"]) if "value" in edit
             else f"tensor {edit['sha256'][:12]}" + (f" from {edit['source']}"
                                                    if "source" in edit else ""))
    where = edit["address"] + (f"[{edit['selection']}]"
                               if "selection" in edit and "cell" not in edit else "")
    lines = [f"whatif {where} {edit['op']} {value} -> {consumer['address']}  "
             f"(mode {result['mode']}, record {result['record'][:16]})"]
    prediction = result["prediction"]
    if prediction["state"] == "refused":
        lines.append(f"  sealed  prediction refused ({prediction['reason']})")
    else:
        linear = prediction["linear"]
        exact = prediction["exact"]
        bound = f"  bound {_claimed(exact['bound'])}" if "bound" in exact else ""
        operands = exact.get("operands", {})
        if operands.get("source") == "forecast":
            bound += f"  (operands forecast from {operands['at']})"
        emulated = prediction.get("emulated", {})
        lines.append(f"  sealed  linear {delta} = "
                     f"{_claimed(linear['value']) if 'value' in linear else num(linear)}   "
                     f"exact {delta} = {_claimed(exact['value'])}{bound}   emulated {delta} = "
                     + (_claimed(emulated["value"]) if "value" in emulated else num(emulated)))
    measured = result["measured"]
    lines.append(f"  native  {delta} = {_claimed(measured['delta'])}   (base "
                 f"{_claimed(measured['base'])} -> edited {_claimed(measured['edited'])})")
    split = result["split"]
    if split.get("state") == "refused":
        lines.append("  error split: none (no prediction)")
    else:
        parts = []
        for term in split["terms"]:
            text = f"{term['name']} {_claimed(term['value'])}"
            if term.get("within_bound") is not None:
                state = term["within_bound"]
                text += f" ({'within' if state else 'OUTSIDE'} bound {_claimed(term['bound'])})"
            parts.append(text)
        lines.append(f"  error split ({split['proof']}): " + " + ".join(parts))
    if "effect" in result:
        lines.append("  " + effect_line(result["effect"], delta,
                                        exact_note=" (written value, measured operands)"))
    session = result["session"]
    lines.append(f"  session {'unchanged' if session['unchanged'] else 'CHANGED'} at "
                 f"{result['where']}  ({len(result['measured']['moves'])} moves, restored)")
    return lines


def effect_line(effect: dict[str, Any], delta: str = "\u0394", exact_note: str = "") -> str:
    """The P20 verdict: native against the emulated change within the floor (1 spacing,
    supplied), the exact change beside it; ``exact_note`` says which exact change it is."""
    numbers = f"native {delta} {_claimed(effect['native'])}"
    if "emulated" in effect:
        numbers += (f", emulated {delta} {_claimed(effect['emulated'])}, residual "
                    f"{_claimed(effect['residual'])} = "
                    f"{effect['residual_spacings']['value']:.2g} spacings")
    if "exact" in effect:
        numbers += f"; exact {delta}{exact_note} {_claimed(effect['exact'])}"
    if "emulated" not in effect:
        numbers += f" ({effect['basis']})"
    count = effect["floor_spacings"]["value"]
    floor = (f"floor {_claimed(effect['floor'])} = {count:g} {effect['dtype']} "
             f"spacing{'' if count == 1 else 's'} ({effect['floor_spacings']['authority']})")
    text = f"verdict (P20): {effect['verdict']}  {numbers}; {floor}"
    for key in ("note", "reason"):
        if effect.get(key):
            text += f"  [{effect[key]}]"
    return text


_PREDICT = {"explain": _explain_lines, "forecast": _forecast_lines, "verify": _verify_lines,
            "whatif": _whatif_lines}


# --------------------------------------------------------------------------- reads/symbols/
# circuit/compare/seek


def _short(item: Any) -> str:
    """A labelled scalar, or the norm of a labelled vector, or a state and reason."""
    if isinstance(item, dict) and "norm" in item and "value" not in item:
        return f"norm {_claimed(item['norm'])}"
    return _claimed(item)


def _verdict(causal: dict[str, Any]) -> str:
    if "delta" not in causal:
        return f"unresolved ({causal['reason']})"
    head = causal["verdict"] + (f" ({causal['reason']})" if causal.get("reason") else "")
    text = f"{head}: Δ {_short(causal['delta'])}"
    effect = causal.get("effect") or {}
    emulated = causal.get("emulated")
    if isinstance(emulated, dict) and emulated.get("state") is None:
        text += f", emulated {_short(emulated)}"
    if "residual" in effect:
        text += (f", residual {_claimed(effect['residual'])} = "
                 f"{effect['residual_spacings']['value']:.2g} spacings")
    predicted = causal.get("predicted")
    if isinstance(predicted, dict) and predicted.get("state") is None:
        text += f"; exact {_short(predicted)}"
    if "floor" in effect:
        text += f"; floor {_claimed(effect['floor'])}"
    return text


def _reads_lines(result: dict[str, Any]) -> list[str]:
    source = result["source"]
    lines = [f"reads {source['address']} at {result['where']}  rows "
             f"{','.join(f't{r}' for r in result['rows'])}  consumer {result['consumer']}  "
             f"({result['total']} reads)"]
    for entry in result["reads"]:
        lines.append(f"  {entry['name']:<10} mass {_claimed(entry['mass'])}  effect "
                     f"{_short(entry['to_consumer'])}")
        lines.append(f"    {_verdict(entry['causal'])}")
    if "omitted" in result:
        omitted = result["omitted"]
        lines.append(f"  +{omitted['count']} more: mass Σ {_claimed(omitted['mass'])}  "
                     f"effect Σ {_short(omitted['to_consumer'])}")
    for name, check in result["checks"].items():
        if check.get("state"):
            continue
        lines.append(f"  {name}: eta_max_abs {_claimed(check['eta_max_abs'])} worst |eta|/bound "
                     f"{_claimed(check['worst_ratio'])} "
                     f"{'within' if check['within_bound'] else 'OUTSIDE'} bound "
                     f"({check['count']} layer rows)")
    lines += [f"  not computed: {item}" for item in result.get("not_computed", ())]
    return lines


def _display_width(text: str) -> int:
    """Terminal columns: wide characters count 2, combining marks 0."""
    import unicodedata

    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF"
               else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(width - _display_width(text), 0)


def _symbol_name(item: dict[str, Any]) -> str:
    """A binding's name; token text is quoted with escapes so spaces and newlines show."""
    name = item["symbol"]
    if "text" in item:
        name += f" {json.dumps(item['text'], ensure_ascii=False)}"
    elif "token" in item:
        name += f" (id {item['token']})"
    return name


def _symbols_lines(result: dict[str, Any]) -> list[str]:
    head = f"symbols at {result['where']}  consumer {result['consumer']}"
    if "consumer_total" in result:
        head += f" = {_claimed(result['consumer_total'])}"
    lines = [head + f"  min_mass {_claimed(result['min_mass'])}",
             "  mass: a sum over the heads and layers that read the binding (n reads, mean per "
             "read)",
             "  life: " + "  ".join(f"{mark} {text}" for mark, text in result["legend"].items())]
    names = [_symbol_name(item) for item in result["bindings"]]
    names += [item["symbol"] for item in result.get("unresolved_symbols", ())] + ["Unbound"]
    width = max(len(n) if n.isascii() else _display_width(n) for n in names) + 1
    for item, name in zip(result["bindings"], names):
        life = item["lifetime"]
        span = f" ({life['first']}..{life['last']})" if life["first"] else ""
        lines.append(f"  {_pad(name, width)}{item['status']:<7}mass {_claimed(item['mass'])} over "
                     f"{item['count']} reads (mean {_short(item['mean_mass'])})  to_consumer "
                     f"{_short(item['to_consumer'])}  life {life['strip']}{span}")
        lines.append(f"    defined {', '.join(item['definitions'])}")
        readers = "; ".join(f"{r['name']} mass {_claimed(r['mass'])} effect "
                            f"{_short(r['effect'])}" for r in item["readers"])
        more = f" (+{item['omitted']} more)" if item.get("omitted") else ""
        lines.append(f"    readers {readers or 'none'}{more}")
        if item.get("edited_by"):
            lines.append(f"    edited by {', '.join(item['edited_by'])}")
    for item in result.get("unresolved_symbols", ()):
        lines.append(f"  {_pad(item['symbol'], width)}unresolved  defined {', '.join(item['definitions'])}"
                     f"  ({item['reason']})")
    unbound = result["unbound"]
    lines.append(f"  {_pad('Unbound', width)}mass {_short(unbound['mass'])}  readout "
                 f"{_short(unbound['readout'])}")
    parts = unbound.get("parts")
    if parts:
        remainder = parts["remainder"]
        text = (f"    = embedding+mlp {_claimed(parts['embedding_mlp'])} + attention not "
                f"listed {_claimed(parts['attention_not_listed'])}")
        if "residual_edits" in parts:
            text += f" + residual edits {_claimed(parts['residual_edits'])}"
        if remainder["within"] is None:
            tail = "bound refused (accumulation premise fails)"
        else:
            tail = (f"bound {_claimed(remainder['bound'])}: "
                    f"{'within' if remainder['within'] else 'OUTSIDE'}")
        lines.append(text + f" + remainder {_claimed(remainder['value'])}, {tail}")
    accounting = result["accounting"]
    partition = accounting["partition"]
    split = (f"partition gap {_claimed(partition['max_gap'])}" if "max_gap" in partition
             else f"partition {num(partition)}")
    lines.append(f"  accounting: {accounting['heads']} heads, listed mass "
                 f"{_claimed(accounting['listed_mass'])}, max |sum p - 1| "
                 f"{_claimed(accounting['max_head_deviation'])} "
                 f"{'<=' if accounting['within_bound'] else '>'} "
                 f"{_claimed(accounting['deviation_bound'])}, {split}")
    page = result["page"]
    shown = len(result["bindings"])
    lines.append(f"  (bindings {page['offset'] + 1}-{page['offset'] + shown} of {page['total']})"
                 if shown else f"  (no bindings at offset {page['offset']} of {page['total']})")
    lines += [f"  not computed: {item}" for item in result.get("not_computed", ())]
    if result.get("stored"):
        lines.append(f"  stored {len(result['stored'])} record(s): {result['stored'][0][:16]}")
    return lines


def _walk(node: dict[str, Any], depth: int = 0, marks: dict[str, str] | None = None,
          texts: dict[str, str] | None = None, circuit: Any = None) -> list[str]:
    marks, texts = marks or {}, texts or {}
    name = node["name"] if node["name"] == node["address"] else (
        f"{node['name']}  {node['address']}")
    if node["id"] in texts:
        name += f" {json.dumps(texts[node['id']])}"
    line = f"{'  ' * depth}{name}  {_short(node['value'])}"
    if "bound" in node:
        line += f" {'within' if node['within'] else 'OUTSIDE'} {_short(node['bound'])}"
    if node.get("unexamined"):
        line += (f"  (not examined, {int(node['unexamined_count']['value'])} subtree(s) "
                 "not expanded)")
    if node["id"] in marks:
        line += f"  [patched in: {'ko' if marks[node['id']] == 'retained' else 'keep'}]"
    lines = [line]
    for child in node.get("children", ()):
        lines += _walk(child, depth + 1, marks, texts, circuit)
    paged = node.get("paged")
    if paged:
        lines.append(f"{'  ' * (depth + 1)}[paged: {_claimed(paged['children'])} children examined, "
                     f"{_claimed(paged['descendants'])} nodes below, stored as page "
                     f"{paged['pointer']['sha256'][:12]}; circuit show {circuit} "
                     f"node={node['id']}]")
    return lines


def _count(item: Any) -> str:
    """A labelled count's value (its label is derived-exact or measured; the caller says which)."""
    return str(int(item["value"]))


def _circuit_test_lines(test: dict[str, Any]) -> list[str]:
    """``circuit test ID ref=NAME``: the reference, the component counts, the four measured
    margins, the two ratios (or why they are refused), the one-spacing readings and the native
    layer calls per pass (wall time is in the result, not printed)."""
    ref = test["reference"]
    where = (f"run, row t{_count(ref['row'])}" if ref["kind"] == "run"
             else f"mean over {_count(ref['items'])} items at their @last")
    lines = [f"circuit {_count(test['id'])} test  {test['consumer']}  "
             f"ref {ref['name']} ({where})"]
    comp = test["components"]
    count = {key: _count(comp[key]) for key in
             ("total", "circuit", "complement", "from_groups", "unexamined")}
    coarse = _count(comp["coarsened"]["source_nodes"])
    lines.append(f"  components [derived-exact] {count['total']}: circuit {count['circuit']}, "
                 f"complement {count['complement']} ({count['from_groups']} from +N more groups, "
                 f"{count['unexamined']} of them never expanded); {coarse} source node(s) "
                 "counted as their head")
    lines.append(f"  {test['components']['rows']}")
    margins = test["margins"]
    for key in ("base", "ref", "ko", "keep", "full"):
        extra = ""
        if key == "full" and "full_equals_ref" in test:
            extra = "  (= ref bit for bit)" if test["full_equals_ref"] else "  (differs from ref)"
        lines.append(f"  {key:<5} {_claimed(margins[key])}  {test['meaning'][key]}{extra}")
    ratios = test["ratios"]
    if ratios.get("state"):
        lines.append(f"  ratios {num(ratios)}")
    else:
        for key, label in (("faithfulness", "faithfulness"), ("knockout_share",
                                                               "knock-out share")):
            lines.append(f"  {label} {ratios['formulas'][key]} = {_claimed(ratios[key])}: "
                         f"{ratios['meaning'][key]}")
    reading, res = test["reading"], test["resolution"]
    lines.append(f"  reading: knock-out {reading['knockout']}; {reading['keep_vs_base']}; "
                 f"{reading['keep_vs_ref']} (one spacing {_claimed(res['spacing'])} "
                 f"{res['dtype']})")
    passes = test["cost"]["per_pass"]
    lines.append("  layer calls completed [measured] " + "  ".join(
        f"{key} {_count(passes[key]['calls']['completed'])}"
        for key in ("ko", "keep", "full")) + "; edits [derived-exact] " + "  ".join(
        f"{key} {_count(passes[key]['edits'])}" for key in ("ko", "keep", "full")))
    return lines


def _circuit_lines(result: dict[str, Any]) -> list[str]:
    if result["action"] == "test":
        lines = _circuit_test_lines(result["test"])
        if result.get("stored"):
            lines.append(f"  stored record {result['stored'][0][:16]}")
        return lines
    if "page" in result:
        page = result["page"]
        lines = [f"circuit {result['id']} node {result['node']}  "
                 f"({'inline subtree' if page.get('state') == 'inline' else 'stored page'})"]
        marks = {}
        lines += ["  " + line for line in _walk(page["node"], 0, marks, result.get("token_text"),
                                                result["id"])]
        return lines
    body = result["record"]["body"]
    lines = [f"circuit {result['id']}  {body['consumer']}  top {body['k']} depth "
             f"{body['depth']}  ({body['count']} nodes, record {result['record']['id'][:16]})"]
    marks = {ref: name for name, refs in result.get("ablated", {}).items() for ref in refs}
    lines += ["  " + line for line in _walk(body["root"], 0, marks, result.get("token_text"),
                                            result["id"])]
    closure = body["closure"]
    lines.append(f"  closure {_short(closure['value'])} "
                 f"{'within' if closure['within'] else 'OUTSIDE'} {_claimed(closure['bound'])} "
                 f"(zero by construction: float64 summation only)")
    within = body["remainders_within"]
    state = ("all within their rounding bounds" if within
             else "NOT all within their rounding bounds" if within is not None
             else "not all bounded (a rounding bound was refused)")
    if body.get("unexamined") and within is not False:
        state = (f"{int(body['unexamined_count']['value'])} dropped subtree(s) not examined; no claim "
                 "about their rounding bounds")
    lines.append(f"  remainders: {state}; {body['edits']['count']} edit node(s)")
    for test in result.get("tests", ()):
        lines += ["  " + line for line in _circuit_test_lines(test)]
    if result.get("stored"):
        lines.append(f"  stored record {result['stored'][0][:16]}")
    return lines


def _compare_lines(result: dict[str, Any]) -> list[str]:
    measured, effects = result["measured"], result["effects"]
    lines = [f"compare A: {result['sides']['A']}  B: {result['sides']['B']}  -> "
             f"{result['consumer']}"]
    lines.append("  " + "  ".join(f"{name} {_short(item.get('value', item))}"
                                  for name, item in measured.items()))
    lines.append("  effects " + "  ".join(f"{name} {_short(value)}"
                                          for name, value in effects.items()))
    verdicts = result.get("verdicts") or {}
    if verdicts:
        lines.append("  verdicts (native only, no exact or emulated chain: effect when edited and "
                     "base differ): " + "  ".join(f"{name} {v['verdict']}"
                                                  for name, v in verdicts.items()))
    interaction = result["interaction"]
    if interaction.get("state"):
        lines.append(f"  interaction {num(interaction)}")
    else:
        lines.append(f"  interaction {interaction['formula']} = {_short(interaction['value'])}")
    if "order" in result:
        lines.append(f"  order: {result['order']}")
    session = result["session"]
    lines.append(f"  session {'unchanged' if session['unchanged'] else 'CHANGED'} at "
                 f"{result['where']}")
    return lines


def _symbol_lines(result: dict[str, Any]) -> list[str]:
    return [f"symbol {result['symbol']} = {', '.join(result['definitions'])}  (declared: "
            f"{', '.join(result['declared'])})"]


def _seek_lines(result: dict[str, Any]) -> list[str]:
    lines = [f"seek {result['target']} via {result['via']}: " + ("; ".join(result["moves"])
                                                                 or "already there")]
    if result.get("dropped_edits"):
        lines.append(f"  dropped edits: {', '.join(result['dropped_edits'])}")
    if result.get("restaged_edits"):
        lines.append(f"  committed edits staged again: {', '.join(result['restaged_edits'])}")
    return lines + _motion(result)


_INSPECT = {"reads": _reads_lines, "symbols": _symbols_lines, "symbol": _symbol_lines,
            "circuit": _circuit_lines, "compare": _compare_lines, "seek": _seek_lines}


def render_error(error: Any) -> str:
    data = error.to_json() if hasattr(error, "to_json") else {"type": "error",
                                                              "message": str(error)}
    return f"error ({data['type']}): {data['message']}"


# --------------------------------------------------------------------------- each / prompt sets


def _cell(item: Any, labelled: bool, digits: int = 5) -> str:
    if isinstance(item, dict) and item.get("state") in ("unresolved", "refused"):
        return item["state"]
    if digits != 5 and isinstance(item["value"], float) and not isinstance(item["value"], bool):
        text = f"{item['value']:.{digits}g}"
        value = "0" if text in ("-0", "0") else text
        return f"{value} {tag(item.get('authority', '?'))}" if labelled else value
    return num(item) if labelled else fmt(item["value"])


def _each_table(rows: list[dict[str, Any]], columns: list[str], uniform: dict[str, str],
                digits: int) -> list[tuple[list[str], str | None]]:
    """The table as (cells, refusal text) rows, header first; a refused row has its name only
    and the refusal text, which does not size any column."""
    has_label = any("label" in row.get("key", {}) for row in rows)
    has_verdict = any("verdict" in row.get("key", {}) for row in rows)
    table: list[tuple[list[str], str | None]] = [
        (["item"] + (["read"] if has_label else []) + columns
         + (["verdict"] if has_verdict else []), None)]
    for row in rows:
        if row["state"] == "refused":
            table.append(([row["name"]], f"refused: {row['reason']}"))
            continue
        key = row.get("key", {})
        cells = [row["name"]] + ([key.get("label", "-")] if has_label else [])
        for name in columns:
            item = key.get("columns", {}).get(name)
            cells.append("-" if item is None else _cell(item, name not in uniform, digits))
        if has_verdict:
            verdict = key.get("verdict", "")
            if row["state"] != "resolved":
                verdict = f"{verdict} ({row['state']})".strip()
            cells.append(verdict)
        table.append((cells, None))
    return table


def _table_lines(table: list[tuple[list[str], str | None]]) -> tuple[list[str], int]:
    """Aligned lines (every column as wide as its widest cell by display width, two spaces
    between, the last column unpadded) and the width of the widest line."""
    count = len(table[0][0])
    widths = [max(_display_width(cells[i]) for cells, refusal in table
                  if refusal is None or i == 0) for i in range(count)]
    out, widest = [], 0
    for cells, refusal in table:
        if refusal is not None:
            line = "  " + _pad(cells[0], widths[0]) + "  " + refusal
        else:
            line = "  " + "  ".join(_pad(c, widths[i]) if i < count - 1 else c
                                    for i, c in enumerate(cells)).rstrip()
            widest = max(widest, _display_width(line))
        out.append(line)
    return out, widest


def _stat_lines(stat: dict[str, Any]) -> list[str]:
    def body(item: dict[str, Any]) -> str:
        agree = item["agree"]
        return (f"n {item['n']}  mean {fmt(item['mean']['value'])}  min "
                f"{fmt(item['min']['value'])}  max {fmt(item['max']['value'])}  sign "
                f"{agree['sign']} {agree['count']}/{item['n']}")

    if stat.get("mixed"):
        lines = [f"  {stat['path']}  (mixed labels, summarised per label)"]
        for label, item in stat["by_authority"].items():
            lines.append(f"    {tag(label)} {body(item)}")
        return lines
    return [f"  {stat['path']}  {tag(stat['mean']['authority'])}", f"    {body(stat)}"]


def _each_lines(result: dict[str, Any]) -> list[str]:
    """``each``: one row per item (name, key numbers, verdict), then the summary lines."""
    rows = result["items"]
    lines = [f"each {result['set']}: {result['line']}",
             f"  {result['total']} items at {result['at']}; runs {result['runs'].split(' ')[0]}; "
             f"result {result['id']}"]
    columns: list[str] = []
    for row in rows:
        for name in row.get("key", {}).get("columns", {}):
            if name not in columns:
                columns.append(name)
    labels: dict[str, set[str]] = {name: set() for name in columns}
    for row in rows:
        for name, item in row.get("key", {}).get("columns", {}).items():
            labels[name].add(item.get("authority", item.get("state", "?")))
    uniform = {name: next(iter(found)) for name, found in labels.items() if len(found) == 1}
    for digits in (5, 4, 3):       # shorten only numbers until the common case fits 80 columns
        table_lines, widest = _table_lines(_each_table(rows, columns, uniform, digits))
        if widest <= 80 or digits == 3:
            break
    lines += table_lines
    if uniform:
        lines.append("  labels: " + ", ".join(f"{name} {uniform[name]}" for name in columns
                                              if name in uniform))
    statistics = result["statistics"]
    lines.append(f"summary: {statistics['exclusion']}")
    numeric = statistics["numeric"]
    shown = numeric[:len(columns) or 2]
    for stat in shown:
        lines += _stat_lines(stat)
    verdicts = [item for item in statistics["verdicts"]
                if item["path"].rsplit(".", 1)[-1] == "verdict" or len(item["counts"]) > 1
                or item["counts"][0]["label"] != "resolved"]
    for item in verdicts[:4]:
        counts = " / ".join(f"{c['label']} {c['count']}" for c in item["counts"])
        lines.append(f"  {item['path']}: {counts}")
    more = (len(numeric) - len(shown) + len(statistics["verdicts"]) - len(verdicts[:4])
            + statistics["omitted"]["count"])
    partial = statistics["partial"]["count"]
    tail = []
    if more:
        tail.append(f"+{more} more in the record")
    if partial:
        tail.append(f"{partial} paths in some items only")
    if tail:
        lines.append("  " + "; ".join(tail) + f" (promptset result {result['id']} [ITEM])")
    if "stored" in result:
        stored = result["stored"]
        lines.append(f"  stored record {stored[:16]}" if isinstance(stored, str)
                     else f"  not stored: {stored['reason']}")
    return lines


def _promptset_lines(result: dict[str, Any]) -> list[str]:
    if result["action"] == "list":
        return [f"{item['set']:<12}{item['total']} items  ({item['source']})"
                for item in result["sets"]] or ["no prompt sets"]
    if result["action"] == "result":
        if "result" in result:
            return [f"{result['id']} item {result['item']}:", _render(result["result"])]
        return _each_lines({**result["summary"], "id": result["id"]})
    lines = [f"set {result['set']}: {result['total']} items  ({result['source']})"]
    shown = result["items"]
    if result["action"] in ("load", "define") and len(shown) > 10:   # show prints them all
        shown = shown[:5]
    for item in shown:
        prompt = (json.dumps(item["prompt"]) if "prompt" in item
                  else " ".join(map(str, item["ids"]))
                  + (" ..." if item["prompt_tokens"] > len(item["ids"]) else ""))
        roles = " ".join(f"@{r['name']}={r['row']}" for r in item["roles"])
        variables = " ".join(f"${v['name']}={v.get('token', json.dumps(v.get('text')))}"
                             for v in item["vars"])
        pair = f" pair {item['pair']}" if "pair" in item else ""
        lines.append(f"  {item['name']:<10}{prompt}  {roles} {variables}{pair}".rstrip())
    if len(shown) < len(result["items"]):
        lines.append(f"  +{len(result['items']) - len(shown)} more (set show {result['set']})")
    return lines


_INSPECT["each"] = _each_lines
_INSPECT["promptset"] = _promptset_lines


# --------------------------------------------------------------------------- procedure / sweep


def _procedure_lines(result: dict[str, Any]) -> list[str]:
    """The prefill split of a run (``prefill``) as one line, when it is not the default (all
    prompt rows but the last batched); the JSON always carries it."""
    prefill = result.get("prefill")
    if not isinstance(prefill, dict) or prefill.get("default", True):
        return []
    return [prefill["text"]]


_HEAT_WIDTH = 80


def _heat_cell(value: float, scale: float, width: int) -> str:
    """One cell of the heat table: the signed tenths of the largest |delta| (``+9`` .. ``-9``),
    ``.`` below a twentieth, ``0`` for an exact zero."""
    if value == 0:
        text = "0"
    elif scale == 0 or abs(value) / scale < 0.05:
        text = "."
    else:
        tenths = min(9, max(1, round(abs(value) / scale * 10)))
        text = ("+" if value > 0 else "-") + str(tenths)
    return text.rjust(width)


def _token_label(item: dict[str, Any], width: int) -> str:
    text = item.get("text")
    if text is None:
        return str(item.get("token", "?"))[-width:].rjust(width)
    shown = text.replace(" ", "·").replace("\n", "↵")
    return shown[:width].rjust(width)


def heat_table(result: dict[str, Any], width: int = _HEAT_WIDTH) -> list[str]:
    """The rows x layers table of a ``sweep`` as text of at most ``width`` columns: one line per
    layer, one column per position (row index, and the token text when a tokenizer exists), a
    cell the signed tenths of the largest |delta|. Positions that do not fit are cut from the
    left and said so (the record holds them all)."""
    table = result["table"]
    rows, layers, grid = table["rows"], table["layers"], table["delta"]["value"]
    scale = max((abs(v) for line in grid for v in line), default=0.0)
    label = max(len(f"L{layers[-1]}"), 3) + 1
    cell = 3 if label + 3 * len(rows) <= width else 2
    fit = max(1, (width - label) // cell)
    start = max(0, len(rows) - fit)
    shown = list(range(start, len(rows)))
    tokens = {item["row"]: item for item in table.get("tokens", ())}
    lines = [" " * label + "".join(f"t{rows[j]}"[-cell:].rjust(cell) for j in shown)]
    if any("text" in tokens.get(rows[j], {}) for j in shown):
        lines.append(" " * label + "".join(_token_label(tokens[rows[j]], cell) for j in shown))
    for i, layer in enumerate(layers):
        lines.append(f"L{layer}".ljust(label)
                     + "".join(_heat_cell(grid[i][j], scale, cell) for j in shown))
    if start:
        lines.append(f"(t{rows[0]}..t{rows[start - 1]} cut to fit; the table holds them)")
    lines.append(f"cell: tenths of max |Δ| {fmt(scale)} [measured] (9 caps); '.' < 1/20, "
                 "'0' exact")
    return lines


def _sweep_lines(result: dict[str, Any]) -> list[str]:
    base = result["base"]
    lines = [f"sweep {result['port']} = {result['origin']}:{result['port']} -> "
             f"{result['consumer']}  ({result['cells']['count']} cells; base "
             f"{num(base)})"]
    lines.extend(f"  {line}" for line in _procedure_lines(result))
    if "table" in result:
        lines.extend(heat_table(result))
    else:
        pointer = result["table_pointer"]
        lines.append(f"  table: {pointer['state']} blob sha256 {pointer['sha256'][:12]} "
                     f"({pointer.get('reason') or str(pointer.get('bytes', '')) + ' bytes'})")
    top = ", ".join(f"t{item['row']}.L{item['layer']} {num(item['delta'])}"
                    for item in result["top"])
    lines.append(f"  top |Δ|: {top}")
    native = result["cost"]["native"]
    lines.append(f"  cost: {native['layer']['value']} layer calls, "
                 f"{native['probe']['value']} probes, "
                 f"{native['transition']['value']} other transitions [measured] "
                 "(wall time in the record)")
    lines.append(f"  session unchanged at {result['where']}")
    return lines


_INSPECT["sweep"] = _sweep_lines
