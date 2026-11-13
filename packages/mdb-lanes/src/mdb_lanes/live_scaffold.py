"""Framework-free role schemas and occurrence graphs over native observations.

The supplied schema is a hypothesis. Resolution supplies token arguments;
native read weights and intervention outcomes remain separate evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any

SCHEMA = "mdb-live-scaffold-v1"


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ReaderRole:
    name: str
    layer: int
    heads: tuple[int, ...]
    column_below: int

    def __post_init__(self):
        if not self.name or self.layer < 0 or self.column_below < 0:
            raise ValueError("invalid reader role")
        if not self.heads or min(self.heads) < 0 or len(set(self.heads)) != len(self.heads):
            raise ValueError("reader heads must be distinct nonnegative indices")


ADDITION_ROLES = (
    ReaderRole("own-column", 22, (6, 8), 0),
    ReaderRole("column-below", 22, (10, 11), 1),
    ReaderRole("carry-sensitive", 20, (1, 2, 5, 10), 2),
)


def scaffold(model_identity: str, roles=ADDITION_ROLES) -> dict[str, Any]:
    body = {"schema": SCHEMA, "model_identity": model_identity,
            "roles": [asdict(role) for role in roles],
            "semantic_authority": "supplied historical addition hypothesis; test new invocations",
            "role_source": "2026-10-30-dynamic-symbol-table; zero-based layers and heads"}
    return {**body, "fingerprint": fingerprint(body)}


def resolve_addition(prompt: str, token_ids: list[int], offsets: list[tuple[int, int]],
                     answer_step: int, roles=ADDITION_ROLES) -> dict[str, Any]:
    """Resolve the final literal addition's columns, without reading its answer.

    The grammar is supplied. Each digit must occupy exactly one native token;
    merged digits and incomplete offsets are refused instead of guessed.
    """
    if type(answer_step) is not int or answer_step < 0:
        raise ValueError("answer_step must be a nonnegative integer")
    if len(token_ids) != len(offsets):
        raise ValueError("token and character-offset counts differ")
    matches = list(re.finditer(r"(?P<a>\d+)\s*\+\s*(?P<b>\d+)\s*=", prompt))
    if not matches:
        raise ValueError("unsupported addition syntax")
    match = matches[-1]
    a, b = match.group("a"), match.group("b")
    if len(a) != len(b):
        raise ValueError("this role schema requires equal operand widths")
    width = len(a)
    if answer_step >= width:
        raise ValueError("answer step outside the declared operand-width horizon")
    columns = {}
    for operand in ("a", "b"):
        for left_index, digit in enumerate(match.group(operand)):
            start = match.start(operand) + left_index
            rows = [row for row, (lo, hi) in enumerate(offsets) if lo <= start < hi]
            if len(rows) != 1 or tuple(offsets[rows[0]]) != (start, start + 1):
                raise ValueError("each operand digit must be one exact token")
            column = width - 1 - left_index
            columns[f"{operand}{column}"] = {
                "symbol": f"{operand}[10^{column}]", "operand": operand,
                "column_from_right": column, "source_row": rows[0],
                "token_id": int(token_ids[rows[0]]), "text": digit,
                "character_span": [start, start + 1], "origin": "caller-input",
                "meaning_status": "supplied-parser-variable",
            }
    bindings = []
    for role in roles:
        column = width - 1 - answer_step - role.column_below
        bindings.append({"role": role.name, "layer": role.layer, "heads": list(role.heads),
                         "column_from_right": column,
                         "status": "resolved" if column >= 0 else "not-applicable",
                         "arguments": [columns[f"{operand}{column}"] for operand in ("a", "b")]
                         if column >= 0 else []})
    return {"operand_width": width, "answer_step": answer_step,
            "input_identity": fingerprint({"prompt": prompt, "token_ids": token_ids}),
            "bindings": bindings, "columns": columns,
            "resolver_authority": "supplied grammar and frozen column-relative rule; no outcome lookup"}


def invocation_graph(schema: dict[str, Any], occurrences: list[dict[str, Any]]) -> dict[str, Any]:
    """Link supplied semantic arguments, observed ports and actual readouts.

    A suffix edge describes declared native execution. Causal status requires
    a separate intervention record, never attention weight alone.
    """
    nodes, edges = {}, []
    for occurrence in occurrences:
        key = fingerprint({"case": occurrence["case"], "history": occurrence["history_sha256"],
                           "answer_step": occurrence["answer_step"]})
        readout = key + ":readout"
        nodes[readout] = {"id": readout, "kind": "readout", "case": occurrence["case"],
                          "answer_step": occurrence["answer_step"],
                          "next_token_id": occurrence["next_token_id"],
                          "next_token_text": occurrence["next_token_text"],
                          "logits_sha256": occurrence["logits_sha256"],
                          "authority": "native-observed"}
        for binding in occurrence["bindings"]:
            if binding["status"] != "resolved":
                continue
            function = key + ":" + binding["role"]
            returned = function + ":return"
            nodes[function] = {"id": function, "kind": "function-invocation",
                               "role": binding["role"], "layer": binding["layer"],
                               "heads": binding["heads"], "scaffold": schema["fingerprint"],
                               "authority": "supplied-role/native-observed-port"}
            nodes[returned] = {"id": returned, "kind": "native-head-return",
                               "sha256": binding.get("head_output_sha256"),
                               "authority": "native-observed"}
            edges.append({"source": function, "target": returned, "kind": "observed-return"})
            for argument in binding["arguments"]:
                source = function + ":arg:" + argument["operand"]
                nodes[source] = {"id": source, "kind": "argument", **argument}
                edges.append({"source": source, "target": function,
                              "kind": "proposed-semantic-binding",
                              "attention": argument.get("attention_by_head"),
                              "authority": "resolver plus native read observation; causal effect separate"})
            edges.append({"source": returned, "target": readout,
                          "kind": "native-suffix-dataflow", "authority": "declared execution, not isolated mediation",
                          "writer_observation": occurrence.get("writer")})
    body = {"schema": SCHEMA, "scaffold": schema, "occurrences": occurrences,
            "nodes": list(nodes.values()), "edges": edges,
            "scope": "candidate role bindings, observed numerical ports and native output; incomplete semantic decomposition"}
    return {**body, "fingerprint": fingerprint(body)}


def render_html(graph: dict[str, Any], destination: str | Path) -> None:
    """Portable occurrence view; no tensor/model/framework hydration."""
    rows = []
    for occurrence in graph["occurrences"]:
        for binding in occurrence["bindings"]:
            args = " · ".join(f"{a['symbol']}={a['text']} @ row {a['source_row']}" for a in binding["arguments"])
            weights = " · ".join(str(a.get("attention_by_head", {})) for a in binding["arguments"])
            values = (occurrence["case"], occurrence["answer_step"], binding["role"],
                      f"L{binding['layer']} / {binding['heads']}", args or binding["status"],
                      weights, occurrence["next_token_text"])
            rows.append("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in values) + "</tr>")
    data = json.dumps(graph, sort_keys=True, allow_nan=False).replace("<", "\\u003c")
    output = """<!doctype html><meta charset="utf-8"><title>Live scaffold invocations</title>
<style>body{font:15px system-ui;margin:32px;color:#182033;background:#f5f7fb}h1{font-size:28px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:10px;border:1px solid #dbe2ed;text-align:left;vertical-align:top}td{font-size:12px}header,details{padding:18px;background:white;margin:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}.flow{font-size:18px;padding:18px;background:#e7efff}</style>
<h1>One scaffold, changing arguments</h1><div class="flow">Source tokens → role-bound reader → native write/suffix → actual output → next history</div>
<header>Role names and grammar are supplied hypotheses. Read weights and head returns are native observations. Intervention effects are recorded separately. This view does not assign semantic causality to an attention weight.</header>
<table><thead><tr><th>Case</th><th>Step</th><th>Role</th><th>Implementation</th><th>Bound arguments</th><th>Native attention</th><th>Actual next token</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table>"
    output += '<details><summary>Complete graph and evidence scope</summary><pre id="raw"></pre></details>'
    output += '<script type="application/json" id="graph">' + data + '</script><script>document.getElementById("raw").textContent=JSON.stringify(JSON.parse(document.getElementById("graph").textContent),null,2)</script>'
    Path(destination).write_text(output)
