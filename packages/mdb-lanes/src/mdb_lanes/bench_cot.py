"""Automatic, source-bound readouts beside native Qwen stepping.

The register describes an instrument's observations. It never supplies model
inputs, infers causal edges from a sequence, or restores a reduced semantic view.
"""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from html import escape
from pathlib import Path
from types import SimpleNamespace

from .errors import BenchError

CONFIG_SCHEMA = "mdb-cot-register-config-v1"
OBSERVATION_SCHEMA = "mdb-cot-register-observation-v1"


def _config(bench):
    from .bench_cot_index import index
    return index(bench).config


def configure(bench, lens_refs=None, *, top_k=5, token_subset=None, enabled=True, optimized=False):
    from .bench_jacobian import _reader

    if type(optimized) is not bool or type(enabled) is not bool or type(top_k) is not int or not 1 <= top_k <= 32:
        raise BenchError("CoT register requires enabled boolean and top_k in 1..32")
    if lens_refs is None:
        previous, _ = _config(bench)
        lens_refs = previous["lens_refs"] if previous else [
            event["pin_ref"] for event in bench.events
            if event.get("kind") == "jacobian-lens-pinned"
        ]
    if not isinstance(lens_refs, (list, tuple)) or not lens_refs:
        raise BenchError("CoT register needs at least one pinned, compatible Jacobian lens")
    if token_subset is not None:
        if (not isinstance(token_subset, (list, tuple)) or not token_subset
                or any(type(token) is not int or not 0 <= token < bench.model.config.vocab_size
                       for token in token_subset)
                or len(set(token_subset)) != len(token_subset)):
            raise BenchError("CoT token_subset requires unique vocabulary token IDs")
        token_subset = list(token_subset)
    boundaries = {}
    if optimized:
        from .bench_frozen_reader import bind
        bind(bench)
    else:
        bench._frozen_readers = None
    for ref in lens_refs:
        if not isinstance(ref, str):
            raise BenchError("CoT lens references must be pinned artifact references")
        reader, _ = _reader(bench, ref)
        boundary = reader.lens.manifest["coordinate_map"]["completed_layer_boundary"]
        if not 1 <= boundary <= bench.vm.runtime.codec.layers:
            raise BenchError("CoT lens boundary is outside the native decoder")
        if str(boundary) in boundaries:
            raise BenchError("CoT register requires one lens per completed layer boundary")
        boundaries[str(boundary)] = ref
    config = {"schema": CONFIG_SCHEMA, "model_identity": bench.identity,
              "session_id": bench.session_id, "enabled": enabled,
              "lens_refs": list(lens_refs), "boundaries": boundaries,
              "options": {"top_k": top_k, "token_subset": token_subset,
                          "logical_address": None}}
    if optimized:
        config["performance"] = {"reader_binding": "managed-frozen-v1",
                                 "capture": "changed-state-v1", "publication": "command-v1",
                                 "hydration": "selected-closure-v1"}
    ref = bench.artifacts.put(config)
    bench.events.append({"kind": "cot-register-configured", "config_ref": ref})
    bench._publish()
    return {"config_ref": ref, "enabled": enabled,
            "completed_layer_boundaries": sorted(map(int, boundaries))}


def _text(value):
    return "".join(char if char.isprintable() else ascii(char)[1:-1]
                   for char in str(value))[:120]


def _token_text(bench, token):
    if bench.tokenizer is None:
        return f"token {token}"
    return _text(bench.tokenizer.decode([token], skip_special_tokens=False))


def _flat(value):
    while isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return value


def capture(bench):
    from .bench_jacobian import read

    config, config_ref = _config(bench)
    execution = bench._backend().execution
    if not config or not config["enabled"] or execution is None or execution.logits is not None:
        return None
    pin_ref = config["boundaries"].get(str(execution.next_layer))
    if pin_ref is None:
        return None
    result = read(bench, pin_ref, config["options"])
    read_receipt = bench.artifacts.get(result["observation_ref"])
    source = read_receipt["source"]
    candidates = [{"token_id": token, "text": _token_text(bench, token), "score": score}
                  for token, score in zip(_flat(result["selected_token_ids"]),
                                         _flat(result["selected_scores"]))]
    row = {"schema": OBSERVATION_SCHEMA, "config_ref": config_ref,
           "observation_ref": result["observation_ref"], "source_cut": result["source_cut"],
           "lens_ref": pin_ref, "session_id": bench.session_id, "branch": bench.active,
           "model_identity": bench.identity, "token_position": source["token_position"],
           "completed_layer_boundary": source["completed_layer_boundary"],
           "activation_sha256": source["activation"]["sha256"], "candidates": candidates,
           "interpretation": "decoded-candidates", "causal_dependency_tested": False,
           "native_mutation": False, "semantic_cot_qualified": False}
    ref = bench.artifacts.put(row)
    bench.events.append({"kind": "cot-register-observation", "register_ref": ref,
                         "source_cut": result["source_cut"]})
    bench._publish()
    return row


def observe_boundary(bench):
    from .bench_jacobian import atomic_sidecar

    config, config_ref = _config(bench)
    execution = bench._backend().execution
    if (not config or not config["enabled"] or execution is None or execution.logits is not None
            or str(execution.next_layer) not in config["boundaries"]):
        return
    try:
        atomic_sidecar(bench, capture)
    except Exception as exc:
        # Optional instrumentation failure does not undo a valid native step.
        bench.events.append({"kind": "cot-register-error", "config_ref": config_ref,
                             "source_cut": bench.branch.cut, "cursor": bench.cursor(),
                             "error": _text(f"{type(exc).__name__}: {exc}")})
        bench._publish()


def _lineage(bench):
    from .bench_cot_index import index
    return index(bench).lineage(bench)


def _observations(bench, cuts):
    from .bench_cot_index import index
    return index(bench).observations(bench, cuts=cuts)


def view(bench, *, limit=8):
    if type(limit) is not int or not 1 <= limit <= 128:
        raise BenchError("CoT view limit must be in 1..128")
    config, _ = _config(bench)
    if config is None:
        return {"enabled": False, "status": "not-configured", "entries": []}
    from .bench_cot_index import index
    indexed = index(bench)
    selected = indexed.node(bench, bench.branch.cut)
    rows = indexed.observations(bench, limit=limit)
    current = bench.cursor()
    latest = rows[-1] if rows else None
    snapshot = selected["row"]
    saved = snapshot.get("execution") or {}
    observed = bool(latest and current["phase"] == "layers"
                    and current["next_layer"] == latest["completed_layer_boundary"]
                    and len(bench.branch.token_ids) == latest["token_position"]
                    and saved.get("hidden", {}).get("sha256") == latest["activation_sha256"])
    errors = indexed.recent_errors(bench)
    return {"enabled": config["enabled"], "status": "observing" if observed else "unobserved-current-boundary",
            "branch": bench.active, "source_cut": bench.branch.cut, "cursor": current,
            "current_boundary_observed": observed, "entry_count": selected["count"],
            "entries": deepcopy(rows), "omitted_entries": max(0, selected["count"]-limit),
            "configured_boundaries": sorted(map(int, config["boundaries"])),
            "errors": deepcopy(errors[-3:]), "semantic_cot_qualified": False,
            "panel_path": str(Path(bench.root) / "cot-register.html"),
            "caption_policy": "source-bound vocabulary candidates; no inferred intentions or causal edges"}


def panel_html(bench, *, limit=32):
    """A self-refreshing native-history graph with deterministic language captions."""
    from .bench_cot_index import index
    indexed = index(bench)
    selected = indexed.node(bench, bench.branch.cut)
    chain = indexed.lineage(bench, limit)
    rows = indexed.observations(bench, cuts={ref for ref, _ in chain})
    observations = {row["source_cut"]: row for row in rows}
    config, _ = _config(bench)
    visible = chain[-limit:]
    height = max(150, len(visible) * 112 + 30)
    parts = [f'<svg viewBox="0 0 1080 {height}" role="img" aria-label="Native execution and decoded vocabulary observations">']
    for position, (ref, cut) in enumerate(visible):
        y = 20 + position * 112
        execution = cut.get("execution")
        layer = execution["next_layer"] if execution else None
        token = len(cut["token_ids"])
        title = f"Token {token} · after layer {layer}" if layer is not None else f"Committed history · {token} tokens"
        current = position == len(visible)-1
        if position:
            parts.append(f'<path d="M 195 {y-26} L 195 {y}" class="native-edge"/>')
        parts.append(f'<g class="{"current" if current else "state"}"><title>{escape(ref)}</title>'
                     f'<rect x="25" y="{y}" width="340" height="84" rx="12"/>'
                     f'<text x="43" y="{y+31}" class="title">{escape(title)}</text>'
                     f'<text x="43" y="{y+57}" class="detail">{escape(_text(cut.get("label", "native state")))}</text></g>')
        observed = observations.get(ref)
        if observed:
            parts.append(f'<path d="M 365 {y+42} L 420 {y+42}" class="decoded-edge"/>')
            terms = ", ".join(repr(item["text"]) for item in observed["candidates"][:5])
            parts.append(f'<g class="observation"><title>{escape(observed["observation_ref"])}</title>'
                         f'<rect x="420" y="{y}" width="620" height="84" rx="12"/>'
                         f'<text x="438" y="{y+28}" class="detail">Decoded vocabulary · dependency untested</text>'
                         f'<text x="438" y="{y+56}" class="terms">{escape(terms[:90])}</text></g>')
        else:
            parts.append(f'<text x="438" y="{y+46}" class="missing">No semantic readout captured at this cut</text>')
    parts.append("</svg>")
    state = "Live observations" if config and config["enabled"] else "Observer disabled"
    omitted = max(0, selected["native_count"]-limit)
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>MDB · CoT register</title><style>'
            'body{margin:0;background:#f5f7fa;color:#17283c;font:15px system-ui,sans-serif}'
            'header{padding:26px 32px;background:white;border-bottom:1px solid #d8e1e9}'
            'h1{margin:0 0 8px;font-size:26px}p{margin:8px 0;color:#526479}'
            'main{padding:24px 32px;max-width:1260px;margin:auto}'
            'button{float:right;padding:9px 14px;border:1px solid #c5d3e1;border-radius:8px;background:white}'
            'svg{width:100%;height:auto}.state rect{fill:white;stroke:#cbd6e0}'
            '.current rect{fill:#e4f5ee;stroke:#2a8b68;stroke-width:2}'
            '.observation rect{fill:#eef2ff;stroke:#c6cef0}.title{font-size:16px;font-weight:600;fill:#17283c}'
            '.detail{font-size:13px;fill:#596c82}.terms{font:15px ui-monospace,monospace;fill:#343d75}'
            '.missing{font-size:13px;fill:#8090a3}.native-edge{stroke:#91a3b6;stroke-width:2;fill:none}'
            '.decoded-edge{stroke:#828ec6;stroke-width:2;stroke-dasharray:5 4;fill:none}'
            '.badge{display:inline-block;padding:5px 9px;border-radius:6px;background:#eef2ff;color:#424d82}'
            '</style><header><button id="refresh">Pause live updates</button>'
            '<h1>MDB · CoT register</h1>'
            f'<span class="badge">{escape(state)}</span> '
            f'<span>Branch {escape(bench.active)} · {selected["count"]} decoded observations</span>'
            '<p>Solid links follow native execution. Dashed links attach decoded vocabulary to its source state.</p>'
            '</header><main><p>These are instrument readouts. Reasoning operations and causal dependencies remain unqualified.</p>'
            f'<p>{omitted} older native cuts omitted from this bounded display; retained history remains available.</p>'
            + "".join(parts)
            + '</main><script>let paused=sessionStorage.getItem("mdb-cot-paused")==="1";'
            'const button=document.getElementById("refresh");'
            'button.textContent=paused?"Resume live updates":"Pause live updates";'
            'button.onclick=()=>{paused=!paused;sessionStorage.setItem("mdb-cot-paused",paused?"1":"0");'
            'button.textContent=paused?"Resume live updates":"Pause live updates";};'
            'setInterval(()=>{if(!paused)location.reload();},1500);</script></html>')


def write_live_panel(bench):
    """Render only the selected durable manifest, never an uncommitted future."""
    selected = json.loads((Path(bench.root) / "selected.json").read_text())["manifest"]
    if getattr(bench, "_cot_panel_manifest", None) == selected:
        return
    manifest = bench.artifacts.get(selected)
    if not any(event.get("kind") == "cot-register-configured" for event in manifest["events"]):
        return
    branch = manifest["branches"][manifest["active"]]
    readonly = SimpleNamespace(root=bench.root, artifacts=bench.artifacts, events=manifest["events"],
                               branch=SimpleNamespace(**branch), active=manifest["active"],
                               identity=manifest["identity"], session_id=manifest["session_id"])
    readonly._cot_index = getattr(bench, "_cot_durable_index", None)
    path = Path(bench.root) / "cot-register.html"
    descriptor, temporary = tempfile.mkstemp(prefix=".cot-register-", dir=bench.root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(panel_html(readonly))
        os.replace(temporary, path)
        bench._cot_panel_manifest = selected
        bench._cot_durable_index = readonly._cot_index
    finally:
        Path(temporary).unlink(missing_ok=True)


def graph(bench, *, limit=128):
    from .state_graph import Evidence, NodeBinding, StateGraph

    if type(limit) is not int or not 1 <= limit <= 128:
        raise BenchError("CoT graph limit must be in 1..128")
    from .bench_cot_index import index
    indexed = index(bench)
    chain = indexed.lineage(bench, limit)
    cuts = {ref for ref, _ in chain}
    result = StateGraph(binding=NodeBinding(model=bench.identity, session=bench.session_id),
                        metadata={"view": "cot-register", "selected_cut": bench.branch.cut,
                                  "omitted_native_cuts": max(0, indexed.node(bench, bench.branch.cut)["native_count"]-limit),
                                  "semantic_cot_qualified": False})
    states = {}
    for ref, snapshot in chain:
        execution = snapshot.get("execution")
        layer = execution["next_layer"] if execution else None
        clock = {"token_position": len(snapshot["token_ids"]), "completed_layer_boundary": layer}
        label = f"Token {clock['token_position']}, after layer {layer}" if layer is not None else f"Committed history: {clock['token_position']} tokens"
        node = result.add_state(cut_ref=ref, binding=NodeBinding(model=bench.identity,
            session=bench.session_id, clock=clock), evidence=Evidence("observed"), labels=(label,))
        states[ref] = node
        if snapshot.get("parent_cut") in states:
            result.add_edge(states[snapshot["parent_cut"]], node, "continues",
                            evidence=Evidence("observed"))
    for row in _observations(bench, cuts):
        terms = ", ".join(escape(candidate["text"], quote=True) for candidate in row["candidates"])
        node = result.add_source(source_ref=row["register_ref"],
            metadata={"interpretation": "decoded-candidates", "observation_ref": row["observation_ref"],
                      "causal_dependency_tested": False}, evidence=Evidence("observed"),
            labels=(f"Decoded candidates: {terms}",))
        result.add_edge(states[row["source_cut"]], node, "annotates", graph="knowledge",
                        evidence=Evidence("observed", metadata={"lens_ref": row["lens_ref"]}))
    return {"snapshot": result.seal().to_dict(), "mermaid": result.to_mermaid(),
            "semantic_cot_qualified": False}


def verify_artifact(value, artifacts, manifest):
    schema = value.get("schema") if isinstance(value, dict) else None
    if schema not in {CONFIG_SCHEMA, OBSERVATION_SCHEMA}:
        return False
    if value.get("model_identity") != manifest["identity"] or value.get("session_id") != manifest["session_id"]:
        raise ValueError("CoT register model/session binding differs")
    if schema == CONFIG_SCHEMA:
        refs = value["lens_refs"]
        if not refs or set(value["boundaries"].values()) != set(refs):
            raise ValueError("CoT register lens inventory differs")
        for ref in refs:
            pin = artifacts.get(ref)
            boundary = pin["lens_manifest"]["coordinate_map"]["completed_layer_boundary"]
            if pin.get("schema") != "mdb-bench-jacobian-pin-v1" or pin["model_identity"] != manifest["identity"] or value["boundaries"].get(str(boundary)) != ref:
                raise ValueError("CoT register lens binding differs")
        return True
    read = artifacts.get(value["observation_ref"])
    config = artifacts.get(value["config_ref"])
    source = read["source"]
    cut = artifacts.get(value["source_cut"])
    if (read.get("schema") != "mdb-bench-jacobian-read-v1"
            or config.get("schema") != CONFIG_SCHEMA or not config["enabled"]
            or source["cut_ref"] != value["source_cut"]
            or read["pin_ref"] != value["lens_ref"]
            or config["boundaries"].get(str(value["completed_layer_boundary"])) != value["lens_ref"]
            or value["token_position"] != source["token_position"]
            or value["completed_layer_boundary"] != source["completed_layer_boundary"]
            or value["activation_sha256"] != source["activation"]["sha256"]
            or cut["execution"]["hidden"]["sha256"] != value["activation_sha256"]
            or value["interpretation"] != "decoded-candidates"
            or value["causal_dependency_tested"] is not False
            or value["native_mutation"] is not False or value["semantic_cot_qualified"] is not False):
        raise ValueError("CoT observation source/interpretation binding differs")
    return True
