"""Offline bidirectional view of canonical symbol/attention joins."""
import json
from pathlib import Path


def causal_label(row):
    """The row's causal label from its own data: a recorded ``causal`` status, else the
    ``CausalStatus`` default (no intervention recorded on these uses)."""
    from .symbol_reads import CausalStatus
    causal = row.get("causal")
    if causal is not None:
        return CausalStatus.from_dict(causal).label
    return CausalStatus().label


def write_view(directory):
    """Qwen join view (``linked-symbols.json``), or the read-record table (``reads.json``)."""
    root = Path(directory)
    if not (root / "linked-symbols.json").exists() and (root / "reads.json").exists():
        return write_reads_view(root)
    source = json.loads((root / "linked-symbols.json").read_text())
    formations = json.loads((root / "formation-symbols.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    rows = []
    for row in source:
        heads = [{**{key: h[key] for key in ("head", "kv_head", "allocation", "source_mass", "entropy_nats",
            "source_value_return_norm", "head_return_norm")}, "use": u}
            for h, u in zip(row["heads"], row["uses"])]
        rows.append({key: row[key] for key in ("session_id", "arm", "step", "layer", "binding", "lifecycle_id", "definition", "output")} |
            {"heads": heads, "token_labels": row["heads"][0]["token_labels"], "causal_label": causal_label(row)})
    data = json.dumps({"rows": rows, "formations": formations, "summary": summary}).replace("<", "\\u003c")
    html = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MDB symbols × attention</title>
<style>body{font:15px system-ui;margin:28px;color:#e2e8f0;background:#0f172a}h1{margin-bottom:8px}p{max-width:1100px;line-height:1.5}a{color:#67e8f9}select,button{padding:8px;margin:4px 10px 4px 0;background:#1e293b;color:#e2e8f0;border:1px solid #64748b;border-radius:5px}button{cursor:pointer}.active{border-color:#fbbf24;color:#fbbf24}.controls{position:sticky;top:0;background:#0f172a;padding:10px 0;z-index:2}table{border-collapse:collapse;font-size:12px;width:max-content;min-width:100%}th,td{padding:9px;border:1px solid #334155;text-align:center;min-width:40px}th{white-space:pre-wrap;max-width:100px}.source{outline:2px solid #fbbf24;cursor:pointer}tr.selected th{color:#fbbf24}.scroll{overflow:auto}.panel{background:#172337;padding:18px;margin:18px 0;border-radius:8px}.nodes{display:flex;gap:10px;flex-wrap:wrap;align-items:stretch}.node{border:1px solid #475569;border-radius:7px;padding:13px;max-width:220px;overflow-wrap:anywhere}.node strong{display:block;color:#67e8f9;margin-bottom:9px}.arrow{align-self:center;font-size:24px}pre{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere}code{overflow-wrap:anywhere;font-size:12px}.muted{color:#94a3b8}.hidden{display:none}.symbolrow{cursor:pointer}.symbolrow:hover{background:#243449}.pill{display:inline-block;border:1px solid #475569;border-radius:14px;padding:5px 10px;font-size:12px;margin:3px}</style>
<h1>MDB symbol table × attention</h1><p>Select a description symbol to inspect its readers, or select a gold source-attention cell to inspect its qualified symbol and native use. The lineage view connects supplied input → native source writer → K/V carrier → query-head use → logits → token commit.</p>
<div id="counts"></div><p class="muted">Canonical public Session records. Semantic role names are supplied; attention is observed use, and individual-head necessity is untested. Other token columns remain visible attention context; gold source cells have registered source bindings. <a href="../../mdb-attention-allocation-20261111/main-01/analysis/attention.html">Compare both runtimes</a>.</p>
<div class="controls"><label>Arm <select id="arm"><option>native</option><option>donor-native</option><option>sham</option><option>donor-late</option><option>donor-all</option></select></label><label>Generated step <select id="step"><option>0</option><option>1</option><option>2</option></select></label><label>Layer <select id="layer"></select></label><label>Query head <select id="head"></select></label><br><button id="mapbutton" onclick="view('map')">Attention map</button><button id="tablebutton" onclick="view('table')">Symbol table</button><button id="lineagebutton" onclick="view('lineage')">Writer → output</button></div>
<div id="selection" class="panel"></div><section id="map"><div id="matrix" class="scroll"></div><div id="use" class="panel"></div></section><section id="table" class="hidden"><div id="symboltable" class="scroll"></div></section><section id="lineage" class="hidden"><div id="chain" class="panel"></div><details><summary>Native writer and source formation records</summary><pre id="formation"></pre></details></section>
<details class="panel"><summary>Selected canonical binding, definition and head use</summary><pre id="canonical"></pre></details><p><a href="linked-symbols.json">Full linked symbols JSON</a> · <a href="outputs.json">Native outputs</a> · <a href="formation-symbols.json">Source definitions</a> · <a href="summary.json">Join audit</a></p>
<script>const data=DATA;
const el=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));let tab='map';
el('layer').innerHTML=Array.from({length:28},(_,i)=>'<option>'+i+'</option>').join('');el('head').innerHTML=Array.from({length:12},(_,i)=>'<option>'+i+'</option>').join('');el('layer').value=20;el('head').value=5;
function current(){return data.rows.find(r=>r.arm===el('arm').value&&r.step===+el('step').value&&r.layer===+el('layer').value)}
function view(name){tab=name;for(const id of ['map','table','lineage']){el(id).classList.toggle('hidden',id!==name);el(id+'button').classList.toggle('active',id===name)}draw()}
function pickHead(h){el('head').value=h;view('map')}
function pickLayer(l){el('layer').value=l;view('map')}
function draw(){const r=current(),h=r.heads[+el('head').value],m=r.definition.metadata,o=r.output;const formations=Object.fromEntries(data.formations.map(f=>[f.binding_fingerprint,f]));const k=formations[m.key_origin_binding],v=formations[m.value_origin_binding];
el('counts').innerHTML=[data.summary.canonical_bindings+' canonical bindings',data.summary.attention_uses+' native attention uses',data.summary.output_bindings+' readouts',data.summary.native_forwards+' new forwards'].map(s=>'<span class="pill">'+esc(s)+'</span>').join('');
el('selection').innerHTML='<strong>'+esc(r.binding.symbol)+'</strong> · source row 18 · query H'+h.head+' / KV'+h.kv_head+'<br>Source attention <strong>'+(100*h.source_mass).toFixed(2)+'%</strong> · K origin: '+esc(m.key_origin_side)+' · V origin: '+esc(m.value_origin_side)+'<br>Actual next token <strong>'+esc(JSON.stringify(o.token_text))+'</strong> ('+o.token_id+')<br>Causal status: <strong>'+esc(r.causal_label)+'</strong><br><span class="muted">Native source cut </span><code>'+esc(r.definition.source_cut)+'</code>';
el('matrix').innerHTML='<table><thead><tr><th>Query / KV</th>'+r.token_labels.map((s,i)=>'<th class="'+(i===18?'source':'')+'">'+i+' '+esc(s)+'</th>').join('')+'</tr></thead><tbody>'+r.heads.map(q=>'<tr class="'+(q.head===h.head?'selected':'')+'"><th><button onclick="pickHead('+q.head+')">H'+q.head+' / KV'+q.kv_head+'</button></th>'+q.allocation.map((p,i)=>'<td class="'+(i===18?'source':'')+'" '+(i===18?'onclick="pickHead('+q.head+')"':'')+' style="background:rgba(14,165,233,'+(p*.94)+')">'+(100*p).toFixed(1)+'%</td>').join('')+'</tr>').join('')+'</tbody></table>';
el('use').innerHTML='<strong>Selected head → registered symbol</strong><p>Query L'+r.layer+'/H'+h.head+' reads the source through KV group '+h.kv_head+'. Its canonical use points to the original native readout receipt, with K/V location selectors and a hash-bound attention array.</p><pre>'+esc(JSON.stringify({use_id:h.use.use_id,binding_fingerprint:h.use.binding_fingerprint,locations:h.use.locations,readout_receipt:h.use.receipt,attention_selector:h.use.metadata.attention_selector,output_binding:h.use.metadata.output_binding},null,2))+'</pre>';
const symbols=data.rows.filter(x=>x.arm===r.arm&&x.step===r.step);el('symboltable').innerHTML='<table><thead><tr><th>Registered source symbol</th><th>Locations</th><th>Head uses</th><th>V origin</th><th>Binding fingerprint</th></tr></thead><tbody>'+symbols.map(x=>'<tr class="symbolrow" onclick="pickLayer('+x.layer+')"><td>'+esc(x.binding.symbol)+'</td><td>'+x.binding.locations.length+'</td><td>'+x.heads.length+'</td><td>'+esc(x.definition.metadata.value_origin_side)+'</td><td><code>'+esc(x.binding.fingerprint)+'</code></td></tr>').join('')+'<tr><td>'+esc(o.binding.symbol)+'</td><td>1 native logit port</td><td>1 sample/commit use</td><td>native readout</td><td><code>'+esc(o.binding.fingerprint)+'</code></td></tr></tbody></table>';
const node=(title,text)=>'<div class="node"><strong>'+esc(title)+'</strong>'+esc(text)+'</div>';el('chain').innerHTML='<div class="nodes">'+[node('Supplied description',m.value_origin_side==='donor'?'boiling':'freezing'),node('Native source writer','Receipt '+v.definitions[0].metadata.writer_receipt),node('Qualified carrier','L'+r.layer+' / K from '+m.key_origin_side+' / V from '+m.value_origin_side+'; exact row bytes checked'),node('Observed reader','H'+h.head+' / KV'+h.kv_head+'; '+(100*h.source_mass).toFixed(2)+'% on source'),node('Native readout',o.readout_receipt),node('Token commit',JSON.stringify(o.token_text)+'; receipt '+o.commit_receipt)].join('<div class="arrow">→</div>')+'</div><p>Writer lineage uses canonical definition references. Attention links point to the actual source cut, native decoder operation and readout receipt. The next-token binding is consumed by the actual greedy sample/commit receipt.</p>';
el('formation').textContent=JSON.stringify({key_origin:k,value_origin:v},null,2);el('canonical').textContent=JSON.stringify({session_id:r.session_id,lifecycle_id:r.lifecycle_id,binding:r.binding,definition:r.definition,selected_head_use:h.use,native_output:o},null,2);
history.replaceState(null,'','#'+new URLSearchParams({arm:r.arm,step:r.step,layer:r.layer,head:h.head,binding:r.binding.fingerprint}).toString());}
const params=new URLSearchParams(location.hash.slice(1));for(const id of ['arm','step','layer','head']){const value=params.get(id);if(value!==null&&Array.from(el(id).options).some(o=>o.value===value))el(id).value=value;el(id).addEventListener('change',draw)}view('map');</script>""".replace("DATA", data)
    (root / "symbols-attention.html").write_text(html)
    return {"file": str(root / "symbols-attention.html"), "source_layer_rows": len(rows),
        "linked_head_uses": sum(len(r["heads"]) for r in rows)}


def write_reads_view(directory):
    """Offline table of ``mdb.symbol_reads`` records (no external assets).

    Top: the scope block (prompts, seeds, independent contrasts, counts per contrast x status)
    and the verdict rule id. Then read sets grouped by contrast; next to every verdict: the
    counted control (in-run, size-matched only or mass-matched, with set and control mass),
    reference controls (never counted), floor and where it came from ("borrowed" when from
    another run), claim / scope / alternatives, and any audit caveat. Then observed reads with
    mass, mass kind and causal label (``unknown`` unless a per-read intervention was recorded).
    """
    from collections import Counter
    from html import escape

    from .symbol_reads import load_reads
    root = Path(directory)
    loaded = load_reads(root / "reads.json")
    reads, sets, meta = loaded["reads"], loaded["sets"], loaded["meta"]
    sources = sorted({r.source.name for r in reads} | {s.source.name for s in sets})
    scope = meta.get("scope") if isinstance(meta.get("scope"), dict) else None
    rule = meta.get("verdict_rule") or {}

    def num(x):
        return "" if x is None else f"{x:.3g}"

    def seeds(values):
        return "" if not values else "/".join(num(v) for v in values)

    def cell(text, cls=""):
        return f'<td class="{cls}">{escape(str(text))}</td>'

    def control_text(c, *, counted):
        if not c:
            return "none in-run" if counted else ""
        where = "in-run" if c.get("in_run") else f"OTHER JOB {c.get('run_id')}"
        if "mass" in (c.get("matched_on") or []):
            matched = "mass-matched"
        elif "set_size" in (c.get("matched_on") or []):
            matched = "size-matched only"
        else:
            matched = "matched on " + ", ".join(c.get("matched_on") or [])
        parts = [f"{c.get('arm')} ({where}; {matched}; judged by {c.get('judged_by')})"]
        if c.get("size") is not None and c.get("set_size") is not None:
            parts.append(f"size {c['size']} vs set {c['set_size']}")
        if c.get("mass_held") is not None:
            parts.append(f"word mass held {seeds(c['mass_held'])} vs set {seeds(c.get('set_mass_held'))}"
                         f" ({c.get('mass_relation', '')})")
        if c.get("picture"):
            parts.append("picture " + " / ".join(c["picture"]))
        if c.get("effect") is not None:
            parts.append("pixel " + num(c["effect"]))
        if c.get("judgement"):
            parts.append("judged " + c["judgement"])
        if not counted:
            parts.append({"none": "not counted", "downgrade_only": "screen: can only downgrade"}.get(
                c.get("verdict_effect"), ""))
        if c.get("rules_out"):
            parts.append("rules out " + ", ".join(c["rules_out"]))
        return "; ".join(p for p in parts if p)

    def floor_text(r):
        if r.floor is None:
            return r.floor_source or ""
        text = f"{num(r.floor)} — {r.floor_source or ''}"
        if r.floor_incomplete:
            text += " [floor incomplete]"
        return text

    by_contrast, totals = {}, Counter()
    for s in sets:
        for r in s.causal.results:
            by_contrast.setdefault(r.contrast_id or "(none)", Counter())[r.verdict] += 1
            totals[r.verdict] += 1
    statuses = ("supported", "not_supported", "unresolved", "control_missing")
    count_rows = "".join(
        "<tr><th>" + escape(c) + "</th>" + "".join(cell(by_contrast[c][v], "v " + v) for v in statuses)
        + cell(sum(by_contrast[c].values())) + "</tr>" for c in sorted(by_contrast))
    count_rows += "<tr><th>all</th>" + "".join(cell(totals[v], "v " + v) for v in statuses) + cell(sum(totals.values())) + "</tr>"
    if scope:
        contrasts = "".join(
            f"<li><strong>{escape(c)}</strong>: {escape(', '.join(v.get('sources', [])))}"
            f"{' (independent cut contrast)' if v.get('independent_cut_contrast') else ''} — {escape(str(v.get('note', '')))}</li>"
            for c, v in sorted((scope.get("contrasts") or {}).items()))
        scope_html = (f'<div class="scope"><h2>Scope</h2><p class="big">{escape(str(scope.get("prompts")))} prompts · '
                      f'{escape(str(scope.get("seeds")))} seeds · about {escape(str(scope.get("independent_contrasts")))} '
                      f'independent contrasts ({escape(", ".join(scope.get("independent_contrast_ids", [])))})</p>'
                      f'<p>{escape(str(scope.get("statement", "")))}</p><p class="muted">{escape(str(scope.get("seeds_note", "")))}</p>'
                      f'<ul>{contrasts}</ul></div>')
    else:
        scope_html = f'<div class="scope"><h2>Scope</h2><p>{escape(str(meta.get("scope", "not stated")))}</p></div>'
    model = meta.get("model") or {}
    identity_html = ("" if not model else
                     f'<p>Model <strong>{escape(str(model.get("name")))}</strong>: identity_status '
                     f'<strong>{escape(str(model.get("identity_status")))}</strong>'
                     f'{" — resolve by: " + escape(str(model.get("resolve_by"))) if model.get("resolve_by") else ""}</p>')
    rule_html = ("" if not rule else
                 f'<details><summary>Verdict rule <code>{escape(str(rule.get("id")))}</code> sha256 '
                 f'<code>{escape(str(rule.get("sha256")))}</code></summary><pre>'
                 f'{escape(json.dumps(rule.get("rule"), indent=1))}</pre></details>')

    head = ("<tr><th>source</th><th>set</th><th>size</th><th>arm</th><th>claim</th><th>verdict</th><th>basis</th>"
            "<th>picture (seeds)</th><th>effect (seeds)</th><th>counted control</th><th>reference controls</th>"
            "<th>floor</th><th>scope / alternatives</th><th>caveat</th><th>reason</th><th>run</th></tr>")
    groups = {}
    for s in sets:
        for r in s.causal.results or (None,):
            contrast = "(no result)" if r is None else (r.contrast_id or "(none)")
            if r is None:
                row = (cell(s.source.name) + cell(s.name) + cell(s.members["count"]) + cell("") + cell("")
                       + cell("unknown", "v unknown") + "".join(cell("") for _ in range(9))
                       + cell("unknown (no intervention on this set)", "muted"))
            else:
                picture = " / ".join((r.picture or {}).get("calls") or [])
                effect = "" if r.effect is None else f"{num(r.effect)} ({seeds(r.effect_per_seed)}) {r.effect_units or ''}"
                refs = " | ".join(control_text(c, counted=False) for c in r.reference_controls)
                scope_text = (f"{r.claim_scope or '—'}; ruled out: {', '.join(r.ruled_out) or 'none'}; "
                              f"remaining: {', '.join(r.remaining_alternatives) or 'none'}")
                caveat = (r.metadata or {}).get("caveat", "")
                row = (cell(s.source.name) + cell(s.name) + cell(s.members["count"])
                       + cell(f"{r.intervention} / {r.cut_type} ({(r.metadata or {}).get('arm', '')})")
                       + cell(r.claim) + cell(r.verdict, "v " + r.verdict) + cell(r.basis) + cell(picture)
                       + cell(effect) + cell(control_text(r.control, counted=True), "ctl") + cell(refs, "ref")
                       + cell(floor_text(r), "floor") + cell(scope_text) + cell(caveat, "caveat")
                       + cell(r.reason or "", "muted") + cell(r.run_id))
            groups.setdefault(contrast, []).append(f'<tr data-source="{escape(s.source.name)}">' + row + "</tr>")
    set_html = "".join(
        f'<h3 id="contrast-{escape(c)}">Contrast: {escape(c)} ({len(rows)} rows)</h3>'
        f'<div class="scroll"><table>{head}{"".join(rows)}</table></div>' for c, rows in sorted(groups.items()))
    read_rows = []
    for r in reads:
        d = r.reader
        read_rows.append(f'<tr data-source="{escape(r.source.name)}">' + cell(r.source.name) + cell(r.source.assignment)
                         + cell(d.block_label or d.block) + cell(d.head) + cell("" if d.kv_group is None else d.kv_group)
                         + cell("all" if d.step is None else d.step) + cell(num(r.mass)) + cell(r.mass_kind, "muted")
                         + cell(r.causal.label, "v " + r.causal.status) + cell(len(r.sets)) + "</tr>")
    options = "".join(f"<option>{escape(n)}</option>" for n in sources)
    html = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MDB symbol reads</title>
<style>:root{{--bg:#0f172a;--fg:#e2e8f0;--muted:#94a3b8;--line:#334155;--ok:#4ade80;--no:#f87171;--un:#fbbf24;--cm:#c084fc}}
body{{font:14px system-ui;margin:24px;color:var(--fg);background:var(--bg)}}table{{border-collapse:collapse;font-size:12px;margin:8px 0 24px}}
th,td{{border:1px solid var(--line);padding:5px 7px;text-align:left;vertical-align:top}}.muted{{color:var(--muted)}}
.v.supported{{color:var(--ok)}}.v.not_supported{{color:var(--no)}}.v.unresolved{{color:var(--un)}}.v.control_missing{{color:var(--cm)}}.v.unknown{{color:var(--muted)}}
td.ctl,td.ref,td.floor,td.caveat{{max-width:340px}}td.caveat{{color:var(--un)}}
.scope{{border:2px solid var(--un);border-radius:8px;padding:8px 16px;margin:12px 0}}.big{{font-size:18px;font-weight:600}}
.scroll{{overflow-x:auto}}select{{padding:6px;background:#1e293b;color:var(--fg);border:1px solid var(--line)}}</style>
<h1>Symbol reads</h1>
{scope_html}
<h2>Results per contrast and status</h2><table><tr><th>contrast</th>{''.join('<th>' + v + '</th>' for v in statuses)}<th>total</th></tr>{count_rows}</table>
{identity_html}{rule_html}
<p>Reads are observations: causal status <strong>unknown</strong> unless a per-read intervention is recorded.
Set results apply to the set as a whole and are not copied onto member heads. Only the counted control (in-run,
matched for the claim) can make a result supported; reference controls are shown but never counted.</p>
<label>Source <select id="src"><option value="">all</option>{options}</select></label>
<h2>Read sets ({len(sets)}), grouped by contrast</h2>{set_html}
<h2>Observed reads ({len(reads)})</h2><div class="scroll"><table><tr><th>source</th><th>role</th><th>block</th><th>head</th><th>KV group</th><th>step</th><th>mass</th><th>mass kind</th><th>causal status</th><th>in sets</th></tr>{''.join(read_rows)}</table></div>
<p class="muted">reads.json fingerprint {escape(loaded['fingerprint'])}</p>
<script>document.getElementById('src').onchange=e=>{{for(const tr of document.querySelectorAll('tr[data-source]'))tr.style.display=!e.target.value||tr.dataset.source===e.target.value?'':'none'}}</script>"""
    (root / "symbol-reads.html").write_text(html)
    return {"file": str(root / "symbol-reads.html"), "reads": len(reads), "sets": len(sets),
            "set_results": sum(len(s.causal.results) for s in sets)}
