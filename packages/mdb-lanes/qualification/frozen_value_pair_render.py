"""Native margins and symbol-linked attention for frozen-pair prompt variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_allocation_audit import array


def render(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    root = Path(root)
    contexts = json.loads((root / "analysis/contexts.json").read_text())
    arms = json.loads((root / "analysis/arms.json").read_text())
    heads = json.loads((root / "analysis/head-uses.json").read_text())
    audit = json.loads((root / "analysis/audit.json").read_text())
    x = np.arange(len(contexts))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    names = [row["case"] for row in contexts]
    values = {name: {r["arm"]: r for r in arms if r["case"] == name} for name in names}
    for offset, (arm, label, color) in enumerate((("native-recipient", "Native freezing", "#94a3b8"),
            ("pair", "Frozen L20 pair", "#0e8190"), ("full-late", "Full L19–27", "#d69a25"),
            ("native-donor", "Native boiling", "#8b6da5"))):
        ax.bar(x + (offset - 1.5) * .19, [values[name][arm]["hot_minus_cold"] for name in names],
            width=.18, label=label, color=color)
    ax.axhline(0, color="#334155", linewidth=.8)
    ax.set_xticks(x, [name.replace("-", "\n") for name in names])
    ax.set_ylabel("Hot − cold margin (logits)")
    ax.set_title("The source-value interface transfers more broadly than the two-cell subset")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.01), fontsize=9)
    fig.text(.02, .02, "Qwen2.5-1.5B · BF16 eager CUDA · original anchor + 4 prospective variants · fixed cells and supplied source-span resolver · 50 native futures", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(root / "analysis" / ("context-effects." + suffix), dpi=180)
    plt.close(fig)
    view_rows = []
    for arm in arms:
        record = json.loads((root / "results/arms" / (arm["case"] + "--" + arm["arm"] + ".json")).read_text())
        case = json.loads((root / "results/cases" / (arm["case"] + ".json")).read_text())
        side = "donor" if arm["arm"] in {"native-donor", "reciprocal-pair"} else "recipient"
        for trace in record["trajectory"][0]["traces"]:
            view_rows.append({"case": arm["case"], "arm": arm["arm"], "layer": trace["layer"],
                "allocation": array(root / "results", trace["attention"])[0, :, 0].tolist(),
                "token_labels": case["token_labels"][side]})
    data = json.dumps({"contexts": contexts, "arms": arms, "heads": heads, "rows": view_rows,
        "audit": audit}).replace("<", "\\u003c")
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MDB frozen pair prompt transfer</title>
<style>body{font:15px system-ui;background:#0f172a;color:#e2e8f0;margin:28px}p{max-width:1150px;line-height:1.5}a{color:#67e8f9}button,select{padding:9px;margin:5px;background:#1e293b;color:#e2e8f0;border:1px solid #64748b;border-radius:5px;cursor:pointer}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #475569;padding:9px;text-align:left}th{white-space:pre-wrap;max-width:100px}tr.pick{cursor:pointer}tr.pick:hover{background:#243449}.panel{padding:18px;background:#172337;margin:18px 0}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.source{outline:2px solid #fbbf24;cursor:pointer}img{max-width:1100px;width:100%;background:white}.controls{position:sticky;top:0;background:#0f172a;padding:8px 0}</style>
<h1>Frozen layer 20 pair across prompt variants</h1><p>One original anchor and four prospective contexts, with the same layer/KV pair and source-span resolver. Select a context or arm to inspect actual attention, canonical source uses and native outputs. Gold attention cells resolve to registered source symbols.</p>
<div id="summary" class="panel"></div><div id="contexts" class="scroll"></div><img src="context-effects.png" alt="Native and intervened margins across five prompt contexts">
<div class="controls"><label>Context <select id="case"></select></label><label>Arm <select id="arm"></select></label><label>Layer <select id="layer"><option>20</option><option>21</option><option>26</option></select></label><label>Head <select id="head"></select></label></div>
<div id="selected" class="panel"></div><div id="matrix" class="scroll"></div><div id="symbols" class="panel"></div><details><summary>Selected canonical head use</summary><pre id="canonical"></pre></details><h2>All arms in this context</h2><div id="arms" class="scroll"></div>
<p>Names and source spans are supplied. Observed head uses do not establish individual-head necessity. <a href="audit.json">Native audit</a> · <a href="contexts.json">Context results</a> · <a href="arms.json">All distributions</a> · <a href="head-uses.json">Canonical uses</a> · <a href="../../proposal.md">Pre-execution plan</a></p>
<script>const data=DATA,el=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
el('case').innerHTML=data.contexts.map(r=>'<option>'+r.case+'</option>').join('');el('case').value='anchor';el('arm').innerHTML=data.arms.filter(r=>r.case==='anchor').map(r=>'<option>'+r.arm+'</option>').join('');el('arm').value='pair';el('head').innerHTML=Array.from({length:12},(_,i)=>'<option>'+i+'</option>').join('');el('head').value=5;
function chooseCase(c){el('case').value=c;draw()}function chooseArm(a){el('arm').value=a;draw()}function chooseHead(h){el('head').value=h;draw()}function chooseLayer(l){el('layer').value=l;draw()}
function draw(){const c=el('case').value,arm=el('arm').value,layer=+el('layer').value,head=+el('head').value,context=data.contexts.find(r=>r.case===c),result=data.arms.find(r=>r.case===c&&r.arm===arm),row=data.rows.find(r=>r.case===c&&r.arm===arm&&r.layer===layer),uses=data.heads.filter(r=>r.case===c&&r.arm===arm&&r.layer===layer),selected=uses.find(r=>r.head===head);
const fresh=data.contexts.filter(r=>r.split!=='anchor-replay');el('summary').textContent=fresh.filter(r=>r.pair_effect>0).length+'/'+fresh.length+' prospective contexts shift toward hot; '+fresh.filter(r=>r.pair_output.trim()==='hot').length+'/'+fresh.length+' pair outputs are hot. '+data.audit.canonical_bindings+' bindings / '+data.audit.counts.canonical_attention_uses+' observed head uses.';
el('contexts').innerHTML='<table><tr><th>Context</th><th>Source / receiver</th><th>Native freezing / boiling</th><th>Pair effect</th><th>Pair first word</th><th>Full late effect / word</th><th>Pair interaction</th></tr>'+data.contexts.map(r=>'<tr class="pick" data-case="'+esc(r.case)+'" onclick="chooseCase(this.dataset.case)"><td>'+esc(r.case)+'</td><td>'+r.source_row+' / '+r.receiver_row+'</td><td>'+esc(r.native_output)+' / '+esc(r.donor_output)+'</td><td>'+r.pair_effect.toFixed(3)+'</td><td>'+esc(r.pair_output)+'</td><td>'+r.full_late_effect.toFixed(3)+' / '+esc(r.full_output)+'</td><td>'+r.interaction.toFixed(3)+'</td></tr>').join('')+'</table>';
el('selected').innerHTML='<strong>'+esc(c+' / '+arm+' / L'+layer+'/H'+head)+'</strong><p>Actual future: <code>'+esc(JSON.stringify(result.own_text))+'</code>. Hot − cold margin '+result.hot_minus_cold.toFixed(3)+', effect relative to '+esc(result.parent_reference)+' '+result.margin_effect.toFixed(3)+'. P(hot) '+(100*result.words.hot.probability).toFixed(2)+'%, P(cold) '+(100*result.words.cold.probability).toFixed(2)+'%, P(boiling) '+(100*result.words.boiling.probability).toFixed(2)+'%.</p><a href="../results/arms/'+c+'--'+arm+'.json">Full native branch, bindings and receipts</a>';
el('matrix').innerHTML='<table><tr><th>Query / KV</th>'+row.token_labels.map((t,i)=>'<th class="'+(i===context.source_row?'source':'')+'">'+i+' '+esc(t)+'</th>').join('')+'</tr>'+row.allocation.map((a,h)=>'<tr><th><button onclick="chooseHead('+h+')">H'+h+' / KV'+Math.floor(h/6)+'</button></th>'+a.map((p,i)=>'<td '+(i===context.source_row?'class="source" onclick="chooseHead('+h+')"':'')+' style="background:rgba(14,165,233,'+(p*.95)+')">'+(p*100).toFixed(1)+'%</td>').join('')+'</tr>').join('')+'</table>';
el('symbols').innerHTML='<strong>Source symbols for this native invocation</strong><p>Select a binding to open its reader layer.</p>'+[20,21,26].map(l=>{const u=data.heads.find(r=>r.case===c&&r.arm===arm&&r.layer===l);return '<button onclick="chooseLayer('+l+')">L'+l+' · '+u.binding_fingerprint.slice(0,16)+'</button>'}).join('');el('canonical').textContent=JSON.stringify(selected,null,2);
el('arms').innerHTML='<table><tr><th>Arm</th><th>Native future</th><th>Margin</th><th>Parent-relative effect</th><th>P hot</th><th>P cold</th></tr>'+data.arms.filter(r=>r.case===c).map(r=>'<tr class="pick" data-arm="'+esc(r.arm)+'" onclick="chooseArm(this.dataset.arm)"><td>'+esc(r.arm)+'</td><td>'+esc(JSON.stringify(r.own_text))+'</td><td>'+r.hot_minus_cold.toFixed(3)+'</td><td>'+r.margin_effect.toFixed(3)+'</td><td>'+(100*r.words.hot.probability).toFixed(2)+'%</td><td>'+(100*r.words.cold.probability).toFixed(2)+'%</td></tr>').join('')+'</table>';
history.replaceState(null,'','#'+new URLSearchParams({case:c,arm,layer,head}).toString());}
const params=new URLSearchParams(location.hash.slice(1));for(const id of ['case','arm','layer','head']){const value=params.get(id);if(value!==null&&Array.from(el(id).options).some(o=>o.value===value))el(id).value=value;el(id).addEventListener('change',draw)}draw();</script>'''.replace("DATA", data)
    (root / "analysis/results.html").write_text(page)
    print(json.dumps({"file": str(root / "analysis/results.html"), "contexts": len(contexts), "native_arms": len(arms)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root)
