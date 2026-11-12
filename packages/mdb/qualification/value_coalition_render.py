"""Plot signed coalition effects and expose their canonical symbol/head records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    root = Path(directory)
    trends = json.loads((root / "trends.json").read_text())
    arms = json.loads((root / "arms.json").read_text())
    heads = json.loads((root / "head-uses.json").read_text())
    cells = trends["cells"]
    labels = [f"L{r['layer']} / KV{r['kv_head']}" for r in cells]
    first = [r["singleton_effect"] for r in cells]
    loss = [r["complement_loss"] for r in cells]
    mass = [max(h["source_mass"] for h in heads if h["arm"] == "native-recipient"
        and h["layer"] == row["layer"] and h["kv_head"] == row["kv_head"]) for row in cells]
    fig, ax = plt.subplots(1, 3, figsize=(13, 7), sharey=True, gridspec_kw={"width_ratios": [1, 1, .85]})
    y = np.arange(len(cells))
    for panel, values, title in ((ax[0], first, "Single donor cell in recipient"),
            (ax[1], loss, "Loss when cell is omitted from full donor")):
        panel.barh(y, values, color=["#137c8b" if v >= 0 else "#cf7045" for v in values])
        panel.axvline(0, color="#334155", linewidth=.8)
        panel.set_title(title, fontsize=10)
        panel.set_xlabel("Change in hot − cold margin (logits)")
        panel.grid(axis="x", alpha=.2)
        panel.set_axisbelow(True)
    ax[0].set_yticks(y, labels)
    ax[0].invert_yaxis()
    ax[2].barh(y, np.asarray(mass) * 100, color="#64748b")
    ax[2].set_xlabel("Largest source attention among 6 heads (%)")
    ax[2].set_title("Native recipient source allocation", fontsize=10)
    ax[2].set_xlim(0, 100)
    ax[2].grid(axis="x", alpha=.2)
    fig.suptitle("Source value influence depends on the surrounding coalition", fontsize=15)
    fig.text(.02, .015, "Qwen2.5-1.5B · BF16 eager CUDA · one freezing/boiling context · 56 branches · observed attention is not individual-head necessity", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    for suffix in ("png", "pdf"):
        fig.savefig(root / ("cell-effects." + suffix), dpi=180)
    plt.close(fig)
    data = json.dumps({"trends": trends, "arms": arms, "heads": heads}).replace("<", "\\u003c")
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MDB causal source value map</title>
<style>body{font:16px system-ui;background:#0f172a;color:#e2e8f0;margin:28px;max-width:1400px}p{line-height:1.5;max-width:1100px}a{color:#67e8f9}button,select{padding:10px;background:#1e293b;color:#e2e8f0;border:1px solid #64748b;border-radius:5px;cursor:pointer;margin:5px}table{border-collapse:collapse;font-size:13px}td,th{padding:9px;border:1px solid #475569;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#172337;padding:16px;font-size:12px}.grid{display:grid;grid-template-columns:repeat(9,minmax(90px,1fr));gap:8px;overflow:auto}.cell{min-width:90px;text-align:center}.panel{background:#172337;padding:18px;margin:18px 0}img{max-width:100%;background:white}.active{outline:2px solid #fbbf24}.muted{color:#94a3b8}</style>
<h1>MDB causal source value map</h1><p>Select a layer/KV cell to compare its isolated transfer with its conditional contribution to the full donor coalition. Each selection exposes the actual native source symbol and six observed query-head uses sharing that KV group.</p>
<p class="muted">One supplied freezing/boiling context. Complement loss measures contribution to this transplanted coalition; individual-head necessity and prompt generalization remain untested. Full output distributions and controls are retained.</p>
<label>Cell effect <select id="mode"><option value="single">Singleton donor transfer</option><option value="complement">Omit cell from full donor</option></select></label><div id="grid" class="grid"></div>
<div class="panel" id="selected"></div><div class="panel" id="readers"></div><details><summary>Canonical uses and native source cut</summary><pre id="canonical"></pre></details>
<h2>All cells and observed attention</h2><img src="cell-effects.png" alt="Singleton and complement margin effects alongside native source attention">
<h2>Controls and coalition outputs</h2><div id="arms"></div><p><a href="audit.json">Independent audit</a> · <a href="trends.json">Signed effects and interactions</a> · <a href="head-uses.json">Canonical head uses</a> · <a href="../../proposal.md">Pre-execution proposal</a></p>
<script>const data=DATA,el=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));let layer=20,kv=0;
function choose(l,k){layer=l;kv=k;draw()}
function draw(){const mode=el('mode').value,cell=data.trends.cells.find(c=>c.layer===layer&&c.kv_head===kv),name=(mode==='single'?'only':'without')+'-L'+layer+'-KV'+kv,arm=data.arms.find(a=>a.arm===name),heads=data.heads.filter(h=>h.arm===name&&h.layer===layer&&h.kv_head===kv);
el('grid').innerHTML=[0,1].map(k=>data.trends.cells.filter(c=>c.kv_head===k).map(c=>{const v=mode==='single'?c.singleton_effect:c.complement_loss;return '<button class="cell '+(c.layer===layer&&k===kv?'active':'')+'" onclick="choose('+c.layer+','+k+')">L'+c.layer+' / KV'+k+'<br>'+v.toFixed(3)+' logits</button>'}).join('')).join('');
const nativeHead=data.heads.filter(h=>h.arm==='native-recipient'&&h.layer===layer&&h.kv_head===kv).sort((a,b)=>b.source_mass-a.source_mass)[0];el('selected').innerHTML='<strong>'+esc(name)+'</strong><p>Singleton effect: '+cell.singleton_effect.toFixed(3)+' logits. Loss from full coalition when omitted: '+cell.complement_loss.toFixed(3)+' logits.</p><p>Actual three-token future: <code>'+esc(JSON.stringify(arm.own_text))+'</code>. Hot − cold margin: '+arm.hot_minus_cold.toFixed(3)+'. P(hot or boiling): '+(100*arm.hot_or_boiling_probability).toFixed(2)+'%. P(cold): '+(100*arm.words.cold.probability).toFixed(2)+'%.</p><a href="../results/arms/'+name+'.json">Full native branch, output bindings and receipts</a> · <a href="../../../mdb-attention-symbols-20261111/join-02/symbols-attention.html#arm=native&step=0&layer='+layer+'&head='+nativeHead.head+'">Matching native reader in the earlier symbol map</a>';
el('readers').innerHTML='<strong>Observed readers of the selected native source symbol</strong><p>'+(mode==='complement'?'The selected source cell retains recipient values while the other 17 carry donor values.':'Only the selected source cell carries donor values.')+'</p><table><tr><th>Head</th><th>Source attention</th><th>Source value return norm</th></tr>'+heads.map(h=>'<tr><td>H'+h.head+' / KV'+h.kv_head+'</td><td>'+(100*h.source_mass).toFixed(2)+'%</td><td>'+h.source_value_return_norm.toFixed(4)+'</td></tr>').join('')+'</table>';
el('canonical').textContent=JSON.stringify(heads,null,2);
const controls=data.arms.filter(a=>(!a.arm.startsWith('without-')&&!a.arm.startsWith('only-L'))||a.arm==='only-L20-both');el('arms').innerHTML='<table><tr><th>Arm</th><th>Native future</th><th>Hot − cold</th><th>P hot or boiling</th><th>P cold</th></tr>'+controls.map(a=>'<tr><td>'+esc(a.arm)+'</td><td>'+esc(JSON.stringify(a.own_text))+'</td><td>'+a.hot_minus_cold.toFixed(3)+'</td><td>'+(100*a.hot_or_boiling_probability).toFixed(2)+'%</td><td>'+(100*a.words.cold.probability).toFixed(2)+'%</td></tr>').join('')+'</table>';
history.replaceState(null,'','#'+new URLSearchParams({layer,kv,mode}).toString());}
const params=new URLSearchParams(location.hash.slice(1));if(data.trends.cells.some(c=>c.layer===+params.get('layer')&&c.kv_head===+params.get('kv'))){layer=+params.get('layer');kv=+params.get('kv')}if(['single','complement'].includes(params.get('mode')))el('mode').value=params.get('mode');el('mode').addEventListener('change',draw);draw();</script>'''.replace("DATA", data)
    (root / "causal-map.html").write_text(page)
    print(json.dumps({"html": str(root / "causal-map.html"), "cells": len(cells), "arms": len(arms)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
