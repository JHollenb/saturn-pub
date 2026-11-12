"""Inspect executed resolver decisions, causal effects and canonical symbol uses."""
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
    analysis = root / "analysis"
    load = lambda name: json.loads((analysis / (name+".json")).read_text())
    contexts, arms, heads, resolutions, audit = [load(n) for n in ("contexts","arms","head-uses","resolutions","audit")]
    names = [c["case"] for c in contexts]
    fig, ax = plt.subplots(figsize=(12,5.5))
    x = np.arange(len(names))
    for i,(arm,label,color) in enumerate((("auto-pair","Attention-addressed L20 pair","#087f8c"),
            ("fixed-distance-pair","Fixed distance from prompt end","#94a3b8"),
            ("auto-full-late","Attention-addressed L19–27","#d59b27"),
            ("auto-wrong-payload","Wrong neighboring payload","#ac718e"))):
        ax.bar(x+(i-1.5)*.2,[next((a["margin_effect"] for a in arms if a["case"]==c and a["arm"]==arm),np.nan) for c in names],
            width=.19,label=label,color=color)
    ax.axhline(0,color="#334155",lw=.8)
    ax.set_xticks(x,[c.replace('-', '\n') for c in names])
    ax.set_ylabel("Change in hot − cold logit margin")
    ax.set_title("Can a native attention reader choose the address used for intervention?", pad=53)
    ax.legend(ncol=2,fontsize=9,loc="lower left",bbox_to_anchor=(0,1.01))
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.02,.02,"Qwen2.5-1.5B · BF16/eager CUDA · frozen L20/H5 resolver · one anchor + four fresh contexts · 3 native own tokens per branch",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1))
    for ext in ('png','pdf'): fig.savefig(analysis/('resolver-effects.'+ext),dpi=180,bbox_inches="tight",pad_inches=.2)
    plt.close(fig)
    traces=[]
    for a in arms:
        case=json.loads((root/'results/cases'/(a['case']+'.json')).read_text())
        arm=json.loads((root/'results/arms'/(a['case']+'--'+a['arm']+'.json')).read_text())
        side='donor' if a['arm'] in {'native-donor','auto-reciprocal'} else 'recipient'
        for t in arm['trajectory'][0]['traces']:
            traces.append({'case':a['case'],'arm':a['arm'],'layer':t['layer'],
                'labels':case['token_labels'][side],'allocation':array(root/'results',t['attention'])[0,:,0].tolist()})
    data=json.dumps(dict(contexts=contexts,arms=arms,heads=heads,resolutions=resolutions,audit=audit,traces=traces)).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MDB executed source resolver</title>
<style>body{font:15px system-ui;margin:28px;background:#0f172a;color:#e2e8f0}p{max-width:1100px;line-height:1.5}a{color:#67e8f9}select{padding:9px;background:#243449;color:white;margin:6px}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #475569;white-space:pre-wrap}.scroll{overflow:auto;margin:20px 0}.source{outline:2px solid #fbbf24;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#172337;padding:18px}img{width:100%;max-width:1150px}.controls{position:sticky;top:0;background:#0f172a;padding:8px}</style>
<h1>The resolver used its own nominated address</h1><p>Native partial pass → L20/H5 attention argmax → exact rewind → typed value edit at nominated rows → unchanged native continuation. Reference source spans were withheld from execution and used afterward for evaluation. The reader, task family and intervention cells were preselected; donor payloads were computed natively.</p>
<p id="summary"></p><div id="nominations" class="scroll"></div><img src="resolver-effects.png" alt="Attention-addressed interventions compared with fixed-position and wrong-payload controls">
<div class="controls">Context <select id="case"></select> Arm <select id="arm"></select> Layer <select id="layer"><option>20</option><option>21</option><option>26</option></select> Head <select id="head"></select></div>
<p id="result"></p><div id="matrix" class="scroll"></div><p>Gold cells name the actual nominated source binding. Click to inspect that head's canonical reader use.</p><pre id="canonical"></pre><div id="arms" class="scroll"></div>
<p><a href="audit.json">Independent audit</a> · <a href="resolutions.json">Nomination proofs</a> · <a href="arms.json">Distributions and native futures</a> · <a href="../../proposal.md">Pre-execution proposal</a></p>
<script>const data=DATA,el=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
el('case').innerHTML=data.contexts.map(c=>'<option>'+c.case+'</option>').join('');el('arm').innerHTML=[...new Set(data.arms.map(a=>a.arm))].map(a=>'<option>'+a+'</option>').join('');el('arm').value='auto-pair';el('head').innerHTML=Array.from({length:12},(_,i)=>'<option>'+i+'</option>').join('');el('head').value=5;
function chooseHead(h){el('head').value=h;draw()}function chooseArm(a){el('arm').value=a;draw()}
function draw(){const c=el('case').value,arm=el('arm').value,layer=+el('layer').value,head=+el('head').value,result=data.arms.find(a=>a.case===c&&a.arm===arm),trace=data.traces.find(t=>t.case===c&&t.arm===arm&&t.layer===layer),use=data.heads.find(h=>h.case===c&&h.arm===arm&&h.layer===layer&&h.head===head);
el('summary').textContent=data.audit.source_matches+'/'+data.audit.nominations+' nominations match withheld source rows; '+data.arms.length+' executed branches; '+data.audit.canonical_bindings+' canonical bindings; '+data.audit.counts.canonical_attention_uses+' observed head uses. This is a bounded resolver, with native model computation still required.';
el('nominations').innerHTML='<table><tr><th>Context / side</th><th>Predicted / reference</th><th>Predicted token</th><th>Attention</th><th>Receiver</th></tr>'+data.resolutions.map(r=>'<tr><td>'+esc(r.case+' / '+r.side)+'</td><td>'+r.predicted_row+' / '+r.reference_rows.join(',')+'</td><td>'+esc(r.source_token)+'</td><td>'+(r.source_mass*100).toFixed(2)+'%</td><td>'+r.receiver_row+'</td></tr>').join('')+'</table>';
if(!result||!trace){el('result').textContent='This arm is inapplicable; inspect the audit record.';el('matrix').innerHTML='';el('canonical').textContent='';return}
el('result').innerHTML='<b>'+esc(c+' / '+arm)+'</b> · native future '+esc(JSON.stringify(result.own_text))+' · margin '+result.hot_minus_cold.toFixed(3)+' · parent-relative effect '+result.margin_effect.toFixed(3)+' · nominated '+result.source_row+', edited '+result.target_row+', donor '+result.payload_row+'. <a href="../results/arms/'+c+'--'+arm+'.json">Native receipts and symbol definitions</a>';
el('matrix').innerHTML='<table><tr><th>Reader</th>'+trace.labels.map((t,i)=>'<th>'+i+' '+esc(t)+'</th>').join('')+'</tr>'+trace.allocation.map((a,h)=>'<tr><th>H'+h+' / KV'+Math.floor(h/6)+'</th>'+a.map((p,i)=>'<td '+(i===result.source_row?'class="source" onclick="chooseHead('+h+')"':'')+' style="background:rgba(14,165,233,'+(p*.95)+')">'+(p*100).toFixed(1)+'%</td>').join('')+'</tr>').join('')+'</table>';
el('canonical').textContent=JSON.stringify(use||{status:'No cached-source binding'},null,2);
el('arms').innerHTML='<table><tr><th>Arm</th><th>Native future</th><th>Margin change</th><th>P hot / cold</th></tr>'+data.arms.filter(a=>a.case===c).map(a=>'<tr data-arm="'+esc(a.arm)+'" onclick="chooseArm(this.dataset.arm)"><td>'+esc(a.arm)+'</td><td>'+esc(JSON.stringify(a.own_text))+'</td><td>'+a.margin_effect.toFixed(3)+'</td><td>'+(a.words.hot.probability*100).toFixed(2)+'% / '+(a.words.cold.probability*100).toFixed(2)+'%</td></tr>').join('')+'</table>';
history.replaceState(null,'','#'+new URLSearchParams({case:c,arm,layer,head}));}
for(const id of ['case','arm','layer','head'])el(id).addEventListener('change',draw);draw();</script>'''.replace('DATA',data)
    (analysis/'results.html').write_text(page)
    print(json.dumps({'view':str(analysis/'results.html'),'branches':len(arms),'head_uses':len(heads)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    render(p.parse_args().root.resolve())
