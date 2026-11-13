"""Audit custody, sparse complete fields, trace witnesses and selected native ports."""
import argparse
import json
import math
import runpy
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())


def analyze(run,destination,discovery,replication):
    from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb_lanes.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/'extracted/interaction-trace-result';custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](run)
    util=runpy.run_path(str(here/'analyze_flux_scene_dependencies.py'));sha=util['sha'];typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));q=typed['quantize_bf16'];route=runpy.run_path(str(here/'analyze_flux_concept_routes.py'));json_tree=runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree']
    index=runpy.run_path(str(here/'analyze_flux_edit_onset.py'))['index_receipts'];public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt']
    config,summary,rows,receipts,selected=[read(data/n) for n in ('plan.json','summary.json','rows.json','receipts.json','selected.json')];request=read(run/'request.json')
    assert config==request['config'] and summary==read(run/'report.json')['report'] and summary['request_sha256']==request['request_sha256']==custody['request_sha256']
    for seed,root in [('60666',discovery),('60667',replication)]:
        assert sha(root/'analysis-v1.json')==config['sources'][seed]['audit_sha256'];old=root/'run-01/extracted/shape-formation-result'
        for name,m in config['sources'][seed]['input_files'].items():assert sha(old/name)==m['sha256'] and (old/name).stat().st_size==m['bytes']
    for name,h in request['source_hashes'].items():assert sha(run/'payload'/name)==h
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256']);assert tx['verified'] and tx['phase']=='closed'
    assert json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/forecasts/trace_policy.json')['predictions']))==config['hypotheses']
    assert summary['counts']==config['expected_counts']==dict(loads=1,encodes=0,noise_draws=0,native_transitions=448,acts=0,receipts=448,images=8,candidate_blocks=8)
    for r in receipts:public(r)
    by=index(receipts);assert len(receipts)==448 and all(r['operation']=='continue' for r in receipts)
    audited=typed['audit_store'](data/'store')
    def slot(fp,s):return route['slot_array'](data/'store',fp,s,typed)
    def interaction(v):return v['11'].astype(np.float64)-v['10']-v['01']+v['00']
    close=lambda x,y:math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-8)
    results={};witnesses=0;complete=0;controls={}
    for seed in ('60666','60667'):
        trace=read(data/('trace-'+seed+'.json'));assert len(trace)==57 and trace[0]['point']['logical_step']==2 and trace[0]['point']['phase']=='project' and trace[-1]['point']['logical_step']==4
        prev={};ranking=[]
        for i,row in enumerate(trace):
            if i:
                assert row['ordinal']==i-1 and row['operator']==trace[i-1]['point']
                for b in ('00','10','01','11'):
                    receipt=next(r['path'][i-1] for r in rows if r['id']==seed+'.'+b);assert receipt==by[receipt['fingerprint']] and receipt['parent']==trace[i-1]['cuts'][b] and receipt['result']==row['cuts'][b]
            for s,m in row['measurements'].items():
                v={b:np.asarray(x,np.float64) for b,x in m['witnesses'].items()};c=interaction(v);assert np.array_equal(c,np.asarray(m['interaction_witnesses']));witnesses+=len(c)
                if row['full_cuts_retained'] or all(fp in audited['cuts'] for fp in row['cuts'].values()):
                    values={b:slot(fp,s) for b,fp in row['cuts'].items()};C=interaction(values);D=values['11'].astype(np.float64)-values['00'];den=float(np.sum(D*D))
                    assert all(np.array_equal(values[b].reshape(-1)[m['indices']],v[b]) for b in v)
                    assert close(float(np.linalg.norm(C)),m['l2']) and close(float(C.mean()),m['signed_mean']) and close(float(np.max(np.abs(C))),m['max_abs'])
                    assert (m['source_contrast_projection'] is None and den==0) or close(float(np.sum(C*D)/den),m['source_contrast_projection']);complete+=1
                if s in prev and 'write_interaction_l2' in m:assert m['write_interaction_l2']>=0
            if i and row['operator']['phase']=='single' and row['operator']['index']<19:
                ranking.append({'point':row['operator'],'ordinal':i-1,'image_write_interaction_l2':row['measurements']['image']['write_interaction_l2']})
            prev=row['measurements']
        ranked=sorted(ranking,key=lambda r:r['image_write_interaction_l2'],reverse=True);sel=selected[seed];assert sel==read(data/('selected-'+seed+'.json'))
        assert sel['point']==ranked[0]['point'] and sel['ordinal']==ranked[0]['ordinal'] and sel['score']==ranked[0]['image_write_interaction_l2']
        before={b:slot(fp,'image') for b,fp in sel['before'].items()};after={b:slot(fp,'image') for b,fp in sel['after'].items()};delta=interaction(after)-interaction(before);assert close(float(np.linalg.norm(delta)),sel['score'])
        ports={b:{k:load_tensor_archive(data,r[k]) for k in ('concat','projection','gate')} for b,r in sel['ports'].items()}
        for b,p in ports.items():
            t=sel['ports'][b]['text_rows'];R=np.concatenate((slot(sel['before'][b],'text'),before[b]),axis=1)
            calculated=q((R.astype(np.float64)+q((p['gate'].astype(np.float64)*p['projection']).astype(np.float32))).astype(np.float32))
            assert np.array_equal(calculated[:,t:],after[b]) and np.array_equal(calculated[:,:t],slot(sel['after'][b],'text'))
            assert np.array_equal(p['gate'],ports['00']['gate'])
        results[seed]={'selected':sel['point'],'score':sel['score'],'top_single_nodes':ranked[:8],'selected_write_rebuilt_l2':float(np.linalg.norm(delta)),
            'project_and_scheduler_trace':[{'point':r['point'],'latent':r['measurements']['latent']} for r in trace if r['point']['phase']=='project'],
            'port_scope':'All4fullconcatenatedA/M, nativeP/gate andtypedpre/postI/T; fullnativeBF16exit equation independentlyrebuilt. UpstreamA/Mformation observed.'}
    for row in rows:
        assert row['png_sha256']==sha(data/row['png']) and row['png_control_exact']==(row['png_sha256']==row['expected_png']);controls[row['id']]=row['png_control_exact'];assert len(row['path'])==56
        previous=row['parent']
        for r in row['path']:assert r==by[r['fingerprint']] and r['parent']==previous;previous=r['result']
        assert previous==row['final_cut']
    report={'schema':'mdb-interaction-trace-audit-v1','job_id':summary['job_id'],'request_sha256':summary['request_sha256'],'mechanics_status':'verified','instrument_status':'sparse-full-field-plus-witness-trace','trend_status':'diagnostic-candidates-await-causal-test','terminal_status':'unassessed','custody':custody,'transaction':tx,'counts':summary['counts'],'distinct_receipt_ids':len(by),'controls':controls,'seeds':results,
        'witness_values_verified':witnesses,'complete_boundary_slot_comparisons':complete,'worker_seconds':summary['worker_seconds'],
        'coverage':'114clockrecords.64fixedwitnesses/fullmetricssource-bound exceptretainedsparse/selectedcuts. Selectedrank recomputedfromsource-boundscoretable; selectedfullwriteL2 independentlyrebuilt. EightPNGnativecontrols. Energyranking diagnosesamplitude, notsemanticresponsibility ornecessity.'}
    with destination.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return {'job_id':summary['job_id'],'controls':controls,'seeds':{k:{'selected':v['selected'],'score':v['score'],'top_single_nodes':v['top_single_nodes']} for k,v in results.items()},'complete_comparisons':complete,'witness_values':witnesses}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--discovery',type=Path,required=True);p.add_argument('--replication',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.run.resolve(),a.output.resolve(),a.discovery.resolve(),a.replication.resolve()),sort_keys=True))
