"""Reconstruct route inputs, verify chronology and retain frozen-transfer misses."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def worker():
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='2'
    import numpy as np

    from mdb_lanes.equation_comparison import EquationCatalog, compare_equations
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction
    h=runpy.run_path('flux_source_receiver.py');read,save,sha,load=[h[k] for k in ('read','save','sha','load_array')]
    program=runpy.run_path('frozen-route-program.py')
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=req['config'];root,source=Path(c['native_root']),Path(c['source_root'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('audit payload/isolation differs')
    if sha(root/'files.json')!=c['native_manifest_sha256'] or sha(source/'files.json')!=c['source_manifest_sha256']:raise ValueError('inventories changed')
    if any(sha(root/n)!=v for n,v in read(root/'files.json').items()):raise ValueError('native data changed')
    native=read(root/'summary.json');plan=read(root/'plan.json')
    if any(sha(source/n)!=v['sha256'] or (source/n).stat().st_size!=v['bytes'] for n,v in plan['inputs'].items()):raise ValueError('source changed')
    if native['counts']!=plan['planned_counts']:raise ValueError('native primitive census differs')
    started=time.monotonic();rows=read(root/'cases.json');anchors=read(root/'source-anchors.json');summaries={};anchor_matches={};recipes=0
    for receiver,items in rows.items():
        anchor_matches[receiver]={}
        for name,row in items.items():
            r=row['head_recipe']
            if r['kind']=='archive':head=load(root,r['archive'])
            else:
                head=load(root,r['base']);alternate=load(root,r['alternate']);width=head.shape[-1]//r['heads_total']
                for i in r['heads']:head[...,i*width:(i+1)*width]=alternate[...,i*width:(i+1)*width]
            bits=(np.ascontiguousarray(head,np.float32).view(np.uint32)>>16).astype('<u2')
            if hashlib.sha256(bits.tobytes()).hexdigest()!=row['head_payload_sha256']:raise ValueError('lossless input recipe differs')
            recipes+=1
        for name in ('base','full'):
            anchor_matches[receiver][name]=dict(head=items[name]['head_recipe']['archive']['sha256']==anchors[receiver][name]['head']['sha256'],
                text=items[name]['text']['sha256']==anchors[receiver][name]['text']['sha256'])
        summaries[receiver]=program['summarize'](items,root,load)
    if recipes!=116:raise ValueError('incomplete route panel')
    verify_transaction(root/'transaction')
    forecast=load_forecast_tree(root/'transaction',read(root/'transaction/forecasts/routes.json')['predictions'])
    left=load_forecast_tree(root/'transaction',read(root/'transaction/frontiers/left.json')['value'])
    if json.dumps(left,sort_keys=True,default=dict)!=json.dumps(read(root/'left-summary.json'),sort_keys=True):raise ValueError('forecast did not bind native left summary')
    timeline=[json.loads(x) for x in (root/'timeline.jsonl').read_text().splitlines()]
    if [x['kind'] for x in timeline]!=['receiver-complete','right-prediction-sealed','receiver-complete'] or timeline[0]['counts']!=timeline[1]['counts'] or timeline[1]['counts']['layernorm']!=58:
        raise ValueError('right forecast chronology differs')
    actual=summaries['right'];comparison={}
    for key in ('solo_heads','removed_heads','solo_bands','removed_bands'):
        p,n=list(forecast[key]),actual[key];size=min(4,len(p))
        comparison[key]=dict(predicted=p,observed=n,top1_matches=p[0]==n[0],top4_overlap=len(set(p[:size])&set(n[:size])),
            rank_correlation=float(np.corrcoef(np.argsort(p),np.argsort(n))[0,1]))
    comparison['solo_signs']=dict(predicted=list(forecast['solo_signs']),observed=actual['solo_signs'],
        matches=sum(p==n for p,n in zip(forecast['solo_signs'],actual['solo_signs'])),
        counterexample_heads=[i for i,(p,n) in enumerate(zip(forecast['solo_signs'],actual['solo_signs'])) if p!=n])
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    result=dict(schema='mdb-reader-route-audit-v1',source_job=native['job_id'],mechanics_status='verified',terminal_status='unassessed',trend_status='observed',
        instrument_status='source-matched' if all(v for r in anchor_matches.values() for p in r.values() for v in p.values()) else 'anchor-mismatch-observed',
        source_anchor_archives=anchor_matches,head_recipes_verified=recipes,counts=native['counts'],summaries=summaries,frozen_prediction_comparison=comparison,
        source_manifest_sha256=sha(root/'files.json'),original_reader_manifest_sha256=sha(source/'files.json'),
        seal_sha256=read(root/'transaction-report.json')['seal_sha256'],chronology=timeline,
        scope=plan['scope'],limitations=['Two known receiving contexts and new intervention states; not fresh seeds or prompts.',
            'Effects measured on next text state; image continuation and object semantics not tested here.',
            'Single-head sufficiency/removal do not establish minimal coalitions or additive attribution.',
            'Signed target fraction is projection on each context full K/V text change, not a semantic score.'])
    save(out/'audit.json',result);catalog=EquationCatalog(out/'catalog/equations');records=[]
    profile=read(root/'native-contract.json')
    for kind,count in (('head',24),('band',4)):
        for i in range(count):
            plus='solo_heads' if kind=='head' else 'solo_bands';minus='removed_heads' if kind=='head' else 'removed_bands'
            predicted=dict(addition_rank=list(forecast[plus]).index(i),removal_rank=list(forecast[minus]).index(i))
            observed=dict(addition_rank=actual[plus].index(i),removal_rank=actual[minus].index(i))
            if kind=='head':predicted['effect_sign']=forecast['solo_signs'][i];observed['effect_sign']=actual['solo_signs'][i]
            context=dict(model='flux2-klein-4b-distilled',source_job=native['job_id'],source_receiver='left',receiver='right',component=f'{kind}{i}',
                clock=dict(step=3,phase='joint',block=0),horizon=1,numerical=dict(dtype='bfloat16',device=profile['device'],bf16_reduced_reduction=profile['bf16_reduced_reduction']))
            record=catalog.record(law_id='law.flux.reader.route-rank-transfer.v1',route_id=f'route.flux.reader.{kind}{i}',alias=f'cmp.flux.reader-route.{kind}{i}',
                binding=dict(model_identity=context['model'],input_identity=plan['parents'],source_identity=native['request_sha256'],execution=context['numerical'],frontier=context['clock'],horizon=1),
                report=compare_equations(predicted,observed),conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],tested_context=context,
                readers=['joint0.text-head','joint0.text-tail'],evidence=dict(audit_sha256=sha(out/'audit.json'),seal_sha256=result['seal_sha256'],
                    role='prospective left-to-right ranking/sign transfer on new reader interventions in known contexts',terminal_status='unassessed'))
            wrong={**context,'horizon':2}
            if len(catalog.find(applicable_to=context))!=1 or catalog.find(applicable_to=wrong):raise ValueError('catalog applicability differs')
            if kind=='head' and i==summaries['left']['solo_heads'][0]:
                save(out/'catalog/context-selected.json',context);save(out/'catalog/context-wrong-horizon.json',wrong)
            records.append(record['fingerprint'])
    save(out/'catalog/provenance.json',dict(records=len(records),fingerprints=records,matching_queries=len(records),wrong_horizon_abstentions=len(records)))
    for name in ('native-contract.json','summary.json','cases.json','profiles.json'):shutil.copyfile(root/name,out/('source-'+name))
    shutil.copytree(root/'transaction',out/'transaction')
    for name in ('parameter-artifacts.json','retrieve_weights.py'):shutil.copyfile(root/name,out/name)
    save(out/'external-evidence.json',dict(host='beast',roots=dict(native=str(root),reader=str(source)),
        manifests=dict(native=read(root/'files.json'),reader=read(source/'files.json')),scope='Full text output tensors retained remotely; input head mixtures encoded losslessly by source arrays and head IDs.'))
    shutil.copyfile('retrieve_reader_evidence.py',out/'retrieve_evidence.py')
    save(out/'plan.json',c);save(out/'request.json',req)
    save(out/'summary.json',dict(schema='mdb-reader-route-compact-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],
        output_root=str(out),source_job=native['job_id'],seconds=time.monotonic()-started,mechanics_status='verified',records=len(records),terminal_status='unassessed'))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--study',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        helper=runpy.run_path(str(Path(__file__).with_name('flux_reader_attention_audit.py')))
        print(json.dumps(helper['submit'](a.run.resolve()) if a.submit else helper['stage'](a.run.resolve(),a.study.resolve(),worker_path=Path(__file__),
            extra_files=[(a.study.resolve()/'payload/flux_reader_routes.py','frozen-route-program.py')],audit_kind='routes',logical_operations=145)))
