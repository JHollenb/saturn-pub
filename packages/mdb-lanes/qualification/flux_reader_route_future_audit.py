"""Verify sealed reader forecasts, first-divergence clocks and native futures."""
import argparse
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def value_identity(record):
    if record is None or (record.get('kind')=='scalar' and record.get('type')=='none' and record.get('value') is None):return None
    digest=record.get('sha256',record.get('tensor'))
    if not isinstance(digest,str) or 'shape' not in record or 'dtype' not in record:raise ValueError('unsupported live slot descriptor')
    return digest,record['dtype'],tuple(record['shape'])


def trajectory(case,reference,field):
    changes=[];layout=[];empty=[]
    for a,b in zip(case['records'],reference['records']):
        if a['point']!=b['point']:raise ValueError('continuation clocks differ')
        x,y=a['fields'].get(field),b['fields'].get(field)
        xi,yi=value_identity(x),value_identity(y)
        changes.append(None if xi is None and yi is None else xi!=yi)
        empty.append(xi is None and yi is None)
        if xi is not None and yi is not None and x.get('stride')!=y.get('stride'):layout.append(a['local'])
    first=next((i for i,v in enumerate(changes) if v),None)
    reunions=[i for i,v in enumerate(changes) if first is not None and i>first and v is False]
    cleared=[i for i in range(1,len(empty)) if empty[i] and not empty[i-1]]
    return dict(first_value_divergence=first,changed_by_local=changes,exact_slot_reunions=reunions,layout_difference_locals=layout,slot_clearance_locals=cleared,
        magnitude_scope='Fingerprints establish exact equality/difference, not intermediate effect sizes or approximate repair.')


def worker():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
    import numpy as np
    from PIL import Image

    from mdb_lanes.equation_comparison import EquationCatalog, compare_equations
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction
    h=runpy.run_path('flux_source_receiver.py');read,save,sha,load=[h[k] for k in ('read','save','sha','load_array')]
    program=runpy.run_path('frozen-future-program.py');ref=runpy.run_path('flux_reentry_reference.py')
    measure=runpy.run_path('analyze_flux_reader_arithmetic.py')['measure']
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=req['config'];root,source=Path(c['native_root']),Path(c['source_root'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('audit source/isolation differs')
    if sha(root/'files.json')!=c['native_manifest_sha256'] or sha(source/'files.json')!=c['source_manifest_sha256']:raise ValueError('inventories changed')
    if any(sha(root/n)!=v for n,v in read(root/'files.json').items()):raise ValueError('native output changed')
    plan=read(root/'plan.json');summary=read(root/'summary.json');native_req=read(root/'request.json')
    if summary['counts']!=plan['planned_counts'] or native_req['config']!=plan or summary['request_sha256']!=native_req['request_sha256']:raise ValueError('native execution census/binding differs')
    if any(sha(source/n)!=v['sha256'] or (source/n).stat().st_size!=v['bytes'] for n,v in plan['inputs'].items()):raise ValueError('source inputs changed')
    started=time.monotonic();cases=read(root/'cases.json');predictions=read(root/'predictions.json');timeline=[json.loads(x) for x in (root/'timeline.jsonl').read_text().splitlines()]
    verify_transaction(root/'transaction',expected_seal_sha256=read(root/'transaction-report.json')['seal_sha256'])
    seals=[i for i,x in enumerate(timeline) if x['kind']=='all-predictions-sealed'];future_events=[i for i,x in enumerate(timeline) if x['kind']=='native-future-complete']
    if len(seals)!=1 or len(future_events)!=16 or min(future_events)<=seals[0] or timeline[seals[0]]['counts']['native_blocks']!=0 or timeline[seals[0]]['counts']['native_sdpa']!=0:
        raise ValueError('forecast/native chronology differs')
    def rgb(key):return np.asarray(Image.open(root/cases[key]['image']['path']).convert('RGB'))
    def effect(base,full,value):
        b,d,v=[np.asarray(x,np.float64) for x in (base,full,value)];actual,predicted=d-b,v-b
        report=measure(actual,predicted);den=float(np.vdot(actual,actual))
        report['target_fraction']=float(np.vdot(actual,predicted)/den) if den else None
        report['changed_elements_from_base']=int(np.count_nonzero(predicted));report['baseline_rms']=float(np.sqrt(np.mean(b*b)))
        return report
    native_old=read(source/'readers.json');prefixes=read(source/'prefixes.json')['operands']
    reports={};controls={};fresh=[]
    for receiver in ('left','right'):
        b,d=[{f:load(source,a) for f,a in prefixes[receiver+'.'+n].items()} for n in ('base','delta')]
        base,full=cases[receiver+'.base'],cases[receiver+'.full']
        base_latent,full_latent=[load(root,x['latent']) for x in (base,full)];base_rgb,full_rgb=rgb(receiver+'.base'),rgb(receiver+'.full')
        for kind,oldkey in (('base','k0v0'),('full','k1v1')):
            old=native_old[receiver+'.'+oldkey];current=cases[receiver+'.'+kind]
            old_latent=old['records'][-1]['outputs']['latent']['archive']
            if sha(source/old['image']['path'])!=old['image']['sha256']:raise ValueError('old control PNG changed')
            load(source,old_latent)
            controls[receiver+'.'+kind]=dict(png_bytes_exact=current['image']['sha256']==old['image']['sha256'],
                latent_archive_exact=current['latent']['sha256']==old_latent['sha256'],
                text_archive_exact=current['first_fields']['text']['sha256']==old['fields']['text']['sha256'])
        for spec in program['specs'](plan['selected_head'],plan['selected_band']):
            key=receiver+'.'+spec['id'];case=cases[key];prediction=predictions[key]
            frozen=load_forecast_tree(root/'transaction',read(root/'transaction/forecasts'/(key+'.json'))['predictions'])
            if json.dumps(frozen,sort_keys=True,default=dict)!=json.dumps(prediction,sort_keys=True):raise ValueError('published prediction differs from seal')
            kv=program['ports'](b,d,spec,ref['bf16'])
            if {f:program['payload_digest'](v) for f,v in kv.items()}!=prediction['ports'] or prediction['ports']!=case['footprint']['kv_sha256']:raise ValueError('sealed/installed/reconstructed KV differs')
            fields={f:measure(load(root,case['first_fields'][f]),load(root,a)) for f,a in prediction['fields'].items()}
            for f in fields:fields[f]['archive_bytes_exact']=case['first_fields'][f]['sha256']==prediction['fields'][f]['sha256']
            target_case=cases[receiver+'.'+prediction['structural_prediction']['reference']]
            image_path=trajectory(case,target_case,'image');text_path=trajectory(case,target_case,'text');latent_path=trajectory(case,target_case,'latent')
            expected=prediction['structural_prediction']['expected_first_image_change']
            candidate_rgb=rgb(key);candidate_latent=load(root,case['latent'])
            result=dict(spec=spec,reader_fields=fields,continuation_reference=prediction['structural_prediction']['reference'],
                image_trajectory=image_path,text_trajectory=text_path,latent_trajectory=latent_path,
                predicted_first_image_divergence=expected,first_image_prediction_matches=expected==image_path['first_value_divergence'],
                first_image_archive_sha256=case['first_image']['sha256'],
                first_image_reference_archive_exact=case['first_image']['sha256']==target_case['first_image']['sha256'],
                final_latent_vs_complete_kv_effect=effect(base_latent,full_latent,candidate_latent),
                final_rgb_vs_complete_kv_effect=effect(base_rgb,full_rgb,candidate_rgb),
                rgb_signed_mean_from_base=[float(x) for x in (candidate_rgb.astype(np.float64)-base_rgb).mean(axis=(0,1))],
                endpoint_vs_reference=dict(latent_exact=case['latent']['sha256']==target_case['latent']['sha256'],
                    png_bytes_exact=case['image']['sha256']==target_case['image']['sha256'],
                    rgb=measure(rgb(receiver+'.'+prediction['structural_prediction']['reference']),candidate_rgb)))
            if spec['kind']=='dose':fresh.append(key)
            reports[key]=result
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    result=dict(schema='mdb-reader-route-future-audit-v1',source_job=summary['job_id'],mechanics_status='verified',terminal_status='unassessed',trend_status='observed',
        instrument_status='controls-replay-qualified' if all(v for x in controls.values() for v in x.values()) else 'control-mismatch-observed',
        counts=summary['counts'],reader_forecasts=reports,new_dose_cases=fresh,historical_controls=controls,
        native_manifest_sha256=sha(root/'files.json'),original_reader_manifest_sha256=sha(source/'files.json'),seal_sha256=read(root/'transaction-report.json')['seal_sha256'],
        limitations=['Four new dose reader states; head/band local exits were already measured in the route screen. All intervention futures are new in this campaign.',
            'Forecasts execute native Flash/linear/normalization/SwiGLU primitives; not an independent Flash kernel implementation.',
            'No numerical whole-image prediction was made; native continuation outcomes and structural timing predictions are separate.',
            'Intermediate trajectory fingerprints do not measure approximate repair magnitudes or object semantics.',
            'Two known contexts, seed127; no fresh prompt/seed generalization claim.'])
    save(out/'audit.json',result);catalog=EquationCatalog(out/'catalog/equations');ids=[]
    profile=read(root/'native-contract.json')
    for key,case in cases.items():
        prediction=predictions[key];report=reports[key]
        p={f:load(root,a) for f,a in prediction['fields'].items()};n={f:load(root,a) for f,a in case['first_fields'].items()}
        p['first_image_divergence']=-1 if report['predicted_first_image_divergence'] is None else report['predicted_first_image_divergence']
        n['first_image_divergence']=-1 if report['image_trajectory']['first_value_divergence'] is None else report['image_trajectory']['first_value_divergence']
        context=dict(model='flux2-klein-4b-distilled',source_job=summary['job_id'],case=key,horizon=1,
            numerical=dict(dtype='bfloat16',device=profile['device'],bf16_reduced_reduction=profile['bf16_reduced_reduction']),clock=dict(step=3,phase='joint',block=0))
        item=catalog.record(law_id='law.flux.reader.native-state-and-consumer-clock.v1',route_id='route.flux.image-KV.text-projection-residual-mlp.v1',alias='cmp.flux.reader-future.'+key,
            binding=dict(model_identity=context['model'],input_identity=plan['parents'],source_identity=native_req['request_sha256'],execution=context['numerical'],frontier=context['clock'],horizon=1),
            report=compare_equations(p,n,max_elements=20_000_000),conditions=[dict(field=f,op='eq',value=v) for f,v in context.items()],tested_context=context,
            readers=['joint0.text-head','joint0.text-mlp','next-image-consumer'],evidence=dict(audit_sha256=sha(out/'audit.json'),seal_sha256=result['seal_sha256'],
                role='sealed local reader-state and first-image-divergence predictions; final image observed',observed_continuation_steps=27,new_dose=key in fresh,terminal_status='unassessed'))
        wrong={**context,'horizon':2}
        if len(catalog.find(applicable_to=context))!=1 or catalog.find(applicable_to=wrong):raise ValueError('catalog applicability differs')
        if key=='right.dose375':save(out/'catalog/context-dose.json',context);save(out/'catalog/context-wrong-horizon.json',wrong)
        ids.append(item['fingerprint'])
    save(out/'catalog/provenance.json',dict(records=len(ids),fingerprints=ids,matching_queries=len(ids),wrong_horizon_abstentions=len(ids)))
    for name in ('native-contract.json','summary.json','cases.json','predictions.json','profiles.json'):shutil.copyfile(root/name,out/('source-'+name))
    for key,case in cases.items():shutil.copyfile(root/case['image']['path'],out/case['image']['path'])
    shutil.copytree(root/'transaction',out/'transaction')
    for name in ('parameter-artifacts.json','retrieve_weights.py'):shutil.copyfile(root/name,out/name)
    save(out/'external-evidence.json',dict(host='beast',roots=dict(native=str(root),reader=str(source)),
        manifests=dict(native=read(root/'files.json'),reader=read(source/'files.json')),scope='Full tensors remain remote. PNGs, full metadata/transaction and compact numerical reports retained here.'))
    shutil.copyfile('retrieve_reader_evidence.py',out/'retrieve_evidence.py')
    save(out/'plan.json',c);save(out/'request.json',req)
    save(out/'summary.json',dict(schema='mdb-reader-route-future-compact-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],
        output_root=str(out),source_job=summary['job_id'],seconds=time.monotonic()-started,records=len(ids),mechanics_status='verified',terminal_status='unassessed'))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--study',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        here=Path(__file__).resolve().parent;helper=runpy.run_path(str(here/'flux_reader_attention_audit.py'))
        print(json.dumps(helper['submit'](a.run.resolve()) if a.submit else helper['stage'](a.run.resolve(),a.study.resolve(),worker_path=Path(__file__),
            extra_files=[(a.study.resolve()/'payload/flux_reader_route_future.py','frozen-future-program.py'),(here/'flux_reentry_reference.py','flux_reentry_reference.py')],audit_kind='route-future',logical_operations=49)))
