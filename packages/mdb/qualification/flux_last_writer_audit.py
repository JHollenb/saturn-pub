"""Saved-only last writer forecasts, operand footprints and native image effects."""
import argparse
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def descriptor_item(mapping,name):
    if mapping.get('kind')!='mapping':raise ValueError('expected tagged Saturn mapping descriptor')
    key=dict(kind='scalar',type='str',value=name)
    found=[x['value'] for x in mapping['items'] if x['key']==key]
    if len(found)!=1:raise ValueError('missing/duplicate declared mapping item: '+name)
    return found[0]


def worker():
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='2'
    import numpy as np
    from PIL import Image
    from mdb.equation_comparison import EquationCatalog,compare_equations
    from mdb.forecast_transaction import load_forecast_tree,verify_transaction
    h=runpy.run_path('flux_source_receiver.py');read,save,sha,load=[h[k] for k in ('read','save','sha','load_array')]
    helper=runpy.run_path('flux_commit_factorial_audit.py');same,factorial,commit=[helper[k] for k in ('same_record','factorial','finite_commit_numpy')]
    measure=runpy.run_path('analyze_flux_reader_arithmetic.py')['measure'];bf16=runpy.run_path('flux_reentry_reference.py')['bf16']
    masks=runpy.run_path('flux_reader_fresh.py')['semantic_masks'];region=runpy.run_path('flux_reader_fresh_audit.py')['region_effect']
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=req['config'];root,source=Path(c['native_root']),Path(c['source_root']);started=time.monotonic()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('audit isolation/source differs')
    if sha(root/'files.json')!=c['native_manifest_sha256'] or sha(source/'files.json')!=c['source_manifest_sha256']:raise ValueError('input inventories differ')
    if any(sha(root/n)!=v for n,v in read(root/'files.json').items()):raise ValueError('native evidence changed')
    plan,summary,native_req=[read(root/n) for n in ('plan.json','summary.json','request.json')]
    if summary['counts']!=plan['planned_counts'] or native_req['config']!=plan or native_req['request_sha256']!=summary['request_sha256']:raise ValueError('native census/request differs')
    if any(sha(source/n)!=v['sha256'] for n,v in plan['inputs'].items()):raise ValueError('source inputs changed')
    cases,predictions,sources=[read(root/n) for n in ('cases.json','predictions.json','sources.json')];tx=read(root/'transaction-report.json')
    expected={f'{r}.r{ri}u{ui}' for r in plan['parents'] for ri,ui in ((0,0),(0,1),(1,0),(1,1))}
    if set(cases)!=set(predictions) or set(cases)!=expected or set(sources)!=set(plan['parents']):raise ValueError('declared case/source keys differ')
    verify_transaction(root/'transaction',expected_seal_sha256=tx['seal_sha256'])
    timeline=[json.loads(x) for x in (root/'timeline.jsonl').read_text().splitlines()];seals=[i for i,x in enumerate(timeline) if x['kind']=='all-predictions-sealed'];futures=[i for i,x in enumerate(timeline) if x['kind']=='native-future-complete']
    if len(seals)!=1 or len(futures)!=16 or min(futures)<=seals[0] or len(cases)!=len(predictions) or len(cases)!=16:raise ValueError('prediction chronology differs')
    at_seal=timeline[seals[0]]['counts']
    if any(at_seal[k]!=v for k,v in plan['expected_prefix_counts'].items()) or at_seal['typed_acts'] or at_seal['write_transplants'] or at_seal['images']:raise ValueError('pretarget work differs')
    control=Path(plan['historical_control_root'])
    if sha(control/'files.json')!=plan['historical_control_manifest_sha256']:raise ValueError('historical control inventory changed')
    control_inventory=read(control/'files.json')
    for n in ('cases.json','source-operands.json'):
        if sha(control/n)!=control_inventory[n]:raise ValueError('historical control metadata changed')
    old=read(control/'cases.json');schedules=read(control/'source-operands.json')
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    reports={};ports={};controls={};semantics={};effects={};ordering={};checkpoints={};count_exact=0;positions=0;ids=[]
    profile=read(root/'native-contract.json');catalog=EquationCatalog(out/'catalog/equations')
    for key,row in cases.items():
        p=predictions[key];s=row['spec'];r=s['receiver']
        frozen=load_forecast_tree(root/'transaction',read(root/'transaction/forecasts'/(key+'.json'))['predictions'])
        if not same(p,frozen):raise ValueError('forecast differs from seal')
        if not same(s,p['spec']):raise ValueError('target labels differ from sealed spec')
        actual={f:load(root,row[f]) for f in ('Y','N2','Z3','image','values')};predicted={f:load(root,p[f]) for f in actual}
        reports[key]={f:dict(**measure(actual[f],predicted[f]),archive_bytes_exact=row[f]['sha256']==p[f]['sha256']) for f in actual}
        count_exact+=sum(reports[key][f]['archive_bytes_exact'] for f in actual);positions+=sum(x.size for x in actual.values())
        R,U,gate=[load(root,x) for x in (row['R'],row['installed_U'],row['gate'])];cpuY=bf16(R+bf16(gate*U))
        reports[key]['saved_numpy_Y']=dict(**measure(actual['Y'],cpuY),raw_bits_exact=bool(np.array_equal(actual['Y'].view(np.uint32),cpuY.view(np.uint32))))
        if row['closure']['sigmas']!=schedules[r]['sigmas_descriptor']:raise ValueError('source schedule differs from bound historical reference')
        sig=np.asarray(schedules[r]['sigmas'],np.float32);cpuZ,product=commit(load(root,sources[r]['Z2']),actual['N2'],sig[3]-sig[2],bf16)
        reports[key]['saved_numpy_Z']=dict(**measure(actual['Z3'],cpuZ),raw_bits_exact=bool(np.array_equal(actual['Z3'].view(np.uint32),cpuZ.view(np.uint32))))
        delta=bf16(np.asarray([sig[3]-sig[2]],np.float32))[0]
        reports[key]['matched_predicted_N2_product']=measure(load(root,p['product']),bf16(delta*predicted['N2']))
        expected_R=load(root,sources[s['residual_source']]['R']);expected_U=load(root,sources[s['write_source']]['U'])
        ports[key]=dict(incoming_R=measure(expected_R,R),installed_U=measure(expected_U,U),
            computed_U_from_R=measure(load(root,sources[s['residual_source']]['U']),load(root,row['computed_U'])),
            own_gate=measure(load(root,sources[r]['gate']),gate),step2_temb_exact=descriptor_item(row['closure']['context'],'temb')==sources[r]['step2_temb'],
            step3_modulation_exact=row['step3_modulation']==sources[r]['step3_modulation'])
        # Every retained target checkpoint is an actual native Session capture.
        checkpoints[key]={}
        if set(row['checkpoints'])!={'readout','scheduler','project'}:raise ValueError('actual checkpoint set incomplete')
        for phase,fp in row['checkpoints'].items():
            meta=read(root/'store/cuts'/(fp+'.json'))
            expected_phase=phase;expected_step=3 if phase=='project' else 2
            point=meta['execution_point']
            if point['logical_step']!=expected_step or point['phase']!=expected_phase:raise ValueError('saved actual checkpoint clock differs')
            local={'readout':0,'scheduler':1,'project':2}[phase];fields=next(x['fields'] for x in row['trajectory'] if x['local']==local)
            if meta['fingerprint']!=fp or meta['manifest']['payload']!=fields:raise ValueError('checkpoint payload/trajectory differs')
            checkpoints[key][phase]=dict(fingerprint=fp,point=point)
        context=dict(model='flux2-klein-4b-distilled',source_job=summary['job_id'],case=key,clock='step2.single19->step3.joint0.V',horizon=5,execution=profile)
        record=catalog.record(law_id='law.flux.single19.residual-gated-write.readout-commit-value.native.v1',route_id='route.flux.single19.readout.commit.imageV.v1',alias='cmp.flux.last-writer.'+key,
            binding=dict(model_identity=context['model'],input_identity={r:v['entry'] for r,v in sources.items()},source_identity=native_req['request_sha256'],execution=profile,frontier=dict(step=2,phase='single',cursor=19),horizon=5),
            report=compare_equations(predicted,actual,max_elements=20_000_000),conditions=[dict(field=f,op='eq',value=v) for f,v in context.items()],tested_context=context,
            readers=['single19.exit','step2.readout','step2.commit','step3.imageV','native-full-image'],
            evidence=dict(seal_sha256=tx['seal_sha256'],role='Sealed conditional Y/N/Z/I/V prediction, RGB native observation',new_fitted_parameters=0,terminal_status='unassessed'))
        ids.append(record['fingerprint'])
        if len(catalog.find(applicable_to=context))!=1 or catalog.find(applicable_to={**context,'horizon':6}):raise ValueError('supplied catalog applicability differs')
    for key,oldkey in plan['historical_controls'].items():
        o=old[oldkey]
        if sha(control/o['png']['path'])!=o['png']['sha256']:raise ValueError('historical PNG changed')
        load(control,o['latent']);load(control,o['component']['latent'])
        controls[key]=dict(png_exact=cases[key]['png']['sha256']==o['png']['sha256'],latent_exact=cases[key]['latent']['sha256']==o['latent']['sha256'],Z3_exact=cases[key]['Z3']['sha256']==o['component']['latent']['sha256'])
    def rgb(k):return np.asarray(Image.open(root/cases[k]['png']['path']).convert('RGB'))
    for r in sources:
        keys={a+b:f'{r}.r{a}u{b}' for a,b in (('0','0'),('0','1'),('1','0'),('1','1'))};base=rgb(keys['00']);regions=masks(base)
        semantics[r]={corner:{name:region(base,rgb(k),mask) for name,mask in regions.items()} for corner,k in keys.items()}
        fields={'rgb':lambda k:rgb(k),'final_latent':lambda k:load(root,cases[k]['latent']),
            **{f:(lambda k,f=f:load(root,cases[k][f])) for f in ('Y','N2','Z3','values')}}
        effects[r]={f:factorial({corner:get(k) for corner,k in keys.items()}) for f,get in fields.items()}
        for report in effects[r].values():report['labels']=dict(carry='incoming R19',flow='raw new U19')
        rp,up=[effects[r]['rgb'][f] for f in ('carry_projection','flow_projection')];rg,ug=[semantics[r][k]['green']['signed_green_score'] for k in ('10','01')]
        ordering[r]=dict(residual_rgb_projection=rp,write_rgb_projection=up,residual_green=rg,write_green=ug,
            predicted_residual_rgb_larger=None if rp is None else rp>up,predicted_residual_green_larger=None if rg is None or ug is None else abs(rg)>abs(ug),
            predicted_negative_residual=None if rg is None else rg<0,predicted_negative_write=None if ug is None else ug<0,
            role='Frozen exploratory predictions; retain every miss and zero/empty-mask abstention.')
    result=dict(schema='mdb-flux-last-writer-audit-v1',source_job=summary['job_id'],mechanics_status='verified',instrument_status='complete-saved-port-and-image-comparisons',trend_status='observed',terminal_status='unassessed',
        counts=summary['counts'],scope=plan['scope'],native_manifest_sha256=sha(root/'files.json'),seal_sha256=tx['seal_sha256'],forecast_archives=80,exact_forecast_archives=count_exact,
        unique_forecast_archives=len({p[f]['sha256'] for p in predictions.values() for f in ('Y','N2','Z3','image','values')}),compared_scalar_positions=positions,
        forecasts=reports,ports=ports,historical_controls=controls,checkpoints=checkpoints,factorials=effects,semantic_contrasts=semantics,frozen_ordering=ordering,semantic_hypotheses=plan['semantic_hypotheses'],
        limitations=['One seed/four dependent recipients. Source/source reuses donorN2 under ownZ2 and should match prior flow-only, not both-source latent.',
            'U is raw fused attention/MLP projection, clamped independently of chosenR. No H/M attribution yet.',
            'Native primitives/original weights compute forecast readout/projection; independent NumPy checks cover only final addition/commit under the qualified scalar profile.',
            'Images are actual native consequences, not numerical image forecasts. Baseline color masks are instruments, not object labels.'])
    save(out/'audit.json',result);save(out/'catalog/provenance.json',dict(records=len(ids),fingerprints=ids,matching_queries=len(ids),wrong_horizon_abstentions=len(ids)))
    for row in cases.values():shutil.copyfile(root/row['png']['path'],out/row['png']['path'])
    for n in ('cases.json','predictions.json','sources.json','native-contract.json','summary.json','plan.json','timeline.jsonl'):shutil.copyfile(root/n,out/('source-'+n))
    shutil.copytree(root/'transaction',out/'transaction')
    for n in ('parameter-artifacts.json','retrieve_weights.py'):shutil.copyfile(root/n,out/n)
    save(out/'external-evidence.json',dict(host='beast',roots=dict(native=str(root),reader=str(source),controls=str(control)),manifests=dict(native=read(root/'files.json'),reader=read(source/'files.json'),controls=control_inventory)))
    shutil.copyfile('retrieve_reader_evidence.py',out/'retrieve_evidence.py');save(out/'plan.json',c);save(out/'request.json',req)
    save(out/'summary.json',dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),source_job=summary['job_id'],seconds=time.monotonic()-started,
        records=len(ids),mechanics_status='verified',new_execution_counts=dict(model_calls=0,native_calls=0,learned_forwards=0,saved_elementwise_writes=16,saved_elementwise_commits=16,saved_elementwise_products=16,images=0)))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        here=Path(__file__).resolve().parent;helper=runpy.run_path(str(here/'flux_reader_attention_audit.py'));p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('run',type=Path);p.add_argument('--study',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        extras=[(here/n,n) for n in ('flux_commit_factorial_audit.py','flux_reentry_reference.py','flux_reader_fresh.py','flux_reader_fresh_audit.py')]
        print(json.dumps(helper['submit'](a.run.resolve()) if a.submit else helper['stage'](a.run.resolve(),a.study.resolve(),worker_path=Path(__file__),extra_files=extras,audit_kind='last-writer',logical_operations=128)))
