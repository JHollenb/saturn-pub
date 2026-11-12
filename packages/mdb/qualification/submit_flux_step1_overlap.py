"""Stage a step1/single6 RAMT cube with a common shape continuation."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,source,onset):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    here=Path(__file__).resolve().parent;u=runpy.run_path(str(here/'flux_predicted_future_campaign.py'));save,sha=u['save'],u['sha'];read=lambda p:json.loads(p.read_text())
    data=source/'run-01/extracted/step1-writers-result';old=read(data/'plan.json');summary=read(data/'summary.json')
    if read(source/'analysis-v1.json')['mechanics_status']!='verified' or read(onset/'analysis-v1.json')['mechanics_status']!='verified':raise ValueError('Verified predecessors required')
    if old['source_audit_sha256']!=sha(onset/'analysis-v1.json'):raise ValueError('Predecessor lineage differs')
    cuts=read(data/'cuts.json');roots={k:cuts['step1'][k]['after_joint4'] for k in ('base','shape')}
    names=set();manifest=read(data/'files.json')
    for fp in roots.values():
        name='store/cuts/'+fp+'.json';names.add(name);cut=read(data/name)
        if cut['execution_point']['logical_step']!=1 or cut['execution_point']['phase']!='single' or cut['execution_point']['index']!=0:raise ValueError('Step1/single0 source required')
        names.update('store/pages/'+h+'.json' for h in cut['slots'].values());names.update('store/blobs/'+h+'.safetensors' for h in cut['physical_blobs'])
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError('Source CAS differs')
        files[name]={'bytes':(data/name).stat().st_size,'sha256':manifest[name]}
    historical={r['id']:r for r in read(onset/'run-01/extracted/edit-onset-result/rows.json')}
    config={'schema':'mdb-step1-overlap-plan-v1',**{k:old[k] for k in ('model','model_path','checkpoint_sha256','seed','prompts','observers')},
        'input_root':summary['output_root']+'/step1-writers-result','input_files':files,'input_cuts':roots,'source_audit_sha256':sha(source/'analysis-v1.json'),'onset_audit_sha256':sha(onset/'analysis-v1.json'),
        'historical_pngs':{'weak':historical['onset2.shape']['png_sha256'],'strong':historical['onset1.shape']['png_sha256']},
        'hypotheses':{'controls':'R0A0M0/Tbase should reproduce onset2.shape; R1A1M1/Tshape should reproduce onset1.shape. Every branch receives shape E beforestep2. These known sourcecontrols test lineage, not promptgeneralization.',
            'factorial':'At step1/single6, independently vary image residual R, observed raw attention A, observed gated MLP M, and postblock working text T. Map all conditional signed effects; allow overlap, redundancy, opposite signs and interactions. No axis is preassigned an eye-only semantic role.',
            'numeric':'Four fullP predictions B([A;M]W^T+b), then eight fullI predictions B(R+B(gP)), sealed beforecachedP/mixedblock/RGB validation. Source00/11P previouslyobserved; mixed10/01P not yet executed. AllA/Mobserved. Nativecached exits should matchfrozen fullblocks exactly; CPU numericerror diagnostic.',
            'future':'Two independently calculated image-exit corners should yield RGB MAE below1 versus nativecached corners, calibrated byprior same-lane errors0.12-0.48. This is a scoredforecast, not discoverygate; record errors and trajectories.',
            'semantics':'Compare narrower-eyelid appearance and finecollateral visually. EyeRGBcontrast projection between sourcecontrols is not a calibratedshape score. No semanticneuron/minimality or completeimageprediction claim.'},
        'expected_counts':runpy.run_path(str(here/'flux_step1_overlap.py'))['census'](),'output_scratch':'/home/beast/.cache/mdb-research','max_wall_seconds':600,
        'scope':'Known60666portrait step1/single6.16nativecached RAMT futures+2CPU-predicted corners, commonshapeE fromstep2.1292native transitions36Acts1328receipts;8counted directblocks,4cachedlinearprojections.1load0encodes/noise. ObservedR/A/M andweights, no upstreamformation orrawpromptRGBforecast.'}
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config)
    worker=here/'flux_step1_overlap_campaign.py';names=['flux_step1_overlap.py','flux_step1_writers.py','flux_conditioning_program.py','flux_single_contraction.py','flux_scene_dependencies.py','flux_predicted_future_campaign.py','flux_family_custody.py']
    extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json')];pre=run/'preflight.v1.json'
    save(pre,{'schema_version':'mrun-preflight-config-v1','hashed_roots':[worker.name,*names,'plan.json',pre.name],'path_token_allowlist':['step1-overlap-result','input-data',config['model_path'],config['input_root'],config['output_scratch']]});extras.append((pre,pre.name))
    raw=(json.dumps(config,sort_keys=True)+'\n').encode()
    request={'schema':'mdb-step1-overlap-request-v1','config':config,'model':config['model'],'task_family':'forward','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'height':512,'width':512,
        'needs':{'cuda':True,'host':'beast'},'execution_environment':intent(),'config_bytes_sha256':hashlib.sha256(raw).hexdigest(),'source_hashes':{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        'scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_operations=30,logical_branches=18,logical_checkpoint_replays=18,max_local_work_items=40,checkpoint_policy='2prefix7+18futures71=1292native,36Acts1328receipts18VAE;8directblocks and4nativecachedP separatelycounted. AllfutureEshape; no retries.'),
        'resource_exception':{'ram_mb':10000,'vram_mb':13000,'disk_gb':6,'calibration_job':'job-2e09f217e4a4','measured_peak_rss_mb':5405.8,'measured_peak_vram_mb':9644,'reason':'Same noencoder residentKlein4B512pxBF16B1. FourP andeightI fields plusoutputweight add<1GB relativetosource lane. Serial18futures, same10GBRAM13GBVRAM. No runtimebenchmark claim.'}}
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request)
    stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return {'run_dir':str(run),'request_sha256':request['request_sha256'],'submitted':False}


def submit(run):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    u=runpy.run_path(str(Path(__file__).with_name('flux_predicted_future_campaign.py')));r=json.loads((run/'request.json').read_text())
    if r['execution_environment']!=intent():raise ValueError('Restage environment before admission')
    if hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='request_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()!=r['request_sha256']:raise ValueError('Request seal differs')
    if (run/'submission.json').exists() or any(u['sha'](run/'payload'/n)!=h for n,h in r['source_hashes'].items()):raise ValueError('Admitted or mutated stage')
    cmd=worker_command('flux_step1_overlap_campaign.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-step1-overlap',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],ram_mb=10000,vram_mb=13000,disk_gb=6,cpu_threads=4,est_wall_s=300,timeout_s=720,retry_on_kill=False,preflight=True,
        note='Matched60666step1/single6 RAMT16cube+2CPUcornerfutures. CommonshapeE from2 toavoid baseErepairconfound.1292native36Acts1328receipts18RGB+8directblocks4cachedP. Hypotheses sealbeforeprefix;4P8I numericsealbeforemixed/cached/RGB.1load0encodes/noise. Qualified10GBRAM13GBVRAM600sworker720slease,noretries. Retainallconditional/collateral effects.')
    u['save'](run/'submission.json',{'job_id':str(job)});return {'job_id':str(job),'run_dir':str(run)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--source',type=Path);p.add_argument('--onset-source',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--collect');a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')))['collect'](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve(),a.onset_source.resolve())
    print(json.dumps(r,sort_keys=True))
