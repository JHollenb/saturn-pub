"""Stage/admit the complete image-pair × text × latent suffix experiment."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,source):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    here=Path(__file__).resolve().parent;u=runpy.run_path(str(here/'flux_predicted_future_campaign.py'));save,sha=u['save'],u['sha'];read=lambda p:json.loads(p.read_text());data=source/'run-01/extracted/receiver-cross-result'
    old=read(data/'plan.json');summary=read(data/'summary.json');binding=read(data/'binding.json');audit=read(source/'analysis-v1.json');rows={r['id']:r for r in read(data/'rows.json')}
    if audit['mechanics_status']!='verified':raise ValueError('Verified source required')
    parents={s:b['after'] for s,b in binding.items()};pairs={s:{'0':parents[s],'1':rows[s+'.from60667.+1']['actions'][-1]['post_act']} for s in parents};manifest=read(data/'files.json');names=set()
    for fp in set(parents.values())|{fp for v in pairs.values() for fp in v.values()}:
        name='store/cuts/'+fp+'.json';names.add(name);cut=read(data/name);names.update('store/pages/'+h+'.json' for h in cut['slots'].values());names.update('store/blobs/'+h+'.safetensors' for h in cut['physical_blobs'])
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError('Source CAS differs')
        files[name]={'bytes':(data/name).stat().st_size,'sha256':manifest[name]}
    counts=runpy.run_path(str(here/'flux_suffix_context.py'))['census']()
    config={'schema':'mdb-suffix-context-plan-v1',**{k:old[k] for k in ('model','model_path','checkpoint_sha256','prompts','observers')},'input_root':summary['output_root']+'/receiver-cross-result','input_files':files,'parents':parents,'pairs':pairs,'source_audit_sha256':sha(source/'analysis-v1.json'),'fixed_payload_source':'60667 attention correction, +1, observed prior native-cached image exits',
        'historical_pngs':{s:{'0':rows[s+'.sham']['png_sha256'],'1':rows[s+'.from60667.+1']['png_sha256']} for s in parents},'expected_counts':counts,'output_scratch':'/mnt/hdd8tb/mdb/research','max_wall_seconds':600,
        'hypotheses':{
            'mechanical_predictions':{'parallel_M':'For fixed complete input image and context, image M19 and gate are bit-identical across text sources. Image attention may change.','latent_independence':'For fixed I/T/context, changing only carried latent leaves post19 I/T and readout noise bit-identical. Latent first enters this tail at scheduler commit.','euler':'Sixteen full Z4 forecasts from observed scheduler inputs seal individually before their own native commit/RGB. Prior BF16 reference predicts exact native Z4; retain numerical discrepancies as observations.'},
            'competing_signed_predictions':{'image_pair_led':'On fixed60667eye RGB axis, pair60666 gives positive corrected-minus-base effect and pair60667 negative for all T/Z contexts.','text_led':'On fixed60667eye RGB axis, T60666 gives positive corrected-minus-base effect, T60667 negative, for both image pairs and either latent.','latent_led':'On fixed60667eye RGB axis, Z60666 gives positive corrected-minus-base effect, Z60667 negative, for both image pairs and either text.'},
            'scoring':'8paired contrasts,2previouslyseen coherent T=Z=pair controls,6new hybrid contrasts. Predictions competing and exploratory, no sign gate. Also retain both fixed saved RGB axes and target/collateral.','controls':'4coherent PNGs exactly match prior shams/corrected native-cached futures. These are replay controls, not new predictive successes.',
            'scope':'Complete I0/I1 pair identical across T/Z, no new A composition or rounding at installation. Select native parent by latent; legal I/T Acts only. Non-I/T/Z closure identical must hold. All generic hypotheses frozen before any tail/capture. Local Euler observations include current native readout, not a complete upstream forecast. No source-free image prediction or semantic geometry law.'},
        'scope':'2fixedcompleteimagepairs x2imagechoices x2nativeT x2nativeZ=16nativeRGB48trans32I/TActs.8directnativeportcaptures separate; fullpre19/post19/scheduler/finalcuts.16localEulerforecasts after observedreadout before owncommit.1load0encodes/noise.'}
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config);worker=here/'flux_suffix_context_campaign.py';names=['flux_suffix_context.py','flux_single_contraction.py','flux_scene_dependencies.py','flux_predicted_future_campaign.py','flux_family_custody.py'];extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json')];pre=run/'preflight.v1.json'
    save(pre,{'schema_version':'mrun-preflight-config-v1','hashed_roots':[worker.name,*names,'plan.json',pre.name],'path_token_allowlist':['suffix-context-result','input-data',config['model_path'],config['input_root'],config['output_scratch']]});extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+'\n').encode();job=read(source/'run-01/job.json')['result']
    request={'schema':'mdb-suffix-context-request-v1','config':config,'model':config['model'],'task_family':'forward','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'height':512,'width':512,'needs':{'cuda':True,'host':'beast'},'execution_environment':intent(),'config_bytes_sha256':hashlib.sha256(raw).hexdigest(),'source_hashes':{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        'scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_operations=16,logical_branches=16,logical_checkpoint_replays=16,max_local_work_items=48,checkpoint_policy='16nativefutures32legalI/TActs48transitions,8directportcaptures separate. Shared completeI pair across T/Z.16localEulerforecasts firstbeforeeachcommit. All serial inprocess,noretries.'),
        'resource_exception':{'ram_mb':10000,'vram_mb':13000,'disk_gb':2,'calibration_job':summary['job_id'],'measured_peak_rss_mb':job['peak_rss_mb'],'measured_peak_vram_mb':job['peak_vram_mb'],'reason':'Same noencoder resident BF16Klein4B512pxB1. No fullW or repeatedmixedconcat forecasts. Native sparseCAScuts and8imageA/M/P captures. Payload/cache/output/log on HDD;2GBroot-admission reservation covers predicted sub-GB transport plus minimal payload, remainingroot3.386GB. Prior10GB/13GB qualified bound,serial.'}}
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras);return {'run_dir':str(run),'request_sha256':request['request_sha256'],'counts':counts}


def submit(run):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    u=runpy.run_path(str(Path(__file__).with_name('flux_predicted_future_campaign.py')));r=json.loads((run/'request.json').read_text())
    if r['execution_environment']!=intent():raise ValueError('Restage environment before admission')
    if hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='request_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()!=r['request_sha256']:raise ValueError('Request seal differs')
    if (run/'submission.json').exists() or any(u['sha'](run/'payload'/n)!=h for n,h in r['source_hashes'].items()):raise ValueError('Admitted or mutated stage')
    cmd=worker_command('flux_suffix_context_campaign.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-suffix-context',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],ram_mb=10000,vram_mb=13000,disk_gb=2,cpu_threads=4,est_wall_s=240,timeout_s=720,retry_on_kill=False,preflight=True,
        note='IdenticalcompleteI0/I1pairs crossedwithnativeT/Z atpre19.16RGB48trans32legalI/TActs; selectZnativeparent, no lateZwrite/adapterchange.8portcaptures testparallelMtextindependence; Zfirstreadatcommit. Freeze3competingdirectionrules+mechanicalpredictions before tail.16localEulerforecasts fromobservednoise beforeowncommit.1load0encodes/noise,serial,10GBRAM13GBVRAM600sworker720slease,no retries,HDDscratch;compactportarrays/CAScuts.')
    u['save'](run/'submission.json',{'job_id':str(job)});return {'job_id':str(job),'run_dir':str(run)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--source',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--collect');a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')))['collect'](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve())
    print(json.dumps(r,sort_keys=True))
