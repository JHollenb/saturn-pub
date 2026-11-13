"""Admit source-assisted attention/MLP/text interaction tests and frozen math."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,source):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    here=Path(__file__).resolve().parent;u=runpy.run_path(str(here/'flux_predicted_future_campaign.py'));save,sha=u['save'],u['sha'];read=lambda p:json.loads(p.read_text());data=source/'run-02/extracted/interaction-trace-result'
    old=read(data/'plan.json');summary=read(data/'summary.json');audit=read(source/'analysis-v1.json');selected=read(data/'selected.json')
    if audit['mechanics_status']!='verified':raise ValueError('Verified trace required')
    manifest=read(data/'files.json');names=set()
    for s in selected.values():
        for fp in (*s['before'].values(),*s['after'].values()):
            name='store/cuts/'+fp+'.json';names.add(name);cut=read(data/name);names.update('store/pages/'+h+'.json' for h in cut['slots'].values());names.update('store/blobs/'+h+'.safetensors' for h in cut['physical_blobs'])
        for p in s['ports'].values():names.update(p[k]['path'] for k in ('concat','projection','gate'))
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError('Source CAS differs')
        files[name]={'bytes':(data/name).stat().st_size,'sha256':manifest[name]}
    H=runpy.run_path(str(here/'flux_interaction_writes.py'));counts=H['census']({k:v['point'] for k,v in selected.items()})
    rows={r['id']:r for r in read(data/'rows.json')}
    config={'schema':'mdb-interaction-writes-plan-v1',**{k:old[k] for k in ('model','model_path','checkpoint_sha256','prompts','observers')},'input_root':summary['output_root']+'/interaction-trace-result','input_files':files,'selected':selected,'source_audit_sha256':sha(source/'analysis-v1.json'),
        'historical_pngs':{seed:rows[seed+'.11']['png_sha256'] for seed in selected},'expected_counts':counts,'output_scratch':'/mnt/hdd8tb/mdb/research','max_wall_seconds':600,
        'hypotheses':{'controls':'Bothsameimage/textshams should reproducehistoricalstrongPNGexactly; nativecached5cases/seed should match source/mixedfullblockimageexits exactly. Preservecontrols separately.',
            'math':'8fullmixedP+8fullI CPUforecasts sealedbeforeanynewnativecachedP/fullmixedblock/RGBfuture. SourceA/M/R/T/gates actualnativeobservations. CPUpredictedAMfuture RGBMAE<1versusnativeAMlinear, calibratedprior0.249/.229, notadiscoverygate; maxerrors retained.',
            'opposite_attention':'Tentative signedconsumerprediction: Alinear andAopposite should shifteyeRGBcontrast inoppositedirections relative tosham. Thisextrapolateslocaldirection, mayfail withnonlinearity/rounding; retainzeroeffects asunmatched.',
            'continuing_text':'Tentativeprediction fromearlierprogramdominance: absoluteeyeRGBcontrastshift fromTlinear exceedsAMlinear. Selectedlateclock mayinstead showimagewriteauthority; retainthatcounterexample.',
            'scope':'LinearA/M/T candidates arecalculatedfromobservedfourcornerstates, notindependentformationforecasts. AllsourcejointI/R/latent/externalcontextretained ineachcandidateexceptdeclaredI/Tslots. Noisolatedsemanticwriter/minimalityclaim; adaptiveamplituderanking isjustacandidate.'},
        'scope':'Two knownseed selectedsingle nodes.8futures/seed:sham,Alinear,Mlinear,AMlinear,Tlinear,AMTlinear,Aopposite,predictedAM. Source-assisted4cornerlinearizedoperands+actualweights; fullnumericalforecastsbeforemixedexecution.10cachedprojections8fullblockvalidationsseparatelycounted,16nativeRGB32Acts. No newprompts/encodes/noise.'}
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config);worker=here/'flux_interaction_writes_campaign.py';names=['flux_interaction_writes.py','flux_single_contraction.py','flux_conditioning_program.py','flux_step1_overlap.py','flux_scene_dependencies.py','flux_predicted_future_campaign.py','flux_family_custody.py'];extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json')];pre=run/'preflight.v1.json'
    save(pre,{'schema_version':'mrun-preflight-config-v1','hashed_roots':[worker.name,*names,'plan.json',pre.name],'path_token_allowlist':['interaction-writes-result','input-data',config['model_path'],config['input_root'],config['output_scratch']]});extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+'\n').encode()
    job=read(source/'run-02/job.json')['result']
    request={'schema':'mdb-interaction-writes-request-v1','config':config,'model':config['model'],'task_family':'forward','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'height':512,'width':512,'needs':{'cuda':True,'host':'beast'},'execution_environment':intent(),'config_bytes_sha256':hashlib.sha256(raw).hexdigest(),'source_hashes':{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        'scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_operations=16,logical_branches=16,logical_checkpoint_replays=16,max_local_work_items=48,checkpoint_policy='16nativefutures8perseed32I/TActs;8fullmixedblockvalidations10cachedprojections separate.8P/8I numericalforecasts first. Allbranchesinprocess,noretries.'),
        'resource_exception':{'ram_mb':10000,'vram_mb':13000,'disk_gb':4,'calibration_job':summary['job_id'],'measured_peak_rss_mb':job['peak_rss_mb'],'measured_peak_vram_mb':job['peak_vram_mb'],'reason':'Same resident noencoderKlein4B512pxBF16B1, serialfutures, bounded8observedconcat fields+8predictedI/P. Fullmatrixforecasts tiledonCPU;10GBRAM13GBVRAM priorqualifiedbound. HDDscratch; no speedupclaim.'}}
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras);return {'run_dir':str(run),'request_sha256':request['request_sha256'],'counts':counts}


def submit(run):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    u=runpy.run_path(str(Path(__file__).with_name('flux_predicted_future_campaign.py')));r=json.loads((run/'request.json').read_text())
    if r['execution_environment']!=intent():raise ValueError('Restage environment before admission')
    if hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='request_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()!=r['request_sha256']:raise ValueError('Request seal differs')
    if (run/'submission.json').exists() or any(u['sha'](run/'payload'/n)!=h for n,h in r['source_hashes'].items()):raise ValueError('Admitted or mutated stage')
    cmd=worker_command('flux_interaction_writes_campaign.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-interaction-writes',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],ram_mb=10000,vram_mb=13000,disk_gb=4,cpu_threads=4,est_wall_s=300,timeout_s=720,retry_on_kill=False,preflight=True,
        note='SelectedI/Tinteractionnodes2knownseeds.8futures/seed16RGB32Acts,10cachedP8fullmixedblockvalidations. Freeze8fullP/8fullI beforemixednativeexecution; sourceA/M/R/Tobserved. TestA/M/Tlinearization, oppositeA,AMTjoint andpredictedAMnativefuture. Sameparent11context/latent retained.1load0encodes/noise.10GBRAM13GBVRAM600sworker720slease,noretries,HDDscratch.')
    u['save'](run/'submission.json',{'job_id':str(job)});return {'job_id':str(job),'run_dir':str(run)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--source',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--collect');a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')))['collect'](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve())
    print(json.dumps(r,sort_keys=True))
