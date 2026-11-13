"""Stage/admit eight saved native suffixes and a bounded operator interaction trace."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,discovery,replication):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    here=Path(__file__).resolve().parent;u=runpy.run_path(str(here/'flux_predicted_future_campaign.py'));save,sha=u['save'],u['sha'];read=lambda p:json.loads(p.read_text());sources={}
    for seed,source in [('60666',discovery),('60667',replication)]:
        data=source/'run-01/extracted/shape-formation-result';old=read(data/'plan.json');summary=read(data/'summary.json')
        if read(source/'analysis-v1.json')['mechanics_status']!='verified':raise ValueError('Verified predecessor required')
        rows={r['id']:r for r in read(data/'rows.json')};keys={'00':'reference.base','10':'handoff6.I','01':'handoff6.T','11':'reference.shape'};cuts={b:rows[k]['trace'][0]['cut'] for b,k in keys.items()};names=set();manifest=read(data/'files.json')
        for fp in cuts.values():
            name='store/cuts/'+fp+'.json';names.add(name);cut=read(data/name)
            if cut['execution_point']['logical_step']!=2 or cut['execution_point']['phase']!='project':raise ValueError('Step2 project source required')
            names.update('store/pages/'+h+'.json' for h in cut['slots'].values());names.update('store/blobs/'+h+'.safetensors' for h in cut['physical_blobs'])
        files={}
        for name in sorted(names):
            if sha(data/name)!=manifest[name]:raise ValueError('Source CAS differs')
            files[name]={'bytes':(data/name).stat().st_size,'sha256':manifest[name]}
        sources[seed]={'input_root':summary['output_root']+'/shape-formation-result','input_files':files,'cuts':cuts,'pngs':{b:rows[k]['png_sha256'] for b,k in keys.items()},'audit_sha256':sha(source/'analysis-v1.json')}
    config={'schema':'mdb-interaction-trace-plan-v1',**{k:old[k] for k in ('model','model_path','checkpoint_sha256','prompts','observers')},'sources':sources,
        'expected_counts':runpy.run_path(str(here/'flux_interaction_trace.py'))['census'](),'output_scratch':'/mnt/hdd8tb/mdb/research','max_wall_seconds':600,
        'hypotheses':{'replay':'All8savedZ2native suffixes should reproducehistoricalPNGsexactly. Fourcorners shareEshape; inheritedI/Tfields atproject arestale, onlylatentlive.',
            'trace':'MeasureJ=X11-X10-X01+X00 andperoperatorchanges forlivefields. DistinguishinheritedJ fromcurrentwriteJ. Fulloperator metrics source-bound except64witnesses andsparsefullcuts; do notuseenergyassemanticgate.',
            'selection':'Choose largestimagewriteinteractionL2 amongsingleblocks0..18 inremainingsteps2/3 separatelyperseed. Exclude19becausepostreadoutimageActunsupported; report alloperators. Diagnosticselection, notsemanticnecessity.',
            'ports':'Atselectednode capture4nativefullconcatenatedA/M operands, projection/gate andnativeI/Treturns. Source-assistednativecaptures, notindependentformationforecasts. Currentlargestwrite maynotcontrolfinaleyes; nextcausaltestrequired.'},
        'scope':'Two knownseedI6xT6factorials, savedstep2parents.448native transitions8RGBreplays8separatecandidateblocks,1load0encodes/noise. Operatortrace with64witnesses/sparsefullcuts, oneadaptivewritable-singlepeakperseed. No causalportclaim fromenergyranking.'}
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config);worker=here/'flux_interaction_trace_campaign.py';names=['flux_interaction_trace.py','flux_single_contraction.py','flux_predicted_future_campaign.py','flux_family_custody.py'];extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json')];pre=run/'preflight.v1.json'
    save(pre,{'schema_version':'mrun-preflight-config-v1','hashed_roots':[worker.name,*names,'plan.json',pre.name],'path_token_allowlist':['interaction-trace-result','input-data',config['model_path'],config['output_scratch'],*[s['input_root'] for s in sources.values()]]});extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+'\n').encode()
    request={'schema':'mdb-interaction-trace-request-v1','config':config,'model':config['model'],'task_family':'forward','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'height':512,'width':512,
        'needs':{'cuda':True,'host':'beast'},'execution_environment':intent(),'config_bytes_sha256':hashlib.sha256(raw).hexdigest(),'source_hashes':{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        'scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_operations=8,logical_branches=8,logical_checkpoint_replays=8,max_local_work_items=24,checkpoint_policy='8*56savednative suffixsteps=448,8VAE,0Acts;4interleavedcontextsperseed,8separatecandidatecaptures. NonretainingSession.capture,sparsefullcuts/topnodefields. Alllocal,noretries.'),
        'resource_exception':{'ram_mb':10000,'vram_mb':13000,'disk_gb':4,'calibration_job':'job-db9e414ff4d6','measured_peak_rss_mb':7287.4,'measured_peak_vram_mb':9970,'reason':'Same noencoder512pxBF16B1 lane,4livecontexts+smallboundedtopnodecache, nohistoryretention/fullweightcopies. Serial8futures. OutputscratchmovedtoHDD toavoidrootdiskpressure.10GBRAM13GBVRAM basedonmeasuredsame-model runs.'}}
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras);return {'run_dir':str(run),'request_sha256':request['request_sha256']}


def submit(run):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    u=runpy.run_path(str(Path(__file__).with_name('flux_predicted_future_campaign.py')));r=json.loads((run/'request.json').read_text())
    if r['execution_environment']!=intent():raise ValueError('Restage environment before admission')
    if hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='request_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()!=r['request_sha256']:raise ValueError('Request seal differs')
    if (run/'submission.json').exists() or any(u['sha'](run/'payload'/n)!=h for n,h in r['source_hashes'].items()):raise ValueError('Admitted or mutated stage')
    cmd=worker_command('flux_interaction_trace_campaign.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-interaction-trace',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],ram_mb=10000,vram_mb=13000,disk_gb=4,cpu_threads=4,est_wall_s=300,timeout_s=720,retry_on_kill=False,preflight=True,
        note='Replay8savedZ2I6xT6 corners over2knownseeds.448native0Acts8RGB8directblockcaptures. Trace4contextslockstep withnonretainingpubliccaptures;64witnesses/sparsefullcuts. Adaptivepeakwritable-single selectedbyimagewriteinteractionL2, notsemanticgate; nativeA/M/Pports capturedfornextcausalcut.1load0encodes/noise.10GBRAM13GBVRAM600sworker720slease,noretries. HDDscratch avoidsrootdiskpressure.')
    u['save'](run/'submission.json',{'job_id':str(job)});return {'job_id':str(job),'run_dir':str(run)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--discovery',type=Path);p.add_argument('--replication',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--collect');a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')))['collect'](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.discovery.resolve(),a.replication.resolve())
    print(json.dumps(r,sort_keys=True))
