"""Stage/admit a fixed attention correction × receiving-state assay."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,source,writers):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    here=Path(__file__).resolve().parent;u=runpy.run_path(str(here/'flux_predicted_future_campaign.py'));save,sha=u['save'],u['sha'];read=lambda p:json.loads(p.read_text());data=source/'run-02/extracted/interaction-trace-result'
    old=read(data/'plan.json');summary=read(data/'summary.json');selected=read(data/'selected.json');wa=read(writers/'analysis-v1.json');wr=read(writers/'run-01/extracted/interaction-writes-result/rows.json')
    if read(source/'analysis-v1.json')['mechanics_status']!='verified' or wa['mechanics_status']!='verified':raise ValueError('Verified source required')
    if any(any(s['point'][k]!=v for k,v in {'logical_step':3,'phase':'single','index':18,'local_clock':18}.items()) for s in selected.values()):raise ValueError('Only qualified single18 clock')
    manifest=read(data/'files.json');names=set()
    for s in selected.values():
        for fp in (s['before']['11'],s['after']['11']):
            name='store/cuts/'+fp+'.json';names.add(name);cut=read(data/name);names.update('store/pages/'+h+'.json' for h in cut['slots'].values());names.update('store/blobs/'+h+'.safetensors' for h in cut['physical_blobs'])
        for p in s['ports'].values():names.update(p[k]['path'] for k in ('concat','gate'))
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError('Source CAS differs')
        files[name]={'bytes':(data/name).stat().st_size,'sha256':manifest[name]}
    counts=runpy.run_path(str(here/'flux_receiver_cross.py'))['census'](selected)
    config={'schema':'mdb-receiver-cross-plan-v1',**{k:old[k] for k in ('model','model_path','checkpoint_sha256','prompts','observers')},'input_root':summary['output_root']+'/interaction-trace-result','input_files':files,'selected':selected,'source_audit_sha256':sha(source/'analysis-v1.json'),'writer_audit_sha256':sha(writers/'analysis-v1.json'),
        'historical_pngs':{seed:{r['case']:r['png_sha256'] for r in wr if r['seed']==seed and r['case'] in ('sham','Alinear','Aopposite')} for seed in selected},'expected_counts':counts,'output_scratch':'/mnt/hdd8tb/mdb/research','max_wall_seconds':600,
        'hypotheses':{
            'competing_directional_models':{'payload_led':'For every receiver, positive delta60666 raises eye RGB contrast projection; positive delta60667 lowers it. Negative doses reverse those signs.','receiver_led':'For either payload, positive doses raise eye RGB contrast projection in receiver60666 and lower it in receiver60667. Negative doses reverse those signs.'},
            'dose':'Tentative: signed eye projection shift at half dose lies between zero and its same-sign full-dose shift. Retain rounding and nonmonotonic counterexamples.',
            'controls':'Shams and own receiver +/-1 endpoints reproduce previously observed exact PNGs. These are controls, not new prediction cases.',
            'math':'Seal 16 complete P/I forecasts before all new native mixed projections, candidate blocks and RGB. Cross CPU I futures should have RGB MAE below1 versus their native cached equivalents, calibrated by prior .080/.070; report all errors, not a discovery gate.',
            'scope':'Source-assisted deltaA=B(A10+A01-A00)-A11 in FP32; exactly same FP32 payload at fixed token/pixel coordinates before recipient BF16 add. Effective rounded delta may differ; measure it. Recipient M/R/gate/native outgoingT/latent/context retained. Two known seeds, same prompt/clock/weights; cross-seed coordinate alignment is not object alignment. No formation forecast or universal semantic portability claim.'},
        'scope':'2 receivers x2 source-assisted attention corrections x4 signed doses,2shams,2 CPU-cross futures.20RGB60native transitions40Acts,16fullblockvalidations18cachedP separately.1load0encodes/noise. All16P/I forecasts sealed first.'}
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config);worker=here/'flux_receiver_cross_campaign.py';names=['flux_receiver_cross.py','flux_interaction_writes.py','flux_single_contraction.py','flux_conditioning_program.py','flux_step1_overlap.py','flux_scene_dependencies.py','flux_predicted_future_campaign.py','flux_family_custody.py'];extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json')];pre=run/'preflight.v1.json'
    save(pre,{'schema_version':'mrun-preflight-config-v1','hashed_roots':[worker.name,*names,'plan.json',pre.name],'path_token_allowlist':['receiver-cross-result','input-data',config['model_path'],config['input_root'],config['output_scratch']]});extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+'\n').encode();job=read(writers/'run-01/job.json')['result']
    request={'schema':'mdb-receiver-cross-request-v1','config':config,'model':config['model'],'task_family':'forward','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'height':512,'width':512,'needs':{'cuda':True,'host':'beast'},'execution_environment':intent(),'config_bytes_sha256':hashlib.sha256(raw).hexdigest(),'source_hashes':{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        'scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_operations=20,logical_branches=20,logical_checkpoint_replays=20,max_local_work_items=60,checkpoint_policy='20 native futures40I/TActs60transitions;16candidateblocks18cachedP separate.16P/Iforecastsfirst. Serial inprocess,noretries.'),
        'resource_exception':{'ram_mb':10000,'vram_mb':13000,'disk_gb':5,'calibration_job':wa['job_id'],'measured_peak_rss_mb':job['peak_rss_mb'],'measured_peak_vram_mb':job['peak_vram_mb'],'reason':'Same noencoder resident BF16Klein4B512pxB1. Serial futures. Bounded2receiver operands +16CPU imageforecasts. Release temporary fourcorner mixes. Previously measured7699MBRSS10264MBVRAM;10GB/13GB. HDDscratch.'}}
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras);return {'run_dir':str(run),'request_sha256':request['request_sha256'],'counts':counts}


def submit(run):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    u=runpy.run_path(str(Path(__file__).with_name('flux_predicted_future_campaign.py')));r=json.loads((run/'request.json').read_text())
    if r['execution_environment']!=intent():raise ValueError('Restage environment before admission')
    if hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='request_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()!=r['request_sha256']:raise ValueError('Request seal differs')
    if (run/'submission.json').exists() or any(u['sha'](run/'payload'/n)!=h for n,h in r['source_hashes'].items()):raise ValueError('Admitted or mutated stage')
    cmd=worker_command('flux_receiver_cross_campaign.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-receiver-cross',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],ram_mb=10000,vram_mb=13000,disk_gb=5,cpu_threads=4,est_wall_s=360,timeout_s=720,retry_on_kill=False,preflight=True,
        note='Disentangle payload vsreceiver:2fixedFP32deltaA x2knownreceivers x4signeddoses plus2shams2CPUcrossfutures.20RGB60trans40Acts16candidateblocks18cachedP. Freeze16fullP/I beforemixednativework. Sameclock3/single18 source-assisted;crosscoordinate alignmentlimited.1residentload0encodes/noise,serial,10GBRAM13GBVRAM600sworker720slease,noretries,HDDscratch.')
    u['save'](run/'submission.json',{'job_id':str(job)});return {'job_id':str(job),'run_dir':str(run)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--source',type=Path);p.add_argument('--writers',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--collect');a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')))['collect'](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve(),a.writers.resolve())
    print(json.dumps(r,sort_keys=True))
