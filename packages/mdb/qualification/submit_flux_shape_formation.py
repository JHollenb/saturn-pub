"""Admit finite early-state handoff/clamp/rescue discovery or new-seed replication."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,source,overlap,fresh_seed=None,discovery=None):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    here=Path(__file__).resolve().parent;u=runpy.run_path(str(here/'flux_predicted_future_campaign.py'));save,sha=u['save'],u['sha'];read=lambda p:json.loads(p.read_text())
    data=source/'run-01/extracted/step1-writers-result';old=read(data/'plan.json');summary=read(data/'summary.json')
    if read(source/'analysis-v1.json')['mechanics_status']!='verified' or read(overlap/'analysis-v1.json')['mechanics_status']!='verified':raise ValueError('Verified predecessors required')
    overlap_data=overlap/'run-01/extracted/step1-overlap-result';prior=read(overlap_data/'plan.json')
    if prior['source_audit_sha256']!=sha(source/'analysis-v1.json'):raise ValueError('Predecessor lineage differs')
    roots={k:read(data/'cuts.json')['step1'][k]['after_joint4'] for k in ('base','shape')};names=set();manifest=read(data/'files.json')
    for fp in roots.values():
        name='store/cuts/'+fp+'.json';names.add(name);cut=read(data/name)
        if cut['execution_point']['logical_step']!=1 or cut['execution_point']['phase']!='single' or cut['execution_point']['index']!=0:raise ValueError('Step1/single0 source required')
        names.update('store/pages/'+h+'.json' for h in cut['slots'].values());names.update('store/blobs/'+h+'.safetensors' for h in cut['physical_blobs'])
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError('Source CAS differs')
        files[name]={'bytes':(data/name).stat().st_size,'sha256':manifest[name]}
    H=runpy.run_path(str(here/'flux_shape_formation.py'));expected=H['census'](fresh=fresh_seed is not None)
    hypotheses={'controls':'Known weak/strong references reproduce prior onset2.shape/onset1.shape PNGs. IT0/IT6 reproduce strongreference. Freshseed references have no historicalPNG expectation.',
        'early_text':'Tentative prediction: T-only handoff atsingle0 iscloser tostrong eyeRGB than I-only atsamecut; no calibratedgeometrypercentage assumed.',
        'writing':'Tentative prediction: Tseed0 followedbyTreset6 retainsmore strong eyecontrast than unmodifiedbase; this testswhether earlyT creates useful I before6. ItsR/T trajectory isexplicit.',
        'clamp_rescue':'Tentative prediction: baseTclamp through6 weakens eyecontrast relative toshape reference; restoring shapeT at6 afterclamps0..5 improvesmatching versuscontinuedclamp. Retain failure/repair/collateral.',
        'scope':'The field amplitudes/causalranks arecontextual. Observe I/T atallboundaries0..7; do notassigna universalwriter orsemanticneuron. AlllaterEshape fixed. Nativeformation/suffix, notindependentneuralformationforecast.'}
    if fresh_seed is not None:
        if discovery is None or fresh_seed==old['seed']:raise ValueError('Freshseed requires auditeddiscovery andunusedseed')
        a=read(discovery/'analysis-v1.json')
        if a['mechanics_status']!='verified':raise ValueError('Discovery must beaudited')
        frozen=read(discovery/'replication-predictions-v1.json')
        if frozen['discovery_audit_sha256']!=sha(discovery/'analysis-v1.json') or frozen['seed']!=fresh_seed:raise ValueError('Frozenreplicationidentity differs')
        hypotheses=frozen['hypotheses']
    config={'schema':'mdb-shape-formation-plan-v1',**{k:old[k] for k in ('model','model_path','checkpoint_sha256','seed','prompts','observers')},'fresh_seed':fresh_seed,
        'input_root':summary['output_root']+'/step1-writers-result','input_files':files,'input_cuts':roots,'source_audit_sha256':sha(source/'analysis-v1.json'),'overlap_audit_sha256':sha(overlap/'analysis-v1.json'),
        'historical_pngs':{} if fresh_seed is not None else prior['historical_pngs'],'recipes':H['recipes'](),'hypotheses':hypotheses,
        'replication_prediction':None if fresh_seed is None else {'path':str(discovery/'replication-predictions-v1.json'),'sha256':sha(discovery/'replication-predictions-v1.json')},
        'expected_counts':expected,'output_scratch':'/home/beast/.cache/mdb-research','max_wall_seconds':600,
        'scope':'Step1/single0..7 I/Tnativeformation, handoffs0/3/6, Tseed/reset6, T/Iclamps0..6 andT rescue6.14nativefutures commonshapeE from2; no candidateblocks. KnownsourceE; freshseed optionallycreatesnewreceivingstateonlyafterhypothesisseal.'}
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config);worker=here/'flux_shape_formation_campaign.py'
    names=['flux_shape_formation.py','flux_step1_writers.py','flux_conditioning_program.py','flux_scene_dependencies.py','flux_predicted_future_campaign.py','flux_family_custody.py'];extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json')];pre=run/'preflight.v1.json'
    save(pre,{'schema_version':'mrun-preflight-config-v1','hashed_roots':[worker.name,*names,'plan.json',pre.name],'path_token_allowlist':['shape-formation-result','input-data',config['model_path'],config['input_root'],config['output_scratch']]});extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+'\n').encode()
    request={'schema':'mdb-shape-formation-request-v1','config':config,'model':config['model'],'task_family':'forward','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'height':512,'width':512,
        'needs':{'cuda':True,'host':'beast'},'execution_environment':intent(),'config_bytes_sha256':hashlib.sha256(raw).hexdigest(),'source_hashes':{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        'scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_operations=14,logical_branches=14,logical_checkpoint_replays=14,max_local_work_items=32,checkpoint_policy='14nativefutures+2shared7step sourceprefixes, typedclamps/rescues. Optionalfreshnoise adds40native/2Acts. Allbrancheslocal,1load0encodes, no retries.'),
        'resource_exception':{'ram_mb':10000,'vram_mb':13000,'disk_gb':4,'calibration_job':'job-3fa5d0f49cf5','measured_peak_rss_mb':8478.6,'measured_peak_vram_mb':9558,'reason':'Same measured noencoderKlein4B512pxBF16B1 lane; no copiedlargeprojectionweights orCPUforecasts. Moretypedcuts retained,14serialfutures.10GBRAM13GBVRAM measuredbound, not runtimegainclaim.'}}
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return {'run_dir':str(run),'request_sha256':request['request_sha256'],'expected_counts':expected}


def submit(run):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    u=runpy.run_path(str(Path(__file__).with_name('flux_predicted_future_campaign.py')));r=json.loads((run/'request.json').read_text())
    if r['execution_environment']!=intent():raise ValueError('Restage environment before admission')
    if hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='request_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()!=r['request_sha256']:raise ValueError('Request seal differs')
    if (run/'submission.json').exists() or any(u['sha'](run/'payload'/n)!=h for n,h in r['source_hashes'].items()):raise ValueError('Admitted or mutated stage')
    cmd=worker_command('flux_shape_formation_campaign.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-shape-formation',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],ram_mb=10000,vram_mb=13000,disk_gb=4,cpu_threads=4,est_wall_s=240,timeout_s=720,retry_on_kill=False,preflight=True,
        note='Step1/single0..7 I/Tnativeformation:handoff0/3/6, Tseed/reset6,T/Iclamps,Trescue6.14RGB commonfutureEshape.1068native47Acts1115receipts; optionalfreshseedadds40native2Acts1noise. Hypotheses sealbeforeprefix/freshnoise.1load0encodes,nocandidateblocks. Fulltypedearlycuts/latenttraces, sourceassistanceexplicit.10GBRAM13GBVRAM600sworker720slease,noretries.')
    u['save'](run/'submission.json',{'job_id':str(job)});return {'job_id':str(job),'run_dir':str(run)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--source',type=Path);p.add_argument('--overlap-source',type=Path);p.add_argument('--fresh-seed',type=int);p.add_argument('--discovery',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--collect');a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')))['collect'](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve(),a.overlap_source.resolve(),a.fresh_seed,a.discovery.resolve() if a.discovery else None)
    print(json.dumps(r,sort_keys=True))
