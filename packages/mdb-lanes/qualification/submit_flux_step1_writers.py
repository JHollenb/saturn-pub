"""Stage/admit a matched step1 pulse, handoff and mathematical writer panel."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,source):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    here=Path(__file__).resolve().parent;u=runpy.run_path(str(here/'flux_predicted_future_campaign.py'));save,sha=u['save'],u['sha']
    read=lambda p:json.loads(p.read_text())
    data=source/'run-01/extracted/edit-onset-result';old=read(data/'plan.json');summary=read(data/'summary.json');audit=read(source/'analysis-v1.json')
    if audit['mechanics_status']!='verified':raise ValueError('Verified predecessor required')
    rows={r['id']:r for r in read(data/'rows.json')};root=read(data/'frontiers.json')['1']
    programs={k:rows['onset1.'+k]['actions'][0]['post_act'] for k in ('base','color','shape','joint')}
    names=set();manifest=read(data/'files.json')
    for fp in (root,*programs.values()):
        name='store/cuts/'+fp+'.json';names.add(name);cut=read(data/name)
        if cut['execution_point']['logical_step']!=1 or cut['execution_point']['phase']!='project':raise ValueError('Step1 project source required')
        names.update('store/pages/'+h+'.json' for h in cut['slots'].values());names.update('store/blobs/'+h+'.safetensors' for h in cut['physical_blobs'])
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError('Source CAS differs')
        files[name]={'bytes':(data/name).stat().st_size,'sha256':manifest[name]}
    expected=runpy.run_path(str(here/'flux_step1_writers.py'))['census']()
    config={'schema':'mdb-step1-writers-plan-v1','model':old['model'],'model_path':old['model_path'],'checkpoint_sha256':old['checkpoint_sha256'],'seed':old['seed'],'prompts':old['prompts'],
        'input_root':summary['output_root']+'/edit-onset-result','input_files':files,'root_cut':root,'program_cuts':programs,'source_audit_sha256':sha(source/'analysis-v1.json'),
        'historical_pngs':{'base':rows['base']['png_sha256'],'shape':rows['onset1.shape']['png_sha256'],'joint':rows['onset1.joint']['png_sha256']},
        'handoffs':{'sites':['after_joint0','after_joint4','after_single6','after_single18'],'modes':['I','T','IT'],'source':'shape','host':'base','later_conditioning':'base beforestep2'},
        'write_rows':{'block':'step1/joint0','bits':['10','01'],'text_roles':['base','shape'],'equation':'B(B(R+B(ga*A_selected))+B(gm*M_selected))','source_values':'Observed matched base/shape R/A/M/gates; no A/M formation forecast','semantics':'Frozenobservedwrites, not sequentialMLPrecomputation afterselectedA'},
        'hypotheses':{'pulse':'Tentative prediction: one step1 shape/joint pulse leaves some narrower-eyelid response after reverting E tobase beforestep2. Retain partialpersistence, repair and absence underthisclock without treating it as mechanismabsence.',
            'handoff':'Tentative orderedprediction: atafter_joint0 T-only eyeRGB iscloser to pulse.shape than I-only; byafter_single6 orafter_single18 I-only iscloser thanT-only. RGBdistance isnot calibratedshape, assessvisualcontrols separately.',
            'writes':'Atfixedjoint0 incomingI/gates, sourceattention introducesimage differences and sequentialMLP maytransformthem. Tentativeconsumerprediction: A1M0 withbaseT iscloser to pulse.shape eyeRGB thanA0M1 withbaseT. Comparethe samefields withshapeT; no fixedsemanticmeaning oruniversalsign assigned.',
            'numeric':'Allfourfullimageexit forecasts fromobserved R/A/M and nativeBF16edges predicted to match nativefrozenwrite substitutions exactly. Forecasts seal aftersourceports butbeforemixedblocks andanyRGBfuture. This doesnot predictsourceA/M orcompleteRGB.',
            'controls':'pulse.base and continuous.shape/joint should reproduce historical PNGs. Allfour IT handoffs should reproducepulse.shape despiteunusedwithin-stepE difference, onceE resetbeforestep2. Preserve controlresults perrow.',
            'collateral':'Coarse blackhair/tealshirt/ivorybackground/frontalportrait predicted toremain, with nonzero finecollateral. Fullimage/text fields arechanged; no localized eye-neuron/automatic objectaddress claim.'},
        'observers':old['observers'],'expected_counts':expected,'output_scratch':'/home/beast/.cache/mdb-research','max_wall_seconds':600,
        'scope':'Known60666portrait fixedbaseL1/clock. Fourstep1E pulses withbaseEfrom2, two ownE continuations,12I/Thandoffs atfourcuts,4predictedjoint0exitfutures crossingfrozenA/M andretainedT.22images1646native steps44Acts1690receipts5counteddirectblocks.1load0encodes/noise. Sourceports observed; exactexitaccounting andnativeconsumerinterventions, not independentMLPformation orfullimageprediction.'}
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config)
    worker=here/'flux_step1_writers_campaign.py';names=['flux_step1_writers.py','flux_conditioning_program.py','flux_scene_dependencies.py','flux_joint_writes.py','flux_predicted_future_campaign.py','flux_family_custody.py']
    extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,{'schema_version':'mrun-preflight-config-v1','hashed_roots':[worker.name,*names,'plan.json',pre.name],
        'path_token_allowlist':['step1-writers-result','input-data',config['model_path'],config['input_root'],config['output_scratch']]});extras.append((pre,pre.name))
    raw=(json.dumps(config,sort_keys=True)+'\n').encode()
    request={'schema':'mdb-step1-writers-request-v1','config':config,'model':config['model'],'task_family':'forward','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'height':512,'width':512,
        'needs':{'cuda':True,'host':'beast'},'execution_environment':intent(),'config_bytes_sha256':hashlib.sha256(raw).hexdigest(),'source_hashes':{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        'scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_operations=27,logical_branches=22,logical_checkpoint_replays=22,max_local_work_items=32,
            checkpoint_policy='Four28step sourceprefixes+4pulse56+2continuous56+3handoffs*(82+78+71+59)+4law82=1646native.44Acts1690receipts22VAE;2sourceblock captures+3fullmixedvalidations separatelycounted.1load0encodes/noise,no retries.'),
        'resource_exception':{'ram_mb':10000,'vram_mb':13000,'disk_gb':6,'calibration_job':'job-13cddbd2f9ab','measured_peak_rss_mb':5729.9,'measured_peak_vram_mb':9298,
            'reason':'Same noencoder residentKlein4B512pxCUDA BF16B1 lane.4sourceE cuts, base/shapecapturedports,4complete3.14Mvalueexitforecasts and3nativevalidationarrays add<256MBretainedtensors; nofullweightcopies. Serial22futures;10GBRAM13GBVRAM measuredlane, not speedupclaim.'}}
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request)
    stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return {'run_dir':str(run),'request_sha256':request['request_sha256'],'submitted':False}


def submit(run):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    u=runpy.run_path(str(Path(__file__).with_name('flux_predicted_future_campaign.py')));r=json.loads((run/'request.json').read_text())
    if r['execution_environment']!=intent():raise ValueError('Restage environment beforeadmission')
    if hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='request_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()!=r['request_sha256']:raise ValueError('Requestseal differs')
    if (run/'submission.json').exists() or any(u['sha'](run/'payload'/n)!=h for n,h in r['source_hashes'].items()):raise ValueError('Admitted ormutated stage')
    cmd=worker_command('flux_step1_writers_campaign.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-step1-writers',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],ram_mb=10000,vram_mb=13000,disk_gb=6,cpu_threads=4,est_wall_s=240,timeout_s=720,retry_on_kill=False,preflight=True,
        note='Fixedportrait60666/baseL1. Fourone-stepEpulses revertbaseE before2;2continuouscontrols;I/T/IT handoffs afterjoint0/joint4/single6/single18;4joint0frozenA/M+Tmathfutures.22RGB1646native44Acts1690receipts+5directblocks. Mainhypotheses sealbeforeproject, fourfullnumericexits sealafterports beforemixedblocks/RGB.1load0encodes/noise;10GBRAM13GBVRAM qualifiedlane600sworker720slease,noretries. Preservepulse repair, collateral andcounterexamples.')
    u['save'](run/'submission.json',{'job_id':str(job)});return {'job_id':str(job),'run_dir':str(run)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--source',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--collect');a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')))['collect'](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve())
    print(json.dumps(r,sort_keys=True))
