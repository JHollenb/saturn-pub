"""Stage one resident clock sweep using the unchanged frozen composition states."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,source):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    here=Path(__file__).resolve().parent;u=runpy.run_path(str(here/'flux_predicted_future_campaign.py'));save,sha=u['save'],u['sha']
    read=lambda p:json.loads(p.read_text())
    data=source/'run-03/extracted/frozen-composition-result';old=read(data/'plan.json');audit=read(source/'analysis-v1.json')
    if audit['mechanics_status']!='verified':raise ValueError('Verified source required')
    binding,frontier,summary=[read(data/n) for n in ('binding.json','frontier.json','summary.json')]
    values={**binding['observed_sources'],**binding['composed'],'joint':read(data/'new-joint.json')['encoding']}
    forecast=read(data/'forecasts.json')['additive'];manifest=read(data/'files.json');names={v['path'] for v in (*values.values(),forecast)}
    for fingerprint in (frontier['initial'],frontier['common_project_frontier']):
        name='store/cuts/'+fingerprint+'.json';names.add(name);cut=read(data/name)
        names.update('store/pages/'+h+'.json' for h in cut['slots'].values());names.update('store/blobs/'+h+'.safetensors' for h in cut['physical_blobs'])
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError('Source hash differs')
        files[name]={'bytes':(data/name).stat().st_size,'sha256':manifest[name]}
    rows=read(data/'rows.json')
    config={'schema':'mdb-edit-onset-plan-v1','model':old['model'],'model_path':old['model_path'],'checkpoint_sha256':old['checkpoint_sha256'],
        'seed':old['seed'],'prompts':old['prompts'],'rules':old['rules'],'onsets':[0,1,2],
        'input_root':summary['output_root']+'/frozen-composition-result','input_files':files,'conditioning':values,'predicted_additive_T':forecast,
        'initial_cut':frontier['initial'],'historical_step3':frontier['common_project_frontier'],'projector_weight_descriptor':binding['weight']['descriptor'],
        'source_audit_sha256':sha(source/'analysis-v1.json'),'source_forecast_seal_sha256':summary['seal']['seal_sha256'],
        'historical_pngs':{'base':next(r['png_sha256'] for r in rows if r['id']=='plain.base')},
        'hypotheses':{'shape_timing':'Starting coherent shape/joint E atstep0or1 should produce visibly narrower eyelids than recordedstep3, with strongest effect atstep0. Compare shapevsbase andjointvscolor separately; retain native prompt noncompliance and nonmonotonicity.',
            'composition':'Earlier additive E should retain violet color and approximate the nativejoint shape atmatchedonset. Prioroldinteraction didnothelp laststep eye error; do not assume its sign generalizes earlier. Score eachonset separately, no new fitting.',
            'repeated_prediction':'The same sealed additive initialT reference should apply at eachprojectread because this linear projection doesnotreadL/time. Repeatedtext replacements should remain within1RGBlevelMAE of nativeadditive futures atsameonset, calibrated by earlier one-read0.196. This is a forecast, not a discoverygate; maxima and accumulation retained.',
            'collateral':'Earlier edits may change more face/hair/clothing geometry. Coarse blackhair/tealshirt/ivorybackground/onefrontalportrait predicted toremain, with no precise identity/pixelpreservation bound.',
            'mechanics':'Loadedinitial state and plainbasereplay must reproduce historicalstep3payload/basePNG; Ebase Acts atallonsets must reproduce baseline. No newencodes/noisedraws. Hypotheses seal before newnativeprojectentry.'},
        'observers':{'windows':{'eyes':[130,180,380,290],'legacy_eyes':[140,155,370,235],'hair':[140,20,380,145],'clothing':[150,375,360,500],'background':[0,0,85,290]},
            'scope':'Fixedwindows before newonsetoutcomes, expandedeyes frompriorreview. Huecounts diagnostic; eyesmaymove at earlieronsets. Geometry evaluatedunblindedagainstnativecolor/shapecontrols, no calibrated classifier orsemanticgate.'},
        'output_scratch':'/home/beast/.cache/mdb-research','max_wall_seconds':600,
        'scope':'Same seed60666/prompts/E/rules as frozencomposition, varyonset0/1/2; originalstep3 retainedseparately. One112-stepbasereplay+21nativeE suffixes+3stackedpredictedT suffixes=2128transitions33Acts2161receipts25VAE. 1load0encodes0noise. AllactualE/savedinitialT sources observed before run; prospectiveclock/futuretest, not unseenpromptorindependentimages claim.'}
    expected=runpy.run_path(str(here/'flux_edit_onset.py'))['census'](config['onsets']);config['expected_counts']=expected
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config)
    worker=here/'flux_edit_onset_campaign.py';names=['flux_edit_onset.py','flux_scene_dependencies.py','flux_predicted_future_campaign.py','flux_family_custody.py']
    extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,{'schema_version':'mrun-preflight-config-v1','hashed_roots':[worker.name,*names,'plan.json',pre.name],
        'path_token_allowlist':['edit-onset-result','input-data',config['model_path'],config['input_root'],config['output_scratch']]});extras.append((pre,pre.name))
    raw=(json.dumps(config,sort_keys=True)+'\n').encode()
    request={'schema':'mdb-edit-onset-request-v1','config':config,'model':config['model'],'task_family':'forward','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'height':512,'width':512,
        'needs':{'cuda':True,'host':'beast'},'execution_environment':intent(),'config_bytes_sha256':hashlib.sha256(raw).hexdigest(),'source_hashes':{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        'scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_operations=25,logical_branches=25,logical_checkpoint_replays=24,max_local_work_items=32,
            checkpoint_policy='112sharedbase+7Evariants*(112+84+56)+3stackedT*(112+84+56)=2128transitions.21EActs+3stackedEActs+9textActs=33Acts2161receipts25VAE. Retainedclocklatents;1load0encodes0noise,noretries.'),
        'resource_exception':{'ram_mb':10000,'vram_mb':13000,'disk_gb':6,'calibration_job':'job-e23befc395c8','reason':'Same noencoder residentKlein4B512pxCUDA BF16B1 lane.7E arrays plus oneinitialT, serialsuffixes, nofullweightcopies or NumPypredictor. Prior10GBRAM13GBVRAM measurednoencoderlane; clocksincreaseiterations not simultaneousactivationgeometry. Not a speedupclaim.'}}
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request)
    stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return {'run_dir':str(run),'request_sha256':request['request_sha256'],'submitted':False}


def submit(run):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    u=runpy.run_path(str(Path(__file__).with_name('flux_predicted_future_campaign.py')));r=json.loads((run/'request.json').read_text())
    if r['execution_environment']!=intent():raise ValueError('Restage execution environment before admission')
    if hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='request_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()!=r['request_sha256']:raise ValueError('Request seal differs')
    if (run/'submission.json').exists() or any(u['sha'](run/'payload'/n)!=h for n,h in r['source_hashes'].items()):raise ValueError('Admitted or mutated stage')
    cmd=worker_command('flux_edit_onset_campaign.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-edit-onset',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],ram_mb=10000,vram_mb=13000,disk_gb=6,cpu_threads=4,est_wall_s=300,timeout_s=720,retry_on_kill=False,preflight=True,
        note='Sameportrait60666/unchanged frozenE rules. Earlieronset0/1/2 x7Eroles plus3stackedadditiveT;1basereplay25images2128transitions33Acts2161receipts.1load0encodes0noise. Sealbefore newnativeproject; preservegeometry/collateral/nonmonotonicity. SavedinitialT reinstalledeachremainingprojectread; clocks/latenttraces retained.10GBRAM13GBVRAM qualifiednoencoderlane;600sworker720slease,noretries.')
    u['save'](run/'submission.json',{'job_id':str(job)});return {'job_id':str(job),'run_dir':str(run)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--source',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--collect');a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')))['collect'](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve())
    print(json.dumps(r,sort_keys=True))
