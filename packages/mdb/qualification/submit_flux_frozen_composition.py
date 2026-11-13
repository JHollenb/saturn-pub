"""Stage/admit one fresh-context, target-quarantined composition experiment."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run, source):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    u=runpy.run_path(str(Path(__file__).with_name('flux_predicted_future_campaign.py')))
    read=lambda p:json.loads(p.read_text())
    save,sha=u['save'],u['sha']
    data=source/'run-01/extracted/conditioning-program-result'
    binding,summary=read(data/'binding.json'),read(data/'summary.json')
    if read(source/'analysis-v1.json')['mechanics_status']!='verified': raise ValueError('Audited predecessor required')
    old={k:binding['conditioning'][k] for k in ('joint','additive')}
    files={v['path']:{'bytes':(data/v['path']).stat().st_size,'sha256':sha(data/v['path'])} for v in old.values()}
    common='Anime head-and-shoulders portrait of one adult woman, straight black shoulder-length hair, teal collared shirt, pale ivory studio background, facing the camera, clean line art, even lighting. '
    prompts={k:common+v for k,v in {'base':'Amber eyes with round, open eyelids.','color':'Violet eyes with round, open eyelids.','shape':'Amber eyes with narrow, almond-shaped eyelids.','joint':'Violet eyes with narrow, almond-shaped eyelids.'}.items()}
    config={'schema':'mdb-frozen-composition-plan-v1','model':'flux2-klein-4b-distilled','model_path':'/mnt/big/llm-models/FLUX.2-klein-4B',
        'checkpoint_sha256':'dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f','seed':60666,'height':512,'width':512,'steps':4,'prompts':prompts,
        'input_root':summary['output_root']+'/conditioning-program-result','input_files':files,'old_conditioning':old,'source_audit_sha256':sha(source/'analysis-v1.json'),
        'output_scratch':'/home/beast/.cache/mdb-research','max_wall_seconds':600,
        'rules':{'additive':'B(Ec_new+Es_new-Eb_new)','plus_old':'B(Eadd_new+Ej_old-Eadd_old)','minus_old':'B(Eadd_new-Ej_old+Eadd_old)','coefficient':1,'alignment':'Unchanged512rowcoordinates; no fitted alignment, no newjointencoding in candidate rule','clock':'step3project on common nativebase L3'},
        'hypotheses':{'primary':'Frozen additive composition should produce violet-looking eyes and narrower eyelids than base while retaining black hair, teal shirt, ivory background and one frontal adult portrait. Compare with newjoint under identicalbaseL3; preserve misses.',
            'transfer':'Exploratory signed prediction: plus_old should reduce eye-window distance to newjoint versus additive; minus_old should increase it. Receiving-state dependence may invalidate this fixed historical correction; no tuning after outcomes.',
            'preservation':'All three candidate states are predicted to retain coarse blackhair/tealshirt/ivorybackground/frontalcomposition. Regional pixelcollateral may remain nonzero; no uniform bound predicted.',
            'numerical':'Predicted initialT for each candidate should yield RGBMAE below1/255 against its matched nativeprojector future, calibrated by two prior cases0.284/0.296. This is a prospective forecast, not an exploration gate; maximum errors retained.',
            'controls':'Plainbase and sameEbase Act must match exactly in latent/RGB. Three sourceencodes and every candidate/initialT forecast sealedbefore newjointencode and anydenoiserexecution.'},
        'observers':{'windows':{'eyes':[140,155,370,235],'hair':[140,20,380,145],'clothing':[150,375,360,500],'background':[0,0,85,290]},'hue_bands':{'gold':[20,80],'violet':[240,320],'min_saturation':.20,'min_value':.10},'scope':'Suppliedpixelwindows, diagnosticHSVcounts and unblindedappearance review. Nativecolor/shapecontrols assessinstrument applicability; no semantic gates or automaticsegmentation.'},
        'scope':'Freshseed60666/newwording+hair/clothing/background. Three observed encodings +fixedoldinteraction; allcandidateE/Tpredictions sealedbefore newjointencode. Commonbasefirst3steps then11nativeimages389transitions10Acts399receipts,4encodes1noise1load. Sourceencoding/nativebasehistory and suffix retained; no rawtextcompleteRGBprediction or speedupclaim.'}
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config)
    here=Path(__file__).resolve().parent;worker=here/'flux_frozen_composition_campaign.py'
    names=['flux_frozen_composition.py','flux_conditioning_program.py','flux_predicted_future_campaign.py','flux_scene_dependencies.py','flux_family_custody.py']
    extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,{'schema_version':'mrun-preflight-config-v1','hashed_roots':[worker.name,*names,'plan.json',pre.name],
        'path_token_allowlist':['frozen-composition-result','input-data',config['model_path'],config['input_root'],config['output_scratch']]});extras.append((pre,pre.name))
    raw=(json.dumps(config,sort_keys=True)+'\n').encode()
    request={'schema':'mdb-frozen-composition-request-v1','config':config,'model':config['model'],'task_family':'forward','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'height':512,'width':512,
        'needs':{'cuda':True,'host':'beast'},'execution_environment':intent(),'config_bytes_sha256':hashlib.sha256(raw).hexdigest(),'source_hashes':{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        'scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_operations=11,logical_branches=11,logical_checkpoint_replays=10,max_local_work_items=24,checkpoint_policy='One84stepbaseprefix+plain28+7native28+3predicted27=389native transitions10Acts399receipts11VAE.4serialencodes including quarantinedjoint afterseal. One finitelease/no retries.'),
        'resource_exception':{'ram_mb':24000,'vram_mb':13000,'disk_gb':6,'calibration_job':'job-64730d03279d','reason':'Qualified fullencoder residentKlein4B CUDA BF16B1 512px lane; serialencodes, releaseencoder beforetransformer/VAE. Priorfullencoder24GBRAM13GBVRAM startupmeasuredprofile;3tiledlinearforecasts, two historicalconditioningarrays. Not a speedupclaim.'}}
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request)
    stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return {'run_dir':str(run),'request_sha256':request['request_sha256'],'submitted':False}


def submit(run):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    u=runpy.run_path(str(Path(__file__).with_name('flux_predicted_future_campaign.py')))
    r=json.loads((run/'request.json').read_text())
    if r['execution_environment'] != intent():
        raise ValueError('Staged execution environment differs from submit environment; restage before admission')
    unsigned={k:v for k,v in r.items() if k!='request_sha256'}
    if hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest()!=r['request_sha256']: raise ValueError('Request seal differs')
    if (run/'submission.json').exists() or any(u['sha'](run/'payload'/n)!=h for n,h in r['source_hashes'].items()): raise ValueError('Admitted or mutated stage')
    cmd=worker_command('flux_frozen_composition_campaign.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-frozen-composition',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],ram_mb=24000,vram_mb=13000,disk_gb=6,cpu_threads=4,est_wall_s=360,timeout_s=720,retry_on_kill=False,preflight=True,
        note='FreshKlein4B BF16B1512px seed60666/newwording. Freeze sourceonlyadditive and +/-historicalinteractionE plus3fullinitialT forecasts before newjointencode or denoiser.1load4encodes1noise84sharedprefix389totaltransitions10Acts399receipts11VAE. CommonbaseL3/unchangednativeconsumer. Retainsemantic/collateralmisses. Qualified24GBRAM13GBVRAM fullencoderlane600sworker720slease,noretries.')
    u['save'](run/'submission.json',{'job_id':str(job)});return {'job_id':str(job),'run_dir':str(run)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--source',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--collect');a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')))['collect'](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve())
    print(json.dumps(r,sort_keys=True))
