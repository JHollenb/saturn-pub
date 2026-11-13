"""Small receiving-latent changes using observed readouts and predicted Euler commits."""
import argparse
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

ALPHAS=(0.,.05,.1,.25,.5,1.)


def read(p):return json.loads(p.read_text())


def mix(a,b,alpha):
    import torch
    if alpha==0:return a.clone()
    if alpha==1:return b.clone()
    return (a.float()+alpha*(b.float()-a.float())).to(a.dtype)


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/'flux_predicted_future_campaign.py'));save,sha,identity=[u[k] for k in ('save','sha','identity')];config,request=[read(Path(os.environ[k])) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config'] or any(sha(p)!=h for p,h in request['source_hashes'].items()):raise ValueError('Staging differs')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch);out=Path('latent-neighborhood-result');out.mkdir();save(out/'plan.json',config);save(out/'request.json',request);root=Path(config['input_root']).resolve(strict=True)
    for name,m in config['input_files'].items():
        src,dst=root/name,Path('input-data')/name
        if src.is_symlink() or not src.resolve().is_relative_to(root) or src.stat().st_size!=m['bytes'] or sha(src)!=m['sha256']:raise ValueError('Source CAS differs')
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(src.read_bytes())
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from mdb_lanes.native_navigation import NativeCutNavigator
    from mdb_lanes.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb_lanes.tensor_archive import write_tensor_archive
    euler=runpy.run_path(str(payload/'flux_scene_dependencies.py'))['predict_euler'];terminal=runpy.run_path(str(payload/'flux_scene_dependencies.py'))['terminal_from_forecast']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts=dict(loads=0,encodes=0,noise_draws=0,transformer_forwards=0,readouts=0,native_transitions=0,acts=0,receipts=0,images=0,euler_forecasts=0,derived_scheduler_initializations=0,derived_terminal_initializations=0);timeline=[];rows=[];forecasts={};values={};receipts=[]
    def event(kind,**kw):
        r={'kind':kind,'seconds':time.monotonic()-started,'counts':dict(counts),**kw};timeline.append(r);print('MDB_LATENT_NEIGHBORHOOD '+json.dumps(r,sort_keys=True),flush=True)
    def array(name,v):return write_tensor_archive(out/'arrays'/(name+'.npy'),v,root=out)
    model=Path(config['model_path']);inventory={p.relative_to(model).as_posix():sha(p) for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']:raise ValueError('Checkpoint differs')
    event('loading');pipe=Flux2KleinPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts['loads']+=1
    for c in (pipe.transformer,pipe.vae):c.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda');original=LocalStore(Path('input-data/store'));store=LocalStore(out/'store')
    def denoiser_called(module,args):counts['transformer_forwards']+=1
    def readout_called(module,args):counts['readouts']+=1
    handles=[m.register_forward_pre_hook(denoiser_called) for m in (pipe.transformer,pipe.transformer.x_embedder,pipe.transformer.context_embedder,*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks)]
    handles.append(pipe.transformer.proj_out.register_forward_pre_hook(readout_called))
    fps={v[k] for v in config['sources'].values() for k in ('scheduler','reference_final','other_parent')};cuts={fp:original.load(fp,device='cuda') for fp in fps}
    for c in cuts.values():
        if c.model_identity!=adapter.model_identity or dict(c.execution)!=dict(adapter.execution):raise ValueError('Lane differs')
        store.save(c)
    save(out/'source.json',{'model_identity':adapter.model_identity,'execution':dict(adapter.execution),'checkpoint_sha256':identity(inventory)})
    tx=ForecastTransaction(out/'transaction',model_identity=adapter.model_identity,input_identity=identity(config['sources']),source_identity=identity(request['source_hashes']),frontier={'kind':'architectural','observed':'Saved complete native scheduler inputs, both native receiving latents and old alpha0/1 images; intermediate-alpha commits/images unexecuted'},execution_scope=config['scope'],budgets={'max_native_calls':1,'max_wall_ms':600000})
    with tx:
        tx.frontier('sources',config['sources'])
        def predict(_):
            for key,b in config['sources'].items():
                s=cuts[b['scheduler']];z0=s.payload['latent'];z1=cuts[b['other_parent']].payload['latent'];dt=float((s.payload['sigmas'][4]-s.payload['sigmas'][3]).item())
                for alpha in ALPHAS:
                    name=key+f'.a{alpha:g}';z=mix(z0,z1,alpha);pred=euler(z.float().cpu().numpy(),s.payload['noise_prediction'].float().cpu().numpy(),dt);value=torch.from_numpy(pred).to('cuda',torch.bfloat16);values[name]={'input':z,'output':value};counts['euler_forecasts']+=1
                    effects={axis:(1-alpha)*b['endpoint_effects']['0'][axis]+alpha*b['endpoint_effects']['1'][axis] for axis in b['endpoint_effects']['0']}
                    forecasts[name]={'source':key,'alpha':alpha,'inputZ3':array(name+'.inputZ3',z),'predictedZ4':array(name+'.predictedZ4',value),'sigma_delta':dt,'predicted_pair_effects':effects}
            save(out/'forecasts.json',forecasts);return {'predictions':{'fields':forecasts,'hypotheses':config['hypotheses'],'counts_at_seal':dict(counts)}}
        tx.predict('latent_neighborhood',predict);seal=tx.seal();event('sealed',seal=seal)
        def execute(_):
            for name,f in forecasts.items():
                b=config['sources'][f['source']];s=cuts[b['scheduler']];validation=None
                if f['alpha']==.1:
                    # Explicitly initialized counterfactual scheduler input, never a late latent Act or historical rewind.
                    state=s.payload;state['latent']=values[name]['input'];session=Session(adapter,state,parent=s.fingerprint);derived=session.capture(retain=False);store.save(derived);counts['derived_scheduler_initializations']+=1;n=NativeCutNavigator(session,max_steps=1);receipt=n.step();receipts.append(receipt.to_dict());counts['native_transitions']+=1;counts['receipts']+=1;actual=n.session.capture(retain=False);store.save(actual);validation={'derived_initial_cut':derived.fingerprint,'receipt':receipt.to_dict(),'native_final_cut':actual.fingerprint,'predicted_exact':bool(torch.equal(actual.payload['latent'],values[name]['output']))}
                session=terminal(adapter,cuts[b['reference_final']],values[name]['output'],parent=s.fingerprint);derived_final=session.capture(retain=False);store.save(derived_final);counts['derived_terminal_initializations']+=1;png=out/(name+'.png');adapter.decode(session,height=512,width=512).save(png);counts['images']+=1
                r={'id':name,'pair':b['pair'],'bit':b['bit'],'alpha':f['alpha'],'scheduler_source':s.fingerprint,'terminal_cut':derived_final.fingerprint,'lineage_kind':'derived-terminal-initialization-from-sealed-Euler-prediction; notnativeforwardorAct','native_validation':validation,'png':png.name,'png_sha256':sha(png)}
                if f['alpha'] in (0.,1.):r.update(expected_png=b['historical_pngs'][str(int(f['alpha']))],png_control_exact=sha(png)==b['historical_pngs'][str(int(f['alpha']))])
                rows.append(r);save(out/'rows'/(name+'.json'),r);event('image_complete',branch=name)
                if time.monotonic()-started>600:raise RuntimeError('Finite neighborhood wall budget exceeded')
            return {'counts':dict(counts),'rows':len(rows)}
        tx.execute('four_native_Euler_checks_and_predicted_terminal_decodes',execute)
    expected=dict(loads=1,encodes=0,noise_draws=0,transformer_forwards=0,readouts=0,native_transitions=4,acts=0,receipts=4,images=24,euler_forecasts=24,derived_scheduler_initializations=4,derived_terminal_initializations=24)
    if counts!=expected:raise ValueError('Completed neighborhood census differs')
    save(out/'rows.json',rows);save(out/'receipts.json',receipts);save(out/'timeline.json',timeline);summary={'schema':'mdb-latent-neighborhood-v1','job_id':os.environ['MRUN_JOB_ID'],'request_sha256':request['request_sha256'],'counts':counts,'seal':seal,'transaction':verify_transaction(tx.root),'worker_seconds':time.monotonic()-started,'output_root':str(Path.cwd()),'mechanics_status':'pending-independent-audit','terminal_status':'unassessed','scope':config['scope']};save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()});runpy.run_path(str(payload/'flux_family_custody.py'))['emit'](out,summary,compression_level=1)


def stage(run,source):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    here=Path(__file__).resolve().parent;u=runpy.run_path(str(here/'flux_predicted_future_campaign.py'));save,sha=u['save'],u['sha'];data=source/'run-01/extracted/suffix-context-result';old=read(data/'plan.json');s=read(data/'summary.json');a=read(source/'analysis-v1.json');rows={r['id']:r for r in read(data/'rows.json')};sources={}
    if a['mechanics_status']!='verified':raise ValueError('Verified source required')
    for pair in ('60666','60667'):
        other='60667' if pair=='60666' else '60666'
        for bit in (0,1):
            r=rows[f'{pair}.I{bit}.T{pair}.Z{pair}'];sources[f'{pair}.I{bit}']={'pair':pair,'bit':bit,'scheduler':r['scheduler_cut'],'reference_final':r['final_cut'],'other_parent':old['parents'][other],'historical_pngs':{'0':r['png_sha256'],'1':rows[f'{pair}.I{bit}.T{pair}.Z{other}']['png_sha256']},'endpoint_effects':{'0':a['paired_observations'][f'{pair}.T{pair}.Z{pair}']['regions']['eyes']['fixed_axes'],'1':a['paired_observations'][f'{pair}.T{pair}.Z{other}']['regions']['eyes']['fixed_axes']}}
    names=set();manifest=read(data/'files.json')
    for fp in {b[k] for b in sources.values() for k in ('scheduler','reference_final','other_parent')}:
        n='store/cuts/'+fp+'.json';names.add(n);v=read(data/n);names.update('store/pages/'+h+'.json' for h in v['slots'].values());names.update('store/blobs/'+h+'.safetensors' for h in v['physical_blobs'])
    files={}
    for n in sorted(names):
        if sha(data/n)!=manifest[n]:raise ValueError('Source CAS differs')
        files[n]={'bytes':(data/n).stat().st_size,'sha256':manifest[n]}
    config={'schema':'mdb-latent-neighborhood-plan-v1',**{k:old[k] for k in ('model','model_path','checkpoint_sha256','observers')},'input_root':s['output_root']+'/suffix-context-result','input_files':files,'sources':sources,'source_audit_sha256':sha(source/'analysis-v1.json'),'output_scratch':'/mnt/hdd8tb/mdb/research','alphas':list(ALPHAS),'hypotheses':{'signed_linear_extrapolation':'For each pair/fixedRGBaxis, predict the paired edit effect at intermediatealpha by straight interpolation of observed0/1 effects. Freeze pointvalues andsigns; allquantitativeerrorsretained. Endpointsarecontrols.','quality':'Tentative: alpha.05/.1 retain coarseportraitstructure substantially better thanwholelatentexchange. Manualunblindedreview, no geometrygate.','math':'24completefinalZforecasts before newexecution/decodes;4nativeEulerchecks atalpha.1 shouldmatchbitwise. Earlier16/16exactcommits and8Zindependentreadoutpairs qualify reuse. Allnoiseobserved; zeronewtransformer/readoutcalls.','scope':'B(FP32(Z0)+alpha*(FP32(Z1)-FP32(Z0))) thenBF16Euler. EachpairkeepsownT/observednoise. ExplicitderivedterminalinitializationsforVAE;4derivedschedulerinitializationsfornativechecks. NolateZAct, nohistoricalrewindclaim, no newpromptsourceforecast.'},'scope':'24predicted-terminalVAEimages:2pairs x2Ichoices x6latentdoses.8historicalendpointcontrols16newimages8newpairedsigncases.4nativeEulerchecks0transformerforwards0readouts0Acts. AllfullZforecastsbeforeexecution.'}
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config);names=['flux_scene_dependencies.py','flux_predicted_future_campaign.py','flux_family_custody.py'];worker=Path(__file__).resolve();extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json')];pre=run/'preflight.v1.json';save(pre,{'schema_version':'mrun-preflight-config-v1','hashed_roots':[worker.name,*names,'plan.json',pre.name],'path_token_allowlist':['latent-neighborhood-result','input-data',config['model_path'],config['input_root'],config['output_scratch']]});extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+'\n').encode();job=read(source/'run-01/job.json')['result']
    request={'schema':'mdb-latent-neighborhood-request-v1','config':config,'model':config['model'],'task_family':'forward','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'height':512,'width':512,'needs':{'cuda':True,'host':'beast'},'execution_environment':intent(),'config_bytes_sha256':hashlib.sha256(raw).hexdigest(),'source_hashes':{n:sha(p) for p,n in [(worker,worker.name),*extras]},'scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_operations=24,logical_branches=24,logical_checkpoint_replays=4,max_local_work_items=24,checkpoint_policy='24predictedterminalVAEdecodes4nativeEulerchecks;0denoiser/readout/Acts. Derivedinitializationsclearlylabelled.1residentload,serial,noretries.'),'resource_exception':{'ram_mb':10000,'vram_mb':13000,'disk_gb':1,'calibration_job':s['job_id'],'measured_peak_rss_mb':job['peak_rss_mb'],'measured_peak_vram_mb':job['peak_vram_mb'],'reason':'Same noencoder residentKleinBF16/VAE512B1;4cheapEulersteps24VAEdecodes. No fullweights/ports exported;24smallZarrays+sourceCAS. HDDscratch/cache.1GBrootadmissioncoverssmalltransport. Prior7675MBRSS9616MBVRAM;10GB/13GB.'}}
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras);return {'run_dir':str(run),'request_sha256':request['request_sha256']}


def submit(run):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    u=runpy.run_path(str(Path(__file__).with_name('flux_predicted_future_campaign.py')));r=read(run/'request.json')
    if r['execution_environment']!=intent() or (run/'submission.json').exists() or any(u['sha'](run/'payload'/n)!=h for n,h in r['source_hashes'].items()):raise ValueError('Stage differs or alreadyadmitted')
    if hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='request_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()!=r['request_sha256']:raise ValueError('Request seal differs')
    cmd=worker_command('flux_latent_neighborhood.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1');job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-latent-neighborhood',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],ram_mb=10000,vram_mb=13000,disk_gb=1,cpu_threads=4,est_wall_s=240,timeout_s=720,retry_on_kill=False,preflight=True,note='ShrinkbrokenwholeZswaps toalpha0,.05,.1,.25,.5,1.24VAEimages from24sealedEulerpredictions usingobservednativevelocity;4nativeEulerchecks at.1. 0transformerforwards/readouts/Acts;explicitderivedterminal/schedulerinitializations.8historicalPNGcontrols,8newpairedsignpredictionsstraightendpointinterpolation.1residentload,serial,noretries,HDDscratch10GBRAM13GBVRAM1GBdisk.')
    u['save'](run/'submission.json',{'job_id':str(job)});return {'job_id':str(job),'run_dir':str(run)}


if __name__=='__main__':
    if 'MRUN_JOB_ID' in os.environ:worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--source',type=Path);p.add_argument('--submit',action='store_true');p.add_argument('--collect');a=p.parse_args()
        if a.collect:r=runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')))['collect'](a.run.resolve(),a.collect)
        elif a.submit:r=submit(a.run.resolve())
        else:r=stage(a.run.resolve(),a.source.resolve())
        print(json.dumps(r,sort_keys=True))
