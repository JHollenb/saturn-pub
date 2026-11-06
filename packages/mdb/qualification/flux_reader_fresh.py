"""Fresh-seed reader predictions with crossed color, shape and spatial contexts."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
H=runpy.run_path(str(HERE/'flux_source_receiver.py'))
read,save,sha=[H[k] for k in ('read','save','sha')]


def semantic_masks(rgb):
    """Baseline-only color masks; shape identity requires separate visual checking."""
    import numpy as np
    x=np.asarray(rgb,np.float64);r,g,b=[x[...,i] for i in range(3)]
    return dict(green=(g>40)&(g>1.15*r)&(g>1.20*b),
        yellow=(r>50)&(g>50)&(r>.8*g)&(g>.8*r)&(r>1.30*b)&(g>1.30*b))


def stage(run,prior,origin_local,factorial,campaign=None):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    old=read(prior/'request.json');c=dict(old['config'])
    binding=read(origin_local/'source-binding.json');inventory=read(origin_local/'files.json')
    origin_parents={r:binding['origins'][r]['cut'] for r in ('c0-o0','c0-o1')}
    inputs={}
    def admit(n):
        p=origin_local/n
        if sha(p)!=inventory[n]:raise ValueError('origin input changed')
        inputs[n]=dict(sha256=sha(p),bytes=p.stat().st_size)
    for fp in origin_parents.values():
        n='store/cuts/'+fp+'.json';admit(n);cut=read(origin_local/n)
        for page in cut['slots'].values():admit('store/pages/'+page+'.json')
        for blob in cut['physical_blobs']:admit('store/blobs/'+blob+'.safetensors')
    admit('source-binding.json')
    source=read(factorial/'source.json');finventory=read(factorial/'files.json')
    conditioning={};extras=[]
    run.mkdir(parents=True,exist_ok=False)
    for r in ('c1-o0','c1-o1'):
        a=binding['origins'][r]['inputs']['conditioning'];p=factorial/a['path']
        if sha(p)!=a['sha256'] or sha(p)!=finventory[a['path']]:raise ValueError('conditioning inventory differs')
        name='conditioning/'+r+'.npy';target=run/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
        conditioning[r]={**a,'path':name};extras.append((target,name))
    c.update(schema='mdb-reader-fresh-plan-v1',seed=193,conditioning=conditioning,origin_parents=origin_parents,
        input_root='/mnt/hdd8tb/mdb/research/job-3272cb7fc922/binding-minimal-result',
        source_files_sha256=sha(origin_local/'files.json'),inputs=inputs,parents={},max_wall_seconds=600,
        conditions=[{**x,'seed':193} for x in source['conditions']],
        case_specs=[dict(id='base',kind='base'),dict(id='band2',kind='band-cross',band=2),
            dict(id='band1',kind='band-cross',band=1),dict(id='own-v',kind='own-v',head=7,scale=1.0625)],
        semantic_hypotheses=dict(stage='exploratory competing predictions, frozen before target suffixes',
            cone='Band2 RGB effect energy per baseline-mask pixel is larger on cone than cylinder, independent of assigned color/order.',
            green='Band2 RGB effect energy per baseline-mask pixel is larger on green than yellow, independent of shape/order.',
            spatial='Band2 RGB effect energy is larger in the left than right image half, independent of color/shape.',
            direction='Color-swap source decreases green score on originally green object and increases it on originally yellow object.',
            controls='Band1 is an adjacent source-row control; own-v scales only own image values for head7, donor-free.',
            score='green_score=(G-R)/(G+R+1); squared RGB delta energy both total and per baseline mask pixel; all signed changes retained.',
            masks='Baseline-only: green G>40,G>1.15R,G>1.20B; yellow R,G>50,R>.8G,G>.8R,R,G>1.30B. Empty masks are instrument-incomplete, not a scientific null.',
            caveat='Prompt color-to-shape assignment requires visual baseline conformance. Pixel response can reflect VAE/color sensitivity rather than a discrete concept.'),
        scope='Fresh seed193, four cached prompt conditions crossed color/order, sixteen sealed QKV-to-text forecasts then native futures. Four native prefixes of85steps each are executed before forecasting and counted; no raw-prompt or full-image numerical forecast. Color-swapped K/V band sources and an own-state no-donor V intervention; shape/color hypotheses exploratory.',
        planned_counts=dict(sdpa=16,staged_matmuls=0,staged_softmax=0,linear=48,layernorm=16,swiglu=16,
            native_sdpa=700,native_blocks=700,native_transitions=772,images=16,model_loads=1,encodes=0,noise_draws=1,prefix_stops=4))
    if campaign:
        allowed={'seed','conditions','case_specs','semantic_hypotheses','panel','reconstruct_values','selection','scope','new_prompt_encoding'}
        if set(campaign)-allowed:raise ValueError('unknown campaign override')
        c.update(campaign)
        if {x['id'] for x in c['conditions']}!={'c0-o0','c0-o1','c1-o0','c1-o1'}:raise ValueError('four matched contexts required')
        c['conditions']=[{**x,'seed':c['seed']} for x in c['conditions']]
        image_kinds={'value-image','image-head-add','image-head-remove'}
        nspec=len(c['case_specs']);n=4*nspec;formed=4 if c.get('reconstruct_values') else 0
        flashes=4*(1+sum(s['kind']!='base' and s['kind'] not in image_kinds for s in c['case_specs']))
        c['planned_counts']=dict(sdpa=flashes,staged_matmuls=0,staged_softmax=0,linear=3*n+formed,layernorm=n+formed,swiglu=n,
            native_sdpa=300+25*n+4*sum(s['kind'] in image_kinds|{'value-text'} for s in c['case_specs']),native_blocks=300+25*n,native_transitions=340+27*n,
            images=n,model_loads=1,encodes=4 if c.get('new_prompt_encoding') else 0,noise_draws=1,prefix_stops=4)
        if formed:
            c['planned_counts']['value_formations']=4
            manifest=read(prior/'parameter-artifacts.json')
            c['value_parameter_sha256']=next(a['source']['payload_sha256'] for a in manifest['artifacts'] if a['source']['tensor']=='transformer_blocks.0.attn.to_v.weight')
        if c.get('new_prompt_encoding'):
            c['checkpoint_files']={k:v for k,v in source['checkpoint_files'].items() if k.startswith(('text_encoder/','tokenizer/'))}
    save(run/'plan.json',c)
    for n in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):
        shutil.copyfile(prior/n,run/n);extras.append((run/n,n))
    names=('flux_reader_route_future.py','flux_reader_routes.py','flux_reader_arithmetic.py','flux_reader_attention.py',
        'flux_source_receiver.py','flux_reentry_reference.py','flux_family_custody.py','flux_latent_reader.py')
    if c.get('new_prompt_encoding'):names=(*names,'retrieve_flux_encoder.py')
    extras.extend((HERE/n,n) for n in names)
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],
        path_token_allowlist=[c['input_root'],c['model_path'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    req=dict(schema='mdb-reader-fresh-request-v1',worker_script=Path(__file__).name,config=c,model=c['model'],
        execution_environment=old['execution_environment'],needs=dict(cuda=True,host='beast'),task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,
        source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},
        resource_exception=dict(ram_mb=10240,vram_mb=10240,cpu_threads=4,disk_gb=2,est_wall_s=240,
            calibration_job='job-4c4cf4ac3624',measured_peak_rss_mb=8209.8,measured_peak_vram_mb=9264.,
            reason='Same resident model/16scalar suffixes measured98.26s/8209.8MBRSS/9264MBVRAM; add300prefixblocks to400suffixblocks, four cached contexts and operand arrays.10GB RAM/VRAM,240s estimate,600s worker/900s outer; no duplicate weights or encoders.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=36,logical_branches=16,logical_checkpoint_replays=16,max_local_work_items=64,
            checkpoint_policy='One resident model;4fresh prefixes,16sealed reader forecasts,16suffixes.772successful native steps/700blocks+4intentional prefix stops,16images;1shared noise draw,0encodes/retries.'))
    if campaign:
        n=4*len(c['case_specs'])
        req['resource_exception'].update(ram_mb=12288 if c.get('new_prompt_encoding') else 10240,disk_gb=4,est_wall_s=360,
            calibration_job='job-6a35eb9ce352',measured_peak_rss_mb=8889.3,measured_peak_vram_mb=9372.,
            reason=f'Fresh80-future job-6a35eb9ce352 measured257.69s/8889.3MBRSS/9372MBVRAM with four new encodes/prefixes. Current{n}suffixes, sequential original encoder then transformer.12GBRAM/10GBVRAM;360s estimate/600s worker. No duplicated per-case KV retention.')
        req['scheduling_contract']=one_lease_scheduling_contract(model_loads=1,logical_operations=4+2*n,logical_branches=n,logical_checkpoint_replays=n,max_local_work_items=4+2*n,
            checkpoint_policy=f'One pipeline acquisition;4prefixes and {n} sealed forecasts then {n} suffixes. Original encoder sequential then released if requested; own image-to-V formation separately counted; no retries.')
    req['request_sha256']=hashlib.sha256(json.dumps(req,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',req)
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    stage_worker(run/'payload',worker=Path(__file__),config=c,request=req,additional_files=extras)
    return dict(request_sha256=req['request_sha256'],seed=c['seed'],cases=4*len(c['case_specs']))


def worker():
    boot=time.monotonic()
    import faulthandler
    import importlib.metadata
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');faulthandler.dump_traceback_later(60,repeat=True)
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=read(os.environ['SATURN_RUN_CONFIG']);source=Path(c['input_root'])
    if req['config']!=c or any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('source/config changed')
    if sha(source/'files.json')!=c['source_files_sha256'] or any(sha(source/n)!=r['sha256'] or (source/n).stat().st_size!=r['bytes'] for n,r in c['inputs'].items()):raise ValueError('origin inventory changed')
    for n,v in c['original_source_sha256'].items():
        if sha(Path(importlib.metadata.distribution('diffusers').locate_file(n)))!=v:raise ValueError('original code changed')
    core=runpy.run_path('flux_reader_arithmetic.py');core['H']['RecipeBank'](read('parameter-artifacts.json'),c['model_path'])
    import diffusers.models.transformers.transformer_flux2 as native
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.flux_final_latent_closure import FluxFinalLatentClosureAdapter
    from mdb.native_navigation import NativeCutNavigator
    from mdb.tensor_archive import load_tensor_archive
    torch.set_num_threads(4);core['configure_eager_determinism']();torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    print('MDB_READER_FRESH loading original transformer/VAE',flush=True)
    if c.get('new_prompt_encoding'):
        import gc
        import inspect
        if any(sha(Path(c['model_path'])/n)!=v for n,v in c['checkpoint_files'].items()):raise ValueError('original encoder/tokenizer checkpoint differs')
        pipe=Flux2KleinPipeline.from_pretrained(c['model_path'],torch_dtype=torch.bfloat16,local_files_only=True)
        pipe.text_encoder.eval().requires_grad_(False).to('cuda');encodings={}
        encoder_source=dict(path=inspect.getfile(type(pipe.text_encoder)),sha256=sha(Path(inspect.getfile(type(pipe.text_encoder)))))
        with torch.inference_mode():
            for condition in c['conditions']:
                value,ids=pipe.encode_prompt(condition['prompt'],device=torch.device('cuda'),num_images_per_prompt=1,max_sequence_length=512)
                encodings[condition['id']]=(value.cpu(),ids.cpu())
        pipe.text_encoder=None;pipe.tokenizer=None;del value,ids;gc.collect();torch.cuda.empty_cache()
    else:pipe=Flux2KleinPipeline.from_pretrained(c['model_path'],torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None)
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=FluxFinalLatentClosureAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    origins={r:LocalStore(source/'store').load(fp,device='cuda') for r,fp in c['origin_parents'].items()}
    if any(p.model_identity!=adapter.model_identity or dict(p.execution)!=dict(adapter.execution) for p in origins.values()):raise ValueError('origin model/backend differs')
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False);store=LocalStore(out/'store')
    if c.get('new_prompt_encoding'):
        save(out/'encoder-source.json',dict(implementation=encoder_source,checkpoint_files=c['checkpoint_files'],prompts=c['conditions'],encodes=4))
        shutil.copyfile('retrieve_flux_encoder.py',out/'retrieve_encoder.py')
    started=time.monotonic()
    def event(kind,**fields):
        row=dict(kind=kind,seconds=time.monotonic()-started,**fields)
        with (out/'timeline.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_READER_FRESH '+json.dumps(row,sort_keys=True),flush=True)
        if row['seconds']>c['max_wall_seconds']:raise RuntimeError('finite fresh campaign budget exceeded')
    counts=dict(native_transitions=0,native_sdpa=0,native_blocks=0,prefix_stops=0,noise_draws=1)
    if c.get('new_prompt_encoding'):counts['encodes']=4
    reader=runpy.run_path('flux_latent_reader.py');active=dict(mode='continuation',text_rows=512);parents={};prefixes={};prefix_rows={}
    handles=[b.register_forward_hook(lambda m,a,v:counts.__setitem__('native_blocks',counts['native_blocks']+1)) for b in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    with torch.inference_mode():
        latent,img_ids=pipe.prepare_latents(1,pipe.transformer.config.in_channels//4,512,512,torch.bfloat16,torch.device('cuda'),torch.Generator(device='cuda').manual_seed(c['seed']))
        def receipt(value,case):
            with (out/'prefix-receipts.jsonl').open('a') as f:f.write(json.dumps(dict(case=case,receipt=value.to_dict()),sort_keys=True)+'\n')
        try:
            with reader['ReaderBoundary'](native,counts,active) as boundary:
                for condition in c['conditions']:
                    r=condition['id'];origin=origins['c0-'+r.split('-')[1]].payload
                    if c.get('new_prompt_encoding'):conditioning,txt_ids=[v.to('cuda') for v in encodings[r]]
                    else:
                        conditioning=(origin['conditioning'] if r.startswith('c0-') else torch.from_numpy(load_tensor_archive(Path.cwd(),c['conditioning'][r]).copy()).to('cuda',torch.bfloat16));txt_ids=origin['txt_ids'].clone()
                    session=adapter.session(latent=latent.clone(),conditioning=conditioning,img_ids=img_ids.clone(),txt_ids=txt_ids,timesteps=origin['timesteps'].clone(),sigmas=origin['sigmas'].clone())
                    initial=session.capture(retain=False);store.save(initial);nav=NativeCutNavigator(session,max_steps=86);active['mode']='continuation'
                    for _ in range(85):receipt(nav.step(),r);counts['native_transitions']+=1
                    parent=session.capture(retain=False);point=dict(parent.execution_point)
                    if (point['logical_step'],point['phase'],point['index'])!=(3,'joint',0):raise ValueError('fresh prefix clock differs')
                    store.save(parent);parents[r]=parent;active['mode']='prefix';boundary.captured=None
                    try:nav.step()
                    except reader['PrefixStop']:pass
                    else:raise ValueError('prefix failed to stop')
                    if boundary.captured is None or describe(session.capture(retain=False).payload)!=describe(parent.payload):raise ValueError('prefix stop mutated parent')
                    receipt(session.attempts[-1],r);prefixes[r]=boundary.captured
                    prefix_rows[r]=dict(parent=parent.fingerprint,initial=initial.fingerprint,initial_latent=describe(initial.payload['latent']),conditioning=describe(conditioning),point=point)
                    event('fresh-prefix-captured',case=r,counts=counts.copy())
        finally:
            for handle in handles:handle.remove()
        expected=dict(native_transitions=340,native_sdpa=300,native_blocks=300,prefix_stops=4,noise_draws=1)
        if c.get('new_prompt_encoding'):expected['encodes']=4
        if counts!=expected:raise ValueError('fresh prefix census differs')
        operand_records={}
        for r in parents:
            other=('c1-' if r.startswith('c0-') else 'c0-')+r.split('-')[1]
            for role,key in (('base',r),('delta',other)):operand_records[r+'.'+role]={f:H['array'](out,v) for f,v in prefixes[key].items()}
        save(out/'prefixes.json',dict(operands=operand_records,counts=counts.copy()));save(out/'fresh-parents.json',prefix_rows)
        save(out/'plan.json',c);save(out/'request.json',req)
        save(out/'native-contract.json',dict(device=torch.cuda.get_device_name(),dtype='bfloat16',torch_version=torch.__version__,cuda_version=torch.version.cuda,
            bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,prefix_native_steps=340,prefix_native_blocks=300,
            boundary='Fresh native positioned text Q/full KV; predictions begin after earlier native prefixes, before target attention.'))
        program=runpy.run_path('flux_reader_route_future.py')
        counts=program['run'](pipe=pipe,adapter=adapter,parents=parents,source=out,output=out,config=c,request=req,event=event,initial_counts=counts)
    for n in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):shutil.copyfile(n,out/n)
    save(out/'summary.json',dict(schema='mdb-reader-fresh-summary-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),counts=counts,seconds=time.monotonic()-started,
        mechanics_status='complete',terminal_status='unassessed',bootstrap_seconds=started-boot,total_worker_seconds=time.monotonic()-boot,scope=c['scope']))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)
    faulthandler.cancel_dump_traceback_later()


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    req=read(run/'request.json');runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=v for n,v in req['source_hashes'].items()):raise ValueError('changed/already submitted run')
    cmd=worker_command(Path(__file__).name);cmd.insert(cmd.index('python')+1,'-B');os.environ.setdefault('MRUN_URL','http://192.168.68.68:9025')
    resources={k:req['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')}
    job=guarded_mrun_launch(cmd,request=req,model=req['model'],task='forward',experiment='mdb-reader-fresh',payload=run/'payload',gpu=True,needs=req['needs'],prefer_host='beast',timeout_s=900,preflight=True,retry_on_kill=False,**resources,
        note=f"Fresh seed{req['config']['seed']}/color-order2x2. One pipeline, encoder released before transformer. Four prefixes/340steps, then {req['config']['planned_counts']['images']} sealed reader forecasts and suffixes. {req['config']['planned_counts']['native_blocks']} nativeblocks, {req['config']['planned_counts']['encodes']} original encodes,1shared noise draw,0retries.600sworker/900souter; earlier native prefix explicitly counted.")
    save(run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--origin-local',type=Path);p.add_argument('--factorial',type=Path);p.add_argument('--campaign',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.prior.resolve(),a.origin_local.resolve(),a.factorial.resolve(),read(a.campaign) if a.campaign else None)))
