"""Head and image-key-row restoration through the qualified native reader."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def mix_heads(base, alternate, selected, heads=24):
    if base.shape!=alternate.shape or base.shape[-1]%heads or any(type(i)is not int or not 0<=i<heads for i in selected):
        raise ValueError('head recipe geometry differs')
    result=base.clone();width=base.shape[-1]//heads
    for i in selected:result[...,i*width:(i+1)*width]=alternate[...,i*width:(i+1)*width]
    return result


def metrics(base, target, value):
    import numpy as np
    b,d,p=[np.asarray(x,np.float64) for x in (base,target,value)]
    effect,actual=p-b,d-b;den=float(np.vdot(actual,actual))
    if den==0:raise ValueError('native reference contrast is empty')
    dot=float(np.vdot(effect,actual));norm=float(np.vdot(effect,effect))
    return dict(target_fraction=dot/den,removed_fraction=1-dot/den,
        relative_error=float(np.linalg.norm(effect-actual)/np.sqrt(den)),
        effect_relative_norm=float(np.sqrt(norm/den)),cosine=float(dot/np.sqrt(norm*den)) if norm else None,
        changed_elements=int(np.count_nonzero(effect)),elements=effect.size)


def summarize(rows, root, load):
    import numpy as np
    base=load(root,rows['base']['text']);target=load(root,rows['full']['text'])
    measures={key:metrics(base,target,load(root,row['text'])) for key,row in rows.items()}
    def order(prefix,field):return sorted(range(24 if 'head' in prefix else 4),key=lambda i:(-measures[f'{prefix}{i}'][field],i))
    return dict(measures=measures,solo_heads=order('head-add-','target_fraction'),
        removed_heads=order('head-remove-','removed_fraction'),solo_bands=order('band-add-','target_fraction'),
        removed_bands=order('band-remove-','removed_fraction'),
        solo_signs=[int(np.sign(measures[f'head-add-{i}']['target_fraction'])) for i in range(24)])


def worker():
    import faulthandler
    import importlib.metadata
    boot=time.monotonic();faulthandler.dump_traceback_later(60,repeat=True)
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    core=runpy.run_path('flux_reader_arithmetic.py');a=runpy.run_path('flux_reader_attention.py');h=core['H']
    read,save,sha,load=[h[k] for k in ('read','save','sha','load_array')]
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=read(os.environ['SATURN_RUN_CONFIG'])
    if req['config']!=c or any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('payload/config differs')
    import torch
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    torch.set_num_threads(4);core['configure_eager_determinism']()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=Path(c['input_root'])
    for n,r in c['inputs'].items():
        if sha(source/n)!=r['sha256'] or (source/n).stat().st_size!=r['bytes']:raise ValueError('input differs')
    for name,v in c['original_source_sha256'].items():
        if sha(Path(importlib.metadata.distribution('diffusers').locate_file(name)))!=v:raise ValueError('native source differs')
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',c);save(out/'request.json',req)
    save(out/'native-contract.json',dict(device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        tf32=False,deterministic=True,dtype='bfloat16',text_shape=[1,512,3072],query_shape=[1,512,24,128],
        source_sha256=c['original_source_sha256'],kernel='Forced FlashAttention; selected ATen operations recorded for each QKV call'))
    bank=h['RecipeBank'](read('parameter-artifacts.json'),c['model_path'])
    weights={k:bank.load(v) for k,v in c['text_parameters'].items()};modules=core['make_modules'](weights,device='cuda',eps=c['eps']);del weights
    parents={r:LocalStore(source/'store').load(fp).payload for r,fp in c['parents'].items()}
    prefixes=read(source/'prefixes.json')['operands'];original=read(source/'readers.json')
    counts=dict(sdpa=0,staged_matmuls=0,staged_softmax=0,linear=0,layernorm=0,swiglu=0,full_models=0,blocks=0,images=0)
    allowed=set(modules['ff'].modules())|{modules['projection'],modules['norm']}
    def guard(m,args):
        if m not in allowed:raise ValueError('undeclared native module')
    handle=torch.nn.modules.module.register_module_forward_pre_hook(guard)
    started=time.monotonic();all_rows={};summaries={};profiles={}
    tensor=lambda x:torch.from_numpy(x.copy()).to('cuda',torch.bfloat16)
    def event(kind,**fields):
        row=dict(kind=kind,seconds=time.monotonic()-started,counts=counts.copy(),**fields)
        with (out/'timeline.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_READER_ROUTES '+json.dumps(row,sort_keys=True),flush=True)
        if row['seconds']>c['max_wall_seconds']:raise RuntimeError('finite route budget exceeded')
    def run_receiver(receiver):
        b,d=[{f:tensor(load(source,r)) for f,r in prefixes[receiver+'.'+n].items()} for n in ('base','delta')]
        nt=b['q'].shape[1]
        if not torch.equal(b['q'],d['q']) or not torch.equal(b['k'][:,:nt],d['k'][:,:nt]) or not torch.equal(b['v'][:,:nt],d['v'][:,:nt]):
            raise ValueError('text operand preservation differs')
        H0,p0=a['attention'](b['q'],b['k'],b['v'],'flash-text',counts)
        H1,p1=a['attention'](b['q'],d['k'],d['v'],'flash-text',counts)
        profiles[receiver]=dict(base=p0,full=p1)
        head_records=dict(base=h['array'](out,H0),full=h['array'](out,H1))
        parent=parents[receiver];text=parent['text'].to('cuda',torch.bfloat16);mod=parent['context']['txt_mod'].to('cuda',torch.bfloat16)
        rows={}
        def retain(name,head,recipe):
            values=core['native_tail'](head,text,mod,modules,counts=counts)
            # Head mixtures are losslessly encoded by two source archives and a
            # list of selected head IDs; no redundant 3MB input per head arm.
            digest=hashlib.sha256(head.view(torch.uint16).cpu().numpy().astype('<u2',copy=False).tobytes()).hexdigest()
            rows[name]=dict(head_recipe=recipe,head_payload_sha256=digest,text=h['array'](out,values['text']))
            save(out/'rows'/(receiver+'.'+name+'.json'),rows[name]);del values
        retain('base',H0,dict(kind='archive',archive=head_records['base']))
        retain('full',H1,dict(kind='archive',archive=head_records['full']))
        for i in range(24):
            for name,base,alternate in [('head-add-',H0,H1),('head-remove-',H1,H0)]:
                recipe=dict(kind='head-mixture',base=head_records['base' if name=='head-add-' else 'full'],
                    alternate=head_records['full' if name=='head-add-' else 'base'],heads=[i],heads_total=24)
                retain(name+str(i),mix_heads(base,alternate,[i]),recipe)
        for i in range(4):
            lo,hi=nt+i*256,nt+(i+1)*256
            for mode in ('add','remove'):
                bk,bv=(b['k'],b['v']) if mode=='add' else (d['k'],d['v'])
                ak,av=(d['k'],d['v']) if mode=='add' else (b['k'],b['v'])
                k,v=bk.clone(),bv.clone();k[:,lo:hi]=ak[:,lo:hi];v[:,lo:hi]=av[:,lo:hi]
                head,profile=a['attention'](b['q'],k,v,'flash-text',counts)
                name=f'band-{mode}-{i}';profiles[receiver][name]=profile
                retain(name,head,dict(kind='archive',archive=h['array'](out,head),source_band=dict(start=lo,stop=hi,mode=mode)))
        all_rows[receiver]=rows;summaries[receiver]=summarize(rows,out,load)
        save(out/(receiver+'-summary.json'),summaries[receiver]);event('receiver-complete',receiver=receiver)
        return summaries[receiver]
    try:
        with torch.inference_mode():
            left=run_receiver('left')
            tx=ForecastTransaction(out/'transaction',model_identity='flux2-klein-4b-distilled',input_identity=c['parents'],
                source_identity=req['request_sha256'],frontier=dict(kind='architectural',boundary='left screen complete; right interventional reader not executed',step=3,block=0),
                execution_scope='Predict right head/band rankings and signs from left only. Base/full right source endpoints are historical known states; right head/band interventions are newly executed. Physical primitives counted separately from one transaction callback.',
                budgets=dict(max_bytes=4<<20,max_wall_ms=180000,max_native_calls=1))
            tx.frontier('left',left)
            tx.predict('routes',lambda ctx:dict(predictions={k:ctx.frontiers['left'][k] for k in ('solo_heads','removed_heads','solo_bands','removed_bands','solo_signs')}))
            seal=tx.seal();verify_transaction(out/'transaction',expected_seal_sha256=seal['seal_sha256']);event('right-prediction-sealed',seal_sha256=seal['seal_sha256'])
            tx.execute('right',lambda _:run_receiver('right'));verify_transaction(out/'transaction',expected_seal_sha256=seal['seal_sha256'])
            save(out/'transaction-report.json',tx.report());tx.close()
    finally:handle.remove()
    if counts!={**counts,'sdpa':20,'linear':348,'layernorm':116,'swiglu':116}:raise ValueError('route primitive census differs')
    save(out/'cases.json',all_rows);save(out/'profiles.json',profiles)
    save(out/'source-anchors.json',{r:{n:original[r+'.'+k]['fields'] for n,k in [('base','k0v0'),('full','k1v1')]} for r in ('left','right')})
    for n in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):shutil.copyfile(n,out/n)
    save(out/'summary.json',dict(schema='mdb-reader-route-screen-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),
        counts=counts,seconds=time.monotonic()-started,bootstrap_seconds=started-boot,mechanics_status='complete',terminal_status='unassessed',scope=c['scope']))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)
    faulthandler.cancel_dump_traceback_later()


def stage(run,prior):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_source_receiver.py'));read,save,sha=h['read'],h['save'],h['sha']
    old=read(prior/'request.json');c=dict(old['config']);c.pop('resource_exception');c.pop('planned_counts')
    c.update(schema='mdb-reader-route-plan-v1',max_wall_seconds=180,
        scope='Two known receivers, new single-head addition/removal and four source-image-row-band addition/removal reader states. Left screen predicts right ranking/sign before right execution in sealed MDB transaction. Original model primitives and three matrices; no new seed/raw-prompt/general semantic or image-continuation claim.',
        planned_counts=dict(sdpa=20,staged_matmuls=0,staged_softmax=0,linear=348,layernorm=116,swiglu=116,full_models=0,blocks=0,images=0))
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',c)
    extras=[(here/n,n) for n in ('flux_reader_arithmetic.py','flux_reader_attention.py','flux_source_receiver.py','flux_reentry_reference.py','flux_family_custody.py')]
    for n in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):
        shutil.copyfile(prior/n,run/n);extras.append((run/n,n))
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],
        path_token_allowlist=[c['input_root'],c['model_path'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    req=dict(schema='mdb-reader-routes-request-v1',config=c,execution_environment=old['execution_environment'],needs=dict(cuda=True,host='beast'),
        task_family='analysis',backend='torch',device='cuda',dtype='bfloat16',source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},
        resource_exception=dict(ram_mb=3072,vram_mb=1536,cpu_threads=4,disk_gb=1,est_wall_s=120,
            calibration_job='job-fb2d2d4395fb',measured_peak_rss_mb=2252.7,measured_peak_vram_mb=824.,reason='Same original three-matrix reader;20ratherthan40SDPA,116ratherthan16serialtails;no staged FP32scorematrices. Bound onecaseactivation and streaming output at existing3GBRAM/1.5GBVRAM. Native operator timeprevious6s,120sestimate includes116outputs/custody/selector.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=116,logical_branches=1,logical_checkpoint_replays=0,
            max_local_work_items=128,extra=dict(task_family='analysis'),checkpoint_policy='One116case scalar route screen:20SDPA/348linear/116norm/116SwiGLU; left-to-right sealed ranking/sign predictions; no fullmodels/blocks/images/retries.'))
    req['request_sha256']=hashlib.sha256(json.dumps(req,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',req)
    runpy.run_path(str(here/'stage_flux_source_receiver.py'))['bind'](req['execution_environment']);stage_worker(run/'payload',worker=Path(__file__),config=c,request=req,additional_files=extras)
    return dict(request_sha256=req['request_sha256'])


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_source_receiver.py'));r=h['read'](run/'request.json')
    runpy.run_path(str(here/'stage_flux_source_receiver.py'))['bind'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('payload drift or repeated submit')
    cmd=worker_command(Path(__file__).name);cmd.insert(cmd.index('python')+1,'-B');os.environ.setdefault('MRUN_URL','http://192.168.68.68:9025')
    resource={k:r['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')}
    job=guarded_mrun_launch(cmd,request=r,model=None,task='analysis',experiment='mdb-flux-reader-routes',payload=run/'payload',gpu=True,needs=r['needs'],prefer_host='beast',timeout_s=600,
        retry_on_kill=False,preflight=True,**resource,note='116same-parent reader states:24head add/24remove+4rowbands add/4remove+2anchors perreceiver.20FlashSDPA/348linear/116LN/116SwiGLU. Left ranking/sign frozen in MDB before rightinterventions.0fullmodels/blocks/images.120sestimate/180sworker/600souter, measured3GBRAM/1.5GBVRAM,no retries.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.prior.resolve())))
