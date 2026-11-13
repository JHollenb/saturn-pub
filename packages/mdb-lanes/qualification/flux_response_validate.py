"""Consume immutable forecasts from a terminal partial run; never recalculate them."""
import argparse
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def expected_counts():
    return dict(loads=1,encodes=0,noise_draws=0,native_transitions=481,acts=18,receipts=40,images=9)


def steps_to_commit(point,joint_blocks,single_blocks):
    phase=point['phase'];index=point['index']
    if phase=='project':return 1+joint_blocks+single_blocks+2
    if phase=='joint'and type(index)is int and 0<=index<joint_blocks:return joint_blocks-index+single_blocks+2
    if phase=='single'and type(index)is int and 0<=index<single_blocks:return single_blocks-index+2
    if phase=='readout':return 2
    if phase=='scheduler':return 1
    raise ValueError('unsupported commit frontier')


def frozen_source(source,seal_sha256):
    from collections.abc import Mapping
    from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
    def tree(v):
        if isinstance(v,Mapping):return {k:tree(x)for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [tree(x)for x in v]
        return v
    read=lambda p:json.loads(p.read_text())
    tx=verify_transaction(source/'transaction',expected_seal_sha256=seal_sha256)
    frozen=tree(load_forecast_tree(source/'transaction',read(source/'transaction/forecasts/response_program.json')['predictions']))
    if not tx['verified']or frozen['forecasts']!=read(source/'forecasts.json')or frozen['entries_at_seal']!={'blocks':0,'target_blocks':0}:raise ValueError('original prediction quarantine differs')
    for case in frozen['forecasts'].values():
        records=[*case['input_fields'].values(),*[r for fields in case['fields'].values()for r in fields.values()]]
        for record in records:
            import hashlib
            p=source/record['path']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('forecast array differs')
    return tx,frozen


def worker():
    payload=Path.cwd();h=runpy.run_path(str(payload/'flux_binding_minimal.py'));sha,save=[h[k]for k in('sha','save')]
    config,request=[json.loads(Path(os.environ[k]).read_text())for k in('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config']or any(sha(payload/n)!=v for n,v in request['source_hashes'].items()):raise ValueError('sealed continuation changed')
    for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    out=Path('binding-minimal-result');out.mkdir();save(out/'plan.json',config);save(out/'request.json',request)
    started=time.monotonic()
    for name,record in config['input_files'].items():
        p=Path(config['input_root'])/name
        if p.stat().st_size!=record['bytes']or sha(p)!=record['sha256']:raise ValueError('frozen source differs: '+name)
        q=out/'source'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act,Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.native_navigation import NativeCutNavigator
    source=out/'source';read=lambda p:json.loads(p.read_text());forecasts=read(source/'forecasts.json')
    tx,frozen=frozen_source(source,config['forecast_seal_sha256'])
    if frozen['config']['cases']!=config['cases']or frozen['config']['parents']!=config['parents']or frozen['config']['checkpoint_sha256']!=config['checkpoint_sha256']:raise ValueError('original specimen differs')
    # The original seal predates every native target. The original transaction
    # remains released after timeout; continuation does not rewrite that history.
    seal_events=[json.loads(line)for line in (source/'timeline.jsonl').read_text().splitlines()if json.loads(line)['kind']=='sealed']
    if len(seal_events)!=1 or seal_events[0]['entries']!={'blocks':0,'target_blocks':0}:raise ValueError('forecast target quarantine differs')
    counts=dict.fromkeys(expected_counts(),0);entries=dict(blocks=0,target_blocks=0);active=dict(branch='setup',at_target=False);receipts=[];rows=[];target_counts={s['id']:0 for s in config['cases']}
    def event(kind,**kw):
        v=dict(kind=kind,seconds=time.monotonic()-started,counts=dict(counts),entries=dict(entries),**kw)
        with(out/'timeline.jsonl').open('a')as f:f.write(json.dumps(v,sort_keys=True)+'\n')
        print('MDB_RESPONSE '+json.dumps(v,sort_keys=True),flush=True)
    def wall():
        if time.monotonic()-started>config['worker_wall_s']:raise RuntimeError('finite validation wall budget')
    def array(name,value):
        p=out/'arrays'/(name+'.npy');p.parent.mkdir(exist_ok=True);value=value.detach().float().cpu().numpy()
        np.save(p,value,allow_pickle=False);return dict(path=p.relative_to(out).as_posix(),sha256=sha(p),shape=list(value.shape),dtype='float32')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise ValueError('CUDA admission required')
    root=Path(config['model_path']);inventory={p.relative_to(root).as_posix():sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.suffix in{'.safetensors','.json','.txt','.model','.tiktoken'}}
    if h['identity'](inventory)!=config['checkpoint_sha256']:raise ValueError('checkpoint identity differs')
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);pipe.set_progress_bar_config(disable=True);counts['loads']=1
    for c in(pipe.transformer,pipe.vae):c.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    def entered(module,args):
        entries['blocks']+=1
        if active['at_target']:entries['target_blocks']+=1;target_counts[active['case']]+=1
    handles=[b.register_forward_pre_hook(entered)for b in[*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    original=LocalStore(source/'store');store=LocalStore(out/'store');base=original.load(config['parents']['base'],device='cuda');control_exit=original.load(config['control_exit'],device='cuda')
    for c in(base,control_exit):
        if c.model_identity!=adapter.model_identity:raise ValueError('model identity differs')
        store.save(c)
    if any(base.execution_point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=1).items())or any(control_exit.execution_point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=2).items()):raise ValueError('source clock differs')
    if any(describe(base.payload[k])!=describe(control_exit.payload[k])for k in base.payload if k not in{'image','text','cursor'}):raise ValueError('clock scaffold changed closure')
    def receipt(value):
        wall();body=value.to_dict();store.receipt(value);record=dict(branch=active['branch'],body=body);receipts.append(record)
        with(out/'receipts.jsonl').open('a')as f:f.write(json.dumps(record,sort_keys=True)+'\n')
        counts['receipts']+=1;counts['native_transitions'if body['operation']=='continue'else'acts']+=body.get('steps',1)
        if any(counts[k]>expected_counts()[k]for k in('native_transitions','acts','receipts')):raise RuntimeError('physical budget exceeded')
    def capture(n):c=n.session.capture(retain=False);store.save(c);return c
    def apply(n,slot,value,label):
        before=capture(n);rec=n.apply(Act.replace(slot,value,name=label));after=capture(n)
        if describe(after.payload)!=describe({**before.payload,slot:value}):raise ValueError('undeclared mutation')
        return dict(slot=slot,before=before.fingerprint,after=after.fingerprint,receipt=rec.to_dict())
    def tensor(record):return torch.from_numpy(np.load(source/record['path'],allow_pickle=False).copy()).to(device='cuda',dtype=torch.bfloat16)
    def finish(n,name,mode,actions,entry,**extra):
        commits=[]
        while n.point['logical_step']<4:
            point=n.point;n.step(steps_to_commit(point,len(pipe.transformer.transformer_blocks),len(pipe.transformer.single_transformer_blocks)))
            if n.point['logical_step']!=point['logical_step']+1:raise ValueError('group did not end at next commit')
            c=capture(n);commits.append(dict(step=n.point['logical_step'],cut=c.fingerprint))
        final=capture(n);p=out/(name+'.'+mode+'.png');adapter.decode(n.session,height=512,width=512).save(p);counts['images']+=1
        row=dict(id=name,mode=mode,entry=entry,actions=actions,commits=commits,final=final.fingerprint,png=p.name,png_sha256=sha(p),**extra)
        rows.append(row);save(out/('completed-'+name+'.'+mode+'.json'),row);event('image_complete',case=name,mode=mode)
    def predicted(spec):
        name=spec['id'];active.update(branch=name+'.predicted',case=name)
        n=NativeCutNavigator(Session.from_cut(adapter,control_exit),max_steps=53,on_receipt=receipt)
        actions=[apply(n,k,tensor(forecasts[name]['fields'][k]['exit']),name+'.predicted.'+k)for k in('image','text')]
        finish(n,name,'predicted',actions,capture(n).fingerprint,clock_scaffold=control_exit.fingerprint,target_entries_before_image=target_counts[name],forecast_source_job=config['source_job'])
    block=pipe.transformer.transformer_blocks[1]
    def native(spec):
        name=spec['id'];active.update(branch=name+'.native',case=name)
        n=NativeCutNavigator(Session.from_cut(adapter,base),max_steps=54,on_receipt=receipt)
        actions=[apply(n,k,tensor(forecasts[name]['input_fields'][k]),name+'.input.'+k)for k in('image','text')];entry=capture(n);ports={}
        def ah(m,args,output):ports.update(attention_image=output[0].detach().clone(),attention_text=output[1].detach().clone())
        def fh(role):
            def hook(m,args,output):ports['mlp_input_'+role]=args[0].detach().clone();ports['mlp_'+role]=output.detach().clone()
            return hook
        hs=[block.attn.register_forward_hook(ah),block.ff.register_forward_hook(fh('image')),block.ff_context.register_forward_hook(fh('text'))]
        active['at_target']=True
        try:n.step()
        finally:
            active['at_target']=False
            for hook in hs:hook.remove()
        exit=capture(n);record=dict(entry=entry.fingerprint,exit=exit.fingerprint,fields={k:array(name+'.native.'+k,v)for k,v in ports.items()});save(out/('ports-'+name+'.json'),record)
        finish(n,name,'native',actions,entry.fingerprint,target=record)
    event('prepared_validation_started',original_transaction=tx,source_job=config['source_job'],source_state='timeout',new_predictions=0)
    predicted(config['cases'][0])
    for spec in config['cases'][1:]:predicted(spec);native(spec)
    save(out/'rows.json',rows);save(out/'receipts.json',receipts)
    if counts!=expected_counts()or entries!=dict(blocks=436,target_blocks=4):raise ValueError('physical census differs')
    summary=dict(schema='mdb-response-prepared-validation-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],counts=counts,entries=entries,source_job=config['source_job'],forecast_seal_sha256=config['forecast_seal_sha256'],new_predictions=0,worker_seconds=time.monotonic()-started,output_root=str(scratch),mechanics_status='pending-independent-audit',terminal_status='unassessed')
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p)for p in sorted(out.rglob('*'))if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=summary['request_sha256'],output_root=str((scratch/out).resolve()),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:raise SystemExit('Use stage_flux_response_validate.py')
