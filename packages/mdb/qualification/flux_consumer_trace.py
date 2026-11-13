"""Replay nine source-qualified J1 exits to their next commit with component captures."""
import hashlib
import inspect
import json
import os
import runpy
import time
from pathlib import Path


def counts():return dict(loads=1,native_transitions=225,receipts=225,acts=0,blocks=207,joint_midpoints=27,norm_out=9,proj_out=9,images=0,encodes=0,noise_draws=0)


def worker():
    payload=Path.cwd();h=runpy.run_path(str(payload/'flux_binding_minimal.py'));sha,save,identity=[h[k] for k in ('sha','save','identity')]
    read=lambda p:json.loads(Path(p).read_text());request=read(os.environ['SATURN_DEBUG_REQUEST']);c=read(os.environ['SATURN_RUN_CONFIG'])
    if c!=request['config'] or any(sha(payload/name)!=digest for name,digest in request['source_hashes'].items()):raise ValueError('trace request/source changed')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    source=Path(c['source_collection']);remote=read(source/'remote-custody.json')
    if sha(source/'remote-custody.json')!=c['source_record_sha256'] or runpy.run_path(str(payload/'flux_family_custody.py'))['verify'](source)!=remote['verification']:raise ValueError('full source custody differs')
    audit=read(c['source_audit'])
    if sha(c['source_audit'])!=c['source_audit_sha256'] or audit['custody']!=remote['verification'] or audit['mechanics_status']!='verified':raise ValueError('source consumer audit differs')
    data=source/'extracted/binding-minimal-result';source_rows=read(data/'rows.json')
    if c['branches']!=[{key:row[key] for key in ('id','mode','entry','commits')} for row in source_rows]:raise ValueError('source branch join differs')
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    save(out/'request.json',request);save(out/'plan.json',c);save(out/'source-binding.json',dict(remote_record=remote,source_audit_sha256=c['source_audit_sha256']))
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.native_navigation import NativeCutNavigator
    started=time.monotonic();actual=dict.fromkeys(counts(),0);arrays={};rows=[];receipts=[];active=dict(branch='setup')
    def event(kind,**kw):
        row=dict(kind=kind,seconds=time.monotonic()-started,counts=dict(actual),**kw)
        with (out/'timeline.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_CONSUMER_TRACE '+json.dumps(row,sort_keys=True),flush=True)
        if row['seconds']>c['worker_wall_s']:raise RuntimeError('finite consumer-trace wall budget')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available() or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:raise ValueError('native CUDA BF16 profile differs')
    root=Path(c['model_path']);inventory={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=c['checkpoint_sha256']:raise ValueError('checkpoint differs')
    event('model_load_start')
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False).to('cuda');pipe.vae.eval().requires_grad_(False);pipe.vae.bn.to('cpu');actual['loads']=1
    adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    reference=read(data/'receipts.json')[0]['body']
    if adapter.model_identity!=reference['model_identity'] or dict(adapter.execution)!=reference['execution']:raise ValueError('source model/execution contract differs')
    parameters=runpy.run_path(str(payload/'flux_trace_parameter_recipes.py'))['recipes'](pipe.transformer,root,inventory,out,payload)
    save(out/'native-contract.json',dict(model_identity=adapter.model_identity,execution=adapter.execution,parameter_recipes=parameters,
        adapter_advance_sha256=hashlib.sha256(inspect.getsource(type(adapter).advance).encode()).hexdigest(),joint_source_sha256=hashlib.sha256(inspect.getsource(type(pipe.transformer.transformer_blocks[2])).encode()).hexdigest(),
        single_source_sha256=hashlib.sha256(inspect.getsource(type(pipe.transformer.single_transformer_blocks[0])).encode()).hexdigest(),
        scheduler_step=inspect.getsource(type(pipe.scheduler).step),bf16_reduced_precision=bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)))
    def entered(module,args):actual['blocks']+=1
    handles=[block.register_forward_pre_hook(entered) for block in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    def no_vae(module,args):raise ValueError('VAE must not execute in this microscope')
    handles.append(pipe.vae.register_forward_pre_hook(no_vae))
    original=LocalStore(data/'store');store=LocalStore(out/'store')
    record=runpy.run_path(str(payload/'flux_hybrid_future.py'))['record_bf16'];array=lambda value:record(out,value,arrays)
    def receipt(value):
        body=value.to_dict();store.receipt(value);receipts.append(dict(branch=active['branch'],body=body));actual['receipts']+=1;actual['native_transitions']+=body.get('steps',0)
        if body['operation']!='continue' or body.get('steps')!=1:raise ValueError('trace must consist of individual native transitions')
    def capture(nav):cut=nav.session.capture(retain=False);store.save(cut);return cut
    def endpoint(cut,label,fields):return dict(id=label,kind='cut',cut=cut.fingerprint,fields=fields,clock=dict(cut.execution_point))
    for spec in c['branches']:
        active['branch']=spec['id']+'.'+spec['mode'];origin=original.load(spec['entry'],device='cuda');destination=original.load(spec['commits'][0],device='cuda');store.save(origin)
        nav=NativeCutNavigator(Session.from_cut(adapter,origin),max_steps=25,on_receipt=receipt)
        if any(nav.point[key]!=value for key,value in dict(logical_step=2,phase='joint',index=2).items()):raise ValueError('trace origin clock differs')
        points=[endpoint(origin,'joint1.exit',['image','text'])];transitions=[]
        for step in range(25):
            before=capture(nav);point=dict(nav.point);ports={};temporary=[]
            if point['phase']=='joint':
                index=point['index'];block=pipe.transformer.transformer_blocks[index]
                def remember(role):
                    def hook(module,args):ports[role]=array(args[0])
                    return hook
                temporary=[block.norm2.register_forward_pre_hook(remember('image')),block.norm2_context.register_forward_pre_hook(remember('text'))]
            elif point['phase']=='readout':
                def norm_hook(module,args,output):ports['image']=array(output);actual['norm_out']+=1
                def projection_hook(module,args,output):actual['proj_out']+=1
                temporary=[pipe.transformer.norm_out.register_forward_hook(norm_hook),pipe.transformer.proj_out.register_forward_hook(projection_hook)]
            try:nav.step()
            finally:
                for handle in temporary:handle.remove()
            after=capture(nav);rec=receipts[-1]['body']
            transitions.append(dict(before=before.fingerprint,after=after.fingerprint,receipt=rec['fingerprint'],point=point))
            if point['phase']=='joint':
                if set(ports)!={'image','text'}:raise ValueError('joint attention midpoint capture missing')
                actual['joint_midpoints']+=1
                points.append(dict(id='joint'+str(index)+'.attention',kind='arrays',arrays=ports,clock=point,before=before.fingerprint,after=after.fingerprint,receipt=rec['fingerprint']))
                points.append(endpoint(after,'joint'+str(index)+'.exit',['image','text']))
            elif point['phase']=='single':points.append(endpoint(after,'single'+str(point['index'])+'.exit',['image','text']))
            elif point['phase']=='readout':
                if set(ports)!={'image'}:raise ValueError('readout norm capture missing')
                points.append(dict(id='readout.norm',kind='arrays',arrays=ports,clock=point,before=before.fingerprint,after=after.fingerprint,receipt=rec['fingerprint']))
                points.append(endpoint(after,'readout.noise',['noise_prediction']))
            elif point['phase']=='scheduler':points.append(endpoint(after,'scheduler.latent',['latent']))
            else:raise ValueError('unexpected trace transition')
        final=capture(nav)
        same=describe(final.payload)==describe(destination.payload) and dict(final.execution_point)==dict(destination.execution_point)
        row=dict(id=spec['id'],mode=spec['mode'],origin=origin.fingerprint,expected_commit=destination.fingerprint,final=final.fingerprint,points=points,transitions=transitions,
            replay_status='qualified' if same else 'instrument-replay-mismatch',full_payload_replay_exact=same)
        rows.append(row);save(out/('trace-'+spec['id']+'.'+spec['mode']+'.json'),row);event('branch_complete',branch=active['branch'],replay_status=row['replay_status'],points=len(points))
    for handle in handles:handle.remove()
    save(out/'traces.json',rows);save(out/'receipts.json',receipts)
    if actual!=counts() or len(rows)!=9 or any(len(row['points'])!=30 or len(row['transitions'])!=25 for row in rows):raise ValueError('complete microscope census differs')
    summary=dict(schema='mdb-consumer-trace-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(out.parent),counts=actual,unique_arrays=len(arrays),
        replay_status={row['id']+'.'+row['mode']:row['replay_status'] for row in rows},seconds=time.monotonic()-started,mechanics_status='complete-pending-independent-audit',terminal_status='unassessed',scope=c['scope'])
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__':worker()
