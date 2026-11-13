"""Prospective receiver-by-dose test of native-parent response anchoring.

The mathematical predictor uses no new native target operands. Historical
half-dose endpoints are explicit calibration anchors, never held-out outcomes.
"""
import hashlib
import inspect
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def cases():
    return [dict(id='three_eighths', receiver='base', anchor='half', dose=.375),
            dict(id='three_eighths.image1', receiver='edited', anchor='half.image1', dose=.375)]


def make_transaction(root, *, model_identity, input_identity, source_identity, wall_s):
    from mdb.forecast_transaction import ForecastTransaction
    return ForecastTransaction(root, model_identity=model_identity, input_identity=input_identity,
        source_identity=source_identity, frontier=dict(kind='architectural', scope=
        'Known half-dose native anchors; two previously unexecuted 3/8 working-text states. Fixed checkpoint, clock and context. No new target ports used.'),
        execution_scope='NumPy full-state absolute and receiver-specific anchored forecasts, then two native CUDA BF16 joint1 blocks.',
        budgets=dict(max_native_calls=1, max_wall_ms=wall_s*1000))


def check_transaction_contract(root):
    from mdb.forecast_transaction import verify_transaction
    tx=make_transaction(root, model_identity='mechanics', input_identity='none', source_identity='none', wall_s=30)
    tx.predict('mechanics', lambda _:dict(predictions=dict(mechanics_only=True)))
    tx.seal()
    with tx: tx.execute('mechanics-only', lambda _:dict(native_calls=0))
    result=verify_transaction(tx.root)
    if not result['verified'] or result['phase']!='closed': raise ValueError('transaction API mismatch')
    return result


def worker():
    payload=Path.cwd(); h=runpy.run_path(str(payload/'flux_binding_minimal.py'))
    sha,save,identity=[h[k] for k in ('sha','save','identity')]
    config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config'] or config['cases']!=cases() or any(sha(payload/n)!=v for n,v in request['source_hashes'].items()):
        raise ValueError('request/source changed')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID']; scratch.mkdir(parents=True,exist_ok=False); os.chdir(scratch)
    out=Path('binding-minimal-result'); out.mkdir(); save(out/'plan.json',config); save(out/'request.json',request)
    started=time.monotonic(); counts=dict(loads=0,native_transitions=0,acts=0,receipts=0,images=0,encodes=0,noise_draws=0)
    entries=dict(blocks=0); receipts=[]
    def event(kind,**kw):
        row=dict(kind=kind,seconds=time.monotonic()-started,counts=dict(counts),entries=dict(entries),**kw)
        with (out/'timeline.jsonl').open('a') as f: f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_CLOSURE '+json.dumps(row,sort_keys=True),flush=True)
    event('verify_source_start')
    for name,item in config['input_files'].items():
        p=Path(config['input_root'])/item['source']
        if sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']: raise ValueError('source custody changed')
        q=out/'source'/name; q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
    event('framework_import_start')
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act,Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.native_navigation import NativeCutNavigator
    from mdb.tensor_archive import write_tensor_archive
    from mdb.forward_recording import tensor_descriptor
    from mdb.forecast_transaction import verify_transaction
    event('framework_import_complete')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction: raise ValueError('historical reduced-BF16 policy differs')
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available(): raise ValueError('CUDA admission required')
    root=Path(config['model_path'])
    inventory={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']: raise ValueError('checkpoint differs')
    event('model_load_start')
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None)
    counts['loads']=1; pipe.set_progress_bar_config(disable=True)
    for component in (pipe.transformer,pipe.vae): component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu')
    adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    block=pipe.transformer.transformer_blocks[1]
    handles=[b.register_forward_pre_hook(lambda m,a:entries.__setitem__('blocks',entries['blocks']+1)) for b in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    source=LocalStore(out/'source/store'); store=LocalStore(out/'store')
    parents={k:source.load(v,device='cuda') for k,v in config['parents'].items()}
    anchors={k:source.load(v['exit'],device='cpu') for k,v in config['history'].items()}
    base=parents['base']; edited=parents['edited']
    for cut in parents.values():
        if cut.model_identity!=adapter.model_identity or any(cut.execution_point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=1).items()): raise ValueError('parent identity/clock differs')
        store.save(cut)
    if any(describe(base.payload[k])!=describe(edited.payload[k]) for k in base.payload if k not in {'image','text','conditioning'}): raise ValueError('receiver closure differs')
    r=runpy.run_path(str(payload/'flux_reentry_reference.py')); math=runpy.run_path(str(payload/'flux_response_reference.py')); B=r['bf16']
    for cls,key in ((type(block),'Flux2TransformerBlock.source_sha256'),(type(block.attn.processor),'Flux2AttnProcessor.source_sha256')):
        if hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()!=r['SOURCE_CONTRACT'][key]: raise ValueError('native operator source differs')
    weights,_=runpy.run_path(str(payload/'flux_response_campaign.py'))['parameter_recipes'](block,root,inventory,r,out,payload)
    event('model_ready'); archive_cache={}
    def array(value):
        if not isinstance(value,torch.Tensor): value=torch.from_numpy(np.asarray(value).copy())
        value=value.detach().cpu().contiguous(); key=identity(tensor_descriptor(value))
        if key not in archive_cache: archive_cache[key]=write_tensor_archive(out/'arrays'/(key+'.npy'),value,root=out)
        return archive_cache[key]
    conv=lambda x:x.detach().float().cpu().numpy()
    T0,T1=conv(base.payload['text']),conv(edited.payload['text'])
    context={k:tuple(conv(x) for x in v) if isinstance(v,tuple) else conv(v) for k,v in base.payload['context'].items()}
    def text(dose):
        T=T0.copy(); T[:,[4,11]]=B(T0[:,[4,11]].astype(float)+dose*(T1[:,[4,11]].astype(float)-T0[:,[4,11]])); return T
    forecasts={}; inputs={}; deltas={}
    for spec in cases():
        if time.monotonic()-started>config['worker_wall_s']: raise RuntimeError('finite wall budget')
        name=spec['id']; I=conv(parents[spec['receiver']].payload['image']); T=text(spec['dose']); H=text(.5)
        historical_entry=source.load(config['history'][spec['anchor']]['entry'],device='cpu')
        if not np.array_equal(I,conv(historical_entry.payload['image'])) or not np.array_equal(H,conv(historical_entry.payload['text'])): raise ValueError('anchor input mismatch')
        if historical_entry.model_identity!=base.model_identity or historical_entry.execution_point!=base.execution_point: raise ValueError('anchor entry identity/clock mismatch')
        if set(historical_entry.payload)!=set(base.payload) or any(describe(historical_entry.payload[k])!=describe(base.payload[k]) for k in base.payload if k not in {'image','text'}): raise ValueError('anchor receiving context mismatch')
        anchor=anchors[spec['anchor']]
        if anchor.model_identity!=base.model_identity or any(anchor.execution_point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=2).items()): raise ValueError('anchor exit identity/clock mismatch')
        cache=math['prepare'](I,H,context,weights,r,heads=pipe.transformer.config.num_attention_heads)
        anchor_prediction=math['predict'](cache,H,compare_dense=False)
        target_prediction=math['predict'](cache,T,compare_dense=False)
        fields={}; deltas[name]={}
        for role in ('image','text'):
            P0=anchor_prediction['fields'][role]['exit']; P1=target_prediction['fields'][role]['exit']; N0=conv(anchors[spec['anchor']].payload[role])
            unrounded=N0.astype(float)+(P1.astype(float)-P0.astype(float)); anchored=B(unrounded)
            deltas[name][role]=P1.astype(float)-P0.astype(float)
            fields[role]=dict(absolute=array(P1),anchored=array(anchored),reference_anchor=array(P0),native_anchor=array(N0),rounding_residual=array(anchored.astype(float)-unrounded))
            fields[role]['zero_response']=fields[role]['native_anchor']
        forecasts[name]=dict(spec=spec,fields=fields,work=target_prediction['work'],cache_seconds=cache['seconds'],anchor_predict_seconds=anchor_prediction['seconds'],target_predict_seconds=target_prediction['seconds'])
        inputs[name]=dict(image=I,text=T); del cache,anchor_prediction,target_prediction
        event('calculated',case=name)
    for spec,other in zip(cases(),reversed(cases())):
        name=spec['id']
        for role in ('image','text'):
            N0=conv(anchors[spec['anchor']].payload[role]); unrounded=N0.astype(float)+deltas[other['id']][role]
            crossed=B(unrounded); fields=forecasts[name]['fields'][role]
            fields['cross_receiver']=array(crossed); fields['cross_rounding_residual']=array(crossed.astype(float)-unrounded)
            fields['cross_receiver_source']=other['id']
    del deltas
    save(out/'forecasts.json',forecasts)
    save(out/'numerical-contract.json',dict(torch=torch.__version__,bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,tf32=torch.backends.cuda.matmul.allow_tf32,deterministic=torch.are_deterministic_algorithms_enabled(),threads=torch.get_num_threads(),batch_size=1,image_shape=[1,1024,3072],text_shape=[1,512,3072],reference='NumPy wide accumulation; legacy QK norm; cached changed-row attention; final BF16 writes',scope='Historical half-dose native anchors, explicit reduced-BF16 full-shape native target profile. No compact-row native GEMM parity claim.'))
    if counts['native_transitions']!=0 or entries['blocks']!=0: raise ValueError('native target entered before seal')
    tx=make_transaction(out/'transaction',model_identity=adapter.model_identity,input_identity=identity(config),source_identity=identity(request['source_hashes']),wall_s=config['worker_wall_s'])
    tx.predict('response_closure',lambda _:dict(predictions=dict(forecasts=forecasts,counts_at_seal=dict(counts),entries_at_seal=dict(entries))))
    seal=tx.seal(); event('sealed',seal=seal); native={}
    def receipt(value):
        body=value.to_dict(); store.receipt(value); receipts.append(body); counts['receipts']+=1
        counts['native_transitions' if body['operation']=='continue' else 'acts']+=body.get('steps',1)
        save(out/('receipt-'+str(len(receipts))+'.json'),body)
    def capture(nav):
        cut=nav.session.capture(retain=False); store.save(cut); return cut
    def execute(spec):
        name=spec['id']; nav=NativeCutNavigator(Session.from_cut(adapter,base),max_steps=1,on_receipt=receipt)
        for role in ('image','text'): nav.apply(Act.replace(role,torch.from_numpy(inputs[name][role].copy()).to('cuda',torch.bfloat16),name=name+'.input.'+role))
        entry=capture(nav); nav.step(); exit=capture(nav)
        native[name]=dict(entry=entry.fingerprint,exit=exit.fingerprint,fields={role:array(conv(exit.payload[role])) for role in ('image','text')})
        save(out/('native-'+name+'.json'),native[name]); event('target_captured',case=name)
    with tx: tx.execute('two-new-native-targets',lambda _:[execute(s) for s in cases()])
    for handle in handles: handle.remove()
    save(out/'native.json',native); save(out/'receipts.json',receipts)
    if counts!=dict(loads=1,native_transitions=2,acts=4,receipts=6,images=0,encodes=0,noise_draws=0) or entries!=dict(blocks=2): raise ValueError('execution census differs')
    summary=dict(schema='mdb-response-closure-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],counts=counts,entries=entries,seal=seal,transaction=verify_transaction(tx.root),output_root=str(scratch),worker_seconds=time.monotonic()-started,mechanics_status='pending-independent-audit',terminal_status='unassessed')
    save(out/'summary.json',summary); save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=summary['request_sha256'],output_root=str((scratch/out).resolve()),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__': worker()
