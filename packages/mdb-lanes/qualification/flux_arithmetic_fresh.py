"""Seal four new complete writes, then trace native affines and attention."""
import argparse
import hashlib
import inspect
import json
import os
import runpy
import time
from pathlib import Path


def cases():
    return [dict(id=pattern+suffix,receiver=receiver,anchor=anchor,row_doses=doses)
        for pattern,doses in [('row4five_eighths',[.625,.375]),('row11five_eighths',[.375,.625])]
        for receiver,anchor,suffix in [('base','half',''),('edited','half.image1','.image1')]]


def counts_expected():
    return dict(loads=1,encodes=0,noise_draws=0,native_transitions=4,acts=8,receipts=12,native_attention_fallbacks=4,images=0)


def worker():
    payload=Path.cwd(); h=runpy.run_path(str(payload/'flux_binding_minimal.py')); sha,save,identity=[h[k] for k in ('sha','save','identity')]
    request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text()); c=request['config']
    if c['cases']!=cases() or any(sha(payload/n)!=v for n,v in request['source_hashes'].items()): raise ValueError('frozen worker/cases changed')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    source=Path(c['input_root'])
    if any(sha(source/n)!=v for n,v in c['inputs'].items()): raise ValueError('source changed')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']; out=scratch/'binding-minimal-result'; out.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    save(out/'plan.json',c); save(out/'request.json',request)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers import transformer_flux2 as native_module
    from saturn_pub import Act,Session
    from saturn_pub.store import LocalStore
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.values import describe
    from mdb_lanes.native_navigation import NativeCutNavigator
    from mdb_lanes.tensor_archive import load_tensor_archive,write_tensor_archive
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb_lanes.attention_partition import dense_attention
    from mdb_lanes.response_closure import measure_error
    counts=dict.fromkeys(counts_expected(),0); entries=dict(blocks=0,dispatch=0); phase='setup'; receipts=[]; archives={}; active=None
    def event(kind,**kw):
        row=dict(kind=kind,seconds=time.monotonic()-started,counts=dict(counts),entries=dict(entries),**kw)
        with (out/'timeline.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_FRESH_ARITHMETIC '+json.dumps(row,sort_keys=True),flush=True)
        if time.monotonic()-started>c['worker_wall_s']: raise RuntimeError('finite worker budget')
    def array(value):
        if not isinstance(value,torch.Tensor):
            value=np.asarray(value,np.float32)
            if not np.array_equal(value,r['bf16'](value)): raise ValueError('forecast archive not BF16')
            value=torch.from_numpy(value.copy()).to(torch.bfloat16)
        value=value.detach().cpu(); key=identity(tensor_descriptor(value))
        if key not in archives: archives[key]=write_tensor_archive(out/'arrays'/(key+'.npy'),value,root=out)
        return archives[key]
    read=lambda p:json.loads(p.read_text()); conv=lambda x:x.detach().float().cpu().numpy()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available() or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction: raise ValueError('numerical profile differs')
    save(out/'native-profile.json',dict(torch=torch.__version__,device=torch.cuda.get_device_name(),threads=torch.get_num_threads(),bf16_reduced=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        tf32=torch.backends.cuda.matmul.allow_tf32,flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),memory_efficient_sdp_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),math_sdp_enabled=torch.backends.cuda.math_sdp_enabled(),scope='Enabled attention backends are recorded, not asserted to be the selected kernel.'))
    root=Path(c['model_path']); inventory={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=c['checkpoint_sha256']: raise ValueError('checkpoint changed')
    event('model_load_start')
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None); pipe.set_progress_bar_config(disable=True); counts['loads']=1
    pipe.transformer.eval().requires_grad_(False).to('cuda'); pipe.vae.eval().requires_grad_(False); pipe.vae.bn.to('cpu')
    adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda'); block=pipe.transformer.transformer_blocks[1]
    r=runpy.run_path(str(payload/'flux_reentry_reference.py')); old=runpy.run_path(str(payload/'flux_local_delta_reference.py')); new=runpy.run_path(str(payload/'flux_arithmetic_prediction.py')); ar=runpy.run_path(str(payload/'flux_arithmetic_reference.py'))
    for cls,key in ((type(block),'Flux2TransformerBlock.source_sha256'),(type(block.attn.processor),'Flux2AttnProcessor.source_sha256')):
        if hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()!=r['SOURCE_CONTRACT'][key]: raise ValueError('native source changed')
    handles=[b.register_forward_pre_hook(lambda m,a:entries.__setitem__('blocks',entries['blocks']+1)) for b in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    guard=torch.nn.modules.module.register_module_forward_pre_hook(lambda m,a:(_ for _ in ()).throw(ValueError('native module during independent prediction')) if phase=='forecast' else None)
    weights,_=runpy.run_path(str(payload/'flux_response_campaign.py'))['parameter_recipes'](block,root,inventory,r,out,payload)
    store=LocalStore(out/'store'); history=LocalStore(source/'store'); parents={k:history.load(v,device='cuda') for k,v in c['parents'].items()}; base,edited=parents['base'],parents['edited']
    for cut in parents.values():
        if cut.model_identity!=adapter.model_identity or any(cut.execution_point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=1).items()): raise ValueError('source model/clock changed')
        store.save(cut)
    T0,T1=conv(base.payload['text']),conv(edited.payload['text']); context={k:tuple(conv(x) for x in v) if isinstance(v,tuple) else conv(v) for k,v in base.payload['context'].items()}
    prior=read(source/'native.json'); inputs={}; forecasts={}; zeros={}; phase='forecast'
    for anchor in ('half','half.image1'):
        spec0=next(s for s in cases() if s['anchor']==anchor); I=conv(parents[spec0['receiver']].payload['image']); calibration=history.load(prior[anchor]['entry'],device='cpu'); T=conv(calibration.payload['text'])
        half=T0.copy(); half[:,[4,11]]=r['bf16'](T0[:,[4,11]].astype(float)+.5*(T1[:,[4,11]].astype(float)-T0[:,[4,11]].astype(float)))
        if calibration.model_identity!=adapter.model_identity or dict(calibration.execution_point)!=dict(base.execution_point): raise ValueError('calibration model/clock differs')
        if not np.array_equal(I,conv(calibration.payload['image'])) or not np.array_equal(T,half): raise ValueError('calibration image/text differs from declared construction')
        if any(describe(calibration.payload[k])!=describe(base.payload[k]) for k in base.payload if k not in {'image','text'}): raise ValueError('calibration closure differs')
        stages={k:load_tensor_archive(source,v) for k,v in prior[anchor]['stages'].items()}
        caches=dict(local=old['prepare'](I,T,context,weights,stages,r),arithmetic=new['prepare'](I,T,context,weights,stages,r,absolute_ports=c['absolute_ports'],affine_policy=c['policy']))
        zeros[anchor]={}
        for mode,cache in caches.items():
            zero=(old if mode=='local' else new)['predict'](cache,T)
            zeros[anchor][mode]={role:measure_error(zero['fields'][role]['exit'],stages[role+'.exit']) for role in ('image','text')}
            if mode=='local' and any(not np.array_equal(zero['fields'][role]['exit'],stages[role+'.exit']) for role in ('image','text')): raise ValueError('anchored mechanics zero changed')
        for spec in cases():
            if spec['anchor']!=anchor: continue
            name=spec['id']; text=T0.copy()
            for row,dose in zip((4,11),spec['row_doses']): text[:,row]=r['bf16'](T0[:,row].astype(float)+dose*(T1[:,row].astype(float)-T0[:,row].astype(float)))
            inputs[name]=dict(image=I,text=text); forecasts[name]={}
            for mode,cache in caches.items():
                observed={}
                def observe(key,value): observed[key]=array(value)
                prediction=old['predict'](cache,text) if mode=='local' else new['predict'](cache,text,observer=observe)
                if mode=='arithmetic':
                    other=[i for i in range(T.shape[1]) if i not in (4,11)]
                    for key in ('q_positioned','k_positioned','v_projection'):
                        if not np.array_equal(load_tensor_archive(out,observed['text.'+key])[:,other],stages['text.'+key][:,other]): raise ValueError('forecast contains undeclared partition operand changes')
                forecasts[name][mode]=dict(fields={role:{k:array(v) for k,v in vals.items() if k=='exit' or mode=='arithmetic'} for role,vals in prediction['fields'].items()},stages=observed,seconds=prediction['seconds'],affine_work=prediction['affine_work'])
            event('predicted',case=name)
        del caches,stages
    save(out/'forecasts.json',forecasts); save(out/'zero-measurements.json',zeros)
    if entries!=dict(blocks=0,dispatch=0): raise ValueError('target computation before seal')
    tx=ForecastTransaction(out/'transaction',model_identity=adapter.model_identity,input_identity=identity(c),source_identity=identity(request['source_hashes']),frontier=dict(kind='architectural',scope='Four unseen5/8x3/8workingtext states on two known receivers; historical normalization/attention calibration remains supplied.'),execution_scope='NumPy complete writes and own positioned textQKV sealed before all4nativeblocks; native attention on ownQKV is a later explicit diagnostic control.',budgets=dict(max_native_calls=1,max_wall_ms=c['worker_wall_s']*1000))
    tx.predict('complete_writes',lambda _:dict(predictions=dict(forecasts=forecasts,zeros=zeros,counts_at_seal=dict(counts),entries_at_seal=dict(entries))))
    seal=tx.seal(); phase='target'; event('sealed',seal=seal); native={}; arguments={}; original_dispatch=native_module.dispatch_attention_fn
    def intercepted(q,k,v,*args,**kwargs):
        arguments[active]=dict(args=args,kwargs=kwargs,description=describe(dict(args=args,kwargs=kwargs)),metadata={label:dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype)) for label,value in [('q',q),('k',k),('v',v)]})
        return original_dispatch(q,k,v,*args,**kwargs)
    native_module.dispatch_attention_fn=intercepted
    def receipt(value):
        body=value.to_dict(); store.receipt(value); receipts.append(dict(case=active,body=body)); counts['receipts']+=1; counts['native_transitions' if body['operation']=='continue' else 'acts']+=body.get('steps',1)
    def capture(nav): cut=nav.session.capture(retain=False); store.save(cut); return cut
    keep={'attention.raw','image.attention','text.attention','image.mlp_input','text.mlp_input','image.mlp','text.mlp','image.after_attention','text.after_attention'}
    keep.update('text.'+k for k in ('modulated','q_projection','k_projection','v_projection','q_normalized','k_normalized','q_positioned','k_positioned'))
    def targets(_):
        nonlocal active
        for spec in cases():
            active=spec['id']; nav=NativeCutNavigator(Session.from_cut(adapter,base),max_steps=1,on_receipt=receipt)
            for role in ('image','text'): nav.apply(Act.replace(role,torch.from_numpy(inputs[active][role].copy()).to('cuda',torch.bfloat16),name=active+'.input.'+role))
            entry=capture(nav); ports={}; unchanged={}
            def remember(key,value):
                if key.startswith('image.') and key.split('.',1)[1] in {'modulated','q_projection','k_projection','v_projection','q_normalized','k_normalized','q_positioned','k_positioned'}:
                    record=prior[spec['anchor']]['stages'][key]; actual=conv(value)
                    if not np.array_equal(actual,load_tensor_archive(source,record)): raise ValueError('unchanged image prefix differs')
                    unchanged[key]=record
                if key in keep: ports[key]=array(value)
            runpy.run_path(str(payload/'flux_precision_cut.py'))['capture_ports'](block,native_module,T0.shape[1],remember,entries,nav.step)
            exit=capture(nav); ports.update({role+'.exit':array(exit.payload[role]) for role in ('image','text')})
            native[active]=dict(entry=entry.fingerprint,exit=exit.fingerprint,stages=ports,unchanged_image_prefix=unchanged,dispatch_metadata=arguments[active]['metadata'],dispatch_arguments=arguments[active]['description'])
            save(out/('native-'+active+'.json'),native[active]); event('native_target',case=active)
        return dict(native=native)
    try:
        with tx: tx.execute('four-native-targets',targets)
    finally: native_module.dispatch_attention_fn=original_dispatch
    save(out/'native.json',native); phase='diagnostic'; controls={}
    def reconstruct(spec,observed):
        anchor=spec['anchor']; records=observed['stages']; p=prior[anchor]['stages']
        return [np.concatenate([load_tensor_archive(out,records['text.'+key]),load_tensor_archive(source,p['image.'+key])],axis=1) for key in ('q_positioned','k_positioned','v_projection')]
    for spec in cases():
        name=spec['id']; owned=reconstruct(spec,forecasts[name]['arithmetic']); tensors=[]; argument=arguments[name]
        for label,value in zip(('q','k','v'),owned):
            meta=argument['metadata'][label]
            if list(value.shape)!=meta['shape'] or meta['dtype']!='torch.bfloat16': raise ValueError('own operand geometry/dtype differs')
            tensor=torch.empty_strided(meta['shape'],meta['stride'],device='cuda',dtype=torch.bfloat16); tensor.copy_(torch.from_numpy(value.copy()).to('cuda',torch.bfloat16)); tensors.append(tensor)
        with torch.no_grad(): fallback=original_dispatch(*tensors,*argument['args'],**argument['kwargs'])
        counts['native_attention_fallbacks']+=1; controls[name]=dict(native_own_attention=array(fallback.flatten(2,3)),dispatch_metadata=argument['metadata'],dispatch_arguments=argument['description'],conditional_qkv={})
        actual=load_tensor_archive(out,native[name]['stages']['text.modulated'])
        for label in ('q','k','v'):
            port='attn.add_'+label+'_proj'; w=weights[port+'.weight']; b=weights.get(port+'.bias'); result=np.empty_like(actual)
            for first in range(0,w.shape[0],256):
                sl=slice(first,first+256); result[...,sl]=r['bf16'](ar['affine_wide'](actual,w[sl],c['policy'],None if b is None else b[sl]))
            controls[name]['conditional_qkv'][label]=array(result.reshape(1,actual.shape[1],24,128))
        native_operands=reconstruct(spec,native[name]); mathematical=r['bf16'](dense_attention(*[x.transpose(0,2,1,3) for x in native_operands]).transpose(0,2,1,3).reshape(1,-1,3072))
        controls[name]['wide_attention_native_operands']=array(mathematical); save(out/('controls-'+name+'.json'),controls[name]); event('diagnostic_controls',case=name)
    guard.remove()
    for handle in handles: handle.remove()
    save(out/'controls.json',controls); save(out/'receipts.json',receipts)
    if counts!=counts_expected() or entries!=dict(blocks=4,dispatch=4): raise ValueError('execution census differs')
    summary=dict(schema='mdb-fresh-affine-arithmetic-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),worker_seconds=time.monotonic()-started,counts=counts,entries=entries,seal=seal,transaction=verify_transaction(tx.root),mechanics_status='pending-independent-audit',terminal_status='unassessed',scope='Four fresh complete-state forecasts, four original blocks, four explicitly native attention controls on own frozen operands. Native target ports only in conditional diagnostics, never forecast inputs. No new image or semantic evidence.')
    save(out/'summary.json',summary); save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=summary['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__': worker()
