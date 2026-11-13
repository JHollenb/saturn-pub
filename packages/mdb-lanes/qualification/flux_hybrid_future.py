"""Prospective receiving-LN comparison, including paired raw attention and RGB futures."""
import hashlib
import inspect
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def cases():
    return [dict(id=pattern+suffix,receiver=receiver,anchor=anchor,row_doses=doses)
        for pattern,doses in [('row4seven_eighths',[.875,.125]),('row11seven_eighths',[.125,.875])]
        for receiver,anchor,suffix in [('base','half',''),('edited','half.image1','.image1')]]


def programs():
    return dict(control='anchored',hybrid='absolute')


def expected_counts():
    return dict(loads=1,encodes=0,noise_draws=0,native_transitions=746,acts=24,receipts=56,images=14)


def record_bf16(root,value,records):
    """CAS sharing with the public archive schema and its independently checked reader."""
    import numpy as np
    import torch
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.tensor_archive import write_tensor_archive
    if isinstance(value,torch.Tensor):value=value.detach().float().cpu().numpy()
    value=np.ascontiguousarray(value,np.float32)
    if not np.isfinite(value).all() or np.any(value.view(np.uint32)&65535):raise ValueError('expected finite BF16 evidence')
    tensor=torch.from_numpy(value.copy()).to(torch.bfloat16);descriptor=tensor_descriptor(tensor)
    key=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest()
    if key not in records:records[key]=write_tensor_archive(root/'arrays'/(key+'.npy'),tensor,root=root)
    return records[key]


def worker():
    payload=Path.cwd(); h=runpy.run_path(str(payload/'flux_binding_minimal.py')); sha,save,identity=[h[k] for k in ('sha','save','identity')]
    request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text()); c=request['config']
    if c['cases']!=cases() or c['programs']!=programs() or any(sha(payload/n)!=v for n,v in request['source_hashes'].items()): raise ValueError('frozen source/cases/programs differ')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']; out=scratch/'binding-minimal-result'; out.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    save(out/'plan.json',c); save(out/'request.json',request)
    for name,item in c['input_files'].items():
        p=Path(c['input_roots'][item['root']])/item['source']
        if sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']: raise ValueError('source custody differs')
        q=out/'source'/name; q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers import transformer_flux2 as native_module
    from saturn_pub import Act,Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.native_navigation import NativeCutNavigator
    from mdb_lanes.forecast_transaction import verify_transaction
    from mdb_lanes.tensor_archive import load_tensor_archive
    common=runpy.run_path(str(payload/'flux_local_delta_campaign.py'))
    counts=dict.fromkeys(expected_counts(),0); entries=dict(blocks=0,target_blocks=0,dispatch=0); target_counts={s['id']:0 for s in cases()}
    active=dict(branch='setup',target=False,phase='setup'); receipts=[]; rows=[]; arrays={}
    def event(kind,**kw):
        row=dict(kind=kind,seconds=time.monotonic()-started,counts=dict(counts),entries=dict(entries),target_counts=dict(target_counts),**kw)
        with (out/'timeline.jsonl').open('a') as f: f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_HYBRID_FUTURE '+json.dumps(row,sort_keys=True),flush=True)
    def wall():
        if time.monotonic()-started>c['worker_wall_s']: raise RuntimeError('finite hybrid-future wall budget')
    def array(value):
        return record_bf16(out,value,arrays)
    def observed(key,value):
        return array(value[:,[4,11]] if key.startswith('text.') and key!='text.after_attention' else value)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available() or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction: raise ValueError('native numerical profile differs')
    root=Path(c['model_path']); inventory={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=c['checkpoint_sha256']: raise ValueError('checkpoint differs')
    event('model_load_start')
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None); pipe.set_progress_bar_config(disable=True); counts['loads']=1
    for component in (pipe.transformer,pipe.vae): component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu'); adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda'); block=pipe.transformer.transformer_blocks[1]
    def no_forecast_forward(module,args):
        if active['phase']=='forecast': raise ValueError('learned module forward inside predictor')
    guards=[m.register_forward_pre_hook(no_forecast_forward) for component in (pipe.transformer,pipe.vae) for m in component.modules()]
    def entered(module,args):
        entries['blocks']+=1
        if active['target']: entries['target_blocks']+=1; target_counts[active['case']]+=1
    handles=[b.register_forward_pre_hook(entered) for b in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    source=LocalStore(out/'source/store'); store=LocalStore(out/'store'); parents={k:source.load(v,device='cuda') for k,v in c['parents'].items()}; base=parents['base']; edited=parents['edited']
    for cut in parents.values():
        if cut.model_identity!=adapter.model_identity or any(cut.execution_point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=1).items()): raise ValueError('parent model/clock differs')
        store.save(cut)
    r=runpy.run_path(str(payload/'flux_reentry_reference.py')); law=runpy.run_path(str(payload/'flux_receiver_norm_prediction.py'))
    for cls,key in ((type(block),'Flux2TransformerBlock.source_sha256'),(type(block.attn.processor),'Flux2AttnProcessor.source_sha256')):
        if hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()!=r['SOURCE_CONTRACT'][key]: raise ValueError('native source differs')
    weights,_=runpy.run_path(str(payload/'flux_response_campaign.py'))['parameter_recipes'](block,root,inventory,r,out,payload)
    conv=lambda x:x.detach().float().cpu().numpy(); T0,T1=conv(base.payload['text']),conv(edited.payload['text']); inputs={}; forecasts={}; baselines={}; anchor_cuts={}; baseline_rgb={}
    active['phase']='forecast'
    for anchor in ('half','half.image1'):
        entry=source.load(c['anchors'][anchor]['entry'],device='cuda'); exit=source.load(c['anchors'][anchor]['exit'],device='cuda'); anchor_cuts[anchor]=exit
        if entry.model_identity!=adapter.model_identity or entry.execution_point!=base.execution_point: raise ValueError('anchor model/clock differs')
        if exit.model_identity!=adapter.model_identity or any(exit.execution_point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=2).items()): raise ValueError('anchor exit clock differs')
        receiver=base if anchor=='half' else edited
        if describe(entry.payload['image'])!=describe(receiver.payload['image']): raise ValueError('anchor receiving image differs')
        half=T0.copy(); half[:,[4,11]]=r['bf16'](T0[:,[4,11]].astype(float)+.5*(T1[:,[4,11]].astype(float)-T0[:,[4,11]].astype(float)))
        if not np.array_equal(conv(entry.payload['text']),half): raise ValueError('anchor half-dose construction differs')
        if any(describe(entry.payload[k])!=describe(base.payload[k]) for k in base.payload if k not in {'image','text'}): raise ValueError('anchor closure differs')
        stages={k:load_tensor_archive(out/'source/prefixes',v) for k,v in c['anchors'][anchor]['stages'].items()}; I,T=conv(entry.payload['image']),conv(entry.payload['text'])
        if any(not np.array_equal(stages[role+'.exit'],conv(exit.payload[role])) for role in ('image','text')): raise ValueError('anchor port/cut differs')
        context={k:tuple(conv(x) for x in v) if isinstance(v,tuple) else conv(v) for k,v in entry.payload['context'].items()}; baselines[anchor]={}
        for mode,norm in programs().items():
            wall(); cache=law['prepare'](I,T,context,weights,stages,r,absolute_ports=c['absolute_ports'],affine_policy=c['policy'],absolute_norms=('text.norm1','text.qk_rms'),attention_mode='anchored',receiver_norm=norm)
            def predict(text):
                seen={}; value=law['predict'](cache,text,observer=lambda k,v:seen.__setitem__(k,observed(k,v)))
                return dict(fields={role:{field:array(x) for field,x in fields.items()} for role,fields in value['fields'].items()},stages=seen,seconds=value['seconds'],affine_work=value['affine_work'],attention_work=value['attention_work'])
            baselines[anchor][mode]=predict(T)
            for spec in [s for s in cases() if s['anchor']==anchor]:
                name=spec['id']; newtext=T0.copy()
                for row,dose in zip((4,11),spec['row_doses']): newtext[:,row]=r['bf16'](T0[:,row].astype(float)+dose*(T1[:,row].astype(float)-T0[:,row].astype(float)))
                inputs[name]=dict(image=I,text=newtext); forecasts.setdefault(name,{})[mode]=predict(newtext); event('calculated',case=name,mode=mode)
            del cache
        del stages
        baseline_rgb[anchor]=np.asarray(Image.open(out/'source/baselines'/(anchor+'.png')).convert('RGB')).copy()
    del weights
    save(out/'forecasts.json',forecasts); save(out/'baselines.json',baselines)
    if entries!=dict(blocks=0,target_blocks=0,dispatch=0): raise ValueError('native computation before write seal')
    tx=common['make_transaction'](out/'write-transaction',model_identity=adapter.model_identity,input_identity=identity(c),source_identity=identity(request['source_hashes']),wall_s=c['worker_wall_s'],scope=c['hypotheses']['scope'])
    tx.predict('affected_writes',lambda _:dict(predictions=dict(forecasts=forecasts,baselines=baselines,counts_at_seal=dict(counts),entries_at_seal=dict(entries),target_counts=dict(target_counts))))
    write_seal=tx.seal(); event('writes_sealed',seal=write_seal); active['phase']='consumer'
    steps=runpy.run_path(str(payload/'flux_response_validate.py'))['steps_to_commit']
    def receipt(value):
        wall(); body=value.to_dict(); store.receipt(value); receipts.append(dict(branch=active['branch'],body=body)); counts['receipts']+=1; counts['native_transitions' if body['operation']=='continue' else 'acts']+=body.get('steps',1)
    def capture(nav): cut=nav.session.capture(retain=False); store.save(cut); return cut
    def finish(nav,name,mode,anchor,entry):
        commits=[]
        while nav.point['logical_step']<4:
            before=nav.point['logical_step']; nav.step(steps(nav.point,len(pipe.transformer.transformer_blocks),len(pipe.transformer.single_transformer_blocks)))
            if nav.point['logical_step']!=before+1: raise ValueError('suffix group crossed wrong commit')
            commits.append(capture(nav).fingerprint)
        final=capture(nav); p=out/(name+'.'+mode+'.png'); adapter.decode(nav.session,height=512,width=512).save(p); counts['images']+=1
        rgb=np.asarray(Image.open(p).convert('RGB')).copy(); row=dict(id=name,mode=mode,anchor=anchor,entry=entry,commits=commits,final=final.fingerprint,png=p.name,png_sha256=sha(p),readout=common['scene_readout'](rgb,baseline_rgb[anchor]),target_counts_at_image=dict(target_counts))
        rows.append(row); save(out/('completed-'+name+'.'+mode+'.json'),row); event('image_complete',case=name,mode=mode); return row
    def candidate_futures(_):
        for anchor in ('half','half.image1'):
            active.update(branch=anchor+'.replay',target=False); nav=NativeCutNavigator(Session.from_cut(adapter,anchor_cuts[anchor]),max_steps=53,on_receipt=receipt)
            row=finish(nav,anchor,'replay',anchor,anchor_cuts[anchor].fingerprint)
            if row['png_sha256']!=c['anchors'][anchor]['png_sha256']: raise ValueError('historical native suffix/PNG replay differs')
        for spec in cases():
            name=spec['id']
            for mode in programs():
                active.update(branch=name+'.'+mode,target=False); scaffold=anchor_cuts[spec['anchor']]; nav=NativeCutNavigator(Session.from_cut(adapter,scaffold),max_steps=53,on_receipt=receipt)
                for role in ('image','text'):
                    value=load_tensor_archive(out,forecasts[name][mode]['fields'][role]['exit'])
                    nav.apply(Act.replace(role,torch.from_numpy(value.copy()).to('cuda',torch.bfloat16),name=name+'.'+mode+'.'+role))
                installed=capture(nav)
                if any(describe(installed.payload[k])!=describe(scaffold.payload[k]) for k in scaffold.payload if k not in {'image','text'}): raise ValueError('candidate changed undeclared closure')
                finish(nav,name,mode,spec['anchor'],installed.fingerprint)
        return dict(images=len(rows),counts=dict(counts))
    with tx: tx.execute('candidate-image-consumers',candidate_futures)
    image_predictions={row['id']+'.'+row['mode']:dict(png_sha256=row['png_sha256'],readout=row['readout']) for row in rows if row['mode']!='replay'}
    save(out/'image-forecasts.json',image_predictions)
    if entries['target_blocks']!=0 or any(target_counts.values()) or counts['images']!=10: raise ValueError('target observed before image seal')
    future=common['make_transaction'](out/'image-transaction',model_identity=adapter.model_identity,input_identity=identity(c),source_identity=identity(dict(sources=request['source_hashes'],write_seal=write_seal)),wall_s=c['worker_wall_s'],scope='Candidate RGB and fixed-source supports measured using native suffix/VAE; four fresh target joint1 writes and futures remain unobserved. Native consumer work fully counted.')
    future.predict('image_effects',lambda _:dict(predictions=dict(forecasts=image_predictions,counts_at_seal=dict(counts),entries_at_seal=dict(entries),target_counts=dict(target_counts))))
    image_seal=future.seal(); event('image_forecasts_sealed',seal=image_seal); native={}
    capture_ports=runpy.run_path(str(payload/'flux_precision_cut.py'))['capture_ports']
    wanted=set(next(iter(baselines['half'].values()))['stages'])|{role+'.'+field for role in ('image','text') for field in ('attention','mlp_input','mlp','exit')}
    def run_target(spec):
        name=spec['id']; active.update(branch=name+'.native',case=name,target=False); nav=NativeCutNavigator(Session.from_cut(adapter,base),max_steps=54,on_receipt=receipt)
        for role in ('image','text'): nav.apply(Act.replace(role,torch.from_numpy(inputs[name][role].copy()).to('cuda',torch.bfloat16),name=name+'.native.'+role))
        entry=capture(nav); ports={}
        def remember(key,value):
            if key in wanted: ports[key]=observed(key,value) if key in baselines['half']['control']['stages'] else array(value)
        active['target']=True
        try: capture_ports(block,native_module,T0.shape[1],remember,entries,nav.step)
        finally: active['target']=False
        exit=capture(nav)
        for role in ('image','text'): remember(role+'.exit',exit.payload[role])
        if set(ports)!=wanted: raise ValueError('required native ports missing')
        native[name]=dict(entry=entry.fingerprint,exit=exit.fingerprint,stages=ports); save(out/('native-'+name+'.json'),native[name]); finish(nav,name,'native',spec['anchor'],entry.fingerprint)
    with future: future.execute('four-unseen-native-targets',lambda _:[run_target(s) for s in cases()])
    for handle in [*handles,*guards]: handle.remove()
    save(out/'native.json',native); save(out/'rows.json',rows); save(out/'receipts.json',receipts)
    if counts!=expected_counts() or entries!=dict(blocks=676,target_blocks=4,dispatch=4) or any(v!=1 for v in target_counts.values()): raise ValueError('physical execution census differs')
    summary=dict(schema='mdb-hybrid-future-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),counts=counts,entries=entries,target_counts=target_counts,
        write_seal=write_seal,image_seal=image_seal,write_transaction=verify_transaction(tx.root),image_transaction=verify_transaction(future.root),worker_seconds=time.monotonic()-started,unique_array_files=len(arrays),mechanics_status='pending-independent-audit',terminal_status='unassessed')
    save(out/'summary.json',summary); save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__': worker()
