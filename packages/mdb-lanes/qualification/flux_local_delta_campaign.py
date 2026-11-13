"""Freeze fresh affected writes and their image forecasts before native targets."""
import hashlib
import inspect
import json
import os
import runpy
import time
from pathlib import Path


def cases():
    return [dict(id=pattern+suffix,receiver=receiver,anchor=anchor,row_doses=doses) for pattern,doses in [('row4heavy',[.75,.25]),('row11heavy',[.25,.75])] for receiver,anchor,suffix in [('base','half',''),('edited','half.image1','.image1')]]


def expected_counts():
    return dict(loads=1,encodes=0,noise_draws=0,native_transitions=746,acts=24,receipts=56,images=14)


def make_transaction(root,*,model_identity,input_identity,source_identity,wall_s,scope):
    from mdb_lanes.forecast_transaction import ForecastTransaction
    return ForecastTransaction(root,model_identity=model_identity,input_identity=input_identity,source_identity=source_identity,
        frontier=dict(kind='architectural',scope=scope),execution_scope='Original CUDA BF16 B1 targets and native image consumers; affected target writes calculated in NumPy using explicit historical calibration.',budgets=dict(max_native_calls=1,max_wall_ms=wall_s*1000))


def preflight_transaction(root):
    from mdb_lanes.forecast_transaction import verify_transaction
    tx=make_transaction(root,model_identity='mechanics',input_identity='none',source_identity='none',wall_s=30,scope='No model mechanics check')
    tx.predict('mechanics',lambda _:dict(predictions=dict(no_model=True))); tx.seal()
    with tx: tx.execute('mechanics',lambda _:dict(native_calls=0))
    if not verify_transaction(tx.root)['verified']: raise ValueError('installed transaction API mismatch')


def scene_readout(rgb,baseline):
    """Fixed source-image supports; not an automatic semantic object decoder."""
    import numpy as np
    source=baseline.astype(float); current=rgb.astype(float); yy,xx=np.indices(rgb.shape[:2]); chroma=(source.max(-1)-source.min(-1))>255*.18
    regions={}; boxes=np.zeros(chroma.shape,bool)
    for name,mask in [('left_source_object',chroma&(xx<256)),('right_source_object',chroma&(xx>=256))]:
        if not mask.any(): raise ValueError('source object readout empty')
        ys,xs=np.nonzero(mask); boxes[max(0,ys.min()-12):min(512,ys.max()+13),max(0,xs.min()-12):min(512,xs.max()+13)]=True
        delta=(current-source)[mask].mean(axis=0); mean=current[mask].mean(axis=0)
        regions[name]=dict(source_pixels=int(mask.sum()),mean_rgb=mean.tolist(),delta_rgb=delta.tolist(),red_minus_blue_delta=float(delta[0]-delta[2]))
    background=~boxes
    return dict(regions=regions,background_outside_source_boxes_mae=float(np.abs(current-source)[background].mean()),
        source_mask_policy=dict(chroma_threshold=.18,left_right_split=256,background_excludes_source_boxes_plus12=True),
        scope='Source RGB supplies fixed foreground supports associated with known left sphere/right cube. Means and background error are pixel observables, not semantic certification.')


def worker():
    payload=Path.cwd(); h=runpy.run_path(str(payload/'flux_binding_minimal.py')); sha,save,identity=[h[k] for k in ('sha','save','identity')]
    request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text()); c=request['config']
    if c['cases']!=cases() or any(sha(payload/n)!=v for n,v in request['source_hashes'].items()): raise ValueError('source/cases changed')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']; out=scratch/'binding-minimal-result'; out.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    save(out/'plan.json',c); save(out/'request.json',request)
    import shutil
    for name,item in c['input_files'].items():
        p=Path(c['input_roots'][item['root']])/item['source']
        if sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']: raise ValueError('source custody changed')
        q=out/'source'/name; q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act,Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.native_navigation import NativeCutNavigator
    from mdb_lanes.forecast_transaction import verify_transaction
    from mdb_lanes.tensor_archive import load_tensor_archive
    counts=dict.fromkeys(expected_counts(),0); entries=dict(blocks=0,target_blocks=0); target_counts={s['id']:0 for s in cases()}; active=dict(branch='setup',target=False); receipts=[]; rows=[]
    def event(kind,**kw):
        row=dict(kind=kind,seconds=time.monotonic()-started,counts=dict(counts),entries=dict(entries),target_counts=dict(target_counts),**kw)
        with (out/'timeline.jsonl').open('a') as f: f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_LOCAL_FUTURE '+json.dumps(row,sort_keys=True),flush=True)
    def wall():
        if time.monotonic()-started>c['worker_wall_s']: raise RuntimeError('finite local-delta campaign wall budget')
    def array(name,value):
        p=out/'arrays'/(name+'.npy'); p.parent.mkdir(exist_ok=True)
        if isinstance(value,torch.Tensor): value=value.detach().float().cpu().numpy()
        with p.open('xb') as f: np.save(f,np.asarray(value,np.float32),allow_pickle=False)
        return dict(path=p.relative_to(out).as_posix(),sha256=sha(p),shape=list(value.shape),dtype='float32')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available() or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction: raise ValueError('native numerical profile differs')
    root=Path(c['model_path']); inventory={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=c['checkpoint_sha256']: raise ValueError('checkpoint differs')
    event('model_load_start')
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None); pipe.set_progress_bar_config(disable=True); counts['loads']=1
    for component in (pipe.transformer,pipe.vae): component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu'); adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda'); block=pipe.transformer.transformer_blocks[1]
    def entered(module,args):
        entries['blocks']+=1
        if active['target']: entries['target_blocks']+=1; target_counts[active['case']]+=1
    handles=[b.register_forward_pre_hook(entered) for b in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    source=LocalStore(out/'source/store'); store=LocalStore(out/'store'); parents={k:source.load(v,device='cuda') for k,v in c['parents'].items()}; base=parents['base']; edited=parents['edited']
    for cut in parents.values():
        if cut.model_identity!=adapter.model_identity or any(cut.execution_point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=1).items()): raise ValueError('parent model/clock differs')
        store.save(cut)
    r=runpy.run_path(str(payload/'flux_reentry_reference.py')); math=runpy.run_path(str(payload/'flux_response_reference.py')); local=runpy.run_path(str(payload/'flux_local_delta_reference.py'))
    for cls,key in ((type(block),'Flux2TransformerBlock.source_sha256'),(type(block.attn.processor),'Flux2AttnProcessor.source_sha256')):
        if hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()!=r['SOURCE_CONTRACT'][key]: raise ValueError('native source differs')
    weights,_=runpy.run_path(str(payload/'flux_response_campaign.py'))['parameter_recipes'](block,root,inventory,r,out,payload)
    conv=lambda x:x.detach().float().cpu().numpy(); T0,T1=conv(base.payload['text']),conv(edited.payload['text']); inputs={}; forecasts={}; anchor_cuts={}; baseline_rgb={}
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
        context={k:tuple(conv(x) for x in v) if isinstance(v,tuple) else conv(v) for k,v in entry.payload['context'].items()}
        lc=local['prepare'](I,T,context,weights,stages,r); zero=local['predict'](lc,T)
        if any(not np.array_equal(zero['fields'][role]['exit'],stages[role+'.exit']) for role in ('image','text')): raise ValueError('zero-edit anchor closure differs')
        mc=math['prepare'](I,T,context,weights,r,heads=pipe.transformer.config.num_attention_heads)
        for spec in [s for s in cases() if s['anchor']==anchor]:
            wall(); name=spec['id']; newtext=T0.copy()
            for row,dose in zip((4,11),spec['row_doses']): newtext[:,row]=r['bf16'](T0[:,row].astype(float)+dose*(T1[:,row].astype(float)-T0[:,row].astype(float)))
            inputs[name]=dict(image=I,text=newtext); forecasts[name]={}
            for mode,prediction in [('local',local['predict'](lc,newtext)),('absolute',math['predict'](mc,newtext,compare_dense=False))]:
                forecasts[name][mode]=dict(fields={role:{key:array(name+'.'+mode+'.'+role+'.'+key,value) for key,value in values.items() if key=='exit' or mode=='local'} for role,values in prediction['fields'].items()},seconds=prediction['seconds'])
            event('calculated',case=name)
        del lc,mc,stages,zero
        baseline_rgb[anchor]=np.asarray(Image.open(out/'source/baselines'/(anchor+'.png')).convert('RGB')).copy()
    del weights
    save(out/'forecasts.json',forecasts)
    if entries!=dict(blocks=0,target_blocks=0): raise ValueError('native model computation before write seal')
    tx=make_transaction(out/'write-transaction',model_identity=adapter.model_identity,input_identity=identity(c),source_identity=identity(request['source_hashes']),wall_s=c['worker_wall_s'],scope='Historical half-dose native ports calibrate local laws; four new asymmetric working-text states, all changed writes NumPy. Native targets unobserved. Native encoder/prefix and image suffix/VAE supplied.')
    tx.predict('affected_writes',lambda _:dict(predictions=dict(forecasts=forecasts,counts_at_seal=dict(counts),entries_at_seal=dict(entries),target_counts=dict(target_counts))))
    write_seal=tx.seal(); event('writes_sealed',seal=write_seal)
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
        rgb=np.asarray(Image.open(p).convert('RGB')).copy(); row=dict(id=name,mode=mode,anchor=anchor,entry=entry,commits=commits,final=final.fingerprint,png=p.name,png_sha256=sha(p),readout=scene_readout(rgb,baseline_rgb[anchor]),target_counts_at_image=dict(target_counts))
        rows.append(row); save(out/('completed-'+name+'.'+mode+'.json'),row); event('image_complete',case=name,mode=mode); return row
    def candidate_futures(_):
        for anchor in ('half','half.image1'):
            active.update(branch=anchor+'.control',target=False); nav=NativeCutNavigator(Session.from_cut(adapter,anchor_cuts[anchor]),max_steps=53,on_receipt=receipt)
            row=finish(nav,anchor,'control',anchor,anchor_cuts[anchor].fingerprint)
            if row['png_sha256']!=c['anchors'][anchor]['png_sha256']: raise ValueError('historical native suffix/PNG replay differs')
        for spec in cases():
            name=spec['id']
            for mode in ('local','absolute'):
                active.update(branch=name+'.'+mode,target=False); scaffold=anchor_cuts[spec['anchor']]; nav=NativeCutNavigator(Session.from_cut(adapter,scaffold),max_steps=53,on_receipt=receipt)
                for role in ('image','text'):
                    record=forecasts[name][mode]['fields'][role]['exit']; value=np.load(out/record['path'],allow_pickle=False)
                    if sha(out/record['path'])!=record['sha256']: raise ValueError('sealed candidate bytes differ')
                    nav.apply(Act.replace(role,torch.from_numpy(value.copy()).to('cuda',torch.bfloat16),name=name+'.'+mode+'.'+role))
                installed=capture(nav)
                if any(describe(installed.payload[k])!=describe(scaffold.payload[k]) for k in scaffold.payload if k not in {'image','text'}): raise ValueError('candidate changed undeclared closure')
                finish(nav,name,mode,spec['anchor'],installed.fingerprint)
        return dict(rows=rows,counts=dict(counts))
    with tx: tx.execute('candidate-image-consumers',candidate_futures)
    image_predictions={row['id']+'.'+row['mode']:dict(png_sha256=row['png_sha256'],readout=row['readout']) for row in rows if row['mode']!='control'}
    save(out/'image-forecasts.json',image_predictions)
    if entries['target_blocks']!=0 or any(target_counts.values()) or counts['images']!=10: raise ValueError('target observed before image forecast seal')
    future=make_transaction(out/'image-transaction',model_identity=adapter.model_identity,input_identity=identity(c),source_identity=identity(dict(sources=request['source_hashes'],write_seal=write_seal)),wall_s=c['worker_wall_s'],scope='Candidate RGB futures and fixed-source-region target/collateral readouts observed; four native target joint1 writes and futures still unobserved. Native suffix work used to form image forecasts is fully counted.')
    future.predict('image_effects',lambda _:dict(predictions=dict(forecasts=image_predictions,counts_at_seal=dict(counts),entries_at_seal=dict(entries),target_counts=dict(target_counts))))
    image_seal=future.seal(); event('image_forecasts_sealed',seal=image_seal); native={}
    def run_target(spec):
        name=spec['id']; active.update(branch=name+'.native',case=name,target=False); nav=NativeCutNavigator(Session.from_cut(adapter,base),max_steps=54,on_receipt=receipt)
        for role in ('image','text'): nav.apply(Act.replace(role,torch.from_numpy(inputs[name][role].copy()).to('cuda',torch.bfloat16),name=name+'.native.'+role))
        entry=capture(nav); ports={}
        def ah(m,args,output): ports.update(attention_image=output[0].detach().clone(),attention_text=output[1].detach().clone())
        def fh(role):
            def hook(m,args,output): ports['mlp_input_'+role]=args[0].detach().clone(); ports['mlp_'+role]=output.detach().clone()
            return hook
        hs=[block.attn.register_forward_hook(ah),block.ff.register_forward_hook(fh('image')),block.ff_context.register_forward_hook(fh('text'))]; active['target']=True
        try: nav.step()
        finally:
            active['target']=False
            for handle in hs: handle.remove()
        exit=capture(nav); native[name]=dict(entry=entry.fingerprint,exit=exit.fingerprint,fields={k:array(name+'.native.'+k,v) for k,v in ports.items()})
        save(out/('native-'+name+'.json'),native[name]); finish(nav,name,'native',spec['anchor'],entry.fingerprint)
    with future: future.execute('four-unseen-native-targets',lambda _:[run_target(s) for s in cases()])
    for handle in handles: handle.remove()
    save(out/'native.json',native); save(out/'rows.json',rows); save(out/'receipts.json',receipts)
    if counts!=expected_counts() or entries!=dict(blocks=676,target_blocks=4) or any(v!=1 for v in target_counts.values()): raise ValueError('physical execution census differs')
    summary=dict(schema='mdb-local-delta-future-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),counts=counts,entries=entries,target_counts=target_counts,
        write_seal=write_seal,image_seal=image_seal,write_transaction=verify_transaction(tx.root),image_transaction=verify_transaction(future.root),worker_seconds=time.monotonic()-started,mechanics_status='pending-independent-audit',terminal_status='unassessed')
    save(out/'summary.json',summary); save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__': worker()
