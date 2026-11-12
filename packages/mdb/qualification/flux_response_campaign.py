"""Seal a reusable changed-row computation, then test new states and native RGB."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import shutil
import time


def cases():
    return [dict(id='base',rows=[],dose=0.,receiver='base',categorical=['red','blue'],role='historical control'),
        dict(id='row4',rows=[4],dose=1.,receiver='base',categorical=['blue','blue'],role='new working-row composition'),
        dict(id='row11',rows=[11],dose=1.,receiver='base',categorical=['red','red'],role='new working-row composition'),
        dict(id='half',rows=[4,11],dose=.5,receiver='base',categorical=None,role='new dose; signed donor effect predicted positive'),
        dict(id='half.image1',rows=[4,11],dose=.5,receiver='edited',categorical=None,role='new receiving state x dose; signed donor effect predicted positive')]


def expected_counts(n):
    return dict(loads=1,encodes=0,noise_draws=0,native_transitions=107*n,acts=4*n,receipts=9*n,images=2*n)


def worker():
    payload=Path.cwd();helper=runpy.run_path(str(payload/'flux_binding_minimal.py'));sha,save,identity=[helper[k]for k in('sha','save','identity')]
    config,request=[json.loads(Path(os.environ[k]).read_text())for k in('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config']or config['cases']!=cases()or any(sha(payload/n)!=v for n,v in request['source_hashes'].items()):raise ValueError('sealed response program differs')
    for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    out=Path('binding-minimal-result');out.mkdir();save(out/'plan.json',config);save(out/'request.json',request)
    for name,record in config['input_files'].items():
        p=Path(config['input_root'])/name
        if p.stat().st_size!=record['bytes']or sha(p)!=record['sha256']:raise ValueError('prefix custody differs')
        q=Path('input-data')/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act,Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb.native_navigation import NativeCutNavigator
    started=time.monotonic();counts=dict.fromkeys(expected_counts(len(cases())),0);entries=dict(blocks=0,target_blocks=0)
    active=dict(branch='setup',at_target=False);receipts=[];rows=[];target_counts={c['id']:0 for c in cases()}
    def event(kind,**kw):
        v=dict(kind=kind,seconds=time.monotonic()-started,counts=dict(counts),entries=dict(entries),**kw)
        with(out/'timeline.jsonl').open('a')as f:f.write(json.dumps(v,sort_keys=True)+'\n')
        print('MDB_RESPONSE '+json.dumps(v,sort_keys=True),flush=True)
    def wall():
        if time.monotonic()-started>config['worker_wall_s']:raise RuntimeError('finite response experiment wall budget')
    def array(name,value):
        p=out/'arrays'/(name+'.npy');p.parent.mkdir(exist_ok=True)
        if isinstance(value,torch.Tensor):value=value.detach().float().cpu().numpy()
        np.save(p,np.asarray(value,np.float32),allow_pickle=False)
        return dict(path=p.relative_to(out).as_posix(),sha256=sha(p),shape=list(value.shape),dtype='float32')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise ValueError('CUDA admission required')
    root=Path(config['model_path']);inventory={p.relative_to(root).as_posix():sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.suffix in{'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']:raise ValueError('checkpoint inventory differs')
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None)
    pipe.set_progress_bar_config(disable=True);counts['loads']=1
    for component in(pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    def entered(module,args):
        entries['blocks']+=1
        if active['at_target']:
            entries['target_blocks']+=1;target_counts[active['case']]+=1
    handles=[b.register_forward_pre_hook(entered)for b in[*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    original=LocalStore('input-data/store');store=LocalStore(out/'store');parents={k:original.load(v,device='cuda')for k,v in config['parents'].items()}
    for c in parents.values():
        if c.model_identity!=adapter.model_identity or any(c.execution_point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=1).items()):raise ValueError('source model/clock differs')
        store.save(c)
    base=parents['base'];edited=parents['edited']
    if any(describe(base.payload[k])!=describe(edited.payload[k])for k in base.payload if k not in{'image','text','conditioning'}):raise ValueError('source closure differs')
    r=runpy.run_path(str(payload/'flux_reentry_reference.py'));math=runpy.run_path(str(payload/'flux_response_reference.py'))
    # Explicit request-sealed helper; loading it does not invoke its worker.
    commit_steps=runpy.run_path(str(payload/'flux_response_validate.py'))['steps_to_commit']
    block=pipe.transformer.transformer_blocks[1]
    import inspect
    for cls,key in((type(block),'Flux2TransformerBlock.source_sha256'),(type(block.attn.processor),'Flux2AttnProcessor.source_sha256')):
        if hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()!=r['SOURCE_CONTRACT'][key]:raise ValueError('qualified native mathematical source differs: '+key)
    if block.norm1.eps!=1e-6 or block.attn.norm_q.eps!=1e-6:raise ValueError('qualified normalization differs')
    weights,recipes=parameter_recipes(block,root,inventory,r,out,payload)
    convert=lambda x:x.detach().float().cpu().numpy()
    context={k:tuple(convert(x)for x in v)if isinstance(v,tuple)else convert(v)for k,v in base.payload['context'].items()}
    T0,T1=convert(base.payload['text']),convert(edited.payload['text']);inputs={};forecasts={};caches={};cache_costs={}
    for spec in cases():
        wall();name=spec['id'];receiver=spec['receiver']
        I=convert(parents[receiver].payload['image']);T=T0.copy();ix=spec['rows'];dose=spec['dose']
        if dose==1:T[:,ix]=T1[:,ix]
        elif ix:T[:,ix]=r['bf16'](T0[:,ix].astype(np.float64)+dose*(T1[:,ix].astype(np.float64)-T0[:,ix]))
        inputs[name]=dict(image=I,text=T)
        if receiver not in caches:
            caches[receiver]=math['prepare'](I,T0,context,weights,r,heads=pipe.transformer.config.num_attention_heads)
            cache_costs[receiver]=caches[receiver]['seconds'];event('cache_prepared',receiver=receiver,cache_seconds=cache_costs[receiver])
        prediction=math['predict'](caches[receiver],T)
        fields={role:{k:array(name+'.predicted.'+role+'.'+k,v)for k,v in values.items()}for role,values in prediction.pop('fields').items()}
        forecasts[name]=dict(**prediction,fields=fields,input_fields={k:array(name+'.input.'+k,v)for k,v in inputs[name].items()})
        event('forecast_calculated',case=name,predictor_seconds=prediction['seconds'])
    save(out/'forecasts.json',forecasts)
    # Parameter and attention caches are calculated before any native target;
    # retaining static constants would duplicate the pinned checkpoint.
    del caches,weights
    def receipt(value):
        wall();body=value.to_dict();store.receipt(value);receipts.append(dict(branch=active['branch'],body=body));counts['receipts']+=1
        counts['native_transitions'if body['operation']=='continue'else'acts']+=body.get('steps',1)
        if any(counts[k]>expected_counts(len(cases()))[k]for k in('native_transitions','acts','receipts')):raise RuntimeError('physical operation budget exceeded')
    def capture(n):c=n.session.capture(retain=False);store.save(c);return c
    def apply(n,slot,value,label):
        before=capture(n);act=Act.replace(slot,value,name=label);rec=n.apply(act);after=capture(n)
        if describe(after.payload)!=describe({**before.payload,slot:value}):raise ValueError('undeclared state change')
        return dict(slot=slot,before=before.fingerprint,after=after.fingerprint,receipt=rec.to_dict())
    tx=ForecastTransaction(out/'transaction',model_identity=adapter.model_identity,input_identity=identity(config),source_identity=identity(request['source_hashes']),
        frontier=dict(kind='architectural',observed='Audited step2/joint1 source cuts and pinned weights. Baseline historical future known; four new working-row/dose/receiver inputs. No current native target inputs or returns used by predictor.'),
        execution_scope='Native CUDA BF16 B1; independent predictor is CPU NumPy FP64 reductions with explicit BF16 boundaries',budgets=dict(max_native_calls=1,max_wall_ms=config['worker_wall_s']*1000))
    tx.predict('response_program',lambda _:dict(predictions=dict(config=config,forecasts=forecasts,cache_seconds=cache_costs,counts_at_seal=dict(counts),entries_at_seal=dict(entries))))
    seal=tx.seal();event('sealed',seal=seal)
    control_exit=None
    def finish(n,label,mode,actions,entry,**extra):
        commits=[]
        while n.point['logical_step']<4:
            point=n.point;n.step(commit_steps(point,len(pipe.transformer.transformer_blocks),len(pipe.transformer.single_transformer_blocks)))
            if n.point['logical_step']!=point['logical_step']+1:raise ValueError('group did not end at next commit')
            c=capture(n);commits.append(dict(step=n.point['logical_step'],cut=c.fingerprint))
        final=capture(n);p=out/(label+'.'+mode+'.png');adapter.decode(n.session,height=512,width=512).save(p);counts['images']+=1
        row=dict(id=label,mode=mode,entry=entry,actions=actions,commits=commits,final=final.fingerprint,png=p.name,png_sha256=sha(p),**extra)
        rows.append(row);save(out/('completed-'+label+'.'+mode+'.json'),row);event('image_complete',case=label,mode=mode)
        return row
    def native(spec):
        nonlocal control_exit
        name=spec['id'];active.update(branch=name+'.native',case=name)
        n=NativeCutNavigator(Session.from_cut(adapter,base),max_steps=54,on_receipt=receipt)
        actions=[apply(n,k,torch.from_numpy(inputs[name][k].copy()).to(device='cuda',dtype=torch.bfloat16),name+'.input.'+k)for k in('image','text')]
        entry=capture(n);ports={}
        def ah(m,args,output):ports.update(attention_image=output[0].detach().clone(),attention_text=output[1].detach().clone())
        def fh(role):
            def hook(m,args,output):ports['mlp_input_'+role]=args[0].detach().clone();ports['mlp_'+role]=output.detach().clone()
            return hook
        hs=[block.attn.register_forward_hook(ah),block.ff.register_forward_hook(fh('image')),block.ff_context.register_forward_hook(fh('text'))]
        active['at_target']=True
        try:n.step()
        finally:
            active['at_target']=False
            for h in hs:h.remove()
        exit=capture(n)
        if name=='base':control_exit=exit
        record=dict(entry=entry.fingerprint,exit=exit.fingerprint,fields={k:array(name+'.native.'+k,v)for k,v in ports.items()})
        save(out/('ports-'+name+'.json'),record)
        finish(n,name,'native',actions,entry.fingerprint,target=record)
    def predicted(spec):
        name=spec['id'];active.update(branch=name+'.predicted',case=name)
        n=NativeCutNavigator(Session.from_cut(adapter,control_exit),max_steps=53,on_receipt=receipt)
        actions=[]
        for role in('image','text'):
            record=forecasts[name]['fields'][role]['exit'];v=np.load(out/record['path'],allow_pickle=False)
            actions.append(apply(n,role,torch.from_numpy(v.copy()).to(device='cuda',dtype=torch.bfloat16),name+'.predicted.'+role))
        installed=capture(n)
        finish(n,name,'predicted',actions,installed.fingerprint,clock_scaffold=control_exit.fingerprint,target_entries_before_image=target_counts[name])
    with tx:
        def execute(_):
            native(cases()[0]);predicted(cases()[0])
            for spec in cases()[1:]:predicted(spec);native(spec)
            return dict(rows=rows,counts=dict(counts),entries=dict(entries))
        tx.execute('native-validation-and-predicted-write-futures',execute)
    save(out/'rows.json',rows);save(out/'receipts.json',receipts)
    if counts!=expected_counts(len(cases()))or entries!=dict(blocks=97*len(cases()),target_blocks=len(cases())):raise ValueError('physical entry/receipt census differs')
    if next(rw for rw in rows if rw['id']=='base'and rw['mode']=='native')['png_sha256']!=config['baseline_png_sha256']:raise ValueError('historical baseline differs')
    summary=dict(schema='mdb-response-program-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],counts=counts,entries=entries,
        seal=seal,transaction=verify_transaction(tx.root),cache_seconds=cache_costs,worker_seconds=time.monotonic()-started,output_root=str(scratch),mechanics_status='pending-independent-audit',terminal_status='unassessed')
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p)for p in sorted(out.rglob('*'))if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str((scratch/out).resolve()),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def parameter_recipes(block,root,inventory,reference,out,payload):
    import base64,io,struct
    import numpy as np
    import torch
    from mdb.parameter_artifacts import validate_manifest
    name='transformer/diffusion_pytorch_model.safetensors';path=root/name
    with path.open('rb')as f:header_bytes=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(header_bytes))
    weights={};artifacts=[]
    for port,p in block.named_parameters():
        full='transformer_blocks.1.'+port;spec=header[full]
        if spec['dtype']!='BF16'or spec['shape']!=list(p.shape):raise ValueError('parameter header differs')
        bits=p.detach().view(torch.uint16).cpu().numpy().astype('<u2',copy=False);raw=bits.tobytes()
        with path.open('rb')as f:
            first,last=spec['data_offsets'];f.seek(8+header_bytes+first);checkpoint_bytes=f.read(last-first)
        if hashlib.sha256(checkpoint_bytes).digest()!=hashlib.sha256(raw).digest():raise ValueError('native parameter differs from pinned checkpoint tensor')
        prefix=io.BytesIO();np.lib.format.write_array_header_1_0(prefix,dict(descr='<u2',fortran_order=False,shape=bits.shape));prefix=prefix.getvalue()
        artifacts.append(dict(path='parameters/'+full+'.npy',bytes=len(prefix)+len(raw),sha256=hashlib.sha256(prefix+raw).hexdigest(),npy_prefix_b64=base64.b64encode(prefix).decode(),
            parameter_ports=[port],checkpoint_tensor_aliases=[full],source=dict(file=name,file_sha256=inventory[name],tensor=full,**spec,payload_sha256=hashlib.sha256(raw).hexdigest())))
        weights[port]=(bits.astype(np.uint32)<<16).view(np.float32)
    reference['_weights'](weights,reference['JOINT_LINEAR_NAMES'],reference['JOINT_NORM_NAMES'])
    manifest=validate_manifest(dict(schema='mdb-parameter-artifacts-v1',artifacts=artifacts,defaults=dict(checkpoint_root=str(root),artifact_root='.',ssh_host='beast',python='python3'),
        provenance=dict(scope='Joint1 parameter artifacts described but never persistently exported. Native parameter bits bound to pinned checkpoint file/name/header; retrieval separately verifies byte hashes.')))
    (out/'parameter-artifacts.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');shutil.copyfile(payload/'retrieve_weights.py',out/'retrieve_weights.py')
    return weights,manifest


def cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('source',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
    helper=runpy.run_path(str(Path(__file__).with_name('stage_flux_response.py')))
    print(json.dumps(helper['submit'if a.submit else'stage'](a.run.resolve(),a.source.resolve()),sort_keys=True))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:cli()
