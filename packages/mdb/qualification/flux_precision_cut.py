"""Two native joint blocks localize a receiver-effect numerical discrepancy."""
import hashlib
import inspect
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def cases():
    return [dict(id='half',receiver='base'),dict(id='half.image1',receiver='edited')]


def expected_counts():
    return dict(loads=1,native_transitions=2,acts=4,receipts=6,images=0,encodes=0,noise_draws=0)


def make_transaction(root,*,model_identity,input_identity,source_identity,wall_s):
    from mdb.forecast_transaction import ForecastTransaction
    return ForecastTransaction(root,model_identity=model_identity,input_identity=input_identity,source_identity=source_identity,
        frontier=dict(kind='architectural',scope='Two historical half-dose inputs. Prior complete outputs are known; new internal stage ports are unobserved. No held-out generalization claim.'),
        execution_scope='CPU NumPy mathematical references with legacy versus extra QK normalized BF16 boundary; actual native CUDA BF16 modules',budgets=dict(max_native_calls=1,max_wall_ms=wall_s*1000))


def check_transaction_contract(root):
    """Exercise the actual installed seal/execute API before acquiring a model."""
    from mdb.forecast_transaction import verify_transaction
    tx=make_transaction(root,model_identity='mechanics-only',input_identity='no-model-input',source_identity='no-model-source',wall_s=30)
    tx.predict('mechanics',lambda _:dict(predictions=dict(mechanics_only=True)))
    tx.seal()
    with tx:tx.execute('no-native-call',lambda _:dict(mechanics_only=True))
    result=verify_transaction(tx.root)
    if not result['verified']or result['phase']!='closed':raise ValueError('transaction API mechanics failed')
    return result


def capture_ports(block,native_module,text_rows,remember,entries,step):
    hs=[];heads=block.attn.heads;dim=block.attn.to_q.out_features//heads
    def hook(key,input=False,reshape=False):
        def f(module,args,output):
            value=args[0]if input else output
            if reshape:value=value.reshape(1,value.shape[1],heads,dim)
            remember(key,value)
        return f
    for role,ff,norm1,norm2,names,norms in[
        ('image',block.ff,block.norm1,block.norm2,('to_q','to_k','to_v'),('norm_q','norm_k')),
        ('text',block.ff_context,block.norm1_context,block.norm2_context,('add_q_proj','add_k_proj','add_v_proj'),('norm_added_q','norm_added_k'))]:
        hs.extend([norm1.register_forward_hook(hook(role+'.norm1')),norm2.register_forward_hook(hook(role+'.norm2')),norm2.register_forward_hook(hook(role+'.after_attention',input=True)),ff.register_forward_hook(hook(role+'.mlp_input',input=True)),ff.register_forward_hook(hook(role+'.mlp')),
            ff.linear_in.register_forward_hook(hook(role+'.mlp_paired')),ff.act_fn.gate_fn.register_forward_hook(hook(role+'.mlp_silu')),ff.act_fn.register_forward_hook(hook(role+'.mlp_activation'))])
        for label,module_name in zip(('q','k','v'),names):hs.append(getattr(block.attn,module_name).register_forward_hook(hook(role+'.'+label+'_projection',reshape=True)))
        for label,module_name in zip(('q','k'),norms):hs.append(getattr(block.attn,module_name).register_forward_hook(hook(role+'.'+label+'_normalized')))
    def ah(m,args,output):remember('image.attention',output[0]);remember('text.attention',output[1])
    def ap(m,args,kwargs):remember('image.modulated',kwargs['hidden_states']);remember('text.modulated',kwargs['encoder_hidden_states'])
    hs.extend([block.attn.register_forward_hook(ah),block.attn.register_forward_pre_hook(ap,with_kwargs=True)])
    original_dispatch=native_module.dispatch_attention_fn
    def dispatch(q,k,v,*args,**kwargs):
        entries['dispatch']+=1
        for role,sl in(('text',slice(0,text_rows)),('image',slice(text_rows,None))):remember(role+'.q_positioned',q[:,sl]);remember(role+'.k_positioned',k[:,sl])
        output=original_dispatch(q,k,v,*args,**kwargs);remember('attention.raw',output.flatten(2,3));return output
    native_module.dispatch_attention_fn=dispatch
    try:return step()
    finally:
        native_module.dispatch_attention_fn=original_dispatch
        for handle in hs:handle.remove()


def worker():
    payload=Path.cwd();h=runpy.run_path(str(payload/'flux_binding_minimal.py'));sha,save,identity=[h[k]for k in('sha','save','identity')]
    config,request=[json.loads(Path(os.environ[k]).read_text())for k in('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config']or any(sha(payload/n)!=v for n,v in request['source_hashes'].items()):raise ValueError('sealed local cut changed')
    for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    out=Path('binding-minimal-result');out.mkdir();save(out/'plan.json',config);save(out/'request.json',request);started=time.monotonic()
    def phase(name,**kw):
        record=dict(phase=name,seconds=time.monotonic()-started,**kw)
        with(out/'setup.jsonl').open('a')as f:f.write(json.dumps(record,sort_keys=True)+'\n')
        print('MDB_PRECISION_SETUP '+json.dumps(record,sort_keys=True),flush=True)
    phase('verify_source_start',files=len(config['input_files']))
    for name,item in config['input_files'].items():
        p=Path(config['input_root'])/item['source']
        if sha(p)!=item['sha256']or p.stat().st_size!=item['bytes']:raise ValueError('source custody differs')
        q=out/'source'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    phase('verify_source_complete',bytes=sum(v['bytes']for v in config['input_files'].values()))
    phase('framework_import_start')
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers import transformer_flux2 as native_module
    from saturn_pub import Act,Session
    from saturn_pub.store import LocalStore
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.values import describe
    from mdb.native_navigation import NativeCutNavigator
    from mdb.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb.tensor_archive import write_tensor_archive,load_tensor_archive
    from mdb.forward_recording import tensor_descriptor
    from mdb.attention_partition import dense_attention
    phase('framework_import_complete')
    counts=dict.fromkeys(expected_counts(),0);entries=dict(blocks=0,dispatch=0);receipts=[];archive_cache={}
    def event(kind,**kw):
        v=dict(kind=kind,seconds=time.monotonic()-started,counts=dict(counts),entries=dict(entries),**kw)
        with(out/'timeline.jsonl').open('a')as f:f.write(json.dumps(v,sort_keys=True)+'\n')
        print('MDB_PRECISION '+json.dumps(v,sort_keys=True),flush=True)
    def wall():
        if time.monotonic()-started>config['worker_wall_s']:raise RuntimeError('finite precision cut budget')
    def array(value):
        if not isinstance(value,torch.Tensor):
            value=np.asarray(value,np.float32)
            if not np.array_equal(r['bf16'](value),value):raise ValueError('non-BF16 reference archive')
            value=torch.from_numpy(value.copy()).to(torch.bfloat16)
        value=value.detach().cpu().contiguous();descriptor=tensor_descriptor(value);key=identity(descriptor)
        if key not in archive_cache:archive_cache[key]=write_tensor_archive(out/'arrays'/(key+'.npy'),value,root=out)
        return archive_cache[key]
    def read(record):return load_tensor_archive(out,record)
    def error(p,n):
        p,n=p.astype(np.float64),n.astype(np.float64);diff=p-n;nr=float(np.sqrt(np.mean(n*n)));pr=float(np.sqrt(np.mean(p*p)))
        return dict(exact=bool(np.array_equal(p,n)),different=int(np.count_nonzero(p!=n)),scalars=int(n.size),rms=float(np.sqrt(np.mean(diff*diff))),reference_rms=nr,
            relative_rms=None if nr==0 else float(np.sqrt(np.mean(diff*diff)))/nr,max_abs=float(np.max(np.abs(diff))),cosine=None if nr==0 or pr==0 else float(np.mean(p*n)/(pr*nr)))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    phase('checkpoint_inventory_start')
    root=Path(config['model_path']);inventory={p.relative_to(root).as_posix():sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.suffix in{'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']:raise ValueError('model inventory differs')
    phase('checkpoint_inventory_complete',files=len(inventory));phase('model_load_start')
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts['loads']=1;pipe.set_progress_bar_config(disable=True)
    for component in(pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda');block=pipe.transformer.transformer_blocks[1]
    phase('model_load_complete')
    handles=[b.register_forward_pre_hook(lambda m,a:entries.__setitem__('blocks',entries['blocks']+1))for b in[*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    source=LocalStore(out/'source/store');store=LocalStore(out/'store');parents={k:source.load(v,device='cuda')for k,v in config['parents'].items()}
    for cut in parents.values():
        if cut.model_identity!=adapter.model_identity or any(cut.execution_point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=1).items()):raise ValueError('prefix identity/clock differs')
        store.save(cut)
    r=runpy.run_path(str(payload/'flux_reentry_reference.py'));math=runpy.run_path(str(payload/'flux_response_reference.py'));B=r['bf16']
    for cls,key in((type(block),'Flux2TransformerBlock.source_sha256'),(type(block.attn.processor),'Flux2AttnProcessor.source_sha256')):
        if hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()!=r['SOURCE_CONTRACT'][key]:raise ValueError('native source contract differs')
    save(out/'native-contract.json',dict(qk_norm_class=type(block.attn.norm_q).__module__+'.'+type(block.attn.norm_q).__name__,qk_forward=inspect.getsource(type(block.attn.norm_q).forward),torch=torch.__version__,
        scope='Actual FLUX uses Torch RMSNorm. Diffusers RMSNorm wrapper and CPU/CUDA arithmetic must not be conflated.'))
    weights,_=runpy.run_path(str(payload/'flux_response_campaign.py'))['parameter_recipes'](block,root,inventory,r,out,payload)
    phase('parameter_recipes_complete',parameters=len(weights))
    conv=lambda x:x.detach().float().cpu().numpy();base=parents['base'];edited=parents['edited'];T0,T1=conv(base.payload['text']),conv(edited.payload['text']);T=T0.copy();T[:,[4,11]]=B(T0[:,[4,11]].astype(float)+.5*(T1[:,[4,11]].astype(float)-T0[:,[4,11]]))
    context={k:tuple(conv(x)for x in v)if isinstance(v,tuple)else conv(v)for k,v in base.payload['context'].items()};inputs={};forecasts={}
    for spec in cases():
        name=spec['id'];I=conv(parents[spec['receiver']].payload['image']);inputs[name]=dict(image=I,text=T)
        forecasts[name]={}
        for mode in('legacy','staged'):
            wall();stages={}
            def observe(key,value):
                expected=I.shape[1]if key.startswith('image.')else T.shape[1]if key.startswith('text.')else I.shape[1]+T.shape[1]
                if value.shape[1]==expected:stages[key]=array(value)
            cache=math['prepare'](I,T,context,weights,r,heads=pipe.transformer.config.num_attention_heads,qk_rounding=mode,observer=observe)
            predicted=math['predict'](cache,T);fields={role:{k:array(v)for k,v in values.items()}for role,values in predicted.pop('fields').items()}
            forecasts[name][mode]=dict(**predicted,fields=fields,stages=stages,cache_seconds=cache['seconds']);del cache
            event('calculated',case=name,mode=mode,stages=len(stages))
    save(out/'forecasts.json',forecasts)
    tx=make_transaction(out/'transaction',model_identity=adapter.model_identity,input_identity=identity(config),source_identity=identity(request['source_hashes']),wall_s=config['worker_wall_s'])
    tx.predict('local_precision',lambda _:dict(predictions=dict(forecasts=forecasts,counts_at_seal=dict(counts),entries_at_seal=dict(entries),candidate='Extra BF16 boundary is unconfirmed; CPU Torch RMSNorm matches legacy while Diffusers wrapper matches staged. Native CUDA trace decides.')))
    seal=tx.seal();event('sealed',seal=seal);native={};conditional={}
    def receipt(value):
        body=value.to_dict();store.receipt(value);receipts.append(body);counts['receipts']+=1;counts['native_transitions'if body['operation']=='continue'else'acts']+=body.get('steps',1)
        save(out/('receipt-'+str(len(receipts))+'.json'),body)
    def capture(nav):c=nav.session.capture(retain=False);store.save(c);return c
    def run_native(spec):
        wall();name=spec['id'];nav=NativeCutNavigator(Session.from_cut(adapter,base),max_steps=1,on_receipt=receipt)
        for slot in('image','text'):nav.apply(Act.replace(slot,torch.from_numpy(inputs[name][slot].copy()).to('cuda',torch.bfloat16),name=name+'.input.'+slot))
        entry=capture(nav);prior=source.load(config['history'][name]['entry'],device='cuda')
        if describe(entry.payload)!=describe(prior.payload):raise ValueError('historical input replay differs')
        ports={}
        def remember(key,value):ports[key]=value.detach().clone()
        capture_ports(block,native_module,T.shape[1],remember,entries,nav.step)
        exit=capture(nav);prior=source.load(config['history'][name]['exit'],device='cuda')
        if describe(exit.payload)!=describe(prior.payload):raise ValueError('historical target replay differs')
        for role in('image','text'):remember(role+'.exit',exit.payload[role])
        records={k:array(v)for k,v in ports.items()};ports.clear();native[name]=dict(entry=entry.fingerprint,exit=exit.fingerprint,stages=records,full_payload_replay_exact=True)
        save(out/('native-'+name+'.json'),native[name]);event('target_captured',case=name,stages=len(records))
        probe={}
        for role in('image','text'):
            for label,norm_name in zip(('q','k'),('norm_q','norm_k')if role=='image'else('norm_added_q','norm_added_k')):
                x=read(records[role+'.'+label+'_projection']);actual=read(records[role+'.'+label+'_normalized']);gamma=weights['attn.'+norm_name+'.weight']
                for mode in('legacy','staged'):
                    value=math['qk_rms'](x,gamma,r,rounding=mode);probe[role+'.'+label+'.'+mode]=dict(input=records[role+'.'+label+'_projection'],output=array(value),native=records[role+'.'+label+'_normalized'],error=error(value,actual))
        conditional[name]=probe;save(out/('conditional-'+name+'.json'),probe);event('conditional_probes_complete',case=name)
    with tx:
        tx.execute('two-local-native-targets',lambda _:[run_native(s)for s in cases()])
    save(out/'native.json',native);save(out/'conditional.json',conditional);save(out/'receipts.json',receipts)
    if counts!=expected_counts()or entries!=dict(blocks=2,dispatch=2):raise ValueError('physical/instrument census differs')
    if any(set(native[name]['stages'])!=set(forecasts[name][mode]['stages'])for name in native for mode in('legacy','staged')):raise ValueError('stage capture incomplete')
    summary=dict(schema='mdb-precision-cut-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],counts=counts,entries=entries,seal=seal,transaction=verify_transaction(tx.root),output_root=str(scratch),worker_seconds=time.monotonic()-started,mechanics_status='pending-independent-audit',terminal_status='unassessed')
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p)for p in sorted(out.rglob('*'))if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=summary['request_sha256'],output_root=str((scratch/out).resolve()),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__':worker()
