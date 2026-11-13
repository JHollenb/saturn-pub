"""Qualify real image-prefix reuse with native affected-state computation."""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
import runpy
import shutil
import time
from pathlib import Path


@contextmanager
def reuse_prefix(bindings,check,events):
    """Scoped interception; original forward bodies are never invoked on hits.

    The caller binds checkpoint/context/clock and native operand geometry. The
    checker validates the complete actual call, including kwargs, before reuse.
    """
    restore=[]
    try:
        for name,module,expected,output in bindings:
            owned='forward' in module.__dict__; original=module.forward
            def cached(*args,_name=name,_expected=expected,_output=output,**kwargs):
                started=time.monotonic(); check(_name,args,kwargs,_expected)
                events.append(dict(module=_name,input_check_seconds=time.monotonic()-started,original_forward_executed=False))
                return _output.clone()
            restore.append((module,owned,original)); module.forward=cached
        yield
    finally:
        for module,owned,original in reversed(restore):
            if owned: module.forward=original
            else: delattr(module,'forward')


def worker():
    payload=Path.cwd(); h=runpy.run_path(str(payload/'flux_binding_minimal.py')); sha,save,identity=[h[k] for k in ('sha','save','identity')]
    request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text()); c=request['config']
    if any(sha(payload/n)!=v for n,v in request['source_hashes'].items()): raise ValueError('source changed')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']; out=scratch/'binding-minimal-result'; out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',c); save(out/'request.json',request); started=time.monotonic()
    def event(kind,**kw):
        row=dict(kind=kind,seconds=time.monotonic()-started,**kw)
        with (out/'timeline.jsonl').open('a') as f: f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_INCREMENTAL '+json.dumps(row,sort_keys=True),flush=True)
    for name,item in c['input_files'].items():
        p=Path(c['input_roots'][item['root']])/item['source']
        if sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']: raise ValueError('source custody differs')
        q=out/'source'/name; q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
    shutil.copyfile(out/'source/prefixes/parameter-artifacts.json',out/'parameter-artifacts.json')
    shutil.copyfile(payload/'retrieve_weights.py',out/'retrieve_weights.py')
    event('framework_import_start')
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.native_navigation import NativeCutNavigator
    from mdb_lanes.tensor_archive import load_tensor_archive
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available() or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction: raise ValueError('native numerical profile differs')
    root=Path(c['model_path']); inventory={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=c['checkpoint_sha256']: raise ValueError('checkpoint differs')
    event('model_load_start')
    pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None); pipe.set_progress_bar_config(disable=True)
    for component in (pipe.transformer,pipe.vae): component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu'); adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda'); block=pipe.transformer.transformer_blocks[1]
    r=runpy.run_path(str(payload/'flux_reentry_reference.py'))
    for cls,key in ((type(block),'Flux2TransformerBlock.source_sha256'),(type(block.attn.processor),'Flux2AttnProcessor.source_sha256')):
        if hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()!=r['SOURCE_CONTRACT'][key]: raise ValueError('native source differs')
    if block.norm1.eps!=1e-6 or block.attn.norm_q.eps!=1e-6 or block.attn.norm_k.eps!=1e-6: raise ValueError('normalization law differs')
    targets=LocalStore(out/'source/targets/store'); prefixes=LocalStore(out/'source/prefixes/store'); store=LocalStore(out/'store')
    reports=[]; receipts=[]; entries=dict(blocks=0); module_entries={key:0 for key in ('norm1','to_q','to_k','to_v','norm_q','norm_k')}
    modules=dict(norm1=block.norm1,**{key:getattr(block.attn,key) for key in module_entries if key!='norm1'})
    def count(name):
        def hook(m,args): module_entries[name]+=1
        return hook
    handles=[module.register_forward_pre_hook(count(name)) for name,module in modules.items()]
    handles.extend(b.register_forward_pre_hook(lambda m,a:entries.__setitem__('blocks',entries['blocks']+1)) for b in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks])
    def receipt(value):
        body=value.to_dict(); store.receipt(value); receipts.append(body); save(out/('receipt-'+str(len(receipts))+'.json'),body)
    event('model_ready')
    for spec in c['cases']:
        entry=targets.load(spec['entry'],device='cuda'); expected=targets.load(spec['exit'],device='cuda'); prefix=prefixes.load(spec['prefix_entry'],device='cuda')
        if entry.model_identity!=prefix.model_identity or entry.model_identity!=adapter.model_identity or entry.execution_point!=prefix.execution_point: raise ValueError('cache identity/clock mismatch')
        if any(entry.execution_point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=1).items()): raise ValueError('cache boundary mismatch')
        if set(entry.payload)!=set(prefix.payload) or any(describe(entry.payload[k])!=describe(prefix.payload[k]) for k in entry.payload if k!='text'): raise ValueError('unchanged image-prefix receiving closure mismatch')
        stages=spec['stages']
        def tensor(key):
            record=stages[key]; value=torch.from_numpy(load_tensor_archive(out/'source/prefixes',record).copy()).to('cuda',torch.bfloat16)
            if list(value.stride())!=record['native_stride']: raise ValueError('cached output native layout differs')
            return value
        mod=tensor('image.modulated'); q,k,v=[tensor('image.'+name+'_projection') for name in ('q','k','v')]
        norm1=tensor('image.norm1'); qn=tensor('image.q_normalized'); kn=tensor('image.k_normalized')
        bindings=[('norm1',modules['norm1'],prefix.payload['image'],norm1),('to_q',modules['to_q'],mod,q.flatten(2)),('to_k',modules['to_k'],mod,k.flatten(2)),('to_v',modules['to_v'],mod,v.flatten(2)),('norm_q',modules['norm_q'],q,qn),('norm_k',modules['norm_k'],k,kn)]
        def check(name,args,kwargs,operand):
            if kwargs or len(args)!=1: raise ValueError('unqualified cached module call')
            actual=args[0]
            if actual.dtype!=operand.dtype or actual.device!=operand.device or actual.shape!=operand.shape or actual.stride()!=operand.stride() or not torch.equal(actual,operand): raise ValueError('changed cached input '+name)
        for mode in ('incremental','full'):
            nav=NativeCutNavigator(Session.from_cut(adapter,entry),max_steps=1,on_receipt=receipt); store.save(entry); hits=[]
            torch.cuda.synchronize(); tick=time.monotonic()
            if mode=='incremental':
                with reuse_prefix(bindings,check,hits): nav.step()
            else: nav.step()
            torch.cuda.synchronize(); seconds=time.monotonic()-tick
            exit=nav.session.capture(retain=False); store.save(exit)
            if describe(exit.payload)!=describe(expected.payload) or exit.execution_point!=expected.execution_point or exit.fingerprint!=expected.fingerprint: raise ValueError('complete native closure differs '+spec['id']+'.'+mode)
            if mode=='incremental' and [row['module'] for row in hits]!=['norm1','to_q','to_k','to_v','norm_q','norm_k']: raise ValueError('cache hit census differs')
            row=dict(case=spec['id'],mode=mode,entry=entry.fingerprint,exit=exit.fingerprint,archived_target=expected.fingerprint,full_payload_exact=True,cache_hits=hits,navigation_step_seconds=seconds,
                skipped_image_affines=3 if mode=='incremental' else 0,skipped_layer_norms=1 if mode=='incremental' else 0,skipped_qk_norms=2 if mode=='incremental' else 0)
            reports.append(row); save(out/('completed-'+spec['id']+'.'+mode+'.json'),row); event('completed',**row)
    for handle in handles: handle.remove()
    if entries['blocks']!=4 or module_entries!={key:4 for key in module_entries} or len(receipts)!=4: raise ValueError('physical call census differs')
    counts=dict(model_loads=1,block_entries=4,incremental_blocks=2,full_native_blocks=2,skipped_image_affines=6,skipped_image_layer_norms=2,skipped_image_qk_norms=4,acts=0,receipts=4,image_futures=0)
    summary=dict(schema='mdb-incremental-native-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),counts=counts,module_entries=module_entries,
        original_module_bodies_executed={key:2 for key in module_entries},reports=reports,worker_seconds=time.monotonic()-started,mechanics_status='pending-independent-audit',terminal_status='unassessed',
        scope='Post-hoc exact incremental execution qualification on two known3/8 cases. Six image affine bodies genuinely skipped; remaining native affected computation retained. Equality checks/synchronous receipt overhead included. Fixed order and n2 do not establish speedup. No independent zero-native prediction or semantic result.')
    save(out/'receipts.json',receipts); save(out/'summary.json',summary); save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__': worker()
