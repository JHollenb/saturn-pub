"""Finite original-parameter CUDA primitives over saved reader operands."""
import faulthandler
import hashlib
import inspect
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path


def worker():
    boot=time.monotonic();faulthandler.dump_traceback_later(60,repeat=True)
    print('MDB_READER_SETUP bootstrap',flush=True)
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    here=Path.cwd();core=runpy.run_path(str(here/'flux_reader_arithmetic.py'));h=core['H'];read,save,sha=h['read'],h['save'],h['sha']
    print('MDB_READER_SETUP numpy-ready '+str(time.monotonic()-boot),flush=True)
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=read(os.environ['SATURN_RUN_CONFIG'])
    if c!=req['config'] or any(sha(here/n)!=v for n,v in req['source_hashes'].items()):raise ValueError('worker binding changed')
    if c.get('python_bytecode') is False and not sys.dont_write_bytecode:raise ValueError('declared no-bytecode startup did not execute')
    import torch
    print('MDB_READER_SETUP torch-ready '+str(time.monotonic()-boot),flush=True)
    from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward
    print('MDB_READER_SETUP modules-ready '+str(time.monotonic()-boot),flush=True)
    from saturn_pub.store import LocalStore
    source=Path(c['input_root'])
    for n,v in c['inputs'].items():
        if sha(source/n)!=v['sha256'] or (source/n).stat().st_size!=v['bytes']:raise ValueError('source operand changed: '+n)
    print('MDB_READER_SETUP inputs-verified '+str(time.monotonic()-boot),flush=True)
    if not torch.cuda.is_available():raise ValueError('CUDA primitive lane required')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    started=time.monotonic()
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',c);save(out/'request.json',req)
    bank=h['RecipeBank'](read(here/'parameter-artifacts.json'),c['model_path'])
    weights={k:bank.load(v) for k,v in c['text_parameters'].items()}
    modules=core['make_modules'](weights,device='cuda',eps=c['eps'])
    store=LocalStore(source/'store');parents={};parent_records={}
    for receiver,fp in c['parents'].items():
        cut=store.load(fp);p=cut.payload
        parents[receiver]=(p['text'].float().numpy().copy(),p['context']['txt_mod'].float().numpy().copy())
        parent_records[receiver]={k:h['array'](out,v) for k,v in zip(('text','mod'),parents[receiver])}
    rows=read(source/'readers.json');cohort=read(source/'cohort.json')
    counts=dict(linear=0,layernorm=0,swiglu=0,mathematical_affines=0,full_model_loads=0,native_blocks=0,native_attention=0,images=0)
    phase={'name':'setup'}
    allowed=set(modules['ff'].modules())|{modules['projection'],modules['norm']}
    def guard(m,a):
        if phase['name']=='math':raise ValueError('native module called during independent arithmetic')
        if m not in allowed:raise ValueError('module outside declared text primitive set')
    handle=torch.nn.modules.module.register_module_forward_pre_hook(guard)
    linear=core['R']['_linear']
    def math_linear(*a,**kw):counts['mathematical_affines']+=1;return linear(*a,**kw)
    core['R']['_linear']=math_linear
    cases={}
    try:
        with torch.inference_mode():
            for key,row in rows.items():
                text,mod=parents[key.split('.')[0]]
                actual={f:h['load_array'](source,r) for f,r in row['fields'].items()}
                previous={f:h['load_array'](source,r) for f,r in cohort['predictions'][key].items()}
                tensor=lambda value:torch.from_numpy(value.copy()).to('cuda',torch.bfloat16)
                native={};default=None
                for name,head,after,reduced in [('native-input',actual['head'],None,original),
                        ('reference-head',previous['head'],None,original),('actual-after',actual['head'],actual['after_attention'],original),
                        ('opposite-reduction',actual['head'],None,not original)]:
                    phase['name']='native';torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced
                    values=core['native_tail'](tensor(head),tensor(text),tensor(mod),modules,
                        after=None if after is None else tensor(after),counts=counts)
                    values=core['wide'](values)
                    native[name]={f:h['array'](out,v) for f,v in values.items()}
                    if name=='native-input':default=values
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=not original
                values=core['native_local_affines']({f:tensor(default[f]) for f in ('head','mlp_input','activation')},modules,counts)
                native['local-opposite']={f:h['array'](out,v) for f,v in core['wide'](values).items()}
                phase['name']='math'
                math={name:{f:h['array'](out,v) for f,v in values.items()} for name,values in [
                    ('actual-head',core['math_tail'](actual['head'],text,mod,weights,eps=c['eps'])),
                    ('actual-after',core['math_tail'](actual['head'],text,mod,weights,eps=c['eps'],after=actual['after_attention'])),
                    ('local',core['math_local'](default,text,mod,weights,eps=c['eps']))]}
                # This independent residual closure uses the original source
                # after-attention/MLP returns, not reconstructed module outputs.
                import numpy as np
                gate=np.split(mod.reshape(1,1,-1),6,axis=-1)[5];b=core['R']['bf16']
                final=h['array'](out,b(actual['after_attention']+b(gate*actual['mlp'])))
                cases[key]=dict(native=native,math=math,source_final=final,source_observed=row['fields'],source_forecast=cohort['predictions'][key])
                save(out/'rows'/(key+'.json'),cases[key])
                event=dict(case=key,seconds=time.monotonic()-started,counts=counts.copy())
                with (out/'timeline.jsonl').open('a') as f:f.write(json.dumps(event,sort_keys=True)+'\n')
                print('MDB_READER_ARITHMETIC '+json.dumps(event,sort_keys=True),flush=True)
                if event['seconds']>c['max_wall_seconds']:raise RuntimeError('finite primitive budget exceeded')
    finally:
        handle.remove();torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original
        core['R']['_linear']=linear
    if counts['linear']!=112 or counts['layernorm']!=32 or counts['swiglu']!=32 or counts['mathematical_affines']!=64:
        raise ValueError('physical primitive census differs')
    save(out/'cases.json',cases);save(out/'parents.json',parent_records)
    for n in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):shutil.copyfile(here/n,out/n)
    save(out/'native-contract.json',dict(device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        input_shape=[1,c['text_rows'],c['width']],dtype='bfloat16',default_bf16_reduced_reduction=original,
        python_bytecode=not sys.dont_write_bytecode,
        fp16_reduced_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_accumulation=getattr(torch.backends.cuda.matmul,'allow_bf16_accumulation',None),tf32=False,deterministic=True,
        inference_mode=True,cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],sdpa='not executed',
        module_source_sha256={name:hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()
            for name,obj in [('Linear',torch.nn.Linear),('LayerNorm',torch.nn.LayerNorm),('Flux2FeedForward',Flux2FeedForward)]}))
    summary=dict(schema='mdb-reader-arithmetic-summary-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],
        output_root=str(out),counts=counts,seconds=time.monotonic()-started,bootstrap_seconds=started-boot,parameter_reads=bank.reads,
        parameter_cache_peak_bytes=bank.peak_retained,mechanics_status='complete',terminal_status='unassessed',scope=c['scope'])
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=req['request_sha256'],output_root=str(out),
        files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':worker()
