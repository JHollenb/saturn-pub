"""Two source-extracted native MLPs localize a saved conditional discrepancy."""
import ast
import hashlib
import inspect
import json
import os
import runpy
import time
from pathlib import Path


CASE='row4seven_eighths'
ANCHOR='half'


def extract_classes(source,expected_sha256):
    """Compile only the two original class definitions, without importing the pipeline."""
    import torch
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('original Flux source changed')
    tree=ast.parse(raw);names={'Flux2FeedForward','Flux2SwiGLU'};nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
    if len(nodes)!=2 or {n.name for n in nodes}!=names:raise ValueError('original MLP classes absent')
    scope=dict(torch=torch,nn=torch.nn,__name__='source_qualified_flux2_mlp')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    return scope['Flux2FeedForward'],{n.name:hashlib.sha256(ast.get_source_segment(raw.decode(),n).encode()).hexdigest() for n in nodes}


def layout(value):
    return dict(shape=list(value.shape),stride=list(value.stride()),storage_offset=value.storage_offset(),dtype=str(value.dtype),device=str(value.device),contiguous=value.is_contiguous())


def capture(module,value,counts):
    fields={};layouts=dict(input=layout(value));handles=[]
    for name,child,kind in [('paired',module.linear_in,'linear'),('silu',module.act_fn.gate_fn,'silu'),('activation',module.act_fn,'product'),('mlp',module.linear_out,'linear')]:
        def hook(m,args,output,name=name,kind=kind):
            counts[kind]+=1;fields[name]=output.detach();layouts[name]=dict(input=layout(args[0]),output=layout(output))
            if name=='paired':
                half=output.shape[-1]//2;layouts['chunks']=dict(u=layout(output[...,:half]),v=layout(output[...,half:]))
        handles.append(child.register_forward_hook(hook))
    try:
        counts['mlp']+=1;module(value)
    finally:
        for handle in handles:handle.remove()
    return fields,layouts


def mathematical_fields(x,stages,native,weights,role,reference,law):
    """Each local comparison gets actual operands; the chained branch owns its operands."""
    import numpy as np
    B=reference['bf16'];prefix='ff' if role=='image' else 'ff_context'
    def first(v):return law['affine_delta'](v,stages['mlp_input'],stages['mlp_paired'],weights[prefix+'.linear_in.weight'],B)[0]
    def second(v):return law['affine_delta'](v,stages['mlp_activation'],stages['mlp'],weights[prefix+'.linear_out.weight'],B)[0]
    def swiglu(z):
        u,v=np.split(z,2,axis=-1);s=reference['_silu'](u);return s,B(s*v)
    z=first(x);s,a=swiglu(z);C=second(a)
    actual_u,actual_v=np.split(native['paired'],2,axis=-1);local_s,local_a=swiglu(native['paired'])
    local=dict(paired=z,silu=local_s,activation=local_a,product=B(native['silu']*actual_v),mlp=second(native['activation']))
    substitutions=dict(C=C,after_native_paired=second(local_a),after_native_activation=local['mlp'])
    return dict(chained=dict(paired=z,silu=s,activation=a,mlp=C),local=local,substitutions=substitutions)


def worker():
    payload=Path.cwd();h=runpy.run_path(str(payload/'flux_binding_minimal.py'));sha,save=h['sha'],h['save'];read=lambda p:json.loads(p.read_text());request=read(Path(os.environ['SATURN_DEBUG_REQUEST']));c=request['config']
    if read(Path(os.environ['SATURN_RUN_CONFIG']))!=c or c['case']!=CASE or c['anchor']!=ANCHOR or any(sha(payload/n)!=v for n,v in request['source_hashes'].items()):raise ValueError('frozen cut differs')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import importlib.metadata
    import numpy as np
    import torch
    from mdb.tensor_archive import load_tensor_archive
    source=Path(c['source_collection']);record=read(source/'remote-custody.json');data=source/'extracted/binding-minimal-result'
    if sha(source/'remote-custody.json')!=c['source_record_sha256'] or sha(Path(c['source_audit']))!=c['source_audit_sha256']:raise ValueError('source custody/audit changed')
    if read(Path(c['source_audit']))['custody']!=record['verification']:raise ValueError('source audit cohort differs')
    for name in ('plan.json','native.json','forecasts.json','baselines.json','parameter-artifacts.json'):
        if sha(data/name)!=record['files'][name]['sha256']:raise ValueError('source metadata changed')
    plan,natives,forecasts,baselines=[read(data/n) for n in ('plan.json','native.json','forecasts.json','baselines.json')]
    old_request=read(source/'request.json')
    if any(request['source_hashes'][name]!=old_request['source_hashes'][name] for name in ('flux_reentry_reference.py','flux_receiver_norm_prediction.py','flux_arithmetic_reference.py')):raise ValueError('frozen numerical law changed')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID'];out=scratch/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();save(out/'plan.json',c);save(out/'request.json',request)
    def event(kind,**kw):
        row=dict(kind=kind,seconds=time.monotonic()-started,**kw)
        with (out/'timeline.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_MLP_PRIMITIVE '+json.dumps(row,sort_keys=True),flush=True)
    def arr(root,item):
        name=(root/item['path']).relative_to(data).as_posix()
        if record['files'][name]['sha256']!=item['sha256']:raise ValueError('array outside verified source')
        return load_tensor_archive(root,item)
    if not torch.cuda.is_available():raise ValueError('CUDA primitive lane required')
    torch.set_num_threads(4);torch.set_deterministic_debug_mode('error');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_reduction=bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    if not original_reduction or not torch.are_deterministic_algorithms_enabled():raise ValueError('original numerical profile differs')
    original_source=Path(importlib.metadata.distribution('diffusers').locate_file('diffusers/models/transformers/transformer_flux2.py'));FeedForward,class_hashes=extract_classes(original_source,c['original_module_source_sha256'])
    bank_type=runpy.run_path(str(payload/'flux_source_receiver.py'))['RecipeBank'];manifest=read(data/'parameter-artifacts.json');bank=bank_type(manifest,c['model_path']);weights={}
    entries={port:item for item in manifest['artifacts'] for port in item['parameter_ports']}
    for port in ('ff.linear_in.weight','ff.linear_out.weight','ff_context.linear_in.weight','ff_context.linear_out.weight'):
        item=entries[port];weights[port]=bank.load(dict(path=item['path'],sha256=item['sha256'],descriptor=dict(sha256=item['source']['payload_sha256']),parameter_tensor=item['source']['tensor'],stored_shape=item['source']['shape']))
    save(out/'parameter-artifacts.json',manifest)
    import shutil
    shutil.copyfile(payload/'retrieve_weights.py',out/'retrieve_weights.py')
    abi=dict(kind='source-extracted original Flux2FeedForward/Flux2SwiGLU classes; pinned installed file, no pipeline import',original_module_source_sha256=sha(original_source),class_source_sha256=class_hashes,
        torch=str(torch.__version__),cuda=str(torch.version.cuda),device=str(torch.cuda.get_device_name()),dtype='bfloat16',bf16_reduced_precision_reduction=bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        fp16_reduced_precision_reduction=bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),bf16_accumulation=None if getattr(torch.backends.cuda.matmul,'allow_bf16_accumulation',None) is None else bool(torch.backends.cuda.matmul.allow_bf16_accumulation),
        tf32=bool(torch.backends.cuda.matmul.allow_tf32),deterministic=bool(torch.are_deterministic_algorithms_enabled()),cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],inference_mode=True,
        linear_forward_sha256=hashlib.sha256(inspect.getsource(torch.nn.Linear.forward).encode()).hexdigest(),silu_forward_sha256=hashlib.sha256(inspect.getsource(torch.nn.SiLU.forward).encode()).hexdigest(),
        input_scope='Exact archived values in full contiguous B1 row geometry. Earlier capture normalized storage; original module input strides were not separately retained. Native output parity is the behavioral closure check.')
    save(out/'native-contract.json',abi);event('contract_bound',profile=abi)
    modules={}
    for role,prefix in [('image','ff'),('text','ff_context')]:
        W1,W2=weights[prefix+'.linear_in.weight'],weights[prefix+'.linear_out.weight']
        with torch.device('meta'):module=FeedForward(W1.shape[1],inner_dim=W2.shape[1],bias=False)
        module.load_state_dict({name:torch.from_numpy(weights[prefix+'.'+name].copy()).to('cuda',torch.bfloat16) for name in ('linear_in.weight','linear_out.weight')},assign=True)
        modules[role]=module.eval().requires_grad_(False)
    allowed={child for module in modules.values() for child in module.modules()};phase=dict(name='setup');counts=dict(mlp=0,linear=0,silu=0,product=0,full_models=0,blocks=0,attention=0,images=0);archive={}
    writer=runpy.run_path(str(payload/'flux_hybrid_future.py'))['record_bf16'];r=runpy.run_path(str(payload/'flux_reentry_reference.py'));law=runpy.run_path(str(payload/'flux_receiver_norm_prediction.py'));math_mlp=runpy.run_path(str(payload/'flux_hybrid_mlp_cut.py'))['mlp']
    def guard(module,args):
        if phase['name']!='native' or module not in allowed:raise ValueError('learned forward outside declared native MLP phase')
    handle=torch.nn.modules.module.register_module_forward_pre_hook(guard);rows={}
    try:
        with torch.inference_mode():
            for role in ('image','text'):
                x=arr(data,natives[CASE]['stages'][role+'.mlp_input']);stages={field:arr(data/'source/prefixes',plan['anchors'][ANCHOR]['stages'][role+'.'+field]) for field in ('mlp_input','mlp_paired','mlp_silu','mlp_activation','mlp')}
                phase['name']='native';native,layouts=capture(modules[role],torch.from_numpy(x.copy()).to('cuda',torch.bfloat16),counts);native={name:value.float().cpu().numpy().copy() for name,value in native.items()};phase['name']='math'
                parity=bool(np.array_equal(native['mlp'],arr(data,natives[CASE]['stages'][role+'.mlp'])))
                calculated=mathematical_fields(x,stages,native,weights,role,r,law);rebuilt={}
                full_stages={role+'.'+name:value for name,value in stages.items()}
                for name,group in [('P1',forecasts[CASE]),('P0',baselines[ANCHOR])]:
                    for mode in ('control','hybrid'):
                        fields=group[mode]['fields'][role];value=math_mlp(arr(data,fields['mlp_input']),full_stages,weights,role,r,law)
                        rebuilt[name+'.'+mode]=bool(np.array_equal(value,arr(data,fields['mlp'])))
                row=dict(native={name:writer(out,value,archive) for name,value in native.items()},math={group:{name:writer(out,value,archive) for name,value in values.items()} for group,values in calculated.items()},layouts=layouts,native_output_parity=parity,owned_reconstruction=rebuilt,
                    source_input=natives[CASE]['stages'][role+'.mlp_input'],source_output=natives[CASE]['stages'][role+'.mlp'])
                rows[role]=row;save(out/('row-'+role+'.json'),row);event('role_complete',role=role,native_output_parity=parity,owned_reconstruction=rebuilt,counts=dict(counts))
                if time.monotonic()-started>c['max_wall_seconds']:raise RuntimeError('finite primitive cut exceeded')
    finally:handle.remove()
    if counts!=dict(mlp=2,linear=4,silu=2,product=2,full_models=0,blocks=0,attention=0,images=0):raise ValueError('physical primitive census differs')
    save(out/'rows.json',rows);save(out/'source-binding.json',dict(collection=str(source),record_sha256=c['source_record_sha256'],source_custody=record['verification'],case=CASE,anchor=ANCHOR,native_entry=natives[CASE]['entry'],native_exit=natives[CASE]['exit'],source_audit_sha256=c['source_audit_sha256'],scope=c['scope']))
    summary=dict(schema='mdb-mlp-primitive-cut-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),counts=counts,seconds=time.monotonic()-started,parameter_reads=bank.reads,parameter_cache_peak_bytes=bank.peak_retained,unique_arrays=len(archive),mechanics_status='pending-independent-audit',terminal_status='unassessed',scope=c['scope'])
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__':worker()
