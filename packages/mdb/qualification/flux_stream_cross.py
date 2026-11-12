"""Typed J2 old/new image-by-text interventions and sparse native continuations."""
import hashlib
import inspect
import json
import os
import runpy
import time
from pathlib import Path


def counts():return dict(loads=1,native_transitions=225,blocks=207,acts=9,receipts=36,attention_midpoints=9,images=0,encodes=0,noise_draws=0)


def worker():
    payload=Path.cwd();h=runpy.run_path(str(payload/'flux_binding_minimal.py'));sha,save,identity=[h[k] for k in ('sha','save','identity')];read=lambda p:json.loads(Path(p).read_text())
    request=read(os.environ['SATURN_DEBUG_REQUEST']);c=read(os.environ['SATURN_RUN_CONFIG'])
    if c!=request['config'] or any(sha(payload/n)!=digest for n,digest in request['source_hashes'].items()):raise ValueError('cross request/source changed')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    source=runpy.run_path(str(payload/'flux_qualified_source.py'))['QualifiedSource'](c['source_collection'],c['source_record_sha256'],c['source_audit'],c['source_audit_sha256'])
    traces=source.json('traces.json');by={(x['id'],x['mode']):x for x in traces};reference=source.json('receipts.json')[0]['body']
    if c['origins']!={name:{mode:by[name,mode]['origin'] for mode in ('native','old','new')} for name in ('half','row4heavy','row11heavy')}:raise ValueError('source input join differs')
    if any(source.audit['replay_status'][name+'.'+mode]!='qualified' for name in c['origins'] for mode in ('native','old','new')):raise ValueError('source future was not qualified')
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    save(out/'request.json',request);save(out/'plan.json',c);save(out/'source-binding.json',dict(record=source.record,record_sha256=c['source_record_sha256'],audit_sha256=c['source_audit_sha256']))
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session,Act
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.native_navigation import NativeCutNavigator
    from mdb.tensor_archive import load_tensor_archive
    started=time.monotonic();actual=dict.fromkeys(counts(),0);arrays={};rows=[];receipts=[];active={}
    def event(kind,**kw):
        row=dict(kind=kind,seconds=time.monotonic()-started,counts=dict(actual),**kw)
        with (out/'timeline.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_STREAM_CROSS '+json.dumps(row,sort_keys=True),flush=True)
        if row['seconds']>c['worker_wall_s']:raise RuntimeError('finite cross wall budget')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available() or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:raise ValueError('native numerical profile differs')
    root=Path(c['model_path']);inventory={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=c['checkpoint_sha256']:raise ValueError('checkpoint differs')
    event('model_load_start');pipe=Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False).to('cuda');pipe.vae.eval().requires_grad_(False);pipe.vae.bn.to('cpu');actual['loads']=1
    adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    if adapter.model_identity!=reference['model_identity'] or dict(adapter.execution)!=reference['execution']:raise ValueError('source model/profile differs')
    parameters=runpy.run_path(str(payload/'flux_trace_parameter_recipes.py'))['recipes'](pipe.transformer,root,inventory,out,payload)
    save(out/'native-contract.json',dict(model_identity=adapter.model_identity,execution=adapter.execution,parameter_recipes=parameters,
        adapter_advance_sha256=hashlib.sha256(inspect.getsource(type(adapter).advance).encode()).hexdigest(),joint_source_sha256=hashlib.sha256(inspect.getsource(type(pipe.transformer.transformer_blocks[2])).encode()).hexdigest(),bf16_reduced_precision=True))
    def entered(module,args):actual['blocks']+=1
    handles=[block.register_forward_pre_hook(entered) for block in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    def no_vae(module,args):raise ValueError('VAE must not execute')
    handles.append(pipe.vae.register_forward_pre_hook(no_vae));original=LocalStore(source.root/'store');store=LocalStore(out/'store')
    array=lambda value:runpy.run_path(str(payload/'flux_hybrid_future.py'))['record_bf16'](out,value,arrays)
    def receipt(value):
        body=value.to_dict();store.receipt(value);receipts.append(dict(branch=active['name'],body=body));actual['receipts']+=1;actual['native_transitions']+=body.get('steps',0);actual['acts']+=body['operation']=='apply'
    def capture(nav):cut=nav.session.capture(retain=False);store.save(cut);return cut
    def load(fp):source.cut(fp);return original.load(fp,device='cuda')
    def fields(cut):return {k:describe(v) for k,v in cut.payload.items()}
    for spec in c['branches']:
        name,corner=spec['id'],spec['corner'];active['name']=name+'.'+corner;old=load(c['origins'][name]['old']);new=load(c['origins'][name]['new']);store.save(old)
        before_fields,other=fields(old),fields(new)
        if set(before_fields)!=set(other) or any(before_fields[k]!=other[k] for k in before_fields if k not in {'image','text'}) or dict(old.execution_point)!=dict(new.execution_point):raise ValueError('old/new nuisance closure differs')
        slot=spec['slot'];donor=old if corner=='00' else new;nav=NativeCutNavigator(Session.from_cut(adapter,old),max_steps=25,on_receipt=receipt)
        if any(nav.point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=2).items()):raise ValueError('J2 input clock differs')
        before=capture(nav);nav.apply(Act.replace(slot,donor.payload[slot],name='J2-stream-corner-'+corner));installed=capture(nav);after_fields=fields(installed)
        if after_fields[slot]!=fields(donor)[slot] or any(before_fields[k]!=after_fields[k] for k in before_fields if k!=slot):raise ValueError('typed single-slot intervention closure differs')
        ports={};block=pipe.transformer.transformer_blocks[2]
        def remember(role):
            def hook(module,args):ports[role]=array(args[0])
            return hook
        hooks=[block.norm2.register_forward_pre_hook(remember('image')),block.norm2_context.register_forward_pre_hook(remember('text'))]
        try:nav.step(1)
        finally:
            for hook in hooks:hook.remove()
        joint=capture(nav)
        if set(ports)!={'image','text'}:raise ValueError('attention midpoint missing')
        actual['attention_midpoints']+=1;nav.step(23);noise=capture(nav)
        if nav.point['phase']!='scheduler' or nav.point['logical_step']!=2:raise ValueError('readout horizon differs')
        nav.step(1);final=capture(nav);control={}
        if corner=='00':
            original_row=by[name,'old'];points={p['id']:p for p in original_row['points']}
            for role,item in points['joint2.attention']['arrays'].items():
                source.member(item['path']);control['attention.'+role]=ports[role]['descriptor']==item['descriptor'] and ports[role]['sha256']==item['sha256']
            for label,cut in [('joint2.exit',joint),('readout.noise',noise),('scheduler.latent',final)]:
                expected=load(points[label]['cut']);control[label]=describe(cut.payload)==describe(expected.payload) and dict(cut.execution_point)==dict(expected.execution_point)
        row=dict(id=name,corner=corner,slot=slot,donor=donor.fingerprint,parent=old.fingerprint,before=before.fingerprint,installed=installed.fingerprint,attention=ports,joint_exit=joint.fingerprint,noise=noise.fingerprint,final=final.fingerprint,
            sham_checks=control,sham_status=('qualified' if all(control.values()) else 'instrument-replay-mismatch') if control else 'not-applicable-crossed-intervention')
        rows.append(row);save(out/('row-'+name+'.'+corner+'.json'),row);event('branch_complete',branch=active['name'],sham_status=row['sham_status'])
    for handle in handles:handle.remove()
    if actual!=counts() or len(rows)!=9:raise ValueError('complete corner census differs')
    save(out/'rows.json',rows);save(out/'receipts.json',receipts);save(out/'consumed-source.json',source.report())
    summary=dict(schema='mdb-stream-cross-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(out.parent),counts=actual,unique_arrays=len(arrays),seconds=time.monotonic()-started,
        sham_status={r['id']:r['sham_status'] for r in rows if r['corner']=='00'},mechanics_status='complete-pending-independent-audit',terminal_status='unassessed',scope=c['scope'])
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__':worker()
