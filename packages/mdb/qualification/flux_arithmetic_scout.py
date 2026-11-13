"""Bounded saved-only arithmetic hypotheses before fresh operand validation."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import tempfile
import time
from pathlib import Path


def selections():
    columns=list(range(0,3072,48))
    return dict(train_rows=[0,4,11,31,255,511],train_columns=columns,
        check_rows=[1,7,19,63,127,256,383,510],check_columns=[i for i in range(17,3072,47) if i not in columns],
        query_rows=[0,1,4,11,31,127,255,511,512,513,543,639,767,895,1023,1151,1279,1407,1503,1535])


def worker():
    payload=Path.cwd(); request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text()); c=request['config']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); read=lambda p:json.loads(p.read_text())
    if any(sha(payload/n)!=v for n,v in request['source_hashes'].items()): raise ValueError('worker changed')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
    import numpy as np
    from mdb.parameter_artifacts import restore,validate_manifest
    from mdb.tensor_archive import load_tensor_archive
    source=Path(c['input_root'])
    if any(sha(source/n)!=v for n,v in c['inputs'].items()): raise ValueError('source changed')
    a=runpy.run_path(str(payload/'flux_arithmetic_reference.py')); s=selections()
    if s!=c['selections'] or a['affine_family']()!=c['affine_family'] or a['attention_family']()!=c['attention_family']: raise ValueError('frozen family changed')
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result'; out.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    save=lambda p,x:p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
    save(out/'plan.json',c); save(out/'request.json',request); native=read(source/'native.json')
    manifest=validate_manifest(read(source/'parameter-artifacts.json'))
    manifest=validate_manifest(dict(manifest,artifacts=[x for x in manifest['artifacts'] if any(p.startswith(('attn.add_q_proj.','attn.add_k_proj.','attn.add_v_proj.')) for p in x['parameter_ports'])]))
    save(out/'parameter-artifacts.json',manifest); shutil.copyfile(payload/'retrieve_weights.py',out/'retrieve_weights.py')
    affine={}; attention={}; results={}; selected=None
    def metric(p,n): return a['metrics'](a['bf16'](p),n)
    with tempfile.TemporaryDirectory(prefix='arithmetic-weights-',dir=out.parent) as tmp:
        save(out/'weight-restoration.json',restore(manifest,tmp,c['checkpoint_root'])); weights={}
        for item in manifest['artifacts']:
            bits=np.load(Path(tmp)/item['path'],allow_pickle=False)
            for port in item['parameter_ports']: weights[port]=(bits.astype(np.uint32)<<16).view(np.float32)
        stages=native['half']['stages']; x=load_tensor_archive(source,stages['text.modulated'])
        def evaluate(policy,label,split):
            rows,columns=s[split+'_rows'],s[split+'_columns']; name='attn.add_'+label+'_proj'
            actual=load_tensor_archive(source,stages['text.'+label+'_projection']).reshape(x.shape)[:,rows][:,:,columns]
            weight=weights[name+'.weight'][columns]; bias=weights.get(name+'.bias'); bias=None if bias is None else bias[columns]
            prediction=a['affine_wide'](x[:,rows],weight,policy,bias)
            return metric(prediction,actual)
        # Selection depends only on text-Q calibration coordinates, never checks.
        for policy in c['affine_family']: affine[policy['id']]=dict(policy=policy,calibration=evaluate(policy,'q','train'))
        selected=min(c['affine_family'],key=lambda p:(affine[p['id']]['calibration']['rms'],p['id']))
        save(out/'affine-selection.json',dict(selected=selected,calibration=affine,selection_rule='minimum calibration RMS then stable ID; no check observations yet'))
        for policy in c['affine_family']:
            affine[policy['id']]['checks']={label:evaluate(policy,label,'check') for label in ('q','k','v')}
        identical={label:bool(np.array_equal(load_tensor_archive(source,native['half']['stages']['text.'+label]),load_tensor_archive(source,native['half.image1']['stages']['text.'+label]))) for label in ('modulated','q_projection','k_projection','v_projection')}
    for case in ('half','half.image1'):
        stages=native[case]['stages']
        def operand(label): return np.concatenate([load_tensor_archive(source,stages[role+'.'+label]) for role in ('text','image')],axis=1).transpose(0,2,1,3)
        q,k,v=[operand(label) for label in ('q_positioned','k_positioned','v_projection')]; q=q[:,:,s['query_rows']]
        n=load_tensor_archive(source,stages['attention.raw'])[:,s['query_rows']].reshape(1,len(s['query_rows']),q.shape[1],q.shape[-1]).transpose(0,2,1,3)
        attention[case]={}; results[case]={}
        for policy in c['attention_family']:
            t=time.monotonic(); prediction=a['bf16'](a['attention_wide'](q,k,v,policy)); results[case][policy['id']]=prediction
            attention[case][policy['id']]=dict(policy=policy,error=a['metrics'](prediction,n),seconds=time.monotonic()-t)
            print('MDB_ARITHMETIC '+json.dumps(dict(case=case,policy=policy['id'],seconds=time.monotonic()-started)),flush=True)
        results[case]['native']=n
    from mdb.response_closure import measure_error
    contrast={policy['id']:measure_error(results['half.image1'][policy['id']].astype(float)-results['half'][policy['id']].astype(float),results['half.image1']['native'].astype(float)-results['half']['native'].astype(float)) for policy in c['attention_family']}
    if any(name in sys.modules for name in ('torch','diffusers','transformers')): raise ValueError('framework imported')
    summary=dict(schema='mdb-arithmetic-scout-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(out.parent),seconds=time.monotonic()-started,
        affine=affine,affine_selection=selected,text_half_banks_identical=identical,attention=attention,attention_receiver_contrast=contrast,
        counts=dict(model_loads=0,native_primitives=0,native_blocks=0,images=0,affine_policies=len(c['affine_family']),attention_policies=len(c['attention_family'])),
        scope='Saved-only numerical development, sampled rows/coordinates on original full-geometry native half calibrations. No fresh-target ports. Selection fixed before disjoint coordinate/projection checks. Text receivers may be duplicates. Numerical surrogates are not recovered CUDA internals.',mechanics_status='pending-independent-audit',terminal_status='unassessed')
    save(out/'summary.json',summary); save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,source):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    from mdb import parameter_artifacts
    here=Path(__file__).resolve().parent; h=runpy.run_path(str(here/'flux_binding_minimal.py')); sha,save=h['sha'],h['save']; read=lambda p:json.loads(p.read_text())
    runpy.run_path(str(here/'flux_family_custody.py'))['verify'](source)
    prior=read(source/'request.json'); h['bind_environment'](prior['execution_environment']); data=source/'extracted/binding-minimal-result'; native=read(data/'native.json'); summary=read(data/'summary.json'); inventory=read(data/'files.json')
    names={'native.json','parameter-artifacts.json','summary.json'}
    for item in native.values():
        for key,record in item['stages'].items():
            if key in ('attention.raw','text.modulated') or key.endswith(('.q_positioned','.k_positioned','.v_projection','.q_projection','.k_projection')): names.add(record['path'])
    inputs={name:inventory[name] for name in sorted(names)}
    if any(sha(data/name)!=v for name,v in inputs.items()): raise ValueError('source inventory changed')
    a=runpy.run_path(str(here/'flux_arithmetic_reference.py'))
    c=dict(input_root=summary['output_root']+'/binding-minimal-result',inputs=inputs,selections=selections(),affine_family=a['affine_family'](),attention_family=a['attention_family'](),
        checkpoint_root=read(data/'parameter-artifacts.json')['defaults']['checkpoint_root'],output_scratch='/mnt/big/mdb/research',directory_custody=True,
        scope='Bounded saved-only hypotheses, trainQcoordinates before disjoint checks; no adaptive family expansion, native operations or fresh target observations.')
    run.mkdir(parents=True,exist_ok=False); save(run/'plan.json',c)
    extras=[(here/n,n) for n in ('flux_family_custody.py','flux_binding_minimal.py','flux_arithmetic_reference.py')]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py')]
    pre=run/'preflight.v1.json'; save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[c['input_root'],c['checkpoint_root'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name)); request=dict(schema='mdb-arithmetic-scout-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='wide-with-declared-rounding',execution_environment=prior['execution_environment'],
        source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=92,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=128,extra=dict(task_family='analysis'),checkpoint_policy='18affine numericalhypotheses xtrainQ/heldQKV;10attentionhypotheses x2savedreceivers,20queries/all1536keys.0nativeprimitives/models/images.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest(); save(run/'request.json',request); stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py'))); r=json.loads((run/'request.json').read_text()); h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()): raise ValueError('already submitted/source changed')
    job=guarded_mrun_launch(worker_command(Path(__file__).name),request=r,model=None,task='analysis',experiment='mdb-attention-arithmetic-scout',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=300,retry_on_kill=False,preflight=True,
        note='Saved-only NumPy numerical scout:18fixed affine schedules on trainQ then disjointQKV coordinates;10attention policies on20queries/allkeys x2knownhalf receivers.0model/nativeprimitives/images;temporary3projectionweights removed. CPU4threads, expectedunder90s/300outer,no retries,intent-first sizing.')
    h['save'](run/'submission.json',dict(job_id=str(job))); return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1': worker()
    else:
        p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path); p.add_argument('source',type=Path,nargs='?'); p.add_argument('--submit',action='store_true'); a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.source.resolve()),sort_keys=True))
