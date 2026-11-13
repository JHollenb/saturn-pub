"""Check one frozen arithmetic schedule on twelve existing native affines."""
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


def operands():
    return [(role,port,source,target) for role,prefix,ff in [('text','add_','ff_context'),('image','to_','ff')]
        for port,source,target in [*[(f'attn.{prefix}{q}_proj' if role=='text' else f'attn.to_{q}','modulated',q+'_projection') for q in ('q','k','v')],
            ('attn.to_add_out' if role=='text' else 'attn.to_out.0','attention.raw','attention'),
            (ff+'.linear_in','mlp_input','mlp_paired'),(ff+'.linear_out','mlp_activation','mlp')]]


def worker():
    payload=Path.cwd(); request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text()); c=request['config']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); read=lambda p:json.loads(p.read_text())
    if any(sha(payload/n)!=v for n,v in request['source_hashes'].items()): raise ValueError('source changed')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
    import numpy as np
    from mdb_lanes.parameter_artifacts import restore,validate_manifest
    from mdb_lanes.tensor_archive import load_tensor_archive
    source=Path(c['input_root'])
    if any(sha(source/n)!=v for n,v in c['inputs'].items()): raise ValueError('bound inputs changed')
    a=runpy.run_path(str(payload/'flux_arithmetic_reference.py')); native=read(source/'native.json'); manifest=validate_manifest(read(source/'parameter-artifacts.json'))
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result'; out.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    save=lambda p,x:p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n'); save(out/'plan.json',c); save(out/'request.json',request)
    save(out/'parameter-artifacts.json',manifest); shutil.copyfile(payload/'retrieve_weights.py',out/'retrieve_weights.py')
    def array(name,value):
        p=out/(name+'.npy'); value=np.asarray(value,np.float32)
        with p.open('xb') as f: np.save(f,value,allow_pickle=False)
        return dict(path=p.name,sha256=sha(p),shape=list(value.shape),dtype=str(value.dtype))
    reports={}
    with tempfile.TemporaryDirectory(prefix='affine-transfer-weights-',dir=out.parent) as tmp:
        save(out/'weight-restoration.json',restore(manifest,tmp,c['checkpoint_root']))
        by_port={port:item for item in manifest['artifacts'] for port in item['parameter_ports']}
        def weight(port):
            bits=np.load(Path(tmp)/by_port[port]['path'],allow_pickle=False)
            return (bits.astype(np.uint32)<<16).view(np.float32)
        for case in ('half','half.image1'):
            reports[case]={}; stages=native[case]['stages']
            for role,port,source_name,target in operands():
                n=load_tensor_archive(source,stages[role+'.'+target]); n=n.reshape(1,n.shape[1],-1)
                x=load_tensor_archive(source,stages['attention.raw' if source_name=='attention.raw' else role+'.'+source_name])
                if source_name=='attention.raw': x=x[:,:512] if role=='text' else x[:,512:]
                x=x.reshape(1,x.shape[1],-1); rows=[i for i in c['rows'] if i<x.shape[1]]; columns=list(range(17,n.shape[-1],max(1,n.shape[-1]//64)))[:64]
                w=weight(port+'.weight')[columns]; bias=weight(port+'.bias')[columns] if port+'.bias' in by_port else None
                actual=n[:,rows][:,:,columns]; predicted=a['bf16'](a['affine_wide'](x[:,rows],w,c['policy'],bias)); wide=a['bf16'](a['affine_wide'](x[:,rows],w,dict(kind='wide'),bias))
                key=role+'.'+port
                reports[case][key]=dict(rows=rows,columns=columns,input_shape=list(x.shape),weight_shape=by_port[port+'.weight']['source']['shape'],policy=a['metrics'](predicted,actual),wide=a['metrics'](wide,actual),
                    samples=dict(predicted=array(case+'.'+key+'.predicted',predicted),native=array(case+'.'+key+'.native',actual),wide=array(case+'.'+key+'.wide',wide)))
                print('MDB_AFFINE_TRANSFER '+json.dumps(dict(case=case,port=port,exact=reports[case][key]['policy']['exact_fraction'],seconds=time.monotonic()-started)),flush=True)
    if any(name in sys.modules for name in ('torch','diffusers','transformers')): raise ValueError('model framework imported')
    summary=dict(schema='mdb-affine-transfer-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(out.parent),seconds=time.monotonic()-started,reports=reports,policy=c['policy'],
        counts=dict(model_loads=0,native_primitives=0,affine_checks=24,samples_per_check=512),mechanics_status='pending-independent-audit',terminal_status='unassessed',
        scope='One already-selected text-Q reduction schedule, unchanged across12affines x2knownhalf states; sampled coordinates from native full-geometry outputs. No adaptive schedule search or fresh target ports. Native half outputs are development evidence; duplicate text projections not independent replicates.')
    save(out/'summary.json',summary); save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,precision,scout):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes import parameter_artifacts
    here=Path(__file__).resolve().parent; h=runpy.run_path(str(here/'flux_binding_minimal.py')); sha,save=h['sha'],h['save']; read=lambda p:json.loads(p.read_text())
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify']
    custody(precision); custody(scout)
    data=precision/'extracted/binding-minimal-result'; result=read(scout/'extracted/binding-minimal-result/summary.json'); prior=read(precision/'request.json'); h['bind_environment'](prior['execution_environment']); inventory=read(data/'files.json'); native=read(data/'native.json')
    names={'parameter-artifacts.json','native.json','summary.json'}
    for row in native.values():
        for role,port,input_name,target in operands():
            for key in ('attention.raw' if input_name=='attention.raw' else role+'.'+input_name,role+'.'+target): names.add(row['stages'][key]['path'])
    inputs={n:inventory[n] for n in names}
    if any(sha(data/n)!=v for n,v in inputs.items()): raise ValueError('source inventory differs')
    c=dict(input_root=read(data/'summary.json')['output_root']+'/binding-minimal-result',inputs=inputs,policy=result['affine_selection'],rows=[1,7,19,63,127,256,383,510],checkpoint_root=read(data/'parameter-artifacts.json')['defaults']['checkpoint_root'],
        selected_scout_summary_sha256=sha(scout/'extracted/binding-minimal-result/summary.json'),output_scratch='/mnt/big/mdb/research',directory_custody=True)
    run.mkdir(parents=True,exist_ok=False); save(run/'plan.json',c)
    extras=[(here/n,n) for n in ('flux_family_custody.py','flux_binding_minimal.py','flux_arithmetic_reference.py')]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py')]
    pre=run/'preflight.v1.json'; save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[c['input_root'],c['checkpoint_root'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name)); request=dict(schema='mdb-affine-transfer-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='BF161024carriedsum',execution_environment=prior['execution_environment'],
        source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=24,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=24,extra=dict(task_family='analysis'),checkpoint_policy='Fixed1024carryRNEschedule vswide on12savedaffines x2receivers,8rows64columns; retainallpredictions/native samples.0nativecalls/images.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest(); save(run/'request.json',request); stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py'))); r=json.loads((run/'request.json').read_text()); h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()): raise ValueError('already submitted/source changed')
    job=guarded_mrun_launch(worker_command(Path(__file__).name),request=r,model=None,task='analysis',experiment='mdb-affine-numerical-transfer',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=300,retry_on_kill=False,preflight=True,
        note='Onefrozen1024carryBF16RNEpolicy on12affines x2knownhalf cases;512samples/check, allpred/native/wide arrays retained.0model/nativeprimitive/imagecalls. Temporarycheckpointweightsdeleted,4CPUthreads,under90s/300outer,no retries,intent-first sizing.')
    h['save'](run/'submission.json',dict(job_id=str(job))); return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1': worker()
    else:
        p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path); p.add_argument('precision',type=Path,nargs='?'); p.add_argument('scout',type=Path,nargs='?'); p.add_argument('--submit',action='store_true'); a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.precision.resolve(),a.scout.resolve()),sort_keys=True))
