"""Eight CUDA matrix primitives diagnose a captured affine numerical contract."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import tempfile
import time
from pathlib import Path


def policies():
    return [('bf16-reduced',True,'bfloat16'),('bf16-full',False,'bfloat16'),('fp32',False,'float32'),('fp64',False,'float64')]


def worker():
    payload=Path.cwd();request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text());c=request['config']
    for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    from mdb.parameter_artifacts import validate_manifest,restore
    from mdb.tensor_archive import load_tensor_archive,write_tensor_archive
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();save=lambda p,v:p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
    if any(sha(payload/n)!=v for n,v in request['source_hashes'].items()):raise ValueError('source changed')
    source=Path(c['input_root'])
    if any(sha(source/n)!=v for n,v in c['inputs'].items()):raise ValueError('native operand identity changed')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID'];out=scratch/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False);save(out/'plan.json',c);save(out/'request.json',request)
    started=time.monotonic();metric=runpy.run_path(str(payload/'analyze_flux_precision.py'))['error'];r=runpy.run_path(str(payload/'flux_reentry_reference.py'))
    full=validate_manifest(read(source/'parameter-artifacts.json'));manifest=validate_manifest(dict(full,artifacts=[a for a in full['artifacts']if any(p in{'attn.to_q.weight','attn.to_q.bias'}for p in a['parameter_ports'])]))
    save(out/'parameter-artifacts.json',manifest);shutil.copyfile(payload/'retrieve_weights.py',out/'retrieve_weights.py')
    native=read(source/'native.json');reports={};predictions={};actuals={};calls=0;original=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise ValueError('CUDA primitive instrument required')
    with tempfile.TemporaryDirectory(prefix='gemm-weights-',dir=scratch)as tmp:
        save(out/'weight-restoration.json',restore(manifest,tmp,c['checkpoint_root']));weights={}
        for a in manifest['artifacts']:
            bits=np.load(Path(tmp)/a['path'],allow_pickle=False);weights[a['parameter_ports'][0]]=(bits.astype(np.uint32)<<16).view(np.float32)
        w=torch.from_numpy(weights['attn.to_q.weight'].copy()).to('cuda',torch.bfloat16);bias=weights.get('attn.to_q.bias');bias=None if bias is None else torch.from_numpy(bias.copy()).to('cuda',torch.bfloat16)
        try:
            with torch.no_grad():
                for case in('half','half.image1'):
                    stages=native[case]['stages'];x=load_tensor_archive(source,stages['image.modulated']);actual=load_tensor_archive(source,stages['image.q_projection']).reshape(x.shape);actuals[case]=actual
                    reference=r['_linear'](x,weights,'attn.to_q',256);predictions[case,'reference']=reference
                    reference_record=write_tensor_archive(out/(case+'.reference.npy'),torch.from_numpy(reference).to(torch.bfloat16),root=out)
                    reports[case]=dict(reference=dict(record=reference_record,vs_native=metric(reference,actual)),policies={});tx=torch.from_numpy(x.copy()).to('cuda',torch.bfloat16)
                    for name,reduced,dtype in policies():
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced;dt=getattr(torch,dtype);a,b=tx.to(dt),w.to(dt);bb=None if bias is None else bias.to(dt)
                        torch.cuda.synchronize();t=time.monotonic();calls+=1;y=torch.nn.functional.linear(a,b,bb);torch.cuda.synchronize();seconds=time.monotonic()-t
                        value=y.to(torch.float32).to(torch.bfloat16).cpu();record=write_tensor_archive(out/(case+'.'+name+'.npy'),value,root=out);v=value.float().numpy();predictions[case,name]=v
                        reports[case]['policies'][name]=dict(record=record,vs_native=metric(v,actual),vs_reference=metric(v,reference),primitive_seconds=seconds)
                        print('MDB_GEMM '+json.dumps(dict(case=case,policy=name,calls=calls,seconds=time.monotonic()-started)),flush=True)
        finally:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original
    contrast={name:metric(predictions['half.image1',name]-predictions['half',name],actuals['half.image1']-actuals['half'])for name in['reference',*[p[0]for p in policies()]]}
    if calls!=8:raise ValueError('primitive census differs')
    result=dict(schema='mdb-cuda-gemm-precision-diagnostic-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],reports=reports,receiver_contrast=contrast,
        counts=dict(full_model_loads=0,learned_weight_sets=1,native_linear_primitives=calls,native_block_forwards=0,image_futures=0),original_reduced_precision=original,seconds=time.monotonic()-started,
        scope='Post-hoc primitive diagnosis on observed native image.modulated operands and checkpoint-identical Q weights. Native target Q is already known. Different arithmetic policies define different numerical lanes, not independent edit predictions.',mechanics_status='pending-independent-audit',terminal_status='unassessed')
    save(out/'summary.json',result);save(out/'files.json',{p.relative_to(out).as_posix():sha(p)for p in sorted(out.rglob('*'))if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=result['job_id'],request_sha256=result['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,study):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    from mdb import parameter_artifacts
    from mdb.edit_search import file_digest as sha
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));read=lambda p:json.loads(p.read_text())
    contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=8,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=8,extra=dict(task_family='analysis'),checkpoint_policy='Eight CUDA Q-projection primitives over two already observed native operands;0fullmodel/block/imageforwards, four arithmetic policies; exact temporary checkpoint weight retrieval.')
    audit=read(study/'audit-v1.json')
    if audit['mechanics_status']!='verified':raise ValueError('audited stage corpus required')
    source=study/'collection-01';data=source/'extracted/binding-minimal-result';prior=read(source/'request.json');h['bind_environment'](prior['execution_environment']);summary=read(data/'summary.json');native=read(data/'native.json')
    inputs={n:sha(data/n)for n in('native.json','summary.json','parameter-artifacts.json')}
    for item in native.values():
        for key in('image.modulated','image.q_projection'):
            record=item['stages'][key];inputs[record['path']]=record['sha256']
    c=dict(input_root=summary['output_root']+'/binding-minimal-result',inputs=inputs,output_scratch='/mnt/big/mdb/research',directory_custody=True,checkpoint_root=read(data/'parameter-artifacts.json')['defaults']['checkpoint_root'],source_audit_sha256=sha(study/'audit-v1.json'))
    run.mkdir(parents=True,exist_ok=False);h['save'](run/'plan.json',c);extras=[(here/n,n)for n in('flux_family_custody.py','flux_reentry_reference.py','analyze_flux_precision.py')]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py')]
    pre=run/'preflight.v1.json';h['save'](pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[c['input_root'],c['output_scratch'],c['checkpoint_root'],'binding-minimal-result']))
    extras.append((pre,pre.name));request=dict(schema='mdb-cuda-gemm-precision-request-v1',config=c,needs=dict(cuda=True,host='beast'),execution_environment=prior['execution_environment'],task_family='analysis',backend='torch',device='cuda',dtype='policy-specific',source_hashes={n:sha(p)for p,n in[(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=contract,
        resource_exception=dict(ram_mb=2048,vram_mb=1024,cpu_threads=4,disk_gb=1,est_wall_s=90,calibration_job='job-a80061c1674b',measured_peak_rss_mb=1684.4,measured_peak_vram_mb=612,reason='Same two native image operands, one Q weight and eight CUDA primitive calls measured1684.4MBRSS/612MBVRAM,46.45s runner.2GB/1GB retain over360MB/410MB headroom plus strict ceiling. No full model/block/VAE forward.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();h['save'](run/'request.json',request);stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if(run/'submission.json').exists()or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('already admitted or source changed')
    resources={k:v for k,v in r.get('resource_exception',{}).items()if k in{'ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s'}}
    job=guarded_mrun_launch(worker_command(Path(__file__).name),request=r,model=None,task='analysis',experiment='mdb-cuda-gemm-precision',payload=run/'payload',gpu=True,needs=r['needs'],prefer_host='beast',timeout_s=360,retry_on_kill=False,preflight=True,**resources,
        note='CUDA primitive microscope:2observed image inputs x4 GEMM policies=8F.linear calls,0fullmodel/block/imageforwards. One temporary pinnedQweight, NumPy reference and perpolicy byte/error comparisons. Intent-first sizing;expectedunder120s/360souter,no retries. Different numerical lanes explicitly diagnostic.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('study',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args();print(json.dumps(submit(a.run.resolve())if a.submit else stage(a.run.resolve(),a.study.resolve()),sort_keys=True))
