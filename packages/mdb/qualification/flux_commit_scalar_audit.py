"""Post-hoc saved-only diagnosis of CUDA/BF16 scheduler scalar promotion."""
import argparse
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def worker():
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='2'
    import numpy as np
    h=runpy.run_path('flux_source_receiver.py');read,save,sha,load=[h[k] for k in ('read','save','sha','load_array')]
    bf16=runpy.run_path('flux_reentry_reference.py')['bf16'];measure=runpy.run_path('analyze_flux_reader_arithmetic.py')['measure']
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=req['config'];root=Path(c['native_root']);started=time.monotonic()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('diagnostic isolation/source differs')
    if sha(root/'files.json')!=c['native_manifest_sha256']:raise ValueError('native inventory differs')
    inventory=read(root/'files.json')
    for name in ('source-operands.json','cases.json','predictions.json'):
        if sha(root/name)!=inventory[name]:raise ValueError('native metadata changed')
    operands,cases,predictions=[read(root/n) for n in ('source-operands.json','cases.json','predictions.json')];rows={}
    for key,p in predictions.items():
        row=cases[key];s=row['spec'];r=s['receiver'];z=load(root,operands[s['z_source']]['latent']);n=load(root,operands[s['n_source']]['noise'])
        sig=np.asarray(operands[r]['sigmas'],np.float32);dt=np.float32(sig[3]-sig[2]);db=bf16(np.asarray([dt],np.float32))[0]
        actual=load(root,row['component']['latent']);actual_product=load(root,p['product'])
        candidates={'fp32-scalar-bf16-product':(bf16(dt*n),dt),'bf16-scalar-bf16-product':(bf16(db*n),db),
            'fp32-scalar-fp32-product':(dt*n,dt)}
        rows[key]=dict(delta_sigma_fp32=float(dt),delta_sigma_bf16=float(db),rules={})
        for name,(product,scalar) in candidates.items():
            value=bf16(z+product)
            rows[key]['rules'][name]=dict(product=measure(actual_product,product),commit=measure(actual,value))
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    save(out/'audit.json',dict(schema='mdb-flux-commit-scalar-diagnostic-v1',source_job=read(root/'summary.json')['job_id'],mechanics_status='verified',instrument_status='saved-operand-comparisons',
        trend_status='observed',terminal_status='unassessed',role='Post-hoc diagnosis after NumPy FP32-scalar rule missed; not a fresh prospective numerical prediction.',
        native_manifest_sha256=sha(root/'files.json'),rows=rows,new_execution_counts=dict(model_calls=0,native_calls=0,learned_forwards=0,saved_elementwise_commit_evaluations=48),
        limitation='A matching saved-array rule characterizes this CUDA/BF16 tensor/scalar profile; independent backend/source profiles remain untested. Original audit and misses stay immutable.'))
    for n in ('parameter-artifacts.json','retrieve_weights.py'):shutil.copyfile(root/n,out/n)
    save(out/'plan.json',c);save(out/'request.json',req)
    save(out/'summary.json',dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),seconds=time.monotonic()-started,mechanics_status='verified'))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_reader_attention_audit.py'));p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('run',type=Path);p.add_argument('--study',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(h['submit'](a.run.resolve()) if a.submit else h['stage'](a.run.resolve(),a.study.resolve(),worker_path=Path(__file__),extra_files=[(here/'flux_reentry_reference.py','flux_reentry_reference.py')],audit_kind='commit-scalar',logical_operations=48)))
