"""Stage a small original-parameter microscope, without another model/image run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
H=runpy.run_path(str(HERE/'flux_source_receiver.py'))
read,save,sha=(H[k] for k in ('read','save','sha'))


def stage(run,prior):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    source=prior/'collection-01/extracted/binding-minimal-result';a=read(prior/'audit-v1.json')
    if a['mechanics_status']!='verified' or a['native_manifest_sha256']!=sha(source/'files.json'):
        raise ValueError('verified reader evidence required')
    # Reuse the original verified collection receipts; raw transport duplicates
    # may be reversibly externalized. Every consumed extracted file is checked
    # below and again on the worker. No new transport-verification claim is made.
    collection=prior/'collection-01';report=read(collection/'report.json')
    archive=read(collection/'archive-custody.json');log=read(collection/'log-custody.json')
    if (report['files_manifest_sha256']!=sha(source/'files.json') or report['job_id']!=a['job']
            or not archive['complete'] or not log['complete'] or archive['archive_sha256']!=report['archive_sha256']):
        raise ValueError('prior verified custody binding differs')
    old=read(source/'request.json');s=read(source/'summary.json');plan=read(source/'plan.json')
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](old['execution_environment'])
    manifest=read(source/'files.json');inputs={}
    def admit(n):
        expected=sha(source/n) if n=='files.json' else manifest[n]
        if sha(source/n)!=expected:raise ValueError('source changed')
        inputs[n]=dict(sha256=expected,bytes=(source/n).stat().st_size)
    for name in ('readers.json','cohort.json','plan.json','summary.json','files.json','parameter-artifacts.json','source-binding.json','retrieve_weights.py'):admit(name)
    for group in (read(source/'readers.json'),read(source/'cohort.json')['predictions']):
        for row in group.values():
            for record in row.get('fields',row).values():
                if isinstance(record,dict) and record.get('encoding'):admit(record['path'])
    parents={k:plan['projects'][k+'.base'] for k in ('left','right')}
    for fp in parents.values():
        n='store/cuts/'+fp+'.json';admit(n);cut=read(source/n)
        for page in cut['slots'].values():admit('store/pages/'+page+'.json')
        for blob in cut['physical_blobs']:admit('store/blobs/'+blob+'.safetensors')
    cfg=dict(schema='mdb-reader-arithmetic-plan-v1',input_root=s['output_root'],inputs=inputs,source_job=s['job_id'],
        source_files_sha256=sha(source/'files.json'),source_audit_sha256=sha(prior/'audit-v1.json'),
        model_path=plan['model_path'],text_parameters=plan['text_parameters'],eps=plan['layernorm_eps'],parents=parents,
        text_rows=512,width=3072,output_scratch='/mnt/big/mdb/research',directory_custody=True,max_wall_seconds=240,python_bytecode=False,
        reused_custody=dict(report_sha256=sha(collection/'report.json'),archive_receipt_sha256=sha(collection/'archive-custody.json'),
            log_receipt_sha256=sha(collection/'log-custody.json'),scope='Prior verified transport receipts plus rechecked extracted inputs; transport archive is not re-emitted or revalidated.'),
        original_module_source_sha256=sha(Path(importlib.metadata.distribution('diffusers').locate_file('diffusers/models/transformers/transformer_flux2.py'))),
        scope='Post-hoc diagnosis on eight known step3/joint0 reader cases. Three checkpoint-identical text matrices; full B1x512 geometry. Native actual-input, prior-reference-head, actual-after and opposite-reduction tails, plus24fixed-input opposite-reduction matrix calls and independent local/actual-input NumPy equations. No new target-free forecasts, full model/block/attention forwards or images.',
        planned_counts=dict(linear=112,layernorm=32,swiglu=32,mathematical_affines=64,full_model_loads=0,native_blocks=0,native_attention=0,images=0))
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',cfg)
    names=['flux_reader_arithmetic.py','flux_source_receiver.py','flux_reentry_reference.py','flux_family_custody.py','analyze_flux_reader_arithmetic.py']
    extras=[(HERE/n,n) for n in names]
    for name in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):
        shutil.copyfile(source/name,run/name);extras.append((run/name,name))
    worker=HERE/'flux_reader_arithmetic_worker.py';pre=run/'preflight.v1.json'
    save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],
        path_token_allowlist=[cfg['input_root'],cfg['model_path'],cfg['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    request=dict(schema='mdb-reader-arithmetic-request-v1',config=cfg,execution_environment=old['execution_environment'],
        needs=dict(cuda=True,host='beast'),task_family='analysis',backend='torch-and-numpy',device='cuda',dtype='bfloat16-native/float64-reference',
        source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=56,logical_branches=1,
            logical_checkpoint_replays=0,max_local_work_items=64,extra=dict(task_family='analysis'),
            checkpoint_policy='One finite operator lease;3 original text matrices,8 observed cases x4 native tails plus24fixed-input opposite-policy affines;112linear/32norm/32SwiGLU and64NumPy affines.0fullmodels/blocks/attention/images;no retries.'),
        resource_exception=dict(ram_mb=3072,vram_mb=1536,cpu_threads=4,disk_gb=2,est_wall_s=150,
            calibration_job='job-a80061c1674b',measured_peak_rss_mb=1684.4,measured_peak_vram_mb=612,
            reason='Prior CUDA Q-primitive lane measured1684.4MBRSS/612MBVRAM. Replace one18MB matrix with188.7MB total BF16 text weights; temporary FP32 weight views377.5MB and full512-row activations.3GBRAM/1.5GBVRAM allow added weights and bounded one-case fields. No full model/VAE or GPU FP64 copies.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=cfg,request=request,additional_files=extras)
    return dict(request_sha256=request['request_sha256'],input_files=len(inputs))


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    r=read(run/'request.json');os.environ.setdefault('MRUN_URL','http://192.168.68.68:9025')
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](r['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('changed or already submitted payload')
    resource={k:r['resource_exception'][k] for k in ('ram_mb','vram_mb','disk_gb','cpu_threads','est_wall_s')}
    command=worker_command('flux_reader_arithmetic_worker.py')
    index=next(i for i in range(len(command)) if command[i:i+4]==['python','-u','-m','mdb_lanes.worker_bootstrap'])
    command.insert(index+1,'-B')
    job=guarded_mrun_launch(command,request=r,model=None,task='analysis',
        experiment='mdb-flux-reader-arithmetic',payload=run/'payload',gpu=True,needs=r['needs'],prefer_host='beast',
        timeout_s=600,retry_on_kill=False,preflight=True,**resource,
        note='Observed-operand reader microscope:3original text matrices;8knowncases,112CUDAlinear/32LayerNorm/32SwiGLU and64independent NumPy affines. FullB1x512 geometry. Torch-only source-transcribed primitive graph; no Diffusers runtime import; original source SHA verified and original-return parity measured.0fullmodels/blocks/attention/VAE/newforecasts.150scompute estimate/240sworker/600souter;no retries.')
    save(run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.prior.resolve())))
