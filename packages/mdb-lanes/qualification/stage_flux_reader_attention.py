"""Stage seven attention arithmetic lanes on the existing eight reader cases."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = runpy.run_path(str(HERE/'flux_source_receiver.py'))
read, save, sha = H['read'], H['save'], H['sha']


def stage(run, prior, arithmetic):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    source = prior/'collection-01/extracted/binding-minimal-result'
    old = read(arithmetic/'run-05/request.json')
    a = read(prior/'audit-v1.json')
    if a['mechanics_status'] != 'verified' or sha(source/'files.json') != a['native_manifest_sha256']:
        raise ValueError('verified source reader campaign required')
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](old['execution_environment'])
    c = old['config']; inputs = dict(c['inputs']); manifest = read(source/'files.json')
    def admit(name):
        expected = sha(source/name) if name == 'files.json' else manifest[name]
        if sha(source/name) != expected:raise ValueError('source inventory differs')
        inputs[name] = dict(sha256=expected,bytes=(source/name).stat().st_size)
    admit('prefixes.json')
    for fields in read(source/'prefixes.json')['operands'].values():
        for record in fields.values():admit(record['path'])
    for name,r in inputs.items():
        if sha(source/name) != r['sha256'] or (source/name).stat().st_size != r['bytes']:
            raise ValueError('previous staged source differs')
    original = {n:sha(Path(importlib.metadata.distribution('diffusers').locate_file(n))) for n in
        ('diffusers/models/attention_dispatch.py','diffusers/models/transformers/transformer_flux2.py')}
    cfg = dict(schema='mdb-reader-attention-plan-v1', input_root=c['input_root'], inputs=inputs,
        source_job=c['source_job'], source_files_sha256=sha(source/'files.json'),
        model_path=c['model_path'], text_parameters=c['text_parameters'], parents=c['parents'], eps=c['eps'], text_rows=512,
        output_scratch='/mnt/big/mdb/research', directory_custody=True, max_wall_seconds=180,
        original_source_sha256=original,
        scope='Known-input numerical diagnosis, eight original step3/joint0 K/V corners. Exact captured positioned text Q/full K/V, synthetic unused query-row geometry controls, native SDPA kernel policies, explicit FP32 score/softmax/value programs, original-weight native text tails. No new raw-prompt prediction, full models/blocks/attention modules, images or learned coefficients.',
        planned_counts=dict(sdpa_max=40, staged_matmuls=32, staged_softmax=16, linear=48, layernorm=16, swiglu=16,
                            full_models=0, blocks=0, attention_modules=0, images=0),
        resource_exception=dict(ram_mb=3072,vram_mb=1536,cpu_threads=4,disk_gb=1,est_wall_s=120,
            calibration_job='job-486f616ef636',measured_peak_rss_mb=2202.,measured_peak_vram_mb=678.,
            reason='Same three188.7MB original text matrices;16rather than32native tails; no CPU FP64 affine work. Added serial512x1536x24 FP32 scores/probabilities, bounded~225MB CUDA temporaries, full1536query auto fused lanes.3GBRAM/1.5GBVRAM covers prior measured2202MBRSS/678MBVRAM plus attention buffers. No full component.'))
    run.mkdir(parents=True,exist_ok=False); save(run/'plan.json',cfg)
    names = ['flux_reader_attention.py','flux_reader_arithmetic.py','flux_source_receiver.py','flux_reentry_reference.py','flux_family_custody.py']
    extras = [(HERE/n,n) for n in names]
    for name in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):
        shutil.copyfile(source/name,run/name); extras.append((run/name,name))
    worker = HERE/'flux_reader_attention_worker.py'; pre = run/'preflight.v1.json'
    save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],
        path_token_allowlist=[cfg['input_root'],cfg['model_path'],cfg['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    req = dict(schema='mdb-reader-attention-request-v1',config=cfg,execution_environment=old['execution_environment'],
        needs=dict(cuda=True,host='beast'),task_family='analysis',backend='torch',device='cuda',dtype='bfloat16-native/fp32-stages',
        source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=72,
            logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=80,extra=dict(task_family='analysis'),
            checkpoint_policy='One original-matrix operator lease,8knowncorners x7attentionlanes plus16native text tails;40SDPA maximum,32stagedmatmuls/16softmax,48linear/16norm/16SwiGLU.0fullmodels/blocks/images;no retries.'))
    req['request_sha256'] = hashlib.sha256(json.dumps(req,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run/'request.json',req); stage_worker(run/'payload',worker=worker,config=cfg,request=req,additional_files=extras)
    return dict(request_sha256=req['request_sha256'])


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    r = read(run/'request.json')
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](r['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=v for n,v in r['source_hashes'].items()):
        raise ValueError('already submitted or changed payload')
    command = worker_command('flux_reader_attention_worker.py')
    command.insert(command.index('python')+1,'-B')
    os.environ.setdefault('MRUN_URL','http://192.168.68.68:9025')
    resources = {k:r['config']['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')}
    job = guarded_mrun_launch(command,request=r,model=None,task='analysis',experiment='mdb-flux-reader-attention',
        payload=run/'payload',gpu=True,needs=r['needs'],prefer_host='beast',timeout_s=600,retry_on_kill=False,preflight=True,
        **resources,note='Fixed-operand reader attention:8knownK/Vcorners,40SDPAmax+32explicitmatmuls/16softmax;2querypaddingcontrols;actualselectedATenops captured;16original-weight native text tails.0fullmodels/blocks/attentionmodules/images/newtargets.120sestimate/180sworker/600souter,no retries. Source-local3GBRAM/1.5GBVRAM measured exception.')
    save(run/'submission.json',dict(job_id=str(job))); return dict(job_id=str(job))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--arithmetic',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.prior.resolve(),a.arithmetic.resolve())))
