"""Source-bound staging and strict, finite mrun admission for binding transfer."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = runpy.run_path(str(HERE / 'flux_binding_generalization.py'))
read, save, sha, identity = [H[k] for k in ('read', 'save', 'sha', 'identity')]


def stage(run, prior, schedule_prior):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    old, schedule_old = read(prior / 'request.json'), read(schedule_prior / 'request.json')
    origin = schedule_old['config']['input_root']
    fp = schedule_old['config']['origin_parents']['c0-o0']
    # Read only the two tiny source schedule tensors; no model execution by SSH.
    code = '''import json,pathlib,struct,hashlib
r=pathlib.Path(ROOT);p=r/'store/cuts'/FP;c=json.loads(p.read_text());result={}
for name in ('timesteps','sigmas'):
 slot=next(x['value'] for x in c['manifest']['payload']['items'] if x['key']['value']==name)
 for blob in c['physical_blobs']:
  q=r/'store/blobs'/(blob+'.safetensors')
  with q.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
   for key,value in header.items():
    if key=='__metadata__' or value['shape']!=slot['shape']:continue
    a,z=value['data_offsets'];f.seek(8+n+a);raw=f.read(z-a)
    if hashlib.sha256(raw).hexdigest()!=slot['sha256']:continue
    result[name]=list(struct.unpack('<'+'f'*((z-a)//4),raw))
assert set(result)=={'timesteps','sigmas'}
result['source']=dict(root=str(r),cut=FP,cut_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((r/'files.json').read_bytes()).hexdigest())
print(json.dumps(result))
'''
    code = code.replace('ROOT', repr(origin)).replace('FP', repr(fp + '.json'))
    schedule = json.loads(subprocess.run(['ssh', 'beast', 'python3', '-I', '-'], input=code,
                                         text=True, capture_output=True, check=True).stdout)
    if schedule['source']['manifest_sha256'] != schedule_old['config']['source_files_sha256']:
        raise ValueError('source schedule inventory differs')
    config = dict(schema='mdb-binding-generalization-plan-v1', model=old['model'],
        model_path=old['config']['model_path'], original_source_sha256=old['config']['original_source_sha256'],
        checkpoint_files=schedule_old['config']['checkpoint_files'], schedule=schedule,
        contexts=H['contexts'](), arms=H['arms'](), output_scratch='/mnt/big/mdb/research',
        directory_custody=True, max_wall_seconds=1080, expected_counts=H['expected_counts'](),
        scope='Eight contexts/two shared noise seeds; paired-source changed-token addresses. Honest natural paths and image-only priming precede144sealed projection forecasts;88native images. Original Linear primitives predict I/T projections only; writer equations are post-hoc known-input audits, no attention or pixel forecast.',
        hypotheses=dict(stage='Exploratory transfer hypotheses frozen before all new encodes and images; counterexamples retained.',
            source_assignment=['IC.early', 'IC.restoreI'], base_assignment=['base-replay', 'I', 'C', 'IC.late', 'IC.control-rows', 'IC.restoreT', 'IC.restoreIT'],
            continuation='Joint I/T restore after step2/joint0 rejoins matched I-only payload after base C reset at project3 and final image exactly.',
            qkv='At step2/joint0 image Q/K/V invariant to C under identical priming/latent/clock; text-dependent attention/MLP may change.',
            applicability='Same pattern may fail across attributes, receiving state, token alignment or native-reference competence.',
            semantics='Review both object attributes, shape/order/visibility/collateral from actual PNGs. RGB contrast projection is not semantic accuracy.'),
        parameter_recipe_sha256=sha(prior / 'parameter-artifacts.json'))
    run.mkdir(parents=True, exist_ok=False)
    save(run / 'plan.json', config)
    extras = []
    for name in ('parameter-artifacts.json', 'retrieve_weights.py', 'source-binding.json'):
        shutil.copyfile(prior / name, run / name)
        extras.append((run / name, name))
    for name in ('flux_source_receiver.py', 'flux_family_custody.py', 'retrieve_flux_encoder.py'):
        extras.append((HERE / name, name))
    worker = HERE / 'flux_binding_generalization.py'
    pre = run / 'preflight.v1.json'
    save(pre, dict(schema_version='mrun-preflight-config-v1', hashed_roots=[worker.name, *[n for _, n in extras], pre.name],
        path_token_allowlist=[config['model_path'], config['output_scratch'], 'binding-minimal-result', origin]))
    extras.append((pre, pre.name))
    req = dict(schema='mdb-binding-generalization-request-v1', worker_script=worker.name, config=config,
        model=config['model'], execution_environment=old['execution_environment'], needs=dict(cuda=True, host='beast'),
        task_family='forward', backend='diffusers', device='cuda', dtype='bfloat16', batch_size=1, height=512, width=512,
        source_hashes={n: sha(p) for p, n in [(worker, worker.name), *extras]},
        resource_exception=dict(ram_mb=12288, vram_mb=10240, cpu_threads=4, disk_gb=8, est_wall_s=840,
            calibration_job='job-6a35eb9ce352', measured_peak_rss_mb=8889.3, measured_peak_vram_mb=9372,
            reason='Measured original encoder sequentially released before transformer:80futures/2500ops257.69s,8889.3MBRSS/9372MBVRAM. Current6008ops/5368blocks/88VAEs/16encodes, stream BF16/CAS evidence and release per-branch tensors. Same resident geometry,12GiBRAM/10GiBVRAM/8GBdisk;840sestimated,1080sworker/1200souter, no parameter duplicates or retries.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=104, logical_branches=88,
            logical_checkpoint_replays=80, max_local_work_items=128,
            checkpoint_policy='One pipeline acquisition;16original prompt encodes then encoder released;2shared noise draws,16natural paths,8image priming prefixes,72sealed new futures;6008nativeops/5368blocks/184Acts/88VAEs. SerialB1, no retries.'))
    # worker_bootstrap's public request envelope uses default JSON separators.
    req['request_sha256'] = hashlib.sha256(json.dumps(req, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / 'request.json', req)
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    stage_worker(run / 'payload', worker=worker, config=config, request=req, additional_files=extras)
    return dict(request_sha256=req['request_sha256'], schedule=schedule, counts=config['expected_counts'])


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    req = read(run / 'request.json')
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    if (run / 'submission.json').exists() or any(sha(run / 'payload' / n) != v for n, v in req['source_hashes'].items()):
        raise ValueError('changed/already submitted payload')
    command = worker_command(req['worker_script'])
    command.insert(command.index('python') + 1, '-B')
    os.environ.setdefault('MRUN_URL', 'http://192.168.68.68:9025')
    resources = {k: req['resource_exception'][k] for k in ('ram_mb', 'vram_mb', 'cpu_threads', 'disk_gb', 'est_wall_s')}
    job = guarded_mrun_launch(command, request=req, model=req['model'], task='forward', experiment='mdb-binding-generalization',
        payload=run / 'payload', gpu=True, needs=req['needs'], prefer_host='beast', timeout_s=1200, preflight=True,
        retry_on_kill=False, **resources,
        note='8matched attribute-binding contexts,2shared seeds.16encodes sequential then release encoder;6008nativeops/5368blocks/184Acts/88VAEs,72newfutures after144projectionforecasts sealed;source-assisted priming/restore,0pixelforecasts. One finiteB1CUDAlease,840sestimate/1080sworker/1200souter,0retries. Retain every context/control/counterexample; nativeprefix explicitly counted.')
    save(run / 'submission.json', dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--schedule-prior', type=Path)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    print(json.dumps(submit(args.run.resolve()) if args.submit else stage(args.run.resolve(), args.prior.resolve(), args.schedule_prior.resolve()), sort_keys=True))
