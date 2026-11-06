"""Stage source-bound CPU forecasts then one resident native MDB panel."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = runpy.run_path(str(HERE/'flux_source_receiver.py'))
read, save, sha = (H[k] for k in ('read','save','sha'))


def bind(environment):
    from mdb.execution_environment import intent
    for key,name in dict(uv_cache_dir='MDB_WORKER_UV_CACHE_DIR',torch_home='MDB_WORKER_TORCH_HOME',
            extra_model_roots='MDB_WORKER_MODEL_ROOTS',python='MDB_WORKER_PYTHON',uv_link_mode='MDB_WORKER_UV_LINK_MODE').items():
        os.environ[name]=environment[key]
    actual=intent()
    if actual != {k:environment[k] for k in actual}:
        raise ValueError('installed worker requirement cohort differs')


def stage(run, prior, phase='forecast', forecast=None):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    source = prior/'run-01/extracted/flux-prompt-factorial-result'
    old = read(prior/'run-01/request.json')
    custody = read(prior/'run-01/collection.json')
    if custody['job_id'] != 'job-95453050cf73' or custody['scheduler_status'] != 'succeeded':
        raise ValueError('authenticated original run required')
    bind(old['execution_environment'])
    inventory = read(source/'files.json')
    parents = {}
    inputs = {}
    def admit(name):
        p = source/name
        if inventory[name] != sha(p):
            raise ValueError('original input inventory changed')
        inputs[name] = dict(sha256=sha(p),bytes=p.stat().st_size)
    for cell in ('c0-o0','c1-o0','c0-o1','c1-o1'):
        rows = [json.loads(line) for line in (source/'native'/f'{cell}.jsonl').read_text().splitlines()]
        r = rows[61]
        parents[cell] = r['capture']['cut']
        name = 'typed-store/cuts/'+parents[cell]+'.json'
        admit(name)
        cut = read(source/name)
        for page in cut['slots'].values():
            admit('typed-store/pages/'+page+'.json')
        for blob in cut['physical_blobs']:
            admit('typed-store/blobs/'+blob+'.safetensors')
    for name in ('binding.json','parameter-artifacts.json','source.json'):
        admit(name)
    binding = read(source/'binding.json')
    origin_parents={cell:binding['origins'][cell]['cut'] for cell in ('c0-o0','c0-o1')}
    for fp in origin_parents.values():
        name='typed-store/cuts/'+fp+'.json';admit(name);cut=read(source/name)
        for page in cut['slots'].values():admit('typed-store/pages/'+page+'.json')
        for blob in cut['physical_blobs']:admit('typed-store/blobs/'+blob+'.safetensors')
    for logical in ('2','3'):
        for key in ('timestep','sigma','sigma_next'):
            admit(binding['pass_schedules'][logical][key]['path'])
    config = dict(schema='mdb-source-receiver-plan-v1',phase=phase,
        model=old['model'],model_path=old['config']['model_path'],
        source_job=custody['job_id'],source_request_sha256=custody['request_sha256'],
        source_root=custody['remote_path'],source_files_sha256=sha(source/'files.json'),
        source_custody_sha256=sha(prior/'run-01/collection.json'),parents=parents,origin_parents=origin_parents,input_files=inputs,
        point=r['capture']['point'],output_scratch='/mnt/hdd8tb/mdb/research',arms=H['arms'](),
        cpu_threads=4,query_chunk=64,projection_chunk=256,neuron_chunk=256,
        max_parameter_cache_bytes=256<<20,
        long_run_review_file='long-run-review-launch-v1.json',
        sample_indices=[0,31,511,512,543,767,1023,1535],image_sample_indices=[0,31,32,255,511,512,767,1023],
        max_wall_seconds=2100 if phase=='forecast' else 750,
        scope='Observed seed127 step2/single0 four natural source cuts; fixed left-layout full working-text delta crossed with left/right receivers. Six own-state fifty-transition mathematical futures plus one sham reuse. New interventions, known source contexts; no encoder/raw-prompt/concept dictionary or kernel-exact claim.',
        hypotheses=dict(single0='Within each receiver, text-only changes preserve all sampled image norm/QKV and parallel MLP operands exactly; attention can change image exit. At single1 the changed image receiver may alter its local MLP.',
            continuation='Own predicted latent feeds next projection and all later reads. Endpoint, signed source effect and receiver interaction errors reported separately; no arbitrary accuracy gate.',
            alternatives=['transferable contextual source','receiver-specific source representation','address dependence','numerical discrepancy','later conditioning overwrite']),
        transport_limits=dict(max_expanded_bytes=3<<30,max_member_bytes=128<<20),
        calibration={cell:read(source/'rows'/f'{cell}.native.json') for cell in ('c0-o0','c0-o1')})
    run.mkdir(parents=True,exist_ok=False)
    for name,src in [('source-binding.json',source/'binding.json'),('parameter-artifacts.json',source/'parameter-artifacts.json'),('retrieve_weights.py',source/'retrieve_weights.py')]:
        shutil.copyfile(src,run/name)
    if phase=='native':
        f = read(forecast/'summary.json')
        if f['counts']['native_model_forwards'] != 0 or f['counts']['mathematical_transitions'] != 300:
            raise ValueError('complete prospective forecast cohort required')
        config.update(forecast_root=f['output_root'],forecast_job=f['job_id'],
            forecast_files_sha256=sha(forecast/'files.json'),forecast_cohort_sha256=sha(forecast/'cohort.json'))
    save(run/'plan.json',config)
    names = ['flux_source_receiver.py','flux_family_custody.py','flux_context_transfer.py',
        'flux_prompt_factorial.py','flux_initial_noise_campaign.py','flux_two_pass_campaign.py',
        'flux_final_latent_campaign.py','flux_joint_tail_campaign.py','flux_joint_chain_campaign.py',
        'flux_reentry_campaign.py','flux_reentry_reference.py','flux_step_boundary_campaign.py',
        'flux_step_boundary_reference.py','flux_transition_reference.py','flux_mlp_replacement.py',
        'flux_single_contexts.py','flux_attention_replacement_campaign.py','flux_native_joint_ports.py',
        'submit_flux_initial_noise.py']
    extras = [(HERE/n,n) for n in names]
    if phase=='native':
        extras += [(HERE/n,n) for n in ('analyze_flux_source_receiver.py','publish_flux_source_receiver.py')]
    extras += [(run/n,n) for n in ('plan.json','source-binding.json','parameter-artifacts.json','retrieve_weights.py')]
    worker = HERE/('flux_source_receiver_'+phase+'.py')
    pre = run/'preflight.v1.json'
    save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],
        path_token_allowlist=[config['source_root'],config['model_path'],config['output_scratch'],'binding-minimal-result',config.get('forecast_root',config['source_root'])]))
    extras.append((pre,pre.name))
    raw = (json.dumps(config,sort_keys=True)+'\n').encode()
    request = dict(schema='mdb-source-receiver-request-v1',config=config,model=config['model'],backend='diffusers',task_family='analysis' if phase=='forecast' else 'forward',
        device='cpu' if phase=='forecast' else 'cuda',dtype='bfloat16',batch_size=1,height=512,width=512,
        needs={'host':'beast',**({'cuda':True} if phase=='native' else {})},execution_environment=old['execution_environment'],
        config_bytes_sha256=hashlib.sha256(raw).hexdigest(),source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=0 if phase=='forecast' else 1,model_free=phase=='forecast',
            logical_operations=6 if phase=='forecast' else 7,logical_branches=6 if phase=='forecast' else 7,
            logical_checkpoint_replays=4 if phase=='forecast' else 7,max_local_work_items=64,
            extra={'task_family':'analysis'} if phase=='forecast' else None,
            checkpoint_policy='CPU-only complete forecast cohort before one resident CUDA original-consumer replay of all seven branches; no native prefix reexecution or encoder. Serial batch1 numerical contract.'),
        resource_exception=dict(ram_mb=6144 if phase=='forecast' else 10000,vram_mb=0 if phase=='forecast' else 13200,disk_gb=4,
            calibration_job='job-8367b9ecabe1' if phase=='forecast' else 'job-4350531b1dcd',
            measured_peak_rss_mb=3922.7 if phase=='forecast' else 8509.1,measured_peak_vram_mb=0 if phase=='forecast' else 12936,
            reason='CPU forecast uses a256MiB immutable originalBF16 LRU over checkpoint offsets verified by originalretriever; no native pipeline. One block working set, no full trajectory resident. Prior original joint2 CPU assay job-8367b9ecabe1 peaked3922.7MB including nativeblock+12complete forecasts;6144MB admits bounded mathematical working set. GPU10000MB uses original full-pipeline8509.1MB measured peak, with encoder omitted and no resident parameter bank;13200MBVRAM exceeds original12936MB peak with a14520MB scheduler ceiling. Replaces13600MB queued-only request blocked by a512MB CUDA reservation; no workload change.'))
    request['request_sha256'] = hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run/'request.json',request)
    stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def launch_command(request):
    command=H['helper']('submit_flux_initial_noise.py')['sealed_worker_command'](request)
    phase=request['config']['phase']
    command[-1]='flux_source_receiver_'+phase+'.py'
    if phase=='forecast':
        if command[0]!='/usr/bin/env':
            raise ValueError('worker environment entrypoint differs')
        command.insert(1,'CUDA_VISIBLE_DEVICES=')
    return command


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    request = read(run/'request.json')
    os.environ.setdefault('MRUN_URL','http://192.168.68.68:9025')
    bind(request['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=v for n,v in request['source_hashes'].items()):
        raise ValueError('changed or previously submitted payload')
    review = read(run.parent/request['config'].get('long_run_review_file','long-run-review.json'))
    if request['config']['phase']=='forecast' and (review['request_sha256']!=request['request_sha256'] or review['blocking_issues']):
        raise ValueError('source-bound long-run review required')
    phase=request['config']['phase']
    command=launch_command(request)
    resource=request['resource_exception']
    job=guarded_mrun_launch(command,request=request,model=None if phase=='forecast' else request['model'],
        task='analysis' if phase=='forecast' else 'forward',experiment='mdb-flux-source-receiver-'+phase,payload=run/'payload',pin='beast',gpu=phase=='native',
        needs=request['needs'],preflight=True,retry_on_kill=False,cpu_threads=4,
        priority=50,
        ram_mb=resource['ram_mb'],vram_mb=resource['vram_mb'],disk_gb=resource['disk_gb'],
        timeout_s=2400 if phase=='forecast' else 1500,est_wall_s=1500 if phase=='forecast' else 600,
        note='Source x receiver computation: '+('CPU-only300 NumPy own-state transitions,6forecasts sealed before nativetargets; original BF16 parameter bits memoized, no GPU reservation.' if phase=='forecast' else 'One CUDA B1 resident load;7same-parent native50-transition continuations/350steps,0encodes/noise; full forecast cohort already sealed; original single0/1 and next-pass joint0 ports;7VAE outputs.')+' No retries;known seed127 sources,new interventions;retain numerical and semantic misses.')
    save(run/'submission.json',dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--phase',choices=['forecast','native'],default='forecast')
    p.add_argument('--forecast',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.prior.resolve(),a.phase,a.forecast),sort_keys=True))
