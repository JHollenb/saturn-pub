"""Stage one resident native suffix campaign after a verified left-only selection."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
H=runpy.run_path(str(HERE/'flux_source_receiver.py'))
read,save,sha=[H[k] for k in ('read','save','sha')]


def stage(run,prior,route_audit):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    root=route_audit/'compact-collection-01/extracted/binding-minimal-result';a=read(root/'audit.json')
    if a['mechanics_status']!='verified' or a['instrument_status']!='source-matched' or a['head_recipes_verified']!=116:
        raise ValueError('verified route instrument required')
    if any(sha(root/n)!=v for n,v in read(root/'files.json').items()):raise ValueError('route audit changed')
    old=read(prior/'request.json');c=dict(old['config']);c.pop('resource_exception');c.pop('planned_counts')
    c.update(schema='mdb-reader-route-future-plan-v1',model='flux2-klein-4b-distilled',max_wall_seconds=600,
        selected_head=a['summaries']['left']['solo_heads'][0],selected_band=a['summaries']['left']['solo_bands'][0],
        route_audit_sha256=sha(root/'audit.json'),route_screen_job=a['source_job'],
        scope='16native B1 futures from two known step3/joint0 parents:4historical anchors,8head/row interventions with known local text effects but new futures,4new K/V dose reader states. All16primitive head/text predictions and structural image-divergence expectations sealed before any native block. Unchanged native suffix and VAE; original weights; no whole-image numerical prediction, new seed/raw-prompt or semantic universality claim.',
        planned_counts=dict(sdpa=12,staged_matmuls=0,staged_softmax=0,linear=48,layernorm=16,swiglu=16,
            native_sdpa=400,native_blocks=400,native_transitions=432,images=16,model_loads=1,encodes=0,noise_draws=0))
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',c)
    save(run/'selection.json',dict(source_audit_sha256=c['route_audit_sha256'],source_screen_job=a['source_job'],receiver='left',
        selected_head=c['selected_head'],selected_band=c['selected_band'],head_order=a['summaries']['left']['solo_heads'],band_order=a['summaries']['left']['solo_bands']))
    names=('flux_reader_route_future.py','flux_reader_routes.py','flux_reader_arithmetic.py','flux_reader_attention.py','flux_source_receiver.py','flux_reentry_reference.py','flux_family_custody.py')
    extras=[(HERE/n,n) for n in names]+[(run/'selection.json','selection.json')]
    for name in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):
        shutil.copyfile(prior/name,run/name);extras.append((run/name,name))
    worker=HERE/'flux_reader_route_future_worker.py';pre=run/'preflight.v1.json'
    save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],
        path_token_allowlist=[c['input_root'],c['model_path'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    req=dict(schema='mdb-reader-route-future-request-v1',config=c,model=c['model'],execution_environment=old['execution_environment'],needs=dict(cuda=True,host='beast'),
        task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,
        source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        resource_exception=dict(ram_mb=9216,vram_mb=10240,cpu_threads=4,disk_gb=2,est_wall_s=150,
            calibration_job='job-4c4cf4ac3624',measured_peak_rss_mb=8209.8,measured_peak_vram_mb=9264.,
            reason='Exact16case B1 512px native transformer/VAE panel measured8209.8MBRSS/9264MBVRAM and98.26s runner.9GBRAM/10GBVRAM adds measured headroom without changing scalar shape, kernels or model/weight residency.150s estimate retains startup variance; worker600s/outer900s unchanged.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=32,logical_branches=16,logical_checkpoint_replays=16,max_local_work_items=64,
            checkpoint_policy='One model acquisition;16primitive reader forecasts all sealed before16native suffixes.12forecastFlash/48linear/16norm/16SwiGLU,432native steps/400blocks/16VAEimages;0encodes/noise/retries. Source head7/band2 selected from left route screen before future execution.'))
    req['request_sha256']=hashlib.sha256(json.dumps(req,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',req)
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](req['execution_environment']);stage_worker(run/'payload',worker=worker,config=c,request=req,additional_files=extras)
    return dict(request_sha256=req['request_sha256'],head=c['selected_head'],band=c['selected_band'])


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    r=read(run/'request.json');runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](r['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('changed/already submitted future campaign')
    command=worker_command('flux_reader_route_future_worker.py');command.insert(command.index('python')+1,'-B');os.environ.setdefault('MRUN_URL','http://192.168.68.68:9025')
    resources={k:r['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')}
    job=guarded_mrun_launch(command,request=r,model=r['model'],task='forward',experiment='mdb-flux-reader-route-future',payload=run/'payload',gpu=True,
        needs=r['needs'],prefer_host='beast',timeout_s=900,preflight=True,retry_on_kill=False,**resources,
        note='16native reader futures, B1/512px:4anchors+8selectedhead/bandinterventions+4new3/8,5/8KVdosecases.16primitive text predictions/structuralfirst-image expectations sealed before firstblock.1transformer/VAE acquisition,432steps/400blocks/16images,0encodes/noise. Measured8209.8MBRAM/9264MBVRAM/98.26s calibration;9GB/10GB reserve,150sestimate/600sworker/900souter,no retries.')
    save(run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--route-audit',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.prior.resolve(),a.route_audit.resolve())))
