"""Stage one finite resident CUDA latent/reader assay from authenticated cuts."""
import argparse
import hashlib
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
    source=prior/'native-collection-01/extracted/binding-minimal-result'
    old=read(source/'request.json');summary=read(source/'summary.json')
    collection=read(prior/'native-collection-01/report.json')
    publication=read(prior/'native-collection-01/publication-join-v1.json')
    if (summary['job_id']!='job-3272cb7fc922' or collection['job_id']!=summary['job_id']
            or publication['publication']['files_manifest_sha256']!=sha(source/'files.json')):
        raise ValueError('authenticated original source-receiver native run required')
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](old['execution_environment'])
    manifest=read(source/'files.json');inputs={};commits={};projects={};calibration={}
    def admit(name):
        if sha(source/name)!=manifest[name]:raise ValueError('source file identity differs')
        inputs[name]=dict(sha256=manifest[name],bytes=(source/name).stat().st_size)
    def admit_cut(fp):
        name='store/cuts/'+fp+'.json';admit(name);cut=read(source/name)
        for page in cut['slots'].values():admit('store/pages/'+page+'.json')
        for blob in cut['physical_blobs']:admit('store/blobs/'+blob+'.safetensors')
    for receiver in ('left','right'):
        for kind in ('base','delta'):
            key=receiver+'.'+kind;row_name='rows/'+key+'.json';admit(row_name);row=read(source/row_name)
            records={(r['logical'],r['local']):r for r in row['records']}
            commits[key]=records[2,27]['cut'];projects[key]=records[3,0]['cut']
            admit_cut(commits[key]);admit_cut(projects[key])
            calibration[key]=dict(text=records[3,1]['outputs']['text']['archive'],
                latent=records[3,27]['outputs']['latent']['archive'],image=row['image'])
            for record in calibration[key].values():admit(record['path'])
    for name in ('source-binding.json','parameter-artifacts.json','retrieve_weights.py'):admit(name)
    spec=read(source/'source-binding.json')['joint0']
    cfg=dict(schema='mdb-latent-reader-plan-v1',model=old['model'],model_path=old['config']['model_path'],
        source_root=summary['output_root'],source_job=summary['job_id'],source_request_sha256=old['request_sha256'],
        source_files_sha256=sha(source/'files.json'),input_files=inputs,commits=commits,projects=projects,
        calibration=calibration,layernorm_eps=spec['layernorm_eps'],query_chunk=64,projection_chunk=256,
        text_parameters={k:v for k,v in spec['parameters'].items() if k.startswith(('attn.to_add_out.','ff_context.linear_in.','ff_context.linear_out.'))},
        output_scratch='/mnt/hdd8tb/mdb/research',directory_custody=True,cpu_threads=4,max_wall_seconds=600,
        scope='Two known seed127 receivers. Closure frontier: observed step3/project. Numerical frontier: actual native positioned textQ/completeKV before step3/joint0 attention. Eight local head/text-write predictions sealed before any new native attention. Sixteen later image futures observed; no full future numerical forecast or raw-prompt/semantic claim.',
        predictions=dict(closure='Baseline latent installed in changed commit reproduces baseline future; reciprocal transplant reproduces changed future. These are complete-live-state equivalence predictions, using known calibrations.',
            text_reader='With baseline image residual/query retained, changed image K+V suffice for the changed text joint0 exit. Split K/V arms decompose the actual read; directions/magnitudes and numerical errors are observations.'),
        counts=dict(prefix_stops=4,mathematical_text_predictions=8,native_transitions=440,completed_blocks=400,
            block_entries=404,native_sdpa=400,latent_acts=8,images=16),transport_limits=dict(max_expanded_bytes=3<<30,max_member_bytes=128<<20))
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',cfg)
    names=['flux_latent_reader.py','flux_source_receiver.py','flux_reentry_reference.py','flux_family_custody.py',
        'submit_flux_initial_noise.py','analyze_flux_latent_reader.py']
    extras=[(HERE/n,n) for n in names]
    for name in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):
        shutil.copyfile(source/name,run/name);extras.append((run/name,name))
    worker=HERE/'flux_latent_reader_worker.py';pre=run/'preflight.v1.json'
    save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],
        path_token_allowlist=[cfg['source_root'],cfg['model_path'],cfg['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name));raw=(json.dumps(cfg,sort_keys=True)+'\n').encode()
    request=dict(schema='mdb-latent-reader-request-v1',config=cfg,model=old['model'],backend='diffusers',task_family='forward',
        device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,needs=dict(cuda=True,host='beast'),
        execution_environment=old['execution_environment'],config_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=20,logical_branches=16,
            logical_checkpoint_replays=20,max_local_work_items=64,checkpoint_policy='One residentB1 model;4 abortedQKV prefixes,8 sealed mathematical localtext predictions,8latent closure futures and8KV reader futures;440successful transitions;no new encoding/noise;no retries.'),
        resource_exception=dict(ram_mb=8192,vram_mb=10240,disk_gb=3,cpu_threads=4,est_wall_s=360,
            calibration_job='job-3272cb7fc922',measured_peak_rss_mb=5662.8,measured_peak_vram_mb=9138,
            reason='Matched B1 512px same original transformer/VAE, no encoder. Prior350steps/13images peaked5662.8MBRSS/9138MBVRAM.8GBRAM admits extra~0.5GB temporaryFP32text weights and~0.1GB observedQKV.10240MBVRAM adds1102MB to measuredpeak, plus scheduler10percentceiling. No batch/dtype/kernel changes.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=cfg,request=request,config_bytes=raw,additional_files=extras)
    return dict(request_sha256=request['request_sha256'],source_files=len(inputs))


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    request=read(run/'request.json');os.environ.setdefault('MRUN_URL','http://192.168.68.68:9025')
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](request['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=v for n,v in request['source_hashes'].items()):
        raise ValueError('changed or previously submitted payload')
    command=H['helper']('submit_flux_initial_noise.py')['sealed_worker_command'](request);command[-1]='flux_latent_reader_worker.py'
    resource=request['resource_exception']
    job=guarded_mrun_launch(command,request=request,model=request['model'],task='forward',experiment='mdb-flux-latent-reader',
        payload=run/'payload',pin='beast',gpu=True,needs=request['needs'],preflight=True,retry_on_kill=False,
        ram_mb=resource['ram_mb'],vram_mb=resource['vram_mb'],disk_gb=resource['disk_gb'],cpu_threads=4,
        est_wall_s=resource['est_wall_s'],timeout_s=900,priority=50,
        note='Latent closure plus key/value reader:1residentCUDA B1load;4prefixstops beforeattention;8localmathforecasts sealed before anyattentiontarget;16futures/440steps/400blocks/16VAEimages;0encodes/noise. Matched9138MBVRAMcalibration;360sestimate/600sworker;no retries.')
    save(run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.prior.resolve())))
