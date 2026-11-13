"""Stage four fresh writes and fourteen counted native image consumers."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,precision,response,diagnostic):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes import parameter_artifacts
    here=Path(__file__).resolve().parent; h=runpy.run_path(str(here/'flux_binding_minimal.py')); sha,save=h['sha'],h['save']; read=lambda p:json.loads(p.read_text())
    if read(precision/'verification-v1.json')['mechanics_status']!='verified' or read(response/'verification-v1.json')['mechanics_status']!='verified' or read(diagnostic/'audit-v1.json')['mechanics_status']!='verified': raise ValueError('verified development/source evidence required')
    roots=dict(prefixes=precision/'collection-01/extracted/binding-minimal-result',history=response/'collection-04/extracted/binding-minimal-result')
    inventories={key:read(root/'files.json') for key,root in roots.items()}; files={}
    def add(key,source,destination):
        source=Path(source); p=roots[key]/source; digest=inventories[key][source.as_posix()]
        if sha(p)!=digest: raise ValueError('source evidence changed')
        item=dict(root=key,source=source.as_posix(),sha256=digest,bytes=p.stat().st_size)
        if destination in files and files[destination]['sha256']!=digest: raise ValueError('source destination collision')
        files[destination]=item
    def cut(key,fp,prefix=Path()):
        name=Path('store/cuts')/(fp+'.json'); record=read(roots[key]/prefix/name); add(key,prefix/name,name.as_posix())
        for page in record['slots'].values():
            name=Path('store/pages')/(page+'.json'); add(key,prefix/name,name.as_posix())
        for blob in record['physical_blobs']:
            name=Path('store/blobs')/(blob+'.safetensors'); add(key,prefix/name,name.as_posix())
    old=read(roots['history']/'plan.json'); rows={(r['id'],r['mode']):r for r in read(roots['history']/'rows.json')}; native=read(roots['prefixes']/'native.json')
    for fp in old['parents'].values(): cut('history',fp,Path('source'))
    anchors={}
    for name in ('half','half.image1'):
        row=rows[name,'native']; target=row['target']
        if any(target[k]!=native[name][k] for k in ('entry','exit')): raise ValueError('precision/history anchor join differs')
        for key in ('entry','exit'): cut('history',target[key])
        add('history',row['png'],'baselines/'+name+'.png')
        for record in native[name]['stages'].values(): add('prefixes',record['path'],'prefixes/'+record['path'])
        anchors[name]=dict(entry=target['entry'],exit=target['exit'],stages=native[name]['stages'],png_sha256=row['png_sha256'])
    prior=read(response/'collection-04/request.json'); h['bind_environment'](prior['execution_environment']); worker=here/'flux_local_delta_campaign.py'; api=runpy.run_path(str(worker))
    c=dict(schema='mdb-local-delta-future-plan-v1',**{k:old[k] for k in ('model','model_path','checkpoint_sha256','parents')},anchors=anchors,cases=api['cases'](),input_files=files,
        input_roots={key:read(root/'summary.json')['output_root']+'/binding-minimal-result' for key,root in roots.items()},output_scratch='/mnt/big/mdb/research',worker_wall_s=750,directory_custody=True,
        algorithm_frozen='native-primitive-anchors-plus-wide-finite-response-v1',diagnostic_audit_sha256=sha(diagnostic/'audit-v1.json'),
        hypotheses=dict(write='Per-primitive native calibration improves fresh own-state finite-response fidelity vs full absolute math; all changed computation NumPy, no native target primitives.',
            image='Candidate complete-write futures predict signed object-region RGB changes and background collateral of unseen native targets. Freeze candidate readouts before targets; report all errors and copy-anchor foil.',
            color_ordering='For each fixed receiver, row4heavy is bluer (lower mean red-minus-blue) than row11heavy on both supplied source-object supports. This direction is frozen before candidate or target futures; retain every counterexample.',
            visible_structure='Both fresh patterns retain two objects with sphere left of cube. Supplied source masks do not certify this; inspect native and candidate images and preserve movement/shading/shape disagreements.',
            scope='Four new asymmetric text combinations on two native-half calibrated image receivers; not new prompt/seed/receiver-general law. Native prefix, historical port calibration and suffix/VAE are supplied.'))
    run.mkdir(parents=True,exist_ok=False); save(run/'plan.json',c); api['preflight_transaction'](run/'transaction-api-preflight')
    extras=[(here/n,n) for n in ('flux_binding_minimal.py','flux_family_custody.py','flux_reentry_reference.py','flux_response_reference.py','flux_local_delta_reference.py','flux_response_campaign.py','flux_response_validate.py')]
    extras.extend([(Path(parameter_artifacts.__file__),'retrieve_weights.py'),(run/'plan.json','plan.json')])
    pre=run/'preflight.v1.json'; save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],path_token_allowlist=['binding-minimal-result','source',*c['input_roots'].values(),c['model_path'],c['output_scratch']]))
    extras.append((pre,pre.name)); raw=(json.dumps(c,sort_keys=True)+'\n').encode()
    request=dict(schema='mdb-local-delta-future-request-v1',config=c,model=c['model'],task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,needs=dict(cuda=True,host='beast'),execution_environment=prior['execution_environment'],
        config_bytes_sha256=hashlib.sha256(raw).hexdigest(),source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=14,logical_branches=14,logical_checkpoint_replays=14,max_local_work_items=14,
            checkpoint_policy='First seal8complete writes;2half controls+8candidate futures before4new targetblocks; second seal freezes candidate RGB target/collateral predictions.14VAE images,746native transitions,24Acts,56grouped receipts,1load,0enc/noise.'),
        resource_exception=dict(ram_mb=13000,vram_mb=10500,disk_gb=3,cpu_threads=4,est_wall_s=360,calibration_jobs=['job-326e9b1270ef','job-96b46e23b8c7','job-83c0610d9c67'],
            measured_native_peak_rss_mb=8088.1,measured_native_peak_vram_mb=9274,
            reason='Same resident noencoder B1 512px image-future lane previously9246-9274MBVRAM. Sequential receiver math retains two reference caches plus native port calibration under3GBextra allowance; weights/caches deleted before image consumers.13GBRAM/10.5GBVRAM, NVMe output. Historical actuals retained; sizing allowance explicit.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest(); save(run/'request.json',request)
    stage_worker(run/'payload',worker=worker,config=c,request=request,config_bytes=raw,additional_files=extras); return dict(run=str(run),request_sha256=request['request_sha256'],counts=api['expected_counts']())


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py'))); r=json.loads((run/'request.json').read_text()); h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()): raise ValueError('already submitted or stage changed')
    resources={k:v for k,v in r['resource_exception'].items() if k in {'ram_mb','vram_mb','disk_gb','cpu_threads','est_wall_s'}}
    job=guarded_mrun_launch(worker_command('flux_local_delta_campaign.py'),request=r,model=r['model'],task='forward',experiment='mdb-flux-local-delta-futures',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],timeout_s=900,retry_on_kill=False,preflight=True,priority=70,**resources,
        note='Fresh independent affected-write test:2asymmetric row4/11 patterns x2receivers=4cases total.8NumPy complete writes sealed;2knownhalf controls+8candidate RGB futures then image-effect seal BEFORE4native targetJ1+futures.14images/746transitions/24Acts/56receipts,1load,0enc/noise. Historical native ports explicitlycalibrate; no targetprimitive in predictor.750worker/900outer,no retries.')
    h['save'](run/'submission.json',dict(job_id=str(job))); return dict(job_id=str(job),run=str(run))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path); p.add_argument('precision',type=Path,nargs='?'); p.add_argument('response',type=Path,nargs='?'); p.add_argument('diagnostic',type=Path,nargs='?'); p.add_argument('--submit',action='store_true'); a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.precision.resolve(),a.response.resolve(),a.diagnostic.resolve()),sort_keys=True))
