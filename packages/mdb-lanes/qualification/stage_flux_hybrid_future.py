"""Freeze two complete programs before four new target states and their images."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,precision,response,study):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes import parameter_artifacts
    here=Path(__file__).resolve().parent; h=runpy.run_path(str(here/'flux_binding_minimal.py')); sha,save=h['sha'],h['save']; read=lambda p:json.loads(p.read_text())
    if any(read(root/'verification-v1.json')['mechanics_status']!='verified' for root in (precision,response,study)): raise ValueError('verified development evidence required')
    selected=study/'receiver'; audit=read(selected/'audit-v1.json'); development=read(selected/'run-01/request.json')
    if audit['mechanics_status']!='verified': raise ValueError('receiving-LN selection not audited')
    worker=here/'flux_hybrid_future.py'; api=runpy.run_path(str(worker)); prior_payload=selected/'run-01/payload'
    for name in ('flux_receiver_norm_prediction.py','flux_arithmetic_reference.py','flux_reentry_reference.py'):
        if sha(here/name)!=development['source_hashes'][name] or sha(prior_payload/name)!=sha(here/name): raise ValueError('executed numerical law changed')
    if any(s['row_doses']==old['row_doses'] for s in api['cases']() for old in development['config']['cases']): raise ValueError('new dose patterns overlap development')
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
    anchors={}; add('prefixes','native.json','prefixes/native.json')
    for name in ('half','half.image1'):
        row=rows[name,'native']; target=row['target']
        if any(target[k]!=native[name][k] for k in ('entry','exit')): raise ValueError('precision/history anchor join differs')
        for key in ('entry','exit'): cut('history',target[key])
        add('history',row['png'],'baselines/'+name+'.png')
        for record in native[name]['stages'].values(): add('prefixes',record['path'],'prefixes/'+record['path'])
        anchors[name]=dict(entry=target['entry'],exit=target['exit'],stages=native[name]['stages'],png_sha256=row['png_sha256'])
    prior=read(response/'collection-04/request.json'); h['bind_environment'](prior['execution_environment'])
    c=dict(schema='mdb-hybrid-future-plan-v1',**{k:old[k] for k in ('model','model_path','checkpoint_sha256','parents')},anchors=anchors,cases=api['cases'](),programs=api['programs'](),input_files=files,
        input_roots={key:read(root/'summary.json')['output_root']+'/binding-minimal-result' for key,root in roots.items()},output_scratch='/mnt/big/mdb/research',worker_wall_s=1050,directory_custody=True,
        absolute_ports=development['config']['absolute_ports'],policy=development['config']['policy'],selection_audit_sha256=sha(selected/'audit-v1.json'),selection_request_sha256=development['request_sha256'],
        algorithm_frozen='corrected-text-LN-QK-RMS+carry1024RNE-five-attention-affines+anchored-attention+receiving-LN-factorial',
        hypotheses=dict(primary='Hybrid reduces the mean image-exit mathematical finite-response relative RMS across four fresh cases compared with control. Signed per-case effects and exceptions retained; no all-rows discovery gate.',
            numerical='Retain own P0 and P1 raw attention, prefix, projected attention, receiving state, MLP input/write and exit. Compare P1-P0 with N1-N0 separately from installed P1-N0. Text and receiver-by-pattern mixed effects are secondary observations.',
            image='Freeze candidate RGB and supplied-source-support readouts before native targets. Report intended change, background collateral, errors and copy-anchor foil; hybrid state improvement need not improve RGB.',
            color_ordering='For each fixed receiver, row4seven_eighths is bluer (lower mean red-minus-blue) than row11seven_eighths on both supplied source-object supports. This direction is frozen before candidate or target futures; retain every counterexample.',
            visible_structure='Both fresh patterns retain two objects with sphere left of cube. Source masks do not certify this; inspect all native/candidate images for movement, shading and shape disagreements.',
            scope='Four fresh dose states on two known calibrated image receivers; not new prompts, seeds or independent receivers. Historical native prefix, weight bytes, numerical calibration and suffix/VAE are supplied. Eight candidates are two programs on four states, not eight independent trials.'))
    run.mkdir(parents=True,exist_ok=False); save(run/'plan.json',c)
    common=runpy.run_path(str(here/'flux_local_delta_campaign.py')); common['preflight_transaction'](run/'transaction-api-preflight')
    save(run/'selection.json',dict(audit=audit,study_verification=read(study/'verification-v1.json'),selected_request=development['request_sha256']))
    extras=[(here/n,n) for n in ('flux_binding_minimal.py','flux_family_custody.py','flux_reentry_reference.py','flux_response_reference.py','flux_local_delta_reference.py','flux_response_campaign.py','flux_response_validate.py','flux_local_delta_campaign.py','flux_receiver_norm_prediction.py','flux_arithmetic_reference.py','flux_precision_cut.py')]
    extras.extend([(Path(parameter_artifacts.__file__),'retrieve_weights.py'),(run/'plan.json','plan.json'),(run/'selection.json','selection.json')])
    pre=run/'preflight.v1.json'; save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],path_token_allowlist=['binding-minimal-result','source',*c['input_roots'].values(),c['model_path'],c['output_scratch']]))
    extras.append((pre,pre.name)); raw=(json.dumps(c,sort_keys=True)+'\n').encode()
    request=dict(schema='mdb-hybrid-future-request-v1',config=c,model=c['model'],task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,needs=dict(cuda=True,host='beast'),execution_environment=prior['execution_environment'],
        config_bytes_sha256=hashlib.sha256(raw).hexdigest(),source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=14,logical_branches=14,logical_checkpoint_replays=14,max_local_work_items=14,
            checkpoint_policy='Seal8P1+4P0 with complete writer/raw attention evidence.2exact-half replays+8candidate futures before4new targetblocks; second seal fixes RGB/readouts.14images/746transitions/24Acts/56receipts,1load,0enc/noise. No framework module forwards during prediction.'),
        resource_exception=dict(ram_mb=13000,vram_mb=10500,disk_gb=10,cpu_threads=4,est_wall_s=480,calibration_jobs=['job-83c0610d9c67','job-96b46e23b8c7'],measured_native_peak_rss_mb=8088.1,measured_native_peak_vram_mb=9274,
            reason='Same14image resident CUDA BF16 noencoder lane previously347s/9274MBVRAM/8088MBRSS. Frozen hybrid/control NumPy caches sequential; prediction arrays streamed BF16 CAS.13GBRAM includes temporary cache allowance.10GBdisk covers copied native calibration, unique new states and full P0/P1 ports. Original checkpoint constants recorded as recipes, not per-run weight bank. Expected5-8minutes;1050worker/1200outer allows cold imports, no retries.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest(); save(run/'request.json',request)
    stage_worker(run/'payload',worker=worker,config=c,request=request,config_bytes=raw,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'],counts=api['expected_counts'](),source_bytes=sum(x['bytes'] for x in files.values()))


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py'))); r=json.loads((run/'request.json').read_text()); h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()): raise ValueError('already submitted or stage changed')
    resources={k:v for k,v in r['resource_exception'].items() if k in {'ram_mb','vram_mb','disk_gb','cpu_threads','est_wall_s'}}
    job=guarded_mrun_launch(worker_command('flux_hybrid_future.py'),request=r,model=r['model'],task='forward',experiment='mdb-flux-hybrid-futures',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],timeout_s=1200,retry_on_kill=False,preflight=True,priority=70,**resources,
        note='Fresh7/8vs1/8 x2knownreceivers.2fixedprograms(control/hybrid)differonlyreceivingLN.8P1+4P0 sealed,pairedrawA retained;2replays+8candidateimages thenRGB/readoutseal BEFORE4targetJ1+images.14images/746transitions/24Acts/56receipts,1load,0enc/noise.Expected5-8min,1050worker/1200outer,noretries.Fullcustody/auditstayBeast;smallreportprojectionlocal.')
    h['save'](run/'submission.json',dict(job_id=str(job))); return dict(job_id=str(job),run=str(run))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path)
    for n in ('precision','response','study'): p.add_argument(n,type=Path,nargs='?')
    p.add_argument('--submit',action='store_true'); a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.precision.resolve(),a.response.resolve(),a.study.resolve()),sort_keys=True))
