"""Freeze the supported affine map before four new internal target states."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,precision,transfer,transfer_audit):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    from mdb import parameter_artifacts
    here=Path(__file__).resolve().parent; h=runpy.run_path(str(here/'flux_binding_minimal.py')); sha,save=h['sha'],h['save']; read=lambda p:json.loads(p.read_text())
    audit=read(transfer_audit)
    if audit['mechanics_status']!='verified': raise ValueError('source transfer audit required')
    receipt=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](transfer)
    if any(receipt[k]!=audit['custody']['transfer'][k] for k in ('job_id','archive_sha256','request_sha256')): raise ValueError('transfer audit belongs to another collection')
    data=precision/'extracted/binding-minimal-result'; old=read(data/'plan.json'); native=read(data/'native.json'); inventory=read(data/'files.json'); prior=read(precision/'request.json')
    h['bind_environment'](prior['execution_environment']); names={'native.json','parameter-artifacts.json','summary.json','plan.json'}
    for item in native.values():
        names.update(record['path'] for record in item['stages'].values())
    for name in inventory:
        if name.startswith('store/'): names.add(name)
    inputs={n:inventory[n] for n in sorted(names)}
    if any(sha(data/n)!=v for n,v in inputs.items()): raise ValueError('source inventory changed')
    policy=read(transfer/'extracted/binding-minimal-result/summary.json')['policy']; worker=here/'flux_arithmetic_fresh.py'; api=runpy.run_path(str(worker))
    if policy!=audit['policy']: raise ValueError('policy differs from audited transfer')
    ports=['attn.add_q_proj','attn.add_k_proj','attn.add_v_proj','attn.to_add_out','attn.to_out.0']
    c=dict(schema='mdb-fresh-affine-arithmetic-plan-v1',**{k:old[k] for k in ('model','model_path','checkpoint_sha256','parents')},policy=policy,absolute_ports=ports,cases=api['cases'](),inputs=inputs,
        input_root=read(data/'summary.json')['output_root']+'/binding-minimal-result',output_scratch='/mnt/big/mdb/research',worker_wall_s=600,directory_custody=True,source_transfer_audit_sha256=sha(transfer_audit),
        map_reason='Frozen1024carryRNE transfers almost exactly across attention affines, but degrades both MLP affine shapes. Apply only five actively computed attention affines; unchanged imageQKV cache, normalization/attention anchors and receiving MLP law remain prior.',
        hypotheses=dict(numerical='Absolute1024carryRNE attention affines improve fresh QKV/projected writes and complete-state effects vs anchoredwide local method; retain baseline/mixed counterexamples.',
            split='Own frozen QKV native-attention controls separate inherited QKV error from same-operand attention arithmetic. Native target ports only diagnose after seal; they never become forecast inputs.',
            scope='Four new asymmetric internal inputs on two fixed historical receivers. No new prompt/seed/image/semantic claim; exact baseline equality is not required for absolute arithmetic.'))
    run.mkdir(parents=True,exist_ok=False); save(run/'plan.json',c)
    extras=[(here/n,n) for n in ('flux_binding_minimal.py','flux_family_custody.py','flux_reentry_reference.py','flux_response_reference.py','flux_local_delta_reference.py','flux_arithmetic_reference.py','flux_arithmetic_prediction.py','flux_response_campaign.py','flux_precision_cut.py')]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],path_token_allowlist=[c['input_root'],c['model_path'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name));request=dict(schema='mdb-fresh-affine-arithmetic-request-v1',config=c,model=c['model'],task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,needs=dict(cuda=True,host='beast'),execution_environment=prior['execution_environment'],
        source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=8,logical_branches=4,logical_checkpoint_replays=4,max_local_work_items=16,
            checkpoint_policy='EightNumPy completewrites and ownQKV before four originalblocks;4nativeSDPAcontrols on ownsealedQKV andposthoc mathematical conditionalQKV.4transitions8Acts12receipts,0images/encodes/noise; allrawstatesBF16 archives, no persistentweightbanks.'),
        resource_exception=dict(ram_mb=10000,vram_mb=10500,cpu_threads=4,disk_gb=2,est_wall_s=300,calibration_job='job-2d7a8ce324b9',measured_peak_rss_mb=4960.8,measured_peak_vram_mb=9292,
            reason='Same original noencoderB1 512px checkpoint/block; preceding14image campaign measured4960.8MBRSS/9292MBVRAM. This run keepsVAE CPU and decodes0images, adds ownSDPAcontrols sequentially and CPUwide1024chunks.10GBRAM/10.5GBVRAM retain explicit headroom; strict admission, no retries.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest(); save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'],counts=api['counts_expected']())


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()): raise ValueError('already submitted/source changed')
    resources={k:v for k,v in r['resource_exception'].items() if k in {'ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s'}}
    job=guarded_mrun_launch(worker_command('flux_arithmetic_fresh.py'),request=r,model=r['model'],task='forward',experiment='mdb-fresh-affine-arithmetic',payload=run/'payload',gpu=True,needs=r['needs'],prefer_host='beast',timeout_s=750,retry_on_kill=False,preflight=True,priority=70,**resources,
        note='Fresh numerical-law extrapolation:4new5/8x3/8cases,8complete writes+ownQKV sealed BEFORE4originalblocks.4nativeSDPAonownoperands explicitly diagnostic; conditionalmath on nativeoperands onlypostseal.0images/encodes/noise,4transitions8Acts12receipts,1load;600worker750outer,no retries. BF16 archives and source references reduce duplicate custody bytes.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    for name in ('precision','transfer','transfer_audit'):p.add_argument(name,type=Path,nargs='?')
    p.add_argument('--submit',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.precision.resolve(),a.transfer.resolve(),a.transfer_audit.resolve()),sort_keys=True))
