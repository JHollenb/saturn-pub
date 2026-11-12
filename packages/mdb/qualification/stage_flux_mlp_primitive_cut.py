"""Bind the two-MLP diagnostic to the verified fresh-state cohort."""
import argparse
import hashlib
import importlib.metadata
import json
import runpy
from pathlib import Path


def stage(run,prior):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    from mdb import parameter_artifacts
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));sha,save=h['sha'],h['save'];read=lambda p:json.loads(p.read_text())
    checked=read(prior/'verification-v2.json')
    if checked['mechanics_status']!='verified':raise ValueError('verified preceding study required')
    pointer=prior/'native-remote-pointer-v1.json';remote=runpy.run_path(str(here/'collect_flux_remote_directory.py'))['checked_pointer'](pointer);old=read(prior/'run-02/request.json');mlp_request=read(prior/'mlp/run-02/request.json');h['bind_environment'](old['execution_environment'])
    if remote['verification']!=checked['remote_scientific_custody']:raise ValueError('source cohort differs')
    for name in ('flux_reentry_reference.py','flux_receiver_norm_prediction.py','flux_arithmetic_reference.py'):
        if sha(here/name)!=old['source_hashes'][name]:raise ValueError('frozen surrogate changed')
    if sha(here/'flux_hybrid_mlp_cut.py')!=mlp_request['source_hashes']['flux_hybrid_mlp_cut.py']:raise ValueError('MLP subprogram changed')
    projected=prior/'audit/collection-01/extracted/binding-minimal-result';mlp=prior/'mlp/collection-01/extracted/binding-minimal-result'
    native=read(projected/'native-proof/native.json');params=read(projected/'native-proof/parameter-artifacts.json');worker=here/'flux_mlp_primitive_cut.py';core=runpy.run_path(str(worker))
    c=dict(case=core['CASE'],anchor=core['ANCHOR'],source_collection=remote['collection'],source_record_sha256=read(pointer)['record_sha256'],source_audit=read(projected/'summary.json')['output_root']+'/binding-minimal-result/audit-v1.json',source_audit_sha256=sha(projected/'audit-v1.json'),
        previous_mlp_root=read(mlp/'summary.json')['output_root']+'/binding-minimal-result',previous_mlp_conditional_sha256=sha(mlp/'conditional.json'),previous_mlp_audit_sha256=sha(mlp/'audit-v1.json'),model_path=params['defaults']['checkpoint_root'],model_identity=read(projected/'native-proof/receipts.json')[0]['body']['model_identity'],native_entry=native[core['CASE']]['entry'],native_exit=native[core['CASE']]['exit'],
        original_module_source_sha256=sha(Path(importlib.metadata.distribution('diffusers').locate_file('diffusers/models/transformers/transformer_flux2.py'))),output_scratch='/mnt/big/mdb/research',directory_custody=True,max_wall_seconds=240,
        hypotheses=dict(primitive='Local same-input first-affine, SiLU, product and output-affine errors locate combined surrogate MLP discrepancy. Existing reader evidence favors checking the output affine but is not step2/J1 evidence.',
            decomposition='Conditional C−N equals ordered first-affine propagation plus nonlinear propagation plus local output-affine discrepancy; preserve all powers and signed pairwise cross terms. Distinct primitive tensor-space errors are never directly added.',
            scope='Post-hoc known-input diagnostic on one preselected row4seven_eighths/base state, image and text MLPs. Actual target intermediate ports supply conditional comparisons only, never future predictions or installed edits.0fullmodels/blocks/attention/images. No new prompt/seed or accuracy-floor/semantic-absence claim.'),
        scope='Two source-extracted original MLP forwards on archived native full-row inputs;4affines/2SiLU/2products. Match archived output bytes to qualify instrument, then score frozen local and propagated laws. Historical half anchors and actual new intermediates are explicit diagnostic operands.')
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',c)
    names=('flux_binding_minimal.py','flux_family_custody.py','flux_source_receiver.py','flux_reentry_reference.py','flux_receiver_norm_prediction.py','flux_arithmetic_reference.py','flux_hybrid_future.py','flux_hybrid_mlp_cut.py')
    extras=[(here/n,n) for n in names]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py'),(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],path_token_allowlist=[c['source_collection'],str(Path(c['source_audit']).parent),c['previous_mlp_root'],c['model_path'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name));request=dict(schema='mdb-mlp-primitive-cut-request-v1',config=c,needs=dict(cuda=True,host='beast'),task_family='analysis',backend='torch-primitive',device='cuda',dtype='bfloat16',execution_environment=old['execution_environment'],source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=2,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=2,extra=dict(task_family='analysis'),checkpoint_policy='Two original-source MLP components only, one native input per role.4linears/2SiLU/2products,fullimage1024/text512geometry. Allmathguarded againstnativeforwards. Nativeoutputparity and scoped conditionaldiagnosis;0fullmodels/blocks/attention/images.'),
        resource_exception=dict(ram_mb=4096,vram_mb=2048,cpu_threads=4,disk_gb=2,est_wall_s=120,calibration_job='job-a80061c1674b',measured_primitive_peak_rss_mb=1684.4,measured_primitive_peak_vram_mb=612,
            reason='Measured CUDA primitive lane1684MBRSS/612MBVRAM. Current four BF16 MLP matrices total~340MB vs prior~19MB; decoded FP32~680MB, fullimage/textpaired/SwiGLU arrays and sequential NumPy256-row-output tiles.4GBRAM/2GBVRAM bounds added matrix/activation footprints; no fullmodel/VAE/GPUFP64, immutable original recipes only. Expectedunder3min,240worker/600outer,no retries.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=c,request=request,additional_files=extras);return dict(run=str(run),request_sha256=request['request_sha256'],case=c['case'])


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('already submitted or source changed')
    resource={k:r['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')};cmd=worker_command('flux_mlp_primitive_cut.py');cmd.insert(cmd.index('python')+1,'-B')
    job=guarded_mrun_launch(cmd,request=r,model=None,task='analysis',experiment='mdb-flux-mlp-primitive-cut',payload=run/'payload',gpu=True,needs=r['needs'],prefer_host='beast',timeout_s=600,retry_on_kill=False,preflight=True,**resource,
        note='Preselectedrow4seven_eighths/base step2J1 image/textnativeMLPinputs.2source-extractedoriginalFFcalls=4affines/2SiLU/2products. Archivedyparityqualifiesinstrument; comparelocalprimitives andorderedoutput-spaceerror withsignedcross terms. Nativeintermediatesdiagnosticonly,0fullmodels/blocks/attention/images.4GBRAM/2GBVRAM,under3min,240worker/600outer,no retries.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('prior',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args();print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.prior.resolve()),sort_keys=True))
