"""Locate a response-ranking reversal at already saved denoising commits."""
import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def worker():
    payload=Path.cwd();request=read(os.environ['SATURN_DEBUG_REQUEST']);c=request['config']
    if read(os.environ['SATURN_RUN_CONFIG'])!=c or any(sha(payload/name)!=digest for name,digest in request['source_hashes'].items()):raise ValueError('saved diagnostic source differs')
    from mdb_lanes.response_closure import audit_response
    import numpy as np
    power=runpy.run_path(str(payload/'flux_own_state_mlp_math.py'))['power_account']
    collection=Path(c['collection']);custody=runpy.run_path(str(payload/'flux_family_custody.py'))['verify'](collection)
    if sha(collection/'remote-custody.json')!=c['remote_record_sha256'] or custody!=read(collection/'remote-custody.json')['verification']:raise ValueError('native scientific custody differs')
    audit=read(c['audit_path'])
    if sha(c['audit_path'])!=c['audit_sha256'] or audit['custody']!=custody or audit['mechanics_status']!='verified':raise ValueError('original independent audit binding differs')
    data=collection/'extracted/binding-minimal-result';by={(row['id'],row['mode']):row for row in read(data/'rows.json')}
    slot=runpy.run_path(str(payload/'analyze_flux_binding_minimal.py'))['read_slot']
    points={};change_accounts={};touched=set()
    def latent(name,mode,step):
        fp=by[name,mode]['commits'][step-3];touched.add(fp)
        cut=read(data/'store/cuts'/(fp+'.json'))
        if cut['execution_point']['logical_step']!=step or cut['execution_point']['phase']!='project':raise ValueError('saved latent clock differs')
        return slot(data/'store',fp,'latent')
    for step in (3,4):
        points[str(step)]={};change_accounts[str(step)]={}
        for name in ('row4heavy','row11heavy'):
            N0,N1=latent('half','native',step),latent(name,'native',step)
            points[str(step)][name]={mode:audit_response(N0,N1,latent('half',mode,step),latent(name,mode,step)) for mode in ('old','new')}
            nd=N1.astype(np.float64)-N0.astype(np.float64)
            errors={mode:latent(name,mode,step).astype(np.float64)-latent('half',mode,step).astype(np.float64)-nd for mode in ('old','new')}
            change_accounts[str(step)][name]=power(dict(old_response_error=errors['old'],program_change=errors['new']-errors['old']),errors['new'])
    if len(touched)!=18:raise ValueError('complete saved latent census differs')
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    h=runpy.run_path(str(payload/'flux_binding_minimal.py'));save=h['save']
    result=dict(schema='mdb-fresh-mlp-saved-continuation-v1',mechanics_status='verified',instrument_status='qualified saved-state measurement',terminal_status='unassessed',custody=custody,original_audit_sha256=c['audit_sha256'],latent_states=18,points=points,old_to_new_change=change_accounts,native_calls=0,model_loads=0,
        scope='Two saved actual denoising commits of nine original futures. Paired responses use each candidate ownP0. This localizes ranking changes at observed boundaries; it cannot assign a finer attention/MLP/VAE operation without further states.')
    if any(name in sys.modules for name in ('torch','diffusers','transformers')):raise ValueError('saved diagnostic imported native framework')
    save(out/'request.json',request);save(out/'diagnostic.json',result)
    summary=dict(schema='mdb-fresh-mlp-saved-continuation-summary-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(out.parent),mechanics_status='verified',native_calls=0,model_loads=0,diagnostic_sha256=sha(out/'diagnostic.json'))
    save(out/'summary.json',summary);save(out/'files.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,study):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));save=h['save'];old=read(study/'run-02/request.json');h['bind_environment'](old['execution_environment'])
    pointer=read(study/'native-remote-pointer-v1.json');independent=read(study/'audit-job-v1.json');audit=study/'audit/collection-01/extracted/binding-minimal-result/audit-v1.json'
    c=dict(collection=pointer['record']['collection'],remote_record_sha256=pointer['record_sha256'],audit_path='/mnt/big/mdb/research/'+independent['job_id']+'/binding-minimal-result/audit-v1.json',audit_sha256=sha(audit),output_scratch='/mnt/big/mdb/research',directory_custody=True)
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',c)
    names=('flux_binding_minimal.py','flux_family_custody.py','analyze_flux_binding_minimal.py','flux_own_state_mlp_math.py');extras=[(here/name,name) for name in names]+[(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[c['collection'],str(Path(c['audit_path']).parent),c['output_scratch'],'binding-minimal-result']));extras.append((pre,pre.name))
    request=dict(schema='mdb-fresh-mlp-saved-continuation-request-v1',config=c,execution_environment=old['execution_environment'],needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=8,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=18,extra=dict(task_family='analysis'),checkpoint_policy='18existing latents at commits3/4; eight paired-response audits. Full original custody/independent-audit joins,0native/framework/MLP/attention/images.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    base=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=read(run/'request.json');base['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=digest for n,digest in r['source_hashes'].items()):raise ValueError('changed or submitted diagnostic')
    cmd=worker_command(Path(__file__).name);cmd.insert(1,'CUDA_VISIBLE_DEVICES=');cmd.insert(cmd.index('python')+1,'-B')
    job=guarded_mrun_launch(cmd,request=r,model=None,task='analysis',experiment='mdb-flux-fresh-mlp-saved-continuation',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=300,retry_on_kill=False,preflight=True,
        note='Saved-only followup to actual mixedRGB result:18existing commit3/4latents,8pairedresponseaudits. No denseMLP/attention/model/image/native calls. Original fullcustody+independentreport joins. Intent-first RAM/CPU,expectedunder1min/300outer,noretries.')
    base['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('study',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args();print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.study.resolve()),sort_keys=True))
