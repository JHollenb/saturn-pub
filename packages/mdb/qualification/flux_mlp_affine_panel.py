"""Saved-only anchored × absolute affine panel on a qualified MLP case."""
import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path


METHODS=('anchored','absolute')


def affine(value,x0,y0,weight,method,reference,law):
    if method=='anchored':return law['affine_delta'](value,x0,y0,weight,reference['bf16'])[0]
    if method=='absolute':return reference['_linear'](value,{'w.weight':weight},'w',256)
    raise ValueError('unknown predeclared affine law')


def worker():
    payload=Path.cwd();h=runpy.run_path(str(payload/'flux_binding_minimal.py'));sha,save=h['sha'],h['save'];read=lambda p:json.loads(p.read_text());request=read(Path(os.environ['SATURN_DEBUG_REQUEST']));c=request['config']
    if read(Path(os.environ['SATURN_RUN_CONFIG']))!=c or any(sha(payload/n)!=digest for n,digest in request['source_hashes'].items()):raise ValueError('saved affine panel changed')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    import numpy as np
    from mdb.tensor_archive import load_tensor_archive
    from mdb.response_closure import audit_response,measure_error
    collection=Path(c['remote_collection']);record=read(collection/'remote-custody.json');data=collection/'extracted/binding-minimal-result';cut=read(collection/'request.json')['config'];source=Path(cut['source_collection']);source_record=read(source/'remote-custody.json');source_data=source/'extracted/binding-minimal-result';audit=read(Path(c['primitive_audit']))
    if sha(collection/'remote-custody.json')!=c['remote_record_sha256'] or sha(Path(c['primitive_audit']))!=c['primitive_audit_sha256'] or audit['custody']!=record['verification'] or audit['instrument_status']!='qualified' or sha(source/'remote-custody.json')!=cut['source_record_sha256']:raise ValueError('qualified primitive/source custody differs')
    def bound(root,relative,inventory):
        p=root/relative
        if sha(p)!=inventory[relative]['sha256']:raise ValueError('source metadata changed')
        return read(p)
    rows=bound(data,'rows.json',record['files']);plan=bound(source_data,'plan.json',source_record['files']);manifest=bound(data,'parameter-artifacts.json',record['files'])
    reference=runpy.run_path(str(payload/'flux_reentry_reference.py'));law=runpy.run_path(str(payload/'flux_receiver_norm_prediction.py'));power=runpy.run_path(str(payload/'analyze_flux_mlp_primitives.py'))['signed_error']
    cut_request=read(collection/'request.json')
    for name in ('flux_reentry_reference.py','flux_receiver_norm_prediction.py','flux_arithmetic_reference.py'):
        if request['source_hashes'][name]!=cut_request['source_hashes'][name]:raise ValueError('frozen reference changed')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID'];out=scratch/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();save(out/'plan.json',c);save(out/'request.json',request);save(out/'parameter-artifacts.json',manifest)
    import shutil
    shutil.copyfile(payload/'retrieve_weights.py',out/'retrieve_weights.py')
    bank=runpy.run_path(str(payload/'flux_source_receiver.py'))['RecipeBank'](manifest,cut['model_path']);entries={port:item for item in manifest['artifacts'] for port in item['parameter_ports']};archive={};write=runpy.run_path(str(payload/'flux_numpy_archive.py'))['record_bf16'];reports={};records={};checks={};counts=dict(model_loads=0,native_calls=0,images=0,affine_calculations=0,swiglu_calculations=0,conditional_mlp_outputs=0)
    def arr(root,item,base,inventory):
        name=(root/item['path']).relative_to(base).as_posix()
        if item['sha256']!=inventory[name]['sha256']:raise ValueError('array outside bound evidence')
        return load_tensor_archive(root,item)
    def native_arr(item):return arr(data,item,data,record['files'])
    for role,prefix in [('image','ff'),('text','ff_context')]:
        row=rows[role];anchor={field:arr(source_data/'source/prefixes',plan['anchors'][cut['anchor']]['stages'][role+'.'+field],source_data,source_record['files']) for field in ('mlp_input','mlp_paired','mlp_silu','mlp_activation','mlp')};target={name:native_arr(item) for name,item in row['native'].items()};target['input']=arr(source_data,row['source_input'],source_data,source_record['files']);weights={}
        for suffix in ('linear_in.weight','linear_out.weight'):
            port=prefix+'.'+suffix;item=entries[port];weights[suffix]=bank.load(dict(path=item['path'],sha256=item['sha256'],descriptor=dict(sha256=item['source']['payload_sha256']),parameter_tensor=item['source']['tensor'],stored_shape=item['source']['shape']))
        endpoints={};local={};record_role=dict(programs={},local_affines={})
        for endpoint in ('P0','P1'):
            is0=endpoint=='P0';x=anchor['mlp_input'] if is0 else target['input'];actual_a=anchor['mlp_activation'] if is0 else target['activation'];endpoints[endpoint]={};local[endpoint]={};record_role['programs'][endpoint]={};record_role['local_affines'][endpoint]={}
            for first in METHODS:
                z=affine(x,anchor['mlp_input'],anchor['mlp_paired'],weights['linear_in.weight'],first,reference,law);counts['affine_calculations']+=1
                local[endpoint]['first.'+first]=z;record_role['local_affines'][endpoint]['first.'+first]=write(out,z,archive);u,v=np.split(z,2,axis=-1);activation=reference['bf16'](reference['_silu'](u)*v);counts['swiglu_calculations']+=1
                for second in METHODS:
                    name=first+'.'+second;y=affine(activation,anchor['mlp_activation'],anchor['mlp'],weights['linear_out.weight'],second,reference,law);counts['affine_calculations']+=1;counts['conditional_mlp_outputs']+=1;endpoints[endpoint][name]=y;record_role['programs'][endpoint][name]=write(out,y,archive)
            for method in METHODS:
                value=affine(actual_a,anchor['mlp_activation'],anchor['mlp'],weights['linear_out.weight'],method,reference,law);counts['affine_calculations']+=1;local[endpoint]['output.'+method]=value;record_role['local_affines'][endpoint]['output.'+method]=write(out,value,archive)
        checks[role]=dict(anchored_target=bool(np.array_equal(endpoints['P1']['anchored.anchored'],native_arr(row['math']['substitutions']['C']))),anchored_anchor=bool(np.array_equal(endpoints['P0']['anchored.anchored'],anchor['mlp'])),local_first=bool(np.array_equal(local['P1']['first.anchored'],native_arr(row['math']['local']['paired']))),local_output=bool(np.array_equal(local['P1']['output.anchored'],native_arr(row['math']['local']['mlp']))))
        local_reports={}
        for site,field in [('first','paired'),('output','mlp')]:
            N0=anchor['mlp_paired' if site=='first' else 'mlp'];N1=target[field];local_reports[site]={method:audit_response(N0,N1,local['P0'][site+'.'+method],local['P1'][site+'.'+method]) for method in METHODS}
        programs={name:audit_response(anchor['mlp'],target['mlp'],endpoints['P0'][name],endpoints['P1'][name]) for name in endpoints['P0']}
        def contrast(values):
            aa,da,ad,dd=[values[name].astype(float) for name in ('anchored.anchored','absolute.anchored','anchored.absolute','absolute.absolute')]
            return power(dict(first_site=da-aa,second_site=ad-aa,interaction=dd-da-ad+aa),dd-aa)
        effects={name:endpoints['P1'][name].astype(float)-endpoints['P0'][name].astype(float) for name in endpoints['P0']}
        reports[role]=dict(local_affines=local_reports,programs=programs,law_change=dict(endpoint=contrast(endpoints['P1']),paired_response=contrast(effects)));records[role]=record_role
        print('MDB_MLP_AFFINE_PANEL '+json.dumps(dict(role=role,checks=checks[role],program_response_relative={name:report['reference_response']['relative_rms'] for name,report in programs.items()},local_response_relative={site:{method:report['reference_response']['relative_rms'] for method,report in methods.items()} for site,methods in local_reports.items()}),sort_keys=True),flush=True)
        del endpoints,local,target,anchor,weights,x,actual_a,z,activation,y,value,effects
        if time.monotonic()-started>c['max_wall_seconds']:raise RuntimeError('finite saved-only affine panel exceeded')
    if counts!=dict(model_loads=0,native_calls=0,images=0,affine_calculations=32,swiglu_calculations=8,conditional_mlp_outputs=16):raise ValueError('saved-only calculation census differs')
    issues=[role+'.'+name for role,values in checks.items() for name,value in values.items() if value is not True]
    if any(name in sys.modules for name in ('torch','diffusers','transformers')):raise ValueError('saved-only panel imported model framework')
    result=dict(schema='mdb-mlp-affine-panel-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),reports=reports,records=records,checks=checks,instrument_issues=issues,mechanics_status='verified-bound-source',instrument_status='qualified-control' if not issues else 'instrument-broken',terminal_status='unassessed',counts=counts,source_custody=record['verification'],source_record_sha256=c['remote_record_sha256'],primitive_audit_sha256=c['primitive_audit_sha256'],seconds=time.monotonic()-started,unique_arrays=len(archive),parameter_reads=bank.reads,scope=c['scope'])
    save(out/'panel-report.json',result);save(out/'summary.json',{key:value for key,value in result.items() if key not in ('reports','records')});save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=result['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,pointer,primitive_projection,native_run):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    from mdb import parameter_artifacts
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));sha,save=h['sha'],h['save'];read=lambda p:json.loads(p.read_text());record=runpy.run_path(str(here/'collect_flux_remote_directory.py'))['checked_pointer'](pointer);prior=read(native_run/'request.json');audit=read(primitive_projection/'audit-v1.json');h['bind_environment'](prior['execution_environment'])
    if audit['custody']!=record['verification'] or audit['instrument_status']!='qualified':raise ValueError('qualified primitive audit required')
    run.mkdir(parents=True,exist_ok=False);c=dict(remote_collection=record['collection'],remote_record_sha256=read(pointer)['record_sha256'],primitive_audit=read(primitive_projection/'summary.json')['output_root']+'/binding-minimal-result/audit-v1.json',primitive_audit_sha256=sha(primitive_projection/'audit-v1.json'),output_scratch='/mnt/big/mdb/research',directory_custody=True,max_wall_seconds=240,methods=list(METHODS),hypothesis='Two specified affine laws, without new schedules or fitted coefficients, distinguish baseline numerical-residue transport from unavoidable arithmetic mismatch. Compare both baselines, local returns and the full2×2 conditional MLP composition; retain signed interactions.',scope='Known-input numerical development on row4seven_eighths/base step2J1 image/textMLPs. Actual native target/anchor inputs supplied; not fresh prediction, native restoration, image edit or selected default.0native/frameworkcalls/images. Full32P0/P1 local/complete outputs retained in CAS; original weights recipe-only.')
    save(run/'plan.json',c);names=('flux_binding_minimal.py','flux_family_custody.py','flux_source_receiver.py','flux_reentry_reference.py','flux_receiver_norm_prediction.py','flux_arithmetic_reference.py','flux_numpy_archive.py','analyze_flux_mlp_primitives.py');extras=[(here/name,name) for name in names]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py'),(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[name for _,name in extras],pre.name],path_token_allowlist=[c['remote_collection'],prior['config']['source_collection'],str(Path(prior['config']['source_audit']).parent),str(Path(c['primitive_audit']).parent),prior['config']['model_path'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name));r=dict(schema='mdb-mlp-affine-panel-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='float64-with-BF16-boundaries',execution_environment=prior['execution_environment'],source_hashes={name:sha(p) for p,name in [(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=16,logical_branches=4,logical_checkpoint_replays=0,max_local_work_items=16,extra=dict(task_family='analysis'),checkpoint_policy='Two fixed affine policies at2sites,2roles,2knownendpoints.16MLPoutputs/16localreturns;32affine/8SwiGLU calculations. Exactsource-controljoins,0native/framework/image. RetainP0/P1and signed interaction, no numerical gate.'),resource_exception=dict(ram_mb=6144,cpu_threads=4,disk_gb=2,est_wall_s=120,calibration_job='job-268ff4637e45',reason='Qualified NumPy primitive audit measured3068MBRSS on current fullgeometry.4policy P0/P1 outputs/localfields and float64 response audits retained sequentially byrole;6GB leaves measured headroom. CPUonly4threads,noTorch,under4min/600outer,noretries.'))
    r['request_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',r);stage_worker(run/'payload',worker=Path(__file__),config=c,request=r,additional_files=extras);return dict(run=str(run),request_sha256=r['request_sha256'])


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/name)!=digest for name,digest in r['source_hashes'].items()):raise ValueError('already submitted or changed stage')
    resource={k:r['resource_exception'][k] for k in ('ram_mb','cpu_threads','disk_gb','est_wall_s')};cmd=worker_command(Path(__file__).name);cmd.insert(cmd.index('python')+1,'-B')
    job=guarded_mrun_launch(cmd,request=r,model=None,task='analysis',experiment='mdb-flux-mlp-affine-panel',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=600,retry_on_kill=False,preflight=True,**resource,note='Saved-only MLP calibration-transport test. Fixed anchored/absolute at2affines,4combinations,2roles/2knownendpoints;16MLPoutputs+16localreturns retained. No new numerical schedules, coefficients, native calls or images. Exactanchoredcontroljoins, P0/P1responseand signedinteraction.6GB/4CPU,under4min/600outer,noretries.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
        for name in ('pointer','primitive_projection','native_run'):p.add_argument(name,type=Path,nargs='?')
        p.add_argument('--submit',action='store_true');a=p.parse_args();print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.pointer.resolve(),a.primitive_projection.resolve(),a.native_run.resolve()),sort_keys=True))
