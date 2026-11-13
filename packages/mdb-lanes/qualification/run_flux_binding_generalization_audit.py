"""Near-data NumPy audit, searchable records and compact custody; no models."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization.py'))


def worker():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    h = helpers()
    read, save, sha = [h[k] for k in ('read', 'save', 'sha')]
    req = read(os.environ['SATURN_DEBUG_REQUEST'])
    c = req['config']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(sha(Path(n)) != v for n, v in req['source_hashes'].items()):
        raise ValueError('saved-only audit binding/isolation differs')
    root = Path(c['native_root'])
    if sha(root / 'files.json') != c['native_manifest_sha256']:
        raise ValueError('native inventory changed')
    start = time.monotonic()
    out = Path(c['output_scratch']) / os.environ['MRUN_JOB_ID'] / 'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False)
    audit_h = runpy.run_path(str(HERE / 'flux_binding_generalization_audit.py'))
    result = audit_h['analyze'](root, out / 'audit.json')
    import numpy as np
    from mdb_lanes.equation_comparison import EquationCatalog, compare_equations
    catalog = EquationCatalog(out / 'catalog/equations')
    native_request, plan, native_summary = [read(root / n) for n in ('request.json', 'plan.json', 'summary.json')]
    parents, profile = read(root / 'parents.json'), read(root / 'native-contract.json')
    records, family_rows = [], []
    def load(ref):
        return (np.load(root / ref['path'], allow_pickle=False).astype(np.uint32) << 16).view(np.float32)
    for case in plan['contexts']:
        cid = case['id']
        pair = {r['field']: r for r in read(root / 'fields.json') if r['case'] == cid and r['arm'] == 'IC.early'}
        context = dict(model=plan['model'], case=cid, prompt=case['prompts']['base'], seed=case['seed'],
                       step=2, phase='project', dtype='bfloat16', device='cuda', batch=1,
                       source_request_sha256=native_request['request_sha256'], receiver_cut=parents[cid]['primed'],
                       primitive_profile_sha256=h['identity'](profile))
        record = catalog.record(law_id='law.flux.conditioning-and-latent-projection.v1',
            route_id='route.flux.attribute-binding.image-prime.conditioning-read.v1', alias='binding.' + cid,
            binding=dict(model_identity=plan['model'], input_identity=context, source_identity=native_request['request_sha256'],
                         execution=read(root / 'native-contract.json'), frontier=dict(step=2, phase='project'), horizon=1),
            report=compare_equations({k:load(v['prediction']) for k,v in pair.items()},
                                     {k:load(v['actual']) for k,v in pair.items()}, max_elements=20_000_000),
            conditions=[dict(field=k, op='eq', value=v) for k,v in context.items()], tested_context=context,
            readers=['joint0.image-attention', 'joint0.image-mlp', 'native.remaining-suffix', 'vae.rgb'],
            evidence=dict(audit_sha256=sha(out / 'audit.json'), numerical_scope='Original Linear primitive projections only.',
                          semantic_scope='Observed native continuation; source-assisted paired address/prime; no pixel forecast.', terminal_status='unassessed'))
        records.append(record['fingerprint'])
        if (len(catalog.find(applicable_to=context)) != 1
                or any(catalog.find(applicable_to={**context, key: wrong}) for key, wrong in
                       (('seed', -1), ('source_request_sha256', 'changed'), ('primitive_profile_sha256', 'changed'), ('receiver_cut', 'changed')))):
            raise ValueError('supplied applicability search differs')
        family_rows.append(dict(id='binding.program.' + cid, family='receiver-priming.timed-context-read.working-text-maintenance',
            model=plan['model'], architecture='FLUX.2 Klein4B distilled', component='project2/joint0 plus native suffix',
            prompt=case['prompts'], coordinates=dict(prime='step1/joint4.image', read='step2/project.conditioning', maintenance='step2/joint0.text'),
            computation='Image-only source priming, changed-token source C read, native attention/MLP write, working-stream restoration and continuation.',
            conditions=case, source_job=native_summary['job_id'], prediction=plan['hypotheses'],
            effects=result['rgb_diagnostics'][cid], controls=result['controls'][cid],
            status='exploratory observation; manual object-attribute review is separate',
            unresolved=['General receiver-only address induction', 'Semantic attribute certainty', 'Other clocks/model/checkpoint', 'Independently predicted attention and image future']))
    save(out / 'catalog/provenance.json', dict(fingerprints=records, exact_context_searches=len(records),
        wrong_seed_abstentions=len(records), wrong_source_abstentions=len(records),
        wrong_profile_abstentions=len(records), wrong_receiver_abstentions=len(records), automatic_rule_induction=False))
    with (out / 'computation-records.jsonl').open('x') as stream:
        for row in family_rows:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    for name in ('rows.json', 'fields.json', 'writes.json', 'actions.json', 'parents.json', 'token-addresses.json',
                 'native-contract.json', 'parameter-artifacts.json', 'retrieve_weights.py', 'retrieve_flux_encoder.py'):
        shutil.copyfile(root / name, out / name)
    for name in ('summary.json', 'request.json', 'plan.json', 'timeline.jsonl'):
        shutil.copyfile(root / name, out / ('source-' + name))
    for row in read(root / 'rows.json'):
        shutil.copyfile(root / row['png'], out / row['png'])
    shutil.copytree(root / 'transaction', out / 'source-transaction')
    save(out / 'external-evidence.json', dict(host='beast', roots=dict(native=str(root)),
        manifests=dict(native=read(root / 'files.json')), scope='Full BF16 activations, CAS and receipts stay near compute; checksum-bound individual retrieval.'))
    shutil.copyfile(HERE / 'retrieve_reader_evidence.py', out / 'retrieve_evidence.py')
    save(out / 'plan.json', c)
    save(out / 'request.json', req)
    save(out / 'summary.json', dict(schema='mdb-binding-generalization-compact-v1', job_id=os.environ['MRUN_JOB_ID'],
        request_sha256=req['request_sha256'], output_root=str(out), source_job=native_summary['job_id'],
        mechanics_status=result['mechanics_status'], seconds=time.monotonic()-start,
        new_execution_counts=dict(model_loads=0, native_primitives=0, images=0), terminal_status='unassessed'))
    save(out / 'files.json', {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY ' + json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'], request_sha256=req['request_sha256'],
        output_root=str(out), files_manifest_sha256=sha(out / 'files.json'), summary_sha256=sha(out / 'summary.json')), sort_keys=True), flush=True)


def stage(run, native):
    from mdb_lanes.job_support import scheduler_api, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    h = helpers()
    read, save, sha = [h[k] for k in ('read', 'save', 'sha')]
    old = read(native / 'request.json')
    jid = read(native / 'submission.json')['job_id']
    _, raw, _ = scheduler_api().request('GET', '/api/jobs/' + jid, timeout_s=20)
    job = json.loads(raw)
    if job['state'] != 'succeeded':
        raise ValueError('completed native campaign required')
    if not (native / 'job.json').exists():
        save(native / 'job.json', job)
    root = Path(old['config']['output_scratch']) / jid / 'binding-minimal-result'
    code = 'import json,hashlib\nfrom pathlib import Path\np=Path(' + repr(str(root)) + ')\nprint(json.dumps(dict(sha=hashlib.sha256((p/"files.json").read_bytes()).hexdigest(),bytes=sum(x.stat().st_size for x in p.rglob("*") if x.is_file()))))\n'
    meta = json.loads(subprocess.run(['ssh','beast','python3','-I','-'], input=code, text=True, capture_output=True, check=True).stdout)
    c = dict(native_root=str(root), native_manifest_sha256=meta['sha'], native_bytes=meta['bytes'], output_scratch='/mnt/big/mdb/research', directory_custody=True)
    run.mkdir(parents=True, exist_ok=False)
    save(run / 'plan.json', c)
    names = ('flux_binding_generalization.py','flux_binding_generalization_audit.py','flux_family_custody.py','retrieve_reader_evidence.py')
    extras = [(HERE / n,n) for n in names]
    pre = run / 'preflight.v1.json'
    save(pre, dict(schema_version='mrun-preflight-config-v1', hashed_roots=[Path(__file__).name,*names,pre.name],
        path_token_allowlist=[str(root),c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    req = dict(schema='mdb-binding-generalization-audit-request-v1', worker_script=Path(__file__).name, config=c,
        execution_environment=old['execution_environment'], needs=dict(cpu=True,host='beast'), task_family='analysis',
        backend='numpy',device='cpu',dtype='float32', source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},
        resource_exception=dict(ram_mb=1536,vram_mb=0,cpu_threads=2,disk_gb=1,est_wall_s=90,
            calibration_job='job-c8a4bc3cc1e4',measured_peak_rss_mb=1328.5,
            reason='Saved-only audit; largest field3.15M elements. Stream one comparison/writer, no Torch/weights/model hydration. Prior9.44M-element audit1328.5MBRSS;1.5GB/2CPU/90sestimate/300souter.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=232,logical_branches=1,
            logical_checkpoint_replays=0,max_local_work_items=256,extra=dict(task_family='analysis'),
            checkpoint_policy='Saved-only144projection+72writer comparisons,endpoint and image diagnostics,8supplied-applicability records;no model/native forwards/retries.'))
    req['request_sha256'] = hashlib.sha256(json.dumps(req,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run / 'request.json',req)
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    stage_worker(run/'payload',worker=Path(__file__),config=c,request=req,additional_files=extras)
    return dict(request_sha256=req['request_sha256'],**meta)


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    h=helpers(); read,save,sha=[h[k] for k in ('read','save','sha')];req=read(run/'request.json')
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=v for n,v in req['source_hashes'].items()):
        raise ValueError('changed/already submitted audit')
    command=worker_command(req['worker_script']);command.insert(1,'CUDA_VISIBLE_DEVICES=');command.insert(command.index('python')+1,'-B')
    resources={k:req['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')}
    job=guarded_mrun_launch(command,request=req,model=None,task='analysis',experiment='mdb-binding-generalization-audit',payload=run/'payload',
        gpu=False,needs=req['needs'],prefer_host='beast',timeout_s=300,preflight=True,retry_on_kill=False,**resources,
        note='Saved-only8context binding audit:144projection/72BF16sequentialwriter comparisons,16endpointjoins,88existingPNGdiagnostics;8searchablecontextrecords.0models/0nativeforwards/0newimages;CPU2/RAM1.5GB/90sestimate/300souter,no retries.')
    save(run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':
        worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--native',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.native.resolve()),sort_keys=True))
