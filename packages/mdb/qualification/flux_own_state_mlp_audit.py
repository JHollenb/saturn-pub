"""Independent near-data recomputation, response audit and compact publication."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def independent(collection, output):
    import numpy as np
    here=Path(__file__).resolve().parent
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](collection)
    request=read(collection/'request.json')
    c=request['config']
    data=collection/'extracted/binding-minimal-result'
    if read(data/'plan.json')!=c or read(data/'request.json')!=request or read(data/'summary.json')!=read(collection/'report.json')['report']:
        raise ValueError('calculation request/report differs')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest:
            raise ValueError('calculation source changed')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest:
            raise ValueError('calculation wheel changed')
    original=runpy.run_path(str(collection/'payload/flux_own_state_mlp.py'))
    helper=original['helpers'](collection/'payload')
    source=original['Sources'](c,helper)
    analysis_module=runpy.run_path(str(collection/'payload/analyze_flux_own_state_mlp.py'))
    evidence=analysis_module['Evidence'](data,c)
    math=helper['flux_own_state_mlp_math.py']
    r,law=[helper[name] for name in ('flux_reentry_reference.py','flux_receiver_norm_prediction.py')]
    B=r['bf16']
    old_request=read(source.previous/'request.json')
    if sha(here/'flux_hybrid_mlp_cut.py')!=old_request['source_hashes']['flux_hybrid_mlp_cut.py']:
        raise ValueError('independent old-control subprogram differs')
    frozen_old=runpy.run_path(str(here/'flux_hybrid_mlp_cut.py'))['mlp']
    bank=helper['flux_source_receiver.py']['RecipeBank'](source.manifest,c['model_path'])
    points={'half','half.image1',*[spec['id'] for spec in c['cases']]}
    if set(evidence.records)!=points or any(set(roles)!={'image','text'} for roles in evidence.records.values()):
        raise ValueError('complete saved cohort differs')
    checked={}
    calculations=0
    receiving_writes=0
    def direct(value,stages,weights):
        # Independent composition of the selected equation from the qualified
        # primitive references; do not call the worker's composite mlp().
        paired=r['_linear'](value,{'linear.weight':weights['in']},'linear',256)
        u,v=np.split(paired,2,axis=-1)
        activation=B(r['_silu'](u)*v)
        return law['affine_delta'](activation,stages['mlp_activation'],stages['mlp'],weights['out'],B)[0]
    for role,prefix in [('image','ff'),('text','ff_context')]:
        weights=original['load_weights'](source,bank,role)
        for anchor in ('half','half.image1'):
            stages={field:source.array(dict(root='prefixes',record=source.plan['anchors'][anchor]['stages'][role+'.'+field])) for field in ('mlp_input','mlp_paired','mlp_activation','mlp')}
            for name in [anchor,*[spec['id'] for spec in c['cases'] if spec['anchor']==anchor]]:
                canonical=source.point(name,anchor,role)
                point=evidence.records[name][role]
                if any(point[key]!=value for key,value in canonical.items()):
                    raise ValueError('own/native input reference changed')
                expected_methods={'old_own','new_own','old_conditional','new_conditional','native'}
                if set(point['states'])!=expected_methods:
                    raise ValueError('five-world census differs')
                own_gate,native_gate=[source.gate(canonical[key],role) for key in ('own_context_cut','native_context_cut')]
                gates={key:math['equality'](expected,evidence.array(point[key])) for key,expected in [('own_gate',own_gate),('native_gate',native_gate)]}
                old=frozen_old(source.array(canonical['own_input']),{role+'.'+key:value for key,value in stages.items()},
                    {prefix+'.linear_in.weight':weights['in'],prefix+'.linear_out.weight':weights['out']},role,r,law)
                new=direct(source.array(canonical['own_input']),stages,weights)
                conditional=direct(source.array(canonical['native_input']),stages,weights)
                calculations+=3
                values=dict(old_own=old,new_own=new,old_conditional=source.array(canonical['old_conditional']),new_conditional=conditional,native=source.array(canonical['native_mlp']))
                checks=dict(gates=gates,mlp={},writes={})
                for method,M in values.items():
                    refs=point['states'][method]
                    if set(refs)!={'after_attention','mlp','gated','exit'}:
                        raise ValueError('receiving ports differ')
                    own=method.endswith('_own')
                    expected_updated=canonical['own_updated' if own else 'native_updated']
                    if refs['after_attention']!=expected_updated:
                        raise ValueError('own/native residual frontier changed')
                    checks['mlp'][method]=math['equality'](M,evidence.array(refs['mlp']))
                    U=source.array(expected_updated)
                    # Independently materialize both finite boundaries.
                    V=B((own_gate if own else native_gate)*M)
                    Y=B(U+V)
                    checks['writes'][method]={field:math['equality'](value,evidence.array(refs[field])) for field,value in [('gated',V),('exit',Y)]}
                    receiving_writes+=1
                checked.setdefault(name,{})[role]=checks
                print('MDB_OWN_MLP_AUDIT '+json.dumps(dict(point=name,role=role,mlp_calculations=calculations,receiving_writes=receiving_writes),sort_keys=True),flush=True)
                del values,old,new,conditional
        del weights,stages
    recomputed=analysis_module['analyze'](data,c)
    if recomputed!=read(data/'analysis-v1.json'):
        raise ValueError('independent response/transport report differs')
    referenced=set(recomputed['output_array_paths'])
    for roles in evidence.records.values():
        for point in roles.values():
            for key in ('own_gate','native_gate'):
                evidence.array(point[key]);referenced.add(point[key]['record']['path'])
    if referenced!={p.relative_to(data).as_posix() for p in (data/'arrays').glob('*.npy')}:
        raise ValueError('new-array/reference census differs')
    issues=[]
    for point,roles in checked.items():
        for role,groups in roles.items():
            for group,items in groups.items():
                for name,detail in items.items():
                    if group=='writes':
                        for field,value in detail.items():
                            if value['raw_bits_exact'] is not True:
                                issues.append(point+'.'+role+'.'+group+'.'+name+'.'+field)
                    elif detail['raw_bits_exact'] is not True:
                        issues.append(point+'.'+role+'.'+group+'.'+name)
    summary=read(data/'summary.json')
    if calculations!=36 or receiving_writes!=60 or len(referenced)!=summary['unique_arrays']:
        raise ValueError('independent work census differs')
    result=dict(schema='mdb-own-state-mlp-independent-audit-v1',custody=custody,mechanics_status='verified',
        instrument_status='qualified' if not issues and not summary['instrument_issues'] else 'instrument-broken',
        instrument_issues=issues,source_control_issues=summary['instrument_issues'],checks=checked,
        source_custody=source.record['verification'],counts=dict(independent_mlp_calculations=calculations,independent_receiving_writes=receiving_writes,
            native_calls=0,model_loads=0,images=0),unique_arrays=len(referenced),primary=recomputed['primary'],
        complete_analysis_reproduced=True,auditor_source_sha256=sha(Path(__file__)),terminal_status='unassessed',
        scope='Separate CPU recomputation of all36MLPs and60receiving writes from canonical source operands, using pinned primitive math/original parameters; full-array bits, actual P0/P1 and complete signed reports checked. No independent CUDA arithmetic, new native execution, fresh case or image claim.')
    with output.open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return result


def publish(collection,audit_path,output):
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog,compare_equations
    data=collection/'extracted/binding-minimal-result'
    request=read(collection/'request.json');c=request['config'];audit=read(audit_path)
    if audit['instrument_status']!='qualified':
        return dict(status='instrument-unqualified-records-not-published')
    module=runpy.run_path(str(collection/'payload/analyze_flux_own_state_mlp.py'))
    evidence=module['Evidence'](data,c)
    report=read(data/'analysis-v1.json')
    source_request=read(Path(c['source_collection'])/'request.json')
    receipt=read(Path(c['source_collection'])/'extracted/binding-minimal-result/receipts.json')[0]['body']
    sources={**request['source_hashes'],**{'wheel:'+k:v for k,v in read(collection/'payload/payload-seal.json')['wheels'].items()}}
    refs=[dict(path=str(p),sha256=sha(p)) for p in (audit_path,collection/'request.json',data/'source-binding.json',data/'parameter-artifacts.json')]
    output.mkdir(parents=True,exist_ok=False)
    catalog=EquationCatalog(output/'catalog');atlas=ComputationAtlas(output/'atlas');records=[];contexts={};limits={}
    natives=read(Path(c['source_collection'])/'extracted/binding-minimal-result/native.json')
    for case in c['cases']:
        name=case['id']
        actual={role:evidence.array(evidence.records[name][role]['states']['native']['exit']) for role in ('image','text')}
        for method in ('old_own','new_own'):
            predicted={role:evidence.array(evidence.records[name][role]['states'][method]['exit']) for role in ('image','text')}
            binding=dict(model_identity=c['model_identity'],source_identity=sources,input_identity=natives[name]['entry'],
                execution=dict(backend='numpy',device='cpu',dtype='float64-with-BF16-boundaries',native_target=receipt['execution']),
                frontier=dict(kind='saved-own-state-refinement',scope='Posthoc development. Candidate uses stored own upstream forecasts; actual native ports only enter separate diagnostic/evaluation worlds.'),
                horizon=dict(equation='complete joint1 image/text exit',consumer='original time gate and finite residual addition; no new native suffix'))
            context=dict(**binding,clock=dict(step=2,joint=1),receiver=case['receiver'],row_doses=case['row_doses'],method=method)
            measures={role:report['reports'][name][role]['methods'][method] for role in ('image','text')}
            record=catalog.record(law_id='flux.joint1.mlp.integration.'+('anchored-first' if method=='old_own' else 'direct-first')+'.v1',
                route_id='flux.step2.joint1.time-gated-mlp.v1',alias='cmp.flux.own-state-mlp.'+name+'.'+method,binding=binding,
                report=compare_equations(predicted,actual,max_elements=5_000_000),conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],
                tested_context=context,readers=['image','text','time-gate','residual-addition'],evidence=dict(artifacts=refs,measurements=measures,
                    source_native_states=4,additional_native_targets=0,source_forecast_request=source_request['request_sha256'],selection='Two preselected laws on four saved cases, not prospective validation or applicability induction.'))
            fp=record['fingerprint'];records.append(record);contexts[fp]=context
            limits[fp]=[dict(claim_id='saved-development-limitations',status='limitation-retained',scope=c['scope'],measurements=measures,provenance=refs)]
    atlas.ingest(output/'catalog',[row['fingerprint'] for row in records],provenance=dict(run_id=audit['custody']['job_id'],artifacts=refs),counterexamples=limits)
    for context in contexts.values():
        if len(atlas.find(applicable_to=context))!=1 or atlas.find(applicable_to={**context,'clock':dict(step=2,joint=2)}) or atlas.find(applicable_to={**context,'input_identity':'unknown'}):
            raise ValueError('applicability/abstention differs')
    result=dict(schema='mdb-own-state-mlp-publication-v1',supplied_programs=2,occurrences=8,exact_queries=8,changed_context_abstentions=16,
        new_native_calls=0,tensor_bytes_copied=0,scope='Exact supplied applicability on saved own-state development, not learned rules, prospective generalization or image evidence.')
    with (output/'provenance.json').open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    return result


def worker():
    payload=Path.cwd();request=read(os.environ['SATURN_DEBUG_REQUEST']);c=request['config']
    if read(os.environ['SATURN_RUN_CONFIG'])!=c or any(sha(payload/name)!=digest for name,digest in request['source_hashes'].items()):
        raise ValueError('audit payload changed')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]='4'
    collection=Path(c['remote_collection']);remote=read(collection/'remote-custody.json')
    if sha(collection/'remote-custody.json')!=c['remote_record_sha256'] or remote['verification']['request_sha256']!=c['source_request_sha256']:
        raise ValueError('own-state custody record changed')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID'];out=scratch/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();save=lambda p,v:p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
    save(out/'request.json',request);save(out/'plan.json',c)
    audit=independent(collection,out/'audit-v1.json')
    if audit['custody']!=remote['verification']:
        raise ValueError('full independent custody differs')
    publication=publish(collection,out/'audit-v1.json',out/'index-v1')
    data=collection/'extracted/binding-minimal-result';copied={}
    def copy(p,name):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        if sha(target)!=sha(p):raise ValueError('projection copy changed bytes')
        copied[name]=dict(sha256=sha(target),bytes=target.stat().st_size,remote_path=str(p))
    copy(collection/'remote-custody.json','remote-custody.json')
    for name in ('request.json','plan.json','summary.json','states.json','checks.json','source-binding.json','source-reads.json','timeline.jsonl','parameter-artifacts.json','retrieve_weights.py','analysis-v1.json'):
        copy(data/name,'calculation-proof/'+name)
    save(out/'projection.json',dict(schema='mdb-compact-scientific-projection-v1',remote_collection=str(collection),remote_record_sha256=c['remote_record_sha256'],
        source_request_sha256=c['source_request_sha256'],copied=copied,scope='Partial report/metadata/catalog projection. Full calculated tensors, original archive/log and complete independent verification stay on Beast. Array references in calculation-proof are explicitly remote, with exact member retrieval.'))
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>32*1024**2:
        raise ValueError('compact projection exceeded bound')
    if any(name in sys.modules for name in ('torch','diffusers','transformers')):
        raise ValueError('independent CPU audit imported model framework')
    summary=dict(schema='mdb-own-state-mlp-remote-audit-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),
        scientific_custody=audit['custody'],mechanics_status='verified',instrument_status=audit['instrument_status'],terminal_status='unassessed',
        audit_sha256=sha(out/'audit-v1.json'),projection_sha256=sha(out/'projection.json'),seconds=time.monotonic()-started,publication=publication,primary=audit['primary'],counts=audit['counts'])
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,pointer,calculation_run):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    here=Path(__file__).resolve().parent;base=runpy.run_path(str(here/'flux_binding_minimal.py'));save=base['save']
    record=runpy.run_path(str(here/'collect_flux_remote_directory.py'))['checked_pointer'](pointer)
    previous=read(calculation_run/'request.json');base['bind_environment'](previous['execution_environment'])
    if previous['request_sha256']!=record['verification']['request_sha256'] or read(calculation_run/'submission.json')['job_id']!=record['verification']['job_id']:
        raise ValueError('audit source job/request differs')
    run.mkdir(parents=True,exist_ok=False)
    c=dict(remote_collection=record['collection'],remote_record_sha256=read(pointer)['record_sha256'],source_request_sha256=previous['request_sha256'],output_scratch='/mnt/big/mdb/research',directory_custody=True)
    save(run/'plan.json',c)
    names=('flux_binding_minimal.py','flux_family_custody.py','flux_hybrid_mlp_cut.py')
    extras=[(here/name,name) for name in names]+[(run/'plan.json','plan.json')]
    preflight=run/'preflight.v1.json';original=previous['config']
    save(preflight,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[name for _,name in extras],preflight.name],
        path_token_allowlist=[c['remote_collection'],original['source_collection'],str(Path(original['source_audit']).parent),original['selection_collection'],original['previous_conditional_root'],original['model_path'],c['output_scratch'],'binding-minimal-result']))
    extras.append((preflight,preflight.name))
    request=dict(schema='mdb-own-state-mlp-audit-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='float64-audit-of-BF16',
        execution_environment=previous['execution_environment'],source_hashes={name:sha(p) for p,name in [(Path(__file__),Path(__file__).name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=36,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=36,
            extra=dict(task_family='analysis'),checkpoint_policy='Full remote custody, canonical own/native source/gate joins, independent36MLP/60finitewrite recomputation, complete response/signed reports,8catalog occurrences.0native/framework/images.'),
        resource_exception=dict(ram_mb=3072,cpu_threads=4,disk_gb=1,est_wall_s=180,calibration_job='job-d37341c851a5',measured_peak_rss_mb=1474.8,
            reason='Current role-sequential36MLP lane measured1475MBRSS. Independent recomputation adds complete archive verification and one4.7Mscalar catalog pair at a time;3GB bounds those extra arrays,4CPU,expectedunder4min/720outer,noretries. Bulk tensors stayremote; compact reports only.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request)
    stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    base=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=read(run/'request.json');base['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/name)!=digest for name,digest in r['source_hashes'].items()):raise ValueError('already submitted or changed audit')
    command=worker_command(Path(__file__).name);command.insert(1,'CUDA_VISIBLE_DEVICES=');command.insert(command.index('python')+1,'-B')
    job=guarded_mrun_launch(command,request=r,model=None,task='analysis',experiment='mdb-flux-own-state-mlp-independent-audit',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=720,retry_on_kill=False,preflight=True,
        **{key:r['resource_exception'][key] for key in ('ram_mb','cpu_threads','disk_gb','est_wall_s')},
        note='CPU independent own-state audit: fullrawcustody,36MLP/60receivingwrite recomputations,raw-bitcontrols,allP0/P1and signed/mixed reports,8exactapplicability records.0native/framework/images;3GB/4CPU from1475MBmeasuredlane,expectedunder4min/720outer,noretries,compactprojectiononly.')
    base['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('pointer',type=Path,nargs='?');parser.add_argument('calculation_run',type=Path,nargs='?');parser.add_argument('--submit',action='store_true');args=parser.parse_args()
        print(json.dumps(submit(args.run.resolve()) if args.submit else stage(args.run.resolve(),args.pointer.resolve(),args.calculation_run.resolve()),sort_keys=True))
