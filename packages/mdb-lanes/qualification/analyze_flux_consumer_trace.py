"""Independent saved-only finite-write/consumer audit; no model-framework imports."""
import base64
import hashlib
import json
import runpy
import struct
from collections import defaultdict
from pathlib import Path
import numpy as np
from mdb_lanes.parameter_artifacts import validate_manifest
from mdb_lanes.response_closure import audit_response
from mdb_lanes.tensor_archive import load_tensor_archive


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')


def audit_parameters(data,c):
    manifest=validate_manifest(read(data/'parameter-artifacts.json'));root=Path(c['model_path']);files={};tensors=set()
    prefixes=tuple(['transformer_blocks.'+str(i)+'.' for i in (2,3,4)]+['single_transformer_blocks.'+str(i)+'.' for i in range(20)]+['norm_out.','proj_out.'])
    for item in manifest['artifacts']:
        source=item['source'];name=source['tensor'];p=root/source['file']
        if source['file'] not in files:
            if sha(p)!=source['file_sha256']:raise ValueError('original parameter file differs')
            with p.open('rb') as f:size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
            files[source['file']]=(8+size,header)
        offset,header=files[source['file']];spec=header[name]
        if (name in tensors or not name.startswith(prefixes) or item['parameter_ports']!=[name]
                or any(source[key]!=spec[key] for key in ('dtype','shape','data_offsets'))):raise ValueError('named parameter recipe/header differs')
        tensors.add(name);first,last=spec['data_offsets'];prefix=base64.b64decode(item['npy_prefix_b64']);payload_hash=hashlib.sha256();archive_hash=hashlib.sha256(prefix)
        with p.open('rb') as f:
            f.seek(offset+first);remaining=last-first
            while remaining:
                block=f.read(min(1<<20,remaining))
                if not block:raise ValueError('truncated parameter')
                remaining-=len(block);payload_hash.update(block);archive_hash.update(block)
        if payload_hash.hexdigest()!=source['payload_sha256'] or archive_hash.hexdigest()!=item['sha256'] or item['bytes']!=len(prefix)+last-first:raise ValueError('exact parameter archive recipe differs')
    required={name for _,header in files.values() for name in header if name!='__metadata__' and name.startswith(prefixes)}
    if tensors!=required:raise ValueError('required suffix parameter census differs')
    return dict(tensors=len(tensors),checkpoint_files=len(files),logical_parameter_bytes=sum(x['bytes'] for x in manifest['artifacts']),persisted_parameter_bytes=0,exact_recipe_hashes=True)


def analyze(collection,output):
    here=Path(__file__).resolve().parent;data=collection/'extracted/binding-minimal-result';payload=collection/'payload'
    collection_record=read(collection/'remote-custody.json')
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](collection,collection_record.get('limits'));request=read(collection/'request.json');c=request['config'];summary=read(data/'summary.json')
    if read(data/'request.json')!=request or read(data/'plan.json')!=c or read(collection/'report.json')['report']!=summary:raise ValueError('executed trace identity differs')
    for name,digest in request['source_hashes'].items():
        if sha(payload/name)!=digest:raise ValueError('trace source differs')
    for name,digest in read(payload/'payload-seal.json')['wheels'].items():
        if sha(payload/name)!=digest:raise ValueError('trace installed wheel differs')
    if summary['counts']!=runpy.run_path(str(payload/'flux_consumer_trace.py'))['counts']():raise ValueError('native physical census differs')
    source=Path(c['source_collection']);remote=read(source/'remote-custody.json');source_data=source/'extracted/binding-minimal-result';original_audit=read(c['source_audit'])
    if (sha(source/'remote-custody.json')!=c['source_record_sha256'] or runpy.run_path(str(here/'flux_family_custody.py'))['verify'](source)!=remote['verification']
            or sha(c['source_audit'])!=c['source_audit_sha256'] or original_audit['custody']!=remote['verification'] or original_audit['mechanics_status']!='verified'
            or read(data/'source-binding.json')!=dict(remote_record=remote,source_audit_sha256=c['source_audit_sha256'])):raise ValueError('original source qualification differs')
    specs=c['branches'];traces=read(data/'traces.json');by={(x['id'],x['mode']):x for x in traces};source_by={(x['id'],x['mode']):x for x in read(source_data/'rows.json')}
    if len(traces)!=9 or set(by)!={(name,mode) for name in ('half','row4heavy','row11heavy') for mode in ('native','old','new')} or specs!=[{k:x[k] for k in ('id','mode','entry','commits')} for x in read(source_data/'rows.json')]:raise ValueError('complete branch cohort differs')
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));store=typed['audit_store'](data/'store');slot=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']
    manifests=store['cuts'];field=lambda fp,key:slot(data/'store',fp,key);clock=lambda fp:typed['unpack'](manifests[fp]['execution_point'])
    original_manifest=lambda fp:read(source_data/'store/cuts'/(fp+'.json'))['manifest']
    public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt'];receipts=read(data/'receipts.json');chains=defaultdict(list)
    contract=read(data/'native-contract.json');reference=read(source_data/'receipts.json')[0]['body'];native_worker=read(data/'summary.json')
    if contract['model_identity']!=reference['model_identity'] or contract['execution']!=reference['execution'] or not contract['bf16_reduced_precision']:raise ValueError('source numerical execution contract differs')
    if sha(data/'retrieve_weights.py')!=request['source_hashes']['retrieve_weights.py']:raise ValueError('standalone parameter retriever differs')
    parameters=audit_parameters(data,c)
    if parameters['tensors']!=contract['parameter_recipes']['tensors'] or parameters['logical_parameter_bytes']!=contract['parameter_recipes']['logical_parameter_bytes']:raise ValueError('native parameter census differs')
    for record in receipts:
        body=record['body'];public(body)
        if (body!=read(data/'store/receipts'/(body['fingerprint']+'.json')) or body['operation']!='continue' or body['steps']!=1
                or body['model_identity']!=contract['model_identity'] or body['execution']!=contract['execution']):raise ValueError('durable native receipt differs')
        chains[record['branch']].append(body)
    if len(receipts)!=225 or len(chains)!=9:raise ValueError('receipt census differs')
    point_ids=['joint1.exit']+[f'joint{i}.{part}' for i in (2,3,4) for part in ('attention','exit')]+[f'single{i}.exit' for i in range(20)]+['readout.norm','readout.noise','scheduler.latent']
    operation_points=[dict(logical_step=2,phase=phase,index=index) for phase,indices in [('joint',range(2,5)),('single',range(20)),('readout',[None]),('scheduler',[None])] for index in indices]
    statuses={};all_array_paths=set()
    for key,row in by.items():
        spec=source_by[key];chain=chains[row['id']+'.'+row['mode']]
        if row['origin']!=spec['entry'] or row['expected_commit']!=spec['commits'][0] or manifests[row['origin']]!=original_manifest(spec['entry']):raise ValueError('trace detached from original observed input')
        if row!=read(data/('trace-'+row['id']+'.'+row['mode']+'.json')) or len(row['transitions'])!=25 or [x['id'] for x in row['points']]!=point_ids or len(chain)!=25:raise ValueError('per-branch trace census differs')
        previous=row['origin']
        for i,(transition,receipt,expected) in enumerate(zip(row['transitions'],chain,operation_points,strict=True)):
            before,after=transition['before'],transition['after'];actual=clock(before)
            if (transition['point']!=actual or any(actual[k]!=v for k,v in expected.items()) or transition['receipt']!=receipt['fingerprint']
                    or receipt['parent']!=before or receipt['result']!=after):raise ValueError('named component/consumer receipt join differs')
            if i==0:
                if manifests[before]['payload']!=manifests[previous]['payload'] or clock(before)!=clock(previous):raise ValueError('first consumer input differs')
            elif before!=previous:raise ValueError('individual operation lineage differs')
            previous=after
        if previous!=row['final'] or clock(previous)['logical_step']!=3 or clock(previous)['phase']!='project':raise ValueError('next commit horizon differs')
        expected=original_manifest(row['expected_commit']);exact=manifests[previous]['payload']==expected['payload'] and manifests[previous]['execution_point']==expected['execution_point']
        if exact!=row['full_payload_replay_exact'] or row['replay_status']!=('qualified' if exact else 'instrument-replay-mismatch'):raise ValueError('declared replay status differs')
        statuses[row['id']+'.'+row['mode']]=row['replay_status']
        for point in row['points']:
            label=point['id'];is_array=label.endswith('.attention') or label=='readout.norm'
            expected_roles={'image'} if label=='readout.norm' else {'noise_prediction'} if label=='readout.noise' else {'latent'} if label=='scheduler.latent' else {'image','text'}
            if point['kind']!=('arrays' if is_array else 'cut') or set(point['arrays'] if is_array else point['fields'])!=expected_roles:raise ValueError('named point kind/role set differs')
            if point['kind']=='cut':
                if point['clock']!=clock(point['cut']):raise ValueError('measurement clock differs')
                expected_cut=row['origin'] if point['id']=='joint1.exit' else row['final'] if point['id']=='scheduler.latent' else next(t['after'] for t in row['transitions'] if ('joint'+str(t['point']['index'])+'.exit' if t['point']['phase']=='joint' else 'single'+str(t['point']['index'])+'.exit' if t['point']['phase']=='single' else 'readout.noise' if t['point']['phase']=='readout' else 'scheduler.latent')==point['id'])
                if point['cut']!=expected_cut:raise ValueError('named endpoint detached from native operation')
            else:
                transition=next(t for t in row['transitions'] if t['receipt']==point['receipt'])
                if any(point[k]!=transition[k] for k in ('before','after')) or point['clock']!=transition['point']:raise ValueError('captured intermediate detached from native operation')
                expected_phase,expected_index=('readout',None) if label=='readout.norm' else ('joint',int(label.split('.')[0][5:]))
                if transition['point']['phase']!=expected_phase or transition['point']['index']!=expected_index:raise ValueError('named midpoint attached to wrong native operation')
                for item in point['arrays'].values():all_array_paths.add(item['path'])
    if all_array_paths!={p.relative_to(data).as_posix() for p in (data/'arrays').glob('*.npy')} or len(all_array_paths)!=summary['unique_arrays']:raise ValueError('intermediate array census differs')
    math=runpy.run_path(str(here/'flux_consumer_trace_math.py'));power=runpy.run_path(str(here/'flux_own_state_mlp_math.py'))['power_account']
    def qualification(case,modes):
        required={name+'.'+mode:by[name,mode]['replay_status'] for name in ('half',case) for mode in ('native',*modes)}
        return dict(status='qualified-original-futures' if all(x=='qualified' for x in required.values()) else 'unqualified-replay-diagnostic',required_worlds=required)
    curves=[];writes=[];previous_worlds=None;observations=[];noise_worlds=None;cohort_qualified=all(x=='qualified' for x in statuses.values())
    for index,label in enumerate(point_ids):
        worlds_by_role=defaultdict(dict)
        for key,row in by.items():
            point=row['points'][index]
            if point['kind']=='arrays':values={role:load_tensor_archive(data,item) for role,item in point['arrays'].items()}
            else:values={('noise' if role=='noise_prediction' else role):field(point['cut'],role) for role in point['fields']}
            for role,value in values.items():
                if not np.isfinite(value).all():raise ValueError('nonfinite trace value')
                expected_shape=[1,1024,3072] if role=='image' else [1,512,3072] if role=='text' else [1,1024,128]
                if list(value.shape)!=expected_shape or value.dtype!=np.float32:raise ValueError('named point geometry/dtype differs')
                worlds_by_role[role][key]=value
        measured={};accounts={}
        for role,worlds in worlds_by_role.items():
            measured[role]={};accounts[role]={}
            for case in ('row4heavy','row11heavy'):
                measured[role][case]={};accounts[role][case]={}
                for mode in ('old','new'):
                    result=audit_response(worlds['half','native'],worlds[case,'native'],worlds['half',mode],worlds[case,mode]);result['qualification']=qualification(case,(mode,));measured[role][case][mode]=result
                    required=[by[k]['replay_status'] for k in [('half','native'),(case,'native'),('half',mode),(case,mode)]]
                    observations.append(dict(kind='observation',specimen=c['checkpoint_sha256'],model_identity=contract['model_identity'],source_job=custody['job_id'],
                        receiver='fresh-seed81119',route='flux.step2.post-joint1.native-suffix',clock=label,role=role,case=case,program=mode,
                        mechanics_status='verified',instrument_status='qualified' if all(x=='qualified' for x in required) else 'instrument-replay-mismatch',terminal_status='unassessed',
                        measurements=dict(response_rms=result['reference_response']['rms'],native_effect_rms=result['native_effect_rms'],relative_response_error=result['reference_response']['relative_rms'],endpoint_rms=result['reference_endpoint']['rms']),
                        scope='Retrospective own-baseline paired response at native component boundary; observed downstream continuation, not a new forecast or semantic edit claim.'))
                    if previous_worlds is not None and index<=26:
                        accounts[role][case][mode]=math['finite_write'](previous_worlds[role],worlds,case,mode,power)
                        accounts[role][case][mode]['qualification']=qualification(case,(mode,))
                errors={mode:math['response_error'](worlds,case,mode) for mode in ('old','new')}
                measured[role][case]['old_to_new']=power(dict(old_response_error=errors['old'],program_change=errors['new']-errors['old']),errors['new'])
                measured[role][case]['old_to_new']['qualification']=qualification(case,('old','new'))
        curves.append(dict(index=index,id=label,roles=measured))
        if previous_worlds is not None and index<=26:writes.append(dict(before=point_ids[index-1],after=label,roles=accounts))
        previous_worlds=worlds_by_role
        if label=='readout.noise':noise_worlds=worlds_by_role['noise']
    inputs={key:field(row['origin'],'latent') for key,row in by.items()};outputs={key:field(row['final'],'latent') for key,row in by.items()};sigmas={key:field(row['origin'],'sigmas') for key,row in by.items()}
    same_bits=runpy.run_path(str(here/'flux_own_state_mlp_math.py'))['same_bits'];first=next(iter(by));common_latent=all(same_bits(v,inputs[first]) for v in inputs.values());common_sigmas=all(same_bits(v,sigmas[first]) for v in sigmas.values())
    if not common_sigmas:raise ValueError('Euler worlds have differing schedules; one h is inapplicable')
    schedule=sigmas[first];h=float(np.float32(schedule[3]-schedule[2]));euler={case:{mode:math['euler_account'](inputs,noise_worlds,outputs,h,case,mode,power) for mode in ('old','new')} for case in ('row4heavy','row11heavy')}
    for case,methods in euler.items():
        for mode,account in methods.items():account['qualification']=qualification(case,(mode,))
    reversals={}
    for role in ('image','text','noise','latent'):
        reversals[role]={}
        for case in ('row4heavy','row11heavy'):
            rows=[p for p in curves if role in p['roles']];deltas=[dict(id=p['id'],signed_new_minus_old_mse=p['roles'][role][case]['new']['defect_power']['change_mse']-p['roles'][role][case]['old']['defect_power']['change_mse']) for p in rows]
            q=qualification(case,('old','new'));first_worse=next((r for r in deltas if r['signed_new_minus_old_mse']>0),None)
            reversals[role][case]=dict(qualification=q,first_observed_new_worse=first_worse if q['status']=='qualified-original-futures' else None,first_observed_trace_new_worse=first_worse,signed_curve=deltas)
    finite_accounts=sum(len(methods) for w in writes for cases in w['roles'].values() for methods in cases.values())
    if len(observations)!=228 or finite_accounts!=208:raise ValueError('actual independent measurement/account census differs')
    result=dict(schema='mdb-native-consumer-trace-audit-v1',mechanics_status='verified',instrument_status='qualified retrospective complete-commit microscope' if cohort_qualified else 'per-row replay mismatches retained',terminal_status='unassessed',custody=custody,
        original_source_custody=remote['verification'],original_audit_sha256=c['source_audit_sha256'],counts=summary['counts'],parameters=parameters,typed_store=store['counts'],unique_intermediate_arrays=len(all_array_paths),
        replay_status=statuses,measurement_counts=dict(observations=len(observations),finite_write_accounts=finite_accounts,euler_accounts=4),curves=curves,finite_writes=writes,euler=dict(h=h,sigmas=schedule.tolist(),common_incoming_latent_bits=common_latent,common_schedule_bits=common_sigmas,accounts=euler),first_observed_reversals=reversals,
        new_native_calls=0,scope=c['scope'],limitations='All futures were previously observed; this is a retrospective component microscope. Joint blocks have attention/MLP residual separation. Single blocks have complete-block boundaries only. Readout maps change coordinates; no residual transport account across those maps. Large finite-write/cross terms are observational leads, not positive causal shares. Relative errors use each boundary own native effect; compare old vs new within a boundary, not amplification across spaces.',
        auditor_source_sha256=sha(Path(__file__)))
    save(output,result)
    with output.with_name('observations.jsonl').open('x') as f:
        for row in observations:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    return result
