"""Independent typed intervention, sham and finite stream-factorial audit."""
import hashlib
import json
import runpy
from collections import defaultdict
from pathlib import Path
import numpy as np
from mdb.response_closure import audit_response
from mdb.tensor_archive import load_tensor_archive


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')


def analyze(collection,output):
    here=Path(__file__).resolve().parent;payload=collection/'payload';data=collection/'extracted/binding-minimal-result'
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](collection);request=read(collection/'request.json');c=request['config'];summary=read(data/'summary.json')
    if read(data/'request.json')!=request or read(data/'plan.json')!=c or read(collection/'report.json')['report']!=summary:raise ValueError('cross identity differs')
    for name,digest in request['source_hashes'].items():
        if sha(payload/name)!=digest:raise ValueError('executed cross source differs')
    for name,digest in read(payload/'payload-seal.json')['wheels'].items():
        if sha(payload/name)!=digest:raise ValueError('cross wheel differs')
    if summary['counts']!=runpy.run_path(str(payload/'flux_stream_cross.py'))['counts']():raise ValueError('native cross census differs')
    source=runpy.run_path(str(here/'flux_qualified_source.py'))['QualifiedSource'](c['source_collection'],c['source_record_sha256'],c['source_audit'],c['source_audit_sha256'])
    if read(data/'source-binding.json')!=dict(record=source.record,record_sha256=c['source_record_sha256'],audit_sha256=c['source_audit_sha256']):raise ValueError('source authority differs')
    consumed=read(data/'consumed-source.json')
    if consumed['inherited_full_custody']!=source.record['verification'] or consumed['inherited_audit_sha256']!=c['source_audit_sha256'] or consumed['inherited_record_sha256']!=c['source_record_sha256']:raise ValueError('native source scope differs')
    for name,item in consumed['files'].items():
        if item!=source.record['files'][name]:raise ValueError('source subset inventory differs')
        source.member(name)
    traces=source.json('traces.json');old={(x['id'],x['mode']):x for x in traces};points={key:{p['id']:p for p in row['points']} for key,row in old.items()}
    if c['origins']!={name:{mode:old[name,mode]['origin'] for mode in ('native','old','new')} for name in ('half','row4heavy','row11heavy')}:raise ValueError('source origin declaration differs')
    if any(source.audit['replay_status'][name+'.'+mode]!='qualified' for name in c['origins'] for mode in ('native','old','new')):raise ValueError('inherited future qualification differs')
    reference=source.json('receipts.json')[0]['body'];contract=read(data/'native-contract.json')
    if contract['model_identity']!=reference['model_identity'] or contract['execution']!=reference['execution'] or not contract['bf16_reduced_precision']:raise ValueError('model/numerical contract differs')
    parameters=runpy.run_path(str(here/'analyze_flux_consumer_trace.py'))['audit_parameters'](data,c)
    if any(parameters[k]!=contract['parameter_recipes'][k] for k in ('tensors','logical_parameter_bytes','persisted_parameter_bytes')) or sha(data/'retrieve_weights.py')!=request['source_hashes']['retrieve_weights.py']:raise ValueError('parameter recipe/retriever differs')
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));store=typed['audit_store'](data/'store');manifests=store['cuts'];slot=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']
    field=lambda fp,key:slot(data/'store',fp,key);clock=lambda fp:typed['unpack'](manifests[fp]['execution_point']);desc=lambda manifest:{x['key']['value']:x['value'] for x in manifest['payload']['items']}
    source_cuts={}
    def source_manifest(fp):
        if fp not in source_cuts:source_cuts[fp]=source.cut(fp)['manifest']
        return source_cuts[fp]
    def source_field(fp,key):source_manifest(fp);return slot(source.root/'store',fp,key)
    public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt'];chains=defaultdict(list);receipts=read(data/'receipts.json')
    for record in receipts:
        body=record['body'];public(body)
        if body!=read(data/'store/receipts'/(body['fingerprint']+'.json')) or body['model_identity']!=contract['model_identity'] or body['execution']!=contract['execution']:raise ValueError('native receipt differs')
        chains[record['branch']].append(body)
    rows=read(data/'rows.json');by={(r['id'],r['corner']):r for r in rows};expected={(name,corner) for name in c['origins'] for corner in ('00','10','01')}
    if len(rows)!=9 or set(by)!=expected or len(receipts)!=36 or len(chains)!=9:raise ValueError('branch/receipt census differs')
    verified_arrays=set();shams={}
    def array(item,root=data,src=False):
        if src:source.member(item['path'])
        else:verified_arrays.add(item['path'])
        value=load_tensor_archive(root,item)
        if value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('finite BF16 archive widening required')
        return value
    same=runpy.run_path(str(here/'flux_own_state_mlp_math.py'))['same_bits']
    for key,row in by.items():
        name,corner=key;chain=chains[name+'.'+corner];parent=c['origins'][name]['old'];donor=c['origins'][name]['old' if corner=='00' else 'new'];selected='text' if corner=='01' else 'image'
        if row!=read(data/('row-'+name+'.'+corner+'.json')) or row['parent']!=parent or row['donor']!=donor or row['slot']!=selected or manifests[parent]!=source_manifest(parent):raise ValueError('corner source/parent differs')
        source_manifest(donor);base=desc(source_manifest(parent));donated=desc(source_manifest(donor));installed=desc(manifests[row['installed']])
        if any(base[k]!=donated[k] for k in base if k not in {'image','text'}) or set(base)!=set(donated) or set(base)!=set(installed):raise ValueError('source nuisance closure differs')
        if installed[selected]!=donated[selected] or any(installed[k]!=base[k] for k in base if k!=selected) or desc(manifests[row['before']])!=base:raise ValueError('single-slot installed closure differs')
        if [x['operation'] for x in chain]!=['apply','continue','continue','continue'] or [x['steps'] for x in chain[1:]]!=[1,23,1]:raise ValueError('bounded consumer operation differs')
        sequence=[row[k] for k in ('before','installed','joint_exit','noise','final')]
        if any(body['parent']!=a or body['result']!=b for a,b,body in zip(sequence,sequence[1:],chain,strict=False)):raise ValueError('named state detached from actual consumer receipt')
        for fp,expected_clock in [(row['before'],(2,'joint',2)),(row['installed'],(2,'joint',2)),(row['joint_exit'],(2,'joint',3)),(row['noise'],(2,'scheduler',None)),(row['final'],(3,'project',None))]:
            p=clock(fp)
            if (p['logical_step'],p['phase'],p['index'])!=expected_clock:raise ValueError('corner endpoint clock differs')
        if set(row['attention'])!={'image','text'}:raise ValueError('J2 attention field census differs')
        actual={role:array(item) for role,item in row['attention'].items()}
        if any(list(v.shape)!=([1,1024,3072] if role=='image' else [1,512,3072]) for role,v in actual.items()):raise ValueError('J2 midpoint geometry differs')
        if corner=='00':
            checks={}
            for role,item in points[name,'old']['joint2.attention']['arrays'].items():checks['attention.'+role]=bool(same(actual[role],array(item,source.root,True)))
            for label,fp in [('joint2.exit',row['joint_exit']),('readout.noise',row['noise']),('scheduler.latent',row['final'])]:
                original=source_manifest(points[name,'old'][label]['cut']);checks[label]=manifests[fp]['payload']==original['payload'] and manifests[fp]['execution_point']==original['execution_point']
            if checks!=row['sham_checks'] or row['sham_status']!=('qualified' if all(checks.values()) else 'instrument-replay-mismatch'):raise ValueError('sham status differs')
            shams[name]=dict(status=row['sham_status'],checks=checks)
    if verified_arrays!={p.relative_to(data).as_posix() for p in (data/'arrays').glob('*.npy')} or len(verified_arrays)!=summary['unique_arrays']:raise ValueError('captured array census differs')
    power=runpy.run_path(str(here/'flux_own_state_mlp_math.py'))['power_account'];math=runpy.run_path(str(here/'flux_stream_cross_math.py'));finite=runpy.run_path(str(here/'flux_consumer_trace_math.py'))['finite_write']
    measured={};observations=[];input_worlds={};native_inputs={};finite_writes={}
    labels=[('input','installed',['image','text']),('attention','attention',['image','text']),('joint_exit','joint_exit',['image','text']),('noise','noise',['noise_prediction']),('latent','final',['latent'])]
    source_labels=dict(input='joint1.exit',attention='joint2.attention',joint_exit='joint2.exit',noise='readout.noise',latent='scheduler.latent')
    def read_source(name,mode,label,role):
        point=points[name,mode][source_labels[label]]
        return array(point['arrays'][role],source.root,True) if point['kind']=='arrays' else source_field(point['cut'],role)
    for label,address,roles in labels:
        measured[label]={}
        for role in roles:
            worlds={};native={}
            for name in c['origins']:
                native[name]=read_source(name,'native',label,role);worlds[name,'11']=read_source(name,'new',label,role)
                for corner in ('00','10','01'):
                    row=by[name,corner];worlds[name,corner]=array(row['attention'][role]) if label=='attention' else field(row[address],role)
            expected_shape=[1,512,3072] if role=='text' else [1,1024,3072] if role=='image' else [1,1024,128]
            if any(list(x.shape)!=expected_shape or x.dtype!=np.float32 or not np.isfinite(x).all() for x in [*native.values(),*worlds.values()]):raise ValueError('factorial field geometry/finite contract differs')
            measured[label][role]={}
            if label=='input':input_worlds[role]=worlds;native_inputs[role]=native
            for case in ('row4heavy','row11heavy'):
                qualification=dict(status='qualified-intervention-and-cached-diagonal' if shams['half']['status']=='qualified' and shams[case]['status']=='qualified' else 'unqualified-sham-diagnostic',shams={k:shams[k]['status'] for k in ('half',case)},inherited_diagonals='source complete-commit replay qualified')
                metrics={corner:audit_response(native['half'],native[case],worlds['half',corner],worlds[case,corner]) for corner in ('00','10','01','11')}
                account=math['factorial'](worlds,native,case,power);account['qualification']=qualification
                measured[label][role][case]=dict(corners=metrics,factorial=account,qualification=qualification)
                step,phase,index={'input':(2,'joint',2),'attention':(2,'joint',2),'joint_exit':(2,'joint',3),'noise':(2,'scheduler',None),'latent':(3,'project',None)}[label]
                for corner,result in metrics.items():observations.append(dict(kind='observation',model_identity=contract['model_identity'],source_job=custody['job_id'],route='flux.step2.J2.stream-input-to-commit3',boundary=label,clock=dict(logical_step=step,phase=phase,index=index,port='attention-residual' if label=='attention' else 'state-cut'),role=role,case=case,corner=corner,
                    mechanics_status='verified',instrument_status=qualification['status'],terminal_status='unassessed',metrics=dict(response_rms=result['reference_response']['rms'],native_effect_rms=result['native_effect_rms'],relative_response_error=result['reference_response']['relative_rms'],endpoint_rms=result['reference_endpoint']['rms']),scope='Typed stream input intervention with ownhalf baseline; cached11/native controls inherited and freshly source-verified. No independent forecast.'))
                if label=='attention':
                    before={**{(name,'native'):x for name,x in native_inputs[role].items()},**input_worlds[role]};after={**{(name,'native'):x for name,x in native.items()},**worlds}
                    finite_writes.setdefault(role,{})[case]={corner:dict(**finite(before,after,case,corner,power),qualification=qualification) for corner in ('00','10','01','11')}
    if len(observations)!=64 or sum(len(v) for cs in finite_writes.values() for v in cs.values())!=16:raise ValueError('independent response/write census differs')
    result=dict(schema='mdb-stream-cross-audit-v1',mechanics_status='verified',instrument_status='qualified typed stream intervention' if all(v['status']=='qualified' for v in shams.values()) else 'per-case sham diagnostics retained',terminal_status='unassessed',custody=custody,
        counts=summary['counts'],typed_store=store['counts'],parameters=parameters,unique_arrays=len(verified_arrays),shams=shams,measurements=measured,finite_attention_writes=finite_writes,measurement_counts=dict(observations=len(observations),factorial_accounts=16,finite_write_accounts=16),source_verification=source.report(),
        new_native_calls=0,scope=c['scope'],auditor_source_sha256=sha(Path(__file__)),limitations='Observed finite input interventions on one known receiver/editfamily. Diagonalold/new/native controls reuse prior audited traces. Mixed corners are not new independent predictions or general programs. Signed factorial terms are context-specific, not positive causal shares. No image/semantic/collateral or later-denoising claim beyond commit3.')
    save(output,result)
    with output.with_name('observations.jsonl').open('x') as f:
        for row in observations:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    return result
