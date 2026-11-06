"""Separate text and image attention carriers from four observed fresh parents."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
H=runpy.run_path(str(HERE/'flux_source_receiver.py'))
read,save,sha=[H[k] for k in ('read','save','sha')]


def stage(run,fresh,audit,panel='streams',operand_study=None):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    evidence=audit/'compact-collection-01/extracted/binding-minimal-result';report=read(evidence/'audit.json')
    if report['mechanics_status']!='verified' or any(sha(evidence/n)!=v for n,v in read(evidence/'files.json').items()):raise ValueError('fresh audit inventory differs')
    old=read(fresh/'request.json');c=dict(old['config']);jid=read(fresh/'submission.json')['job_id']
    if jid!=report['source_job']:raise ValueError('fresh audit/request differs')
    root=Path(c['output_scratch'])/jid/'binding-minimal-result';parents=read(evidence/'source-fresh-parents.json')
    code='import json\nfrom pathlib import Path\nr=Path('+repr(str(root))+')\nparents='+repr({r:p['parent'] for r,p in parents.items()})+'\n'
    code+='''inventory=json.loads((r/'files.json').read_text());names={'prefixes.json','fresh-parents.json','cases.json'}
prefixes=json.loads((r/'prefixes.json').read_text())['operands']
for fields in prefixes.values():
 for a in fields.values():names.add(a['path'])
for fp in parents.values():
 n='store/cuts/'+fp+'.json';names.add(n);cut=json.loads((r/n).read_text())
 for page in cut['slots'].values():names.add('store/pages/'+page+'.json')
 for blob in cut['physical_blobs']:names.add('store/blobs/'+blob+'.safetensors')
print(json.dumps({n:dict(sha256=inventory[n],bytes=(r/n).stat().st_size) for n in sorted(names)}))
'''
    inputs=json.loads(subprocess.run(['ssh','beast','python3','-I','-'],input=code,text=True,capture_output=True,check=True).stdout)
    for k in ('conditioning','origin_parents'):c.pop(k,None)
    c.update(schema='mdb-reader-streams-plan-v1',input_root=str(root),source_files_sha256=report['native_manifest_sha256'],inputs=inputs,
        parents={r:p['parent'] for r,p in parents.items()},expected_prefix_counts=dict(native_blocks=0,native_sdpa=0,native_transitions=0),
        case_specs=[dict(id='base',kind='base'),dict(id='band2',kind='band-cross',band=2),
            dict(id='text-only',kind='band-text',band=2),dict(id='image-only',kind='band-image',band=2)],
        historical_controls={r+'.'+arm:r+'.'+arm for r in parents for arm in ('base','band2')},
        semantic_hypotheses=dict(stage='Exploratory path separation; all predictions frozen before target execution.',
            timing='Text-only attention output changes working text at joint0, first image at joint1; image-only output preserves current text through joint0 and changes current image.',
            alternatives=['color response mainly current image attention carrier','color response mainly continuing text carrier','both paths contribute with interaction'],
            consumer='Compare signed green-score response and RGB energy for text-only/image-only/full band2 against the same baseline. Retain interaction full-text-image+base.',
            masks='Same baseline-only green/yellow masks frozen in fresh-seed campaign; visual baseline conformance checked separately.',
            caveat='Separates attention output streams, not individual heads or semantic concepts. Full and base are eight historical controls; eight stream-split futures are new.'),
        scope='Four already observed seed193 color/order parents;8historical base/band2 controls+8new text-only/image-only attention-output futures. Each new split computes native full attention for base and changedKV at identical Q/shape/strides then routes one complete stream. All16local reader forecasts seal before any target block. Native image continuation observed, not numerically predicted.',
        planned_counts=dict(sdpa=12,staged_matmuls=0,staged_softmax=0,linear=48,layernorm=16,swiglu=16,
            native_sdpa=408,native_blocks=400,native_transitions=432,images=16,model_loads=1,encodes=0,noise_draws=0))
    if panel not in ('streams','kv','value-streams'):raise ValueError('unknown reader panel')
    c['panel']=panel
    if panel=='kv':
        c.update(case_specs=[dict(id='base',kind='base'),dict(id='band2',kind='band-cross',band=2),
                dict(id='keys-only',kind='band-key',band=2),dict(id='values-only',kind='band-value',band=2)],
            semantic_hypotheses=dict(stage='Exploratory operand factorial; frozen before target suffixes.',
                alternatives=['value payload dominates signed color response','changed keys dominate routing response','joint key/value interaction is required'],
                prediction='Values-only has larger absolute signed green-score response on originally green object than keys-only, and the same negative sign as full band2. Each context retained, including misses.',
                arithmetic='All16complete text reader head/residual/MLP/text fields predicted from own modified K/V before any target block.',
                controls='Eight base/full historical controls; eight new K-only/V-only band cases; receiving textKV/query/state held fixed.',
                caveat='Native final image observed, not numerically forecast. K/V edits affect both query streams; image dominance of jointKV does not certify dominance for each operand separately.'),
            scope='Four observed seed193 color/order parents,8historical base/full controls+8new band2K-only/V-only futures. Freeze16primitive reader predictions and value-dominance/sign hypotheses before any targetblock. Full original native suffix;0prefix regeneration/encodes/noise/retries. Signed operand effects and factorial interaction retained.',
            planned_counts={**c['planned_counts'],'sdpa':16,'native_sdpa':400})
    elif panel=='value-streams':
        if operand_study is None:raise ValueError('audited operand controls required')
        control_evidence=operand_study/'compact-collection-01/extracted/binding-minimal-result'
        control_audit=read(control_evidence/'audit.json')
        if control_audit['mechanics_status']!='verified' or any(sha(control_evidence/n)!=v for n,v in read(control_evidence/'files.json').items()):raise ValueError('operand control audit differs')
        c.update(case_specs=[dict(id='base',kind='base'),dict(id='band2',kind='band-value',band=2),
                dict(id='text-only',kind='value-text',band=2),dict(id='image-only',kind='value-image',band=2)],
            historical_control_root=str(Path(c['output_scratch'])/control_audit['source_job']/'binding-minimal-result'),
            historical_control_manifest_sha256=control_audit['native_manifest_sha256'],
            historical_controls={r+'.'+arm:r+'.'+oldarm for r in parents for arm,oldarm in (('base','base'),('band2','values-only'))},
            semantic_hypotheses=dict(stage='V-only output-stream mediation; frozen before target execution.',
                prediction='Image-only V-source output has larger absolute green-score response on originally green object than text-only and retains negative sign in all four contexts; retain misses.',
                timing='Text-only first changes image at joint1 and text at joint0; image-only has reciprocal timing. Current exit closures match base/full V-only stream choices.',
                controls='Eight base/full-V historical controls from independently audited operand panel; eight new V-only stream-split futures.',
                caveat='Routes complete streams, not individual heads. Native final image observed, not numerically forecast; applies only to tested contexts/clock/source band.'),
            scope='Four observed seed193 parents;8historical base/values-only controls+8new V-only text/image output splits.16local primitive reader forecasts and image-dominance/sign hypotheses sealed before targets. Copy source imageVband2 only; K/textV/Q fixed, then route complete native output streams. No raw-prompt/independentFlash/full-image forecast or universal symbol claim.')
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',c)
    extras=[(HERE/n,n) for n in ('flux_reader_route_future.py','flux_reader_routes.py','flux_reader_arithmetic.py','flux_reader_attention.py',
        'flux_source_receiver.py','flux_reentry_reference.py','flux_family_custody.py')]
    for n in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):
        shutil.copyfile(fresh/n,run/n);extras.append((run/n,n))
    worker=HERE/'flux_reader_route_future_worker.py';pre=run/'preflight.v1.json'
    save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],path_token_allowlist=[str(root),c['model_path'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    req=dict(schema='mdb-reader-'+panel+'-request-v1',worker_script=worker.name,config=c,model=c['model'],execution_environment=old['execution_environment'],
        needs=dict(cuda=True,host='beast'),task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,
        source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        resource_exception=dict(ram_mb=9216,vram_mb=10240,cpu_threads=4,disk_gb=2,est_wall_s=150,
            calibration_job='job-4c4cf4ac3624',measured_peak_rss_mb=8209.8,measured_peak_vram_mb=9264.,
            reason='Same scalar B1/512px model/VAE and16suffixes measured98.26s,8209.8MBRSS/9264MBVRAM. At most8extra attention calls for stream splits; no extra blocks, encoders or weights.9GBRAM/10GBVRAM reserve,150sestimate.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=32,logical_branches=16,logical_checkpoint_replays=16,max_local_work_items=64,
            checkpoint_policy='One resident model/16sealed predictions then16native suffixes;'+str(c['planned_counts']['sdpa'])+'forecastFlash+48linears,432steps/400blocks/'+str(c['planned_counts']['native_sdpa'])+'nativeSDPA/16VAE;0encodes/noise/retries.8historical controls+8new '+panel+' factorial cases.'))
    req['request_sha256']=hashlib.sha256(json.dumps(req,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',req)
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](req['execution_environment']);stage_worker(run/'payload',worker=worker,config=c,request=req,additional_files=extras)
    return dict(request_sha256=req['request_sha256'],cases=16)


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    r=read(run/'request.json');runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](r['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('changed/already submitted streams campaign')
    cmd=worker_command(r['worker_script']);cmd.insert(cmd.index('python')+1,'-B');os.environ.setdefault('MRUN_URL','http://192.168.68.68:9025')
    resources={k:r['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')}
    panel=r['config'].get('panel','streams');pc=r['config']['planned_counts']
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-reader-'+panel,payload=run/'payload',gpu=True,needs=r['needs'],prefer_host='beast',timeout_s=900,preflight=True,retry_on_kill=False,**resources,
        note='16B1/512px '+panel+' futures:8historical base/band2+8new factorial cases. All16local forecasts/semantic hypotheses seal before targetblocks.1model acquisition,432steps/400blocks/'+str(pc['native_sdpa'])+'nativeSDPA/16VAE,0encodes/noise/retries.150sestimate/600sworker/900souter; calibrated9GBRAM/10GBVRAM.')
    save(run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--fresh',type=Path);p.add_argument('--audit',type=Path);p.add_argument('--panel',choices=('streams','kv','value-streams'),default='streams');p.add_argument('--operand-study',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.fresh.resolve(),a.audit.resolve(),a.panel,a.operand_study.resolve() if a.operand_study else None)))
