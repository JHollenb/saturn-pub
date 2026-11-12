"""Build a labelled analysis view; preserve both raw execution histories."""
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path


def assemble(partial,continuation,view):
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    here=Path(__file__).resolve().parent;m=runpy.run_path(str(here/'flux_family_custody.py'))
    custody=m['verify'](continuation)
    old=partial/'extracted/binding-minimal-result';new=continuation/'extracted/binding-minimal-result'
    forensic=read(partial/'forensic-custody.json');recovery=read(old/'recovery.json');new_summary=read(new/'summary.json')
    if read(partial/'job.json')['state']!='timeout'or forensic['archive_sha256']!=sha(partial/'archive.tar.gz'):raise ValueError('original timeout custody differs')
    for name,item in recovery['original_inventory'].items():
        if sha(old/name)!=item['sha256']or(old/name).stat().st_size!=item['bytes']:raise ValueError('original bytes changed')
    if new_summary['source_job']!=forensic['job_id']or read(new/'plan.json')['source_inventory_sha256']!=sha(old/'recovery.json'):raise ValueError('continuation lineage differs')
    selected=[read(old/'completed-base.native.json'),*read(new/'rows.json')]
    if len(selected)!=10 or len({(r['id'],r['mode'])for r in selected})!=10:raise ValueError('incomplete case matrix')
    # Recover only the completed native control chain. Keep the interrupted
    # predicted chain in the raw forensic source, never silently count it away.
    base=read(old/'plan.json')['parents']['base'];body={r['result']:r for p in(old/'store/receipts').glob('*.json')for r in[read(p)]}
    cursor=selected[0]['final'];chain=[]
    while cursor!=base:
        r=body[cursor];chain.append(dict(branch='base.native',body=r));cursor=r['parent']
        if len(chain)>56:raise ValueError('control chain loop')
    chain.reverse()
    if len(chain)!=56:raise ValueError('control receipt count differs')
    receipts=chain+read(new/'receipts.json')
    view.mkdir(parents=True,exist_ok=False);data=view/'extracted/binding-minimal-result';data.mkdir(parents=True)
    origins={}
    def put(src,relative):
        dst=data/relative;digest=sha(src)
        if dst.exists():
            if sha(dst)!=digest:raise ValueError('analysis-view collision: '+relative)
        else:dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
        origins.setdefault(relative,[]).append(dict(path=str(src.resolve()),sha256=digest))
    for root in(old,new):
        for prefix in('arrays','store'):
            for p in sorted((root/prefix).rglob('*')):
                if p.is_file():put(p,p.relative_to(root).as_posix())
    for name in('plan.json','forecasts.json','parameter-artifacts.json','retrieve_weights.py'):
        put(old/name,name)
    for p in(old/'transaction').rglob('*'):
        if p.is_file():put(p,p.relative_to(old).as_posix())
    for row in selected:put((old if row['mode']=='native'and row['id']=='base'else new)/row['png'],row['png'])
    shutil.copytree(partial/'payload',view/'payload');shutil.copyfile(partial/'request.json',view/'request.json')
    counts=dict(loads=2,encodes=0,noise_draws=0,native_transitions=535,acts=20,receipts=96,images=10)
    save=m['_save'];save(data/'rows.json',selected);save(data/'receipts.json',receipts)
    timeline=[json.loads(l)for l in(old/'timeline.jsonl').read_text().splitlines()]
    seal=next(e['seal']for e in timeline if e['kind']=='sealed')
    from mdb.forecast_transaction import load_forecast_tree
    frozen=load_forecast_tree(old/'transaction',read(old/'transaction/forecasts/response_program.json')['predictions'])
    summary=dict(schema='mdb-response-derived-analysis-view-v1',counts=counts,entries=dict(blocks=485,target_blocks=5),seal=seal,cache_seconds=dict(frozen['cache_seconds']),worker_seconds=None,
        scope='Selected complete10-image comparison across two runs. Not a successful reclassification of the original timeout. Costs below retain interrupted work.')
    save(data/'summary.json',summary);save(view/'report.json',dict(report=summary))
    durable_old=[read(p)for p in(old/'store/receipts').glob('*.json')]
    actual=dict(model_loads=2,completed_images=10,durable_receipts=len(durable_old)+new_summary['counts']['receipts'],
        durable_native_transitions=sum(r.get('steps',1)for r in durable_old if r['operation']=='continue')+new_summary['counts']['native_transitions'],
        durable_acts=sum(r['operation']!='continue'for r in durable_old)+new_summary['counts']['acts'],
        scope='Durable receipt lower bound; an operation interrupted before receipt persistence cannot be reconstructed. Includes abandoned base.predicted prefix.')
    provenance=dict(schema='mdb-response-derived-analysis-view-v1',partial=str(partial.resolve()),continuation=str(continuation.resolve()),original_state='timeout',continuation_custody=custody,
        original_forensic_custody=forensic,selected_complete_comparison_counts=counts,actual_attempt_cost=actual,origins=origins,source_job_records=[str(partial/'job.json'),str(continuation/'job.json')],new_model_calls=0)
    save(view/'analysis-view.json',provenance)
    return provenance
