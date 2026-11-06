"""Additional saved-only verification of actual exit clocks and receipt seals."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def verify(native,output):
    from saturn_pub.values import canonical
    h=runpy.run_path(str(Path(__file__).with_name('flux_source_receiver.py')))
    cfg=h['read'](native/'plan.json');checked=0
    for arm in h['arms']():
        row=h['read'](native/'rows'/(arm['id']+'.json'))
        for r in row['records']:
            logical,local=r['logical'],r['local']
            if local==0:phase,index='joint',0
            elif local<5:phase,index='joint',local
            elif local==5:phase,index='single',0
            elif local<25:phase,index='single',local-5
            elif local==25:phase,index='readout',None
            elif local==26:phase,index='scheduler',None
            else:phase,index='project',None;logical+=1
            operation={'project':'native_input_projection','readout':'norm_and_projection',
                'scheduler':'flow_scheduler_commit'}.get(phase,f'{phase}_block.{index}')
            expected=dict(family='flux2-klein',logical_step=logical,phase=phase,operator=phase,
                index=index,edge='before',next_operation=operation,
                local_clock=20 if phase in ('readout','scheduler') else (index or 0))
            if r['point']!=expected:raise ValueError('actual native exit clock differs: '+str((arm['id'],r['logical'],local)))
            checked+=1
    receipts=[json.loads(s)['receipt'] for s in (native/'receipts.jsonl').read_text().splitlines()]
    for receipt in receipts:
        if hashlib.sha256(canonical({k:v for k,v in receipt.items() if k!='fingerprint'})).hexdigest()!=receipt['fingerprint']:
            raise ValueError('receipt content seal differs')
    if checked!=350 or len(receipts)!=361:raise ValueError('incomplete native clock/receipt coverage')
    result=dict(schema='mdb-source-receiver-clock-receipt-verification-v1',mechanics_status='verified',
        actual_exit_clocks=checked,receipt_rows=len(receipts),unique_receipt_seals=len({r['fingerprint'] for r in receipts}),
        native_files_sha256=h['sha'](native/'files.json'),request_sha256=h['read'](native/'request.json')['request_sha256'],
        source_point=cfg['point'],new_model_calls=0,verifier_sha256=h['sha'](Path(__file__)),
        scope='Additional actual execution-point and receipt-content verification; no mathematical forecast recomputation.')
    h['save'](output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('native',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.native.resolve(),a.output.resolve()),sort_keys=True))
