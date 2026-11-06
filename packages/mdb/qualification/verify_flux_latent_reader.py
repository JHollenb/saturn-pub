"""Additional saved-only clock, lineage and intentional-prefix-refusal checks."""
import argparse
import collections
import json
import runpy
from pathlib import Path


def verify(root,output):
    h=runpy.run_path(str(Path(__file__).with_name('flux_source_receiver.py')))
    read=h['read'];cfg=read(root/'plan.json')
    receipts=[json.loads(x) for x in (root/'receipts.jsonl').read_text().splitlines()]
    census=collections.Counter((x['kind'],x['receipt'].get('status'),x['receipt']['operation']) for x in receipts)
    if census!={('native',None,'continue'):440,('latent-act',None,'apply'):8,('expected-prefix-stop','refused','continue'):4}:
        raise ValueError('receipt operation/status census differs')
    for row in receipts:
        r=row['receipt']
        if row['kind']=='expected-prefix-stop' and (r['parent']!=r['result'] or r['parent']!=cfg['projects'][row['arm']] or r['error']!='PrefixStop'):
            raise ValueError('prefix rollback identity differs')
    def point(local):
        if local<0:phase,index,clock,step='project',None,0,3
        elif local<5:phase,index,clock,step='joint',local,local,3
        elif local<25:phase,index,clock,step='single',local-5,local-5,3
        elif local==25:phase,index,clock,step='readout',None,20,3
        elif local==26:phase,index,clock,step='scheduler',None,20,3
        else:phase,index,clock,step='project',None,0,4
        operation={'project':'native_input_projection','readout':'norm_and_projection','scheduler':'flow_scheduler_commit'}.get(phase,f'{phase}_block.{index}')
        return dict(family='flux2-klein',logical_step=step,phase=phase,index=index,local_clock=clock,edge='before',operator=phase,next_operation=operation)
    count=0
    for group in ('closure','readers'):
        for key,row in read(root/(group+'.json')).items():
            native=[x['receipt'] for x in receipts if x['arm']==key and x['kind']=='native']
            first=0 if group=='closure' else 1
            if [x['local'] for x in row['records']]!=list(range(first,28)):raise ValueError('missing native clocks')
            prior=row['input_cut'] if group=='closure' else cfg['projects'][key.split('.')[0]+'.base']
            for record,receipt in zip(row['records'],native,strict=True):
                local=record['local'];expected=point(local)
                if record['point']!=expected or receipt['execution_point']!=expected or receipt['result']!=record['cut'] or receipt['parent']!=prior:
                    raise ValueError('native receipt/clock/lineage join differs')
                transitions=receipt['transitions']
                if receipt['steps']!=1 or len(transitions)!=1 or transitions[0]['before']!=point(local-1) or transitions[0]['after']!=expected:
                    raise ValueError('actual native transition differs')
                prior=receipt['result'];count+=1
    result=dict(mechanics_status='verified',native_clocks=count,successful_receipts=448,intentional_prefix_refusals=4,
        unique_receipt_seals=len({x['receipt']['fingerprint'] for x in receipts}),new_model_calls=0,verifier_sha256=h['sha'](Path(__file__)))
    h['save'](output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.root.resolve(),a.output.resolve()),sort_keys=True))
