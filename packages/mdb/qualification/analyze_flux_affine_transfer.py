"""Join retained samples to original native operands and numerical reports."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from mdb.edit_search import file_digest as sha
from mdb.tensor_archive import load_tensor_archive


def analyze(collection,precision,output):
    here=Path(__file__).resolve().parent; read=lambda p:json.loads(p.read_text()); custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify']
    receipts=dict(transfer=custody(collection),precision=custody(precision)); data=collection/'extracted/binding-minimal-result'; source=precision/'extracted/binding-minimal-result'
    request=read(collection/'request.json'); c=request['config']; summary=read(data/'summary.json'); ar=runpy.run_path(str(here/'flux_arithmetic_reference.py'))
    if c!=read(data/'plan.json') or summary!=read(collection/'report.json')['report']: raise ValueError('request/report differs')
    if sha(here/'flux_arithmetic_reference.py')!=request['source_hashes']['flux_arithmetic_reference.py']: raise ValueError('reference source differs')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest: raise ValueError('payload differs')
    for name,digest in c['inputs'].items():
        if sha(source/name)!=digest: raise ValueError('consumed native evidence differs')
    native=read(source/'native.json'); reports={}; files=set()
    def array(record):
        p=data/record['path']; value=np.load(p,allow_pickle=False)
        if sha(p)!=record['sha256'] or str(value.dtype)!=record['dtype'] or list(value.shape)!=record['shape']: raise ValueError('sample differs')
        files.add(record['path']); return value
    for case,rows in summary['reports'].items():
        reports[case]={}
        for key,row in rows.items():
            role=key.split('.',1)[0]; port=key.split('.',1)[1]
            target={'attn.add_q_proj':'q_projection','attn.add_k_proj':'k_projection','attn.add_v_proj':'v_projection','attn.to_q':'q_projection','attn.to_k':'k_projection','attn.to_v':'v_projection','attn.to_add_out':'attention','attn.to_out.0':'attention'}.get(port,'mlp_paired' if port.endswith('linear_in') else 'mlp')
            actual=load_tensor_archive(source,native[case]['stages'][role+'.'+target]); actual=actual.reshape(1,actual.shape[1],-1)[:,row['rows']][:,:,row['columns']]
            retained=array(row['samples']['native'])
            if not np.array_equal(actual,retained): raise ValueError('sample disconnected from original native return')
            reports[case][key]={}
            for method in ('policy','wide'):
                measured=ar['metrics'](array(row['samples']['predicted' if method=='policy' else method]),retained)
                for field,value in measured.items():
                    other=row[method][field]
                    if isinstance(value,float):
                        if not np.isclose(value,other,rtol=1e-12,atol=1e-12): raise ValueError('metric differs')
                    elif value!=other: raise ValueError('metric differs')
                reports[case][key][method]=measured
    result=dict(schema='mdb-affine-transfer-audit-v1',mechanics_status='verified',custody=receipts,policy=c['policy'],reports=reports,sample_files_verified=len(files),
        native_calls=0,source_sha256=sha(Path(__file__)),terminal_status='unassessed',scope='Independent custody and sample-to-native joins, all retained prediction metrics rebuilt. Numerical schedule calculation is source-qualified, not independently re-executed. No new states/native calls.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',samples=len(files))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','precision','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(analyze(a.collection.resolve(),a.precision.resolve(),a.output.resolve())))
