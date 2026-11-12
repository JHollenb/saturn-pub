"""Independent saved-state audit of actual prefix bypass and native closure."""
import argparse
import json
import runpy
from pathlib import Path
from mdb.edit_search import file_digest as sha
from mdb.tensor_archive import load_tensor_archive


def analyze(collection,output):
    here=Path(__file__).resolve().parent; read=lambda p:json.loads(p.read_text())
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](collection)
    data=collection/'extracted/binding-minimal-result'; request=read(collection/'request.json'); summary=read(data/'summary.json'); plan=read(data/'plan.json')
    if summary!=read(collection/'report.json')['report'] or plan!=request['config']: raise ValueError('request/report mismatch')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest: raise ValueError('worker source changed')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest: raise ValueError('runtime wheel changed')
    for name,item in plan['input_files'].items():
        if sha(data/'source'/name)!=item['sha256'] or (data/'source'/name).stat().st_size!=item['bytes']: raise ValueError('input copy mismatch')
    expected=dict(model_loads=1,block_entries=4,incremental_blocks=2,full_native_blocks=2,skipped_image_affines=6,skipped_image_layer_norms=2,skipped_image_qk_norms=4,acts=0,receipts=4,image_futures=0)
    if summary['counts']!=expected: raise ValueError('physical work census mismatch')
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py')); audit=typed['audit_store']
    stores={key:audit(data/key) for key in ('store','source/targets/store','source/prefixes/store')}
    receipts=read(data/'receipts.json'); reports=summary['reports']; names=['norm1','to_q','to_k','to_v','norm_q','norm_k']
    if len(reports)!=4 or len(receipts)!=4 or summary['module_entries']!={key:4 for key in names}: raise ValueError('module/receipt census mismatch')
    hits={key:0 for key in names}; rows=[]
    for spec in plan['cases']:
        for record in spec['stages'].values(): load_tensor_archive(data/'source/prefixes',record)
        prefix=stores['source/prefixes/store']['cuts'][spec['prefix_entry']]; target_entry=stores['source/targets/store']['cuts'][spec['entry']]
        fields=lambda p:{v['key']['value']:v['value'] for v in p['payload']['items']}
        left,right=fields(prefix),fields(target_entry)
        if prefix['execution_point']!=target_entry['execution_point'] or any(left[k]!=right[k] for k in left if k!='text'): raise ValueError('prefix applicability mismatch')
        for mode in ('incremental','full'):
            index=len(rows); row=reports[index]; receipt=receipts[index]
            if row['case']!=spec['id'] or row['mode']!=mode or row['entry']!=spec['entry'] or row['exit']!=spec['exit'] or row['archived_target']!=spec['exit']: raise ValueError('comparison identity mismatch')
            if not row['full_payload_exact'] or stores['store']['cuts'][row['exit']]!=stores['source/targets/store']['cuts'][spec['exit']]: raise ValueError('complete cut differs')
            if receipt['operation']!='continue' or receipt['steps']!=1 or receipt['parent']!=spec['entry'] or receipt['result']!=spec['exit'] or receipt!=read(data/'store/receipts'/(receipt['fingerprint']+'.json')): raise ValueError('receipt closure mismatch')
            expected_hits=names if mode=='incremental' else []
            if [hit['module'] for hit in row['cache_hits']]!=expected_hits: raise ValueError('cache hit schedule mismatch')
            for hit in row['cache_hits']:
                if hit['original_forward_executed'] is not False or hit['input_check_seconds']<0: raise ValueError('cache body evidence mismatch')
                hits[hit['module']]+=1
            rows.append(dict(case=row['case'],mode=mode,complete_cut_exact=True,skipped_image_affines=row['skipped_image_affines'],navigation_step_seconds=row['navigation_step_seconds'],input_check_seconds=sum(hit['input_check_seconds'] for hit in row['cache_hits'])))
    bodies={key:summary['module_entries'][key]-hits[key] for key in names}
    if hits!={key:2 for key in names} or bodies!=summary['original_module_bodies_executed']: raise ValueError('executed/skipped census mismatch')
    result=dict(schema='mdb-incremental-native-audit-v1',mechanics_status='verified',instrument_status='complete archived/current/incremental typed-cut equality; body bypass supported by source and focused mechanics tests',terminal_status='unassessed',
        custody=custody,counts=expected,cache_hits=hits,original_bodies_executed_by_census=bodies,rows=rows,
        auditor_source_sha256=sha(Path(__file__)),worker_seconds=summary['worker_seconds'],new_native_calls_in_audit=0,
        scope='Two known internal states, six manually selected image-prefix modules. Exact skip qualification with native affected-state execution. Fixed-order n2 navigation timings include equality-check synchronization and receipt overhead; no established speedup, automatic dependency inference or zero-native forecasting.')
    with output.open('x') as f: json.dump(result,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
    return dict(mechanics_status='verified',counts=expected,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('collection',type=Path); p.add_argument('output',type=Path); a=p.parse_args(); print(json.dumps(analyze(a.collection.resolve(),a.output.resolve()),sort_keys=True))
