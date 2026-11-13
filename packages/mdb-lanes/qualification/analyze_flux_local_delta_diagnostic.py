"""Verify saved-case local responses and exact input joins without model calls."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from mdb_lanes.edit_search import file_digest as sha
from mdb_lanes.tensor_archive import load_tensor_archive
from mdb_lanes.response_closure import measure_error


def analyze(collection,precision,response,output):
    read=lambda p:json.loads(p.read_text()); here=Path(__file__).resolve().parent
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](collection); data=collection/'extracted/binding-minimal-result'
    request=read(collection/'request.json'); summary=read(data/'summary.json'); forecasts=read(data/'forecasts.json')
    if summary!=read(collection/'report.json')['report'] or read(data/'plan.json')!=request['config'] or any(summary[k]!=0 for k in ('model_loads','native_forwards','native_primitives')) or not summary['zero_edit_native_closure']: raise ValueError('development assay contract differs')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest: raise ValueError('source differs')
    source=precision/'collection-01/extracted/binding-minimal-result'; target=response/'collection-01/extracted/binding-minimal-result'
    for key,root in [('prefix',source),('targets',target)]:
        for name,item in request['config']['inputs'][key].items():
            if sha(root/name)!=item['sha256'] or (root/name).stat().st_size!=item['bytes']: raise ValueError('consumed source evidence differs')
    slot=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']; B=runpy.run_path(str(here/'flux_reentry_reference.py'))['bf16']
    sn=read(source/'native.json'); tn=read(target/'native.json'); plan=read(source/'plan.json')
    T0=slot(source/'source/store',plan['parents']['base'],'text'); T1=slot(source/'source/store',plan['parents']['edited'],'text'); T=T0.copy(); T[:,[4,11]]=B(T0[:,[4,11]].astype(float)+.375*(T1[:,[4,11]].astype(float)-T0[:,[4,11]]))
    joins={}
    for name,anchor in [('three_eighths','half'),('three_eighths.image1','half.image1')]:
        if not np.array_equal(T,slot(target/'store',tn[name]['entry'],'text')) or not np.array_equal(slot(source/'store',sn[anchor]['entry'],'image'),slot(target/'store',tn[name]['entry'],'image')): raise ValueError('dose/receiver input join differs')
        def cut(root,fp): return read(root/'cuts'/(fp+'.json'))
        a=cut(source/'store',sn[anchor]['entry']); b=cut(target/'store',tn[name]['entry'])
        if a['execution_point']!=b['execution_point'] or any(a['slots'][k]!=b['slots'][k] for k in a['slots'] if k not in ('image','text')): raise ValueError('receiving closure join differs')
        joins[name]=dict(receiving_closure_exact=True,text_construction_exact=True)
        for role in ('image','text'):
            record=forecasts[name]['fields'][role]['exit']; p=data/record['path']; value=np.load(p,allow_pickle=False)
            if sha(p)!=record['sha256'] or list(value.shape)!=record['shape'] or str(value.dtype)!=record['dtype']: raise ValueError('external forecast differs')
            actual=load_tensor_archive(target,tn[name]['fields'][role]); baseline=load_tensor_archive(source,sn[anchor]['stages'][role+'.exit'])
            measured={'endpoint':measure_error(value,actual),'dose_response':measure_error(value.astype(float)-baseline.astype(float),actual.astype(float)-baseline.astype(float))}
            for key,metrics in measured.items():
                for field,item in metrics.items():
                    expected=summary['reports'][name][role][key][field]
                    if item is None:
                        if expected is not None: raise ValueError('metric differs')
                    elif not np.isclose(item,expected,rtol=1e-12,atol=1e-15): raise ValueError('saved metric recomputation differs')
    result=dict(schema='mdb-local-delta-development-audit-v1',mechanics_status='verified',terminal_status='unassessed',custody=custody,input_joins=joins,
        reports=summary['reports'],factorial=summary['factorial'],new_native_calls=0,auditor_source_sha256=sha(Path(__file__)),
        scope='Known-case development diagnostic. Explicit historical native calibration; complete own-state candidate computed without Torch/native primitives. No fresh semantic result.')
    with output.open('x') as f: json.dump(result,f,indent=2,sort_keys=True); f.write('\n')
    return dict(mechanics_status='verified',input_joins=joins)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','precision','response','output'): p.add_argument(name,type=Path)
    a=p.parse_args(); print(json.dumps(analyze(*[getattr(a,n).resolve() for n in ('collection','precision','response','output')]),sort_keys=True))
