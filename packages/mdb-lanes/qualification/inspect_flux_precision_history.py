"""Compare unsealed diagnostic calculations to explicitly known old endpoints."""
import argparse
import json
import runpy
from pathlib import Path

import numpy as np
from mdb_lanes.edit_search import file_digest
from mdb_lanes.tensor_archive import load_tensor_archive


def inspect(snapshot,history,output):
    here=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());data=snapshot/'extracted/binding-minimal-result';old=history/'collection-04/extracted/binding-minimal-result'
    f=read(data/'forecasts.json');config=read(data/'plan.json');rows={(r['id'],r['mode']):r for r in read(old/'rows.json')}
    r=runpy.run_path(str(here/'analyze_flux_precision.py'));metric=r['error']
    runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'))['audit_store'](data/'source/store')
    slot=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']
    def actual(name,role,key):
        target=rows[name,'native']['target']
        if target['exit']!=config['history'][name]['exit']:raise ValueError('historical endpoint identity differs')
        if key=='exit':return slot(data/'source/store',target['exit'],role)
        record=target['fields'][key+'_'+role];path=old/record['path']
        if file_digest(path)!=record['sha256']:raise ValueError('historical field bytes differ')
        return np.load(path,allow_pickle=False)
    reports={name:{mode:{role:{key:metric(load_tensor_archive(data,f[name][mode]['fields'][role][key]),actual(name,role,key))for key in('attention','mlp_input','mlp','exit')}for role in('image','text')}for mode in('legacy','staged')}for name in('half','half.image1')}
    contrast={mode:{role:{key:metric(load_tensor_archive(data,f['half.image1'][mode]['fields'][role][key])-load_tensor_archive(data,f['half'][mode]['fields'][role][key]),actual('half.image1',role,key)-actual('half',role,key))for key in('attention','mlp_input','mlp','exit')}for role in('image','text')}for mode in('legacy','staged')}
    result=dict(schema='mdb-historical-precision-inspection-v1',mechanics_status='verified',terminal_status='unassessed',reports=reports,receiver_contrast=contrast,new_native_calls=0,
        scope='Unsealed calculations from a failed pre-target diagnostic versus already known historical endpoints. This is post-hoc inspection, not a fresh prediction or the new internal-port assay.',
        provenance=dict(forensic_custody_sha256=file_digest(snapshot/'forensic-custody.json'),historical_verification_sha256=file_digest(history/'verification-v1.json'),inspector_sha256=file_digest(Path(__file__))))
    with output.open('x')as p:json.dump(result,p,indent=2,sort_keys=True);p.write('\n')
    return dict(endpoint_image_error={name:{mode:reports[name][mode]['image']['exit']['relative_rms']for mode in('legacy','staged')}for name in reports},receiver_exit_error={mode:{role:contrast[mode][role]['exit']['relative_rms']for role in('image','text')}for mode in contrast})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in('snapshot','history','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(inspect(*[getattr(a,n).resolve()for n in('snapshot','history','output')]),sort_keys=True))
