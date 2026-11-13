"""Isolate affine arithmetic using saved rows with bit-identical full inputs."""
import argparse
import json
import runpy
from pathlib import Path

import numpy as np
from mdb_lanes.edit_search import file_digest
from mdb_lanes.tensor_archive import load_tensor_archive
from mdb_lanes.parameter_artifacts import validate_manifest


def inspect(study,output):
    read=lambda p:json.loads(p.read_text());data=study/'collection-01/extracted/binding-minimal-result';audit=read(study/'audit-v1.json')
    if audit['mechanics_status']!='verified':raise ValueError('complete stage audit required')
    f=read(data/'forecasts.json');n=read(data/'native.json');manifest=validate_manifest(read(data/'parameter-artifacts.json'))
    metric=runpy.run_path(str(Path(__file__).with_name('analyze_flux_precision.py')))['error'];reports={}
    for case in('half','half.image1'):
        reports[case]={}
        for role in('image','text'):
            p=f[case]['legacy']['stages'];actual=n[case]['stages']
            x,y=[load_tensor_archive(data,t[role+'.modulated'])for t in(p,actual)]
            same=np.all(x==y,axis=-1)[0]
            projections={}
            for label in('q','k','v'):
                key=role+'.'+label+'_projection'
                if np.count_nonzero(same):projections[label]=metric(load_tensor_archive(data,p[key])[:,same],load_tensor_archive(data,actual[key])[:,same])
            reports[case][role]=dict(equal_input_rows=int(np.count_nonzero(same)),total_rows=int(same.size),operator='row-local affine projection with checkpoint-identical parameters',projection_errors=projections)
    result=dict(schema='mdb-identical-row-affine-diagnostic-v1',mechanics_status='verified',terminal_status='unassessed',reports=reports,new_native_calls=0,new_affine_evaluations=0,
        base_audit_sha256=file_digest(study/'audit-v1.json'),inspector_sha256=file_digest(Path(__file__)),parameter_manifest_sha256=file_digest(data/'parameter-artifacts.json'),parameter_artifacts=len(manifest['artifacts']),
        scope='Post-hoc subset of already calculated full-shape projections. Selected rows have bit-identical complete modulated operands. Native parameters were byte-compared to the pinned checkpoint in the source-sealed worker. Different outputs on these rows cannot be attributed to upstream row differences; this isolates the affine numerical contract, without identifying its exact kernel reduction policy.')
    with output.open('x')as p:json.dump(result,p,indent=2,sort_keys=True);p.write('\n')
    return dict(reports=reports,new_native_calls=0,new_affine_evaluations=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    print(json.dumps(inspect(a.study.resolve(),a.output.resolve()),sort_keys=True))
