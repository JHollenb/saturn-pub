"""Join saved operator diagnostics to the audited full native stage trace."""
import argparse
import json
import runpy
from pathlib import Path

import numpy as np
from mdb_lanes.edit_search import file_digest
from mdb_lanes.tensor_archive import load_tensor_archive
from mdb_lanes.parameter_artifacts import validate_manifest


def verify(study,output):
    here=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());metric=runpy.run_path(str(here/'analyze_flux_precision.py'))['error'];custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify']
    audit=read(study/'audit-v1.json');equal=read(study/'equal-input-rows-v1.json');source=study/'collection-01/extracted/binding-minimal-result';native=read(source/'native.json')
    if audit['mechanics_status']!='verified'or equal['base_audit_sha256']!=file_digest(study/'audit-v1.json'):raise ValueError('base evidence join differs')
    def checked(run):
        proof=custody(run);data=run/'extracted/binding-minimal-result';summary=read(data/'summary.json');request=read(run/'request.json')
        if summary!=read(run/'report.json')['report']or read(data/'plan.json')!=request['config']:raise ValueError('report/config differs')
        for name,digest in request['source_hashes'].items():
            if file_digest(run/'payload'/name)!=digest:raise ValueError('diagnostic source differs')
        for name,digest in read(run/'payload/payload-seal.json')['wheels'].items():
            if file_digest(run/'payload'/name)!=digest:raise ValueError('diagnostic wheel differs')
        for name,digest in request['config']['inputs'].items():
            if file_digest(source/name)!=digest:raise ValueError('native operand/source join differs')
        return proof,data,summary,request
    def compare(a,b):
        for key in('exact','different','scalars','max_abs'):
            if a[key]!=b[key]:raise ValueError('reported metric count/extreme differs')
        # Mechanics only: equivalent float64 aggregate reductions across BLAS
        # implementations. This tolerance has no scientific acceptance role.
        for key in('rms','reference_rms','relative_rms','cosine'):
            if a[key]is None or b[key]is None:
                if a[key]!=b[key]:raise ValueError('undefined metric differs')
            elif not np.isclose(a[key],b[key],rtol=1e-12,atol=1e-15):raise ValueError('recomputed aggregate differs')
    a_proof,ad,attention,ar=checked(study/'attention/collection-01');a_values={};a_actual={}
    if any(attention[k]!=0 for k in('model_loads','native_forwards','native_transitions'))or attention['source_summary_sha256']!=file_digest(source/'summary.json'):raise ValueError('attention scope differs')
    for case,row in attention['reports'].items():
        path=ad/row['path'];v=np.load(path,allow_pickle=False)
        if file_digest(path)!=row['sha256']or list(v.shape)!=row['shape']or str(v.dtype)!=row['dtype']:raise ValueError('attention result framing differs')
        n=load_tensor_archive(source,native[case]['stages']['attention.raw']);compare(metric(v,n),row['error']);a_values[case]=v;a_actual[case]=n
    compare(metric(a_values['half.image1']-a_values['half'],a_actual['half.image1']-a_actual['half']),attention['receiver_contrast'])
    g_proof,gd,gemm,gr=checked(study/'gemm/collection-01');g_values={};g_actual={}
    if gemm['counts']!=dict(full_model_loads=0,learned_weight_sets=1,native_linear_primitives=8,native_block_forwards=0,image_futures=0):raise ValueError('primitive census differs')
    if gr['config']['source_audit_sha256']!=file_digest(study/'audit-v1.json'):raise ValueError('primitive audit lineage differs')
    manifest=validate_manifest(read(gd/'parameter-artifacts.json'));full=validate_manifest(read(source/'parameter-artifacts.json'))
    if any(entry not in full['artifacts']for entry in manifest['artifacts'])or file_digest(gd/'retrieve_weights.py')!=gr['source_hashes']['retrieve_weights.py']:raise ValueError('weight recipes differ')
    events=[json.loads(line[len('MDB_GEMM '):])for line in(gd.parents[1]/'scheduler-metadata.log').read_text().splitlines()if line.startswith('MDB_GEMM ')]
    if [r['calls']for r in events]!=list(range(1,9))or len({(r['case'],r['policy'])for r in events})!=8:raise ValueError('primitive progress census differs')
    for case,row in gemm['reports'].items():
        stages=native[case]['stages'];n=load_tensor_archive(source,stages['image.q_projection']).reshape(1,1024,3072);g_actual[case]=n
        ref=load_tensor_archive(gd,row['reference']['record']);compare(metric(ref,n),row['reference']['vs_native']);g_values[case,'reference']=ref
        for name,values in row['policies'].items():
            v=load_tensor_archive(gd,values['record']);compare(metric(v,n),values['vs_native']);compare(metric(v,ref),values['vs_reference']);g_values[case,name]=v
    for name,values in gemm['receiver_contrast'].items():compare(metric(g_values['half.image1',name]-g_values['half',name],g_actual['half.image1']-g_actual['half']),values)
    result=dict(schema='mdb-precision-diagnostic-verification-v1',mechanics_status='verified',terminal_status='unassessed',
        first_large_discrepancy='QKV affine numerical contract; unchanged full input rows retain0.23-0.25percent relative error',
        source_seal=read(source/'summary.json')['seal'],main_counts=audit['counts'],
        attention=dict(custody=a_proof,endpoint_relative_rms={case:row['error']['relative_rms']for case,row in attention['reports'].items()},receiver_relative_rms=attention['receiver_contrast']['relative_rms'],native_model_calls=0),
        gemm=dict(custody=g_proof,counts=gemm['counts'],reduced_matches_native={case:row['policies']['bf16-reduced']['vs_native']['exact']for case,row in gemm['reports'].items()},
            gpu_fp64_matches_numpy_reference={case:row['policies']['fp64']['vs_reference']['exact']for case,row in gemm['reports'].items()},original_reduced_precision=gemm['original_reduced_precision'],
            reduced_vs_full_accumulation={case:row['policies']['bf16-full']for case,row in gemm['reports'].items()},receiver_contrast=gemm['receiver_contrast']),
        limitations=['Historical receiver inputs and prior complete endpoints; new intermediate ports were unobserved before the two-block assay.',
                     'Native-operand restorations and matrix-policy tests are post-hoc diagnostics, not independent predictions.',
                     'No corrected complete-block own-state prediction, fresh-dose confirmation, new semantic edit or end-to-end speedup is established.',
                     'The BF16 policy flag is causally isolated for this Q projection/geometry. Exact internal CUDA tile/reduction sequence and attention kernel emulation remain unresolved.'],
        evidence={name:file_digest(study/name)for name in('audit-v1.json','equal-input-rows-v1.json','historical-inspection-v1.json','attempt-cost-v1.json')},verifier_source_sha256=file_digest(Path(__file__)),new_native_calls_in_verification=0)
    with output.open('x')as p:json.dump(result,p,indent=2,sort_keys=True);p.write('\n')
    return dict(mechanics_status='verified',reduced_matches_native=result['gemm']['reduced_matches_native'],fp64_matches_reference=result['gemm']['gpu_fp64_matches_numpy_reference'],attention_receiver_error=result['attention']['receiver_relative_rms'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(verify(a.study.resolve(),a.output.resolve()),sort_keys=True))
