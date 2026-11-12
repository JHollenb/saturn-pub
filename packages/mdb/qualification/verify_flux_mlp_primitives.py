"""Offline closure of a compact primitive audit and bound remote raw custody."""
import argparse
import hashlib
import json
import math
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path


def verify(study,output):
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_family_custody.py'));read=lambda p:json.loads(p.read_text());sha=lambda p:h['_sha'](p)[0]
    pointer=read(study/'native-remote-pointer-v1.json');record=pointer['record'];canonical=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()
    if hashlib.sha256(canonical).hexdigest()!=pointer['record_sha256']:raise ValueError('remote record digest differs')
    native_request=h['_request'](study/'run-01');jobs={kind:read(study/(kind+'-job-v1.json')) for kind in ('native','audit')}
    if jobs['native']['state']!='succeeded' or jobs['native']['job_id']!=record['verification']['job_id'] or native_request['request_sha256']!=record['verification']['request_sha256']:raise ValueError('native request/job binding differs')
    collection=study/'audit/collection-01';custody=h['verify'](collection);data=collection/'extracted/binding-minimal-result';audit=read(data/'audit-v1.json');summary=read(data/'summary.json');projection=read(data/'projection.json')
    if jobs['audit']['state']!='succeeded' or jobs['audit']['job_id']!=custody['job_id'] or audit['custody']!=record['verification'] or summary['scientific_custody']!=record['verification'] or read(data/'remote-custody.json')!=record or sha(data/'remote-custody.json')!=pointer['record_sha256']:raise ValueError('independent remote audit chain differs')
    if audit['mechanics_status']!='verified' or audit['instrument_status']!='qualified' or audit['instrument_issues']:raise ValueError('primitive instrument unqualified')
    if sha(data/'audit-v1.json')!=summary['audit_sha256'] or sha(data/'projection.json')!=summary['projection_sha256'] or sha(collection/'payload/analyze_flux_mlp_primitives.py')!=audit['auditor_source_sha256']:raise ValueError('audit/source digest differs')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('installed wheel differs')
    for name,item in projection['copied'].items():
        if sha(data/name)!=item['sha256'] or (data/name).stat().st_size!=item['bytes']:raise ValueError('projection copy differs')
        if name.startswith('native-proof/') and record['files'][name[len('native-proof/'):]]['sha256']!=item['sha256']:raise ValueError('native metadata/source differs')
    if read(data/'native-proof/request.json')!=native_request or sha(data/'native-proof/summary.json')!=record['summary_sha256']:raise ValueError('native proof differs')
    expected=dict(mlp=2,linear=4,silu=2,product=2,full_models=0,blocks=0,attention=0,images=0)
    if audit['counts']!=expected or summary['counts']!=dict(model_loads=0,native_calls=0,images=0) or audit['independent_math_evaluations']!=12:raise ValueError('physical work census differs')
    census=dict(native_replays=0,owned_reconstructions=0,saved_math_fields=0,previous_conditional_joins=0)
    for role,check in audit['checks'].items():
        if check['native_output_parity'] is not True or check['worker_parity_agrees'] is not True:raise ValueError('native replay differs')
        census['native_replays']+=1
        for key,target in [('owned_reconstruction','owned_reconstructions'),('saved_math','saved_math_fields'),('previous_conditional','previous_conditional_joins')]:
            for value in check[key].values():
                if value is not True:raise ValueError('exact reconstruction instrument differs')
                census[target]+=1
        # Check the algebraic power account without imposing an accuracy gate.
        for measurement in audit['reports'][role]['ordered_output_space'].values():
            summed=sum(measurement['component_mse'].values())+sum(measurement['twice_signed_cross_moments'].values())
            if not math.isclose(summed,measurement['total_mse'],rel_tol=1e-12,abs_tol=1e-20) or measurement['identity_residual_rms']!=0:raise ValueError('output-space error identity differs')
    if census!=dict(native_replays=2,owned_reconstructions=8,saved_math_fields=24,previous_conditional_joins=4):raise ValueError('exact check census differs')
    catalog=EquationCatalog(data/'index-v1/catalog');atlas=ComputationAtlas(data/'index-v1/atlas');records=catalog.find();exact=0;abstain=0
    for row in records:
        context=row['tested_context'];exact+=len(atlas.find(applicable_to=context));abstain+=not bool(atlas.find(applicable_to={**context,'clock':dict(step=2,joint=2)}));abstain+=not bool(atlas.find(applicable_to={**context,'input_identity':'unknown'}))
        if context['frontier']['kind']!='post-hoc-native-operands':raise ValueError('diagnostic frontier lost')
    if len(records)!=10 or exact!=10 or abstain!=20:raise ValueError('catalog/applicability differs')
    test=ET.parse(study/'mechanics-v2.xml').getroot();suites=[test] if test.tag=='testsuite' else list(test.iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=4 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped')):raise ValueError('installed mechanics tests incomplete')
    result=dict(schema='mdb-mlp-primitives-study-verification-v1',mechanics_status='verified',instrument_status='qualified',terminal_status='unassessed',remote_scientific_custody=record['verification'],remote_record_sha256=pointer['record_sha256'],local_projection_custody=custody,checks=census,atlas=dict(supplied_laws=5,occurrences=10,exact_queries=exact,changed_context_abstentions=abstain),installed_mechanics_tests=4,counts=expected,jobs={kind:dict(job_id=job['job_id'],state=job['state'],result=job['result']) for kind,job in jobs.items()},scope='Offline proof of complete local audit/projection and its source-bound full remote primitive evidence. Native raw arrays remain on Beast; no complete-local-raw, fresh-prediction, new-image or speed claim. Numeric errors and signed cancellations are observations, not qualification gates.')
    with output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',instrument_status='qualified',checks=census,atlas_occurrences=10)


def verify_panel(study,output):
    h=runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')));read=lambda p:json.loads(p.read_text());sha=lambda p:h['_sha'](p)[0];root=study/'panel';pointer=read(root/'remote-pointer-v2.json');record=pointer['record'];request=h['_request'](root/'run-02');native_request=h['_request'](study/'run-01');job=read(root/'job-v2.json');report=read(root/'panel-report-v2.json');prior=read(study/'native-remote-pointer-v1.json')['record'];primitive=study/'audit/collection-01/extracted/binding-minimal-result/audit-v1.json'
    if hashlib.sha256((json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()).hexdigest()!=pointer['record_sha256']:raise ValueError('panel remote record hash differs')
    if job['state']!='succeeded' or job['job_id']!=record['verification']['job_id'] or request['request_sha256']!=record['verification']['request_sha256'] or report['request_sha256']!=request['request_sha256'] or report['job_id']!=job['job_id']:raise ValueError('panel source/job differs')
    encoder=record['recovery_encoder']
    if encoder['source_request_sha256']!=native_request['request_sha256'] or encoder['sha256']!=native_request['source_hashes']['flux_family_custody.py']:raise ValueError('explicit recovery encoder binding differs')
    if sha(root/'panel-report-v2.json')!=record['files']['panel-report.json']['sha256'] or report['source_custody']!=prior['verification'] or sha(primitive)!=report['primitive_audit_sha256']:raise ValueError('panel report/source custody differs')
    expected=dict(model_loads=0,native_calls=0,images=0,affine_calculations=32,swiglu_calculations=8,conditional_mlp_outputs=16)
    if report['counts']!=expected or report['instrument_status']!='qualified-control' or report['instrument_issues']:raise ValueError('panel work/control differs')
    checks=0;logical=0;paths=set()
    for role,values in report['checks'].items():
        for name,value in values.items():
            if value is not True:raise ValueError('anchored reconstruction differs')
            checks+=1
        for by_endpoint in report['records'][role].values():
            for fields in by_endpoint.values():
                for item in fields.values():
                    logical+=1;paths.add(item['path'])
                    if record['files'][item['path']]['sha256']!=item['sha256'] or item['encoding']!='bf16-bits-v1':raise ValueError('panel array reference differs')
        for power in report['reports'][role]['law_change'].values():
            if power['identity_residual_rms']!=0 or not math.isclose(sum(power['component_mse'].values())+sum(power['twice_signed_cross_moments'].values()),power['total_mse'],rel_tol=1e-12,abs_tol=1e-20):raise ValueError('panel signed power identity differs')
    if checks!=8 or logical!=32 or len(paths)!=report['unique_arrays']:raise ValueError('panel output/reference census differs')
    failed=read(root/'failed-job-v1.json');old=h['_request'](root/'run-01');lines=(root/'failed-log-prefix-v1.txt').read_text().splitlines();observed=[json.loads(line[len('MDB_MLP_AFFINE_PANEL '):]) for line in lines if line.startswith('MDB_MLP_AFFINE_PANEL ')]
    if failed['state']!='failed' or failed['job_id']!=read(root/'run-01/submission.json')['job_id'] or len(observed)!=2:raise ValueError('failed attempt not retained')
    for name in ('flux_reentry_reference.py','flux_receiver_norm_prediction.py','flux_arithmetic_reference.py'):
        if old['source_hashes'][name]!=request['source_hashes'][name]:raise ValueError('repair changed numerical law')
    for row in observed:
        current=report['reports'][row['role']]
        if row['program_response_relative']!={name:value['reference_response']['relative_rms'] for name,value in current['programs'].items()}:raise ValueError('export repair changed completed response measurements')
    xml=ET.parse(study/'instrument-mechanics-v3.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=19 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped')):raise ValueError('combined instrument tests incomplete')
    result=dict(schema='mdb-mlp-affine-panel-verification-v1',mechanics_status='verified',instrument_status='qualified-control',terminal_status='unassessed',remote_scientific_custody=record['verification'],source_native_custody=prior['verification'],counts=expected,owned_control_joins=checks,logical_arrays=logical,unique_arrays=len(paths),installed_instrument_tests=19,failed_attempt=dict(job_id=failed['job_id'],state=failed['state'],seconds=failed['result']['elapsed_s'],same_completed_response_measurements=True),scope='Full remote legacy custody plus locally authenticated report, source/control joins and count checks. Mathematical development results are from the source-qualified panel worker; no separate second recomputation of the new four-law panel. All predictions remain remote and can be retrieved exactly. No fresh-prediction/default/image/speed claim.')
    with output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',instrument_status='qualified-control',logical_arrays=logical,unique_arrays=len(paths),native_calls=0,installed_instrument_tests=19)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('output',type=Path);p.add_argument('--panel',action='store_true');a=p.parse_args();print(json.dumps((verify_panel if a.panel else verify)(a.study.resolve(),a.output.resolve()),sort_keys=True))
