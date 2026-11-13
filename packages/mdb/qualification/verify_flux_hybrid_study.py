"""Offline closure of local projections and the authenticated remote scientific chain."""
import argparse
import hashlib
import json
import runpy
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


def verify(study,output):
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog
    from mdb.forecast_transaction import verify_transaction
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_family_custody.py'));read=lambda p:json.loads(p.read_text());sha=lambda p:h['_sha'](p)[0]
    pointer=read(study/'native-remote-pointer-v1.json');record=pointer['record'];canonical=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()
    if hashlib.sha256(canonical).hexdigest()!=pointer['record_sha256']:raise ValueError('remote pointer record hash differs')
    native_request=h['_request'](study/'run-02');jobs={kind:read(study/name) for kind,name in [('native','native-job-v1.json'),('audit','audit-job-v1.json'),('mlp','mlp-job-v1.json')]}
    if jobs['native']['state']!='succeeded' or jobs['native']['job_id']!=record['verification']['job_id'] or native_request['request_sha256']!=record['verification']['request_sha256']:raise ValueError('native job/request differs')
    collections={key:study/path for key,path in [('audit','audit/collection-01'),('mlp','mlp/collection-01')]};custody={key:h['verify'](root) for key,root in collections.items()}
    for key,root in collections.items():
        if jobs[key]['state']!='succeeded' or jobs[key]['job_id']!=custody[key]['job_id']:raise ValueError('audit/diagnostic job differs')
        for name,digest in read(root/'payload/payload-seal.json')['wheels'].items():
            if sha(root/'payload'/name)!=digest:raise ValueError('wheel differs')
    data=collections['audit']/'extracted/binding-minimal-result';mlp=collections['mlp']/'extracted/binding-minimal-result';audit=read(data/'audit-v1.json');summary=read(data/'summary.json');m=read(mlp/'audit-v1.json');ms=read(mlp/'summary.json')
    if audit['mechanics_status']!='verified' or m['mechanics_status']!='verified' or audit['custody']!=record['verification'] or ms['source_custody']!=record['verification'] or read(data/'remote-custody.json')!=record or sha(data/'remote-custody.json')!=pointer['record_sha256']:raise ValueError('full remote scientific custody chain differs')
    if summary['audit_sha256']!=sha(data/'audit-v1.json') or summary['scientific_custody']!=record['verification'] or sha(data/'native-proof/summary.json')!=record['summary_sha256']:raise ValueError('source scientific summary/audit differs')
    if audit['auditor_source_sha256']!=sha(collections['audit']/'payload/analyze_flux_hybrid_future.py') or m['auditor_source_sha256']!=sha(collections['mlp']/'payload/analyze_flux_hybrid_mlp.py'):raise ValueError('auditor source differs')
    projection=read(data/'projection.json')
    for name,item in projection['copied'].items():
        if sha(data/name)!=item['sha256'] or (data/name).stat().st_size!=item['bytes']:raise ValueError('compact projection differs')
        if name.startswith('native-proof/') and not name.startswith(('native-proof/write-transaction/','native-proof/image-transaction/')):
            if record['files'][name[len('native-proof/'):]]['sha256']!=item['sha256']:raise ValueError('native metadata differs from full source')
    tx={}
    for key in ('write','image'):
        tx[key]=verify_transaction(data/'native-proof'/(key+'-transaction'),expected_seal_sha256=read(data/'native-proof/summary.json')[key+'_seal']['seal_sha256'])
        if not tx[key]['verified'] or tx[key]['phase']!='closed':raise ValueError('copied transaction not verified')
    primary={mode:statistics.mean(row[mode]['image']['exit']['reference_response']['relative_rms'] for row in audit['numerical'].values()) for mode in ('control','hybrid')}
    if primary!=audit['primary']['mean_image_response_error']:raise ValueError('primary aggregation differs')
    if ms['counts']!=dict(model_loads=0,native_calls=0,images=0,mlp_calculations=36,conditional_outputs=12,owned_reconstructions=24) or len(m['owned_reconstructions'])!=24 or not all(m['owned_reconstructions'].values()):raise ValueError('subprogram reconstruction mechanics failed')
    bits=runpy.run_path(str(collections['mlp']/'payload/flux_attention_anchor.py'))['load_bits'];conditional=read(mlp/'conditional.json');occurrences=0
    for roles in [*conditional['anchors'].values(),*conditional['cases'].values()]:
        for item in roles.values():bits(mlp,item);occurrences+=1
    if occurrences!=12:raise ValueError('conditional output census differs')
    catalog=EquationCatalog(data/'index-v1/catalog');atlas=ComputationAtlas(data/'index-v1/atlas');rows=catalog.find();exact=0;abstentions=0
    for row in rows:
        context=row['tested_context'];exact+=len(atlas.find(applicable_to=context));abstentions+=not bool(atlas.find(applicable_to={**context,'clock':dict(step=2,joint=2)}));abstentions+=not bool(atlas.find(applicable_to={**context,'input_identity':'unknown'}))
    if len(rows)!=8 or exact!=8 or abstentions!=16:raise ValueError('atlas closure differs')
    test=ET.parse(study/'storage-mechanics/tests-v2.xml').getroot();suites=[test] if test.tag=='testsuite' else list(test.iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=9 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped')):raise ValueError('mechanics tests incomplete')
    result=dict(schema='mdb-hybrid-study-verification-v1',mechanics_status='verified',terminal_status='unassessed',local_custody=custody,remote_scientific_custody=record['verification'],remote_record_sha256=pointer['record_sha256'],transactions=tx,primary=primary,atlas=dict(programs=2,occurrences=8,exact_queries=exact,changed_context_abstentions=abstentions),conditional=dict(outputs=12,owned_exact=24,new_native_calls=0),installed_mechanics_tests=9,jobs={k:dict(job_id=v['job_id'],state=v['state'],result=v['result']) for k,v in jobs.items()},scope='Offline verification of complete local compact-projection/MLP collections and their bound full remote scientific audit. Full native tensors remain remote; this does not rerun the full remote verifier or claim a complete local native collection. All numerical accuracy/mixed/consumer counterexamples retained.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',primary=primary,conditional_outputs=12,owned_exact=24,atlas_occurrences=8)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(verify(a.study.resolve(),a.output.resolve()),sort_keys=True))
