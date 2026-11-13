"""Offline closure of the audited fresh receiver's compact projection and remote evidence."""
import argparse
import hashlib
import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path


def verify(study, output):
    from mdb.forecast_transaction import verify_transaction
    from mdb.equation_comparison import EquationCatalog
    from mdb.computation_atlas import ComputationAtlas
    here=Path(__file__).resolve().parent; h=runpy.run_path(str(here/'flux_family_custody.py'))
    read=lambda p:json.loads(p.read_text()); sha=lambda p:h['_sha'](p)[0]
    pointer=read(study/'native-remote-pointer-v1.json'); record=pointer['record']
    if hashlib.sha256((json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()).hexdigest()!=pointer['record_sha256']:raise ValueError('remote record digest differs')
    request=h['_request'](study/'run-02'); jobs={kind:read(study/(kind+'-job-v1.json')) for kind in ('native','audit')}
    if jobs['native']['state']!='succeeded' or jobs['native']['job_id']!=record['verification']['job_id'] or request['request_sha256']!=record['verification']['request_sha256']:raise ValueError('native job/request differs')
    collection=study/'audit/collection-01'; custody=h['verify'](collection); data=collection/'extracted/binding-minimal-result'
    audit,summary,projection=[read(data/name) for name in ('audit-v1.json','summary.json','projection.json')]
    if jobs['audit']['state']!='succeeded' or jobs['audit']['job_id']!=custody['job_id'] or audit['custody']!=record['verification'] or summary['scientific_custody']!=record['verification'] or read(data/'remote-custody.json')!=record or sha(data/'remote-custody.json')!=pointer['record_sha256']:raise ValueError('independent scientific custody chain differs')
    if audit['mechanics_status']!='verified' or sha(data/'audit-v1.json')!=summary['audit_sha256'] or sha(data/'projection.json')!=summary['projection_sha256'] or sha(collection/'payload/analyze_flux_fresh_mlp_future.py')!=audit['auditor_source_sha256']:raise ValueError('audit/projection/source differs')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('audit wheel differs')
    for name,item in projection['copied'].items():
        if sha(data/name)!=item['sha256'] or (data/name).stat().st_size!=item['bytes']:raise ValueError('projection copy differs')
        if name.startswith('native-proof/') and record['files'][name[len('native-proof/'):]]['sha256']!=item['sha256']:raise ValueError('projected original member differs')
        if name.startswith('images/') and record['files'][name[len('images/'):]]['sha256']!=item['sha256']:raise ValueError('original PNG projection differs')
    proof=data/'native-proof'; native_summary=read(proof/'summary.json')
    if read(proof/'request.json')!=request or sha(proof/'summary.json')!=record['summary_sha256']:raise ValueError('native proof differs')
    for name in ('write','image'):
        checked=verify_transaction(proof/(name+'-transaction'),expected_seal_sha256=native_summary[name+'_seal']['seal_sha256'])
        if not checked['verified'] or checked['phase']!='closed':raise ValueError('projected transaction differs')
    expected=dict(loads=1,encodes=0,noise_draws=1,native_transitions=538,acts=15,receipts=37,images=9)
    if audit['counts']!=expected or native_summary['counts']!=expected or audit['independent_forecast_field_comparisons']!=72 or summary['counts']!=dict(model_loads=0,native_calls=0,scientific_images=9,independent_forecast_fields=72):raise ValueError('physical/recomputation census differs')
    xml=ET.parse(study/'mechanics-v3.xml').getroot(); suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=8 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped')):raise ValueError('installed instrument tests incomplete')
    catalog=EquationCatalog(data/'index-v1/catalog'); atlas=ComputationAtlas(data/'index-v1/atlas'); rows=catalog.find(); exact=abstain=0
    for row in rows:
        ctx=row['tested_context'];exact+=len(atlas.find(applicable_to=ctx));abstain+=not bool(atlas.find(applicable_to={**ctx,'seed':81120}));abstain+=not bool(atlas.find(applicable_to={**ctx,'clock':dict(step=2,joint=2)}))
        if ctx['frontier']['kind']!='fresh-receiver-shared-calibration':raise ValueError('calibrated transfer scope lost')
    if len(rows)!=4 or exact!=4 or abstain!=8:raise ValueError('atlas occurrence/applicability census differs')
    observations=dict(primary=audit['primary'],image={name:{mode:{key:values[key] for key in ('endpoint','response','baseline')} for mode,values in modes.items()} for name,modes in audit['outcomes'].items()},color_ordering=audit['color_ordering'])
    result=dict(schema='mdb-fresh-mlp-study-verification-v1',mechanics_status='verified',instrument_status=audit['instrument_status'],terminal_status='unassessed',remote_scientific_custody=record['verification'],local_projection_custody=custody,
        independent_forecast_fields=72,installed_mechanics_tests=8,counts=expected,atlas=dict(occurrences=4,exact_queries=exact,changed_context_abstentions=abstain),observations=observations,
        jobs={kind:dict(job_id=job['job_id'],elapsed_s=job['result']['elapsed_s'],peak_rss_mb=job['result']['peak_rss_mb'],peak_vram_mb=job['result']['peak_vram_mb']) for kind,job in jobs.items()},
        storage=dict(remote_scientific_bytes=sum(item['bytes'] for item in record['files'].values()),remote_array_bytes=sum(item['bytes'] for name,item in record['files'].items() if name.startswith('arrays/')),unique_arrays=audit['unique_arrays'],local_scientific_projection_bytes=sum(p.stat().st_size for p in data.rglob('*') if p.is_file()),local_archive_bytes=custody['archive_bytes']),
        scope='Source-assisted historical two-row intervention on one fresh calibrated receiver. Full remote custody and separate CPU forecast replay; exact named state-to-consumer receipt joins. Numerical/visual countertrends retained. No newly discovered semantic address, calibration-free portability, independent CUDA arithmetic or inference speed claim.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',independent_forecast_fields=72,atlas_occurrences=4,primary=audit['primary'],storage=result['storage'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(verify(a.study.resolve(),a.output.resolve()),sort_keys=True))
