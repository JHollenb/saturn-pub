"""Offline compact-projection closure for the audited native consumer trace."""
import argparse
import hashlib
import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path


def verify(study,output):
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_family_custody.py'));read=lambda p:json.loads(p.read_text());sha=lambda p:h['_sha'](p)[0]
    pointer=read(study/'native-remote-pointer-v1.json');record=pointer['record'];request=h['_request'](study/'run-02')
    if hashlib.sha256((json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()).hexdigest()!=pointer['record_sha256']:raise ValueError('remote record digest differs')
    jobs={kind:read(study/(kind+'-job-v1.json')) for kind in ('native','audit')}
    if jobs['native']['state']!='succeeded' or jobs['native']['job_id']!=record['verification']['job_id'] or request['request_sha256']!=record['verification']['request_sha256']:raise ValueError('native job/request differs')
    collection=study/'audit/collection-01';custody=h['verify'](collection);data=collection/'extracted/binding-minimal-result'
    audit,summary,projection=[read(data/name) for name in ('audit-v1.json','summary.json','projection.json')]
    if (jobs['audit']['state']!='succeeded' or jobs['audit']['job_id']!=custody['job_id'] or audit['custody']!=record['verification'] or summary['scientific_custody']!=record['verification']
            or read(data/'remote-custody.json')!=record or sha(data/'remote-custody.json')!=pointer['record_sha256']):raise ValueError('independent scientific custody chain differs')
    if audit['mechanics_status']!='verified' or sha(data/'audit-v1.json')!=summary['audit_sha256'] or sha(data/'projection.json')!=summary['projection_sha256'] or sha(collection/'payload/analyze_flux_consumer_trace.py')!=audit['auditor_source_sha256']:raise ValueError('audit/projection/source differs')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('audit wheel differs')
    for name,item in projection['copied'].items():
        if sha(data/name)!=item['sha256'] or (data/name).stat().st_size!=item['bytes']:raise ValueError('projection copy differs')
        if name.startswith('native-proof/') and record['files'][name[len('native-proof/'):]]['sha256']!=item['sha256']:raise ValueError('original projected member differs')
    proof=data/'native-proof';native_summary=read(proof/'summary.json')
    if read(proof/'request.json')!=request or sha(proof/'summary.json')!=record['summary_sha256']:raise ValueError('native proof differs')
    expected=runpy.run_path(str(study/'run-02/payload/flux_consumer_trace.py'))['counts']()
    if audit['counts']!=expected or native_summary['counts']!=expected or audit['measurement_counts']!=dict(observations=228,finite_write_accounts=208,euler_accounts=4) or summary['counts']!=dict(model_loads=0,native_calls=0,**audit['measurement_counts']):raise ValueError('physical/independent measurement census differs')
    lines=[json.loads(x) for x in (data/'observations.jsonl').read_text().splitlines()]
    if len(lines)!=228 or any(x['kind']!='observation' or x['source_job']!=jobs['native']['job_id'] for x in lines):raise ValueError('observation stream differs')
    xml=ET.parse(study/'mechanics-v5.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=14 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped')):raise ValueError('installed instrument tests incomplete')
    result=dict(schema='mdb-consumer-trace-study-verification-v1',mechanics_status='verified',instrument_status=audit['instrument_status'],terminal_status='unassessed',remote_scientific_custody=record['verification'],local_projection_custody=custody,
        installed_mechanics_tests=14,counts=expected,measurement_counts=audit['measurement_counts'],replay_status=audit['replay_status'],first_observed_reversals=summary['first_observed_reversals'],
        jobs={kind:dict(job_id=job['job_id'],elapsed_s=job['result']['elapsed_s'],peak_rss_mb=job['result']['peak_rss_mb'],peak_vram_mb=job['result']['peak_vram_mb']) for kind,job in jobs.items()},
        storage=dict(remote_scientific_bytes=sum(item['bytes'] for item in record['files'].values()),remote_typed_blob_bytes=sum(item['bytes'] for name,item in record['files'].items() if name.startswith('store/blobs/')),remote_intermediate_array_bytes=sum(item['bytes'] for name,item in record['files'].items() if name.startswith('arrays/')),
            unique_intermediate_arrays=audit['unique_intermediate_arrays'],local_scientific_projection_bytes=sum(p.stat().st_size for p in data.rglob('*') if p.is_file()),local_archive_bytes=custody['archive_bytes'],limits=record.get('limits')),
        scope='Retrospective component microscope of nine original observed futures. Full remote custody/typed state/source/receipt/parameter/complete-commit joins audited in a separate CPU process. This local check verifies its exact compact projection; it does not locally hydrate or reread the remote scientific tensor tree. Signed countertrends remain observations, not gates.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',instrument_status=result['instrument_status'],measurement_counts=result['measurement_counts'],storage=result['storage'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(verify(a.study.resolve(),a.output.resolve()),sort_keys=True))
