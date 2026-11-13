"""Offline exact-projection closure for the J2 stream intervention study."""
import argparse
import hashlib
import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path


def verify(study,output):
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_family_custody.py'));read=lambda p:json.loads(p.read_text());sha=lambda p:h['_sha'](p)[0]
    pointer=read(study/'native-remote-pointer-v1.json');record=pointer['record'];request=h['_request'](study/'run-01');jobs={kind:read(study/(kind+'-job-v1.json')) for kind in ('native','audit')}
    if hashlib.sha256((json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()).hexdigest()!=pointer['record_sha256']:raise ValueError('native custody record differs')
    if jobs['native']['state']!='succeeded' or jobs['native']['job_id']!=record['verification']['job_id'] or request['request_sha256']!=record['verification']['request_sha256']:raise ValueError('native job/request differs')
    collection=study/'audit/collection-01';custody=h['verify'](collection);data=collection/'extracted/binding-minimal-result';audit,summary,projection=[read(data/n) for n in ('audit-v1.json','summary.json','projection.json')]
    if (jobs['audit']['state']!='succeeded' or jobs['audit']['job_id']!=custody['job_id'] or audit['custody']!=record['verification'] or summary['scientific_custody']!=record['verification']
            or read(data/'remote-custody.json')!=record or sha(data/'remote-custody.json')!=pointer['record_sha256']):raise ValueError('independent scientific custody chain differs')
    if audit['mechanics_status']!='verified' or sha(data/'audit-v1.json')!=summary['audit_sha256'] or sha(data/'projection.json')!=summary['projection_sha256'] or sha(collection/'payload/analyze_flux_stream_cross.py')!=audit['auditor_source_sha256']:raise ValueError('audit/source/projection differs')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('installed audit wheel differs')
    for name,item in projection['copied'].items():
        if sha(data/name)!=item['sha256'] or (data/name).stat().st_size!=item['bytes']:raise ValueError('compact projection copy differs')
        if name.startswith('native-proof/') and record['files'][name[len('native-proof/'):]]['sha256']!=item['sha256']:raise ValueError('projected native member differs')
    proof=data/'native-proof';native=read(proof/'summary.json');c=request['config']
    if read(proof/'request.json')!=request or sha(proof/'summary.json')!=record['summary_sha256']:raise ValueError('original native proof differs')
    expected=runpy.run_path(str(study/'run-01/payload/flux_stream_cross.py'))['counts']()
    if native['counts']!=expected or audit['counts']!=expected or audit['measurement_counts']!=dict(observations=64,factorial_accounts=16,finite_write_accounts=16) or summary['counts']!=dict(model_loads=0,native_calls=0,**audit['measurement_counts']):raise ValueError('work/measurement census differs')
    source=audit['source_verification']
    if source['inherited_audit_sha256']!=c['source_audit_sha256'] or source['inherited_record_sha256']!=c['source_record_sha256'] or source['inherited_full_custody']!=read(proof/'source-binding.json')['record']['verification']:raise ValueError('inherited source qualification differs')
    observations=[json.loads(x) for x in (data/'observations.jsonl').read_text().splitlines()]
    if len(observations)!=64 or any(x['source_job']!=jobs['native']['job_id'] for x in observations) or any(x['clock']['logical_step']!=(3 if x['boundary']=='latent' else 2) for x in observations):raise ValueError('observation/clock census differs')
    xml=ET.parse(study/'mechanics-v3.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=3 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped')):raise ValueError('instrument mechanics tests incomplete')
    result=dict(schema='mdb-stream-cross-study-verification-v1',mechanics_status='verified',instrument_status=audit['instrument_status'],terminal_status='unassessed',remote_scientific_custody=record['verification'],local_projection_custody=custody,
        inherited_source=dict(custody=source['inherited_full_custody'],record_sha256=c['source_record_sha256'],audit_sha256=c['source_audit_sha256'],consumed_files=len(source['files'])),shams=audit['shams'],installed_mechanics_tests=3,counts=expected,measurement_counts=audit['measurement_counts'],response_errors=summary['response_errors'],
        jobs={kind:dict(job_id=j['job_id'],elapsed_s=j['result']['elapsed_s'],peak_rss_mb=j['result']['peak_rss_mb'],peak_vram_mb=j['result']['peak_vram_mb']) for kind,j in jobs.items()},
        storage=dict(remote_scientific_bytes=sum(x['bytes'] for x in record['files'].values()),remote_typed_blob_bytes=sum(x['bytes'] for n,x in record['files'].items() if n.startswith('store/blobs/')),remote_array_bytes=sum(x['bytes'] for n,x in record['files'].items() if n.startswith('arrays/')),
            local_scientific_projection_bytes=sum(p.stat().st_size for p in data.rglob('*') if p.is_file()),local_archive_bytes=custody['archive_bytes'],native_archive_bytes=record['verification']['archive_bytes'],custody_seconds=record['seconds']),
        scope='Offline exact local projection of a separately audited typed intervention. Full current remote custody checked nearcompute; prior full source qualification inherited, every consumed source member freshly verified. No local hydration of the remote raw tensor tree, independent forecast, newsemanticaddress, image or speedup claim.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',instrument_status=result['instrument_status'],shams=result['shams'],measurement_counts=result['measurement_counts'],storage=result['storage'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(verify(a.study.resolve(),a.output.resolve()),sort_keys=True))
