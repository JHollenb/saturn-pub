"""Verify closure of the three saved numerical cuts, atlas and source snapshots."""
import argparse
import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path
from mdb_lanes.computation_atlas import ComputationAtlas
from mdb_lanes.edit_search import file_digest as sha


def verify(root):
    here=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'];panels={};references=[]
    for phase in ('boundary','full','receiver'):
        base=root if phase=='boundary' else root/phase;collection=base/'collection-01';checked=custody(collection);request=read(collection/'request.json');audit=read(base/'audit-v1.json');summary=read(collection/'extracted/binding-minimal-result/summary.json');job=read(collection/'job.json')
        if audit['mechanics_status']!='verified' or job['state']!='succeeded' or summary!=read(collection/'report.json')['report']:raise ValueError('incomplete panel')
        if any(audit['custody'][k]!=checked[k] for k in ('job_id','request_sha256','archive_sha256')):raise ValueError('audit custody mismatch')
        for name,digest in request['source_hashes'].items():
            if sha(collection/'payload'/name)!=digest:raise ValueError('payload source changed')
            if (here/name).exists() and name!='retrieve_weights.py' and sha(here/name)!=digest:raise ValueError('executed qualification source changed')
        if summary['counts']['native_calls']!=0 or summary['counts']['model_loads']!=0:raise ValueError('declared zero-native scope differs')
        panels[phase]=dict(custody=checked,counts=summary['counts'],worker_seconds=summary['seconds'],process_seconds=job['result']['elapsed_s'],peak_rss_mb=job['result']['peak_rss_mb'],peak_vram_mb=job['result']['peak_vram_mb'])
        references.extend([base/'audit-v1.json',collection/'request.json',collection/'report.json'])
    if read(root/'receiver/audit-v1.json')['source_full_audit_sha256']!=sha(root/'full/audit-v1.json') or read(root/'full/audit-v1.json')['source_attention_audit_sha256']!=sha(root/'audit-v1.json'):raise ValueError('cut chain disconnected')
    if read(root/'storage-verified-v2.json')['mechanics_status']!='verified':raise ValueError('storage unverified')
    atlas=ComputationAtlas(root/'index-v2/atlas');records=atlas.find()
    if len(records)!=24 or len({r['equation']['law_id'] for r in records})!=6:raise ValueError('atlas census differs')
    for record in records:
        context=record['equation']['tested_context']
        if len(atlas.find(applicable_to=context))!=1 or atlas.find(applicable_to={**context,'clock':dict(step=2,joint=2)}) or atlas.find(applicable_to={**context,'input_identity':'unknown'}):raise ValueError('applicability differs')
        for item in record['provenance']['artifacts']:
            if sha(item['path'])!=item['sha256']:raise ValueError('atlas provenance drift')
    suites=ET.parse(root/'mechanics-v3.xml').getroot().findall('testsuite')
    if sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites):raise ValueError('mechanics tests fail')
    offloads=sorted(root.glob('prior-*-transport-externalized-v1.json'));freed=0
    for p in offloads:
        value=read(p);freed+=value['freed_bytes']
        if value['freed_bytes']!=sum(r['bytes'] for r in value['records']):raise ValueError('transport census differs')
        for r in value['records']:
            marker=Path(r['local']).parent/'transport-externalized-v1.json'
            if read(marker)['sha256']!=sha(p):raise ValueError('offload receipt marker differs')
    references.extend([root/n for n in ('README.md','advice/astra-review.md','advice/results-review.md','mechanics-v3.xml','index-v2/provenance.json','index-v1/failure-note.json','storage-plan-v2.json','storage-verified-v2.json','storage-compacted-v1.json','storage-compacted-v2.json','evidence-coverage-v1.json')]);references.extend(offloads)
    names=('flux_attention_anchor.py','analyze_flux_attention_anchor.py','flux_attention_law_prediction.py','flux_attention_law_diagnostic.py','analyze_flux_attention_law.py','flux_receiver_norm_prediction.py','flux_receiver_norm_diagnostic.py','analyze_flux_receiver_norm.py','publish_flux_attention_laws.py','verify_flux_attention_law_study.py')
    result=dict(schema='mdb-attention-consumer-law-study-verification-v1',mechanics_status='verified',instrument_status='Dedicated source/custody/numerical audits; generic array-only evidence adapter incomplete',terminal_status='unassessed',panels=panels,atlas=dict(programs=6,occurrences=24,source_native_states=4,new_native_targets=0,exact_queries=24,changed_context_abstentions=48),installed_mechanics_tests=sum(int(s.attrib['tests']) for s in suites),transport_bytes_externalized=freed,evidence=[dict(path=str(p.resolve()),sha256=sha(p)) for p in references],qualification_sources={n:sha(here/n) for n in names},new_native_calls_in_verification=0,
        limitations='Fixed-law development on known states, not an unseen-state result. Full-program raw P0 attention was not retained; output projection is the earliest retained port demonstrating reversal, not proven first divergence. Weak mixed interactions remain unresolved. Remote transport hashes were checked during offload, not re-read by this local verifier. Current study custody remains local; older externalized collections require exact restoration for full legacy transport verification. No new semantic/image or speed qualification.')
    with (root/'verification-v1.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',panels=list(panels),atlas=result['atlas'],installed_mechanics_tests=result['installed_mechanics_tests'],new_native_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(verify(a.root.resolve()),sort_keys=True))
