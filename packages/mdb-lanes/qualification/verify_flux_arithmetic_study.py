"""Close the arithmetic study's source, custody, audits and atlas joins."""
import argparse
import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path
from mdb_lanes.edit_search import file_digest as sha
from mdb_lanes.computation_atlas import ComputationAtlas


def verify(root):
    read=lambda p:json.loads(p.read_text());here=Path(__file__).resolve().parent
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'];panels={};references=[]
    for phase in ('scout','transfer','fresh','normalization'):
        collection=root/phase/'collection-01';checked=custody(collection);request=read(collection/'request.json')
        if read(collection/'job.json')['state']!='succeeded':raise ValueError('incomplete run')
        for name,digest in request['source_hashes'].items():
            if sha(collection/'payload'/name)!=digest:raise ValueError('sealed source drift')
            current=here/name
            if current.exists() and name!='retrieve_weights.py' and sha(current)!=digest:raise ValueError('executed qualification source changed')
        data=collection/'extracted/binding-minimal-result';summary=read(data/'summary.json')
        if summary!=read(collection/'report.json')['report']:raise ValueError('summary differs')
        panels[phase]=dict(custody=checked,counts=summary.get('counts'),worker_seconds=summary.get('seconds',summary.get('worker_seconds')))
        references.extend([collection/'request.json',collection/'report.json'])
        if phase!='scout':
            audit=root/phase/'audit-v1.json';a=read(audit)
            if a['mechanics_status']!='verified':raise ValueError('unverified audit')
            binding=a['custody'][phase] if phase in ('fresh','transfer') else a['custody']
            if any(binding[k]!=checked[k] for k in ('job_id','archive_sha256','request_sha256')):raise ValueError('audit custody join differs')
            references.append(audit)
    a=read(root/'normalization/audit-v1.json')
    if a['source_fresh_audit_sha256']!=sha(root/'fresh/audit-v1.json'):raise ValueError('diagnostic source audit differs')
    atlas=ComputationAtlas(root/'index-v1/atlas');records=atlas.find()
    if len(records)!=8:raise ValueError('atlas census differs')
    for record in records:
        context=record['equation']['tested_context']
        if len(atlas.find(applicable_to=context))!=1 or atlas.find(applicable_to={**context,'clock':dict(step=2,joint=2)}) or atlas.find(applicable_to={**context,'input_identity':'unknown'}):raise ValueError('applicability differs')
        for ref in record['provenance']['artifacts']:
            if sha(ref['path'])!=ref['sha256']:raise ValueError('atlas provenance drift')
    suites=ET.parse(root/'mechanics-v4.xml').getroot().findall('testsuite')
    if sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites):raise ValueError('mechanics tests not clean')
    if read(root/'storage-verified-v2.json')['mechanics_status']!='verified':raise ValueError('shared storage not verified')
    references.extend([root/'index-v1/provenance.json',root/'mechanics-v4.xml',root/'README.md',root/'advice/results-review.md',root/'advice/normalization-review.md',root/'prior-transport-externalized-v1.json',root/'normalization/run-01/failure-note.json',root/'normalization/run-01/job.json',root/'storage-verified-v2.json',root/'storage-compacted-v1.json',root/'storage-compacted-v2.json'])
    result=dict(schema='mdb-attention-arithmetic-study-verification-v1',mechanics_status='verified',instrument_status='Dedicated source/custody/typed-state/forecast audits; no semantic image instrument in this cut',terminal_status='unassessed',panels=panels,atlas=dict(programs=2,occurrences=8,prospective_native_states=4,exact_queries=8,changed_context_abstentions=16),installed_mechanics_tests=sum(int(s.attrib['tests']) for s in suites),
        evidence=[dict(path=str(p.resolve()),sha256=sha(p)) for p in references],qualification_sources={p.name:sha(p) for p in [here/n for n in ('flux_arithmetic_reference.py','flux_arithmetic_scout.py','flux_affine_transfer.py','analyze_flux_affine_transfer.py','flux_arithmetic_prediction.py','flux_arithmetic_fresh.py','stage_flux_arithmetic_fresh.py','analyze_flux_arithmetic_fresh.py','flux_absolute_norm_prediction.py','flux_absolute_norm_diagnostic.py','analyze_flux_absolute_norm.py','publish_flux_arithmetic.py','verify_flux_arithmetic_study.py')]},new_native_calls_in_verification=0,
        limitations='Normalization variants are post-hoc development; the selected update needs prospective validation. Same-input affine matches do not imply accurate owned inputs. Small mixed effects and attention arithmetic remain unresolved. No new image, semantic or speed qualification. Historical transport offload is separately reversible; original extracted evidence retained.')
    with (root/'verification-v1.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',phases=list(panels),atlas=result['atlas'],installed_mechanics_tests=result['installed_mechanics_tests'],new_native_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(verify(a.root.resolve()),sort_keys=True))
