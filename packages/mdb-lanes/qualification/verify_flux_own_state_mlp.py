"""Offline custody closure and compact observations for the own-state MLP study."""
import argparse
import hashlib
import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path


def verify(study, output):
    from mdb_lanes.computation_atlas import ComputationAtlas
    from mdb_lanes.equation_comparison import EquationCatalog
    h = runpy.run_path(str(Path(__file__).with_name('flux_family_custody.py')))
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: h['_sha'](p)[0]
    pointer = read(study/'remote-pointer-v1.json')
    record = pointer['record']
    canonical = (json.dumps(record, sort_keys=True, allow_nan=False)+'\n').encode()
    if hashlib.sha256(canonical).hexdigest() != pointer['record_sha256']:
        raise ValueError('remote scientific record digest differs')
    request = h['_request'](study/'run-02')
    jobs = {kind: read(study/(kind+'-job-v1.json')) for kind in ('calculation', 'audit')}
    if (jobs['calculation']['state'] != 'succeeded'
            or jobs['calculation']['job_id'] != record['verification']['job_id']
            or request['request_sha256'] != record['verification']['request_sha256']):
        raise ValueError('calculation request/job binding differs')
    collection = study/'audit/collection-01'
    custody = h['verify'](collection)
    data = collection/'extracted/binding-minimal-result'
    audit, summary, projection = [read(data/name) for name in ('audit-v1.json', 'summary.json', 'projection.json')]
    if (jobs['audit']['state'] != 'succeeded' or jobs['audit']['job_id'] != custody['job_id']
            or audit['custody'] != record['verification']
            or summary['scientific_custody'] != record['verification']
            or read(data/'remote-custody.json') != record
            or sha(data/'remote-custody.json') != pointer['record_sha256']):
        raise ValueError('independent audit/scientific custody chain differs')
    if (audit['mechanics_status'] != 'verified' or audit['instrument_status'] != 'qualified'
            or audit['instrument_issues'] or audit['source_control_issues']
            or audit['complete_analysis_reproduced'] is not True):
        raise ValueError('independent recomputation incomplete')
    if (sha(data/'audit-v1.json') != summary['audit_sha256']
            or sha(data/'projection.json') != summary['projection_sha256']
            or sha(collection/'payload/flux_own_state_mlp_audit.py') != audit['auditor_source_sha256']):
        raise ValueError('audit/source/projection digest differs')
    for name, digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name) != digest:
            raise ValueError('installed wheel differs')
    for name, item in projection['copied'].items():
        if sha(data/name) != item['sha256'] or (data/name).stat().st_size != item['bytes']:
            raise ValueError('projection copy differs')
        if name.startswith('calculation-proof/'):
            member = name[len('calculation-proof/'):]
            if record['files'][member]['sha256'] != item['sha256']:
                raise ValueError('calculation metadata differs from full remote inventory')
    proof = data/'calculation-proof'
    calculated, analysis, controls, states = [read(proof/name) for name in ('summary.json', 'analysis-v1.json', 'checks.json', 'states.json')]
    if read(proof/'request.json') != request or sha(proof/'summary.json') != record['summary_sha256']:
        raise ValueError('original calculation proof differs')
    expected = dict(mlp_calculations=36, affine_calculations=72, swiglu_calculations=36,
                    receiving_writes=60, model_loads=0, native_calls=0, images=0)
    if (calculated['counts'] != expected or audit['counts'] != dict(independent_mlp_calculations=36,
            independent_receiving_writes=60, model_loads=0, native_calls=0, images=0)
            or audit['unique_arrays'] != calculated['unique_arrays']
            or audit['primary'] != analysis['primary'] or summary['primary'] != analysis['primary']):
        raise ValueError('physical work/report census differs')
    census = {}
    for roles in controls.values():
        for groups in roles.values():
            for name, value in groups.items():
                if value is not True:
                    raise ValueError('source reconstruction control differs')
                census[name] = census.get(name, 0)+1
    if census != dict(native_write_closure=12, old_own_exit=12, old_own_mlp=12, selected_conditional_join=4):
        raise ValueError('source control census differs')
    comparisons = 0
    for point, roles in audit['checks'].items():
        if point not in states or set(roles) != {'image', 'text'}:
            raise ValueError('independent point census differs')
        for role, groups in roles.items():
            if set(groups) != {'gates', 'mlp', 'writes'}:
                raise ValueError('independent port census differs')
            for group, items in groups.items():
                details = [value for fields in items.values() for value in fields.values()] if group == 'writes' else list(items.values())
                for value in details:
                    if value['raw_bits_exact'] is not True:
                        raise ValueError('independent raw-bit comparison differs')
                    comparisons += 1
    if comparisons != 204 or set(audit['checks']) != set(states):
        raise ValueError('independent raw-bit comparison census differs')
    catalog = EquationCatalog(data/'index-v1/catalog')
    atlas = ComputationAtlas(data/'index-v1/atlas')
    records, exact, abstain = catalog.find(), 0, 0
    for row in records:
        context = row['tested_context']
        if context['frontier']['kind'] != 'saved-own-state-refinement':
            raise ValueError('development frontier lost')
        exact += len(atlas.find(applicable_to=context))
        abstain += not bool(atlas.find(applicable_to={**context, 'clock': dict(step=2, joint=2)}))
        abstain += not bool(atlas.find(applicable_to={**context, 'input_identity': 'unknown'}))
    if len(records) != 8 or exact != 8 or abstain != 16:
        raise ValueError('catalog/applicability census differs')
    test = ET.parse(study/'mechanics-v2.xml').getroot()
    suites = [test] if test.tag == 'testsuite' else list(test.iter('testsuite'))
    if sum(int(s.get('tests', '0')) for s in suites) != 5 or any(int(s.get(k, '0')) for s in suites for k in ('failures', 'errors', 'skipped')):
        raise ValueError('installed mechanics tests incomplete')
    # Scientific measurements below are observations, never qualification gates.
    ordinary, biases, powers = {}, {}, {}
    for name, roles in analysis['reports'].items():
        ordinary[name] = {}
        biases[name] = {}
        powers[name] = {}
        for role, row in roles.items():
            ordinary[name][role] = {method: {field: fields['reference_response']['relative_rms']
                for field, fields in values['fields'].items()} for method, values in row['methods'].items()}
            biases[name][role] = {method: values['fields']['exit']['defect_power']
                for method, values in row['methods'].items()}
            powers[name][role] = {method: values['finite_write']['prediction_error']
                for method, values in row['methods'].items()}
    mixed = {role: {field: {method: value['contrasts']['receiver_by_dose']
        for method, value in by_method.items()} for field, by_method in by_field.items()}
        for role, by_field in analysis['mixed'].items()}
    scientific_bytes = sum(item['bytes'] for item in record['files'].values())
    array_bytes = sum(item['bytes'] for name, item in record['files'].items() if name.startswith('arrays/'))
    result = dict(schema='mdb-own-state-mlp-study-verification-v1', mechanics_status='verified',
        instrument_status='qualified', terminal_status='unassessed',
        remote_scientific_custody=record['verification'], remote_record_sha256=pointer['record_sha256'],
        local_projection_custody=custody, source_controls=census, independent_bit_comparisons=comparisons,
        atlas=dict(supplied_programs=2, occurrences=8, exact_queries=exact, changed_context_abstentions=abstain),
        installed_mechanics_tests=5, calculation_counts=expected, independent_counts=audit['counts'],
        primary=analysis['primary'], observations=dict(ordinary=ordinary, exit_defects=biases,
            finite_write_error=powers, receiver_by_pattern=mixed),
        storage=dict(remote_scientific_bytes=scientific_bytes, remote_array_bytes=array_bytes,
            unique_output_arrays=audit['unique_arrays'], compact_projection_bytes=sum(p.stat().st_size for p in data.rglob('*') if p.is_file()),
            compact_archive_bytes=custody['archive_bytes']),
        jobs={kind: dict(job_id=job['job_id'], state=job['state'], elapsed_s=job['result']['elapsed_s'],
            peak_rss_mb=job['result']['peak_rss_mb'], peak_vram_mb=job['result']['peak_vram_mb']) for kind, job in jobs.items()},
        scope='Full remote scientific custody verified there, separate 36-MLP/60-write raw-bit recomputation, complete signed report reproduced, and offline compact custody closure. Scientific errors remain observations. Saved development only: no fresh native prediction, independent CUDA arithmetic, image, default change, or inference speed claim.')
    with output.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    return dict(mechanics_status='verified', instrument_status='qualified', independent_bit_comparisons=comparisons,
        source_controls=census, atlas_occurrences=8, primary=analysis['primary'], storage=result['storage'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('study', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.study.resolve(), a.output.resolve()), sort_keys=True))
