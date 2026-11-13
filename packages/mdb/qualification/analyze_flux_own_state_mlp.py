"""Source-bound paired-response and finite-write analysis of saved own-state edits."""
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from mdb.response_closure import audit_factorial, audit_response, measure_error
from mdb.tensor_archive import load_tensor_archive


def read(path):
    return json.loads(Path(path).read_text())


class Evidence:
    def __init__(self, output, config):
        self.output, self.config = Path(output), config
        self.source = Path(config['source_collection']) / 'extracted/binding-minimal-result'
        self.previous = Path(config['previous_conditional_root'])
        self.inventory = read(Path(config['source_collection']) / 'remote-custody.json')['files']
        self.records = read(self.output / 'states.json')
        self.bits = runpy.run_path(str(Path(__file__).with_name('flux_attention_anchor.py')))['load_bits']
        self.files = set()

    def array(self, ref):
        item, root = ref['record'], ref['root']
        if root == 'conditional':
            return self.bits(self.previous, item)
        roots = dict(output=self.output, source=self.source, prefixes=self.source/'source/prefixes')
        if root not in roots:
            raise ValueError('explicit evidence root required')
        base = roots[root]
        if root != 'output':
            name = (base/item['path']).relative_to(self.source).as_posix()
            if self.inventory[name]['sha256'] != item['sha256']:
                raise ValueError('source array outside verified inventory')
        else:
            self.files.add(item['path'])
        return load_tensor_archive(base, item)

    def state(self, point, role, method):
        return {key: self.array(ref) for key, ref in self.records[point][role]['states'][method].items()}


def analyze(output, config):
    evidence = Evidence(output, config)
    math = runpy.run_path(str(Path(__file__).with_name('flux_own_state_mlp_math.py')))
    reports = {}
    methods = ('old_own', 'new_own', 'old_conditional', 'new_conditional')
    for spec in config['cases']:
        name, anchor = spec['id'], spec['anchor']
        reports[name] = {}
        for role in ('image', 'text'):
            N0, N1 = [evidence.state(point, role, 'native') for point in (anchor, name)]
            states = {method: [evidence.state(point, role, method) for point in (anchor, name)] for method in methods}
            row = dict(methods={}, inherited_vs_conditional={})
            for method, (P0, P1) in states.items():
                fields = {field: audit_response(N0[field], N1[field], P0[field], P1[field]) for field in ('mlp', 'gated', 'exit')}
                row['methods'][method] = dict(fields=fields, finite_write=math['transport'](N0, N1, P0, P1))
            for policy in ('old', 'new'):
                P0, P1 = states[policy+'_own']
                C0, C1 = states[policy+'_conditional']
                row['inherited_vs_conditional'][policy] = {}
                for field in ('mlp', 'exit'):
                    p = P1[field].astype(float)-P0[field].astype(float)
                    c = C1[field].astype(float)-C0[field].astype(float)
                    n = N1[field].astype(float)-N0[field].astype(float)
                    row['inherited_vs_conditional'][policy][field] = math['power_account'](
                        dict(inherited_program_state=p-c, conditional_law=c-n), p-n)
            row['new_minus_old'] = {}
            for field in ('mlp', 'gated', 'exit'):
                old0, old1 = states['old_own']
                new0, new1 = states['new_own']
                row['new_minus_old'][field] = measure_error(new1[field].astype(float)-new0[field].astype(float), old1[field].astype(float)-old0[field].astype(float))
            reports[name][role] = row
            del states, N0, N1, P0, P1, C0, C1, old0, old1, new0, new1
    # Four corners: receiving image (first/second), pattern (row11/row4-heavy).
    order = ('row11seven_eighths', 'row4seven_eighths', 'row11seven_eighths.image1', 'row4seven_eighths.image1')
    if set(order) != {spec['id'] for spec in config['cases']}:
        raise ValueError('declared four-corner cohort differs')
    mixed = {}
    for role in ('image', 'text'):
        mixed[role] = {}
        for field in ('mlp', 'gated', 'exit'):
            actual = {key: evidence.array(evidence.records[name][role]['states']['native'][field]) for key, name in zip(('00','01','10','11'), order)}
            mixed[role][field] = {}
            for method in methods:
                predicted = {key: evidence.array(evidence.records[name][role]['states'][method][field]) for key, name in zip(('00','01','10','11'), order)}
                mixed[role][field][method] = audit_factorial(actual, predicted)
    primary = {method: float(np.mean([reports[spec['id']]['image']['methods'][method]['fields']['exit']['reference_response']['relative_rms'] for spec in config['cases']])) for method in ('old_own','new_own')}
    return dict(schema='mdb-own-state-mlp-analysis-v1', reports=reports, mixed=mixed,
        primary=dict(mean_image_complete_response_error=primary), output_array_paths=sorted(evidence.files),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        terminal_status='unassessed', scope='Saved-case development on stored own upstream predictions. Conditional native-input worlds are explicitly separate diagnostics. Actual P0/P1, finite gated-write transport and signed cancellation retained. No fresh prediction, native call, image or speed claim.')
