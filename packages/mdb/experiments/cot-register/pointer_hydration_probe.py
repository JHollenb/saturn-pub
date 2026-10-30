"""Model-free thin-view and selected-payload probe over a retained live trial."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import sys
from pathlib import Path


class NoWorkspace(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'saturn', 'saturn_pub', 'common', 'common_harness', 'mrun'}:
            raise ImportError('model-free probe attempted upstream import: ' + fullname)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--installed', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.installed.resolve()))
    sys.meta_path.insert(0, NoWorkspace())
    from mdb.circuit_runtime import DirectoryArtifactStore
    from mdb.future_forest import PersistentFutureForest
    from mdb.vm_payload_cache import PayloadStore

    job = args.job.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    native = job / 'artifacts'
    artifacts = DirectoryArtifactStore(native / 'bench/artifacts')
    payloads = PayloadStore(native / 'bench/vm/radix/cas', read_only=True)
    report = json.loads((native / 'report.json').read_text())
    manifest_bytes = (native / 'artifact-manifest.json').read_bytes()
    rows, cases = [], []
    for path in sorted((native / 'rows').glob('*-observed.json')):
        history = json.loads(path.read_text())
        cases.append({'id': history['id'], 'prompt': history['prompt'],
                      'generated_text': history['generated_text']})
        for observation in history['observations']:
            sealed = artifacts.get(observation['register_ref'])
            read = artifacts.get(sealed['observation_ref'])
            if sealed['model_identity'] != report['model_identity']:
                raise ValueError('observation model binding differs')
            if read['source']['cut_ref'] != sealed['source_cut']:
                raise ValueError('observation source cut differs')
            if read['source_ref']['sha256'] != sealed['activation_sha256']:
                raise ValueError('observation source activation differs')
            rows.append({'case': history['id'], 'branch': sealed['branch'],
                         'cut_ref': sealed['source_cut'], 'register_ref': observation['register_ref'],
                         'observation_ref': sealed['observation_ref'], 'lens_ref': sealed['lens_ref'],
                         'clock': read['source']['cursor'], 'token_position': sealed['token_position'],
                         'residual_ref': read['source_ref'], 'candidates': sealed['candidates']})
    view = {'schema': 'mdb-cot-thin-pointer-view-probe-v1', 'origin_job': job.name,
            'origin_output_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'model_identity': report['model_identity'], 'profile': report['profile'],
            'native_backing': 'external original session; references are not embedded payloads',
            'semantic_cot_qualified': False, 'cases': cases, 'observations': rows}
    forest = PersistentFutureForest(output / 'thott-view', forest_id='cot-pointer-probe')
    ref = forest.put_json_object(view, media_type='application/vnd.mdb.cot-pointer-view+json')
    node = forest.create_root(state_refs={'cot-view': ref}, metadata={'origin_job': job.name,
                              'scope': 'descriptive readouts and native pointers; no execution authority'})
    pointer = {'schema': 'mdb-cot-thin-view-pointer-probe-v1', 'forest_id': forest.forest_id,
               'node_id': node, 'view_ref': ref, 'origin_job': job.name}
    pointer_bytes = json.dumps(pointer, sort_keys=True, separators=(',', ':')).encode()
    (output / 'pointer.json').write_bytes(pointer_bytes + b'\n')

    reopened = PersistentFutureForest.reopen(output / 'thott-view')
    reopened.verify_node(node)
    state = reopened.effective_state(node)
    raw_view = reopened.get_blob(state['cot-view'])
    restored_view = json.loads(raw_view)
    if restored_view != view:
        raise ValueError('thin view differs after Thott reopen')
    before = payloads.audit_summary()
    selected = next(row for row in restored_view['observations'] if row['case'] == 'france')
    source = selected['residual_ref']
    hydrated = payloads.audit(source, expected_shape=(1, 1, 1536),
                             expected_dtype='torch.bfloat16', expected_bytes=3072)
    if hashlib.sha256(hydrated.payload).hexdigest() != source['sha256']:
        raise ValueError('selected residual hydration differs')
    after = payloads.audit_summary()
    result = {'schema': 'mdb-cot-pointer-hydration-probe-v1', 'origin_job': job.name,
              'mechanics_status': 'valid', 'model_execution_calls': 0,
              'observations': len(rows), 'pointer_json_bytes': len(pointer_bytes) + 1,
              'thin_view_blob_bytes': len(raw_view), 'view_payload_reads': before['read_attempts'],
              'view_native_payload_bytes': before['bytes_read'],
              'selected_source_cut': selected['cut_ref'],
              'selected_residual_sha256': source['sha256'],
              'selected_residual_bytes': len(hydrated.payload), 'hydration_audit': after,
              'source_integrity_checks': 'existing DirectoryArtifactStore, Thott node, and PayloadStore',
              'boundary': 'metadata rendering and selected residual byte hydration only; no rewind, native continuation, live integration, or speedup claim'}
    (output / 'report.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
