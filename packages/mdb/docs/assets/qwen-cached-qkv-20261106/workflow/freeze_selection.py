"""Freeze two row-sign cases using authenticated metadata only."""
import hashlib
import json
from pathlib import Path

BASE = Path('/Users/jakeholl/domains')
ROOT = Path(__file__).resolve().parent
PARENT = BASE / 'artifacts/mdb-qwen-sign-shortcut-20261106'
ROW = BASE / 'artifacts/mdb-qwen-row-sign-20261106'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_path = ROW / 'selection-v1/protocol.json'
    if sha(old_path) != 'd942001a79784ebb1bc5d5ed3f3946eaa03b4629f2fe13c7493afbaaf85b51e9':
        raise ValueError('Original metadata selection changed')
    old = json.loads(old_path.read_text())
    pack = PARENT / 'compact-collection-01/extracted/binding-minimal-result'
    manifest_path = pack / 'source-files.json'
    if sha(manifest_path) != old['parent_raw_manifest_sha256']:
        raise ValueError('Original raw inventory changed')
    inventory = json.loads(manifest_path.read_text())['files']
    forecast_path = PARENT / 'frontier-metadata-v1/cases/long-description/forecast.json'
    raw = forecast_path.read_bytes()
    if inventory['cases/long-description/forecast.json'] != dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()):
        raise ValueError('Original positive forecast metadata changed')
    forecast = json.loads(raw)
    arrays = {key: value for key, value in old['source_arrays'].items() if key not in ('gate', 'up', 'product')}
    arrays.update({key: forecast['ports']['plus'][key] for key in ('q', 'k', 'v')})
    for name, value in arrays.items():
        if inventory[value['path']] != {key: value[key] for key in ('bytes', 'sha256')}:
            raise ValueError('Source array metadata changed: ' + name)
    arms = [dict(id='last-row-negative', negative_ranges=[[511, 512]], windows=[[0, 511], [511, 512]]),
            dict(id='alternating64', negative_ranges=[[64, 128], [192, 256], [320, 384], [448, 512]],
                 windows=[[start, start + 64] for start in range(0, 512, 64)])]
    for arm in arms:
        signs = [-1 if any(start <= row < end for start, end in arm['negative_ranges']) else 1 for row in range(512)]
        arm['row_signs'] = signs
        arm['structural_support_closed_rows'] = sum(all(signs[row] == signs[key] for key in range(row + 1)) for row in range(512))
        arm['permitted_cross_sign_edges'] = sum(signs[row] != signs[key] for row in range(512) for key in range(row + 1))
    protocol = {key: old[key] for key in ('parent_job', 'parent_request_sha256', 'parent_raw_root',
        'parent_raw_manifest_sha256', 'parent_global_seal_sha256', 'source_case', 'sequence_rows', 'edit_index', 'horizon',
        'original_component', 'attention_dispatch', 'actual_native_mask', 'actual_native_is_causal', 'cache', 'frontier', 'limited_to')}
    protocol.update(schema='mdb-qwen-cached-qkv-selection-v1', source_arrays=arrays, arms=arms,
        basis='Inherited own-functional positive positioned Q/K/V, H18, baseline ports and input/RoPE metadata; no negative or native target ports.',
        predictor='Qprime=sQ; Kprime=sK; Vprime=sV; recompute same-profile SDPA, output projection, residual, postnorm, gate/up, SiLU product, down projection and H19.',
        layout_policy='Restore original declared dense BHTD strides from canonical NPY coordinates before SDPA; record and compare actual operand geometry.',
        baseline='Archived functional positive basis versus fresh standalone native positive baseline, reported separately.',
        comparisons=['cached-qkv-minus-full-functional', 'full-functional-minus-native', 'cached-qkv-minus-native'],
        expectations=['Source-row negation is a candidate finite QKV relation; retain numerical, bit and signed-zero differences separately.',
            'Recomputed attention may account for mixed-support responses missed by the cheaper R/G/U shortcut.',
            'Neither structural cross-sign edges nor mixed support require a nonzero measured effect.',
            'Seal both complete predictions, transformed QKV and full signed response hashes before all three native component targets.'],
        physical_plan=dict(full_functional_edited_blocks=2, cached_attention_downstream_evaluations=2, native_block_calls=3,
            functional_SDPA=2, cached_SDPA=2, native_SDPA=3, functional_linear=14, cached_linear=8, native_linear=21,
            component_hydrations=1, full_model=0, LM=0, embedding=0, tokenizer=0, FLUX=0, VAE=0, noise=0),
        scientific_gates=[], raw_array_or_weight_bytes_read_by_selection=0)
    output = ROOT / 'selection-v1'
    output.mkdir(exist_ok=False)
    with (output / 'protocol.json').open('x') as stream:
        json.dump(protocol, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(protocol_sha256=sha(output / 'protocol.json'), arrays=len(arrays),
        inherited_array_bytes=sum(value['bytes'] for value in arrays.values()),
        arms=[{key: arm[key] for key in ('id', 'structural_support_closed_rows', 'permitted_cross_sign_edges')} for arm in arms])))


if __name__ == '__main__':
    main()
