"""Mechanics of the finite campaign; no model execution or semantic mock scores."""
import runpy
from pathlib import Path

H = runpy.run_path(str(Path(__file__).parents[1] / 'qualification/flux_binding_generalization.py'))


def test_schedule_geometry_and_complete_factorial():
    rows, arms = H['contexts'](), H['arms']()
    assert len(rows) == 8 and len(arms) == 9
    assert len({r['id'] for r in rows}) == 8
    assert {r['seed'] for r in rows} == {109, 347}
    assert len({(r['primed'], r['early']) for r in arms[:4]}) == 4
    expected = H['expected_counts']()
    assert expected['native_transitions'] == 8 * (2*112 + 23 + 9*56)
    assert expected['native_blocks'] == 8 * (2*100 + 21 + 9*50)
    assert expected['acts'] == 8 * (1 + 9*2 + 1+1+2)
    assert expected['receipts'] == expected['acts'] + expected['native_transitions']


def token(ids, valid=None):
    return dict(ids=ids, mask=valid or [1]*len(ids), offsets=[[i, i+1] for i in range(len(ids))])


def test_row_resolver_is_paired_and_preserves_shift_diagnostic():
    a, b = token([1, 2, 3, 4, 5, 0], [1, 1, 1, 1, 1, 0]), token([1, 7, 8, 4, 5, 9])
    result = H['resolve_rows'](a, b)
    assert result['changed'] == [1, 2, 5]
    assert result['control'] == [0, 3, 4]
    assert result['token_lengths_equal'] is False


def test_single_factor_bridges_and_frozen_material_texture():
    cases = {c['id']:c for c in H['contexts']()}
    assert cases['anchor']['prompts'] == cases['seed']['prompts']
    assert cases['anchor']['seed'] != cases['seed']['seed']
    assert cases['material']['attribute_class'] == 'material'
    assert cases['texture']['attribute_class'] == 'texture'
    assert cases['anchor']['prompts']['base'].startswith('A red sphere to the left of a blue cube')
    assert cases['anchor']['prompts']['source'].startswith('A blue sphere to the left of a red cube')
