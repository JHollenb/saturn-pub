"""Pair-closure mechanics before a source-assisted I/T cross is interpretable."""
from pathlib import Path
from copy import deepcopy
import runpy
import json
import pytest

H = runpy.run_path(str(Path(__file__).parents[1] / 'qualification/flux_binding_carrier_cross.py'))


def pair():
    p = dict(image=dict(kind='tensor', dtype='torch.bfloat16', shape=[1, 1024, 3072], sha256='image-P'),
             text=dict(kind='tensor', dtype='torch.bfloat16', shape=[1, 512, 3072], sha256='text-P'),
             conditioning={'sha256': 'base-C'}, latent={'sha256': 'same-Z'}, context={'gate': 'same-clock'})
    q = deepcopy(p); q['image']['sha256'] = 'image-Q'; q['text']['sha256'] = 'text-Q'
    return p, q


def test_stream_differences_are_allowed_but_hidden_context_change_refuses():
    p, q = pair()
    assert H['compatible_payloads'](p, q)['common_nonstream_slots'] == ['conditioning', 'context', 'latent']
    q['context']['gate'] = 'different-clock'
    with pytest.raises(ValueError, match='outside image/text'): H['compatible_payloads'](p, q)


def test_conditioning_and_slot_set_are_part_of_the_mutable_closure():
    p, q = pair(); q['conditioning']['sha256'] = 'source-C'
    with pytest.raises(ValueError, match='outside image/text'): H['compatible_payloads'](p, q)
    p, q = pair(); q['unaccounted-cache'] = {'sha256': 'hidden'}
    with pytest.raises(ValueError, match='slot set'): H['compatible_payloads'](p, q)


def test_incompatible_stream_geometry_refuses():
    p, q = pair(); q['text']['shape'] = [1, 511, 3072]
    with pytest.raises(ValueError, match='geometry'): H['compatible_payloads'](p, q)


def test_public_description_tags_stay_inside_each_named_slot():
    from saturn_pub.values import describe
    payload = {'conditioning': None, 'context': {'time': 2, 'ids': (0, 1)}}
    assert describe(payload)['kind'] == 'mapping'
    slots = H['payload_slot_descriptors'](payload)
    assert set(slots) == set(payload)
    assert slots['context'] == describe(payload['context'])
    assert slots['context']['kind'] == 'mapping'
    assert slots['conditioning'] == {'kind': 'scalar', 'type': 'none', 'value': None}


def test_public_immutable_cut_landmark_serializes_without_mutating_cut():
    from saturn_pub import StateCut
    point = dict(family='flux2', logical_step=2, phase='joint', index=1)
    cut = StateCut(model_identity='tiny-mechanics', execution={}, boundary='joint1',
                   parent=None, _payload={key: None for key in ('image', 'text', 'latent', 'conditioning')},
                   execution_point=point)
    with pytest.raises(TypeError):
        json.dumps(cut.execution_point)
    record = H['landmark_record'](cut, 'anchor', 'I.P_T.Q', 'entry')
    assert json.loads(json.dumps(record))['point'] == point
    record['point']['index'] = 9
    assert cut.execution_point['index'] == 1
    cut.verify()
