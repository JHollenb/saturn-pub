"""Recovery preserves source bytes and reconstructs plain CAS metadata only."""
import hashlib
import json
from pathlib import Path
import runpy
from types import MappingProxyType

import pytest


H = runpy.run_path(str(Path(__file__).parents[1] / 'qualification/recover_flux_binding_carrier_cross.py'))


def test_physical_inventory_preserves_malformed_partial_and_refuses_symlinks(tmp_path):
    source = tmp_path / 'failed'; source.mkdir()
    partial = b'[{"point": '
    (source / 'trajectory.json').write_bytes(partial)
    value = H['physical_inventory'](source)
    assert value['files']['trajectory.json'] == dict(sha256=hashlib.sha256(partial).hexdigest(), bytes=len(partial))
    assert (source / 'trajectory.json').read_bytes() == partial
    (source / 'alias').symlink_to(source / 'trajectory.json')
    with pytest.raises(ValueError): H['physical_inventory'](source)


def test_reconstructed_points_are_plain_json_and_landmarks_are_complete():
    point = MappingProxyType(dict(logical_step=2, phase='joint', index=1))
    described = {slot: dict(kind='scalar', value=None) for slot in ('image', 'text', 'latent', 'conditioning')}
    class Evidence:
        def cut(self, fingerprint):
            return dict(execution_point=point), described
    rows = [dict(case='fixture', arm='I.P_T.P', marks={label: label for label in H['LABELS']})]
    value = H['reconstruct_trajectory'](rows, Evidence())
    assert len(value) == 7 and {row['label'] for row in value} == set(H['LABELS'])
    assert type(value[0]['point']) is dict
    assert json.loads(json.dumps(value))[0]['point']['index'] == 1


def test_actual_environment_log_extraction_and_request_seal_are_explicit():
    environment = dict(packages=dict(torch='observed', diffusers='observed-too'), cuda_driver='logged-device-description')
    log = 'MDB_EXECUTION_ENVIRONMENT=' + json.dumps(environment)
    assert H['execution_environment_from_log'](log) == environment
    assert H['execution_environment_from_log']('no environment record') is None
    with pytest.raises(ValueError): H['execution_environment_from_log'](log + '\n' + log)
    request = dict(config=dict(fixture=True))
    request['request_sha256'] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert H['request_identity'](request) == request['request_sha256']
    request['config']['fixture'] = False
    with pytest.raises(ValueError): H['request_identity'](request)
