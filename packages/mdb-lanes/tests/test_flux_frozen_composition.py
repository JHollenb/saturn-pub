import runpy
from pathlib import Path
import numpy as np
import pytest

HELPER=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/flux_frozen_composition.py'))


def test_source_only_composition_signed_transfer_and_nonmutation():
    values={k:np.full((1,4,8),v,np.float32) for k,v in [('base',1),('color',3),('shape',5)]}
    old_joint=np.full((1,4,8),11,np.float32);old_add=np.full((1,4,8),7,np.float32)
    before={k:v.copy() for k,v in values.items()}
    result=HELPER['compose'](values,old_joint,old_add)
    for k,expected in [('additive',7),('plus_old',11),('minus_old',3)]: assert np.array_equal(result[k],np.full((1,4,8),expected,np.float32))
    assert all(np.array_equal(v,before[k]) for k,v in values.items())
    with pytest.raises(ValueError,match='three source'): HELPER['compose']({**values,'joint':old_joint},old_joint,old_add)


def test_hue_diagnostic_separates_gold_violet_neutral():
    gold=np.array([[[255,170,0]]],np.uint8);violet=np.array([[[150,40,220]]],np.uint8);neutral=np.array([[[128,128,128]]],np.uint8)
    assert HELPER['hue_observer'](gold)['gold_band']==1
    assert HELPER['hue_observer'](violet)['violet_band']==1
    assert HELPER['hue_observer'](neutral)['chromatic']==0


def test_source_only_frontier_uses_public_transaction_contract(tmp_path):
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    tx=ForecastTransaction(tmp_path/'transaction',model_identity='checkpoint',input_identity='three-source',source_identity='frozen-rule',
        frontier=HELPER['forecast_frontier']({'new_joint_encoded':False}),execution_scope='Three observed source encodings; no joint outcome available')
    with tx:
        tx.frontier('binding',{'new_joint_encoded':False})
        tx.predict('rule',lambda _: {'predictions':{'value':7}})
        tx.seal()
        tx.execute('native-window',lambda _: {'outcome':'retained'})
    assert verify_transaction(tx.root)['verified']


def test_environment_drift_refused_before_scheduler_admission(tmp_path,monkeypatch):
    import json
    from mdb_lanes.execution_environment import intent
    staged=intent()
    (tmp_path/'request.json').write_text(json.dumps({'execution_environment':staged}))
    monkeypatch.setenv('MDB_WORKER_UV_CACHE_DIR','/different-task-cache')
    launcher=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/submit_flux_frozen_composition.py'))
    with pytest.raises(ValueError,match='restage before admission'):
        launcher['submit'](tmp_path)
