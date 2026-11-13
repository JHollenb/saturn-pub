"""Bound source pairing, one gate and target-only return surgery."""
import json
import runpy
from pathlib import Path
import numpy as np
from mdb_lanes.forecast_transaction import ForecastTransaction,load_forecast_tree,verify_transaction

Q=Path(__file__).resolve().parents[1]/'qualification'
W=runpy.run_path(str(Q/'flux_last_writer.py'));P=runpy.run_path(str(Q/'flux_commit_factorial.py'))
B=runpy.run_path(str(Q/'flux_reentry_reference.py'))['bf16']
A=runpy.run_path(str(Q/'flux_last_writer_audit.py'))


def test_actual_saturn_tagged_context_descriptor():
    from saturn_pub.values import describe
    context=describe({'temb':3,'img_mod':5})
    assert A['descriptor_item'](context,'temb')==describe(3)
    import pytest
    with pytest.raises(ValueError):A['descriptor_item'](context,'missing')


def test_crosses_keep_recipient_and_same_layout():
    rows=W['cases'](['c0-o0','c0-o1','c1-o0','c1-o1'])
    assert len(rows)==16
    for row in rows.values():
        assert row['residual_source'].split('-')[1]==row['write_source'].split('-')[1]==row['receiver'].split('-')[1]
        assert (row['residual_source']==row['receiver'])==(row['R']==0)
        assert (row['write_source']==row['receiver'])==(row['U']==0)
    assert not W['transplant_active']('source-writer') and not W['transplant_active']('first-reader') and not W['transplant_active']('continuation')
    assert W['transplant_active']('target-writer')


def test_raw_write_gate_applied_once_and_signed_parts_preserved():
    r=np.array([1.,-2.],np.float32);u=np.array([4.,2.],np.float32);gate=np.array([.5,-1.],np.float32)
    expected=B(r+B(gate*u))
    assert np.array_equal(expected,[3.,-4.])
    assert not np.array_equal(expected,B(r+B(gate*B(gate*u))))


def test_exact_forecast_packet_contract(tmp_path):
    root=tmp_path/'tx';tx=ForecastTransaction(root,model_identity='tiny',input_identity={'r':'cut'},source_identity='source',
        frontier=dict(kind='architectural',step=2,phase='single',cursor=19),execution_scope='Tiny last-writer transaction API check')
    tx.frontier('sources',{'r':{'R':{'shape':[1,1536,3072]}}})
    value={f:{'shape':[1,2,2],'sha256':'dummy'} for f in ('Y','N2','Z3','image','values','product')}
    tx.predict('r.r0u1',lambda _:P['forecast_record'](value));s=tx.seal();verify_transaction(root,expected_seal_sha256=s['seal_sha256'])
    actual=load_forecast_tree(root,json.loads((root/'forecasts/r.r0u1.json').read_text())['predictions'])
    assert list(actual['Y']['shape'])==[1,2,2]
    tx.execute('r.r0u1',lambda _:dict(outputs=value));verify_transaction(root,expected_seal_sha256=s['seal_sha256']);tx.close()
