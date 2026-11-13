"""Route recipes and sealed scalar predictions before a new intervention panel."""
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

C=runpy.run_path(str(Path(__file__).parents[1]/'qualification/flux_reader_routes.py'))


def test_head_restore_preserves_other_heads_and_inputs():
    base=torch.arange(48).reshape(1,3,16).to(torch.bfloat16);alternate=-base
    result=C['mix_heads'](base,alternate,[1,3],heads=4)
    for i in range(4):
        assert torch.equal(result[...,i*4:(i+1)*4],(alternate if i in (1,3) else base)[...,i*4:(i+1)*4])
    assert torch.equal(base,torch.arange(48).reshape(1,3,16).to(torch.bfloat16))
    assert torch.equal(C['mix_heads'](base,alternate,[],4),base)
    assert torch.equal(C['mix_heads'](base,alternate,list(range(4)),4),alternate)


def test_scalar_effect_frontier_is_serializable_and_sealed_before_execution(tmp_path):
    b=np.zeros((1,3,4),np.float32);target=np.ones_like(b)
    m=C['metrics'](b,target,target*.5)
    assert m['target_fraction']==.5 and m['relative_error']==.5 and m['cosine']==1.
    with pytest.raises(ValueError):C['metrics'](b,b,target)
    tx=ForecastTransaction(tmp_path/'tx',model_identity='tiny',input_identity='parent',source_identity='source',
        frontier=dict(kind='architectural',boundary='left complete, right not executed'))
    tx.frontier('left',dict(measure=m,heads=[3,1,0,2]))
    tx.predict('rank',lambda ctx:dict(predictions=dict(heads=ctx.frontiers['left']['heads'])))
    seal=tx.seal();verify_transaction(tx.root,expected_seal_sha256=seal['seal_sha256'])
    tx.execute('right',lambda _:dict(heads=[1,3,0,2]))
    assert tuple(tx.prediction('rank')['heads'])==(3,1,0,2)
    verify_transaction(tx.root,expected_seal_sha256=seal['seal_sha256']);tx.close()
