"""Intervention construction preserves text ports and unselected source rows."""
import runpy
from pathlib import Path

import numpy as np

Q=Path(__file__).parents[1]/'qualification'
C=runpy.run_path(str(Q/'flux_reader_route_future.py'))
B=runpy.run_path(str(Q/'flux_reentry_reference.py'))['bf16']


def test_kv_doses_and_source_bands_preserve_text_and_other_rows():
    base={k:np.zeros((1,1026,1,2),np.float32) for k in ('k','v')}
    changed={k:np.full_like(v,8.) for k,v in base.items()}
    for k in base:base[k][:,:2]=3;changed[k][:,:2]=3
    recipes=C['specs'](7,2)
    for spec in recipes:
        result=C['ports'](base,changed,spec,B,nt=2)
        for key,value in result.items():
            assert np.array_equal(value[:,:2],base[key][:,:2])
            if spec['kind']=='dose':assert np.all(value[:,2:]==(3 if spec['dose']==.375 else 5))
            if spec['kind']=='band-add':
                assert np.all(value[:,514:770]==8) and np.all(value[:,2:514]==0) and np.all(value[:,770:]==0)
            if spec['kind']=='band-remove':
                assert np.all(value[:,514:770]==0) and np.all(value[:,2:514]==8) and np.all(value[:,770:]==8)
    assert np.all(base['k'][:,2:]==0) and np.all(changed['k'][:,2:]==8)


def test_payload_digest_agrees_between_exact_widening_and_native_bits():
    import torch
    value=torch.tensor([[-0.,1.,-2.,.375]],dtype=torch.bfloat16)
    assert C['payload_digest'](value)==C['payload_digest'](value.float().numpy())


def test_image_cohorts_protect_text_and_complements_for_every_head():
    import pytest
    import torch
    base=torch.arange(1*5*24*2).reshape(1,5,24,2)
    changed=base+1000
    for i in range(24):
        add=C['image_head_route'](base,changed,dict(kind='image-head-add',heads=[i]),2)
        remove=C['image_head_route'](base,changed,dict(kind='image-head-remove',heads=[i]),2)
        assert torch.equal(add[:,:2],base[:,:2]) and torch.equal(remove[:,:2],base[:,:2])
        assert torch.equal(add[:,2:,i],changed[:,2:,i]) and torch.equal(remove[:,2:,i],base[:,2:,i])
        others=[j for j in range(24) if j!=i]
        assert torch.equal(add[:,2:,others],base[:,2:,others])
        assert torch.equal(remove[:,2:,others],changed[:,2:,others])
    for heads in ([],[1,1],[-1],[24],[True]):
        with pytest.raises(ValueError):C['image_head_route'](base,changed,dict(kind='image-head-add',heads=heads),2)
