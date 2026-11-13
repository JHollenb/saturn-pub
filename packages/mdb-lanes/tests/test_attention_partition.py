import numpy as np
import pytest
from mdb_lanes.attention_partition import dense_attention,prepare_attention_partition


@pytest.mark.parametrize('indices',[(1,),(1,4),tuple(range(7))])
def test_partition_matches_dense_with_changed_queries_and_global_max_shift(indices):
    rng=np.random.default_rng(101)
    q,k,v=[rng.normal(size=(1,3,7,8))for _ in range(3)]
    partition=prepare_attention_partition(q,k,v,indices,query_chunk=3)
    new_k=k[:,:,indices,:].copy()+80;new_v=v[:,:,indices,:].copy()-10
    changed=(1,5);new_q=q[:,:,changed,:].copy()*-4
    actual,work=partition.evaluate(new_k,new_v,query_rows=new_q,query_indices=changed)
    full_k=k.copy();full_v=v.copy();full_q=q.copy()
    full_k[:,:,indices,:]=new_k;full_v[:,:,indices,:]=new_v;full_q[:,:,changed,:]=new_q
    expected=dense_attention(full_q,full_k,full_v,query_chunk=2)
    np.testing.assert_allclose(actual,expected,rtol=256*np.finfo(float).eps,atol=256*np.finfo(float).eps)
    assert work['changed_queries']==2
    assert not partition.q.flags.writeable


def test_changing_the_old_dominant_key_does_not_cancel_the_remaining_mass():
    q=np.ones((1,1,2,1));k=np.array([1000.,0.]).reshape(1,1,2,1);v=np.array([4.,7.]).reshape(1,1,2,1)
    p=prepare_attention_partition(q,k,v,[0])
    new_k=np.array([-1000.]).reshape(1,1,1,1);new_v=np.array([9.]).reshape(1,1,1,1)
    result,_=p.evaluate(new_k,new_v)
    np.testing.assert_array_equal(result,np.full((1,1,2,1),7.))


def test_rejects_ambiguous_indices_nonfinite_inputs_and_missing_changed_queries():
    x=np.ones((1,1,3,2))
    with pytest.raises(ValueError,match='mutable key rows'):prepare_attention_partition(x,x,x,[1,1])
    with pytest.raises(ValueError,match='finite'):prepare_attention_partition(x*np.nan,x,x,[1])
    p=prepare_attention_partition(x,x,x,[1])
    with pytest.raises(ValueError,match='supplied together'):p.evaluate(x[:,:,:1],x[:,:,:1],query_indices=[1])
