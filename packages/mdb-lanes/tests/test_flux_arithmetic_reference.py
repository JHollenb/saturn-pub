"""Numerical-instrument mechanics, without loading a model."""
import runpy
from pathlib import Path
import numpy as np

R=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/flux_arithmetic_reference.py'))


def test_rounding_sign_and_tie():
    x=np.array([1+1/256,-1-1/256,1+3/256,-1-3/256],np.float32)
    np.testing.assert_array_equal(R['bf16'](x),[1,-1,1+1/64,-1-1/64])
    np.testing.assert_array_equal(R['bf16'](x,'rtz'),[1,-1,1+1/128,-1-1/128])


def test_wide_affine_retains_bias_and_hypotheses():
    x=np.array([[1.,2.,3.,4.]]); w=np.array([[1.,-1.,2.,-.5]])
    for policy in R['affine_family']():
        np.testing.assert_array_equal(R['affine_wide'](x,w,policy,np.array([.5])),[[3.5]])
    assert len({p['id'] for p in R['affine_family']()})==18


def test_online_wide_attention_preserves_query_key_roles():
    rng=np.random.default_rng(91)
    q,k,v=[rng.normal(size=s).astype(np.float32) for s in [(1,2,3,4),(1,2,301,4),(1,2,301,5)]]
    expected=R['attention_wide'](q,k,v,R['attention_family']()[0])
    for policy in R['attention_family']():
        actual=R['attention_wide'](q,k,v,policy)
        assert actual.shape==(1,2,3,5) and np.isfinite(actual).all()
        if policy['probability']=='wide': np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)


def test_calibration_and_check_coordinates_disjoint():
    worker=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/flux_arithmetic_scout.py'))
    s=worker['selections']()
    assert not set(s['train_rows'])&set(s['check_rows'])
    assert not set(s['train_columns'])&set(s['check_columns'])


def test_absolute_text_affine_preserves_partition_complement():
    prediction=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/flux_arithmetic_prediction.py'))
    x=np.arange(32,dtype=np.float32).reshape(1,8,4); weight=np.eye(4,dtype=np.float32); native=np.full_like(x,99)
    actual=prediction['absolute_affine'](x,native,weight,dict(kind='carry',chunk=2,rounding='rne'),R['bf16'],mutable_rows=[1,4])
    np.testing.assert_array_equal(actual[:,[1,4]],x[:,[1,4]])
    np.testing.assert_array_equal(actual[:,[0,2,3,5,6,7]],native[:,[0,2,3,5,6,7]])


def test_partition_and_fallback_use_identical_effective_operands():
    from mdb_lanes.attention_partition import prepare_attention_partition,dense_attention
    prediction=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/flux_arithmetic_prediction.py'))
    rng=np.random.default_rng(75); ix=[1,4]
    anchors=[R['bf16'](rng.normal(size=(1,8,4))) for _ in range(3)]
    inputs=[R['bf16'](rng.normal(size=(1,8,4))) for _ in range(3)]
    # The absolute law disagrees with cached returns even outside the edit.
    anchors[2][:,6]=99
    owned=[prediction['absolute_affine'](x,a,np.eye(4,dtype=np.float32),dict(kind='wide'),R['bf16'],mutable_rows=ix) for x,a in zip(inputs,anchors)]
    layout=lambda x:x.reshape(1,8,1,4).transpose(0,2,1,3)
    partition=prepare_attention_partition(*map(layout,anchors),ix)
    q,k,v=map(layout,owned)
    from_partition,_=partition.evaluate(k[:,:,ix],v[:,:,ix],query_rows=q[:,:,ix],query_indices=ix)
    np.testing.assert_array_equal(owned[2][:,6],anchors[2][:,6])
    np.testing.assert_allclose(from_partition,dense_attention(q,k,v),rtol=1e-12,atol=1e-12)


def test_anchored_normalization_transports_rounding_phase():
    prediction=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/flux_arithmetic_prediction.py'))
    # A correct real-valued law can still acquire an extra BF16 ULP from y0's rounding residual.
    x0=np.array([[1.003]],np.float32); x1=np.array([[1.005]],np.float32)
    anchor=R['bf16'](x0)
    anchored=prediction['normalized_delta'](x1,x0,anchor,lambda x:x,R['bf16'])
    np.testing.assert_array_equal(anchored,[[1.]])
    np.testing.assert_array_equal(R['bf16'](x1),[[1.0078125]])
