"""Small NumPy mechanics checks; no native modules or real models."""
from __future__ import annotations

import copy
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "qualification"
A = runpy.run_path(str(HERE / "analyze_flux_mlp_replacement.py"))
Q = runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py"))["quantize_bf16"]


def test_chunked_full_field_has_paired_banks_and_explicit_rounding():
    image = np.array([[[-1, 1], [1, -1], [-1, 1]]], np.float32)
    mod = np.array([[1, 2, 0, 1, 100, 100]], np.float32)
    # eps3 makes norm ±.5; affine for first query [.5,3].
    weights = np.array([[1,0], [0,1], [1,1], [0,1], [1,0], [1,-1]], np.float32)
    result = A['rebuild_full_mlp'](image,mod,weights,eps=3.,mlp_width=3,query_chunk=1,neuron_chunk=2,quantize=Q)
    u = np.array([.5,3,3.5],np.float32)
    v = np.array([3,.5,-2.5],np.float32)
    expected = Q(Q(A['ENTRY']['stable_silu'](u).astype(np.float32))*v)
    assert np.array_equal(result[0,0],expected) and np.array_equal(result[0,2],expected)
    whole = A['rebuild_full_mlp'](image,mod,weights,eps=3.,mlp_width=3,query_chunk=3,neuron_chunk=3,quantize=Q)
    assert A['exact_bits'](result,whole)
    with pytest.raises(ValueError,match='geometry'):
        A['rebuild_full_mlp'](image,mod,weights,eps=3.,mlp_width=3,query_chunk=0,quantize=Q)


def test_full_field_census_retains_signed_zero_and_zero_reference_energy():
    actual = np.array([[[0.,1.],[-0.,-2.]]],np.float32)
    reference = np.zeros_like(actual)
    result = A['field_error'](actual,reference,chunk=1)
    assert result['sample_count']==4 and result['bitwise_mismatch_count']==3
    assert result['l2']==np.sqrt(5) and result['signed_mean']==-.25
    assert result['relative_rms'] is None and result['cosine'] is None


@pytest.mark.parametrize('changed',['text','attention','M'])
def test_replacement_footprint_rejects_undeclared_or_wrong_source_operand(changed):
    original = np.arange(24,dtype=np.float32).reshape(1,4,6)
    replacement = np.full((1,2,4),100,np.float32)
    used = original.copy();used[:,2:,2:]=replacement
    assert all(A['replacement_recipe'](original,used,replacement,text_rows=2,attention_width=2).values())
    index = {'text':(0,0,0),'attention':(0,2,0),'M':(0,2,2)}[changed]
    used[index]+=1
    with pytest.raises(ValueError,match='undeclared consumer'):
        A['replacement_recipe'](original,used,replacement,text_rows=2,attention_width=2)


def test_wrong_address_roll_is_exact_position_permutation_and_keeps_shape():
    value = np.arange(24,dtype=np.float32).reshape(1,4,6)
    rolled = A['roll_field'](value,shift=2)
    assert rolled.shape==value.shape and A['exact_bits'](rolled[:,2:],value[:,:2])
    assert np.array_equal(np.sort(rolled.ravel().view(np.uint32)),np.sort(value.ravel().view(np.uint32)))
    assert not A['exact_bits'](rolled,value)


def test_identical_predicted_and_measured_roll_can_share_contiguous_archive_record():
    original=np.arange(48,dtype=np.float32).reshape(1,4,12)
    measured=original[:,:,4:]
    predicted=measured.copy()
    measured_roll=A['roll_field'](measured,shift=2)
    predicted_roll=A['roll_field'](predicted,shift=2)
    # The first archived occurrence is contiguous. A later predicted payload
    # can use the native sliced layout while sharing its dtype/shape/value key.
    record={'descriptor':{'dtype':'torch.bfloat16'},'native_stride':[32,8,1]}
    assert A['exact_bits'](measured_roll,predicted_roll)
    result=A['payload_archive_witness'](record,measured_roll,predicted_roll)
    assert result['declared_values_exact'] is True
    assert result['stored_representative_native_stride'] != [48,12,1]
    changed=predicted_roll.copy();changed[0,0,0]+=1
    with pytest.raises(ValueError,match='lossless declared'):
        A['payload_archive_witness'](record,measured_roll,changed)


def test_error_over_zero_effect_is_undefined_not_a_failed_threshold():
    assert A['error_effect_ratio'](1,0)['ratio'] is None
    assert A['error_effect_ratio'](0,0)['status']=='undefined-zero-effect'
    assert A['error_effect_ratio'](2,4)['ratio']==.5


def test_auditor_import_is_framework_free_and_never_runs_staged_predictor():
    code='''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','diffusers','transformers','saturn','common'}:raise RuntimeError(fullname)
sys.meta_path.insert(0,Block())
runpy.run_path(sys.argv[1])
'''
    result=subprocess.run([sys.executable,'-I','-c',code,str(HERE/'analyze_flux_mlp_replacement.py')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def timing_fixture():
    seal = {"seal_sha256":"sealed-before-native"}
    before = {"candidate_calls":0,"adapter_steps":108}
    events = [{"kind":"full_field_forecast_sealed","seal":seal,"counts":before,"target_entries_before_seal":0}]
    for index,name in enumerate(A['ROW_IDS']):
        events += [{"kind":"candidate_start","candidate":name,"counts":{"candidate_calls":index}},
                   {"kind":"candidate_complete","candidate":name,"counts":{"candidate_calls":index+1}}]
    events += [{"kind":"branch_complete","row":name} for name in A['ROW_IDS']]
    for index,event in enumerate(events):event['sequence']=index
    return events,{"data":Path('/not-read'),"summary":{"seal":seal,"target_calls":11},"binding":{"counts_before_seal":before}}


def test_public_seal_timing_rejects_native_entry_before_seal_and_missing_arm(monkeypatch):
    events,parent = timing_fixture()
    monkeypatch.setitem(A['SHARED'],'read_jsonl',lambda path:events)
    assert A['audit_timing'](parent,{"target_calls":11})['direct_calls_after_seal']==9
    changed = copy.deepcopy(events)
    changed[0]['target_entries_before_seal']=1
    monkeypatch.setitem(A['SHARED'],'read_jsonl',lambda path:changed)
    with pytest.raises(ValueError,match='ordering/target'):
        A['audit_timing'](parent,{"target_calls":11})
    changed = copy.deepcopy(events)
    changed[3]['candidate']='zero'
    monkeypatch.setitem(A['SHARED'],'read_jsonl',lambda path:changed)
    with pytest.raises(ValueError,match='ordering/target'):
        A['audit_timing'](parent,{"target_calls":11})


def test_downstream_error_effect_keeps_roll_countertrend_and_zero_effect():
    base=np.zeros((1,2,2),np.float32)
    images={name:base.copy() for name in A['ROW_IDS']}
    latents={name:base.copy() for name in A['ROW_IDS']}
    pixels={name:np.zeros((2,2,3),np.uint8) for name in A['ROW_IDS']}
    images['M.predicted'][0,0,0]=1
    latents['M.measured'].fill(3)
    latents['M.predicted'].fill(5)
    latents['M.measured.roll'].fill(1)
    _,pairs,ratios=A['endpoint_analysis'](images,pixels,latents)
    assert ratios['high_M']['post_image_L2']['status']=='undefined-zero-effect'
    assert ratios['high_M']['latent_L2']['ratio']==pytest.approx(2/3)
    assert pairs['measured_placement']['latent']['signed_mean']==-2
