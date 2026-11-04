"""Model-free pre-entry arithmetic, custody and timing audit checks."""
from __future__ import annotations

import hashlib
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "qualification"
A = runpy.run_path(str(HERE / "analyze_flux_single_mlp.py"))
Q = runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py"))["quantize_bf16"]


def test_independent_layernorm_time_affine_and_swiglu_scalar_reference():
    x = np.array([[[-1, 1], [1, -1]]], np.float32)
    mod = np.array([[1, 2, 0, 1, 999, 999]], np.float32)
    weights = np.array([[1, 0], [0, 1]], np.float32)
    out = A["entry_equations"](x, mod, weights, None, eps=3., queries=[0], neurons=[0], hidden_width=2, mlp_width=3, quantize=Q)
    assert np.array_equal(out["fp64"]["norm"], [[[-.5, .5]]])
    assert np.array_equal(out["fp64"]["modulated"], [[[.5, 3]]])
    assert out["fp64"]["u"].item() == .5 and out["fp64"]["v"].item() == 3
    assert out["fp64"]["activation"].item() == pytest.approx(1.5 / (1 + np.exp(-.5)))
    assert out["bf16_reference"]["activation"].item() == Q(Q(np.array([.5 / (1 + np.exp(-.5))], np.float32)) * 3).item()


def test_paired_fused_rows_are_after_qkv_with_second_mlp_bank_offset():
    assert A["paired_row_indices"](3072, 9216, [0, 288, 8928]) == [9216, 9504, 18144, 18432, 18720, 27360]
    for neurons in ([0, 0], [9216], [True]):
        with pytest.raises(ValueError, match="row selector"):
            A["paired_row_indices"](3072, 9216, neurons)


def test_silu_stable_extremes_and_invalid_source_shapes():
    out = A["stable_silu"](np.array([-1000., 0., 1000.]))
    assert np.isfinite(out).all() and out.tolist() == [-0., 0., 1000.]
    with pytest.raises(ValueError, match="operands"):
        A["entry_equations"](np.zeros((1, 2, 2)), np.zeros((1, 6)), np.zeros((2, 2)), None,
            eps=1e-6, queries=[True], neurons=[0], hidden_width=2, mlp_width=3, quantize=Q)


def test_prepared_timing_has_no_fictitious_six_step_panel_replay():
    events = []
    counts = {**A["PANEL_COUNTS"], "candidate_calls": 0, "adapter_steps": 0, "acts": 0,
              "images": 0, "vae_calls": 0, "receipt_emissions": 0}
    events.append({"kind": "hypotheses_sealed", "counts": counts, "blocked_native_entries": 0})
    for role in ("zero", "text"):
        events.extend([{"kind": "candidate_start", "candidate": role}, {"kind": "candidate_complete", "candidate": role}])
    events.append({"kind": "projection_forecast_sealed", "counts": {**counts, "candidate_calls": 2}})
    for role in ("A", "M", "AM"):
        events.extend([{"kind": "candidate_start", "candidate": role}, {"kind": "candidate_complete", "candidate": role}])
    events = [{**v, "sequence": i} for i, v in enumerate(events)]
    assert A["SHARED"]["projection_timing"](events, replay_steps=0)["qualified"]
    assert not A["SHARED"]["projection_timing"](events)["qualified"]


def test_framework_free_import_does_not_execute_native_or_staged_helper():
    code = '''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','diffusers','transformers','saturn','common'}:raise RuntimeError(fullname)
sys.meta_path.insert(0,Block())
runpy.run_path(sys.argv[1])
'''
    result = subprocess.run([sys.executable, "-I", "-c", code, str(HERE / "analyze_flux_single_mlp.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_bf16_bits_archive_is_lossless_immutable_and_native_hash_checked(tmp_path):
    from mdb.tensor_archive import TensorArchiveError

    bits = np.array([[0x0000, 0x8000, 0x3f80, 0xc000, 0x0001]], dtype='<u2')
    file = tmp_path / 'sample.npy'
    np.save(file, bits, allow_pickle=False)
    record = {'path':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
        'descriptor':{'dtype':'torch.bfloat16','shape':list(bits.shape),'sha256':hashlib.sha256(bits.tobytes()).hexdigest()},
        'stored_dtype':'uint16','stored_shape':list(bits.shape),'native_stride':[5,1],
        'encoding':'bf16-bits-v1','native_byteorder':'little'}
    e = A['SavedEvidence'](tmp_path)
    actual = e.array(record)
    assert np.array_equal(actual.view(np.uint32), bits.astype(np.uint32) << 16)
    assert actual[0,2] == 1 and actual[0,3] == -2 and not actual.flags.writeable
    with pytest.raises(ValueError):
        actual.setflags(write=True)
    bits[0,2] = 0x4000
    np.save(file,bits,allow_pickle=False)
    record['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    with pytest.raises(TensorArchiveError,match='native tensor digest'):
        e.array(record)


def test_fresh_contrast_collapse_is_observation_but_other_state_drift_is_invalid():
    fields = {'low':{'image':'i','text':'t','conditioning':'c'},'high':{'image':'i','text':'t','conditioning':'c'}}
    e = SimpleNamespace(fields=lambda key:fields[key],point=lambda key:{'phase':'single','index':6},
                        slot=lambda cut,key:np.zeros((1,2),np.float32))
    observed = A['fresh_pair'](e,'low','high')
    assert observed['image_exact'] and observed['text_exact']
    fields['high']['conditioning']='other'
    with pytest.raises(ValueError,match='outside I/T'):
        A['fresh_pair'](e,'low','high')


def test_prompt_text_ids_require_native_int64_bytes_not_integer_valued_floats():
    ids = np.arange(1536, dtype=np.int64).reshape(512,3)
    assert A['encoding_ids_exact'](ids,ids.copy())
    assert not A['encoding_ids_exact'](ids.astype(np.float32),ids)
    assert not A['encoding_ids_exact'](ids,ids.astype(np.float32))
    changed = ids.copy()
    changed[0,0] = 1
    assert not A['encoding_ids_exact'](changed,ids)


@pytest.mark.parametrize('signed,ordering',[(-2.,'A-closer'),(2.,'M-closer'),(0.,'tie')])
def test_fresh_ordering_is_observation_not_inherited_historical_prediction(signed,ordering):
    local = {'H_M_over_A':{'observed':'miss','scope':'Predeclared tentative', 'signed_A_minus_M_L2_to_high':signed}}
    A['fresh_local_ordering'](local)
    assert 'H_M_over_A' not in local
    assert local['A_vs_M_observation']['observed_ordering'] == ordering
    assert local['A_vs_M_observation']['scope'].startswith('Unpredicted exploratory ordering;')


def test_bitwise_mismatch_census_includes_signed_zero_without_quality_gate():
    actual = np.array([0.,1.,2.],np.float32)
    reference = np.array([-0.,1.,3.],np.float32)
    assert A['bitwise_mismatch_counts'](actual,reference) == {'sample_count':3,'bf16_reference_mismatch_count':2}
