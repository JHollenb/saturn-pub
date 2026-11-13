"""The intervention is always formed from the untouched receiving state."""
import runpy
from pathlib import Path
import numpy as np
import pytest


def test_targets_do_not_accumulate_anchor_rounding():
    here=Path(__file__).resolve().parents[1]/'qualification'
    p=runpy.run_path(str(here/'flux_fresh_mlp_program.py'))
    B=runpy.run_path(str(here/'flux_reentry_reference.py'))['bf16']
    base=np.ones((1,16,4),np.float32)
    # A half-step rounds back to 1, while the .875-step crosses a BF16 boundary.
    D=np.full((1,2,4),.006,np.float64)
    anchor=p['make_text'](base,D,[.5,.5],B)
    target=p['make_text'](base,D,[.875,.125],B)
    wrong=p['make_text'](anchor,D,[.375,0],B)
    assert np.array_equal(anchor,base)
    assert not np.array_equal(target,wrong)
    assert np.array_equal(target[:,4],B(base[:,4].astype(float)+.875*D[:,0]))
    assert np.array_equal(base,np.ones_like(base))
    other=[i for i in range(16) if i not in (4,11)]
    assert np.array_equal(target[:,other],base[:,other])
    with pytest.raises(ValueError,match='doses'):p['make_text'](base,D,[np.nan,.5],B)
    with pytest.raises(ValueError,match='geometry'):p['make_text'](base,D[:,:1],[.5,.5],B)


def test_named_prediction_must_be_the_consumed_cut():
    import copy
    h=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/analyze_flux_fresh_mlp_future.py'))
    chain=[dict(operation='apply',parent='anchor',result='i'),dict(operation='apply',parent='i',result='installed'),dict(operation='continue',parent='installed',result='commit3'),dict(operation='continue',parent='commit3',result='commit4')]
    row=dict(entry='installed',source_exit='anchor',commits=['commit3','commit4'],final='commit4')
    h['verify_consumed_state'](row,chain,'anchor')
    detached={**row,'entry':'another-correct-looking-cut'}
    with pytest.raises(ValueError,match='detached'):h['verify_consumed_state'](detached,chain,'anchor')
    wrong={**row,'commits':['unconsumed','commit4']}
    with pytest.raises(ValueError,match='commits'):h['verify_consumed_state'](wrong,chain,'anchor')
    native=copy.deepcopy(chain);native[1].update(operation='continue',steps=1)
    native_row={**row,'source_exit':'installed'}
    h['verify_consumed_state'](native_row,native,'anchor',dict(entry='i',exit='installed'))
    with pytest.raises(ValueError,match='detached'):h['verify_consumed_state'](native_row,native,'anchor',dict(entry='i',exit='unconsumed'))
