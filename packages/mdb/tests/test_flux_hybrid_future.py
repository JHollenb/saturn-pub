"""Exercise the actual qualification writer through the installed public archive reader."""
import runpy
from pathlib import Path

import numpy as np
import pytest
from mdb.tensor_archive import load_tensor_archive


def test_prediction_and_native_archive_roundtrip_and_sharing(tmp_path):
    torch=pytest.importorskip('torch')
    h=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/flux_hybrid_future.py'));cache={}
    x=np.array([[[1.0,-2.0,0.0],[.125,16.0,-.5]]],dtype=np.float32)
    predicted=h['record_bf16'](tmp_path,x,cache);native=h['record_bf16'](tmp_path,torch.from_numpy(x.copy()).to(torch.bfloat16),cache)
    assert predicted==native and len(cache)==1
    assert np.array_equal(load_tensor_archive(tmp_path,predicted),x)
    other=h['record_bf16'](tmp_path,x.reshape(1,3,2),cache)
    assert other!=predicted and len(cache)==2
    assert np.array_equal(load_tensor_archive(tmp_path,other),x.reshape(1,3,2))
    with pytest.raises(ValueError,match='finite BF16'):h['record_bf16'](tmp_path,np.array([1.0001],np.float32),cache)
