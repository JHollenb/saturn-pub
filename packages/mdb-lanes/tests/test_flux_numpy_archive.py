"""Check actual public writer compatibility and a fresh framework-free process."""
import json
import runpy
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from mdb_lanes.tensor_archive import load_tensor_archive


HELPER=Path(__file__).resolve().parents[1]/'qualification/flux_numpy_archive.py'


def test_numpy_writer_matches_native_writer_bytes_and_records(tmp_path):
    torch=pytest.importorskip('torch');native=runpy.run_path(str(HELPER.with_name('flux_hybrid_future.py')))['record_bf16'];writer=runpy.run_path(str(HELPER))['record_bf16'];a=tmp_path/'native';b=tmp_path/'numpy';first={};second={}
    x=np.array([[[0.,-0.,1.],[-2.,2**-133,3.25]]],np.float32)
    p=native(a,x,first);q=writer(b,x,second)
    assert p==q and (a/p['path']).read_bytes()==(b/q['path']).read_bytes()
    assert load_tensor_archive(b,q).tobytes()==x.tobytes()
    assert writer(b,x,second)==q and len(second)==1
    z=writer(b,x.reshape(1,3,2),second)
    assert z!=q and len(second)==2


def test_numpy_writer_refuses_lossy_and_nonfinite_values(tmp_path):
    writer=runpy.run_path(str(HELPER))['record_bf16']
    for x in (np.array([1.0001],np.float32),np.array([np.inf],np.float32),np.array([],np.float32)):
        with pytest.raises(ValueError,match='finite BF16'):writer(tmp_path,x,{})


def test_numpy_writer_is_framework_free_in_fresh_process(tmp_path):
    script='''import runpy,sys,json
from pathlib import Path
import numpy as np
from mdb_lanes.tensor_archive import load_tensor_archive
writer=runpy.run_path(sys.argv[1])['record_bf16'];r=writer(Path(sys.argv[2]),np.array([1.,-2.],np.float32),{})
assert load_tensor_archive(Path(sys.argv[2]),r).tolist()==[1.,-2.]
assert not any(n in sys.modules for n in ('torch','diffusers','transformers'))
print(json.dumps(r))
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(HELPER),str(tmp_path)],text=True,capture_output=True,check=True)
    assert json.loads(result.stdout)['encoding']=='bf16-bits-v1'
