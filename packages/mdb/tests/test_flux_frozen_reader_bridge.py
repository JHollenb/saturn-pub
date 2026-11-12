"""Independent mechanics checks for the development-only causal bridge."""
from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

WORKER = Path(__file__).resolve().parents[1] / "qualification/flux_frozen_reader_bridge.py"


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_interaction_preserves_negative_differences(dtype):
    operation = runpy.run_path(str(WORKER))["factorial"]
    # Independent complete factorial: Q=100, R=90, W=80, RW=50 -> -20.
    result = operation({name: np.array([value], dtype=dtype) for name, value in
                        [("whole_Q", 100), ("residual_only", 90), ("write_only", 80), ("both_RW", 50)]})
    assert result.dtype == np.float64
    assert result.tolist() == [-20.0]


def test_incompatible_factorial_shape_is_mechanics_failure():
    operation = runpy.run_path(str(WORKER))["factorial"]
    with pytest.raises(ValueError, match="shape"):
        operation({"whole_Q": np.zeros(2), "residual_only": np.zeros(2),
                   "write_only": np.zeros(2), "both_RW": np.zeros(3)})


def test_equal_values_with_changed_layout_are_rejected():
    import torch
    from saturn_pub.values import describe

    check = runpy.run_path(str(WORKER))["checked_tensor"]
    original = torch.arange(24).reshape(1, 4, 6).transpose(1, 2)
    descriptor = describe(original)
    assert check(original, descriptor) is original
    with pytest.raises(ValueError, match="layout"):
        check(original.contiguous(), descriptor)


def test_registered_read_uses_identity_and_excludes_adjacent_future():
    from mdb.titan import ActRegistry, ActSpec, Frame

    dispatch = runpy.run_path(str(WORKER))["dispatch_reader"]
    spec = ActSpec("test.bridge_reader", "v1", reads=("current",), writes=("H0",),
                   write_kinds={"H0": "tensor"}, write_abis={"H0": {"shape": [2], "dtype": "float32"}})
    frame = Frame.from_values({"current": np.array([1, 2], np.float32), "historical_future": "forbidden"})
    registry = ActRegistry()

    def read(values):
        assert set(values) == {"current"}
        return {"H0": values["current"] * 2}, {}

    registry.register(spec, "frozen-reader", read)
    output, evidence = dispatch(registry, spec, frame)
    assert output.values["H0"].tolist() == [2, 4]
    assert evidence.act == spec.identity
    assert evidence.input_frame_id == frame.frame_id


def test_guard_blocks_existing_future_before_release(tmp_path):
    # Separate process because Python audit hooks cannot be uninstalled.
    script = r'''
import json,runpy
from pathlib import Path
from types import SimpleNamespace
r=Path(ROOT); source=r/'native'; model=r/'model'; coeff=r/'old-predictor'/'fit';
for p in [source,model,coeff]:p.mkdir(parents=True)
for name in ['image.png','cut.json']:(source/name).write_bytes(b'real future canary')
(coeff/'weight.npy').write_bytes(b'allowed frozen coefficient')
e=SimpleNamespace(root=source,closure=lambda _ref:[])
bridge={'cases':{'c':{'input_cut_after_conditioning_reset':{},'previous_velocity':{'cut':{}},
 'evaluation_only':{'png':{'source_path':'image.png'},'velocity3':{'cut':{'member':'cut.json'}}}}}}
fn=runpy.run_path(WORKER)['install_guard']
phase=fn(e,bridge,{'modelRoot':str(model),'imageAssets':{}},{'fields':{'weight':{'path':str(coeff/'weight.npy')}}},{'fields':{}})
assert len(phase['denied'])==2 and phase['future_opens']==0
assert (coeff/'weight.npy').read_bytes()==b'allowed frozen coefficient'
try:(source/'image.png').read_bytes()
except PermissionError:pass
else:raise AssertionError('future leaked')
phase['released']=True
assert (source/'image.png').read_bytes()==b'real future canary'
assert phase['future_opens']==1
print(json.dumps({'denied':len(phase['denied']),'released_opens':phase['future_opens']}))
'''.replace("ROOT", repr(str(tmp_path)), 1).replace("WORKER", repr(str(WORKER)), 1)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"denied": 3, "released_opens": 1}
