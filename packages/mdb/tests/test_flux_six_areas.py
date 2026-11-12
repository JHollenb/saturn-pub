"""Small scientific-instrument checks; no model/framework import."""
import runpy
from pathlib import Path
import numpy as np
import pytest

H=runpy.run_path(str(Path(__file__).resolve().parents[1]/"qualification/flux_six_areas.py"))


def test_factorial_retains_interaction_and_rejects_incomplete_geometry():
    x={"b0r0":np.zeros((1,2,3)),"b1r0":np.ones((1,2,3)),"b0r1":np.full((1,2,3),2.),"b1r1":np.full((1,2,3),4.)}
    assert H["factorial"](x)["interaction_rms"]==1.
    assert H["factorial"](x)["interaction_over_joint"]==.25
    with pytest.raises(ValueError):H["factorial"]({k:v for k,v in x.items() if k!="b1r1"})


def test_routes_use_all_keys_and_keep_padding_visible():
    qkv={"q":np.zeros((1,6,2,4),np.float32),"k":np.ones((1,6,2,4),np.float32)}
    result=H["route_witness"](qkv,2,{"red":[0],"padding":[1]},queries=3)
    assert result["probability"].shape==(1,2,3,6)
    assert np.allclose(result["all_text"],1/3)
    assert np.allclose(result["padding"],1/6)
    assert np.allclose(result["probability"].sum(-1),1)
    assert np.all(result["rows"]>=2)


def test_resident_entrypoint_uses_worker_not_launcher_parser(monkeypatch):
    monkeypatch.setenv("SATURN_RESIDENT_WORKER","1")
    monkeypatch.delenv("SATURN_RUN_CONFIG",raising=False)
    path=Path(__file__).resolve().parents[1]/"qualification/flux_binding_program.py"
    # Missing worker configuration must fail at worker binding, before any
    # framework/model import. The prior bug instead exited argparse with2.
    with pytest.raises(KeyError,match="SATURN_RUN_CONFIG"):
        runpy.run_path(str(path),run_name="__main__")
