"""Tiny native closure and exact scheduler-law checks for legal dependency cuts."""
import runpy
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from saturn_pub import Session
from saturn_pub.values import describe

from mdb_lanes.native_navigation import NativeCutNavigator

HERE=Path(__file__).parents[1]
q=runpy.run_path(str(HERE/"qualification/flux_scene_dependencies.py"))
tiny=runpy.run_path(str(HERE/"tests/test_flux_single_contexts.py"))


def test_legal_project_reconstruction_reproduces_current_frontier():
    _model,adapter,session=tiny["tiny_session"]()
    n=NativeCutNavigator(session,max_steps=112);n.until(logical_step=3,phase="single",index=6)
    source=n.capture();root=q["project_frontier"](adapter,source)
    assert root.execution_point["phase"]=="project" and root.execution_point["logical_step"]==3
    restored=NativeCutNavigator(Session.from_cut(adapter,root),max_steps=12)
    for _ in range(12):restored.step()
    assert describe(restored.capture().payload)==describe(source.payload)


def test_numpy_euler_predicts_actual_scheduler_with_bf16_product_rounding():
    generator=torch.Generator().manual_seed(81)
    for factor in (.3,.182137,.125):
        sample=torch.randn(1,16,12,generator=generator).to(torch.bfloat16)
        velocity=torch.randn(1,16,12,generator=generator).to(torch.bfloat16)
        s=FlowMatchEulerDiscreteScheduler()
        s.timesteps=torch.tensor([1000.,800.,600.,300.]);s.sigmas=torch.tensor([1.,.8,.6,factor,0.])
        s.set_begin_index(3)
        actual=s.step(velocity,s.timesteps[3],sample,return_dict=False)[0]
        prediction=q["predict_euler"](sample.float().numpy(),velocity.float().numpy(),-float(s.sigmas[3]))
        np.testing.assert_array_equal(prediction,actual.float().numpy())


def test_forecast_terminal_is_native_decodable_state_without_transformer_steps():
    _model,adapter,session=tiny["tiny_session"]()
    n=NativeCutNavigator(session,max_steps=112);n.until(logical_step=4,phase="project")
    final=n.capture();latent=final.payload["latent"]+torch.tensor(.125,dtype=torch.bfloat16)
    predicted=q["terminal_from_forecast"](adapter,final,latent,parent=final.fingerprint)
    assert predicted.capture().execution_point["logical_step"]==4
    assert not predicted.receipts
    assert torch.equal(predicted.read("latent"),latent)


def test_continuation_observer_reads_only_applicable_live_ports():
    _model,adapter,session=tiny["tiny_session"]()
    n=NativeCutNavigator(session,max_steps=112);n.until(logical_step=3,phase="single",index=6)
    points=[]
    while True:
        values=q["observe_live"](n);points.append((n.point,values))
        if n.point["logical_step"]==4:break
        n.step()
    assert len(points)==17
    for point,values in points:
        assert (values["noise_prediction"]["kind"]=="tensor")== (point["phase"]=="scheduler")
    assert points[-1][1]["image"]==describe(None)
