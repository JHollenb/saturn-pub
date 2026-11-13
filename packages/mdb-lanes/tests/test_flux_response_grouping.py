"""Native commit grouping preserves the actual tiny FLUX program and state."""
import runpy
from pathlib import Path
from saturn_pub import Session
from saturn_pub.values import describe
from mdb_lanes.native_navigation import NativeCutNavigator


def test_grouped_commit_matches_every_native_transition_and_full_state():
    here=Path(__file__).resolve().parent
    adapter,_,reference,_=runpy.run_path(str(here/'test_flux_forecast_closure.py'))['fixture']()
    initial=reference.capture();grouped=Session.from_cut(adapter,initial)
    step_to_commit=runpy.run_path(str(here.parent/'qualification/flux_response_validate.py'))['steps_to_commit']
    nav=NativeCutNavigator(grouped,max_steps=14);native_calls=0;group_calls=0
    while nav.point['logical_step']<2:
        count=step_to_commit(nav.point,2,2);transitions=[]
        for _ in range(count):
            receipt=reference.continue_().to_dict();transitions.extend(receipt['transitions']);native_calls+=1
        grouped_receipt=nav.step(count).to_dict();group_calls+=1
        assert grouped_receipt['steps']==len(grouped_receipt['transitions'])==count
        for actual,expected in zip(grouped_receipt['transitions'],transitions,strict=True):
            for key in('before','after','native_operation','structural_operation'):assert actual[key]==expected[key]
        assert describe(grouped.capture(retain=False).payload)==describe(reference.capture(retain=False).payload)
    assert native_calls==nav.steps==14 and group_calls==2


def test_real_campaign_geometry_keeps_native_work_separate_from_receipt_count():
    worker=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/flux_response_validate.py'))
    f=worker['steps_to_commit']
    assert [f(dict(phase='joint',index=2),5,20),f(dict(phase='project',index=None),5,20)]==[25,28]
    counts=worker['expected_counts']()
    assert counts['native_transitions']==53+4*(53+54)==481
    assert counts['receipts']==4+4*(4+5)==40
