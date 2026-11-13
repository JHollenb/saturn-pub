"""Clock comparisons distinguish value changes, layouts and absent slots."""
import runpy
from pathlib import Path

C=runpy.run_path(str(Path(__file__).parents[1]/'qualification/flux_reader_route_future_audit.py'))


def test_first_value_change_is_not_a_stride_change_or_missing_slot():
    def value(digest,stride=1):return dict(sha256=digest,dtype='torch.bfloat16',shape=[1,2],stride=[2,stride])
    def case(values):return dict(records=[dict(local=i,point=dict(index=i+1),fields={'image':v} if v is not None else {}) for i,v in enumerate(values)])
    base=case([value('a'),value('b'),None,value('c')])
    branch=case([value('a',2),value('changed'),None,value('c')])
    result=C['trajectory'](branch,base,'image')
    assert result['first_value_divergence']==1
    assert result['changed_by_local']==[False,True,None,False]
    assert result['exact_slot_reunions']==[3] and result['layout_difference_locals']==[0]


def test_typed_empty_working_slots_are_disposal_not_exact_repair():
    empty=dict(kind='scalar',type='none',value=None)
    def case(digest):return dict(records=[dict(local=0,point='joint1',fields=dict(text=dict(tensor=digest,dtype='torch.bfloat16',shape=[1,2]))),
        dict(local=1,point='project-next-pass',fields=dict(text=empty))])
    result=C['trajectory'](case('changed'),case('base'),'text')
    assert result['first_value_divergence']==0 and result['changed_by_local']==[True,None]
    assert result['exact_slot_reunions']==[] and result['slot_clearance_locals']==[1]
