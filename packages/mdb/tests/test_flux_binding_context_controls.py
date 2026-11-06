"""Contextual controls must follow changed content, rather than chat prefixes."""
import copy
from pathlib import Path
import runpy


H=runpy.run_path(str(Path(__file__).parents[1]/'qualification/flux_binding_context_controls.py'))


def fixture():
    base='A red sphere and a blue cube on white.'
    source='A blue sphere and a red cube on white.'
    context=dict(prompts=dict(base=base,source=source))
    def tokens(prompt,ids):
        return dict(ids=ids,mask=[1]*8+[0],
            offsets=[[0,1],[22,25],[26,32],[33,36],[39,43],[44,48],[49,51],[100,110],[0,0]],
            pieces=['<|im_start|>','red','sphere','and','blue','cube','on','<|im_end|>','<|endoftext|>'],
            attribute_spans=[dict(attribute=a,start=20+prompt.index(a))for a in ('red','blue')])
    return context,dict(base=tokens(base,[1,2,3,4,5,6,7,8,0]),source=tokens(source,[1,9,3,4,10,6,7,8,0]))


def test_select_contextual_unchanged_content_after_both_attributes():
    context,tokens=fixture()
    selected=H['select_rows'](context,tokens,dict(changed=[1,4]))
    assert selected['rows']==[5,6]
    assert all(i>selected['last_changed_id_row']for i in selected['rows'])
    assert selected['pieces']==['cube','on']


def test_special_padding_and_source_outside_prompt_never_substitute():
    context,tokens=fixture()
    tokens['source']['offsets'][6]=[100,105]
    try:H['select_rows'](context,tokens,dict(changed=[1,4]))
    except ValueError:pass
    else:raise AssertionError('invalid source-content row admitted')
    context,tokens=fixture()
    tokens['base']['pieces'][6]='<|im_end|>'
    try:H['select_rows'](context,tokens,dict(changed=[1,4]))
    except ValueError:pass
    else:raise AssertionError('special row admitted')


def test_selector_and_count_integrity():
    context,tokens=fixture()
    try:H['select_rows'](context,tokens,dict(changed=[1]))
    except ValueError:pass
    else:raise AssertionError('changed-ID scope silently changed')
    counts=H['expected_counts']()
    assert counts['native_transitions']==448 and counts['native_blocks']==400
    assert counts['acts']==16 and counts['receipts']==464 and counts['images']==8
    assert counts['encodes']==counts['noise_draws']==0
