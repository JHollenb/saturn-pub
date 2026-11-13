"""Stream-factorial identities and source qualification are instrument contracts."""
import hashlib
import json
import runpy
from pathlib import Path
import numpy as np
import pytest


def helper(name):return runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification'/name))


def test_factorial_uses_own_baselines_and_retains_interaction():
    m=helper('flux_stream_cross_math.py');power=helper('flux_own_state_mlp_math.py')['power_account']
    worlds={(name,corner):np.array([v],np.float32) for name,values in [('half',dict(zip(('00','10','01','11'),(2,4,3,6)))),('edit',dict(zip(('00','10','01','11'),(6,12,10,20))))] for corner,v in values.items()}
    result=m['factorial'](worlds,dict(half=np.array([0.]),edit=np.array([5.])),'edit',power)
    assert result['program_change']['component_mse']==dict(image_switch=16.,text_switch=9.,image_text_interaction=9.)
    assert result['program_change']['total_mse']==100 and result['revised_error']['total_mse']==81
    assert result['revised_error']['identity_residual_rms']==0
    assert result['revised_error']['twice_signed_cross_moments']['old_error × image_switch']==-8


def test_qualified_source_binds_inherited_audit_and_fresh_consumed_bytes(tmp_path):
    p=tmp_path/'collection';root=p/'extracted/binding-minimal-result';root.mkdir(parents=True);item=root/'value.json';item.write_text('{"value":1}\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();record=dict(verification=dict(job_id='source'),files={'value.json':dict(bytes=item.stat().st_size,sha256=sha(item))})
    (p/'remote-custody.json').write_text(json.dumps(record));audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(mechanics_status='verified',custody=record['verification'])))
    source=helper('flux_qualified_source.py')['QualifiedSource'](p,sha(p/'remote-custody.json'),audit,sha(audit));assert source.json('value.json')==dict(value=1)
    assert source.report()['inherited_audit_sha256']==sha(audit)
    item.write_text('{"value":2}\n')
    with pytest.raises(ValueError,match='consumed source'):source.json('value.json')
    with pytest.raises(ValueError,match='unsafe'):source.member('../outside')


def test_source_cut_verifies_the_complete_referenced_page_blob_closure(tmp_path):
    root=tmp_path/'collection';data=root/'extracted/binding-minimal-result';fp='abc';page='def';blob='012'
    files={'store/cuts/'+fp+'.json':json.dumps(dict(fingerprint=fp,slots=dict(image=page))).encode(),'store/pages/'+page+'.json':json.dumps(dict(tensor_blob=blob)).encode(),'store/blobs/'+blob+'.safetensors':b'tensor-bytes'}
    for name,raw in files.items():p=data/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();record=dict(verification=dict(job_id='source'),files={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in files.items()})
    (root/'remote-custody.json').write_text(json.dumps(record));audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(mechanics_status='verified',custody=record['verification'])))
    source=helper('flux_qualified_source.py')['QualifiedSource'](root,sha(root/'remote-custody.json'),audit,sha(audit));source.cut(fp)
    assert set(source.report()['files'])==set(files)
    (data/('store/blobs/'+blob+'.safetensors')).write_bytes(b'broken-bytes')
    with pytest.raises(ValueError,match='consumed source'):source.cut(fp)
