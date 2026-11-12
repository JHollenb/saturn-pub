"""Exercise the saved-operand worker with a tiny, explicitly synthetic corpus."""
import hashlib
import json
import runpy
import shutil
from pathlib import Path


def test_saved_attention_axes_and_model_free_report(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from mdb.tensor_archive import write_tensor_archive
    from mdb.attention_partition import dense_attention
    here=Path(__file__).resolve().parents[1]/'qualification'
    module=runpy.run_path(str(here/'flux_precision_attention.py'))
    B=runpy.run_path(str(here/'flux_reentry_reference.py'))['bf16']
    source=tmp_path/'source';source.mkdir();native={};rng=np.random.default_rng(31)
    for case in('half','half.image1'):
        stages={};operands=[]
        for label in('q_positioned','k_positioned','v_projection'):
            parts=[]
            for role,rows in(('text',3),('image',4)):
                value=B(rng.normal(size=(1,rows,2,4)))
                stages[role+'.'+label]=write_tensor_archive(source/(case+'.'+role+'.'+label+'.npy'),torch.from_numpy(value).to(torch.bfloat16),root=source)
                parts.append(value)
            operands.append(np.concatenate(parts,axis=1).transpose(0,2,1,3))
        output=B(dense_attention(*operands).transpose(0,2,1,3).reshape(1,7,8))
        stages['attention.raw']=write_tensor_archive(source/(case+'.raw.npy'),torch.from_numpy(output).to(torch.bfloat16),root=source)
        native[case]=dict(stages=stages)
    (source/'native.json').write_text(json.dumps(native));(source/'summary.json').write_text('{}')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    work=tmp_path/'payload';work.mkdir()
    for name in('flux_reentry_reference.py','analyze_flux_precision.py'):shutil.copyfile(here/name,work/name)
    config=dict(input_root=str(source),inputs={p.name:sha(p)for p in source.iterdir()},output_scratch=str(tmp_path/'output'))
    request=dict(config=config,source_hashes={p.name:sha(p)for p in work.iterdir()},request_sha256='synthetic-unit-only')
    (work/'request.json').write_text(json.dumps(request));monkeypatch.chdir(work);monkeypatch.setenv('SATURN_DEBUG_REQUEST',str(work/'request.json'));monkeypatch.setenv('MRUN_JOB_ID','synthetic-test')
    module['worker']()
    report=json.loads((tmp_path/'output/synthetic-test/binding-minimal-result/summary.json').read_text())
    assert report['model_loads']==report['native_forwards']==report['native_transitions']==0
    assert all(row['error']['exact']for row in report['reports'].values())
    assert report['receiver_contrast']['exact']
