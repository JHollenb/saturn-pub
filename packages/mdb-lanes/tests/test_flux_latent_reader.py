"""Tiny native prefix/reader/closure qualification; no pretrained model or VAE."""
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from saturn_pub import Act, Session
from saturn_pub.values import describe

Q=Path(__file__).parents[1]/'qualification'
C=runpy.run_path(str(Q/'flux_latent_reader.py'))


def test_complete_native_prefix_forecast_closure_and_key_value_campaign(tmp_path):
    fixture=runpy.run_path(str(Path(__file__).with_name('test_flux_prompt_factorial.py')))
    model,adapter,sessions=fixture['tiny']();model.requires_grad_(False)
    commits={};projects={};known={}
    for receiver,source in [('left','c0-o0'),('right','c0-o1')]:
        session=sessions[source];session.continue_(84)
        commits[receiver+'.base']=session.capture(retain=False)
        changed=Session.from_cut(adapter,commits[receiver+'.base'])
        changed.apply(Act.replace('latent',(changed.read('latent')+.125).to(torch.bfloat16)))
        commits[receiver+'.delta']=changed.capture(retain=False)
        for kind in ('base','delta'):
            s=Session.from_cut(adapter,commits[receiver+'.'+kind]);s.continue_()
            projects[receiver+'.'+kind]=s.capture(retain=False);s.continue_()
            known[receiver+'.'+kind]=s.read('text')
    weights={k:v.detach().float().numpy().copy() for k,v in model.transformer_blocks[0].state_dict().items()
        if k.startswith(('attn.to_add_out.','ff_context.linear_in.','ff_context.linear_out.'))}
    import diffusers.models.transformers.transformer_flux2 as native
    original=native.dispatch_attention_fn;decoded=[]
    out=tmp_path/'result'
    def decode(session,key):
        decoded.append(key)
        bits=hashlib.sha256(session.read('latent').float().numpy().tobytes()).digest()
        image=Image.fromarray(np.full((2,2,3),list(bits[:3]),np.uint8));path=out/(key+'.png');image.save(path)
        return dict(path=path.name,sha256=C['H']['sha'](path))
    result=C['run_campaign'](model=model,adapter=adapter,commits=commits,projects=projects,weights=weights,
        output=out,config=dict(layernorm_eps=1e-6,query_chunk=2,projection_chunk=5),request=dict(request_sha256='tiny'),decode=decode)
    assert native.dispatch_attention_fn is original
    assert result['counts']['native_transitions']==440 and result['counts']['native_sdpa']==400
    assert result['counts']['prefix_stops']==4 and result['counts']['completed_blocks']==400
    cohort=json.loads((out/'cohort.json').read_text());assert cohort['counts']['native_sdpa']==0
    closure=json.loads((out/'closure.json').read_text());readers=json.loads((out/'readers.json').read_text())
    for receiver in ('left','right'):
        assert closure[receiver+'.restore']['final_payload']==closure[receiver+'.base']['final_payload']
        assert closure[receiver+'.transplant']['final_payload']==closure[receiver+'.delta']['final_payload']
        for corner,kind in [('k0v0','base'),('k1v1','delta')]:
            actual=C['H']['load_array'](out,readers[receiver+'.'+corner]['fields']['text'])
            assert np.array_equal(actual,known[receiver+'.'+kind].float().numpy())
        for kind in ('base','delta'):
            assert describe(projects[receiver+'.'+kind].payload['text'])==describe(projects[receiver+'.base'].payload['text'])
    assert len(decoded)==16
    for block in [*model.transformer_blocks,*model.single_transformer_blocks]:
        assert not block._forward_pre_hooks and not block._forward_hooks
    # Exercise the exact saved-only auditor and public equation catalog on this
    # synthetic fixture. The census's one load denotes this tiny model only.
    save=C['H']['save'];calibrations={}
    for key,value in known.items():
        calibrations[key]=dict(text=C['H']['array'](out,value),image=closure[key]['image'])
    config=dict(model='tiny',projects={k:v.fingerprint for k,v in projects.items()},
        scope='Synthetic tiny fixture, fake image consumer; no pretrained scientific evidence.')
    request=dict(config=config,request_sha256='tiny')
    save(out/'known-calibrations.json',dict(records=calibrations))
    save(out/'request.json',request);save(out/'plan.json',config)
    save(out/'summary.json',dict(job_id='tiny',request_sha256='tiny',cohort_sha256=result['cohort_sha256'],
        counts={**result['counts'],'model_loads':1}))
    save(out/'files.json',{p.relative_to(out).as_posix():C['H']['sha'](p) for p in out.rglob('*') if p.is_file()})
    audited=runpy.run_path(str(Q/'analyze_flux_latent_reader.py'))['audit'](out,tmp_path/'audit.json')
    assert audited['mechanics_status']=='verified'
    assert all(audited['text_identity_exact'].values())
    errors=json.loads((tmp_path/'audit.json').read_text())['errors']
    assert max(e['relative_to_actual_rms'] for row in errors.values() for e in row.values())<.05
    published=runpy.run_path(str(Q/'publish_flux_latent_reader.py'))['publish'](out,tmp_path/'audit.json',tmp_path/'records')
    assert published['occurrences']==14 and published['changed_horizon_abstentions']==14
    verified=runpy.run_path(str(Q/'verify_flux_latent_reader.py'))['verify'](out,tmp_path/'clocks.json')
    assert verified['native_clocks']==440
