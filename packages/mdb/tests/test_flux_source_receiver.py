"""Small causal/locality and recipe mechanics; no pretrained local execution."""
import runpy
from pathlib import Path

import numpy as np
import pytest

Q = Path(__file__).parents[1]/'qualification'
H = runpy.run_path(str(Q/'flux_source_receiver.py'))


def test_sealed_forecast_cohort_serializes_public_immutable_tree(tmp_path):
    from mdb.forecast_transaction import ForecastTransaction, load_forecast_tree
    tx=ForecastTransaction(tmp_path/'tx',model_identity='model',input_identity='input',source_identity='source',
        frontier=dict(kind='architectural',boundary='entry'),execution_scope='tiny archive',
        budgets=dict(max_native_calls=1,max_bytes=1<<20,max_wall_ms=10000))
    value=np.arange(8,dtype=np.float32).reshape(1,2,4)
    archive=H['array'](tmp_path,value)
    prediction=dict(arm=dict(id='tiny'),records=[dict(outputs=dict(latent=dict(archive=archive)))])
    tx.predict('program',lambda _:dict(predictions=prediction));tx.seal()
    frozen=load_forecast_tree(tmp_path/'tx',H['read'](tmp_path/'tx/forecasts/program.json')['predictions'])
    H['save'](tmp_path/'cohort.json',dict(programs={'tiny':frozen}))
    assert H['read'](tmp_path/'cohort.json')==dict(programs={'tiny':prediction})
    assert np.array_equal(H['load_array'](tmp_path,frozen['records'][0]['outputs']['latent']['archive']),value)


def test_launch_hides_cuda_only_for_the_cpu_forecaster(monkeypatch):
    import json
    import os
    import subprocess
    import sys
    stage=runpy.run_path(str(Q/'stage_flux_source_receiver.py'))
    environment=H['helper']('submit_flux_initial_noise.py')['execution_intent']()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7')
    for phase,expected in [('forecast',''),('native','7')]:
        command=stage['launch_command'](dict(config=dict(phase=phase),execution_environment=environment))
        env_prefix=command[:command.index('uv')]
        actual=subprocess.check_output([*env_prefix,sys.executable,'-c',
            'import os,json;print(json.dumps(os.environ.get("CUDA_VISIBLE_DEVICES")))'],env=os.environ)
        assert json.loads(actual)==expected


def test_source_delta_reconstructs_left_and_preserves_receiver_choice():
    values = {k:np.full((1,4,2),v,np.float32) for k,v in
        [('c0-o0',1.),('c1-o0',3.),('c0-o1',8.),('c1-o1',16.)]}
    result = {a['id']:H['text_input'](values,a) for a in H['arms']()}
    assert np.array_equal(result['left.delta'],values['c1-o0'])
    assert np.all(result['right.delta']==10.)
    assert np.array_equal(result['right.sham'],values['c0-o1'])
    assert np.array_equal(result['right.donor'],values['c1-o1'])
    assert len(H['arms']())==7 and sum(a['kind']!='sham' for a in H['arms']())==6


def test_bf16_archive_roundtrip_and_tamper(tmp_path):
    v=np.array([[[1.,-2.],[.5,0.]]],np.float32)
    record=H['array'](tmp_path,v)
    assert np.array_equal(H['load_array'](tmp_path,record),v)
    (tmp_path/record['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):H['load_array'](tmp_path,record)


def test_bf16_view_identity_and_legacy_sealed_views(tmp_path):
    a=np.arange(8,dtype=np.float32).reshape(1,2,4)
    b=a.reshape(1,2,2,2)
    ra,rb=H['array'](tmp_path,a),H['array'](tmp_path,b)
    assert ra['path']!=rb['path']
    assert np.array_equal(H['load_array'](tmp_path,ra),a)
    assert np.array_equal(H['load_array'](tmp_path,rb),b)
    legacy=dict(ra,encoding='bf16-bits-v1',shape=list(b.shape))
    assert np.array_equal(H['load_array'](tmp_path,legacy),b)
    with pytest.raises(ValueError):H['load_array'](tmp_path,dict(ra,shape=list(b.shape)))
    with pytest.raises(ValueError):H['load_array'](tmp_path,dict(legacy,shape=[9]))


def test_recipe_bits_match_original_and_decoded_edits_cannot_mutate_cache(tmp_path):
    torch=pytest.importorskip('torch')
    from safetensors.torch import save_file
    model=torch.nn.Linear(4,3,bias=True).to(torch.bfloat16)
    path=tmp_path/'weights.safetensors';save_file(model.state_dict(),path)
    cp=H['helper']('flux_initial_noise_campaign.py')['CheckpointParameters'](tmp_path,path.name,H['sha'](path),model)
    record=cp.record(model.weight)
    bias=cp.record(model.bias)
    manifest=dict(schema='mdb-parameter-artifacts-v1',artifacts=list(cp.entries.values()))
    bank=H['RecipeBank'](manifest,tmp_path)
    actual=bank.load(record)
    assert np.array_equal(actual,model.weight.detach().float().numpy())
    actual[:]=0
    assert np.array_equal(bank.load(record),model.weight.detach().float().numpy())
    changed=dict(record,sha256='0'*64)
    with pytest.raises(ValueError):bank.load(changed)
    bounded=H['RecipeBank'](manifest,tmp_path,max_cache_bytes=24)
    first=bounded.load(record);bounded.load(bias);again=bounded.load(record)
    assert np.array_equal(first,again) and bounded.payload_reads==3 and bounded.peak_retained<=24
    path.touch()
    with pytest.raises(ValueError):bounded.load(record)


def test_native_single_observer_preserves_values_and_exposes_parallel_locality():
    torch=pytest.importorskip('torch')
    native=pytest.importorskip('diffusers.models.transformers.transformer_flux2')
    torch.set_num_threads(1)
    torch.manual_seed(127)
    block=native.Flux2SingleTransformerBlock(dim=8,num_attention_heads=2,attention_head_dim=4,mlp_ratio=3.).eval().to(torch.bfloat16)
    state=torch.randn(1,8,8,dtype=torch.bfloat16)
    changed=state.clone();changed[:,:4,0]+=1
    mod=torch.randn(1,24,dtype=torch.bfloat16)
    kwargs=dict(encoder_hidden_states=None,temb_mod=mod,image_rotary_emb=(torch.ones(8,4),torch.zeros(8,4)))
    original=native.dispatch_attention_fn
    with torch.inference_mode():expected=block(hidden_states=state,**kwargs)
    fields=[]
    for x in (state,changed):
        obs=H['SinglePorts'](block,native,[1,3,4,7])
        with obs,torch.inference_mode():actual=block(hidden_states=x,**kwargs)
        if x is state:assert torch.equal(expected,actual)
        assert native.dispatch_attention_fn is original
        fields.append(obs.fields)
    assert set(fields[0])=={'layernorm','modulated','q_raw','k_raw','q_norm','k_norm','q_rope','k_rope','v','attention','mlp_full','projection'}
    for key in ('layernorm','modulated','q_raw','k_raw','q_norm','k_norm','q_rope','k_rope','v','mlp_full'):
        assert np.array_equal(fields[0][key][:,2:],fields[1][key][:,2:])
    assert not np.array_equal(fields[0]['attention'][:,2:],fields[1]['attention'][:,2:])
    assert len(block.attn.to_out._forward_hooks)==len(block.attn.to_out._forward_pre_hooks)==0


def test_fifty_step_suffix_joins_the_existing_fifty_six_step_program(tmp_path):
    pytest.importorskip('torch')
    from mdb.tensor_archive import load_tensor_archive
    fixture=runpy.run_path(str(Path(__file__).with_name('test_flux_prompt_factorial.py')))
    model,adapter,sessions=fixture['tiny']()
    session=sessions['c0-o0'];session.continue_(56)
    before=session.capture(retain=False)
    two=H['helper']('flux_two_pass_campaign.py')
    transition=H['helper']('flux_transition_reference.py')
    boundary=H['helper']('flux_step_boundary_reference.py')
    reentry=H['helper']('flux_reentry_reference.py')
    cfg=dict(sample_indices=[0,2,3,6],image_sample_indices=[0,1,3],query_chunk=2,projection_chunk=5,neuron_chunk=5)
    writer=H['helper']('flux_step_boundary_campaign.py')['make_array_writer'](tmp_path,lambda:None)
    binding=two['bind_inputs'](model=model,pre_project=before,array=writer,counts={},prefix_ref={},config=cfg,production_profile=False)
    complete=two['predict_two_pass'](binding,archive_root=tmp_path,transition=transition,boundary=boundary,reentry=reentry,config=cfg,event=lambda *a,**k:None)
    def wide(x):
        if isinstance(x,dict):return {k:wide(v) for k,v in x.items()}
        if isinstance(x,tuple):return tuple(wide(v) for v in x)
        return (x.float() if x.is_floating_point() else x).numpy() if hasattr(x,'numpy') else x
    p=wide(before.payload)
    first=complete['passes'][0]
    p.update(image=first['joints']['4']['image'],text=first['joints']['4']['text'],context=first['project']['context'])
    class Bank:
        load=staticmethod(lambda r:None if r is None else load_tensor_archive(tmp_path,r))
    calls=[]
    hooks=[block.register_forward_pre_hook(lambda *a:calls.append(True)) for block in [*model.transformer_blocks,*model.single_transformer_blocks]]
    try:
        result=list(H['suffix_events'](p,binding,lambda r:load_tensor_archive(tmp_path,r),Bank(),cfg))
    finally:
        for hook in hooks:hook.remove()
    assert not calls and len(result)==50
    assert [(q,l) for q,l,_,_ in result]==[(2,i) for i in range(6,28)]+[(3,i) for i in range(28)]
    assert np.array_equal(result[0][2]['image'],first['single_steps'][0]['prediction']['output'][:,3:])
    assert np.array_equal(result[-1][2]['latent'],complete['passes'][-1]['commit']['latent'])
    assert p['img_ids'].dtype==np.int64
    assert 'indices' not in result[0][3] and 'neuron_indices' not in result[0][3]
    for _,_,_,ports in result:
        for value in ports.values():
            assert np.array_equal(H['load_array'](tmp_path,H['array'](tmp_path,value)),value)
