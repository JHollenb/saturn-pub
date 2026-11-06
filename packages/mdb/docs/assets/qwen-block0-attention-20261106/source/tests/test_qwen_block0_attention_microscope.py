"""Recipe-backed tiny block0 mechanics; no original checkpoints or tokenizer."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import runpy
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from safetensors.torch import save_file
from transformers import Qwen3Config
from transformers.models.qwen3 import modeling_qwen3

from mdb import parameter_artifacts

HERE=Path(__file__).parents[1]
W=runpy.run_path(str(HERE/'qualification/qwen_block0_attention_microscope.py'))


def fixture(root):
    torch.set_num_threads(1)
    torch.manual_seed(127)
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    q=Qwen3Config(vocab_size=37,hidden_size=20,intermediate_size=24,num_hidden_layers=2,
                 num_attention_heads=4,num_key_value_heads=2,head_dim=4,max_position_embeddings=32,
                 attention_dropout=0.,attention_bias=False,pad_token_id=0,rope_theta=1000000)
    q._attn_implementation='sdpa'
    block=modeling_qwen3.Qwen3DecoderLayer(q,0).eval().to(torch.bfloat16)
    weights={'model.layers.0.'+name:value for name,value in block.state_dict().items()}
    weights['model.embed_tokens.weight']=torch.randn(q.vocab_size,q.hidden_size,dtype=torch.bfloat16)
    checkpoint=root/'checkpoint'
    checkpoint.mkdir()
    path=checkpoint/'tiny.safetensors'
    save_file(weights,str(path))
    header_bytes=path.read_bytes()
    header=json.loads(header_bytes[8:8+struct.unpack('<Q',header_bytes[:8])[0]])
    entries=[]
    for name,value in weights.items():
        array=value.contiguous().view(torch.uint16).numpy()
        buffer=io.BytesIO()
        np.save(buffer,array,allow_pickle=False)
        raw=array.tobytes()
        encoded=buffer.getvalue()
        entries.append({'path':'parameters/'+name+'.npy','bytes':len(encoded),'sha256':hashlib.sha256(encoded).hexdigest(),
                        'npy_prefix_b64':base64.b64encode(encoded[:-len(raw)]).decode(),
                        'source':{'file':path.name,'file_sha256':W['sha'](path),'tensor':name,'dtype':'BF16',
                                  'shape':list(value.shape),'data_offsets':header[name]['data_offsets'],
                                  'payload_sha256':hashlib.sha256(raw).hexdigest()}})
    W['save'](root/'qwen-config.json',q.to_dict())
    W['save'](root/'qwen-parameter-artifacts.json',{'schema':'mdb-parameter-artifacts-v1','artifacts':entries})
    shutil.copyfile(parameter_artifacts.__file__,root/'retrieve_weights.py')
    inputs=root/'inputs'
    inputs.mkdir()
    graph,_=W['helpers']()
    rotary=modeling_qwen3.Qwen3RotaryEmbedding(q,device=torch.device('cpu'))
    source_specs={}
    for case in W['CASES']:
        base=inputs/'cases'/case
        ids=torch.tensor([[3,7,9,11,13,17,0,0]]) if case=='short' else torch.tensor([[3,7,9,19,23,29,31,33]])
        mask=torch.tensor([[1,1,1,1,1,1,0,0]]) if case=='short' else torch.ones_like(ids)
        hidden=functional.embedding(ids,weights['model.embed_tokens.weight'])
        source_mask,_,_=graph['_mask_for_attention'](mask,batch_size=1,sequence_length=8,backend='sdpa',dtype=torch.bfloat16,device=ids.device)
        positions=torch.arange(8)[None]
        with torch.inference_mode():
            result=block(hidden_states=hidden,attention_mask=source_mask,position_ids=positions,
                         position_embeddings=rotary(hidden,positions),use_cache=False)
        for lane in ('forecast','native'):
            record={'input_ids':W['tensor_record'](base,lane+'/ids',ids),
                    'attention_mask':W['tensor_record'](base,lane+'/mask',mask),
                    'hidden_states':[W['tensor_record'](base,lane+'/H0',hidden),W['tensor_record'](base,lane+'/H1',result)]}
            W['save'](base/(lane+'-qwen.json'),record)
        source_specs[case]={lane+'_qwen':'cases/'+case+'/'+lane+'-qwen.json' for lane in ('forecast','native')}
    input_files={str(path.relative_to(inputs)):{'bytes':path.stat().st_size,'sha256':W['sha'](path)} for path in inputs.rglob('*') if path.is_file()}
    W['save'](inputs/'files.json',{'schema':'mdb-image-tail-files-v1','job_id':'tiny-parent','request_sha256':'a'*64,
                                 'files':input_files,'expanded_bytes':sum(row['bytes'] for row in input_files.values())})
    import transformers.integrations.sdpa_attention as sdpa
    import transformers.masking_utils as masking
    config={'schema':'mdb-qwen-block0-attention-plan-v1','profile':'tiny-cpu','device':'cpu','source_reduction_policy':True,
            'cases':source_specs,'input_root':str(inputs),'input_files':input_files,'parameter_root':str(checkpoint),
            'source_provenance':{'job_id':'tiny-parent','request_sha256':'a'*64,'manifest_sha256':W['sha'](inputs/'files.json')},
            'qwen_config':str(root/'qwen-config.json'),'qwen_manifest':str(root/'qwen-parameter-artifacts.json'),
            'retrieval_script':str(root/'retrieve_weights.py'),'fixed_bank_case':'long','shared_prefix_rows':3,
            'max_wall_seconds':60,'cpu_threads':1,'profile_sdpa':False,'output':str(root/'result'),
            'native_sources':{'qwen3':W['sha'](modeling_qwen3.__file__),'sdpa':W['sha'](sdpa.__file__),'masking':W['sha'](masking.__file__)}}
    for key in ('qwen_config','qwen_manifest','retrieval_script'):
        config[key+'_sha256']=W['sha'](config[key])
    return config


def test_actual_recipe_backed_two_native_blocks_and_three_triangle_arms(tmp_path):
    config=fixture(tmp_path)
    result=W['run'](config)
    assert result['actual']['sdpa_calls']==7
    assert result['actual']['native_block_calls']==2
    assert result['actual']['source_parameter_reads']==12
    rows=W['read'](Path(config['output'])/'sdpa.json')
    assert [row['label'] for row in rows]==['functional.short','functional.long',*W['ARMS'],'native.short','native.long']
    assert rows[2]['mask']['dtype']=='torch.bool' and not rows[2]['is_causal']
    assert rows[3]['mask'] is None and rows[3]['is_causal'] and not rows[3]['enable_gqa']
    assert rows[4]['mask'] is None and rows[4]['enable_gqa']
    cases=W['read'](Path(config['output'])/'cases.json')
    for case in W['CASES']:
        assert cases[case]['historical_replay']['native']['H0']['bitwise_exact']
        assert cases[case]['historical_replay']['native']['H1']['bitwise_exact']
        assert all(row['bitwise_exact'] for row in cases[case]['matched_functional_vs_native'].values())
    assert not list(Path(config['output']).rglob('*.safetensors'))
    assert W['read'](Path(config['output'])/'functional-seal.json')['native_block_calls']==0
    assert result['pair_comparisons']['native']['q']['bitwise_exact']


def test_exact_array_serializer_bf16_stride_integer_and_tamper(tmp_path):
    value=torch.tensor([[0.,-0.,1.,-2.],[3.,4.,5.,6.]],dtype=torch.bfloat16).t()
    record=W['tensor_record'](tmp_path,'value',value)
    assert record['stride']==list(value.stride())
    actual=W['load_input'](tmp_path,record,'cpu')
    assert torch.equal(actual.view(torch.uint16),value.contiguous().view(torch.uint16))
    byte=np.array([0,255,3],dtype=np.uint8)
    np.save(tmp_path/'bytes.npy',byte,allow_pickle=False)
    minimal={'path':'bytes.npy','bytes':(tmp_path/'bytes.npy').stat().st_size,'sha256':W['sha'](tmp_path/'bytes.npy'),'shape':[3],'dtype':'uint8'}
    assert W['load_input'](tmp_path,minimal,'cpu').dtype==torch.uint8
    (tmp_path/record['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='source array'):
        W['load_input'](tmp_path,record,'cpu')


def test_canonical_group_mapping_and_broken_repeated_bank():
    value=torch.arange(2*5*4,dtype=torch.float32).reshape(1,2,5,4)
    repeated=value[:,:,None].expand(1,2,4,5,4).reshape(1,8,5,4)
    assert torch.equal(W['canonical_kv'](repeated,2,4),value)
    broken=repeated.clone()
    broken[:,1,0,0]+=1
    with pytest.raises(ValueError,match='canonical duplicates'):
        W['canonical_kv'](broken,2,4)


def test_triangle_keeps_operands_and_does_not_enter_native_modules(tmp_path):
    config=fixture(tmp_path)
    graph,_=W['helpers']()
    q=Qwen3Config.from_dict(W['read'](config['qwen_config']))
    torch.manual_seed(77)
    bank={'q':torch.randn(1,4,8,4,dtype=torch.bfloat16),'k':torch.randn(1,2,8,4,dtype=torch.bfloat16),
          'v':torch.randn(1,2,8,4,dtype=torch.bfloat16),'H0':torch.randn(1,8,20,dtype=torch.bfloat16)}
    block=modeling_qwen3.Qwen3DecoderLayer(q,0).to(torch.bfloat16)
    weights={'model.layers.0.'+name:value for name,value in block.state_dict().items()}
    before={name:value.clone() for name,value in bank.items()}
    calls=[]

    def call(name,query,key,value,mask,causal,gqa,scale):
        calls.append((name,query,key,value,mask,causal,gqa))
        return functional.scaled_dot_product_attention(query,key,value,attn_mask=mask,is_causal=causal,enable_gqa=gqa,scale=scale)

    with torch.inference_mode(),graph['module_entry_guard'](block) as entries:
        result,_=W['triangle'](bank,weights,q,graph,call)
    assert entries==[] and len(calls)==3
    assert calls[0][1] is calls[1][1] is calls[2][1]
    assert calls[0][2] is calls[1][2]
    assert calls[2][2] is bank['k']
    assert all(torch.equal(value,before[name]) for name,value in bank.items())
    assert set(result)==set(W['ARMS'])


def test_profiler_failure_does_not_repeat_sdpa_and_recorder_restores(tmp_path,monkeypatch):
    calls=[]
    original=functional.scaled_dot_product_attention

    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)

    monkeypatch.setattr(functional,'scaled_dot_product_attention',counted)
    monkeypatch.setattr(torch.profiler,'profile',lambda **kwargs:(_ for _ in ()).throw(RuntimeError('unsupported profiler')))
    rows=[]
    q=torch.randn(1,2,3,4,dtype=torch.bfloat16)
    with W['sdpa_recorder'](rows,'tiny',output=tmp_path,profile=True):
        functional.scaled_dot_product_attention(q,q,q,is_causal=True)
    assert len(calls)==1 and rows[0]['profiler']['status']=='unavailable'
    assert functional.scaled_dot_product_attention is counted


def test_supported_cpu_profiler_observes_existing_call_without_repeat(tmp_path,monkeypatch):
    calls=[]
    original=functional.scaled_dot_product_attention

    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)

    monkeypatch.setattr(functional,'scaled_dot_product_attention',counted)
    rows=[]
    q=torch.randn(1,2,3,4,dtype=torch.bfloat16)
    with W['sdpa_recorder'](rows,'tiny',output=tmp_path,profile=True):
        functional.scaled_dot_product_attention(q,q,q,is_causal=True)
    assert len(calls)==1 and len(rows)==1
    assert rows[0]['profiler']['status'] in {'observed','unavailable','instrument-broken'}
    if rows[0]['profiler']['status']=='observed':
        assert any('scaled_dot_product' in name for name in rows[0]['profiler']['events'])


def test_signed_telescope_negative_cross_and_zero_alignment_are_not_gates():
    result=W['vector_account'](torch.tensor([2.,0.]),torch.tensor([-1.,1.]),torch.tensor([1.,1.]))
    assert result['signed_cross_term']<0 and result['energy_identity_residual']==0
    zero=W['vector_account'](torch.zeros(2),torch.ones(2),torch.zeros(2))
    assert zero['first_vs_historical_cosine'] is None


def test_actual_worker_cli_finalizes_fresh_directory(tmp_path):
    config=fixture(tmp_path)
    W['save'](tmp_path/'config.json',config)
    process=subprocess.run([sys.executable,'-I','-B',str(HERE/'qualification/qwen_block0_attention_microscope.py'),'--config',str(tmp_path/'config.json')],cwd=tmp_path,capture_output=True,text=True,check=False)
    assert process.returncode==0,process.stdout+'\n'+process.stderr
    assert 'MDB_DIRECTORY_READY ' in process.stdout
    manifest=W['read'](Path(config['output'])/'files.json')
    assert manifest['schema']=='mdb-image-tail-files-v1'
    assert 'cases.json' in manifest['files'] and 'triangle.json' in manifest['files']
