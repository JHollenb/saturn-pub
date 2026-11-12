"""Small model-free mechanics: real own-state consumer, prewrite refusal, skip."""
from dataclasses import replace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb.ar.causal_context_qwen import CausalContextQwenRuntime, CausalContextQwenMount
from mdb.qwen_state_span import QwenStateSpanProducer, install_span, use_state_span


def fixture():
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=24, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64, attn_implementation="eager")).eval().to(torch.bfloat16).requires_grad_(False)
    runtime = CausalContextQwenRuntime(model, None, model_abi_sha256="a"*64, backend="native-eager")
    empty = tuple((torch.empty(1,2,0,4,dtype=torch.bfloat16),torch.empty(1,2,0,4,dtype=torch.bfloat16)) for _ in range(4))
    delta = tuple((torch.zeros(1,2,24,4,dtype=torch.bfloat16),torch.zeros(1,2,24,4,dtype=torch.bfloat16)) for _ in range(4))
    state = CausalContextQwenMount("m", "p", "a"*64, ("n",), empty, delta, 0, 0, 24)
    return runtime, state


def test_dense_control_closes_recurring_own_token_and_later_ingress():
    runtime, parent = fixture()
    a,b = parent.fork(),parent.fork()
    producer = QwenStateSpanProducer(runtime,1,3)
    token = 4
    for i in range(8):
        # Two later caller tokens are actually ingested into each owned history.
        token = {3:9,4:11}.get(i,token)
        left,_ = runtime.step(a,token)
        with use_state_span(runtime,producer):
            right,_ = runtime.step(b,token)
        assert torch.equal(left,right)
        assert a.delta_rows == b.delta_rows == i+1
        for x,y in zip(a.delta,b.delta):
            for k,v in zip(x,y): assert torch.equal(k,v)
        token = int(right.argmax())
    assert parent.length == 0


def test_packet_refusal_is_prewrite_and_rank_producer_skips_original_modules():
    runtime,state = fixture()
    producer = QwenStateSpanProducer(runtime,1,3,bases={i:torch.eye(16)[:4] for i in (1,2)})
    execution = runtime.begin_token(state,4)
    runtime.advance_token(execution)
    packet = producer.predict(execution)
    before = tuple(tuple(t.clone() for t in p) for p in state.delta)
    bad = replace(packet, rows=packet.rows[:-1]+((packet.rows[-1][0].float(),packet.rows[-1][1]),))
    with pytest.raises(ValueError): install_span(runtime,execution,bad)
    assert execution.next_layer == 1 and state.delta_rows == 0
    for p,q in zip(before,state.delta):
        for x,y in zip(p,q): assert torch.equal(x,y)
    install_span(runtime,execution,packet)
    assert execution.next_layer == 3 and state.delta_rows == 0
    assert execution.hidden is packet.hidden
    with pytest.raises(ValueError): install_span(runtime,execution,packet)
    def refuse(*args): raise AssertionError("native span module entered")
    hooks = [module.register_forward_pre_hook(refuse) for i in (1,2) for module in runtime.model.model.layers[i].modules()]
    try:
        with use_state_span(runtime,producer): runtime.step(state,7)
    finally:
        for hook in hooks: hook.remove()
    assert state.delta_rows == 1


def test_frozen_basis_and_inference_bank_contract():
    runtime,state=fixture()
    basis={i:torch.eye(16)[:4] for i in (1,2)}
    producer=QwenStateSpanProducer(runtime,1,3,bases=basis)
    fixed=producer.factors[1][0].clone()
    basis[1].zero_()
    assert torch.equal(producer.factors[1][0],fixed)
    with torch.inference_mode():
        state.delta=tuple(tuple(t.clone() for t in p) for p in state.delta)
        execution=runtime.begin_token(state,5)
        runtime.advance_token(execution)
        packet=producer.predict(execution)
        execution.hidden.add_(1)
    with pytest.raises(ValueError,match="stale"): install_span(runtime,execution,packet)
    with use_state_span(runtime,producer):
        with pytest.raises(ValueError,match="monitor"):
            runtime.step(state,5,monitor=True)
