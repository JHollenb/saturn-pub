from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest
import torch

from mdb.ar.causal_context_qwen import CausalContextQwenError, CausalContextQwenRuntime
from mdb.bench_qwen import QwenBenchSession, tensor_digest
from mdb.causal_radix_store import CausalRadixStore
from mdb.model_os_causal_context import build_causal_context_package


def _model(model_type: str):
    transformers = pytest.importorskip("transformers")
    config_cls = transformers.Qwen3Config if model_type == "qwen3" else transformers.Qwen2Config
    model_cls = (
        transformers.Qwen3ForCausalLM
        if model_type == "qwen3"
        else transformers.Qwen2ForCausalLM
    )
    config = config_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    model = model_cls(config).eval().to(dtype=torch.bfloat16)
    model.config._attn_implementation = "eager"
    return model


def _model_surfaces(model):
    return {
        "parameters": [
            (name, id(value), value.data_ptr(), value._version)
            for name, value in model.named_parameters()
        ],
        "hooks": [
            (
                name,
                tuple(module._forward_hooks),
                tuple(module._forward_pre_hooks),
                tuple(module._backward_hooks),
            )
            for name, module in model.named_modules()
        ],
    }


def test_qwen3_native_eager_matches_hf_one_token_and_binds_distinct_abi(tmp_path):
    torch.manual_seed(53)
    model = _model("qwen3")
    before = _model_surfaces(model)
    abi = sha256(b"tiny-qwen3-bf16-native-eager").hexdigest()
    runtime = CausalContextQwenRuntime(
        model,
        CausalRadixStore(tmp_path / "radix"),
        model_abi_sha256=abi,
        backend="native-eager",
    )
    prompt = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        prefilled = model(prompt, use_cache=True)
        rows = runtime.codec.encode_cache_rows(prefilled.past_key_values)
        mount_id = runtime.store.publish(model_abi_sha256=abi, pages={"main": rows})
        state = runtime.mount(mount_id, "main", extra_rows=4)
        baseline_cache = deepcopy(prefilled.past_key_values)
        baseline = model(
            torch.tensor([[5]]),
            past_key_values=baseline_cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits[:, -1, :]
        candidate, trace = runtime.step(state, 5, monitor=True)
    assert runtime.codec.model_type == "qwen3"
    assert len(trace["layers"]) == 2
    assert torch.allclose(candidate.float(), baseline.float(), atol=0.02, rtol=0.02)
    package = build_causal_context_package(
        runtime, model_id="tiny-qwen3", tokenizer_identity="token-ids-only"
    )
    assert package.model_abi.family == "qwen3"
    assert package.model_abi.numerical_contract == "qwen3-bf16-hf-eager-v1"
    assert package.model_abi.graph_identity == "qwen3-one-token-full-layer-v1"
    assert package.consumers[0].consumer_id == "saturn.qwen3.causal-context-direct-read"
    assert package.package_id == "model-os.qwen3.causal-context-direct-read"
    assert _model_surfaces(model) == before
    with pytest.raises(CausalContextQwenError, match="only for native-eager"):
        CausalContextQwenRuntime(
            model,
            CausalRadixStore(tmp_path / "unqualified"),
            model_abi_sha256=abi,
            backend="torch",
        )


def test_qwen3_bench_step_checkpoint_restore_self_and_family_annotations(tmp_path):
    torch.manual_seed(59)
    model = _model("qwen3")
    before = _model_surfaces(model)
    bench = QwenBenchSession.from_resident(
        model,
        root=tmp_path / "bench",
        model_id="tiny-qwen3",
        revision="test",
        model_sha256=sha256(b"tiny-qwen3-checkpoint").hexdigest(),
        session_id="qwen3-native",
        lane="native",
        attention_backend="native-eager",
        continuation_capacity=16,
    )
    try:
        prompt = [1, 2, 3, 4]
        with torch.inference_mode():
            cold = model(torch.tensor([prompt]), use_cache=True).logits[:, -1, :]
        parent = bench.prefill(prompt)
        assert torch.allclose(
            bench.resolve_value("logits").float(), cold.float(), atol=0.0, rtol=0.0
        )
        assert bench.capabilities()["family"] == "qwen3"
        assert bench.vm.package.model_abi.family == "qwen3"
        assert bench.vm.package.model_abi.numerical_contract == "qwen3-bf16-hf-eager-v1"
        bench.stage_input(5)
        first = bench.step()
        first_logits = bench.resolve_value("logits").detach().clone()
        bench.restore(parent)
        assert bench.branch.token_ids == prompt
        bench.stage_input(5)
        second = bench.step()
        second_logits = bench.resolve_value("logits").detach().clone()
        assert first["token_id"] == second["token_id"] == 5
        assert tensor_digest(first_logits) == tensor_digest(second_logits)
        cut = bench.source_frames.cuts.query(bench.source_frames.inspect()["statecut_ref"])
        assert cut["family"] == "qwen3"
        assert _model_surfaces(model) == before
    finally:
        bench.close()


def test_qwen2_legacy_package_identities_are_unchanged(tmp_path):
    model = _model("qwen2")
    runtime = CausalContextQwenRuntime(
        model,
        CausalRadixStore(tmp_path / "radix"),
        model_abi_sha256=sha256(b"tiny-qwen2-legacy").hexdigest(),
        backend="native-eager",
    )
    package = build_causal_context_package(
        runtime, model_id="tiny-qwen2", tokenizer_identity="token-ids-only"
    )
    assert package.package_id == "model-os.qwen2.causal-context-direct-read"
    assert package.state_schema.name == "saturn-model-os-causal-context-qwen-v1"
    assert package.model_abi.family == "qwen2"
    assert package.model_abi.numerical_contract == "qwen2-bf16-hf-eager-v1"
    assert package.model_abi.graph_identity == "qwen2-one-token-full-layer-v1"
    assert package.consumers[0].consumer_id == "saturn.qwen2.causal-context-direct-read"
