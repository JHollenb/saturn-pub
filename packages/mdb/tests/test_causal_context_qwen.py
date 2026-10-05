from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest
import torch

from mdb.ar.causal_context_qwen import (
    CausalContextQwenError,
    CausalContextQwenRuntime,
    cache_prefix_digest,
)
from mdb.ar.qwen_shared_ancestor_attention import qwen_cache_layer_tensors
from mdb.causal_radix_store import CausalRadixStore, RadixStoreError
from mdb.model_os_causal_context import (
    CausalContextQwenModelBackend,
    TOKEN_ADDRESS,
    build_causal_context_package,
)
from mdb.model_os_kernel import ModelOSKernel


def test_radix_mount_reaches_full_qwen_continuation_without_flat_cache(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(7)
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = transformers.Qwen2ForCausalLM(config).eval().to(dtype=torch.bfloat16)
    abi = sha256(b"tiny-qwen2-bf16-test").hexdigest()
    store = CausalRadixStore(tmp_path / "radix")
    runtime = CausalContextQwenRuntime(model, store, model_abi_sha256=abi, backend="torch")
    assert runtime.codec.head_dim == 8
    with torch.inference_mode():
        prefix = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True).past_key_values
        caches = {}
        for name, suffix in (("left", (5, 6)), ("right", (5, 7))):
            cache = deepcopy(prefix)
            for token in suffix:
                model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            caches[name] = cache
        assert cache_prefix_digest(caches["left"], 2, 4) == cache_prefix_digest(
            caches["right"], 2, 4
        )
        rows = {name: runtime.codec.encode_cache_rows(cache) for name, cache in caches.items()}
        mount_id = store.publish(model_abi_sha256=abi, pages=rows)
        states = tuple(runtime.mount(mount_id, name, extra_rows=2) for name in caches)
        assert states[0].ancestor is states[1].ancestor
        assert states[0].node_sha256s[0] == states[1].node_sha256s[0]
        assert states[0].ancestor_rows == 5
        assert states[0].delta_rows == 1
        assert runtime.physical_bytes(states)["ancestor_bytes"] > 0
        for cache, state in zip(caches.values(), states):
            for layer in range(runtime.codec.layers):
                native_key, native_value = qwen_cache_layer_tensors(cache, layer)
                for placed, native in zip(state.ancestor[layer], (native_key, native_value)):
                    assert torch.equal(placed, native[:, :, :state.ancestor_rows, :])
                for placed, native in zip(state.delta[layer], (native_key, native_value)):
                    assert torch.equal(
                        placed[:, :, :state.delta_rows, :], native[:, :, state.ancestor_rows:, :]
                    )
        package = build_causal_context_package(
            runtime, model_id="tiny-qwen2", tokenizer_identity="tiny-test-tokens",
        )
        backend = CausalContextQwenModelBackend(runtime, states[0], package)
        kernel = ModelOSKernel.install(package, backend)
        parent = kernel.current_statecut
        refused = kernel.execute(
            updates={TOKEN_ADDRESS: 9}, horizon=1,
            verifier=lambda _observation: {"accepted": False, "reason": "test rollback"},
        )
        assert refused.status == "restored"
        assert kernel.current_statecut.handle == parent.handle
        assert states[0].delta_rows == 1
        parent_visible_key = states[0].delta[0][0][:, :, :1, :].clone()
        accepted = kernel.execute(updates={TOKEN_ADDRESS: 9}, horizon=1)
        assert accepted.status == "committed"
        assert kernel.current_statecut.parent_handle == parent.handle
        assert backend.latest_private_output[0].shape == (1, 64)
        assert backend.current_mount.delta is states[0].delta
        assert torch.equal(states[0].delta[0][0][:, :, :1, :], parent_visible_key)
        committed_mount = backend.current_mount
        first_future_key = committed_mount.delta[0][0].clone()
        kernel.restore(parent)
        alternative = kernel.execute(updates={TOKEN_ADDRESS: 10}, horizon=1)
        assert alternative.status == "committed"
        assert backend.current_mount.delta is not committed_mount.delta
        assert torch.equal(committed_mount.delta[0][0], first_future_key)
        for (name, cache), state in zip(caches.items(), states):
            baseline = model(
                torch.tensor([[9]]), past_key_values=cache, use_cache=True,
                logits_to_keep=1,
            ).logits[:, -1, :]
            candidate, trace = runtime.step(state, 9, monitor=True)
            assert candidate.shape == baseline.shape
            assert len(trace["layers"]) == 2
            assert torch.allclose(candidate.float(), baseline.float(), atol=0.1, rtol=0.1)
            before = state.delta_rows
            fork = state.fork()
            runtime.step(fork, 10)
            assert state.delta_rows == before
            assert state.ancestor is fork.ancestor
        with pytest.raises(RadixStoreError, match="ABI"):
            store.read(mount_id, "left", model_abi_sha256="0" * 64)
        states[0].model_abi_sha256 = "0" * 64
        with pytest.raises(CausalContextQwenError, match="ABI"):
            runtime.step(states[0], 10)
