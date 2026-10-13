"""Tiny native forwards: edits propagate and temporary hooks never leak."""

import pytest

from mdb import BenchError, BenchSession

pytest.importorskip("torch")


def tiny_qwen():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(91)
    return transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            attn_implementation="eager",
        )
    ).eval()


def test_transformers_capture_edit_propagation_and_cleanup():
    import torch

    model = tiny_qwen()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    point = "module:model.embed_tokens:output"
    session = BenchSession.from_backend(
        "transformers", model=model, identity={"model_id": "random-qwen"}
    )
    original = model(**inputs).logits
    result = session.forward(inputs, capture=[point], interventions={point: {"op": "suppress"}})
    assert torch.count_nonzero(result.captured[point]) == 0
    assert not torch.equal(result.output.logits, original)
    assert torch.equal(session.forward(inputs).output.logits, original)
    before = len(model.model.embed_tokens._forward_hooks)

    def broken(value):
        raise RuntimeError("failed edit")

    with pytest.raises(RuntimeError, match="failed edit"):
        session.forward(inputs, interventions={point: broken})
    assert len(model.model.embed_tokens._forward_hooks) == before
    with pytest.raises(BenchError, match="shape"):
        session.forward(inputs, interventions={point: lambda value: value[:, :1]})
    with pytest.raises(BenchError, match="non-finite"):
        session.forward(inputs, interventions={point: lambda value: value * float("nan")})
    assert torch.equal(session.forward(inputs).output.logits, original)
    with pytest.raises(BenchError, match="unknown intervention fields"):
        session.forward(inputs, interventions={point: {"op": "suppress", "position": 0}})


def test_transformers_retains_caller_hooks_and_gradients():
    import torch

    model = tiny_qwen()
    seen = []
    handle = model.model.embed_tokens.register_forward_hook(lambda *args: seen.append(1))
    try:
        session = BenchSession.from_backend(
            "transformers", model=model, identity={"model_id": "random-qwen"}
        )
        point = "module:model.embed_tokens:output"
        result = session.forward(
            {"input_ids": torch.tensor([[1, 2]])},
            interventions={point: {"op": "scale", "value": 0.5}},
        )
        result.output.logits.sum().backward()
        assert model.model.embed_tokens.weight.grad is not None
        assert seen == [1] and len(model.model.embed_tokens._forward_hooks) == 1
    finally:
        handle.remove()


def test_existing_resident_adapter_preserves_native_cut_registry(tmp_path):
    import torch

    from mdb.bench_debugger import BenchDebuggerAdapter
    from mdb.bench_verify import verify

    model = tiny_qwen().to(torch.bfloat16)
    resident = BenchSession.from_resident(model, root=tmp_path / "native", continuation_capacity=8)
    resident.prefill([1, 2, 3])
    with BenchSession.from_backend(
        BenchDebuggerAdapter(resident), identity=resident.identity
    ) as session:
        parent = session.checkpoint("registered-parent")["cut_id"]
        before = resident.cursor()
        session.step(1)
        assert resident.cursor() != before
        session.restore(parent)
        assert resident.cursor() == before
        assert parent in resident.cuts
        assert session.debugger().capabilities()["exact_replay"]
        with pytest.raises(BenchError, match="positive"):
            session.step(-1)
    assert not resident.closed
    resident.export(tmp_path / "export")
    resident.close()
    assert verify(tmp_path / "export")["mechanics_status"] == "valid"


def tiny_lens():
    lens = pytest.importorskip("transformer_lens")
    # v3 legacy and v4 native bridges have distinct constructors/hook names.
    try:
        from transformer_lens.config import TransformerBridgeConfig
        from transformer_lens.model_bridge import TransformerBridge

        config = TransformerBridgeConfig(
            n_layers=1,
            d_model=16,
            n_ctx=16,
            d_head=8,
            n_heads=2,
            d_mlp=32,
            d_vocab=32,
            act_fn="relu",
            device="cpu",
            seed=91,
        )
        return TransformerBridge.boot_native(config), "blocks.0.hook_in"
    except (ImportError, AttributeError):
        config = lens.HookedTransformerConfig(
            n_layers=1,
            d_model=16,
            n_ctx=16,
            d_head=8,
            n_heads=2,
            d_mlp=32,
            d_vocab=32,
            act_fn="relu",
            device="cpu",
            seed=91,
        )
        return lens.HookedTransformer(config), "blocks.0.hook_resid_pre"


def test_real_transformerlens_temporary_hooks_and_recovery():
    import torch

    model, point = tiny_lens()
    inputs = torch.tensor([[1, 2, 3]])
    session = BenchSession.from_backend(
        "transformerlens", model=model, identity={"model_id": "random-lens"}
    )
    original = session.forward(inputs).output
    result = session.forward(inputs, capture=[point], interventions={point: {"op": "suppress"}})
    assert torch.count_nonzero(result.captured[point]) == 0
    assert not torch.equal(result.output, original)
    assert torch.equal(session.forward(inputs).output, original)

    def broken(value):
        raise RuntimeError("failed edit")

    with pytest.raises(RuntimeError, match="failed edit"):
        session.forward(inputs, interventions={point: broken})
    assert torch.equal(session.forward(inputs).output, original)
    with pytest.raises(BenchError, match="lifecycle"):
        session.forward(inputs, reset_hooks_end=False)


def test_non_firing_module_and_lens_hooks_refuse():
    import torch

    from mdb import TransformerLensRunner

    class Conditional(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unused = torch.nn.Identity()

        def forward(self, value):
            return value

    session = BenchSession.from_backend(
        "transformers", model=Conditional(), identity={"model_id": "conditional"}
    )
    with pytest.raises(BenchError, match="did not fire"):
        session.forward(torch.ones(2), interventions={"module:unused:output": {"op": "suppress"}})

    class BrokenLens:
        hook_dict = {"hidden": object()}

        def run_with_hooks(self, inputs, **kwargs):
            return inputs

    runner = TransformerLensRunner(BrokenLens())
    with pytest.raises(BenchError, match="did not fire"):
        runner.forward(torch.ones(2), interventions={"hidden": {"op": "suppress"}})


def test_declarative_only_lens_driver_never_uses_observation_hooks():
    import torch

    from mdb import TransformerLensRunner

    class Driver:
        supported_hook_points = {"hidden"}
        non_fireable_hook_points = {"fused"}

        def supports(self, feature):
            return False

    class Remote:
        hook_dict = {"hidden": object(), "fused": object()}
        _driver = Driver()
        calls = 0

        def run_with_hooks(self, *args, **kwargs):
            raise AssertionError("observation hooks cannot intervene on remote execution")

        def run_with_cache(self, inputs, *, intervene, names_filter, **kwargs):
            self.calls += 1
            spec = intervene["hidden"]
            assert spec == {"op": "scale", "factor": 0.5}
            value = inputs * spec["factor"]
            return value + 1, {"hidden": value} if "hidden" in names_filter else {}

    model = Remote()
    session = BenchSession.from_backend(
        TransformerLensRunner(model), identity={"model_id": "fake-remote"}
    )
    result = session.forward(
        torch.ones(2), capture=["hidden"], interventions={"hidden": {"op": "scale", "value": 0.5}}
    )
    assert torch.equal(result.output, torch.full((2,), 1.5))
    assert "fused" not in session.capabilities()["capture_points"]
    with pytest.raises(BenchError, match="callbacks"):
        session.forward(torch.ones(2), interventions={"hidden": lambda value: value * 0})
    assert model.calls == 1
