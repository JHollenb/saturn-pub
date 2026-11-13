"""``mdb.trace``: passive capture is bitwise non-intrusive, agrees with the stepper bit for bit,
refuses what it cannot observe, keeps calls and batch rows apart, and always removes its hooks."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.core.errors import BadArgs, Refused  # noqa: E402
from mdb.core.pointers import MAX_RECORD_BYTES, check_bounded, seal_digest  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.engine.ports import program_for, tensor_digest  # noqa: E402
from mdb.engine.trace import DEFAULT_MAX_BYTES  # noqa: E402

CASES = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32")]
TOKENS = [3, 17, 5, 29, 11]
BATCH = torch.tensor([[3, 17, 5, 29, 11], [8, 2, 30, 1, 19], [4, 4, 22, 13, 7]])


def every_port(model):
    names = program_for(model.config.model_type).names
    return [f"L{i}.{p}" for i in range(model.config.num_hidden_layers) for p in names]


def hook_count(model):
    return sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()
    )


def modeling(model):
    return program_for(model.config.model_type).modeling_module()


@pytest.mark.parametrize("family,dtype", CASES)
def test_logits_are_bitwise_equal_with_and_without_the_trace(family, dtype):
    model = tiny_model(family, dtype)
    with torch.no_grad():
        plain = model(BATCH).logits
        with mdb.trace(model, ports=every_port(model)) as t:
            traced = model(BATCH).logits
    assert tensor_digest(plain) == tensor_digest(traced)
    assert all(t.info(a)["state"] == "resolved" for a in every_port(model))
    assert t["L1.h_out"].shape[:2] == BATCH.shape


@pytest.mark.parametrize("family,dtype", CASES)
def test_captured_values_equal_the_stepping_debugger_bit_for_bit(family, dtype):
    dbg = mdb.open_tiny(family, dtype)
    model, names = dbg.model, dbg.program.names
    run = dbg.run(TOKENS, lookahead="layer")
    stepped = {}
    for layer in range(model.config.num_hidden_layers):
        run.until(f"L{layer}.m")
        for port in names[:-1]:
            stepped[f"L{layer}.{port}"] = run.read(f"L{layer}.{port}")
        run.until(f"L{layer}.h_out")
        stepped[f"L{layer}.h_out"] = run.read(f"L{layer}.h_out")
    assert sorted(stepped) == sorted(every_port(model))
    ids = torch.tensor([TOKENS])
    with torch.no_grad():
        prefix = model(ids[:, :-1], use_cache=True)
        with mdb.trace(model, ports=list(stepped)) as t:  # the stepper's native call
            model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True)
    for address, value in stepped.items():
        got = t[address]
        assert (got.shape, got.dtype) == (value.shape, value.dtype), address
        assert tensor_digest(got) == tensor_digest(value), address
    s = t.info("L0.s")
    assert (s["kind"], s["authority"], s["replica_equal"]) == ("replica", "derived-exact", True)
    p = t.info("L0.p")
    assert (p["kind"], p["authority"], p["axes"]) == ("native", "measured",
                                                      ["batch", "head", "row", "src"])


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_sdpa_refuses_the_eager_ports_and_stays_bitwise_passive(family):
    model = tiny_model(family, "fp32")
    model.set_attn_implementation("sdpa")
    with torch.no_grad():
        plain = model(BATCH).logits
        with mdb.trace(model, ports=every_port(model)) as t:
            traced = model(BATCH).logits
    assert tensor_digest(plain) == tensor_digest(traced)
    for address in every_port(model):
        info = t.info(address)
        if address.split(".")[1] in ("qr", "kr", "s", "p"):
            assert info["state"] == "refused" and info["authority"] is None
            assert "'sdpa'" in info["reason"]
            with pytest.raises(Refused, match="eager attention"):
                t[address]
        else:
            assert info["state"] == "resolved", address
    assert t.info("L0.A")["authority"] == "measured"  # the value mix is still native


def test_hooks_are_removed_after_a_normal_and_a_raising_forward():
    model = tiny_model("qwen2", "bf16")
    before, original = hook_count(model), modeling(model).eager_attention_forward
    with torch.no_grad(), mdb.trace(model, ports=every_port(model)):
        assert hook_count(model) > before
        model(BATCH)
    assert hook_count(model) == before
    assert modeling(model).eager_attention_forward is original

    def boom(module, args):
        raise RuntimeError("boom in layer 1")

    user = model.model.layers[1].register_forward_pre_hook(boom)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with torch.no_grad(), mdb.trace(model, ports=["L0.m", "L1.h", "L1.p"]) as t:
                model(BATCH)
    finally:
        user.remove()
    assert hook_count(model) == before
    assert modeling(model).eager_attention_forward is original
    assert t.info("L0.m")["state"] == "resolved"  # layer 0 ran before the error
    for address in ("L1.h", "L1.p"):
        info = t.info(address)
        assert info["state"] == "refused" and info["reason"] == "layer 1 did not run in call 0"


def test_batch_rows_and_several_calls_are_kept_apart():
    model = tiny_model("qwen3", "fp32")
    with torch.no_grad(), mdb.trace(model, ports=["L0.m", "L0.p", "L1.kr"]) as t:
        model(BATCH)
        model(BATCH[:2, :3])
    assert t.calls == 2
    assert tuple(t["L0.m", 0].shape) == (3, 5, 32)
    assert tuple(t["L0.m", 1].shape) == (2, 3, 32)
    assert tuple(t["L0.p", 0].shape) == (3, 4, 5, 5)  # (batch, head, row, src)
    assert tuple(t["L1.kr", -1].shape) == (2, 2, 3, 8)  # (batch, kv_head, src, head_dim)
    with pytest.raises(BadArgs, match="2 calls were traced"):
        t["L0.m"]
    with pytest.raises(BadArgs, match="out of range"):
        t.info("L0.m", call=2)
    with torch.no_grad(), mdb.trace(model, ports=["L0.m"]) as alone:
        model(BATCH[:2, :3])
    assert torch.equal(alone["L0.m"], t["L0.m", 1])


def test_captures_are_detached_copies_on_the_same_device_even_with_grad():
    model = tiny_model("qwen2", "fp32")
    with mdb.trace(model, ports=["L0.h_out", "L0.q"]) as t:
        out = model(BATCH)
    out.logits.sum().backward()  # the trace left the autograd graph intact
    value = t["L0.h_out"]
    assert not value.requires_grad and value.grad_fn is None
    assert value.device == next(model.parameters()).device


def test_dropout_refuses_s_and_p():
    model = tiny_model("qwen2", "fp32").train()
    for layer in model.model.layers:
        layer.self_attn.attention_dropout = 0.5
    with torch.no_grad(), mdb.trace(model, ports=["L0.qr", "L0.s", "L0.p"]) as t:
        model(BATCH)
    assert t.info("L0.qr")["state"] == "resolved"
    for address in ("L0.s", "L0.p"):
        assert "dropout" in t.info(address)["reason"]


def test_max_bytes_stops_capture_and_refuses_the_rest():
    model = tiny_model("qwen2", "fp32")
    one = 3 * 5 * 16 * 4  # bytes of one (3, 5, 16) fp32 residual value
    with torch.no_grad(), mdb.trace(model, ports=["L0.h", "L0.a", "L0.kr"],
                                    max_bytes=one + one // 2) as t:
        model(BATCH)
    assert t.info("L0.h")["state"] == "resolved"
    # L0.kr is (3, 1, 5, 8) fp32 = one // 2 bytes and would fit: the stop is sticky
    for address in ("L0.a", "L0.kr"):
        info = t.info(address)
        assert info["state"] == "refused" and "max_bytes" in info["reason"]
    assert t.record()["budget"] == {"max_bytes": one + one // 2, "held_bytes": one}
    assert DEFAULT_MAX_BYTES == 1 << 30


def test_nested_trace_project_and_bad_addresses_are_refused():
    model = tiny_model("qwen2", "bf16")
    with mdb.trace(model, ports=["L0.m"]) as outer:
        with pytest.raises(Refused, match="already active"):
            with mdb.trace(model, ports=["L1.m"]):
                pass
        with torch.no_grad():
            model(BATCH)
    assert outer["L0.m"].shape[0] == 3  # the outer trace kept working
    with pytest.raises(Refused, match="not built yet"):
        mdb.trace(model, ports=["L0.m"], project="runs/x")
    for bad, message in (("L0.qn", "no port 'qn'"), ("L9.m", "does not exist"),
                         ("t3.L0.m", "L<layer>.<port>")):
        with pytest.raises(BadArgs, match=message):
            mdb.trace(model, ports=[bad])
    with mdb.trace(model, ports=["L0.m"]) as idle:
        pass
    assert idle.info("L0.m")["reason"].startswith("no forward")


def test_record_is_sealed_bounded_and_matches_the_captures():
    model = tiny_model("qwen3", "bf16")
    model.set_attn_implementation("sdpa")
    with torch.no_grad(), mdb.trace(model, ports=["L0.m", "L0.p", "L1.s"]) as t:
        model(BATCH)
        model(BATCH[:1])
    record = t.record()
    body = {k: v for k, v in record.items() if k != "seal"}
    assert record["seal"] == seal_digest(body)
    assert check_bounded(record) == record
    assert len(seal_digest(record)) == 64 and len(str(record)) < MAX_RECORD_BYTES
    assert record["calls_total"] == 2 and record["model"]["family"] == "qwen3"
    first = record["calls"][0]["values"]
    assert first["L0.m"]["sha256"] == tensor_digest(t["L0.m", 0])
    assert first["L0.m"]["authority"] == "measured"
    assert first["L0.p"]["state"] == "refused" and "sha256" not in first["L0.p"]
    assert [c["call"] for c in t.record(calls=[1])["calls"]] == [1]
