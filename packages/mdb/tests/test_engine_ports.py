import json
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402

from mdb.engine import ports as ports_module  # noqa: E402
from mdb.engine import selfcheck  # noqa: E402
from mdb.engine.ports import (  # noqa: E402
    PortController,
    PortEdit,
    check_deterministic,
    observe_all,
    program_for,
)

CASES = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32")]
PROMPT = [3, 17, 5, 29, 11, 42, 8]  # 7 tokens
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def build_model(family, dtype, seed=1234):
    """3 layers, 4 heads over 2 KV groups, head_dim 16, non-degenerate weights."""
    torch.manual_seed(seed)
    common = dict(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    if family == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM

        model = Qwen2ForCausalLM(Qwen2Config(**common)).eval()
    else:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        model = Qwen3ForCausalLM(Qwen3Config(head_dim=16, **common)).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=generator))
            elif name.endswith("bias"):
                p.copy_(0.5 * torch.randn(p.shape, generator=generator))
            elif "q_proj" in name or "k_proj" in name:
                p.copy_(0.4 * torch.randn(p.shape, generator=generator))
            else:
                p.copy_(0.15 * torch.randn(p.shape, generator=generator))
    return model.to(DTYPES[dtype])


def head_dim(adapter):
    config = adapter.model.config
    return getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads


@pytest.fixture(scope="module", params=CASES, ids=lambda case: "-".join(case))
def adapter(request):
    family, dtype = request.param
    return DecoderAdapter(build_model(family, dtype), granularity="operation")


def _at_layer(adapter, layer, tokens=PROMPT):
    session = adapter.session(tokens)
    for _ in range(layer + 1):
        session.continue_()
    return session


def _full(adapter, session, layer, edits=()):
    controller = PortController(program_for(adapter), layer, edits=edits)
    assert session.probe(controller)["stopped"] is False
    return controller


def _no_hooks(adapter):
    for module in adapter.model.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
    modeling = program_for(adapter).modeling_module()
    assert not getattr(modeling.eager_attention_forward, "_mdb_attention_observer", False)


def test_fixtures_cover_grouped_query_attention(adapter):
    assert adapter.layers >= 3 and adapter.num_heads >= 4 and adapter.num_kv_heads >= 2
    assert adapter.num_heads > adapter.num_kv_heads and len(PROMPT) >= 6
    assert adapter.execution["attention"] == "eager"


def test_program_order_is_native_execution_order(adapter):
    program = program_for(adapter)
    if adapter.spec.model_type == "qwen3":
        assert program.names[2:7] == ("q", "qn", "k", "kn", "v")
    else:
        assert program.names[2:5] == ("q", "k", "v")
    for layer in range(adapter.layers):
        controller = _full(adapter, _at_layer(adapter, layer), layer)
        assert tuple(controller.order) == program.names
        assert set(controller.captures) == set(program.names)
    json.dumps(program.to_dict())


def test_t1_hooks_inert_receipts_cuts_and_logits_identical(adapter):
    plain, observed = adapter.session(PROMPT), adapter.session(PROMPT)
    controller = observe_all(program_for(adapter))
    generated = 3
    while plain.read("tokens").shape[1] < len(PROMPT) + generated:
        plain.continue_()
        observed.continue_(1, ports=controller)
        left, right = plain.capture(retain=False), observed.capture(retain=False)
        assert left.fingerprint == right.fingerprint
        if left.payload["logits"] is not None:
            assert (
                left.payload["logits"].view(torch.uint8).tolist()
                == right.payload["logits"].view(torch.uint8).tolist()
            )
    assert controller.installs == adapter.layers * generated
    assert [r._body for r in plain.receipts] == [r._body for r in observed.receipts]
    assert observed.read("tokens").tolist() == plain.read("tokens").tolist()
    _no_hooks(adapter)


def test_t2_stop_probes_equal_one_pass_and_never_touch_state(adapter):
    program = program_for(adapter)
    for tokens in (PROMPT, [7]):
        session = adapter.session(tokens)
        session.continue_()
        for layer in range(adapter.layers):
            before = session.capture(retain=False)
            cache = [t.clone() for t in before.payload["keys"] + before.payload["values"]]
            full = _full(adapter, session, layer)
            for port in program.names[:-1]:
                stop = PortController(program, layer, stop=port, capture=(port,))
                assert session.probe(stop)["stopped"] is True
                assert stop.order[-1] == port
                assert torch.equal(stop.captures[port], full.captures[port]), port
                after = session.capture(retain=False)
                assert after.fingerprint == before.fingerprint
                payload = after.payload
                assert all(
                    torch.equal(a, b) for a, b in zip(cache, payload["keys"] + payload["values"])
                )
            assert (len(session.attempts), len(session.history)) == (0, 0)
            receipts = len(session.receipts)
            session.continue_()
            assert len(session.receipts) == receipts + 1
    _no_hooks(adapter)


def test_s_replica_and_value_mix_are_bitwise_native(adapter):
    for layer in range(adapter.layers):
        controller = _full(adapter, _at_layer(adapter, layer), layer)
        assert controller.checks["s"] == {"equal": True}
        assert controller.checks["mix"] == {"equal": True}
        s, p = controller.captures["s"], controller.captures["p"]
        assert torch.equal(torch.softmax(s, dim=-1, dtype=torch.float32).to(p.dtype), p)
        assert s.shape == p.shape == (1, adapter.num_heads, 1, len(PROMPT))
        assert controller.captures["kr"].shape == (
            1,
            adapter.num_kv_heads,
            len(PROMPT),
            head_dim(adapter),
        )


def _changed(base, edited, names):
    return {name for name in names if not torch.equal(base.captures[name], edited.captures[name])}


def _noise(like, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(like.shape, generator=generator).to(like.dtype)


def test_t6_o_edit_reaches_skip_and_mlp(adapter):
    session = _at_layer(adapter, 0)
    base = _full(adapter, session, 0)
    edited = _full(adapter, session, 0, [PortEdit("o", _noise(base.captures["o"]), "add")])
    names = program_for(adapter).names
    upstream = names[: names.index("o")]
    assert not _changed(base, edited, upstream)
    assert {"o", "r", "n", "g", "u", "sg", "f", "m", "h_out"} <= _changed(base, edited, names)
    c = edited.captures
    assert torch.equal(c["r"], c["h"] + c["o"])
    assert torch.equal(c["h_out"], c["r"] + c["m"])
    assert torch.equal(edited.originals["o"], base.captures["o"])


def test_t6_n_edit_reaches_mlp_only(adapter):
    session = _at_layer(adapter, 1)
    base = _full(adapter, session, 1)
    edited = _full(adapter, session, 1, [PortEdit("n", _noise(base.captures["n"]))])
    names = program_for(adapter).names
    assert _changed(base, edited, names) == {"n", "g", "u", "sg", "f", "m", "h_out"}
    c = edited.captures
    assert torch.equal(c["r"], base.captures["r"])
    assert torch.equal(c["h_out"], c["r"] + c["m"])


def test_t6_a_edit_reaches_attention_branch_but_not_skip(adapter):
    session = _at_layer(adapter, 0)
    base = _full(adapter, session, 0)
    edited = _full(adapter, session, 0, [PortEdit("a", _noise(base.captures["a"]), "add")])
    changed = _changed(base, edited, program_for(adapter).names)
    assert {"a", "q", "k", "v", "qr", "s", "p", "A", "o", "r", "h_out"} <= changed
    assert "h" not in changed
    c = edited.captures
    assert torch.equal(c["h"], base.captures["h"])
    assert torch.equal(c["r"], base.captures["h"] + c["o"])


def test_t6_h_edit_reaches_skip_and_attention(adapter):
    session = _at_layer(adapter, 0)
    base = _full(adapter, session, 0)
    edited = _full(adapter, session, 0, [PortEdit("h", 0.5, "scale")])
    changed = _changed(base, edited, program_for(adapter).names)
    assert {"h", "a", "q", "o", "r", "h_out"} <= changed
    c = edited.captures
    assert torch.equal(c["h"], base.captures["h"] * 0.5)
    assert torch.equal(c["r"], c["h"] + c["o"])


def test_t6_r_edit_is_realised_as_o_and_reports_its_error(adapter):
    session = _at_layer(adapter, 0)
    layer = adapter.model.model.layers[0]
    base = _full(adapter, session, 0)
    target = base.captures["r"] + _noise(base.captures["r"])
    edited = _full(adapter, session, 0, [PortEdit("r", target)])
    c = edited.captures
    assert torch.equal(c["o"], target - c["h"])
    assert torch.equal(c["r"], c["h"] + c["o"])
    assert torch.equal(c["h"], base.captures["h"])
    with torch.inference_mode():
        assert torch.equal(c["n"], layer.post_attention_layernorm(c["r"]))
    assert torch.equal(c["h_out"], c["r"] + c["m"])
    [record] = edited.manifest()
    error = float((c["r"].float() - target.float()).abs().max())
    assert record["realised_at"] == "o"
    # the cost of realising r through o is recorded under realisation_*; realised_* is measured
    # against x op requested in float64, as at every other port (here target is the request)
    assert record["realisation_max_abs_error"] == error
    assert record["realisation_exact"] is (error == 0.0)
    assert record["realised_max_abs_error"] == error and record["realised_exact"] is (error == 0.0)
    exact = _full(adapter, session, 0, [PortEdit("r", 0.0, "add")])
    [exact_record] = exact.manifest()
    assert exact_record["realised_exact"] is True and exact_record["realisation_exact"] is True
    assert torch.equal(exact.captures["h_out"], base.captures["h_out"])


def test_t6_p_and_s_edits_use_the_replica_mix(adapter):
    session = _at_layer(adapter, 1)
    base = _full(adapter, session, 1)
    uniform = torch.full_like(base.captures["p"], 1.0 / base.captures["p"].shape[-1])
    edited = _full(adapter, session, 1, [PortEdit("p", uniform)])
    assert edited.checks["mix"] == {"replica": True, "native_unedited": False}
    assert edited.checks["mix_replica"] == {"equal": True}
    assert torch.equal(edited.captures["p"], uniform)
    assert {"A", "o", "r", "h_out"} <= _changed(base, edited, ("A", "o", "r", "h_out"))
    scores = _full(adapter, session, 1, [PortEdit("s", _noise(base.captures["s"]), "add")])
    s, p = scores.captures["s"], scores.captures["p"]
    assert torch.equal(torch.softmax(s, dim=-1, dtype=torch.float32).to(p.dtype), p)
    assert torch.equal(scores.originals["p"], base.captures["p"])
    with pytest.raises(ValueError, match="not editable"):
        PortController(program_for(adapter), 1, edits=[PortEdit("kr", 1.0, "scale")])


def test_t6_committed_edit_is_in_receipt_and_changes_the_cut(adapter):
    plain, edited = _at_layer(adapter, 0), _at_layer(adapter, 0)
    value = _noise(torch.empty(1, 1, adapter.num_kv_heads * head_dim(adapter), dtype=adapter.dtype))
    controller = PortController(
        program_for(adapter), 0, edits=[PortEdit("k", value, "add")], capture=False
    )
    plain.continue_()
    receipt = edited.continue_(1, ports=controller)
    entry = receipt.to_dict()["transitions"][0]
    [manifest] = entry["port_edits"]
    assert manifest["port"] == "k" and manifest["op"] == "add" and manifest["layer"] == 0
    assert manifest["value"]["kind"] == "tensor" and len(manifest["value"]["sha256"]) == 64
    assert len(json.dumps(entry["port_edits"])) < 768  # incl. the labels map
    left, right = plain.capture(retain=False), edited.capture(retain=False)
    assert left.fingerprint != right.fingerprint
    keys, edited_keys = left.payload["keys"][0], right.payload["keys"][0]
    assert torch.equal(keys[..., :-1, :], edited_keys[..., :-1, :])
    assert not torch.equal(keys[..., -1, :], edited_keys[..., -1, :])
    assert controller.manifest() is None


def test_hooks_removed_after_stop_and_after_errors(adapter):
    session = _at_layer(adapter, 0)
    program = program_for(adapter)
    session.probe(PortController(program, 0, stop="qr"))
    _no_hooks(adapter)
    bad = PortController(program, 0, edits=[PortEdit("g", torch.zeros(3, dtype=adapter.dtype))])
    with pytest.raises(ValueError, match="needs shape"):
        session.probe(bad)
    _no_hooks(adapter)
    assert (len(session.receipts), len(session.attempts)) == (1, 0)
    with pytest.raises(ValueError, match="h_out is the layer output"):
        PortController(program, 0, stop="h_out")
    with pytest.raises(ValueError, match="explicit layer"):
        PortController(program, None, edits=[PortEdit("g", 1.0, "scale")])


# --- fix 1: the manifest lists exactly the edits that fired during that transition -----------


def test_probe_with_edit_never_leaks_into_a_later_receipt(adapter):
    program = program_for(adapter)
    session = _at_layer(adapter, 1)
    controller = PortController(program, 1, edits=[PortEdit("m", 0.0, "scale")])
    session.probe(controller)
    plain = _at_layer(adapter, 1)
    first = session.continue_(1, ports=PortController(program, 2))
    assert first._body == plain.continue_()._body
    receipts, attempts = len(session.receipts), len(session.attempts)
    # layer 2 would run: the controller addresses layer 1, so the call is refused up front
    with pytest.raises(ValueError, match="carries edits for layer 1"):
        session.continue_(1, ports=controller)
    assert (len(session.receipts), len(session.attempts)) == (receipts, attempts)
    assert session.continue_()._body == plain.continue_()._body  # no edit leaked into layer 2
    _no_hooks(adapter)


def test_one_controller_over_three_transitions_records_only_where_applied(adapter):
    program = program_for(adapter)
    session = _at_layer(adapter, 0)
    controller = PortController(program, 1, edits=[PortEdit("m", 0.5, "scale")], capture=False)
    transitions = []
    with pytest.raises(ValueError, match="carries edits for layer 1"):  # layer 0 runs: refused
        session.continue_(1, ports=controller)
    transitions += session.continue_(2, ports=controller).to_dict()["transitions"]  # layers 0, 1
    transitions += session.continue_(1).to_dict()["transitions"]  # layer 2, plain
    assert ["port_edits" in entry for entry in transitions] == [False, True, False]
    [edit] = transitions[1]["port_edits"]
    assert edit["port"] == "m" and edit["layer"] == 1
    many = _at_layer(adapter, 0)
    entries = many.continue_(3, ports=controller).to_dict()["transitions"]
    assert ["port_edits" in entry for entry in entries] == [False, True, False]
    # normalization / readout / sample transitions never install the controller
    idle = PortController(program, 1, capture=False)
    for _ in range(4):
        entry = many.continue_(1, ports=idle).to_dict()["transitions"][0]
        assert "port_edits" not in entry


def test_unfired_edit_for_the_running_layer_fails_the_transition(adapter, monkeypatch):
    program = program_for(adapter)
    session = _at_layer(adapter, 1)
    before = session.capture(retain=False).fingerprint
    receipts = len(session.receipts)
    original = PortController._out

    def silent(self, name):
        return (lambda module, args, output: None) if name == "m" else original(self, name)

    monkeypatch.setattr(PortController, "_out", silent)
    controller = PortController(program, 1, edits=[PortEdit("m", 0.0, "scale")])
    with pytest.raises(ValueError, match="staged edits were not realised: m"):
        session.continue_(1, ports=controller)
    assert session.capture(retain=False).fingerprint == before
    assert (len(session.receipts), len(session.attempts)) == (receipts, 1)
    _no_hooks(adapter)
    monkeypatch.undo()
    assert session.continue_(1, ports=controller).to_dict()["transitions"][0]["port_edits"]


# --- fix 3: the cast of an edit value is recorded ----------------------------------------


def _expected_cast_error(value, dtype):
    return float((value.double() - value.to(dtype).double()).abs().max())


@pytest.mark.parametrize("source", [torch.float64, torch.float32])
def test_cast_error_of_tensor_edits_is_recorded(adapter, source):
    session = _at_layer(adapter, 0)
    native = _full(adapter, session, 0).captures["a"]
    generator = torch.Generator().manual_seed(9)
    requested = (native.double() + 1e-3 * torch.randn(native.shape, generator=generator)).to(source)
    controller = _full(adapter, session, 0, [PortEdit("a", requested)])
    [record] = controller.manifest()
    expected = _expected_cast_error(requested, adapter.dtype)
    assert record["realised_dtype"] == str(adapter.dtype)
    assert record["realised_max_abs_error"] == expected == record["cast_max_abs_error"]
    assert record["realised_exact"] is (expected == 0.0)
    assert record["realised_sha256"] == ports_module.tensor_digest(requested.to(adapter.dtype))
    assert controller.captures["a"].dtype == adapter.dtype
    if adapter.dtype == torch.bfloat16:
        assert expected > 0 and record["realised_exact"] is False
    assert len(json.dumps(record)) < 1024


def test_exactly_representable_and_scalar_edits(adapter):
    session = _at_layer(adapter, 0)
    native = _full(adapter, session, 0).captures["a"]
    exact = _full(adapter, session, 0, [PortEdit("a", native.double())])
    [record] = exact.manifest()
    assert record["realised_exact"] is True and record["realised_max_abs_error"] == 0.0
    half = _full(adapter, session, 0, [PortEdit("a", 0.5, "scale")])
    assert half.manifest()[0]["realised_exact"] is True
    tenth = _full(adapter, session, 0, [PortEdit("a", 0.1, "scale")])
    [record] = tenth.manifest()
    operand = torch.tensor(0.1, dtype=adapter.dtype)
    # the arithmetic used the recorded cast operand, so the captured value is x * bf16(0.1)
    assert torch.equal(tenth.captures["a"], native * operand)
    cast_error = abs(0.1 - float(operand))
    assert record["cast_exact"] is False
    assert record["cast_max_abs_error"] == pytest.approx(cast_error, rel=1e-9)
    ideal = native.double() * 0.1
    effect = float((tenth.captures["a"].double() - ideal).abs().max())
    assert record["realised_max_abs_error"] == effect and record["realised_exact"] is False
    assert record["realised_sha256"] == ports_module.tensor_digest(operand)
    assert 0.0 < record["changed_fraction"] <= 1.0
    replaced = _full(adapter, session, 0, [PortEdit("a", 0.1)])
    assert replaced.manifest()[0]["realised_exact"] is False


def test_r_edit_keeps_its_own_error_and_records_the_cast_separately(adapter):
    session = _at_layer(adapter, 0)
    base = _full(adapter, session, 0)
    target = (base.captures["r"].double() + 1e-3).to(torch.float64)
    controller = _full(adapter, session, 0, [PortEdit("r", target)])
    [record] = controller.manifest()
    r_error = float((controller.captures["r"].double() - target.to(adapter.dtype).double()).abs().max())
    assert record["realised_at"] == "o"
    assert record["cast_max_abs_error"] == _expected_cast_error(target, adapter.dtype)
    assert record["cast_exact"] is (record["cast_max_abs_error"] == 0.0)
    assert isinstance(record["realised_max_abs_error"], float) and r_error >= 0.0
    assert record["realised_dtype"] == str(adapter.dtype)


# --- fix 4: replica labelling --------------------------------------------------------------


def test_s_and_p_entries_are_labelled_replica_and_verified(adapter):
    session = _at_layer(adapter, 1)
    base = _full(adapter, session, 1)
    uniform = torch.full_like(base.captures["p"], 1.0 / base.captures["p"].shape[-1])
    p_edit = _full(adapter, session, 1, [PortEdit("p", uniform)])
    [record] = p_edit.manifest()
    assert record["mix"] == "replica" and record["replica_verified"] is True
    assert record["is_distribution"] is True
    s_edit = _full(adapter, session, 1, [PortEdit("s", _noise(base.captures["s"]), "add")])
    [record] = s_edit.manifest()
    assert record["mix"] == "replica" and record["replica_verified"] is True
    assert "is_distribution" not in record
    other = _full(adapter, session, 1, [PortEdit("m", 0.5, "scale")])
    assert "mix" not in other.manifest()[0]
    assert s_edit.capture_notes["s"] == {"replica_equal": True, "max_abs_diff": 0.0}


def test_failed_replica_check_is_visible_on_the_entry_and_the_capture(adapter, monkeypatch):
    session = _at_layer(adapter, 1)
    base = _full(adapter, session, 1)
    monkeypatch.setattr(
        ports_module, "_equality", lambda a, b: {"equal": False, "max_abs_diff": 0.25}
    )
    controller = _full(adapter, session, 1, [PortEdit("p", torch.zeros_like(base.captures["p"]))])
    [record] = controller.fired.values()
    assert record["replica_verified"] is False
    assert controller.capture_notes["s"] == {"replica_equal": False, "max_abs_diff": 0.25}
    assert controller.replica_verified is False
    assert controller.replica_max_abs_diff == 0.25
    with pytest.raises(ValueError, match="layer 1: replica check failed.*0.25.*p edit"):
        controller.manifest()  # a commit would be refused (see the session-level test below)


def test_non_distribution_p_edit_is_accepted_and_described(adapter):
    session = _at_layer(adapter, 1)
    base = _full(adapter, session, 1)
    shape = base.captures["p"].shape
    doubled = _full(adapter, session, 1, [PortEdit("p", torch.full(shape, 2.0 / shape[-1]))])
    [record] = doubled.manifest()
    assert record["is_distribution"] is False
    assert record["row_sum_min"] == pytest.approx(2.0, abs=2e-2)
    assert record["row_sum_max"] == pytest.approx(2.0, abs=2e-2)
    assert record["min"] > 0
    negative = base.captures["p"].clone()
    negative[..., 0] = -0.25
    [record] = _full(adapter, session, 1, [PortEdit("p", negative)]).manifest()
    assert record["is_distribution"] is False and record["min"] == -0.25
    assert set(record) >= {"row_sum_min", "row_sum_max", "min", "is_distribution"}


# --- fix 5: errors, read info, atomic install ---------------------------------------------


def test_non_broadcastable_values_name_port_and_shapes(adapter):
    session = _at_layer(adapter, 0)
    for op in ("add", "scale"):
        bad = PortController(
            program_for(adapter), 0, edits=[PortEdit("g", torch.zeros(5, 7, dtype=adapter.dtype), op)]
        )
        with pytest.raises(ValueError, match=r"port g.*port shape \(1, 1, 128\).*value shape \(5, 7\)"):
            session.probe(bad)
    _no_hooks(adapter)


def test_probe_of_a_layer_that_is_not_pending_is_refused(adapter):
    session = _at_layer(adapter, 0)
    before = session.capture(retain=False).fingerprint
    controller = PortController(program_for(adapter), 1, stop="q")
    with pytest.raises(ValueError, match="addresses layer 1 but the probed transition ran layer 0"):
        session.probe(controller)
    with pytest.raises(ValueError, match="addresses layer 2"):
        session.probe(PortController(program_for(adapter), 2, edits=[PortEdit("m", 1.0, "add")]))
    assert session.capture(retain=False).fingerprint == before
    assert controller.captures == {}


def test_o_capture_under_a_staged_r_edit_is_marked_as_realisation(adapter):
    session = _at_layer(adapter, 0)
    base = _full(adapter, session, 0)
    target = base.captures["r"] + _noise(base.captures["r"])
    edited = _full(adapter, session, 0, [PortEdit("r", target)])
    assert edited.edited_captures["o"] == {"by": "r", "kind": "realisation"}
    assert edited.edited_captures["r"] == {"by": "r", "kind": "edit"}
    assert not torch.equal(edited.captures["o"], base.captures["o"])
    assert base.edited_captures == {}
    s_edit = _full(adapter, session, 0, [PortEdit("s", _noise(base.captures["s"]), "add")])
    assert s_edit.edited_captures["s"]["kind"] == "edit"
    assert s_edit.edited_captures["p"] == {"by": "s", "kind": "replica"}


def test_interceptor_restored_after_an_error_inside_it(adapter):
    program = program_for(adapter)
    modeling = program.modeling_module()
    original = modeling.eager_attention_forward
    session = _at_layer(adapter, 0)
    bad = PortController(program, 0, edits=[PortEdit("qr", torch.zeros(3, dtype=adapter.dtype))])
    with pytest.raises(ValueError, match="needs shape"):
        session.probe(bad)
    assert modeling.eager_attention_forward is original
    _no_hooks(adapter)
    assert not _full(adapter, session, 0).checks == {}


def test_nested_install_on_one_module_is_refused_and_restores(adapter):
    program = program_for(adapter)
    modeling = program.modeling_module()
    original = modeling.eager_attention_forward
    layer = adapter.model.model.layers[0]
    outer = PortController(program, 0)
    with outer.installed(adapter, 0, layer):
        inner = PortController(program, 0)
        with pytest.raises(ValueError, match="already installed on this attention module"):
            with inner.installed(adapter, 0, layer):
                pass
        assert modeling.eager_attention_forward._mdb_attention_observer
    assert modeling.eager_attention_forward is original
    _no_hooks(adapter)


def test_check_deterministic_refuses_training_modules():
    model = build_model("qwen2", "fp32")
    check_deterministic(model)
    model.model.layers[1].self_attn.attention_dropout = 0.5
    model.train()
    with pytest.raises(ValueError, match="eval-mode.*0.5"):
        check_deterministic(model)


# --- fix 5d: selfcheck ---------------------------------------------------------------------


def _selfcheck(tmp_path, *args):
    out = tmp_path / "report.json"
    code = selfcheck.main([*args, "--json", str(out)])
    return code, json.loads(out.read_text())


def test_selfcheck_tiny_with_prompt_runs_and_passes(tmp_path):
    code, report = _selfcheck(tmp_path, "--tiny", "qwen2", "--prompt", "port stepping", "--tokens", "1")
    assert code == 0 and report["passed"] is True
    assert report["prompt_tokens"] == len("port stepping")
    assert "bytes modulo" in report["prompt_encoding"]
    probes = report["probe_consistency"]
    assert all(item == {"equal": True} for item in probes["s_replica"] + probes["mix_replica"])
    assert report["exactness"]["observed_layer_calls"] == report["exactness"]["expected_layer_calls"]


def test_selfcheck_fails_when_a_replica_check_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ports_module, "_equality", lambda a, b: {"equal": False, "max_abs_diff": 1.0})
    code, report = _selfcheck(tmp_path, "--tiny", "qwen2", "--token-ids", "1,2,3,4,5,6", "--tokens", "1")
    assert code == 1 and report["passed"] is False


def test_selfcheck_fails_when_the_controller_never_installs(tmp_path, monkeypatch):
    original = PortController.installed

    @contextmanager
    def lazy(self, adapter, index, module):
        if self.layer is None:
            yield self
            return
        with original(self, adapter, index, module) as controller:
            yield controller

    monkeypatch.setattr(PortController, "installed", lazy)
    code, report = _selfcheck(tmp_path, "--tiny", "qwen2", "--token-ids", "1,2,3,4,5,6", "--tokens", "1")
    assert report["exactness"]["observed_layer_calls"] == 0
    assert code == 1 and report["passed"] is False


# --- round 2: scalar operands, masks, honest stops, interceptor registry, labels --------------


def _bits(value):
    return value.contiguous().view({2: torch.int16, 4: torch.int32}[value.element_size()])


@pytest.mark.parametrize("op", ["add", "scale"])
def test_scalar_edit_computes_with_the_recorded_cast_operand(adapter, op):
    session = _at_layer(adapter, 0)
    x = _full(adapter, session, 0).captures["a"]
    edited = _full(adapter, session, 0, [PortEdit("a", 0.1, op)])
    [record] = edited.manifest()
    operand = torch.tensor(0.1, dtype=adapter.dtype)  # independent expectation
    expected = x + operand if op == "add" else x * operand
    assert torch.equal(edited.captures["a"], expected)
    assert record["realised_sha256"] == ports_module.tensor_digest(operand)
    assert record["cast_max_abs_error"] == abs(0.1 - float(operand.double()))
    ideal = x.double() + 0.1 if op == "add" else x.double() * 0.1
    effect = float((expected.double() - ideal).abs().max())
    assert record["realised_max_abs_error"] == effect
    assert record["realised_exact"] is (effect == 0.0)
    assert record["changed_fraction"] == float((_bits(x) != _bits(expected)).double().mean())


def test_tensor_operand_effect_error_is_separate_from_the_cast_error(adapter):
    session = _at_layer(adapter, 0)
    x = _full(adapter, session, 0).captures["a"]
    generator = torch.Generator().manual_seed(21)
    delta = (1e-3 * torch.randn(x.shape, generator=generator, dtype=torch.float64)).to(torch.float32)
    for op in ("add", "scale"):
        [record] = _full(adapter, session, 0, [PortEdit("a", delta, op)]).manifest()
        written = delta.to(adapter.dtype)
        got = x + written if op == "add" else x * written
        ideal = x.double() + delta.double() if op == "add" else x.double() * delta.double()
        assert record["cast_max_abs_error"] == float((written.double() - delta.double()).abs().max())
        assert record["realised_max_abs_error"] == float((got.double() - ideal).abs().max())


def test_an_add_that_rounds_away_is_visible(adapter):
    session = _at_layer(adapter, 0)
    x = _full(adapter, session, 0).captures["a"]
    tiny = torch.full(x.shape, 2.0**-14, dtype=torch.float64)
    controller = _full(adapter, session, 0, [PortEdit("a", tiny, "add")])
    [record] = controller.manifest()
    got = controller.captures["a"]
    unchanged = float((_bits(got) == _bits(x)).double().mean())
    assert record["cast_exact"] is True  # 2^-14 is representable
    assert record["changed_fraction"] == pytest.approx(1.0 - unchanged, abs=0)
    if adapter.dtype == torch.bfloat16:
        assert record["realised_exact"] is False and record["changed_fraction"] < 0.5
        assert record["realised_max_abs_error"] > 0.0


def test_masked_edit_leaves_unselected_elements_bit_identical(adapter):
    session = _at_layer(adapter, 0)
    base = _full(adapter, session, 0)
    x = base.captures["g"]
    mask = torch.zeros(x.shape, dtype=torch.bool)
    mask[..., ::3] = True
    for op, value in (("replace", 0.5), ("add", 0.25), ("scale", 2.0)):
        edit = PortEdit("g", value, op, mask=mask, selection="every third unit")
        controller = _full(adapter, session, 0, [edit])
        got = controller.captures["g"]
        assert torch.equal(_bits(got)[~mask], _bits(x)[~mask])
        operand = torch.tensor(value, dtype=adapter.dtype)
        full = {"replace": operand.expand(x.shape), "add": x + operand, "scale": x * operand}[op]
        assert torch.equal(got[mask], full[mask])
        [record] = controller.manifest()
        assert record["selection"] == "every third unit"
        assert record["mask_sha256"] == ports_module.tensor_digest(mask)
        assert record["mask_count"] == int(mask.sum())
        assert record["changed_fraction"] <= float(mask.double().mean()) + 1e-12
        manifest = edit.manifest()
        assert manifest["selection"] == "every third unit" and manifest["mask_count"] == int(mask.sum())
    with pytest.raises(ValueError, match="mask at port g does not broadcast"):
        PortEdit("g", 1.0, mask=torch.ones(7, dtype=torch.bool)).validate(x.shape)
    with pytest.raises(ValueError, match="bool tensor"):
        PortEdit("g", 1.0, mask=torch.ones(3))


def test_masked_r_edit_keeps_unselected_elements_of_r_native(adapter):
    session = _at_layer(adapter, 0)
    base = _full(adapter, session, 0)
    mask = torch.zeros_like(base.captures["r"], dtype=torch.bool)
    mask[..., :8] = True
    edited = _full(adapter, session, 0, [PortEdit("r", 0.5, "add", mask=mask)])
    r, native = edited.captures["r"], base.captures["r"]
    assert torch.equal(_bits(r)[~mask], _bits(native)[~mask])
    assert not torch.equal(r[mask], native[mask])
    [record] = edited.manifest()
    assert record["mask_count"] == 8 and record["realised_at"] == "o"


def test_program_shapes_equal_captured_shapes(adapter):
    program = program_for(adapter)
    session = _at_layer(adapter, 1)
    controller = _full(adapter, session, 1)
    for name in program.names:
        derived = program.shape(name, adapter.model.config, src=len(PROMPT))
        assert derived == tuple(controller.captures[name].shape), name


def test_producer_edges_name_earlier_ports_and_match_the_operator_text(adapter):
    import re

    program = program_for(adapter)
    names = program.names
    for spec in program.ports:
        assert spec.inputs, spec.name
        for source in spec.inputs:
            assert source in ("cache", "prev_layer") or names.index(source) < names.index(spec.name)
        rhs = spec.operator.split("=", 1)[1]
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs))
        named = {w for w in words if w in names}
        if "layer input" in rhs:
            named.add("prev_layer")
        if "cache" in words:
            named.add("cache")
        assert named == set(spec.inputs), (spec.name, named, spec.inputs)
    if adapter.spec.model_type == "qwen3":
        assert program.spec("qr").inputs == ("qn",)
        assert program.spec("kr").inputs == ("kn", "cache")
        assert "RoPE(qn)" in program.spec("qr").operator
        assert "RoPE(kn)" in program.spec("kr").operator
    else:
        assert program.spec("qr").inputs == ("q",) and program.spec("kr").inputs == ("k", "cache")
    assert all("inputs" in port for port in program.to_dict()["ports"])


def test_rope_ports_declare_only_the_pair_expansion_proof(adapter):
    program = program_for(adapter)
    assert program.spec("qr").proofs == ("P04",) and program.spec("kr").proofs == ("P04",)
    assert all("P26" not in port.proofs for port in program.ports)


def test_no_port_is_labelled_with_the_bare_word_derived(adapter):
    import re

    text = json.dumps(program_for(adapter).to_dict())
    assert not re.search(r"derived(?!-)", text)
    assert program_for(adapter).spec("s").kind == "replica"


# --- E4a: a stop at p computes A natively but neither holds nor checks it ----------------------


def test_p_stop_does_not_hold_or_check_the_value_mix(adapter):
    program = program_for(adapter)
    session = _at_layer(adapter, 1)
    at_p = PortController(program, 1, stop="p", capture=program.names[: program.index("p") + 1])
    assert session.probe(at_p)["stopped"] is True
    assert "A" not in at_p.captures and "A" not in at_p.order
    assert "mix_replica" not in at_p.checks and "mix" not in at_p.checks
    assert at_p.replica_verified is None and at_p.checks["s"] == {"equal": True}
    at_a = PortController(program, 1, stop="A", capture=("A",))
    assert session.probe(at_a)["stopped"] is True
    assert at_a.checks["mix_replica"] == {"equal": True} and at_a.replica_verified is True


# --- E5: replica checks are judged against native, not against themselves ----------------------


def _perturbing_native(monkeypatch, modeling, *, weights=False, output=False):
    native = modeling.eager_attention_forward

    def wrapped(module, query, key, value, mask, *args, **kwargs):
        out, w = native(module, query, key, value, mask, *args, **kwargs)
        if weights:
            w = w.clone()
            _bits(w).view(-1)[0] += 1  # one ulp in one element
        if output:
            out = out.clone()
            _bits(out).view(-1)[0] += 1
        return out, w

    monkeypatch.setattr(modeling, "eager_attention_forward", wrapped)


def test_one_ulp_in_native_weights_fails_the_s_replica_check(adapter, monkeypatch):
    program = program_for(adapter)
    session = _at_layer(adapter, 1)
    _perturbing_native(monkeypatch, program.modeling_module(), weights=True)
    controller = _full(adapter, session, 1)
    assert controller.checks["s"]["equal"] is False
    assert controller.capture_notes["s"]["replica_equal"] is False
    assert controller.replica_verified is False
    _no_hooks(adapter)


def test_a_perturbed_native_output_fails_the_mix_replica_check(adapter, monkeypatch):
    program = program_for(adapter)
    session = _at_layer(adapter, 1)
    _perturbing_native(monkeypatch, program.modeling_module(), output=True)
    controller = _full(adapter, session, 1)
    assert controller.checks["s"] == {"equal": True}
    assert controller.checks["mix_replica"]["equal"] is False
    assert controller.replica_verified is False


@pytest.mark.parametrize("mode", ["weights", "output"])
def test_selfcheck_fails_against_a_perturbed_native_attention(tmp_path, monkeypatch, mode):
    modeling = program_for("qwen2").modeling_module()
    _perturbing_native(monkeypatch, modeling, weights=mode == "weights", output=mode == "output")
    code, report = _selfcheck(tmp_path, "--tiny", "qwen2", "--token-ids", "1,2,3,4,5,6", "--tokens", "1")
    assert code == 1 and report["passed"] is False


# --- E6: one interceptor, a registry keyed by attention module, pass-through elsewhere ---------


def test_plain_forward_from_another_thread_is_untouched_while_installed(adapter):
    import threading

    program = program_for(adapter)
    ids = torch.tensor([PROMPT])
    with torch.inference_mode():
        reference = adapter.model(ids).logits.clone()
    controller = PortController(program, 0)
    seen = {}
    with controller.installed(adapter, 0, adapter.model.model.layers[0]):

        def plain():
            with torch.inference_mode():
                seen["logits"] = adapter.model(ids).logits.clone()

        thread = threading.Thread(target=plain)
        thread.start()
        thread.join()
        assert controller.order == [] and controller._realised == [] and controller.captures == {}
    assert torch.equal(_bits(seen["logits"]), _bits(reference))
    _no_hooks(adapter)


def test_registry_is_empty_and_patch_restored_after_an_exception_inside_the_body(adapter):
    program = program_for(adapter)
    modeling = program.modeling_module()
    original = modeling.eager_attention_forward
    controller = PortController(program, 0)
    with pytest.raises(RuntimeError, match="boom"):
        with controller.installed(adapter, 0, adapter.model.model.layers[0]):
            assert modeling.eager_attention_forward is not original
            raise RuntimeError("boom")
    assert modeling.eager_attention_forward is original and ports_module._PATCHES == {}
    assert ports_module._LOCAL.active is None
    _no_hooks(adapter)


def test_two_models_install_at_once_and_each_sees_only_its_own_controller():
    models = [DecoderAdapter(build_model("qwen2", "fp32", seed=seed), granularity="operation") for seed in (1, 2)]
    program = program_for("qwen2")
    modeling = program.modeling_module()
    original = modeling.eager_attention_forward
    sessions = [_at_layer(m, 0) for m in models]
    outer = PortController(program, 0)
    inner = PortController(program, 0)
    with outer.installed(models[0], 0, models[0].model.model.layers[0]):
        assert set(ports_module._PATCHES) == {modeling.__name__}
        with inner.installed(models[1], 0, models[1].model.model.layers[0]):
            assert len(ports_module._PATCHES[modeling.__name__].owners) == 2
    assert modeling.eager_attention_forward is original and ports_module._PATCHES == {}
    # each model, probed alone afterwards, gives what the nested install would have given it
    solo = [_full(m, s, 0) for m, s in zip(models, sessions)]
    assert solo[0].checks["s"] == {"equal": True} and solo[1].checks["s"] == {"equal": True}


# --- E7 ---------------------------------------------------------------------------------------


def test_layer_beyond_the_model_is_refused(adapter):
    program = program_for(adapter)
    with pytest.raises(ValueError, match="does not exist in a 3-layer model"):
        PortController(program, 3, edits=[PortEdit("m", 0.0, "scale")], layers=adapter.layers)
    session = _at_layer(adapter, 0)
    before = session.capture(retain=False).fingerprint
    ghost = PortController(program, 99, edits=[PortEdit("m", 0.0, "scale")], capture=False)
    with pytest.raises(ValueError, match="carries edits for layer 99"):
        session.continue_(3, ports=ghost)
    assert session.capture(retain=False).fingerprint == before
    _no_hooks(adapter)


@pytest.mark.parametrize(
    "dtype,tolerance",
    [(torch.bfloat16, 2.0**-7), (torch.float16, 2.0**-10), (torch.float32, 1e-5), (torch.float64, 1e-12)],
)
def test_row_sum_tolerance_is_dtype_specific_and_labelled(dtype, tolerance):
    uniform = torch.full((1, 2, 1, 8), 0.125, dtype=dtype)
    stats = ports_module._row_statistics(uniform)
    assert stats["row_sum_tolerance"] == tolerance
    assert "row_sum_tolerance_authority" not in stats
    assert ports_module._with_labels(dict(stats))["labels"] == {
        "row_sum_min": "measured", "row_sum_max": "measured", "min": "measured",
        "row_sum_tolerance": "supplied",
    }
    assert stats["is_distribution"] is True
    off = uniform * (1.0 + 4 * tolerance)
    assert ports_module._row_statistics(off)["is_distribution"] is False


# --- E11 --------------------------------------------------------------------------------------


def test_engine_does_not_import_the_legacy_module():
    import subprocess
    import sys
    from pathlib import Path

    engine = Path(ports_module.__file__).parent
    assert not [f.name for f in engine.glob("*.py") if "local_debugger" in f.read_text()]
    code = (
        "import sys; import mdb.engine.selfcheck, mdb.engine.run, mdb.engine.models; "
        "bad = [m for m in sys.modules if m == 'mdb.local_debugger']; "
        "print(len([m for m in sys.modules if m.startswith('mdb.')]), bad)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-1500:]
    count, bad = result.stdout.split(" ", 1)
    assert bad.strip() == "[]" and int(count) < 30, result.stdout


def test_tiny_model_builder_takes_layers():
    from mdb.engine.models import tiny_model

    assert tiny_model("qwen3", torch.float32, layers=3).config.num_hidden_layers == 3
    assert selfcheck.tiny_model is tiny_model
    with pytest.raises(ValueError, match="qwen2 or qwen3"):
        tiny_model("llama")


def test_selfcheck_model_path_goes_through_load_local(tmp_path, monkeypatch):
    seen = {}

    def fake(path, *, dtype, device):
        seen.update(path=path, dtype=dtype, device=device)
        return selfcheck.tiny_model("qwen2", dtype)

    monkeypatch.setattr(selfcheck, "load_local", fake)
    code, report = _selfcheck(tmp_path, "--model", str(tmp_path), "--token-ids", "1,2,3,4,5,6", "--tokens", "1")
    assert code == 0 and seen["path"] == str(tmp_path) and report["source"] == "checkpoint"
    assert report["labels"]["exactness.expected_layer_calls"] == "derived-exact"
    assert report["labels"]["generated"] == "supplied" and report["labels"]["prompt_tokens"] == "supplied"


# --- round 3 -----------------------------------------------------------------------------------


def _ideal64(x, op, request, mask):
    """x op request in float64, on the selected elements (independent of PortEdit)."""
    wide = x.detach().to(torch.float64)
    req = torch.as_tensor(request, dtype=torch.float64)
    ideal = {"replace": req.expand(wide.shape), "add": wide + req, "scale": wide * req}[op]
    return ideal if mask is None else torch.where(mask, ideal, wide)


@pytest.mark.parametrize("op", ["replace", "add", "scale"])
@pytest.mark.parametrize("masked", [False, True], ids=["whole", "masked"])
@pytest.mark.parametrize("kind", ["scalar", "tensor"])
def test_r_edit_errors_are_measured_against_the_float64_result_at_every_port(
    adapter, op, masked, kind
):
    """F2: ``realised_*`` at r use the reference every other port uses (x op requested in
    float64); the cost of realising r through o is ``realisation_*`` (|r_realised - target|)."""
    session = _at_layer(adapter, 1)
    base = _full(adapter, session, 1)
    x = base.captures["r"]
    gen = torch.Generator().manual_seed(11)
    if kind == "scalar":
        request = {"replace": 0.1, "add": 0.1, "scale": 1.1}[op]
    else:
        request = torch.randn(x.shape, generator=gen) * 0.37 + (1.0 if op == "scale" else 0.0)
    mask = None
    if masked:
        mask = torch.zeros(x.shape, dtype=torch.bool)
        mask[..., ::3] = True
    controller = _full(adapter, session, 1, [PortEdit("r", request, op, mask=mask)])
    [record] = controller.manifest()
    r = controller.captures["r"]
    ideal = _ideal64(x, op, request, mask)
    assert record["realised_max_abs_error"] == float((r.double() - ideal).abs().max())
    assert record["realised_exact"] is bool((r.double() == ideal).all())
    # the dtype-arithmetic target the realisation aims at, recomputed independently
    written = torch.as_tensor(request).to(adapter.dtype)
    target = {"replace": written.expand(x.shape), "add": x + written, "scale": x * written}[op]
    if mask is not None:
        target = torch.where(mask, target, x)
    assert record["realisation_max_abs_error"] == float((r.double() - target.double()).abs().max())
    assert record["realisation_exact"] is bool(torch.equal(r, target))
    if masked:
        assert torch.equal(r[~mask], x[~mask]) and record["mask_count"] == int(mask.sum())
        assert torch.equal(controller.captures["o"][~mask], base.captures["o"][~mask])


def test_manifest_numbers_carry_labels(adapter):
    session = _at_layer(adapter, 1)
    base = _full(adapter, session, 1)
    uniform = torch.full_like(base.captures["p"], 2.0 / base.captures["p"].shape[-1])
    [p_entry] = _full(adapter, session, 1, [PortEdit("p", uniform)]).manifest()
    assert p_entry["labels"] == {
        "cast_max_abs_error": "measured", "realised_max_abs_error": "measured",
        "changed_fraction": "measured", "row_sum_min": "measured", "row_sum_max": "measured",
        "min": "measured", "row_sum_tolerance": "supplied",
    }
    assert "row_sum_tolerance_authority" not in p_entry
    mask = torch.zeros(base.captures["r"].shape, dtype=torch.bool)
    mask[..., :4] = True
    [r_entry] = _full(adapter, session, 1, [PortEdit("r", 0.5, "add", mask=mask)]).manifest()
    assert r_entry["labels"] == {
        "cast_max_abs_error": "measured", "realised_max_abs_error": "measured",
        "realisation_max_abs_error": "measured", "changed_fraction": "measured",
    }
    assert "mask_count" not in r_entry["labels"] and "mask_shape" not in r_entry["labels"]


def _one_ulp_native(monkeypatch, adapter):
    modeling = program_for(adapter).modeling_module()
    native = modeling.eager_attention_forward

    def wrapped(module, q, k, v, mask, *a, **kw):
        out, weights = native(module, q, k, v, mask, *a, **kw)
        weights = weights.clone()
        flat = weights.view(-1)
        flat[0] = torch.nextafter(flat[0], torch.tensor(2.0, dtype=flat.dtype))
        return out, weights

    monkeypatch.setattr(modeling, "eager_attention_forward", wrapped)


@pytest.mark.parametrize("port", ["s", "p"])
def test_commit_with_an_s_or_p_edit_is_refused_when_the_replica_check_fails(
    adapter, monkeypatch, port
):
    program = program_for(adapter)
    session = _at_layer(adapter, 1)
    probe = _full(adapter, session, 1)
    edit = (
        PortEdit("s", 0.5, "add") if port == "s"
        else PortEdit("p", torch.full_like(probe.captures["p"], 0.125))
    )
    before = session.capture(retain=False).fingerprint
    receipts = len(session.receipts)
    _one_ulp_native(monkeypatch, adapter)
    controller = PortController(program, 1, edits=[edit], capture=False)
    with pytest.raises(ValueError, match=rf"layer 1: replica check failed.*{port} edit"):
        session.continue_(1, ports=controller)
    assert session.capture(retain=False).fingerprint == before and len(session.receipts) == receipts
    assert controller.replica_verified is False and controller.replica_max_abs_diff > 0
    # an edit elsewhere in the same layer is still committed (the replica is not involved)
    other = PortController(program, 1, edits=[PortEdit("m", 0.5, "add")], capture=False)
    assert session.continue_(1, ports=other).to_dict()["transitions"][0]["port_edits"]


def test_nested_forward_of_the_same_model_runs_natively_and_leaves_the_outer_run_alone(adapter):
    """F4: a user hook that calls ``model(...)`` from inside the pending layer."""
    program = program_for(adapter)
    session = _at_layer(adapter, 1)
    sequence = torch.tensor([PROMPT])
    with torch.no_grad():
        reference_logits = adapter.model(sequence).logits.clone()
    edit = PortEdit("m", 0.5, "scale")
    reference = _full(adapter, session, 1, [edit])
    nested: dict = {}

    def reenter(module, args, output):
        if not nested:
            nested["entered"] = True
            nested["logits"] = adapter.model(sequence).logits.clone()

    handle = adapter.model.model.layers[1].mlp.gate_proj.register_forward_hook(reenter)
    try:
        controller = _full(adapter, session, 1, [edit])
    finally:
        handle.remove()
    assert torch.equal(nested["logits"], reference_logits)  # the nested call was not staged
    assert controller.order == list(program.names)  # no port recorded twice
    for name in program.names:
        assert torch.equal(controller.captures[name], reference.captures[name]), name
    assert len(controller.manifest()) == 1
    _no_hooks(adapter)
    # the controller is engaged again at the next outermost entry
    again = _full(adapter, session, 1, [edit])
    assert torch.equal(again.captures["h_out"], reference.captures["h_out"])


def test_a_masked_r_edit_leaves_o_native_outside_the_mask(adapter):
    """F7 E2d: r = h + o is elementwise, so o changes only where the mask selects."""
    session = _at_layer(adapter, 1)
    base = _full(adapter, session, 1)
    mask = torch.zeros(base.captures["r"].shape, dtype=torch.bool)
    mask[..., :8] = True
    edited = _full(adapter, session, 1, [PortEdit("r", 0.5, "add", mask=mask)])
    assert torch.equal(edited.captures["o"][~mask], base.captures["o"][~mask])
    assert not torch.equal(edited.captures["o"][mask], base.captures["o"][mask])
    assert torch.equal(edited.captures["r"][~mask], base.captures["r"][~mask])
    assert torch.equal(edited.originals["o"], base.captures["o"])  # original o under an r edit


def test_each_concurrent_stop_probe_stops_at_its_own_port(adapter):
    """F7 E6a: two models, two threads, both installs live at once in one shared patch."""
    import threading

    family, dtype = adapter.spec.model_type, {v: k for k, v in DTYPES.items()}[adapter.dtype]
    adapters = [adapter, DecoderAdapter(build_model(family, dtype, seed=999), granularity="operation")]
    sessions = [_at_layer(a, 0) for a in adapters]
    program = program_for(adapter)
    stops = ["s", "p"]
    solo = [
        PortController(program, 0, stop=stop, capture=program.names[: program.index(stop) + 1])
        for stop in stops
    ]
    for controller, session in zip(solo, sessions):
        session.probe(controller)
    barrier = threading.Barrier(2, timeout=30)
    overlap = {"n": 0}

    def meet(module, args, output):
        if threading.current_thread().name.startswith("runner"):
            try:
                barrier.wait()
                overlap["n"] += 1
            except threading.BrokenBarrierError:
                pass

    handles = [a._layer_modules[0].self_attn.q_proj.register_forward_hook(meet) for a in adapters]
    results: list = [None, None]

    def runner(i):
        controller = PortController(
            program, 0, stop=stops[i], capture=program.names[: program.index(stops[i]) + 1]
        )
        try:
            results[i] = (sessions[i].probe(controller), controller)
        except Exception as error:  # noqa: BLE001
            results[i] = error
            barrier.abort()

    threads = [threading.Thread(target=runner, args=(i,), name=f"runner{i}") for i in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
    finally:
        for handle in handles:
            handle.remove()
    assert overlap["n"] == 2, results
    for i, stop in enumerate(stops):
        outcome, controller = results[i]
        assert outcome["stopped"] is True and controller.order[-1] == stop
        assert controller.order == solo[i].order
        for name in controller.order:
            assert torch.equal(controller.captures[name], solo[i].captures[name]), (stop, name)
    assert ports_module._PATCHES == {}
    _no_hooks(adapter)


def test_continue_with_ports_refuses_edits_for_a_layer_that_will_not_run(adapter):
    """F3: refused before anything runs; no receipt, no attempt, cut unchanged."""
    program = program_for(adapter)
    session = _at_layer(adapter, 1)  # pending layer 1
    fresh = adapter.session(PROMPT)  # embed transition pending
    for target, name, controller in (
        (session, "pending L1 with an L0 controller", PortController(program, 0, edits=[PortEdit("m", 0.5, "add")], capture=False)),
        (session, "pending L1 with a stop at L0", PortController(program, 0, stop="m", capture=False)),
        (fresh, "embed with an L0 controller", PortController(program, 0, edits=[PortEdit("m", 0.5, "add")], capture=False)),
    ):
        cut = target.capture(retain=False).fingerprint
        counts = (len(target.receipts), len(target.attempts), len(target.history))
        with pytest.raises(ValueError, match="nothing was run"):
            target.continue_(1, ports=controller)
        assert target.capture(retain=False).fingerprint == cut, name
        assert (len(target.receipts), len(target.attempts), len(target.history)) == counts, name
    # the transition after the next one is reachable: continue_(2) fires on step 2
    controller = PortController(program, 1, edits=[PortEdit("m", 0.5, "add")], capture=False)
    entries = _at_layer(adapter, 0).continue_(2, ports=controller).to_dict()["transitions"]
    assert ["port_edits" in e for e in entries] == [False, True]
    _no_hooks(adapter)


def test_a_controller_with_edits_that_was_never_installed_cannot_finish_cleanly(adapter):
    """F3 defence in depth: ``end_call`` raises when the call never installed the controller."""
    program = program_for(adapter)
    controller = PortController(program, 1, edits=[PortEdit("m", 0.5, "add")], capture=False)
    controller.begin_call()
    with pytest.raises(ValueError, match="never ran that layer"):
        controller.end_call()
    quiet = PortController(program, 1, capture=False)
    quiet.begin_call()
    quiet.end_call()  # nothing carried: nothing to lose
