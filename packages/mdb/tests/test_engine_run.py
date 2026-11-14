import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import saturn_pub  # noqa: E402
from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from test_engine_ports import PROMPT, build_model, head_dim  # noqa: E402,F401

import mdb  # noqa: E402
from mdb.engine import Run  # noqa: E402
from mdb.engine.ports import tensor_digest  # noqa: E402
from mdb.engine.selfcheck import tiny_model  # noqa: E402

CASES = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32"),
         ("llama", "bf16"), ("llama", "fp32"), ("mistral", "bf16"), ("mistral", "fp32")]
ROW = len(PROMPT) - 1  # the first generated row's index; addresses read t{ROW}.<stop>
NEXT = ROW + 1


def _adapter(family, dtype):
    return DecoderAdapter(build_model(family, dtype), granularity="operation")


@pytest.fixture(scope="module", params=CASES, ids=lambda case: "-".join(case))
def case(request):
    return request.param


@pytest.fixture(params=["none", "layer"])
def mode(request):
    return request.param


def Run_(adapter, mode, **kwargs):  # noqa: N802
    return Run(adapter, PROMPT, lookahead=mode, **kwargs)





@pytest.fixture(scope="module")
def adapter(case):
    return _adapter(*case)


def _walk(run, count):
    trace = []
    for _ in range(count):
        run.step("port")
        trace.append((run.where, tensor_digest(run.read())))
    return trace


def test_cursor_stops_units_and_reads(adapter, mode):
    run = Run_(adapter, mode)
    names = run.program.names
    last = adapter.layers - 1
    assert run.where == f"t{ROW}.start"
    assert run.lookahead == mode
    assert run.step() == f"t{ROW}.emb"
    assert run.step() == f"t{ROW}.L0.h"
    assert torch.equal(run.read(), run.read("emb"))
    assert run.next() == f"t{ROW}.L0.r"
    assert run.next() == f"t{ROW}.L0.h_out"
    assert run.step("layer") == f"t{ROW}.L1.h_out"
    assert run.step("layer", last - 1) == f"t{ROW}.L{last}.h_out"
    assert run.step("port", 3) == f"t{ROW}.sample"
    assert run.step("token") == f"t{NEXT}.start"
    assert run.until("L1.g") == f"t{NEXT}.L1.g"
    g = run.read()
    assert run.finish() == f"t{NEXT}.L1.h_out"
    assert run.until(f"t{NEXT}.logits") == f"t{NEXT}.logits"
    assert torch.equal(run.read("L1.g"), g)
    assert torch.equal(run.read(f"t{NEXT}.L1.h_out"), run.read("L1.h_out"))
    first = run.session.read("tokens")[0, NEXT].item()
    assert run.read(f"t{ROW}.sample").tolist() == [[first]]
    assert run.read(f"t{NEXT}.logits").shape == (1, adapter.vocab_size)
    with pytest.raises(ValueError, match="behind the cursor"):
        run.until(f"t{NEXT}.L0.a")
    with pytest.raises(ValueError, match="not been computed"):
        run.read(f"t{NEXT}.sample")
    with pytest.raises(ValueError, match="before the start"):
        run.read(f"t{ROW - 2}.fn")
    assert len(names) == len(set(names))


def test_beyond_the_cursor_is_refused_in_both_modes(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L1.q")
    with pytest.raises(ValueError, match="not been computed"):
        run.read("L1.k")
    with pytest.raises(ValueError, match="not been computed"):
        run.read("L1.h_out")
    assert run.read("L1.q").shape[-1] == adapter.num_heads * head_dim(adapter)


def test_step_through_a_layer_by_lookahead_mode(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L0.h")
    controller = run._cache
    seen = [run.where]
    for _ in range(len(run.program.names) - 2):
        run.step()
        seen.append(run.where)
        if mode == "layer":
            assert run._cache is controller
        else:
            assert run._cache is not controller and run._cache.installs == 1
            assert run._cache.stop == run._port
            controller = run._cache
    assert seen[-1] == f"t{ROW}.L0.m"
    if mode == "layer":
        assert controller.installs == 1
    run.step()
    assert run.where == f"t{ROW}.L0.h_out" and run._cache is None


def test_port_digests_are_identical_across_lookahead_modes(adapter):
    traces = {}
    for name in ("none", "layer"):
        run = Run_(adapter, name)
        run.until("L1.h")
        run.set("L1.g", 2.0, "scale")
        traces[name] = _walk(run, 2 * len(run.program.names) + 9)
        traces[name].append(run.checkpoint()["cut"])
    assert traces["none"] == traces["layer"]


def test_t3_rewind_within_and_across_layers_and_tokens(adapter, mode):
    run = Run_(adapter, mode)
    last = adapter.layers - 1
    for start, crossing in (("L0.f", f"t{ROW}.L1.h"), (f"t{ROW}.L{last}.m", f"t{NEXT}.start")):
        run.until(start)
        checkpoint = run.checkpoint(start)
        assert checkpoint["lookahead"] == mode
        assert len(json.dumps(checkpoint)) < 1024
        first = _walk(run, 7)
        assert crossing in [where for where, _ in first]
        assert run.restore(checkpoint) == checkpoint["where"]
        assert _walk(run, 7) == first
        run.restore(checkpoint)
        reversed_trace = []
        run.step("port", 7)
        for _ in range(7):
            reversed_trace.append((run.where, tensor_digest(run.read())))
            run.reverse_step()
        assert run.where == checkpoint["where"]
        assert reversed_trace[::-1] == first
        run.step("port", 7)
    assert run.reverse_step("token") == f"t{NEXT}.start"
    assert run.reverse_step("layer") == f"t{ROW}.sample"
    assert run.reverse_step("sublayer", 3) == f"t{ROW}.L{last}.h_out"


def test_t3_rewind_with_staged_and_committed_edits(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L0.u")
    noise = torch.randn(run.read("L0.v").shape, generator=torch.Generator().manual_seed(5))
    run.set("L0.v", noise.to(adapter.dtype), "add")
    run.set("L0.g", 2.0, "scale")
    edited_u = run.read()
    run.finish()
    run.until("L1.a")
    run.set("L1.m", 0.25, "add")
    checkpoint = run.checkpoint("edited")
    assert [entry["layer"] for entry in checkpoint["path_edits"]] == [0]
    first = _walk(run, 7)
    run.reverse_step("token")
    assert run.where == f"t{ROW}.start"
    assert run.restore(checkpoint) == f"t{ROW}.L1.a"
    assert run.edits == checkpoint["edits"]
    assert _walk(run, 7) == first
    run.until(f"t{NEXT}.emb")
    back = len(run.stops()) - run._index["L0.u"] + run._index["emb"]
    run.reverse_step("port", back)
    assert run.where == f"t{ROW}.L0.u"
    assert torch.equal(run.read(), edited_u)
    assert [edit["port"] for edit in run.edits] == ["v", "g"]


def test_set_commit_abort_and_refusals(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L1.n")
    base = {name: run.read(f"L1.{name}") for name in ("r", "n")}
    manifest = run.set("n", 3.0, "scale")
    assert manifest == {"port": "n", "op": "scale", "value": {"kind": "scalar", "value": 3.0}}
    assert not torch.equal(run.read("L1.n"), base["n"])
    assert torch.equal(run.read("L1.r"), base["r"])
    run.abort()
    assert run.edits == [] and torch.equal(run.read("L1.n"), base["n"])
    run.set("m", 1.0, "add")
    before = run.checkpoint()
    assert run.commit() == f"t{ROW}.L1.h_out"
    entry = run.session.receipts[-1].to_dict()["transitions"][0]
    assert entry["port_edits"][0]["port"] == "m"
    assert run.checkpoint()["cut"] != before["cut"]
    with pytest.raises(ValueError, match="pending layer"):
        run.set("L1.g", 1.0, "add")
    run.step()
    with pytest.raises(ValueError, match="pending layer"):
        run.set("L0.g", 1.0)
    with pytest.raises(ValueError, match="pending layer"):
        Run_(adapter, mode).set("L0.g", 1.0, "scale")
    fresh = Run_(adapter, mode)
    fresh.step()
    with pytest.raises(ValueError, match="not editable"):
        fresh.set("L0.kr", 1.0, "scale")
    fresh = Run_(adapter, mode)
    fresh.until("L0.q")
    with pytest.raises(ValueError, match="needs shape"):
        fresh.set("L0.g", torch.zeros(2))
    assert fresh.edits == [] and fresh.where == f"t{ROW}.L0.q"
    with pytest.raises(ValueError, match="port g.*port shape.*value shape"):
        fresh.set("L0.g", torch.zeros(5, 7), "add")
    assert fresh.edits == [] and fresh.where == f"t{ROW}.L0.q"
    with pytest.raises(ValueError, match="granularity='operation'"):
        Run(DecoderAdapter(tiny_model("qwen2"), granularity="layer"), PROMPT)
    with pytest.raises(ValueError, match="lookahead must be one of"):
        Run(adapter, PROMPT, lookahead="whole")


def test_retained_rows_replay_dropped_cuts(adapter, mode):
    run = Run_(adapter, mode, retain_rows=0)
    run.until(f"t{ROW}.L1.h_out")
    hidden = run.read()
    run.step("token", 2)
    dropped = [mark for mark in run._trail if mark.cut is None]
    assert dropped and all(mark.label != "start" for mark in dropped)
    assert torch.equal(run.read(f"t{ROW}.L1.h_out"), hidden)
    run.reverse_step("token", 2)
    assert run.where == f"t{ROW}.start"
    run.until(f"t{ROW}.L1.h_out")
    assert torch.equal(run.read(), hidden)


# --- fix 6: honest stops ---------------------------------------------------------------------


def test_none_mode_computes_nothing_beyond_the_cursor_and_layer_mode_does(adapter):
    program_names = Run_(adapter, "none").program.names
    for mode_name in ("none", "layer"):
        run = Run_(adapter, mode_name)
        fired = []
        hook = adapter.model.model.layers[1].mlp.down_proj.register_forward_hook(
            lambda module, args, output: fired.append(1)
        )
        try:
            run.until("L1.q")
            cursor = program_names.index("q")
            controller = run._cache
            if mode_name == "none":
                assert not fired
                assert set(controller.captures) == set(program_names[: cursor + 1])
                assert set(controller.order) <= set(program_names[: cursor + 1])
                assert controller.order[-1] == "q"
                assert set(controller.originals) <= set(program_names[: cursor + 1])
                with pytest.raises(ValueError, match="not been computed"):
                    run.read("L1.m")
                run.step("port", 3)  # k, v, qr
                assert set(run._cache.captures) == set(program_names[: cursor + 4])
                run.reverse_step()
                assert set(run._cache.captures) == set(program_names[: cursor + 3])
                assert not fired
            else:
                assert fired and set(controller.captures) == set(program_names)
        finally:
            hook.remove()


def test_status_and_checkpoint_state_the_lookahead_mode(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L0.a")
    assert run.status() == {
        "where": f"t{ROW}.L0.a",
        "port": "a",
        "pending_layer": 0,
        "lookahead": mode,
        # layer mode: the full-layer probe has already computed every port after the cursor
        "computed_past_cursor": [] if mode == "none" else list(run.program.names[2:]),
    }
    assert run.checkpoint()["lookahead"] == mode


def test_edit_beyond_the_cursor_is_validated_without_holding_later_ports(adapter):
    run = Run_(adapter, "none")
    run.until("L0.a")
    run.set("L0.m", 0.5, "scale")
    assert set(run._cache.captures) == set(run.program.names[:2])
    with pytest.raises(ValueError, match="needs shape"):
        run.set("L0.g", torch.zeros(2))
    assert [edit["port"] for edit in run.edits] == ["m"]
    reference = Run_(adapter, "layer")
    reference.until("L0.a")
    reference.set("L0.m", 0.5, "scale")
    assert run.finish() == reference.finish()
    assert run.checkpoint()["cut"] == reference.checkpoint()["cut"]


def test_r_edit_realises_o_at_the_o_stop_in_none_mode(adapter):
    plain = Run_(adapter, "none")
    plain.until("L0.o")
    native_o = plain.read()
    runs = {}
    for mode_name in ("none", "layer"):
        run = Run_(adapter, mode_name)
        run.until("L0.a")
        run.set("L0.r", torch.randn(1, 1, 64, generator=torch.Generator().manual_seed(2)).to(adapter.dtype), "add")
        run.until("L0.o")
        runs[mode_name] = run
    assert torch.equal(runs["none"].read(), runs["layer"].read())
    assert not torch.equal(runs["none"].read(), native_o)
    for run in runs.values():
        info = run.read_info("L0.o")
        assert info["edited"] is False and info["by"] == "r" and info["kind"] == "realisation"
        assert info["authority"] in ("derived-exact", "derived-local")
    native = plain.read_info("L0.o")
    assert native["by"] is None and native["authority"] == "measured" and native["kind"] == "native"
    assert runs["none"].read_info("L0.a")["edited"] is False
    assert runs["none"].read_info("emb")["authority"] == "measured"


def test_read_info_for_supplied_s_and_replica_p(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L1.k")
    shape = run.program.spec("s")
    del shape
    run.until("L1.p")
    base_s = run.read("L1.s")
    noise = torch.randn(base_s.shape, generator=torch.Generator().manual_seed(4)).to(base_s.dtype)
    run.set("L1.s", noise, "add")
    assert run.read_info("L1.s")["by"] == "s"
    assert run.read_info("L1.s")["authority"] == "derived-exact"  # add: native + supplied operand
    assert run.read_info("L1.s")["replica_equal"] is True
    info = run.read_info("L1.p")
    assert info["authority"] == "derived-exact" and info["by"] == "s" and info["kind"] == "replica"
    run.abort()
    assert run.read_info("L1.s")["edited"] is False


def test_p_and_s_edits_committed_through_run_carry_replica_labels(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L1.a")
    p = run.read("L1.a")
    del p
    run.until("L1.p")
    base_p = run.read("L1.p")
    base_s = run.read("L1.s")
    uniform = torch.full_like(base_p, 1.0 / base_p.shape[-1])
    run.set("L1.p", uniform)
    run.set("L1.s", torch.randn(base_s.shape, generator=torch.Generator().manual_seed(6)).to(base_s.dtype), "add")
    run.commit()
    entries = run.session.receipts[-1].to_dict()["transitions"][0]["port_edits"]
    by_port = {entry["port"]: entry for entry in entries}
    assert set(by_port) == {"p", "s"}
    for entry in entries:
        assert entry["mix"] == "replica" and entry["replica_verified"] is True
    assert by_port["p"]["is_distribution"] is True
    assert set(by_port["p"]) >= {"row_sum_min", "row_sum_max", "min", "is_distribution"}
    assert "mix" not in {k for e in entries if e["port"] not in ("p", "s") for k in e}


def test_non_distribution_p_edit_through_run_is_recorded(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L1.p")
    base_p = run.read("L1.p")
    run.set("L1.p", torch.full_like(base_p, 0.5))
    run.commit()
    [entry] = run.session.receipts[-1].to_dict()["transitions"][0]["port_edits"]
    assert entry["is_distribution"] is False
    assert entry["row_sum_min"] == entry["row_sum_max"] == 0.5 * base_p.shape[-1]
    assert entry["min"] == 0.5 and entry["mix"] == "replica"


# --- fix 2: training mode / dropout refused ---------------------------------------------------


def _train_model(family="qwen2"):
    model = build_model(family, "fp32")
    for layer in model.model.layers:
        layer.self_attn.attention_dropout = 0.5
    model.config.attention_dropout = 0.5
    return model


def test_train_mode_with_dropout_refused_at_construction(mode):
    model = _train_model()
    adapter = DecoderAdapter(model, granularity="operation")
    model.train()
    with pytest.raises(ValueError, match="eval-mode"):
        Run_(adapter, mode)
    model.eval()
    Run_(adapter, mode)


def test_train_mode_refused_at_commit_without_edits(mode):
    model = _train_model()
    run = Run_(DecoderAdapter(model, granularity="operation"), mode)
    run.step()
    before = run.checkpoint()["cut"]
    model.train()
    with pytest.raises(ValueError, match="eval-mode"):
        run.step("token")
    assert run.checkpoint()["cut"] == before
    model.eval()
    run.step("token")


def test_train_mode_refused_at_commit_with_edits(mode):
    model = _train_model()
    run = Run_(DecoderAdapter(model, granularity="operation"), mode)
    run.until("L0.a")
    run.set("L0.m", 0.5, "scale")
    model.train()
    with pytest.raises(ValueError, match="eval-mode"):
        run.commit()
    assert run.where == f"t{ROW}.L0.a" and len(run.session.receipts) == 1
    model.eval()
    run.commit()


def test_train_mode_refused_at_probe(mode):
    model = _train_model("qwen3")
    run = Run_(DecoderAdapter(model, granularity="operation"), mode)
    run.until("L0.h")
    model.train()
    if mode == "none":  # lookahead="layer" serves the next port from its cache: no probe runs
        with pytest.raises(ValueError, match="eval-mode"):
            run.step()
    with pytest.raises(ValueError, match="eval-mode"):
        run.set("L0.g", 1.0, "scale")
    assert run.edits == [] and len(run.session.receipts) == 1
    model.eval()
    run.until("L0.q")


_CHILD = """
import json, sys
sys.path.insert(0, sys.argv[2])
from saturn_pub.adapters.decoder import DecoderAdapter
from mdb.engine import Run
from mdb.engine.ports import tensor_digest
from test_engine_ports import build_model
import torch
args = json.loads(sys.argv[1])
scope = args["checkpoint"]["numeric_scope"]  # a cut binds the numeric scope: match the parent's
torch.set_num_threads(scope["torch_num_threads"])
torch.use_deterministic_algorithms(scope["deterministic_algorithms"])
torch.set_float32_matmul_precision(scope["float32_matmul_precision"])
torch.backends.cuda.matmul.allow_tf32 = scope["tf32_matmul"]
adapter = DecoderAdapter(build_model(args["family"], args["dtype"]), granularity="operation")
run = Run(adapter, args["tokens"], lookahead=args["lookahead"])
run.restore(args["checkpoint"])
trace = [run.where]
for _ in range(7):
    run.step("port")
    trace.append([run.where, tensor_digest(run.read())])
print(json.dumps(trace))
"""


def test_t3_restore_in_a_fresh_process(case, adapter, mode):
    run = Run_(adapter, mode)
    run.until("L0.p")
    run.set("L0.a", 0.5, "scale")
    run.finish()
    run.until(f"t{ROW}.L1.sg")
    run.set("L1.u", 0.125, "add")
    checkpoint = json.loads(json.dumps(run.checkpoint("child")))
    expected = [run.where] + [list(item) for item in _walk(run, 7)]
    paths = [str(Path(module.__file__).parents[1]) for module in (saturn_pub, mdb)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    payload = json.dumps(
        {
            "family": case[0],
            "dtype": case[1],
            "tokens": PROMPT,
            "checkpoint": checkpoint,
            "lookahead": mode,
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, payload, str(Path(__file__).parent)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == expected


def test_restore_names_the_numeric_scope_that_differs():
    before = torch.get_num_threads()
    other = 2 if before != 2 else 3
    run = Run_(_adapter("qwen2", "fp32"), "none")
    run.until("L0.p")
    checkpoint = json.loads(json.dumps(run.checkpoint("scoped")))
    assert checkpoint["numeric_scope"]["torch_num_threads"] == before
    try:
        torch.set_num_threads(other)
        if torch.get_num_threads() != other:
            pytest.skip("this torch build does not change its thread count")
        fresh = Run_(_adapter("qwen2", "fp32"), "none")
        with pytest.raises(ValueError, match="torch_num_threads: checkpoint .* this process"):
            fresh.restore(checkpoint)
        assert fresh.where == Run_(_adapter("qwen2", "fp32"), "none").where  # nothing moved
    finally:
        torch.set_num_threads(before)
    legacy = {key: value for key, value in checkpoint.items() if key != "numeric_scope"}
    assert Run_(_adapter("qwen2", "fp32"), "none").restore(legacy) == run.where


@pytest.mark.parametrize("flag", ["bf16", "fp16"])
def test_reduced_precision_flag_is_bound_and_refused(flag):
    """M1: the CUDA reduced-precision reduction flags are in the numeric scope; a checkpoint
    taken under one value refuses to restore under the other and names the field."""
    attribute = f"allow_{flag}_reduced_precision_reduction"
    field = f"{flag}_reduced_precision_reduction"
    matmul = torch.backends.cuda.matmul
    before = getattr(matmul, attribute)
    try:
        setattr(matmul, attribute, True)
        run = Run_(_adapter("qwen2", "bf16"), "none")
        run.until("L0.p")
        assert run.numeric_scope[field] is True
        checkpoint = json.loads(json.dumps(run.checkpoint("flagged")))
        setattr(matmul, attribute, False)
        with pytest.raises(ValueError, match=f"{field}: recorded True, current False"):
            run.step()  # the live adapter refuses to execute under the drifted flag
        fresh = Run_(_adapter("qwen2", "bf16"), "none")  # rewrapped under the other flag
        assert fresh.numeric_scope[field] is False
        with pytest.raises(ValueError, match=f"{field}: checkpoint True, this process False"):
            fresh.restore(checkpoint)
        with pytest.raises(ValueError, match=f"numeric_scope.{field}: cut True, adapter False"):
            Session.from_cut(fresh.adapter, run._trail[-1].cut)
    finally:
        setattr(matmul, attribute, before)


# --- round 2 ---------------------------------------------------------------------------------

INFO_KEYS = {
    "edited", "by", "kind", "authority", "state", "reason", "ops", "selective",
    "upstream_edits", "replica_equal", "replica_checked", "max_abs_diff", "max_abs_diff_authority",
    "realisation_exact", "realisation_max_abs_error", "same_port_edit",
}  # fmt: skip


def _noise(run, address, seed=3):
    shape = run.port_shape(address.split(".")[-1])
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _info_at(run, address):
    run.until(address)
    info = run.read_info()
    assert set(info) == INFO_KEYS
    return info


def test_read_info_native_edit_and_masked_authorities(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L1.h")
    native = run.read_info("L1.h")
    assert native == {
        "edited": False, "by": None, "kind": "native", "authority": "measured",
        "state": "resolved", "reason": None, "ops": [], "selective": False,
        "upstream_edits": [], "replica_equal": None, "replica_checked": None, "max_abs_diff": None,
        "max_abs_diff_authority": None, "realisation_exact": None,
        "realisation_max_abs_error": None, "same_port_edit": None,
    }  # fmt: skip
    dtype = adapter.dtype
    mask = torch.zeros(run.port_shape("m"), dtype=torch.bool)
    mask[..., :4] = True
    run.set("L1.g", _noise(run, "g").to(dtype))
    run.set("L1.u", 2.0, "scale")
    run.set("L1.f", _noise(run, "f").to(dtype), "add")
    run.set("L1.m", 0.5, mask=mask, selection="first four")
    run.until("L1.m")
    for port, ops, authority, selective in (
        ("g", ["replace"], "supplied", False),
        ("u", ["scale"], "derived-exact", False),
        ("f", ["add"], "derived-exact", False),
        # a masked replace is the cast of the supplied tensor only on the masked elements
        ("m", ["replace"], "derived-exact", True),
    ):
        info = run.read_info(f"L1.{port}")
        assert set(info) == INFO_KEYS
        assert (info["edited"], info["by"], info["kind"]) == (True, port, "edit")
        assert (info["ops"], info["authority"], info["selective"]) == (ops, authority, selective)
    masked = run.read_info("L1.m", select=mask)
    assert (masked["edited"], masked["authority"], masked["same_port_edit"]) == (True, "supplied", None)
    outside = run.read_info("L1.m", select=~mask)
    assert (outside["edited"], outside["by"], outside["kind"]) == (False, None, "native")
    assert outside["authority"] == "measured"
    assert outside["same_port_edit"] == {"port": "m", "ops": ["replace"]}
    assert run.read_info("L1.sg")["kind"] == "native"
    assert run.read_info("L1.sg")["authority"] == "measured"
    sg = run.read_info("L1.sg")["upstream_edits"]
    assert [(e["port"], e["op"], e["staged"]) for e in sg] == [("g", "replace", True)]
    assert sg[0]["row"] == ROW and sg[0]["layer"] == 1


def test_read_info_r_edit_realisation_and_replicas(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L1.a")
    run.set("L1.r", _noise(run, "r").to(adapter.dtype), "add")
    info = _info_at(run, "L1.o")
    assert (info["edited"], info["by"], info["kind"]) == (False, "r", "realisation")
    run.until("L1.n")
    [record] = run.staged_info()
    assert record["fired"] is True
    assert info["authority"] == "derived-exact"  # o' = target - h is computed exactly
    assert info["realisation_exact"] is record["realisation_exact"]
    assert info["realisation_max_abs_error"] == record["realisation_max_abs_error"]
    r = run.read_info("L1.r")
    assert (r["edited"], r["by"]) == (True, "r") and r["realisation_exact"] is record["realisation_exact"]
    if record["realisation_exact"]:
        assert (r["kind"], r["authority"]) == ("edit", "derived-exact")  # an add, not a replace
    else:  # r is what the native add h + o' produced
        assert (r["kind"], r["authority"]) == ("realisation", "measured")
    n = run.read_info()
    assert n["kind"] == "native" and [e["port"] for e in n["upstream_edits"]] == ["r"]
    run.abort()
    fresh = Run_(adapter, mode)
    s = _info_at(fresh, "L1.s")
    assert (s["kind"], s["by"], s["edited"]) == ("replica", None, False)
    assert s["replica_equal"] is (None if mode == "none" else True)
    if mode == "none":  # the stop at s ends before the native call: the replica is unchecked
        # derived from the captured operands by the stated formula; only the comparison is pending
        assert (s["state"], s["authority"], s["reason"]) == ("resolved", "derived-exact", None)
        assert s["replica_checked"] is False
        fresh.step()
    assert fresh.read_info("L1.s")["replica_checked"] is True
    assert fresh.read_info("L1.s")["replica_equal"] is True
    assert fresh.read_info("L1.s")["max_abs_diff_authority"] == "measured"
    assert (fresh.read_info("L1.s")["state"], fresh.read_info("L1.s")["authority"]) == (
        "resolved", "derived-exact")
    for edited, by in (("s", "s"), ("p", "p"), ("sp", "s,p")):
        run = Run_(adapter, mode)
        run.until("L1.a")
        if "s" in edited:
            run.set("L1.s", _noise(run, "s").to(adapter.dtype), "add")
        if "p" in edited:
            run.set("L1.p", torch.full(run.port_shape("p"), 1.0 / len(PROMPT)).to(adapter.dtype))
        a = _info_at(run, "L1.A")
        assert (a["edited"], a["kind"], a["authority"], a["by"]) == (False, "replica", "derived-exact", by)
        assert a["replica_equal"] is True and a["max_abs_diff"] == 0.0
        p = run.read_info("L1.p")
        if edited == "s":
            assert (p["kind"], p["authority"], p["by"]) == ("replica", "derived-exact", "s")
        if "p" in edited:
            assert p["edited"] is True and p["kind"] == "edit"
            assert p["by"] == by.replace("s,", "s,") and p["authority"] == "supplied"
    plain = Run_(adapter, mode)
    assert _info_at(plain, "L1.A")["kind"] == "native"


def test_upstream_edits_carry_the_selection_staged_and_committed(adapter, mode):
    """M37: an upstream entry names the edit's selector (``h0``), None for a whole port."""
    run = Run_(adapter, mode)
    run.until("L0.h")
    heads = torch.zeros(run.port_shape("a"), dtype=torch.bool)
    heads[..., :4] = True
    run.set("L0.a", 0.5, "scale", mask=heads, selection="h0")
    assert [e["selection"] for e in _info_at(run, "L0.n")["upstream_edits"]] == ["h0"]  # staged
    run.finish()
    run.until("L2.q")
    ups = run.read_info()["upstream_edits"]
    assert [(e["port"], e["staged"], e["selection"]) for e in ups] == [("a", False, "h0")]
    run.set("L2.m", 0.25, "add")  # a whole-port edit has no selection
    run.finish()
    run.step("token")
    run.until("L0.q")
    assert [(e["port"], e["selection"]) for e in run.read_info()["upstream_edits"]] == [
        ("a", "h0"), ("m", None)]


def test_read_info_n_under_an_a_edit_and_committed_upstream_edits(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L0.h")
    run.set("L0.a", 0.5, "scale")
    n = _info_at(run, "L0.n")
    assert (n["kind"], n["authority"], n["edited"]) == ("native", "measured", False)
    assert [(e["port"], e["op"], e["staged"]) for e in n["upstream_edits"]] == [("a", "scale", True)]
    run.finish()
    run.until("L2.q")
    q = run.read_info()
    assert q["kind"] == "native" and q["authority"] == "measured"
    assert q["upstream_edits"] == [{"row": ROW, "layer": 0, "port": "a", "op": "scale", "staged": False,
                                    "selection": None}]
    # committed layer-0 value re-probed after the fact still lists its own edit's position
    assert run.read_info("L0.m")["upstream_edits"][0]["staged"] is False
    run.set("L2.m", 0.25, "add")
    run.finish()
    run.step("token")
    run.until("L0.q")
    later = run.read_info()
    assert [(e["row"], e["layer"], e["port"], e["staged"]) for e in later["upstream_edits"]] == [
        (ROW, 0, "a", False),
        (ROW, 2, "m", False),
    ]
    assert run.read_info(f"t{ROW}.L2.h_out")["upstream_edits"][0]["port"] == "a"


# --- E4: honest stops ------------------------------------------------------------------------


def test_p_stop_discloses_the_native_a_and_s_stop_runs_no_attention(adapter):
    modeling = program_for_run(adapter).modeling_module()
    native = modeling.eager_attention_forward
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return native(*args, **kwargs)

    modeling.eager_attention_forward = counting
    try:
        run = Run_(adapter, "none")
        run.until("L1.kr")
        calls.clear()  # the walk through layer 0 and the L1.h..kr probes ran native attention
        assert run.status()["computed_past_cursor"] == [] and not calls
        run.step()
        assert run.where.endswith("L1.s") and not calls
        run.step()
        assert run.where.endswith("L1.p") and len(calls) == 1
        assert run.status()["computed_past_cursor"] == ["A"]
        assert "A" not in run._cache.captures and "mix_replica" not in run.checks
        run.step()
        assert run.status()["computed_past_cursor"] == [] and run.checks["mix_replica"] == {"equal": True}
        layer = Run_(adapter, "layer")
        layer.until("L1.p")
        names = layer.program.names
        assert layer.status()["computed_past_cursor"] == list(names[names.index("p") + 1 :])
        assert layer.status()["computed_past_cursor"][0] == "A"
    finally:
        modeling.eager_attention_forward = native


def program_for_run(adapter):
    from mdb.engine.ports import program_for

    return program_for(adapter)


@pytest.mark.parametrize("masked", [False, True])
def test_set_ahead_of_the_cursor_executes_nothing_past_it(adapter, masked):
    run = Run_(adapter, "none")
    layer = adapter.model.model.layers[1]
    fired = []
    handles = [
        module.register_forward_hook(lambda m, a, o, n=name: fired.append(n))
        for name, module in layer.named_modules()
        if name
    ]
    try:
        run.until("L1.a")
        assert fired == ["input_layernorm"]
        fired.clear()
        mask = None
        if masked:
            mask = torch.ones(run.port_shape("m"), dtype=torch.bool)
        run.set("L1.m", 0.5, "scale", mask=mask)
        run.set("L1.p", torch.full(run.port_shape("p"), 0.125).to(adapter.dtype), mask=None)
        # only the (re-)probe up to the cursor ran; nothing after input_layernorm
        assert set(fired) <= {"input_layernorm"}
        before = list(fired)
        with pytest.raises(ValueError, match="needs shape"):
            run.set("L1.g", torch.zeros(2))
        with pytest.raises(ValueError, match="port shape"):
            run.set("L1.g", torch.zeros(5, 7), "add")
        assert fired[len(before):] == []
    finally:
        for handle in handles:
            handle.remove()


def test_set_at_a_boundary_runs_no_native_code(adapter):
    run = Run_(adapter, "none")
    run.until("emb")
    seen = []
    handle = adapter.model.register_forward_hook(lambda m, a, o: seen.append(1))
    hooks = [
        m.register_forward_hook(lambda m_, a, o: seen.append(1)) for m in adapter.model.modules()
        if m is not adapter.model
    ]
    try:
        run.set("L0.m", 0.5, "scale")
        run.set("L0.p", 0.5, "scale", mask=torch.ones(run.port_shape("p"), dtype=torch.bool))
    finally:
        handle.remove()
        for h in hooks:
            h.remove()
    assert seen == [] and len(run.edits) == 2


def test_port_shape_matches_probe_shapes(adapter):
    run = Run_(adapter, "layer")
    run.until("L0.m")
    for name in run.program.names[:-1]:
        assert run.port_shape(name) == tuple(run.read(f"L0.{name}").shape), name


# --- E2: atomic set, masks, edit_mask --------------------------------------------------------


@pytest.mark.parametrize("failure", [RuntimeError("boom"), KeyboardInterrupt()])
def test_set_is_atomic_when_the_probe_fails(adapter, mode, failure, monkeypatch):
    run = Run_(adapter, mode)
    run.until("L1.k")
    run.set("L1.g", 2.0, "scale")
    cache, edits = run._cache, run.edits
    original = Run._ensure_cache

    def failing(self, port=None):
        raise failure

    monkeypatch.setattr(Run, "_ensure_cache", failing)
    with pytest.raises(type(failure)):
        run.set("L1.q", 0.5, "scale", mask=torch.ones(run.port_shape("q"), dtype=torch.bool))
    monkeypatch.setattr(Run, "_ensure_cache", original)
    assert run.edits == edits and run._cache is cache
    assert run.edit_mask("L1.q") is None and len(run.staged_info()) == 1
    run.set("L1.q", 0.5, "scale")
    assert [e["port"] for e in run.edits] == ["g", "q"]


def test_edit_mask_is_the_union_of_what_edits_write(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L1.a")
    assert run.edit_mask("L1.g") is None
    first = torch.zeros(run.port_shape("g"), dtype=torch.bool)
    first[..., :5] = True
    second = torch.zeros(run.port_shape("g"), dtype=torch.bool)
    second[..., 3:9] = True
    run.set("L1.g", 1.0, "add", mask=first, selection="a")
    assert torch.equal(run.edit_mask("L1.g"), first) and run.edit_mask("L1.u") is None
    run.set("L1.g", 1.0, "add", mask=second[..., :], selection="b")
    assert torch.equal(run.edit_mask("L1.g"), first | second)
    run.set("L1.u", 3.0, "scale")
    whole = run.edit_mask("L1.u")
    assert whole.dtype == torch.bool and bool(whole.all()) and tuple(whole.shape) == run.port_shape("u")
    assert run.edit_mask("L0.g") is None
    run.abort()
    assert run.edit_mask("L1.g") is None


def test_masked_run_edit_is_bit_identical_elsewhere_and_recorded(adapter, mode):
    reference = Run_(adapter, mode)
    reference.until("L1.f")
    base = reference.read()
    run = Run_(adapter, mode)
    run.until("L1.a")
    mask = torch.zeros(run.port_shape("f"), dtype=torch.bool)
    mask[..., ::2] = True
    manifest = run.set("L1.f", 0.0, mask=mask, selection="even units")
    assert manifest["selection"] == "even units" and manifest["mask_count"] == int(mask.sum())
    assert len(manifest["mask_sha256"]) == 64
    run.until("L1.f")
    got = run.read()
    assert torch.equal(got[..., 1::2], base[..., 1::2]) and bool((got[..., ::2] == 0).all())
    info = run.read_info()
    assert info["selective"] is True and info["authority"] == "derived-exact"
    assert run.read_info(select=mask)["authority"] == "supplied"
    run.commit()
    [entry] = run.session.receipts[-1].to_dict()["transitions"][0]["port_edits"]
    assert entry["selection"] == "even units" and entry["mask_count"] == int(mask.sum())
    [record] = run.committed_edits()
    assert record["selection"] == "even units" and record["changed_fraction"] > 0


# --- E8, E9 ----------------------------------------------------------------------------------


def test_staged_info_reports_requests_and_realised_fields(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L1.a")
    assert run.staged_info() == []
    run.set("L1.m", 0.5, "scale")
    run.set("L1.p", torch.full(run.port_shape("p"), 1.0 / len(PROMPT)).to(adapter.dtype))
    m, p = run.staged_info()
    assert (m["port"], m["op"], m["selection"], m["layer"]) == ("m", "scale", None, 1)
    assert m["value"] == {"kind": "scalar", "value": 0.5}
    assert m["fired"] is (mode == "layer") and (mode == "none" or m["realised_exact"] is True)
    run.until("L1.m")
    m, p = run.staged_info()
    assert m["fired"] and p["fired"]
    assert {"realised_exact", "realised_max_abs_error", "cast_exact", "cast_max_abs_error", "changed_fraction"} <= set(m)
    assert p["mix"] == "replica" and p["replica_verified"] is True and p["is_distribution"] is True
    none_p = Run_(adapter, "none")
    none_p.until("L1.a")
    none_p.set("L1.p", torch.full(none_p.port_shape("p"), 0.125).to(adapter.dtype))
    none_p.until("L1.p")
    [record] = none_p.staged_info()
    assert record["fired"] and record["replica_verified"] is None  # mix is checked at A


def test_read_cache_reads_the_slot_without_hashing(adapter, mode, monkeypatch):
    import hashlib

    run = Run_(adapter, mode)
    run.until("L1.h")
    reference = run.session.capture(retain=False).payload
    sizes = []
    real = hashlib.sha256

    def counting(data=b"", *args, **kwargs):
        sizes.append(len(data))
        return real(data, *args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", counting)
    got = {(layer, kind): run.read_cache(layer, kind) for layer in range(adapter.layers) for kind in "kv"}
    # Session.read re-verifies the execution ABI (one small config digest per call); no tensor
    # or state is hashed
    small = list(sizes)
    assert len(small) == len(got) and max(small) < 1 << 16
    sizes.clear()
    run.session.capture(retain=False)
    assert max(sizes) > 4 * max(small)  # a capture does hash the state
    monkeypatch.undo()
    for (layer, kind), tensor in got.items():
        assert torch.equal(tensor, reference["keys" if kind == "k" else "values"][layer])
    with pytest.raises(ValueError, match="kind"):
        run.read_cache(0, "x")
    with pytest.raises(ValueError, match="layer"):
        run.read_cache(adapter.layers, "k")


# --- E5 (D2), E6, E7 --------------------------------------------------------------------------


def test_only_the_final_norm_in_train_mode_refuses_a_probe():
    model = build_model("qwen2", "fp32")
    run = Run(DecoderAdapter(model, granularity="operation"), PROMPT, lookahead="none")
    run.until("L0.h")
    model.model.norm.train()
    try:
        with pytest.raises(ValueError, match="eval-mode"):
            run.step()
        assert run.where == f"t{ROW}.L0.h"
    finally:
        model.eval()
    run.step()


def test_two_runs_on_two_models_interleave_exactly(case):
    family, dtype = case
    a, b = _adapter(family, dtype), DecoderAdapter(build_model(family, dtype, seed=77), granularity="operation")

    def script(run):
        run.until("L0.a")
        run.set("L0.m", 0.5, "scale")
        run.set("L0.p", torch.full(run.port_shape("p"), 1.0 / len(PROMPT)).to(run.adapter.dtype))

    def trace(run, count):
        return [(run.step(), tensor_digest(run.read())) for _ in range(count)]

    expected = []
    for adapter in (a, b):
        run = Run(adapter, PROMPT)
        script(run)
        expected.append(trace(run, 30))
    run_a, run_b = Run(a, PROMPT), Run(b, PROMPT)
    script(run_a), script(run_b)
    got = ([], [])
    for _ in range(30):
        for index, run in enumerate((run_a, run_b)):
            got[index].append((run.step(), tensor_digest(run.read())))
    assert list(got) == expected
    assert got[0] != got[1]


def test_plain_forward_with_staged_edits_is_unhooked(adapter, mode):
    ids = torch.tensor([PROMPT])
    with torch.inference_mode():
        reference = adapter.model(ids).logits.clone()
    run = Run_(adapter, mode)
    run.until("L1.a")
    run.set("L1.m", 0.0, "scale")
    run.set("L1.p", torch.full(run.port_shape("p"), 0.125).to(adapter.dtype))
    cache, order = run._cache, list(run._cache.order)
    with torch.inference_mode():
        again = adapter.model(ids).logits
    assert torch.equal(again.view(torch.uint8), reference.view(torch.uint8))
    assert run._cache is cache and run._cache.order == order
    assert run.read("L1.a").shape[-1] == adapter.model.config.hidden_size
    for module in adapter.model.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_layer_beyond_the_model_is_refused_at_set(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L0.a")
    with pytest.raises(ValueError, match="no stop L9.m"):
        run.set("L9.m", 1.0)
    assert run.edits == []


# --- E13 -------------------------------------------------------------------------------------


def test_committed_edits_original_and_row_captures(adapter, mode):
    reference = Run_(adapter, mode)
    reference.until("L0.h")
    ref_h = reference.read()
    h_only = Run_(adapter, mode)
    h_only.until("L0.h")
    h_only.set("L0.h", 0.5, "scale")
    h_only.until("L0.o")
    ref_o_after_h = h_only.read()
    reference.until("L0.o")
    ref_o = reference.read()
    assert not torch.equal(ref_o, ref_o_after_h)

    run = Run_(adapter, mode)
    run.until("L0.h")
    run.set("L0.h", 0.5, "scale")
    run.set("L0.o", 0.25, "add", selection="all of o")
    run.until("L0.u")
    pending_o, pending_h = run.original("L0.o"), run.original("L0.h")
    assert torch.equal(pending_h, ref_h) and torch.equal(pending_o, ref_o_after_h)
    assert run.original("L0.q") is None
    run.finish()
    run.step("token", 2)
    assert run.where == f"t{NEXT + 1}.start"
    committed = run.committed_edits()
    assert [(e["row"], e["layer"], e["port"], e["op"]) for e in committed] == [
        (ROW, 0, "h", "scale"),
        (ROW, 0, "o", "add"),
    ]
    assert committed[1]["selection"] == "all of o" and committed[0]["selection"] is None
    for entry in committed:
        assert {"realised_exact", "cast_exact", "changed_fraction", "value"} <= set(entry)

    state = (
        run.where, len(run.session.receipts), len(run._trail), run.checkpoint()["cut"],
        run.session.capture(retain=False).fingerprint,
    )  # fmt: skip
    address = f"t{ROW}.L0"
    assert run.original(f"{address}.h") is None and run.original(f"{address}.o") is None
    assert torch.equal(run.original(f"{address}.h", recompute=True), ref_h)
    assert torch.equal(run.original(f"{address}.o", recompute=True), ref_o_after_h)
    assert run.original(f"{address}.q", recompute=True) is None
    assert run.original(f"{address}.h_out", recompute=True) is None
    assert state == (
        run.where, len(run.session.receipts), len(run._trail), run.checkpoint()["cut"],
        run.session.capture(retain=False).fingerprint,
    )  # fmt: skip

    rows = run.row_captures(ROW)
    assert rows is not None and {"emb", "L0.h_out"} <= set(rows)
    assert torch.equal(rows["L0.h_out"], run.read(f"t{ROW}.L0.h_out"))
    assert run.row_captures(10_000) is None
    dropped = Run_(adapter, mode, retain_rows=0)
    dropped.until("L0.h")
    dropped.set("L0.h", 0.5, "scale")
    dropped.finish()
    dropped.step("token", 3)
    kept = dropped.row_captures(ROW)
    assert kept is None or set(kept) == {"start"}


# --- round 3: read_info authority (F1), row captures (F5) --------------------------------------


def _one_ulp_native(monkeypatch, adapter):
    modeling = program_for_run(adapter).modeling_module()
    native = modeling.eager_attention_forward

    def wrapped(module, q, k, v, mask, *a, **kw):
        out, weights = native(module, q, k, v, mask, *a, **kw)
        weights = weights.clone()
        flat = weights.view(-1)
        flat[0] = torch.nextafter(flat[0], torch.tensor(2.0, dtype=flat.dtype))
        return out, weights

    monkeypatch.setattr(modeling, "eager_attention_forward", wrapped)


def test_supplied_means_the_value_is_bitwise_the_cast_of_the_supplied_tensor(adapter, mode):
    """F1a/b: replace is supplied, any other combination is derived-exact with every op."""
    shape_run = Run_(adapter, mode)
    shape_run.until("L1.m")
    shape = shape_run.port_shape("m")
    supplied = (torch.randn(shape, generator=torch.Generator().manual_seed(5)) * 0.5).to(adapter.dtype)
    cases = {
        "replace": ([("replace", supplied)], "supplied", ["replace"]),
        "add": ([("add", 0.25)], "derived-exact", ["add"]),
        "scale": ([("scale", 0.5)], "derived-exact", ["scale"]),
        "replace_then_add": ([("replace", supplied), ("add", 0.25)], "derived-exact", ["replace", "add"]),
        "replace_then_scale": ([("replace", supplied), ("scale", 2.0)], "derived-exact", ["replace", "scale"]),
        "add_then_replace": ([("add", 0.25), ("replace", supplied)], "supplied", ["add", "replace"]),
    }
    for name, (edits, authority, ops) in cases.items():
        run = Run_(adapter, mode)
        run.until("L1.a")
        for op, value in edits:
            run.set("L1.m", value, op)
        run.until("L1.m")
        info = run.read_info()
        assert (info["authority"], info["ops"], info["kind"], info["edited"]) == (
            authority, ops, "edit", True), name
        assert info["state"] == "resolved" and info["reason"] is None
        assert torch.equal(run.read(), supplied) is (authority == "supplied"), name


def test_a_masked_edit_is_supplied_only_where_the_replace_wrote(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L1.a")
    mask = torch.zeros(run.port_shape("m"), dtype=torch.bool)
    mask[..., :8] = True
    run.set("L1.m", 0.5, "replace", mask=mask)
    run.set("L1.m", 1.0, "add", mask=torch.zeros_like(mask) | ~mask)
    run.until("L1.m")
    whole = run.read_info()
    assert (whole["authority"], whole["ops"]) == ("derived-exact", ["replace", "add"])
    inside, outside = run.read_info(select=mask), run.read_info(select=~mask)
    assert (inside["authority"], inside["edited"]) == ("supplied", True)  # the add missed these
    assert (outside["authority"], outside["edited"]) == ("derived-exact", True)  # add on native


def test_an_edit_that_wrote_nothing_keeps_the_native_label(adapter, mode):
    """F1g: an all-false mask (or a selection outside the mask) wrote zero elements."""
    run = Run_(adapter, mode)
    run.until("L1.a")
    none = torch.zeros(run.port_shape("m"), dtype=torch.bool)
    run.set("L1.m", 9.0, "replace", mask=none)
    run.until("L1.m")
    info = run.read_info()
    assert (info["edited"], info["by"], info["kind"], info["authority"]) == (False, None, "native", "measured")
    assert info["same_port_edit"] == {"port": "m", "ops": ["replace"]}
    assert info["selective"] is True
    reference = Run_(adapter, mode)
    reference.until("L1.m")
    assert torch.equal(run.read(), reference.read())


def test_o_under_an_r_edit_is_a_derived_realisation_and_composes_with_an_o_edit(adapter, mode):
    """F1b/d: o replaced and r edited is no longer 'supplied'; the value is neither."""
    run = Run_(adapter, mode)
    run.until("L1.o")
    supplied = (torch.randn(run.port_shape("o"), generator=torch.Generator().manual_seed(3)) * 0.5).to(adapter.dtype)
    run.set("L1.o", supplied, "replace")
    run.set("L1.r", 0.37, "add")
    info = run.read_info()
    assert not torch.equal(run.read(), supplied)
    assert (info["authority"], info["by"], info["kind"], info["ops"]) == (
        "derived-exact", "o,r", "edit", ["replace", "add"])
    only_r = Run_(adapter, mode)
    only_r.until("L1.o")
    only_r.set("L1.r", 0.37, "add")
    o = only_r.read_info()
    assert (o["authority"], o["kind"], o["by"], o["edited"]) == ("derived-exact", "realisation", "r", False)


def test_r_is_supplied_only_when_the_realised_value_is_the_supplied_one(adapter, mode):
    """F1d: never ``supplied`` for a tensor that differs from the supplied one."""
    for seed, scale in ((4, 2.0), (5, 0.001)):
        run = Run_(adapter, mode)
        run.until("L1.r")
        target = (torch.randn(run.port_shape("r"), generator=torch.Generator().manual_seed(seed)) * scale).to(adapter.dtype)
        run.set("L1.r", target, "replace")
        info = run.read_info()
        exact = torch.equal(run.read(), target)
        assert info["realisation_exact"] is exact
        assert (info["edited"], info["by"]) == (True, "r")
        if exact:
            assert (info["kind"], info["authority"]) == ("edit", "supplied")
        else:
            assert (info["kind"], info["authority"]) == ("realisation", "measured")
            assert info["realisation_max_abs_error"] == float(
                (run.read().double() - target.double()).abs().max())
    inexact_seen = [True]
    del inexact_seen
    # a request that is exact by construction (the native r itself) is supplied
    run = Run_(adapter, mode)
    run.until("L1.r")
    native = run.read()
    run.set("L1.r", native.clone(), "replace")
    info = run.read_info()
    assert torch.equal(run.read(), native) and info["realisation_exact"] is True
    assert (info["kind"], info["authority"]) == ("edit", "supplied")


def test_a_committed_h_out_edit_is_visible_on_the_boundary_read(adapter, mode):
    """F1c: ``L1.h_out`` and ``L2.h`` are one value and both say it was edited."""
    run = Run_(adapter, mode)
    run.until("L1.m")
    value = torch.full(run.port_shape("h_out"), 0.125, dtype=adapter.dtype)
    for op, authority in (("replace", "supplied"), ("add", "derived-exact")):
        run = Run_(adapter, mode)
        run.until("L1.m")
        run.set("L1.h_out", value, op)
        run.finish()
        boundary = run.read_info("L1.h_out")
        assert (boundary["edited"], boundary["by"], boundary["kind"], boundary["authority"]) == (
            True, "h_out", "edit", authority)
        assert boundary["upstream_edits"] == [] and boundary["ops"] == [op]
        assert (torch.equal(run.read("L1.h_out"), value)) is (op == "replace")
        run.step()
        assert run.where.endswith("L2.h")
        entry = run.read_info()
        assert (entry["edited"], entry["by"], entry["authority"]) == (True, "h_out", authority)
        assert entry["upstream_edits"] == [] and torch.equal(run.read(), run.read("L1.h_out"))
        run.step()  # a later port of the next layer only has it upstream
        later = run.read_info()
        assert later["edited"] is False and [e["port"] for e in later["upstream_edits"]] == ["h_out"]
    masked = Run_(adapter, mode)
    masked.until("L1.m")
    half = torch.zeros(masked.port_shape("h_out"), dtype=torch.bool)
    half[..., :32] = True
    masked.set("L1.h_out", 0.0, "replace", mask=half)
    masked.finish()
    info = masked.read_info("L1.h_out")
    assert (info["authority"], info["selective"]) == ("derived-exact", True)
    assert masked.read_info("L1.h_out", select=half)["authority"] == "supplied"


def test_replica_values_are_refused_only_when_the_check_failed(adapter, monkeypatch):
    run = Run_(adapter, "none")
    run.until("L1.s")
    info = run.read_info()
    assert (info["state"], info["authority"], info["reason"]) == ("resolved", "derived-exact", None)
    assert info["kind"] == "replica" and info["replica_checked"] is False and info["replica_equal"] is None
    run.until("L1.p")
    assert run.read_info("L1.s")["replica_checked"] is True and run.read_info("L1.s")["replica_equal"] is True
    _one_ulp_native(monkeypatch, adapter)
    run = Run_(adapter, "layer")
    run.until("L1.s")
    s = run.read_info()
    assert (s["state"], s["authority"]) == ("refused", None)
    assert s["reason"].startswith("replica check failed: max |replica \u2212 native| = ")
    run.set("L1.s", 1.0, "add")
    run.until("L1.A")
    for port in ("A", "p"):
        info = run.read_info(f"L1.{port}")
        assert info["kind"] == "replica" and (info["state"], info["authority"]) == ("refused", None), port
        assert "replica check failed" in info["reason"]
    # native-only values of the same layer are untouched by the replica check
    assert run.read_info("L1.q")["state"] == "resolved"
    # commit is refused (nothing committed, edits stay staged); an edit-free layer commits
    where, receipts = run.where, len(run.session.receipts)
    with pytest.raises(ValueError, match="layer 1: replica check failed.*s edit"):
        run.commit()
    assert run.where == where and len(run.session.receipts) == receipts
    assert [e["port"] for e in run.edits] == ["s"]
    run.set("L1.s", 0.0, "scale", mask=torch.zeros(run.port_shape("s"), dtype=torch.bool))
    run.abort()
    assert run.edits == []
    run.finish()  # no s/p edit: the one-ulp replica is not involved


def test_original_of_o_and_A_under_the_edit_that_rewrote_them(adapter, mode):
    """F1h: the port's value changed, so there is an original (the native value)."""
    plain = Run_(adapter, mode)
    plain.until("L1.A")
    native_a = plain.read("L1.A")
    plain.until("L1.o")
    native_o = plain.read("L1.o")
    run = Run_(adapter, mode)
    run.until("L1.a")
    run.set("L1.r", 0.5, "add")
    run.until("L1.o")
    assert torch.equal(run.original("L1.o"), native_o)
    assert not torch.equal(run.read("L1.o"), native_o)
    assert run.original("L1.q") is None
    run.finish()
    assert run.original("L1.o") is None
    assert torch.equal(run.original("L1.o", recompute=True), native_o)
    run = Run_(adapter, mode)
    run.until("L1.a")
    run.set("L1.s", 2.0, "scale")  # (a uniform add would leave the softmax unchanged)
    run.until("L1.A")
    assert torch.equal(run.original("L1.A"), native_a)
    assert not torch.equal(run.read("L1.A"), native_a)
    run.finish()
    assert run.original("L1.A") is None
    assert torch.equal(run.original("L1.A", recompute=True), native_a)


def test_row_captures_hold_only_ports_at_or_before_the_cursor_in_both_modes(adapter, mode):
    run = Run_(adapter, mode)
    run.until("L1.a")
    rows = run.row_captures(ROW)
    held = {k.split(".", 1)[1] for k in rows if k.startswith("L1.")}
    assert held == {"h", "a"}
    run.until("L1.v")
    held = {k.split(".", 1)[1] for k in run.row_captures(ROW) if k.startswith("L1.")}
    names = list(run.program.names)
    assert held == set(names[: names.index("v") + 1])
    run.finish()
    assert not any(k.startswith("L2.") for k in run.row_captures(ROW))


# --- round 3, part 2: interrupted moves (G1), committed masks (G2), two edits (G4) -------------

from saturn_pub.core import Session  # noqa: E402


class _Boom(BaseException):
    """A stand-in for KeyboardInterrupt (also a BaseException; pytest keeps it local)."""


def _inject(monkeypatch, method, *, after=True):
    """Make the first ``Session.<method>`` call raise ``_Boom`` once, after it ran (the native
    session has moved) or before."""
    original = getattr(Session, method)
    fired = {"n": 0}

    def wrapped(self, *args, **kwargs):
        if fired["n"]:
            return original(self, *args, **kwargs)
        fired["n"] = 1
        if not after:
            raise _Boom
        original(self, *args, **kwargs)
        raise _Boom

    monkeypatch.setattr(Session, method, wrapped)
    return fired


def _in_sync(run):
    return run.session.capture(retain=False).fingerprint == run._trail[-1].fingerprint


def _twin_at(adapter, mode, where, edits=()):
    twin = Run_(adapter, mode)
    twin.until(where)
    return twin


MOTIONS = {
    "step-token": lambda r: r.step("token"),
    "step-layer": lambda r: r.step("layer", 2),
    "finish": lambda r: r.finish(),
    "commit": lambda r: r.commit(),
    "until": lambda r: r.until(f"t{NEXT}.L1.o"),
}


@pytest.mark.parametrize("motion", sorted(MOTIONS))
@pytest.mark.parametrize("staged", [False, True], ids=["plain", "staged-edit"])
def test_an_interrupt_after_the_native_transition_leaves_cursor_and_session_in_step(
    adapter, mode, monkeypatch, motion, staged
):
    run = Run_(adapter, mode)
    run.until("L0.q")
    if staged:
        run.set("L0.v", 0.5, "add")
    where, edits = run.where, [e["port"] for e in run.edits]
    receipts = len(run.session.receipts)
    _inject(monkeypatch, "continue_", after=True)
    with pytest.raises(_Boom):
        MOTIONS[motion](run)
    monkeypatch.undo()
    assert _in_sync(run), "session is a transition ahead of the trail"
    assert [e["port"] for e in run.edits] == edits  # the staged edit is still staged
    assert run.committed_edits() == [] and run.where == where
    assert len(run.session.receipts) >= receipts  # the abandoned transition stays in the log
    # the run is usable and equals an uninterrupted twin: finish the layer and compare
    run.finish()
    twin = Run_(adapter, mode)
    twin.until("L0.q")
    if staged:
        twin.set("L0.v", 0.5, "add")
    twin.finish()
    assert run.where == twin.where
    assert torch.equal(run.read(), twin.read()) and run.committed_edits() == twin.committed_edits()
    assert run.read_info()["upstream_edits"] == twin.read_info()["upstream_edits"]
    run.until("L1.m")
    twin.until("L1.m")
    assert torch.equal(run.read(), twin.read())
    assert run.read_info()["upstream_edits"] == twin.read_info()["upstream_edits"]
    assert bool(staged) is bool(run.read_info()["upstream_edits"])  # still "under edit"


@pytest.mark.parametrize("how", ["reverse", "restore-checkpoint", "restore-root"])
def test_an_interrupt_after_a_session_restore_leaves_cursor_and_session_in_step(
    adapter, mode, monkeypatch, how
):
    run = Run_(adapter, mode)
    run.until("L0.q")
    run.set("L0.v", 0.5, "add")
    run.until("L1.q")  # L0 committed with the edit; the cursor is in L1
    checkpoint = run.checkpoint("here")
    run.until("L1.m")
    where = run.where
    trail_length = len(run._trail)
    committed = run.committed_edits()
    _inject(monkeypatch, "restore", after=True)
    action = {
        "reverse": lambda: run.reverse_step("layer", 2),
        "restore-checkpoint": lambda: run.restore({**checkpoint, "cut": run._trail[0].fingerprint,
                                                    "transitions": 0, "port": None, "edits": [],
                                                    "path_edits": []}),
        "restore-root": lambda: run._restore_mark(0),
    }[how]
    with pytest.raises(_Boom):
        action()
    monkeypatch.undo()
    if how == "reverse":  # the first of the two steps completed in place; the second was undone
        probe = Run_(adapter, mode)
        probe.until("L0.q")
        probe.set("L0.v", 0.5, "add")
        probe.until("L1.m")
        where = probe.reverse_step("layer", 1)
    assert _in_sync(run) and run.where == where and len(run._trail) == trail_length
    assert run.committed_edits() == committed
    assert [e["port"] for e in run.edits] == []
    twin = Run_(adapter, mode)
    twin.until("L0.q")
    twin.set("L0.v", 0.5, "add")
    twin.until("L1.m")
    run.until("L1.m")
    assert torch.equal(run.read(), twin.read())
    assert run.read_info()["upstream_edits"] == twin.read_info()["upstream_edits"]
    run.reverse_step("layer", 1)  # and the interrupted move works when repeated
    assert _in_sync(run)
    run.until("L1.m")
    assert torch.equal(run.read(), twin.read())


def test_a_refused_abort_changes_nothing(adapter, mode, monkeypatch):
    run = Run_(adapter, mode)
    run.until("L0.q")
    run.set("L0.v", 0.5, "add")
    original = Run._ensure_cache
    calls = {"n": 0}

    def refusing(self, port=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("refused")
        return original(self, port)

    monkeypatch.setattr(Run, "_ensure_cache", refusing)
    with pytest.raises(ValueError, match="refused"):
        run.abort()
    monkeypatch.undo()
    assert [e["port"] for e in run.edits] == ["v"]  # the edit was not dropped
    assert run.read_info()["upstream_edits"] == []
    run.abort()
    assert run.edits == []


def test_masked_edit_labels_are_the_same_staged_and_committed(adapter, mode):
    """G2: a committed masked edit still labels each selection by what the mask covered."""
    run = Run_(adapter, mode)
    run.until("L0.a")
    mask = torch.zeros(run.port_shape("m"), dtype=torch.bool)
    mask[..., :8] = True
    run.set("L0.m", 0.5, "replace", mask=mask)
    run.until("L0.m")
    staged = {
        name: run.read_info(select=sel) for name, sel in (("in", mask), ("out", ~mask), ("all", None))
    }
    mask_staged = run.edit_mask("L0.m").clone()
    run.finish()
    run.until("L1.a")
    assert torch.equal(run.edit_mask("L0.m"), mask_staged)  # served from the occurrence's mark
    for name, sel in (("in", mask), ("out", ~mask), ("all", None)):
        info = run.read_info("L0.m", select=sel)
        for key in ("edited", "authority", "kind", "by", "ops", "same_port_edit", "selective"):
            assert info[key] == staged[name][key], (name, key)
    assert run.edit_mask("L1.m") is None and run.edit_mask("L0.g") is None


def test_two_edits_on_one_port_label_each_selection_by_its_own_operations(adapter, mode):
    """G4: supplied only where every covering edit's last write is a replace."""
    run = Run_(adapter, mode)
    run.until("L0.a")
    shape = run.port_shape("q")
    scale_mask = torch.zeros(shape, dtype=torch.bool)
    scale_mask[:, 0] = True if len(shape) == 3 else False
    flat = run.port_shape("m")
    first = torch.zeros(flat, dtype=torch.bool)
    first[..., :4] = True
    second = torch.zeros(flat, dtype=torch.bool)
    second[..., 8:12] = True
    run.set("L0.m", 0.5, "scale", mask=first)
    run.set("L0.m", 1.0, "replace", mask=second)
    run.until("L0.m")
    for sel, want in ((first, ("derived-exact", ["scale", "replace"])), (second, ("supplied", ["scale", "replace"]))):
        info = run.read_info(select=sel)
        assert (info["authority"], info["ops"]) == want
        assert info["edited"] is True
    rest = ~(first | second)
    info = run.read_info(select=rest)
    assert (info["edited"], info["authority"], info["same_port_edit"]) == (
        False, "measured", {"port": "m", "ops": ["scale", "replace"]})
    both = run.read_info(select=first | second)
    assert both["authority"] == "derived-exact"  # one region scaled, one replaced
    run.finish()
    again = run.read_info("L0.m", select=second)
    assert again["authority"] == "supplied" and run.read_info("L0.m", select=first)["authority"] == "derived-exact"


# --- restore(defer_interrupt=True): an interrupt during the restore cannot roll it back -----------


def _interrupt_at(monkeypatch, method, nth, *, after):
    """Raise ``KeyboardInterrupt`` once, at the ``nth`` call of ``Session.<method>`` (before the
    call ran, or after it ran)."""
    original = getattr(Session, method)
    state = {"calls": 0, "fired": False}

    def wrapped(self, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] != nth:
            return original(self, *args, **kwargs)
        state["fired"] = True
        if not after:
            raise KeyboardInterrupt
        original(self, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(Session, method, wrapped)
    return state


def _excursion(adapter, mode, replay):
    """A run away from a checkpoint (``t{ROW}.L1.v``: L0 committed with an edit, an edit staged
    at L1.v) and that checkpoint. Rewind: the run took the checkpoint, then moved on to the next
    row. Replay: a second run over the same tokens (holding the edit tensors) that never reached
    the checkpoint's cut, so restoring it replays the committed path."""
    run = Run_(adapter, mode)
    run.until("L0.v")
    run.set("L0.v", 0.5, "add")
    run.until("L1.v")
    run.set("L1.v", 2.0, "scale")
    checkpoint = run.checkpoint("here")
    if replay:
        other = Run_(adapter, mode)
        other.hold(run.edit_tensors())
        other.until("L1.m")
        return other, checkpoint
    run.step("token")
    return run, checkpoint


FIRED: set = set()
POINTS = [("restore", 1, False), ("restore", 1, True), ("capture", 1, False),
          ("capture", 1, True), ("probe", 1, False), ("probe", 1, True), ("continue_", 1, False),
          ("continue_", 2, True), ("capture", 2, False)]  # capture 2: the replay's second mark


@pytest.mark.parametrize("point", POINTS, ids=lambda p: f"{p[0]}-{p[1]}-{'after' if p[2] else 'before'}")
@pytest.mark.parametrize("replay", [False, True], ids=["rewind", "replay"])
def test_a_deferred_interrupt_completes_the_restore_then_raises(adapter, mode, monkeypatch, point,
                                                                replay):
    method, nth, after = point
    run, checkpoint = _excursion(adapter, mode, replay)
    twin, _ = _excursion(adapter, mode, replay)
    twin.restore(checkpoint)
    state = _interrupt_at(monkeypatch, method, nth, after=after)
    try:
        run.restore(checkpoint, defer_interrupt=True)
        raised = False
    except KeyboardInterrupt:
        raised = True
    monkeypatch.undo()
    assert raised is state["fired"]  # raised exactly when it fired, and only after completing
    FIRED.add((method, nth, after, replay, state["fired"]))
    assert run.where == checkpoint["where"] == twin.where
    assert [e["port"] for e in run.edits] == [e["port"] for e in twin.edits] == ["v"]
    assert len(run.committed_edits()) == 1
    assert run.committed_edits() == twin.committed_edits()
    assert _in_sync(run)
    assert torch.equal(run.read(), twin.read())
    assert run.read_info() == twin.read_info()


def test_without_defer_the_same_interrupt_leaves_the_excursion(adapter, mode, monkeypatch):
    """The control: the default restore is all-or-nothing, so an interrupt inside it rolls back to
    where the excursion stood (what ``whatif`` must never leave behind)."""
    run, checkpoint = _excursion(adapter, mode, False)
    where = run.where
    _interrupt_at(monkeypatch, "restore", 1, after=True)
    with pytest.raises(KeyboardInterrupt):
        run.restore(checkpoint)
    monkeypatch.undo()
    assert run.where == where != checkpoint["where"]
    assert _in_sync(run)


def test_the_deferred_interrupt_points_fire():
    """Every injection point above fired in at least one case (the sweep is not vacuous)."""
    if not FIRED:
        pytest.skip("run with the deferred-interrupt sweep")
    fired = {(m, n, a, r) for m, n, a, r, f in FIRED if f}
    assert {(m, n, a) for m, n, a, _ in fired} == set(POINTS), sorted(FIRED)


# --- restore sets the edits to the checkpoint's even when the cut is bit-identical --------------


@pytest.mark.parametrize("edit", [("m", 0.0, "add"), ("m", 1.0, "scale"), ("m", 0.5, "add")],
                         ids=["add-0", "scale-1", "add-0.5"])
def test_restore_drops_an_edit_whose_result_is_bit_identical(adapter, edit):
    """``add 0`` / ``scale 1`` commit an edit whose native cut equals the checkpoint's: the
    restore must still leave the checkpoint's (empty) edit list, staged and committed."""
    port, value, op = edit
    run = Run(adapter, PROMPT)
    run.until(f"t{NEXT}.logits")
    checkpoint = run.checkpoint()
    while run.parse(run.where) >= run.parse(f"t{NEXT}.L1.h"):
        run.reverse_step("layer")
    run.until(f"t{NEXT}.L1.h")
    run.set(f"t{NEXT}.L1.{port}", value, op)
    run.until(f"t{NEXT}.logits")
    assert len(run.committed_edits()) == 1
    run.restore(checkpoint)
    assert run.committed_edits() == [] and run.edits == []
    assert run.where == checkpoint["where"]
    assert run.checkpoint()["path_edits"] == checkpoint["path_edits"] == []
