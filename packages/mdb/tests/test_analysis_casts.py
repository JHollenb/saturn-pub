"""Casts and held memory in the analysis layer, on a tiny Qwen2 (BF16).

The rule: a value is cast to float64 at most once, a slice is taken before the cast, and
nothing wider than a result is kept after a call returns. Each test pins one place the rule was
broken (a whole weight or logits vector cast to read one or two rows, a weight cast once per
reader row, row models or masks kept past their use, a recursion as deep as the row chain).
The numbers themselves are pinned by the module tests and the REPL goldens.
"""

import gc
import sys
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import (  # noqa: E402
    circuits,
    compare,
    explain,
    finite,
    forecast,
    propagate,
    reads,
    symbols,
)
from mdb.analysis.forecast import Margin, RowModel  # noqa: E402
from mdb.analysis.propagate import Edit, whatif  # noqa: E402
from mdb.analysis.reads import layer_module  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops.debugger import attach, open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3]
V = 101  # no other dimension of the model is 101: a full-vocabulary tensor is recognisable
MARGIN = Margin(3, 4, 9)
MODULES = (finite, forecast, propagate, explain, reads, circuits, symbols, compare)


@pytest.fixture(scope="module")
def debugger():
    return attach(tiny_model("qwen2", "bf16", vocab_size=V))


@pytest.fixture()
def run(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    return run


@pytest.fixture()
def casts(monkeypatch):
    """Every tensor the analysis modules hand to ``f64``, with the function that cast it."""
    seen: list[tuple[object, str]] = []
    real = finite.f64

    def spy(x):
        seen.append((x, sys._getframe(1).f_code.co_name))
        return real(x)

    for module in MODULES:
        monkeypatch.setattr(module, "f64", spy)
    return seen


def _embedding(run):
    return getattr(run.adapter.backbone, run.adapter.spec.embed).weight


def test_readout_and_embedding_rows_are_indexed_then_cast(run, casts):
    head, embedding = run.adapter.model.lm_head.weight, _embedding(run)
    explain.explain_margin(run, 4, 9)
    explain.explain_port(explain.captures_from_run(run), 0, "h")
    circuits.build(run, MARGIN, top=2, depth=1)
    symbols.table(run, consumer=MARGIN)
    whole = [where for x, where in casts if x is head or x is embedding]
    assert whole == []
    # the theorem helpers read two rows of the head too
    w = torch.randn(V, 8, dtype=torch.bfloat16)
    casts.clear()
    finite.margin_error_bound(w, 3, 7, torch.randn(8, dtype=torch.float64))
    finite.readout_margin(torch.randn(8), w, torch.ones(8), 1e-6, 3, 7)
    finite.embedding_lookup(w, 5)
    assert [where for x, where in casts if x is w] == []


def test_two_logits_are_indexed_then_cast(debugger, casts):
    run = debugger.run(PROMPT)
    run.until("t3.L1.n")
    forecast.seal(run, MARGIN)
    run.until("t3.logits")
    forecast.verify(run)                                    # forecast._native
    compare.measure(run, [], MARGIN)                        # compare.consumer_value
    whatif(run, Edit("t3.L1.m", 0.5, "add"), MARGIN)        # propagate._scalar
    explain.explain_margin(run, 4, 9, first_order=True)     # explain_margin, _first_order
    reads.readout(run, MARGIN)
    # a row model's captured logits are its value (all V of them); every other reader needs two
    full = {where for x, where in casts if x.numel() == V and where != "_compute"}
    assert full == set()


def test_winner_certificate_forms_c_b_a_block_at_a_time_with_the_same_bits(monkeypatch):
    g = torch.Generator().manual_seed(7)
    w = torch.randn(V, 12, generator=g).to(torch.bfloat16)
    h = torch.randn(12, generator=g, dtype=torch.float64)
    gamma = torch.rand(12, generator=g, dtype=torch.float64) + 0.5
    delta = 0.01 * torch.randn(12, generator=g, dtype=torch.float64)
    whole = finite.winner_certificate(h, w, gamma, 1e-6, delta=delta, bias_free=True)
    # the formula before blocking: c_b for every b != a as one (V - 1, d) matrix
    w64, rho = finite.f64(w), finite._rho(h, 1e-6)
    logits = (rho * gamma * h) @ w64.T
    a = int(torch.argmax(logits))
    others = [i for i in range(V) if i != a]
    cs = gamma[None, :] * (w64[a][None, :] - w64[others])
    margins = logits[a] - logits[others]
    norms = cs.norm(dim=-1)
    assert torch.equal(whole.total, (margins / (rho * norms)).min())
    assert whole.extras["condition_holds"] == bool(((cs @ delta) > -margins / rho).all())
    monkeypatch.setattr(finite, "_WINNER_CHUNK", 5 * 12)    # five rows per block
    blocked = finite.winner_certificate(h, w, gamma, 1e-6, delta=delta, bias_free=True)
    assert torch.equal(blocked.total, whole.total) and blocked.extras == whole.extras


def test_o_proj_is_cast_once_per_layer_per_verb_and_the_cache_is_sliced_first(debugger, casts):
    run = debugger.run(PROMPT)
    run.until("t5.logits")      # stepped rows t3, t4, t5 (t0..t2 were prefilled)
    layers = run.adapter.layers
    o_proj = [layer_module(run.adapter, layer).self_attn.o_proj.weight for layer in range(layers)]

    def count():
        return sum(1 for x, _ in casts if any(x is w for w in o_proj))

    symbols.table(run, consumer=Margin(5, 4, 9), rows="all")       # 3 reader rows x 2 layers
    assert count() == layers
    casts.clear()
    result = reads.reads(run, "t3", rows="all", causal=False)
    assert count() == layers
    # the result keeps no value mix (each held its own float64 o_proj and (H, T, d) parts)
    assert not any(isinstance(v, reads.HeadMix) for v in result.tensors.values())
    casts.clear()
    reads.head_mix(run, 3, 0)       # row 3 reads cache rows 0..3 of the six cached
    dh = run.adapter.head_dim
    cache = [x for x, _ in casts if x.dim() >= 3 and x.shape[-1] == dh]
    assert cache and all(int(x.shape[-2]) == 4 for x in cache)


def _walk_tensors(obj, seen=None):
    """Every tensor reachable from ``obj`` through dataclass fields, tuples, lists and dicts."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_tensors(value, seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _walk_tensors(value, seen)
    elif hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            yield from _walk_tensors(getattr(obj, name), seen)


@pytest.mark.parametrize("consumer", [MARGIN, "t3.L1.h_out"], ids=str)
def test_a_circuit_builds_nothing_for_a_folded_child_and_keeps_no_mask(run, consumer,
                                                                       monkeypatch):
    made = []
    init = circuits.Node.__init__

    def record(self, *args, **kwargs):
        init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(circuits.Node, "__init__", record)
    downs = [layer_module(run.adapter, layer).mlp.down_proj.weight
             for layer in range(run.adapter.layers)]
    down_casts, abs_copies = [], []
    real_f64, real_abs = circuits.f64, torch.Tensor.abs

    def f64(x):
        if any(x is w for w in downs):
            down_casts.append(x)
        return real_f64(x)

    def absolute(self):
        if self.dim() == 2 and tuple(self.shape) == tuple(downs[0].shape):
            abs_copies.append(self.shape)
        return real_abs(self)

    monkeypatch.setattr(circuits, "f64", f64)
    monkeypatch.setattr(torch.Tensor, "abs", absolute)
    circuit = circuits.build(run, consumer, top=1, depth=3)
    monkeypatch.undo()
    tree = {id(n) for n in circuits._walk(circuit.root)}
    folded = [n for n in made if id(n) not in tree and n.kind in ("neuron", "head", "source")]
    assert folded == []                     # no Node for a neuron, head or source not kept
    kept_mlp = sum(1 for n in circuits._walk(circuit.root) if n.kind == "mlp" and n.children)
    assert len(down_casts) == kept_mlp <= 1  # down_proj cast only where its neurons are listed
    assert abs_copies == []                  # |W_down| is taken in place
    members = [m for g in circuits._groups(circuit.root) for m in g.members]
    assert members and all(isinstance(m, circuits.Member) and m.value is None for m in members)
    assert not [t for t in _walk_tensors(circuit) if t.dtype == torch.bool]
    # the complement's prediction is kept instead: one number (or one vector) per circuit
    assert circuit.complement is not None
    widest = max(t.numel() for t in _walk_tensors(circuit))
    assert widest <= run.adapter.model.config.hidden_size


def test_whatif_keeps_one_chain_of_row_models_alive_at_a_time(debugger, monkeypatch):
    live: weakref.WeakSet = weakref.WeakSet()
    peak, at_measure = [0], []
    init = RowModel.__init__

    def record(self, *args, **kwargs):
        init(self, *args, **kwargs)
        live.add(self)
        peak[0] = max(peak[0], len(live))

    measure = propagate._measure

    def measured(*args, **kwargs):
        gc.collect()
        at_measure.append(len(live))
        return measure(*args, **kwargs)

    monkeypatch.setattr(RowModel, "__init__", record)
    monkeypatch.setattr(propagate, "_measure", measured)
    run = debugger.run(PROMPT)
    run.until("t3.L0.a")
    whatif(run, Edit("t3.L0.o", 0.5, "add"), Margin(6, 4, 9))    # rows t3..t6
    assert at_measure == [0]       # the prediction's chains are gone before anything native
    assert peak[0] == 4            # base, edited, emulated base and edited, first-order, realised: in turn


def test_a_whatif_chain_far_past_the_cursor_has_no_depth_limit():
    run = open_tiny("qwen2", "bf16").run(PROMPT * 10)
    row = run.parse(run.where)[0]
    run.until(f"t{row}.L0.a")
    # 31 rows of 2 layers: a recursive evaluation needs about 1,200 frames here
    result = whatif(run, Edit(f"t{row}.L0.o", 0.5, "add"), Margin(row + 30, 4, 9))
    assert result.report["prediction"]["state"] == "resolved"


def test_seal_copies_no_operand_twice(debugger, monkeypatch):
    clones = []
    real = torch.Tensor.clone

    def spy(self, *args, **kwargs):
        if sys._getframe(1).f_code.co_filename.endswith("forecast.py"):
            clones.append(self.shape)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", spy)
    run = debugger.run(PROMPT)
    run.until("t3.L1.a")
    forecast.seal(run, "t3.L1.m")
    forecast.seal(run, MARGIN)
    assert clones == []


def test_first_order_prefills_once_and_the_gradient_leaves_the_session_alone(run, monkeypatch):
    adapter = run.adapter
    sessions = []
    real = adapter.session

    def session(tokens):
        sessions.append(len(tokens))
        return real(tokens)

    monkeypatch.setattr(adapter, "session", session)
    plain = len(sessions)
    explain.explain_margin(run, 4, 9)
    without = len(sessions) - plain
    out = explain.explain_margin(run, 4, 9, first_order=True)
    assert out["first_order"]["state"] == "resolved"
    assert len(sessions) - plain - 2 * without == 1      # one prefill for every layer
    monkeypatch.undo()
    probe = adapter.session(PROMPT)
    probe.continue_()
    before = probe.capture(retain=False).fingerprint
    explain.consumer_gradient(probe, 4, 9)
    assert probe.capture(retain=False).fingerprint == before
