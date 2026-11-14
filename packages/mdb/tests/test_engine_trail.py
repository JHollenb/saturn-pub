"""M2: trail cuts share what a transition did not change, older cuts cannot be altered, and an
in-place write of a shared tensor is refused; what a run no longer needs is released.

Bytes are measured on a tiny model (3 layers, 2 KV heads, head_dim 16) through
``Run.held_bytes`` (distinct storages, each counted once)."""

import gc
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from saturn_pub.core import InPlaceWrite, Session  # noqa: E402
from saturn_pub.values import describe  # noqa: E402
from test_engine_ports import PROMPT, build_model  # noqa: E402

from mdb.engine import Run  # noqa: E402

CASES = [("qwen2", "bf16"), ("qwen3", "fp32"), ("llama", "fp32"), ("mistral", "bf16")]


@pytest.fixture(scope="module", params=CASES, ids=lambda case: "-".join(case))
def adapter(request):
    return DecoderAdapter(build_model(*request.param), granularity="operation")


def _row_bytes(adapter):
    """K and V bytes of one token row in one layer."""
    size = torch.empty((), dtype=adapter.dtype).element_size()
    return 2 * adapter.num_kv_heads * adapter.head_dim * size


def test_trail_growth_per_row_is_constant_and_small(adapter):
    """Each generated row adds the same bytes to the trail whatever the context length: per
    layer transition one row of that layer's K/V (plus the hidden carrier, the logits and the
    token), never a copy of the whole state."""
    run = Run(adapter, PROMPT)
    run.step("token")
    growth, held = [], run.held_bytes()["trail"]
    for _ in range(8):
        run.step("token")
        now = run.held_bytes()["trail"]
        growth.append(now - held)
        held = now
    assert len(set(growth)) == 1, growth  # constant: independent of the row count
    hidden = adapter.hidden_size * _row_bytes(adapter) // (2 * adapter.num_kv_heads
                                                          * adapter.head_dim)
    logits = adapter.vocab_size * hidden // adapter.hidden_size
    marks = adapter.layers + 5
    bound = adapter.layers * _row_bytes(adapter) + marks * hidden + 2 * logits + 64
    assert growth[0] <= bound, (growth[0], bound)
    whole_state = sum(t.numel() * t.element_size()
                      for t in (*run.session.read("keys"), *run.session.read("values")))
    assert growth[0] < whole_state  # one mark used to hold a whole state copy


def test_consecutive_marks_share_unchanged_layers(adapter):
    run = Run(adapter, PROMPT)
    run.until("L1.h")  # marks: start, emb, L0.h_out
    before, after = run._trail[-2].cut, run._trail[-1].cut  # emb, L0.h_out (layer 0 ran)
    shared = [index for index in range(adapter.layers)
              if before._payload["keys"][index] is after._payload["keys"][index]]
    assert shared == list(range(1, adapter.layers))  # every layer but the one that ran
    pair = before.nbytes() + after.nbytes()
    from saturn_pub.core import held_bytes

    assert held_bytes(before, after) < pair  # the union counts the shared storage once


def _walk(run):
    run.until("L1.p")
    run.set("L1.m", 0.5, "scale")
    run.step("token")
    run.until("L2.g")
    run.set_cell(0, "v", 0.25, "add", rows=[1], kv=[0])
    run.step("token")


def test_nothing_public_alters_an_older_cut(adapter):
    """Edits, cell edits, rewinds, restores, aborts and reads leave every sealed cut as sealed:
    each still verifies against its fingerprint and holds the same bytes."""
    run = Run(adapter, PROMPT)
    _walk(run)
    sealed = [(mark.cut, describe(mark.cut.payload)) for mark in run._trail]
    first = run.checkpoint("first")
    reads = {address: run.read(address) for address in
             (f"t{len(PROMPT) - 1}.L1.m", f"t{len(PROMPT)}.L2.g", f"t{len(PROMPT)}.logits")}
    run.until("L0.v")
    run.set("L0.v", 2.0, "add")
    run.set_cell(1, "k", 0.0, "scale", rows=[0, 2])
    run.step("layer", 2)
    run.reverse_step("token")
    run.until("L2.o")
    run.set("L2.o", 3.0, "scale")
    run.abort()
    run.original(f"t{len(PROMPT) - 1}.L1.m", recompute=True)
    run.restore(first)
    run.step("token", 2)
    for cut, payload in sealed:
        cut.verify()
        assert describe(cut.payload) == payload
    run.restore(first)
    for address, value in reads.items():
        assert torch.equal(run.read(address), value)  # bitwise, after all of it


def test_a_hook_writing_its_input_in_place_cannot_reach_a_cut(adapter):
    """A user hook that writes the layer input in place (the hidden carrier the state and the
    sealed cuts share) writes the private copy native modules get: on a commit and on a probe
    the cuts keep their bytes, as before shares_state (when every transition ran on a copy)."""
    run = Run(adapter, PROMPT)
    run.until("L0.h_out")
    cut = run._trail[-1].cut
    held = describe(cut.payload)

    def write(module, args):
        args[0].mul_(2)

    handle = adapter.model.model.layers[1].register_forward_pre_hook(write)
    try:
        run.until("L1.q")  # a probe of layer 1
        run.step("layer")  # the commit of layer 1
    finally:
        handle.remove()
    cut.verify()
    assert describe(cut.payload) == held
    run.reverse_step("layer")
    assert describe(run.session.capture(retain=False).payload) == held


def test_a_refused_write_never_reaches_a_restore(adapter):
    """Restoring from a cut whose shared tensor was written in place is refused (the guard is
    the version counter: no rehash needed)."""
    session = adapter.session(PROMPT)
    session.continue_(2)
    cut = session.capture(retain=False)
    cut._payload["hidden"].add_(1)
    with pytest.raises(InPlaceWrite, match="hidden"):
        Session.from_cut(adapter, cut)


def test_aborted_and_rewound_edit_tensors_are_released(adapter):
    run = Run(adapter, PROMPT)
    run.until("L1.m")
    shape = run.port_shape("m")
    run.set("L1.m", torch.ones(shape), "add")
    run.set("L1.m", torch.ones(shape) * 2, "add", mask=torch.ones(shape, dtype=torch.bool))
    assert len(run._values) == 2 and len(run._masks) == 1
    run.abort()
    assert run._values == {} and run._masks == {}
    run.set("L1.m", torch.ones(shape) * 3, "add")
    saved = run.checkpoint("kept")  # a checkpoint names the tensor: kept for its restore
    run.abort()
    assert len(run._values) == 1
    run.restore(saved)
    assert run.edits[0]["value"]["kind"] == "tensor"
    run.step("layer")
    run.reverse_step("layer")  # back before the edit: the committed edit is gone ...
    assert len(run._values) == 1  # ... but the checkpoint still names its tensor


def test_port_controllers_keep_no_call_scoped_tensors(adapter):
    run = Run(adapter, PROMPT)
    run.until("L1.p")
    run.set("L1.r", 0.5, "scale")
    run.until("L1.n")
    assert run._cache._live == {}


def test_a_rewound_branch_is_released(adapter):
    """Cuts of marks a rewind discards are freed (nothing else holds them)."""
    run = Run(adapter, PROMPT)
    run.step("token")
    late = weakref.ref(run._trail[-1].cut)
    run.reverse_step("token")
    gc.collect()
    assert late() is None
    assert run.held_bytes()["reads"] == 0


def test_the_session_keeps_a_bounded_number_of_receipts(adapter):
    from mdb.engine.run import _RECEIPTS

    run = Run(adapter, PROMPT)
    run.step("token", _RECEIPTS // (adapter.layers + 5) + 2)
    assert run._trail[-1].transitions > _RECEIPTS
    assert len(run.session.receipts) == _RECEIPTS
    assert run.session.receipts[-1].to_dict()["result"] == run._trail[-1].fingerprint
