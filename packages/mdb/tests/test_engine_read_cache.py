"""Reads of committed layers (``Run._read``): the full-layer read cache for unedited layers and
the reused hydrated Session for edited ones give the same bytes and labels as per-port
stop-probes on a fresh ``Session.from_cut`` (``Run.cache_reads = False``)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from saturn_pub.core import Session  # noqa: E402
from test_engine_ports import PROMPT, build_model  # noqa: E402

from mdb.engine import Run  # noqa: E402

CASES = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32")]
ROW = len(PROMPT) - 1  # 6: the first stepped row


@pytest.fixture(scope="module", params=CASES, ids=lambda case: "-".join(case))
def adapter(request):
    return DecoderAdapter(build_model(*request.param), granularity="operation")


def _scenario(adapter, retain_rows=None):
    """Row 6: layer 0 carries a port edit, layer 1 a cell edit; row 7 is unedited."""
    run = Run(adapter, PROMPT, retain_rows=retain_rows)
    run.until(f"t{ROW}.L0.m")
    run.set("m", 0.5, "scale")
    run.until(f"t{ROW}.L1.h")
    run.set_cell(0, "v", 0.25, "add", rows=[2])
    run.until(f"t{ROW + 2}.L0.h")
    return run


def _addresses(run):
    return [f"t{row}.L{layer}.{port}" for row in (ROW, ROW + 1)
            for layer in range(run.adapter.layers) for port in run.program.names]


def _bytes(value):
    return value.contiguous().view(torch.uint8).flatten().tolist(), value.dtype, tuple(value.shape)


def _reads(run, cached):
    run.cache_reads = cached
    run.close()
    return {address: (_bytes(run.read(address)), run.read_info(address))
            for address in _addresses(run)}


@pytest.mark.parametrize("retain_rows", [None, 0])
def test_committed_reads_equal_with_cache_on_and_bypassed(adapter, retain_rows):
    run = _scenario(adapter, retain_rows)
    on, off = _reads(run, True), _reads(run, False)
    assert on.keys() == off.keys()
    for address in on:
        assert on[address][0] == off[address][0], address
        assert on[address][1] == off[address][1], address
    # the scenario reaches both paths: an edited layer, a cell layer and unedited layers
    assert on[f"t{ROW}.L0.m"][1]["edited"]
    assert any(e.get("cell") for e in on[f"t{ROW}.L1.A"][1]["upstream_edits"])


def _counting(monkeypatch):
    calls = []
    original = Session.from_cut

    def counted(cls, adapter, cut):
        calls.append(cut.fingerprint)
        return original(adapter, cut)

    monkeypatch.setattr(Session, "from_cut", classmethod(counted))
    return calls


@pytest.mark.parametrize("row,layer", [(ROW + 1, 1), (ROW, 0), (ROW, 1)],
                         ids=["unedited", "port-edit", "cell-edit"])
def test_reading_every_port_of_a_committed_layer_hydrates_once(adapter, monkeypatch, row, layer):
    run = _scenario(adapter)
    calls = _counting(monkeypatch)
    for port in run.program.names:
        run.read(f"t{row}.L{layer}.{port}")
        run.read_info(f"t{row}.L{layer}.{port}")
    assert len(calls) == 1
    run.cache_reads = False
    run.close()
    del calls[:]
    for port in run.program.names:
        run.read(f"t{row}.L{layer}.{port}")
    assert len(calls) == len(run.program.names) - (1 if "h_out" in run.program.names else 0)


def test_probe_leaves_the_hydrated_session_unchanged(adapter):
    run = _scenario(adapter)
    for layer in (0, 1):  # a port-edit layer and a cell-edit layer
        index, _ = run._layer_mark(run.parse(f"t{ROW}.L{layer}.h"))
        for port in run.program.names:
            run.read(f"t{ROW}.L{layer}.{port}")
        fingerprint, session = run._hydrated
        assert fingerprint == run._trail[index].fingerprint
        assert session.capture(retain=False).fingerprint == fingerprint


def test_returned_values_do_not_alias_the_cache(adapter):
    run = _scenario(adapter)
    for address in (f"t{ROW + 1}.L1.q", f"t{ROW}.L0.m"):
        first = run.read(address)
        kept = first.clone()
        first.fill_(7)
        assert torch.equal(run.read(address), kept)


def test_cache_is_dropped_on_restore_reverse_prune_and_close(adapter):
    run = _scenario(adapter)
    saved = run.checkpoint()

    def warm():
        run.read(f"t{ROW + 1}.L1.q")
        run.read(f"t{ROW}.L0.q")
        assert run._reads and run._hydrated is not None

    warm()
    run.close()
    assert not run._reads and run._hydrated is None
    warm()
    run.reverse_step("port")
    assert not run._reads and run._hydrated is None
    warm()
    run.restore(saved)
    assert not run._reads and run._hydrated is None
    pruned = _scenario(adapter, retain_rows=0)
    pruned.read(f"t{ROW + 1}.L1.q")
    assert pruned._reads
    pruned.until(f"t{ROW + 3}.L0.h")  # commits past a row: older cuts are dropped
    assert not pruned._reads and pruned._hydrated is None
