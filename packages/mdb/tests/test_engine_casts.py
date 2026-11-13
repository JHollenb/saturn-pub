"""Casts and copies in the engine: each edit operand is computed once and crosses to the host
once, reads copy only what they return, ``mix DOSE`` is one rounding of the float64 result, and
differences labelled ``measured`` are float64 differences."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from saturn_pub.core import StateCut  # noqa: E402
from test_engine_ports import PROMPT, build_model  # noqa: E402

import mdb.engine.run as run_module  # noqa: E402
from mdb.engine import Run  # noqa: E402
from mdb.engine.edits import CellEdit  # noqa: E402
from mdb.engine.ports import PortEdit, _equality, wide  # noqa: E402

ROW = len(PROMPT) - 1


@pytest.fixture(scope="module", params=[("qwen2", "bf16"), ("qwen3", "fp32")],
                ids=lambda case: "-".join(case))
def adapter(request):
    return DecoderAdapter(build_model(*request.param), granularity="operation")


def _bits(value):
    return value.view(torch.int16 if value.element_size() == 2 else torch.int32)


# --- P1: mix DOSE is computed in float64 and rounded once ---------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("operand", ["scalar", "f32 tensor", "f64 tensor", "same dtype"])
def test_mix_is_one_rounding_of_the_float64_result(dtype, operand):
    generator = torch.Generator().manual_seed(3)
    block = (torch.randn((1, 2, 3, 8), generator=generator, dtype=torch.float64) * 3).to(dtype)
    shape = (1, 2, 3, 8)
    value = {
        "scalar": float(torch.randn((), generator=generator)) * 3,
        "f32 tensor": torch.randn(shape, generator=generator) * 3,
        "f64 tensor": torch.randn(shape, generator=generator, dtype=torch.float64) * 3,
        "same dtype": (torch.randn(shape, generator=generator) * 3).to(dtype),
    }[operand]
    v64 = value[0].double() if isinstance(value, torch.Tensor) else torch.tensor(value).double()
    x64 = block.double()
    differ = 0
    for dose in (0.1, 0.25, 0.5, 0.7, 0.9):
        edit = CellEdit(0, "v", (0, 1, 2), value, "mix", dose=dose)
        once = (x64 + dose * (v64 - x64)).to(dtype)
        assert torch.equal(_bits(edit.apply_block(block)), _bits(once))
        fp32 = block.float() + dose * (v64.float() - block.float())  # the old path
        differ += int((_bits(fp32.to(dtype)) != _bits(once)).sum())
    if dtype == torch.float32:
        assert differ > 0  # the float32 interpolation the mix used to compute rounds differently


def test_mix_zero_is_the_identity_and_mix_one_the_copy():
    block = torch.tensor([-0.0, 0.0, 1.5, -2.25, 3.0e-41], dtype=torch.float32).reshape(1, 1, 1, 5)
    value = torch.full((1, 1, 1, 5), 7.0)
    zero = CellEdit(0, "k", (0,), value, "mix", dose=0.0).apply_block(block)
    assert torch.equal(_bits(zero), _bits(block))  # -0.0 stays -0.0, the subnormal stays
    one = CellEdit(0, "k", (0,), value, "mix", dose=1.0).apply_block(block)
    assert torch.equal(_bits(one), _bits(value[0].to(block.dtype).expand(block.shape)))


def test_the_cast_record_describes_the_cast_on_each_operators_path():
    cache = torch.zeros((1, 1, 2, 4), dtype=torch.bfloat16)
    operand = 1.0 + 2.0**-12  # not a bf16 number: its cast rounds
    mixed = CellEdit(0, "k", (1,), operand, "mix", dose=0.5).realise(cache)[1]
    assert "cast_exact" not in mixed and "cast_max_abs_error" not in mixed  # mix casts no operand
    assert "cast_max_abs_error" not in mixed["labels"]
    for op in ("replace", "add", "scale"):
        record = CellEdit(0, "k", (1,), operand, op).realise(cache)[1]
        cast = torch.tensor(operand, dtype=torch.float64).to(torch.bfloat16).double()
        assert record["cast_max_abs_error"] == abs(float(cast) - operand)
        assert record["cast_exact"] is False
    copy = CellEdit(0, "k", (1,), operand, "mix", dose=1.0).realise(cache)[1]
    assert copy["cast_max_abs_error"] == abs(float(cast) - operand)  # mix 1 is the cast copy


# --- item 8: operands once, one host crossing per tensor -----------------------------------


def test_a_cell_edit_computes_its_operand_once(monkeypatch):
    cache = torch.randn((1, 2, 4, 8))
    counts = {"_cast": 0, "_requested": 0}
    for name in counts:
        original = getattr(CellEdit, name)

        def counted(self, *args, _name=name, _original=original):
            counts[_name] += 1
            return _original(self, *args)

        monkeypatch.setattr(CellEdit, name, counted)
    CellEdit(0, "v", (1, 2), torch.randn((1, 2, 2, 8)), "add").realise(cache)
    assert counts == {"_cast": 1, "_requested": 1}


def test_a_port_edit_computes_its_operands_once(monkeypatch):
    calls = []
    original = PortEdit._operands
    monkeypatch.setattr(PortEdit, "_operands",
                        lambda self, dtype: calls.append(dtype) or original(self, dtype))
    x = torch.randn((1, 3, 8))
    mask = torch.zeros((1, 3, 8), dtype=torch.bool)
    mask[..., :4] = True
    out, record = PortEdit("q", torch.randn((1, 3, 8)), "add", mask=mask).realise(x)
    assert len(calls) == 1 and record["mask_count"] == 12
    assert torch.equal(out[~mask], x[~mask])


def _host_transfers(monkeypatch):
    """Count tensors moved from a non-CPU device to the host (``.to(cpu...)`` / ``.cpu()``)."""
    moved = []
    to, cpu = torch.Tensor.to, torch.Tensor.cpu

    def counted_to(self, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if self.device.type != "cpu" and target is not None and not isinstance(target, torch.dtype):
            if torch.device(target).type == "cpu":
                moved.append(tuple(self.shape))
        return to(self, *args, **kwargs)

    def counted_cpu(self, *args, **kwargs):
        if self.device.type != "cpu":
            moved.append(tuple(self.shape))
        return cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", counted_to)
    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    return moved


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a non-CPU device")
@pytest.mark.parametrize("op", ["replace", "add", "mix"])
def test_cell_and_port_edits_cross_to_the_host_once_per_tensor(monkeypatch, op):
    device = torch.device("mps")
    cache = torch.randn((1, 2, 4, 8)).to(device)
    value = torch.randn((1, 2, 2, 8))
    moved = _host_transfers(monkeypatch)
    out, record = CellEdit(0, "v", (1, 2), value, op, dose=0.5 if op == "mix" else None
                           ).realise(cache)
    assert moved == [(2, 2, 8), (2, 2, 8)]  # the picked block and the written one
    moved.clear()
    x = torch.randn((1, 3, 8)).to(device)
    port_op = "replace" if op == "replace" else "add"
    PortEdit("q", torch.randn((1, 3, 8)).to(device), port_op).realise(x)
    # x, the result and the value once each; the cast operand is made on the host
    assert sorted(moved) == [(1, 3, 8)] * 3
    monkeypatch.undo()
    assert out.device.type == "mps" and record["realised_exact"] in (True, False)


# --- item 4: reads copy only what they return ----------------------------------------------


def test_reads_copy_one_slot_one_layer_and_no_discarded_value(adapter, monkeypatch):
    run = Run(adapter, PROMPT)
    run.until(f"t{ROW}.L1.q")
    payloads = []
    monkeypatch.setattr(StateCut, "payload",
                        property(lambda self: payloads.append(1) or self.slot("hidden")))
    boundary = run.read(f"t{ROW}.emb")  # a boundary of a mark that is not the last
    assert payloads == []
    monkeypatch.undo()
    assert torch.equal(boundary, run._trail[1].cut.payload["hidden"])
    clones = []
    original = run_module.clone
    monkeypatch.setattr(run_module, "clone", lambda v: clones.append(v) or original(v))
    run.read(f"t{ROW}.L0.o")  # warm the committed-layer read cache
    clones.clear()
    run.read_info(f"t{ROW}.L0.o")
    run.read_info(f"t{ROW}.L1.q")
    assert clones == []  # read_info builds the context only
    monkeypatch.undo()
    sizes = []
    clone = torch.Tensor.clone
    monkeypatch.setattr(torch.Tensor, "clone",
                        lambda self, *a, **k: sizes.append(self.numel()) or clone(self, *a, **k))
    layer = run.read_cache(0, "k")
    assert sizes == [layer.numel()]


# --- P3: measured differences are float64 differences -------------------------------------


def test_replica_differences_are_float64_differences():
    replica = torch.tensor([1.0 + 2.0**-23, 3.0], dtype=torch.float32)
    native = torch.tensor([-(2.0**-30), 3.0], dtype=torch.float32)
    result = _equality(replica, native)
    exact = (replica.double() - native.double()).abs().max()
    assert result["max_abs_diff"] == float(exact)
    assert float((replica - native).abs().max()) != float(exact)  # float32 would round it
    assert wide(replica).dtype == torch.float64 and wide(replica).device == replica.device
