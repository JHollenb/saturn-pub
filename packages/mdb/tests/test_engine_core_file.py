"""Core files (gap audit P2), now checkpoint files (#517), on tiny Qwen2 (bf16) and Qwen3 (fp32).

A dump reloads as a steppable run whose logits, ports and edits are bit-identical to the source
run's on the same model and dtype (null check 0, argmax equal); a split at a stepped row keeps the
stepped K/V; the same file reloads on an fp32 copy of a bf16 model (weights upcast exactly) only
as a replica asked for (``replica=True``), labelled, whose null check is recorded; a changed
tensor or another model shape is refused. ``test_engine_checkpoint.py`` covers the rest."""

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.engine import core_file  # noqa: E402
from mdb.engine.models import cast_weights, tiny_model  # noqa: E402
from mdb.ops.debugger import attach, open_tiny  # noqa: E402

P = [1, 5, 9, 3, 7]


@pytest.fixture(scope="module", params=[("qwen2", "bf16"), ("qwen3", "fp32")],
                ids=lambda c: "-".join(c))
def dbg(request):
    return open_tiny(*request.param)


def test_dump_reload_is_bit_identical(dbg, tmp_path):
    src = dbg.run(P, name="src", prefill=2)
    src.until("t4.logits")
    record = src.dump(tmp_path / "core.pt", row=2, capture=("t4.logits", "t4.L1.h_out"))
    assert record["kind"] == "checkpoint-state" and record["body"]["cut"]["row"] == 2
    run = dbg.run_from(tmp_path / "core.pt", name="replay")
    assert run.where == "t2.start" and run.core["replica"] is False
    check = run.core_check
    assert check["ok"] and check["same_dtype_and_identity"]
    assert check["checks"]["t4.logits"]["max_abs_diff"] == 0.0
    run.until("t4.logits")
    assert torch.equal(run.read("t4.logits"), src.read("t4.logits"))
    assert torch.equal(run.read("t3.L0.m"), src.read("t3.L0.m"))       # stepped rows have ports
    a = dbg.run(P, name="edit-src", prefill=2)
    a.until("t4.L0.h")
    a.set("t4.L0.m", 0.25, "add")
    a.until("t4.logits")
    run.rewind("t4.L0.h")
    run.set("t4.L0.m", 0.25, "add")
    run.until("t4.logits")
    assert torch.equal(run.read("t4.logits"), a.read("t4.logits"))


def test_split_at_a_stepped_row(dbg, tmp_path):
    src = dbg.run(P, name="src3", prefill=1)
    src.until("t4.logits")
    src.dump(tmp_path / "c3.pt", row=3)
    loaded = core_file.load(tmp_path / "c3.pt")
    run = loaded.run(dbg.adapter)
    assert run.where == "t3.start" and run.core_check["ok"]
    run.until("t4.logits")
    assert torch.equal(run.read("t4.logits"), src.read("t4.logits"))
    with pytest.raises(ValueError):
        src.dump(tmp_path / "bad.pt", row=0)                  # t0 was prefilled: no start mark


def test_tamper_and_shape_refusals(dbg, tmp_path):
    src = dbg.run(P, name="srct")
    src.until("t4.logits")
    src.dump(tmp_path / "t.pt")
    data = torch.load(tmp_path / "t.pt", weights_only=True)
    key = sorted(data["tensors"])[0]
    data["tensors"][key] = data["tensors"][key].clone()
    data["tensors"][key].view(-1)[0] += 1
    torch.save(data, tmp_path / "t2.pt")
    with pytest.raises(ValueError):
        core_file.load(tmp_path / "t2.pt")
    other = open_tiny("llama", "fp32")
    with pytest.raises(ValueError):
        core_file.load(tmp_path / "t.pt").run(other.adapter)


def test_fp32_replica_of_a_bf16_dump(tmp_path):
    model = tiny_model("qwen2", "bf16", layers=2)
    dbg = attach(model)
    src = dbg.run(P, name="bf16")
    src.until("t4.logits")
    src.dump(tmp_path / "bf.pt")
    wide = cast_weights(copy.deepcopy(model), torch.float32)   # exact upcast, buffers untouched
    dbg32 = attach(wide)
    with pytest.raises(ValueError, match="replica=True"):
        dbg32.run_from(tmp_path / "bf.pt")
    src.rewind("t4.start")
    src.until("t4.logits")
    src.dump(tmp_path / "bf.pt", at="t4.start")              # captures the t4 logits
    run = dbg32.run_from(tmp_path / "bf.pt", replica=True)
    assert run.core["replica"] and run.core["source_dtype"] == "bfloat16"
    check = run.core_check
    assert "argmax_equal" in check["checks"]["t4.logits"]
    assert check["same_dtype_and_identity"] is False
    run.until("t4.logits")
    assert run.read("t4.logits").dtype == torch.float32
