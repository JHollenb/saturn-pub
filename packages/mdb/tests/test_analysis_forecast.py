"""Sealed forecasts on tiny Qwen2/Qwen3 (BF16 and FP32): next-port forecasts land within their
stated rounding bound, sealing precedes the native value, a wrong operand fails ``verify``, an
edit after sealing is reported as ``edited`` and an unreached target stays open."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import forecast  # noqa: E402
from mdb.core.pointers import AUTHORITIES, MAX_RECORD_BYTES  # noqa: E402
from mdb.core.records import verify_record  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3]
CONFIGS = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32")]


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def labelled_numbers(node):
    """Every labelled number ``{value, authority, ...}`` in a JSON tree."""
    if isinstance(node, dict):
        if "authority" in node and "value" in node:
            return [node]
        return [n for value in node.values() for n in labelled_numbers(value)]
    if isinstance(node, list):
        return [n for value in node for n in labelled_numbers(value)]
    return []


def test_next_port_forecast_is_within_its_bound_at_every_port_with_an_identity(debugger):
    run = debugger.run(PROMPT)
    verified, refused = {}, {}
    while run.where != "t5.start":
        where = run.where
        try:
            view = forecast.seal(run)
        except forecast.Refusal as error:
            refused[where] = str(error)
            run.step()
            continue
        run.step()
        [result] = forecast.verify(run, [view["id"]])
        assert result["verdict"] == "within-bound", (where, result)
        assert result["residual_max_abs"]["authority"] in ("measured", "derived-exact")
        assert result["bound_max"]["authority"] == "bound"
        verified[view["target"]] = result
    ports = {target.split(".", 1)[1].split(".")[-1] for target in verified}
    # every port of the program but A and every global stop with an identity was forecast and
    # checked one stop ahead; A is computed by the native call that stops at p, so it is
    # forecast from the stop before p (two stops ahead)
    assert set(run.program.names) - {"A"} <= ports
    assert {"emb", "fn", "logits", "sample"} <= ports
    for layer in range(run.adapter.layers):
        again = debugger.run(PROMPT)
        again.until(f"t3.L{layer}.s")
        view = forecast.seal(again, f"t3.L{layer}.A")
        again.until(f"t3.L{layer}.o")
        [result] = forecast.verify(again, [view["id"]])
        assert result["verdict"] == "within-bound", result
    # the stops with no forecast at the cursor before them, and why: the token list of the next
    # row has no identity, and A is computed by the native call that stops at p
    assert forecast.NO_IDENTITY == {"start": forecast.NO_IDENTITY["start"]}
    assert set(refused) == {"t3.L0.p", "t3.L1.p", "t3.sample", "t4.L0.p", "t4.L1.p",
                            "t4.sample"}
    assert all("computed_past_cursor" in refused[f"t{r}.L{i}.p"] for r in (3, 4) for i in (0, 1))
    assert all("token list" in refused[f"t{r}.sample"] for r in (3, 4))


def test_forecast_record_binds_cut_cursor_method_operands_and_is_sealed(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L0.k")
    view = forecast.seal(run, "t3.L0.p")
    record = forecast.ledger(run)[view["id"]].record
    assert verify_record(record) == record
    assert len(json.dumps(record)) <= MAX_RECORD_BYTES
    body = record["body"]
    assert body["sealed_at"]["where"] == "t3.L0.k"
    assert body["sealed_at"]["cut"] == run.checkpoint()["cut"]
    assert [step["stop"] for step in body["method"]["chain"]][-1] == "t3.L0.p"
    assert all(step["proof"] for step in body["method"]["chain"])
    addresses = {item["address"] for item in body["operands"]}
    assert {"t3.L0.k", "cache.L0.k[row=0:3]"} <= addresses
    assert all(len(item["sha256"]) == 64 for item in body["operands"])
    assert body["predicted"]["sha256"] == record["pointers"][0]["value"]["sha256"]
    for claim in labelled_numbers(body):
        assert claim["authority"] in AUTHORITIES


def test_seal_precedes_native_and_a_computed_target_is_refused(debugger):
    run = debugger.run(PROMPT)
    view = forecast.seal(run, forecast.Margin(3, 4, 9))
    run.until("t3.sample")
    [result] = forecast.verify(run, [view["id"]])
    order = result["order"]
    assert order["sealed_before_target"] and order["descends"] and not order["same_cut"]
    assert (order["sealed"]["row"], order["sealed"]["index"]) < (
        order["target"]["row"], order["target"]["index"])
    assert order["receipts_after_seal"]["count"] > 0
    with pytest.raises(forecast.Refusal, match="already computed"):
        forecast.seal(run, "t3.logits")
    with pytest.raises(forecast.Refusal, match="already computed"):
        forecast.seal(run, forecast.Margin(3, 4, 9))
    run2 = debugger.run(PROMPT)
    run2.until("t3.L1.p")
    with pytest.raises(forecast.Refusal, match="computed_past_cursor"):
        forecast.seal(run2, "t3.L1.A")
    with pytest.raises(forecast.Refusal, match="another row"):
        forecast.seal(run2, "t4.L0.a")


def test_a_wrong_operand_makes_verify_fail(debugger, monkeypatch):
    run = debugger.run(PROMPT)
    read = run.read

    def perturbed(target=None):
        value = read(target)
        return value + 0.25 if target == "t3.L0.h" else value

    monkeypatch.setattr(run, "read", perturbed)
    view = forecast.seal(run)          # predicts t3.L0.a from the perturbed h
    monkeypatch.setattr(run, "read", read)
    run.step()
    [result] = forecast.verify(run, [view["id"]])
    assert result["verdict"] == "outside-bound"
    assert result["within_bound"] is False
    assert result["worst_excess"]["value"] > 0


def test_edit_after_sealing_is_reported_not_failed_and_unreached_stays_open(debugger):
    run = debugger.run(PROMPT)
    edited = forecast.seal(run, "t3.L0.v")
    later = forecast.seal(run, "t3.L1.a")
    run.set("t3.L0.v", 0.5, "add")
    with pytest.raises(forecast.Refusal, match="staged edit"):
        forecast.seal(run, "t3.L0.o")
    run.until("t3.L0.o")
    results = {r["id"]: r for r in forecast.verify(run)}
    assert results[edited["id"]]["verdict"] == "edited"
    assert "residual_max_abs" not in results[edited["id"]]
    assert results[later["id"]]["verdict"] == "open"
    assert results[later["id"]]["status"] == "open"
    run.until("t3.L1.a")
    [again] = forecast.verify(run, [later["id"]])
    assert again["verdict"] == "edited"  # the committed L0.v edit reaches L1.a upstream


def test_rewind_and_replay_to_the_target_still_verifies_with_its_order(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L0.n")
    view = forecast.seal(run, "t3.L1.h")
    run.reverse_step("layer")      # back to the boundary the sealing stop belongs to
    run.until("t3.L1.h")
    [result] = forecast.verify(run, [view["id"]])
    assert result["verdict"] == "within-bound"
    assert result["order"]["descends"] and result["order"]["sealed_before_target"]


def test_a_large_weight_is_multiplied_block_by_block_and_no_float64_weight_is_kept(
        debugger, monkeypatch):
    """The readout must not need a float64 copy of the whole unembedding: with the block size
    forced below the vocabulary the logits and their bound are the ones of the single block."""
    import torch

    run = debugger.run(PROMPT)
    run.until("t3.fn")
    row = int(run.where.split(".")[0][1:])

    def known(label):
        return run.read(f"t{row}.{label}") if label == "fn" else None

    whole = forecast.RowModel(run, row, known)
    value, bound = whole.get("logits"), whole.bound("logits")
    width = int(run.adapter.model.lm_head.weight.shape[1])
    monkeypatch.setattr(forecast, "_CHUNK", 3 * width)  # three vocabulary rows per block
    blocks = forecast.RowModel(run, row, known)
    assert int(value.shape[-1]) > 3
    assert torch.allclose(blocks.get("logits"), value, rtol=1e-13, atol=0.0)
    assert torch.allclose(blocks.bound("logits"), bound, rtol=1e-13, atol=0.0)
    # a row model holds values, never weights
    held = [v for v in vars(blocks).values() if isinstance(v, dict)]
    widest = max((t.numel() for d in held for t in d.values() if isinstance(t, torch.Tensor)),
                 default=0)
    assert widest <= int(value.numel())
    assert not hasattr(blocks, "_weights")
