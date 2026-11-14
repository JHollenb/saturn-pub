"""Sealed forecasts on tiny Qwen2/Qwen3 (BF16 and FP32): next-port forecasts land within their
stated rounding bound, sealing precedes the native value, verify decides on the emulated
prediction (a wrong operand, prediction or identity is not confirmed; drift whose every stage
is the native step is unresolved), an edit after sealing is reported as ``edited`` and an
unreached target stays open."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import forecast  # noqa: E402
from mdb.core.pointers import AUTHORITIES, MAX_RECORD_BYTES  # noqa: E402
from mdb.core.records import verify_record  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3]
CONFIGS = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32"),
           ("llama", "bf16"), ("llama", "fp32"), ("mistral", "bf16"), ("mistral", "fp32")]


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
        assert result["verdict"] == "confirmed", (where, result)
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
        assert result["verdict"] == "confirmed", result
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
    chain = forecast.ledger(run)[view["id"]].chain
    assert [step["stop"] for step in chain][-1] == "t3.L0.p"
    assert all(step["proof"] for step in chain)
    assert body["method"]["summary"]["count"] == len(chain)
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
    assert result["verdict"] == "not confirmed"
    assert "operand is not the one sealed" in result["reason"]
    assert result["within_bound"] is False          # information: against the exact prediction
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
    # 28 random layers: the worst-case bound may be infinite, which never decides the verdict:
    # the measured residual is within the dtype floor
    assert result["verdict"] == "confirmed" and result["state"] == "resolved"
    assert result["floor"]["authority"] == "supplied" and result["floor"]["proof"] == "P20"
    assert result["residual_max_abs"]["value"] < 0.05
    assert result["residual_spacings"]["authority"] == "measured"
    assert result["residual_spacings"]["value"] <= result["floor_spacings"]["value"]
    assert result["order"]["descends"] and result["order"]["sealed_before_target"]


def test_a_wrong_prediction_is_not_confirmed_and_confirm_reads_the_floor():
    import torch

    from mdb.analysis import verdict

    # bf16 near 20: spacing 0.125; floor 1 (supplied) x 0.125
    inside = verdict.confirm(torch.tensor(0.125, dtype=torch.float64), 20.0, torch.bfloat16)
    outside = verdict.confirm(torch.tensor(0.126, dtype=torch.float64), 20.0, torch.bfloat16)
    assert (inside["verdict"], outside["verdict"]) == ("confirmed", "not confirmed")
    assert outside["floor"]["value"] == 0.125 and outside["floor"]["authority"] == "supplied"
    assert outside["floor_spacings"]["value"] == verdict.FLOOR_SPACINGS == 1.0
    assert outside["floor_spacings"]["authority"] == "supplied" and outside["floor_spacings"]["support"]
    assert outside["residual_spacings"]["authority"] == "measured"
    assert abs(outside["residual_spacings"]["value"] - 0.126 / 0.125) < 1e-12
    assert not hasattr(verdict, "COEFFICIENT") and not hasattr(verdict, "COEFFICIENT_SUPPORT")
    # a margin is a difference of two rounded values: the floor takes both spacings
    margin = verdict.confirm(torch.tensor(0.1), 20.0, torch.bfloat16, partner=1.0)
    assert margin["spacing"]["value"] == 0.125 + 2.0 ** -7 and margin["verdict"] == "confirmed"
    assert verdict.confirm(torch.tensor(1), 5, torch.int64)["verdict"] == "not confirmed"
    # a vector is judged per element: one element above the floor decides, and is named
    gap = torch.zeros(8, dtype=torch.float64)
    gap[3] = 0.07                                  # 1.1 spacings of 8.0 (spacing 0.0625)
    vector = verdict.confirm(gap, 8.0, torch.bfloat16)
    assert vector["verdict"] == "not confirmed" and vector["index"] == 3


def _tie_model(layers=2):
    """Tiny bf16 Qwen2 whose layer-0 keys share a large bias on the two lowest-frequency RoPE
    dimensions of each head (RoPE leaves them nearly unrotated) and whose queries share a
    bias there too: every score is about 1.06e4 (bf16 spacing 64) and the score differences
    between keys (below 1) round away, so the native scores tie, as Qwen2.5-1.5B's k bias makes
    them tie at L0 (M42)."""
    import torch

    from mdb.engine.models import tiny_model
    from mdb.ops.debugger import attach

    model = tiny_model("qwen2", "bf16", layers=layers)
    attn = model.model.layers[0].self_attn
    dh = attn.head_dim
    with torch.no_grad():
        attn.k_proj.weight.mul_(0.01)
        for bias, value in ((attn.k_proj.bias, 300.0), (attn.q_proj.bias, 50.0)):
            for start in range(0, bias.shape[0], dh):
                bias[start + dh // 2 - 1] = value
                bias[start + dh - 1] = value
    return attach(model, None)


TIE_PROMPT = [1, 5, 9, 3, 7, 2, 11, 4]


def test_rounded_score_ties_put_the_exact_chain_tens_of_spacings_off_and_emulation_on_native():
    """M42 on a constructed tiny model: rounding the scores to bf16 ties them, so the native
    softmax is not the softmax of the real-number scores. The exact chain is tens of spacings
    off from L0.p on (the old rule, 3 x sqrt(n) spacings, did not confirm it); the emulated
    prediction is the native value, and the sealed origin names the score spacing."""
    run = _tie_model().run(TIE_PROMPT)
    row = run.parse(run.where)[0]
    ids = [forecast.seal(run, f"t{row}.L0.{port}")["id"] for port in ("p", "A", "m")]
    run.until(f"t{row}.L0.m")
    scores = run.read(f"t{row}.L0.s").float().reshape(2, -1)
    assert bool((scores == scores[:, :1]).all())              # every native score tied
    for result in forecast.verify(run, ids):
        assert result["verdict"] == "confirmed", result
        assert result["residual_spacings"]["value"] == 0.0     # emulated: the native bits
        assert result["rounding_effect"]["spacings"]["value"] > 15, result["rounding_effect"]
        assert result["rounding_effect"]["max_abs"]["authority"] == "measured"
        origin = result["rounding_origin"]
        assert origin["stop"] == f"t{row}.L0.p" and origin["score_spacing"]["value"] == 64.0
        assert origin["text"].startswith(f"t{row}.L0.p[h") and "score spacing 64 at |s|" in \
            origin["text"]


@pytest.mark.parametrize("family,dtype,layers", [(f, d, n) for f in ("qwen2", "qwen3")
                                                  for d in ("bf16", "fp32")
                                                  for n in (2, 6, 12, 28)])
def test_correct_margin_and_vector_forecasts_are_confirmed_with_their_residual_in_spacings(
        family, dtype, layers):
    """Forecasts sealed at L0.h (margin, the last layer's m, fn, logits): the emulated
    prediction is within the floor of native. The exact chain is up to about 10 spacings off in
    fp32 at 28 layers, which the emulated one is not (measured: 0)."""
    from mdb.engine.models import tiny_model
    from mdb.ops.debugger import attach

    run = attach(tiny_model(family, dtype, layers=layers), None).run(PROMPT)
    ids = [forecast.seal(run, forecast.Margin(3, 3, 7))["id"],
           forecast.seal(run, f"t3.L{layers - 1}.m")["id"],
           forecast.seal(run, "t3.fn")["id"], forecast.seal(run, "t3.logits")["id"]]
    run.until("t3.logits")
    for result in forecast.verify(run, ids):
        assert result["verdict"] == "confirmed", result
        assert result["residual_spacings"]["value"] <= result["floor_spacings"]["value"]
        assert result["rounded_steps"]["value"] >= layers          # information
        assert result["rounding_effect"]["max_abs"]["authority"] == "measured"


def test_a_forecast_whose_prediction_is_perturbed_is_not_confirmed(debugger):
    run = debugger.run(PROMPT)
    view = forecast.seal(run)
    entry = forecast.ledger(run)[view["id"]]
    entry.emulated = entry.emulated + 1.0   # a deliberately wrong prediction
    run.step()
    [result] = forecast.verify(run, [view["id"]])
    assert result["verdict"] == "not confirmed" and result["state"] == "resolved"
    assert result["residual_max_abs"]["value"] > result["floor"]["value"]
    assert result["residual_spacings"]["value"] > result["floor_spacings"]["value"]
    assert "sealed digest" in result["reason"]


def test_a_wrong_identity_is_not_confirmed_at_the_stage_it_gets_wrong(monkeypatch):
    """Both chains use a wrong SiLU (x 1.01): the prediction is its chain's value, but the native
    L0.sg lies outside the bound of the (wrong) identity on its native operand."""
    from mdb.engine.models import tiny_model
    from mdb.ops.debugger import attach

    def wrong(original):
        def silu(self, g, bg):
            value, bound = original(self, g, bg)
            return value * 1.01, bound
        return silu

    monkeypatch.setattr(forecast.RowModel, "_silu", wrong(forecast.RowModel._silu))
    monkeypatch.setattr(forecast.EmulatedRowModel, "_silu",
                        wrong(forecast.EmulatedRowModel._silu))
    run = attach(tiny_model("qwen2", "bf16"), None).run(PROMPT)
    run.until("t3.L0.n")
    view = forecast.seal(run, "t3.L0.m")
    run.until("t3.L0.m")
    [result] = forecast.verify(run, [view["id"]])
    assert result["verdict"] == "not confirmed", result
    assert result["parting"]["stage"] == "t3.L0.sg" and result["parting"]["stop"] == "t3.L0.sg"


def test_drift_whose_every_stage_is_the_native_step_is_unresolved_and_names_the_tie(monkeypatch):
    """A one-spacing difference of the emulated scores (as a backend kernel differing by one
    rounding would make it) at a native tie: the emulated softmax splits that head differently,
    every stage reproduces native from its native operands, so the verdict is unresolved (not a
    wider floor) with the parting stop and the tie named."""

    def shifted(original):
        def scores(self, q, bq, k, bk, scaling):
            value, bound = original(self, q, bq, k, bk, scaling)
            value = value.clone()
            value[1, 0] += 64.0                    # one spacing at 1.06e4, head 1, key 0
            return value, bound
        return scores

    monkeypatch.setattr(forecast.EmulatedRowModel, "_scores",
                        shifted(forecast.EmulatedRowModel._scores))
    run = _tie_model().run(TIE_PROMPT)
    row = run.parse(run.where)[0]
    view = forecast.seal(run, f"t{row}.L0.A")
    run.until(f"t{row}.L0.A")
    [result] = forecast.verify(run, [view["id"]])
    assert result["verdict"] == "unresolved" and result["state"] == "unresolved", result
    # the scores part by one spacing (within the floor); the softmax is where they part
    assert result["parting"]["stop"] == f"t{row}.L0.p"
    assert result["parting"]["stages_checked"]["count"] == 2       # L0.p and L0.A
    assert any(tie.startswith(f"t{row}.L0.p[h1]: native top scores tie") for tie in
               result["parting"]["ties"]), result["parting"]
    assert "rounding drift" in result["reason"] and "tie" in result["reason"]


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


def _deep_margin(run_debugger, store):
    """Seal ``M[3,7]`` of the last row's logits from ``L0.h`` on a 28-layer tiny model."""
    run = run_debugger.run(PROMPT)
    run.until("L0.h")
    view = forecast.seal(run, forecast.Margin(3, 3, 7), store=store)
    return run, view


def _deep_debugger():
    from mdb.engine.models import tiny_model
    from mdb.ops.debugger import attach

    return attach(tiny_model("qwen2", "bf16", layers=28), None)


def _check_deep(run, view, store):
    entry = forecast.ledger(run)[view["id"]]
    record = entry.record
    assert len(entry.chain) > 64                      # the old record refused this chain
    assert verify_record(record) == record
    assert len(json.dumps(record, sort_keys=True, separators=(",", ":"))) < 64 * 1024
    summary = record["body"]["method"]["summary"]
    assert summary["count"] == len(entry.chain)
    assert sum(item["count"] for item in summary["proof_counts"]) >= len(entry.chain)
    assert summary["first"] == entry.chain[:3] and summary["last"] == entry.chain[-3:]
    assert sum(item["count"] for item in summary["layer_counts"]) + summary["global_steps"]["count"] == len(entry.chain)
    run.until("logits")
    before = (entry.predicted.clone(), None if entry.bound is None else entry.bound.clone())
    (result,) = forecast.verify(run, [view["id"]])
    # 28 random layers: the worst-case bound may be infinite, which never decides the verdict:
    # the measured residual is within the dtype floor
    assert result["verdict"] == "confirmed" and result["state"] == "resolved"
    assert result["floor"]["authority"] == "supplied" and result["floor"]["proof"] == "P20"
    assert result["residual_max_abs"]["value"] < 0.05
    assert result["residual_spacings"]["value"] <= result["floor_spacings"]["value"]
    assert float(entry.predicted) == float(before[0])
    return entry, summary, result


def test_a_chain_longer_than_the_record_cap_is_summarised_without_a_project():
    debugger = _deep_debugger()
    run, view = _deep_margin(debugger, None)
    entry, summary, result = _check_deep(run, view, None)
    assert summary["chain_pointer"]["state"] == "absent" and "no project" in summary["chain_pointer"]["reason"]
    assert entry.record["pointers"] == []
    assert view["method"] == entry.chain[:forecast.MAX_ITEMS] and view["method_summary"] == summary
    assert view["method_listed"]["count"] == len(entry.chain)      # M41: the view obeys the bound
    with pytest.raises(forecast.Refusal, match="no chain pointer"):
        forecast.load_chain(entry.record, None)


def test_a_chain_longer_than_the_record_cap_is_stored_by_pointer_with_a_project(tmp_path):
    from mdb.core.project import Project

    project = Project(tmp_path / "p")
    debugger = _deep_debugger()
    run, view = _deep_margin(debugger, project.store)
    entry, summary, result = _check_deep(run, view, project.store)
    assert summary["chain_pointer"]["state"] == "stored"
    assert forecast.load_chain(entry.record, project.store) == entry.chain
    # the same forecast without a store verifies to the same numbers
    other, other_view = _deep_margin(debugger, None)
    other_entry, _, other_result = _check_deep(other, other_view, None)
    assert float(other_entry.predicted) == float(entry.predicted)
    assert other_result["residual_max_abs"] == result["residual_max_abs"]
    assert other_result["native"] == result["native"]


def test_rope_reads_the_model_inv_freq_buffer_not_the_theta_formula():
    """M49: RoPE takes the inverse frequencies from the model's float32 ``rotary_emb.inv_freq``
    buffer. A model whose buffer is not ``theta^(-2i/d)`` (here scaled by 1.5) runs with the
    buffer, and so must the forecast: the old formula predicted the rotation of another model."""
    from mdb.engine.models import tiny_model
    from mdb.ops.debugger import attach

    model = tiny_model("qwen2", "fp32")
    model.model.rotary_emb.inv_freq.mul_(1.5)
    run = attach(model).run(PROMPT)
    forecast.seal(run, "t3.L0.qr")
    run.until("t3.L0.kr")
    result = forecast.verify(run)[0]
    assert result["verdict"] == "confirmed", result
