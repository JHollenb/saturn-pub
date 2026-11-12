"""Model-free algebra, real public state/Act/transaction and serializer mechanics.

No pretrained or synthetic neural model is constructed or forwarded here.
"""
from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from saturn_pub.core import Adapter, Session
from saturn_pub.store import LocalStore
from saturn_pub.values import describe
from torch.nn import functional as F

from mdb.forecast_transaction import (
    ForecastTransaction,
    ForecastTransactionError,
    verify_transaction,
)

FILE = Path(__file__).parents[1] / "qualification/flux_gauge_spectral_campaign.py"
M = runpy.run_path(str(FILE))


class ArithmeticAdapter(Adapter):
    """Typed state transport only: no model, learned parameter or neural call."""
    model_identity = "model-free-gauge-custody-fixture"
    execution = dict(family="arithmetic-mechanics", version=1)

    def addresses(self, state):
        return ("image", "text")

    def validate(self, state):
        assert set(state) == {"image", "text", "clock", "fixed"}
        assert state["image"].dtype == state["text"].dtype == torch.bfloat16

    def boundary(self, state):
        return "arithmetic/" + str(state["clock"])

    def advance(self, state):
        return dict(state, clock=state["clock"] + 1, image=state["image"] + .125)


def test_public_common_parent_act_restore_and_cas(tmp_path):
    adapter = ArithmeticAdapter()
    state = dict(image=torch.arange(32).reshape(1, 4, 8).to(torch.bfloat16), text=torch.ones(1, 2, 8, dtype=torch.bfloat16), clock=0, fixed="untouched")
    session = Session(adapter, state); parent = session.capture(retain=False)
    store = LocalStore(tmp_path / "store"); store.save(parent)
    branch = session.fork(parent)
    for slot in ("image", "text"):
        replacement = M["gauge_replacement"](branch.read(slot), "gauge")
        before, after, receipt = M["apply_slot_edit"](branch, slot, replacement)
        store.save(before); store.save(after)
        assert receipt.to_dict()["parent"] == before.fingerprint and receipt.to_dict()["result"] == after.fingerprint
        assert after.payload["fixed"] == "untouched"
    branch.continue_()
    restored = branch.restore(parent)
    assert restored.to_dict()["result"] == parent.fingerprint and restored.to_dict()["verified_exact"]
    assert describe(branch.capture(retain=False).payload) == describe(parent.payload)
    assert describe(store.load(parent.fingerprint).payload) == describe(parent.payload)
    assert not branch.history


def test_gauge_centered_control_and_effective_rounding_are_distinct():
    own = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    gauge, control = (M["gauge_replacement"](own, name) for name in ("gauge", "mean-zero-control"))
    assert torch.equal(gauge.double().norm(dim=-1), control.double().norm(dim=-1))
    assert torch.count_nonzero(control.double().mean(-1)) == 0
    assert M["leakage"](own, gauge)["centered_leakage_rms"] == 0
    large = torch.tensor([[[256., .125, .5, 1., 2., 4., 8., 16.]]], dtype=torch.bfloat16)
    measured = M["leakage"](large, M["gauge_replacement"](large, "gauge"))
    assert measured["centered_leakage_rms"] > 0
    assert torch.equal(M["gauge_replacement"](large, "sham").view(torch.int16), large.view(torch.int16))


def test_bounded_source_factor_ritz_directions_and_work():
    identity = torch.eye(4, dtype=torch.bfloat16)
    weight_in = torch.cat([identity, identity]); weight_out = identity
    directions, report = M["factor_directions"](weight_in, weight_out, torch.ones(4), iterations=4, random_seed=602)
    assert report["directions"]["top"]["eigenvalue"] > 0 and report["directions"]["bottom"]["eigenvalue"] < 0
    assert report["directions"]["top"]["residual_norm"] < 1e-12
    assert report["directions"]["bottom"]["residual_norm"] < 1e-12
    assert report["work"]["factor_matvecs"] == 4 * report["work"]["operator_calls"]
    assert report["basis_rank"] <= 4
    again, _ = M["factor_directions"](weight_in, weight_out, torch.ones(4), iterations=4, random_seed=602)
    assert all(torch.equal(directions[name], again[name]) for name in directions)
    assert all(float(value.norm()) == pytest.approx(1.) for value in directions.values())


def test_antipodal_operand_even_write_and_original_add_order():
    x = torch.tensor([[[.125, -2., 4., 1.], [0., 0., 0., 0.]]], dtype=torch.bfloat16)
    direction = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    positive, negative = M["directional_inputs"](x, direction)
    assert torch.equal((-positive).view(torch.int16), negative.view(torch.int16))
    assert float(positive[0, 0].double().norm()) == pytest.approx(float(x[0, 0].double().norm()), rel=.005)
    plus, minus = F.silu(positive) * positive, F.silu(negative) * negative
    even = M["even_write"](plus, minus)
    residual, gate, base = x, torch.tensor(.75, dtype=torch.bfloat16), x * .25
    expected = residual + gate * (base + .25 * ((plus + minus) * .5))
    assert torch.equal(M["augmented_endpoint"](residual, gate, base, even).view(torch.int16), expected.view(torch.int16))


def plain_equation_fixture():
    """Plain tensor factor records, deliberately no torch.nn.Module instances."""
    def linear(width_in, width_out, start):
        weight = torch.arange(start, start + width_in * width_out).reshape(width_out, width_in).remainder(7).sub(3).to(torch.bfloat16) / 16
        return SimpleNamespace(weight=weight, bias=None)
    norm = lambda: SimpleNamespace(normalized_shape=(4,), weight=None, bias=None, eps=1e-6)
    rms = lambda: SimpleNamespace(normalized_shape=(4,), weight=torch.ones(4, dtype=torch.bfloat16), eps=1e-6)
    attn = SimpleNamespace(head_dim=4, to_q=linear(4, 4, 0), to_k=linear(4, 4, 2), to_v=linear(4, 4, 4),
        add_q_proj=linear(4, 4, 6), add_k_proj=linear(4, 4, 8), add_v_proj=linear(4, 4, 10),
        to_out=[linear(4, 4, 12)], to_add_out=linear(4, 4, 14), norm_q=rms(), norm_k=rms(), norm_added_q=rms(), norm_added_k=rms())
    ff = lambda start: SimpleNamespace(linear_in=linear(4, 8, start), linear_out=linear(4, 4, start + 2))
    block = SimpleNamespace(attn=attn, norm1=norm(), norm1_context=norm(), norm2=norm(), norm2_context=norm(), ff=ff(16), ff_context=ff(20))
    model = SimpleNamespace(transformer_blocks=[block])
    state = dict(image=torch.arange(8).reshape(1, 2, 4).to(torch.bfloat16), text=torch.arange(8, 16).reshape(1, 2, 4).to(torch.bfloat16),
        phase="joint", cursor=0, step=0, context=dict(img_mod=torch.ones(1, 24, dtype=torch.bfloat16) / 8,
        txt_mod=torch.ones(1, 24, dtype=torch.bfloat16) / 16, rope=(torch.ones(4, 4), torch.zeros(4, 4))))
    return model, state


def test_selected_joint0_matches_frozen_formula_without_module_execution():
    graph = runpy.run_path(str(FILE.with_name("flux_functional_denoiser.py")))
    model, state = plain_equation_fixture()
    expected = graph["functional_joint"](model, 0, state["image"], state["text"], state["context"])
    actual, account = M["selected_joint0"](graph, model, state)
    assert torch.equal(expected[0].view(torch.int16), actual["image"].view(torch.int16))
    assert torch.equal(expected[1].view(torch.int16), actual["text"].view(torch.int16))
    assert set(account) == {"R_attention", "x", "M_base", "gate_ff", "image_ln", "text_ln"}


def test_actual_transaction_script_global_seal_before_any_target(tmp_path):
    calls = []
    tx = ForecastTransaction(tmp_path / "tx", model_identity="model-free", input_identity="a" * 64, source_identity="b" * 64,
        frontier=dict(kind="architectural"), array_storage="files-v1", budgets=dict(max_predictions=1, max_executions=1, max_native_calls=1, max_bytes=1 << 20))
    with pytest.raises(ForecastTransactionError):
        tx.execute("premature", lambda _ctx: calls.append("bad"))
    assert not calls
    # A failed operation's journal is intentionally not recycled for a clean claim.
    tx.close()
    clean = ForecastTransaction(tmp_path / "clean", model_identity="model-free", input_identity="a" * 64, source_identity="b" * 64,
        frontier=dict(kind="architectural"), array_storage="files-v1", budgets=dict(max_predictions=1, max_executions=1, max_native_calls=1, max_bytes=1 << 20))
    clean.run_script([dict(op="frontier", name="own-parent", value=dict(cut="c" * 64)),
        dict(op="predict", name="all-futures", callback=lambda _ctx: dict(predictions=dict(futures=20, targets_at_seal=0))), dict(op="seal"),
        dict(op="execute", name="native-targets", callback=lambda _ctx: calls.append("after-seal") or dict(calls=1))])
    clean.close(); report = verify_transaction(tmp_path / "clean")
    assert calls == ["after-seal"] and report["verified"]


def test_real_rich_and_byte_serializers_and_signed_responses(tmp_path):
    old = M["helpers"]()
    tensor = torch.tensor([-.0, .5, -2.], dtype=torch.bfloat16)
    rich = old["_save_tensor"](tmp_path, "rich.npy", tensor)
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    minimal = old["_save_bytes"](tmp_path, "pixels.npy", pixels)
    assert torch.equal(M["load_tensor"](tmp_path, rich, "cpu").view(torch.int16), tensor.view(torch.int16))
    assert np.array_equal(M["load_tensor"](tmp_path, minimal, "cpu").numpy(), pixels)
    delta = M["signed_fields"](torch.tensor([2., -1.]), torch.tensor([1., 1.]))["/"]
    assert delta["elements"] == 2 and delta["rms"] == pytest.approx(np.sqrt(2.5)) and delta["signed_mean"] == -.5
    (tmp_path / "rich.npy").write_bytes((tmp_path / "rich.npy").read_bytes() + b"bad")
    with pytest.raises(ValueError, match="sealed array changed"):
        M["load_tensor"](tmp_path, rich, "cpu")


def test_config_census_shared_prefixes_and_no_huge_frame_retention():
    config = M["make_config"]({})
    declared = M["contract"]()
    assert sum(len(M["PRIMARY_ARMS"] if row["primary"] else M["CROSSED_ARMS"]) for row in M["CONTEXTS"]) == 20
    assert declared["functional_transitions"] == declared["native_target_transitions"] == 20 * 111 - 12
    assert declared["functional_vae_calls"] + declared["native_vae_reference_calls"] + declared["native_vae_repeat_calls"] == 44
    assert config["cuda_placements"] == 5 and config["max_gpu_components"] == 1
    assert config["max_wall_seconds"] == 10800
    assert M["retain_transition"](dict(step=0, phase="joint", index=1))
    assert not M["retain_transition"](dict(step=0, phase="single", index=5))
    assert M["retain_transition"](dict(step=3, phase="scheduler", index=20))
    assert "capture(retain=True)" not in FILE.read_text()


def test_reduced_law_complete_rows_hashes_ordering_without_learned_work(monkeypatch):
    import hashlib

    x = torch.tensor([[[1., 2., 0., 0.], [2., 1., 1., 0.]]], dtype=torch.bfloat16)
    coordinate = torch.tensor([1., -1., 0., 0.], dtype=torch.float64) / np.sqrt(2.)
    directions = dict(top=torch.tensor([1., 0., 0., 0.]), bottom=torch.tensor([0., 1., 0., 0.]), random=torch.tensor([0., 0., 1., 0.]))
    rows = {}
    monkeypatch.setattr(F, "linear", lambda *_args, **_kwargs: pytest.fail("reduced law must add no learned projection"))
    for name, direction in directions.items():
        positive, negative = M["directional_inputs"](x, direction)
        even = M["even_write"](F.silu(positive) * positive, F.silu(negative) * negative)
        rayleigh = float((coordinate * direction.double().square()).sum() / 2)
        record = M["reduced_law_prediction"](x, direction, rayleigh, even, coordinate, positive, negative)
        ideal = x.double().square().sum(-1).numpy() * rayleigh
        assert record["ideal_unrounded"]["values"] == ideal.reshape(-1).tolist()
        assert record["ideal_unrounded"]["sha256"] == hashlib.sha256(ideal.astype("<f8").tobytes()).hexdigest()
        assert record["full_finite_forecast"]["elements"] == 2
        assert record["actual_positive_operand_norm_squared"]["values"] == record["actual_negative_operand_norm_squared"]["values"]
        rows[name] = record
    ordering = M["reduced_ordering"](rows)
    assert ordering["ideal_unrounded"]["descending_per_row"] == [[0, 2, 1], [0, 2, 1]]
    assert len(ordering["full_finite_forecast"]["descending_per_row"]) == 2


def test_repeat_decoder_consumes_exact_native_final_and_truthful_origin(tmp_path):
    calls = []
    native_final = dict(latent=torch.zeros(1, 2, 4, dtype=torch.bfloat16))
    def decoder(root, _vae, state, _config, origin, *_args):
        calls.append((root, state, origin))
        return dict(origin=origin, decoded=dict(path="decoder/native-reference/decoded.npy"), png=dict(path="decoder/native-reference/image.png"))
    record = M["native_repeat_decode"]({"_native_decoder": decoder}, tmp_path, None, native_final, {}, {}, None, "n" * 64)
    assert len(calls) == 1 and calls[0][1] is native_final
    assert calls[0][2] == record["origin"] == "same_native_final_latent_repeat"
    assert record["decoded"]["path"] == "native-repeat/decoder/native-reference/decoded.npy"
    assert record["source_native_final_cut"] == "n" * 64 and record["primary_reference"] == "decoder-native-reference.json"


def test_worker_request_custody_and_real_finalizer_without_science(tmp_path, monkeypatch, capfd):
    """Outer request/marker mechanics only; config/source science validation is separate."""
    module = runpy.run_path(str(FILE)); worker = module["worker"]; namespace = worker.__globals__
    config = dict(output_scratch=str(tmp_path / "output"), output_name="model-free-result",
        transport_limits=dict(max_expanded_bytes=1 << 20, max_member_bytes=1 << 20))
    config_path = tmp_path / "config.json"; module["save"](config_path, config)
    source = tmp_path / FILE.name; source.write_bytes(FILE.read_bytes())
    request = dict(config=config, source_hashes={source.name: module["sha"](source)})
    import hashlib
    import json
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    request_path = tmp_path / "request.json"; module["save"](request_path, request)
    monkeypatch.setenv("SATURN_DEBUG_REQUEST", str(request_path)); monkeypatch.setenv("MRUN_JOB_ID", "job-mechanics")
    monkeypatch.setitem(namespace, "validate_config", lambda _config: module["contract"]())
    calls = []
    def no_science(derived, **_kwargs):
        calls.append(derived); output = Path(derived["output"]); output.mkdir(parents=True)
        result = dict(job_id=derived["job_id"], request_sha256=derived["request_sha256"], scope="model-free bootstrap fixture")
        module["save"](output / "summary.json", result)
        return result
    monkeypatch.setitem(namespace, "run", no_science)
    ready = worker(config_path)
    assert len(calls) == 1 and ready["job_id"] == "job-mechanics" and ready["request_sha256"] == request["request_sha256"]
    marker = next(line for line in capfd.readouterr().out.splitlines() if line.startswith("MDB_DIRECTORY_READY "))
    assert json.loads(marker.removeprefix("MDB_DIRECTORY_READY ")) == ready
    source.write_bytes(source.read_bytes() + b"\n# corrupted synthetic source copy\n")
    try:
        with pytest.raises(ValueError, match="executed source changed"):
            worker(config_path)
        assert len(calls) == 1
        Path(config_path).write_bytes(module["canonical"](dict(config, changed=True)))
        with pytest.raises(ValueError, match="request/config identity differs"):
            worker(config_path)
        assert len(calls) == 1
    finally:
        namespace["faulthandler"].cancel_dump_traceback_later() if "faulthandler" in namespace else __import__("faulthandler").cancel_dump_traceback_later()
