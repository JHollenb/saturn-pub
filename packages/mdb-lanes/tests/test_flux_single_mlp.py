"""Entry-only MLP equations, scoped native witnesses and prepared panel."""
from __future__ import annotations

import hashlib
import json
import math
import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
from mdb_lanes.tensor_archive import load_tensor_archive, write_tensor_archive

HERE = Path(__file__).parents[1]
mlp = runpy.run_path(str(HERE / "qualification/flux_single_mlp.py"))
tiny = runpy.run_path(str(HERE / "tests/test_flux_single_contexts.py"))
engine = runpy.run_path(str(HERE / "qualification/flux_single_contexts.py"))
panel_engine = runpy.run_path(str(HERE / "qualification/flux_single_contraction_campaign.py"))
fresh = runpy.run_path(str(HERE / "qualification/flux_single_mlp_campaign.py"))


def test_entry_equation_matches_scalar_norm_modulation_paired_rows_and_casts():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(1, 4, 5)).astype(np.float32)
    mod = rng.normal(size=(1, 15)).astype(np.float32)
    weights = rng.normal(size=(4, 5)).astype(np.float32)
    bias = rng.normal(size=4).astype(np.float32)
    result = mlp["predict_entry_samples"](x, mod, weights, bias, eps=1e-6, query_indices=[0, 3], neuron_indices=[1, 7], hidden_width=5, mlp_width=9)
    assert result["weight_row_indices"] == [16, 22, 25, 31]
    shift, scale, _ = np.split(mod.reshape(1, 1, 15).astype(np.float64), 3, axis=-1)
    for qi, position in enumerate((0, 3)):
        values = [float(v) for v in x[0, position]]
        mean = math.fsum(values) / 5
        variance = math.fsum((v - mean) ** 2 for v in values) / 5
        norm = [(v - mean) / math.sqrt(variance + 1e-6) for v in values]
        expected_mod = [(1 + float(s)) * n + float(b) for n, s, b in zip(norm, scale[0, 0], shift[0, 0], strict=True)]
        projected = [math.fsum(float(w) * v for w, v in zip(row, expected_mod, strict=True)) + float(b) for row, b in zip(weights, bias, strict=True)]
        expected = [u / (1 + math.exp(-u)) * v for u, v in zip(projected[:2], projected[2:], strict=True)]
        np.testing.assert_allclose(result["analytic"]["layernorm"][0, qi], norm, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(result["analytic"]["activation"][0, qi], expected, rtol=1e-13, atol=1e-13)
    def cast(value):
        return torch.from_numpy(np.array(value, copy=True)).float().to(torch.bfloat16).float().numpy()
    reference = result["bf16_reference"]
    norm_b, scale_b = cast(result["analytic"]["layernorm"]), cast(1 + scale)
    product_b = cast(scale_b * norm_b)
    mod_b = cast(product_b.astype(np.float64) + shift)
    uv_b = cast(mod_b.astype(np.float64) @ weights.astype(np.float64).T + bias.astype(np.float64))
    np.testing.assert_array_equal(reference["modulated"], mod_b)
    np.testing.assert_array_equal(reference["u"], uv_b[:, :, :2])
    np.testing.assert_array_equal(reference["v"], uv_b[:, :, 2:])
    np.testing.assert_array_equal(reference["activation"], cast(reference["silu_u"] * reference["v"]))
    assert not reference["activation"].flags.writeable
    for kwargs in ({"eps": 0.}, {"query_indices": [0, 0]}, {"neuron_indices": [1, 1]}):
        with pytest.raises(ValueError):
            mlp["predict_entry_samples"](x, mod, weights, bias, **{ "eps": 1e-6, "query_indices": [0, 3], "neuron_indices": [1, 7], "hidden_width": 5, "mlp_width": 9, **kwargs})


@pytest.mark.parametrize("collapsed", [False, True])
def test_early_transaction_then_two_native_witnesses_and_complete_prepared_panel(tmp_path, monkeypatch, collapsed):
    monkeypatch.chdir(tmp_path)
    shutil.copyfile(HERE / "qualification/flux_single_kv.py", "flux_single_kv.py")
    shutil.copyfile(HERE / "qualification/flux_single_contraction.py", "flux_single_contraction.py")
    shutil.copyfile(HERE / "qualification/flux_single_contexts.py", "flux_single_contexts.py")
    model, adapter, session = tiny["tiny_session"]()
    cuts, builder = engine["build_sources"](adapter=adapter, session=session, output=tmp_path / "builder", context={"id": "tiny"})
    cuts, prep = fresh["prepare_sources"](adapter=adapter, cuts=cuts, output=tmp_path / "prep", builder_ref={"path": "builder/builder.json"}, wall=lambda: None)
    assert builder["counts"]["adapter_steps"] + prep["counts"]["adapter_steps"] == 108
    assert builder["counts"]["acts"] + prep["counts"]["acts"] == 4
    if collapsed:
        cuts = {"zero": cuts["zero"], "text": cuts["zero"]}
    block = model.single_transformer_blocks[6]
    h, i, neurons = block.attn.query_dim, block.attn.mlp_hidden_dim, [0, 3]
    indices = mlp["mlp_weight_row_indices"](hidden_width=h, mlp_width=i, neuron_indices=neurons)
    weights = block.attn.to_qkv_mlp_proj.weight[indices].detach().float().numpy().copy()
    bias = block.attn.to_qkv_mlp_proj.bias
    bias = bias[indices].detach().float().numpy().copy() if bias is not None else None
    incoming = {role: (cut.payload["image"].float().numpy().copy(), cut.payload["context"]["single_mod"].float().numpy().copy()) for role, cut in cuts.items()}
    def consumer(state, **kwargs):
        digest = hashlib.sha256(state.read("latent").float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2, 2, 3), list(digest[:3]), dtype=np.uint8))
    adapter.decode = consumer  # Synthetic native test consumer, no VAE qualification.
    cfg = {"branches": runpy.run_path("flux_single_contraction.py")["BRANCHES"], "max_adapter_steps": 137, "max_acts": 10,
        "max_candidate_calls": 5, "max_images": 9, "max_vae_calls": 9, "max_wall_seconds": 300,
        "query_indices": [0, 3], "output_dims": [0, 7], "array_storage": "bf16-bits", "hypotheses": {}, "scope": "tiny prepared mechanics"}
    calls = []
    handle = block.register_forward_pre_hook(lambda *_: calls.append("native"))
    with ForecastTransaction(tmp_path / "early", model_identity=adapter.model_identity, input_identity="entry", source_identity="source", frontier={"kind": "architectural"}) as tx:
        tx.predict("entry_mlp", lambda _: {"predictions": {role: mlp["predict_entry_samples"](*values, weights, bias,
            eps=block.norm.eps, query_indices=[0, 3], neuron_indices=neurons, hidden_width=h, mlp_width=i) for role, values in incoming.items()}})
        assert calls == []
        seal = tx.seal()
        def execute(_):
            with mlp["capture_source_calls"](block, text_rows=cuts["zero"].payload["text"].shape[1], query_indices=[0, 3], neuron_indices=neurons) as captured:
                result = panel_engine["run_panel"](pipe=SimpleNamespace(transformer=model, vae=torch.nn.Identity()), adapter=adapter,
                    inputs=cuts, output=tmp_path / "panel", config=cfg, request={"job_id": "parent", "request_sha256": "r" * 64, "source_hashes": {}},
                    metadata={}, selection={"kind": "fresh-builder"}, prepared_inputs=True)
            assert captured["calls"] == 7 and set(captured["rows"]) == {"zero", "text"}
            for index, role in enumerate(("zero", "text")):
                row = captured["rows"][role]
                assert row["call_index"] == index and row["u"].shape == row["activation"].shape == (1, 2, 2)
                assert row["layernorm"].shape == row["modulated_input"].shape == (1, 2, h)
                assert torch.equal(torch.nn.functional.silu(row["u"]) * row["v"], row["activation"])
                for name in ("layernorm", "modulated_input", "u", "v", "activation"):
                    record = write_tensor_archive(tmp_path / (role + "-" + name + ".npy"), row[name])
                    np.testing.assert_array_equal(load_tensor_archive(tmp_path, record), row[name].float().numpy())
            return {"counts": result["counts"]}
        tx.execute("native", execute)
    handle.remove()
    assert verify_transaction(tx.root, expected_seal_sha256=seal["seal_sha256"])["verified"]
    observed = tx.observation("native")["counts"]
    assert observed["adapter_steps"] == 137 and observed["receipt_emissions"] == 147 and observed["acts"] == 10
    source = json.loads((tmp_path / "panel/source.json").read_text())
    assert source["prepared_inputs"] and source["array_storage"] == "bf16-bits" and source["preseal_counts"]["adapter_steps"] == 0
    if collapsed:
        assert source["source_contrast_observations"] == {"image_exact": True, "text_exact": True}
        assert len(list((tmp_path / "panel/rows").glob("*.json"))) == 9
    replay = json.loads((tmp_path / "panel/source-replay.json").read_text())
    assert replay["kind"] == "prepared-inputs" and replay["counts"]["adapter_steps"] == 0
    records = json.loads((tmp_path / "panel/candidates.json").read_text())["candidates"]
    assert records["zero"]["ports"]["original_concat"]["stored_dtype"] == "uint16"
    assert not any(module._forward_pre_hooks or module._forward_hooks for module in model.modules())


def test_native_witness_hooks_cleanup_on_forward_failure(tmp_path, monkeypatch):
    model, adapter, session = tiny["tiny_session"]()
    from mdb_lanes.native_navigation import NativeCutNavigator
    n = NativeCutNavigator(session, max_steps=100)
    n.until(logical_step=3, phase="single", index=6)
    cut, block = n.capture(), model.single_transformer_blocks[6]
    def broken(*args, **kwargs):
        raise RuntimeError("tiny source failed")
    monkeypatch.setattr(block.attn.mlp_act_fn, "forward", broken)
    with pytest.raises(RuntimeError, match="tiny source failed"):
        with mlp["capture_source_calls"](block, text_rows=cut.payload["text"].shape[1], query_indices=[0], neuron_indices=[0]):
            block(hidden_states=torch.cat([cut.payload["text"], cut.payload["image"]], dim=1), encoder_hidden_states=None,
                temb_mod=cut.payload["context"]["single_mod"], image_rotary_emb=cut.payload["context"]["rope"])
    assert not any(module._forward_pre_hooks or module._forward_hooks for module in model.modules())
