"""Original-source arithmetic, saved custody and real public SET mechanics."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from saturn_pub.core import Session
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb_lanes.forecast_transaction import ForecastTransaction

ROOT = Path(__file__).parents[1]
FILE = ROOT / "qualification/flux_binding_residual_write.py"
M = runpy.run_path(str(FILE))
H = runpy.run_path(str(FILE.with_name("flux_binding_residual_sources.py")))
OLD_TEST = runpy.run_path(str(Path(__file__).with_name("test_flux_binding_support_sources.py")))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def inventory(root, source):
    files = {path.relative_to(root).as_posix(): dict(sha256=H["sha"](path), bytes=path.stat().st_size)
             for path in root.rglob("*") if path.is_file() and path.name != "files.json"}
    write(root / "files.json", dict(job_id=source["job_id"], request_sha256=source["request_sha256"], files=files))
    source["manifest_sha256"] = H["sha"](root / "files.json")


def source_fixture(tmp_path):
    """Actual old writer -> one wrapped lifecycle-shaped source -> new writer."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    old_config, source_root, source_store, adapter, _ = OLD_TEST["source_fixture"](legacy)
    old_helper = OLD_TEST["M"]
    prepared = old_helper["prepare_binding_sources"](old_config, source_root, source_store, adapter)["toy"]
    _, base, old = M["common_helpers"]()
    case = dict(P=base["cut_record"](prepared["pcut"]), Q=base["cut_record"](prepared["qcut"]),
                support=prepared["support"], text_payloads=prepared["witnesses"]["text_payloads"],
                base_conditioning=prepared["witnesses"]["base_conditioning"], witnesses=prepared["witnesses"])
    write(source_root / "binding-sources.json", dict(toy=case))
    index = {}
    q = prepared["qcut"]
    for arm in ("whole_Q", "support_P"):
        session = Session(adapter, q.payload, parent=q.parent)
        _, entry, _ = base["apply_slot_edit"](session, "text", prepared["text_payloads"][arm])
        source_store.save(entry)
        final_state = copy.deepcopy(dict(entry.payload))
        final_state.update(step=4, phase="project", cursor=0)
        final = Session(adapter, final_state, parent=entry.fingerprint).capture(retain=False)
        source_store.save(final)
        branch = "cases/binding-toy/" + arm
        write(source_root / branch / "native-rows.json", [dict(retained=base["cut_record"](entry))])
        png = source_root / branch / "decoder/native-reference/image.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"PNG saved byte fixture " + arm.encode())
        write(source_root / branch / "decoder-native-reference.json", dict(png=dict(
            path=png.relative_to(source_root / branch).as_posix(), sha256=H["sha"](png), bytes=png.stat().st_size)))
        index["binding-toy/" + arm] = dict(root=branch, native_final=final.fingerprint)
    write(source_root / "state-index.json", index)
    config = dict(contexts=["toy"], binding_plan=old_config["binding_plan"])
    metadata_pins = {key + "_sha256": "a" * 64 for key in old["_source_keys"]()}
    config.update(metadata_pins)
    request = dict(config=dict(metadata_pins), source_hashes={"flux_lifecycle_campaign.py": M["HELPERS"]["flux_lifecycle_campaign.py"]})
    request_sha = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    request["request_sha256"] = request_sha
    source = dict(root=str(source_root.resolve()), request_sha256=request_sha, job_id="job-fixture", manifest_sha256="")
    config["source"] = source
    write(source_root / "request.json", request)
    write(source_root / "plan.json", dict(request["config"], job_id=source["job_id"], request_sha256=request_sha, output=source["root"]))
    tx = ForecastTransaction(source_root / "forecast-transaction", model_identity=adapter.model_identity,
                             input_identity="fixture", source_identity="fixture", execution_scope="source observations",
                             frontier=dict(kind="architectural", origin="actual arithmetic StateCut fixture"))
    tx.frontier("observed-parents", dict(source="actual public CAS fixture"))
    tx.predict("all-futures", lambda _ctx: dict(predictions=dict(cases=["toy"])))
    seal = tx.seal()
    tx.execute("original-native-targets", lambda _ctx: dict(observation="fixture, no model"))
    tx.close()
    write(source_root / "summary.json", dict(job_id=source["job_id"], request_sha256=request_sha, forecast_seal=seal["seal_sha256"]))
    write(source_root / "source.json", dict(job_id=source["job_id"], request_sha256=request_sha))
    inventory(source_root, source)
    output = tmp_path / "new-output"
    output.mkdir()
    return config, output, LocalStore(output / "state-store"), adapter, base, old


def test_definitions_are_framework_free_and_helpers_are_hash_checked(tmp_path):
    script = "import runpy,sys; runpy.run_path(sys.argv[1]); assert not {'torch','numpy','saturn_pub'} & sys.modules.keys()"
    for path in (FILE, FILE.with_name("flux_binding_residual_sources.py")):
        subprocess.run([sys.executable, "-I", "-B", "-c", script, str(path)], check=True)
    M["common_helpers"]()
    assert M["make_config"]({})["contexts"] == ["anchor", "wording"]


def test_masked_decomposition_cast_order_and_nonadditive_endpoint_are_observed(monkeypatch):
    pin = torch.zeros(1, 3, 2, dtype=torch.bfloat16)
    qin = pin.clone()
    pout, qout = torch.ones_like(pin), torch.ones_like(pin)
    pin[:, 1] = 2**-8
    qin[:, 1] = -(2**-30)
    pout[:, 1] = 1 + 2**-7
    qout[:, 0, 0] = -0.
    seen = []
    original_to = torch.Tensor.to

    def casting(value, *args, **kwargs):
        if args and args[0] == torch.bfloat16:
            seen.append(value.dtype)
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", casting)
    result = H["decompose_sources"](pin, qin, pout, qout, [1])
    assert seen == [torch.float64] * 3
    for arm in M["ARMS"]:
        assert OLD_TEST["M"]["same_bits"](result["text_payloads"][arm][:, [0, 2]], qout[:, [0, 2]])
    assert torch.signbit(result["text_payloads"]["residual_only"][0, 0, 0])
    np_dr = pin.double().numpy() - qin.double().numpy()
    np_dw = (pout.double().numpy() - pin.double().numpy()) - (qout.double().numpy() - qin.double().numpy())
    assert np.array_equal(result["deltas"]["DR"].numpy(), np_dr)
    assert np.array_equal(result["deltas"]["DW"].numpy(), np_dw)
    assert torch.count_nonzero(result["deltas"]["closure"]) == 0
    assert torch.count_nonzero(result["masked_deltas"]["DW"][:, [0, 2]]) == 0
    # The declared operator is tested directly, including actual CPU double rounding.
    midpoint_sides = torch.tensor([1 + 2**-8 - 2**-30, 1 + 2**-8 + 2**-30], dtype=torch.float64)
    actual = original_to(midpoint_sides, torch.bfloat16)
    assert torch.equal(result["text_payloads"]["residual_only"][:, 1].reshape(-1).view(torch.int16), actual[1].expand(2).view(torch.int16))
    rounded = result["text_payloads"]
    expected_remainder = (rounded["both_RW"].double() - rounded["residual_only"].double()) - (rounded["write_only"].double() - rounded["whole_Q"].double())
    assert torch.equal(result["entry_nonadditive_remainder"], expected_remainder)
    assert torch.count_nonzero(expected_remainder) > 0


def test_zero_and_singular_directions_are_retained_but_bad_sources_reject():
    value = torch.ones(1, 3, 2, dtype=torch.bfloat16)
    result = H["decompose_sources"](value, value, value, value, [1])
    assert result["analysis"]["geometry"]["cosine"] is None
    assert all(torch.equal(endpoint, value) for endpoint in result["text_payloads"].values())
    with pytest.raises(ValueError, match="CPU BF16"):
        H["decompose_sources"](value.float(), value, value, value, [1])
    with pytest.raises(ValueError, match="support"):
        H["decompose_sources"](value, value, value, value, [1, 1])


def test_actual_saved_source_writer_and_public_acts_preserve_all_nontext(tmp_path, monkeypatch):
    config, output, store, adapter, base, old = source_fixture(tmp_path)
    source_root = Path(config["source"]["root"])
    before = {path: H["sha"](path) for path in source_root.rglob("*") if path.is_file()}
    monkeypatch.setattr(torch.nn.Module, "__call__", lambda *_args, **_kwargs: pytest.fail("no model forward"))
    prepared, sidecar = H["prepare_sources"](config, output, store, adapter, base, old)
    case = prepared["toy"]
    saved = sidecar["cases"]["toy"]
    q = case["qcut"]
    master = Session(adapter, q.payload, parent=q.parent)
    for arm in M["ARMS"]:
        branch = master.fork(q)
        before_cut, after, receipt = base["apply_slot_edit"](branch, "text", case["text_payloads"][arm])
        assert receipt.to_dict()["parent"] == before_cut.fingerprint
        assert OLD_TEST["M"]["same_bits"](after.payload["text"], case["text_payloads"][arm])
        assert describe({key: value for key, value in after.payload.items() if key != "text"}) == describe({key: value for key, value in q.payload.items() if key != "text"})
        branch.restore(q)
        assert branch.capture(retain=False).fingerprint == q.fingerprint
        restored_array = base["load_tensor"](output, saved["text_payloads"][arm], "cpu")
        assert OLD_TEST["M"]["same_bits"](restored_array, case["text_payloads"][arm])
    assert sidecar["new_model_calls"] == 0 and sidecar["source_transaction"]["seal_sha256"]
    assert saved["known_endpoints"]["both_RW"]["historical_arm"] == "support_P"
    assert set(saved["deltas"]) == {"DR", "DW", "DR_plus_DW", "output_difference", "closure"}
    assert all(H["sha"](path) == checksum for path, checksum in before.items())


def test_source_hash_strict_model_and_known_combined_control_are_authoritative(tmp_path):
    config, output, store, adapter, base, old = source_fixture(tmp_path)
    adapter.model_identity = "different-original-model"
    with pytest.raises(ValueError, match="strict model identity"):
        H["prepare_sources"](config, output, store, adapter, base, old)
    adapter.model_identity = OLD_TEST["SavedArithmetic"].model_identity
    root = Path(config["source"]["root"])
    source_cases = json.loads((root / "binding-sources.json").read_text())
    source_cases["toy"]["text_payloads"]["support_P"] = source_cases["toy"]["text_payloads"]["whole_Q"]
    write(root / "binding-sources.json", source_cases)
    inventory(root, config["source"])
    with pytest.raises(ValueError, match="known endpoint payload differs"):
        H["prepare_sources"](config, output, store, adapter, base, old)
    evidence = H["Evidence"](config["source"])
    with pytest.raises(ValueError, match="unsafe source member"):
        evidence.member("../outside")
    (root / "summary.json").write_text("{}\n")
    with pytest.raises(ValueError, match="member changed"):
        evidence.read("summary.json")


def test_fresh_binding_adapter_preserves_original_identity_and_rejects_loader_drift(tmp_path, monkeypatch):
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    monkeypatch.setattr(torch.nn.Module, "__call__", lambda *_args, **_kwargs: pytest.fail("no model forward"))
    flux = Flux2Transformer2DModel(in_channels=8, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
        timestep_guidance_channels=8, axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).to(torch.bfloat16)
    flux.register_to_config(_name_or_path="/raid/yiyi/klein-4b-distilled-diffusers/transformer")
    vae = AutoencoderKLFlux2(down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,), norm_num_groups=1, layers_per_block=1, latent_channels=2,
        mid_block_add_attention=False, sample_size=8).to(torch.bfloat16)
    root = tmp_path / "checkpoint"
    Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, transformer=flux,
        text_encoder=None, tokenizer=None, is_distilled=True).save_pretrained(root)
    native = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16,
        local_files_only=True, text_encoder=None, tokenizer=None)
    config = dict(flux_parameter_root=str(root), flux_scheduler_config=str(root / "scheduler/scheduler_config.json"))
    lifecycle, _ = M["frozen_helpers"]()
    native_adapter = Flux2KleinAdapter.from_pipeline(native, granularity="operation", device="cpu")
    before = describe(native.transformer.state_dict())
    fresh = M["fresh_binding_adapter"](native.transformer, native.vae, config, lifecycle, device="cpu")
    fresh.validate_execution()
    assert fresh.model_identity == native_adapter.model_identity and fresh.execution == native_adapter.execution
    assert describe(native.transformer.state_dict()) == before
    native.transformer.register_to_config(_name_or_path="foreign-loader")
    with pytest.raises(ValueError, match="unqualified transformer loader path"):
        M["fresh_binding_adapter"](native.transformer, native.vae, config, lifecycle, device="cpu")
    with pytest.raises(ValueError, match="configuration changed"):
        fresh.validate_execution()
