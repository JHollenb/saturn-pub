"""Tiny CPU saved-QKV cross; public wheels, no trained weights or jobs."""
from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers.models.transformers import transformer_flux2 as native

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/flux_image_query_bank.py"
C = runpy.run_path(str(PATH))
R = runpy.run_path(str(HERE / "qualification/flux_image_reader_forecast.py"))
F = runpy.run_path(str(HERE / "tests/test_flux_image_reader_forecast.py"))


@pytest.fixture(scope="module")
def saved_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("source-reader")
    config, parents, weights = F["fixture"](root)
    config["request_sha256"] = "1" * 64
    previous = os.environ.get("MRUN_JOB_ID")
    os.environ["MRUN_JOB_ID"] = "tiny-reader-source"
    try:
        result = R["run"](config)
    finally:
        if previous is None:
            os.environ.pop("MRUN_JOB_ID", None)
        else:
            os.environ["MRUN_JOB_ID"] = previous
    R["finalize_directory"](config["output"], job_id="tiny-reader-source", request_sha256=config["request_sha256"])
    source_config = json.loads(json.dumps(config))
    source_config.update(source_root=config["output"], source_job_id="tiny-reader-source", source_request_sha256="1" * 64,
                         source_manifest_sha256=F["OLD"]["C"]["sha"](Path(config["output"]) / "files.json"),
                         source_input_seal_sha256=result["input_seal_sha256"], source_forecast_seal_sha256=result["forecast_seal_sha256"],
                         request_sha256="2" * 64)
    for name in (C["READER"], C["INVENTORY"]):
        source_config["helper_source_sha256"][name] = F["OLD"]["C"]["sha"](HERE / "qualification" / name)
    return source_config, parents, weights


def configuration(saved_source, root):
    config = json.loads(json.dumps(saved_source[0]))
    config["output"] = str(root / "query-bank")
    return config


@pytest.fixture(scope="module")
def completed(saved_source, tmp_path_factory):
    config = configuration(saved_source, tmp_path_factory.mktemp("cross"))
    events = []
    result = C["run"](config, progress=lambda event: events.append(event))
    return config, result, events


def test_all_four_writes_sealed_before_exactly_four_targets(completed):
    config, summary, events = completed
    assert [row["kind"] for row in events] == ["source_bound", "cohort_sealed"] + ["target_completed"] * 4
    assert events[0]["target_entries"] == events[1]["target_entries"] == 0
    assert summary["branches"] == ["q0-b0", "q0-b1", "q1-b0", "q1-b1"]
    assert summary["forecast_primitive_counts"] == dict(native_attention=4, linear=12, layernorm=4, swiglu=4)
    assert summary["target_counts"] == dict(native_blocks=4, native_attention=4, linear=48, layernorm=16, rmsnorm=16, swiglu=8)
    assert summary["complete_native_targets"] == 4 and summary["new_prefix_entries"] == summary["denied_target_attempts"] == 0
    assert summary["vae_calls"] == summary["suffix_steps"] == summary["encodes"] == summary["noise_draws"] == 0
    from mdb_lanes.forecast_transaction import verify_transaction
    verified = verify_transaction(Path(config["output"]) / "transaction", expected_seal_sha256=summary["forecast_seal_sha256"])
    assert verified["verified"]


def test_actual_receiving_and_consumed_qkv_are_unambiguous_and_exact(completed):
    config, _, _ = completed
    _, m, core = C["helpers"](config)
    source = C["load_source"](config, R, m, core)
    data = Path(config["output"])
    receiving_hashes = set()
    for branch in C["BRANCHES"]:
        target = json.loads((data / "targets" / (branch["id"] + ".json")).read_text())
        assert not {"q", "k", "v"} & target["actual"].keys()
        assert target["forecast_minus_native"]["changed_values"] == 0
        assert target["operation"]["original_sdpa_calls"] == 1 and not target["operation"]["return_replaced"]
        assert target["operation"]["act_receipt"] is None
        receiving_hashes.add(tuple(target["receiving_descriptors"][name]["sha256"] for name in ("q", "k", "v")))
        for name in ("q", "k", "v"):
            index = str(branch["sourceQindex"] if name == "q" else branch["sourceBankindex"])
            actual = core["H"]["load_array"](data, target["consumed_qkv"][name])
            assert np.array_equal(actual, source["sources"][index]["values"][name])
            assert target["consumed_descriptors"][name]["stride"] == source["sources"][index]["layouts"][name]["stride"]
    assert len(receiving_hashes) == 1  # Every unchanged native block formed midpoint QKV.
    changes = json.loads((data / "source-qkv-changes.json").read_text())
    assert all(row["text_rows"]["changed_values"] == 0 for row in changes.values())
    assert changes["q"]["image_rows"]["changed_values"] > 0


def test_retrospective_controls_and_fixed_recipient_denominators(completed):
    config, _, _ = completed
    _, _, core = C["helpers"](config)
    data = Path(config["output"])
    for key in ("q0-b0", "q1-b1"):
        row = json.loads((data / "targets" / (key + ".json")).read_text())
        assert row["retrospective_control"]["native_minus_prior"]["changed_values"] == 0
    actual = {branch["id"]: core["H"]["load_array"](data, json.loads((data / "targets" / (branch["id"] + ".json")).read_text())["actual"]["image"])
              for branch in C["BRANCHES"]}
    result = json.loads((data / "contrasts.json").read_text())
    total = actual["q1-b1"].astype(np.float64) - actual["q0-b0"].astype(np.float64)
    interaction = (actual["q1-b1"].astype(np.float64) - actual["q1-b0"].astype(np.float64)
                   - actual["q0-b1"].astype(np.float64) + actual["q0-b0"].astype(np.float64))
    assert result["native_total_rms"] == float(np.sqrt(np.mean(total * total)))
    assert result["records"]["interaction"]["native"]["rms"] == float(np.sqrt(np.mean(interaction * interaction)))
    assert "identical midpoint" in result["denominator"]


def test_zero_total_denominator_remains_undefined():
    values = {branch["id"]: np.ones((1, 2, 3), np.float32) for branch in C["BRANCHES"]}
    report = C["contrast_accounts"](values, values)
    assert report["native_total_rms"] == 0
    assert all(row["native"]["relative_to_native_total_rms"] is None for row in report["records"].values())
    assert all(row["cosine"] is None for row in report["records"].values())


def test_source_sidecar_must_join_original_sealed_frontier(saved_source, tmp_path):
    config = configuration(saved_source, tmp_path)
    source = tmp_path / "source-copy"
    shutil.copytree(config["source_root"], source)
    sham = json.loads((source / "prefixes/sham.json").read_text())
    changed = json.loads((source / "prefixes/midpoint.json").read_text())
    sham["values"]["q"] = changed["values"]["q"]
    path = source / "prefixes/sham.json"
    path.write_text(json.dumps(sham))
    manifest = json.loads((source / "files.json").read_text())
    _, m, _ = C["helpers"](config)
    manifest["files"]["prefixes/sham.json"] = dict(sha256=m["sha"](path), bytes=path.stat().st_size)
    (source / "files.json").write_text(json.dumps(manifest))
    config.update(source_root=str(source), source_manifest_sha256=m["sha"](source / "files.json"))
    with pytest.raises(ValueError, match="sealed prefix QKV"):
        C["run"](config)
    assert not (Path(config["output"]) / "targets").exists()


@pytest.mark.parametrize("field", ("source_job_id", "source_forecast_seal_sha256"))
def test_wrong_historical_identity_refused_before_native(saved_source, tmp_path, field):
    config = configuration(saved_source, tmp_path)
    config[field] = "wrong" if field == "source_job_id" else "0" * 64
    with pytest.raises(Exception):
        C["run"](config)
    assert not (Path(config["output"]) / "targets").exists()


def test_dispatch_intervention_refuses_wrong_payload_and_restores(saved_source):
    config, parents, weights = saved_source
    _, m, _ = C["helpers"](config)
    block = native.Flux2TransformerBlock(**config["module_config"]).eval().to(torch.bfloat16)
    block.load_state_dict({name: torch.from_numpy(value).to(torch.bfloat16) for name, value in weights.items()})
    p = parents["sham"].payload
    from mdb_lanes.forward_recording import tensor_descriptor
    q = torch.zeros(1, 7, 2, 4, dtype=torch.bfloat16)
    chosen = {name: q for name in ("q", "k", "v")}
    declared = {name: dict(tensor_descriptor(value), stride=list(value.stride())) for name, value in chosen.items()}
    declared["k"]["sha256"] = "0" * 64
    original = native.dispatch_attention_fn
    counts = dict.fromkeys(("native_blocks", "native_attention", "linear", "layernorm", "rmsnorm", "swiglu"), 0)
    before = {name: (len(module._forward_hooks), len(module._forward_pre_hooks)) for name, module in block.named_modules()}
    with pytest.raises(ValueError, match="sealed QKV"), torch.inference_mode():
        C["capture_intervened"](block, p, native, chosen, declared, m, counts)
    assert native.dispatch_attention_fn is original
    assert before == {name: (len(module._forward_hooks), len(module._forward_pre_hooks)) for name, module in block.named_modules()}


def test_corrupt_new_seal_blocks_all_native_targets(saved_source, tmp_path):
    config = configuration(saved_source, tmp_path)
    completions = []
    handle = torch.nn.modules.module.register_module_forward_hook(lambda module, args, result: completions.append(True) if isinstance(module, native.Flux2TransformerBlock) else None)
    def tamper(row):
        if row["kind"] == "cohort_sealed":
            (Path(config["output"]) / "transaction/seal.json").write_text("{}")
    try:
        with pytest.raises(Exception, match="seal changed"):
            C["run"](config, progress=tamper)
    finally:
        handle.remove()
    assert not completions and not (Path(config["output"]) / "targets").exists()


def test_forecast_guard_latches_even_caught_target_attempts(saved_source, tmp_path, monkeypatch):
    config = configuration(saved_source, tmp_path)
    original_forward, original_dispatch, created = native.Flux2TransformerBlock.forward, native.dispatch_attention_fn, []
    def track(self, *args, **kwargs):
        created.append(self)
        return original_forward(self, *args, **kwargs)
    # The meta-constructed target is discovered by a temporary constructor wrapper;
    # no legitimate full forward is expected before cohort release.
    original_init = native.Flux2TransformerBlock.__init__
    def made(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)
    monkeypatch.setattr(native.Flux2TransformerBlock, "__init__", made)
    monkeypatch.setattr(native.Flux2TransformerBlock, "forward", track)
    def intrusive(*args, **kwargs):
        p = saved_source[1]["sham"].payload
        try:
            created[0](p["image"], p["text"], p["context"]["img_mod"], p["context"]["txt_mod"], p["context"]["rope"])
        except ValueError:
            pass
        return original_dispatch(*args, **kwargs)
    monkeypatch.setattr(native, "dispatch_attention_fn", intrusive)
    with pytest.raises(ValueError, match="target attempted"):
        C["run"](config)
    assert not (Path(config["output"]) / "targets").exists()


def test_cli_finalization_and_recipe_only_output(saved_source, tmp_path):
    config = configuration(saved_source, tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH), "--config", str(path)], cwd=tmp_path,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = json.loads(next(line.split(" ", 1)[1] for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    _, m, _ = C["helpers"](config)
    data = Path(config["output"])
    assert marker["files_manifest_sha256"] == m["sha"](data / "files.json")
    assert marker["expanded_bytes"] < 32 << 20
    recipes = json.loads((data / "parameter-artifacts.json").read_text())
    assert all(not (data / row["path"]).exists() for row in recipes["artifacts"])
