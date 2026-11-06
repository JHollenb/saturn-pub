"""Installed tiny imported-StateCut mechanics; no real checkpoint/model jobs."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from PIL import Image
from safetensors.torch import load_file
from saturn_pub import Session
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb.flux_final_latent_closure import FluxFinalLatentClosureAdapter
from mdb.forecast_transaction import load_forecast_tree
from mdb.tensor_archive import load_tensor_archive

HERE = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    data, frozen = (os.environ.get(name) for name in ("MDB_FACTORIAL_TINY_DATA", "MDB_FACTORIAL_FROZEN_PAYLOAD"))
    if not data or not frozen:
        pytest.skip("requires explicit saved tiny fixture and immutable original producer source cohort")
    root = tmp_path_factory.mktemp("native-corner")
    for path in Path(frozen).glob("*.py"):
        shutil.copyfile(path, root / path.name)
    shutil.copyfile(HERE / "qualification/flux_native_corner.py", root / "flux_native_corner.py")
    module = runpy.run_path(str(root / "flux_native_corner.py"))
    torch.set_num_threads(1)
    data = Path(data)
    recipes = json.loads((data / "parameter-artifacts.json").read_text())
    weights = Path(recipes["defaults"]["checkpoint_root"]) / "transformer/diffusion_pytorch_model.safetensors"
    model = Flux2Transformer2DModel(in_channels=4, num_layers=5, num_single_layers=20,
        attention_head_dim=8, num_attention_heads=1, joint_attention_dim=6, timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval().to(torch.bfloat16)
    model.load_state_dict(load_file(weights))
    adapter = FluxFinalLatentClosureAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    store = LocalStore(data / "typed-store")
    parents, predictions, references = {}, {}, {}
    for cell in module["CELLS"]:
        native = [json.loads(line) for line in (data / "native" / (cell + ".jsonl")).read_text().splitlines()]
        parents[cell] = store.load(native[27]["capture"]["cut"])
        transaction = data / "transactions" / cell
        program = module["support"]()["prior"]()["util"]()["json_value"](load_forecast_tree(transaction,
            json.loads((transaction / "forecasts/program.json").read_text())["predictions"]))
        record = program["transitions"][27]["outputs"]["latent"]
        value = torch.from_numpy(np.array(load_tensor_archive(data, record["archive"]), copy=True)).bfloat16()
        predictions[cell] = {"latent": value, "binding": {"record": record, "scope": "saved synthetic fixture"}}
        references[cell] = {"transitions": native, "image": json.loads((data / "rows" / (cell + ".native.json")).read_text())["image"],
            "binding": {"scope": "saved synthetic fixture"}}
    counts = dict.fromkeys(module["COUNTS"], 0)
    counts["physical_batch"] = 1
    decoded, events = [], []
    vae = torch.nn.Identity()
    def decode(session, **kwargs):
        decoded.append(dict(counts))
        value = vae(session.read("latent"))
        digest = hashlib.sha256(value.float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2, 2, 3), list(digest[:3]), dtype=np.uint8))
    adapter.decode = decode
    output = root / "result"
    with torch.inference_mode():
        summary = module["run"](adapter=adapter, parents=parents, predictions=predictions, references=references,
            output=output, config=module["make_config"](), request={"job_id": "tiny-native-corner", "request_sha256": "fixture",
                "source_hashes": {"flux_native_corner.py": module["sha"](root / "flux_native_corner.py")}},
            source={"scope": "Synthetic small model, no encoder, deterministic2x2 pixel consumer, not actual VAE."},
            counts=counts, parameter_manifest=recipes, retrieval_source=data / "retrieve_weights.py",
            expectations={"scope": "tiny mechanics only, numerical expectations not gates"},
            response_contract={"scope": "tiny paired response R-M, replay M-Nold separate"}, guard_components=(model, vae),
            progress=events.append, production_profile=False)
    return {"module": module, "summary": summary, "output": output, "parents": parents, "predictions": predictions,
        "references": references, "adapter": adapter, "events": events, "decoded": decoded, "root": root}


def test_336_native_four_acts_four_futures_with_both_seals_first(completed):
    x = completed
    assert x["summary"]["counts"] == {**x["module"]["COUNTS"], "model_loads": 0}
    kinds = [e["kind"] for e in x["events"]]
    assert kinds.count("condition_sealed") == 2
    assert kinds.index("cohort_sealed") < kinds.index("native_transition")
    assert [row["adapter_steps"] for row in x["decoded"]] == [84, 168, 252, 336]
    cohort = json.loads((x["output"] / "cohort-seal.json").read_text())
    assert cohort["counts"]["adapter_steps"] == cohort["counts"]["acts"] == 0
    assert set(cohort["target_calls"].values()) == {0}
    assert x["summary"]["target_calls"] == x["module"]["target_counts"](12)
    assert all(v == x["module"]["target_counts"](6) for v in x["summary"]["target_calls_per_condition"].values())


def test_imported_latent_act_preserves_all12_other_slots_and_original_layout(completed):
    x = completed
    for cell in x["module"]["CELLS"]:
        parent = x["parents"][cell]
        for role in ("N1.sham", "P1"):
            name = cell + "." + role
            row = x["summary"]["rows"][name]
            action = row["act"]
            assert action["writes"] == ["latent"] and action["invalidates"] == []
            assert set(action["preconditions"]) == {"latent", "conditioning"}
            entry = LocalStore(x["output"] / "typed-store").load(row["entry"]["cut"])
            expected = parent.payload["latent"] if role == "N1.sham" else x["predictions"][cell]["latent"]
            assert describe(entry.payload) == describe({**parent.payload, "latent": expected})
            assert entry.payload["latent"].stride() == parent.payload["latent"].stride()
            assert row["source_parent"] == parent.fingerprint


def test_metadata_receipts_and_declared28_full_exits_only(completed):
    x = completed
    retained_count = 0
    for name in x["module"]["make_config"]()["branch_order"]:
        rows = [json.loads(line) for line in (x["output"] / (name + ".jsonl")).read_text().splitlines()]
        assert len(rows) == 84
        for position, row in enumerate(rows):
            logical, local = 1 + position // 28, position % 28
            keep = x["module"]["retained"](logical, local)
            assert row["logical_step"] == logical and row["local_index"] == local
            assert row["capture"]["retained_full_cut"] == keep
            assert all((v["archive"] is not None) == keep for v in row["outputs"].values())
            assert all(len(v["witness"]["values"]) <= 64 for v in row["outputs"].values())
            retained_count += keep
    assert retained_count == 28
    ledger = [json.loads(line) for line in (x["output"] / "receipt-emissions.jsonl").read_text().splitlines()]
    assert len(ledger) == 340 and [r["sequence"] for r in ledger] == list(range(340))
    manifest = json.loads((x["output"] / "parameter-artifacts.json").read_text())
    assert len(manifest["artifacts"]) > 0  # Tiny equal constants deduplicate; production requires169.
    assert all(not (x["output"] / r["path"]).exists() for r in manifest["artifacts"])


def test_replay_diagnostics_retained_without_quality_gate(completed):
    for row in completed["summary"]["rows"].values():
        if row["input_role"] == "N1.sham":
            assert len(row["native_replay_diagnostics"]) == 84
            assert all(all(r["field_hash_matches"].values()) for r in row["native_replay_diagnostics"])
            assert row["baseline_png_exact_prior"] is True
        else:
            assert row["native_replay_diagnostics"] == []


def test_changed_latent_is_consumed_without_refresh_and_bad_shape_refused(completed):
    x = completed
    parent = x["parents"]["c1-o0"]
    with torch.inference_mode():
        changed = parent.payload["latent"].clone()
        changed[0, 0, 0] += 1
        session = Session.from_cut(x["adapter"], parent)
        session.apply(x["module"]["latent_act"](parent, changed, "tiny-changed"))
        session.continue_()
        actual = session.capture(retain=False)
        assert torch.equal(actual.payload["latent"], changed)
        original = Session.from_cut(x["adapter"], parent)
        original.continue_()
        baseline = original.capture(retain=False)
        assert not torch.equal(actual.payload["image"], baseline.payload["image"])
        assert torch.equal(actual.payload["text"], baseline.payload["text"])
        assert describe(actual.payload["context"]) == describe(baseline.payload["context"])
        with pytest.raises(ValueError, match="shape/dtype"):
            x["module"]["latent_act"](parent, changed[:, :1], "tiny-bad-shape")


def test_strict_fixed_census_and_no_late_clock_admission(completed):
    m = completed["module"]
    config = m["make_config"]()
    m["validate_config"](config)
    changed = copy.deepcopy(config)
    changed["counts"]["adapter_steps"] = 335
    with pytest.raises(ValueError, match="finite contract"):
        m["validate_config"](changed)
    final = next(iter(completed["summary"]["rows"].values()))["final"]["cut"]
    cut = LocalStore(completed["output"] / "typed-store").load(final)
    with pytest.raises(ValueError, match="native N1"):
        m["latent_act"](cut, cut.payload["latent"], "wrong-clock")


def test_noncontiguous_source_layout_with_contiguous_prediction(completed):
    x = completed
    parent = x["parents"]["c1-o0"]
    with torch.inference_mode():
        original = parent.payload["latent"]
        reference = original.transpose(1, 2).contiguous().transpose(1, 2)
        assert reference.shape == original.shape and reference.stride() != original.stride()
        synthetic = Session(x["adapter"], {**parent.payload, "latent": reference}).capture(retain=False)
        candidate = x["predictions"]["c1-o0"]["latent"].contiguous()
        action = x["module"]["latent_act"](synthetic, candidate, "hostile-layout")
        session = Session.from_cut(x["adapter"], synthetic)
        session.apply(action)
        entry = session.capture(retain=False)
        assert entry.payload["latent"].stride() == reference.stride()
        assert torch.equal(entry.payload["latent"], candidate)
        assert describe(entry.payload) == describe({**synthetic.payload, "latent": action.parameters["value"]})
