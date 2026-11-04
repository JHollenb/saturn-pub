"""Tiny native bounded chains, both-stream installs and complete bridge custody."""
from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from saturn_pub.store import LocalStore

from mdb.forecast_transaction import ForecastTransaction, verify_transaction
from mdb.forward_recording import tensor_descriptor
from mdb.tensor_archive import load_tensor_archive, write_tensor_archive

HERE = Path(__file__).parents[1]


def plan(campaign, indices=None):
    declared = campaign["contract"]({} if indices is None else {"block_indices": indices})
    return {"context": declared["context"], "height": 512, "width": 512, "steps": 4, "block_indices": declared["block_indices"],
        "selector": {"logical_step": 3, "phase": "single", "index": 6}, "branches": declared["branches"],
        "branch_execution_order": declared["order"], "observation_plan": declared["observation_plan"],
        "sample_indices": campaign["SAMPLE_INDICES"], "query_chunk": 64, "projection_chunk": 256,
        "neuron_chunk": 256, "retain_fields": False, "array_storage": "bf16-bits", "archive_integer_ids": "native-int64",
        "cpu_threads": 4, "max_wall_seconds": 900, "archive_compression_level": 1,
        **{"max_" + key: declared["counts"][key] for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls")}}


@pytest.mark.parametrize(("indices", "corruption"), [([6, 7], None), ([6, 7], "text"),
    ([6, 7, 8], None), ([6, 7, 8], "text"), ([6, 7, 8], "image"), ([6, 7, 8], "join")])
def test_complete_sealed_chain_drives_both_streams_and_native_futures(tmp_path, monkeypatch, indices, corruption):
    monkeypatch.chdir(tmp_path)
    for name in ("flux_single_contexts.py", "flux_single_kv.py", "flux_single_mlp_campaign.py",
            "flux_single_contraction.py", "flux_mlp_replacement.py", "flux_attention_replacement_campaign.py",
            "flux_transition_reference.py"):
        shutil.copyfile(HERE / "qualification" / name, name)
    tiny = runpy.run_path(str(HERE / "tests/test_flux_single_contexts.py"))
    campaign = runpy.run_path(str(HERE / "qualification/flux_transition_chain_campaign.py"))
    helper = runpy.run_path("flux_transition_reference.py")
    util = runpy.run_path("flux_single_mlp_campaign.py")
    previous = runpy.run_path("flux_attention_replacement_campaign.py")
    monkeypatch.setitem(campaign["predict_cases"].__globals__, "SAMPLE_INDICES", [0, 2, 3, 6])
    model, adapter, session = tiny["tiny_session"]()
    cuts, builder = runpy.run_path("flux_single_contexts.py")["build_sources"](adapter=adapter, session=session,
        output=tmp_path / "builder", context={"id": "tiny-chain79"})
    cuts, preparation = util["prepare_sources"](adapter=adapter, cuts=cuts, output=tmp_path / "preparation",
        builder_ref={"path": "builder/builder.json"}, wall=lambda: None)
    assert builder["counts"]["adapter_steps"] + preparation["counts"]["adapter_steps"] == 108
    output = tmp_path / "result"
    output.mkdir()
    arrays = {}
    def array(value):
        descriptor = tensor_descriptor(value)
        key = hashlib.sha256(json.dumps({name: descriptor[name] for name in ("dtype", "shape", "sha256")}, sort_keys=True).encode()).hexdigest()
        if key not in arrays:
            arrays[key] = write_tensor_archive(output / "arrays" / (key + ".npy"), value, root=output)
        return arrays[key]
    # The test's final consumer is synthetic. Native transformer, scheduler,
    # StateCut, Act and receipt mechanics are exercised without a real VAE.
    def consumer(state, **kwargs):
        digest = hashlib.sha256(state.read("latent").float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2, 2, 3), list(digest[:3]), dtype=np.uint8))
    adapter.decode = consumer
    counts = {key: 0 for key in campaign["COUNTS"]}
    counts.update(adapter_steps=108, acts=4, receipt_emissions=112, physical_batch=1)
    config = plan(campaign, indices)
    declared = campaign["contract"](config)
    suffix, last = declared["suffix_steps"], indices[-1]
    common_key = "common_post" + str(last)
    config.update(query_chunk=2, projection_chunk=5, neuron_chunk=5)
    binding, operands = campaign["bind_inputs"]({index: model.single_transformer_blocks[index] for index in indices},
        cuts, array=array, counts=counts, builder_ref={"path": "builder/builder.json"},
        preparation_ref={"path": "preparation/preparation.json"}, config=config, profile=previous["attention_profile"])
    util["save"](output / "binding.json", binding)
    released, calls = {"native": False}, dict.fromkeys(indices, 0)
    def guard(module, args):
        assert released["native"], "native current target called before complete chain seal"
        for index in calls:
            if module is model.single_transformer_blocks[index]:
                calls[index] += 1
    handles = [model.single_transformer_blocks[index].register_forward_pre_hook(guard) for index in calls]
    try:
        with ForecastTransaction(output / "transaction", model_identity=adapter.model_identity,
                input_identity="tiny-pre6", source_identity="tiny-whole-chain", frontier={"kind": "architectural"}) as tx:
            tx.frontier("source_binding", binding)
            tx.predict("transition_chain", lambda _: {"predictions": campaign["predict_cases"](operands, helper=helper, config=config)})
            predictions = tx.prediction("transition_chain")["cases"]
            for role in predictions:
                chain = predictions[role]
                for position in range(1, len(indices)):
                    assert chain["steps"][position]["input_source"] == "previous predicted full output"
                    assert chain["steps"][position]["input_descriptor"] == chain["steps"][position-1]["prediction"]["output_descriptor"]
            assert calls == dict.fromkeys(indices, 0)
            if corruption:
                # A low-bit corruption must be refused at the consumer join,
                # rather than silently cast into another payload after seal.
                steps = [dict(step) for step in predictions["zero"]["steps"]]
                if corruption == "join":
                    steps[-1]["input_descriptor"] = {**steps[-1]["input_descriptor"], "bf16_le_sha256": "0"*64}
                else:
                    bad = dict(steps[0]["prediction"])
                    bad["output"] = bad["output"].copy()
                    bad["output"].view(np.uint32)[0, 3 if corruption == "image" else 0, 0] |= np.uint32(1)
                    steps[0]["prediction"] = bad
                fake = {**predictions, "zero": {**predictions["zero"], "steps": steps}}
            seal = tx.seal()
            released["native"] = True
            def execute(context):
                return campaign["run_panel"](adapter=adapter, cuts=cuts, output=output, config=config,
                    store=LocalStore(output / "typed-store"), array=array, wall=lambda: None,
                    event=lambda *args, **kwargs: None, counts=counts,
                    predictions=fake if corruption else context.predictions["transition_chain"]["cases"])
            if corruption:
                with pytest.raises(ValueError, match="own-state" if corruption == "join" else "exact BF16 widenings"):
                    tx.execute("native", execute)
                assert len(list((output / "rows").glob("*.json"))) == 3
                return
            tx.execute("native", execute)
    finally:
        for handle in handles:
            handle.remove()
    assert verify_transaction(tx.root, expected_seal_sha256=seal["seal_sha256"])["verified"]
    assert calls == {int(key): value for key, value in declared["target_calls"].items()}
    assert counts == {**declared["counts"], "model_loads": 0, "prompt_encodings": 0, "noise_draws": 0}
    result = tx.observation("native")
    assert all(result["control_observations"].values())
    rows = {row["id"]: row for row in result["rows"]}
    assert list(rows) == declared["order"] and len(list(output.glob("*.png"))) == len(declared["branches"])
    prefixes = json.loads((output / "native-boundaries.json").read_text())["prefixes"]
    store = LocalStore(output / "typed-store")
    for name, row in rows.items():
        assert len(row["continuation_receipts"]) == suffix
        assert row[common_key]["point"]["index"] == row[common_key]["clock_state"]["cursor"] == last+1
        assert row["parent"] == prefixes[row["base"]]["post"+str(row["installation_boundary"])]["cut"]
        if row["mode"] == "native":
            assert not row["acts"] and not row["bridge_receipts"]
            continue
        assert [action["slot"] for action in row["acts"]] == ["image", "text"]
        assert row["acts"][1]["parent"] == row["acts"][0]["post_act"]
        assert len(row["bridge_receipts"]) == last-row["installation_boundary"]
        if len(indices) == 3:
            assert len(row["bridge_boundaries"]) == len(row["bridge_receipts"])
            for position, (bridge, receipt) in enumerate(zip(row["bridge_boundaries"], row["bridge_receipts"], strict=True)):
                assert bridge["cut"] == receipt["result"]
                assert bridge["point"]["index"] == row["installation_boundary"]+position+2
                saved = store.load(bridge["cut"])
                for slot in ("image", "text"):
                    np.testing.assert_array_equal(load_tensor_archive(output, util["json_value"](bridge[slot])), saved.payload[slot].float().numpy())
            if row["bridge_boundaries"]:
                assert util["json_value"](row["bridge_boundaries"][-1]) == util["json_value"](row[common_key])
        else:
            assert "bridge_boundaries" not in row
        if row["mode"] == "sham":
            continue
        predicted = predictions[row["base"]]["steps"][row["installation_boundary"] - 6]["prediction"]["output"]
        installed = store.load(row["installed"]["cut"])
        np.testing.assert_array_equal(installed.payload["text"].float().numpy(), predicted[:, :3])
        np.testing.assert_array_equal(installed.payload["image"].float().numpy(), predicted[:, 3:])
        original = store.load(row["parent"])
        for slot in ("image", "text"):
            assert installed.payload[slot].stride() == original.payload[slot].stride()
        assert util["identity"](row["installed"]["non_stream_state"]) == util["identity"](binding["inputs"][row["base"]]["non_stream_state"])
        common = row[common_key]
        assert load_tensor_archive(output, util["json_value"](common["image"])).shape == (1, 4, 8)
    emitted = [json.loads(line) for line in (output / "receipt-emissions.jsonl").read_text().splitlines()]
    assert [item["sequence"] for item in emitted] == list(range(declared["panel_counts"]["receipt_emissions"]))
    expected = ["prefix.zero"] * len(indices) + ["prefix.text"] * len(indices)
    for row in result["rows"]:
        expected += [row["id"]] * (suffix + len(row["acts"]) + len(row["bridge_receipts"]))
    assert [item["branch"] for item in emitted] == expected
    observed = json.loads((output / "continuation-observations.json").read_text())
    assert set(observed["rows"]) == set(declared["order"])
    for name, boundaries in observed["rows"].items():
        assert len(boundaries) == suffix+1
        for index, boundary in enumerate(boundaries):
            assert boundary["complete_cut_retained"] == (index in {0, suffix})
            if index == 0:
                assert boundary["cut"] == rows[name][common_key]["cut"]
            if index == suffix:
                assert boundary["slots"]["latent"]["meaning"]["phase"] == "committed-Z4"
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("indices", [[6, 7], [6, 7, 8]])
def test_frozen_chain_plan_refuses_teacher_forcing_clock_and_budget_drift(indices):
    campaign = runpy.run_path(str(HERE / "qualification/flux_transition_chain_campaign.py"))
    config = plan(campaign, indices)
    campaign["validate_config"](config)
    for changes in ({"block_indices": [7, 6]}, {"selector": {"logical_step": 3, "phase": "single", "index": 7}},
            {"max_adapter_steps": 213}, {"branch_execution_order": list(reversed(config["branch_execution_order"]))},
            {"retain_fields": True}, {"cpu_threads": True}):
        with pytest.raises(ValueError):
            campaign["validate_config"]({**config, **changes})


def test_pure_contract_preserves_defaults_and_declares_finite_three_block_variant():
    campaign = runpy.run_path(str(HERE / "qualification/flux_transition_chain_campaign.py"))
    legacy = campaign["contract"]({})
    assert legacy["context"] == campaign["CONTEXT"]
    assert legacy["branches"] == campaign["BRANCHES"]
    assert legacy["order"] == campaign["ORDER"]
    assert legacy["counts"] == campaign["COUNTS"]
    assert legacy["observation_plan"] == campaign["OBSERVATION_PLAN"]
    fresh = campaign["contract"]({"block_indices": [6, 7, 8]})
    assert fresh["context"]["seed"] == 83 and fresh["context"]["id"] == "p1-s83"
    assert fresh["counts"] == {**campaign["COUNTS"], "adapter_steps": 237, "acts": 18,
        "receipt_emissions": 255, "images": 9, "vae_calls": 9}
    assert fresh["panel_counts"] == {"adapter_steps": 129, "acts": 14, "receipt_emissions": 143,
        "candidate_calls": 0, "images": 9, "vae_calls": 9}
    assert fresh["target_calls"] == {"6": 2, "7": 4, "8": 6}
    assert fresh["suffix_steps"] == 13 and fresh["observation_plan"]["boundaries_per_branch"] == 14
    fresh["branches"][0]["mode"] = "corrupted"
    legacy["observation_plan"]["slots"].clear()
    assert campaign["contract"]({})["branches"][0]["mode"] == "native"
    assert campaign["contract"]({})["observation_plan"]["slots"]
    for invalid in ([6], [6, 8], [6, 7, 8, 9], (6, 7), [6, True]):
        with pytest.raises(ValueError):
            campaign["contract"]({"block_indices": invalid})
