"""Framework-free custody, formula and countertrend mechanics; no model runs."""
from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "qualification"
A = runpy.run_path(str(HERE / "analyze_flux_single_contexts.py"))


def sources():
    zero = {key: np.zeros((1, 4, 1, 2), dtype=np.float32) for key in ("q", "k", "v")}
    zero["v"][:] = np.arange(8, dtype=np.float32).reshape(1, 4, 1, 2)
    text = {key: value.copy() for key, value in zero.items()}
    text["k"][:, :2] = 7
    text["v"][:, :2] += 100
    return zero, text


def test_offline_formula_uniform_queries_select_only_text_bank():
    zero, text = sources()
    result = A["recompute_attention"](zero, text, text_rows=2, query_indices=[0, 1])
    for name, offset in (("K", 0), ("V", 50), ("KV", 50)):
        expected = np.array([3., 4.]) + offset
        np.testing.assert_array_equal(result[name]["attention"], np.broadcast_to(expected, (1, 2, 1, 2)))
        np.testing.assert_array_equal(result[name]["text_mass"], np.full((1, 1, 2), .5))
    assert zero["v"][0, 0, 0, 0] == 0  # no operand mutation
    text["v"][:, 2:] = 10000
    changed = A["recompute_attention"](zero, text, text_rows=2, query_indices=[0, 1])
    for name in result:
        np.testing.assert_array_equal(changed[name]["attention"], result[name]["attention"])


@pytest.mark.parametrize("change", ["query", "partition", "shape", "nonfinite", "inventory"])
def test_formula_refuses_invalid_operand_contract(change):
    zero, text = sources()
    indices, rows = [0, 1], 2
    if change == "query":
        indices = [0, 0]
    elif change == "partition":
        rows = 4
    elif change == "shape":
        text["v"] = text["v"][:, :3]
    elif change == "nonfinite":
        zero["q"][0, 0, 0, 0] = np.nan
    else:
        del text["q"]
    with pytest.raises(ValueError):
        A["recompute_attention"](zero, text, text_rows=rows, query_indices=indices)


def test_formula_disagreement_is_retained_without_numerical_or_semantic_gate():
    zero, text = sources()
    q = runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py"))["quantize_bf16"]
    rebuilt = A["recompute_attention"](zero, text, text_rows=2, query_indices=[0])
    prediction = {"instrument_status": "applicable", "query_indices": [0], "cases": {
        name: {"attention": row["attention"] + 9, "attention_bf16": q((row["attention"] + 9).astype(np.float32)),
               "text_mass": row["text_mass"]} for name, row in rebuilt.items()}}
    result = A["forecast_math"](prediction, {"zero": zero, "text": text}, text_rows=2, quantize=q)
    assert result["instrument_status"] == "applicable"
    assert result["cases"]["K"]["saved_fp64_minus_offline_reference"]["signed_mean"] == 9
    assert not result["cases"]["K"]["saved_bf16_reference_bit_exact_offline"]
    assert A["forecast_math"]({"instrument_status": "not-applicable"}, {}, text_rows=2,
                              quantize=q)["instrument_status"] == "not-applicable"


def test_endpoint_direction_retains_negative_zero_and_observable_disagreement():
    values = {"text.text": (2, 4), "K.image": (3, 3), "V.image": (1, 4), "KV.image": (2, 5)}
    comparisons = {name: {"text": {"latent": {"base_l2": a}, "rgb": {"base_mae": b}}}
                   for name, (a, b) in values.items()}
    result = A["text_direction"](comparisons)
    assert result["latent_l2"]["K.image"]["signed_margin"] == 1
    assert result["rgb_mae"]["K.image"]["signed_margin"] == -1
    assert result["latent_l2"]["V.image"]["direction"] == "miss"
    assert result["rgb_mae"]["V.image"]["direction"] == "tie"
    assert result["latent_l2"]["KV.image"]["direction"] == "tie"
    assert result["disagreements"] == ["K.image", "V.image", "KV.image"]
    assert result["terminal_status"] == "unassessed"


def census_fixture():
    initial = {"model_loads": 1, "prompt_encodings": 2, "noise_draws": 0, "adapter_steps": 0,
        "acts": 0, "candidate_calls": 0, "images": 0, "vae_calls": 0, "receipt_emissions": 0, "physical_batch": 1}
    current, rows, sources = dict(initial), {}, {}
    for context_id in A["CONTEXT_IDS"]:
        current["noise_draws"] += 1
        sources[context_id] = {"parent_counts_before_builder": dict(current)}
        builder = {"adapter_steps": 96, "acts": 3, "receipt_emissions": 99}
        for key, value in builder.items():
            current[key] += value
        sources[context_id]["parent_counts_after_builder"] = dict(current)
        panel = dict(A["PANEL_COUNTS"])
        for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls", "receipt_emissions"):
            current[key] += panel[key]
        rows[context_id] = {"builder_counts": builder, "panel_counts": panel, "counts_after": dict(current)}
    return rows, initial, current, sources


def test_physical_census_keeps_one_parent_load_and_actual_offsets():
    args = census_fixture()
    result = A["cumulative_census"](*args)
    assert result["global"]["receipt_emissions"] == 900
    assert result["per_context"]["total_native_steps"] == 287
    assert result["per_context"]["total_acts"] == 13
    assert result["global"]["model_loads"] == 1
    args[0]["p1-s47"]["panel_counts"]["model_loads"] = 1
    with pytest.raises(ValueError, match="per-panel census"):
        A["cumulative_census"](*args)


def test_census_rejects_unrecorded_work_between_builder_and_panel():
    args = census_fixture()
    args[3]["p1-s53"]["parent_counts_before_builder"]["candidate_calls"] += 1
    with pytest.raises(ValueError, match="starting offset"):
        A["cumulative_census"](*args)


def test_builder_act_rejects_extra_slot_write_and_wrong_parent():
    before, after = {"conditioning": "own", "image": "image"}, {"conditioning": "zero", "image": "image"}
    point = {"logical_step": 2, "phase": "project", "index": None}
    act = {"writes": ["conditioning"], "operation": {"id": "saturn.builtin.replace"},
           "parameters": {"value": "zero"}}
    receipt = {"fingerprint": "r", "operation": "apply", "act": act, "parent": "a", "result": "b",
               "writes": {"conditioning": "zero"}}
    action = {"slot": "conditioning", "undeclared_slots_exact": True, "receipt": receipt,
              "receipt_id": "r", "parent": "a", "post_act": "b", "act": act, "point": point}
    # Descriptors retain stride as in actual typed pages.
    descriptors = {key: {"value": key, "stride": [1]} for key in ("own", "zero", "image")}
    before = {key: descriptors[value] for key, value in before.items()}
    after = {key: descriptors[value] for key, value in after.items()}
    act["parameters"]["value"] = descriptors["zero"]
    receipt["writes"]["conditioning"] = descriptors["zero"]
    expected = np.zeros((1,), dtype=np.float32)
    e = SimpleNamespace(receipts={"r": receipt}, fields=lambda cut: before if cut == "a" else after,
        typed={"unpack": lambda value: value}, point=lambda _: point, slot=lambda *_: expected)
    assert A["builder_act"](e, action, "a", expected, slot="conditioning") == "b"
    after["image"] = descriptors["own"]
    with pytest.raises(ValueError, match="undeclared state"):
        A["builder_act"](e, action, "a", expected, slot="conditioning")
    with pytest.raises(ValueError, match="source differs"):
        A["builder_act"](e, action, "other-parent", expected, slot="conditioning")


def test_builder_native_path_uses_actual_receipts_and_requires_exact_census():
    p0 = {"next_operation": "project"}
    p1 = {"next_operation": "joint"}
    receipt = {"fingerprint": "r", "parent": "a", "result": "b", "operation": "continue", "steps": 1,
               "transitions": [{"before": p0, "after": p1,
                   "native_operation": {"footprint": "declared", "operation": "project"}}]}
    e = SimpleNamespace(receipts={"r": receipt}, point=lambda cut: p0 if cut == "a" else p1)
    assert A["builder_path"](e, [receipt], "a", "b", expected_steps=1) == ["r"]
    with pytest.raises(ValueError, match="step count"):
        A["builder_path"](e, [receipt], "a", "b", expected_steps=2)
    with pytest.raises(ValueError, match="different cut"):
        A["builder_path"](e, [receipt], "a", "c", expected_steps=1)


def test_parent_custody_is_called_once_before_any_context_array(tmp_path):
    h = runpy.run_path(str(HERE / "summarize_context_window.py"))
    write = lambda path, value: (path.parent.mkdir(parents=True, exist_ok=True), path.write_text(json.dumps(value)))
    contexts = [{"id": key, "prompt_id": prompt, "seed": seed} for key, prompt, seed in
                (("p0-s53", "p0", 53), ("p1-s47", "p1", 47), ("p1-s53", "p1", 53))]
    plan = {"contexts": contexts, "expected_checkpoint_sha256": h["sha"](h["canonical"]({}))}
    required = ("flux_single_contexts_campaign.py", "flux_single_contexts.py", "flux_single_kv.py",
                "flux_family_custody.py", "plan.json", "preflight.v1.json")
    request = {"config": plan, "source_hashes": {key: "source-hash" for key in required}, "request_sha256": "request"}
    source = {"job_id": "parent", "request_sha256": "request", "source_hashes": request["source_hashes"],
              "model_identity": "model", "checkpoint_sha256": plan["expected_checkpoint_sha256"],
              "checkpoint_files": {}, "contexts": contexts}
    data = tmp_path / "extracted/flux-single-contexts-result"
    write(data / "plan.json", plan)
    write(data / "source.json", source)
    summary = {"source_sha256": A["sha_file"](data / "source.json"), "job_id": "parent", "request_sha256": "request",
               "model_identity": "model", "contexts": {key: {} for key in A["CONTEXT_IDS"]}}
    write(data / "summary.json", summary)
    write(data / "plan-frozen.json", {"plan_sha256": A["sha_file"](data / "plan.json"), "request_sha256": "request",
        "job_id": "parent", "source_identity": h["sha"](h["canonical"](request["source_hashes"])),
        "work": {key: 0 for key in ("model_loads", "prompt_encodings", "noise_draws", "adapter_steps", "candidate_calls")}})
    write(tmp_path / "request.json", request)
    write(tmp_path / "job.json", {"job_id": "parent", "state": "succeeded"})
    write(tmp_path / "report.json", {"job_id": "parent", "request_sha256": "request",
        "archive_root": "flux-single-contexts-result", "report": summary})
    write(tmp_path / "payload/plan.json", plan)
    events = [{"kind": "prompt_encoded", "prompt_id": name} for name in ("p0", "p1")]
    for key in A["CONTEXT_IDS"]:
        events += [{"kind": "context_start", "context_id": key}, {"kind": "context_complete", "context_id": key}]
    (data / "timeline.jsonl").write_text("\n".join(json.dumps({"sequence": i, **event}) for i, event in enumerate(events)))
    calls = []
    def custody(root, *, limits):
        calls.append(root)
        assert not list(data.rglob("*.npy"))
        return {"mechanics_status": "verified", "job_id": "parent", "request_sha256": "request"}
    result = A["parent_run"](tmp_path, custody_verify=custody)
    assert calls == [tmp_path.resolve()]
    assert result["job"]["job_id"] == "parent"  # no synthetic child job/report
    source["job_id"] = "invented-child"
    write(data / "source.json", source)
    with pytest.raises(ValueError, match="parent plan/source"):
        A["parent_run"](tmp_path, custody_verify=custody)


def test_event_lines_are_bounded_and_auditor_imports_without_frameworks(tmp_path):
    event = tmp_path / "timeline.jsonl"
    event.write_bytes(b'\n{"sequence":0}\n')
    assert A["read_jsonl"](event) == [{"sequence": 0}]
    event.write_bytes(b"x" * 101)
    with pytest.raises(ValueError, match="line exceeds"):
        A["read_jsonl"](event, max_line_bytes=100)
    script = '''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','diffusers','transformers','saturn','common'}:raise RuntimeError(fullname)
sys.meta_path.insert(0,Block())
runpy.run_path(sys.argv[1])
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(HERE / "analyze_flux_single_contexts.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
