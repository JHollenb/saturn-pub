"""Retrospective saved-fixture mechanics only; no native/model construction."""
from __future__ import annotations

import copy
import json
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1]


def module():
    return runpy.run_path(str(HERE / "qualification/flux_matched_pass_replay.py"))


def test_signed_vector_identity_keeps_negative_cross_term_and_common_denominator():
    m = module()
    n, q, p = [np.array(x, dtype=np.float32) for x in ([10., 10.], [12., 8.], [11., 9.])]
    result = m["vector_account"](q, p, n)
    assert result["dot_A_B"] == -4 and result["twice_dot_A_B"] == -8
    assert result["A_Q_minus_N"]["squared_l2"] == 8
    assert result["B_P_minus_Q"]["squared_l2"] == result["E_P_minus_N"]["squared_l2"] == 2
    assert result["squared_norm_identity"]["residual"] == result["vector_identity_max_abs"] == 0
    assert [result[key]["native_rms"] for key in ("A_Q_minus_N", "B_P_minus_Q", "E_P_minus_N")] == [10., 10., 10.]
    assert result["B_P_minus_Q"]["native_relative_rms"] == .1
    zero = m["vector_account"](q, p, np.zeros_like(n))
    assert zero["E_P_minus_N"]["native_relative_rms"] is None


def test_no_shape_or_nonfinite_account_and_no_future_activation_read():
    m = module()
    with pytest.raises(ValueError, match="operands"):
        m["vector_account"](np.ones(3), np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="operands"):
        m["vector_account"](np.array([np.nan]), np.ones(1), np.ones(1))
    class E:
        def __init__(self):
            self.reads = []
        def array(self, record):
            self.reads.append(record)
            return np.ones(1, dtype=np.float32)
    evidence = E()
    reader = m["InputOnlyReader"](evidence, [{"path": "actual-N1"}])
    reader.array({"path": "actual-N1"})
    reader.array({"path": "bank", "storage": "checkpoint-recipe-v1"})
    with pytest.raises(ValueError, match="future activation"):
        reader.array({"path": "native-pass1-target"})
    with pytest.raises(ValueError, match="future activation"):
        reader.array({"path": "actual-N1", "descriptor": "different"})
    assert len(evidence.reads) == 2


def test_archive_identity_includes_shape_not_just_flat_value_bytes(tmp_path):
    m = module()
    a = runpy.run_path(str(HERE / "qualification/analyze_flux_prompt_factorial.py"))
    (tmp_path / "arrays").mkdir()
    values = np.arange(4, dtype=np.float32)
    flat = m["archive"](tmp_path, values, a["describe_field"](values))
    square = m["archive"](tmp_path, values.reshape(2, 2), a["describe_field"](values.reshape(2, 2)))
    assert flat["descriptor"]["fp32_le_sha256"] == square["descriptor"]["fp32_le_sha256"]
    assert flat["path"] != square["path"]
    assert np.array_equal(m["load_q"](tmp_path, flat, a["describe_field"]), values)
    assert np.array_equal(m["load_q"](tmp_path, square, a["describe_field"]), values.reshape(2, 2))


@pytest.fixture(scope="module")
def saved_replay(tmp_path_factory):
    data, cache = (os.environ.get(name) for name in ("MDB_FACTORIAL_TINY_DATA", "MDB_FACTORIAL_TINY_PARAMETER_ROOT"))
    if not data or not cache:
        pytest.skip("Requires explicit prior qualified saved tiny fixture, never constructs a model")
    m = module()
    output = tmp_path_factory.mktemp("matched-pass") / "output"
    events = []
    result = m["run"](Path(data), analyzer_source=HERE / "qualification/analyze_flux_prompt_factorial.py",
        memoization_source=HERE / "qualification/memoize_flux_audit_parameters.py", parameter_root=Path(cache),
        output=output, max_decoded_bytes=8 << 20, max_retained_bytes=8 << 20, production_profile=False,
        progress=lambda row: events.append(row), custody={"scope": "explicit synthetic saved fixture only"})
    return {"m": m, "result": result, "output": output, "events": events, "data": Path(data), "cache": Path(cache)}


def test_real_saved_fixture_56_independent_operations_and_all_own_state_joins(saved_replay):
    x = saved_replay
    assert x["result"]["math_transitions"] == len(x["events"]) == 56
    assert x["result"]["native_forwards"] == 0
    a = runpy.run_path(str(HERE / "qualification/analyze_flux_prompt_factorial.py"))
    binding = json.loads((x["data"] / "binding.json").read_text())
    for cell in ("c1-o0", "c0-o1"):
        q = json.loads((x["output"] / (cell + ".Q.json")).read_text())
        source = q["origin"]
        assert {key: source["point"][key] for key in ("logical_step", "phase", "index")} == {
            "logical_step": 1, "phase": "project", "index": None}
        assert q["chain_length"] == len(q["transitions"]) == 28
        assert q["transitions"][0]["inputs"]["latent"] == a["describe_field"](
            a["independent"]()["Evidence"](x["data"], x["cache"], {}).array(source["inputs"]["latent"]))
        for key in ("conditioning", "txt_ids", "img_ids"):
            assert source["inputs"][key] == binding["origins"][cell]["inputs"][key]
        previous = None
        for row in q["transitions"]:
            local = row["local_index"]
            assert (row["kind"], row["block_index"]) == a["kind_at"](local)
            if 1 <= local <= 26:
                assert row["inputs"]["image"] == previous["image"]["descriptor"]
                if local <= 25:
                    assert row["inputs"]["text"] == previous["text"]["descriptor"]
            if local == 27:
                assert row["inputs"]["noise"] == previous["noise"]["descriptor"]
                assert row["inputs"]["latent"] == q["transitions"][0]["inputs"]["latent"]
            assert all((v["archive"] is not None) == (local in x["m"]["LANDMARKS"]) for v in row["outputs"].values())
            previous = row["outputs"]
    assert x["result"]["parameter_read_cache"]["verified"]
    assert x["result"]["parameter_read_cache"]["verification_runs"] == 1
    assert x["result"]["parameter_read_cache"]["storage_codec"] == "bf16-bits"


def test_q_seal_precedes_target_results_and_all_full_signed_accounts(saved_replay):
    x = saved_replay
    seal = json.loads((x["output"] / "Q-seal.json").read_text())
    assert seal["target_arrays_read_before_Q_seal"] == 0
    assert seal["kind"] == "retrospective-completed-Q-custody-seal"
    assert x["result"]["Q_seal_sha256"] == x["m"]["sha"](x["output"] / "Q-seal.json")
    count = 0
    for cell, group in x["result"]["comparisons"].items():
        assert [row["local_index"] for row in group["rows"]] == [0, 5, 25, 26, 27]
        for row in group["rows"]:
            for field in row["fields"].values():
                account = field["account"]
                assert account["vector_identity_max_abs"] == 0
                assert abs(account["squared_norm_identity"]["residual"]) < 1e-12
                assert account["A_Q_minus_N"]["native_rms"] == account["B_P_minus_Q"]["native_rms"] == account["E_P_minus_N"]["native_rms"]
                count += 1
        assert seal["programs"][cell]["sha256"] == x["m"]["sha"](x["output"] / (cell + ".Q.json"))
    assert count == 28


def test_stale_pass_schedule_refused_before_equations(saved_replay, tmp_path):
    x = saved_replay
    a = runpy.run_path(str(HERE / "qualification/analyze_flux_prompt_factorial.py"))
    old = a["independent"]()
    recipes = {row["path"]: row for row in json.loads((x["data"] / "parameter-artifacts.json").read_text())["artifacts"]}
    evidence = old["Evidence"](x["data"], x["cache"], recipes)
    binding = json.loads((x["data"] / "binding.json").read_text())
    origin, _ = x["m"]["selected_native_origin"](a, old, evidence, binding, x["data"], "c1-o0")
    binding = copy.deepcopy(binding)
    binding["pass_schedules"]["1"] = binding["pass_schedules"]["2"]
    allowed = list(origin["inputs"].values()) + [binding["pass_schedules"]["1"][name] for name in ("timestep", "sigma", "sigma_next")]
    def forbidden(*args, **kwargs):
        raise AssertionError("Equation invoked before source clock check")
    with pytest.raises(ValueError, match="clock/schedule"):
        x["m"]["replay_pass"](x["m"]["InputOnlyReader"](evidence, allowed), binding, origin,
            two={"independent_pass_events": forbidden}, describe=a["describe_field"], context_fields=a["context_fields"],
            kind_at=a["kind_at"], output=tmp_path)


def test_wrong_frozen_source_refused_before_output(saved_replay, tmp_path):
    x = saved_replay
    with pytest.raises(ValueError, match="source binding"):
        x["m"]["run"](x["data"], analyzer_source=HERE / "qualification/analyze_flux_prompt_factorial.py",
            memoization_source=HERE / "qualification/memoize_flux_audit_parameters.py", parameter_root=x["cache"],
            output=tmp_path / "never-created", expected_analyzer_sha256="0" * 64)
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("change", ["future-latent", "clock", "conditioning"])
def test_native_N1_source_cannot_borrow_later_state_or_other_context(saved_replay, tmp_path, change):
    x = saved_replay
    a = runpy.run_path(str(HERE / "qualification/analyze_flux_prompt_factorial.py"))
    old = a["independent"]()
    recipes = {row["path"]: row for row in json.loads((x["data"] / "parameter-artifacts.json").read_text())["artifacts"]}
    evidence = old["Evidence"](x["data"], x["cache"], recipes)
    binding = json.loads((x["data"] / "binding.json").read_text())
    lines = [json.loads(line) for line in (x["data"] / "native/c1-o0.jsonl").read_text().splitlines()]
    if change == "future-latent":
        lines[27]["outputs"]["latent"]["archive"] = lines[55]["outputs"]["latent"]["archive"]
    elif change == "clock":
        lines[27]["capture"]["point"]["logical_step"] = 2
    else:
        binding["origins"]["c1-o0"]["inputs"]["conditioning"] = binding["origins"]["c0-o1"]["inputs"]["conditioning"]
    (tmp_path / "native").mkdir()
    (tmp_path / "native/c1-o0.jsonl").write_text("\n".join(json.dumps(row) for row in lines) + "\n")
    with pytest.raises(ValueError):
        x["m"]["selected_native_origin"](a, old, evidence, binding, tmp_path, "c1-o0")
