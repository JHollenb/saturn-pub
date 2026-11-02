"""Tiny mechanics only; no semantic model qualification or training benchmark."""

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from runtime import initialize_runtime, language_observation

from saturn_pub.training import TrainingIntervalError


def scores(value):
    return {
        "validation": {"bpb": value},
        "autonomous_rollout": {"entropy": 1.0},
        "trace_endpoints": {"carrier_rms": 1.0},
    }


def make_runtime(tmp_path):
    torch.manual_seed(12)
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(0.1))
    model.register_buffer("scratch", torch.ones(2), persistent=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    cursor = {"step": 0, "sampler_position": 0}
    generator = torch.Generator().manual_seed(31)
    cache = {"value": torch.ones(2)}
    runtime = initialize_runtime(
        model,
        optimizer,
        cursor,
        tmp_path / "runtime",
        "tiny",
        generators={"sampler": generator},
        mutable={
            "cache": (
                lambda: cache["value"].clone(),
                lambda state: cache.update(value=state.clone()),
            )
        },
    )
    return runtime, model, optimizer, cursor, generator, cache


def test_rejected_interval_restores_every_declared_state_and_archives(tmp_path):
    runtime, model, optimizer, cursor, generator, cache = make_runtime(tmp_path)

    def step(_):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.defaults["lr"] = 0.3
        model.scratch.add_(1)
        cursor["step"] += 1
        cursor["sampler_position"] += 3
        cache["value"].add_(5)
        torch.rand(2, generator=generator)
        random.random()
        return {"sampler_position": cursor["sampler_position"]}

    runtime.establish_baseline(lambda: scores(1.0))
    parent = runtime.capture()
    receipt = runtime.run_interval(
        interval_id="bad-future", steps=1, train_step=step, evaluator=lambda: scores(2.0)
    ).to_dict()
    assert receipt["decision"]["action"] == "restore-prior"
    assert receipt["rollback"]["verified_exact"]
    assert runtime.capture().fingerprint == parent.fingerprint
    archive = runtime.last_candidate_archive
    candidate = runtime.load_snapshot(archive["path"])
    assert candidate.fingerprint != parent.fingerprint
    assert receipt["candidate"]["evaluation"]["evidence"]["candidate_archive"] == archive
    runtime.restore(candidate)
    assert cursor == {"step": 1, "sampler_position": 3}
    assert cache["value"].tolist() == [6.0, 6.0]
    assert optimizer.state
    runtime.restore(parent)
    assert not optimizer.state
    assert cursor == {"step": 0, "sampler_position": 0}
    assert runtime.capture().fingerprint == parent.fingerprint


def test_candidate_evaluation_rng_and_mode_isolation(tmp_path):
    runtime, model, optimizer, cursor, _, _ = make_runtime(tmp_path)

    def evaluate():
        model.eval()
        cursor["step"] = 900
        torch.rand(3)
        return scores(1.0)

    parent = runtime.capture()
    runtime.establish_baseline(evaluate)
    assert runtime.capture().fingerprint == parent.fingerprint
    assert model.training
    assert cursor["step"] == 0


def test_broken_training_future_retains_partial_payload_and_restores(tmp_path):
    runtime, model, _, cursor, _, _ = make_runtime(tmp_path)
    runtime.establish_baseline(lambda: scores(1.0))
    parent = runtime.capture()

    def broken(_):
        with torch.no_grad():
            next(model.parameters()).add_(1)
        cursor["step"] = 1
        raise RuntimeError("deliberate interrupted future")

    with pytest.raises(TrainingIntervalError) as error:
        runtime.run_interval(
            interval_id="broken", steps=1, train_step=broken, evaluator=lambda: scores(0.0)
        )
    assert error.value.receipt.to_dict()["rollback"]["verified_exact"]
    assert Path(runtime.last_candidate_archive["path"]).is_file()
    assert runtime.capture().fingerprint == parent.fingerprint


def test_native_panel_common_parent_intervention_repair_and_saturn(tmp_path):
    runtime, model, _, _, _, _ = make_runtime(tmp_path)
    parent = runtime.capture()
    initial = {
        "carrier": torch.ones(1, 2),
        "suffix": torch.ones(1, 2),
        "target": torch.tensor([1], dtype=torch.long),
    }

    def consumer(state):
        logits = model(state["carrier"] + state["suffix"])
        generated = logits.argmax(-1).reshape(1, 1)
        return {"logits": logits, "generated": generated}, language_observation(
            logits, state["target"], generated
        )

    panel = runtime.native_panel(
        initial_state=initial, writable=("carrier",), consumer=consumer, panel_id="mechanics"
    )
    assert panel["mechanics"] == {
        "no_op_exact": True,
        "repair_exact": True,
        "restore_exact": True,
        "saturn_no_op_exact": True,
    }
    assert panel["mechanics_status"] == "valid"
    assert panel["terminal_status"] == "not-assessed"
    assert panel["physical_consumer_calls"] == 7
    rows = {row["arm"]: row for row in panel["rows"]}
    assert rows["dose-0"]["alias_of"] == "no-op"
    assert (
        rows["dose-2"]["observation"]["logits_sha256"]
        != rows["native"]["observation"]["logits_sha256"]
    )
    assert rows["native"]["cut"]["parent_handle"] == rows["dose-2"]["cut"]["parent_handle"]
    assert rows["repair"]["parent"] == "dose-2"
    assert Path(rows["dose-2"]["output_artifact"]).is_file()
    assert Path(panel["session_root"], "selected.json").is_file()
    assert runtime.capture().fingerprint == parent.fingerprint
    with pytest.raises(FileExistsError):
        runtime.native_panel(
            initial_state=initial, writable=("carrier",), consumer=consumer, panel_id="mechanics"
        )


def test_panel_refuses_invalid_input_and_restores_on_consumer_failure(tmp_path):
    runtime, _, _, _, _, _ = make_runtime(tmp_path)
    parent = runtime.capture()

    def consumer(_):
        raise RuntimeError("deliberate broken instrument")

    with pytest.raises(RuntimeError, match="broken instrument"):
        runtime.native_panel(
            initial_state={"carrier": torch.ones(2)}, writable=("carrier",), consumer=consumer
        )
    assert runtime.capture().fingerprint == parent.fingerprint
    with pytest.raises(ValueError, match="floating"):
        runtime.native_panel(
            initial_state={"carrier": torch.ones(2, dtype=torch.long)},
            writable=("carrier",),
            consumer=consumer,
            panel_id="invalid",
        )


def test_spf_real_carrier_seam_native_suffix_and_generated_history(tmp_path):
    from models import build_spf_model

    model = build_spf_model(vocab_size=8, d_model=4, n_layers=1, state_size=2, seed=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    runtime = initialize_runtime(
        model,
        optimizer,
        {"sampler_position": 0},
        tmp_path / "spf",
        {"model_id": "tiny-spf-reference", "family": "spf-reference"},
        mutable={
            "diagnostics": (
                lambda: model.last_diagnostics,
                lambda value: setattr(model, "last_diagnostics", value),
            )
        },
    )
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    carrier = model.capture_carrier(ids, 2)
    state = {
        **carrier["state_planes"],
        "suffix": ids[:, 2:],
        "targets": torch.tensor([[4, 5]]),
        "next_logits": carrier["next_logits"],
    }

    def consume(inputs):
        local = {**carrier, "state_planes": {key: inputs[key] for key in carrier["state_planes"]}}
        logits = model.continue_from_carrier(local, inputs["suffix"])
        generated = model.generate_from_carrier(local, inputs["next_logits"], 3)
        return {"logits": logits, "generated": generated}, language_observation(
            logits, inputs["targets"], generated
        )

    parent = runtime.capture()
    panel = runtime.native_panel(
        initial_state=state,
        writable=tuple(carrier["state_planes"]),
        consumer=consume,
        panel_id="spf-carrier",
    )
    assert panel["mechanics_status"] == "valid"
    assert runtime.capture().fingerprint == parent.fingerprint
    rows = {row["arm"]: row for row in panel["rows"]}
    assert len(rows["native"]["observation"]["generated_history"][0]) == 3
    assert (
        rows["native"]["observation"]["logits_sha256"]
        != rows["dose-2"]["observation"]["logits_sha256"]
    )
