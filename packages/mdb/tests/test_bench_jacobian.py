"""Tiny Qwen mechanics for bound reads and native residual writes."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchSession
from mdb.bench_jacobian import _header_hash, _model_identity, application_profile
from mdb.bench_verify import verify
from mdb.errors import BenchError
from mdb.jacobian_workspace import JacobianWorkspaceError, metadata_sha256, tensor_payload_identity


@pytest.fixture
def resident():
    torch.manual_seed(91)
    return (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=64,
                attn_implementation="eager",
            )
        )
        .eval()
        .to(torch.bfloat16)
    )


def lens(bench):
    matrix = torch.eye(16, dtype=torch.bfloat16)
    profile = application_profile(bench)
    body = {
        "schema": "saturn-jacobian-lens-v1",
        "model_identity": {"tokenizer_identity_policy": "token-ids-only-v1"},
        "execution_compatibility": {
            "fitting_lane": profile["application_lane"],
            "permitted_application_lanes": [profile["application_lane"]],
        },
        "coordinate_map": {
            "residual_width": 16,
            "source_layer": 0,
            "completed_layer_boundary": 1,
            "hook_convention": "decoder-block-output",
            "source_position_semantics": "absolute source token",
        },
        "transport": {
            "frozen": True,
            "shape": [16, 16],
            "dtype": str(matrix.dtype),
            "sha256": tensor_payload_identity(matrix)["sha256"],
        },
        "decoder": {
            "residual_width": 16,
            "vocab_size": 32,
            "final_norm": "model.norm",
            "output_head": "lm_head",
            "final_norm_weight_sha256": _header_hash(bench.model.model.norm.weight),
            "unembedding_weight_sha256": _header_hash(bench.model.lm_head.weight),
        },
        "estimator": {"name": "target-sum/source-mean"},
        "dictionary": {
            "atom_derivation": "decoder-weight-times-transport",
            "norm_scale_convention": "unit-l2-after-transport",
            "sparse_interpretation": "unperformed",
        },
    }
    body["model_identity"] = _model_identity(bench, body)
    return {**body, "fingerprint": metadata_sha256(body)}, matrix


def opened(resident, root):
    b = BenchSession.from_resident(resident, root=root, lane="native", continuation_capacity=8)
    b.prefill([1, 2, 3])
    manifest, transport = lens(b)
    pin = b.pin_jacobian_lens(manifest, transport)
    b.step("layer")
    read = b.read_jacobian(
        pin["lens_ref"],
        token_subset=[3, 7, 11],
        top_k=2,
        logical_address={"symbol": "fixture.animal", "instance": "one", "role": "remembered"},
    )
    return b, pin, read


def test_read_native_parity_and_explicit_writer_commit_abort(resident, tmp_path):
    b, pin, read = opened(resident, tmp_path / "bench")
    source = b.resolve_value("residual")
    with torch.inference_mode():
        scores = b.model.lm_head(b.model.model.norm(source))
    selected = b._tensor_load(
        b.artifacts.get(read["observation_ref"])["readout_refs"]["selected_normalized_scores"]
    )
    ids = b._tensor_load(
        b.artifacts.get(read["observation_ref"])["readout_refs"]["selected_token_ids"]
    )
    assert torch.equal(selected, scores.gather(-1, ids))
    assert torch.equal(source, b.resolve_value("residual"))
    recipe = {
        "operation": "steer",
        "token_id": 7,
        "dose": 0.5,
        "consumer": {"name": "fixture-later-query", "horizon": 2},
    }
    act = b.preview_jacobian_edit(read["observation_ref"], recipe)
    assert torch.equal(b.resolve_value("residual"), source)
    assert act["plan"]["writes"] == ["bench.residual"]
    candidate = b._tensor_load(b.artifacts.get(act["jacobian_candidate_ref"])["candidate_ref"])
    done = b.commit_edit(act["act_id"])
    assert done["jacobian_candidate_ref"] == act["jacobian_candidate_ref"]
    assert b.artifacts.get(done["receipt_ref"])["decision"] == "commit"
    assert torch.equal(b.resolve_value("residual"), candidate)
    b.restore(read["source_cut"])
    noop = b.preview_jacobian_edit(read["observation_ref"], {**recipe, "dose": 0})
    assert noop["writer_metadata"]["recipe"]["noop_exact"]
    b.abort_edit(noop["act_id"])
    assert torch.equal(b.resolve_value("residual"), source)
    assert verify(b.root)["counts"]["jacobian_artifacts"] >= 4
    b.close()


def test_stale_branch_cut_and_layer_are_rejected(resident, tmp_path):
    b, pin, read = opened(resident, tmp_path / "bench")
    recipe = {
        "operation": "span",
        "token_ids": [3, 7],
        "target_coordinates": [0, 1],
        "dose": 1,
        "consumer": {"name": "fixture-query", "horizon": 1},
    }
    b.fork("other")
    with pytest.raises(BenchError, match="stale"):
        b.preview_jacobian_edit(read["observation_ref"], recipe)
    b.checkout("main")
    b.restore(read["source_cut"])
    act = b.preview_jacobian_edit(read["observation_ref"], recipe)
    b.abort_edit(act["act_id"])
    # Even an exact tensor restore has a different cut identity after abort.
    with pytest.raises(BenchError, match="stale"):
        b.preview_jacobian_edit(read["observation_ref"], recipe)
    b.restore(read["source_cut"])
    b.step("layer")
    with pytest.raises(BenchError, match="completed layer"):
        b.read_jacobian(pin["lens_ref"])
    b.close()


def test_pin_rejects_corruption_identity_lane_and_decoder(resident, tmp_path):
    b = BenchSession.from_resident(resident, root=tmp_path / "bench", lane="native")
    b.prefill([1, 2])
    manifest, matrix = lens(b)
    bad = matrix.clone()
    bad[0, 0] = 2
    with pytest.raises(JacobianWorkspaceError, match="payload SHA"):
        b.pin_jacobian_lens(manifest, bad)
    for section, key, value in (
        ("model_identity", "weight_sha256", "f" * 64),
        ("execution_compatibility", "permitted_application_lanes", ["cuda-float32-eager-tf32off"]),
        ("decoder", "unembedding_weight_sha256", "f" * 64),
        ("decoder", "final_norm", "different normalization"),
        ("decoder", "output_head", "nonlinear head"),
        ("decoder", "logit_softcap", 30.0),
    ):
        changed = deepcopy(manifest)
        changed[section][key] = value
        changed["fingerprint"] = metadata_sha256(
            {k: v for k, v in changed.items() if k != "fingerprint"}
        )
        with pytest.raises((BenchError, JacobianWorkspaceError)):
            b.pin_jacobian_lens(changed, matrix)
    b.close()


def test_export_reopen_and_publication_failure_preserve_native_source(
    resident, tmp_path, monkeypatch
):
    b, pin, read = opened(resident, tmp_path / "bench")
    source = b.resolve_value("residual")
    selected = (b.root / "selected.json").read_bytes()
    original_publish = b._publish

    def fail():
        raise OSError("injected sidecar publication failure")

    monkeypatch.setattr(b, "_publish", fail)
    with pytest.raises(OSError):
        b.read_jacobian(pin["lens_ref"], token_subset=[3, 7])
    assert (b.root / "selected.json").read_bytes() == selected
    assert torch.equal(b.resolve_value("residual"), source)
    assert b.branch.cut == read["source_cut"]
    monkeypatch.setattr(b, "_publish", original_publish)
    b.export(tmp_path / "export")
    b.close()
    fresh = BenchSession.resume(resident, root=tmp_path / "export")
    fresh.restore(read["source_cut"])
    act = fresh.preview_jacobian_edit(
        read["observation_ref"],
        {
            "operation": "span",
            "token_ids": [3, 7],
            "target_coordinates": [0, 1],
            "dose": 0,
            "consumer": {"name": "fixture-query", "horizon": 1},
        },
    )
    fresh.commit_edit(act["act_id"])
    assert torch.equal(fresh.resolve_value("residual"), source)
    assert verify(fresh.root)["counts"]["jacobian_artifacts"] >= 3
    fresh.close()


def test_deferred_selection_failure_restores_all_sidecar_cursors(resident, tmp_path, monkeypatch):
    b, pin, read = opened(resident, tmp_path / "bench")
    before = {
        "selected": (b.root / "selected.json").read_bytes(),
        "frame": b.source_frames.current,
        "cut": b.branch.cut,
        "node": b.branch.node,
        "checkpoint": b.vm._session(b.active).checkpoint_ref,
        "events": deepcopy(b.events),
        "cuts": deepcopy(b.cuts),
        "hidden": b.resolve_value("residual"),
    }
    original_select = b.publication.select

    def fail_final_select(manifest):
        if b.publication._atomic_selection:
            return original_select(manifest)
        raise OSError("injected final selection failure")

    monkeypatch.setattr(b.publication, "select", fail_final_select)
    with pytest.raises(OSError, match="final selection"):
        b.read_jacobian(pin["lens_ref"], token_subset=[3, 7])
    assert (b.root / "selected.json").read_bytes() == before["selected"]
    assert b.source_frames.current == before["frame"]
    assert b.branch.cut == before["cut"] == read["source_cut"]
    assert b.branch.node == before["node"]
    assert b.vm._session(b.active).checkpoint_ref == before["checkpoint"]
    assert b.events == before["events"] and b.cuts == before["cuts"]
    assert torch.equal(b.resolve_value("residual"), before["hidden"])
    monkeypatch.setattr(b.publication, "select", original_select)
    b.close()


@pytest.mark.parametrize("decision", ["commit", "abort"])
def test_failed_native_publication_cannot_leak_a_completed_event(
    resident, tmp_path, monkeypatch, decision
):
    b, _, read = opened(resident, tmp_path / "bench")
    source = b.resolve_value("residual")
    act = b.preview_jacobian_edit(
        read["observation_ref"],
        {
            "operation": "steer",
            "token_id": 7,
            "dose": 0.5,
            "consumer": {"name": "fixture-query", "horizon": 1},
        },
    )
    selected = (b.root / "selected.json").read_bytes()
    previous_events, previous_cuts = deepcopy(b.events), deepcopy(b.cuts)
    original_select = b.publication.select

    def fail(manifest):
        raise OSError("injected native publication failure")

    monkeypatch.setattr(b.publication, "select", fail)
    with pytest.raises(OSError):
        getattr(b, f"{decision}_edit")(act["act_id"])
    assert (b.root / "selected.json").read_bytes() == selected
    assert b.events == previous_events and b.cuts == previous_cuts
    assert not b.branch.intervention_lineage
    assert b._edit is None and b._jacobian_pending is None
    assert torch.equal(b.resolve_value("residual"), source)
    monkeypatch.setattr(b.publication, "select", original_select)
    with b.publication.transaction():
        b._publish()
    assert b.events == previous_events
    verify(b.root)
    b.close()


def test_export_rejects_pending_native_act_before_checkpoint(resident, tmp_path):
    b, _, read = opened(resident, tmp_path / "bench")
    act = b.preview_jacobian_edit(
        read["observation_ref"],
        {
            "operation": "steer",
            "token_id": 7,
            "dose": 0,
            "consumer": {"name": "fixture-query", "horizon": 1},
        },
    )
    selected, cut = (b.root / "selected.json").read_bytes(), b.branch.cut
    with pytest.raises(BenchError, match="pending Act before export"):
        b.export(tmp_path / "export")
    assert not (tmp_path / "export").exists()
    assert (b.root / "selected.json").read_bytes() == selected and b.branch.cut == cut
    assert b._edit[0] == act["act_id"]
    b.abort_edit(act["act_id"])
    b.close()


def test_repeated_common_parent_act_has_distinct_attempt_custody(resident, tmp_path):
    b, _, read = opened(resident, tmp_path / "bench")
    recipe = {
        "operation": "steer",
        "token_id": 7,
        "dose": 0.5,
        "consumer": {"name": "fixture-query", "horizon": 1},
    }
    first = b.preview_jacobian_edit(read["observation_ref"], recipe)
    b.commit_edit(first["act_id"])
    b.restore(read["source_cut"])
    second = b.preview_jacobian_edit(read["observation_ref"], recipe)
    assert first["act_id"] == second["act_id"]
    assert first["jacobian_candidate_ref"] != second["jacobian_candidate_ref"]
    b.abort_edit(second["act_id"])
    terminal = [
        event
        for event in b.events
        if event.get("kind") in {"residual-act", "act-rejected"}
        and event.get("jacobian_candidate_ref")
        in {first["jacobian_candidate_ref"], second["jacobian_candidate_ref"]}
    ]
    assert {event["kind"] for event in terminal} == {"residual-act", "act-rejected"}
    verify(b.root)
    b.close()
