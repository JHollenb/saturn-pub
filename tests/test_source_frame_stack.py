from dataclasses import replace
import json

import pytest
import torch

from mdb.ar.source_set import ArSourceSet
from mdb.source_frame_stack import FrameSourceBinding, SourceFrameError, SourceFrameStack
from mdb.source_program_surface import artifact_from_ar_source_set, materialize_ar_source_set
from mdb.spectral_address import (
    CanonicalSymbolKey, SpectralFrame, SpectralInstanceKey, SpectralLogicalAddress,
    SpectralQuery, SpectralRoleTransformSpec,
)


def address(instance="one"):
    return SpectralLogicalAddress(CanonicalSymbolKey("fact", (0.1, 0.2), "memory", "program", "fixture"),
        SpectralInstanceKey.deterministic("episode", instance, 2),
        SpectralRoleTransformSpec("source", (0., 0.), 1, 0, "program", "program"), SpectralFrame("scope"))


def query(a):
    return SpectralQuery("query", a.phases, "memory", "program", "scope",
        exact_symbol_id="fact", exact_scene_id="episode", exact_instance_id=a.instance_key.instance_id,
        exact_role_id="source")


def artifact(name, offset=0):
    key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4) + offset
    source = ArSourceSet(model_id="fixture", revision="r1", source_token_ids=(1, 2),
        layers=((key, key + 1),), admitted=(True, True))
    return artifact_from_ar_source_set(source, store_id=name)


def cut(stack, name, parent=None, identity=None):
    return stack.cuts.capture(cut_id=name, family="qwen2", state_schema_id="fixture-v1",
        model_identity=identity or {"model_id": "fixture", "revision": "r1"}, boundary="token",
        index=0, total=2, state={"pending_token": 3}, parent_cut=parent)


def fixture_stack(tmp_path):
    stack = SourceFrameStack(tmp_path)
    root_cut = cut(stack, "root")
    root = stack.push(label="root", statecut_ref=root_cut.ref, artifacts={"a": artifact("a")},
        bindings=(FrameSourceBinding(address(), "a"),))
    return stack, root, root_cut


def test_nested_shadow_pop_and_reopen_preserve_sibling_sources(tmp_path):
    stack, root, parent_cut = fixture_stack(tmp_path)
    child_cut = cut(stack, "child", parent_cut.ref)
    child = stack.push(label="child", statecut_ref=child_cut.ref, artifacts={"b": artifact("b", 100)},
        bindings=(FrameSourceBinding(address(), "b"),))
    selected = stack.resolve(query(address()))
    assert selected["selection"]["store_id"] == "b"
    artifacts, plan = stack.materialize_plan(selected)
    assert materialize_ar_source_set(artifacts, plan, base_store_id="b").layers[0][0][0, 0, 0, 0] == 100
    assert stack.pop() == root
    assert stack.resolve(query(address()))["selection"]["store_id"] == "a"
    sibling = stack.push(label="sibling", statecut_ref=parent_cut.ref)
    assert stack.resolve(query(address()))["selection"]["store_id"] == "a"
    reopened = SourceFrameStack(tmp_path)
    assert reopened.current == sibling
    reopened.checkout(child)
    assert reopened.resolve(query(address()))["selection"]["store_id"] == "b"
    assert reopened.restore_statecut().state["pending_token"] == 3
    assert reopened.debug_view()["scope_depth"] == 2
    reopened.register.forest.verify_closure()


def test_wrong_scope_role_instance_and_forged_selection_refuse(tmp_path):
    stack, _, _ = fixture_stack(tmp_path)
    for q in (replace(query(address()), exact_role_id="wrong"),
              replace(query(address()), exact_instance_id="absent"),
              replace(query(address()), frame_id="other")):
        resolution = stack.resolve(q)
        assert resolution["abstained"]
        with pytest.raises(SourceFrameError, match="abstained"):
            stack.materialize_plan(resolution)
    resolution = stack.resolve(query(address()))
    resolution["selection"]["store_id"] = "forged"
    with pytest.raises(SourceFrameError, match="sealed frame scope"):
        stack.materialize_plan(resolution)


def test_bad_child_cut_and_store_rebinding_leave_active_parent(tmp_path):
    stack, root, parent_cut = fixture_stack(tmp_path)
    for bad in (cut(stack, "unrelated"), cut(stack, "wrong-model", parent_cut.ref,
                 {"model_id": "wrong", "revision": "r1"})):
        with pytest.raises(SourceFrameError):
            stack.push(label="bad", statecut_ref=bad.ref)
        assert stack.current == root
    with pytest.raises(SourceFrameError, match="immutable source store"):
        stack.push(label="bad", statecut_ref=parent_cut.ref, artifacts={"a": artifact("a", 10)})
    assert stack.current == root


def test_statecut_tamper_is_detected_before_native_hydration(tmp_path):
    stack, _, parent_cut = fixture_stack(tmp_path)
    path = stack.cuts.manifest_path(parent_cut)
    body = json.loads(path.read_text())
    body["model_identity"]["revision"] = "tampered"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        stack.restore_statecut()


def test_source_address_policy_nomination_is_bound_and_can_abstain(tmp_path):
    from mdb.ar.source_address_policy import SourceAddressPolicy
    stack, _, _ = fixture_stack(tmp_path)
    policy = SourceAddressPolicy(hidden_size=4, bottleneck=2, layer=0,
        model_identity="fixture", tokenizer_identity="tok", seed=0)
    # Bind the fixture cut to its policy tokenizer, using a new root.
    stack = SourceFrameStack(tmp_path / "policy")
    parent = cut(stack, "root", identity={"model_id": "fixture", "revision": "r1", "tokenizer_sha256": "tok"})
    stack.push(label="root", statecut_ref=parent.ref, artifacts={"a": artifact("a")},
        bindings=(FrameSourceBinding(address(), "a"),))
    with torch.no_grad():
        policy.abstain.bias.fill_(-100)
    resolved = stack.resolve(query(address()), policy=policy, query_state=torch.ones(1, 4),
        candidate_states={address().fingerprint: torch.ones(4)})
    assert not resolved["abstained"]
    stack.materialize_plan(resolved)
    alias = replace(address(), canonical_key=replace(address().canonical_key, identity_phases=(0.3, 0.4)))
    stack.push(label="conflicting-exact-name", statecut_ref=parent.ref,
        artifacts={"b": artifact("b", 100)}, bindings=(FrameSourceBinding(alias, "b"),))
    ambiguous = stack.resolve(query(address()), policy=policy, query_state=torch.ones(1, 4),
        candidate_states={a.fingerprint: torch.ones(4) for a in (address(), alias)})
    assert ambiguous["abstained"]
    assert ambiguous["reason"] == "ambiguous-payload-binding"
    stack.pop()
    with torch.no_grad():
        policy.abstain.bias.fill_(100)
    assert stack.resolve(query(address()), policy=policy, query_state=torch.ones(1, 4),
        candidate_states={address().fingerprint: torch.ones(4)})["abstained"]
    policy.model_identity = "other"
    with pytest.raises(SourceFrameError, match="identity"):
        stack.resolve(query(address()), policy=policy, query_state=torch.ones(1, 4),
            candidate_states={address().fingerprint: torch.ones(4)})


def test_debugger_exposes_frames_beside_cuts(tmp_path, capsys):
    from test_debugger_protocol import make_session
    from mdb.debugger_cli import DebugShell
    stack, root, _ = fixture_stack(tmp_path)
    session, worker = make_session()
    worker.source_frame_stack = stack
    assert session.frames()["frames"][0]["node_id"] == root
    assert session.cuts()["source_frames"]["scope_depth"] == 1
    assert "frames" in session.capabilities()["optional_operations"]
    shell = DebugShell(session)
    shell.execute("frames")
    assert json.loads(capsys.readouterr().out)["scope_depth"] == 1
