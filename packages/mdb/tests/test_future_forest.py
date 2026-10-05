from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from mdb.future_forest import FutureForestError, PersistentFutureForest


def _forest(tmp_path: Path) -> tuple[PersistentFutureForest, str, dict, dict, dict]:
    forest = PersistentFutureForest(tmp_path / "forest", forest_id="test-forest")
    base = forest.put_blob(b"base", media_type="tensor/raw")
    ability_a = forest.put_blob(b"ability-a", media_type="act/state")
    ability_b = forest.put_blob(b"ability-b", media_type="act/state")
    root = forest.create_root(state_refs={"model/base": base}, metadata={"step": 0})
    return forest, root, base, ability_a, ability_b


def test_closure_snapshot_reads_each_node_once_and_does_not_hide_later_corruption(tmp_path, monkeypatch):
    forest, root, base, ability_a, ability_b = _forest(tmp_path)
    parent = root
    for index in range(12):
        parent = forest.fork(parent, transition={"kind": "step", "index": index},
                             diff_refs={"model/base": base, "a": ability_a, "b": ability_b})
    expected = forest._verify_closure()
    original = forest._read_json
    reads = []
    def tracked(path):
        if path.parent == forest.nodes: reads.append(path)
        return original(path)
    monkeypatch.setattr(forest, "_read_json", tracked)
    with forest.verification_snapshot():
        assert forest.verify_closure() == expected
        for node in forest.list_nodes():
            forest.verify_node(node)
            forest.effective_state(node)
    assert len(reads) == expected["nodes"]
    assert len(set(reads)) == expected["nodes"]
    path = forest.nodes / f"{root}.json"
    body = json.loads(path.read_text())
    body["metadata"]["step"] = 99
    path.write_text(json.dumps(body))
    with pytest.raises(FutureForestError): forest.verify_closure()


def test_verification_snapshot_resets_after_exception(tmp_path):
    forest, root, *_ = _forest(tmp_path)
    with pytest.raises(RuntimeError, match="stop audit"):
        with forest.verification_snapshot():
            forest.verify_node(root)
            raise RuntimeError("stop audit")
    path = forest.nodes / f"{root}.json"
    body = json.loads(path.read_text())
    body["metadata"]["tampered"] = True
    path.write_text(json.dumps(body))
    with pytest.raises(FutureForestError):
        forest.verify_node(root)


def test_branches_share_parent_and_rejected_evidence_survives(tmp_path: Path) -> None:
    forest, root, _base, ability_a, ability_b = _forest(tmp_path)
    branch_a = forest.fork(
        root,
        transition={"kind": "train", "checkpoint": 10, "address": "layer3/head1"},
        diff_refs={"capability/a": ability_a},
    )
    branch_b = forest.fork(
        root,
        transition={"kind": "train", "checkpoint": 10, "address": "layer4/head2"},
        diff_refs={"capability/b": ability_b},
    )
    observation = forest.observe(
        branch_b,
        stage="native-consumer",
        scores={"target": 0.2, "collateral": -0.8},
    )
    decision = forest.decide(
        branch_b, decision="prune", observation_ids=(observation,), reason="collateral regression"
    )
    assert forest.head() == root
    assert forest.verify_decision(decision)["decision"] == "prune"
    assert forest.effective_state(branch_a)["model/base"] == forest.effective_state(branch_b)[
        "model/base"
    ]
    closure = forest.verify_closure()
    assert closure["nodes"] == 3
    assert closure["content_objects"] == 3
    assert closure["logical_effective_state_references"] == 5


def test_promote_rewind_cherry_pick_and_three_way_merge(tmp_path: Path) -> None:
    forest, root, _base, ability_a, ability_b = _forest(tmp_path)
    left = forest.fork(
        root, transition={"kind": "train", "branch": "left"}, diff_refs={"cap/a": ability_a}
    )
    right = forest.fork(
        root,
        transition={"kind": "train", "branch": "right"},
        diff_refs={"cap/b": ability_b},
    )
    observed = forest.observe(left, stage="rollout", scores={"capability": 1.0})
    forest.decide(left, decision="promote", observation_ids=(observed,), reason="consumer benefit")
    assert forest.head() == left
    forest.rewind(root, reason="try another future")
    assert forest.head() == root

    cherry = forest.cherry_pick(left, right, addresses=("cap/a",))
    state = forest.effective_state(cherry)
    assert state["cap/a"] == ability_a and state["cap/b"] == ability_b

    merged = forest.three_way_merge(base_id=root, left_id=left, right_id=right)
    assert forest.effective_state(merged) == state
    assert set(forest.verify_node(merged)["parents"]) == {left, right}


def test_conflicting_merge_requires_explicit_resolution(tmp_path: Path) -> None:
    forest, root, _base, ability_a, ability_b = _forest(tmp_path)
    left = forest.fork(
        root, transition={"kind": "train"}, diff_refs={"shared/address": ability_a}
    )
    right = forest.fork(
        root, transition={"kind": "train"}, diff_refs={"shared/address": ability_b}
    )
    with pytest.raises(FutureForestError, match="unresolved"):
        forest.three_way_merge(base_id=root, left_id=left, right_id=right)
    merged = forest.three_way_merge(
        base_id=root,
        left_id=left,
        right_id=right,
        resolutions={"shared/address": "right"},
    )
    assert forest.effective_state(merged)["shared/address"] == ability_b


def test_promotion_cannot_jump_between_sibling_histories(tmp_path: Path) -> None:
    forest, root, _base, ability_a, ability_b = _forest(tmp_path)
    left = forest.fork(
        root, transition={"kind": "train"}, diff_refs={"cap/a": ability_a}
    )
    right = forest.fork(
        root, transition={"kind": "train"}, diff_refs={"cap/b": ability_b}
    )
    left_observation = forest.observe(left, stage="rollout", scores={"target": 1.0})
    right_observation = forest.observe(right, stage="rollout", scores={"target": 1.0})
    forest.decide(
        left,
        decision="promote",
        observation_ids=(left_observation,),
        reason="choose left",
    )
    with pytest.raises(FutureForestError, match="does not descend"):
        forest.decide(
            right,
            decision="promote",
            observation_ids=(right_observation,),
            reason="invalid sibling jump",
        )


def test_reopen_and_tamper_rejection(tmp_path: Path) -> None:
    forest, root, _base, ability_a, _ability_b = _forest(tmp_path)
    branch = forest.fork(root, transition={"kind": "train"}, diff_refs={"cap/a": ability_a})
    reopened = PersistentFutureForest.reopen(forest.root)
    assert reopened.effective_state(branch)["cap/a"] == ability_a

    node_path = forest.nodes / f"{branch}.json"
    payload = json.loads(node_path.read_text())
    payload["metadata"]["tampered"] = True
    node_path.write_text(json.dumps(payload))
    with pytest.raises(FutureForestError, match="content hash changed"):
        reopened.verify_node(branch)


def test_closure_rejects_orphan_content_object(tmp_path: Path) -> None:
    forest, _root, _base, _ability_a, _ability_b = _forest(tmp_path)
    forest.put_blob(b"unreferenced", media_type="tensor/raw")
    with pytest.raises(FutureForestError, match="orphaned"):
        forest.verify_closure()


def test_immutable_publish_falls_back_when_volume_rejects_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_hard_link(source: str | bytes | Path, destination: str | bytes | Path) -> None:
        del source, destination
        raise OSError(errno.EPERM, "hard links unsupported by mounted filesystem")

    monkeypatch.setattr(os, "link", reject_hard_link)
    forest = PersistentFutureForest(tmp_path / "forest", forest_id="no-hard-links")
    reference = forest.put_blob(b"payload", media_type="application/octet-stream")
    forest.create_root(state_refs={"payload": reference}, metadata={})
    assert (forest.root / "manifest.json").is_file()
    assert forest.get_blob(reference) == b"payload"
    assert forest.verify_closure()["content_objects"] == 1

    # Re-publishing equal bytes is idempotent, while different bytes at the
    # same immutable path still fail closed under the fallback.
    target = forest.root / "immutable.bin"
    forest._publish_bytes(target, b"same")
    forest._publish_bytes(target, b"same")
    with pytest.raises(FutureForestError, match="different bytes"):
        forest._publish_bytes(target, b"different")
