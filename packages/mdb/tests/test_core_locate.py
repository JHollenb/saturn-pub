"""``mdb.core.locate``: a model name resolves to a local checkpoint, never to a download."""
from __future__ import annotations

import os
import time

import pytest

from mdb.core.errors import BadArgs
from mdb.core.locate import cached_models, locate_model


def _repo(cache, identifier, revisions=("abc",), main=None, config=True):
    repo = cache / ("models--" + identifier.replace("/", "--"))
    for index, revision in enumerate(revisions):
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        if config:
            (snapshot / "config.json").write_text("{}")
        stamp = time.time() - 100 + index
        os.utime(snapshot, (stamp, stamp))
    if main is not None:
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(main + "\n")
    return repo


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # no real cache behind the fake one
    _repo(hub, "Qwen/Qwen2.5-1.5B-Instruct", ("old", "new"), main="old")
    _repo(hub, "Qwen/Qwen2.5-0.5B-Instruct")
    _repo(hub, "Qwen/Qwen3-0.6B", ("r1", "r2"))
    _repo(hub, "EleutherAI/broken", config=False)
    return hub


def test_a_directory_is_taken_as_given(tmp_path, cache):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    with pytest.raises(BadArgs, match="config.json missing"):
        locate_model(str(checkpoint))
    (checkpoint / "config.json").write_text("{}")
    assert locate_model(str(checkpoint)) == checkpoint.resolve()


def test_names_resolve_in_order_full_id_then_name_then_unique_substring(cache):
    main = cache / "models--Qwen--Qwen2.5-1.5B-Instruct" / "snapshots" / "old"
    assert locate_model("Qwen/Qwen2.5-1.5B-Instruct") == main  # refs/main wins over mtime
    assert locate_model("qwen2.5-1.5b-instruct") == main
    assert locate_model("qwen2.5-1.5b") == main
    newest = cache / "models--Qwen--Qwen3-0.6B" / "snapshots" / "r2"
    assert locate_model("qwen3") == newest  # no refs/main: the newest snapshot


def test_ambiguous_and_unknown_names_are_refused_with_candidates(cache):
    with pytest.raises(BadArgs) as caught:
        locate_model("qwen2.5")
    assert "matches 2 cached models" in caught.value.message
    assert "Qwen/Qwen2.5-0.5B-Instruct" in caught.value.message
    with pytest.raises(BadArgs) as caught:
        locate_model("llama")
    assert "nothing is downloaded" in caught.value.message and "Qwen/Qwen3-0.6B" in \
        caught.value.message
    with pytest.raises(BadArgs):
        locate_model("  ")


def test_a_repository_without_a_config_is_not_a_model(cache):
    assert "EleutherAI/broken" not in cached_models()
    with pytest.raises(BadArgs):
        locate_model("broken")
