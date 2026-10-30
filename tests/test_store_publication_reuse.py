from pathlib import Path

import pytest

from saturn_pub import store


def test_identical_existing_artifact_skips_tempfile_and_fsync(tmp_path, monkeypatch):
    path = tmp_path / "pages" / "page.json"
    path.parent.mkdir()
    payload = b'{"schema":"page"}'
    path.write_bytes(payload)

    def unexpected(*args, **kwargs):
        raise AssertionError("immutable reuse should not allocate or fsync a temporary file")

    monkeypatch.setattr(store.tempfile, "NamedTemporaryFile", unexpected)
    monkeypatch.setattr(store.os, "fsync", unexpected)

    assert store._publish(path, payload) is False
    assert path.read_bytes() == payload


def test_corrupt_existing_artifact_is_rejected_without_replacement(tmp_path, monkeypatch):
    path = tmp_path / "pages" / "page.json"
    path.parent.mkdir()
    corrupt = b"corrupt"
    path.write_bytes(corrupt)

    monkeypatch.setattr(
        store.tempfile,
        "NamedTemporaryFile",
        lambda *args, **kwargs: pytest.fail("corrupt CAS object must fail before temp publication"),
    )
    with pytest.raises(ValueError, match="existing artifact does not match its content address"):
        store._publish(path, b"expected")
    assert path.read_bytes() == corrupt


def test_first_publish_and_link_race_are_atomic_and_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "blobs" / "blob"
    payload = b"immutable payload"

    assert store._publish(path, payload) is True
    assert path.read_bytes() == payload
    assert [item.name for item in path.parent.iterdir()] == [path.name]
    assert store._publish(path, payload) is False

    raced = tmp_path / "raced" / "blob"
    raced.parent.mkdir()
    calls = []

    def competing_link(temporary, destination):
        calls.append((Path(temporary), Path(destination)))
        Path(destination).write_bytes(payload)
        raise FileExistsError(destination)

    monkeypatch.setattr(store.os, "link", competing_link)
    assert store._publish(raced, payload) is False
    assert calls == [(calls[0][0], raced)]
    assert raced.read_bytes() == payload
    assert [item.name for item in raced.parent.iterdir()] == [raced.name]
