import errno
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from saturn_pub.shared_store import SharedStore, configure, configured_store


def test_whole_file_dedup_preserves_paths_bytes_and_seals(tmp_path):
    pool = SharedStore(tmp_path / "pool")
    paths = [tmp_path / "one.npy", tmp_path / "two.npy"]
    for path in paths:
        path.write_bytes(b"immutable tensor bytes")
    results = [pool.intern(path) for path in paths]
    assert not results[0]["relinked"] and results[1]["relinked"]
    assert results[1]["released_inode_logical_bytes"] == len(paths[0].read_bytes())
    assert os.path.samefile(*paths)
    assert not paths[0].is_symlink() and paths[0].stat().st_mode & 0o222 == 0
    assert pool.intern(paths[1])["released_inode_logical_bytes"] == 0
    assert paths[0].read_bytes() == b"immutable tensor bytes"


def test_changed_source_or_corrupt_object_never_replaces_source(tmp_path):
    pool = SharedStore(tmp_path / "pool")
    path = tmp_path / "input.npy"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="frozen storage plan"):
        pool.intern(path, expected_sha256="0" * 64)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    obj = pool.object_path(digest)
    obj.parent.mkdir(); obj.write_bytes(b"corrupt!"); obj.chmod(0o444)
    with pytest.raises(ValueError, match="content-address mismatch"):
        pool.intern(path)
    assert path.read_bytes() == b"original" and not os.path.samefile(path, obj)


def test_interrupt_before_replace_retains_both_original_and_canonical(tmp_path, monkeypatch):
    pool = SharedStore(tmp_path / "pool")
    first, second = tmp_path / "one.npy", tmp_path / "two.npy"
    first.write_bytes(b"same"); second.write_bytes(b"same")
    pool.intern(first)
    def fail(*args):
        raise OSError("simulated replacement interruption")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="interruption"):
        pool.intern(second)
    assert first.read_bytes() == second.read_bytes() == b"same"
    assert not list(tmp_path.glob(".mdb-share-*"))


def test_symlink_and_cross_device_link_fail_without_losing_data(tmp_path, monkeypatch):
    pool = SharedStore(tmp_path / "pool")
    path = tmp_path / "one.npy"; path.write_bytes(b"safe")
    alias = tmp_path / "alias.npy"; alias.symlink_to(path)
    with pytest.raises(ValueError, match="symbolic links"):
        pool.intern(alias)
    def fail(*args):
        raise OSError(errno.EXDEV, "cross-device link")
    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError, match="cross-device"):
        pool.intern(path)
    assert path.read_bytes() == b"safe"


def test_concurrent_publish_and_existing_content_validation(tmp_path):
    pool = SharedStore(tmp_path / "pool")
    paths = [tmp_path / str(i) / "array.npy" for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert all(workers.map(lambda p: pool.publish(p, b"same value"), paths))
    assert len(list(pool.objects.glob("*/*"))) == 1
    assert all(os.path.samefile(paths[0], p) for p in paths)
    assert not pool.publish(paths[0], b"same value")
    with pytest.raises(ValueError, match="existing artifact differs"):
        pool.publish(paths[0], b"different")


def test_configured_scope_and_environment(tmp_path, monkeypatch):
    scope, root = tmp_path / "artifacts", tmp_path / "pool"
    configure(scope, root)
    assert configured_store(scope / "run/store/blobs/a.safetensors").root == root
    assert configured_store(tmp_path / "elsewhere/file.npy") is None
    monkeypatch.setenv("MDB_SHARED_TENSOR_STORE", str(root))
    assert configured_store(tmp_path / "elsewhere/file.npy").root == root
    with pytest.raises(ValueError, match="differently"):
        configure(scope, tmp_path / "another-pool")


def test_native_statecut_identity_and_legacy_loading_across_shared_stores(tmp_path):
    torch = pytest.importorskip("torch")
    from saturn_pub import StateCut
    from saturn_pub.store import LocalStore

    configure(tmp_path / "runs", tmp_path / "pool")
    cut = StateCut("tiny", {"backend": "cpu"}, "boundary", None,
                   {"value": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)})
    stores = [LocalStore(tmp_path / "runs" / name) for name in ["one", "two"]]
    for store in stores:
        assert store.save(cut) == cut.fingerprint
        loaded = store.load(cut.fingerprint)
        assert loaded.fingerprint == cut.fingerprint
        assert torch.equal(loaded.payload["value"], cut.payload["value"])
    blobs = [next((s.root / "blobs").glob("*.safetensors")) for s in stores]
    assert os.path.samefile(*blobs)
