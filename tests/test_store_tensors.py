import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from saturn_pub.store import LocalStore

ARRAYS = {
    "float32": np.arange(12, dtype=np.float32).reshape(3, 4) / 7,
    "float16": np.array([[1.5, -2.25], [65504.0, 0.0]], dtype=np.float16),
    "int64": np.array([-(2**62), 0, 2**62 + 5], dtype=np.int64),
    "bool": np.array([[True, False, True]]),
    "big-endian": np.arange(5, dtype=">i4"),
    "strided": np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2],
    "scalar": np.float32(3.5),
    "empty": np.zeros((0, 3), dtype=np.float32),
}


@pytest.mark.parametrize("name", ARRAYS)
def test_numpy_round_trip(tmp_path, name):
    store = LocalStore(tmp_path)
    value = ARRAYS[name]
    ref = store.put_tensor(value)
    restored = store.get_tensor(ref["ref"])
    expected = np.asarray(value)
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == expected.dtype.newbyteorder("=") and restored.shape == expected.shape
    assert np.array_equal(restored, expected)
    little = expected.astype(expected.dtype.newbyteorder("<"))
    assert ref == {"ref": ref["ref"], "shape": list(expected.shape), "dtype": expected.dtype.name,
                   "sha256": hashlib.sha256(little.tobytes()).hexdigest(),
                   "bytes": expected.nbytes}
    assert store.tensor_info(ref["ref"]) == ref
    assert store.has_tensor(ref["ref"])


def test_numpy_tensors_need_no_torch(tmp_path):
    code = (
        "import sys, numpy as np\n"
        "from saturn_pub.store import LocalStore\n"
        f"s = LocalStore({str(tmp_path)!r})\n"
        "r = s.put_tensor(np.arange(6, dtype=np.int64).reshape(2, 3))\n"
        "assert (s.get_tensor(r['ref']) == np.arange(6).reshape(2, 3)).all()\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_idempotent_content_address(tmp_path):
    store = LocalStore(tmp_path)
    value = np.arange(10, dtype=np.float32)
    first, second = store.put_tensor(value), store.put_tensor(value.copy())
    assert first == second
    assert sorted(p.name for p in (tmp_path / "tensors").iterdir()) == [
        f"{first['ref']}.safetensors"]
    # same bytes under another dtype or shape are another tensor
    assert store.put_tensor(value.view(np.int32))["ref"] != first["ref"]
    assert store.put_tensor(value.reshape(2, 5))["ref"] != first["ref"]
    assert store.put_tensor(value.reshape(2, 5))["sha256"] == first["sha256"]
    raw = (tmp_path / "tensors" / f"{first['ref']}.safetensors").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == first["ref"]


def test_corruption_is_detected(tmp_path):
    store = LocalStore(tmp_path)
    ref = store.put_tensor(np.arange(8, dtype=np.float32))["ref"]
    path = tmp_path / "tensors" / f"{ref}.safetensors"
    good = path.read_bytes()
    path.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    with pytest.raises(ValueError, match="tensor blob digest mismatch"):
        store.get_tensor(ref)
    # A file that hashes to its own name but whose payload differs from its sealed sha256.
    size = int.from_bytes(good[:8], "little")
    forged = good[:8 + size] + bytes(len(good) - 8 - size)
    forged_ref = hashlib.sha256(forged).hexdigest()
    (tmp_path / "tensors" / f"{forged_ref}.safetensors").write_bytes(forged)
    with pytest.raises(ValueError, match="tensor payload digest mismatch"):
        store.get_tensor(forged_ref)
    # A header whose shape disagrees with the payload length.
    header = json.loads(good[8:8 + size])
    header["tensor"]["shape"] = [9]
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    bad = len(text).to_bytes(8, "little") + text + good[8 + size:]
    bad_ref = hashlib.sha256(bad).hexdigest()
    (tmp_path / "tensors" / f"{bad_ref}.safetensors").write_bytes(bad)
    with pytest.raises(ValueError, match="does not match its header"):
        store.get_tensor(bad_ref)
    with pytest.raises(ValueError, match="SHA-256"):
        store.get_tensor("../" + ref)
    with pytest.raises(FileNotFoundError):
        store.get_tensor("0" * 64)


def test_refuses_unsupported_values(tmp_path):
    store = LocalStore(tmp_path)
    for value in ([1, 2], np.array(["a"]), np.array([1 + 2j]), np.array([object()])):
        with pytest.raises(ValueError, match="put_tensor expects|unsupported"):
            store.put_tensor(value)
    ref = store.put_tensor(np.zeros(2, np.float32))["ref"]
    with pytest.raises(ValueError, match="framework"):
        store.get_tensor(ref, framework="jax")


def test_bfloat16_torch_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    from saturn_pub.values import describe

    store = LocalStore(tmp_path)
    value = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 3).to(torch.bfloat16)
    ref = store.put_tensor(value[:, 1:])  # non-contiguous view
    assert ref["dtype"] == "bfloat16" and ref["shape"] == [2, 2, 4] and ref["bytes"] == 32
    assert ref["sha256"] == describe(value[:, 1:].contiguous())["sha256"]
    restored = store.get_tensor(ref["ref"])
    assert restored.dtype == torch.bfloat16 and torch.equal(restored, value[:, 1:])
    assert store.put_tensor(value[:, 1:].clone()) == ref
    with pytest.raises(ValueError, match="needs framework='torch'"):
        store.get_tensor(ref["ref"], framework="numpy")
    as_torch = store.get_tensor(store.put_tensor(np.arange(3, dtype=np.int64))["ref"],
                                framework="torch")
    assert as_torch.dtype == torch.int64 and as_torch.tolist() == [0, 1, 2]
    assert store.get_tensor(ref["ref"], device="cpu").device.type == "cpu"
    flags = torch.tensor([True, False])
    assert torch.equal(store.get_tensor(store.put_tensor(flags)["ref"], framework="torch"), flags)
    from safetensors.torch import load_file

    loaded = load_file(tmp_path / "tensors" / f"{ref['ref']}.safetensors")
    assert torch.equal(loaded["tensor"], value[:, 1:])


def _blob(tmp_path, ref):
    return tmp_path / "tensors" / f"{ref}.safetensors"


def test_putting_again_repairs_a_corrupt_blob_atomically(tmp_path, monkeypatch):
    import os

    store = LocalStore(tmp_path)
    value = np.arange(8, dtype=np.float32)
    ref = store.put_tensor(value)["ref"]
    good = _blob(tmp_path, ref).read_bytes()
    for damaged in (good[:-1] + bytes([good[-1] ^ 1]), good[: len(good) // 2], b"", b"junk"):
        _blob(tmp_path, ref).write_bytes(damaged)
        with pytest.raises(ValueError):
            store.get_tensor(ref)
        calls = []
        real_fsync, real_replace = os.fsync, os.replace
        monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
        monkeypatch.setattr(os, "replace", lambda a, b: (calls.append("replace"),
                                                        real_replace(a, b))[1])
        assert store.put_tensor(value)["ref"] == ref          # repairs instead of raising
        monkeypatch.undo()
        assert calls[:2] == ["fsync", "replace"]               # synced before it was renamed
        assert _blob(tmp_path, ref).read_bytes() == good
        assert np.array_equal(store.get_tensor(ref), value)
        assert sorted(p.name for p in (tmp_path / "tensors").iterdir()) == [f"{ref}.safetensors"]
    # an intact blob is left alone (not rewritten)
    before = _blob(tmp_path, ref).stat().st_ino
    store.put_tensor(value)
    assert _blob(tmp_path, ref).stat().st_ino == before


def test_a_failed_repair_leaves_the_damaged_file_and_no_temporary(tmp_path, monkeypatch):
    import os

    store = LocalStore(tmp_path)
    value = np.arange(4, dtype=np.int32)
    ref = store.put_tensor(value)["ref"]
    _blob(tmp_path, ref).write_bytes(b"damaged")
    monkeypatch.setattr(os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError, match="injected"):
        store.put_tensor(value)
    monkeypatch.undo()
    assert _blob(tmp_path, ref).read_bytes() == b"damaged"
    assert [p.name for p in (tmp_path / "tensors").iterdir()] == [f"{ref}.safetensors"]
    store.put_tensor(value)
    assert np.array_equal(store.get_tensor(ref), value)


def test_has_tensor_verifies_size_and_hash_on_request(tmp_path):
    store = LocalStore(tmp_path)
    ref = store.put_tensor(np.arange(8, dtype=np.float32))["ref"]
    path = _blob(tmp_path, ref)
    good = path.read_bytes()
    assert store.has_tensor(ref) and store.has_tensor(ref, verify=True)
    assert not store.has_tensor("0" * 64) and not store.has_tensor("0" * 64, verify=True)
    path.write_bytes(good[:-4])                                   # wrong size
    assert store.has_tensor(ref) and not store.has_tensor(ref, verify=True)
    path.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))           # right size, wrong hash
    assert store.has_tensor(ref) and not store.has_tensor(ref, verify=True)
    path.write_bytes(b"")
    assert store.has_tensor(ref) and not store.has_tensor(ref, verify=True)
    path.write_bytes(good)
    assert store.has_tensor(ref, verify=True)
    with pytest.raises(ValueError, match="SHA-256"):
        store.has_tensor("../" + ref)


def test_lazy_negation_and_conjugation_views_are_materialised(tmp_path):
    torch = pytest.importorskip("torch")

    store = LocalStore(tmp_path)
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3) - 2.5
    lazy = torch._neg_view(base)
    assert lazy.is_neg()
    stored = store.put_tensor(lazy)
    assert store.get_tensor(stored["ref"]).tolist() == (-base).tolist()
    assert stored == store.put_tensor(-base)                      # same bytes, same address
    from saturn_pub.values import describe

    assert stored["sha256"] == describe((-base).contiguous())["sha256"]
    assert store.put_tensor(base.conj()) == store.put_tensor(base)   # a no-op on real dtypes
    neg_bf16 = torch._neg_view(base.to(torch.bfloat16))
    assert store.get_tensor(store.put_tensor(neg_bf16)["ref"]).float().tolist() == (-base).tolist()
    complex_view = torch.view_as_complex(torch.zeros(2, 2)).conj()
    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        store.put_tensor(complex_view)


# --------------------------------------------------------------------------- round 3 (C9)

def _rewrite_header(good, mutate):
    """The blob ``good`` with its header edited by ``mutate(header)``, re-addressed by content."""
    size = int.from_bytes(good[:8], "little")
    header = json.loads(good[8:8 + size])
    mutate(header)
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    raw = len(text).to_bytes(8, "little") + text + good[8 + size:]
    ref = hashlib.sha256(raw).hexdigest()
    return ref, raw


def test_c9_stale_temporaries_are_removed_at_the_next_put_and_live_ones_are_kept(tmp_path):
    import os
    import time

    store = LocalStore(tmp_path)
    store.put_tensor(np.zeros(2, np.float32))
    tensors = tmp_path / "tensors"
    old, fresh, other = tensors / "tmpold1", tensors / "tmplive2", tensors / "keep-me.txt"
    stale_other = tensors / "notatmp"
    for path in (old, fresh, other, stale_other):
        path.write_bytes(b"x")
    ancient = time.time() - 2 * 3600
    for path in (old, other, stale_other):
        os.utime(path, (ancient, ancient))
    young = time.time() - 600
    os.utime(fresh, (young, young))                       # a live writer's file: 10 minutes old
    # nothing is removed before a successful put
    assert old.exists()
    ref = store.put_tensor(np.arange(3, dtype=np.int32))["ref"]
    assert not old.exists()                               # older than an hour: swept
    assert fresh.exists()                                 # younger: possibly being written
    assert other.exists() and stale_other.exists()        # only tmp* files are ever touched
    assert np.array_equal(store.get_tensor(ref), np.arange(3, dtype=np.int32))


def test_c9_a_failed_put_removes_nothing(tmp_path, monkeypatch):
    import os
    import time

    store = LocalStore(tmp_path)
    store.put_tensor(np.zeros(2, np.float32))
    stale = tmp_path / "tensors" / "tmpold"
    stale.write_bytes(b"x")
    ancient = time.time() - 2 * 3600
    os.utime(stale, (ancient, ancient))
    import saturn_pub.store as module
    monkeypatch.setattr(module, "_publish", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError):
        store.put_tensor(np.ones(2, np.float32))
    assert stale.exists()
    monkeypatch.undo()
    store.put_tensor(np.ones(2, np.float32))
    assert not stale.exists()


def test_c9_a_directory_or_a_non_text_ref_is_a_clear_value_error(tmp_path):
    store = LocalStore(tmp_path)
    value = np.arange(4, dtype=np.float32)
    ref = store.put_tensor(value)["ref"]
    path = _blob(tmp_path, ref)
    path.unlink()
    path.mkdir()
    for call in (lambda: store.get_tensor(ref), lambda: store.tensor_info(ref),
                 lambda: store.put_tensor(value)):
        with pytest.raises(ValueError, match="is a directory"):
            call()
    assert store.has_tensor(ref) is False                 # is_file: a directory is not a blob
    assert store.has_tensor(ref, verify=True) is False
    path.rmdir()
    store.put_tensor(value)
    assert np.array_equal(store.get_tensor(ref), value)
    for bad in (5, None, b"0" * 64, ["0" * 64], 1.5):
        for call in (store.get_tensor, store.tensor_info, store.has_tensor,
                     lambda r: store.has_tensor(r, verify=True)):
            with pytest.raises(ValueError, match="SHA-256"):
                call(bad)


def test_c9_numpy_reads_are_writable_copies(tmp_path):
    store = LocalStore(tmp_path)
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    ref = store.put_tensor(value)["ref"]
    first = store.get_tensor(ref)
    assert first.flags.writeable and first.flags.owndata
    first[0, 0] = 99.0                                    # mutating a read does not touch the store
    second = store.get_tensor(ref)
    assert second[0, 0] == 0.0 and np.array_equal(second, value)
    empty = store.get_tensor(store.put_tensor(np.zeros((0, 3), np.float32))["ref"])
    assert empty.shape == (0, 3) and empty.flags.writeable
    scalar = store.get_tensor(store.put_tensor(np.float32(2.5))["ref"])
    assert scalar.flags.writeable
    scalar[()] = 1.0


def test_c9_header_types_are_checked_on_read(tmp_path):
    store = LocalStore(tmp_path)
    ref = store.put_tensor(np.arange(8, dtype=np.float32))["ref"]
    good = _blob(tmp_path, ref).read_bytes()

    def put(mutate):
        bad_ref, raw = _rewrite_header(good, mutate)
        _blob(tmp_path, bad_ref).write_bytes(raw)
        return bad_ref

    def entry(**changes):
        return lambda h: h["tensor"].update(changes)

    cases = {
        "float shape": entry(shape=[8.0]), "bool shape": entry(shape=[True]),
        "negative shape": entry(shape=[-8]), "text shape": entry(shape="8"),
        "nested shape": entry(shape=[[8]]), "mapping shape": entry(shape={"a": 8}),
        "null shape": entry(shape=None),
        "list dtype": entry(dtype=["F32"]), "number dtype": entry(dtype=5),
        "unknown dtype": entry(dtype="F33"), "null dtype": entry(dtype=None),
        "float offsets": entry(data_offsets=[0.0, 32.0]), "bool offsets": entry(data_offsets=[False, 32]),
        "text offsets": entry(data_offsets="ab"), "short offsets": entry(data_offsets=[0]),
        "mapping offsets": entry(data_offsets={"a": 0, "b": 32}),
        "mapping entry replaced by list": lambda h: h.update(tensor=[1, 2]),
        "metadata is text": lambda h: h.update(__metadata__="schema"),
        "metadata is a list": lambda h: h.update(__metadata__=[]),
        "extra section": lambda h: h.update(extra={}),
    }
    for name, mutate in cases.items():
        bad = put(mutate)
        with pytest.raises(ValueError, match="malformed|does not match") as caught:
            store.get_tensor(bad)
        assert type(caught.value) is ValueError, name        # never TypeError/AttributeError
        assert store.has_tensor(bad, verify=True) is False, name
    # the unmutated header (re-addressed, identical bytes) still reads
    same, raw = _rewrite_header(good, lambda h: None)
    assert np.array_equal(store.get_tensor(ref), np.arange(8, dtype=np.float32)) and same == ref


def test_a_deeply_nested_header_is_malformed_not_a_recursion_error(tmp_path):
    """Audit round 3: a header of '[' * 200000 escaped json.loads as a raw RecursionError."""
    store = LocalStore(tmp_path)
    store.put_tensor(np.arange(2, dtype=np.float32))          # creates the tensors directory
    header = b"[" * 200000
    raw = len(header).to_bytes(8, "little") + header
    ref = hashlib.sha256(raw).hexdigest()
    _blob(tmp_path, ref).write_bytes(raw)
    for call in (lambda: store.get_tensor(ref), lambda: store.tensor_info(ref)):
        with pytest.raises(ValueError, match="malformed") as caught:
            call()
        assert type(caught.value) is ValueError
    assert store.has_tensor(ref) is True                        # the file is there ...
    assert store.has_tensor(ref, verify=True) is False          # ... but not a sound tensor


def test_c9_a_corrupt_blob_hard_linked_into_a_shared_pool_is_refused_naming_the_pool(tmp_path):
    import os
    import stat

    from saturn_pub.shared_store import configure

    pool_root = tmp_path / "pool"
    configure(tmp_path / "runs", pool_root)
    store = LocalStore(tmp_path / "runs" / "one")
    value = np.arange(16, dtype=np.float32)
    info = store.put_tensor(value)
    blob = tmp_path / "runs" / "one" / "tensors" / f"{info['ref']}.safetensors"
    pooled = pool_root / "sha256" / info["ref"][:2] / info["ref"]
    assert os.path.samefile(blob, pooled)                 # shared by hard link
    good = blob.read_bytes()
    os.chmod(pooled, 0o644)
    pooled.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))  # same inode: both names are corrupt
    os.chmod(pooled, stat.S_IMODE(0o444))
    with pytest.raises(ValueError, match="pooled object .* is corrupt") as caught:
        store.put_tensor(value)
    assert str(pooled) in str(caught.value) and str(blob) in str(caught.value)
    assert "repair it in the shared pool" in str(caught.value)
    assert os.path.samefile(blob, pooled) and blob.read_bytes() != good   # nothing was replaced
    # repairing in the pool (remove the object and its links) lets the store put again
    os.chmod(pooled, 0o644)
    pooled.unlink()
    blob.unlink()
    again = store.put_tensor(value)
    assert again == info and np.array_equal(store.get_tensor(info["ref"]), value)
    # an unpooled corrupt blob is still repaired in place
    plain = LocalStore(tmp_path / "plain")
    ref = plain.put_tensor(value)["ref"]
    plain_blob = tmp_path / "plain" / "tensors" / f"{ref}.safetensors"
    plain_blob.write_bytes(b"junk")
    assert plain.put_tensor(value)["ref"] == ref and plain_blob.read_bytes() == good
