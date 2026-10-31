from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mdb.forecast_transaction as forecast_module
from mdb.forecast_transaction import (
    ForecastPayload,
    ForecastTransaction,
    ForecastTransactionError,
    load_forecast_array,
    load_forecast_tree,
    verify_transaction,
)


def transaction(root, *, storage=None, **kwargs):
    if storage is not None:
        kwargs["array_storage"] = storage
    return ForecastTransaction(root, model_identity="tiny-numeric-contract", input_identity="raw",
        source_identity={"fixture": "storage-mechanics"}, frontier={"kind": "raw-input"}, **kwargs)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def descriptors(value):
    if isinstance(value, dict):
        if value.get("kind") == "array":
            yield value
        else:
            for child in value.values():
                yield from descriptors(child)
    elif isinstance(value, list):
        for child in value:
            yield from descriptors(child)


def predict(tx, values=None):
    values = np.arange(6, dtype=np.float64).reshape(2, 3) if values is None else values
    return tx.predict("p", lambda context: {"predictions": {"a": values, "again": values.copy()},
        "payloads": {"edit": ForecastPayload(values, dtype="float32")}})


def array_descriptor(record):
    return record["predictions"]["items"]["a"]


MAGIC = b"MDBNPYBUNDLE1\n"


def test_bundle_framing_preserves_original_npy_and_content_seals(tmp_path):
    value = np.arange(6, dtype=np.float64).reshape(2, 3)
    tx = transaction(tmp_path / "tx")
    record = predict(tx, value)
    leaves = list(descriptors(record))
    assert len({leaf["path"] for leaf in leaves}) == 1
    assert len({leaf["sha256"] for leaf in leaves}) == 2
    assert record["predictions"]["items"]["a"] == record["predictions"]["items"]["again"]
    originals = {digest(raw): raw for raw in (npy_bytes(value), npy_bytes(value.astype(np.float32)))}
    packed = MAGIC + b"".join(originals[key] for key in sorted(originals))
    path = tx.root / leaves[0]["path"]
    assert path.read_bytes() == packed
    assert leaves[0]["path"] == "arrays/" + digest(packed) + ".npyb"
    offset = len(MAGIC)
    for identifier in sorted(originals):
        raw = originals[identifier]
        same_leaf = [leaf for leaf in leaves if leaf["sha256"] == identifier]
        for leaf in same_leaf:
            assert leaf["storage"] == "bundle-v1" and leaf["bundle_sha256"] == digest(packed)
            assert leaf["offset"] == offset and leaf["nbytes"] == len(raw)
            assert digest(packed[leaf["offset"]:leaf["offset"] + leaf["nbytes"]]) == identifier
            array = load_forecast_array(tx.root, leaf)
            assert leaf["content_sha256"] == digest(array.tobytes())
        offset += len(raw)
    assert offset == len(packed)
    tx.close()


def test_bundle_dedup_reuses_closed_descriptors_across_frontier_forecast_and_observation(tmp_path, monkeypatch):
    tx = transaction(tmp_path / "tx", weights={"W": np.eye(2, dtype=np.float32)})
    value = np.arange(5, dtype=np.uint64)
    frontier = tx.frontier("input", value)
    original = frontier["value"]
    before = set((tx.root / "arrays").iterdir())
    original_link = os.link
    array_links = []

    def linked(source, destination, *args, **kwargs):
        if Path(destination).parent.name == "arrays":
            array_links.append(Path(destination))
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(forecast_module.os, "link", linked)
    record = tx.predict("same", lambda context: {"predictions": {"value": value}})
    assert record["predictions"]["items"]["value"] == original
    assert set((tx.root / "arrays").iterdir()) == before and array_links == []
    tx.seal()
    tx.execute("same", lambda context: value)
    observed = json.loads((tx.root / "observations/same.json").read_text())
    assert observed["value"] == original
    assert set((tx.root / "arrays").iterdir()) == before and array_links == []
    tx.close()
    assert verify_transaction(tx.root)["verified"]


def test_public_tree_reader_reads_each_verified_bundle_once_per_call(tmp_path, monkeypatch):
    tx = transaction(tmp_path / "tx")
    record = predict(tx)
    target = (tx.root / array_descriptor(record)["path"]).resolve()
    original_open = Path.open
    reads = []

    def opened(path, *args, **kwargs):
        if path.resolve() == target:
            reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    first = load_forecast_tree(tx.root, record["predictions"])
    assert len(reads) == 1
    second = load_forecast_tree(tx.root, record["predictions"])
    assert len(reads) == 2
    np.testing.assert_array_equal(first["a"], second["a"])
    assert first["a"] is not second["a"] and not np.shares_memory(first["a"], second["a"])
    with pytest.raises(ValueError):
        second["again"].setflags(write=True)
    tx.close()


@pytest.mark.parametrize("field,value", [
    ("offset", -1), ("offset", 0), ("offset", True), ("offset", 1.5),
    ("offset", "13"), ("offset", 2**80), ("nbytes", -1), ("nbytes", 0),
    ("nbytes", True), ("nbytes", 1.5), ("nbytes", "128"), ("nbytes", 2**80),
    ("bundle_sha256", "0" * 64), ("sha256", "0" * 64),
    ("content_sha256", "0" * 64), ("shape", [1, 6]), ("dtype", "float32"),
    ("storage", "unknown-layout"),
])
def test_packed_reader_refuses_malformed_offsets_geometry_and_seals(tmp_path, field, value):
    tx = transaction(tmp_path / "tx")
    descriptor = array_descriptor(predict(tx))
    altered = {**descriptor, field: value}
    with pytest.raises(ForecastTransactionError):
        load_forecast_array(tx.root, altered)
    tx.close()


@pytest.mark.parametrize("storage", ["files-v1", "bundle-v1"])
@pytest.mark.parametrize("path_kind", ["traversal", "absolute", "symlink-file", "symlink-parent", "missing"])
def test_public_reader_refuses_unsafe_paths_and_missing_artifacts(tmp_path, storage, path_kind):
    tx = transaction(tmp_path / "tx", storage=storage)
    descriptor = array_descriptor(predict(tx))
    original = tx.root / descriptor["path"]
    if path_kind == "traversal":
        descriptor["path"] = "../" + tx.root.name + "/" + descriptor["path"]
    elif path_kind == "absolute":
        descriptor["path"] = str(original.resolve())
    elif path_kind == "symlink-file":
        target = tmp_path / "external"
        target.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(target)
    elif path_kind == "symlink-parent":
        outside = tmp_path / "external-arrays"
        original.parent.rename(outside)
        original.parent.symlink_to(outside, target_is_directory=True)
    else:
        original.unlink()
    with pytest.raises(ForecastTransactionError):
        load_forecast_array(tx.root, descriptor)
    tx.close()


@pytest.mark.parametrize("corruption", ["truncate", "flip-leaf", "flip-header"])
def test_bundle_corruption_blocks_public_read_offline_verification_and_native_release(tmp_path, corruption):
    tx = transaction(tmp_path / "tx")
    descriptor = array_descriptor(predict(tx))
    tx.seal()
    path = tx.root / descriptor["path"]
    raw = bytearray(path.read_bytes())
    if corruption == "truncate":
        raw = raw[:-1]
    else:
        index = descriptor["offset"] + descriptor["nbytes"] - 1 if corruption == "flip-leaf" else 0
        raw[index] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ForecastTransactionError):
        load_forecast_array(tx.root, descriptor)
    with pytest.raises(ForecastTransactionError):
        verify_transaction(tx.root)
    executed = []
    with pytest.raises(ForecastTransactionError):
        tx.execute("refused", lambda context: executed.append(True))
    assert not executed and tx.phase == "failed"
    tx.close()


def test_reader_refuses_rehashed_bad_magic_and_valid_npy_embedded_inside_another_leaf(tmp_path):
    tx = transaction(tmp_path / "tx")
    value = np.array([1., 2.], dtype=np.float32)
    original = array_descriptor(predict(tx, value))
    packed = bytearray((tx.root / original["path"]).read_bytes())
    packed[0] ^= 1
    bad_header = {**original, "path": "arrays/" + digest(packed) + ".npyb", "bundle_sha256": digest(packed)}
    (tx.root / bad_header["path"]).write_bytes(packed)
    with pytest.raises(ForecastTransactionError):
        load_forecast_array(tx.root, bad_header)
    inner = npy_bytes(value)
    outer = npy_bytes(np.frombuffer(inner, dtype=np.uint8))
    packed = MAGIC + outer
    embedded = {**original, "path": "arrays/" + digest(packed) + ".npyb", "bundle_sha256": digest(packed),
                "offset": len(packed) - len(inner), "nbytes": len(inner), "sha256": digest(inner),
                "shape": list(value.shape), "dtype": str(value.dtype), "content_sha256": digest(value.tobytes())}
    (tx.root / embedded["path"]).write_bytes(packed)
    # Hash/geometry/range validation alone would accept this valid inner NPY.
    assert packed[embedded["offset"]:] == inner
    with pytest.raises(ForecastTransactionError):
        load_forecast_array(tx.root, embedded)
    tx.close()


def test_bundle_byte_budget_refuses_large_pending_pack_without_publishing_forecast(tmp_path):
    tx = transaction(tmp_path / "tx", budgets={"max_bytes": 16 * 1024})
    with pytest.raises(ForecastTransactionError, match="byte|budget"):
        tx.predict("large", lambda context: {"predictions": np.zeros(32 * 1024, dtype=np.uint8)})
    assert tx.phase == "failed"
    assert not (tx.root / "forecasts/large.json").exists()
    assert not (tx.root / "seal.json").exists()
    assert not list(tx.root.rglob(".publish-*"))
    tx.close()


def test_bundle_is_fsynced_and_linked_before_referencing_forecast_metadata(tmp_path, monkeypatch):
    tx = transaction(tmp_path / "tx")
    original_link, original_fsync = os.link, os.fsync
    events = []

    def synced(fd):
        events.append("sync-file" if stat.S_ISREG(os.fstat(fd).st_mode) else "sync-directory")
        return original_fsync(fd)

    def linked(source, destination, *args, **kwargs):
        destination = Path(destination)
        if destination.parent.name == "arrays":
            events.append("link-bundle")
        elif destination.parent.name == "forecasts":
            events.append("link-forecast")
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(forecast_module.os, "link", linked)
    monkeypatch.setattr(forecast_module.os, "fsync", synced)
    predict(tx)
    bundle_index = events.index("link-bundle")
    assert events[bundle_index - 1:bundle_index + 2] == ["sync-file", "link-bundle", "sync-directory"]
    assert events.index("link-forecast") > bundle_index + 1
    tx.close()


def test_transaction_root_and_every_new_directory_entry_are_durable_before_children_publish(tmp_path, monkeypatch):
    original_mkdir, original_fsync, original_link = os.mkdir, os.fsync, os.link
    events = []

    def inode(value):
        return value.st_dev, value.st_ino

    def made(path, *args, **kwargs):
        result = original_mkdir(path, *args, **kwargs)
        created = Path(path).resolve()
        events.append(("mkdir", created, inode(created.parent.stat())))
        return result

    def synced(fd):
        result = original_fsync(fd)
        information = os.fstat(fd)
        if stat.S_ISDIR(information.st_mode):
            events.append(("sync-directory", inode(information)))
        return result

    def linked(source, destination, *args, **kwargs):
        result = original_link(source, destination, *args, **kwargs)
        events.append(("publish", Path(destination).resolve()))
        return result

    monkeypatch.setattr(forecast_module.os, "mkdir", made)
    monkeypatch.setattr(forecast_module.os, "fsync", synced)
    monkeypatch.setattr(forecast_module.os, "link", linked)
    tx = transaction(tmp_path / "new-parent" / "nested" / "tx")
    tx.frontier("roots", np.array([1], dtype=np.float32))
    predict(tx)
    tx.seal()
    tx.execute("actual", lambda context: np.array([2], dtype=np.float32))
    tx.close()
    created = {event[1] for event in events if event[0] == "mkdir"}
    expected = {tx.root, tx.root.parent, tx.root.parent.parent}
    expected.update(tx.root / name for name in ("arrays", "frontiers", "forecasts", "observations", "events",
                                               "custody", "custody/receipts"))
    assert expected <= created
    for index, event in enumerate(events):
        if event[0] != "mkdir":
            continue
        directory, parent_inode = event[1:]
        # A later fsync of the child itself does not persist its name in the
        # parent. Require the correct parent inode before any child publication.
        next_child = next((position for position in range(index + 1, len(events))
            if events[position][0] in {"mkdir", "publish"}
            and events[position][1].is_relative_to(directory)), len(events))
        assert any(item[0] == "sync-directory" and item[1] == parent_inode
                   for item in events[index + 1:next_child]), f"unsynced parent entry for {directory}"


@pytest.mark.parametrize("storage", ["files-v1", "bundle-v1"])
@pytest.mark.parametrize("kind,dtype", [("u", "uint24"), ("f", "float24")])
def test_rehashed_counterfeit_npy_dtype_is_refused_by_loader_and_stdlib_verifier(tmp_path, storage, kind, dtype):
    tx = transaction(tmp_path / "tx", storage=storage)
    record = tx.predict("p", lambda context: {"predictions": np.ones(1, dtype=np.uint8)})
    original = record["predictions"]
    tx.seal()
    tx.close()
    # This is well-framed NPY with three payload bytes and a forbidden numeric
    # item size. All file/content/global-seal hashes below are recomputed, so a
    # checksum mismatch cannot substitute for dtype-format validation.
    header_text = repr({"descr": "<" + kind + "3", "fortran_order": False, "shape": (1,)})
    header_length = ((10 + len(header_text) + 1 + 63) // 64) * 64 - 10
    header = (header_text + " " * (header_length - len(header_text) - 1) + "\n").encode()
    leaf = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\0\0\0"
    descriptor = {"kind": "array", "sha256": digest(leaf), "shape": [1], "dtype": dtype,
                  "content_sha256": digest(b"\0\0\0")}
    packed = MAGIC + leaf if storage == "bundle-v1" else leaf
    if storage == "bundle-v1":
        descriptor.update(storage=storage, bundle_sha256=digest(packed), offset=len(MAGIC), nbytes=len(leaf))
    descriptor["path"] = "arrays/" + digest(packed) + (".npyb" if storage == "bundle-v1" else ".npy")
    (tx.root / descriptor["path"]).write_bytes(packed)
    (tx.root / original["path"]).unlink()
    record["predictions"] = descriptor
    forecast_raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    (tx.root / "forecasts/p.json").write_bytes(forecast_raw)
    manifest = json.loads((tx.root / "seal.json").read_text())
    del manifest["files"][original["path"]]
    manifest["files"][descriptor["path"]] = {"sha256": digest(packed), "bytes": len(packed)}
    manifest["files"]["forecasts/p.json"] = {"sha256": digest(forecast_raw), "bytes": len(forecast_raw)}
    seal_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    (tx.root / "seal.json").write_bytes(seal_raw)
    for relative, reference in manifest["files"].items():
        raw = (tx.root / relative).read_bytes()
        assert digest(raw) == reference["sha256"] and len(raw) == reference["bytes"]
    with pytest.raises(ForecastTransactionError, match="item size|dtype"):
        load_forecast_array(tx.root, descriptor)
    code = """
import builtins, importlib.util, pathlib, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'jax', 'tensorflow', 'isa_lab'}:
        raise RuntimeError('stdlib validation imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
source = pathlib.Path(sys.argv[1])
package_spec = importlib.util.spec_from_file_location('mdb', source.with_name('__init__.py'),
    submodule_search_locations=[str(source.parent)])
package = importlib.util.module_from_spec(package_spec)
sys.modules['mdb'] = package
package_spec.loader.exec_module(package)
spec = importlib.util.spec_from_file_location('mdb.forecast_transaction', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
try:
    module.verify_transaction(sys.argv[2], expected_seal_sha256=sys.argv[3])
except module.ForecastTransactionError as error:
    assert 'item size' in str(error) or 'dtype' in str(error), str(error)
else:
    raise AssertionError('counterfeit unsupported dtype was admitted')
assert not set(sys.modules) & {'numpy', 'torch', 'jax', 'tensorflow', 'isa_lab'}
"""
    completed = subprocess.run([sys.executable, "-I", "-c", code,
        str(Path(forecast_module.__file__).resolve()), str(tx.root), digest(seal_raw)],
        capture_output=True, text=True, timeout=20, env=dict(os.environ))
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("failed_target", ["arrays", "forecasts"])
def test_atomic_publication_failure_never_exposes_a_partial_forecast(tmp_path, monkeypatch, failed_target):
    tx = transaction(tmp_path / "tx")
    original_link = os.link
    value = np.arange(4, dtype=np.float64)

    def crash(source, destination, *args, **kwargs):
        if Path(destination).parent.name == failed_target:
            raise OSError("injected publication failure")
        return original_link(source, destination, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(forecast_module.os, "link", crash)
        with pytest.raises(OSError, match="injected publication"):
            tx.predict("p", lambda context: {"predictions": value})
    assert tx.phase == "failed" and tx.report()["forecasts"] == []
    assert not (tx.root / "forecasts/p.json").exists() and not (tx.root / "seal.json").exists()
    assert not list(tx.root.rglob(".publish-*"))
    bundles = list((tx.root / "arrays").glob("*.npyb"))
    if failed_target == "arrays":
        assert bundles == []
    else:
        assert len(bundles) == 1
        assert bundles[0].read_bytes() == MAGIC + npy_bytes(value)
        assert bundles[0].stem == digest(bundles[0].read_bytes())
    tx.close()


def test_default_matches_explicit_bundle_and_preserves_legacy_leaf_identities(tmp_path):
    value = np.arange(12, dtype=np.float64).reshape(3, 4)
    records = {}
    for label, storage in (("default", None), ("explicit", "bundle-v1"), ("legacy", "files-v1")):
        tx = transaction(tmp_path / label, storage=storage, weights={"W": value.copy()})
        record = predict(tx, value)
        records[label] = record
        tx.seal()
        tx.execute("actual", lambda context: context.predictions["p"]["a"] + 1)
        np.testing.assert_array_equal(tx.observation("actual"), value + 1)
        tx.close()
        assert verify_transaction(tx.root, expected_seal_sha256=tx.report()["seal_sha256"],
                                  expected_event_head=tx.report()["event_head"])["verified"]
        if label != "legacy":
            assert tx.declarations["array_storage"] == "bundle-v1"
            assert list((tx.root / "arrays").glob("*.npyb"))
            assert not list((tx.root / "arrays").glob("*.npy"))
    assert records["default"] == records["explicit"]
    for packed, legacy in zip(descriptors(records["default"]), descriptors(records["legacy"]), strict=True):
        assert packed["storage"] == "bundle-v1"
        assert {key: packed[key] for key in ("sha256", "content_sha256", "shape", "dtype")} == {
            key: legacy[key] for key in ("sha256", "content_sha256", "shape", "dtype")}


def test_explicit_legacy_keeps_original_npy_descriptor_bytes(tmp_path):
    value = np.array([[1.25, -2.5], [0., 7.]], dtype=np.float32)
    raw = npy_bytes(value)
    expected = {"kind": "array", "path": "arrays/" + digest(raw) + ".npy", "sha256": digest(raw),
                "content_sha256": digest(value.tobytes()), "shape": [2, 2], "dtype": "float32"}
    tx = transaction(tmp_path / "legacy", storage="files-v1")
    record = predict(tx, value)
    assert "array_storage" not in tx.declarations
    assert array_descriptor(record) == expected
    assert (tx.root / expected["path"]).read_bytes() == raw
    np.testing.assert_array_equal(load_forecast_array(tx.root, expected), value)
    tx.seal()
    tx.close()
    assert verify_transaction(tx.root)["verified"]


@pytest.mark.parametrize("storage", ["files-v1", "bundle-v1"])
def test_public_reader_and_callback_views_are_fresh_immutable_and_numerically_equal(tmp_path, storage):
    value = np.arange(12, dtype=np.float64).reshape(3, 4)[::-1, ::2]
    expected = value.copy()
    tx = transaction(tmp_path / storage, storage=storage)
    tx.frontier("roots", {"input": value})
    record = predict(tx, value)
    value[:] = -999
    descriptor = array_descriptor(record)
    first = load_forecast_array(tx.root, descriptor)
    second = load_forecast_array(tx.root, descriptor)
    np.testing.assert_array_equal(first, expected)
    assert first.dtype == expected.dtype and first.shape == expected.shape
    assert first is not second and not np.shares_memory(first, second)
    for array in (first, second, tx.prediction("p", path=("a",)), tx.payload("p", "edit")):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 42
    mutable = first.copy()
    mutable[:] = 123
    np.testing.assert_array_equal(load_forecast_array(tx.root, descriptor), expected)
    tx.seal()
    tx.execute("actual", lambda context: {"head": context.predictions["p"]["a"] + 1})
    np.testing.assert_array_equal(tx.observation("actual", path=("head",)), expected + 1)
    tx.close()


def test_bundles_reduce_physical_files_and_fsync_without_relaxing_durability(tmp_path, monkeypatch):
    original_fsync = os.fsync
    calls = []

    def synced(fd):
        calls.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr(forecast_module.os, "fsync", synced)
    values = {str(index): np.array([index, index + .25], dtype=np.float32) for index in range(40)}
    measurements = {}
    for storage in ("files-v1", "bundle-v1"):
        tx = transaction(tmp_path / storage, storage=storage)
        before = len(calls)
        record = tx.predict("many", lambda context: {"predictions": values})
        files = {descriptor["path"] for descriptor in descriptors(record)}
        measurements[storage] = {"fsync": len(calls) - before, "array_files": len(files)}
        for descriptor in descriptors(record):
            assert (tx.root / descriptor["path"]).is_file()
        tx.seal()
        tx.close()
        assert verify_transaction(tx.root)["verified"]
    assert measurements["files-v1"]["array_files"] == 40
    assert measurements["bundle-v1"]["array_files"] < measurements["files-v1"]["array_files"]
    assert measurements["bundle-v1"]["fsync"] >= 2
    assert measurements["bundle-v1"]["fsync"] < measurements["files-v1"]["fsync"]


def test_file_budget_counts_physical_bundle_files_and_offline_bounds_still_apply(tmp_path):
    values = {str(index): np.array([index], dtype=np.int64) for index in range(40)}
    bundled = transaction(tmp_path / "bundle", budgets={"max_files": 20})
    bundled.predict("many", lambda context: {"predictions": values})
    bundled.seal()
    bundled.close()
    assert verify_transaction(bundled.root)["verified"]
    files = [path for path in bundled.root.rglob("*") if path.is_file()]
    with pytest.raises(ForecastTransactionError, match="bound"):
        verify_transaction(bundled.root, max_files=len(files) - 1)
    with pytest.raises(ForecastTransactionError, match="bound"):
        verify_transaction(bundled.root, max_bytes=sum(path.stat().st_size for path in files) - 1)
    legacy = transaction(tmp_path / "legacy", storage="files-v1", budgets={"max_files": 20})
    with pytest.raises(ForecastTransactionError, match="budget"):
        legacy.predict("many", lambda context: {"predictions": values})
        legacy.seal()
    assert not (legacy.root / "seal.json").exists()


@pytest.mark.parametrize("storage", ["files-v1", "bundle-v1"])
def test_legacy_and_bundle_verify_cli_stays_framework_and_numpy_free(tmp_path, storage):
    tx = transaction(tmp_path / storage, storage=storage)
    predict(tx)
    seal = tx.seal()
    tx.close()
    code = """
import builtins, importlib.util, pathlib, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'jax', 'tensorflow', 'isa_lab'}:
        raise RuntimeError('offline metadata CLI imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
source = pathlib.Path(sys.argv[1])
package_spec = importlib.util.spec_from_file_location('mdb', source.with_name('__init__.py'),
    submodule_search_locations=[str(source.parent)])
package = importlib.util.module_from_spec(package_spec)
sys.modules['mdb'] = package
package_spec.loader.exec_module(package)
spec = importlib.util.spec_from_file_location('mdb.forecast_transaction', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.main(['verify', sys.argv[2], '--expected-seal', sys.argv[3]]) == 0
assert not set(sys.modules) & {'numpy', 'torch', 'jax', 'tensorflow', 'isa_lab'}
"""
    completed = subprocess.run([sys.executable, "-I", "-c", code,
        str(Path(forecast_module.__file__).resolve()), str(tx.root), seal["seal_sha256"]],
        capture_output=True, text=True, timeout=20, env=dict(os.environ))
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["verified"]
