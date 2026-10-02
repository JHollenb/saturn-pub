from __future__ import annotations

import json

import pytest

from mdb.vm_payload_cache import (
    PAYLOAD_AUTHORITY_CACHE_ONLY,
    PayloadCacheError,
    PayloadCacheMiss,
    PayloadIntegrityError,
    PayloadStore,
)


def test_physical_payload_roundtrip_measures_hit_miss_and_authority(tmp_path) -> None:
    store = PayloadStore(tmp_path / "payloads")
    ref = store.put_bytes(b"01234567", shape=(2,), dtype="float32", role="scene-latent")

    hydrated = store.hydrate(
        ref,
        expected_shape=(2,),
        expected_dtype="float32",
        native_consumer="scene-native-renderer",
    )
    assert hydrated.payload == b"01234567"
    assert hydrated.receipt.hit is True
    assert hydrated.receipt.miss is False
    assert hydrated.receipt.bytes_read == 8
    assert hydrated.receipt.checksum_verified is True
    assert hydrated.receipt.authority_boundary == {
        "payload_store": PAYLOAD_AUTHORITY_CACHE_ONLY,
        "native_consumer": "scene-native-renderer",
        "native_consumer_authority": "external-native-consumer",
        "semantic_equivalence": "not_assessed",
    }

    with pytest.raises(PayloadCacheMiss) as missing:
        store.hydrate("cas://vm-payload/sha256/" + "f" * 64)
    assert missing.value.receipt is not None
    assert missing.value.receipt.miss is True
    assert store.metrics["hits"] == 1
    assert store.metrics["misses"] == 1
    assert store.metrics["bytes_read"] == 8


def test_shape_dtype_and_size_are_verified_and_tampering_fails_closed(tmp_path) -> None:
    store = PayloadStore(tmp_path / "payloads")
    ref = store.put_bytes(b"\x00" * 8, shape=(2,), dtype="float32")

    with pytest.raises(PayloadIntegrityError, match="shape mismatch"):
        store.hydrate(ref, expected_shape=(4,))
    with pytest.raises(PayloadIntegrityError, match="dtype mismatch"):
        store.hydrate(ref, expected_dtype="float16")

    store.object_path(ref).write_bytes(b"tampered")
    with pytest.raises(PayloadIntegrityError, match="checksum mismatch"):
        store.hydrate(ref)
    assert store.metrics["integrity_failures"] >= 3


def test_put_is_immutable_and_receipt_is_durable(tmp_path) -> None:
    store = PayloadStore(tmp_path / "payloads")
    first = store.put_with_receipt(b"abc", role="source")
    second = store.put_with_receipt(b"abc", role="source")
    assert first.ref.ref == second.ref.ref
    assert second.receipt.cache_hit is True
    assert second.receipt.bytes_written == 0
    assert list((tmp_path / "payloads" / "objects").glob("*.bin"))
    receipt_path = next((tmp_path / "payloads" / "receipts").glob("*.json"))
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["schema"]


def test_bounded_resident_tier_reuses_bytes_and_falls_back_to_disk(tmp_path) -> None:
    store = PayloadStore(tmp_path / "payloads", resident_capacity_bytes=8)
    ref = store.put_bytes(b"abcd", shape=(4,), dtype="uint8")

    disk = store.hydrate(ref)
    resident = store.hydrate(ref)
    assert disk.source == "disk"
    assert disk.receipt.bytes_read == 4
    assert resident.source == "resident"
    assert resident.receipt.bytes_read == 0
    assert resident.receipt.resident_bytes_read == 4
    assert resident.receipt.resident is True
    assert resident.payload == disk.payload
    assert resident.ref.ref == ref.ref
    assert store.metrics["resident_hits"] == 1
    assert store.metrics["resident_bytes_read"] == 4

    assert store.evict(ref) == 1
    assert store.resident_snapshot() == ()
    hydrated_again = store.hydrate(ref)
    assert hydrated_again.source == "disk"
    assert hydrated_again.receipt.bytes_read == 4
    assert store.metrics["bytes_read"] == 8
    assert store.metrics["resident_evictions"] == 1


def test_resident_lru_is_deterministic_and_pins_are_respected(tmp_path) -> None:
    store = PayloadStore(tmp_path / "payloads", resident_capacity_bytes=8)
    first = store.put_bytes(b"1111", shape=(4,), dtype="uint8")
    second = store.put_bytes(b"2222", shape=(4,), dtype="uint8")
    third = store.put_bytes(b"3333", shape=(4,), dtype="uint8")

    store.hydrate(first)
    store.hydrate(second)
    assert store.resident_snapshot() == (first.sha256, second.sha256)
    store.hydrate(first)  # Promote first to most-recently-used.
    store.hydrate(third)
    assert store.resident_snapshot() == (first.sha256, third.sha256)

    assert store.pin(first) is True
    store.hydrate(second)
    assert store.resident_snapshot() == (first.sha256, second.sha256)
    with pytest.raises(PayloadCacheError, match="pinned"):
        store.evict(first)
    assert store.unpin(first) is True
    assert store.evict(first) == 1
