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


def test_byte_audit_uses_disk_and_preserves_durable_receipts(tmp_path):
    store = PayloadStore(tmp_path, resident_capacity_bytes=8)
    ref = store.put_bytes(b"abcdefgh", shape=(8,), dtype="uint8")
    store.hydrate(ref, native_consumer="native-test")
    before = {p.name: p.read_bytes() for p in store.receipts.glob("*.json")}
    result = store.audit(ref, expected_shape=(8,), expected_dtype="uint8", expected_bytes=8)
    assert result.payload == b"abcdefgh"
    assert result.source == "disk"
    assert result.receipt.authority_boundary["native_consumer"] is None
    assert result.receipt.checksum_verified
    with pytest.raises(PayloadIntegrityError, match="shape mismatch"):
        store.audit(ref, expected_shape=(4,))
    with pytest.raises(PayloadIntegrityError, match="dtype mismatch"):
        store.audit(ref, expected_dtype="float32")
    with pytest.raises(PayloadIntegrityError, match="byte count mismatch"):
        store.audit(ref, expected_bytes=4)
    store.object_path(ref).write_bytes(b"corrupt!")
    with pytest.raises(PayloadIntegrityError, match="checksum mismatch"):
        store.audit(ref)  # Must refuse even though the uncorrupted bytes are resident.
    with pytest.raises(PayloadCacheMiss):
        store.audit("cas://vm-payload/sha256/"+"f"*64)
    assert before == {p.name: p.read_bytes() for p in store.receipts.glob("*.json")}
    # Consumer hydration still publishes its failure evidence durably.
    with pytest.raises(PayloadIntegrityError):
        store.hydrate(ref, prefer_resident=False)
    assert len(list(store.receipts.glob("*.json"))) > len(before)


def test_existing_read_only_audit_is_deterministic_and_never_creates_store(tmp_path):
    missing = tmp_path / "absent"
    with pytest.raises(PayloadCacheError) as error:
        PayloadStore(missing, read_only=True)
    assert not missing.exists()
    assert error.value.audit_summary["mechanics_status"] == "invalid"
    writer = PayloadStore(tmp_path / "store")
    ref = writer.put_bytes(b"abcd", shape=(4,), dtype="uint8")
    audits = []
    before = {p.relative_to(writer.root): p.read_bytes() for p in writer.root.rglob("*") if p.is_file()}
    for _ in range(2):
        reader = PayloadStore(writer.root, read_only=True)
        reader.audit(ref)
        audits.append(reader.audit_summary())
        with pytest.raises(PayloadCacheError, match="cannot put"):
            reader.put_bytes(b"other")
        with pytest.raises(PayloadCacheError, match="requires audit"):
            reader.hydrate(ref)
    assert audits[0] == audits[1]
    assert audits[0]["unique_payloads"] == audits[0]["checksum_verified"] == audits[0]["geometry_verified"] == 1
    assert audits[0]["bytes_read"] == audits[0]["unique_verified_bytes"] == 4
    assert audits[0]["failures"] == audits[0]["misses"] == 0
    assert before == {p.relative_to(writer.root): p.read_bytes() for p in writer.root.rglob("*") if p.is_file()}


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
