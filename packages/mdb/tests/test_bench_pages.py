import torch
from mdb.bench_pages import BenchPages


def test_durable_native_layout_and_private_resume(tmp_path):
    pages = BenchPages(tmp_path, device="cpu", budget=4096)
    original = {
        "transpose": torch.arange(24).reshape(4, 6).T,
        "channels_last": torch.arange(48).reshape(1, 3, 4, 4).contiguous(memory_format=torch.channels_last),
        "expanded": torch.arange(4).reshape(1, 4).expand(3, 4),
        "offset": torch.arange(32)[4:20:2],
    }
    encoded = pages.encode(original)
    restored = pages.decode(encoded, private=True)
    for key, value in original.items():
        assert torch.equal(restored[key], value)
        assert restored[key].stride() == value.stride()
    restored["transpose"][0, 0] = 99
    assert original["transpose"][0, 0] == 0
    assert pages.decode(encoded)["transpose"][0, 0] == 0
    # Earlier v2 descriptors omitted layout and remain readable.
    del encoded["transpose"]["$geometry"]["stride"]
    assert torch.equal(pages.decode(encoded)["transpose"], original["transpose"])


def test_chunk_dirtiness_and_explicit_readonly_reuse(tmp_path):
    pages = BenchPages(tmp_path, device="cpu", budget=4096, page_bytes=256)
    value = torch.arange(256, dtype=torch.float32)
    first = pages.encode({"value": value})
    assert len(first["value"]["$chunks"]) == 4
    clone = value.clone(); clone[70] += 1
    second = pages.encode({"value": clone})
    before, after = first["value"]["$chunks"], second["value"]["$chunks"]
    assert [a == b for a, b in zip(before, after)] == [True, False, True, True]
    changed = pages.stats["encoded_bytes"]
    shared = pages.encode({"value": value}, previous_state={"value": value}, previous_encoded=first,
        readonly_ids={id(value)})
    assert shared == first and pages.stats["encoded_bytes"] == changed
    # Ordinary calls never trust inference tensor identity/version counters.
    with torch.inference_mode():
        mutable = torch.zeros(64)
        old = pages.encode(mutable)
        mutable[0] = 1
        assert pages.encode(mutable) != old
    fresh = BenchPages(tmp_path, device="cpu", budget=4096)
    assert torch.equal(fresh.decode(second)["value"], clone)
    private = fresh.decode(second, private=True)["value"]
    private.zero_()
    assert torch.equal(fresh.decode(second)["value"], clone)


def test_legacy_whole_tensor_reader(tmp_path):
    pages = BenchPages(tmp_path, device="cpu", budget=4096)
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    raw = value.view(torch.uint8).numpy().tobytes()
    ref = pages.store.put_bytes(raw, shape=(len(raw),), dtype="torch.uint8", role="bench-family-page")
    legacy = {"$tensor": ref.to_dict(), "$geometry": {"shape": [3, 4], "dtype": "torch.float32", "device": "cpu"}}
    assert torch.equal(pages.decode(legacy), value)
    assert pages.encode(value, previous_state=value, previous_encoded=legacy,
        readonly_ids={id(value)}, verify_readonly=True) == legacy


def test_corrupt_chunk_geometry_rejected(tmp_path):
    import pytest
    from mdb.bench import BenchError
    pages = BenchPages(tmp_path, device="cpu", budget=4096, page_bytes=256)
    value = pages.encode(torch.zeros(128))
    value["$chunks"].pop()
    with pytest.raises(BenchError, match="chunked"):
        pages.decode(value)


def test_prior_identity_requires_readonly_declaration_and_can_be_audited(tmp_path):
    import pytest
    from mdb.bench import BenchError
    pages = BenchPages(tmp_path, device="cpu", budget=4096)
    with torch.inference_mode():
        value = torch.zeros(64)
        old = pages.encode(value)
        value[0] = 1
        assert pages.encode(value, previous_state=value, previous_encoded=old) != old
        with pytest.raises(BenchError, match="read-only"):
            pages.encode(value, previous_state=value, previous_encoded=old,
                readonly_ids={id(value)}, verify_readonly=True)


def test_memory_counts_retained_backing_as_well_as_visible_branches(tmp_path):
    pages = BenchPages(tmp_path, device="cpu", budget=4096, page_bytes=256)
    first = pages.encode(torch.arange(128, dtype=torch.float32))
    second = pages.encode(torch.arange(128, dtype=torch.float32)+1)
    pages.decode(first)
    working = pages.decode(second)
    memory = pages.inspect({"visible": second}, working)
    assert memory["retained_resident_pages"] == 1
    assert memory["unique_backed_bytes"] == 1024
    assert memory["resident_bytes"] == 1024
