from __future__ import annotations

import json

import pytest

from mdb.vm_cache import (
    VM_CACHE_PLANES,
    VM_CACHE_REF_PREFIX,
    VMCacheEntry,
    VMCacheError,
    VMCachePlane,
    VMCachePointer,
    VMCacheStore,
    build_address_cache_manifest,
    build_operator_cache_manifest,
    build_renderer_cache_manifest,
    build_source_cache_manifest,
    diff_vm_cache_manifests,
    main,
)


class _TensorLike:
    shape = (1, 2)


def _identities() -> dict[str, dict[str, str]]:
    return {
        "model": {"family": "fixture", "revision": "r1"},
        "input": {"prompt": "a bounded circuit", "split": "dev"},
        "abi": {"name": "fixture-vm", "version": "1"},
    }


def _entry(entry_id: str, *, offset: int = 0) -> VMCacheEntry:
    return VMCacheEntry(
        entry_id=entry_id,
        role="route",
        pointers=(
            VMCachePointer.from_source_handle(
                f"thott://fixture/source/{entry_id}",
                role="source",
                metadata={"offset": offset},
            ),
            VMCachePointer.from_artifact_ref(
                f"cas://sha256/{offset + 1:064x}",
                role="descriptor",
                sha256=f"{offset + 2:064x}",
                bytes=64,
                shape=(1, 4),
                dtype="float32",
            ),
        ),
        metadata={"address": f"token:{offset}", "operator": "mix"},
        similarity={"metric": "cosine", "score": 0.9 - offset / 100},
    )


def _source_manifest(*, entries: tuple[VMCacheEntry, ...] = ()):
    ids = _identities()
    return build_source_cache_manifest(
        manifest_id="source-fixture",
        model_identity=ids["model"],
        input_identity=ids["input"],
        abi_identity=ids["abi"],
        parent_lineage=("cas://sha256/" + "a" * 64,),
        entries=entries,
        similarity={"metric": "fingerprint", "score": 1.0},
    )


def test_plane_manifest_is_content_addressed_and_order_invariant() -> None:
    first = _source_manifest(entries=(_entry("b", offset=1), _entry("a")))
    second = build_source_cache_manifest(
        manifest_id="source-fixture",
        model_identity={"revision": "r1", "family": "fixture"},
        input_identity={"split": "dev", "prompt": "a bounded circuit"},
        abi_identity={"version": "1", "name": "fixture-vm"},
        parent_lineage=("cas://sha256/" + "a" * 64,),
        entries=(_entry("a"), _entry("b", offset=1)),
        similarity={"score": 1.0, "metric": "fingerprint"},
    )

    assert first.plane == VMCachePlane.SOURCE.value
    assert first.fingerprint == second.fingerprint
    assert first.to_dict()["content_sha256"] == first.fingerprint
    serialized = json.dumps(first.to_dict()).lower()
    assert all(token not in serialized for token in ("payload", "tensor", "weights", "latents"))
    assert first.summary()["pointer_count"] == 4
    assert first.parent_fingerprint == "a" * 64


def test_pointer_requires_exact_external_identity_and_rejects_payloads() -> None:
    with pytest.raises(VMCacheError, match="filesystem path"):
        VMCachePointer.from_artifact_ref("/tmp/tensor.pt")
    with pytest.raises(VMCacheError, match="payload field"):
        VMCachePointer.from_source_handle("thott://fixture/source", metadata={"raw_bytes": b"x"})
    with pytest.raises(VMCacheError, match="payload field"):
        VMCacheEntry(entry_id="bad", metadata={"payload": [1, 2, 3]})
    with pytest.raises(VMCacheError, match="tensor-like payload"):
        VMCacheEntry(entry_id="tensor", metadata={"descriptor": _TensorLike()})
    descriptive = VMCacheEntry(
        entry_id="descriptor",
        metadata={"values": ["source", "target"], "array": "shape-only"},
    )
    assert descriptive.metadata["values"] == ["source", "target"]


def test_four_planes_remain_explicit_and_diffs_do_not_claim_equivalence() -> None:
    ids = _identities()
    kwargs = {
        "manifest_id": "fixture",
        "model_identity": ids["model"],
        "input_identity": ids["input"],
        "abi_identity": ids["abi"],
        "entries": (_entry("a"),),
    }
    source = build_source_cache_manifest(**kwargs)
    address = build_address_cache_manifest(**kwargs)
    operator = build_operator_cache_manifest(**kwargs)
    renderer = build_renderer_cache_manifest(**kwargs)
    changed = build_source_cache_manifest(**{**kwargs, "entries": (_entry("a", offset=2),)})

    assert {source.plane, address.plane, operator.plane, renderer.plane} == set(VM_CACHE_PLANES)
    cross_plane = diff_vm_cache_manifests(source, address)
    assert cross_plane["identity_changes"]["plane"] == {"left": "source", "right": "address"}
    assert cross_plane["semantic_equivalence"] == "not_assessed"
    assert diff_vm_cache_manifests(source, changed)["entries"]["changed"][0]["entry_id"] == "a"

    left_pointer = VMCachePointer.from_source_handle("thott://fixture/shared")
    right_pointer = VMCachePointer.from_source_handle("thott://fixture/right-only")
    left = build_source_cache_manifest(
        **{**kwargs, "entries": (VMCacheEntry(entry_id="p", pointers=(left_pointer,)),)}
    )
    right = build_source_cache_manifest(
        **{
            **kwargs,
            "entries": (VMCacheEntry(entry_id="p", pointers=(left_pointer, right_pointer)),),
        }
    )
    assert (
        diff_vm_cache_manifests(left, right)["structural_similarity"]["shared_pointer_jaccard"]
        == 0.5
    )


def test_store_roundtrip_lineage_and_lightweight_cli(tmp_path, capsys) -> None:
    store = VMCacheStore(tmp_path / "vm-cache")
    base = _source_manifest(entries=(_entry("a"),))
    child = base.child(manifest_id="source-fixture-child", entries=(_entry("a", offset=1),))

    base_ref = store.put(base)
    child_ref = store.put(child)
    assert base_ref == VM_CACHE_REF_PREFIX + base.fingerprint
    assert store.get(base_ref).to_dict() == base.to_dict()
    assert len(store.list()) == 2
    assert len(store.list("source")) == 2
    assert store.diff(base_ref, child_ref)["semantic_equivalence"] == "not_assessed"
    assert store.get(child_ref).parent_fingerprint == base.fingerprint

    assert main(["list", str(store.root), "--plane", "source"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["manifests"][0]["plane"] == "source"
    assert output["manifests"][0]["entry_count"] == 1


def test_empty_metadata_only_entry_is_valid() -> None:
    entry = VMCacheEntry(entry_id="descriptor", role="operator")
    manifest = build_address_cache_manifest(
        manifest_id="address-only",
        model_identity="fixture-model",
        input_identity="fixture-input",
        abi_identity="fixture-abi",
        entries=(entry,),
    )
    assert manifest.plane == "address"
    assert manifest.entry("descriptor").pointers == ()
