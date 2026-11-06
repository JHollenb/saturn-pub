"""I/O adapter mechanics and frozen tiny factorial equivalence; no models."""
from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1]


def module():
    return runpy.run_path(str(HERE / "qualification/memoize_flux_audit_parameters.py"))


def record(**changes):
    result = {"storage": "checkpoint-recipe-v1", "path": "parameters/fixed.npy", "sha256": "a" * 64,
        "encoding": "bf16-bits-v1", "descriptor": {"shape": [2], "dtype": "torch.bfloat16", "sha256": "b" * 64},
        "stored_shape": [2], "stored_dtype": "uint16", "native_stride": [1], "parameter_tensor": "a.weight"}
    result.update(changes)
    return result


def fixture():
    class Evidence:
        def __init__(self, *, allowed=None):
            self.calls = []
            self.allowed = allowed

        def array(self, value):
            self.calls.append(copy.deepcopy(value))
            if value.get("storage") == "checkpoint-recipe-v1":
                if self.allowed is not None and value != self.allowed:
                    raise ValueError("exact original recipe allowlist refused")
                return np.array([1., -2.], dtype=np.float32)
            return np.array([len(self.calls)], dtype=np.float32)
    cohort = {"Evidence": Evidence}
    namespace = {"independent": lambda: cohort}
    return cohort, namespace, Evidence


def test_first_exact_decode_then_immutable_bitwise_hits_and_factory_restore():
    a = module()
    cohort, ns, original = fixture()
    handle = a["install"](ns, max_decoded_bytes=64)
    reader = cohort["Evidence"](allowed=record())
    first, second = reader.array(record()), reader.array(record())
    assert first is second and len(reader.calls) == 1
    assert first.dtype == np.float32 and np.array_equal(first.view(np.uint32), np.array([1., -2.], np.float32).view(np.uint32))
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0] = 3
    with pytest.raises(ValueError):
        first.setflags(write=True)
    with pytest.raises(ValueError):
        first.view(np.ndarray).setflags(write=True)
    report = handle.verify()
    assert report["verified"] and report["decoded_bytes"] == report["last_verified_bytes"] == 8
    assert report["cache_hits"] == 1 and report["original_parameter_decodes"] == 1
    assert report["verified_entries"][0]["native_descriptor"] == record()["descriptor"]
    json.dumps(report, allow_nan=False)
    handle.uninstall()
    assert cohort["Evidence"] is original


@pytest.mark.parametrize("field,value", [("parameter_tensor", "b.weight"), ("sha256", "c" * 64),
    ("native_stride", [2]), ("metadata", {"alias": "another"}), ("descriptor", {"shape": [2], "dtype": "torch.bfloat16", "sha256": "d" * 64})])
def test_complete_record_difference_never_reuses_path_only_cache(field, value):
    a = module()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 64)
    reader = cohort["Evidence"](allowed=record())
    reader.array(record())
    other = record(**{field: value})
    with pytest.raises(ValueError, match="original recipe allowlist"):
        reader.array(other)
    assert len(reader.calls) == 2 and handle.stats()["cache_hits"] == 0


def test_distinct_evidence_instance_cannot_bypass_its_original_allowlist():
    a = module()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 64)
    first = cohort["Evidence"](allowed=record())
    first.array(record())
    second = cohort["Evidence"](allowed=record(parameter_tensor="other"))
    with pytest.raises(ValueError, match="allowlist"):
        second.array(record())
    assert len(second.calls) == 1 and handle.stats()["validation_domains"] == 2


def test_valid_same_file_aliases_are_separately_decoded_and_key_order_is_canonical():
    a = module()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 64)
    reader = cohort["Evidence"]()
    first = reader.array(record())
    second_record = record(parameter_tensor="valid.alias.weight")
    second = reader.array(second_record)
    assert first is not second and np.array_equal(first.view(np.uint32), second.view(np.uint32))
    assert reader.array(dict(reversed(list(second_record.items())))) is second
    assert len(reader.calls) == 2 and handle.stats()["cached_entries"] == 2
    assert handle.verify()["decoded_bytes"] == 16


def test_cache_owns_bytes_and_verification_report_does_not_alias_its_metadata():
    a = module()
    backing = np.array([1., 2.], dtype=np.float32)
    class Evidence:
        def array(self, value):
            return backing
    cohort = {"Evidence": Evidence}
    handle = a["install"]({"independent": lambda: cohort}, 64)
    value = cohort["Evidence"]().array(record())
    backing[0] = 99
    assert value[0] == 1
    report = handle.verify()
    report["verified_entries"][0]["decoded_identity"]["shape"][0] = 99
    assert handle.verify()["verified_entries"][0]["decoded_identity"]["shape"] == [2]


def test_nonparameter_activation_target_and_unknown_storage_always_delegate():
    a = module()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 64)
    reader = cohort["Evidence"]()
    for value in ({"path": "activation.npy"}, {"path": "output.npy", "storage": "native"},
                  {"path": "parameters/fixed.npy", "storage": "checkpoint-recipe-v0"}):
        x, y = reader.array(value), reader.array(value)
        assert x.item() != y.item()
    assert len(reader.calls) == 6 and handle.stats()["nonparameter_delegations"] == 6
    assert handle.verify()["decoded_bytes"] == 0


def test_explicit_byte_bound_rejects_without_eviction_and_keeps_prior_entries():
    a = module()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 8)
    reader = cohort["Evidence"]()
    first = reader.array(record())
    with pytest.raises(a["MemoizationBoundError"]):
        reader.array(record(parameter_tensor="b.weight"))
    assert handle.stats()["decoded_bytes"] == 8 and handle.stats()["cached_entries"] == 1
    assert handle.stats()["bound_rejections"] == 1 and reader.array(record()) is first
    assert handle.verify()["verified"]


@pytest.mark.parametrize("mutation", ["bytes", "shape", "dtype"])
def test_hostile_cached_mutation_is_detected_at_final_verification(mutation):
    a = module()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 64)
    value = cohort["Evidence"]().array(record())
    if mutation == "bytes":
        # Deliberate hostile memory write on a tiny owned test allocation only.
        replacement = ctypes.c_float(99.)
        ctypes.memmove(value.ctypes.data, ctypes.byref(replacement), ctypes.sizeof(replacement))
    elif mutation == "shape":
        value.shape = (1, 2)
    else:
        value.dtype = np.int32
    with pytest.raises(a["MemoizationIntegrityError"]):
        handle.verify()
    assert handle.stats()["integrity_failures"] == 1


def test_input_record_mutation_gets_new_validation_not_a_cached_value():
    a = module()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 64)
    value = record()
    reader = cohort["Evidence"](allowed=copy.deepcopy(value))
    reader.array(value)
    value["descriptor"]["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="allowlist"):
        reader.array(value)
    assert len(reader.calls) == 2 and handle.stats()["cache_hits"] == 0


def test_invalid_bound_duplicate_install_and_other_factory_replacement_refused():
    a = module()
    cohort, ns, original = fixture()
    for bound in (0, -1, True, 1.5, float("inf")):
        with pytest.raises(ValueError):
            a["install"](ns, bound)
    handle = a["install"](ns, 64)
    with pytest.raises(ValueError, match="already installed"):
        a["install"](ns, 64)
    cohort["Evidence"] = original
    with pytest.raises(a["MemoizationIntegrityError"]):
        handle.uninstall()


def test_adapter_imports_no_native_framework():
    import subprocess
    path = HERE / "qualification/memoize_flux_audit_parameters.py"
    code = "import runpy,sys;runpy.run_path(sys.argv[1]);assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-I", "-c", code, str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_real_frozen_tiny_all960_reconstruction_hashes_unchanged():
    data = os.environ.get("MDB_FACTORIAL_TINY_DATA")
    cache = os.environ.get("MDB_FACTORIAL_TINY_PARAMETER_ROOT")
    baseline = os.environ.get("MDB_FACTORIAL_TINY_BASELINE_AUDIT")
    if not all((data, cache, baseline)):
        pytest.skip("requires explicit saved qualified tiny fixture; no model construction in this test")
    analyzer = runpy.run_path(str(HERE / "qualification/analyze_flux_prompt_factorial.py"))
    frozen_hash = "86084b8c751519dcb0ce4cbd5e8640a5cc84ccf12a986e6074c2e694002d40e5"
    assert hashlib.sha256((HERE / "qualification/analyze_flux_prompt_factorial.py").read_bytes()).hexdigest() == frozen_hash
    before = json.loads(Path(baseline).read_text())
    original_entrypoint = analyzer["analyze_data"]
    handle = module()["install"](analyzer, 8 << 20)
    after = analyzer["analyze_data"](Path(data), parameter_root=Path(cache), expected_job="tiny-factorial", production_profile=False)
    verification = handle.verify()
    assert analyzer["analyze_data"] is original_entrypoint
    assert after["numeric"] == before["numeric"]
    assert after["endpoint_comparisons"] == before["endpoint_comparisons"]
    assert after["coverage"] == before["coverage"]
    assert after["coverage"]["complete_predicted_fields_reconstructed"] == 960
    assert all(row["complete_field_hashes_match"] for row in after["numeric"].values())
    assert verification["verified"] and verification["cache_hits"] > verification["cache_misses"]
    assert verification["original_parameter_decodes"] == verification["cached_entries"]
    destination = os.environ.get("MDB_FACTORIAL_MEMOIZATION_REPORT")
    if destination:
        def tree_hash(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        with Path(destination).open("x") as stream:
            json.dump({"verification": verification, "baseline_sha256": hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
                "numeric_unchanged": True, "endpoints_unchanged": True, "coverage": after["coverage"],
                "numeric_sha256": {"uncached": tree_hash(before["numeric"]), "cached": tree_hash(after["numeric"])},
                "endpoints_sha256": {"uncached": tree_hash(before["endpoint_comparisons"]), "cached": tree_hash(after["endpoint_comparisons"])}},
                stream, indent=2, sort_keys=True)


@pytest.mark.parametrize("codec", ["float32", "bf16-bits"])
def test_both_codecs_return_exact_readonly_f32_and_bind_stored_identity(codec):
    a = module()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 64, storage_codec=codec, max_retained_bytes=1 << 20)
    reader = cohort["Evidence"](allowed=record())
    first, second = reader.array(record()), reader.array(record())
    assert len(reader.calls) == 1 and first.dtype == second.dtype == np.float32
    assert np.array_equal(first.view(np.uint32), second.view(np.uint32))
    assert (first is second) == (codec == "float32")
    for value in (first, second):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
        with pytest.raises(ValueError):
            value[0] = 0
    report = handle.verify()
    assert report["logical_decoded_bytes"] == report["decoded_bytes"] == 8
    assert report["retained_bytes"] == (8 if codec == "float32" else 4)
    assert report["last_verified_retained_bytes"] == report["retained_bytes"]
    assert report["retained_allocation_bytes"] > report["retained_bytes"]
    assert report["retained_allocation_bytes"] <= report["max_retained_bytes"]
    assert report["retained_allocation_bound_enforced"] is True
    entry = report["verified_entries"][0]
    assert entry["decoded_identity"]["dtype"] == np.dtype(np.float32).str
    assert entry["stored_identity"]["dtype"] == np.dtype(np.float32 if codec == "float32" else np.uint16).str
    assert report["expanded_parameter_reads"] == (0 if codec == "float32" else 2)
    assert report["expanded_bytes_emitted"] == (0 if codec == "float32" else 16)
    assert report["largest_expanded_read_bytes"] == (0 if codec == "float32" else 8)


@pytest.mark.parametrize("codec", ["float32", "bf16-bits"])
def test_both_codecs_respect_full_record_allowlist_and_bypass_activations(codec):
    a = module()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 1024, storage_codec=codec, max_retained_bytes=1 << 20)
    reader = cohort["Evidence"](allowed=record())
    reader.array(record())
    for value in (record(parameter_tensor="hostile.alias"), record(sha256="0" * 64),
                  record(descriptor={"shape": [2], "dtype": "torch.bfloat16", "sha256": "0" * 64})):
        with pytest.raises(ValueError, match="allowlist"):
            reader.array(value)
    activation = {"path": "output.npy"}
    first, second = reader.array(activation), reader.array(activation)
    assert first.item() != second.item()
    assert handle.stats()["cache_hits"] == 0 and handle.stats()["nonparameter_delegations"] == 2
    other = cohort["Evidence"](allowed=record(parameter_tensor="different.instance"))
    with pytest.raises(ValueError, match="allowlist"):
        other.array(record())
    assert len(other.calls) == 1
    assert handle.verify()["cached_entries"] == 1


@pytest.mark.parametrize("codec", ["float32", "bf16-bits"])
@pytest.mark.parametrize("bound", ["logical", "retained"])
def test_independent_logical_and_physical_bounds_reject_without_eviction(codec, bound):
    a = module()
    cohort, ns, _ = fixture()
    first_physical = 8 if codec == "float32" else 4
    probe = a["install"](ns, 64, storage_codec=codec, max_retained_bytes=1 << 20)
    cohort["Evidence"]().array(record())
    allocation_cap = probe.verify()["retained_allocation_bytes"] + 64
    probe.uninstall()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 8 if bound == "logical" else 64, storage_codec=codec,
        max_retained_bytes=1 << 20 if bound == "logical" else allocation_cap)
    reader = cohort["Evidence"]()
    expected = reader.array(record()).copy()
    with pytest.raises(a["MemoizationBoundError"]):
        reader.array(record(parameter_tensor="second.weight"))
    assert np.array_equal(reader.array(record()).view(np.uint32), expected.view(np.uint32))
    stats = handle.verify()
    assert stats["decoded_bytes"] == 8 and stats["retained_bytes"] == first_physical
    assert stats["cached_entries"] == 1 and stats["bound_rejections"] == 1
    assert stats["retained_allocation_bytes"] <= stats["max_retained_bytes"]
    assert stats["logical_bound_rejections"] == int(bound == "logical")
    assert stats["retained_bound_rejections"] == int(bound == "retained")


def test_bf16_codec_refuses_nonrepresentable_f32_instead_of_rounding():
    a = module()
    class Evidence:
        def array(self, value):
            return np.array([1.000001, -2.], dtype=np.float32)
    cohort = {"Evidence": Evidence}
    handle = a["install"]({"independent": lambda: cohort}, 64, storage_codec="bf16-bits", max_retained_bytes=1 << 20)
    with pytest.raises(a["MemoizationIntegrityError"], match="rounding is forbidden"):
        cohort["Evidence"]().array(record())
    stats = handle.verify()
    assert stats["non_bf16_rejections"] == 1 and stats["cached_entries"] == 0
    assert stats["decoded_bytes"] == stats["retained_bytes"] == 0


def test_bf16_codec_is_bit_exact_for_signed_zero_subnormals_and_finite_extremes():
    a = module()
    words = np.array([0, 0x80000000, 0x00010000, 0x80010000, 0x3f800000, 0x7f7f0000, 0xff7f0000], dtype=np.uint32)
    class Evidence:
        def array(self, value):
            return words.view(np.float32)
    cohort = {"Evidence": Evidence}
    value = record(stored_shape=[7], descriptor={"shape": [7], "dtype": "torch.bfloat16", "sha256": "b" * 64})
    handle = a["install"]({"independent": lambda: cohort}, 64, storage_codec="bf16-bits", max_retained_bytes=1 << 20)
    reader = cohort["Evidence"]()
    assert np.array_equal(reader.array(value).view(np.uint32), words)
    assert np.array_equal(reader.array(value).view(np.uint32), words)
    assert handle.verify()["retained_bytes"] == 14


@pytest.mark.parametrize("mutation", ["bytes", "shape", "dtype"])
def test_bf16_cached_uint16_mutation_is_detected(mutation):
    a = module()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 64, storage_codec="bf16-bits", max_retained_bytes=1 << 20)
    cohort["Evidence"]().array(record())
    stored = next(iter(handle._entries.values())).value
    assert stored.dtype == np.uint16 and not stored.flags.writeable
    with pytest.raises(ValueError):
        stored.setflags(write=True)
    if mutation == "bytes":
        replacement = ctypes.c_uint16(0x3f81)
        ctypes.memmove(stored.ctypes.data, ctypes.byref(replacement), ctypes.sizeof(replacement))
    elif mutation == "shape":
        stored.shape = (1, 2)
    else:
        stored.dtype = np.int16
    with pytest.raises(a["MemoizationIntegrityError"]):
        handle.verify()


def test_bf16_expanded_read_buffers_are_not_retained_as_a_full_f32_cache():
    import gc
    import weakref

    a = module()
    cohort, ns, _ = fixture()
    handle = a["install"](ns, 64, storage_codec="bf16-bits", max_retained_bytes=1 << 20)
    reader = cohort["Evidence"]()
    value = reader.array(record())
    reference = weakref.ref(value)
    del value
    gc.collect()
    assert reference() is None
    assert all(entry.value.dtype == np.uint16 for entry in handle._entries.values())
    assert np.array_equal(reader.array(record()), np.array([1., -2.], dtype=np.float32))
    assert handle.verify()["retained_bytes"] == 4


def test_new_codec_and_retained_bound_validation_preserve_default_api():
    a = module()
    cohort, ns, _ = fixture()
    for codec in (None, "unknown", True):
        with pytest.raises(ValueError):
            a["install"](ns, 64, storage_codec=codec)
    for bound in (0, -1, True, 1.5, float("inf"), a["BASE_ALLOCATION_ALLOWANCE"] - 1):
        with pytest.raises(ValueError):
            a["install"](ns, 64, max_retained_bytes=bound)
    handle = a["install"](ns, 64)
    assert handle.stats()["storage_codec"] == "float32" and handle.stats()["max_retained_bytes"] is None
    assert handle.stats()["schema"] == "mdb-flux-audit-parameter-memoization-v1"
    assert handle.stats()["retained_allocation_bound_enforced"] is False
    handle.uninstall()
    assert cohort["Evidence"] is not handle.Evidence
    with pytest.raises(ValueError, match="explicit max_retained_bytes"):
        a["install"](ns, 64, storage_codec="bf16-bits")


def test_bf16_real_frozen_tiny_all960_reconstruction_hashes_unchanged():
    data = os.environ.get("MDB_FACTORIAL_TINY_DATA")
    cache = os.environ.get("MDB_FACTORIAL_TINY_PARAMETER_ROOT")
    baseline = os.environ.get("MDB_FACTORIAL_TINY_BASELINE_AUDIT")
    if not all((data, cache, baseline)):
        pytest.skip("requires explicit saved qualified tiny fixture; no model construction in this test")
    analyzer = runpy.run_path(str(HERE / "qualification/analyze_flux_prompt_factorial.py"))
    assert hashlib.sha256((HERE / "qualification/analyze_flux_prompt_factorial.py").read_bytes()).hexdigest() == (
        "86084b8c751519dcb0ce4cbd5e8640a5cc84ccf12a986e6074c2e694002d40e5")
    before = json.loads(Path(baseline).read_text())
    original_entrypoint = analyzer["analyze_data"]
    handle = module()["install"](analyzer, 8 << 20, storage_codec="bf16-bits", max_retained_bytes=4 << 20)
    after = analyzer["analyze_data"](Path(data), parameter_root=Path(cache), expected_job="tiny-factorial", production_profile=False)
    verification = handle.verify()
    assert analyzer["analyze_data"] is original_entrypoint
    assert after["numeric"] == before["numeric"] and after["endpoint_comparisons"] == before["endpoint_comparisons"]
    assert after["coverage"] == before["coverage"] and after["coverage"]["complete_predicted_fields_reconstructed"] == 960
    assert all(row["complete_field_hashes_match"] for row in after["numeric"].values())
    assert verification["verified"] and verification["cached_entries"] == verification["original_parameter_decodes"] == 169
    assert verification["retained_bytes"] * 2 == verification["decoded_bytes"]
    assert verification["expanded_parameter_reads"] == verification["parameter_reads"]
    destination = os.environ.get("MDB_FACTORIAL_BF16_MEMOIZATION_REPORT")
    if destination:
        def tree_hash(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        with Path(destination).open("x") as stream:
            json.dump({"verification": verification, "baseline_sha256": hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
                "numeric_unchanged": True, "endpoints_unchanged": True, "coverage": after["coverage"],
                "numeric_sha256": {"uncached": tree_hash(before["numeric"]), "bf16_cached": tree_hash(after["numeric"])},
                "endpoints_sha256": {"uncached": tree_hash(before["endpoint_comparisons"]), "bf16_cached": tree_hash(after["endpoint_comparisons"])}},
                stream, indent=2, sort_keys=True)
