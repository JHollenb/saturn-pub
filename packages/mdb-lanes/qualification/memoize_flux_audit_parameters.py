"""Optional bounded parameter I/O adapter for the frozen saved-only auditor.

Only checkpoint-recipe-v1 reads through its original Evidence decoder may be
memoized. Numerical functions, native/activation reads and prospective data are
untouched. A caller must run handle.verify() after its audit and retain the
returned source-bound counters/identities alongside the audit report.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

BASE_ALLOCATION_ALLOWANCE = 8192
ENTRY_INDEX_ALLOWANCE = 512


class MemoizationIntegrityError(ValueError):
    """The declared cache identity or its immutable decoded value changed."""


class MemoizationBoundError(MemoryError):
    """A first validated decode would exceed a finite cache byte budget."""


def canonical(record):
    """The whole serialized record is its identity, including aliases/metadata."""
    try:
        return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise MemoizationIntegrityError("parameter record is not canonical finite JSON") from error


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def decoded_identity(value):
    return {"shape": list(value.shape), "dtype": value.dtype.str, "bytes": int(value.nbytes),
        "sha256": digest_bytes(memoryview(value).cast("B"))}


@dataclass(frozen=True, slots=True)
class _Entry:
    context: int
    record_json: str
    record_sha256: str
    descriptor_json: str
    decoded: dict
    stored: dict
    value: np.ndarray


def allocation_charge(entry):
    """Conservative owned-object charge; repeated global atoms overcount safely.

    Walk the entry metadata and the ndarray/base/bytes ownership chain. A fixed
    per-entry allowance covers both cache indexes, their keys and weak-domain
    bookkeeping. This is explicit allocation accounting, not a measured RSS.
    """
    seen = set()
    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        result = sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            if value.base is not None:
                result += size(value.base)
        elif isinstance(value, _Entry):
            result += sum(size(getattr(value, field.name)) for field in fields(value))
        elif isinstance(value, dict):
            result += sum(size(key) + size(child) for key, child in value.items())
        elif isinstance(value, (tuple, list)):
            result += sum(size(child) for child in value)
        return result
    return size(entry) + ENTRY_INDEX_ALLOWANCE


class ParameterMemoization:
    """One explicit adapter installation and its non-evicting decoded cache.

    Each Evidence instance gets a separate validation domain. Even identical
    records in a different instance must pass that instance's original allowlist
    and decoder once. Logical decoded and physical retained-byte bounds apply
    across domains. BF16-bits retains only exact uint16 constants, expanding a
    fresh immutable FP32 value on each read. Decoder/math/read transients are
    additional memory, not a retained full-FP32 cache.
    """

    def __init__(self, cohort, original, max_decoded_bytes, storage_codec, max_retained_bytes):
        self._cohort = cohort
        self._original = original
        self._contexts = weakref.WeakKeyDictionary()
        self._next_context = 0
        self._entries = {}
        self._allocation_charges = {}
        self._max_decoded_bytes = max_decoded_bytes
        self._max_retained_bytes = max_retained_bytes
        self._storage_codec = storage_codec
        self._counters = {"parameter_reads": 0, "cache_hits": 0, "cache_misses": 0,
            "original_parameter_decodes": 0, "nonparameter_delegations": 0,
            "decoded_bytes": 0, "peak_decoded_bytes": 0, "bound_rejections": 0,
            "retained_bytes": 0, "peak_retained_bytes": 0, "logical_bound_rejections": 0,
            "retained_allocation_bytes": BASE_ALLOCATION_ALLOWANCE,
            "peak_retained_allocation_bytes": BASE_ALLOCATION_ALLOWANCE,
            "retained_bound_rejections": 0, "non_bf16_rejections": 0,
            "expanded_parameter_reads": 0, "expanded_bytes_emitted": 0,
            "largest_expanded_read_bytes": 0,
            "integrity_failures": 0, "verification_runs": 0, "last_verified_bytes": 0,
            "last_verified_retained_bytes": 0}
        self._active = True
        owner = self

        class MemoizedEvidence(original):
            _mdb_parameter_memoization_owner = owner

            def array(self, record):
                return owner._read(self, record)

        MemoizedEvidence.__name__ = "ParameterMemoized" + original.__name__
        MemoizedEvidence.__qualname__ = MemoizedEvidence.__name__
        self.Evidence = MemoizedEvidence

    def _fail(self, message):
        self._counters["integrity_failures"] += 1
        raise MemoizationIntegrityError(message)

    def _context(self, instance):
        if instance not in self._contexts:
            self._next_context += 1
            self._contexts[instance] = self._next_context
        return self._contexts[instance]

    def _check_metadata(self, entry):
        value = entry.value
        expected = entry.stored
        if (not isinstance(value, np.ndarray) or value.dtype.str != expected["dtype"]
                or list(value.shape) != expected["shape"] or value.nbytes != expected["bytes"]
                or not value.flags.c_contiguous or value.flags.writeable):
            self._fail("cached stored parameter geometry/dtype/readonly contract changed")

    def _expanded(self, entry, *, count_read):
        if self._storage_codec == "float32":
            return entry.value
        # Integer-only bit expansion, including signs and BF16 subnormals.
        # The immutable bytes owner prevents setflags(True) on returned views.
        words = entry.value.astype(np.uint32) << np.uint32(16)
        result = np.frombuffer(words.tobytes(order="C"), dtype=np.float32).reshape(entry.decoded["shape"])
        if count_read:
            self._counters["expanded_parameter_reads"] += 1
            self._counters["expanded_bytes_emitted"] += int(result.nbytes)
            self._counters["largest_expanded_read_bytes"] = max(self._counters["largest_expanded_read_bytes"], int(result.nbytes))
        return result

    def _read(self, instance, record):
        if not isinstance(record, Mapping) or record.get("storage") != "checkpoint-recipe-v1":
            self._counters["nonparameter_delegations"] += 1
            return self._original.array(instance, record)
        if not self._active:
            self._fail("parameter cache installation has been uninstalled")
        self._counters["parameter_reads"] += 1
        serialized = canonical(record)
        key = self._context(instance), serialized
        entry = self._entries.get(key)
        if entry is not None:
            self._check_metadata(entry)
            self._counters["cache_hits"] += 1
            return self._expanded(entry, count_read=True)
        self._counters["cache_misses"] += 1
        self._counters["original_parameter_decodes"] += 1
        # This is the original class method, including its complete recipe,
        # archive, named native port and descriptor validation. No fast reader.
        decoded = self._original.array(instance, record)
        if canonical(record) != serialized:
            self._fail("original decoder mutated its parameter record")
        if (not isinstance(decoded, np.ndarray) or decoded.dtype != np.dtype("float32")
                or not decoded.size or not decoded.flags.c_contiguous):
            self._fail("original parameter decoder returned another dtype/empty/noncontiguous value")
        if not np.isfinite(decoded).all():
            self._fail("original parameter decoder returned nonfinite data")
        if (list(decoded.shape) != record.get("stored_shape")
                or record.get("descriptor", {}).get("shape") != list(decoded.shape)):
            self._fail("validated decoded parameter shape differs from its complete record")
        needed = int(decoded.nbytes)
        if self._storage_codec == "bf16-bits" and np.any(decoded.view(np.uint32) & np.uint32(65535)):
            self._counters["non_bf16_rejections"] += 1
            self._fail("BF16-bits cache refuses nonrepresentable FP32 values; rounding is forbidden")
        logical_excess = self._counters["decoded_bytes"] + needed > self._max_decoded_bytes
        if logical_excess:
            self._counters["bound_rejections"] += 1
            self._counters["logical_bound_rejections"] += 1
            raise MemoizationBoundError("validated parameter exceeds logical decoded byte bound; no eviction or insertion")
        # An immutable bytes owner prevents ordinary writes and setflags(True),
        # including through ndarray views. No alias to the decoder's buffers.
        if self._storage_codec == "float32":
            value = np.frombuffer(decoded.tobytes(order="C"), dtype=decoded.dtype).reshape(decoded.shape)
        else:
            bits = (decoded.view(np.uint32) >> np.uint32(16)).astype("<u2")
            value = np.frombuffer(bits.tobytes(order="C"), dtype="<u2").reshape(decoded.shape)
            del bits
        entry = _Entry(key[0], serialized, digest_bytes(serialized.encode()),
            canonical(record["descriptor"]), decoded_identity(decoded), decoded_identity(value), value)
        self._check_metadata(entry)
        charge = allocation_charge(entry)
        if (self._max_retained_bytes is not None
                and self._counters["retained_allocation_bytes"] + charge > self._max_retained_bytes):
            self._counters["bound_rejections"] += 1
            self._counters["retained_bound_rejections"] += 1
            raise MemoizationBoundError("validated parameter exceeds physical retained allocation bound; no eviction or insertion")
        self._entries[key] = entry
        self._allocation_charges[key] = charge
        self._counters["decoded_bytes"] += needed
        self._counters["peak_decoded_bytes"] = max(self._counters["peak_decoded_bytes"], self._counters["decoded_bytes"])
        self._counters["retained_bytes"] += int(value.nbytes)
        self._counters["peak_retained_bytes"] = max(self._counters["peak_retained_bytes"], self._counters["retained_bytes"])
        self._counters["retained_allocation_bytes"] += charge
        self._counters["peak_retained_allocation_bytes"] = max(self._counters["peak_retained_allocation_bytes"],
            self._counters["retained_allocation_bytes"])
        del decoded
        return self._expanded(entry, count_read=True)

    def stats(self):
        """JSON-safe counters only; final integrity requires verify(), not this."""
        return {"schema": "mdb-flux-audit-parameter-memoization-v1" if self._storage_codec == "float32"
                else "mdb-flux-audit-parameter-memoization-v2", **self._counters,
            "active": self._active, "max_decoded_bytes": self._max_decoded_bytes,
            "logical_decoded_bytes": self._counters["decoded_bytes"], "max_retained_bytes": self._max_retained_bytes,
            "storage_codec": self._storage_codec,
            "retained_allocation_bound_enforced": self._max_retained_bytes is not None,
            "retained_overhead_charge_bytes": self._counters["retained_allocation_bytes"] - self._counters["retained_bytes"],
            "cached_entries": len(self._entries), "validation_domains": self._next_context,
            "original_evidence": self._original.__module__ + "." + self._original.__qualname__,
            "adapter_source_sha256": digest_bytes(Path(__file__).read_bytes()),
            "identity_policy": "Complete canonical JSON record, separately validated per Evidence instance",
            "eviction_policy": "None; first validated decode exceeding either finite byte bound raises",
            "cache_scope": "checkpoint-recipe-v1 only; native, activation, output and target arrays always delegate",
            "memory_scope": "decoded_bytes is logical FP32 size; retained_bytes is stored array payload. retained_allocation_bytes is a conservative charge including owned array/base/bytes and entry metadata plus fixed allowances, not measured RSS. Explicit max_retained_bytes limits that total. Positional FP32 API with max_retained_bytes=None preserves its original logical-only bound.",
            "allocation_charge_policy": f"{BASE_ALLOCATION_ALLOWANCE} fixed bytes plus recursive per-entry owned objects plus {ENTRY_INDEX_ALLOWANCE} bytes per entry for indexes/domain bookkeeping; shared atoms may be counted repeatedly conservatively",
            "expansion_scope": "Fresh FP32 read buffers are never cached in BF16 mode. Largest emitted payload and cumulative emitted bytes are counted; live caller-held, original-decoder, conversion and numerical temporaries are not tracked and are additional.",
            "numerical_functions_modified": False}

    def verify(self):
        """Hash all retained values and check bound/identities; raises on change."""
        if self._active and self._cohort.get("Evidence") is not self.Evidence:
            self._fail("another installation changed the Evidence factory")
        records, total, retained, charged = [], 0, 0, BASE_ALLOCATION_ALLOWANCE
        for key, entry in self._entries.items():
            self._check_metadata(entry)
            if decoded_identity(entry.value) != entry.stored:
                self._fail("cached stored parameter bytes changed")
            if decoded_identity(self._expanded(entry, count_read=False)) != entry.decoded:
                self._fail("reconstructed cached parameter differs from its original FP32 decode")
            if digest_bytes(entry.record_json.encode()) != entry.record_sha256:
                self._fail("cached complete-record identity changed")
            record = json.loads(entry.record_json)
            if canonical(record["descriptor"]) != entry.descriptor_json:
                self._fail("cached native descriptor identity changed")
            total += entry.decoded["bytes"]
            retained += entry.stored["bytes"]
            charge = self._allocation_charges[key]
            if allocation_charge(entry) > charge:
                self._fail("cached owned allocation grew beyond its retained charge")
            charged += charge
            records.append({"validation_domain": entry.context, "complete_record_sha256": entry.record_sha256,
                "path": record["path"], "parameter_tensor": record["parameter_tensor"],
                "native_descriptor": json.loads(entry.descriptor_json), "decoded_identity": json.loads(canonical(entry.decoded)),
                "stored_identity": json.loads(canonical(entry.stored)), "storage_codec": self._storage_codec,
                "retained_allocation_charge_bytes": charge})
        if total != self._counters["decoded_bytes"] or total > self._max_decoded_bytes:
            self._fail("cached logical decoded-byte census differs from its finite bound")
        if retained != self._counters["retained_bytes"]:
            self._fail("cached physical retained-byte census differs from its finite bound")
        if (charged != self._counters["retained_allocation_bytes"]
                or (self._max_retained_bytes is not None and charged > self._max_retained_bytes)):
            self._fail("cached retained allocation census differs from its explicit finite bound")
        self._counters["verification_runs"] += 1
        self._counters["last_verified_bytes"] = total
        self._counters["last_verified_retained_bytes"] = retained
        return {**self.stats(), "verified": True, "verified_entries": records,
            "scope": "I/O-only cache integrity. No numerical accuracy or prospective scientific claim is assigned here."}

    def uninstall(self):
        """Verify, then restore only the exact factory replaced by this handle."""
        if not self._active or self._cohort.get("Evidence") is not self.Evidence:
            self._fail("another installation changed the Evidence factory; restore refused")
        report = self.verify()
        self._cohort["Evidence"] = self._original
        self._active = False
        return report


def install(analyzer_namespace, max_decoded_bytes, *, storage_codec="float32", max_retained_bytes=None):
    """Wrap cached cohort Evidence reads, preserving the analyzer entrypoint.

    The caller must retain the adapter source digest and byte bound in its own
    request, then append handle.verify() to its final report. No numerical
    callbacks or existing files are edited. No native framework is imported.
    Default positional float32 behavior/logical bound is unchanged. An explicit
    max_retained_bytes limits allocation-inclusive conservative charges; BF16
    mode requires that independent physical bound. Decoder/math/read transients
    are not retained-cache bytes and require separate resource admission.
    """
    if type(max_decoded_bytes) is not int or max_decoded_bytes <= 0:
        raise ValueError("max_decoded_bytes must be an explicit positive finite integer")
    if type(storage_codec) is not str or storage_codec not in ("float32", "bf16-bits"):
        raise ValueError("storage_codec must be float32 or bf16-bits")
    if max_retained_bytes is not None and (type(max_retained_bytes) is not int or max_retained_bytes < BASE_ALLOCATION_ALLOWANCE):
        raise ValueError("max_retained_bytes must cover the positive finite base allocation allowance")
    if storage_codec == "bf16-bits" and max_retained_bytes is None:
        raise ValueError("bf16-bits requires an explicit max_retained_bytes allocation bound")
    if not isinstance(analyzer_namespace, Mapping) or not callable(analyzer_namespace.get("independent")):
        raise ValueError("a frozen analyzer namespace with independent() is required")
    cohort = analyzer_namespace["independent"]()
    if not isinstance(cohort, dict) or not inspect.isclass(cohort.get("Evidence")):
        raise ValueError("the cached independent cohort has no Evidence class")
    original = cohort["Evidence"]
    if getattr(original, "_mdb_parameter_memoization_owner", None) is not None:
        raise ValueError("parameter memoization is already installed on this cohort")
    if not callable(getattr(original, "array", None)):
        raise ValueError("the original Evidence class has no array decoder")
    handle = ParameterMemoization(cohort, original, max_decoded_bytes, storage_codec, max_retained_bytes)
    cohort["Evidence"] = handle.Evidence
    return handle
