"""Durable tensor trees and a bounded immutable resident page pool.

Family adapters own mutable working tensors; the pool only exposes immutable
pages. Copies used for interventions or native in-place writes are explicit.
"""
from collections import OrderedDict
from collections.abc import Mapping
import torch
from hashlib import sha256
from time import perf_counter

from .bench_qwen import BenchError
from .vm_payload_cache import PayloadStore, PayloadRef


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


def allocations(value):
    """Physical storage bytes, including storage retained by offset views."""
    result = {}
    for tensor in tensors(value):
        storage = tensor.untyped_storage()
        result[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return result


def private_tree(value, *, readonly_ids=frozenset()):
    """Private wrappers and native-layout tensors, with declared read-only leaves."""
    if isinstance(value, torch.Tensor):
        if id(value) in readonly_ids: return value
        shape, stride = value.shape, value.stride()
        compact = [1 if s == 0 and n > 0 else n for n, s in zip(shape, stride)]
        source = value
        for axis, (n, s) in enumerate(zip(shape, stride)):
            if s == 0 and n > 1: source = source.narrow(axis, 0, 1)
        result = torch.empty_strided(compact, stride, dtype=value.dtype, device=value.device)
        result.copy_(source)
        return result.expand(shape) if tuple(compact) != tuple(shape) else result
    if isinstance(value, dict): return {k: private_tree(v, readonly_ids=readonly_ids) for k, v in value.items()}
    if isinstance(value, list): return [private_tree(v, readonly_ids=readonly_ids) for v in value]
    if isinstance(value, tuple): return tuple(private_tree(v, readonly_ids=readonly_ids) for v in value)
    return value


class BenchPages:
    def __init__(self, root, *, device, budget, page_bytes=64 << 10):
        if type(budget) is not int or budget <= 0:
            raise BenchError("resident state budget must be positive")
        self.store = PayloadStore(root)
        self.device, self.budget = torch.device(device), budget
        self.pool = OrderedDict()
        self.pool_refs = {}
        self.events = []
        if type(page_bytes) is not int or page_bytes < 256 or page_bytes > 16 << 20:
            raise BenchError("page bytes must be between 256 and 16 MiB")
        self.page_bytes = page_bytes
        self.stats = {"encoded_bytes": 0, "encoded_tensors": 0, "encoded_pages": 0,
            "reused_bytes": 0, "reused_tensors": 0, "encode_seconds": 0.0,
            "decode_seconds": 0.0, "private_copy_bytes": 0}

    def encode(self, value, *, previous_state=None, previous_encoded=None, readonly_ids=frozenset(),
            verify_readonly=False, _reuse=None):
        """Reuse only adapter-declared read-only leaves, never pointer memoization.

        The previous tree is supplied by the admitted transition owner after its
        adapter has isolated all writes. Inference tensors have no version counter.
        Arbitrary encode calls therefore always serialize their supplied bytes.
        """
        if _reuse is None:
            _reuse = {}
            def collect(old, encoded):
                if (isinstance(old, torch.Tensor) and id(old) in readonly_ids
                    and isinstance(encoded, dict) and ("$tensor" in encoded or "$chunks" in encoded)):
                    _reuse[id(old)] = (old, encoded)
                elif isinstance(old, dict) and isinstance(encoded, dict):
                    for key in old: collect(old[key], encoded.get(key))
                elif isinstance(old, (list, tuple)):
                    items = encoded.get("$tuple") if isinstance(encoded, dict) else encoded
                    if isinstance(items, list):
                        for a, b in zip(old, items): collect(a, b)
            collect(previous_state, previous_encoded)
        if isinstance(value, torch.Tensor):
            reused = _reuse.get(id(value))
            if reused is not None and reused[0] is value:
                if verify_readonly:
                    descriptor = reused[1]
                    refs = descriptor.get("$chunks", [descriptor.get("$tensor")])
                    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
                    expected = b"".join(self.store.hydrate(PayloadRef.from_dict(r), native_consumer="bench-readonly-audit").payload for r in refs)
                    geometry = descriptor.get("$geometry", {})
                    if (raw != expected or (geometry and (geometry["shape"] != list(value.shape)
                        or geometry.get("stride", list(value.contiguous().stride())) != list(value.stride())
                        or geometry["dtype"] != str(value.dtype)))):
                        raise BenchError("adapter read-only declaration was mutated")
                self.stats["reused_tensors"] += 1
                self.stats["reused_bytes"] += value.numel()*value.element_size()
                return reused[1]
            started = perf_counter()
            t = value.detach().contiguous()
            raw = t.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            # PayloadStore binds one descriptor to byte identity. Native forwards
            # routinely reshape the same bytes, so geometry belongs to the view,
            # while the physical immutable allocation is always a raw byte page.
            refs = [self.store.put_bytes(raw[start:start+self.page_bytes], shape=(len(raw[start:start+self.page_bytes]),),
                dtype="torch.uint8", role="bench-family-page").to_dict() for start in range(0, len(raw), self.page_bytes)]
            self.stats["encoded_bytes"] += len(raw); self.stats["encoded_tensors"] += 1
            self.stats["encoded_pages"] += len(refs); self.stats["encode_seconds"] += perf_counter()-started
            return {"$chunks": refs, "$bytes": len(raw), "$page_bytes": self.page_bytes,
                "$geometry": {"shape": list(t.shape), "stride": list(value.stride()), "dtype": str(t.dtype), "device": t.device.type}}
        if isinstance(value, tuple):
            old = previous_encoded.get("$tuple", []) if isinstance(previous_encoded, dict) else []
            return {"$tuple": [self.encode(v, previous_state=previous_state[i] if isinstance(previous_state, tuple) and i < len(previous_state) else None,
                previous_encoded=old[i] if i < len(old) else None, verify_readonly=verify_readonly, _reuse=_reuse) for i, v in enumerate(value)]}
        if isinstance(value, list):
            return [self.encode(v, previous_state=previous_state[i] if isinstance(previous_state, list) and i < len(previous_state) else None,
                previous_encoded=previous_encoded[i] if isinstance(previous_encoded, list) and i < len(previous_encoded) else None,
                verify_readonly=verify_readonly, _reuse=_reuse) for i, v in enumerate(value)]
        if isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise BenchError("checkpoint maps require string keys")
            return {k: self.encode(v, previous_state=previous_state.get(k) if isinstance(previous_state, dict) else None,
                previous_encoded=previous_encoded.get(k) if isinstance(previous_encoded, dict) else None,
                verify_readonly=verify_readonly, _reuse=_reuse) for k, v in value.items()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise BenchError(f"unsupported checkpoint payload {type(value).__name__}")

    def decode(self, value, *, private=False):
        if isinstance(value, dict) and ("$tensor" in value or "$chunks" in value):
            started = perf_counter()
            refs = [PayloadRef.from_dict(r) for r in value["$chunks"]] if "$chunks" in value else [PayloadRef.from_dict(value["$tensor"])]
            size = sum(r.bytes for r in refs)
            if "$chunks" in value:
                page_bytes = value.get("$page_bytes")
                if (type(value.get("$bytes")) is not int or value["$bytes"] != size or type(page_bytes) is not int
                    or not 256 <= page_bytes <= 16 << 20 or any(r.dtype != "torch.uint8" or r.shape != (r.bytes,)
                    or r.bytes != min(page_bytes, size-i*page_bytes) for i, r in enumerate(refs))):
                    raise BenchError("invalid chunked tensor closure")
            digest = refs[0].sha256 if len(refs) == 1 else sha256("".join(r.sha256 for r in refs).encode()).hexdigest()
            placement = value.get("$geometry", {}).get("device", self.device.type)
            if placement not in {"cpu", self.device.type}:
                raise BenchError("checkpoint tensor placement incompatible with native device")
            target = torch.device("cpu") if placement == "cpu" else self.device
            ref_dtype = refs[0].dtype if refs else "torch.uint8"
            ref_shape = refs[0].shape if len(refs) == 1 else (size,)
            key = (digest, ref_dtype, tuple(ref_shape), placement)
            if key not in self.pool:
                raw = b"".join(self.store.hydrate(ref, native_consumer="bench-native-resume").payload for ref in refs)
                dtype = getattr(torch, ref_dtype.removeprefix("torch."))
                self.pool[key] = (torch.frombuffer(bytearray(raw), dtype=dtype).reshape(ref_shape).to(target)
                    if raw else torch.empty(ref_shape, dtype=dtype, device=target))
                self.events.append({"kind": "hydrate", "sha256": digest, "bytes": len(raw), "backing_pages": len(refs)})
                self.pool_refs[key] = tuple(refs)
            self.pool.move_to_end(key)
            result = self.pool[key]
            layout_owned = False
            if "$geometry" in value:
                geometry = value["$geometry"]
                dtype = getattr(torch, geometry["dtype"].removeprefix("torch."), None)
                shape = geometry["shape"]
                if dtype is None or any(type(n) is not int or n < 0 for n in shape):
                    raise BenchError("invalid native tensor view geometry")
                try: result = result.view(dtype).reshape(shape)
                except RuntimeError as exc: raise BenchError("tensor view differs from physical byte extent") from exc
                stride = geometry.get("stride", list(result.stride()))
                if len(stride) != len(shape) or any(type(n) is not int or n < 0 for n in stride):
                    raise BenchError("invalid native tensor stride")
                if tuple(stride) != result.stride():
                    # Preserve native layout: CUDA kernels may select different
                    # arithmetic for contiguous and transposed/channel-last views.
                    compact_shape = [1 if s == 0 and n > 0 else n for n, s in zip(shape, stride)]
                    source = result
                    for axis, (n, s) in enumerate(zip(shape, stride)):
                        if s == 0 and n > 1: source = source.narrow(axis, 0, 1)
                    extent = 0 if any(n == 0 for n in compact_shape) else 1+sum((n-1)*s for n, s in zip(compact_shape, stride))
                    if extent*result.element_size() > self.budget:
                        raise BenchError("native strided view exceeds resident state budget")
                    restored = torch.empty_strided(compact_shape, stride, dtype=dtype, device=target)
                    try: restored.copy_(source)
                    except RuntimeError as exc: raise BenchError("unsupported overlapping native tensor layout") from exc
                    result = restored.expand(shape)
                    layout_owned = True
            self.stats["decode_seconds"] += perf_counter()-started
            if private:
                self.stats["private_copy_bytes"] += result.numel()*result.element_size()
                return result if layout_owned else result.clone(memory_format=torch.preserve_format)
            return result
        if isinstance(value, dict) and "$tuple" in value:
            return tuple(self.decode(v, private=private) for v in value["$tuple"])
        if isinstance(value, list):
            return [self.decode(v, private=private) for v in value]
        if isinstance(value, dict):
            return {k: self.decode(v, private=private) for k, v in value.items()}
        return value

    def admit(self, working=None):
        live = allocations(working)
        if sum(live.values()) > self.budget:
            raise BenchError("active mutable closure exceeds resident state budget")
        union = allocations(list(self.pool.values())); union.update(live)
        total = sum(union.values())
        for key in list(self.pool):
            if total <= self.budget:
                break
            t = self.pool.pop(key)
            self.pool_refs.pop(key, None)
            storage = (str(t.device), t.untyped_storage().data_ptr())
            if storage not in live:
                total -= union.pop(storage, 0)
            self.events.append({"kind": "evict", "sha256": key[0], "bytes": t.numel()*t.element_size()})

    def inspect(self, branches, working):
        owners = {}
        assemblies = set()
        visible_bytes, logical_views = 0, 0
        def walk(value, owner, path=""):
            nonlocal visible_bytes, logical_views
            if isinstance(value, dict) and ("$tensor" in value or "$chunks" in value):
                refs = [PayloadRef.from_dict(r) for r in value["$chunks"]] if "$chunks" in value else [PayloadRef.from_dict(value["$tensor"])]
                visible_bytes += sum(r.bytes for r in refs)
                logical_views += 1
                placement = value.get("$geometry", {}).get("device", self.device.type)
                digest = refs[0].sha256 if len(refs) == 1 else sha256("".join(r.sha256 for r in refs).encode()).hexdigest()
                assembly = (digest, refs[0].dtype if refs else "torch.uint8", tuple(refs[0].shape) if len(refs) == 1 else (sum(r.bytes for r in refs),), placement)
                assemblies.add(assembly)
                for index, ref in enumerate(refs):
                    key = (ref.sha256, ref.dtype, tuple(ref.shape), placement)
                    row = owners.setdefault(key, {"sha256": ref.sha256, "bytes": ref.bytes,
                                                  "owners": [], "views": [], "backed": True, "resident": False})
                    row["resident"] |= assembly in self.pool
                    row["views"].append({"owner": owner, "path": path, "page": index, "geometry": value.get("$geometry")})
                    if owner not in row["owners"]: row["owners"].append(owner)
            elif isinstance(value, dict):
                for key, v in value.items(): walk(v, owner, f"{path}.{key}".strip("."))
            elif isinstance(value, list):
                for index, v in enumerate(value): walk(v, owner, f"{path}.{index}".strip("."))
        for owner, state in branches.items(): walk(state, owner)
        live = allocations(list(self.pool.values()))
        live.update(allocations(working))
        for key, row in owners.items():
            row["ownership"] = "shared" if len(row["owners"]) > 1 else "private"
        retained = [{"sha256": key[0], "bytes": tensor.numel()*tensor.element_size(), "owners": [],
            "views": [], "backed": True, "resident": True, "ownership": "retained-pool"}
            for key, tensor in self.pool.items() if key not in assemblies]
        backing = {r["sha256"]: r["bytes"] for r in owners.values()}
        for refs in self.pool_refs.values():
            backing.update({r.sha256: r.bytes for r in refs})
        return {"budget_bytes": self.budget,
                "resident_bytes": sum(live.values()),
                "unique_allocations": [*owners.values(), *retained],
                "unique_backed_bytes": sum(backing.values()),
                "visible_bytes": visible_bytes, "logical_views": logical_views,
                "retained_resident_pages": len(retained),
                "evictions": sum(e["kind"] == "evict" for e in self.events),
                "hydrations": sum(e["kind"] == "hydrate" for e in self.events),
                "events": list(self.events), "payload_stats": self.store.stats.to_dict(),
                "backing_page_bytes": self.page_bytes, "resident_allocation_kind": "native-tensor-assembly",
                "page_stats": dict(self.stats)}
