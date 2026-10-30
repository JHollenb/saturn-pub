"""Durable deterministic CPU ATen frontiers in the public LocalStore CAS.

Backing storages are byte tensors; a separately sealed geometry table restores
aliases, offsets, permutations and broadcast views. No Python object pickle or
model import path is serialized. The profile is deliberately CPU-only and
requires the same Torch version, architecture and byte order when resumed.
"""
from __future__ import annotations

import copy
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

from saturn_pub import Receipt, StateCut
from saturn_pub.store import LocalStore
from saturn_pub.values import canonical, digest

SCHEMA = "mdb-discovery-durable-frontier-v1"
PROFILE = "deterministic-cpu-aten-v1"
DEFAULT_MAX_STORAGE_BYTES = 8 << 30
DEFAULT_MAX_TENSOR_ELEMENTS = 1 << 31
_DTYPES = {"bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16",
           "float32", "float64", "complex64", "complex128"}


class DiscoveryStorageError(ValueError):
    pass


def _store(value):
    if isinstance(value, LocalStore):
        return value
    return LocalStore(Path(value))


def _platform():
    import torch

    return {"torch":str(torch.__version__), "architecture":platform.machine(),
            "byteorder":sys.byteorder, "device":"cpu", "intraop_threads":torch.get_num_threads()}


def _restore_platform(saved):
    import torch

    if not isinstance(saved,dict):
        raise DiscoveryStorageError("missing durable CPU platform binding")
    current = _platform()
    if {key:value for key,value in saved.items() if key != "intraop_threads"} != {
            key:value for key,value in current.items() if key != "intraop_threads"}:
        raise DiscoveryStorageError("durable CPU frontier platform/Torch applicability differs")
    threads = saved.get("intraop_threads")
    if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
        raise DiscoveryStorageError("durable CPU thread contract exceeds host thread capacity")
    if current["intraop_threads"] != threads:
        torch.set_num_threads(threads)
    if torch.get_num_threads() != threads:
        raise DiscoveryStorageError("durable CPU thread contract could not be restored")
    return {"scope":"process-wide Torch intra-op threads", "before":current["intraop_threads"],
            "after":threads,"changed":current["intraop_threads"] != threads}


def _json_copy(value):
    import json

    # Canonical encoding rejects tensors, arbitrary objects, and nonfinite JSON.
    return json.loads(canonical(value))


def _integer(value, label, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise DiscoveryStorageError(f"invalid {label}")
    return value


def _validate_prefix(trace, pc, steps, acts, environment):
    from .forward_recording import tensor_descriptor

    _integer(pc,"frontier PC",len(trace["events"]))
    if not isinstance(steps,list) or len(steps) != pc:
        raise DiscoveryStorageError("frontier prefix length differs from PC")
    for index,row in enumerate(steps):
        if (set(row) != {"pc","event","exact_outputs"} or row["pc"] != index
                or row["event"] != trace["events"][index] or type(row["exact_outputs"]) is not bool):
            raise DiscoveryStorageError("frontier prefix event identity changed")
    if not isinstance(acts,list):
        raise DiscoveryStorageError("frontier Acts must be a list")
    last, previous_pc = {},0
    for receipt in acts:
        if not isinstance(receipt,dict) or receipt.get("fingerprint") != digest({
                key:value for key,value in receipt.items() if key != "fingerprint"}):
            raise DiscoveryStorageError("frontier Act receipt seal differs")
        at = receipt.get("pc")
        _integer(at,"Act PC",pc)
        if at < previous_pc or receipt.get("operation") != "mdb.discovery.ssa-act":
            raise DiscoveryStorageError("frontier Act ordering or operation differs")
        previous_pc = at
        key = receipt.get("value_id")
        descriptor = trace["values"].get(key)
        if (descriptor is None or descriptor["producer"] is None or descriptor["producer"] >= at
                or any(key in event["inputs"] for event in trace["events"][:at])):
            raise DiscoveryStorageError("frontier Act targets a root, future or consumed value")
        if key in last and last[key]["after"] != receipt["before"]:
            raise DiscoveryStorageError("frontier Act value lineage differs")
        last[key] = receipt
    for key,receipt in last.items():
        if tensor_descriptor(environment[key]) != receipt["after"]:
            raise DiscoveryStorageError("frontier edited value differs from its last Act")


def _validate_environment(trace, pc, environment):
    import torch

    from .forward_recording import tensor_descriptor

    expected = {key for key,row in trace["values"].items()
                if row["producer"] is None or row["producer"] < pc}
    if set(environment) != expected:
        raise DiscoveryStorageError("frontier must contain exactly roots and completed definitions")
    for key,value in environment.items():
        row = trace["values"][key]
        if (not isinstance(value,torch.Tensor) or value.device.type != "cpu"
                or value.layout != torch.strided or value.is_conj() or value.is_neg()):
            raise DiscoveryStorageError("frontier supports ordinary CPU strided tensor views only")
        if list(value.shape) != row["shape"] or str(value.dtype) != row["dtype"]:
            raise DiscoveryStorageError("frontier tensor geometry differs from trace")
        if row["producer"] is None:
            if not value.is_contiguous() or value.storage_offset() != 0:
                raise DiscoveryStorageError("frontier roots differ from the dense root layout contract")
            if tensor_descriptor(value)["sha256"] != row["sha256"]:
                raise DiscoveryStorageError("frontier declared root bytes differ from trace")
        elif not bool(torch.isfinite(value).all()):
            raise DiscoveryStorageError("frontier produced tensor violates finite-value contract")


def _pack_environment(environment):
    import torch

    groups,views,slots = {},{},{}
    for key,value in sorted(environment.items()):
        storage = value.untyped_storage()
        if storage._cdata not in groups:
            name = f"storage_{len(groups):06d}"
            groups[storage._cdata] = name
            # StateCut/LocalStore own the copied bytes. Whole storage, including
            # gaps and unused tails, is necessary for exact alias restoration.
            slots[name] = torch.empty(0,dtype=torch.uint8).set_(
                storage,0,(storage.nbytes(),),(1,)).clone()
        views[key] = {"storage":groups[storage._cdata], "shape":list(value.shape),
                      "stride":list(value.stride()), "offset":value.storage_offset(),
                      "dtype":str(value.dtype)}
    return slots,views


def save_frontier(runtime, store, *, metadata=None,storage_layout="bundle-v1"):
    """Publish a complete, resumable native frontier; return its CAS identity.

    ``metadata`` is optional JSON (for example lane/epoch/driver bindings). It
    shares the frontier's seal; this function does not interpret controller state.
    ``bundle-v1`` places the independent backing storages in one CAS tensor
    slot, retaining individual hashes and view geometry. ``slots-v1`` preserves
    the original publication layout. Both use LocalStore's full durable sync.
    """
    from .discovery_runtime import ReplayForward
    from .forward_symbols import verify_trace

    if not isinstance(runtime,ReplayForward):
        raise DiscoveryStorageError("durable storage requires ReplayForward")
    if storage_layout not in {"bundle-v1","slots-v1"}:
        raise DiscoveryStorageError("unsupported durable storage layout")
    verify_trace(runtime.trace)
    cut = runtime.capture()
    cut.state_cut.verify()
    environment = cut._environment
    _validate_environment(runtime.trace,cut.pc,environment)
    _validate_prefix(runtime.trace,cut.pc,cut._steps,cut._acts,environment)
    slots,views = _pack_environment(environment)
    compact_steps = [{"pc":row["pc"], "exact_outputs":row["exact_outputs"]} for row in cut._steps]
    binding = {"schema":SCHEMA, "profile":PROFILE, "platform":_platform(),
        "storage_layout":storage_layout,
        "trace":copy.deepcopy(runtime.trace), "program_id":runtime.catalog.program_id,
        "trace_fingerprint":runtime.trace["fingerprint"], "pc":cut.pc,"views":views,
        "prefix":compact_steps,"acts":copy.deepcopy(cut._acts),"owner":cut._owner,
        "work":dict(runtime.work),"last_error":_json_copy(runtime.last_error),
        "metadata":_json_copy({} if metadata is None else metadata)}
    storage_payload = {"storage_bundle":slots} if storage_layout == "bundle-v1" else slots
    durable = StateCut(model_identity=runtime.trace["model_identity"],execution=runtime.trace["execution"],
        boundary=SCHEMA,parent=runtime.trace["fingerprint"],_payload={"binding":binding,**storage_payload},
        execution_point={"next_event":cut.pc},surface={"restore_support":"native-durable",
            "profile":PROFILE,"alias_scope":"complete backing storage and view geometry",
            "rng_closure":False,"gpu_hydration":False})
    target = _store(store)
    identifier = target.save(durable)
    return {"schema":SCHEMA,"cut_id":identifier,"pc":cut.pc,"program_id":runtime.catalog.program_id,
            "trace_fingerprint":runtime.trace["fingerprint"],"restore_mode":"native-durable",
            "profile":PROFILE,"storage_bytes":sum(value.numel() for value in slots.values()),
            "storage_groups":len(slots),"storage_layout":storage_layout,"admission":target.last_admission}


def _preflight_manifest(manifest, max_storage_bytes):
    surface = manifest.get("surface",{})
    described_profile = {row["key"].get("value"):row["value"].get("value")
                         for row in surface.get("items",[]) if row["key"].get("kind") == "scalar"}
    if manifest.get("boundary") != SCHEMA or described_profile.get("profile") != PROFILE:
        raise DiscoveryStorageError("unsupported durable frontier profile")
    total = 0
    pending = [manifest["payload"]]
    while pending:
        item = pending.pop()
        if not isinstance(item,dict):
            raise DiscoveryStorageError("invalid typed frontier manifest")
        kind = item.get("kind")
        if kind == "tensor":
            shape = item.get("shape")
            if (item.get("dtype") != "torch.uint8" or not isinstance(shape,list) or len(shape) != 1
                    or type(shape[0]) is not int or shape[0] < 0 or item.get("stride") != [1]):
                raise DiscoveryStorageError("durable frontier pages must be contiguous storage bytes")
            total += shape[0]
            if total > max_storage_bytes:
                raise DiscoveryStorageError("durable frontier exceeds storage byte budget before hydration")
        elif kind == "mapping":
            pending.extend(row["value"] for row in item["items"])
            pending.extend(row["key"] for row in item["items"])
        elif kind in {"list","tuple"}:
            pending.extend(item["items"])
        elif kind != "scalar":
            raise DiscoveryStorageError("unknown typed frontier manifest value")
    return total


def inspect_frontier(store, identifier, *, max_storage_bytes=DEFAULT_MAX_STORAGE_BYTES):
    """Verify the immutable metadata and resource declaration without Torch."""
    _integer(max_storage_bytes,"storage byte budget")
    manifest = _store(store).inspect(identifier)
    total = _preflight_manifest(manifest,max_storage_bytes)
    return {"schema":SCHEMA,"cut_id":identifier,"restore_mode":"native-durable",
            "profile":PROFILE,"storage_bytes":total,"manifest":manifest,
            "verification":"StateCut manifest identity; tensor payload verification occurs on load"}


def _unpack_environment(binding, slots, max_tensor_elements):
    import torch

    views = binding.get("views")
    if not isinstance(views,dict):
        raise DiscoveryStorageError("missing durable frontier view table")
    used,environment = set(),{}
    for key,row in views.items():
        if not isinstance(row,dict) or set(row) != {"storage","shape","stride","offset","dtype"}:
            raise DiscoveryStorageError("invalid durable frontier view descriptor")
        storage = row["storage"]
        if storage not in slots:
            raise DiscoveryStorageError("view references unavailable backing storage")
        backing = slots[storage]
        if (not isinstance(backing,torch.Tensor) or backing.dtype != torch.uint8
                or backing.device.type != "cpu" or backing.ndim != 1 or not backing.is_contiguous()
                or backing.storage_offset() != 0):
            raise DiscoveryStorageError("backing storage is not owned contiguous CPU bytes")
        dtype_name = row["dtype"]
        if not isinstance(dtype_name,str) or dtype_name.removeprefix("torch.") not in _DTYPES:
            raise DiscoveryStorageError("unsupported durable tensor dtype")
        dtype = getattr(torch,dtype_name.removeprefix("torch."))
        shape,stride,offset = row["shape"],row["stride"],row["offset"]
        if (not isinstance(shape,list) or not isinstance(stride,list) or len(shape) != len(stride)
                or len(shape)>32 or any(type(v) is not int or v < 0 for v in shape+stride)):
            raise DiscoveryStorageError("invalid durable tensor geometry")
        _integer(offset,"tensor storage offset")
        elements = math.prod(shape)
        if elements > max_tensor_elements:
            raise DiscoveryStorageError("durable tensor exceeds logical element budget")
        width = torch.empty((),dtype=dtype).element_size()
        required = offset*width if elements == 0 else (offset+sum((n-1)*s for n,s in zip(shape,stride))+1)*width
        if required > backing.numel():
            raise DiscoveryStorageError("durable tensor view exceeds backing storage")
        environment[key] = torch.empty(0,dtype=dtype).set_(
            backing.untyped_storage(),offset,tuple(shape),tuple(stride))
        used.add(storage)
    if used != set(slots):
        raise DiscoveryStorageError("durable frontier contains unreferenced backing storage")
    return environment


def load_frontier_record(store, identifier, *, expected_program_id=None,
                         expected_trace_fingerprint=None,max_storage_bytes=DEFAULT_MAX_STORAGE_BYTES,
                         max_tensor_elements=DEFAULT_MAX_TENSOR_ELEMENTS):
    """Hydrate a native cursor and its bound JSON metadata, without prefix replay."""
    from .discovery_catalog import StructuralCatalog
    from .discovery_runtime import ReplayForward
    from .forward_recording import ATenStepper
    from .forward_symbols import verify_trace

    _integer(max_tensor_elements,"logical element budget")
    source = _store(store)
    inspect_frontier(source,identifier,max_storage_bytes=max_storage_bytes)
    durable = source.load(identifier)
    durable.verify()
    payload = durable.payload
    binding = payload.pop("binding",None)
    if not isinstance(binding,dict) or binding.get("schema") != SCHEMA or binding.get("profile") != PROFILE:
        raise DiscoveryStorageError("unsupported durable frontier binding")
    storage_layout = binding.get("storage_layout","slots-v1")
    if storage_layout == "bundle-v1":
        if set(payload) != {"storage_bundle"} or not isinstance(payload["storage_bundle"],dict):
            raise DiscoveryStorageError("durable storage bundle inventory differs")
        payload = payload["storage_bundle"]
    elif storage_layout != "slots-v1":
        raise DiscoveryStorageError("unsupported durable storage layout")
    thread_restore = _restore_platform(binding.get("platform"))
    trace = binding.get("trace")
    verify_trace(trace)
    if trace["fingerprint"] != binding.get("trace_fingerprint"):
        raise DiscoveryStorageError("durable trace identity differs")
    catalog = StructuralCatalog(trace)
    if catalog.program_id != binding.get("program_id"):
        raise DiscoveryStorageError("durable program identity differs")
    if expected_program_id is not None and expected_program_id != catalog.program_id:
        raise DiscoveryStorageError("requested program binding differs from durable frontier")
    if expected_trace_fingerprint is not None and expected_trace_fingerprint != trace["fingerprint"]:
        raise DiscoveryStorageError("requested trace binding differs from durable frontier")
    if (durable.model_identity != trace["model_identity"] or dict(durable.execution) != trace["execution"]
            or durable.parent != trace["fingerprint"]):
        raise DiscoveryStorageError("durable StateCut binding differs from trace")
    pc = _integer(binding.get("pc"),"frontier PC",len(trace["events"]))
    if dict(durable.execution_point or {}) != {"next_event":pc}:
        raise DiscoveryStorageError("durable execution point differs from frontier PC")
    environment = _unpack_environment(binding,payload,max_tensor_elements)
    _validate_environment(trace,pc,environment)
    prefix = binding.get("prefix")
    if not isinstance(prefix,list) or any(not isinstance(row,dict) or set(row) != {"pc","exact_outputs"}
                                          for row in prefix):
        raise DiscoveryStorageError("invalid durable prefix descriptors")
    steps = [{**row,"event":trace["events"][i]} for i,row in enumerate(prefix) if i<len(trace["events"])]
    if len(steps) != len(prefix):
        raise DiscoveryStorageError("durable prefix exceeds program")
    acts = binding.get("acts")
    _validate_prefix(trace,pc,steps,acts,environment)
    work = binding.get("work")
    if not isinstance(work,dict) or any(not isinstance(key,str) or type(value) is not int or value < 0
                                         for key,value in work.items()):
        raise DiscoveryStorageError("invalid durable physical work counters")
    required_work = {"replay_primitives","attempted_primitives","restores","forks"}
    if not required_work.issubset(work):
        raise DiscoveryStorageError("durable work counter inventory differs")
    owner = binding.get("owner")
    if not isinstance(owner,str) or not owner:
        raise DiscoveryStorageError("durable input ownership is unavailable")
    runtime = object.__new__(ReplayForward)
    runtime.trace,runtime.catalog,runtime._owner = trace,catalog,owner
    runtime._machine = object.__new__(ATenStepper)
    runtime._machine.trace,runtime._machine.pc = trace,pc
    runtime._machine.environment,runtime._machine.steps = environment,steps
    runtime._acts,runtime.work = acts,dict(work)
    runtime.work["durable_loads"] = runtime.work.get("durable_loads",0)+1
    runtime.last_error = _json_copy(binding.get("last_error"))
    runtime.durable_origin = {"cut_id":identifier,"restore_mode":"native-durable","profile":PROFILE,
                              "storage_layout":storage_layout,
                              "prefix_replayed":False,"physical_attempts_after_checkpoint":"not-in-snapshot",
                              "cpu_thread_restore":thread_restore}
    return runtime,_json_copy(binding.get("metadata",{}))


def load_frontier(store, identifier, **kwargs):
    return load_frontier_record(store,identifier,**kwargs)[0]


SESSION_SCHEMA = "mdb-discovery-durable-session-v1"


def _snapshot_driver(driver):
    if driver is None:
        return None
    from .discovery_isa import ISADriver

    if type(driver) is not ISADriver:
        raise DiscoveryStorageError("durable session supports the explicit R3 ISA driver only")
    return {"state":driver.state(),"commits":_json_copy(driver.commits)}


def _restore_driver(saved):
    if saved is None:
        return None
    from .discovery_isa import ISADriver

    state = saved["state"]
    if state.get("driver") != "r3-predicted-length":
        raise DiscoveryStorageError("unsupported durable instruction driver")
    driver = ISADriver(state["program_hex"],state["register_words"],state["flags_CF_ZF_SF_OF"],
        offset=state["byte_offset"],max_instructions=state["max_instructions"],
        store_policy=state["store_policy"],syscall_policy=state["syscall_policy"])
    driver.instruction_index = _integer(state["instruction_index"],"driver instruction index",
                                       state["max_instructions"])
    if driver.state() != state:
        raise DiscoveryStorageError("durable driver state/binding differs")
    commits = saved.get("commits")
    if not isinstance(commits,list) or len(commits) != driver.instruction_index:
        raise DiscoveryStorageError("durable driver commit history differs from instruction index")
    previous = None
    for index,commit in enumerate(commits):
        if (not isinstance(commit,dict) or commit.get("committed") is not True
                or commit["before"]["instruction_index"] != index
                or commit["after"]["instruction_index"] != index+1
                or (previous is not None and previous != commit["before"])):
            raise DiscoveryStorageError("durable driver commit lineage differs")
        previous = commit["after"]
    if previous is not None and previous != state:
        raise DiscoveryStorageError("durable driver architecture differs from last commit")
    driver.commits = _json_copy(commits)
    return driver


def save_session_frontier(session,lane_id,store,*,storage_layout="bundle-v1"):
    """Durably select one lane/epoch/branch and its current architectural state.

    Other branches, archived inputs, in-flight commands and the retry cache are
    not silently serialized. Model factories are rebound explicitly on resume.
    Existing alias/break/watch bindings are retained as descriptive metadata.
    """
    from .discovery import DiscoverySession

    if not isinstance(session,DiscoverySession):
        raise DiscoveryStorageError("durable session requires DiscoverySession")
    if storage_layout not in {"bundle-v1","slots-v1"}:
        raise DiscoveryStorageError("unsupported durable storage layout")
    target = _store(store)
    with session._lock:
        if lane_id not in session.lanes:
            raise DiscoveryStorageError("unknown durable lane")
        lane = session.lanes[lane_id]
        if lane.status in {"running","closed"}:
            raise DiscoveryStorageError("durable session capture requires a stopped, open lane")
        branch = lane.branch
        runtime = branch.runtime
        native = save_frontier(runtime,target,storage_layout=storage_layout) if runtime is not None else None
        model_runtime = runtime
        if model_runtime is None:
            prior = [cut["runtime"] for cut in session.cuts.values()
                     if cut["runtime"] is not None and cut["target"]["lane_id"] == lane.id
                     and cut["target"]["epoch_id"] == lane.epoch_id]
            model_runtime = prior[-1] if prior else None
        model_binding = None if model_runtime is None else {
            "model_identity":model_runtime.trace["model_identity"],
            "execution":model_runtime.trace["execution"],"program_id":model_runtime.catalog.program_id}
        watches = []
        for row in session.watches.values():
            copied = {key:copy.deepcopy(value) for key,value in row.items() if key != "baseline"}
            copied["baseline"] = [{"target":list(key),"value":value}
                                   for key,value in row.get("baseline",{}).items()]
            watches.append(copied)
        snapshot = {"schema":SESSION_SCHEMA,"profile":PROFILE,"platform":_platform(),
            "native_cut":None if native is None else native["cut_id"],"model_binding":model_binding,
            "target":session._target(lane),"status":lane.status,"phase":branch.phase,
            "forward_index":branch.forward_index,"blocked":copy.deepcopy(branch.blocked),
            "suppressed":[list(item) for item in sorted(branch.suppressed)],
            "driver":_snapshot_driver(branch.driver),"budgets":dict(session.budgets),
            "work":dict(session.work),"reference_operations":lane.epoch.reference_operations,
            "reference_forwards":lane.epoch.reference_forwards,
            "aliases":copy.deepcopy(session.aliases),"breakpoints":copy.deepcopy(session.breakpoints),
            "watches":watches,"events":copy.deepcopy(session.events),
            "cut_sequence":session._sequence,
            "elapsed_wall_ms":max(0,(time.monotonic()-session._started)*1000),
            "saved_wall_time":time.time(),"retry_cache_restored":False,
            "restored_scope":"selected lane/input epoch/branch only"}
        # JSON validation happens before publication of the selected session cut.
        snapshot = _json_copy(snapshot)
        cut = StateCut(model_identity="mdb.discovery.no-active-model" if model_binding is None else
            model_binding["model_identity"],execution={} if model_binding is None else model_binding["execution"],
            boundary=SESSION_SCHEMA,parent=None if native is None else native["cut_id"],
            _payload={"session":snapshot},execution_point={"lane_id":lane.id,"epoch_id":lane.epoch_id,
                "branch_id":lane.branch_id,"forward_index":branch.forward_index},
            surface={"restore_support":"native-durable","profile":PROFILE,
                "scope":"selected lane/input epoch/branch","factory_binding":"explicit-on-resume",
                "durable_retry":False})
        identifier = target.save(cut)
        return {"schema":SESSION_SCHEMA,"cut_id":identifier,
                "native_cut":snapshot["native_cut"],"target":snapshot["target"],
                "restore_mode":"native-durable","profile":PROFILE,
                "storage_layout":storage_layout,
                "scope":snapshot["restored_scope"],"admission":target.last_admission}


def inspect_session_frontier(store,identifier,*,max_storage_bytes=DEFAULT_MAX_STORAGE_BYTES):
    """Verify session metadata and referenced native manifest without Torch."""
    source = _store(store)
    manifest = source.inspect(identifier)
    if manifest.get("boundary") != SESSION_SCHEMA:
        raise DiscoveryStorageError("unsupported durable session schema")
    pending = [manifest["payload"]]
    while pending:
        item = pending.pop()
        if item.get("kind") == "tensor":
            raise DiscoveryStorageError("durable session metadata cannot contain tensor pages")
        if item.get("kind") == "mapping":
            pending.extend(row["value"] for row in item["items"])
            pending.extend(row["key"] for row in item["items"])
        elif item.get("kind") in {"list","tuple"}:
            pending.extend(item["items"])
    cut = source.load(identifier)
    if cut.boundary != SESSION_SCHEMA:
        raise DiscoveryStorageError("unsupported durable session schema")
    snapshot = cut.payload.get("session")
    if not isinstance(snapshot,dict) or snapshot.get("schema") != SESSION_SCHEMA:
        raise DiscoveryStorageError("invalid durable session binding")
    native = snapshot.get("native_cut")
    if cut.parent != native:
        raise DiscoveryStorageError("durable session/native custody differs")
    native_metadata = inspect_frontier(source,native,max_storage_bytes=max_storage_bytes) if native is not None else None
    return {"schema":SESSION_SCHEMA,"cut_id":identifier,"restore_mode":"native-durable",
            "profile":PROFILE,"session":snapshot,"native_frontier":native_metadata,
            "verification":"session metadata hydrated; native tensor bytes verified on resume"}


def resume_session_frontier(store,identifier,*,factory=None,
                            max_storage_bytes=DEFAULT_MAX_STORAGE_BYTES,
                            max_tensor_elements=DEFAULT_MAX_TENSOR_ELEMENTS,
                            admission=None,continuation_budget=None):
    """Resume the selected native/architectural head with an optional new factory.

    With no factory, finishing the saved forward and committing its existing
    heads remain available. Creating another forward reports missing binding
    before native work is charged. A supplied callable is never deserialized.
    Default wall accounting includes offline elapsed time. An explicit new
    ``admission={"id": ...}`` and complete ``continuation_budget`` may admit a
    new finite lease: counter ceilings are absolute, retain historical work,
    and ``max_wall_ms`` starts at this new admission. Resident IDs must match
    the current mrun job; no admission is silently inferred.
    """
    from .discovery import DiscoveryError, DiscoverySession, _Branch, _Epoch, _Lane
    from .discovery_runtime import ReplayForward

    inspected = inspect_session_frontier(store,identifier,max_storage_bytes=max_storage_bytes)
    snapshot = inspected["session"]
    if snapshot.get("profile") != PROFILE:
        raise DiscoveryStorageError("durable session platform applicability differs")
    thread_restore = _restore_platform(snapshot.get("platform"))
    binding = snapshot.get("model_binding")
    native_id = snapshot["native_cut"]
    runtime = None if native_id is None else load_frontier(store,native_id,
        expected_program_id=None if binding is None else binding["program_id"],
        max_storage_bytes=max_storage_bytes,max_tensor_elements=max_tensor_elements)
    driver = _restore_driver(snapshot["driver"])
    if runtime is not None and binding != {"model_identity":runtime.trace["model_identity"],
            "execution":runtime.trace["execution"],"program_id":runtime.catalog.program_id}:
        raise DiscoveryStorageError("durable session model binding differs from native frontier")
    if factory is None:
        def bound_factory(*_args):
            raise DiscoveryError("durable resume requires a supplied qualified factory before another forward")
        bound_factory.__discovery_missing_binding__ = True
    elif not callable(factory):
        raise DiscoveryStorageError("resume factory must be an explicit callable")
    else:
        def bound_factory(*args):
            from .forward_recording import _leaves, tensor_descriptor

            created = factory(*args)
            if not isinstance(created,ReplayForward) or created.pc != 0:
                raise DiscoveryError("rebound factory must return a root-only ReplayForward")
            actual = {"model_identity":created.trace["model_identity"],"execution":created.trace["execution"],
                      "program_id":created.catalog.program_id}
            if binding is None or actual != binding:
                raise DiscoveryError("rebound factory model/program/shape applicability differs")
            if driver is not None:
                # Recorder root order follows first native use rather than
                # Python argument order. Compare content/geometry multisets,
                # deduplicating repeated references to the same input tensor.
                supplied = {id(value):value for value in _leaves(args)}
                fields = ("shape","dtype","sha256")
                supplied_descriptors = [tensor_descriptor(value) for value in supplied.values()]
                expected_roots = Counter(digest({key:row[key] for key in fields})
                                         for row in supplied_descriptors)
                actual_roots = Counter(digest({key:row[key] for key in fields})
                                       for row in created.trace["values"].values() if row["origin"] == "input")
                if actual_roots != expected_roots:
                    raise DiscoveryError("rebound factory input roots differ from supplied driver arguments")
            return created
    session = DiscoverySession(snapshot["budgets"])
    target = snapshot["target"]
    if not isinstance(target,dict) or set(target) != {"lane_id","epoch_id","branch_id","revision"}:
        raise DiscoveryStorageError("invalid durable session target")
    for key in ("lane_id","epoch_id","branch_id"):
        if not isinstance(target[key],str) or not target[key]:
            raise DiscoveryStorageError("invalid durable session identity")
    _integer(target["revision"],"lane revision")
    phase = snapshot["phase"]
    if phase not in {"input_ready","forward_active","forward_complete","commit_pending",
                      "effect_blocked","driver_terminated"}:
        raise DiscoveryStorageError("invalid durable execution phase")
    if phase in {"forward_active","forward_complete","effect_blocked","commit_pending"} and runtime is None:
        raise DiscoveryStorageError("durable active phase has no native frontier")
    if phase == "forward_complete" and not runtime.complete:
        raise DiscoveryStorageError("durable forward completion has unfinished native work")
    if phase in {"input_ready","effect_blocked","commit_pending","driver_terminated"} and driver is None:
        raise DiscoveryStorageError("durable architectural phase has no instruction driver")
    if phase == "driver_terminated" and not driver.termination:
        raise DiscoveryStorageError("durable terminated phase has a runnable instruction driver")
    if snapshot["status"] not in {"ready","paused","completed","errored"}:
        raise DiscoveryStorageError("invalid durable stopped lane status")
    for key in ("reference_operations","reference_forwards","forward_index"):
        _integer(snapshot[key],key)
    epoch = _Epoch(target["epoch_id"],bound_factory,snapshot["reference_operations"],snapshot["reference_forwards"])
    branch = _Branch(target["branch_id"],runtime,driver,phase=phase,forward_index=snapshot["forward_index"],
                     parent_cut=None,blocked=copy.deepcopy(snapshot["blocked"]))
    lane = _Lane(target["lane_id"],{epoch.id:epoch},epoch.id,target["branch_id"],
                 target["revision"],snapshot["status"])
    epoch.branches[branch.id] = branch
    session.lanes[lane.id] = lane
    session.focus = lane.id
    work = snapshot["work"]
    if (not isinstance(work,dict) or set(work) != set(session.work)
            or any(type(value) is not int or value < 0 for value in work.values())):
        raise DiscoveryStorageError("invalid durable session physical work inventory")
    session.work = dict(work)
    session.aliases = copy.deepcopy(snapshot["aliases"])
    session.breakpoints = copy.deepcopy(snapshot["breakpoints"])
    session.watches = {}
    for watch in snapshot["watches"]:
        watch = copy.deepcopy(watch)
        watch["baseline"] = {tuple(row["target"]):row["value"] for row in watch["baseline"]}
        session.watches[watch["id"]] = watch
    session.events = copy.deepcopy(snapshot["events"])
    session._sequence = _integer(snapshot.get("cut_sequence"),"cut allocator sequence")
    elapsed = snapshot["elapsed_wall_ms"]
    saved_time = snapshot["saved_wall_time"]
    if any(type(value) not in {int,float} or not math.isfinite(value) or value < 0 for value in (elapsed,saved_time)):
        raise DiscoveryStorageError("invalid durable wall budget metadata")
    session._started = time.monotonic()-elapsed/1000-max(0,time.time()-saved_time)
    admission_receipt = None
    if admission is not None or continuation_budget is not None:
        if (not isinstance(admission,dict) or not isinstance(admission.get("id"),str)
                or not admission["id"].strip() or not isinstance(continuation_budget,dict)
                or set(continuation_budget) != set(DiscoverySession.DEFAULT_BUDGETS)):
            raise DiscoveryStorageError("new admission requires an ID and every finite absolute budget ceiling")
        if os.environ.get("SATURN_RESIDENT_WORKER") and admission["id"] != os.environ.get("MRUN_JOB_ID"):
            raise DiscoveryStorageError("resident continuation admission ID must match MRUN_JOB_ID")
        revised = DiscoverySession(continuation_budget).budgets
        counters = {"max_native_operations":"native_operations","max_model_forwards":"model_forwards",
                    "max_branches":"branches","max_commands":"commands",
                    "max_instructions":"architectural_commits"}
        if any(revised[key] < session.work[counter] for key,counter in counters.items()):
            raise DiscoveryStorageError("absolute continuation ceilings cannot erase historical spent work")
        admitted = Receipt.make(operation="mdb.discovery.continuation-admission",status="admitted",
            admission=_json_copy(admission),source_cut=identifier,before_budgets=dict(session.budgets),
            after_budgets=dict(revised),spent_work=dict(session.work),
            ceilings="absolute cumulative counters; max_wall_ms is the newly admitted lease duration",
            historical_elapsed_wall_ms=elapsed,offline_wall_ms=max(0,time.time()-saved_time)*1000,
            durable_retry=False)
        _store(store).receipt(admitted)
        admission_receipt = admitted.to_dict()
        session.budgets = dict(revised)
        session._started = time.monotonic()
        session._event("continuation_admitted",receipt=admission_receipt)
    # Old local cut handles are metadata only; this creates a fresh retained
    # selected head. Retry IDs do not acquire durable exactly-once semantics.
    session._capture(lane,name="resumed")
    session._prime_watches(lane,"durable-resume")
    branch.suppressed = {tuple(item) for item in snapshot["suppressed"]}
    session._event("durable_resumed",cut_id=identifier,target=session._target(lane),
                   native_cut=native_id,prefix_replayed=False,factory_bound=factory is not None,
                   retry_cache_restored=False,restored_scope=snapshot["restored_scope"],
                   cpu_thread_restore=thread_restore,continuation_admission=admission_receipt)
    session.durable_origin = {"cut_id":identifier,"native_cut":native_id,"profile":PROFILE,
                              "restored_scope":snapshot["restored_scope"]}
    return session


__all__ = ["DiscoveryStorageError","save_frontier","load_frontier","load_frontier_record",
           "inspect_frontier","save_session_frontier","resume_session_frontier",
           "inspect_session_frontier","PROFILE"]
