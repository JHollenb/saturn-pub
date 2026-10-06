"""Readable and transactional hidden K/V memory for the Qwen Open Bench.

Reads hydrate exact durable checkpoint bytes without selecting the cut.  Writes
reuse the successor-KV interchange backend: a complete private mount is staged,
checked for masked locality, admitted by Model OS, and published by the owning
bench only after the native StateCut commit succeeds.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from typing import Any

import torch

from .bench_qwen import BENCH_CUT_SCHEMA, BenchError, tensor_digest
from .bench_qwen_memory import (
    ADDRESS,
    PENDING_TOKEN_POLICY,
    SuccessorKVBackend,
    _geometry,
    _locality,
    _stage,
    _visible,
    normalize_selector,
)
from .model_os_contracts import (
    ConsumerContract,
    ModelPackage,
    ModelProgram,
    ProgramInstruction,
    StateSchema,
    StateSlot,
    StateSurface,
    content_digest,
)
from .model_os_kernel import ModelOSKernel

READ_SCHEMA = "saturn-bench-hidden-memory-read-v1"
DIFF_SCHEMA = "saturn-bench-hidden-memory-diff-v1"
WRITE_SCHEMA = "saturn-bench-hidden-memory-write-v1"
DEFAULT_MAX_VALUES = 16
MAX_VALUES = 256
DEFAULT_MAX_ROWS = 32
MAX_ROWS = 256
COMPONENT_INDEX = {"key": 0, "value": 1}


def logical_address(layer: int, component: str, row: int) -> str:
    return f"kv_cache.layer_{layer:03d}.{component}.row_{row:06d}"


def _json_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchError(f"{name} must be an object")
    return dict(value)


def _cut(bench, ref: str | None) -> tuple[str, dict[str, Any]]:
    if not bench.branches:
        raise BenchError("prefill a Qwen prompt before reading hidden memory")
    ref = ref or bench.branch.cut
    if not isinstance(ref, str) or not ref:
        raise BenchError("hidden memory requires a durable cut")
    try:
        cut = bench.artifacts.get(ref)
    except Exception as error:
        raise BenchError("unknown hidden-memory cut") from error
    if (
        cut.get("schema") != BENCH_CUT_SCHEMA
        or cut.get("identity") != bench.identity
        or cut.get("lane") != bench.lane
        or cut.get("numerical_contract") != bench.vm.package.model_abi.fingerprint
    ):
        raise BenchError("hidden-memory cut is incompatible with this Qwen session")
    if cut.get("execution") is not None:
        raise BenchError("hidden memory is readable only at a committed token boundary")
    return ref, cut


def _layers(bench, cut: dict[str, Any]):
    checkpoint = bench.vm._load_checkpoint(cut["vm_checkpoint"])
    read = bench.vm.store.read(
        checkpoint["mount_sha256"],
        checkpoint["page_id"],
        model_abi_sha256=bench.model_abi,
    )
    if len(read.segments) not in {1, 2}:
        raise BenchError("hidden-memory checkpoint has unsupported physical segmentation")
    codec = bench.vm.runtime.codec
    ancestor = codec.decode_segment(read.segments[0], torch.device("cpu"))
    raw_delta = bench.vm._delta_bytes(checkpoint, consumer="bench-hidden-memory-read")
    delta = codec.decode_segment(raw_delta, torch.device("cpu")) if raw_delta else tuple(
        (
            torch.empty((1, codec.groups, 0, codec.head_dim), dtype=codec.dtype),
            torch.empty((1, codec.groups, 0, codec.head_dim), dtype=codec.dtype),
        )
        for _ in range(codec.layers)
    )
    if (
        checkpoint["ancestor_rows"] != ancestor[0][0].shape[2]
        or checkpoint["delta_rows"] != delta[0][0].shape[2]
        or checkpoint["ancestor_rows"] + checkpoint["delta_rows"] != len(cut["token_ids"])
    ):
        raise BenchError("hidden-memory checkpoint row closure changed")
    visible = tuple(
        tuple(torch.cat((ancestor[layer][component], delta[layer][component]), dim=2)
              for component in range(2))
        for layer in range(codec.layers)
    )
    return checkpoint, visible


def _read_options(spec: Any, *, allowed: set[str]) -> dict[str, Any]:
    spec = {} if spec is None else _json_object(spec, name="hidden-memory spec")
    if set(spec) - allowed:
        raise BenchError("hidden-memory spec has unknown fields")
    values = spec.get("values", True)
    maximum = spec.get("max_values", DEFAULT_MAX_VALUES)
    max_rows = spec.get("max_rows", DEFAULT_MAX_ROWS)
    if type(values) is not bool:
        raise BenchError("hidden-memory values must be boolean")
    if type(maximum) is not int or not 1 <= maximum <= MAX_VALUES:
        raise BenchError(f"hidden-memory max_values must be in [1, {MAX_VALUES}]")
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS:
        raise BenchError(f"hidden-memory max_rows must be in [1, {MAX_ROWS}]")
    return {**spec, "values": values, "max_values": maximum, "max_rows": max_rows}


def _page_selector(spec, *, layers: int, rows: int):
    requested = normalize_selector(
        {key: spec[key] for key in ("layers", "row_start", "row_stop", "components") if key in spec},
        layers=layers, rows=rows)
    selected_total = requested["row_stop"] - requested["row_start"]
    selector = dict(requested)
    selector["row_stop"] = min(requested["row_stop"], requested["row_start"] + spec["max_rows"])
    truncated = selector["row_stop"] < requested["row_stop"]
    return selector, {"total_rows": rows, "selected_total_rows": selected_total,
                      "next_row_start": selector["row_stop"] if truncated else None,
                      "truncated": truncated}


def _token_text(bench, token_id: int) -> str | None:
    if bench.tokenizer is None:
        return None
    try:
        return bench.tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        converter = getattr(bench.tokenizer, "convert_ids_to_tokens", None)
        return str(converter(token_id)) if converter is not None else None


def _physical(checkpoint, *, layer: int, component: str, row: int) -> dict[str, Any]:
    ancestor_rows = checkpoint["ancestor_rows"]
    return {
        "mount_sha256": checkpoint["mount_sha256"],
        "page_id": checkpoint["page_id"],
        "region": "ancestor" if row < ancestor_rows else "private",
        "region_row": row if row < ancestor_rows else row - ancestor_rows,
        "layer": layer,
        "component": component,
    }


def _summary(tensor: torch.Tensor, *, include_values: bool, max_values: int) -> dict[str, Any]:
    flat = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    result = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": tensor_digest(tensor),
        "norm": float(torch.linalg.vector_norm(flat).item()),
        "max_abs": float(flat.abs().max().item()) if flat.numel() else 0.0,
        "num_values": int(flat.numel()),
    }
    if include_values:
        count = min(int(flat.numel()), max_values)
        result["values"] = flat[:count].tolist()
        result["values_truncated"] = count < flat.numel()
    return result


@torch.inference_mode()
def memory_read(bench, spec=None):
    """Read selected committed Qwen K/V rows from the current or named cut."""
    if bench.publication.mode != "journal":
        raise BenchError("hidden-memory access requires journal publication mode")
    spec = _read_options(
        spec,
        allowed={"cut_id", "layers", "row_start", "row_stop", "components", "values", "max_values", "max_rows"},
    )
    if "cut_id" not in spec and bench._backend().execution is not None:
        raise BenchError("current hidden memory is unavailable during an in-flight token; name a committed cut_id")
    cut_id, cut = _cut(bench, spec.get("cut_id"))
    checkpoint, layers = _layers(bench, cut)
    selector, page = _page_selector(spec, layers=len(layers), rows=len(cut["token_ids"]))
    rows = []
    for row in range(selector["row_start"], selector["row_stop"]):
        components = {}
        for layer in selector["layers"]:
            for component in selector["components"]:
                address = logical_address(layer, component, row)
                tensor = layers[layer][COMPONENT_INDEX[component]][:, :, row : row + 1, :]
                components[address] = {
                    "layer": layer,
                    "component": component,
                    "logical_address": address,
                    "physical_address": _physical(
                        checkpoint, layer=layer, component=component, row=row
                    ),
                    **_summary(
                        tensor,
                        include_values=spec["values"],
                        max_values=spec["max_values"],
                    ),
                }
        token_id = cut["token_ids"][row]
        rows.append(
            {
                "row": row,
                "token_id": token_id,
                "token_text": _token_text(bench, token_id),
                "components": components,
            }
        )
    return {
        "schema": READ_SCHEMA,
        "cut_id": cut_id,
        "model_identity": bench.identity,
        "numerical_contract": bench.vm.package.model_abi.fingerprint,
        "selector": selector,
        "values": spec["values"],
        "max_values": spec["max_values"],
        "max_rows": spec["max_rows"],
        **page,
        "rows": rows,
    }


@torch.inference_mode()
def memory_diff(bench, spec):
    """Diff aligned committed K/V rows without checking out either cut."""
    if bench.publication.mode != "journal":
        raise BenchError("hidden-memory access requires journal publication mode")
    spec = _read_options(
        spec,
        allowed={"left_cut", "right_cut", "layers", "row_start", "row_stop", "components", "values", "max_values", "max_rows"},
    )
    if set(("left_cut", "right_cut")) - set(spec):
        raise BenchError("memory_diff requires left_cut and right_cut")
    left_id, left = _cut(bench, spec["left_cut"])
    right_id, right = _cut(bench, spec["right_cut"])
    if (
        left["token_ids"] != right["token_ids"]
        or left["prompt_length"] != right["prompt_length"]
    ):
        raise BenchError("memory_diff requires cuts with identical committed token history")
    left_checkpoint, left_layers = _layers(bench, left)
    right_checkpoint, right_layers = _layers(bench, right)
    selector, page = _page_selector(spec, layers=len(left_layers), rows=len(left["token_ids"]))
    rows = []
    for row in range(selector["row_start"], selector["row_stop"]):
        components = {}
        for layer in selector["layers"]:
            for component in selector["components"]:
                address = logical_address(layer, component, row)
                index = COMPONENT_INDEX[component]
                a = left_layers[layer][index][:, :, row : row + 1, :]
                b = right_layers[layer][index][:, :, row : row + 1, :]
                delta = b.to(torch.float32) - a.to(torch.float32)
                components[address] = {
                    "layer": layer,
                    "component": component,
                    "logical_address": address,
                    "left_physical_address": _physical(left_checkpoint, layer=layer, component=component, row=row),
                    "right_physical_address": _physical(right_checkpoint, layer=layer, component=component, row=row),
                    "equal": bool(torch.equal(a, b)),
                    "left": _summary(a, include_values=spec["values"], max_values=spec["max_values"]),
                    "right": _summary(b, include_values=spec["values"], max_values=spec["max_values"]),
                    "delta": _summary(delta, include_values=spec["values"], max_values=spec["max_values"]),
                }
        token_id = left["token_ids"][row]
        rows.append({"row": row, "token_id": token_id, "token_text": _token_text(bench, token_id),
                     "components": components})
    return {"schema": DIFF_SCHEMA, "left_cut": left_id, "right_cut": right_id,
            "model_identity": bench.identity, "selector": selector,
            "max_rows": spec["max_rows"], **page, "rows": rows}


def _selected_addresses(selector: dict[str, Any]) -> list[str]:
    return [logical_address(layer, component, row)
            for row in range(selector["row_start"], selector["row_stop"])
            for layer in selector["layers"] for component in selector["components"]]


def normalize_write_spec(bench, spec) -> dict[str, Any]:
    """Strictly validate a hidden-memory write before any branch or native work."""
    if bench.publication.mode != "journal":
        raise BenchError("hidden-memory writes require journal publication mode")
    if not bench.branches or bench._edit or bench._backend().execution is not None or bench.cursor()["phase"] != "token":
        raise BenchError("hidden-memory writes require a committed token boundary and no pending Act")
    spec = _json_object(spec, name="memory_write spec")
    if set(spec) - {"mode", "selector", "donor_cut", "value", "values"}:
        raise BenchError("memory_write spec has unknown fields")
    mode = spec.get("mode")
    if mode not in {"copy", "zero", "scale", "replace"}:
        raise BenchError("memory_write mode must be copy, zero, scale, or replace")
    geometry = _geometry(bench, bench._backend().current_mount)
    selector = normalize_selector(spec.get("selector"), layers=geometry["layers"], rows=geometry["rows"])
    result = {"mode": mode, "selector": selector}
    allowed = {
        "copy": {"mode", "selector", "donor_cut"},
        "zero": {"mode", "selector"},
        "scale": {"mode", "selector", "value"},
        "replace": {"mode", "selector", "values"},
    }[mode]
    if set(spec) - allowed:
        raise BenchError(f"memory_write {mode} has incompatible payload fields")
    if mode == "copy":
        if not isinstance(spec.get("donor_cut"), str) or not spec["donor_cut"]:
            raise BenchError("copy requires an explicit donor_cut")
        donor_id, donor = _cut(bench, spec["donor_cut"])
        if donor["token_ids"] != bench.branch.token_ids or donor["prompt_length"] != bench.branch.prompt_length:
            raise BenchError("copy requires a donor with identical committed token history")
        result["donor_cut"] = donor_id
    elif mode == "scale":
        value = spec.get("value")
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise BenchError("scale requires one finite scalar value")
        native = torch.as_tensor(float(value), dtype=bench.vm.runtime.codec.dtype)
        if not bool(torch.isfinite(native)):
            raise BenchError("scale value is not finite in the native dtype")
        result["value"] = float(value)
    elif mode == "replace":
        values = spec.get("values")
        if not isinstance(values, dict):
            raise BenchError("replace requires values keyed by logical address")
        expected = _selected_addresses(selector)
        if set(values) != set(expected):
            raise BenchError("replace values must cover exactly the selected logical addresses")
        shape = (1, geometry["kv_heads"], 1, geometry["head_dim"])
        frozen = {}
        for address in expected:
            try:
                tensor = torch.as_tensor(values[address], dtype=geometry_dtype(bench))
            except (TypeError, ValueError, OverflowError) as error:
                raise BenchError(f"replace value for {address} is not numeric") from error
            if tuple(tensor.shape) != shape:
                raise BenchError(f"replace value for {address} must have exact shape {shape}")
            if not bool(torch.isfinite(tensor).all()):
                raise BenchError(f"replace value for {address} is non-finite in the native dtype")
            frozen[address] = tensor.to(torch.float32).tolist()
        result["values"] = frozen
    return result


def geometry_dtype(bench):
    return bench.vm.runtime.codec.dtype


def _recipient_layers(recipient):
    return tuple(
        (
            tuple(t.clone() for t in recipient.ancestor[layer]),
            tuple(t.clone() for t in recipient.delta[layer]),
        )
        for layer in range(len(recipient.ancestor))
    )


def _transform_layers(bench, recipient, normalized):
    mode, selector = normalized["mode"], normalized["selector"]
    if mode == "copy":
        donor = bench.artifacts.get(normalized["donor_cut"])
        _, visible = _layers(bench, donor)
        device = recipient.ancestor[0][0].device
        return tuple((
            tuple(visible[layer][component][:, :, :recipient.ancestor_rows, :].to(device)
                  for component in range(2)),
            tuple(visible[layer][component][:, :, recipient.ancestor_rows:, :].to(device)
                  for component in range(2)),
        ) for layer in range(len(visible)))
    donor_layers = _recipient_layers(recipient)
    start, stop = selector["row_start"], selector["row_stop"]
    for layer in selector["layers"]:
        for component in selector["components"]:
            index = COMPONENT_INDEX[component]
            visible = torch.cat((donor_layers[layer][0][index],
                                 donor_layers[layer][1][index][:, :, :recipient.delta_rows]), dim=2)
            for row in range(start, stop):
                target = visible[:, :, row : row + 1, :]
                if mode == "zero":
                    target.zero_()
                elif mode == "scale":
                    target.mul_(normalized["value"])
                else:
                    address = logical_address(layer, component, row)
                    target.copy_(torch.as_tensor(normalized["values"][address], dtype=target.dtype,
                                                 device=target.device))
            ancestor = donor_layers[layer][0][index]
            ancestor.copy_(visible[:, :, :recipient.ancestor_rows])
            if stop > recipient.ancestor_rows:
                donor_layers[layer][1][index][:, :, :recipient.delta_rows].copy_(
                    visible[:, :, recipient.ancestor_rows:]
                )
    return donor_layers


def _persist_candidate(bench, candidate):
    """Turn the private candidate into an immutable mount before admission."""
    codec = bench.vm.runtime.codec
    visible = tuple(tuple(_visible(candidate, layer, component) for component in range(2))
                    for layer in range(codec.layers))
    raw = codec.encode_layers(visible, candidate.length)
    rows = tuple(raw[offset : offset + codec.row_bytes]
                 for offset in range(0, len(raw), codec.row_bytes))
    page_id = f"hidden-memory-{content_digest({'rows': [tensor_digest(t) for pair in visible for t in pair]})[:24]}"
    mount = bench.vm.store.publish(model_abi_sha256=bench.model_abi, pages={page_id: rows})
    return bench.vm.runtime.mount(mount, page_id, extra_rows=candidate.delta_capacity)


def _transaction(bench, normalized, *, parent_cut: str):
    owner = bench._backend()
    recipient = owner.current_mount
    geometry = _geometry(bench, recipient)
    selector = normalized["selector"]
    donor_layers = _transform_layers(bench, recipient, normalized)
    candidate = _stage(recipient, donor_layers, selector)
    locality = _locality(recipient, candidate, donor_layers, selector)
    candidate = _persist_candidate(bench, candidate)
    # The durable private mount, rather than only the transient staging copy,
    # must satisfy the same selected-payload and masked-locality contract.
    locality = _locality(recipient, candidate, donor_layers, selector)
    selected = {
        logical_address(layer, component, row): tensor_digest(
            _visible(candidate, layer, COMPONENT_INDEX[component])[:, :, row : row + 1, :]
        )
        for row in range(selector["row_start"], selector["row_stop"])
        for layer in selector["layers"] for component in selector["components"]
    }
    donor_cut = normalized.get("donor_cut", parent_cut)
    donor = bench.artifacts.get(donor_cut)
    history = {"length": len(bench.branch.token_ids),
               "token_ids_sha256": content_digest(list(bench.branch.token_ids))}
    pending = {"recipient_token_id": bench.branch.pending,
               "donor_token_id": donor["pending"], "policy": PENDING_TOKEN_POLICY}
    payload = {key: normalized[key] for key in ("donor_cut", "value") if key in normalized}
    if "values" in normalized:
        payload["values_sha256"] = content_digest(normalized["values"])
    act_id = content_digest({"schema": WRITE_SCHEMA, "parent_cut": parent_cut,
        "model_abi": bench.model_abi, "native_abi": bench.vm.package.model_abi.fingerprint,
        "mode": normalized["mode"], "selector": selector, "history": history,
        "pending": pending, "payload": payload, "selected_payload": selected})
    schema = StateSchema(name="saturn-bench-qwen-hidden-memory-v1", version=1, slots=(StateSlot(
        address=ADDRESS, value_type="typed-kv-selection", owner="bench-qwen",
        dtype=geometry["dtype"],
        shape=(geometry["batch"], geometry["layers"], 2, geometry["kv_heads"],
               geometry["rows"], geometry["head_dim"]),
        read_scopes=("model-os",), write_scopes=("model-os",), lifetime="branch",
        branchable=True, persistable=True, freshness="candidate",
        lineage=donor_cut,
        metadata={"mode": normalized["mode"], "geometry": geometry, "selector": selector,
                  "pending_token_policy": PENDING_TOKEN_POLICY}),))
    abi = replace(bench.vm.package.model_abi, state_schema=schema.fingerprint)
    consumer = ConsumerContract(consumer_id="bench.qwen.hidden-memory-install", reads=(ADDRESS,),
        output_kind="hidden-memory-write-receipt", continuation_tokens=0, native=True)
    program = ModelProgram(program_id="bench.qwen.hidden-memory-install.v1", version=1,
        instructions=(ProgramInstruction(instruction_id="install", op="Write", reads=(ADDRESS,),
            writes=(ADDRESS,), outputs=(ADDRESS,), authority="model-os"),),
        portable=False, terminal_claim=False)
    package = ModelPackage(package_id="bench.qwen.hidden-memory-install", version=1,
        model_abi=abi, state_schema=schema,
        state_surface=StateSurface.from_schema(schema, surface_id="bench-hidden-memory", owner="bench-qwen"),
        program=program, consumers=(consumer,),
        permissions={"subjects": {"model-os": {"read": [ADDRESS], "write": [ADDRESS]}},
                     "authorities": ["model-os"]})
    reference = {"schema": WRITE_SCHEMA, "act_id": act_id, "mode": normalized["mode"],
        "parent_cut": parent_cut, "donor_cut": donor_cut, "history": history, "pending": pending,
        "geometry": geometry, "selector": selector, "payload": payload,
        "selected_payload": selected, "pending_token_policy": PENDING_TOKEN_POLICY}
    backend = SuccessorKVBackend(model_abi=abi, state_schema=schema)
    backend.owner, backend.candidate_mount, backend.reference = owner, candidate, reference
    backend.geometry, backend.selector, backend.locality = geometry, selector, locality
    backend.act_id = act_id
    backend.donor_cut = donor_cut
    return ModelOSKernel.install(package, backend), {ADDRESS: reference}, {
        "schema": WRITE_SCHEMA, "act_id": act_id, "mode": normalized["mode"],
        "parent_cut": parent_cut, "donor_cut": donor_cut, "history": history, "pending": pending,
        "geometry": geometry, "selector": selector, "payload": payload,
        "selected_payload": selected, "locality": locality,
        "pending_token_policy": PENDING_TOKEN_POLICY}


@torch.inference_mode()
def memory_write(bench, spec):
    """Atomically commit one selected Qwen K/V transform at the current cut."""
    normalized = normalize_write_spec(bench, spec)
    parent = bench.branch.cut
    previous_lineage = list(bench.branch.intervention_lineage)
    previous_events = list(bench.events)
    previous_cuts = deepcopy(bench.cuts)
    kernel, updates, plan = _transaction(bench, normalized, parent_cut=parent)
    try:
        with bench.publication.atomic_selection():
            result = kernel.execute(updates=updates, horizon=0, metadata={
                "session_id": bench.session_id, "branch": bench.active, "parent_cut": parent,
                "mode": normalized["mode"], "donor_cut": plan["donor_cut"]})
            if result.receipt.decision != "commit":
                raise BenchError("hidden-memory Act did not commit")
            receipt_ref = bench.artifacts.put(result.receipt.to_payload())
            bench.branch.intervention_lineage.append(plan["act_id"])
            cut = bench.checkpoint("hidden-memory-committed")
            frozen = {**plan, "cut_id": cut, "parent_cut": parent, "receipt_ref": receipt_ref,
                "plan": {"writes": ["successor_kv"], "authority": "model-os",
                         "mode": plan["mode"], "selector": plan["selector"]}}
            act_ref = bench.artifacts.put(frozen)
            bench.events.append({"kind": "hidden-memory-write", "act_ref": act_ref,
                "act_id": plan["act_id"], "parent": parent, "cut_id": cut,
                "receipt_ref": receipt_ref, "mode": plan["mode"]})
            bench._publish()
            return {**frozen, "act_ref": act_ref}
    except BaseException:
        bench.branch.intervention_lineage = previous_lineage
        bench.events = previous_events
        bench.cuts = previous_cuts
        bench._restore(parent)
        raise


def verify_write_act(act, artifacts, vm_artifacts=None):
    """Verify a frozen hidden-memory Act and its durable transaction closure."""
    if not isinstance(act, dict) or act.get("schema") != WRITE_SCHEMA:
        raise ValueError("hidden-memory Act schema changed")
    required = {"act_id", "mode", "parent_cut", "donor_cut", "receipt_ref", "cut_id",
                "geometry", "selector", "locality", "pending_token_policy", "history",
                "pending", "payload", "selected_payload"}
    if required - set(act):
        raise ValueError("hidden-memory Act closure is incomplete")
    parent = artifacts.get(act["parent_cut"])
    donor = artifacts.get(act["donor_cut"])
    cut = artifacts.get(act["cut_id"])
    receipt = artifacts.get(act["receipt_ref"])
    receipt_body = dict(receipt)
    receipt_digest = receipt_body.pop("content_sha256", None)
    if receipt_digest != content_digest(receipt_body) or receipt.get("decision") != "commit":
        raise ValueError("hidden-memory receipt is not a valid commit")
    if any(item.get("schema") != BENCH_CUT_SCHEMA for item in (parent, donor, cut)):
        raise ValueError("hidden-memory Act references a non-bench cut")
    if (donor["identity"] != parent["identity"] or donor["lane"] != parent["lane"]
            or donor["numerical_contract"] != parent["numerical_contract"]
            or donor["token_ids"] != parent["token_ids"]
            or donor["prompt_length"] != parent["prompt_length"]):
        raise ValueError("hidden-memory donor/history binding changed")
    expected_history = {"length": len(parent["token_ids"]),
                        "token_ids_sha256": content_digest(parent["token_ids"])}
    expected_pending = {"recipient_token_id": parent["pending"],
                        "donor_token_id": donor["pending"],
                        "policy": PENDING_TOKEN_POLICY}
    if act["history"] != expected_history or act["pending"] != expected_pending:
        raise ValueError("hidden-memory history or pending-token binding changed")
    if act["mode"] not in {"copy", "zero", "scale", "replace"}:
        raise ValueError("hidden-memory write mode changed")
    payload = act["payload"]
    if not isinstance(payload, dict):
        raise ValueError("hidden-memory payload binding changed")
    if act["mode"] == "copy" and payload.get("donor_cut") != act["donor_cut"]:
        raise ValueError("hidden-memory copy donor binding changed")
    if act["mode"] == "zero" and payload:
        raise ValueError("hidden-memory zero payload changed")
    if act["mode"] == "scale" and (set(payload) != {"value"}
            or type(payload["value"]) not in {int, float} or not math.isfinite(float(payload["value"]))):
        raise ValueError("hidden-memory scale payload changed")
    if act["mode"] == "replace" and (set(payload) != {"values_sha256"}
            or not isinstance(payload["values_sha256"], str) or len(payload["values_sha256"]) != 64):
        raise ValueError("hidden-memory replacement payload changed")
    geometry, selector = act["geometry"], act["selector"]
    if (not isinstance(geometry, dict) or geometry.get("rows") != len(parent["token_ids"])
            or not isinstance(selector, dict) or set(selector) != {"layers", "row_start", "row_stop", "components"}
            or not 0 <= selector["row_start"] < selector["row_stop"] <= geometry["rows"]
            or any(type(layer) is not int or not 0 <= layer < geometry["layers"] for layer in selector["layers"])
            or any(component not in COMPONENT_INDEX for component in selector["components"])):
        raise ValueError("hidden-memory geometry or selector changed")
    expected_addresses = set(_selected_addresses(selector))
    selected = act["selected_payload"]
    if (not isinstance(selected, dict) or set(selected) != expected_addresses
            or any(not isinstance(digest, str) or len(digest) != 64 for digest in selected.values())):
        raise ValueError("hidden-memory selected-payload closure changed")
    locality = act["locality"]
    if (not isinstance(locality, dict) or locality.get("selected_payload_exact") is not True
            or locality.get("masked_locality_exact") is not True
            or type(locality.get("changed_elements")) is not int or locality["changed_elements"] < 0):
        raise ValueError("hidden-memory locality receipt changed")
    expected_act_id = content_digest({"schema": WRITE_SCHEMA, "parent_cut": act["parent_cut"],
        "model_abi": content_digest(parent["identity"]), "native_abi": parent["numerical_contract"],
        "mode": act["mode"], "selector": selector, "history": expected_history,
        "pending": expected_pending, "payload": payload, "selected_payload": selected})
    if act["act_id"] != expected_act_id:
        raise ValueError("hidden-memory Act identity changed")
    if (cut["parent_cut"] != act["parent_cut"] or cut["identity"] != parent["identity"]
            or cut["lane"] != parent["lane"] or cut["token_ids"] != parent["token_ids"]
            or cut["prompt_length"] != parent["prompt_length"] or cut["pending"] != parent["pending"]
            or not cut["intervention_lineage"] or cut["intervention_lineage"][-1] != act["act_id"]):
        raise ValueError("hidden-memory committed cut binding changed")
    continuation, backend = receipt.get("continuation", {}), receipt.get("backend", {})
    if (receipt.get("schema") != "saturn-model-os-execution-receipt-v1"
            or receipt.get("consumer_id") != "bench.qwen.hidden-memory-install"
            or continuation.get("accepted") is not True
            or continuation.get("consumer_id") != receipt["consumer_id"]
            or continuation.get("horizon") != 0
            or continuation.get("execution_mode") != "native-successor-kv-install"
            or continuation.get("act_id") != act["act_id"]
            or continuation.get("donor_cut") != act["donor_cut"]
            or continuation.get("recipient_policy") != PENDING_TOKEN_POLICY
            or continuation.get("geometry") != geometry or continuation.get("selector") != selector
            or continuation.get("locality") != locality
            or backend.get("execution_mode") != "native-successor-kv-install"
            or backend.get("locality") != locality
            or not isinstance(backend.get("native_statecut"), dict)):
        raise ValueError("hidden-memory native receipt closure changed")
    statecut = dict(backend["native_statecut"])
    statecut_digest = statecut.pop("content_sha256", None)
    if statecut_digest != content_digest(statecut):
        raise ValueError("hidden-memory native StateCut identity changed")
    if vm_artifacts is not None:
        checkpoint = vm_artifacts.get(cut["vm_checkpoint"])
        if checkpoint["ancestor_rows"] + checkpoint["delta_rows"] != len(cut["token_ids"]):
            raise ValueError("hidden-memory VM checkpoint row closure changed")
    return True


__all__ = [
    "DEFAULT_MAX_ROWS", "DEFAULT_MAX_VALUES", "DIFF_SCHEMA", "MAX_ROWS", "MAX_VALUES", "READ_SCHEMA", "WRITE_SCHEMA",
    "logical_address", "memory_diff", "memory_read", "memory_write", "normalize_write_spec",
    "verify_write_act",
]
