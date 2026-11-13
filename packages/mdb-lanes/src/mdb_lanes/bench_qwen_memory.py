"""Typed successor-KV interchange for the Qwen Open Bench.

The operation is deliberately narrower than an arbitrary cache writer.  It
copies a rectangular row/depth/component selection between two committed cuts
with identical history and native geometry.  Model OS admits the write; the
native backend privately stages a complete mount and publishes it atomically.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from .ar.causal_context_qwen import CausalContextQwenMount
from .bench_qwen import BenchError, tensor_digest
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
from .model_os_kernel import InMemoryModelBackend, ModelOSExecutionError, ModelOSKernel

ADDRESS = "bench.successor_kv"
PENDING_TOKEN_POLICY = "preserve-recipient-history-pending-logits;fixed-query-before-consumer"


def normalize_selector(value: Any, *, layers: int, rows: int) -> dict[str, Any]:
    """Validate and freeze the bounded public selector."""
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - {
        "layers",
        "row_start",
        "row_stop",
        "components",
    }:
        raise BenchError("Qwen memory_selector has unknown fields")
    selected_layers = value.get("layers", list(range(layers)))
    if (
        not isinstance(selected_layers, list)
        or not selected_layers
        or any(type(layer) is not int or not 0 <= layer < layers for layer in selected_layers)
        or len(set(selected_layers)) != len(selected_layers)
    ):
        raise BenchError("Qwen memory_selector layers must be distinct native decoder depths")
    start = value.get("row_start", 0)
    stop = value.get("row_stop", rows)
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= rows:
        raise BenchError("Qwen memory_selector row span is outside the committed successor history")
    components = value.get("components", ["key", "value"])
    if (
        not isinstance(components, list)
        or not components
        or any(component not in {"key", "value"} for component in components)
        or len(set(components)) != len(components)
    ):
        raise BenchError("Qwen memory_selector components must select key, value, or both")
    return {
        "layers": sorted(selected_layers),
        "row_start": start,
        "row_stop": stop,
        "components": sorted(components),
    }


def _geometry(bench, mount: CausalContextQwenMount) -> dict[str, Any]:
    codec = bench.vm.runtime.codec
    return {
        "batch": 1,
        "layers": codec.layers,
        "kv_heads": codec.groups,
        "rows": mount.length,
        "head_dim": codec.head_dim,
        "ancestor_rows": mount.ancestor_rows,
        "private_rows": mount.delta_rows,
        "private_capacity": mount.delta_capacity,
        "dtype": str(codec.dtype),
        "device": str(bench.vm.runtime.device),
        "row_bytes": codec.row_bytes,
        "layout": "batch,kv_head,token,head_dim",
    }


def _visible(mount: CausalContextQwenMount, layer: int, component: int) -> torch.Tensor:
    return torch.cat(
        (mount.ancestor[layer][component], mount.delta[layer][component][:, :, : mount.delta_rows]),
        dim=2,
    )


def _donor_layers(bench, cut: dict[str, Any], recipient: CausalContextQwenMount):
    checkpoint = bench.vm._load_checkpoint(cut["vm_checkpoint"])
    if (
        checkpoint["mount_sha256"] != recipient.mount_sha256
        or checkpoint["page_id"] != recipient.page_id
        or checkpoint["ancestor_rows"] != recipient.ancestor_rows
        or checkpoint["delta_rows"] != recipient.delta_rows
        or checkpoint["delta_capacity"] != recipient.delta_capacity
    ):
        raise BenchError("Qwen donor/recipient KV mount geometry or shared history changed")
    raw = bench.vm._delta_bytes(checkpoint, consumer="bench-successor-kv-interchange")
    decoded = bench.vm.runtime.codec.decode_segment(raw, bench.vm.runtime.device) if raw else ()
    if recipient.delta_rows and len(decoded) != bench.vm.runtime.codec.layers:
        raise BenchError("Qwen donor KV private closure is incomplete")
    return tuple(
        (recipient.ancestor[layer], decoded[layer] if decoded else recipient.delta[layer])
        for layer in range(bench.vm.runtime.codec.layers)
    )


def _stage(recipient: CausalContextQwenMount, donor_layers, selector):
    ancestor = tuple((key.clone(), value.clone()) for key, value in recipient.ancestor)
    delta = tuple((key.clone(), value.clone()) for key, value in recipient.delta)
    candidate = CausalContextQwenMount(
        recipient.mount_sha256,
        recipient.page_id,
        recipient.model_abi_sha256,
        recipient.node_sha256s,
        ancestor,
        delta,
        recipient.ancestor_rows,
        recipient.delta_rows,
        recipient.delta_capacity,
    )
    start, stop = selector["row_start"], selector["row_stop"]
    component_indices = {"key": 0, "value": 1}
    for layer in selector["layers"]:
        for component in selector["components"]:
            index = component_indices[component]
            donor = torch.cat(
                (
                    donor_layers[layer][0][index],
                    donor_layers[layer][1][index][:, :, : recipient.delta_rows],
                ),
                dim=2,
            )
            if start < recipient.ancestor_rows:
                ancestor_stop = min(stop, recipient.ancestor_rows)
                candidate.ancestor[layer][index][:, :, start:ancestor_stop].copy_(
                    donor[:, :, start:ancestor_stop]
                )
            if stop > recipient.ancestor_rows:
                delta_start = max(start, recipient.ancestor_rows) - recipient.ancestor_rows
                delta_stop = stop - recipient.ancestor_rows
                candidate.delta[layer][index][:, :, delta_start:delta_stop].copy_(
                    donor[:, :, max(start, recipient.ancestor_rows) : stop]
                )
    return candidate


def _locality(recipient, candidate, donor_layers, selector):
    selected = {
        (layer, {"key": 0, "value": 1}[component])
        for layer in selector["layers"]
        for component in selector["components"]
    }
    start, stop = selector["row_start"], selector["row_stop"]
    selected_exact = True
    masked_exact = True
    changed = 0
    for layer in range(len(recipient.ancestor)):
        for component in range(2):
            before = _visible(recipient, layer, component)
            after = _visible(candidate, layer, component)
            donor = torch.cat(
                (
                    donor_layers[layer][0][component],
                    donor_layers[layer][1][component][:, :, : recipient.delta_rows],
                ),
                dim=2,
            )
            mask = torch.zeros(before.shape, dtype=torch.bool, device=before.device)
            if (layer, component) in selected:
                mask[:, :, start:stop] = True
                selected_exact = selected_exact and torch.equal(after[mask], donor[mask])
            masked_exact = masked_exact and torch.equal(after[~mask], before[~mask])
            changed += int(torch.count_nonzero(after != before))
    if not selected_exact or not masked_exact:
        raise BenchError(
            "Qwen successor KV staging violated selected-write or masked-locality contract"
        )
    return {
        "selected_payload_exact": selected_exact,
        "masked_locality_exact": masked_exact,
        "changed_elements": changed,
    }


class SuccessorKVBackend(InMemoryModelBackend):
    def preview(self, parent, plan, updates):
        if set(updates) != {ADDRESS} or updates[ADDRESS] != self.reference:
            raise ModelOSExecutionError("successor KV payload changed after admission")
        candidate, result = super().preview(parent, plan, updates)
        return candidate, {
            **result,
            "geometry": self.geometry,
            "selector": self.selector,
            "locality": self.locality,
        }

    def continue_execution(self, candidate, *, consumer, horizon):
        super().continue_execution(candidate, consumer=consumer, horizon=horizon)
        if horizon != 0 or self.owner.execution is not None:
            raise ModelOSExecutionError("successor KV boundary moved during preview")
        return {
            "accepted": True,
            "consumer_id": consumer.consumer_id,
            "horizon": 0,
            "execution_mode": "native-successor-kv-install",
            "act_id": self.act_id,
            "donor_cut": self.donor_cut,
            "recipient_policy": PENDING_TOKEN_POLICY,
            "geometry": self.geometry,
            "selector": self.selector,
            "locality": self.locality,
        }

    def commit(self, candidate):
        receipt = super().commit(candidate)
        parent = self.owner.current_statecut
        committed = self.owner._new_cut(
            self.candidate_mount, parent=parent.handle, status="committed", token=None
        )
        self.owner._current = committed.handle
        return {
            **receipt,
            "execution_mode": "native-successor-kv-install",
            "native_statecut": committed.to_payload(),
            "locality": self.locality,
        }


def transaction(bench, *, parent_cut: str, donor_cut: str, donor: dict[str, Any], selector):
    owner = bench._backend()
    recipient = owner.current_mount
    geometry = _geometry(bench, recipient)
    selector = normalize_selector(selector, layers=geometry["layers"], rows=geometry["rows"])
    donor_layers = _donor_layers(bench, donor, recipient)
    candidate = _stage(recipient, donor_layers, selector)
    locality = _locality(recipient, candidate, donor_layers, selector)
    selected_summary = {
        f"layer_{layer:03d}.{component}": tensor_digest(
            _visible(candidate, layer, {"key": 0, "value": 1}[component])[
                :, :, selector["row_start"] : selector["row_stop"]
            ]
        )
        for layer in selector["layers"]
        for component in selector["components"]
    }
    history = {
        "length": len(bench.branch.token_ids),
        "token_ids_sha256": content_digest(list(bench.branch.token_ids)),
    }
    pending = {
        "recipient_token_id": bench.branch.pending,
        "donor_token_id": donor["pending"],
        "policy": PENDING_TOKEN_POLICY,
    }
    act_id = content_digest(
        {
            "parent_cut": parent_cut,
            "donor_cut": donor_cut,
            "model_abi": bench.model_abi,
            "native_abi": bench.vm.package.model_abi.fingerprint,
            "history": bench.branch.token_ids,
            "pending": pending,
            "geometry": geometry,
            "selector": selector,
            "pending_token_policy": PENDING_TOKEN_POLICY,
            "selected_payload": selected_summary,
        }
    )
    schema = StateSchema(
        name="saturn-bench-qwen-successor-kv-v1",
        version=1,
        slots=(
            StateSlot(
                address=ADDRESS,
                value_type="typed-kv-selection",
                owner="bench-qwen",
                dtype=geometry["dtype"],
                shape=(
                    geometry["batch"],
                    geometry["layers"],
                    2,
                    geometry["kv_heads"],
                    geometry["rows"],
                    geometry["head_dim"],
                ),
                read_scopes=("model-os",),
                write_scopes=("model-os",),
                lifetime="branch",
                branchable=True,
                persistable=True,
                freshness="candidate",
                lineage=donor_cut,
                metadata={
                    "geometry": geometry,
                    "selector": selector,
                    "pending_token_policy": PENDING_TOKEN_POLICY,
                },
            ),
        ),
    )
    abi = replace(bench.vm.package.model_abi, state_schema=schema.fingerprint)
    consumer = ConsumerContract(
        consumer_id="bench.qwen.successor-kv-install",
        reads=(ADDRESS,),
        output_kind="successor-kv-install-receipt",
        continuation_tokens=0,
        native=True,
    )
    program = ModelProgram(
        program_id="bench.qwen.successor-kv-install.v1",
        version=1,
        instructions=(
            ProgramInstruction(
                instruction_id="install",
                op="Write",
                reads=(ADDRESS,),
                writes=(ADDRESS,),
                outputs=(ADDRESS,),
                authority="model-os",
            ),
        ),
        portable=False,
        terminal_claim=False,
    )
    package = ModelPackage(
        package_id="bench.qwen.successor-kv-install",
        version=1,
        model_abi=abi,
        state_schema=schema,
        state_surface=StateSurface.from_schema(
            schema, surface_id="bench-successor-kv", owner="bench-qwen"
        ),
        program=program,
        consumers=(consumer,),
        permissions={
            "subjects": {"model-os": {"read": [ADDRESS], "write": [ADDRESS]}},
            "authorities": ["model-os"],
        },
    )
    reference = {
        "act_id": act_id,
        "donor_cut": donor_cut,
        "geometry": geometry,
        "selector": selector,
        "selected_payload": selected_summary,
        "history": history,
        "pending": pending,
        "pending_token_policy": PENDING_TOKEN_POLICY,
    }
    backend = SuccessorKVBackend(model_abi=abi, state_schema=schema)
    backend.owner = owner
    backend.candidate_mount = candidate
    backend.reference = reference
    backend.geometry, backend.selector, backend.locality = geometry, selector, locality
    backend.act_id, backend.donor_cut = act_id, donor_cut
    return (
        ModelOSKernel.install(package, backend),
        {ADDRESS: reference},
        {
            "act_id": act_id,
            "geometry": geometry,
            "selector": selector,
            "locality": locality,
            "pending_token_policy": PENDING_TOKEN_POLICY,
            "history": history,
            "pending": pending,
            "selected_payload": selected_summary,
        },
    )


__all__ = ["PENDING_TOKEN_POLICY", "normalize_selector", "transaction"]
