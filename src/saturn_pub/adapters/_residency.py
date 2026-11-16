"""Adapter-owned block-streamed weight residency for models larger than device memory.

Two modes share one call site:

``resident``
    Frozen weights already live on the execution device; native modules are called
    directly with zero overhead and unchanged numerics.

``streamed``
    Frozen weights stay in host memory (optionally pinned). For each native module the
    adapter copies exactly that module's parameters and buffers to the device, runs the
    module functionally on the device, and lets the transient device copies fall out of
    scope. The host weights themselves never move, so the adapter's frozen-model guard
    (parameter ``id``/``_version``/``data_ptr``) still holds after every block.

Copying a parameter with ``Tensor.to(device)`` is a byte-preserving move (no dtype or
layout change), and ``torch.func.functional_call`` substitutes the supplied tensors
without altering the module, so a streamed block computes the same bits a resident block
would on the same device. That equality is proved by tests and measured on real weights.

Two optional transport refinements change only *when* and *which* bytes cross the bus,
never the arithmetic:

``prefetch`` (requires ``pin_host``)
    Learns each module's successor from the call order (a decode visits embed, rotary,
    layers, norm, head in the same order every token) and copies the predicted next
    module's weights on a side CUDA stream while the current module computes. The compute
    stream waits on that copy's event before use. A wrong prediction only wastes a copy.

``host_embedding``
    An ``nn.Embedding`` lookup is a row gather. Streamed residency would otherwise copy the
    whole table (1.24 GB for Qwen3-8B) to read a handful of rows; this gathers the rows in
    host memory and copies only them. A gather moves bytes and does no arithmetic, so the
    output bytes are those of the device lookup.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.func import functional_call


class BlockResidency:
    """Place frozen weights for a declared residency and run native modules under it."""

    def __init__(
        self,
        device: Any,
        *,
        mode: str = "resident",
        pin_host: bool = False,
        prefetch: bool = False,
        host_embedding: bool = False,
    ):
        if mode not in {"resident", "streamed"}:
            raise ValueError("residency mode must be 'resident' or 'streamed'")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            # A bare "cuda" has index None, but tensors moved to it become "cuda:N";
            # resolve the concrete index so state device checks compare equal.
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.mode = mode
        self.pin_host = bool(pin_host)
        self.prefetch = bool(prefetch)
        self.host_embedding = bool(host_embedding)
        if pin_host and mode != "streamed":
            raise ValueError("host pinning only applies to streamed residency")
        if pin_host and self.device.type != "cuda":
            raise ValueError("host pinning is only useful for a CUDA execution device")
        if (prefetch or host_embedding) and mode != "streamed":
            raise ValueError("prefetch and host embedding only apply to streamed residency")
        if prefetch and not pin_host:
            raise ValueError("prefetch needs pin_host=True: pageable copies cannot overlap")
        self._successor: dict[int, Any] = {}
        self._previous: Any = None
        self._staged: dict[int, tuple[dict[str, torch.Tensor], Any]] = {}
        self._copy_stream = torch.cuda.Stream(device=self.device) if self.prefetch else None
        self.stats = {"prefetch_hits": 0, "prefetch_misses": 0, "prefetch_wasted": 0,
                      "host_embedding_calls": 0}

    def place(self, *modules: Any) -> BlockResidency:
        """Position frozen weights for this residency; call before freezing identity.

        ``resident`` moves weights onto the execution device (matching direct calls).
        ``streamed`` parks weights in host memory and optionally pins them so later
        device copies are asynchronous. Pinning reassigns storage, so this must run
        before the adapter records its frozen-parameter guard.
        """
        for module in modules:
            if module is None:
                continue
            if self.mode == "resident":
                module.to(self.device)
            else:
                module.to("cpu")
                if self.pin_host:
                    for parameter in module.parameters(recurse=True):
                        if not parameter.data.is_pinned():
                            parameter.data = parameter.data.pin_memory()
        return self

    def _copy(self, module: Any) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for name, parameter in module.named_parameters():
            tensors[name] = parameter.to(self.device, non_blocking=self.pin_host)
        for name, buffer in module.named_buffers():
            tensors[name] = buffer.to(self.device, non_blocking=self.pin_host)
        return tensors

    def _stage(self, module: Any) -> None:
        """Start copying ``module`` on the side stream (it waits for nothing on compute)."""
        key = id(module)
        if key in self._staged or self._copy_stream is None:
            return
        with torch.cuda.stream(self._copy_stream):
            tensors = self._copy(module)
            event = torch.cuda.Event()
            event.record(self._copy_stream)
        self._staged[key] = (tensors, event)

    def _fetch(self, module: Any) -> dict[str, torch.Tensor]:
        if self._copy_stream is None:
            return self._copy(module)
        # Learn the successor edge, then serve this module from its staged copy if present.
        if self._previous is not None:
            self._successor[id(self._previous)] = module
        self._previous = module
        staged = self._staged.pop(id(module), None)
        for key in list(self._staged):  # a mispredicted copy is dropped, never reused
            self._staged.pop(key)
            self.stats["prefetch_wasted"] += 1
        compute = torch.cuda.current_stream(self.device)
        if staged is None:
            self.stats["prefetch_misses"] += 1
            self._stage(module)
            staged = self._staged.pop(id(module))
        else:
            self.stats["prefetch_hits"] += 1
        tensors, event = staged
        compute.wait_event(event)
        for tensor in tensors.values():
            # Memory allocated on the copy stream is used on the compute stream: keep the
            # caching allocator from handing it out again before compute is done with it.
            tensor.record_stream(compute)
        nxt = self._successor.get(id(module))
        if nxt is not None:
            self._stage(nxt)
        return tensors

    def run(self, module: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute ``module`` under the declared residency, returning native output."""
        if self.mode == "resident":
            return module(*args, **kwargs)
        if (
            self.host_embedding
            and isinstance(module, torch.nn.Embedding)
            and module.max_norm is None
            and not kwargs
            and len(args) == 1
        ):
            self.stats["host_embedding_calls"] += 1
            rows = module(args[0].to("cpu"))
            return rows.to(self.device, non_blocking=False)
        return functional_call(module, self._fetch(module), args, kwargs)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "mode": self.mode,
            "device": {"type": self.device.type, "index": self.device.index},
            "host_pinned": self.pin_host,
        }
        # Transport-only options are recorded only when on, so receipts written before they
        # existed stay byte-identical.
        if self.prefetch:
            out["prefetch"] = True
        if self.host_embedding:
            out["host_embedding"] = True
        return out
