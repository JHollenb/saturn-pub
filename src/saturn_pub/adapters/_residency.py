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
"""

from __future__ import annotations

from typing import Any

import torch
from torch.func import functional_call


class BlockResidency:
    """Place frozen weights for a declared residency and run native modules under it."""

    def __init__(self, device: Any, *, mode: str = "resident", pin_host: bool = False):
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
        if pin_host and mode != "streamed":
            raise ValueError("host pinning only applies to streamed residency")
        if pin_host and self.device.type != "cuda":
            raise ValueError("host pinning is only useful for a CUDA execution device")

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
                        parameter.data = parameter.data.pin_memory()
        return self

    def run(self, module: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute ``module`` under the declared residency, returning native output."""
        if self.mode == "resident":
            return module(*args, **kwargs)
        tensors: dict[str, torch.Tensor] = {}
        for name, parameter in module.named_parameters():
            tensors[name] = parameter.to(self.device, non_blocking=self.pin_host)
        for name, buffer in module.named_buffers():
            tensors[name] = buffer.to(self.device, non_blocking=self.pin_host)
        return functional_call(module, tensors, args, kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "device": {"type": self.device.type, "index": self.device.index},
            "host_pinned": self.pin_host,
        }
