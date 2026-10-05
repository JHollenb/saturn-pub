"""Immutable paged, copy-on-write KV storage for AR branching.

``PagedKVLayer`` is the storage ABI for the page-native attention primitive in
``saturn.ar.paged_attention``.  Forks and layer/token cuts share untouched
pages, and only a partial tail is copied when a branch appends.  Materializing
the layer remains available as an explicit compatibility/reference operation;
the page-native attention path does not require it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mdb.qwen2_machine import KVLayerState, NormalizedKVCache, Qwen2MachineError

PAGED_KV_SCHEMA = "saturn-ar-paged-kv-v1"


class PagedKVError(ValueError):
    """Raised when a paged cache violates its storage ABI."""


def _validate_block_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PagedKVError("block_size must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class KVPage:
    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.key, torch.Tensor) or not isinstance(self.value, torch.Tensor):
            raise Qwen2MachineError("KV pages require tensor key/value payloads")
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise Qwen2MachineError("KV pages require rank-four tensors")
        if self.key.shape != self.value.shape:
            raise Qwen2MachineError("KV page key/value shapes must match")
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise Qwen2MachineError("KV page key/value device and dtype must match")

    @property
    def length(self) -> int:
        return int(self.key.shape[-2])

    @property
    def sequence_length(self) -> int:
        return self.length


@dataclass(frozen=True, slots=True)
class PagedKVLayer:
    pages: tuple[KVPage, ...]
    length: int
    block_size: int

    def __post_init__(self) -> None:
        block_size = _validate_block_size(self.block_size)
        object.__setattr__(self, "pages", tuple(self.pages))
        object.__setattr__(self, "block_size", block_size)
        if self.length < 0:
            raise PagedKVError("paged KV length cannot be negative")
        if sum(page.length for page in self.pages) != self.length:
            raise PagedKVError("paged KV page lengths do not add up to layer length")
        for index, page in enumerate(self.pages):
            if page.length < 1 or page.length > block_size:
                raise PagedKVError("KV page length is outside the configured block size")
            if index < len(self.pages) - 1 and page.length != block_size:
                raise PagedKVError("only the final KV page may be partial")

    @classmethod
    def from_tensors(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        block_size: int = 128,
    ) -> PagedKVLayer:
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise Qwen2MachineError("paged KV tensors must be matching rank-four tensors")
        block_size = _validate_block_size(block_size)
        pages = tuple(
            KVPage(
                key[..., start : start + block_size, :],
                value[..., start : start + block_size, :],
            )
            for start in range(0, int(key.shape[-2]), block_size)
        )
        return cls(pages=pages, length=int(key.shape[-2]), block_size=block_size)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> PagedKVLayer:
        """Return a branch with the append; untouched full pages are shared."""

        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise PagedKVError("paged KV append must be matching rank-four tensors")
        if key.shape[:2] != self.shape[:2] or key.shape[-1] != self.shape[-1]:
            raise PagedKVError("paged KV append shape prefix does not match the layer")
        if key.device != self.device or value.device != self.device:
            raise PagedKVError("paged KV append device does not match the layer")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise PagedKVError("paged KV append dtype does not match the layer")
        if int(key.shape[-2]) == 0:
            return self

        pages = list(self.pages)
        cursor = 0
        if pages and pages[-1].length < self.block_size:
            tail = pages.pop()
            take = min(self.block_size - tail.length, int(key.shape[-2]))
            pages.append(
                KVPage(
                    torch.cat((tail.key, key[..., :take, :]), dim=-2),
                    torch.cat((tail.value, value[..., :take, :]), dim=-2),
                )
            )
            cursor = take
        while cursor < int(key.shape[-2]):
            page_key = key[..., cursor : cursor + self.block_size, :]
            page_value = value[..., cursor : cursor + self.block_size, :]
            pages.append(KVPage(page_key, page_value))
            cursor += int(page_key.shape[-2])
        return PagedKVLayer(
            pages=tuple(pages),
            length=self.length + int(key.shape[-2]),
            block_size=self.block_size,
        )

    def replace_range(
        self,
        start: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> PagedKVLayer:
        """Return a branch with one token range replaced copy-on-write.

        Only pages intersecting ``[start, start + replacement_length)`` are
        rebuilt.  Every other page keeps object identity with the parent.
        This is the storage primitive needed by StateCut/KV surgery: a branch
        that changes one query slot must not duplicate an otherwise immutable
        prefix.
        """

        if isinstance(start, bool) or not isinstance(start, int):
            raise PagedKVError("paged KV replacement start must be an integer")
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise PagedKVError("paged KV replacement must be matching rank-four tensors")
        if not self.pages:
            raise PagedKVError("cannot replace a range in an empty paged KV layer")
        if key.shape[:2] != self.shape[:2] or key.shape[-1] != self.shape[-1]:
            raise PagedKVError("paged KV replacement shape prefix does not match the layer")
        if key.device != self.device or value.device != self.device:
            raise PagedKVError("paged KV replacement device does not match the layer")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise PagedKVError("paged KV replacement dtype does not match the layer")
        replacement_length = int(key.shape[-2])
        stop = start + replacement_length
        if replacement_length < 1:
            raise PagedKVError("paged KV replacement cannot be empty")
        if start < 0 or stop > self.length:
            raise PagedKVError("paged KV replacement range is out of bounds")

        pages: list[KVPage] = []
        page_start = 0
        for page in self.pages:
            page_stop = page_start + page.length
            overlap_start = max(start, page_start)
            overlap_stop = min(stop, page_stop)
            if overlap_start >= overlap_stop:
                pages.append(page)
                page_start = page_stop
                continue

            local_start = overlap_start - page_start
            local_stop = overlap_stop - page_start
            replacement_start = overlap_start - start
            replacement_stop = overlap_stop - start
            key_parts = (
                page.key[..., :local_start, :],
                key[..., replacement_start:replacement_stop, :],
                page.key[..., local_stop:, :],
            )
            value_parts = (
                page.value[..., :local_start, :],
                value[..., replacement_start:replacement_stop, :],
                page.value[..., local_stop:, :],
            )
            pages.append(
                KVPage(
                    torch.cat(key_parts, dim=-2),
                    torch.cat(value_parts, dim=-2),
                )
            )
            page_start = page_stop
        return PagedKVLayer(
            pages=tuple(pages),
            length=self.length,
            block_size=self.block_size,
        )

    def replace_value_range(self, start: int, value: torch.Tensor) -> PagedKVLayer:
        """Replace VALUE slots while retaining every original KEY tensor.

        Query-slot physiology shows that some branch families change values
        only.  Rebuilding their keys would be pure transport.  The touched
        page object is new, but its key tensor is the parent's exact object;
        only intersecting value pages allocate storage.
        """

        if isinstance(start, bool) or not isinstance(start, int):
            raise PagedKVError("paged KV value replacement start must be an integer")
        if not isinstance(value, torch.Tensor) or value.ndim != 4:
            raise PagedKVError("paged KV value replacement must be rank four")
        if not self.pages:
            raise PagedKVError("cannot replace values in an empty paged KV layer")
        if value.shape[:2] != self.shape[:2] or value.shape[-1] != self.shape[-1]:
            raise PagedKVError("paged KV value replacement shape prefix does not match the layer")
        if value.device != self.device:
            raise PagedKVError("paged KV value replacement device does not match the layer")
        if value.dtype != self.dtype:
            raise PagedKVError("paged KV value replacement dtype does not match the layer")
        replacement_length = int(value.shape[-2])
        stop = start + replacement_length
        if replacement_length < 1:
            raise PagedKVError("paged KV value replacement cannot be empty")
        if start < 0 or stop > self.length:
            raise PagedKVError("paged KV value replacement range is out of bounds")

        pages: list[KVPage] = []
        page_start = 0
        for page in self.pages:
            page_stop = page_start + page.length
            overlap_start = max(start, page_start)
            overlap_stop = min(stop, page_stop)
            if overlap_start >= overlap_stop:
                pages.append(page)
                page_start = page_stop
                continue
            local_start = overlap_start - page_start
            local_stop = overlap_stop - page_start
            replacement_start = overlap_start - start
            replacement_stop = overlap_stop - start
            pages.append(
                KVPage(
                    page.key,
                    torch.cat(
                        (
                            page.value[..., :local_start, :],
                            value[..., replacement_start:replacement_stop, :],
                            page.value[..., local_stop:, :],
                        ),
                        dim=-2,
                    ),
                )
            )
            page_start = page_stop
        return PagedKVLayer(
            pages=tuple(pages),
            length=self.length,
            block_size=self.block_size,
        )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        if not self.pages:
            raise PagedKVError("empty paged KV layer has no tensor shape")
        first = self.pages[0].key
        return (int(first.shape[0]), int(first.shape[1]), self.length, int(first.shape[3]))

    @property
    def device(self) -> torch.device:
        return self.pages[0].key.device

    @property
    def dtype(self) -> torch.dtype:
        return self.pages[0].key.dtype

    def materialize(self) -> KVLayerState:
        key, value = self.materialize_tensors()
        return KVLayerState(key, value)

    def materialize_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contiguous compatibility tensors without assigning identity.

        The ledger assigns producer/incremental identity after hashing the
        append once.  A one-page layer is already materialized, so avoid two
        redundant ``cat`` launches in the short-prefix case.
        """

        if not self.pages:
            raise PagedKVError("cannot materialize an empty paged KV layer")
        if len(self.pages) == 1:
            page = self.pages[0]
            return page.key, page.value
        return (
            torch.cat(tuple(page.key for page in self.pages), dim=-2),
            torch.cat(tuple(page.value for page in self.pages), dim=-2),
        )

    def iter_pages(self) -> tuple[KVPage, ...]:
        """Return the immutable page sequence for a page-native consumer."""

        return self.pages

    def shared_page_count(self, other: PagedKVLayer) -> int:
        return sum(left is right for left, right in zip(self.pages, other.pages, strict=False))

    @property
    def sequence_length(self) -> int:
        return self.length


@dataclass(frozen=True, slots=True)
class PagedKVCache:
    """Immutable cache table whose layers are persistent page trees."""

    layers: tuple[PagedKVLayer | None, ...]
    block_size: int = 128
    schema: str = PAGED_KV_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PAGED_KV_SCHEMA:
            raise Qwen2MachineError(f"unsupported paged KV schema {self.schema!r}")
        _validate_block_size(self.block_size)
        object.__setattr__(self, "layers", tuple(self.layers))

    @classmethod
    def empty(cls, layer_count: int, *, block_size: int = 128) -> PagedKVCache:
        if layer_count < 1:
            raise ValueError("layer_count must be positive")
        return cls(tuple(None for _ in range(layer_count)), block_size=block_size)

    @classmethod
    def from_normalized(
        cls,
        cache: NormalizedKVCache,
        *,
        block_size: int = 128,
    ) -> PagedKVCache:
        return cls(
            tuple(
                None
                if layer is None
                else PagedKVLayer.from_tensors(
                    layer.key,
                    layer.value,
                    block_size=block_size,
                )
                for layer in cache.layers
            ),
            block_size=block_size,
        )

    def append_layer(
        self,
        index: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> PagedKVCache:
        if not 0 <= index < len(self.layers):
            raise Qwen2MachineError(f"paged KV layer {index} is out of range")
        current = self.layers[index]
        if current is None:
            updated = PagedKVLayer.from_tensors(key, value, block_size=self.block_size)
        else:
            updated = current.append(key, value)
        layers = list(self.layers)
        layers[index] = updated
        return PagedKVCache(tuple(layers), block_size=self.block_size)

    def append(
        self,
        index: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> PagedKVCache:
        return self.append_layer(index, key, value)

    def replace_layer(self, index: int, layer: PagedKVLayer) -> PagedKVCache:
        """Return a cache with one already-built persistent layer installed.

        Native page consumers build a layer once per attention call and hand
        it back to the owner.  Keeping replacement on the cache makes that
        ownership transition explicit without exposing a mutable page table.
        """

        if not 0 <= index < len(self.layers):
            raise Qwen2MachineError(f"paged KV layer {index} is out of range")
        if not isinstance(layer, PagedKVLayer):
            raise PagedKVError("replacement must be a PagedKVLayer")
        if layer.block_size != self.block_size:
            raise PagedKVError("replacement page size does not match the cache")
        layers = list(self.layers)
        layers[index] = layer
        return PagedKVCache(tuple(layers), block_size=self.block_size)

    def replace_range(
        self,
        index: int,
        start: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> PagedKVCache:
        """Replace one layer range while sharing every untouched page/layer."""

        if not 0 <= index < len(self.layers):
            raise Qwen2MachineError(f"paged KV layer {index} is out of range")
        current = self.layers[index]
        if current is None:
            raise PagedKVError("cannot replace a range in an uninitialized KV layer")
        return self.replace_layer(index, current.replace_range(start, key, value))

    def replace_value_range(
        self,
        index: int,
        start: int,
        value: torch.Tensor,
    ) -> PagedKVCache:
        """Replace VALUE slots while sharing all keys and untouched pages."""

        if not 0 <= index < len(self.layers):
            raise Qwen2MachineError(f"paged KV layer {index} is out of range")
        current = self.layers[index]
        if current is None:
            raise PagedKVError("cannot replace values in an uninitialized KV layer")
        return self.replace_layer(index, current.replace_value_range(start, value))

    def to_normalized(self) -> NormalizedKVCache:
        return NormalizedKVCache(
            tuple(None if layer is None else layer.materialize() for layer in self.layers)
        )

    def materialize(self) -> NormalizedKVCache:
        return self.to_normalized()

    def to_normalized_cache(self) -> NormalizedKVCache:
        return self.to_normalized()

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def common_sequence_length(self, *, complete: bool = True) -> int:
        if complete and any(layer is None for layer in self.layers):
            raise PagedKVError("paged KV cache has uninitialized layers")
        lengths = {layer.sequence_length for layer in self.layers if layer is not None}
        if len(lengths) > 1:
            raise PagedKVError("paged KV layers do not share one sequence length")
        return next(iter(lengths), 0)

    def fork(self) -> PagedKVCache:
        """Fork by reference; the next append performs the required COW."""

        return PagedKVCache(self.layers, block_size=self.block_size)

    @property
    def page_count(self) -> int:
        return sum(0 if layer is None else len(layer.pages) for layer in self.layers)

    def shared_page_count(self, other: PagedKVCache) -> int:
        return sum(
            0
            if left is None or right is None
            else left.shared_page_count(right)
            for left, right in zip(self.layers, other.layers, strict=False)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "block_size": self.block_size,
            "layer_count": len(self.layers),
            "page_count": self.page_count,
            "lengths": [None if layer is None else layer.length for layer in self.layers],
            "storage": "persistent-copy-on-write-pages",
            "attention_boundary": "page-native-or-reference-materialize",
        }


__all__ = [
    "KVPage",
    "PAGED_KV_SCHEMA",
    "PagedKVCache",
    "PagedKVError",
    "PagedKVLayer",
]
