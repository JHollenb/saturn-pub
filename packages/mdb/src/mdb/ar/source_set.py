"""Permutation-invariant paired source records for cached AR decoders.

An :class:`ArSourceSet` views the pre-query prefix cache as vertical records:
one prompt position bound to its post-position-encoding key and value at every
decoder layer.  Physical row order is a storage chart, not source identity.

Admission masks keep the original cache geometry and absolute query position.
This is essential for causal deletion: physically shortening a cache would
confound source removal with a positional-clock change.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mdb.tensor_hash import tensor_bytes_sha256

if TYPE_CHECKING:
    from mdb.ar.prefix_program import ArPrefixProgram

AR_SOURCE_SET_SCHEMA = "saturn-ar-source-set-v1"


class ArSourceSetError(ValueError):
    """A source set violates its paired-record or admission contract."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ArSourceSet:
    """A prompt-static set of vertically bound K/V source records.

    ``layers`` use the DynamicCache geometry ``[batch, heads, source, width]``
    and remain in their native physical order. ``admitted`` controls which
    source records the live query may read without changing source length.
    """

    model_id: str
    revision: str
    source_token_ids: tuple[int, ...]
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    admitted: tuple[bool, ...]
    cache_abi: str = "transformers-dynamic-cache-post-position-encoding-kv-v1"

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision or not self.layers:
            raise ArSourceSetError("source-set identity and layers must be non-empty")
        count = len(self.source_token_ids)
        if count < 1 or len(self.admitted) != count:
            raise ArSourceSetError(
                "source tokens and admission mask must have equal nonzero length"
            )
        if any(token < 0 for token in self.source_token_ids):
            raise ArSourceSetError("source token IDs must be non-negative")
        for index, (key, value) in enumerate(self.layers):
            if key.ndim != 4 or key.shape != value.shape:
                raise ArSourceSetError(f"layer {index} has unsupported paired K/V geometry")
            if int(key.shape[0]) != 1 or int(key.shape[-2]) != count:
                raise ArSourceSetError(f"layer {index} does not match source-record count")
            if key.device.type != "cpu" or value.device.type != "cpu":
                raise ArSourceSetError("durable source records must be CPU resident")

    @classmethod
    def from_program(cls, program: ArPrefixProgram) -> ArSourceSet:
        """Create the fully admitted pre-query source set of a prefix program."""

        source_tokens = tuple(program.source_token_ids[:-1])
        return cls(
            model_id=program.model_id,
            revision=program.revision,
            source_token_ids=source_tokens,
            layers=program.query_cache_layers,
            admitted=(True,) * len(source_tokens),
            cache_abi=program.cache_abi,
        )

    @property
    def record_count(self) -> int:
        return len(self.source_token_ids)

    @property
    def admitted_indices(self) -> tuple[int, ...]:
        return tuple(index for index, keep in enumerate(self.admitted) if keep)

    @property
    def admitted_count(self) -> int:
        return sum(self.admitted)

    @property
    def total_tensor_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for pair in self.layers
            for tensor in pair
        )

    @property
    def admitted_tensor_bytes(self) -> int:
        total = 0
        for key, value in self.layers:
            per_record = (
                key[:, :, :1, :].numel() * key.element_size()
                + value[:, :, :1, :].numel() * value.element_size()
            )
            total += self.admitted_count * per_record
        return total

    def with_admitted(self, indices: Iterable[int]) -> ArSourceSet:
        selected = frozenset(int(index) for index in indices)
        if any(index < 0 or index >= self.record_count for index in selected):
            raise ArSourceSetError("admitted source index is out of range")
        return ArSourceSet(
            model_id=self.model_id,
            revision=self.revision,
            source_token_ids=self.source_token_ids,
            layers=self.layers,
            admitted=tuple(index in selected for index in range(self.record_count)),
            cache_abi=self.cache_abi,
        )

    def permuted(self, order: Sequence[int]) -> ArSourceSet:
        """Jointly permute tokens, K, V, and admission while preserving pairing."""

        normalized = tuple(int(index) for index in order)
        if sorted(normalized) != list(range(self.record_count)):
            raise ArSourceSetError("source permutation must contain every record exactly once")
        index = torch.tensor(normalized, dtype=torch.long)
        return ArSourceSet(
            model_id=self.model_id,
            revision=self.revision,
            source_token_ids=tuple(self.source_token_ids[item] for item in normalized),
            layers=tuple(
                (
                    key.index_select(-2, index).contiguous(),
                    value.index_select(-2, index).contiguous(),
                )
                for key, value in self.layers
            ),
            admitted=tuple(self.admitted[item] for item in normalized),
            cache_abi=self.cache_abi,
        )

    def record_fingerprints(self) -> tuple[str, ...]:
        """Hash each vertical prompt record with its K/V binding intact."""

        rows = []
        for position in range(self.record_count):
            rows.append(
                _digest(
                    {
                        "token_id": self.source_token_ids[position],
                        "admitted": self.admitted[position],
                        "layers": [
                            {
                                "layer": layer,
                                "key": tensor_bytes_sha256(key[:, :, position : position + 1, :]),
                                "value": tensor_bytes_sha256(
                                    value[:, :, position : position + 1, :]
                                ),
                            }
                            for layer, (key, value) in enumerate(self.layers)
                        ],
                    }
                )
            )
        return tuple(rows)

    @property
    def fingerprint(self) -> str:
        """Return identity invariant to a coherent permutation of source records."""

        return _digest(
            {
                "schema": AR_SOURCE_SET_SCHEMA,
                "model_id": self.model_id,
                "revision": self.revision,
                "cache_abi": self.cache_abi,
                "records": sorted(self.record_fingerprints()),
            }
        )

    def attention_mask(
        self,
        *,
        dynamic_admitted: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Keep source geometry fixed and append admitted live/pending records."""

        if dynamic_admitted < 1:
            raise ArSourceSetError("a query step must admit at least its pending/live token")
        values = [int(value) for value in self.admitted] + [1] * int(dynamic_admitted)
        return torch.tensor([values], dtype=torch.long, device=device)

    def manifest(self) -> dict[str, Any]:
        body = {
            "schema": AR_SOURCE_SET_SCHEMA,
            "model_id": self.model_id,
            "revision": self.revision,
            "cache_abi": self.cache_abi,
            "source_token_ids": list(self.source_token_ids),
            "record_count": self.record_count,
            "admitted": list(self.admitted),
            "admitted_indices": list(self.admitted_indices),
            "admitted_count": self.admitted_count,
            "total_tensor_bytes": self.total_tensor_bytes,
            "admitted_tensor_bytes": self.admitted_tensor_bytes,
            "record_fingerprints": list(self.record_fingerprints()),
            "permutation_invariant_fingerprint": self.fingerprint,
            "causal_contract": {
                "record": "one prompt position's paired K/V across every decoder layer",
                "deletion": "attention-mask non-admission with physical cache length preserved",
                "position": "pending and generated token positions remain on the native clock",
            },
        }
        return body
