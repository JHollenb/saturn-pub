"""Explicit model selection for Saturn's dense autoregressive lane.

The numerical machine is intentionally independent of a checkpoint name, but
launchers still need one auditable way to turn a user-facing model key into a
registry entry, model identity, and architecture capability.  This module is
that boundary.  It uses mrun's canonical model registry lazily so importing
the AR core does not acquire or load model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mdb.qwen2_machine import MODEL_ID as PINNED_MODEL_ID
from mdb.qwen2_machine import MODEL_REVISION as PINNED_MODEL_REVISION
from mdb.qwen2_machine import Qwen2MachineError

DEFAULT_AR_MODEL = "qwen2.5-0.5b"
LOCAL_MATERIALIZED_REVISION = "local-materialized"
SUPPORTED_AR_MODEL_TYPES = frozenset(("qwen2", "qwen3"))


class ArModelSelectionError(Qwen2MachineError):
    """Raised when a requested checkpoint is outside the dense AR ABI."""


@dataclass(frozen=True, slots=True)
class ArModelSpec:
    """Resolved model identity and loader key for one AR worker."""

    name: str
    registry_name: str
    model_id: str
    revision: str
    family: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.name,
                self.registry_name,
                self.model_id,
                self.revision,
                self.family,
            )
        ):
            raise ArModelSelectionError("AR model identity fields must be non-empty strings")
        if self.family not in SUPPORTED_AR_MODEL_TYPES:
            supported = ", ".join(sorted(SUPPORTED_AR_MODEL_TYPES))
            raise ArModelSelectionError(
                f"model {self.name!r} has family {self.family!r}; dense AR supports {supported}"
            )

    @property
    def supports_v1(self) -> bool:
        """Whether the historical pinned Qwen2 comparison arm applies."""

        return self.family == "qwen2"

    @property
    def identity(self) -> str:
        return f"{self.model_id}@{self.revision}"

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "registry_name": self.registry_name,
            "model_id": self.model_id,
            "revision": self.revision,
            "family": self.family,
            "supports_v1": self.supports_v1,
        }


def resolve_ar_model(
    model: str | None = None,
    *,
    revision: str | None = None,
) -> ArModelSpec:
    """Resolve a mrun registry key into a fail-closed dense AR model spec.

    ``qwen2.5-0.5b`` keeps its immutable Hub revision.  Other materialized
    fleet checkpoints receive an explicit local provenance label unless the
    caller supplies a revision.  Their config/tokenizer fingerprints still
    enter every Saturn prefix identity, so the revision label cannot make two
    different resident models interchangeable.
    """

    requested = str(model or DEFAULT_AR_MODEL)
    try:
        from mrun.models import resolve_model

        resolved = resolve_model(requested)
    except (ImportError, ValueError) as exc:
        raise ArModelSelectionError(f"cannot resolve AR model {requested!r}: {exc}") from exc

    family = str(getattr(resolved, "family", ""))
    if family not in SUPPORTED_AR_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_AR_MODEL_TYPES))
        raise ArModelSelectionError(
            f"model {requested!r} resolves to family {family!r}; dense AR supports {supported}"
        )

    registry_name = str(getattr(resolved, "name", requested))
    model_id = str(getattr(resolved, "hf_id", ""))
    if not model_id:
        raise ArModelSelectionError(f"model {requested!r} has no Hugging Face model ID")
    selected_revision = revision
    if selected_revision is None:
        selected_revision = (
            PINNED_MODEL_REVISION
            if model_id == PINNED_MODEL_ID and registry_name == DEFAULT_AR_MODEL
            else LOCAL_MATERIALIZED_REVISION
        )
    if not isinstance(selected_revision, str) or not selected_revision:
        raise ArModelSelectionError("AR model revision must be a non-empty string")
    return ArModelSpec(
        name=registry_name,
        registry_name=registry_name,
        model_id=model_id,
        revision=selected_revision,
        family=family,
    )


def available_ar_models() -> tuple[str, ...]:
    """Return registered dense Qwen keys that can be requested by AR workers."""

    from mrun.models import REGISTRY

    return tuple(
        sorted(
            name
            for name, spec in REGISTRY.items()
            if str(getattr(spec, "family", "")) in SUPPORTED_AR_MODEL_TYPES
        )
    )


def load_ar_model(
    model: str | None = None,
    *,
    revision: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    attn_implementation: str = "sdpa",
    local_files_only: bool = True,
) -> tuple[ArModelSpec, Any, Any]:
    """Load one selected model through mrun and verify its realized family.

    Returns ``(spec, model, tokenizer)``.  The revision is identity metadata;
    mrun's local materialized directory remains the byte source for offline
    fleet workers.
    """

    spec = resolve_ar_model(model, revision=revision)
    try:
        from mrun.models import load_hf_model, load_tokenizer

        loader_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if spec.revision != LOCAL_MATERIALIZED_REVISION:
            loader_kwargs["revision"] = spec.revision
        tokenizer = load_tokenizer(spec.registry_name, **loader_kwargs)
        loaded = load_hf_model(
            spec.registry_name,
            torch_dtype=torch_dtype,
            device=device,
            attn_implementation=attn_implementation,
            **loader_kwargs,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise ArModelSelectionError(
            f"could not load selected AR model {spec.identity}: {exc}"
        ) from exc

    realized = str(getattr(getattr(loaded, "config", None), "model_type", ""))
    if realized != spec.family:
        raise ArModelSelectionError(
            f"selected {spec.registry_name!r} resolved as {spec.family!r} but loaded model "
            f"reports model_type={realized!r}"
        )
    return spec, loaded, tokenizer


__all__ = [
    "ArModelSelectionError",
    "ArModelSpec",
    "DEFAULT_AR_MODEL",
    "LOCAL_MATERIALIZED_REVISION",
    "SUPPORTED_AR_MODEL_TYPES",
    "available_ar_models",
    "load_ar_model",
    "resolve_ar_model",
]
