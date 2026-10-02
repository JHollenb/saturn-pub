"""Token, coordinate, and runtime-usage receipts for the Saturn debugger.

The debugger has two different kinds of evidence:

* token/coordinate manifests describe *what was supplied to the model*; and
* runtime usage events describe *what the running program consumed at a
  particular boundary*.

Both are metadata-only.  Tensor payloads remain owned by mrun and the worker;
these receipts are small enough to persist beside a checkpoint or stream to a
debugger without changing frame identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TOKEN_STREAM_SCHEMA = "saturn-token-stream-v1"
TOKEN_MANIFEST_SCHEMA = "saturn-token-manifest-v1"
IMAGE_COORDINATE_SCHEMA = "saturn-image-coordinate-manifest-v1"
COORDINATE_MANIFEST_SCHEMA = "saturn-coordinate-manifest-v1"
RUNTIME_USAGE_SCHEMA = "saturn-runtime-usage-v1"
RUNTIME_TRACE_SCHEMA = "saturn-runtime-telemetry-trace-v1"


class RuntimeTelemetryError(ValueError):
    """Raised when a telemetry receipt cannot be made unambiguous."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    return repr(value)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _tolist(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            value = detach().cpu()
        except (AttributeError, RuntimeError, TypeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except (RuntimeError, TypeError):
            pass
    return value


def _first_sequence(value: Any) -> list[Any]:
    value = _tolist(value)
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        if value and isinstance(value[0], (tuple, list)):
            return list(value[0])
        return list(value)
    return []


def _tokenizer_name(tokenizer: Any) -> str | None:
    if tokenizer is None:
        return None
    for attribute in ("name_or_path", "_name_or_path"):
        value = getattr(tokenizer, attribute, None)
        if value:
            return str(value)
    return type(tokenizer).__name__


def _chat_template(tokenizer: Any) -> str | None:
    value = getattr(tokenizer, "chat_template", None)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return repr(value)


def _finite_positive_int(value: Any) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if resolved <= 0 or resolved > 1_000_000:
        return None
    return resolved


def _tokenizer_capacity(tokenizer: Any, fallback: int) -> int:
    configured = _finite_positive_int(getattr(tokenizer, "model_max_length", None))
    return configured or int(fallback)


def _token_strings(tokenizer: Any, token_ids: Sequence[int]) -> tuple[str, ...]:
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(converter):
        return ()
    try:
        values = converter([int(value) for value in token_ids])
    except (TypeError, ValueError, RuntimeError):
        return ()
    if isinstance(values, str):
        return (values,)
    if not isinstance(values, (tuple, list)):
        return ()
    return tuple(str(value) for value in values)


@dataclass(frozen=True, slots=True)
class TokenStreamManifest:
    """Exact token positions observed at one tokenizer/encoder stream."""

    stream: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    tokens: tuple[str, ...] = ()
    tokenizer: str | None = None
    chat_template: str | None = None
    capacity_tokens: int | None = None
    model_max_position_embeddings: int | None = None
    capture_source: str = "encoder_forward"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stream:
            raise RuntimeTelemetryError("token stream requires a stream name")
        ids = tuple(int(value) for value in self.input_ids)
        mask = tuple(int(bool(value)) for value in self.attention_mask)
        if len(ids) != len(mask):
            raise RuntimeTelemetryError(
                f"token stream {self.stream!r} input_ids and attention_mask differ in length"
            )
        tokens = tuple(str(value) for value in self.tokens)
        if tokens and len(tokens) != len(ids):
            raise RuntimeTelemetryError(
                f"token stream {self.stream!r} tokens and input_ids differ in length"
            )
        capacity = len(ids) if self.capacity_tokens is None else int(self.capacity_tokens)
        if capacity < len(ids) or capacity <= 0:
            raise RuntimeTelemetryError(
                f"token stream {self.stream!r} has invalid capacity {capacity}"
            )
        if any(value not in {0, 1} for value in mask):
            raise RuntimeTelemetryError("attention_mask must contain only zero/one values")
        if sum(mask) > capacity:
            raise RuntimeTelemetryError(
                f"token stream {self.stream!r} has more active tokens than capacity"
            )
        object.__setattr__(self, "input_ids", ids)
        object.__setattr__(self, "attention_mask", mask)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "capacity_tokens", capacity)
        object.__setattr__(
            self,
            "model_max_position_embeddings",
            (
                None
                if self.model_max_position_embeddings is None
                else int(self.model_max_position_embeddings)
            ),
        )
        object.__setattr__(self, "metadata", dict(_jsonable(self.metadata)))

    @property
    def active_positions(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.attention_mask) if value)

    @property
    def active_token_count(self) -> int:
        return len(self.active_positions)

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "schema": TOKEN_STREAM_SCHEMA,
                "stream": self.stream,
                "input_ids": list(self.input_ids),
                "attention_mask": list(self.attention_mask),
                "tokenizer": self.tokenizer,
                "chat_template": self.chat_template,
            }
        )

    @property
    def layout_fingerprint(self) -> str:
        """Identity for offset-sensitive debugging, including truncation geometry."""

        return _digest(
            {
                "schema": "saturn-token-stream-layout-v1",
                "stream_fingerprint": self.fingerprint,
                "tokens": list(self.tokens),
                "capacity_tokens": self.capacity_tokens,
                "model_max_position_embeddings": self.model_max_position_embeddings,
                "capture_source": self.capture_source,
                "raw_token_count": self.metadata.get("raw_token_count"),
                "raw_token_ids_sha256": self.metadata.get("raw_token_ids_sha256"),
                "truncated": self.metadata.get("truncated"),
                "truncation_side": self.metadata.get("truncation_side"),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        percent = 100.0 * self.active_token_count / int(self.capacity_tokens)
        return {
            "schema": TOKEN_STREAM_SCHEMA,
            "stream": self.stream,
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "active_positions": list(self.active_positions),
            "active_token_count": self.active_token_count,
            "padded_length": len(self.input_ids),
            "capacity_tokens": int(self.capacity_tokens),
            "context_window_percent": round(percent, 6),
            "tokens": list(self.tokens),
            "tokenizer": self.tokenizer,
            "chat_template": self.chat_template,
            "model_max_position_embeddings": self.model_max_position_embeddings,
            "capture_source": self.capture_source,
            "token_ids_sha256": _digest(list(self.input_ids)),
            "fingerprint": self.fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TokenManifest:
    """A stable collection of all text-tokenizer streams for one prompt."""

    streams: Mapping[str, TokenStreamManifest]
    primary_stream: str | None = None
    prompt_sha256: str | None = None
    status: str = "captured"
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        streams = dict(self.streams)
        if not streams:
            raise RuntimeTelemetryError("token manifest requires at least one stream")
        for name, stream in streams.items():
            if str(name) != stream.stream:
                raise RuntimeTelemetryError("token manifest stream keys must match stream names")
        primary = self.primary_stream or next(iter(streams))
        if primary not in streams:
            raise RuntimeTelemetryError(f"unknown primary token stream {primary!r}")
        if self.status not in {"captured", "partial"}:
            raise RuntimeTelemetryError(f"unsupported token manifest status {self.status!r}")
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "primary_stream", primary)
        object.__setattr__(self, "errors", tuple(str(value) for value in self.errors))
        object.__setattr__(self, "metadata", dict(_jsonable(self.metadata)))

    @property
    def primary(self) -> TokenStreamManifest:
        return self.streams[str(self.primary_stream)]

    @property
    def active_token_count(self) -> int:
        return self.primary.active_token_count

    @property
    def total_active_token_count(self) -> int:
        return sum(stream.active_token_count for stream in self.streams.values())

    @property
    def context_limit_tokens(self) -> int:
        return int(self.primary.capacity_tokens)

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "schema": TOKEN_MANIFEST_SCHEMA,
                "primary_stream": self.primary_stream,
                "streams": {
                    name: stream.fingerprint for name, stream in self.streams.items()
                },
                "prompt_sha256": self.prompt_sha256,
            }
        )

    @property
    def layout_fingerprint(self) -> str:
        return _digest(
            {
                "schema": "saturn-token-manifest-layout-v1",
                "primary_stream": self.primary_stream,
                "streams": {
                    name: stream.layout_fingerprint
                    for name, stream in self.streams.items()
                },
                "prompt_sha256": self.prompt_sha256,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TOKEN_MANIFEST_SCHEMA,
            "status": self.status,
            "primary_stream": self.primary_stream,
            "active_token_count": self.active_token_count,
            "total_active_token_count": self.total_active_token_count,
            "context_limit_tokens": self.context_limit_tokens,
            "context_window_percent": round(
                100.0 * self.active_token_count / self.context_limit_tokens, 6
            ),
            "prompt_sha256": self.prompt_sha256,
            "errors": list(self.errors),
            "streams": {
                name: stream.to_dict() for name, stream in self.streams.items()
            },
            "fingerprint": self.fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
            "metadata": dict(self.metadata),
        }


def _stream_from_inputs(
    stream: str,
    input_ids: Any,
    attention_mask: Any,
    *,
    tokenizer: Any = None,
    capacity_tokens: int | None = None,
    capture_source: str = "encoder_forward",
    metadata: Mapping[str, Any] | None = None,
) -> TokenStreamManifest:
    ids = tuple(int(value) for value in _first_sequence(input_ids))
    mask_values = _first_sequence(attention_mask)
    mask = tuple(int(bool(value)) for value in (mask_values or [1] * len(ids)))
    if len(mask) < len(ids):
        mask = (*mask, *([0] * (len(ids) - len(mask))))
    if len(mask) > len(ids):
        mask = mask[: len(ids)]
    config = getattr(getattr(tokenizer, "config", None), "max_position_embeddings", None)
    model_max = _finite_positive_int(config)
    return TokenStreamManifest(
        stream=stream,
        input_ids=ids,
        attention_mask=mask,
        tokens=_token_strings(tokenizer, ids),
        tokenizer=_tokenizer_name(tokenizer),
        chat_template=_chat_template(tokenizer),
        capacity_tokens=capacity_tokens or len(ids),
        model_max_position_embeddings=model_max,
        capture_source=capture_source,
        metadata=metadata or {},
    )


def _render_for_tokenizer(tokenizer: Any, prompt: str) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template) or not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return str(
            apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        return str(apply_template(messages, tokenize=False, add_generation_prompt=True))


def _tokenize_fallback(
    stream: str,
    tokenizer: Any,
    prompt: str,
    *,
    max_sequence_length: int | None = None,
) -> TokenStreamManifest:
    if tokenizer is None or not callable(tokenizer):
        raise RuntimeTelemetryError(f"tokenizer for stream {stream!r} is unavailable")
    # Flux.2 deliberately bounds its Qwen conditioner to the pipeline's
    # 512-position prompt stream even though the underlying Qwen model may
    # advertise a much larger autoregressive context.  Chat-template streams
    # therefore default to the pipeline conditioner capacity; callers can
    # override it explicitly for another runtime.
    capacity = max_sequence_length or (
        512 if getattr(tokenizer, "chat_template", None) else _tokenizer_capacity(tokenizer, 512)
    )
    rendered = _render_for_tokenizer(tokenizer, prompt)
    raw_metadata: dict[str, Any] = {
        "truncation_side": str(getattr(tokenizer, "truncation_side", "right")),
    }
    try:
        raw_encoded = tokenizer(
            rendered,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        if isinstance(raw_encoded, Mapping) and raw_encoded.get("input_ids") is not None:
            raw_ids = [int(value) for value in _first_sequence(raw_encoded["input_ids"])]
            raw_metadata.update(
                {
                    "raw_token_count": len(raw_ids),
                    "raw_token_ids_sha256": _digest(raw_ids),
                    "truncated": len(raw_ids) > capacity,
                }
            )
    except (RuntimeTelemetryError, TypeError, ValueError, RuntimeError):
        raw_metadata["truncation_observation"] = "unavailable"
    encoded = tokenizer(
        rendered,
        padding="max_length",
        truncation=True,
        max_length=capacity,
        return_tensors="pt",
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise RuntimeTelemetryError(f"tokenizer stream {stream!r} returned no input_ids")
    return _stream_from_inputs(
        stream,
        encoded["input_ids"],
        encoded.get("attention_mask"),
        tokenizer=tokenizer,
        capacity_tokens=capacity,
        capture_source="tokenizer_fallback",
        metadata=raw_metadata,
    )


def _pipeline_tokenizers(pipeline: Any) -> list[tuple[str, Any, Any]]:
    result: list[tuple[str, Any, Any]] = []
    seen: set[int] = set()
    for index in range(1, 5):
        encoder_name = "text_encoder" if index == 1 else f"text_encoder_{index}"
        tokenizer_name = "tokenizer" if index == 1 else f"tokenizer_{index}"
        encoder = getattr(pipeline, encoder_name, None)
        tokenizer = getattr(pipeline, tokenizer_name, None)
        if encoder is None and tokenizer is None:
            continue
        identity = id(encoder) if encoder is not None else id(tokenizer)
        if identity in seen:
            continue
        seen.add(identity)
        result.append((encoder_name, encoder, tokenizer))
    return result


def _primary_stream(streams: Mapping[str, Any]) -> str:
    """Select the sequence conditioner, not Flux.1's pooled CLIP stream."""

    if "text_encoder_2" in streams:
        return "text_encoder_2"
    if "text_encoder" in streams:
        return "text_encoder"
    return next(iter(streams))


def token_manifest_from_pipeline(
    pipeline: Any,
    prompt: str,
    *,
    max_sequence_length: int | None = None,
) -> dict[str, Any]:
    """Capture tokenizer-visible positions without running an encoder."""

    streams: dict[str, TokenStreamManifest] = {}
    errors: list[str] = []
    for stream, _encoder, tokenizer in _pipeline_tokenizers(pipeline):
        if tokenizer is None:
            continue
        try:
            streams[stream] = _tokenize_fallback(
                stream,
                tokenizer,
                prompt,
                max_sequence_length=max_sequence_length,
            )
        except (RuntimeTelemetryError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"{stream}: {exc}")
    if not streams:
        raise RuntimeTelemetryError(
            "no pipeline tokenizer could produce a token manifest"
            + (f" ({'; '.join(errors)})" if errors else "")
        )
    manifest = TokenManifest(
        streams=streams,
        primary_stream=_primary_stream(streams),
        prompt_sha256=_digest(prompt),
        status="partial" if errors else "captured",
        errors=tuple(errors),
        metadata={"source": "pipeline_tokenizer"},
    )
    return manifest.to_dict()


def capture_prompt_token_manifest(
    pipeline: Any,
    encode: Callable[[], Any],
    prompt: str,
    *,
    max_sequence_length: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run the normal encoder while recording its exact token inputs.

    Hooks are attached only around the supplied encode call.  If an embedding
    cache bypasses the encoder, the function falls back to the pipeline's
    tokenizer and marks the source explicitly instead of silently claiming an
    encoder observation.
    """

    captures: dict[str, TokenStreamManifest] = {}
    hooks: list[Any] = []

    def make_hook(stream: str, tokenizer: Any) -> Callable[..., None]:
        def hook(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            _output: Any,
        ) -> None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None:
                return
            attention_mask = kwargs.get("attention_mask")
            captures[stream] = _stream_from_inputs(
                stream,
                input_ids,
                attention_mask,
                tokenizer=tokenizer,
                # The encoder's actual padded input length is the capacity
                # for this conditioner stream.  A caller-supplied global
                # sequence limit must not turn Flux.1's 77-token CLIP stream
                # into a fictitious 512-token context.
                capacity_tokens=None,
                capture_source="encoder_forward",
            )

        return hook

    try:
        for stream, encoder, tokenizer in _pipeline_tokenizers(pipeline):
            if encoder is None or not callable(getattr(encoder, "register_forward_hook", None)):
                continue
            hook = make_hook(stream, tokenizer)
            try:
                handle = encoder.register_forward_hook(hook, with_kwargs=True)
            except TypeError:
                def legacy_hook(
                    module: Any,
                    args: tuple[Any, ...],
                    output: Any,
                    hook: Callable[..., None] = hook,
                ) -> None:
                    hook(module, args, {}, output)

                handle = encoder.register_forward_hook(legacy_hook)
            hooks.append(handle)
        embeds = encode()
    finally:
        for handle in hooks:
            handle.remove()

    errors: list[str] = []
    for stream, _encoder, tokenizer in _pipeline_tokenizers(pipeline):
        if stream in captures or tokenizer is None:
            continue
        try:
            captures[stream] = _tokenize_fallback(
                stream,
                tokenizer,
                prompt,
                max_sequence_length=max_sequence_length,
            )
        except (RuntimeTelemetryError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"{stream}: {exc}")
    if not captures:
        raise RuntimeTelemetryError(
            "encoder produced no observable token streams"
            + (f" ({'; '.join(errors)})" if errors else "")
        )
    manifest = TokenManifest(
        streams=captures,
        primary_stream=_primary_stream(captures),
        prompt_sha256=_digest(prompt),
        status="partial" if errors else "captured",
        errors=tuple(errors),
        metadata={"source": "encoder_forward_or_tokenizer_fallback"},
    )
    return embeds, manifest.to_dict()


def _image_rows(value: Any) -> list[list[int]]:
    rows = _tolist(value)
    while (
        isinstance(rows, list)
        and rows
        and isinstance(rows[0], list)
        and rows[0]
        and isinstance(rows[0][0], list)
    ):
        rows = rows[0]
    if not isinstance(rows, list):
        return []
    return [
        [int(item) for item in row]
        for row in rows
        if isinstance(row, (tuple, list)) and len(row) >= 2
    ]


def image_token_count(value: Any) -> int:
    return len(_image_rows(value))


def build_image_coordinate_manifest(
    latent_ids: Any,
    *,
    height: int,
    width: int,
    reference_token_count: int = 0,
) -> dict[str, Any]:
    """Map native FLUX image IDs to grid and approximate output-pixel cells."""

    rows = _image_rows(latent_ids)
    if not rows:
        raise RuntimeTelemetryError("image coordinate mapping requires native latent IDs")
    native_rows = sorted({row[1] if len(row) >= 3 else row[0] for row in rows})
    native_cols = sorted({row[2] if len(row) >= 3 else row[1] for row in rows})
    row_index = {value: index for index, value in enumerate(native_rows)}
    col_index = {value: index for index, value in enumerate(native_cols)}
    grid_height = len(native_rows)
    grid_width = len(native_cols)
    tokens: list[dict[str, Any]] = []
    for flat_index, native in enumerate(rows):
        native_row = native[1] if len(native) >= 3 else native[0]
        native_col = native[2] if len(native) >= 3 else native[1]
        grid_row = row_index[native_row]
        grid_col = col_index[native_col]
        x0 = (grid_col * int(width)) // grid_width
        x1 = ((grid_col + 1) * int(width)) // grid_width
        y0 = (grid_row * int(height)) // grid_height
        y1 = ((grid_row + 1) * int(height)) // grid_height
        tokens.append(
            {
                "flat_index": flat_index,
                "native_id": native,
                "native_row": native_row,
                "native_col": native_col,
                "grid_row": grid_row,
                "grid_col": grid_col,
                "pixel_bbox": [x0, y0, x1, y1],
            }
        )
    payload = {
        "schema": IMAGE_COORDINATE_SCHEMA,
        "coordinate_space": "flux.latent_position",
        "native_axes": (
            ["time", "height", "width", "layer"]
            if len(rows[0]) >= 4
            else ["height", "width", "layer"]
        ),
        "resolution": [int(height), int(width)],
        "grid": {"height": grid_height, "width": grid_width},
        "token_count": len(tokens),
        "reference_token_count": int(reference_token_count),
        "mapping_semantics": "approximate_latent_cell_not_pixel_ownership",
        "latent_ids_sha256": _digest(rows),
        "tokens": tokens,
    }
    payload["fingerprint"] = _digest(
        {key: value for key, value in payload.items() if key != "fingerprint"}
    )
    return payload


def build_coordinate_manifest(
    token_manifest: Mapping[str, Any],
    latent_ids: Any,
    *,
    height: int,
    width: int,
    reference_token_count: int = 0,
) -> dict[str, Any]:
    """Join prompt-token and image-latent coordinates into one debugger receipt."""

    image = build_image_coordinate_manifest(
        latent_ids,
        height=height,
        width=width,
        reference_token_count=reference_token_count,
    )
    payload = {
        "schema": COORDINATE_MANIFEST_SCHEMA,
        "mapping_version": "native-text-and-latent-grid-v1",
        "text": _jsonable(token_manifest),
        "image": image,
        "join_semantics": (
            "text positions are tokenizer coordinates; image positions are native latent "
            "coordinates; neither is a semantic object label"
        ),
    }
    payload["fingerprint"] = _digest(
        {key: value for key, value in payload.items() if key != "fingerprint"}
    )
    return payload


@dataclass(frozen=True, slots=True)
class RuntimeUsage:
    """One live, metadata-only usage snapshot."""

    phase: str
    event: str
    event_index: int
    context_tokens: int
    context_limit_tokens: int | None
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    prefill_tokens: int = 0
    completion_tokens_visible: int = 0
    generated_tokens_raw: int = 0
    accepted_tokens: int = 0
    decode_steps: int = 0
    step_index: int | None = None
    total_steps: int | None = None
    observed_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    token_uses: tuple[Mapping[str, Any], ...] = ()
    session_id: str | None = None
    run_id: str | None = None
    token_manifest_fingerprint: str | None = None
    coordinate_manifest_fingerprint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "event_index",
            "context_tokens",
            "prompt_tokens",
            "cached_prompt_tokens",
            "prefill_tokens",
            "completion_tokens_visible",
            "generated_tokens_raw",
            "accepted_tokens",
            "decode_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise RuntimeTelemetryError(f"runtime usage {name} must be non-negative")
        if self.context_limit_tokens is not None:
            if isinstance(self.context_limit_tokens, bool) or int(self.context_limit_tokens) <= 0:
                raise RuntimeTelemetryError("context_limit_tokens must be positive")
            if int(self.context_tokens) > int(self.context_limit_tokens):
                raise RuntimeTelemetryError("context_tokens exceeds context_limit_tokens")
        if int(self.cached_prompt_tokens) > int(self.prompt_tokens):
            raise RuntimeTelemetryError("cached_prompt_tokens exceeds prompt_tokens")
        object.__setattr__(
            self,
            "token_uses",
            tuple(dict(_jsonable(item)) for item in self.token_uses),
        )
        object.__setattr__(self, "metadata", dict(_jsonable(self.metadata)))

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens_visible)

    @property
    def context_window_fraction(self) -> float | None:
        if self.context_limit_tokens is None:
            return None
        return float(self.context_tokens) / float(self.context_limit_tokens)

    @property
    def context_window_percent(self) -> float | None:
        fraction = self.context_window_fraction
        return None if fraction is None else round(100.0 * fraction, 6)

    @property
    def remaining_context_tokens(self) -> int | None:
        if self.context_limit_tokens is None:
            return None
        return int(self.context_limit_tokens) - int(self.context_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_USAGE_SCHEMA,
            "phase": self.phase,
            "event": self.event,
            "event_index": int(self.event_index),
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "observed_at_utc": self.observed_at_utc,
            "context_tokens": int(self.context_tokens),
            "context_limit_tokens": self.context_limit_tokens,
            "context_window_fraction": self.context_window_fraction,
            "context_window_percent": self.context_window_percent,
            "remaining_context_tokens": self.remaining_context_tokens,
            "prompt_tokens": int(self.prompt_tokens),
            "cached_prompt_tokens": int(self.cached_prompt_tokens),
            "prefill_tokens": int(self.prefill_tokens),
            "completion_tokens_visible": int(self.completion_tokens_visible),
            "generated_tokens_raw": int(self.generated_tokens_raw),
            "accepted_tokens": int(self.accepted_tokens),
            "total_tokens": self.total_tokens,
            "decode_steps": int(self.decode_steps),
            "token_uses": [dict(item) for item in self.token_uses],
            "session_id": self.session_id,
            "run_id": self.run_id,
            "token_manifest_fingerprint": self.token_manifest_fingerprint,
            "coordinate_manifest_fingerprint": self.coordinate_manifest_fingerprint,
            "metadata": dict(self.metadata),
        }


def build_diffusion_usage(
    token_manifest: Mapping[str, Any],
    *,
    step_index: int,
    total_steps: int,
    event_index: int,
    image_token_count_value: int,
    reference_token_count: int = 0,
    event: str = "denoise_step",
    session_id: str | None = None,
    run_id: str | None = None,
    coordinate_manifest_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Build a diffusion usage event without calling image latents LLM tokens."""

    streams = token_manifest.get("streams") if isinstance(token_manifest, Mapping) else None
    streams = streams if isinstance(streams, Mapping) else {}
    primary_name = str(token_manifest.get("primary_stream") or next(iter(streams), "text"))
    primary = streams.get(primary_name) if isinstance(streams, Mapping) else None
    primary = primary if isinstance(primary, Mapping) else {}
    context_tokens = int(
        primary.get("active_token_count")
        or token_manifest.get("active_token_count")
        or 0
    )
    context_limit = token_manifest.get("context_limit_tokens")
    context_limit = None if context_limit is None else int(context_limit)
    token_uses: list[dict[str, Any]] = []
    for name, stream in streams.items():
        active = int((stream or {}).get("active_token_count") or 0)
        token_uses.append(
            {
                "name": "text_conditioning",
                "stream": str(name),
                "tokens": active,
                "role": "conditioning",
                "cadence": "per_denoise_step",
            }
        )
    token_uses.append(
        {
            "name": "image_latent",
            "tokens": int(image_token_count_value),
            "role": "latent_state",
            "cadence": "per_denoise_step",
        }
    )
    if int(reference_token_count):
        token_uses.append(
            {
                "name": "reference_latent",
                "tokens": int(reference_token_count),
                "role": "reference_conditioning",
                "cadence": "per_denoise_step",
            }
        )
    usage = RuntimeUsage(
        phase="diffusion",
        event=event,
        event_index=event_index,
        step_index=int(step_index),
        total_steps=int(total_steps),
        context_tokens=context_tokens,
        context_limit_tokens=context_limit,
        prompt_tokens=sum(
            int((stream or {}).get("active_token_count") or 0)
            for stream in streams.values()
        ),
        prefill_tokens=sum(
            int((stream or {}).get("active_token_count") or 0)
            for stream in streams.values()
        ),
        decode_steps=0,
        token_uses=tuple(token_uses),
        session_id=session_id,
        run_id=run_id,
        token_manifest_fingerprint=token_manifest.get("fingerprint"),
        coordinate_manifest_fingerprint=coordinate_manifest_fingerprint,
        metadata={
            "context_kind": "fixed_diffusion_conditioning",
            "token_usage_basis": "conditioning_and_latent_observation",
            "not_autoregressive_completion": True,
            "observation_kind": (
                "snapshot"
                if event in {"capture_complete", "replay_context_loaded"}
                else (
                    "conditioning_prefill"
                    if event == "conditioning_prefill"
                    else "denoise_step"
                )
            ),
            "work_executed": event not in {"capture_complete", "replay_context_loaded"},
            "observed_token_work_tokens": (
                0
                if event in {"capture_complete", "replay_context_loaded"}
                else sum(int(item["tokens"]) for item in token_uses)
            ),
            "physical_tokens_per_denoise_step": sum(
                int(item["tokens"]) for item in token_uses
            ),
        },
    )
    return usage.to_dict()


class RuntimeUsageLedger:
    """Accumulate committed prefill/decode usage for a resident Saturn run.

    The ledger deliberately takes the current context length from the caller
    on every commit.  That makes sliding-window eviction and paged-KV
    implementations observable without pretending that ``prompt + generated``
    is always the resident context length.
    """

    def __init__(
        self,
        context_limit_tokens: int | None,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        token_manifest_fingerprint: str | None = None,
        coordinate_manifest_fingerprint: str | None = None,
    ) -> None:
        if context_limit_tokens is not None and int(context_limit_tokens) <= 0:
            raise RuntimeTelemetryError("context_limit_tokens must be positive")
        self.context_limit_tokens = (
            None if context_limit_tokens is None else int(context_limit_tokens)
        )
        self.session_id = session_id
        self.run_id = run_id
        self.token_manifest_fingerprint = token_manifest_fingerprint
        self.coordinate_manifest_fingerprint = coordinate_manifest_fingerprint
        self._prefilled = False
        self._context_tokens = 0
        self._prompt_tokens = 0
        self._cached_prompt_tokens = 0
        self._prefill_tokens = 0
        self._completion_tokens_visible = 0
        self._generated_tokens_raw = 0
        self._accepted_tokens = 0
        self._decode_steps = 0
        self._event_index = 0
        self._events: list[dict[str, Any]] = []

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._events)

    @property
    def latest(self) -> dict[str, Any] | None:
        return None if not self._events else dict(self._events[-1])

    def _emit(
        self,
        *,
        event: str,
        token_uses: Sequence[Mapping[str, Any]] = (),
        step_index: int | None = None,
        total_steps: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        usage = RuntimeUsage(
            phase="inference",
            event=event,
            event_index=self._event_index,
            context_tokens=self._context_tokens,
            context_limit_tokens=self.context_limit_tokens,
            prompt_tokens=self._prompt_tokens,
            cached_prompt_tokens=self._cached_prompt_tokens,
            prefill_tokens=self._prefill_tokens,
            completion_tokens_visible=self._completion_tokens_visible,
            generated_tokens_raw=self._generated_tokens_raw,
            accepted_tokens=self._accepted_tokens,
            decode_steps=self._decode_steps,
            step_index=step_index,
            total_steps=total_steps,
            token_uses=tuple(token_uses),
            session_id=self.session_id,
            run_id=self.run_id,
            token_manifest_fingerprint=self.token_manifest_fingerprint,
            coordinate_manifest_fingerprint=self.coordinate_manifest_fingerprint,
            metadata={
                "context_kind": "committed_kv_length",
                **dict(metadata or {}),
            },
        )
        payload = usage.to_dict()
        self._events.append(payload)
        self._event_index += 1
        return payload

    def prefill(
        self,
        prompt_tokens: int,
        *,
        cached_prompt_tokens: int = 0,
        context_tokens: int | None = None,
        token_uses: Sequence[Mapping[str, Any]] = (),
        event: str = "prefill",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record the exact input work and initial committed KV length."""

        if self._prefilled:
            raise RuntimeTelemetryError("runtime usage ledger already has a prefill")
        prompt_tokens = int(prompt_tokens)
        cached_prompt_tokens = int(cached_prompt_tokens)
        if prompt_tokens < 0 or cached_prompt_tokens < 0:
            raise RuntimeTelemetryError("prefill token counts must be non-negative")
        if cached_prompt_tokens > prompt_tokens:
            raise RuntimeTelemetryError("cached_prompt_tokens exceeds prompt_tokens")
        self._prefilled = True
        self._prompt_tokens = prompt_tokens
        self._cached_prompt_tokens = cached_prompt_tokens
        self._prefill_tokens = prompt_tokens - cached_prompt_tokens
        self._context_tokens = (
            prompt_tokens if context_tokens is None else int(context_tokens)
        )
        if self._context_tokens < 0:
            raise RuntimeTelemetryError("context_tokens must be non-negative")
        return self._emit(event=event, token_uses=token_uses, metadata=metadata)

    def commit(
        self,
        accepted_tokens: int,
        *,
        visible_tokens: int | None = None,
        context_tokens: int | None = None,
        token_uses: Sequence[Mapping[str, Any]] = (),
        event: str = "decode_commit",
        step_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record tokens accepted by the model and the post-commit KV length."""

        if not self._prefilled:
            raise RuntimeTelemetryError("decode commit requires a prefill")
        accepted_tokens = int(accepted_tokens)
        visible = accepted_tokens if visible_tokens is None else int(visible_tokens)
        if accepted_tokens < 0 or visible < 0:
            raise RuntimeTelemetryError("committed token counts must be non-negative")
        if visible > accepted_tokens:
            raise RuntimeTelemetryError("visible_tokens exceeds accepted_tokens")
        self._accepted_tokens += accepted_tokens
        self._generated_tokens_raw += accepted_tokens
        self._completion_tokens_visible += visible
        self._decode_steps += 1
        self._context_tokens = (
            self._context_tokens + accepted_tokens
            if context_tokens is None
            else int(context_tokens)
        )
        if self._context_tokens < 0:
            raise RuntimeTelemetryError("context_tokens must be non-negative")
        return self._emit(
            event=event,
            token_uses=token_uses,
            step_index=step_index,
            metadata={
                "accepted_tokens_this_commit": accepted_tokens,
                "visible_tokens_this_commit": visible,
                **dict(metadata or {}),
            },
        )


__all__ = [
    "COORDINATE_MANIFEST_SCHEMA",
    "IMAGE_COORDINATE_SCHEMA",
    "RUNTIME_USAGE_SCHEMA",
    "RUNTIME_TRACE_SCHEMA",
    "TOKEN_MANIFEST_SCHEMA",
    "TOKEN_STREAM_SCHEMA",
    "RuntimeTelemetryError",
    "RuntimeUsageLedger",
    "RuntimeUsage",
    "TokenManifest",
    "TokenStreamManifest",
    "build_coordinate_manifest",
    "build_diffusion_usage",
    "build_image_coordinate_manifest",
    "capture_prompt_token_manifest",
    "image_token_count",
    "token_manifest_from_pipeline",
]
