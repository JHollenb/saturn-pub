"""Versioned control protocol for Saturn's live debugger.

The protocol is deliberately transport-agnostic.  mrun may carry these messages over
its authenticated job control API, while tests and local workers can use the in-memory
transport.  Values crossing the boundary are JSON metadata only; tensor payloads are
StateCut/custody references and are never implicit raw-memory handles.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEBUGGER_PROTOCOL_SCHEMA = "saturn-debugger-protocol-v1"
DEBUGGER_CAPABILITY_SCHEMA = "saturn-debugger-capability-v1"
DEBUGGER_REQUEST_SCHEMA = "saturn-debugger-request-v1"
DEBUGGER_RESPONSE_SCHEMA = "saturn-debugger-response-v1"
DISCOVERY_RELAY_BINDING_SCHEMA = "mdb-discovery-relay-binding-v1"
DISCOVERY_RELAY_FIELD = "$discovery_relay"

_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_SAFE_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_INT_FIELDS = {"int", "integer", "index", "step", "row"}


class DebuggerProtocolError(RuntimeError):
    """A refused, malformed, stale, or unavailable debugger operation."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _id(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DebuggerProtocolError(f"{field} must be a non-empty string")
    return value.strip()


def _json_mapping(value: Mapping[str, Any] | None, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DebuggerProtocolError(f"{field} must be an object")
    try:
        return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DebuggerProtocolError(f"{field} must be JSON data") from exc


@dataclass(frozen=True, slots=True)
class DebuggerAddressSpec:
    """A declarative, typed logical address exposed by a native adapter.

    The template is metadata, not executable code.  A worker may use a richer
    native resolver behind the same name, but a client can never smuggle a
    Python expression or a raw pointer into an address declaration.
    """

    name: str
    kind: str
    path_template: str
    readable: bool = True
    writable: bool = False
    fields: Mapping[str, Any] = field(default_factory=dict)
    struct_name: str | None = None
    source_write: Mapping[str, Any] = field(default_factory=dict)
    boundaries: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    numerical_contract: str = "exact"

    def __post_init__(self) -> None:
        name = _id(self.name, "address name")
        kind = _id(self.kind, "address kind")
        template = _id(self.path_template, "address path_template")
        if not _SAFE_NAME.fullmatch(name):
            raise DebuggerProtocolError(f"address name is not a safe identifier: {name!r}")
        if not _SAFE_NAME.fullmatch(kind):
            raise DebuggerProtocolError(f"address kind is not a safe identifier: {kind!r}")
        fields = _json_mapping(self.fields, "address fields")
        placeholders = {
            field_name
            for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(
                template
            )
            if field_name is not None
        }
        parsed_fields = [
            field_name
            for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(
                template
            )
            if field_name is not None
        ]
        if any(not _SAFE_FIELD.fullmatch(field_name) for field_name in parsed_fields):
            raise DebuggerProtocolError(
                "address path_template fields must be simple safe identifiers"
            )
        if len(parsed_fields) != len(set(parsed_fields)):
            raise DebuggerProtocolError("address path_template fields may not repeat")
        unknown = placeholders - set(fields)
        if unknown:
            raise DebuggerProtocolError(
                f"address template fields lack declarations: {sorted(unknown)}"
            )
        if self.struct_name is not None:
            _id(self.struct_name, "address struct_name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "path_template", template)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "source_write", _json_mapping(self.source_write, "source_write"))
        object.__setattr__(self, "boundaries", tuple(str(value) for value in self.boundaries))
        object.__setattr__(self, "invalidates", tuple(str(value) for value in self.invalidates))
        object.__setattr__(
            self,
            "numerical_contract",
            _id(self.numerical_contract, "numerical_contract"),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DebuggerAddressSpec:
        if not isinstance(value, Mapping):
            raise DebuggerProtocolError("address spec must be an object")
        return cls(
            name=str(value.get("name") or ""),
            kind=str(value.get("kind") or "custom"),
            path_template=str(value.get("path_template") or ""),
            readable=bool(value.get("readable", True)),
            writable=bool(value.get("writable", False)),
            fields=dict(value.get("fields") or {}),
            struct_name=value.get("struct_name"),
            source_write=dict(value.get("source_write") or {}),
            boundaries=tuple(value.get("boundaries") or ()),
            invalidates=tuple(value.get("invalidates") or ()),
            numerical_contract=str(value.get("numerical_contract") or "exact"),
        )

    def _match_pattern(self) -> re.Pattern[str]:
        parts: list[str] = []
        for literal, field_name, _format_spec, _conversion in string.Formatter().parse(
            self.path_template
        ):
            parts.append(re.escape(literal))
            if field_name is not None:
                # The path grammar is deliberately bounded to one segment per
                # field.  A site such as ``joint.4`` remains one segment; a
                # field cannot escape its brackets or separators.
                parts.append(r"(?P<" + field_name + r">[^\[\]]+)")
        return re.compile("^" + "".join(parts) + "$")

    def match(self, path: str) -> dict[str, Any] | None:
        """Resolve a path to typed field values, or return ``None``."""

        if not isinstance(path, str):
            return None
        match = self._match_pattern().match(path)
        if match is None:
            return None
        resolved: dict[str, Any] = dict(match.groupdict())
        for field_name, descriptor in self.fields.items():
            if not isinstance(descriptor, Mapping):
                descriptor = {"type": descriptor}
            field_type = str(descriptor.get("type") or "string").lower()
            raw = resolved.get(field_name)
            if field_type in _INT_FIELDS:
                try:
                    parsed: Any = int(str(raw))
                except (TypeError, ValueError):
                    return None
                if str(parsed) != str(raw):
                    return None
                if parsed < int(descriptor.get("minimum", 0)):
                    return None
                maximum = descriptor.get("maximum")
                if maximum is not None and parsed > int(maximum):
                    return None
                resolved[field_name] = parsed
            elif field_type in {"slug", "name"} and not _SAFE_NAME.fullmatch(str(raw)):
                return None
            choices = descriptor.get("choices")
            if choices is not None and str(raw) not in {str(choice) for choice in choices}:
                return None
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "path_template": self.path_template,
            "readable": self.readable,
            "writable": self.writable,
            "fields": dict(self.fields),
            "struct_name": self.struct_name,
            "source_write": dict(self.source_write),
            "boundaries": list(self.boundaries),
            "invalidates": list(self.invalidates),
            "numerical_contract": self.numerical_contract,
        }


@dataclass(frozen=True, slots=True)
class DebuggerStructSpec:
    """A user/runtime-defined typed struct that can own logical addresses."""

    name: str
    schema: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    address_specs: tuple[DebuggerAddressSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _id(self.name, "struct name")
        schema = _id(self.schema, "struct schema")
        if not _SAFE_NAME.fullmatch(name):
            raise DebuggerProtocolError(f"struct name is not a safe identifier: {name!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "fields", _json_mapping(self.fields, "struct fields"))
        object.__setattr__(self, "address_specs", tuple(self.address_specs))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "struct metadata"))
        for address in self.address_specs:
            if address.struct_name not in {None, name}:
                raise DebuggerProtocolError(
                    f"struct address {address.name!r} belongs to another struct"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DebuggerStructSpec:
        if not isinstance(value, Mapping):
            raise DebuggerProtocolError("struct spec must be an object")
        raw_addresses = value.get("address_specs") or value.get("addresses") or ()
        if not isinstance(raw_addresses, (list, tuple)):
            raise DebuggerProtocolError("struct address_specs must be a list")
        name = str(value.get("name") or "")
        addresses = []
        for raw in raw_addresses:
            address = DebuggerAddressSpec.from_dict(raw)
            if address.struct_name is None:
                address = DebuggerAddressSpec(
                    name=address.name,
                    kind=address.kind,
                    path_template=address.path_template,
                    readable=address.readable,
                    writable=address.writable,
                    fields=address.fields,
                    struct_name=name,
                    source_write=address.source_write,
                    boundaries=address.boundaries,
                    invalidates=address.invalidates,
                    numerical_contract=address.numerical_contract,
                )
            addresses.append(address)
        return cls(
            name=name,
            schema=str(value.get("schema") or "saturn-debugger-struct-v1"),
            fields=dict(value.get("fields") or {}),
            address_specs=tuple(addresses),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema": self.schema,
            "fields": dict(self.fields),
            "address_specs": [address.to_dict() for address in self.address_specs],
            "metadata": dict(self.metadata),
        }


def conditioner_row_path(stream: str, site: str, step: int, row: int) -> str:
    """Return the canonical logical path for one FLUX conditioner row."""

    stream = _id(stream, "conditioner stream")
    site = _id(site, "conditioner site")
    if not _SAFE_NAME.fullmatch(stream) or not _SAFE_NAME.fullmatch(site):
        raise DebuggerProtocolError("conditioner stream and site must be safe names")
    if int(step) < 0 or int(row) < 0:
        raise DebuggerProtocolError("conditioner step and row must be non-negative")
    return f"conditioner[{stream}][{site}][step={int(step)}][row={int(row)}]"


def conditioner_row_address(path: str) -> dict[str, Any] | None:
    """Parse a canonical conditioner-row path without touching native state."""

    pattern = re.compile(
        r"^conditioner\[(?P<stream>[^\[\]]+)\]"
        r"\[(?P<site>[^\[\]]+)\]\[step=(?P<step>\d+)\]\[row=(?P<row>\d+)\]$"
    )
    match = pattern.match(str(path))
    if match is None:
        return None
    return {
        "kind": "conditioner-row",
        "stream": match.group("stream"),
        "site": match.group("site"),
        "step": int(match.group("step")),
        "row": int(match.group("row")),
        "path": str(path),
    }


@dataclass(frozen=True, slots=True)
class DebugCapability:
    """Capabilities advertised by one instrumented resident worker."""

    job_id: str
    session_id: str
    family: str
    model_identity: dict[str, Any]
    boundaries: tuple[str, ...]
    inspectable_slots: tuple[str, ...]
    writable_slots: tuple[str, ...]
    exact_replay: bool
    endpoint: str | None = None
    state_schema_id: str = "unknown"
    backend: str = "unknown"
    consumer: str = "native"
    source_fingerprint: str | None = None
    boundary_catalog: tuple[Mapping[str, Any], ...] = ()
    address_specs: tuple[DebuggerAddressSpec, ...] = ()
    structs: tuple[DebuggerStructSpec, ...] = ()
    algebra_ops: tuple[str, ...] = ()
    optional_operations: tuple[str, ...] = ()
    checkpoint_requires_pause: bool = False

    def __post_init__(self) -> None:
        _id(self.job_id, "job_id")
        _id(self.session_id, "session_id")
        _id(self.family, "family")
        if not self.model_identity:
            raise DebuggerProtocolError("model_identity must not be empty")
        if not self.boundaries:
            raise DebuggerProtocolError("at least one typed boundary is required")
        unknown_catalog = {
            str(item.get("kind"))
            for item in self.boundary_catalog
            if isinstance(item, Mapping) and item.get("kind") is not None
        } - set(self.boundaries)
        if unknown_catalog:
            raise DebuggerProtocolError(
                f"boundary catalog contains undeclared kinds: {sorted(unknown_catalog)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEBUGGER_CAPABILITY_SCHEMA,
            "protocol": DEBUGGER_PROTOCOL_SCHEMA,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "family": self.family,
            "model_identity": self.model_identity,
            "boundaries": list(self.boundaries),
            "inspectable_slots": list(self.inspectable_slots),
            "writable_slots": list(self.writable_slots),
            "exact_replay": self.exact_replay,
            "endpoint": self.endpoint,
            "state_schema_id": self.state_schema_id,
            "backend": self.backend,
            "consumer": self.consumer,
            "source_fingerprint": self.source_fingerprint,
            "boundary_catalog": [dict(item) for item in self.boundary_catalog],
            "address_specs": [item.to_dict() for item in self.address_specs],
            "structs": [item.to_dict() for item in self.structs],
            "algebra_ops": list(self.algebra_ops),
            "optional_operations": list(self.optional_operations),
            "checkpoint_requires_pause": self.checkpoint_requires_pause,
        }


@dataclass(frozen=True, slots=True)
class BreakpointSpec:
    kind: str
    index: int | None = None
    expression: str | None = None
    enabled: bool = True
    breakpoint_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        if not isinstance(self.kind, str) or not _SAFE_NAME.fullmatch(self.kind):
            raise DebuggerProtocolError(f"unsupported breakpoint kind: {self.kind}")
        if self.kind == "if" and not self.expression:
            raise DebuggerProtocolError("conditional breakpoint requires expression")
        if self.kind != "if" and (self.index is None or self.index < 0):
            raise DebuggerProtocolError("boundary breakpoint requires a non-negative index")
        return {
            "breakpoint_id": self.breakpoint_id,
            "kind": self.kind,
            "index": self.index,
            "expression": self.expression,
            "enabled": self.enabled,
        }


@dataclass(frozen=True, slots=True)
class WatchSpec:
    path: str
    predicate: str | None = None
    mode: str = "change"
    watch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        if self.mode not in {"change", "level", "edge"}:
            raise DebuggerProtocolError(
                "watch mode must be one of: change, level, edge"
            )
        return {
            "watch_id": self.watch_id,
            "path": _id(self.path, "watch path"),
            "predicate": self.predicate,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class ActSpec:
    """A typed, previewable write with optional safe algebra/address metadata."""

    path: str
    value: Any
    parent_cut: str
    act_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sign: str = "replace"
    numerical_contract: str = "exact"
    expression: str | None = None
    address: Mapping[str, Any] = field(default_factory=dict)
    struct_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        _id(self.path, "Act path")
        _id(self.parent_cut, "parent_cut")
        if self.sign not in {"replace", "add", "scale"}:
            raise DebuggerProtocolError(f"unsupported Act sign: {self.sign}")
        try:
            json.dumps(self.value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DebuggerProtocolError(
                "Act value must be JSON data; use worker-side algebra for native tensors"
            ) from exc
        if self.expression is not None:
            expression = _id(self.expression, "Act expression")
            if len(expression) > 4096:
                raise DebuggerProtocolError("Act expression is too long")
        if self.struct_name is not None:
            _id(self.struct_name, "Act struct_name")
        address = _json_mapping(self.address, "Act address")
        return {
            "act_id": self.act_id,
            "path": self.path,
            "value": self.value,
            "parent_cut": self.parent_cut,
            "sign": self.sign,
            "numerical_contract": self.numerical_contract,
            "expression": self.expression,
            "address": address,
            "struct_name": self.struct_name,
        }


@dataclass(frozen=True, slots=True)
class DebugRequest:
    operation: str
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sequence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEBUGGER_REQUEST_SCHEMA,
            "protocol": DEBUGGER_PROTOCOL_SCHEMA,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "operation": _id(self.operation, "operation"),
            "payload": self.payload,
        }


@dataclass(frozen=True, slots=True)
class DebugResponse:
    request_id: str
    sequence: int
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEBUGGER_RESPONSE_SCHEMA,
            "protocol": DEBUGGER_PROTOCOL_SCHEMA,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
        }

    def require_ok(self) -> dict[str, Any]:
        if not self.ok:
            raise DebuggerProtocolError(self.error or "debugger operation refused")
        return self.result


class DebugTransport(Protocol):
    def request(self, request: DebugRequest) -> DebugResponse: ...


def bind_discovery_relay_request(request: DebugRequest) -> DebugRequest:
    """Bind relay cache keys to bytes while retaining the logical mutation ID.

    Older mrun brokers cache by request ID before reaching the worker. A content
    and sequence bound wire ID prevents that cache from hiding changed commands
    or caching a sequence-refusal forever under a logical mutation identity.
    """
    if not request.operation.startswith("discovery."):
        return request
    if DISCOVERY_RELAY_FIELD in request.payload:
        raise DebuggerProtocolError("reserved discovery relay metadata cannot be supplied")
    binding = {"schema": DISCOVERY_RELAY_BINDING_SCHEMA,
               "logical_request_id": request.request_id,
               "command_digest": digest({"operation": request.operation, "payload": request.payload})}
    wire_id = "discovery-" + digest({"logical_request_id": request.request_id,
        "operation": request.operation, "payload": request.payload, "sequence": request.sequence})
    return DebugRequest(request.operation, {**request.payload, DISCOVERY_RELAY_FIELD: binding},
                        request_id=wire_id, sequence=request.sequence)


def unwrap_discovery_relay_request(request: DebugRequest) -> DebugRequest:
    """Verify optional relay metadata before worker retry/sequence dispatch."""
    if not request.operation.startswith("discovery.") or DISCOVERY_RELAY_FIELD not in request.payload:
        return request
    payload = dict(request.payload)
    binding = payload.pop(DISCOVERY_RELAY_FIELD)
    if not isinstance(binding, Mapping) or binding.get("schema") != DISCOVERY_RELAY_BINDING_SCHEMA:
        raise DebuggerProtocolError("invalid discovery relay binding")
    logical_id = _id(binding.get("logical_request_id"), "logical discovery request_id")
    logical = DebugRequest(request.operation, payload, request_id=logical_id, sequence=request.sequence)
    expected = bind_discovery_relay_request(logical)
    if request.request_id != expected.request_id or dict(binding) != expected.payload[DISCOVERY_RELAY_FIELD]:
        raise DebuggerProtocolError("discovery relay content/identity binding differs")
    return logical


class HttpDebugTransport:
    """Authenticated mrun job transport.

    The server endpoint is intentionally explicit: a 404 means the job was not
    launched with the Saturn debug capability, rather than silently degrading to
    log polling or arbitrary process inspection.
    """

    def __init__(self, endpoint: str, *, token: str | None = None, timeout: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, request: DebugRequest) -> DebugResponse:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["X-Mrun-Token"] = self.token
        body = _canonical(request.to_dict())
        http_request = Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(http_request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                raise DebuggerProtocolError(
                    "mrun job has no Saturn debug endpoint; launch it with debugger instrumentation"
                ) from exc
            raise DebuggerProtocolError(f"mrun debugger endpoint returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise DebuggerProtocolError(f"cannot reach mrun debugger endpoint: {exc}") from exc
        return response_from_dict(raw)


class MrunDebugWorkerPoller:
    """Worker-side bridge for the mrun Saturn debug broker.

    mrun owns authentication and lease routing; the worker owns model state and
    calls its :class:`~saturn.debugger_worker.DebugWorkerController` at safe
    points.  A long-running worker can call ``serve_once`` between boundaries.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        token: str | None = None,
        agent_headers: dict[str, str] | None = None,
        debug_credential: str | None = None,
        debug_credential_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if bool(debug_credential) != bool(debug_credential_id):
            raise DebuggerProtocolError(
                "debug credential and credential ID must be supplied together"
            )
        if debug_credential and (token or agent_headers):
            raise DebuggerProtocolError(
                "a scoped debugger credential cannot be combined with broad worker auth"
            )
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.agent_headers = dict(agent_headers or {})
        self.debug_credential = debug_credential
        self.debug_credential_id = debug_credential_id
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", **self.agent_headers}
        if self.debug_credential:
            headers["X-MRun-Debug-Credential"] = self.debug_credential
            headers["X-MRun-Debug-Credential-ID"] = str(self.debug_credential_id)
        if self.token:
            headers["X-Mrun-Token"] = self.token
        return headers

    def register(self, capability: DebugCapability) -> dict[str, Any]:
        body = _canonical({"capability": capability.to_dict()})
        request = Request(
            self.endpoint + "/register",
            data=body,
            headers={**self._headers(), "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DebuggerProtocolError(
                f"cannot register Saturn debugger with mrun: {exc}"
            ) from exc

    def serve_once(self, worker: Any) -> bool:
        return self._serve_once(worker, wait_s=0.0)

    def _serve_once(self, worker: Any, *, wait_s: float) -> bool:
        separator = "&" if "?" in self.endpoint else "?"
        poll_url = f"{self.endpoint}{separator}wait_s={float(wait_s):.3f}"
        poll = Request(poll_url, headers=self._headers(), method="GET")
        try:
            with urlopen(poll, timeout=self.timeout) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise DebuggerProtocolError(f"cannot poll Saturn debugger from mrun: {exc}") from exc
        request_data = envelope.get("request")
        if not request_data:
            return False
        request = DebugRequest(
            operation=request_data["operation"],
            payload=dict(request_data.get("payload") or {}),
            request_id=request_data["request_id"],
            sequence=request_data.get("sequence"),
        )
        response = worker.handle(request)
        body = _canonical(response.to_dict())
        reply = Request(
            self.endpoint + "/respond",
            data=body,
            headers={**self._headers(), "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(reply, timeout=self.timeout):
                pass
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DebuggerProtocolError(f"cannot publish Saturn debugger response: {exc}") from exc
        return True

    def serve_until_resumed(self, worker: Any, *, poll_s: float = 1.0) -> None:
        """Service commands while a worker is paused at a typed safe point."""
        while True:
            self._serve_once(worker, wait_s=poll_s)
            status = worker.status()
            if status.get("state") not in {"paused", "pause-pending"}:
                return


def response_from_dict(value: dict[str, Any]) -> DebugResponse:
    if value.get("schema") != DEBUGGER_RESPONSE_SCHEMA:
        raise DebuggerProtocolError("unsupported debugger response schema")
    return DebugResponse(
        request_id=_id(value.get("request_id"), "response request_id"),
        sequence=int(value.get("sequence", 0)),
        ok=bool(value.get("ok")),
        result=dict(value.get("result") or {}),
        error=value.get("error"),
    )


class InMemoryDebugTransport:
    """Deterministic worker transport used by mechanics tests and local adapters."""

    def __init__(self, worker: Any) -> None:
        self.worker = worker

    def request(self, request: DebugRequest) -> DebugResponse:
        return self.worker.handle(request)


def capability_from_dict(value: dict[str, Any]) -> DebugCapability:
    if value.get("schema") != DEBUGGER_CAPABILITY_SCHEMA:
        raise DebuggerProtocolError("unsupported debugger capability schema")
    return DebugCapability(
        job_id=_id(value.get("job_id"), "job_id"),
        session_id=_id(value.get("session_id"), "session_id"),
        family=_id(value.get("family"), "family"),
        model_identity=dict(value.get("model_identity") or {}),
        boundaries=tuple(value.get("boundaries") or ()),
        inspectable_slots=tuple(value.get("inspectable_slots") or ()),
        writable_slots=tuple(value.get("writable_slots") or ()),
        exact_replay=bool(value.get("exact_replay")),
        endpoint=value.get("endpoint"),
        state_schema_id=str(value.get("state_schema_id") or "unknown"),
        backend=str(value.get("backend") or "unknown"),
        consumer=str(value.get("consumer") or "native"),
        source_fingerprint=value.get("source_fingerprint"),
        boundary_catalog=tuple(value.get("boundary_catalog") or ()),
        address_specs=tuple(
            DebuggerAddressSpec.from_dict(item) for item in (value.get("address_specs") or ())
        ),
        structs=tuple(
            DebuggerStructSpec.from_dict(item) for item in (value.get("structs") or ())
        ),
        algebra_ops=tuple(str(item) for item in (value.get("algebra_ops") or ())),
        optional_operations=tuple(
            str(item) for item in (value.get("optional_operations") or ())
        ),
        checkpoint_requires_pause=bool(value.get("checkpoint_requires_pause", False)),
    )


def now_utc() -> float:
    return time.time()
