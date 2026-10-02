"""Compact prompt/program recording for Saturn compiler debug runs.

The default artifact is deliberately small and tensor-free.  Prompt text is
stored once in a content-addressed prompt table, logical programs are stored
once in a content-addressed program table, and executions join the two tables
by digest.  Exact prompt strings embedded inside a program are replaced by a
typed reference, so recording an ``OpenPromptCompilation`` does not retain the
same prompt three times.

Set ``SATURN_PROGRAM_RECORD_PATH`` to opt in at call sites using
:func:`record_program_if_enabled`.  Rehydration is explicit and reconstructs a
human/tool-facing document containing the prompt list, program list, and their
bindings.  No model is loaded and no family-local tensor payload is exported.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROGRAM_RECORD_SCHEMA = "saturn-ar-program-record-v1"
PROGRAM_RECORD_REHYDRATED_SCHEMA = "saturn-ar-program-record-rehydrated-v1"
PROGRAM_RECORD_ENV = "SATURN_PROGRAM_RECORD_PATH"
PROMPT_REF_KEY = "$saturn_prompt_ref"


class ProgramRecordError(ValueError):
    """A program recording is malformed, ambiguous, or has changed bytes."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProgramRecordError("program record must be canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _require_digest(value: Any, *, label: str) -> str:
    candidate = str(value)
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ProgramRecordError(f"{label} must be a lowercase SHA-256 digest")
    return candidate


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical(value).decode("utf-8"))
    except ProgramRecordError as exc:
        raise ProgramRecordError(f"{label} is not tensor-free canonical JSON") from exc


def _normalize_prompt(prompt: Any) -> str:
    if not isinstance(prompt, str) or not prompt:
        raise ProgramRecordError("recorded prompts must be non-empty strings")
    return prompt


@contextmanager
def _record_lock(path: Path):
    """Serialize the read/merge/replace transaction for one output artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path.with_name(f"{path.name}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _externalize_prompt_text(value: Any, prompt_ids: Mapping[str, str]) -> Any:
    """Replace exact prompt strings with typed refs and reject tag collisions."""

    if isinstance(value, str):
        digest = prompt_ids.get(value)
        return {PROMPT_REF_KEY: digest} if digest is not None else value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProgramRecordError("program values must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProgramRecordError("program object keys must be strings")
            if key == PROMPT_REF_KEY:
                raise ProgramRecordError(f"program uses reserved key {PROMPT_REF_KEY!r}")
            result[key] = _externalize_prompt_text(item, prompt_ids)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_externalize_prompt_text(item, prompt_ids) for item in value]
    raise ProgramRecordError(
        f"program contains unsupported non-JSON value {type(value).__name__}"
    )


def _rehydrate_prompt_text(value: Any, prompts: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {PROMPT_REF_KEY}:
            digest = _require_digest(value[PROMPT_REF_KEY], label="prompt reference")
            try:
                return prompts[digest]
            except KeyError as exc:
                raise ProgramRecordError("program references an unknown prompt") from exc
        if PROMPT_REF_KEY in value:
            raise ProgramRecordError("prompt reference must be a single-key object")
        return {str(key): _rehydrate_prompt_text(item, prompts) for key, item in value.items()}
    if isinstance(value, list):
        return [_rehydrate_prompt_text(item, prompts) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ProgramBinding:
    """A deduplicated prompt-to-program observation."""

    prompt_sha256s: tuple[str, ...]
    program_sha256: str
    label: str
    call_count: int = 1

    def __post_init__(self) -> None:
        if not self.prompt_sha256s or len(set(self.prompt_sha256s)) != len(
            self.prompt_sha256s
        ):
            raise ProgramRecordError("binding prompts must be a non-empty unique sequence")
        for digest in self.prompt_sha256s:
            _require_digest(digest, label="binding prompt hash")
        _require_digest(self.program_sha256, label="binding program hash")
        if not isinstance(self.label, str) or not self.label:
            raise ProgramRecordError("binding label must be a non-empty string")
        if (
            isinstance(self.call_count, bool)
            or not isinstance(self.call_count, int)
            or self.call_count < 1
        ):
            raise ProgramRecordError("binding call_count must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_sha256s": list(self.prompt_sha256s),
            "program_sha256": self.program_sha256,
            "label": self.label,
            "call_count": self.call_count,
        }


@dataclass(slots=True)
class ProgramRecorder:
    """In-memory recorder with atomic compact-manifest persistence."""

    output_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _prompts: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _programs: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _bindings: dict[tuple[tuple[str, ...], str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.output_path is not None:
            self.output_path = Path(self.output_path)
        self.metadata = _json_copy(dict(self.metadata), label="recording metadata")

    @classmethod
    def open(cls, path: str | Path) -> ProgramRecorder:
        """Open a compact artifact for additive recording."""

        payload = read_program_record(path)
        recorder = cls(Path(path), metadata=payload["metadata"])
        recorder._load(payload)
        return recorder

    @classmethod
    def from_environment(cls) -> ProgramRecorder | None:
        path = os.environ.get(PROGRAM_RECORD_ENV)
        if not path:
            return None
        candidate = Path(path)
        return cls.open(candidate) if candidate.exists() else cls(candidate)

    def _load(self, payload: Mapping[str, Any]) -> None:
        for row in payload["prompts"]:
            self._prompts[str(row["sha256"])] = str(row["text"])
        for row in payload["programs"]:
            self._programs[str(row["sha256"])] = _json_copy(
                row["program"], label="recorded program"
            )
        for row in payload["bindings"]:
            key = (
                tuple(str(value) for value in row["prompt_sha256s"]),
                str(row["program_sha256"]),
                str(row["label"]),
            )
            self._bindings[key] = int(row["call_count"])

    def add(
        self,
        *,
        prompts: str | Sequence[str],
        program: Mapping[str, Any],
        label: str = "logical-program",
    ) -> tuple[tuple[str, ...], str]:
        """Record one association, retaining each unique prompt/program once."""

        prompt_values = (prompts,) if isinstance(prompts, str) else tuple(prompts)
        prompt_texts = tuple(dict.fromkeys(_normalize_prompt(value) for value in prompt_values))
        if not prompt_texts:
            raise ProgramRecordError("recorded prompt list cannot be empty")
        if not isinstance(program, Mapping):
            raise ProgramRecordError("recorded program must be a mapping")
        binding_label = str(label).strip()
        if not binding_label:
            raise ProgramRecordError("recorded program label cannot be empty")
        prompt_sha256s: list[str] = []
        for prompt_text in prompt_texts:
            prompt_sha256 = _prompt_digest(prompt_text)
            prior_prompt = self._prompts.setdefault(prompt_sha256, prompt_text)
            if prior_prompt != prompt_text:
                raise ProgramRecordError("prompt SHA-256 collision")
            prompt_sha256s.append(prompt_sha256)
        externalized = _externalize_prompt_text(
            _json_copy(dict(program), label="recorded program"),
            dict(zip(prompt_texts, prompt_sha256s, strict=True)),
        )
        program_sha256 = _digest(externalized)
        prior_program = self._programs.setdefault(program_sha256, externalized)
        if prior_program != externalized:
            raise ProgramRecordError("program SHA-256 collision")
        prompt_identity = tuple(prompt_sha256s)
        key = (prompt_identity, program_sha256, binding_label)
        self._bindings[key] = self._bindings.get(key, 0) + 1
        return prompt_identity, program_sha256

    def manifest(self) -> dict[str, Any]:
        prompts = [
            {"sha256": digest, "text": self._prompts[digest]}
            for digest in sorted(self._prompts)
        ]
        programs = [
            {
                "sha256": digest,
                "schema": self._programs[digest].get("schema"),
                "declared_content_sha256": self._programs[digest].get("content_sha256"),
                "program": self._programs[digest],
            }
            for digest in sorted(self._programs)
        ]
        bindings = [
            ProgramBinding(*key, call_count=count).to_dict()
            for key, count in sorted(self._bindings.items())
        ]
        body = {
            "schema": PROGRAM_RECORD_SCHEMA,
            "prompts": prompts,
            "programs": programs,
            "bindings": bindings,
            "metadata": dict(self.metadata),
            "statistics": {
                "unique_prompts": len(prompts),
                "unique_programs": len(programs),
                "unique_bindings": len(bindings),
                "recorded_calls": sum(row["call_count"] for row in bindings),
            },
            "storage_contract": {
                "prompt_text": "stored-once-by-sha256",
                "program": "stored-once-by-canonical-json-sha256",
                "embedded_prompt_text": "externalized-to-typed-reference",
                "tensor_payloads": "not-recorded",
            },
            "authority": {
                "artifact_only": True,
                "execution_authorized": False,
                "semantic_equivalence": "not_assessed",
                "native_consumer_validation_required": True,
            },
        }
        return {**body, "content_sha256": _digest(body)}

    def write(self, path: str | Path | None = None) -> Path:
        destination = Path(path) if path is not None else self.output_path
        if destination is None:
            raise ProgramRecordError("program recorder has no output path")
        payload = self.manifest()
        validate_program_record(payload)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ) + "\n"
        from mdb.bench_publication import stage_write
        if stage_write(destination, encoded.encode("utf-8"), immutable=False):
            self.output_path = destination
            return destination
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.output_path = destination
        return destination


def validate_program_record(payload: Mapping[str, Any]) -> None:
    """Verify hashes, references, statistics, and the tensor-free compact ABI."""

    if not isinstance(payload, Mapping) or payload.get("schema") != PROGRAM_RECORD_SCHEMA:
        raise ProgramRecordError("unsupported program record schema")
    expected_keys = {
        "schema",
        "prompts",
        "programs",
        "bindings",
        "metadata",
        "statistics",
        "storage_contract",
        "authority",
        "content_sha256",
    }
    if set(payload) != expected_keys:
        raise ProgramRecordError("program record fields changed")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != _digest(body):
        raise ProgramRecordError("program record fingerprint changed")
    prompts_value = payload.get("prompts")
    programs_value = payload.get("programs")
    bindings_value = payload.get("bindings")
    if not all(
        isinstance(value, list)
        for value in (prompts_value, programs_value, bindings_value)
    ):
        raise ProgramRecordError("program record tables must be lists")

    prompts: dict[str, str] = {}
    for row in prompts_value:
        if not isinstance(row, Mapping) or set(row) != {"sha256", "text"}:
            raise ProgramRecordError("prompt record fields changed")
        text = _normalize_prompt(row["text"])
        digest = _require_digest(row["sha256"], label="prompt hash")
        if digest != _prompt_digest(text) or digest in prompts:
            raise ProgramRecordError("prompt table is duplicated or has changed bytes")
        prompts[digest] = text

    programs: dict[str, Any] = {}
    for row in programs_value:
        if not isinstance(row, Mapping) or set(row) != {
            "sha256",
            "schema",
            "declared_content_sha256",
            "program",
        }:
            raise ProgramRecordError("program table fields changed")
        digest = _require_digest(row["sha256"], label="program hash")
        program = _json_copy(row["program"], label="recorded program")
        if digest != _digest(program) or digest in programs:
            raise ProgramRecordError("program table is duplicated or has changed bytes")
        _rehydrate_prompt_text(program, prompts)
        programs[digest] = program

    binding_keys: set[tuple[str, str, str]] = set()
    total_calls = 0
    for row in bindings_value:
        if not isinstance(row, Mapping) or set(row) != {
            "prompt_sha256s",
            "program_sha256",
            "label",
            "call_count",
        }:
            raise ProgramRecordError("program binding fields changed")
        binding = ProgramBinding(
            prompt_sha256s=tuple(str(value) for value in row["prompt_sha256s"]),
            program_sha256=str(row["program_sha256"]),
            label=str(row["label"]),
            call_count=row["call_count"],
        )
        key = (binding.prompt_sha256s, binding.program_sha256, binding.label)
        if key in binding_keys:
            raise ProgramRecordError("program binding table is duplicated")
        if (
            any(digest not in prompts for digest in binding.prompt_sha256s)
            or binding.program_sha256 not in programs
        ):
            raise ProgramRecordError("program binding references an unknown object")
        binding_keys.add(key)
        total_calls += binding.call_count

    statistics = payload.get("statistics")
    if statistics != {
        "unique_prompts": len(prompts),
        "unique_programs": len(programs),
        "unique_bindings": len(binding_keys),
        "recorded_calls": total_calls,
    }:
        raise ProgramRecordError("program record statistics changed")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or authority.get("artifact_only") is not True:
        raise ProgramRecordError("program record granted executable authority")
    if authority.get("execution_authorized") is not False:
        raise ProgramRecordError("program record granted executable authority")


def read_program_record(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProgramRecordError(f"cannot read program record: {candidate}") from exc
    if not isinstance(value, dict):
        raise ProgramRecordError("program record must be a JSON object")
    validate_program_record(value)
    return value


def rehydrate_program_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Expand prompt references while preserving deduplicated association tables."""

    validate_program_record(payload)
    prompts = {str(row["sha256"]): str(row["text"]) for row in payload["prompts"]}
    programs = {
        str(row["sha256"]): _rehydrate_prompt_text(row["program"], prompts)
        for row in payload["programs"]
    }
    return {
        "schema": PROGRAM_RECORD_REHYDRATED_SCHEMA,
        "record_content_sha256": payload["content_sha256"],
        "prompts": [
            {"sha256": digest, "text": prompts[digest]} for digest in sorted(prompts)
        ],
        "programs": [
            {"sha256": digest, "program": programs[digest]} for digest in sorted(programs)
        ],
        "bindings": [dict(row) for row in payload["bindings"]],
        "metadata": dict(payload["metadata"]),
        "authority": dict(payload["authority"]),
    }


def compact_rehydrated_program_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Import a full/rehydrated document back into the compact representation."""

    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != PROGRAM_RECORD_REHYDRATED_SCHEMA
    ):
        raise ProgramRecordError("unsupported rehydrated program record schema")
    prompt_rows = payload.get("prompts")
    program_rows = payload.get("programs")
    binding_rows = payload.get("bindings")
    if not all(isinstance(value, list) for value in (prompt_rows, program_rows, binding_rows)):
        raise ProgramRecordError("rehydrated program record tables must be lists")
    prompts = {
        _require_digest(row["sha256"], label="rehydrated prompt hash"): _normalize_prompt(
            row["text"]
        )
        for row in prompt_rows
        if isinstance(row, Mapping)
    }
    programs = {
        _require_digest(row["sha256"], label="rehydrated program hash"): row["program"]
        for row in program_rows
        if isinstance(row, Mapping)
    }
    if len(prompts) != len(prompt_rows) or len(programs) != len(program_rows):
        raise ProgramRecordError("rehydrated program record contains malformed rows")
    recorder = ProgramRecorder(metadata=payload.get("metadata") or {})
    for row in binding_rows:
        if not isinstance(row, Mapping):
            raise ProgramRecordError("rehydrated binding must be an object")
        prompt_sha_values = row.get("prompt_sha256s")
        if not isinstance(prompt_sha_values, list) or not prompt_sha_values:
            raise ProgramRecordError("rehydrated binding prompt list is invalid")
        prompt_shas = tuple(
            _require_digest(value, label="binding prompt hash")
            for value in prompt_sha_values
        )
        program_sha = _require_digest(row.get("program_sha256"), label="binding program hash")
        if any(value not in prompts for value in prompt_shas) or program_sha not in programs:
            raise ProgramRecordError("rehydrated binding references an unknown object")
        for prompt_sha in prompt_shas:
            if _prompt_digest(prompts[prompt_sha]) != prompt_sha:
                raise ProgramRecordError("rehydrated prompt bytes changed")
        calls = row.get("call_count")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
            raise ProgramRecordError("rehydrated binding call_count is invalid")
        for _ in range(calls):
            observed_prompt_shas, observed_program_sha = recorder.add(
                prompts=tuple(prompts[value] for value in prompt_shas),
                program=programs[program_sha],
                label=str(row.get("label") or ""),
            )
            if observed_prompt_shas != prompt_shas or observed_program_sha != program_sha:
                raise ProgramRecordError("rehydrated prompt identity changed during import")
    compact = recorder.manifest()
    declared = payload.get("record_content_sha256")
    if declared is not None and compact["content_sha256"] != declared:
        raise ProgramRecordError("rehydrated import does not reconstruct the compact artifact")
    return compact


def record_program_if_enabled(
    *,
    prompts: str | Sequence[str],
    program: Mapping[str, Any],
    label: str = "logical-program",
) -> dict[str, Any] | None:
    """Append to the env-selected compact artifact, or do nothing when disabled."""

    path = os.environ.get(PROGRAM_RECORD_ENV)
    if not path:
        return None
    destination = Path(path)
    with _record_lock(destination):
        recorder = (
            ProgramRecorder.open(destination)
            if destination.exists()
            else ProgramRecorder(destination)
        )
        recorder.add(prompts=prompts, program=program, label=label)
        recorder.write()
        return recorder.manifest()


def _write_json(path: Path | None, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path is None:
        print(encoded, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and rehydrate Saturn program records")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="verify one compact program record")
    validate.add_argument("record", type=Path)
    rehydrate = sub.add_parser("rehydrate", help="expand prompt references")
    rehydrate.add_argument("record", type=Path)
    rehydrate.add_argument("--output", type=Path)
    export = sub.add_parser("export", help="write one explicit full/rehydrated export")
    export.add_argument("record", type=Path)
    export.add_argument("output", type=Path)
    import_ = sub.add_parser("import", help="compact and verify a full export")
    import_.add_argument("export", type=Path)
    import_.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        payload = read_program_record(args.record)
        _write_json(
            None,
            {
                "schema": PROGRAM_RECORD_SCHEMA,
                "content_sha256": payload["content_sha256"],
                "statistics": payload["statistics"],
                "status": "valid",
            },
        )
        return 0
    if args.command in {"rehydrate", "export"}:
        expanded = rehydrate_program_record(read_program_record(args.record))
        output = args.output
        _write_json(output, expanded)
        return 0
    try:
        expanded = json.loads(args.export.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProgramRecordError(f"cannot read rehydrated export: {args.export}") from exc
    compact = compact_rehydrated_program_record(expanded)
    recorder = ProgramRecorder(args.output, metadata=compact["metadata"])
    recorder._load(compact)
    recorder.write()
    return 0


__all__ = [
    "PROGRAM_RECORD_ENV",
    "PROGRAM_RECORD_REHYDRATED_SCHEMA",
    "PROGRAM_RECORD_SCHEMA",
    "PROMPT_REF_KEY",
    "ProgramBinding",
    "ProgramRecordError",
    "ProgramRecorder",
    "compact_rehydrated_program_record",
    "main",
    "read_program_record",
    "record_program_if_enabled",
    "rehydrate_program_record",
    "validate_program_record",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the wrapper script
    raise SystemExit(main())
