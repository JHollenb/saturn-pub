"""Package-independent mrun admission and direct collection contracts."""
from __future__ import annotations
import base64
import gzip
import hashlib
import importlib
import inspect
import json
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from .local_data_plane import reject_external_operation
from .tools import configure_workspace, scheduler_api
from .payload import stage_worker, worker_command
_TERMINAL_STATES = {"succeeded", "failed", "killed_ram", "killed_vram", "timeout", "cancelled", "lost"}
RESIDENT_WORKER_ENV = "SATURN_RESIDENT_WORKER"
from .scheduling_contract import DIRECT_COLLECTION_POLICY, FINITE_RESTART_PROOF_EXECUTION_MODEL, FINITE_SEQUENTIAL_MODELS_EXECUTION_MODEL, LEASE_LOCAL_ARTIFACT_REUSE_POLICY, LOCAL_CHECKPOINT_REPLAY_POLICY, MAX_LOCAL_WORK_ITEMS, NO_AUTOMATIC_RETRY_POLICY, NO_LOCAL_CHECKPOINT_REPLAY_POLICY, NO_SUBPROCESS_RETRY_POLICY, OUTER_MRUN_CONTROL_PLANE, RESIDENT_WORKER_DATA_PLANE, RESTART_PROOF_LEASE_CONTRACT_SCHEMA, RESTART_PROOF_LEASE_SCOPE, SEQUENTIAL_MODELS_LEASE_CONTRACT_SCHEMA, SEQUENTIAL_MODELS_LEASE_SCOPE, SEQUENTIAL_RECONSTRUCTION_POLICY, is_no_automatic_retry_policy, require_finite_no_retry_contract, require_single_lease_contract

def _reject_nested_submission() -> None:
    """Fail closed if a resident data-plane worker tries to submit to mrun."""
    reject_external_operation('mrun submission')
    if os.environ.get(RESIDENT_WORKER_ENV) == '1':
        raise RuntimeError('resident Saturn workers cannot submit nested mrun jobs; keep checkpoint branches inside the current worker lease')

def _mrun_launch() -> Any:
    """Return the newest mrun launcher, with a compatibility bridge for older clients.

    MDB stages installed mrun bytes into remote workers. The local
    control process may have an older installed client. Keep
    the intent-first launch contract here so a stale client cannot turn a
    valid dry-run into an import error or silently drop CUDA admission.
    """
    module = importlib.import_module('mrun.client.submit')
    launch = getattr(module, 'launch', None)
    if callable(launch):
        supported = inspect.signature(launch).parameters
        accepts_kwargs = any((parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in supported.values()))
        if 'guarded' not in supported and (not accepts_kwargs):
            raise RuntimeError('mrun client launcher lacks guarded submission support; refusing ordinary append-only Saturn submission')
        if 'needs' not in supported and (not accepts_kwargs):
            raise RuntimeError('mrun client launcher lacks explicit needs support; upgrade mrun instead of weakening admission')
        extensions = {'disk_gb', 'cpu_threads', 'est_wall_s'} - set(supported)
        if extensions and not accepts_kwargs:
            def reservation_compatibility_launch(command: list[str], **kwargs: Any) -> Any:
                active = {key: kwargs.get(key) for key in extensions if kwargs.get(key) is not None}
                if not active:
                    return launch(command, **{key: value for key, value in kwargs.items()
                                               if key not in extensions})
                submit = getattr(module, 'submit', None)
                if not callable(submit):
                    raise RuntimeError('mrun client cannot preserve declared resource reservations')
                parameters = inspect.signature(submit).parameters
                variadic = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
                required = {'reservation', 'guarded', 'needs', 'preflight', 'retry_on_kill', 'prefer_host'}
                if not variadic and not required.issubset(parameters):
                    raise RuntimeError('mrun submitter cannot preserve resource/admission controls')
                cpu, wall = kwargs.get('cpu_threads'), kwargs.get('est_wall_s')
                if cpu is not None and (isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 1):
                    raise ValueError('cpu_threads must be a positive integer')
                import math
                if wall is not None and (isinstance(wall, bool) or not isinstance(wall, (int, float))
                                         or not math.isfinite(wall) or wall <= 0):
                    raise ValueError('est_wall_s must be positive and finite')
                if (cpu is not None or wall is not None) and kwargs.get('ram_mb') is None:
                    raise ValueError('cpu_threads and est_wall_s require a declared ram_mb')
                reservation = {'ram_mb': kwargs.get('ram_mb'), 'vram_mb': kwargs.get('vram_mb') or 0.0}
                reservation.update({key: kwargs[key] for key in ('disk_gb', 'cpu_threads', 'est_wall_s')
                                    if kwargs.get(key) is not None})
                needs = dict(kwargs.get('needs') or {})
                if kwargs.get('gpu'):
                    needs['cuda'] = True
                translated = {key: kwargs.get(key) for key in ('experiment', 'config', 'payload',
                    'prefer_host', 'env_alias', 'model', 'task', 'timeout_s', 'priority', 'note',
                    'retry_on_kill', 'api', 'preflight', 'guarded', 'guarded_ttl_s') if key in kwargs}
                translated.update(cmd=command, host=kwargs.get('pin'), reservation=reservation, needs=needs)
                job_id = submit(**translated)
                return job_id if kwargs.get('detach', False) else module.attach(job_id, api=kwargs.get('api'))
            return reservation_compatibility_launch
        return launch
    raise RuntimeError('mrun client exposes no qualified launch() API; upgrade mrun instead of weakening admission')

def reject_staged_target_collisions(staging: Path, sources: Iterable[tuple[Path, Path]], *, reserved_targets: Iterable[Path]=()) -> tuple[tuple[Path, Path], ...]:
    """Validate and normalize source files before copying them into a stage.

    ``stage_worker`` historically derived the destination for a worker from
    ``Path.name``.  That is safe only while every source has a distinct
    basename.  Dependencies that intentionally share a basename must instead
    provide distinct relative targets through ``additional_files``.  Compare
    the resolved target paths here so equivalent spellings cannot silently
    overwrite one another.
    """
    root = Path(staging).resolve()
    seen: dict[Path, Path] = {}
    normalized: list[tuple[Path, Path]] = []

    def normalize_target(raw_target: Path) -> Path:
        target = Path(raw_target)
        if target.is_absolute():
            raise ValueError(f'staged source target must be relative: {target}')
        if target == Path('.') or not target.parts:
            raise ValueError('staged source target must name a file')
        if '..' in target.parts:
            raise ValueError(f'staged source target cannot escape the stage: {target}')
        resolved = (root / target).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f'staged source target cannot escape the stage: {target}') from exc
        return relative
    for (source, raw_target) in sources:
        source_path = Path(source)
        target = normalize_target(Path(raw_target))
        previous = seen.get(target)
        if previous is not None:
            raise ValueError(f'staged source target collision at {target}: {previous} and {source_path}')
        seen[target] = source_path
        normalized.append((source_path, target))
    for raw_target in reserved_targets:
        target = normalize_target(Path(raw_target))
        previous = seen.get(target)
        if previous is not None:
            raise ValueError(f'staged source target collides with a reserved runtime file at {target}: {previous}')
    return tuple(normalized)

def source_hash_manifest(sources: Mapping[str, Path] | Iterable[tuple[str | Path, Path]]) -> dict[str, str]:
    """Hash sources under explicit relative manifest targets.

    A mapping keyed by ``Path.name`` is lossy when two source paths share a
    basename.  Callers should provide the target names used by the staged
    ABI (for example ``projection_base.py`` and ``open_axis_base.py``).  The
    same target collision check used by :func:`stage_worker` is applied before
    any bytes are hashed, and iterable inputs additionally make duplicate
    entries observable instead of allowing a dict comprehension to erase one.
    """
    entries = sources.items() if isinstance(sources, Mapping) else sources
    staged = tuple(((Path(source), Path(target)) for (target, source) in entries))
    normalized = reject_staged_target_collisions(Path('.'), staged)
    return {target.as_posix(): hashlib.sha256(source.read_bytes()).hexdigest() for (source, target) in normalized}

def guarded_mrun_launch(command: list[str], *, request: Mapping[str, Any], experiment: str, task: str, model: str | None, payload: Path, mrun_config: Mapping[str, Any] | None=None, prefer_host: str | None=None, pin: str | None=None, ram_mb: float | None=None, vram_mb: float | None=None, disk_gb: float | None=None, cpu_threads: int | None=None, est_wall_s: float | None=None, gpu: bool=True, needs: Mapping[str, Any] | None=None, env_alias: str | None=None, timeout_s: float | None=None, priority: int=0, note: str | None=None, detach: bool=True, retry_on_kill: bool=False, preflight: bool=True, api: Any | None=None) -> str:
    """Submit a custom-staged worker through Saturn's guarded one-lease seam.

    Custom workers may need a staging layout that ``submit_worker`` does not
    know how to construct.  They still must use this helper so request
    validation, request fingerprinting, mrun's guarded/idempotent admission,
    and the explicit one-lease contract cannot drift between launchers.
    """
    _reject_nested_submission()
    if 'scheduling_contract' not in request:
        raise ValueError('guarded_mrun_launch requires request.scheduling_contract')
    require_finite_no_retry_contract(request)
    if retry_on_kill:
        raise ValueError('finite Saturn workers cannot enable retry_on_kill; admit a corrected specimen explicitly after diagnosing a failure')
    configure_workspace()
    launch_config = dict(mrun_config or request)
    declared_needs = dict(request.get('needs') or {})
    if needs is None:
        resolved_needs = declared_needs
    else:
        resolved_needs = dict(needs)
        if declared_needs and resolved_needs != declared_needs:
            raise ValueError(f'guarded_mrun_launch needs drift from request.needs: request={declared_needs!r}, explicit={resolved_needs!r}')
    expected_contract = request['scheduling_contract']
    if launch_config.get('scheduling_contract') != expected_contract:
        raise ValueError('guarded_mrun_launch requires mrun_config.scheduling_contract to match request')
    launch_config['saturn_request_sha256'] = _request_sha256(request)
    scheduler = api if api is not None else scheduler_api()
    launch = _mrun_launch()
    job_id = launch(command, experiment=experiment, task=task, model=model, payload=payload, prefer_host=prefer_host, pin=pin, ram_mb=ram_mb, vram_mb=vram_mb, disk_gb=disk_gb, cpu_threads=cpu_threads, est_wall_s=est_wall_s, gpu=gpu, needs=resolved_needs, config=launch_config, env_alias=env_alias, timeout_s=timeout_s, priority=priority, note=note, detach=detach, retry_on_kill=retry_on_kill, preflight=preflight, guarded=True, api=scheduler)
    return str(job_id)

def one_lease_scheduling_contract(*, lease_scope: str='entire-worker', execution_model: str='finite_local_program', checkpoint_policy: str='one outer Saturn worker lease; all logical branches and analysis remain in process', logical_branches: int | None=None, logical_operations: int | None=None, logical_checkpoint_replays: int | None=None, model_loads: int=1, model_free: bool=False, simultaneous_models: bool=False, max_local_work_items: int=MAX_LOCAL_WORK_ITEMS, conditioner_policy: str | None=None, extra: Mapping[str, Any] | None=None) -> dict[str, Any]:
    """Build the canonical scheduler contract for one finite Saturn worker."""
    if not isinstance(model_free, bool):
        raise ValueError('model_free must be boolean')
    if not isinstance(simultaneous_models, bool):
        raise ValueError('simultaneous_models must be boolean')
    if isinstance(model_loads, bool) or not isinstance(model_loads, int):
        raise ValueError('model_loads must be an integer count')
    if model_free and (model_loads != 0 or simultaneous_models):
        raise ValueError('model-free analysis requires model_loads=0 without simultaneous models')
    if not model_free and model_loads != (2 if simultaneous_models else 1):
        raise ValueError('model-backed workers require one model load, or exactly two explicitly simultaneous models')
    contract: dict[str, Any] = {'schema': 'saturn-single-lease-scheduling-contract-v1', 'lease_scope': lease_scope, 'execution_model': execution_model, 'max_mrun_submissions': 1, 'per_branch_mrun_submissions': 0, 'per_transition_mrun_submissions': 0, 'model_loads': int(model_loads), **({'simultaneous_models': True, 'resident_model_concurrency': 2} if simultaneous_models else {}), 'max_local_work_items': int(max_local_work_items), 'additional_scheduler_submissions': 0, 'branch_execution': 'in_process', 'control_plane': OUTER_MRUN_CONTROL_PLANE, 'data_plane': RESIDENT_WORKER_DATA_PLANE, 'replay_policy': NO_LOCAL_CHECKPOINT_REPLAY_POLICY if logical_checkpoint_replays == 0 else LOCAL_CHECKPOINT_REPLAY_POLICY, 'mrun_retry_policy': NO_AUTOMATIC_RETRY_POLICY, 'collection_policy': DIRECT_COLLECTION_POLICY, 'interactive': False, 'checkpoint_policy': checkpoint_policy}
    if model_free:
        contract['model_free'] = True
    if logical_branches is not None:
        contract['logical_branches'] = int(logical_branches)
    if logical_operations is not None:
        contract['logical_operations'] = int(logical_operations)
    if logical_checkpoint_replays is not None:
        contract['logical_checkpoint_replays'] = int(logical_checkpoint_replays)
    if conditioner_policy is not None:
        contract['conditioner_policy'] = conditioner_policy
    if extra:
        conflicts = sorted((str(key) for (key, value) in extra.items() if str(key) in contract and contract[str(key)] != value and (not (str(key) == 'mrun_retry_policy' and is_no_automatic_retry_policy(contract[str(key)]) and is_no_automatic_retry_policy(value)))))
        if conflicts:
            raise ValueError('one-lease scheduling extras cannot override canonical fields: ' + ', '.join(conflicts))
        for (key, value) in extra.items():
            normalized_key = str(key)
            if normalized_key not in contract or normalized_key == 'mrun_retry_policy':
                contract[normalized_key] = value
    require_single_lease_contract({'scheduling_contract': contract})
    return contract

def one_lease_restart_proof_scheduling_contract(*, student_process_reconstructions: int, checkpoint_policy: str='one finite CUDA lease; acquire the registered checkpoint once, then reconstruct one student at a time in sequential child processes', logical_branches: int | None=None, logical_operations: int | None=None, logical_checkpoint_replays: int | None=None, max_local_work_items: int=MAX_LOCAL_WORK_ITEMS, extra: Mapping[str, Any] | None=None) -> dict[str, Any]:
    """Build an honest one-lease contract for a cross-process restart proof.

    ``model_loads`` is the total number of student reconstructions, not peak
    residency.  Peak residency is separately fixed at one, and every child
    reopens the single checkpoint acquired into lease-local custody.  This API
    is deliberately separate from ``one_lease_scheduling_contract`` so an
    ordinary model-backed worker remains a strict one-load program.
    """
    contract: dict[str, Any] = {'schema': RESTART_PROOF_LEASE_CONTRACT_SCHEMA, 'lease_scope': RESTART_PROOF_LEASE_SCOPE, 'execution_model': FINITE_RESTART_PROOF_EXECUTION_MODEL, 'task_family': 'student-restart-proof', 'max_mrun_submissions': 1, 'per_branch_mrun_submissions': 0, 'per_transition_mrun_submissions': 0, 'additional_scheduler_submissions': 0, 'nested_mrun_submissions': 0, 'student_process_reconstructions': student_process_reconstructions, 'model_loads': student_process_reconstructions, 'resident_model_concurrency': 1, 'registered_checkpoint_downloads': 1, 'student_process_execution': SEQUENTIAL_RECONSTRUCTION_POLICY, 'checkpoint_artifact_policy': LEASE_LOCAL_ARTIFACT_REUSE_POLICY, 'subprocess_retry_policy': NO_SUBPROCESS_RETRY_POLICY, 'max_local_work_items': max_local_work_items, 'control_plane': OUTER_MRUN_CONTROL_PLANE, 'data_plane': RESIDENT_WORKER_DATA_PLANE, 'replay_policy': LOCAL_CHECKPOINT_REPLAY_POLICY, 'mrun_retry_policy': NO_AUTOMATIC_RETRY_POLICY, 'collection_policy': DIRECT_COLLECTION_POLICY, 'interactive': False, 'checkpoint_policy': checkpoint_policy}
    for (field, value) in (('logical_branches', logical_branches), ('logical_operations', logical_operations), ('logical_checkpoint_replays', logical_checkpoint_replays)):
        if value is not None:
            contract[field] = value
    if extra:
        conflicts = sorted((str(key) for (key, value) in extra.items() if str(key) in contract and contract[str(key)] != value and (not (str(key) == 'mrun_retry_policy' and is_no_automatic_retry_policy(contract[str(key)]) and is_no_automatic_retry_policy(value)))))
        if conflicts:
            raise ValueError('restart-proof scheduling extras cannot override canonical fields: ' + ', '.join(conflicts))
        for (key, value) in extra.items():
            normalized_key = str(key)
            if normalized_key not in contract or normalized_key == 'mrun_retry_policy':
                contract[normalized_key] = value
    require_finite_no_retry_contract({'scheduling_contract': contract})
    return contract

def one_lease_sequential_models_scheduling_contract(*, model_processes: int, task_family: str, checkpoint_policy: str, max_local_work_items: int=MAX_LOCAL_WORK_ITEMS, extra: Mapping[str, Any] | None=None) -> dict[str, Any]:
    """Build a finite one-lease contract for heterogeneous sequential model loads."""
    contract: dict[str, Any] = {'schema': SEQUENTIAL_MODELS_LEASE_CONTRACT_SCHEMA, 'lease_scope': SEQUENTIAL_MODELS_LEASE_SCOPE, 'execution_model': FINITE_SEQUENTIAL_MODELS_EXECUTION_MODEL, 'task_family': str(task_family), 'max_mrun_submissions': 1, 'per_branch_mrun_submissions': 0, 'per_transition_mrun_submissions': 0, 'additional_scheduler_submissions': 0, 'nested_mrun_submissions': 0, 'model_processes': int(model_processes), 'model_loads': int(model_processes), 'resident_model_concurrency': 1, 'model_process_execution': SEQUENTIAL_RECONSTRUCTION_POLICY, 'subprocess_retry_policy': NO_SUBPROCESS_RETRY_POLICY, 'max_local_work_items': int(max_local_work_items), 'control_plane': OUTER_MRUN_CONTROL_PLANE, 'data_plane': RESIDENT_WORKER_DATA_PLANE, 'mrun_retry_policy': NO_AUTOMATIC_RETRY_POLICY, 'collection_policy': DIRECT_COLLECTION_POLICY, 'interactive': False, 'checkpoint_policy': str(checkpoint_policy)}
    if extra:
        conflicts = sorted((str(key) for (key, value) in extra.items() if str(key) in contract and contract[str(key)] != value))
        if conflicts:
            raise ValueError('sequential-model scheduling extras cannot override canonical fields: ' + ', '.join(conflicts))
        contract.update({str(key): value for (key, value) in extra.items()})
    require_finite_no_retry_contract({'scheduling_contract': contract})
    return contract

def _request_sha256(request: Mapping[str, Any]) -> str:
    """Return the stable Saturn request marker carried into mrun config."""
    encoded = json.dumps(dict(request), sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def request_sha256(request: Mapping[str, Any]) -> str:
    """Return the stable digest used to bind a Saturn request to mrun."""
    return _request_sha256(request)

def terminal_job(job_id: str, *, wait: bool=False, max_wait_s: float=7200.0, follow_retry_lineage: bool=False) -> tuple[str, dict[str, Any], str]:
    """Collect one mrun job ID and return its metadata + logs.

    Collection remains non-blocking by default.  Transactional callers set
    ``wait=True`` to poll the same scheduler job until it reaches a terminal
    state.  The default is deliberately direct: a retry child is a distinct
    scheduler specimen and must not replace the requested job in a scientific
    receipt.  ``follow_retry_lineage=True`` is an explicit diagnostic escape
    hatch for inspecting legacy jobs that predate the finite no-retry policy.
    """
    reject_external_operation('mrun job polling')
    api = scheduler_api()
    current = job_id
    started = time.monotonic()
    seen: set[str] = set()
    cycle_detected = False
    while True:
        if current in seen:
            cycle_detected = True
        else:
            seen.add(current)
        (status, body, _) = api.request('GET', f'/api/jobs/{current}', timeout_s=60.0)
        if status >= 400:
            raise RuntimeError(f'job read failed: HTTP {status}')
        payload = json.loads(body.decode('utf-8'))
        retry = (payload.get('meta') or {}).get('auto_retry_job_id')
        retry = None if retry is None else str(retry)
        if not follow_retry_lineage and retry and (retry != current):
            raise RuntimeError(f'mrun job {current} exposes auto_retry_job_id={retry}; direct Saturn collection refuses to substitute a retry child. Use an explicit corrected run or pass follow_retry_lineage=True for legacy diagnosis.')
        if not follow_retry_lineage and retry == current:
            raise RuntimeError(f'mrun job {current} exposes a cyclic auto-retry hint; direct Saturn collection refuses to follow it')
        if follow_retry_lineage and (not cycle_detected) and retry and (retry != current) and (retry not in seen):
            current = retry
            continue
        if follow_retry_lineage and retry and (retry == current or retry in seen):
            cycle_detected = True
        if wait and str(payload.get('state')) not in _TERMINAL_STATES:
            if time.monotonic() - started >= float(max_wait_s):
                raise TimeoutError(f'timed out waiting for mrun job {current}')
            time.sleep(1.0)
            continue
        (log_status, log_body, _) = api.request('GET', f'/api/jobs/{current}/logs?offset=0&wait_s=1', timeout_s=60.0)
        if log_status >= 400:
            raise RuntimeError(f'log read failed: HTTP {log_status}')
        return (current, payload, log_body.decode('utf-8', errors='replace'))

def guarded_job_receipt(job_id: str) -> dict[str, Any]:
    """Read mrun's immutable guarded admission/custody receipt."""
    reject_external_operation('mrun job receipt access')
    api = scheduler_api()
    (status, body, _) = api.request('GET', f'/api/jobs/{str(job_id)}/receipt', timeout_s=60.0)
    if status >= 400:
        raise RuntimeError(f'guarded job receipt failed: HTTP {status}')
    payload = json.loads(body.decode('utf-8'))
    if not isinstance(payload, dict):
        raise RuntimeError('guarded job receipt must be an object')
    return payload

def decode_marker(log_text: str, marker: str) -> dict[str, Any] | None:
    for line in reversed(log_text.splitlines()):
        if not line.startswith(marker):
            continue
        try:
            return json.loads(gzip.decompress(base64.b64decode(line[len(marker):].strip())))
        except Exception as exc:
            return {'status': 'error', 'error': f'marker_decode:{type(exc).__name__}: {exc}'}
    return None
