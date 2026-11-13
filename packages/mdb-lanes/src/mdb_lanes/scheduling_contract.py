"""Fail-closed scheduling contracts for Saturn debugger workers.

``mrun`` is a job/admission control plane, not an interactive replay RPC.  A
Saturn request may therefore describe many local branches, but it must never
silently turn those branches into additional scheduler submissions.  This
module keeps that invariant at the request boundary so a malformed worker
request fails before it reaches the scheduler.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SINGLE_LEASE_CONTRACT_SCHEMA = "saturn-single-lease-scheduling-contract-v1"
RESTART_PROOF_LEASE_CONTRACT_SCHEMA = (
    "saturn-single-lease-restart-proof-scheduling-contract-v1"
)
SEQUENTIAL_MODELS_LEASE_CONTRACT_SCHEMA = (
    "saturn-single-lease-sequential-models-scheduling-contract-v1"
)
BOUNDED_DEBUGGER_LEASE_CONTRACT_SCHEMA = (
    "saturn-single-lease-bounded-debugger-scheduling-contract-v1"
)
FINITE_DEBUGGER_EXECUTION_MODEL = "finite_debugger_session"
BOUNDED_DEBUGGER_LEASE_SCOPE = "entire-bounded-debugger-worker"
ONE_LEASE_EXECUTION_SCHEMA = "saturn-one-lease-execution-v1"
FINITE_LOCAL_LEASE_SCOPES = frozenset(
    {
        "entire-batch-worker",
        "entire-session-batch-worker",
        "entire-worker",
    }
)
FINITE_LOCAL_EXECUTION_MODEL = "finite_local_program"
FINITE_RESTART_PROOF_EXECUTION_MODEL = "finite_sequential_restart_proof"
FINITE_SEQUENTIAL_MODELS_EXECUTION_MODEL = "finite_sequential_models"
RESTART_PROOF_LEASE_SCOPE = "entire-restart-proof-worker"
SEQUENTIAL_MODELS_LEASE_SCOPE = "entire-sequential-model-worker"
SEQUENTIAL_RECONSTRUCTION_POLICY = "sequential_subprocesses"
LEASE_LOCAL_ARTIFACT_REUSE_POLICY = "download_once_reuse_lease_local"
NO_SUBPROCESS_RETRY_POLICY = "none"
# These values make the scheduler/data-plane split explicit in every newly
# emitted request.  mrun owns admission and custody for the outer worker;
# checkpoint restores and suffix branches stay in that resident process.
OUTER_MRUN_CONTROL_PLANE = "mrun_outer_lease"
RESIDENT_WORKER_DATA_PLANE = "resident_worker"
LOCAL_CHECKPOINT_REPLAY_POLICY = "local_checkpoint_restore"
NO_LOCAL_CHECKPOINT_REPLAY_POLICY = "none_source_only"
# Saturn's finite workers are scientific specimens, not scheduler jobs that
# should be cloned after a ceiling kill.  Keep the policy in the request so a
# launcher cannot accidentally inherit mrun's general-purpose retry default.
NO_AUTOMATIC_RETRY_POLICY = "none_for_finite_lease"
NO_AUTOMATIC_RETRY_POLICY_PREFIX = "none_for_"
# A collector must not silently change the specimen whose receipt it returns.
# Retry-lineage traversal remains an explicit diagnostic escape hatch.
DIRECT_COLLECTION_POLICY = "direct_job_only"
# Operational guardrail: a larger panel should be split into deliberately
# separate finite jobs instead of turning one mrun lease into an unbounded
# resident worker. This is not a scientific sample-size limit.
MAX_LOCAL_WORK_ITEMS = 256


class SchedulingContractError(ValueError):
    """Raised when a Saturn request violates its declared lease contract."""


def is_no_automatic_retry_policy(value: Any) -> bool:
    """Return whether a declared policy disables mrun's automatic cloning."""

    return (
        isinstance(value, str)
        and value.startswith(NO_AUTOMATIC_RETRY_POLICY_PREFIX)
        and len(value) > len(NO_AUTOMATIC_RETRY_POLICY_PREFIX)
    )


def check_one_lease_execution(
    execution: Mapping[str, Any],
    *,
    prefix: str = "",
    local_count_field: str | None = None,
    expected_local_count: int | None = None,
) -> dict[str, bool]:
    """Return independent checks for one outer mrun execution.

    The scheduler is deliberately treated as a coarse control plane.  Local
    branch/suffix work may be numerous, but it must be represented by a
    counter in the worker receipt rather than by additional scheduler jobs.
    Verifiers use this helper so boundary, replay, and session reports cannot
    drift into subtly different lease definitions.
    """

    if not isinstance(execution, Mapping):
        raise SchedulingContractError("execution ledger must be an object")
    checks = {
        f"{prefix}one_scheduler_job": execution.get("scheduler_jobs") == 1,
        f"{prefix}one_mrun_submission": execution.get("mrun_submissions") == 1,
        f"{prefix}one_model_load": execution.get("model_loads") == 1,
        f"{prefix}no_additional_scheduler_submissions": (
            execution.get("additional_scheduler_submissions") == 0
        ),
    }
    if local_count_field is not None and expected_local_count is not None:
        checks[f"{prefix}local_count"] = (
            execution.get(local_count_field) == int(expected_local_count)
        )
    return checks


def validate_one_lease_execution(
    execution: Mapping[str, Any],
    *,
    local_count_field: str | None = None,
    expected_local_count: int | None = None,
) -> dict[str, bool]:
    """Fail closed when a worker receipt claims more than one mrun execution."""

    checks = check_one_lease_execution(
        execution,
        local_count_field=local_count_field,
        expected_local_count=expected_local_count,
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SchedulingContractError(
            "one-lease execution proof failed: " + ", ".join(failed)
        )
    return checks


def build_one_lease_execution(
    *,
    execution_model: str,
    details: Mapping[str, Any] | None = None,
    local_count_field: str | None = None,
    local_count: int | None = None,
) -> dict[str, Any]:
    """Build and validate the canonical flat one-lease receipt ledger.

    Workers may add diagnostic fields, but scheduler custody fields are owned
    here so a receipt cannot accidentally nest or omit them.  ``local_count``
    is deliberately separate from scheduler counters: it records work done by
    the already-loaded worker, not additional mrun submissions.
    """

    resolved_model = str(execution_model or "")
    if not resolved_model:
        raise SchedulingContractError("one-lease execution requires an execution_model")
    if (local_count_field is None) != (local_count is None):
        raise SchedulingContractError(
            "local_count_field and local_count must be supplied together"
        )
    ledger = dict(details or {})
    reserved = {
        "scheduler_jobs",
        "mrun_submissions",
        "model_loads",
        "additional_scheduler_submissions",
        "execution_model",
    }
    conflicts = sorted(reserved.intersection(ledger))
    if conflicts:
        raise SchedulingContractError(
            "one-lease scheduler fields are owned by the contract: "
            + ", ".join(conflicts)
        )
    if local_count_field is not None:
        if not str(local_count_field):
            raise SchedulingContractError("local_count_field must be non-empty")
        if local_count_field in reserved:
            raise SchedulingContractError(
                f"local_count_field cannot overwrite scheduler field {local_count_field!r}"
            )
        if local_count_field in ledger:
            raise SchedulingContractError(
                f"local count field is owned by the contract: {local_count_field}"
            )
        if isinstance(local_count, bool) or not isinstance(local_count, int) or local_count < 0:
            raise SchedulingContractError("local_count must be a non-negative integer")
        ledger[local_count_field] = int(local_count)
    ledger.update(
        {
            "scheduler_jobs": 1,
            "mrun_submissions": 1,
            "model_loads": 1,
            "additional_scheduler_submissions": 0,
            "execution_model": resolved_model,
        }
    )
    validate_one_lease_execution(
        ledger,
        local_count_field=local_count_field,
        expected_local_count=local_count,
    )
    return ledger


def validate_restart_proof_lease_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one finite lease containing sequential student reconstructions.

    A restart proof is intentionally not an ordinary one-load worker.  It may
    reconstruct the student in a declared number of fresh child processes,
    while preserving the resource property that at most one reconstruction is
    resident at a time.  The downloaded checkpoint remains in lease-local
    custody and is reopened by each process; no process may submit to mrun or
    retry itself.
    """

    if not isinstance(contract, Mapping):
        raise SchedulingContractError("restart-proof scheduling contract must be an object")
    row = dict(contract)
    expected_strings = {
        "schema": RESTART_PROOF_LEASE_CONTRACT_SCHEMA,
        "lease_scope": RESTART_PROOF_LEASE_SCOPE,
        "execution_model": FINITE_RESTART_PROOF_EXECUTION_MODEL,
        "task_family": "student-restart-proof",
        "control_plane": OUTER_MRUN_CONTROL_PLANE,
        "data_plane": RESIDENT_WORKER_DATA_PLANE,
        "replay_policy": LOCAL_CHECKPOINT_REPLAY_POLICY,
        "collection_policy": DIRECT_COLLECTION_POLICY,
        "student_process_execution": SEQUENTIAL_RECONSTRUCTION_POLICY,
        "checkpoint_artifact_policy": LEASE_LOCAL_ARTIFACT_REUSE_POLICY,
        "subprocess_retry_policy": NO_SUBPROCESS_RETRY_POLICY,
    }
    for field, expected in expected_strings.items():
        if row.get(field) != expected:
            raise SchedulingContractError(
                f"restart-proof {field} must be {expected!r}"
            )
    for field, expected in (
        ("max_mrun_submissions", 1),
        ("per_branch_mrun_submissions", 0),
        ("per_transition_mrun_submissions", 0),
        ("additional_scheduler_submissions", 0),
        ("nested_mrun_submissions", 0),
        ("resident_model_concurrency", 1),
        ("registered_checkpoint_downloads", 1),
    ):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise SchedulingContractError(
                f"restart-proof {field} must be the integer {expected}"
            )
    if row.get("interactive") is not False:
        raise SchedulingContractError("restart-proof workers must be finite, not interactive")
    if not is_no_automatic_retry_policy(row.get("mrun_retry_policy")):
        raise SchedulingContractError(
            "restart-proof workers must disable mrun automatic retries"
        )
    if row.get("retry_on_kill", False) is not False:
        raise SchedulingContractError("restart-proof retry_on_kill must remain false")
    if row.get("model_free", False) is not False:
        raise SchedulingContractError("restart-proof workers reconstruct a model")

    reconstructions = row.get("student_process_reconstructions")
    if (
        isinstance(reconstructions, bool)
        or not isinstance(reconstructions, int)
        or reconstructions < 2
    ):
        raise SchedulingContractError(
            "student_process_reconstructions must be an integer of at least two"
        )
    model_loads = row.get("model_loads")
    if (
        isinstance(model_loads, bool)
        or not isinstance(model_loads, int)
        or model_loads != reconstructions
    ):
        raise SchedulingContractError(
            "restart-proof model_loads must equal student_process_reconstructions"
        )
    max_local_items = row.get("max_local_work_items", MAX_LOCAL_WORK_ITEMS)
    if (
        isinstance(max_local_items, bool)
        or not isinstance(max_local_items, int)
        or max_local_items < 1
        or max_local_items > MAX_LOCAL_WORK_ITEMS
    ):
        raise SchedulingContractError(
            "restart-proof max_local_work_items must be within Saturn's finite cap"
        )
    if reconstructions > max_local_items:
        raise SchedulingContractError(
            "student_process_reconstructions exceeds the bounded local-work budget"
        )
    for field in (
        "logical_branches",
        "logical_operations",
        "logical_checkpoint_replays",
    ):
        if field not in row:
            continue
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SchedulingContractError(f"{field} must be a positive integer")
        if value > max_local_items:
            raise SchedulingContractError(
                f"{field} exceeds the bounded local-work budget ({max_local_items})"
            )
    checkpoint_policy = row.get("checkpoint_policy")
    if not isinstance(checkpoint_policy, str) or not checkpoint_policy.strip():
        raise SchedulingContractError(
            "restart-proof checkpoint_policy must describe the finite proof"
        )
    return row


def validate_sequential_models_lease_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one finite lease that loads heterogeneous models one at a time."""

    if not isinstance(contract, Mapping):
        raise SchedulingContractError("sequential-model scheduling contract must be an object")
    row = dict(contract)
    expected_strings = {
        "schema": SEQUENTIAL_MODELS_LEASE_CONTRACT_SCHEMA,
        "lease_scope": SEQUENTIAL_MODELS_LEASE_SCOPE,
        "execution_model": FINITE_SEQUENTIAL_MODELS_EXECUTION_MODEL,
        "control_plane": OUTER_MRUN_CONTROL_PLANE,
        "data_plane": RESIDENT_WORKER_DATA_PLANE,
        "collection_policy": DIRECT_COLLECTION_POLICY,
        "model_process_execution": SEQUENTIAL_RECONSTRUCTION_POLICY,
        "subprocess_retry_policy": NO_SUBPROCESS_RETRY_POLICY,
    }
    for field, expected in expected_strings.items():
        if row.get(field) != expected:
            raise SchedulingContractError(
                f"sequential-model {field} must be {expected!r}"
            )
    task_family = row.get("task_family")
    if not isinstance(task_family, str) or not task_family.strip():
        raise SchedulingContractError("sequential-model task_family must be non-empty")
    for field, expected in (
        ("max_mrun_submissions", 1),
        ("per_branch_mrun_submissions", 0),
        ("per_transition_mrun_submissions", 0),
        ("additional_scheduler_submissions", 0),
        ("nested_mrun_submissions", 0),
        ("resident_model_concurrency", 1),
    ):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise SchedulingContractError(
                f"sequential-model {field} must be the integer {expected}"
            )
    model_processes = row.get("model_processes")
    if (
        isinstance(model_processes, bool)
        or not isinstance(model_processes, int)
        or model_processes < 2
    ):
        raise SchedulingContractError("model_processes must be an integer of at least two")
    if row.get("model_loads") != model_processes:
        raise SchedulingContractError("model_loads must equal model_processes")
    max_local_items = row.get("max_local_work_items", MAX_LOCAL_WORK_ITEMS)
    if (
        isinstance(max_local_items, bool)
        or not isinstance(max_local_items, int)
        or not 1 <= max_local_items <= MAX_LOCAL_WORK_ITEMS
        or model_processes > max_local_items
    ):
        raise SchedulingContractError("sequential model count exceeds the finite work bound")
    if row.get("interactive") is not False:
        raise SchedulingContractError("sequential-model workers must be finite")
    if not is_no_automatic_retry_policy(row.get("mrun_retry_policy")):
        raise SchedulingContractError("sequential-model workers must disable automatic retries")
    if row.get("retry_on_kill", False) is not False:
        raise SchedulingContractError("sequential-model retry_on_kill must remain false")
    checkpoint_policy = row.get("checkpoint_policy")
    if not isinstance(checkpoint_policy, str) or not checkpoint_policy.strip():
        raise SchedulingContractError("sequential-model checkpoint_policy must be non-empty")
    return row


def validate_single_lease_contract(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate an explicitly declared one-lease contract.

    Requests without a contract remain compatible with older Saturn workers;
    requests that do declare one are checked before mrun submission.  The
    verifier intentionally checks only scheduler-facing fields.  The worker's
    receipt remains authoritative for the observed execution counts.
    """

    raw = request.get("scheduling_contract")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise SchedulingContractError("scheduling_contract must be an object")

    contract = dict(raw)
    if contract.get("schema") == BOUNDED_DEBUGGER_LEASE_CONTRACT_SCHEMA:
        return validate_bounded_debugger_lease_contract(contract)
    if contract.get("schema") == RESTART_PROOF_LEASE_CONTRACT_SCHEMA:
        return validate_restart_proof_lease_contract(contract)
    if contract.get("schema") == SEQUENTIAL_MODELS_LEASE_CONTRACT_SCHEMA:
        return validate_sequential_models_lease_contract(contract)
    lease_scope = str(contract.get("lease_scope") or "")
    if lease_scope in FINITE_LOCAL_LEASE_SCOPES:
        if contract.get("execution_model") != FINITE_LOCAL_EXECUTION_MODEL:
            raise SchedulingContractError(
                f"{lease_scope} must declare execution_model="
                f"{FINITE_LOCAL_EXECUTION_MODEL}"
            )
        if contract.get("interactive") is True:
            raise SchedulingContractError(
                f"{lease_scope} cannot declare interactive execution"
            )
    if contract.get("max_mrun_submissions") != 1:
        raise SchedulingContractError(
            "Saturn debugger requests must allow exactly one mrun submission"
        )
    for field in ("per_branch_mrun_submissions", "per_transition_mrun_submissions"):
        if field in contract and contract[field] != 0:
            raise SchedulingContractError(f"{field} must be zero inside one mrun lease")
    if contract.get("additional_scheduler_submissions", 0) != 0:
        raise SchedulingContractError(
            "additional_scheduler_submissions must be zero inside one mrun lease"
        )
    retry_policy = contract.get("mrun_retry_policy")
    if retry_policy is not None and not is_no_automatic_retry_policy(retry_policy):
        raise SchedulingContractError(
            "finite Saturn contracts must disable mrun automatic retries; "
            "use a mrun_retry_policy beginning with 'none_for_'"
        )
    collection_policy = contract.get("collection_policy")
    if collection_policy is not None and collection_policy != DIRECT_COLLECTION_POLICY:
        raise SchedulingContractError(
            f"collection_policy must be {DIRECT_COLLECTION_POLICY!r}"
        )
    for field, expected in (
        ("control_plane", OUTER_MRUN_CONTROL_PLANE),
        ("data_plane", RESIDENT_WORKER_DATA_PLANE),
    ):
        if field in contract and contract[field] != expected:
            raise SchedulingContractError(
                f"{field} must be {expected!r}; mrun is the outer control plane"
            )
    if "interactive" in contract and contract["interactive"] is not False:
        raise SchedulingContractError(
            "one-lease Saturn workers are finite local programs, not interactive mrun services"
        )
    if "branch_execution" in contract and contract["branch_execution"] != "in_process":
        raise SchedulingContractError("branch_execution must be in_process inside one mrun lease")
    replay_policy = contract.get("replay_policy")
    if replay_policy not in {
        None,
        LOCAL_CHECKPOINT_REPLAY_POLICY,
        NO_LOCAL_CHECKPOINT_REPLAY_POLICY,
    }:
        raise SchedulingContractError(
            "replay_policy must describe local checkpoint restore or an explicit "
            "source-only no-replay worker"
        )
    for field in ("logical_branches", "logical_operations"):
        if field not in contract:
            continue
        value = contract[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SchedulingContractError(f"{field} must be a positive integer")
    if "logical_checkpoint_replays" in contract:
        replay_count = contract["logical_checkpoint_replays"]
        if isinstance(replay_count, bool) or not isinstance(replay_count, int) or replay_count < 0:
            raise SchedulingContractError(
                "logical_checkpoint_replays must be a non-negative integer"
            )
        expected_policy = (
            NO_LOCAL_CHECKPOINT_REPLAY_POLICY
            if replay_count == 0
            else LOCAL_CHECKPOINT_REPLAY_POLICY
        )
        if replay_policy != expected_policy:
            raise SchedulingContractError(
                "replay_policy does not match logical_checkpoint_replays"
            )
    model_free = contract.get("model_free", False)
    if not isinstance(model_free, bool):
        raise SchedulingContractError("model_free must be boolean when declared")
    simultaneous_models = contract.get("simultaneous_models", False)
    if not isinstance(simultaneous_models, bool):
        raise SchedulingContractError("simultaneous_models must be boolean when declared")
    if model_free and simultaneous_models:
        raise SchedulingContractError("model-free work cannot declare simultaneous models")
    expected_model_loads = 0 if model_free else (2 if simultaneous_models else 1)
    model_loads = contract.get("model_loads")
    if (
        isinstance(model_loads, bool)
        or not isinstance(model_loads, int)
        or model_loads != expected_model_loads
    ):
        raise SchedulingContractError(
            "model-free analysis workers must load zero models; model-backed Saturn "
            "workers must load one model or exactly two explicitly simultaneous models"
        )
    expected_concurrency = 2 if simultaneous_models else (0 if model_free else 1)
    if contract.get("resident_model_concurrency", expected_concurrency) != expected_concurrency:
        raise SchedulingContractError("resident_model_concurrency disagrees with model loads")
    if model_free and contract.get("task_family") != "analysis":
        raise SchedulingContractError(
            "model_free=True is reserved for explicit task_family=analysis workers"
        )
    max_local_items = contract.get("max_local_work_items", MAX_LOCAL_WORK_ITEMS)
    if (
        isinstance(max_local_items, bool)
        or not isinstance(max_local_items, int)
        or max_local_items < 1
    ):
        raise SchedulingContractError(
            "max_local_work_items must be a positive integer"
        )
    if max_local_items > MAX_LOCAL_WORK_ITEMS:
        raise SchedulingContractError(
            "max_local_work_items exceeds Saturn's finite local-work cap"
        )
    for field in ("logical_branches", "logical_operations"):
        if field in contract and int(contract[field]) > max_local_items:
            raise SchedulingContractError(
                f"{field} exceeds the bounded local-work budget "
                f"({max_local_items})"
            )
    if contract.get("lease_scope") == "single-replay-worker" and contract.get(
        "operator_ack"
    ) not in {"single-shot", "dry-run-only"}:
        raise SchedulingContractError(
            "single replay submissions require an explicit single-shot acknowledgement"
        )
    return contract


def validate_bounded_debugger_lease_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """An explicit finite interactive lease, separate from legacy batch workers.

    Interactive commands remain local to one admitted model owner. Mandatory
    command, time and model-work ceilings make this a bounded debugger session,
    never an unbounded cross-job serving process.
    """
    if not isinstance(contract, Mapping):
        raise SchedulingContractError("bounded-debugger scheduling contract must be an object")
    row = dict(contract)
    for field, expected in {
        "schema": BOUNDED_DEBUGGER_LEASE_CONTRACT_SCHEMA,
        "lease_scope": BOUNDED_DEBUGGER_LEASE_SCOPE,
        "execution_model": FINITE_DEBUGGER_EXECUTION_MODEL,
        "task_family": "discovery-debugger",
        "control_plane": OUTER_MRUN_CONTROL_PLANE,
        "data_plane": RESIDENT_WORKER_DATA_PLANE,
        "collection_policy": DIRECT_COLLECTION_POLICY,
        "replay_policy": LOCAL_CHECKPOINT_REPLAY_POLICY,
        "branch_execution": "in_process",
    }.items():
        if row.get(field) != expected:
            raise SchedulingContractError(f"bounded-debugger {field} must be {expected!r}")
    if row.get("interactive") is not True:
        raise SchedulingContractError("bounded debugger must honestly declare interactive=True")
    for field, expected in (("max_mrun_submissions", 1), ("per_branch_mrun_submissions", 0),
                            ("per_transition_mrun_submissions", 0),
                            ("additional_scheduler_submissions", 0), ("nested_mrun_submissions", 0),
                            ("model_loads", 1), ("resident_model_concurrency", 1)):
        value = row.get(field)
        if type(value) is not int or value != expected:
            raise SchedulingContractError(f"bounded-debugger {field} must be the integer {expected}")
    for field in ("max_debug_commands", "max_wall_ms", "max_native_operations", "max_model_forwards",
                  "max_branches"):
        value = row.get(field)
        if type(value) is not int or value < 1:
            raise SchedulingContractError(f"bounded-debugger {field} must be a positive finite integer")
    if row["max_branches"] > MAX_LOCAL_WORK_ITEMS:
        raise SchedulingContractError("bounded-debugger max_branches exceeds finite local-work cap")
    if row.get("retry_on_kill") is not False:
        raise SchedulingContractError("bounded-debugger retry_on_kill must remain false")
    if row.get("model_free", False) is not False or row.get("simultaneous_models", False) is not False:
        raise SchedulingContractError("bounded debugger owns exactly one resident model")
    checkpoint_policy = row.get("checkpoint_policy")
    if not isinstance(checkpoint_policy, str) or not checkpoint_policy.strip():
        raise SchedulingContractError("bounded-debugger checkpoint_policy must be nonempty")
    if not is_no_automatic_retry_policy(row.get("mrun_retry_policy")):
        raise SchedulingContractError("bounded debugger must disable automatic retries")
    # Reuse the existing common scheduler/custody invariants through an internal
    # validation projection. The published row retains interactive=True and its
    # distinct schema; legacy contracts never acquire this permission.
    common = {**row, "schema": SINGLE_LEASE_CONTRACT_SCHEMA,
              "lease_scope": "entire-worker", "execution_model": FINITE_LOCAL_EXECUTION_MODEL,
              "interactive": False}
    validate_single_lease_contract({"scheduling_contract": common})
    return row


def require_single_lease_contract(request: Mapping[str, Any]) -> dict[str, Any]:
    """Require the canonical schema at a live Saturn submission boundary.

    ``validate_single_lease_contract`` intentionally accepts older receipts
    and hand-authored historical evidence.  A new mrun admission is stricter:
    it must carry the versioned contract emitted by
    ``saturn.one_lease_scheduling_contract``.
    """

    contract = validate_single_lease_contract(request)
    if contract is None:
        raise SchedulingContractError(
            "Saturn mrun submission requires scheduling_contract"
        )
    supported_schemas = {
        SINGLE_LEASE_CONTRACT_SCHEMA,
        RESTART_PROOF_LEASE_CONTRACT_SCHEMA,
        SEQUENTIAL_MODELS_LEASE_CONTRACT_SCHEMA,
        BOUNDED_DEBUGGER_LEASE_CONTRACT_SCHEMA,
    }
    if contract.get("schema") not in supported_schemas:
        raise SchedulingContractError(
            "Saturn mrun submission requires schema in: "
            + ", ".join(sorted(supported_schemas))
        )
    return contract


def require_finite_no_retry_contract(request: Mapping[str, Any]) -> dict[str, Any]:
    """Require the stricter live-launch policy for a finite Saturn worker.

    Historical receipts and offline compilers may use the compatibility
    validator above.  A new mrun admission must state both that the worker is
    finite and that mrun must not create a retry child for it.
    """

    contract = require_single_lease_contract(request)
    if not is_no_automatic_retry_policy(contract.get("mrun_retry_policy")):
        raise SchedulingContractError(
            "live Saturn submissions require mrun_retry_policy=none_for_*"
        )
    if contract.get("collection_policy") != DIRECT_COLLECTION_POLICY:
        raise SchedulingContractError(
            "live Saturn submissions require collection_policy=direct_job_only"
        )
    return contract


__all__ = [
    "SINGLE_LEASE_CONTRACT_SCHEMA",
    "RESTART_PROOF_LEASE_CONTRACT_SCHEMA",
    "SEQUENTIAL_MODELS_LEASE_CONTRACT_SCHEMA",
    "BOUNDED_DEBUGGER_LEASE_CONTRACT_SCHEMA",
    "FINITE_DEBUGGER_EXECUTION_MODEL",
    "BOUNDED_DEBUGGER_LEASE_SCOPE",
    "ONE_LEASE_EXECUTION_SCHEMA",
    "FINITE_LOCAL_LEASE_SCOPES",
    "FINITE_LOCAL_EXECUTION_MODEL",
    "FINITE_RESTART_PROOF_EXECUTION_MODEL",
    "FINITE_SEQUENTIAL_MODELS_EXECUTION_MODEL",
    "RESTART_PROOF_LEASE_SCOPE",
    "SEQUENTIAL_MODELS_LEASE_SCOPE",
    "SEQUENTIAL_RECONSTRUCTION_POLICY",
    "LEASE_LOCAL_ARTIFACT_REUSE_POLICY",
    "NO_SUBPROCESS_RETRY_POLICY",
    "OUTER_MRUN_CONTROL_PLANE",
    "RESIDENT_WORKER_DATA_PLANE",
    "LOCAL_CHECKPOINT_REPLAY_POLICY",
    "NO_AUTOMATIC_RETRY_POLICY",
    "NO_AUTOMATIC_RETRY_POLICY_PREFIX",
    "DIRECT_COLLECTION_POLICY",
    "MAX_LOCAL_WORK_ITEMS",
    "SchedulingContractError",
    "build_one_lease_execution",
    "check_one_lease_execution",
    "require_single_lease_contract",
    "require_finite_no_retry_contract",
    "is_no_automatic_retry_policy",
    "validate_single_lease_contract",
    "validate_restart_proof_lease_contract",
    "validate_sequential_models_lease_contract",
    "validate_bounded_debugger_lease_contract",
    "validate_one_lease_execution",
]
