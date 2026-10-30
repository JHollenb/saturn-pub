from __future__ import annotations

import pytest

from mdb.discovery_remote import bounded_debugger_scheduling_contract
from mdb.job_support import one_lease_scheduling_contract
from mdb.scheduling_contract import (
    BOUNDED_DEBUGGER_LEASE_CONTRACT_SCHEMA,
    SchedulingContractError,
    require_finite_no_retry_contract,
    require_single_lease_contract,
    validate_single_lease_contract,
)


def contract():
    return bounded_debugger_scheduling_contract(max_debug_commands=128,
        max_wall_ms=90000, max_native_operations=131072, max_model_forwards=8)


def test_bounded_interactive_schema_is_explicit_and_admissible_without_relaxing_legacy():
    row = contract()
    assert row["schema"] == BOUNDED_DEBUGGER_LEASE_CONTRACT_SCHEMA
    assert row["interactive"] is True and row["execution_model"] == "finite_debugger_session"
    assert require_single_lease_contract({"scheduling_contract": row}) == row
    assert require_finite_no_retry_contract({"scheduling_contract": row}) == row
    legacy = one_lease_scheduling_contract()
    assert legacy["interactive"] is False
    assert validate_single_lease_contract({"scheduling_contract": legacy}) == legacy
    legacy["interactive"] = True
    with pytest.raises(SchedulingContractError, match="interactive"):
        validate_single_lease_contract({"scheduling_contract": legacy})


@pytest.mark.parametrize("field", ["max_debug_commands", "max_wall_ms", "max_native_operations",
                                   "max_model_forwards", "max_branches"])
@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5, "10"])
def test_every_debugger_work_ceiling_is_required_finite_integer(field, value):
    row = contract()
    row[field] = value
    with pytest.raises(SchedulingContractError, match=field):
        require_finite_no_retry_contract({"scheduling_contract": row})
    del row[field]
    with pytest.raises(SchedulingContractError, match=field):
        require_finite_no_retry_contract({"scheduling_contract": row})


@pytest.mark.parametrize(("field", "value"), [
    ("interactive", False), ("model_loads", 2), ("model_loads", True),
    ("max_mrun_submissions", 2), ("max_mrun_submissions", True),
    ("per_transition_mrun_submissions", 1), ("per_branch_mrun_submissions", 1),
    ("nested_mrun_submissions", 1), ("additional_scheduler_submissions", 1),
    ("resident_model_concurrency", 2), ("mrun_retry_policy", "automatic"),
    ("retry_on_kill", True), ("collection_policy", "follow_retry_lineage"),
    ("branch_execution", "scheduler_job"), ("control_plane", "resident_worker"),
    ("data_plane", "mrun_outer_lease"), ("max_branches", 257)])
def test_bounded_debugger_cannot_expand_lease_ownership_or_hide_interactivity(field, value):
    row = contract()
    row[field] = value
    with pytest.raises(SchedulingContractError):
        require_finite_no_retry_contract({"scheduling_contract": row})


def test_helper_extra_metadata_cannot_overwrite_owned_ceilings():
    with pytest.raises(ValueError, match="owned"):
        bounded_debugger_scheduling_contract(max_debug_commands=128, max_wall_ms=90000,
            max_native_operations=131072, max_model_forwards=8, extra={"interactive": False})
