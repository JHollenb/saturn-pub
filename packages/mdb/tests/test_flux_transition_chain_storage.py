"""Declared full-weight members must fit before admitting another model run."""
import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "qualification"


def test_full_fused_weight_storage_rejects_inherited_member_bound_without_allocation():
    plan = runpy.run_path(str(HERE / "submit_flux_transition_chain.py"))["storage_plan"]
    with pytest.raises(ValueError, match="transport member bound"):
        plan({"transport_limits": {"max_expanded_bytes": 3072 << 20}})
    value = plan({"transport_limits": {"max_member_bytes": 256 << 20, "max_expanded_bytes": 3072 << 20}})
    assert value["members"]["fused_weight"]["bytes"] == 169869440
    assert value["members"]["fused_weight"]["header_bytes"] == 128
    assert value["two_block_parameter_bytes_lower_bound"] == 2 * (169869440 + 75497600)


def test_parameter_expanded_lower_bound_is_an_integrity_check():
    plan = runpy.run_path(str(HERE / "submit_flux_transition_chain.py"))["storage_plan"]
    with pytest.raises(ValueError, match="expanded transport bound"):
        plan({"transport_limits": {"max_member_bytes": 256 << 20, "max_expanded_bytes": 400 << 20}})


def test_actual_member_inventory_keeps_bounded_enforcement(tmp_path):
    custody = runpy.run_path(str(HERE / "flux_family_custody.py"))
    (tmp_path / "test.bin").write_bytes(b"123456789")
    with pytest.raises(custody["CustodyError"], match="member exceeds"):
        custody["_inventory"](tmp_path, custody["_limits"]({"max_member_bytes": 8}))
    inventory = custody["_inventory"](tmp_path, custody["_limits"]({"max_member_bytes": 16}))
    assert inventory["test.bin"]["bytes"] == 9
