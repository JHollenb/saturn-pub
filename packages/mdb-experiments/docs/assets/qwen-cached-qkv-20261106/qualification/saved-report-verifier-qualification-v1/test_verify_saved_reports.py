"""Compact tiny JSON only; actual installed CLI, no tensors or scheduler."""
import copy
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
V = runpy.run_path(str(HERE / "mdb-qwen-cached-qkv-20261106/verify_saved_reports.py"))
SAVED = HERE / "saved-report"


def fixture_report():
    report = V["read"](SAVED / "audit.json")
    report["layout_restoration"] = V["read"](HERE / "layout-restoration.json")
    return report, V["read"](HERE / "protocol.json")


def test_actual_saved_scalar_channels_variable_windows_and_separate_QKV_baseline():
    report, protocol = fixture_report()
    result = V["rollup"](report, protocol)
    for channel in V["CHANNELS"]:
        assert result["channels"][channel]["endpoint"]["whole"]["fields"] == 8
        assert result["channels"][channel]["response_error"]["whole"]["fields"] == 8
        assert result["channels"][channel]["endpoint"]["windows"]["fields"] == 24
        assert result["positioned_QKV"][channel]["whole"]["fields"] == 6
    assert result["positive_baseline_primary"]["fields"] == 4 and result["positive_baseline_QKV"]["fields"] == 3
    assert result["layout"]["arrays"] == 13 and set(result["layout"]["changed_staging_to_restored_stride"]) == {"q", "k", "v"}
    broken = copy.deepcopy(report)
    arm = protocol["arms"][0]["id"]
    broken["arms"][arm]["channels"][V["CHANNELS"][0]]["head"]["endpoint"]["whole"]["sample_count"] += 1
    with pytest.raises(ValueError, match="whole/protocol-window"):
        V["rollup"](broken, protocol)


def test_logical_bit_and_zero_counts_from_scalar_report_remain_distinct():
    value = dict(sample_count=10, bitwise_exact=False, changed_values=2, maxabs=.5, rms=.1, predicted_rms=1., native_rms=1., signed_mean=.02,
        signed_zero=dict(shared=4, defects=3))
    result = V["metrics"]([value])
    assert result["changed_values"] == 2 and result["signed_zero"]["defects"] == 3 and result["logical_bit_changed_elements"] == 5


def test_actual_cached_collector_metadata_schema_without_fabricated_reader_field():
    receipt = V["read"](HERE / "collection-receipt.json")
    assert "reader_sha256" not in receipt
    V["collection_binding"](receipt, receipt["native_job_id"], receipt["job_id"], dict(request_sha256=receipt["request_sha256"]))
    wrong = dict(receipt, collector_sha256="0"*64)
    with pytest.raises(ValueError, match="exact cached collector"):
        V["collection_binding"](wrong, receipt["native_job_id"], receipt["job_id"], dict(request_sha256=receipt["request_sha256"]))


def test_actual_installed_lookup_aliases_and_changed_source_profile(tmp_path):
    calls = V["H"]["cli_accounts"](SAVED, tmp_path)
    assert len(calls) == 12 and sum(row["matches"] for row in calls) == 4
    assert {row["api"] for row in calls} == {"equations", "atlas"}
    assert all(row["matches"] == (1 if row["contrast"] == "exact" else 0) for row in calls)


def test_reused_compact_inventory_rejects_changed_member_and_help_is_local(tmp_path):
    collection = tmp_path / "col"
    pack = collection / "extracted/binding-minimal-result"
    shutil.copytree(SAVED, pack)
    V["save"](pack / "files.json", {path.relative_to(pack).as_posix(): V["sha"](path) for path in pack.rglob("*") if path.is_file()})
    V["save"](collection / "collection-receipt.json", dict(files_manifest_sha256=V["sha"](pack / "files.json")))
    assert not V["H"]["H"]["compact_account"](collection, pack)["retained_array_members"]
    (pack / "audit.json").write_text("{}")
    with pytest.raises(ValueError, match="compact member differs"):
        V["H"]["H"]["compact_account"](collection, pack)
    result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "mdb-qwen-cached-qkv-20261106/verify_saved_reports.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--native-job" in result.stdout and "--expected-compact-manifest" in result.stdout
