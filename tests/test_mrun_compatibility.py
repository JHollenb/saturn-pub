from types import SimpleNamespace

import pytest

from mdb.job_support import _mrun_launch


def test_pinned_client_preserves_declared_reservation_and_admission(monkeypatch):
    calls = []
    def launch(command, *, guarded, needs, model, ram_mb=None, vram_mb=None):
        pytest.fail("old launch cannot carry the declared CPU/disk reservation")
    def submit(**kwargs):
        calls.append(kwargs)
        return "job-123456789abc"
    monkeypatch.setattr("mdb.job_support.importlib.import_module",
                        lambda name: SimpleNamespace(launch=launch, submit=submit))
    job = _mrun_launch()(["worker"], guarded=True, needs={"cuda": True}, gpu=True,
        model="mamba-130m", ram_mb=6144, vram_mb=1024, cpu_threads=4, disk_gb=2,
        est_wall_s=60, preflight=True, retry_on_kill=False, prefer_host="beast", detach=True)
    assert job == "job-123456789abc"
    assert calls[0]["reservation"] == {"ram_mb": 6144, "vram_mb": 1024,
        "cpu_threads": 4, "disk_gb": 2, "est_wall_s": 60}
    assert calls[0]["needs"] == {"cuda": True}
    assert calls[0]["guarded"] and calls[0]["preflight"]
    assert calls[0]["retry_on_kill"] is False
    assert calls[0]["prefer_host"] == "beast"


def test_pinned_client_drops_only_absent_extension_values(monkeypatch):
    calls = []
    def launch(command, *, guarded, needs):
        calls.append((command, guarded, needs))
        return "job-123456789abc"
    monkeypatch.setattr("mdb.job_support.importlib.import_module", lambda name: SimpleNamespace(launch=launch))
    assert _mrun_launch()(["worker"], guarded=True, needs={"cuda": True},
                          disk_gb=None, cpu_threads=None, est_wall_s=None) == "job-123456789abc"
    assert calls == [(["worker"], True, {"cuda": True})]


def test_pinned_client_rejects_missing_admission_control(monkeypatch):
    def launch(command, *, guarded, needs):
        pass
    def submit(*, reservation):
        pytest.fail("unsafe submit must never execute")
    monkeypatch.setattr("mdb.job_support.importlib.import_module",
                        lambda name: SimpleNamespace(launch=launch, submit=submit))
    with pytest.raises(RuntimeError, match="admission controls"):
        _mrun_launch()(["worker"], cpu_threads=4, ram_mb=6144, guarded=True)


def test_stale_launcher_without_needs_is_refused(monkeypatch):
    def launch(command, *, guarded, gpu=False):
        pytest.fail("stale launcher must never execute")
    monkeypatch.setattr("mdb.job_support.importlib.import_module",
                        lambda name: SimpleNamespace(launch=launch))
    with pytest.raises(RuntimeError, match="explicit needs support"):
        _mrun_launch()


def test_submit_only_stale_client_is_refused(monkeypatch):
    def submit(*, cmd, guarded, reservation=None):
        pytest.fail("unqualified submit-only fallback must never execute")
    monkeypatch.setattr("mdb.job_support.importlib.import_module",
                        lambda name: SimpleNamespace(submit=submit))
    with pytest.raises(RuntimeError, match="no qualified launch"):
        _mrun_launch()
