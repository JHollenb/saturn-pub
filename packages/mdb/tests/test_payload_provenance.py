from pathlib import Path
from types import SimpleNamespace

import pytest

from mdb import payload


def test_runtime_wheel_refuses_workspace_shadow_of_installed_distribution(monkeypatch, tmp_path):
    expected = tmp_path/"installed/mrun/__init__.py"
    source = tmp_path/"workspace/mrun/__init__.py"
    distribution = SimpleNamespace(files=[Path("mrun/__init__.py")], locate_file=lambda entry: expected)
    monkeypatch.setattr(payload.importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(payload.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(source)))
    with pytest.raises(RuntimeError, match="shadowed"):
        payload._require_distribution_entry("mrun", "mrun-pub")
    monkeypatch.setattr(payload.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(expected)))
    payload._require_distribution_entry("mrun", "mrun-pub")


def test_declared_editable_distribution_keeps_its_resolution(monkeypatch):
    distribution = SimpleNamespace(files=[Path("_mdb_runtime.pth")])
    monkeypatch.setattr(payload.importlib.metadata, "distribution", lambda name: distribution)
    payload._require_distribution_entry("mdb", "mdb-runtime")
