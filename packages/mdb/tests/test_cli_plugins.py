"""The engine CLI and package exports reach lane code only through entry points.

``mdb.core.plugins`` reads the ``mdb.lanes`` (verbs) and ``mdb.lanes.exports`` (attributes)
groups. In this source tree they come from the stub ``mdb_lanes-0.1.0.dist-info`` written
from ``mdb/lanes_compat.py``; after the split they come from ``packages/mdb-lanes``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import mdb
from mdb import cli
from mdb.core import plugins


def _run(code):
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=120, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


@pytest.fixture()
def no_plugin(monkeypatch):
    """The engine as installed without mdb-lanes: no entry point of either group."""
    cached = plugins.entries
    cached.cache_clear()
    monkeypatch.setattr(plugins, "entries", lambda group: {})
    for name in ("BenchSession", "Project"):
        mdb.__dict__.pop(name, None)
    yield
    cached.cache_clear()


@pytest.fixture()
def plugin():
    """The lane plug-in, present: skipped when the engine is tested without mdb-lanes."""
    plugins.clear()
    if not plugins.installed():
        pytest.skip("mdb-lanes is not on the path; plug-in dispatch has nothing to dispatch to")


def test_the_stub_distribution_matches_the_shim_tables():
    lanes_compat = pytest.importorskip("mdb_lanes.lanes_compat")
    stub = lanes_compat.stub_dir()
    for name, text in lanes_compat.stub_files().items():
        assert (stub / name).read_text() == text, f"stale {name}: python -m mdb.lanes_compat --write"
    assert set(plugins.entries(plugins.VERB_GROUP)) >= set(lanes_compat.VERBS)
    assert set(plugins.entries(plugins.EXPORT_GROUP)) >= set(lanes_compat.EXPORTS)


def test_engine_verbs_need_no_plugin(capsys, no_plugin):
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == mdb.__version__
    assert cli.main([]) == 0
    assert "usage: mdb" in capsys.readouterr().out


def test_a_lane_verb_without_the_plugin_exits_2_and_says_so(capsys, no_plugin):
    assert cli.main(["project", "/nowhere"]) == 2
    assert "mdb-lanes plug-in is not installed" in capsys.readouterr().err
    assert cli.main(["lanes"]) == 2
    assert "mdb lanes has no commands: the mdb-lanes plug-in is not installed" in \
        capsys.readouterr().err
    assert cli.main(["lanes", "bench"]) == 2


def test_a_lane_export_without_the_plugin_names_mdb_lanes(no_plugin):
    with pytest.raises(AttributeError, match="mdb-lanes plug-in"):
        mdb.BenchSession  # noqa: B018
    assert mdb.attach.__module__ == "mdb.ops.debugger"


def test_lanes_lists_and_runs_plugin_verbs(capsys, plugin):
    assert cli.main(["lanes"]) == 0
    listed = capsys.readouterr().out.split()
    assert {"project", "bench", "open", "debug", "capabilities", "help"} <= set(listed)
    assert cli.main(["lanes", "help"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() and not out.err


def test_a_bare_legacy_verb_falls_through_with_one_notice(capsys, plugin):
    assert cli.main(["help"]) == 0
    out = capsys.readouterr()
    assert out.err.strip().splitlines() == [
        "mdb: 'help' is a lanes command (mdb-lanes plug-in); running it as 'mdb lanes help'"]
    assert out.out.strip()


def test_an_unknown_verb_exits_2(capsys, plugin):
    assert cli.main(["no-such-verb"]) == 2
    assert "unknown verb" in capsys.readouterr().err


def test_lane_exports_resolve_through_the_entry_point_group(plugin):
    # mdb.project before the split, mdb_lanes.project after it: whatever the group declares.
    declared = plugins.entries(plugins.EXPORT_GROUP)["Project"].value.split(":")[0]
    assert _run("import mdb; print(mdb.Project.__module__, mdb.StateGraph.__name__)").split() \
        == [declared, "StateGraph"]


def test_serve_verb_delegates_without_loading_fastapi_at_import():
    out = _run("import sys; from mdb.cli import main; "
               "rc = main(['serve', '--project', '/nowhere', '--port', '0']); "
               "print(rc, 'fastapi' in sys.modules, 'uvicorn' in sys.modules)")
    assert out.split() == ["2", "False", "False"]


def test_importing_the_engine_cli_loads_no_lane_module_and_no_framework():
    engine = ("core", "engine", "analysis", "ops", "repl", "serve", "cli", "__main__")
    loaded = json.loads(_run(
        "import sys, json, mdb, mdb.cli, mdb.core; "
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('torch', 'numpy', 'fastapi', 'mdb_lanes') or (m.startswith('mdb.') and "
        f"m.split('.')[1] not in {engine!r}))))"))
    assert loaded == []
