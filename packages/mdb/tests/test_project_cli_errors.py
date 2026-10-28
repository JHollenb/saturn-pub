"""Expected offline CLI refusals are readable and keep failed data untrusted."""

import sqlite3

import pytest
from test_saturn_pub_adapter import opened

from mdb.project import Project, main


def refused(capsys, arguments, message):
    assert main(arguments) == 1
    output = capsys.readouterr()
    assert not output.out
    assert output.err.startswith("mdb project: ")
    assert message in output.err
    assert "Traceback" not in output.err


def test_binding_and_slice_refusals_keep_stdout_empty(tmp_path, capsys):
    project, adapter, owner, debug = opened(tmp_path)
    try:
        debug.step()
        adapter.command(
            "project symbol-bind count --port value --context panel "
            '--context-slot value --claim "Supplied count"'
        )
        sliced = project.slice("experiment", symbol="count")
        cut = sliced["cut_id"]
        prefix = [str(project.root)]
        refused(
            capsys,
            prefix + ["slice", "--symbol", "count", "--context", "wrong"],
            "symbol context or model mismatch",
        )
        refused(capsys, prefix + ["slice-show", "absent"], "slice is not retained")
        refused(capsys, prefix + ["list", "--session-id", "absent"], "unknown project session")
        debug.step()
        refused(capsys, prefix + ["slice", "--symbol", "count"], "symbol input context changed")
        selected = project.session("experiment")["selected"]
        with project._db() as db:
            db.execute("UPDATE slices SET body='{}' WHERE id=?", (sliced["slice_id"],))
        refused(capsys, prefix + ["slice-show", sliced["slice_id"]], "project slice seal changed")
        assert project.session("experiment")["selected"] == selected
        assert cut in project.cut_ids("experiment")
    finally:
        owner.close()


@pytest.mark.parametrize("broken", ["schema", "database", "cut"])
def test_project_open_and_storage_errors_are_readable(tmp_path, capsys, broken):
    project = Project(tmp_path / "broken")
    action = ["sessions"]
    if broken == "schema":
        with sqlite3.connect(project.root / "project.sqlite") as db:
            db.execute("PRAGMA user_version = 999")
        message = "unsupported MDB project schema"
    elif broken == "database":
        (project.root / "project.sqlite").write_bytes(b"not a SQLite database")
        message = "not a database"
    else:
        project, _, owner, debug = opened(tmp_path)
        cut = debug.checkpoint("retained")["cut_id"]
        owner.close()
        (project.store.root / "cuts" / f"{cut}.json").unlink()
        action = ["cut", cut]
        message = "No such file"
    refused(capsys, [str(project.root), *action], message)


def test_unexpected_programming_errors_still_propagate(tmp_path, monkeypatch):
    from mdb import project_navigation

    def broken(*args, **kwargs):
        raise RuntimeError("unexpected implementation failure")

    monkeypatch.setattr(project_navigation, "execute", broken)
    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        main([str(tmp_path / "project"), "list"])
