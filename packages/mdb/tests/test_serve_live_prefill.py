"""The live timeline starts at the run's root (M54): with ``prefill=0`` every prompt row is a
stepped row with its stops counted, not only the last. Needs the ``serve`` extra (run in the
common venv)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from mdb.ops import dispatch  # noqa: E402
from mdb.serve.live import LiveSession, open_debugger  # noqa: E402
from mdb.serve.views.live import timeline  # noqa: E402


def test_the_timeline_root_is_the_first_stepped_row(tmp_path):
    session = LiveSession(open_debugger(tmp_path / "p", tiny="qwen2"))
    try:
        dispatch(session.ctx, "start", {"prompt": [1, 5, 9, 3], "prefill": 0})
        dispatch(session.ctx, "until", {"target": "t2.L1.h"})
        view, _ = timeline(session)
        stops = len(view.stops)
        assert (view.root.row, view.root.index) == (0, 0)
        assert [row.count for row in view.rows] == [stops, stops, view.cursor.index + 1]
    finally:
        session.close()
