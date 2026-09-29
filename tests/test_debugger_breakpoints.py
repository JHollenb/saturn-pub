import pytest

from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.cli import Debugger


def test_breakpoint_continue_and_exact_rewind():
    debugger = Debugger(QwenAdapter.tiny().session([5, 7, 11]))
    debugger.execute("break layer:1")
    stopped = debugger.execute("continue")
    assert stopped["stop"] == "breakpoint"
    assert stopped["steps"] == 2
    assert stopped["frame"]["boundary"] == "layer:1"
    debugger.execute("capture parent")
    debugger.execute("next")
    assert debugger.execute("where")["boundary"] != "layer:1"
    assert debugger.execute("restore parent")["verified_exact"]
    # A breakpoint hit does not trap continue at the current boundary.
    assert debugger.execute("continue")["steps"] > 1
    debugger.execute("delete all")
    assert debugger.execute("breakpoints") == {"breakpoints": []}


def test_bounded_unreachable_stop_and_wildcard():
    debugger = Debugger(QwenAdapter.tiny().session([5, 7, 11]))
    assert debugger.execute("until nonexistent 2")["stop"] == "step-limit"
    assert debugger.execute("until layer:* 4")["stop"] == "breakpoint"
    with pytest.raises(ValueError, match="positive"):
        debugger.execute("until layer:* 0")
