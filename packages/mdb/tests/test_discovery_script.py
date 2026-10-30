import json

import pytest

from mdb.discovery_script import (
    DiscoveryJournal,
    DiscoveryScriptError,
    ScriptRunner,
    compile_command,
    parse_script,
)


def test_dialect_and_finite_expansion():
    with pytest.raises(DiscoveryScriptError):
        parse_script("step law")
    with pytest.raises(DiscoveryScriptError):
        parse_script("interface mdb-discovery-v1\nrepeat 999999\nstep law\nend")
    parsed = parse_script("interface mdb-discovery-v1\nrepeat 2\nstep law\nend")
    assert len(parsed["commands"]) == 2
    assert compile_command("step law --max-ops 5") == ("motion", {"action": "step", "unit": "law", "max_ops": 5})


class Fake:
    def __init__(self, blocker=False):
        self.calls = []
        self.blocker = blocker

    def call(self, operation, payload, **kwargs):
        self.calls.append((operation, payload))
        causes = []
        if len(self.calls) == 1:
            causes = [{"kind": "breakpoint", "id": "b"}]
            if self.blocker:
                causes.append({"kind": "budget"})
        return {"ok": True, "lane_id": "lane_00", "stop_causes": causes}


def test_handlers_defer_once_and_blockers_cannot_be_overridden(tmp_path):
    script = "interface mdb-discovery-v1\non-hit b\ninspect state\ncontinue --defer\nend\nstep law"
    fake = Fake()
    runner = ScriptRunner(fake, journal=DiscoveryJournal(tmp_path / "journal"))
    result = runner.run(script)
    assert result["ok"]
    assert [x[0] for x in fake.calls] == ["motion", "inspect", "motion", "status"]
    assert fake.calls[1][1]["target"]["lane_id"] == "lane_00"
    blocked = Fake(True)
    ScriptRunner(blocked).run(script)
    assert [x[0] for x in blocked.calls] == ["motion", "inspect", "status"]
    rows = DiscoveryJournal(tmp_path / "journal").verify()
    assert len(rows) == 4
    path = next((tmp_path / "journal/objects").glob("*.json"))
    path.write_text(json.dumps({"tampered": True}))
    with pytest.raises(ValueError, match="digest"):
        DiscoveryJournal(tmp_path / "journal")


def test_full_syntax_checked_before_mutation():
    fake = Fake()
    with pytest.raises(DiscoveryScriptError):
        ScriptRunner(fake).run("interface mdb-discovery-v1\nstep law\nexecute arbitrary python")
    assert not fake.calls
