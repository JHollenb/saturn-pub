import json
import subprocess
import sys

import pytest

from saturn_pub import Act, Adapter, Session, StateCut
from saturn_pub.cli import Debugger
from saturn_pub.dependencies import Cell, DependencyGraph
from saturn_pub.investigation import Arm, Investigation
from saturn_pub.program import Instruction, Program
from saturn_pub.store import LocalStore


class Counter(Adapter):
    model_identity = "counter-v1"
    execution = {"family": "counter"}

    def boundary(self, state):
        return "step"

    def validate(self, state):
        if type(state["value"]) is not int or state["value"] < 0:
            raise ValueError("nonnegative integer required")

    def addresses(self, state):
        return ("value",)

    def advance(self, state):
        if state["value"] == 99:
            raise RuntimeError("injected consumer failure")
        return {"value": state["value"] + 1}


def test_branch_commit_restore_and_stale_commit():
    session = Session(Counter(), {"value": 0})
    parent = session.capture()
    left, right = session.fork(parent), session.fork(parent)
    left.apply(Act.add("value", 2))
    left.continue_()
    assert right.read("value") == session.read("value") == 0
    session.commit(left)
    assert session.read("value") == 3
    with pytest.raises(ValueError, match="current parent"):
        session.commit(right)
    session.restore(parent)
    assert session.capture().fingerprint == parent.fingerprint
    assert left.read("value") == 3


def test_failed_act_and_continuation_are_atomic():
    session = Session(Counter(), {"value": 98})
    parent = session.capture()
    with pytest.raises(RuntimeError):
        session.continue_(2)
    assert session.capture().fingerprint == parent.fingerprint
    with pytest.raises(ValueError):
        session.apply(Act.replace("value", -1))
    assert session.capture().fingerprint == parent.fingerprint
    with pytest.raises(ValueError, match="declared writes"):
        session.apply(Act("bad", ("value",), ("value",), lambda _: {"secret": 1}))
    assert session.read("value") == 98


def test_receipt_names_current_cut_and_abort_keeps_parent():
    session = Session(Counter(), {"value": 0})
    receipt = session.apply(Act.add("value", 1))
    assert receipt.to_dict()["result"] == session.capture().fingerprint
    parent = session.capture()
    child = session.fork(parent)
    child.continue_()
    assert session.abort(child).to_dict()["result"] == parent.fingerprint
    assert session.capture().fingerprint == parent.fingerprint
    assert child.read("value") == 2
    with pytest.raises(ValueError):
        session.commit(child)


def test_declared_read_visibility_and_cut_tamper():
    session = Session(Counter(), {"value": 0, "secret": 42})
    seen = []
    session.apply(
        Act(
            "read",
            ("value",),
            ("value",),
            lambda x: (seen.append(set(x)), {"value": x["value"]})[1],
        )
    )
    assert seen == [{"value"}]
    cut = session.capture()
    cut._payload["value"] = 10
    with pytest.raises(ValueError, match="changed"):
        session.restore(cut)
    wrong = StateCut("wrong", {}, "step", None, {"value": 0})
    with pytest.raises(ValueError, match="incompatible"):
        session.restore(wrong)


def test_store_fresh_process_and_metadata_tamper(tmp_path):
    session = Session(Counter(), {"value": 3})
    store = LocalStore(tmp_path)
    identifier = store.save(session.capture())
    code = (
        "from saturn_pub.store import LocalStore; import sys; "
        "s=LocalStore(sys.argv[1]); c=s.load(sys.argv[2]); "
        "assert c.payload['value']==3; assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code, str(tmp_path), identifier], check=True)
    manifest = tmp_path / "cuts" / f"{identifier}.json"
    data = json.loads(manifest.read_text())
    data["manifest"]["boundary"] = "tampered"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        store.load(identifier)
    with pytest.raises(ValueError):
        store.load("../../escape")


def test_debugger_delegates_to_session():
    debugger = Debugger(Session(Counter(), {"value": 0}))
    debugger.execute("capture start")
    debugger.execute("fork candidate start")
    debugger.execute("use candidate")
    debugger.execute("add value 3")
    debugger.execute("step 2")
    assert debugger.execute("read value") == 5
    debugger.execute("use root")
    debugger.execute("commit candidate")
    assert debugger.execute("read value") == 5
    debugger.execute("restore start")
    assert debugger.execute("read value") == 0


def test_panels_preserve_errors_and_program_support():
    session = Session(Counter(), {"value": 0})
    act = Act.add("value", 2)
    plan = Investigation(
        "effect",
        (Arm("native"), Arm("candidate", (act,)), Arm("broken", (Act.replace("value", -1),))),
        1,
    )
    panel = plan.run(session, lambda branch: {"value": branch.read("value")})
    assert [row["status"] for row in panel["rows"]] == [
        "completed",
        "completed",
        "instrument-error",
    ]
    assert panel["rows"][1]["metrics"]["value"] == 3
    assert session.read("value") == 0
    program = Program.compile(session, panel, "candidate", (Instruction(0, act),), context="tiny")
    assert program.run(session, context="tiny").read("value") == 3
    with pytest.raises(ValueError, match="support"):
        program.run(session, context="other")
    with pytest.raises(ValueError, match="budget"):
        program.run(session, context="tiny", continuation_steps=2)
    with pytest.raises(ValueError, match="match"):
        Program.compile(
            session, panel, "candidate", (Instruction(0, Act.add("value", 7)),), context="tiny"
        )
    assert Act.add("value", 2).manifest() != Act.add("value", 7).manifest()


def test_dependency_reuse_and_feedback_invalidation():
    calls = []
    graph = DependencyGraph(
        (
            Cell("source", ("input",), lambda x: x["input"] * 2, "1"),
            Cell("writer", ("source", "latent"), lambda x: x["source"] + x["latent"], "1"),
            Cell("schedule", ("writer",), lambda x: x["writer"] + 1, "1"),
        )
    )

    def consumer(x):
        calls.append(x["schedule"])
        return x["schedule"]

    output, _ = graph.run({"input": 2, "latent": 1}, consumer)
    assert output == 6
    _, receipt = graph.run({"input": 2, "latent": 1}, consumer)
    assert len(receipt["reused"]) == 3 and len(calls) == 2
    _, changed = graph.run({"input": 2, "latent": 5}, consumer)
    assert changed["reused"] == ["source"]
    assert graph.dirty({"latent"}) == {"latent", "writer", "schedule"}
