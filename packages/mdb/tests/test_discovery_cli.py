import json

import torch

from mdb.discovery import DiscoverySession
from mdb.discovery_cli import main, save_capture
from mdb.discovery_runtime import ReplayForward
from mdb.discovery_storage import save_session_frontier
from mdb.forward_recording import record_forward


class Arithmetic(torch.nn.Module):
    def forward(self, value):
        return (value + 1).square()


def recording():
    return record_forward(Arithmetic().eval(), (torch.ones(2, 3),),
        model_identity="cli-dispatch-test", execution={"backend": "cpu-fp32"}, input_identity="ones")


def test_replay_repeated_command_flags_do_not_replace_subcommand(tmp_path):
    recipe = tmp_path / "recipe"
    save_capture(recording(), recipe)
    output = tmp_path / "result"
    assert main(["replay", str(recipe), "--command", "step forward", "--command", "inspect outputs",
                 "--output", str(output)]) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["ok"] and result["commands_executed"] == 2


def test_resume_cli_selects_durable_path_and_executes_only_suffix(tmp_path):
    capture = recording()
    session = DiscoverySession()
    session.add_lane("a", lambda: ReplayForward(capture))
    assert session.call("motion", {"unit": "op"})["ok"]
    saved = save_session_frontier(session, "a", tmp_path / "store")
    output = tmp_path / "resumed"
    assert main(["resume", str(tmp_path / "store"), saved["cut_id"],
                 "--command", "step forward", "--output", str(output)]) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["status"]["lanes"][0]["pc"] == len(capture.trace["events"])
    assert result["status"]["work"]["replay_native_operations"] == len(capture.trace["events"])
    assert main(["inspect", str(tmp_path / "store"), "--cut", saved["cut_id"]]) == 0
