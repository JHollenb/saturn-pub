import json

from mdb.bench_cli import main
from mdb.bench_loading import BenchModelSpec
import mdb.job_support as launch


def test_model_identity_does_not_replace_planning_registry_name(tmp_path, monkeypatch):
    spec = BenchModelSpec("qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct", "local", "qwen2")
    monkeypatch.setattr("mdb.bench_loading.resolve_bench_model", lambda name: spec)
    captured = {}
    def submit(command, **kwargs):
        captured.update(kwargs)
        return "test-job"
    monkeypatch.setattr(launch, "guarded_mrun_launch", submit)
    monkeypatch.setattr(launch, "stage_worker", lambda *args, **kwargs: None)
    commands = tmp_path / "commands.json"
    commands.write_text("[]")
    assert main(["open", "--root", str(tmp_path / "session"), "--prompt", "hi",
                 "--commands", str(commands)]) == 0
    assert captured["request"]["model"] == spec.to_dict()
    assert captured["mrun_config"]["model"] == spec.registry_name
    assert captured["mrun_config"]["model_identity"] == spec.to_dict()
    assert captured["mrun_config"]["task_family"] == "forward"
    assert captured["retry_on_kill"] is False
    assert captured["preflight"] is True
