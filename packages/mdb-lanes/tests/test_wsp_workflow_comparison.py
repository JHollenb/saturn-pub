"""Integrity of the matched numerical editor experiment, not quality thresholds."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1] / "qualification/wsp_workflow_comparison"


def load(name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(__import__("sys").modules, name, module)
    spec.loader.exec_module(module)
    return module


def setup(monkeypatch):
    protocol = load("protocol", monkeypatch)
    return protocol, load("worker", monkeypatch)


def test_search_fixture_excludes_evaluation_and_true_gain_restores_function(monkeypatch):
    protocol, worker = setup(monkeypatch)
    model, splits, truth = protocol.fixture(7193, "sign", include_evaluation=False)
    assert set(splits) == {"discovery", "preservation"}
    assert worker.score(model, splits)["task_mse"] > 0
    with torch.no_grad():
        dict(model.named_modules())[truth["fault_site"]].weight.div_(truth["fault_gain"])
    assert worker.score(model, splits)["task_mse"] < 1e-29
    assert worker.score(model, splits)["preservation_mse"] < 1e-29


def test_edit_and_numerical_policy_refuse_invalid_evidence(monkeypatch):
    p, _ = setup(monkeypatch)
    for edits in ([{"site": "a", "gain": float("nan")}], [{"site": "missing", "gain": 1}],
                  [{"site": "a", "gain": 1}, {"site": "a", "gain": 2}]):
        with pytest.raises(ValueError):
            p.validate_edits(edits, ["a"])
    for code in ("import os", "x.__class__", "np.load('secret')", "while True: pass",
                 "model.net.forward(input_ids=x)", "model.net.train()", "model.net.forward = lambda x: x"):
        with pytest.raises(ValueError):
            p.validate_program(code)
    p.validate_program("out = [probe([{'site': s, 'gain': .5}]) for s in sites]")
    p.validate_program("estimates[np.min(distance, axis=1) < 0.035**2] = -np.inf")
    with pytest.raises(ValueError):
        p.validate_program("model.input_ids[0] = 0")
    assert p.public_getattr(torch.nn.Linear(2, 2), "bias") is not None
    assert p.public_getattr(object(), "missing", 3) == 3
    for name in ("__dict__", "forward", "load", "train"):
        with pytest.raises(ValueError):
            p.public_getattr(torch.nn.Linear(2, 2), name)


def test_native_budget_retains_completed_probes_and_restores_parameters(tmp_path, monkeypatch):
    _, worker = setup(monkeypatch)
    row = worker.search_one({"round": 1, "calls_per_round": 2, "program_seconds": 12},
        {"id": "blind-pytorch", "arm": "pytorch_hooks", "specimen": {"id": "blind", "seed": 7193, "kind": "sign"},
         "proposal": {"analysis_python": "for g in [0., .5, 1.]:\n    probe([{'site': sites[0], 'gain': g}])",
                      "forecast": "unknown"}}, tmp_path, [])
    assert len(row["probes"]) == 2
    assert row["native_work"]["completed_native_calls"] == 2
    assert row["native_work"]["rejected_native_calls"] == 1
    assert row["instrument_status"] == "evidence-incomplete"
    assert not row["parameter_drift"]
    assert (tmp_path / "probe-1/outcome.json").exists()
    assert not (tmp_path / "probe-2/outcome.json").exists()


def test_native_local_capture_is_typed_replay_of_live_final_position(tmp_path, monkeypatch):
    import numpy as np

    setup(monkeypatch)
    native = load("native_workflow", monkeypatch)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.write = torch.nn.Linear(4, 3, bias=False)
            self.last_readout = None

        def forward(self, rows):
            x = rows.float().reshape(-1, 1, 1).expand(-1, 2, 4)
            out = self.write(x)
            self.last_readout = {"logp": out[:, -1].detach().numpy().copy()}
            return out

    model = Tiny().eval().requires_grad_(False)
    rows = torch.arange(3)
    expected = model(rows).detach().numpy()
    taps, receipt = native.local_capture(model, rows, ["write"], tmp_path / "capture", "tiny")
    assert np.array_equal(taps["write"]["output"].numpy(), expected[:, -1])
    assert np.array_equal(model.last_readout["logp"], expected[:, -1])
    assert receipt["full_model_forwards"] == receipt["local_projection_replays"] == 1
    assert taps["write"]["input_port"] and taps["write"]["output_port"]
    assert not model.write._forward_hooks


def test_mdb_actual_native_ports_and_patch_run_under_same_budget(tmp_path, monkeypatch):
    _, worker = setup(monkeypatch)
    row = worker.search_one({"round": 1, "calls_per_round": 2, "program_seconds": 12},
        {"id": "blind-mdb", "arm": "mdb", "specimen": {"id": "blind", "seed": 7193, "kind": "sign"},
         "proposal": {"analysis_python": "taps = capture()\nr = linear_response(weights[sites[0]], weights[sites[0]]*.5, taps[sites[0]]['input'])\nprobe([{'site': sites[0], 'gain': .5}])",
                      "forecast": "local change known, task response unknown"}}, tmp_path, [])
    assert row["instrument_status"] == "observed", row["error"]
    assert row["native_work"]["completed_native_calls"] == 2
    assert len(row["captures"]) == len(row["probes"]) == 1
    assert not row["parameter_drift"]
    assert json.loads((tmp_path / "probe-0/verification.json").read_text())["outcome_manifest"]


def test_direct_parameter_mutation_is_not_a_search_result(tmp_path, monkeypatch):
    _, worker = setup(monkeypatch)
    row = worker.search_one({"round": 1, "calls_per_round": 2, "program_seconds": 12},
        {"id": "mutation", "arm": "wsp_python", "specimen": {"id": "blind", "seed": 7193, "kind": "sign"},
         "proposal": {"analysis_python": "for p in model.parameters():\n    p.zero_()", "forecast": "unknown"}}, tmp_path, [])
    assert row["instrument_status"] == "instrument-broken"
    assert row["parameter_drift"]


def test_normal_python_unpacking_and_numpy_runtime_imports(tmp_path, monkeypatch):
    _, worker = setup(monkeypatch)
    row = worker.search_one({"round": 1, "calls_per_round": 2, "program_seconds": 12},
        {"id": "numerical", "arm": "pytorch_hooks", "specimen": {"id": "blind", "seed": 7193, "kind": "sign"},
         "proposal": {"analysis_python": "u, _, vh = np.linalg.svd(weights[sites[0]].numpy())\nprint(torch.tensor([1., 2.]))\nprobe([])",
                      "forecast": "unknown"}}, tmp_path, [])
    assert row["instrument_status"] == "observed", row["error_traceback"]
    assert len(row["probes"]) == 1


def test_independent_reader_uses_actual_outputs_and_refuses_selection_drift(monkeypatch):
    import copy

    import numpy as np

    setup(monkeypatch)
    audit = load("audit_report", monkeypatch)
    x = np.array([[1., 0., -.5, .2], [.3, -.2, .7, 1.]])
    target, original = x + .1*np.tanh(x), x + .2*np.tanh(x)
    mse = float(np.mean((original-target)**2))
    native = {"task_mse": mse, "preservation_mse": mse, "objective": -2*mse,
              "task_outputs": original.tolist(), "preservation_outputs": original.tolist()}
    edited = {"task_mse": 0., "preservation_mse": 0., "objective": 0.,
              "task_outputs": target.tolist(), "preservation_outputs": target.tolist()}
    edits = [{"site": "s", "gain": .5}]
    trial = {"trial": "t", "arm": "mdb", "specimen": "o", "selection": edits,
        "native": native, "edited": edited, "restored": native, "oracle_reveal": {"kind": "amplification"},
        "evaluation_inputs": x.tolist(), "evaluation_targets": target.tolist(),
        "preservation_inputs": x.tolist(), "preservation_targets": target.tolist()}
    bundle = {"plan": {"arms": ["mdb"]}, "packets": {"o": {"sites": ["s"],
        "weights": {"s.weight": (.2*np.eye(4)).tolist()}, "baseline": native,
        "discovery": [x.tolist(), target.tolist()], "preservation": [x.tolist(), target.tolist()]}},
        "selections": [{"id": "t", "selection": edits}], "evaluation": [trial],
        "discovery": [{"trial": "t", "parameter_drift": False, "probes": [{"edits": edits, "result": edited}],
            "program_seconds": .01, "native_work": {"started_native_calls": 1, "native_rows": 4}, "error": None}],
        "proposals": [{"id": "t", "proposal": {"analysis_python": "probe(edits)"},
            "editor_receipt": {"usage": [{"input_tokens": 100, "output_tokens": 20}], "seconds": 1.}}]}
    report = audit.analyze(bundle)
    assert report["rows"][0]["numpy_consumer_comparisons"]["edited_task"]["maximum_absolute_difference"] == 0
    assert report["rows"][0]["target_mse_change"] == -mse
    altered = copy.deepcopy(bundle)
    altered["selections"][0]["selection"] = []
    with pytest.raises(ValueError, match="discovery-only"):
        audit.analyze(altered)


def test_native_consumer_uses_true_token_nll_and_original_distribution_kl(monkeypatch):
    from types import SimpleNamespace

    setup(monkeypatch)
    profile = load("qwen_profile", monkeypatch)

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            ids = torch.arange(len(texts)).reshape(-1, 1).repeat(1, 3)
            return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))

        def encode(self, text, **kwargs):
            return [int(text) + 4]

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Embedding(16, 24)

        def forward(self, input_ids, attention_mask, **kwargs):
            return SimpleNamespace(logits=self.head(input_ids[:, -1]).unsqueeze(1))

    net = Net().eval().requires_grad_(False)
    consumer = profile.Consumer(net, Tokenizer(), 7134)
    splits = {"discovery": (torch.arange(8), torch.zeros(8, 1)),
              "preservation": (torch.arange(8, 16), torch.zeros(8, 1))}
    original = consumer.comparison_score(splits, ("discovery", "preservation"))
    direct = consumer.anchor_logp[torch.arange(8), consumer.target_ids[:8]]
    assert original["task_mean_logp"] == pytest.approx(float(direct.mean()), rel=1e-6)
    assert original["preservation_kl"] == 0
    for invalid in (torch.tensor([0., 1.]), torch.tensor([0, 0]), torch.tensor([-1]), torch.tensor([16])):
        with pytest.raises(ValueError):
            consumer(invalid)
    reversed_splits = {"discovery": (torch.arange(7, -1, -1), torch.zeros(8, 1)),
                       "preservation": splits["preservation"]}
    with pytest.raises(ValueError, match="fixed ordered"):
        consumer.comparison_score(reversed_splits, ("discovery", "preservation"))
    with torch.no_grad():
        net.head.weight[:, 4] += .4
    changed = consumer.comparison_score(splits, ("discovery", "preservation"))
    actual = consumer.last_readout["logp"]
    expected = (consumer.anchor_logp[8:].exp() * (consumer.anchor_logp[8:] - torch.tensor(actual[8:]))).sum(-1).mean()
    assert changed["preservation_kl"] == pytest.approx(float(expected), rel=1e-6)
    assert changed["objective"] == pytest.approx(changed["task_mean_logp"]-changed["preservation_kl"], rel=1e-6)


def test_native_keyword_root_calls_share_probe_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace

    _, worker = setup(monkeypatch)
    profile = load("qwen_profile", monkeypatch)

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            ids = torch.arange(len(texts)).reshape(-1, 1)
            return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))

        def encode(self, text, **kwargs):
            return [int(text)+4]

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Embedding(16, 24)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.head(input_ids))

    net = Net().eval().requires_grad_(False)
    model = profile.Consumer(net, Tokenizer(), 1)
    ctx = {"model": model, "sites": ["net.head"], "guard_model": net,
        "input_guard": model.comparison_input_identity,
        "input_identity": {"native_table": model.input_identity},
        "splits": {"discovery": (torch.arange(8), torch.zeros(8, 1)),
                   "preservation": (torch.arange(8, 16), torch.zeros(8, 1))}}
    row = worker.search_one({"round": 1, "calls_per_round": 1, "program_seconds": 12},
        {"id": "native-budget", "arm": "wsp_python", "specimen": {"id": "native"},
         "proposal": {"analysis_python": "model.net(input_ids=model.input_ids)\nprobe([])", "forecast": "unknown"}},
        tmp_path, [], context=ctx)
    assert row["native_work"]["completed_native_calls"] == 1
    assert row["native_work"]["rejected_native_calls"] == 1
    assert row["instrument_status"] == "evidence-incomplete"
    assert not row["parameter_drift"] and not row["input_drift"]
    bound = tmp_path / "bound-probe"
    bound.mkdir()
    observed = worker.search_one({"round": 1, "calls_per_round": 1, "program_seconds": 12},
        {"id": "native-binding", "arm": "mdb", "specimen": {"id": "native"},
         "proposal": {"analysis_python": "probe([])", "forecast": "unchanged consumer"}},
        bound, [], context=ctx)
    assert observed["instrument_status"] == "observed", observed["error"]
    declarations = json.loads((bound / "probe-0/study/declarations.json").read_text())
    assert declarations["input_identity"]["native_table"] == model.input_identity


def test_native_target_boundary_refuses_suffix_retokenization(monkeypatch):
    setup(monkeypatch)
    load("qwen_profile", monkeypatch)
    native = load("native_workflow", monkeypatch)

    class Retokenizing:
        def encode(self, text, **kwargs):
            return [len(text)]

    with pytest.raises(ValueError, match="prompt boundary"):
        native.native_context(None, Retokenizing(), 1, "compact_modulo_space")


def test_native_staging_keeps_editor_and_custody_limits_distinct(tmp_path, monkeypatch):
    import zipfile

    setup(monkeypatch)
    launcher = load("followup_launch", monkeypatch)
    wheels = []
    for name, package in (("mdb_runtime", "mdb"), ("saturn_pub", "saturn_pub"), ("mrun_pub", "mrun")):
        path = tmp_path / (name+"-0.1-py3-none-any.whl")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(package+"/__init__.py", "")
        wheels.append(path)
    monkeypatch.setattr(launcher.runpy, "run_path", lambda path: {"wheels": lambda: wheels, "command": lambda request: ["not-executed"]})
    launcher.stage(tmp_path / "run", "prepare", config_extra={"limits": {"max_prompt_bytes": 96000}})
    request = json.loads((tmp_path / "run/request.json").read_text())
    assert request["config"]["limits"] == {"max_prompt_bytes": 96000}
    assert request["config"]["custody_limits"]["max_archive_bytes"] > 0
    seal = json.loads((tmp_path / "run/payload/payload-seal.json").read_text())
    assert seal["files"] == request["source_hashes"]


def test_timeout_proposal_has_explicit_fallback_and_no_fabricated_editor_usage(tmp_path, monkeypatch):
    setup(monkeypatch)
    launcher = load("launch", monkeypatch)
    directory = tmp_path / "editor-round-2/timed-out"
    directory.mkdir(parents=True)
    fallback = {"analysis_python": "print('no editor response')", "mechanism": "controller fallback", "forecast": "not applicable"}
    values = {"prompt.txt": "original prompt", "command.json": "[]", "failure.json": "{}",
              "controller-fallback.json": json.dumps(fallback)}
    for n, value in values.items():
        (directory / n).write_text(value)
    receipt = {"inference_status": "timeout", "usage": [], "usage_status": "unavailable",
        "prompt_sha256": launcher.sha(directory / "prompt.txt"), "command_sha256": launcher.sha(directory / "command.json"),
        "failure_sha256": launcher.sha(directory / "failure.json"), "fallback_sha256": launcher.sha(directory / "controller-fallback.json")}
    rows = [{"id": "timed-out", "proposal": fallback, "editor_receipt": receipt}]
    proposals = tmp_path / "editor-round-2/proposals.json"
    proposals.write_text(json.dumps(rows))
    source = tmp_path / "source"
    source.mkdir()
    (source / "request.json").write_text(json.dumps({"source_hashes": {"proposals.json": launcher.sha(proposals)}}))
    assert launcher.verified_proposals(tmp_path, 2, source) == rows
    assert not (directory / "answer.json").exists() and not (directory / "events.jsonl").exists()
    (directory / "controller-fallback.json").write_text("{}")
    with pytest.raises(ValueError, match="custody differs"):
        launcher.verified_proposals(tmp_path, 2, source)


def test_editor_timeout_stops_process_group_and_keeps_partial_output(tmp_path, monkeypatch):
    import sys

    setup(monkeypatch)
    launcher = load("launch", monkeypatch)
    result, timed_out = launcher.run_editor_process(
        [sys.executable, "-u", "-c", "import time; print('partial-event',flush=True); time.sleep(10)"],
        "", tmp_path, .2)
    assert timed_out and result.returncode != 0
    assert "partial-event" in result.stdout
