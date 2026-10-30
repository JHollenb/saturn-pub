"""Live-register mechanics; random weights establish no semantic CoT claim."""

import json
from copy import deepcopy

import pytest
import torch
from test_bench_jacobian import lens

from mdb.bench import BenchSession
from mdb.bench_debugger import BenchDebuggerAdapter
from mdb.bench_verify import verify
from mdb.debugger_render import frame
from mdb.errors import BenchError
from mdb.jacobian_io import FrozenJacobianLens
from mdb.jacobian_workspace import metadata_sha256
from mdb.local_debugger import connect_local, tiny_qwen


@pytest.fixture
def resident():
    return tiny_qwen(seed=91)


def opened(model, root, *, attach=True, boundaries=(1, 2)):
    bench = BenchSession.from_resident(model, root=root, lane="native", continuation_capacity=8)
    bench.prefill([1, 2, 3])
    if attach:
        refs = []
        for boundary in boundaries:
            manifest, transport = lens(bench)
            manifest["coordinate_map"]["source_layer"] = boundary - 1
            manifest["coordinate_map"]["completed_layer_boundary"] = boundary
            manifest["fingerprint"] = metadata_sha256(
                {key: value for key, value in manifest.items() if key != "fingerprint"})
            refs.append(bench.pin_jacobian_lens(manifest, transport)["lens_ref"])
        bench.configure_cot_register(refs, top_k=3)
    return bench


def test_automatic_reads_preserve_native_future_and_memory(resident, tmp_path):
    plain = opened(resident, tmp_path / "plain", attach=False)
    watched = opened(resident, tmp_path / "watched")
    random_state = torch.get_rng_state().clone()
    for _ in range(2):
        native, observed = plain.step("token"), watched.step("token")
        assert native["token_id"] == observed["token_id"]
        assert native["next_token_id"] == observed["next_token_id"]
        assert native["logits_sha256"] == observed["logits_sha256"]
        assert torch.equal(plain.last_logits, watched.last_logits)
        for layer in range(2):
            for component in ("key", "value"):
                path = f"kv_cache.layer_{layer}.{component}.private"
                assert torch.equal(plain.resolve_value(path), watched.resolve_value(path))
    assert torch.equal(random_state, torch.get_rng_state())
    register = watched.cot_register()
    assert register["entry_count"] == 4
    assert {row["completed_layer_boundary"] for row in register["entries"]} == {1, 2}
    assert register["semantic_cot_qualified"] is False
    assert register["current_boundary_observed"] is False
    assert verify(watched.root)["counts"]["cot_artifacts"] == 5
    plain.close()
    watched.close()


def test_register_and_graph_follow_native_restore_and_siblings(resident, tmp_path):
    bench = opened(resident, tmp_path / "bench")
    initial = bench.branch.cut
    bench.step("layer")
    parent = bench.branch.cut
    assert bench.cot_register()["current_boundary_observed"]
    bench.fork("child")
    bench.step("layer")
    child = bench.cot_register()
    assert child["entry_count"] == 2
    assert child["entries"][-1]["branch"] == "child"
    panel = (bench.root / "cot-register.html").read_text()
    assert "Branch child" in panel and "Decoded vocabulary" in panel
    graph = bench.cot_graph()
    assert "Decoded candidates:" in graph["mermaid"]
    assert "annotates" in graph["mermaid"]
    assert all(row["causal_dependency_tested"] is False for row in child["entries"])
    bench.checkout("main")
    assert bench.cot_register()["entry_count"] == 1
    assert "Branch main" in (bench.root / "cot-register.html").read_text()
    bench.restore(initial)
    assert bench.cot_register()["entry_count"] == 0
    bench.restore(parent)
    assert bench.cot_register()["entry_count"] == 1
    bench.close()


def test_local_step_renders_register_without_an_inspect_query(resident, tmp_path):
    bench = opened(resident, tmp_path / "bench", boundaries=(1,))
    owner, session, _ = connect_local(BenchDebuggerAdapter(bench), identity=bench.identity, bench=bench)
    session.step(1, kind="layer")
    text = frame(session.status())
    assert "CoT register: token 3, after layer 1" in text
    assert "Vocabulary readout:" in text
    assert "Dependency has not been causally tested." in text
    session.step(1, kind="layer")
    assert "Historical observation" in frame(session.status())
    assert session.inspect("cot.graph")["semantic_cot_qualified"] is False
    owner.close()
    bench.close()


def test_observer_failure_keeps_native_execution_and_reports_error(resident, tmp_path, monkeypatch):
    import mdb.bench_cot as cot

    plain = opened(resident, tmp_path / "plain", attach=False)
    bench = opened(resident, tmp_path / "watched")

    def broken(_):
        raise RuntimeError("broken instrument")

    monkeypatch.setattr(cot, "capture", broken)
    native, observed = plain.step("token"), bench.step("token")
    assert native["logits_sha256"] == observed["logits_sha256"]
    register = bench.cot_register()
    assert register["entry_count"] == 0
    assert "broken instrument" in frame(bench.inspect())
    assert register["errors"]
    plain.close()
    bench.close()


def test_export_resume_retains_register_and_automatic_observation(resident, tmp_path):
    bench = opened(resident, tmp_path / "bench")
    bench.step("layer")
    before = bench.cot_register()["entries"]
    bench.export(tmp_path / "export")
    bench.close()
    resumed = BenchSession.resume(resident, root=tmp_path / "export")
    assert resumed.cot_register()["entries"] == before
    resumed.step("layer")
    assert resumed.cot_register()["entry_count"] == 2
    assert resumed.cot_register()["current_boundary_observed"]
    verify(resumed.root)
    resumed.close()


def test_edit_does_not_misrepresent_historical_readout_as_current(resident, tmp_path):
    bench = opened(resident, tmp_path / "bench")
    bench.step("layer")
    assert bench.cot_register()["current_boundary_observed"]
    act = bench.preview_edit("residual", 0.5, mode="scale")
    bench.commit_edit(act["act_id"])
    view = bench.cot_register()
    assert view["entry_count"] == 1
    assert not view["current_boundary_observed"]
    assert "Historical observation" in frame(bench.inspect())
    bench.close()


def test_configuration_is_atomic_and_invalid_bindings_refused(resident, tmp_path, monkeypatch):
    bench = opened(resident, tmp_path / "bench")
    events, cut = deepcopy(bench.events), bench.branch.cut
    config = bench.artifacts.get(events[-1]["config_ref"])
    with pytest.raises(BenchError, match="one lens"):
        bench.configure_cot_register([config["lens_refs"][0]] * 2)
    with pytest.raises(BenchError, match="top_k"):
        bench.configure_cot_register(top_k=True)
    with pytest.raises(BenchError, match="unique vocabulary"):
        bench.configure_cot_register(token_subset=[2, 2])
    assert bench.events == events and bench.branch.cut == cut
    original_publish = bench._publish

    def fail():
        raise OSError("injected publication failure")

    monkeypatch.setattr(bench, "_publish", fail)
    with pytest.raises(OSError):
        bench.configure_cot_register(enabled=False)
    assert bench.events == events and bench.branch.cut == cut
    monkeypatch.setattr(bench, "_publish", original_publish)
    bench.configure_cot_register(enabled=False)
    bench.step("token")
    assert bench.cot_register()["entry_count"] == 0
    bench.close()


def test_offline_verification_refuses_rebound_observation(resident, tmp_path):
    from mdb.bench_cot import verify_artifact

    bench = opened(resident, tmp_path / "bench")
    bench.step("layer")
    row = deepcopy(bench.cot_register()["entries"][-1])
    row["semantic_cot_qualified"] = True
    with pytest.raises(ValueError, match="interpretation binding"):
        verify_artifact(row, bench.artifacts, {"identity": bench.identity, "session_id": bench.session_id})
    assert "true" not in json.dumps(bench.cot_register()["semantic_cot_qualified"])
    bench.close()


def test_local_cli_attaches_saved_lens_and_shows_live_register(resident, tmp_path, capsys):
    from mdb.debugger_cli import main

    builder = BenchSession.from_resident(
        resident, root=tmp_path / "builder", model_id="tiny-qwen2-seed-91")
    builder.prefill([1, 2, 3])
    manifest, matrix = lens(builder)
    FrozenJacobianLens(manifest, matrix).save(tmp_path / "lens")
    builder.close()
    assert main(["debug", "--tiny", "qwen2", "--seed", "91", "--root", str(tmp_path / "cli"),
                 "--cot-lens", str(tmp_path / "lens"), "--command", "step layer 1"]) == 0
    text = capsys.readouterr().out
    assert "CoT register: token 3, after layer 1" in text
    assert (tmp_path / "cli" / "cot-register.html").is_file()


def test_live_graph_escapes_language_and_keeps_native_state(resident, tmp_path):
    from mdb.bench_cot import panel_html

    bench = opened(resident, tmp_path / "bench")
    bench.step("layer")
    row = deepcopy(bench.cot_register()["entries"][-1])
    row.pop("register_ref")
    row["candidates"][0]["text"] = '<script>alert("bad")</script>'
    ref = bench.artifacts.put(row)
    bench.events[-1]["register_ref"] = ref
    cut, residual = bench.branch.cut, bench.resolve_value("residual")
    text = panel_html(bench)
    assert '<script>alert("bad")</script>' not in text
    assert "&lt;script&gt;" in text
    assert bench.branch.cut == cut and torch.equal(residual, bench.resolve_value("residual"))
    bench.close()


def test_graph_render_failure_does_not_change_native_future(resident, tmp_path, monkeypatch):
    import mdb.bench_cot as cot

    plain = opened(resident, tmp_path / "plain", attach=False)
    bench = opened(resident, tmp_path / "bench")

    def fail(_):
        raise OSError("unavailable graph file")

    monkeypatch.setattr(cot, "write_live_panel", fail)
    native, observed = plain.step("token"), bench.step("token")
    assert native["logits_sha256"] == observed["logits_sha256"]
    assert "unavailable graph file" in frame(bench.inspect())
    assert bench.cot_register()["entry_count"] == 2
    plain.close()
    bench.close()
