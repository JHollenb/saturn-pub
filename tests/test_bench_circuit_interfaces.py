from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import MambaConfig, MambaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchSession
from mdb.bench_cli import run_commands
from mdb.bench_debugger import BenchDebuggerAdapter
from mdb.debugger_cli import DebugShell
from mdb.debugger_protocol import (
    BreakpointSpec,
    DebugCapability,
    InMemoryDebugTransport,
)
from mdb.debugger_session import DebugSession
from mdb.debugger_worker import DebugWorkerController


@pytest.fixture(params=["qwen2", "mamba"])
def family(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def model(family: str):
    torch.manual_seed(113)
    if family == "mamba":
        return MambaForCausalLM(
            MambaConfig(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=2,
                state_size=4,
                expand=2,
                conv_kernel=4,
            )
        ).eval().to(device="cpu", dtype=torch.float32)
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            attn_implementation="eager",
        )
    ).eval().to(device="cpu", dtype=torch.bfloat16)


def _spec(family: str) -> dict:
    return {
        "name": "interface-circuit",
        "sites": [0],
        "contrasts": [{"name": "weaken", "mode": "scale", "value": 0.9}],
        "observable": {
            "kind": "token_margin",
            "token_id": 7,
            "against_token_id": 9,
        },
        "query_token_ids": [4],
        "continuation_tokens": 1,
        "memory_mediation": family == "mamba",
        "max_native_operations": 128,
        "max_capture_bytes": 64 * 1024 * 1024,
        "seed": 0,
    }


def _bench(model, tmp_path: Path) -> BenchSession:
    return BenchSession.from_resident(
        model,
        root=tmp_path / "bench",
        continuation_capacity=8,
    )


def _last_json(capsys: pytest.CaptureFixture[str]) -> dict:
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines
    return json.loads(lines[-1])


def _debug_session(bench: BenchSession) -> tuple[DebugSession, DebugWorkerController]:
    adapter = BenchDebuggerAdapter(bench)
    caps = adapter.capabilities
    capability = DebugCapability(
        job_id=f"job-{bench.session_id}",
        session_id=bench.session_id,
        family=caps.family,
        model_identity=bench.identity,
        boundaries=caps.boundaries,
        inspectable_slots=caps.inspectable_slots,
        writable_slots=caps.writable_slots,
        exact_replay=caps.exact_replay,
        state_schema_id=caps.state_schema_id,
        backend=caps.backend,
        consumer=caps.consumer,
        boundary_catalog=caps.boundary_catalog,
        address_specs=caps.address_specs,
        structs=caps.structs,
        algebra_ops=caps.algebra_ops,
        optional_operations=adapter.optional_operations(),
        checkpoint_requires_pause=caps.checkpoint_requires_pause,
    )
    worker = DebugWorkerController(capability, adapter)
    session = DebugSession(
        InMemoryDebugTransport(worker),
        job_id=capability.job_id,
        session_id=bench.session_id,
    )
    return session, worker


def test_finite_bench_json_dispatch_exposes_measurements(
    model, family: str, tmp_path: Path
) -> None:
    bench = _bench(model, tmp_path)
    try:
        bench.prefill([1, 2, 3])
        commands = [
            {"op": "measure_circuit", "spec": _spec(family)},
            {"op": "inspect", "path": "measurements"},
        ]
        results = run_commands(bench, commands)
        measured = results[0]["result"]
        python_status = bench.inspect()
        assert measured["session_id"] == python_status["session_id"]
        assert measured["cut_id"] == python_status["cut_id"]
        assert results[0]["session_id"] == python_status["session_id"]
        assert results[0]["cut_id"] == python_status["cut_id"]
        assert measured["report_ref"].startswith("cas://sha256/")
        assert measured["report"]["session_id"] == python_status["session_id"]
        assert measured["report"]["parent_cut"]

        measurements = results[1]["result"]
        assert measurements == bench.inspect("measurements")
        assert any(
            row["report_ref"] == measured["report_ref"] for row in measurements
        )
    finally:
        bench.close()


def test_live_debugger_shell_measure_and_replay_match_bench_identity(
    model, family: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench = _bench(model, tmp_path)
    spec = _spec(family)
    spec_file = tmp_path / "circuit-spec.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    try:
        bench.prefill([1, 2, 3])
        session, worker = _debug_session(bench)
        shell = DebugShell(session, json_output=True)

        session.add_breakpoint(BreakpointSpec("token", 0))
        worker.safe_point("token", 0, {"phase": "committed-token"})
        assert worker.status()["state"] == "paused"

        assert shell.execute(f"measure-circuit @{spec_file}")
        measured = _last_json(capsys)
        python_status = bench.inspect()
        assert measured["session_id"] == python_status["session_id"]
        assert measured["cut_id"] == python_status["cut_id"]
        assert session.current_cut == python_status["cut_id"]
        assert measured["report"]["session_id"] == python_status["session_id"]
        assert any(
            row["report_ref"] == measured["report_ref"]
            for row in bench.inspect("measurements")
        )

        assert shell.execute("inspect measurements")
        inspected = _last_json(capsys)
        assert inspected["value"] == bench.inspect("measurements")

        evidence_ref = measured["report"]["rows"][0]["evidence_ref"]
        assert shell.execute(f"replay-circuit {evidence_ref}")
        replayed = _last_json(capsys)
        python_after_replay = bench.inspect()
        assert replayed["exact"] is True
        assert replayed["evidence_ref"] == evidence_ref
        assert replayed["session_id"] == python_after_replay["session_id"]
        assert replayed["cut_id"] == python_after_replay["cut_id"]
        assert session.current_cut == python_after_replay["cut_id"]
    finally:
        bench.close()
