import os

def debugger(bench):
    from contextlib import redirect_stdout
    from io import StringIO
    from mdb.bench_debugger import BenchDebuggerAdapter
    from mdb.debugger_cli import DebugShell
    from mdb.debugger_protocol import DebugCapability, InMemoryDebugTransport
    from mdb.debugger_session import DebugSession
    from mdb.debugger_worker import DebugWorkerController
    adapter = BenchDebuggerAdapter(bench); caps = adapter.capabilities
    capability = DebugCapability(job_id=os.environ["MRUN_JOB_ID"], session_id=bench.session_id,
        family=caps.family, model_identity=bench.identity, boundaries=caps.boundaries,
        inspectable_slots=caps.inspectable_slots, writable_slots=caps.writable_slots, exact_replay=True)
    worker = DebugWorkerController(capability, adapter, source_frame_stack=bench.source_frames)
    session = DebugSession(InMemoryDebugTransport(worker), job_id=os.environ["MRUN_JOB_ID"])
    shell = DebugShell(session, json_output=True); session.pause()
    kind = "layer" if bench.family == "mamba" else "block"
    worker.safe_point(kind, 0)
    output = StringIO()
    with redirect_stdout(output):
        for line in [f"break {kind} 1", f"step {kind} 1", "inspect residual", "inspect memory", "checkpoint debugger-parent",
                     "fork debugger-candidate", "set residual 1 scale", "preview", "abort", "set residual 1 scale", "preview", "commit", "cuts"]:
            if not shell.execute(line): raise RuntimeError(f"debugger command failed: {line}: {output.getvalue()[-1000:]}")
            c = bench.cursor(); worker.safe_point(kind, c.get("next_layer", c.get("block")) or 0)
    return {"transport": "in-process protocol; deployed fleet broker remains pending",
            "session_id": bench.session_id, "cut_id": bench.branch.cut, "receipt": session.write_receipt(),
            "commands": output.getvalue(), "same_python_identity": session.capabilities()["session_id"] == bench.session_id}
