"""One resident pretrained decoder; bounded project/CLI mechanics qualification."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def qualify(native_adapter, token_ids, root):
    import torch

    from mdb import BenchSession, Project
    from mdb.debugger_cli import DebugShell

    project = Project(root)
    with BenchSession.from_saturn_pub(
        native_adapter.session(token_ids), project=project, session_id="qualification"
    ) as owner:
        debug = owner.debugger()
        parent = debug.checkpoint("parent")["cut_id"]
        debug.step(4)  # Embedding and three decoder-layer transitions.
        hidden = owner.adapter.session.read("hidden").detach().clone()
        debug.restore(parent)
        debug.step(4)
        if not torch.equal(owner.adapter.session.read("hidden"), hidden):
            raise RuntimeError("same-program replay changed hidden state")
        shell = DebugShell(debug, json_output=True)
        commands = [
            "project analyze",
            "project callers port:hidden",
            "project consumers port:hidden",
            "project trace port:hidden --depth 3",
            "project backtrace hidden",
            "fork intervention",
            "set hidden 0",
            "commit",
            'project finding port:hidden --title "explicit zero write" --kind observation',
            "project findings",
        ]
        for command in commands:
            if not shell.execute(command):
                raise RuntimeError(f"project qualification command failed: {command}")
        expected = project.findings("qualification")[0]["annotation_id"]
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'mrun', 'saturn', 'common'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
from mdb import Project
p = Project(sys.argv[1])
assert p.analyze('qualification')['result']['authority'] == 'structural-description'
assert p.trace('qualification', 'port:hidden')['edges']
assert p.findings('qualification')[0]['annotation_id'] == sys.argv[2]
assert p.graph().verify().ok
"""
    child = subprocess.run(
        [sys.executable, "-I", "-c", code, str(project.root), expected],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if child.returncode:
        raise RuntimeError("offline project reopen failed: " + child.stderr[-3000:])
    analysis = Project(root).analyze("qualification")
    return {
        "same_program_hidden_replay": True,
        "offline_reopen_without_frameworks": True,
        "analysis_id": analysis["analysis_id"],
        "analysis_binding": analysis["binding"],
        "evidence": analysis["evidence"],
        "finding_id": expected,
        "commands": commands,
        "project_root": str(project.root),
    }


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from mrun.models import snapshot_dir
    from saturn_pub.adapters.decoder import DecoderAdapter
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    job = os.environ["MRUN_JOB_ID"]
    if not job.startswith("job-") or not job.removeprefix("job-").isalnum():
        raise ValueError("invalid job identity")
    root = Path("/mnt/big/mdb/project-analysis") / job
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(113)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    path = snapshot_dir(config["model"])
    print("PROJECT_QUALIFICATION_PHASE=load", flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
        )
        .to("cuda")
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    ids = tokenizer.encode("The capital of France is")[:8]
    print("PROJECT_QUALIFICATION_PHASE=workflow", flush=True)
    result = qualify(DecoderAdapter(model), ids, root / "project")
    result.update(
        schema="mdb-project-analysis-qualification-v1",
        model=config["model"],
        device="cuda",
        dtype="float32",
        prompt_tokens=len(ids),
        physical_transitions=8,
        elapsed_seconds=time.monotonic() - started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        mechanics_status="valid",
        instrument_status="native-transitions-ran",
        trend_status="not-evaluated",
        terminal_status="bounded-workflow-mechanics-only",
    )
    (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print("MDB_PROJECT_ANALYSIS_REPORT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
