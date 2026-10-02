"""Copy the focused upstream runtime regressions with namespace-only changes."""
from pathlib import Path
import ast
import re

root = Path(__file__).resolve().parents[1]
source = root.parent / "saturn" / "tests"
names = sorted(source.glob("test_bench*.py"))
names += [source / f"test_{name}.py" for name in (
    "causal_context_qwen", "causal_context_vm", "mamba_state_vm",
    "source_frame_stack", "hyperion", "vm_cache", "vm_payload_cache",
    "future_forest", "debugger_protocol")]
for path in names:
    if path.exists():
        text = re.sub(r"\b(from|import) saturn\b", r"\1 mdb", path.read_text())
        (root / "tests" / path.name).write_text(text)

worker = (root.parent / "saturn/experiments/2026-10-01-open-bench-families/worker.py").read_text()
function = next(node for node in ast.parse(worker).body if isinstance(node, ast.FunctionDef) and node.name == "debugger")
helper = ast.get_source_segment(worker, function).replace("from saturn.", "from mdb.")
(root / "tests/debugger_fixture.py").write_text("import os\n\n" + helper + "\n")
for name in ("test_bench_family_debugger.py", "test_bench_diffusion.py"):
    test = root / "tests" / name
    text = test.read_text().replace('worker = runpy.run_path(str(Path(__file__).parents[1]/"experiments/2026-10-01-open-bench-families/worker.py"))', 'from debugger_fixture import debugger')
    test.write_text(text.replace('worker["debugger"](b)', 'debugger(b)'))
