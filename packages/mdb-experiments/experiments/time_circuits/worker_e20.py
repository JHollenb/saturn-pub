"""Finite Qwen E20 reproduction; no upstream checkouts or nested leases."""
import base64
import gzip
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
sys.path.insert(0, str(Path(__file__).parent))
import paper_e20 as paper
from e20_bridge import execute_context

MARKER = "MDB_PAPER_REPORT_GZIP_B64="


class PaperOutput(io.StringIO):
    def write(self, value):
        if value.startswith(paper.MARKER):
            return super().write(value)
        return sys.__stdout__.write(value)


def main():
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    payload = request["payload"]
    root = Path(payload["output_root"]) / request["request_sha256"][:16]
    root.mkdir(parents=True, exist_ok=False)
    from mrun.models import snapshot_dir
    model_path = str(snapshot_dir("qwen2.5-1.5b"))
    weight = Path(model_path) / "model.safetensors"
    digest = paper.rsc.sha256_file(weight)
    if digest != payload["expected_weights_sha256"]:
        raise RuntimeError("E20 original paper checkpoint identity changed")
    paper.resolve_model_path = lambda *_args, **_kwargs: model_path
    native_run_context = paper.run_context
    sessions = []
    identity = {"model_id": "Qwen/Qwen2.5-1.5B", "family": "qwen2", "weights_sha256": digest,
        "dtype": "torch.float32", "numerical_contract": "paper-e20-fp32-eager-prefix-suffix-v1",
        "source_hashes": request["source_hashes"], "request_sha256": request["request_sha256"]}

    def run_context(cr, **settings):
        paper.run_context = native_run_context
        try:
            result, record = execute_context(paper, cr, root=root / f"context-{settings['context_index']}",
                identity=identity, **settings)
            sessions.append(record)
            return result
        finally:
            paper.run_context = run_context
    paper.run_context = run_context
    capture = PaperOutput()
    from contextlib import redirect_stdout
    with redirect_stdout(capture):
        status = paper.main()
    encoded = capture.getvalue().split(paper.MARKER)[-1].strip()
    report = json.loads(gzip.decompress(base64.b64decode(encoded)))
    report["mdb"] = {"schema": "mdb-paper-reproduction-v1", "sessions": sessions,
        "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"],
        "model_weights_sha256": digest, "root": str(root),
        "integration": "native research adapter; formed-KV tensor cuts and Model OS writes",
        "limitations": ["prefix residual formation uses the frozen paper native forward",
                        "these cuts are research inputs, not BenchSession token/layer cuts"]}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(MARKER + base64.b64encode(gzip.compress(json.dumps(report).encode())).decode(), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
