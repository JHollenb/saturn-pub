"""Pinned E6 feature interchange through MDB native input cuts."""
import base64
import gzip
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
from contextlib import redirect_stdout
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
sys.path.insert(0, str(Path(__file__).parent))
import paper_e6 as paper
from e6_bridge import FeatureSessions, graph_reference

MARKER = "MDB_PAPER_REPORT_GZIP_B64="


class PaperOutput(io.StringIO):
    def write(self, value):
        if value.startswith(paper.MARKER):
            return super().write(value)
        return sys.__stdout__.write(value)


def digest(path):
    h = sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def main():
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    payload = request["payload"]
    root = Path(payload["output_root"]) / request["request_sha256"][:16]
    root.mkdir(parents=True, exist_ok=False)
    import circuit_tracer
    import torch
    from circuit_tracer import ReplacementModel
    from circuit_tracer.graph import compute_graph_scores, compute_node_influence
    from transformers import AutoConfig, AutoTokenizer

    from mdb.circuit_runtime import DirectoryArtifactStore
    versions = {n: importlib.metadata.version(n) for n in ("torch", "transformers", "nnsight", "circuit-tracer", "numpy")}
    if versions["transformers"] != "4.57.3" or versions["nnsight"] != "0.8.0rc1" or versions["torch"] != "2.13.0":
        raise RuntimeError("E6 pinned instrument toolchain changed")
    repo = Path(circuit_tracer.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != payload["circuit_tracer_commit"]:
        raise RuntimeError("E6 circuit-tracer source revision changed")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--", "circuit_tracer"], text=True).strip():
        raise RuntimeError("E6 pinned instrument sources have uncommitted changes")
    hub = Path(os.environ["HF_HOME"]) / "hub"
    expected_references = {
        "models--google--gemma-2-2b": "25319945f7fd83b8b903e12081777b7eef2ba993",
        "models--mwhanna--gemma-scope-transcoders": "bd5773156dea09893636c801df1237d0410307d2",
    }
    for name, reference in expected_references.items():
        if (hub / name / "refs/main").read_text().strip() != reference:
            raise RuntimeError("E6 offline instrument asset revision changed: " + name)
    weight = Path(payload["model_local_path"]) / "model.safetensors"
    weights_sha = digest(weight)
    if weights_sha != payload["expected_weights_sha256"]:
        raise RuntimeError("E6 Gemma checkpoint changed")
    # Native dependency bytes are evidence too; a dirty pinned checkout is not
    # silently described as identical to its Git revision.
    instrument_sources = {p.relative_to(repo).as_posix(): digest(p)
                          for p in sorted((repo / "circuit_tracer").rglob("*.py"))}
    deterministic = bool(payload.get("deterministic_algorithms", True))
    torch.use_deterministic_algorithms(deterministic)
    environment = {"python": platform.python_version(), "packages": versions, "asset_references": expected_references,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "circuit_tracer_commit": commit, "instrument_sources": instrument_sources}
    print("MDB_EXECUTION_ENVIRONMENT=" + json.dumps(environment, sort_keys=True), flush=True)
    (root / "execution-environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(payload["model_local_path"])
    # The frozen worker patches the tokenizer's offline Mistral check before
    # loading. Delay our tokenizer until after that patch has executed.
    identity = {"model_id": "unsloth/gemma-2-2b", "revision": "25319945f7fd83b8b903e12081777b7eef2ba993",
        "weights_sha256": weights_sha, "family": "gemma2-circuit-tracer", "dtype": "torch.bfloat16",
        "numerical_contract": "paper-e6-nnsight-bf16-clean-error-" + ("deterministic-v1" if deterministic else "native-repeat-v1"),
        "deterministic_algorithms": deterministic, "instrument": environment,
        "source_hashes": request["source_hashes"], "request_sha256": request["request_sha256"],
        "operator": payload["e6_mode"]}
    original = ReplacementModel.from_pretrained
    original_attribute = circuit_tracer.attribute
    graph_store = DirectoryArtifactStore(root / "graphs")
    graph_refs = []
    def attribute(inputs, model, **kwargs):
        graph = original_attribute(inputs, model, **kwargs)
        replacement, completeness = compute_graph_scores(graph)
        n = len(graph.selected_features)
        weights = torch.zeros(graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device)
        weights[-len(graph.logit_targets):] = graph.logit_probabilities
        influence = compute_node_influence(graph.adjacency_matrix, weights)[:n].float().cpu().tolist()
        artifact = graph_reference(graph, identity=identity, prompt=inputs,
            replacement=replacement, completeness=completeness, influence=influence)
        graph_refs.append({"prompt": inputs, "reference": graph_store.put(artifact)})
        return graph
    circuit_tracer.attribute = attribute
    owner = None
    def from_pretrained(cls, *args, **kwargs):
        nonlocal owner
        model = original(*args, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(payload["model_local_path"])
        owner = FeatureSessions(root=root, identity=identity, tokenizer=tokenizer,
            layers=config.num_hidden_layers, features=int(model.transcoders.d_transcoder), operator=payload["e6_mode"],
            reference_repeats=int(payload.get("reference_repeats", 1)))
        owner.attach(model)
        return model
    ReplacementModel.from_pretrained = classmethod(from_pretrained)
    capture = PaperOutput()
    try:
        with redirect_stdout(capture):
            status = paper.main()
        report = json.loads(gzip.decompress(base64.b64decode(capture.getvalue().split(paper.MARKER)[-1].strip())))
        summaries = owner.finish()
        report["mdb"] = {"schema": "mdb-paper-reproduction-v1", "sessions": summaries,
            "calls": owner.records, "graphs": graph_refs, "native_reference_seconds": owner.reference_seconds,
            "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"],
            "root": str(root), "model_weights_sha256": weights_sha,
            "integration": "native circuit-tracer consumer; admitted integer feature addresses and durable input cuts",
            "limitations": ["Gemma interactive BenchSession adapter remains unqualified",
                            "native output repeatability is measured explicitly; bit equivalence is not presumed",
                            "clean reconstruction error terms remain the original paper instrument policy",
                            "native KV head-to-head in the paper uses a separate historical panel",
                            "external attribution graph internals are not arbitrary resumable model cuts"]}
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(MARKER + base64.b64encode(gzip.compress(json.dumps(report).encode())).decode(), flush=True)
        if deterministic and any(not replay["exact"] for session in summaries for replay in session["replays"]):
            return 2
        return 2 if any(item.get("error") for item in report["items"]) else status
    finally:
        ReplacementModel.from_pretrained = original
        circuit_tracer.attribute = original_attribute
        if owner:
            owner.detach()


if __name__ == "__main__":
    raise SystemExit(main())
