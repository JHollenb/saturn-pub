"""Fresh-process installed-wheel mechanics proof; no pretrained model is used."""

from __future__ import annotations

import argparse
import importlib.abc
import json
import sys
from pathlib import Path


class BlockPrivate(importlib.abc.MetaPathFinder):
    def __init__(self, *, framework_free=False):
        self.blocked = {"saturn", "common", "common_harness"}
        if framework_free:
            self.blocked |= {"torch", "transformers", "numpy"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise ImportError("qualification blocked import: " + fullname)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "view"))
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    sys.meta_path.insert(0, BlockPrivate(framework_free=args.mode == "view"))
    import saturn_pub

    import mdb

    installed = args.installed.resolve()
    for module in (mdb, saturn_pub):
        if installed not in Path(module.__file__).resolve().parents:
            raise RuntimeError("qualification escaped the installed wheel: " + module.__name__)
    root = args.root.resolve()
    if args.mode == "view":
        from mdb.circuit_discovery_view import typed_events
        from mdb.temporal_causal_viewer import normalize_evidence, render_html

        report = json.loads((root / "report.json").read_text())
        events = typed_events(report)
        bundle = normalize_evidence([events])
        (root / "circuit.html").write_text(render_html(bundle))
        output = {"framework_free": True, "events": len(bundle["events"]),
                  "semantic_cot_qualified": report["semantic_cot_qualified"]}
        (root / "viewer-qualification.json").write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output))
        return

    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from mdb.bench import BenchSession
    from mdb.bench_verify import verify

    torch.manual_seed(603)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attn_implementation="eager",
    )).eval().to(torch.bfloat16).requires_grad_(False)
    root.mkdir(parents=True, exist_ok=False)
    bench = BenchSession.from_resident(model, root=root / "session", continuation_capacity=16,
                                       attention_backend="native-eager")
    bench.prefill([1, 2, 3])
    result = bench.discover_circuit({
        "plan": [{"kind": "input", "token_ids": [7]}, {"kind": "generate", "count": 1},
                 {"kind": "input", "token_ids": [8]}, {"kind": "generate", "count": 1}],
        "receiver_offsets": [0, 2], "candidates_per_receiver": 1, "scales": [0.0, 0.5],
        "observable": {"kind": "token_margin", "token_id": 9, "against_token_id": 11},
    })
    report = result["report"]
    (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    native = verify(bench.root)
    exported = bench.export(root / "export")
    bench.close()
    saved = verify(exported["root"])
    resumed = BenchSession.resume(model, root=exported["root"])
    actual = []
    for token in (7, None, 8, None):
        if token is not None:
            resumed.stage_input(token)
        actual.append(resumed.step()["logits_sha256"])
    resumed.close()
    expected = [row["logits_sha256"] for row in report["baseline"]["trajectory"]]
    if actual != expected:
        raise RuntimeError("export/resume changed the native future")
    output = {"installed_mdb": str(Path(mdb.__file__).resolve()),
              "installed_saturn_pub": str(Path(saturn_pub.__file__).resolve()),
              "torch": torch.__version__, "specimen": "tiny random CPU BF16 Qwen2",
              "native_counts": native["counts"], "export_counts": saved["counts"],
              "restored_future_exact": True, "selected_candidates": 2, "dose_tests": 4,
              "first_query_and_keys_exact": all(t["local_prediction"]["query_and_keys_exact"] for t in report["tests"]),
              "semantic_cot_qualified": False}
    (root / "qualification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
